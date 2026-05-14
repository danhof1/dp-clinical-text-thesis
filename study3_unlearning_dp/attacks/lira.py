"""
LiRA — Likelihood Ratio Attack (Carlini et al. 2022) and its unlearning
adaptation U-LiRA (Hayes et al. 2024).

Given shadow models with known membership masks, LiRA fits per-example
Gaussians over a SIGNAL f(x) (we use logit of confidence for LLM-next-token
prediction — see Carlini's paper for why this is near-optimal):

  f_IN_i   = shadow losses on x when x IN the shadow's training set
  f_OUT_i  = shadow losses on x when x OUT of the shadow's training set

We fit Normal(μ_IN, σ_IN) and Normal(μ_OUT, σ_OUT) per example. Then for
a target model, we compute its loss on x and report the likelihood ratio:

  score(x) = logpdf(target_loss; μ_IN, σ_IN) - logpdf(target_loss; μ_OUT, σ_OUT)

Higher = more likely IN.

Offline vs. Online
------------------
- OFFLINE-LiRA: fit only the OUT Gaussian and use a global IN distribution.
  Much cheaper because we don't need an IN shadow for every target example.
- ONLINE-LiRA: fit both IN and OUT Gaussians per example. Needs many
  shadow models per example; we run this only in the headline ε configuration.

U-LiRA
------
For each shadow model i with the target example IN, we additionally have the
UNLEARNED counterpart (trained by shadow_training, with cfg.train_unlearned_counterpart=True).
U-LiRA fits a different pair of Gaussians:

  f_UNLEARNED_i = unlearned-shadow loss on x   (x was IN pre-unlearning)
  f_OUT_i       = OUT-shadow loss on x

The target is the *unlearned* target model. This tests whether unlearning
genuinely erased the membership signal or only obscured it.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_from_disk
from peft import PeftModel
from scipy.stats import norm
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import AttackResult, save_results, summarize
from .common import compute_token_log_probs, loss_from_log_probs


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("attacks.lira")


# ------- Scoring primitives -------

def example_loss(
    model, tokenizer, text: str, max_length: int = 1024, device: str = "cuda",
) -> float:
    tok_lp = compute_token_log_probs(model, tokenizer, text, max_length, device)
    return loss_from_log_probs(tok_lp)


def logit_of_confidence(loss: float) -> float:
    """
    LiRA signal: logit-transform of model confidence, per Carlini et al. 2022 Sec 3.4.
    We approximate using the per-sequence loss. p = exp(-loss) is the
    geometric mean next-token probability. logit = log(p / (1-p)).

    This transform makes the IN and OUT distributions much closer to Gaussian,
    which makes the likelihood ratio better-calibrated.
    """
    p = float(np.exp(-loss))
    p = min(max(p, 1e-8), 1 - 1e-8)
    return float(np.log(p / (1 - p)))


# ------- Shadow score collection -------

def score_texts_with_model(
    model_path: str,
    base_model: str,
    texts: list[str],
    tokenizer,
    max_length: int,
    device: str,
    has_adapter: bool = True,
) -> np.ndarray:
    """
    Load model, compute per-text loss, return array of logit-confidences.
    If has_adapter=True, model_path is a PEFT adapter on top of base_model.
    If has_adapter=False, model_path is a full model checkpoint.
    """
    dtype = torch.bfloat16
    if has_adapter:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype, attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, attn_implementation="sdpa",
        )
    model.eval().to(device)

    out = np.zeros(len(texts), dtype=np.float32)
    for i, t in enumerate(tqdm(texts, desc=f"score {Path(model_path).name}")):
        out[i] = logit_of_confidence(example_loss(model, tokenizer, t, max_length, device))

    del model
    torch.cuda.empty_cache()
    return out


def collect_shadow_scores(
    shadow_root: Path,
    base_model: str,
    texts: list[str],
    tokenizer,
    max_length: int,
    device: str,
    use_unlearned: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      scores: [n_shadows, n_texts] — logit-confidence from each shadow on each text
      membership_matrix: [n_shadows, n_texts_in_population] — IN/OUT record

    Note: `texts` must be the population in the same order as population_ids.json.
    """
    shadow_root = Path(shadow_root)
    mm = np.load(shadow_root / "membership_matrix.npy")
    n_shadows = mm.shape[0]

    all_scores = np.zeros((n_shadows, len(texts)), dtype=np.float32)
    for i in range(n_shadows):
        if use_unlearned:
            ckpt_dir = shadow_root / f"shadow_{i:03d}_unlearned"
            # find final checkpoint
            final_pointer = ckpt_dir / "final.json"
            if final_pointer.exists():
                ckpt = Path(json.loads(final_pointer.read_text())["final_checkpoint"])
            else:
                ckpts = sorted(ckpt_dir.glob("step_*"))
                if not ckpts:
                    raise FileNotFoundError(f"no unlearned ckpt for shadow {i}")
                ckpt = ckpts[-1]
            has_adapter = False  # unlearn/run.py saves a full model snapshot
        else:
            ckpt = shadow_root / f"shadow_{i:03d}"
            has_adapter = True

        all_scores[i] = score_texts_with_model(
            str(ckpt), base_model, texts, tokenizer, max_length, device, has_adapter,
        )
    return all_scores, mm


# ------- LiRA proper -------

def lira_per_example_score(
    target_score: float,
    in_scores: np.ndarray,  # shadow scores where IN
    out_scores: np.ndarray, # shadow scores where OUT
    mode: str = "online",   # "online" or "offline"
    fix_variance: bool = True,
) -> float:
    """
    Likelihood ratio score for ONE target example.
    Higher = more likely IN.
    """
    if len(out_scores) < 2:
        return float("nan")

    mu_out = float(out_scores.mean())
    sd_out = float(out_scores.std(ddof=1)) if len(out_scores) > 1 else 1.0
    sd_out = max(sd_out, 1e-3)

    if mode == "offline":
        # No IN Gaussian; just use z-score vs OUT
        return (target_score - mu_out) / sd_out

    # online LiRA
    if len(in_scores) < 2:
        # Fall back to offline for this example
        return (target_score - mu_out) / sd_out

    mu_in = float(in_scores.mean())
    sd_in = float(in_scores.std(ddof=1))
    sd_in = max(sd_in, 1e-3)

    if fix_variance:
        # Carlini et al. found fixing variance to the global estimate works
        # better than per-example variance with few shadows.
        sd_in = sd_out = float(np.concatenate([in_scores, out_scores]).std(ddof=1))
        sd_in = sd_out = max(sd_in, 1e-3)

    log_in = norm.logpdf(target_score, mu_in, sd_in)
    log_out = norm.logpdf(target_score, mu_out, sd_out)
    return float(log_in - log_out)


def run_lira(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shadow_root = Path(cfg["shadow_root"])

    # Load population (must match what shadow_training used)
    population = load_from_disk(str(shadow_root / "population"))
    with open(shadow_root / "population_ids.json") as f:
        population_ids = json.load(f)
    assert len(population) == len(population_ids)

    texts = [r["text"] for r in population]
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = cfg.get("max_length", 1024)

    # --- Collect shadow scores ---
    log.info("collecting shadow scores (trained) over %d examples", len(texts))
    shadow_scores, mm = collect_shadow_scores(
        shadow_root, cfg["base_model"], texts, tokenizer, max_length, device,
        use_unlearned=False,
    )

    is_ulira = cfg.get("ulira", False)
    if is_ulira:
        log.info("collecting unlearned-shadow scores for U-LiRA")
        unl_scores, _ = collect_shadow_scores(
            shadow_root, cfg["base_model"], texts, tokenizer, max_length, device,
            use_unlearned=True,
        )

    # --- Target model score ---
    log.info("scoring target model %s", cfg["target_model_path"])
    target_scores = score_texts_with_model(
        cfg["target_model_path"], cfg["base_model"], texts, tokenizer,
        max_length, device,
        has_adapter=cfg.get("target_has_adapter", True),
    )

    # --- Per-example LiRA ---
    mode = cfg.get("mode", "online")
    per_ex = np.zeros(len(texts), dtype=np.float64)
    for j in range(len(texts)):
        in_mask = mm[:, j] == 1
        out_mask = mm[:, j] == 0
        in_s = shadow_scores[in_mask, j]
        out_s = shadow_scores[out_mask, j]

        if is_ulira:
            # U-LiRA: IN distribution is replaced with UNLEARNED-when-was-IN
            in_s = unl_scores[in_mask, j]

        per_ex[j] = lira_per_example_score(
            target_scores[j], in_s, out_s, mode=mode, fix_variance=True,
        )

    # --- Split into target membership labels ---
    # Target membership in the TARGET model, not the shadow models.
    # The target is the real DP-LoRA / unlearn+DP model, whose "in" set is splits["finetune"].
    splits_path = cfg["splits_path"]
    splits = load_from_disk(splits_path)
    finetune_ids = set()
    for r in splits["finetune"]:
        fid = r.get("doc_id") or __import__("hashlib").sha256(
            r["text"].encode()).hexdigest()[:16]
        finetune_ids.add(fid)

    member_idx = [j for j, pid in enumerate(population_ids) if pid in finetune_ids]
    nonmember_idx = [j for j, pid in enumerate(population_ids) if pid not in finetune_ids]
    log.info("target-model labels: %d members, %d non-members in population",
             len(member_idx), len(nonmember_idx))

    results = [
        summarize(
            f"lira_{mode}{'_ulira' if is_ulira else ''}",
            per_ex[member_idx], per_ex[nonmember_idx],
        )
    ]

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_results(results, out_dir / f"lira_{mode}{'_ulira' if is_ulira else ''}.json")
    np.save(out_dir / f"lira_per_example_scores.npy", per_ex)

    log.info("\n=== LiRA summary ===")
    for r in results:
        log.info(
            "%-20s AUC=%.3f TPR@1%%=%.3f TPR@0.1%%=%.3f",
            r.name, r.auc, r.tpr_at_1pct_fpr, r.tpr_at_0p1pct_fpr,
        )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--target_model_path_override", default=None)
    ap.add_argument("--output_dir_override", default=None)
    ap.add_argument("--mode", default=None, help="override online/offline")
    ap.add_argument("--ulira", action="store_true", help="U-LiRA mode")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.target_model_path_override:
        cfg["target_model_path"] = args.target_model_path_override
    if args.output_dir_override:
        cfg["output_dir"] = args.output_dir_override
    if args.mode:
        cfg["mode"] = args.mode
    if args.ulira:
        cfg["ulira"] = True

    run_lira(cfg)


if __name__ == "__main__":
    main()

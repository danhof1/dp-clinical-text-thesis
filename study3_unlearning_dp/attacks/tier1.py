"""
Tier 1 attacks: fast, reference-free or single-reference attacks that run on
every model configuration (every ε, every unlearning method).

Implements:
  - loss                         (Yeom et al. 2018)
  - Min-K% Prob                  (Shi et al. 2024)
  - Min-K%++                     (Zhang et al. 2024, improves over Min-K%)
  - reference-model loss ratio   (requires a reference model; use base LLM)
  - zlib ratio                   (Carlini et al. 2021; weak but cheap)

For each attack we compute scores on:
  - members: the fine-tuning set (MTSamples notes we trained on)
  - nonmembers: held-out MTSamples notes (same distribution, presumed in pretraining)
  - clean_nonmembers: held-out from a corpus NOT in pretraining

Reporting members vs. nonmembers vs. clean_nonmembers lets us DECOMPOSE
the MIA signal into (pretraining contamination) + (fine-tuning memorization),
which is the key analysis for the paper.
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
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import (
    AttackResult,
    compute_token_log_probs,
    loss_from_log_probs,
    min_k_percent_score,
    save_results,
    summarize,
    zlib_ratio_score,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("attacks.tier1")


def min_k_pp_score(model_tok_lp: np.ndarray, ref_tok_lp: np.ndarray, k_pct: float = 20.0) -> float:
    """
    Min-K%++ (Zhang et al. 2024): normalize each token's log-prob by the
    expected log-prob under the model, then take the mean of the bottom-k%.
    Requires access to the full distribution, which compute_token_log_probs
    doesn't give us directly. For practical use we approximate with a
    reference-normalized version: (tok_lp_model - tok_lp_ref).mean() of bottom-k.

    Higher = more likely member (same convention as Min-K%).
    """
    if model_tok_lp.size == 0:
        return -float("inf")
    n = min(len(model_tok_lp), len(ref_tok_lp))
    normalized = model_tok_lp[:n] - ref_tok_lp[:n]
    k = max(1, int(n * k_pct / 100.0))
    lowest = np.sort(normalized)[:k]
    return float(lowest.mean())


def reference_ratio_score(model_loss: float, ref_loss: float) -> float:
    """
    ref_loss - model_loss. Higher = model is more confident than reference,
    which is evidence of memorization.
    """
    return ref_loss - model_loss


def score_dataset(
    model,
    tokenizer,
    dataset,
    max_length: int,
    device: str,
    reference_model=None,
) -> dict[str, np.ndarray]:
    """
    Run all Tier 1 attacks on one dataset split.

    Returns a dict of score-arrays, one per attack, of length len(dataset).
    """
    scores = {
        "loss": [],           # note: we negate at summarize time (lower loss = member)
        "min_k_20": [],
        "min_k_10": [],
        "zlib_ratio": [],
    }
    if reference_model is not None:
        scores["ref_ratio"] = []
        scores["min_k_pp_20"] = []

    for row in tqdm(dataset, desc="scoring"):
        text = row["text"]
        tok_lp = compute_token_log_probs(model, tokenizer, text, max_length, device)
        loss = loss_from_log_probs(tok_lp)

        # higher = more member: invert loss
        scores["loss"].append(-loss)
        scores["min_k_20"].append(min_k_percent_score(tok_lp, 20.0))
        scores["min_k_10"].append(min_k_percent_score(tok_lp, 10.0))
        scores["zlib_ratio"].append(-zlib_ratio_score(text, loss))  # invert: lower ratio = member

        if reference_model is not None:
            ref_tok_lp = compute_token_log_probs(
                reference_model, tokenizer, text, max_length, device,
            )
            ref_loss = loss_from_log_probs(ref_tok_lp)
            scores["ref_ratio"].append(reference_ratio_score(loss, ref_loss))
            scores["min_k_pp_20"].append(min_k_pp_score(tok_lp, ref_tok_lp, 20.0))

    return {k: np.array(v, dtype=np.float64) for k, v in scores.items()}


def run_tier1(cfg: dict) -> list[AttackResult]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = load_from_disk(cfg["splits_path"])

    # --- Load target model (possibly with adapter) ---
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=dtype, attn_implementation="sdpa",
    )
    if cfg.get("adapter_path"):
        model = PeftModel.from_pretrained(model, cfg["adapter_path"])
        log.info("loaded adapter %s", cfg["adapter_path"])
    model.eval().to(device)

    # --- Optional reference model for ref_ratio and min-k++ ---
    reference_model = None
    if cfg.get("reference_model"):
        log.info("loading reference model %s", cfg["reference_model"])
        reference_model = AutoModelForCausalLM.from_pretrained(
            cfg["reference_model"], torch_dtype=dtype, attn_implementation="sdpa",
        )
        reference_model.eval().to(device)

    max_length = cfg.get("max_length", 1024)

    # --- Score each split ---
    log.info("scoring members")
    member_scores = score_dataset(
        model, tokenizer, splits["members"], max_length, device, reference_model,
    )
    log.info("scoring nonmembers (same-distribution holdout)")
    nonmember_scores = score_dataset(
        model, tokenizer, splits["nonmembers"], max_length, device, reference_model,
    )
    log.info("scoring clean_nonmembers (out-of-pretraining holdout)")
    clean_scores = score_dataset(
        model, tokenizer, splits["clean_nonmembers"], max_length, device, reference_model,
    )

    # Also score canaries — they are members AND high-signal, so a strong audit metric.
    if "canaries" in splits:
        log.info("scoring canaries")
        canary_scores = score_dataset(
            model, tokenizer, splits["canaries"], max_length, device, reference_model,
        )
    else:
        canary_scores = None

    # --- Assemble results: two comparisons matter ---
    # (A) members vs. nonmembers       -> reflects BOTH pretraining + FT
    # (B) members vs. clean_nonmembers -> isolates combined leakage vs clean baseline
    # (C) canaries vs. clean_nonmembers -> DP audit lower bound
    results = []
    for attack_name in member_scores.keys():
        results.append(summarize(
            f"{attack_name}_vs_samedist_nonmem",
            member_scores[attack_name], nonmember_scores[attack_name],
        ))
        results.append(summarize(
            f"{attack_name}_vs_clean_nonmem",
            member_scores[attack_name], clean_scores[attack_name],
        ))
        if canary_scores is not None:
            results.append(summarize(
                f"{attack_name}_canaries_vs_clean",
                canary_scores[attack_name], clean_scores[attack_name],
            ))

    # --- Save per-example scores too so we can re-analyze later ---
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = {
        "members": {k: v.tolist() for k, v in member_scores.items()},
        "nonmembers": {k: v.tolist() for k, v in nonmember_scores.items()},
        "clean_nonmembers": {k: v.tolist() for k, v in clean_scores.items()},
    }
    if canary_scores is not None:
        raw["canaries"] = {k: v.tolist() for k, v in canary_scores.items()}
    with open(out_dir / "per_example_scores.json", "w") as f:
        json.dump(raw, f)

    save_results(results, out_dir / "tier1_results.json")

    # Pretty print summary
    log.info("\n=== Tier 1 summary ===")
    for r in results:
        log.info(
            "%-45s AUC=%.3f TPR@1%%=%.3f TPR@0.1%%=%.3f",
            r.name, r.auc, r.tpr_at_1pct_fpr, r.tpr_at_0p1pct_fpr,
        )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base_model_override", default=None)
    ap.add_argument("--adapter_path_override", default=None)
    ap.add_argument("--output_dir_override", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.base_model_override:
        cfg["base_model"] = args.base_model_override
    if args.adapter_path_override:
        cfg["adapter_path"] = args.adapter_path_override
    if args.output_dir_override:
        cfg["output_dir"] = args.output_dir_override

    run_tier1(cfg)


if __name__ == "__main__":
    main()

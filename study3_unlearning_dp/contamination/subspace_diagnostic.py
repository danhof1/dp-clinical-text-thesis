#!/usr/bin/env python3
"""
Subspace diagnostic: measure how paraphrased PMC projects onto ReGLU's erased directions.

Recomputes forget_dirs from BioMistral base + splits_v1_pmc at layers {16, 20, 24}.
Uses MTSamples retain set as calibration reference (score=0.0) and original PMC (score=1.0).
Reports where each paraphrase aggressiveness level falls on this normalized scale.

Threshold interpretation:
  score ~0.0  → indistinguishable from MTSamples (fully outside erased subspace)
  score ~1.0  → indistinguishable from original PMC (fully inside erased subspace)
  score <0.3  → plausibly learnable after ReGLU (outside the erasure target)

Usage:
  python subspace_diagnostic.py
  python subspace_diagnostic.py --layers 16 20 24 --n_eval 100
"""
import argparse
import json
import logging
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("subspace_diagnostic")

# ─── Paths ───────────────────────────────────────────────────────
BASE_MODEL = "/fs1/shared/model/llm/BioMistral-7B"
SPLITS = (
    "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/"
    "New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/"
    "outputs/splits_v1_pmc"
)
PARAPHRASE_DIR = (
    "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/"
    "New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/"
    "outputs/paraphrased_pmc"
)
OUT_DIR = (
    "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/"
    "New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/"
    "outputs/subspace_diagnostic"
)
CACHE_DIR = "/fs1/projects/unlearning_pretraining/Proj_code/.cache"

# ─── ReGLU parameters (must match original run) ─────────────────
BETA = 0.5
LORA_R = 16  # number of eigenvectors to extract
N_REPR_BATCHES = 10  # batches for covariance estimation
BATCH_SIZE = 4
MAX_SEQ_LENGTH = 512
SEED = 42


# ─── Hidden state extraction ────────────────────────────────────

@torch.no_grad()
def extract_hidden(model, loader, layer_id, device, n_batches=None):
    model.eval()
    parts = []
    for i, batch in enumerate(loader):
        if n_batches is not None and i >= n_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
        h = out.hidden_states[layer_id + 1]  # +1 because index 0 is embeddings
        mask = batch["attention_mask"].bool()
        parts.append(h[mask].float().cpu())
    return torch.cat(parts, dim=0)


def compute_forget_dirs(H_forget, H_retain, beta, r):
    H_F = H_forget - H_forget.mean(0, keepdim=True)
    H_R = H_retain - H_retain.mean(0, keepdim=True)

    CovF = (H_F.T @ H_F) / max(H_F.shape[0] - 1, 1)
    CovR = (H_R.T @ H_R) / max(H_R.shape[0] - 1, 1)
    CovDelta = (1 - beta) * CovF - beta * CovR

    vals, vecs = torch.linalg.eigh(CovDelta)
    forget_dirs = vecs[:, -r:].T.contiguous()  # [r, hidden_dim]

    log.info(
        "  CovDelta top eigenvalues: %s",
        [f"{v:.2f}" for v in vals[-r:].flip(0).tolist()],
    )
    return forget_dirs, vals[-r:].flip(0)


def mean_squared_projection(H, forget_dirs):
    proj = H @ forget_dirs.T  # [n_tokens, r]
    return (proj ** 2).sum(dim=1).mean().item()


# ─── Tokenize raw texts into a DataLoader ───────────────────────

def texts_to_loader(texts, tokenizer, batch_size, max_length):
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    ds = torch.utils.data.TensorDataset(
        encodings["input_ids"], encodings["attention_mask"],
    )

    def collate(batch):
        ids = torch.stack([b[0] for b in batch])
        mask = torch.stack([b[1] for b in batch])
        return {"input_ids": ids, "attention_mask": mask}

    return DataLoader(ds, batch_size=batch_size, collate_fn=collate)


# ─── Main ────────────────────────────────────────────────────────

def run(layers, n_eval):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model + tokenizer
    log.info("Loading base model %s", BASE_MODEL)
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, cache_dir=CACHE_DIR,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device).eval()

    # Load splits
    log.info("Loading splits from %s", SPLITS)
    splits = load_from_disk(SPLITS)

    def tok_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
        )

    forget_ds = splits["finetune"].map(
        tok_fn, batched=True, remove_columns=splits["finetune"].column_names,
    )
    retain_ds = splits["retain"].map(
        tok_fn, batched=True, remove_columns=splits["retain"].column_names,
    )
    forget_ds.set_format("torch")
    retain_ds.set_format("torch")

    forget_loader = DataLoader(forget_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    retain_loader = DataLoader(retain_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Load paraphrased texts
    paraphrase_dir = Path(PARAPHRASE_DIR)
    paraphrase_texts = {}
    for lvl in ["light", "moderate", "aggressive"]:
        jsonl_path = paraphrase_dir / f"paraphrased_{lvl}.jsonl"
        if not jsonl_path.exists():
            log.warning("Paraphrase file not found: %s — skipping", jsonl_path)
            continue
        records = []
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        paraphrase_texts[lvl] = records
        log.info("Loaded %d paraphrases at level=%s", len(records), lvl)

    if not paraphrase_texts:
        log.error("No paraphrase files found in %s — run paraphrase_pmc.py first", paraphrase_dir)
        return

    # Prepare eval subsets (held out from eigendecomposition)
    # eigendecomp uses first N_REPR_BATCHES * BATCH_SIZE = 40 shuffled examples
    # We use a separate slice for eval to avoid circularity
    eval_pmc_texts = [splits["finetune"][i]["text"] for i in range(100, 100 + n_eval)]
    eval_mts_texts = [splits["retain"][i]["text"] for i in range(100, 100 + n_eval)]

    eval_pmc_loader = texts_to_loader(eval_pmc_texts, tokenizer, BATCH_SIZE, MAX_SEQ_LENGTH)
    eval_mts_loader = texts_to_loader(eval_mts_texts, tokenizer, BATCH_SIZE, MAX_SEQ_LENGTH)

    # Paraphrase loaders
    para_loaders = {}
    for lvl, records in paraphrase_texts.items():
        texts = [r["paraphrase_text"] for r in records[:n_eval]]
        para_loaders[lvl] = texts_to_loader(texts, tokenizer, BATCH_SIZE, MAX_SEQ_LENGTH)

    # ── Run diagnostic at each layer ─────────────────────────────
    results = {}

    for layer_id in layers:
        log.info("=" * 60)
        log.info("Layer %d: computing forget directions", layer_id)
        log.info("=" * 60)

        # Recompute forget_dirs (deterministic with seed=42)
        torch.manual_seed(SEED)
        H_F = extract_hidden(model, forget_loader, layer_id, device, N_REPR_BATCHES)
        H_R = extract_hidden(model, retain_loader, layer_id, device, N_REPR_BATCHES)
        log.info("  Forget tokens: %d, Retain tokens: %d", len(H_F), len(H_R))

        forget_dirs, eigenvalues = compute_forget_dirs(H_F, H_R, BETA, LORA_R)
        forget_dirs_dev = forget_dirs.to(device=device, dtype=dtype)
        del H_F, H_R

        # Measure projections
        log.info("  Extracting eval hidden states...")

        H_pmc = extract_hidden(model, eval_pmc_loader, layer_id, device)
        proj_pmc = mean_squared_projection(H_pmc.to(forget_dirs.device), forget_dirs)
        del H_pmc

        H_mts = extract_hidden(model, eval_mts_loader, layer_id, device)
        proj_mts = mean_squared_projection(H_mts.to(forget_dirs.device), forget_dirs)
        del H_mts

        log.info("  Original PMC projection:  %.4f", proj_pmc)
        log.info("  MTSamples projection:     %.4f", proj_mts)

        layer_results = {
            "layer": layer_id,
            "eigenvalues_top5": eigenvalues[:5].tolist(),
            "proj_original_pmc": proj_pmc,
            "proj_mtsample_calibration": proj_mts,
            "paraphrase_levels": {},
        }

        denom = proj_pmc - proj_mts
        if abs(denom) < 1e-8:
            log.warning("  PMC and MTSamples projections nearly identical — normalization unstable")
            denom = 1.0

        for lvl, loader in para_loaders.items():
            H_para = extract_hidden(model, loader, layer_id, device)
            proj_para = mean_squared_projection(H_para.to(forget_dirs.device), forget_dirs)
            del H_para

            normalized = (proj_para - proj_mts) / denom
            pct_of_mts = (proj_para / proj_mts * 100) if proj_mts > 0 else float("inf")

            layer_results["paraphrase_levels"][lvl] = {
                "raw_projection": proj_para,
                "normalized_score": normalized,
                "pct_of_mtsample": pct_of_mts,
            }
            log.info(
                "  %-12s projection: %.4f  |  normalized: %.3f  |  %% of MTS: %.1f%%",
                lvl, proj_para, normalized, pct_of_mts,
            )

        results[f"layer_{layer_id}"] = layer_results

        # Save forget_dirs for potential reuse
        torch.save(
            {"forget_dirs": forget_dirs, "eigenvalues": eigenvalues},
            out_dir / f"forget_dirs_layer{layer_id}.pt",
        )

    # ── Summary ──────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info("Scale: MTSamples=0.0 (calibration), Original PMC=1.0 (erased subspace)")
    log.info("Threshold: score <0.3 suggests paraphrases are outside erased subspace\n")

    for layer_id in layers:
        lr = results[f"layer_{layer_id}"]
        log.info("Layer %d:", layer_id)
        log.info("  PMC (raw): %.4f  |  MTS (raw): %.4f", lr["proj_original_pmc"], lr["proj_mtsample_calibration"])
        for lvl in ["light", "moderate", "aggressive"]:
            if lvl in lr["paraphrase_levels"]:
                pl = lr["paraphrase_levels"][lvl]
                verdict = "INSIDE erased subspace" if pl["normalized_score"] > 0.5 else (
                    "BORDERLINE" if pl["normalized_score"] > 0.3 else "OUTSIDE erased subspace"
                )
                log.info(
                    "  %-12s → normalized=%.3f (%% of MTS=%.1f%%)  %s",
                    lvl, pl["normalized_score"], pl["pct_of_mtsample"], verdict,
                )
        log.info("")

    # Save results
    out_path = out_dir / "diagnostic_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", out_path)

    # Print go/no-go recommendation
    layer20 = results.get("layer_20", {})
    if layer20:
        scores = {
            lvl: layer20["paraphrase_levels"][lvl]["normalized_score"]
            for lvl in layer20.get("paraphrase_levels", {})
        }
        viable = [lvl for lvl, s in scores.items() if s < 0.3]
        if viable:
            best = min(viable, key=lambda l: scores[l])
            log.info(
                "RECOMMENDATION: %s paraphrase (score=%.3f at layer 20) is viable "
                "for the three-stage pipeline. Proceed to step 3.",
                best.upper(), scores[best],
            )
        else:
            borderline = [lvl for lvl, s in scores.items() if s < 0.5]
            if borderline:
                log.info(
                    "BORDERLINE: No level clearly outside erased subspace at layer 20. "
                    "Consider testing with a stronger shift (back-translation)."
                )
            else:
                log.info(
                    "NOT VIABLE: All paraphrase levels project heavily onto erased subspace. "
                    "ReGLU erasure appears semantically bound, not surface-form bound."
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure paraphrased PMC projection onto ReGLU's erased subspace",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[16, 20, 24],
        help="Layers to compute forget_dirs at (default: 16 20 24)",
    )
    parser.add_argument(
        "--n_eval", type=int, default=100,
        help="Number of samples per test set for projection measurement (default: 100)",
    )
    args = parser.parse_args()
    run(args.layers, args.n_eval)

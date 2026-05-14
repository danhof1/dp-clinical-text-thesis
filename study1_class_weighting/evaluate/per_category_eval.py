"""
per_category_eval.py

For each run × ICD category:
  1. Adherence: score 50 sampled synthetic notes with Asclepius judge (0-1 scale)
  2. MAUVE: compute per-category MAUVE using Asclepius last-token embeddings

Reuses real note embeddings across all runs — computed once from data0_base.

Outputs:
  eval/per_category_adherence.csv  — run_id × category → n, mean_adherence, match_rate
  eval/per_category_mauve.csv      — run_id × category → mauve_score, n_syn, n_real

Usage:
    python per_category_eval.py [--runs run1,run2] [--adh_per_cat 50]
"""

from __future__ import annotations
import argparse, csv, json, math, os, random, re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

PROJ      = "/fs1/projects/unlearning_pretraining/Proj_code"
GEN_ROOT  = f"{PROJ}/generated/mimic"
EVAL_DIR  = f"{PROJ}/eval"
JUDGE     = "/fs1/shared/model/llm/Asclepius-Llama3-8B"
REAL_SRC  = "data0_base"   # use original_bhc from this run as real reference
SEED      = 42
random.seed(SEED)
np.random.seed(SEED)

# ── data loading ──────────────────────────────────────────────────────────────

def load_run(run_name: str) -> dict[str, list[dict]]:
    """Return {category: [row, ...]} for a run."""
    path = os.path.join(GEN_ROOT, run_name, "synthetic_bhc.jsonl")
    cats: dict[str, list] = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            cat = r.get("control_codes", {}).get("icd_category", "Unknown")
            cats[cat].append(r)
    return dict(cats)


def discover_runs(requested: list[str] | None) -> list[str]:
    all_runs = sorted(
        d for d in os.listdir(GEN_ROOT)
        if os.path.isfile(os.path.join(GEN_ROOT, d, "synthetic_bhc.jsonl"))
    )
    if requested:
        return [r for r in all_runs if r in requested]
    return all_runs


# ── Asclepius utilities ───────────────────────────────────────────────────────

class ForceDigit(LogitsProcessor):
    def __init__(self, ids):
        self.ids = ids
    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        for i in self.ids: mask[:, i] = 0.0
        return scores + mask


def get_digit_ids(tok):
    ids = []
    for d in "12345":
        for v in (d, f" {d}"):
            t = tok.encode(v, add_special_tokens=False)
            if len(t) == 1: ids.append(t[0])
    return list(set(ids))


@torch.no_grad()
def score_adherence_batch(texts: list[str], cats: list[str],
                          model, tok, force_proc, snippet_tokens=300) -> list[float]:
    """Score one note at a time (generate is hard to batch); returns 0-1 scores."""
    results = []
    for text, cat in zip(texts, cats):
        toks   = tok.encode(text, add_special_tokens=False)
        snip   = tok.decode(toks[:snippet_tokens], skip_special_tokens=True)
        prompt = (
            f"ICD-10 category: {cat}\n\n"
            "Does the following hospital course note reflect the above ICD-10 category?\n"
            "1 = no match at all  2 = weak  3 = moderate  4 = strong  5 = clearly matches\n"
            "Reply with a single digit (1-5) only.\n\n"
            f"Note:\n{snip}\n\nScore:"
        )
        inp = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inp, max_new_tokens=1, do_sample=False,
            pad_token_id=tok.eos_token_id,
            logits_processor=[force_proc],
        )
        raw = tok.decode(out[0][inp["input_ids"].shape[-1]:],
                         skip_special_tokens=True).strip()
        score = -1.0
        for ch in raw:
            if ch in "12345":
                score = (int(ch) - 1) / 4.0
                break
        results.append(score)
    return results


@torch.no_grad()
def embed_texts(texts: list[str], model, tok, max_length=512,
                batch_size=16) -> np.ndarray:
    """Last-token pooled embeddings from Asclepius hidden states."""
    vecs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc   = tok(chunk, return_tensors="pt", truncation=True,
                    max_length=max_length, padding=True).to(model.device)
        out   = model(**enc, output_hidden_states=True)
        h     = out.hidden_states[-1]                  # [B, T, H]
        lens  = enc["attention_mask"].sum(dim=1) - 1   # last real token idx
        for j, l in enumerate(lens):
            vecs.append(h[j, l].cpu().float().numpy())
    return np.stack(vecs)


def compute_mauve_from_features(p_feats: np.ndarray,
                                q_feats: np.ndarray,
                                device_id: int = 0) -> float | None:
    try:
        import mauve as mauve_lib
        result = mauve_lib.compute_mauve(
            p_features=p_feats, q_features=q_feats,
            device_id=device_id, verbose=False,
            num_buckets=min(500, len(p_feats) // 2),
        )
        return float(result.mauve)
    except Exception as e:
        print(f"    MAUVE error: {e}")
        return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None,
                    help="Comma-separated subset of run dirs (default: all)")
    ap.add_argument("--adh_per_cat", type=int, default=50,
                    help="Notes to score per category for adherence")
    ap.add_argument("--min_mauve_n", type=int, default=30,
                    help="Min samples per category to compute MAUVE")
    ap.add_argument("--skip_adherence", action="store_true")
    ap.add_argument("--skip_mauve", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = discover_runs(args.runs.split(",") if args.runs else None)
    print(f"Evaluating {len(runs)} runs:", runs)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading Asclepius from {JUDGE}...")
    tok = AutoTokenizer.from_pretrained(JUDGE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="auto",
    )
    model.eval()
    force_proc = ForceDigit(get_digit_ids(tok))
    print("  Loaded.\n")

    # ── Real embeddings: compute once from data0_base original_bhc ────────────
    real_embeds_by_cat: dict[str, np.ndarray] = {}
    if not args.skip_mauve:
        print(f"Computing real embeddings from {REAL_SRC}...")
        real_data = load_run(REAL_SRC)
        for cat, rows in tqdm(real_data.items(), desc="real cats"):
            real_texts = [r["original_bhc"] for r in rows if r.get("original_bhc")]
            if len(real_texts) < args.min_mauve_n:
                continue
            real_embeds_by_cat[cat] = embed_texts(real_texts, model, tok)
        print(f"  Cached embeddings for {len(real_embeds_by_cat)} categories.\n")

    # ── Per-run evaluation ────────────────────────────────────────────────────
    adh_rows:  list[dict] = []
    mauve_rows: list[dict] = []

    for run_name in runs:
        print(f"\n{'═'*70}")
        print(f"  Run: {run_name}")
        run_data = load_run(run_name)

        # ── Adherence ─────────────────────────────────────────────────────────
        if not args.skip_adherence:
            print(f"  Adherence scoring ({args.adh_per_cat}/cat)...")
            for cat, rows in sorted(run_data.items()):
                sample = random.sample(rows, min(args.adh_per_cat, len(rows)))
                texts  = [r.get("synthetic_bhc", "").strip() for r in sample]
                cats   = [cat] * len(texts)
                # filter too-short notes
                valid  = [(t, c) for t, c in zip(texts, cats)
                          if len(tok.encode(t, add_special_tokens=False)) >= 10]
                if not valid:
                    continue
                vt, vc = zip(*valid)
                scores = score_adherence_batch(list(vt), list(vc), model, tok, force_proc)
                valid_scores = [s for s in scores if s >= 0]
                if not valid_scores:
                    continue
                adh_rows.append({
                    "run_id":       run_name,
                    "category":     cat,
                    "n_scored":     len(valid_scores),
                    "mean_adherence": round(np.mean(valid_scores), 4),
                    "match_rate":   round(sum(1 for s in valid_scores if s >= 0.6)
                                          / len(valid_scores), 4),
                })
                print(f"    {cat[:45]:<45}  n={len(valid_scores):3d}  "
                      f"adh={adh_rows[-1]['mean_adherence']:.3f}  "
                      f"match={adh_rows[-1]['match_rate']:.2f}")

        # ── MAUVE ─────────────────────────────────────────────────────────────
        if not args.skip_mauve:
            print(f"  MAUVE per category...")
            for cat, rows in sorted(run_data.items()):
                if cat not in real_embeds_by_cat:
                    continue
                syn_texts = [r.get("synthetic_bhc", "").strip() for r in rows
                             if r.get("synthetic_bhc")]
                if len(syn_texts) < args.min_mauve_n:
                    print(f"    SKIP {cat[:40]} (n={len(syn_texts)} < {args.min_mauve_n})")
                    continue
                syn_feats  = embed_texts(syn_texts, model, tok)
                real_feats = real_embeds_by_cat[cat]
                mauve_score = compute_mauve_from_features(real_feats, syn_feats)
                print(f"    {cat[:45]:<45}  n_syn={len(syn_texts):4d}  "
                      f"n_real={len(real_feats):4d}  mauve={mauve_score:.4f}" if mauve_score else
                      f"    {cat[:45]:<45}  FAILED")
                mauve_rows.append({
                    "run_id":      run_name,
                    "category":    cat,
                    "mauve_score": mauve_score,
                    "n_synthetic": len(syn_texts),
                    "n_real":      len(real_feats),
                })

    # ── Write outputs ─────────────────────────────────────────────────────────
    os.makedirs(EVAL_DIR, exist_ok=True)

    if adh_rows:
        out = f"{EVAL_DIR}/per_category_adherence.csv"
        fields = ["run_id", "category", "n_scored", "mean_adherence", "match_rate"]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(adh_rows)
        print(f"\nWrote {len(adh_rows)} rows → {out}")

    if mauve_rows:
        out = f"{EVAL_DIR}/per_category_mauve.csv"
        fields = ["run_id", "category", "mauve_score", "n_synthetic", "n_real"]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(mauve_rows)
        print(f"Wrote {len(mauve_rows)} rows → {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()

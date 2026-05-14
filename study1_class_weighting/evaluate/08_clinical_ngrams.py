"""
08_clinical_ngrams.py — Clinical N-gram Term Overlap (Track A, MIMIC)

For each synthetic run:
  1. Extracts top-K clinical unigrams and bigrams from real BHC notes
  2. Computes the fraction of those n-grams present in synthetic notes
  3. Computes KL divergence between real and synthetic n-gram frequency distributions
  4. Reports coverage and distributional similarity per run

Output: eval/clinical_ngram_results.csv
  run_id, topk_unigram_coverage, topk_bigram_coverage,
  kl_unigram, kl_bigram, n_syn, n_real

Usage:
    python Scripts_2.0/08_clinical_ngrams.py [--runs run1,run2] [--topk 500]
"""

from __future__ import annotations
import argparse, csv, json, math, re
from collections import Counter
from pathlib import Path

import numpy as np

PROJ      = "/fs1/projects/unlearning_pretraining/Proj_code"
GEN_ROOT  = f"{PROJ}/generated/mimic"
# 2026-04-28: Switched from data0_base/original_bhc (9,178 stratified subset) to
# full train.jsonl (284,702 records). Reason: gold-standard reference should be
# the complete ground-truth corpus, not a distribution-matched subset.
# Previous reference inflated rare-category bigrams into top-K vocabulary.
REAL_SRC  = f"{PROJ}/data/train.jsonl"
EVAL_DIR  = f"{PROJ}/eval"

STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "was", "were",
    "for", "on", "at", "by", "with", "he", "she", "his", "her", "this",
    "that", "it", "be", "as", "are", "been", "have", "has", "had",
    "but", "from", "they", "we", "their", "will", "also", "no", "not",
    "patient", "pt", "who", "which", "than", "then", "s", "x", "per",
    "history", "hx", "noted", "denies", "reports", "year", "old", "male",
    "female", "man", "woman", "yo", "day", "days", "including", "given",
    "due", "well", "mg", "dose", "daily"
}


def tokenize(text):
    tokens = re.findall(r"\b[a-z][a-z\-]*[a-z]\b", text.lower())
    return [t for t in tokens if t not in STOP and len(t) >= 3]


def get_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def build_corpus_counter(texts, n):
    counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        counter.update(get_ngrams(tokens, n))
    return counter


def kl_divergence(p_counter, q_counter, topk):
    vocab = [w for w, _ in p_counter.most_common(topk)]
    p_total = sum(p_counter[w] for w in vocab) + 1e-10
    q_total = sum(q_counter.get(w, 0) for w in vocab) + 1e-10
    kl = 0.0
    for w in vocab:
        p_i = (p_counter[w] + 1e-9) / p_total
        q_i = (q_counter.get(w, 0) + 1e-9) / q_total
        kl += p_i * math.log(p_i / q_i)
    return kl


def coverage(real_counter, syn_counter, topk):
    top_real = {w for w, _ in real_counter.most_common(topk)}
    present = sum(1 for w in top_real if syn_counter.get(w, 0) > 0)
    return present / len(top_real) if top_real else 0.0


def load_real_texts():
    texts = []
    with open(REAL_SRC) as f:
        for line in f:
            r = json.loads(line)
            t = (r.get("bhc_text") or "").strip()
            if t:
                texts.append(t)
    return texts


def load_synthetic_texts(run_name):
    path = Path(GEN_ROOT) / run_name / "synthetic_bhc.jsonl"
    texts = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            t = r.get("synthetic_bhc", "").strip()
            if t:
                texts.append(t)
    return texts


def discover_runs(requested=None):
    all_runs = sorted(
        d.name for d in Path(GEN_ROOT).iterdir()
        if (Path(GEN_ROOT) / d.name / "synthetic_bhc.jsonl").exists()
    )
    if requested:
        return [r for r in all_runs if r in requested]
    return all_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None)
    ap.add_argument("--topk", type=int, default=500)
    args = ap.parse_args()

    runs = discover_runs(args.runs.split(",") if args.runs else None)
    print(f"Runs: {runs}")
    print(f"Top-K: {args.topk}\n")

    print("Loading real notes...")
    real_texts = load_real_texts()
    print(f"  {len(real_texts)} real notes loaded")

    print("Building real n-gram distributions...")
    real_uni = build_corpus_counter(real_texts, 1)
    real_bi  = build_corpus_counter(real_texts, 2)
    print(f"  unigram vocab: {len(real_uni):,}  bigram vocab: {len(real_bi):,}")

    results = []
    for run in runs:
        print(f"\n[{run}]")
        try:
            syn_texts = load_synthetic_texts(run)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  {len(syn_texts)} synthetic notes")
        syn_uni = build_corpus_counter(syn_texts, 1)
        syn_bi  = build_corpus_counter(syn_texts, 2)

        cov_uni = coverage(real_uni, syn_uni, args.topk)
        cov_bi  = coverage(real_bi,  syn_bi,  args.topk)
        kl_uni  = kl_divergence(real_uni, syn_uni, args.topk)
        kl_bi   = kl_divergence(real_bi,  syn_bi,  args.topk)

        print(f"  unigram coverage={cov_uni:.3f}  KL={kl_uni:.4f}")
        print(f"  bigram  coverage={cov_bi:.3f}  KL={kl_bi:.4f}")

        results.append({
            "run_id":                run,
            "n_syn":                 len(syn_texts),
            "n_real":                len(real_texts),
            "topk_unigram_coverage": round(cov_uni, 4),
            "topk_bigram_coverage":  round(cov_bi,  4),
            "kl_unigram":            round(kl_uni,  4),
            "kl_bigram":             round(kl_bi,   4),
        })

    out = f"{EVAL_DIR}/clinical_ngram_results.csv"
    Path(EVAL_DIR).mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

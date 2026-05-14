"""
03.3_rerank_candidates.py — Offline re-ranking of saved generation candidates.

# 2026-04-28: Created to enable term-aware candidate selection without regeneration.
# Reason: clinical bigram coverage (ng_bi) is our strongest quality signal, but the
# PPL-only quality maximiser selects for fluency over clinical specificity. This script
# re-ranks saved candidates using a combined score:
#     score = alpha * (1 - norm_ppl) + (1 - alpha) * term_overlap
# where term_overlap measures coverage of category-specific clinical terms.
#
# Requires generation with --save_candidates flag (03.2_generate.py).

Inputs:
    - synthetic_bhc.jsonl with 'candidates' field (from --save_candidates)
    - Reference term lists (built from data0_base per ICD category)

Outputs:
    - New synthetic_bhc.jsonl with re-ranked selections
    - rerank_stats.json with before/after comparison

Usage:
    python 03.3_rerank_candidates.py \\
        --input generated/mimic/eps4_power03cap10_mimic_cands/synthetic_bhc.jsonl \\
        --output generated/mimic/eps4_power03cap10_mimic_termfilt/ \\
        --alpha 0.5

    # Alpha controls the PPL vs term-overlap tradeoff:
    #   alpha=1.0 → pure PPL (same as original)
    #   alpha=0.0 → pure term overlap
    #   alpha=0.5 → equal weight (default)
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJ = "/fs1/projects/unlearning_pretraining/Proj_code"
GEN_ROOT = f"{PROJ}/generated/mimic"
REAL_SRC = f"{PROJ}/data/train.jsonl"

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


def get_bigrams(tokens):
    return [tuple(tokens[i:i+2]) for i in range(len(tokens) - 1)]


def build_category_terms(topk=200):
    """
    Build per-ICD-category reference term sets from real data.
    Uses data0_base original_bhc as the reference (same as 08_clinical_ngrams.py).
    Returns {category: set of top-K bigrams}.
    """
    print(f"Building per-category reference terms from {REAL_SRC}...")
    cat_counters = defaultdict(Counter)

    with open(REAL_SRC) as f:
        for line in f:
            r = json.loads(line)
            cat = r.get("control_codes", {}).get("icd_category", "Unknown")
            text = r.get("bhc_text", "")
            if text:
                tokens = tokenize(text)
                bigrams = get_bigrams(tokens)
                cat_counters[cat].update(bigrams)

    cat_terms = {}
    for cat, counter in cat_counters.items():
        cat_terms[cat] = set(bg for bg, _ in counter.most_common(topk))

    print(f"  Built term sets for {len(cat_terms)} categories ({topk} bigrams each)")
    return cat_terms


def term_overlap_score(text, reference_terms):
    """
    Fraction of reference bigrams present in the candidate text.
    Returns 0-1 (higher = more clinical terms present).
    """
    if not reference_terms:
        return 0.0
    tokens = tokenize(text)
    candidate_bigrams = set(get_bigrams(tokens))
    overlap = candidate_bigrams & reference_terms
    return len(overlap) / len(reference_terms)


def combined_score(ppl, term_score, alpha, ppl_min, ppl_max):
    """
    Combined ranking score. Higher = better candidate.
    alpha=1 → pure PPL (lower is better, so we invert).
    alpha=0 → pure term overlap.
    """
    if ppl_max == ppl_min:
        norm_ppl = 0.5
    else:
        norm_ppl = 1.0 - (ppl - ppl_min) / (ppl_max - ppl_min)
    norm_ppl = max(0.0, min(1.0, norm_ppl))
    return alpha * norm_ppl + (1.0 - alpha) * term_score


def main():
    ap = argparse.ArgumentParser(
        description="Re-rank generation candidates using term-aware scoring"
    )
    ap.add_argument("--input", required=True,
                    help="Input synthetic_bhc.jsonl (must have 'candidates' field)")
    ap.add_argument("--output", required=True,
                    help="Output directory for re-ranked results")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Weight for PPL component (0=pure terms, 1=pure PPL)")
    ap.add_argument("--topk_terms", type=int, default=200,
                    help="Top-K bigrams per category for reference")
    args = ap.parse_args()

    # Load input
    print(f"Loading candidates from {args.input}")
    records = []
    with open(args.input) as f:
        for line in f:
            r = json.loads(line.strip())
            if "candidates" not in r:
                print(f"ERROR: Record {r.get('note_id', '?')} has no 'candidates' field.")
                print("       Generation must be run with --save_candidates flag.")
                sys.exit(1)
            records.append(r)
    print(f"  Loaded {len(records)} records with {len(records[0]['candidates'])} candidates each")

    # Build reference terms
    cat_terms = build_category_terms(topk=args.topk_terms)

    # Re-rank
    print(f"Re-ranking with alpha={args.alpha} (0=pure terms, 1=pure PPL)...")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "synthetic_bhc.jsonl"

    stats = {
        "n_records": len(records),
        "alpha": args.alpha,
        "topk_terms": args.topk_terms,
        "selection_changed": 0,
        "ppl_before": [],
        "ppl_after": [],
        "term_score_before": [],
        "term_score_after": [],
    }

    with open(output_path, "w") as out_f:
        for rec in records:
            cat = rec.get("control_codes", {}).get("icd_category", "Unknown")
            ref_terms = cat_terms.get(cat, set())

            candidates = rec["candidates"]
            ppls = [c["ppl"] for c in candidates]
            ppl_min = min(p for p in ppls if p < float("inf"))
            ppl_max = max(p for p in ppls if p < float("inf"))

            # Score each candidate
            scored = []
            for c in candidates:
                ts = term_overlap_score(c["text"], ref_terms)
                cs = combined_score(c["ppl"], ts, args.alpha, ppl_min, ppl_max)
                scored.append((c["text"], c["ppl"], ts, cs))

            # Original winner (lowest PPL = first in sorted list)
            original_winner = candidates[0]["text"]
            original_ppl = candidates[0]["ppl"]
            original_ts = term_overlap_score(original_winner, ref_terms)

            # New winner (highest combined score)
            scored.sort(key=lambda x: -x[3])
            new_winner, new_ppl, new_ts, new_cs = scored[0]

            changed = (new_winner != original_winner)
            if changed:
                stats["selection_changed"] += 1

            stats["ppl_before"].append(original_ppl)
            stats["ppl_after"].append(new_ppl)
            stats["term_score_before"].append(original_ts)
            stats["term_score_after"].append(new_ts)

            # Write result (same schema as original, drop candidates for size)
            result = {
                "note_id": rec["note_id"],
                "control_codes": rec["control_codes"],
                "prefix": rec["prefix"],
                "synthetic_bhc": new_winner,
                "perplexity": new_ppl,
                "passed_filter": True,
                "original_bhc": rec.get("original_bhc", ""),
                "n_candidates": len(candidates),
            }
            out_f.write(json.dumps(result) + "\n")

    # Stats
    import numpy as np
    ppl_b = [p for p in stats["ppl_before"] if p < float("inf")]
    ppl_a = [p for p in stats["ppl_after"] if p < float("inf")]
    ts_b = stats["term_score_before"]
    ts_a = stats["term_score_after"]

    summary = {
        "n_records": stats["n_records"],
        "alpha": args.alpha,
        "topk_terms": args.topk_terms,
        "selection_changed": stats["selection_changed"],
        "change_rate": stats["selection_changed"] / stats["n_records"],
        "ppl_before_mean": float(np.mean(ppl_b)),
        "ppl_after_mean": float(np.mean(ppl_a)),
        "term_score_before_mean": float(np.mean(ts_b)),
        "term_score_after_mean": float(np.mean(ts_a)),
        "term_score_lift": float(np.mean(ts_a)) - float(np.mean(ts_b)),
    }

    stats_path = output_dir / "rerank_stats.json"
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults:")
    print(f"  Selection changed: {summary['selection_changed']}/{summary['n_records']} "
          f"({summary['change_rate']:.1%})")
    print(f"  PPL:  {summary['ppl_before_mean']:.2f} -> {summary['ppl_after_mean']:.2f}")
    print(f"  Term: {summary['term_score_before_mean']:.4f} -> {summary['term_score_after_mean']:.4f} "
          f"(+{summary['term_score_lift']:.4f})")
    print(f"\nWrote {stats['n_records']} records -> {output_path}")
    print(f"Stats -> {stats_path}")


if __name__ == "__main__":
    main()

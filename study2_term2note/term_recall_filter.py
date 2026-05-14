"""
term_recall_filter.py — Post-generation filter based on conditioning term recall.

Works on both Track A (ICD-category reference terms) and T2N (section SNOMED terms).

For T2N: each section has explicit conditioning terms. Measures what fraction
of those terms appear in the generated text. Notes with mean section term recall
below threshold are dropped.

For Track A: builds per-ICD-category reference terms from training data.
Measures what fraction of category reference bigrams appear in the generated note.
Notes below threshold are dropped.

Usage:
    # T2N mode (uses conditioning terms from sections)
    python term_recall_filter.py \
        --input outputs/generated/term2note_v3_eps4/synthetic_term2note.jsonl \
        --output outputs/generated/term2note_v3_eps4/synthetic_term2note_filtered.jsonl \
        --mode t2n --threshold 0.1

    # Track A mode (uses ICD-category reference bigrams)
    python term_recall_filter.py \
        --input generated/mimic/eps4_power03cap10/synthetic_bhc.jsonl \
        --output generated/mimic/eps4_power03cap10_filtered/synthetic_bhc.jsonl \
        --mode track_a --threshold 0.02 \
        --real_data data/train.jsonl
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

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


# ── T2N mode: term recall against conditioning SNOMED terms ──

def t2n_section_term_recall(section):
    """Fraction of conditioning terms present in section text."""
    terms_str = section.get("terms", "none")
    text = section.get("text", "")
    if not terms_str or terms_str == "none" or not text:
        return None
    terms = [t.strip().lower() for t in terms_str.split(",") if t.strip()]
    if not terms:
        return None
    text_lower = text.lower()
    hits = sum(1 for t in terms if t in text_lower)
    return hits / len(terms)


def t2n_note_term_recall(record):
    """Mean term recall across sections with conditioning terms."""
    sections = record.get("sections", [])
    recalls = []
    for sec in sections:
        tr = t2n_section_term_recall(sec)
        if tr is not None:
            recalls.append(tr)
    return sum(recalls) / len(recalls) if recalls else 0.0


# ── Track A mode: term recall against ICD-category reference bigrams ──

def build_category_terms(real_path, topk=200):
    cat_counters = defaultdict(Counter)
    with open(real_path) as f:
        for line in f:
            r = json.loads(line)
            cat = r.get("control_codes", {}).get("icd_category", "Unknown")
            text = r.get("bhc_text", "")
            if text:
                tokens = tokenize(text)
                cat_counters[cat].update(get_bigrams(tokens))
    cat_terms = {}
    for cat, counter in cat_counters.items():
        cat_terms[cat] = set(bg for bg, _ in counter.most_common(topk))
    return cat_terms


def track_a_note_term_recall(record, cat_terms):
    cat = record.get("control_codes", {}).get("icd_category", "Unknown")
    ref = cat_terms.get(cat, set())
    if not ref:
        return 0.0
    text = record.get("synthetic_bhc", "")
    tokens = tokenize(text)
    note_bigrams = set(get_bigrams(tokens))
    return len(note_bigrams & ref) / len(ref)


def main():
    ap = argparse.ArgumentParser(description="Term recall filter for generated notes")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", required=True, choices=["t2n", "track_a"])
    ap.add_argument("--threshold", type=float, default=0.1,
                    help="Min term recall to keep (default 0.1 for T2N, 0.02 for Track A)")
    ap.add_argument("--real_data", default=None,
                    help="Path to train.jsonl (required for track_a mode)")
    ap.add_argument("--topk", type=int, default=200)
    args = ap.parse_args()

    if args.mode == "track_a" and not args.real_data:
        print("ERROR: --real_data required for track_a mode")
        sys.exit(1)

    cat_terms = None
    if args.mode == "track_a":
        print(f"Building category terms from {args.real_data}...")
        cat_terms = build_category_terms(args.real_data, topk=args.topk)
        print(f"  {len(cat_terms)} categories")

    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    print(f"Loaded {len(records)} records from {args.input}")

    recalls = []
    for r in records:
        if args.mode == "t2n":
            tr = t2n_note_term_recall(r)
        else:
            tr = track_a_note_term_recall(r, cat_terms)
        recalls.append(tr)

    kept = []
    dropped = []
    for r, tr in zip(records, recalls):
        if tr >= args.threshold:
            kept.append((r, tr))
        else:
            dropped.append((r, tr))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r, tr in kept:
            f.write(json.dumps(r) + "\n")

    import numpy as np
    all_tr = np.array(recalls)
    kept_tr = np.array([tr for _, tr in kept])
    dropped_tr = np.array([tr for _, tr in dropped])

    stats = {
        "mode": args.mode,
        "threshold": args.threshold,
        "total": len(records),
        "kept": len(kept),
        "dropped": len(dropped),
        "drop_rate": len(dropped) / len(records) if records else 0,
        "mean_recall_all": float(np.mean(all_tr)),
        "mean_recall_kept": float(np.mean(kept_tr)) if len(kept_tr) else 0,
        "mean_recall_dropped": float(np.mean(dropped_tr)) if len(dropped_tr) else 0,
        "median_recall_all": float(np.median(all_tr)),
        "p10_recall": float(np.percentile(all_tr, 10)),
        "p25_recall": float(np.percentile(all_tr, 25)),
    }

    stats_path = args.output.replace(".jsonl", "_filter_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nResults:")
    print(f"  Kept: {stats['kept']}/{stats['total']} ({1-stats['drop_rate']:.1%})")
    print(f"  Dropped: {stats['dropped']} ({stats['drop_rate']:.1%})")
    print(f"  Mean recall (all): {stats['mean_recall_all']:.4f}")
    print(f"  Mean recall (kept): {stats['mean_recall_kept']:.4f}")
    print(f"  P10/P25/Median: {stats['p10_recall']:.4f} / {stats['p25_recall']:.4f} / {stats['median_recall_all']:.4f}")
    print(f"\nWrote {stats['kept']} records -> {args.output}")
    print(f"Stats -> {stats_path}")


if __name__ == "__main__":
    main()

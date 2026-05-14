"""
sample_generations.py
Samples synthetic BHC notes across generation conditions for qualitative review.
For each condition: prints per-category counts, then 1 note per ICD tier.
"""

import json
import random
import sys
import os
from collections import defaultdict, Counter

PROJ = "/fs1/projects/unlearning_pretraining/Proj_code/generated"
SEED = 42
random.seed(SEED)

CONDITIONS = [
    ("data0_base",           "BASE (no DP, no fine-tune)"),
    ("data2_eps4_weighted",  "INV-FREQ ε=4  [COLLAPSED]"),
    ("eps4",                 "UNWEIGHTED ε=4"),
    ("eps4_sqrt10_mimic",    "SQRT-CAP10 ε=4"),
    ("eps_inf",              "UNWEIGHTED ε=∞"),
    ("epsinf_sqrt10_mimic",  "SQRT-CAP10 ε=∞"),
]

def load(dirname):
    path = os.path.join(PROJ, dirname, "synthetic_bhc.jsonl")
    records = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            cat = r["control_codes"]["icd_category"]
            records[cat].append(r)
    return records

def tier(n):
    if n > 10000: return "MAJORITY"
    if n >= 500:  return "MEDIUM"
    return "MINORITY"

def sample_condition(label, records):
    counts = {c: len(v) for c, v in records.items()}
    total  = sum(counts.values())

    print(f"\n{'#'*72}")
    print(f"  CONDITION: {label}   ({total:,} records, {len(counts)} categories)")
    print(f"{'#'*72}")

    # Per-category count table
    print(f"\n  {'Category':<55} {'n':>5}  tier")
    print(f"  {'-'*55}  {'-'*5}  {'-'*8}")
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<55} {n:>5}  {tier(n)}")

    # Sample 1 note from each tier
    by_tier = defaultdict(list)
    for cat, recs in records.items():
        t = tier(counts[cat])
        for r in recs:
            by_tier[t].append((cat, counts[cat], r))

    for t in ["MAJORITY", "MEDIUM", "MINORITY"]:
        pool = by_tier[t]
        if not pool:
            print(f"\n  [No {t} records]")
            continue
        cat, n, r = random.choice(pool)
        requested = r["control_codes"]["icd_category"]
        synth     = r.get("synthetic_bhc", "").strip()
        ppl       = r.get("perplexity", float("nan"))
        print(f"\n{'─'*72}")
        print(f"  TIER: {t}  |  Requested: {requested}  (n={n:,})")
        print(f"  PPL: {ppl:.2f}")
        print(f"{'─'*72}")
        print(synth)

for dirname, label in CONDITIONS:
    try:
        records = load(dirname)
        sample_condition(label, records)
    except FileNotFoundError:
        print(f"\n[SKIP] {label}: file not found")

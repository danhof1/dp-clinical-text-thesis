"""
11_distinct_n.py — Intra-corpus diversity (distinct-n)

For each generation run, computes:
  distinct-1 through distinct-4: unique n-grams / total n-grams
  self-BLEU-4 (avg pairwise BLEU on 500 random pairs)

Output: eval/distinct_n_results.csv
"""
from __future__ import annotations
import argparse, csv, json, random
from collections import Counter
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

PROJ     = "/fs1/projects/unlearning_pretraining/Proj_code"
GEN_ROOT = f"{PROJ}/generated/mimic"
EVAL_DIR = f"{PROJ}/eval"
SEED     = 42
random.seed(SEED)


def load_texts(run_name):
    path = Path(GEN_ROOT) / run_name / "synthetic_bhc.jsonl"
    texts = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            t = r.get("synthetic_bhc", "").strip()
            if t:
                texts.append(t)
    return texts


def distinct_n(texts, n):
    total_ngrams = Counter()
    for t in texts:
        tokens = t.lower().split()
        for i in range(len(tokens) - n + 1):
            total_ngrams[tuple(tokens[i:i+n])] += 1
    total_count = sum(total_ngrams.values())
    if total_count == 0:
        return 0.0
    return len(total_ngrams) / total_count


def self_bleu(texts, n_pairs=500):
    if len(texts) < 2:
        return 0.0
    smoothie = SmoothingFunction().method1
    indices = list(range(len(texts)))
    pairs = []
    for _ in range(n_pairs):
        i, j = random.sample(indices, 2)
        pairs.append((i, j))
    scores = []
    for i, j in pairs:
        ref = texts[i].lower().split()
        hyp = texts[j].lower().split()
        if len(hyp) < 4 or len(ref) < 4:
            continue
        s = sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25),
                          smoothing_function=smoothie)
        scores.append(s)
    return sum(scores) / len(scores) if scores else 0.0


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
    ap.add_argument("--runs", default=None,
                    help="Comma-separated run names (default: all)")
    ap.add_argument("--n_pairs", type=int, default=500,
                    help="Number of pairs for self-BLEU")
    args = ap.parse_args()

    runs = discover_runs(args.runs.split(",") if args.runs else None)
    print(f"Evaluating {len(runs)} runs for diversity metrics")

    results = []
    for run_name in runs:
        print(f"  {run_name}...", end=" ", flush=True)
        texts = load_texts(run_name)
        if not texts:
            print("EMPTY - skipping")
            continue
        d1 = distinct_n(texts, 1)
        d2 = distinct_n(texts, 2)
        d3 = distinct_n(texts, 3)
        d4 = distinct_n(texts, 4)
        sb = self_bleu(texts, n_pairs=args.n_pairs)
        results.append({
            "run_id": run_name,
            "n_notes": len(texts),
            "distinct_1": round(d1, 5),
            "distinct_2": round(d2, 5),
            "distinct_3": round(d3, 5),
            "distinct_4": round(d4, 5),
            "self_bleu_4": round(sb, 5),
        })
        print(f"d1={d1:.4f} d2={d2:.4f} d4={d4:.4f} sbleu={sb:.4f}")

    out = f"{EVAL_DIR}/distinct_n_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {len(results)} rows -> {out}")


if __name__ == "__main__":
    main()

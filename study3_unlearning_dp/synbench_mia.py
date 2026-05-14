"""
SynBench-style Membership Inference Attack on synthetic text.

Implements the ΔP methodology from:
  "SynBench: A Benchmark for Differentially Private Text Generation" (Sun et al.)

Pipeline:
  1. Train n-gram LM on synthetic data
  2. Train reference n-gram LM on auxiliary (non-member) data
  3. For each target record, compute ΔP = P_synth(x) - P_ref(x)
  4. Use ΔP as membership score → compute AUC

Usage:
  python synbench_mia.py \
    --synthetic_path outputs/generated/bio_pmc_eps8.jsonl \
    --splits_path outputs/splits_v1_pmc \
    --output_path outputs/attacks/synbench_bio_pmc_eps8.json \
    --n_gram 3
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("synbench_mia")


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class NgramLM:
    def __init__(self, n: int = 3, delta: float = 0.5):
        self.n = n
        self.delta = delta
        self.ngram_counts: dict[tuple, Counter] = defaultdict(Counter)
        self.context_totals: dict[tuple, int] = defaultdict(int)
        self.vocab: set[str] = set()

    def train(self, documents: list[str]):
        for doc in documents:
            tokens = tokenize(doc)
            self.vocab.update(tokens)
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i : i + self.n - 1])
                word = tokens[i + self.n - 1]
                self.ngram_counts[context][word] += 1
                self.context_totals[context] += 1
        log.info(
            "trained %d-gram LM: %d contexts, vocab=%d",
            self.n, len(self.context_totals), len(self.vocab),
        )

    def log_prob(self, text: str) -> float:
        tokens = tokenize(text)
        if len(tokens) < self.n:
            return float("-inf")
        total_lp = 0.0
        count = 0
        V = max(len(self.vocab), 1)
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i : i + self.n - 1])
            word = tokens[i + self.n - 1]
            num = self.ngram_counts[context][word] + self.delta
            denom = self.context_totals[context] + self.delta * V
            total_lp += math.log(num / denom)
            count += 1
        return total_lp / max(count, 1)


def load_jsonl_texts(path: Path, text_key: str = "text", max_records: int = 0) -> list[str]:
    texts = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            t = rec.get(text_key, "")
            if t:
                texts.append(t)
            if max_records and len(texts) >= max_records:
                break
    return texts


def load_split(splits_path: Path, split_name: str, max_records: int = 0) -> list[str]:
    split_dir = splits_path / split_name
    texts = []
    for jsonl in sorted(split_dir.glob("*.jsonl")):
        with open(jsonl) as f:
            for line in f:
                rec = json.loads(line)
                t = rec.get("text", rec.get("note", rec.get("content", "")))
                if t:
                    texts.append(t)
                if max_records and len(texts) >= max_records:
                    return texts
    if not texts:
        dataset_path = split_dir / "dataset.json"
        if dataset_path.exists():
            data = json.loads(dataset_path.read_text())
            for rec in data:
                t = rec.get("text", rec.get("note", ""))
                if t:
                    texts.append(t)
                if max_records and len(texts) >= max_records:
                    return texts
    if not texts:
        try:
            from datasets import load_from_disk
            ds = load_from_disk(str(split_dir))
            for rec in ds:
                t = rec.get("text", rec.get("note", ""))
                if t:
                    texts.append(t)
                if max_records and len(texts) >= max_records:
                    return texts
        except Exception:
            pass
    return texts


def run_synbench_mia(
    synthetic_texts: list[str],
    member_texts: list[str],
    nonmember_texts: list[str],
    auxiliary_texts: list[str],
    n_gram: int = 3,
) -> dict:
    log.info(
        "synthetic=%d, members=%d, nonmembers=%d, auxiliary=%d",
        len(synthetic_texts), len(member_texts), len(nonmember_texts), len(auxiliary_texts),
    )

    synth_lm = NgramLM(n=n_gram)
    synth_lm.train(synthetic_texts)

    ref_lm = NgramLM(n=n_gram)
    ref_lm.train(auxiliary_texts)

    member_scores = []
    for text in member_texts:
        dp = synth_lm.log_prob(text) - ref_lm.log_prob(text)
        member_scores.append(dp)

    nonmember_scores = []
    for text in nonmember_texts:
        dp = synth_lm.log_prob(text) - ref_lm.log_prob(text)
        nonmember_scores.append(dp)

    labels = [1] * len(member_scores) + [0] * len(nonmember_scores)
    scores = member_scores + nonmember_scores

    labels_arr = np.array(labels)
    scores_arr = np.array(scores)

    valid = np.isfinite(scores_arr)
    if valid.sum() < 10:
        log.warning("too few valid scores (%d), returning AUC=0.5", valid.sum())
        return {"auc": 0.5, "n_valid": int(valid.sum())}

    labels_arr = labels_arr[valid]
    scores_arr = scores_arr[valid]

    auc = float(roc_auc_score(labels_arr, scores_arr))

    fpr, tpr, thresholds = roc_curve(labels_arr, scores_arr)
    tpr_at_1pct = float(tpr[np.searchsorted(fpr, 0.01, side="right") - 1]) if np.any(fpr <= 0.01) else 0.0
    tpr_at_01pct = float(tpr[np.searchsorted(fpr, 0.001, side="right") - 1]) if np.any(fpr <= 0.001) else 0.0
    tpr_at_10pct = float(tpr[np.searchsorted(fpr, 0.10, side="right") - 1]) if np.any(fpr <= 0.10) else 0.0
    tpr_at_20pct = float(tpr[np.searchsorted(fpr, 0.20, side="right") - 1]) if np.any(fpr <= 0.20) else 0.0

    return {
        "auc": auc,
        "tpr_at_1pct_fpr": tpr_at_1pct,
        "tpr_at_0p1pct_fpr": tpr_at_01pct,
        "tpr_at_10pct_fpr": tpr_at_10pct,
        "tpr_at_20pct_fpr": tpr_at_20pct,
        "member_dp_mean": float(np.mean(member_scores)),
        "member_dp_std": float(np.std(member_scores)),
        "nonmember_dp_mean": float(np.mean(nonmember_scores)),
        "nonmember_dp_std": float(np.std(nonmember_scores)),
        "n_members": len(member_scores),
        "n_nonmembers": len(nonmember_scores),
        "n_synthetic": len(synthetic_texts),
        "n_auxiliary": len(auxiliary_texts),
        "n_gram": n_gram,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic_path", required=True, help="JSONL of synthetic texts")
    ap.add_argument("--splits_path", required=True, help="Directory with members/nonmembers/clean_nonmembers splits")
    ap.add_argument("--output_path", required=True, help="JSON output for results")
    ap.add_argument("--n_gram", type=int, default=3)
    ap.add_argument("--max_members", type=int, default=0, help="Cap on member records (0=all)")
    ap.add_argument("--max_nonmembers", type=int, default=0, help="Cap on nonmember records (0=all)")
    ap.add_argument("--auxiliary_split", default="nonmembers",
                    help="Split to use as auxiliary/reference (default: nonmembers)")
    ap.add_argument("--text_key", default="text", help="JSON key for text in synthetic file")
    args = ap.parse_args()

    splits = Path(args.splits_path)
    synthetic_texts = load_jsonl_texts(Path(args.synthetic_path), text_key=args.text_key)
    log.info("loaded %d synthetic texts", len(synthetic_texts))

    member_texts = load_split(splits, "members", args.max_members)
    log.info("loaded %d member texts", len(member_texts))

    nonmember_texts = load_split(splits, "nonmembers", args.max_nonmembers)
    log.info("loaded %d nonmember texts", len(nonmember_texts))

    auxiliary_texts = load_split(splits, args.auxiliary_split)
    log.info("loaded %d auxiliary texts from '%s'", len(auxiliary_texts), args.auxiliary_split)

    results = {}

    log.info("--- same-dist MIA (members vs nonmembers, ref=nonmembers) ---")
    results["samedist"] = run_synbench_mia(
        synthetic_texts, member_texts, nonmember_texts, auxiliary_texts, args.n_gram,
    )
    log.info("same-dist AUC = %.4f", results["samedist"]["auc"])

    clean_texts = load_split(splits, "clean_nonmembers")
    if clean_texts:
        log.info("loaded %d clean nonmember texts", len(clean_texts))
        log.info("--- clean MIA (members vs clean_nonmembers, ref=clean_nonmembers) ---")
        results["clean"] = run_synbench_mia(
            synthetic_texts, member_texts, clean_texts, clean_texts, args.n_gram,
        )
        log.info("clean AUC = %.4f", results["clean"]["auc"])

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info("wrote results to %s", out)


if __name__ == "__main__":
    main()

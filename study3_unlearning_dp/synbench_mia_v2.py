"""
SynBench-aligned Membership Inference Attack on synthetic text.

Replicates the methodology from:
  "SynBench: A Benchmark for Differentially Private Text Generation" (Sun et al.)

Key alignment with SynBench:
  - Stage 1: LoF outlier detection using sentence-t5-base embeddings
  - Stage 2: 4 reference LMs with balanced target membership (2 include, 2 exclude)
  - Stage 3: ΔP = P_synth(x) - P̄_ref(x) as attack signal
  - n=2 bigrams (SynBench's best-performing setting)
  - Targets: top 1% farthest + LoF outliers (worst-case auditing)

Simplification vs full SynBench:
  - Single synthetic dataset per config (not 100 subsets — infeasible with 8B model)
  - sentence-t5-base via transformers (not sentence-transformers package)

Usage:
  python synbench_mia_v2.py \
    --synthetic_path outputs/generated/psytar_eps8.jsonl \
    --splits_path outputs/splits_psytar \
    --output_path outputs/attacks/synbench_v2_psytar_eps8.json
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
from sklearn.neighbors import LocalOutlierFactor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("synbench_v2")


# ── Tokenization & n-gram LM ──────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class NgramLM:
    """Add-delta smoothed n-gram language model."""
    def __init__(self, n: int = 2, delta: float = 0.5):
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


# ── Embedding & Outlier Detection (Stage 1) ───────────────────────────

def compute_embeddings(texts: list[str], model_name: str = "sentence-transformers/sentence-t5-base"):
    """Compute sentence embeddings using sentence-t5-base encoder via transformers."""
    import torch
    from transformers import AutoTokenizer, T5EncoderModel

    log.info("loading embedding model: %s", model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    embeddings = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        # Mean pooling over non-padding tokens
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings)


def detect_outliers(texts: list[str], top_pct: float = 0.01):
    """
    Stage 1: Identify outlier records using LoF + farthest-point detection.
    Returns indices of outlier records.
    """
    embeddings = compute_embeddings(texts)

    # LoF outliers
    n_neighbors = min(20, len(texts) - 1)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination="auto")
    lof_labels = lof.fit_predict(embeddings)
    lof_outlier_idx = set(np.where(lof_labels == -1)[0])
    log.info("LoF detected %d outliers", len(lof_outlier_idx))

    # Top 1% farthest from centroid
    centroid = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    n_farthest = max(1, int(len(texts) * top_pct))
    farthest_idx = set(np.argsort(distances)[-n_farthest:])
    log.info("farthest %d records (top %.1f%%)", len(farthest_idx), top_pct * 100)

    outlier_idx = sorted(lof_outlier_idx | farthest_idx)
    log.info("total outlier targets: %d", len(outlier_idx))
    return outlier_idx


# ── Data Loading ──────────────────────────────────────────────────────

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

    # Try JSONL files
    for jsonl in sorted(split_dir.glob("*.jsonl")):
        with open(jsonl) as f:
            for line in f:
                rec = json.loads(line)
                t = rec.get("text", rec.get("note", rec.get("content", "")))
                if t:
                    texts.append(t)
                if max_records and len(texts) >= max_records:
                    return texts
    if texts:
        return texts

    # Try HuggingFace dataset
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


# ── SynBench Attack (Stages 2–3) ─────────────────────────────────────

def build_reference_lms(
    auxiliary_texts: list[str],
    target_text: str,
    n_gram: int,
    n_refs: int = 4,
) -> list[NgramLM]:
    """
    Build M=4 reference LMs from auxiliary data.
    Per SynBench: 2 include the target, 2 exclude it.
    """
    rng = np.random.RandomState(42)
    indices = list(range(len(auxiliary_texts)))
    rng.shuffle(indices)

    chunk_size = len(indices) // n_refs
    ref_lms = []
    for r in range(n_refs):
        start = r * chunk_size
        end = start + chunk_size if r < n_refs - 1 else len(indices)
        chunk_texts = [auxiliary_texts[i] for i in indices[start:end]]

        # Include target in first 2 refs, exclude from last 2
        if r < n_refs // 2:
            chunk_texts = chunk_texts + [target_text]

        lm = NgramLM(n=n_gram)
        lm.train(chunk_texts)
        ref_lms.append(lm)

    return ref_lms


def run_synbench_attack(
    synthetic_texts: list[str],
    member_texts: list[str],
    nonmember_texts: list[str],
    auxiliary_texts: list[str],
    n_gram: int = 2,
    use_outlier_detection: bool = True,
) -> dict:
    """
    Full SynBench-aligned MIA attack.

    Stage 1: Identify outlier targets from members + nonmembers
    Stage 2: Build synthetic LM + 4 reference LMs per target
    Stage 3: Compute ΔP scores, ROC, AUC
    """
    log.info(
        "synthetic=%d, members=%d, nonmembers=%d, auxiliary=%d, n_gram=%d",
        len(synthetic_texts), len(member_texts), len(nonmember_texts),
        len(auxiliary_texts), n_gram,
    )

    # Stage 1: Outlier detection
    all_texts = member_texts + nonmember_texts
    all_labels = [1] * len(member_texts) + [0] * len(nonmember_texts)

    if use_outlier_detection and len(all_texts) >= 20:
        outlier_idx = detect_outliers(all_texts)
        target_texts = [all_texts[i] for i in outlier_idx]
        target_labels = [all_labels[i] for i in outlier_idx]
        log.info(
            "targeting %d outliers: %d members, %d nonmembers",
            len(target_texts),
            sum(target_labels),
            len(target_labels) - sum(target_labels),
        )
    else:
        target_texts = all_texts
        target_labels = all_labels
        log.info("using all %d records as targets (outlier detection disabled)", len(target_texts))

    # Stage 2: Train synthetic n-gram LM
    synth_lm = NgramLM(n=n_gram)
    synth_lm.train(synthetic_texts)
    log.info("trained synthetic %d-gram LM: %d contexts", n_gram, len(synth_lm.context_totals))

    # Stage 3: Score each target
    scores = []
    for text, label in zip(target_texts, target_labels):
        p_synth = synth_lm.log_prob(text)

        # Build 4 reference LMs with balanced target membership
        ref_lms = build_reference_lms(auxiliary_texts, text, n_gram, n_refs=4)
        ref_probs = [lm.log_prob(text) for lm in ref_lms]
        p_ref = np.mean(ref_probs)

        dp = p_synth - p_ref
        scores.append({"text_prefix": text[:80], "label": label, "dp": dp,
                        "p_synth": p_synth, "p_ref": p_ref})

    labels_arr = np.array([s["label"] for s in scores])
    scores_arr = np.array([s["dp"] for s in scores])

    valid = np.isfinite(scores_arr)
    if valid.sum() < 10:
        log.warning("too few valid scores (%d)", valid.sum())
        return {"auc": 0.5, "n_valid": int(valid.sum()), "method": "synbench_v2"}

    labels_v = labels_arr[valid]
    scores_v = scores_arr[valid]

    auc = float(roc_auc_score(labels_v, scores_v))
    fpr, tpr, _ = roc_curve(labels_v, scores_v)

    def tpr_at_fpr(fpr_arr, tpr_arr, threshold):
        idx = np.searchsorted(fpr_arr, threshold, side="right") - 1
        return float(tpr_arr[max(idx, 0)]) if np.any(fpr_arr <= threshold) else 0.0

    # Per-group stats
    member_scores = scores_arr[labels_arr == 1]
    nonmember_scores = scores_arr[labels_arr == 0]
    member_valid = member_scores[np.isfinite(member_scores)]
    nonmember_valid = nonmember_scores[np.isfinite(nonmember_scores)]

    results = {
        "method": "synbench_v2",
        "auc": auc,
        "tpr_at_1pct_fpr": tpr_at_fpr(fpr, tpr, 0.01),
        "tpr_at_0p1pct_fpr": tpr_at_fpr(fpr, tpr, 0.001),
        "tpr_at_10pct_fpr": tpr_at_fpr(fpr, tpr, 0.10),
        "tpr_at_20pct_fpr": tpr_at_fpr(fpr, tpr, 0.20),
        "member_dp_mean": float(np.mean(member_valid)) if len(member_valid) else 0,
        "member_dp_std": float(np.std(member_valid)) if len(member_valid) else 0,
        "nonmember_dp_mean": float(np.mean(nonmember_valid)) if len(nonmember_valid) else 0,
        "nonmember_dp_std": float(np.std(nonmember_valid)) if len(nonmember_valid) else 0,
        "n_targets": len(target_texts),
        "n_target_members": int(sum(target_labels)),
        "n_target_nonmembers": int(len(target_labels) - sum(target_labels)),
        "n_synthetic": len(synthetic_texts),
        "n_auxiliary": len(auxiliary_texts),
        "n_gram": n_gram,
        "n_refs": 4,
        "outlier_detection": use_outlier_detection,
    }

    # Also run on ALL records with a single pre-built reference LM (fast)
    if use_outlier_detection and len(target_texts) < len(all_texts):
        log.info("--- also scoring all records (single ref LM, fast) ---")
        ref_lm_single = NgramLM(n=n_gram)
        ref_lm_single.train(auxiliary_texts)
        all_scores = []
        for text in all_texts:
            p_s = synth_lm.log_prob(text)
            p_r = ref_lm_single.log_prob(text)
            all_scores.append(p_s - p_r)

        all_scores_arr = np.array(all_scores)
        all_labels_arr = np.array(all_labels)
        av = np.isfinite(all_scores_arr)
        if av.sum() >= 10:
            all_auc = float(roc_auc_score(all_labels_arr[av], all_scores_arr[av]))
            results["all_records_auc"] = all_auc
            log.info("all-records AUC = %.4f", all_auc)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic_path", required=True)
    ap.add_argument("--splits_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--n_gram", type=int, default=2, help="n-gram order (SynBench uses 2)")
    ap.add_argument("--auxiliary_split", default="auxiliary")
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--no_outlier_detection", action="store_true",
                    help="Disable LoF outlier targeting (use all records)")
    args = ap.parse_args()

    splits = Path(args.splits_path)

    synthetic_texts = load_jsonl_texts(Path(args.synthetic_path), text_key=args.text_key)
    log.info("loaded %d synthetic texts", len(synthetic_texts))

    member_texts = load_split(splits, "members")
    log.info("loaded %d member texts", len(member_texts))

    nonmember_texts = load_split(splits, "nonmembers")
    log.info("loaded %d nonmember texts", len(nonmember_texts))

    auxiliary_texts = load_split(splits, args.auxiliary_split)
    if not auxiliary_texts:
        log.warning("no auxiliary split '%s', falling back to 'retain'", args.auxiliary_split)
        auxiliary_texts = load_split(splits, "retain")
    if not auxiliary_texts:
        log.warning("no auxiliary data found, splitting nonmembers in half")
        mid = len(nonmember_texts) // 2
        auxiliary_texts = nonmember_texts[:mid]
        nonmember_texts = nonmember_texts[mid:]
    log.info("loaded %d auxiliary texts", len(auxiliary_texts))

    results = run_synbench_attack(
        synthetic_texts=synthetic_texts,
        member_texts=member_texts,
        nonmember_texts=nonmember_texts,
        auxiliary_texts=auxiliary_texts,
        n_gram=args.n_gram,
        use_outlier_detection=not args.no_outlier_detection,
    )

    log.info("outlier-targeted AUC = %.4f", results["auc"])
    if "all_records_auc" in results:
        log.info("all-records AUC = %.4f", results["all_records_auc"])

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info("wrote results to %s", out)


if __name__ == "__main__":
    main()

"""
07_mia_v2.py
============
Membership Inference Attack — adapted from 07_mia.py to support both
MIMIC (single JSONL) and splits-based (HuggingFace datasets) workflows.

STEP 09 — Black-box MIA on synthetic output (SynBench protocol, Sun et al. 2025)
    Adversary has access only to the synthetic dataset, not the model.
    Tests whether synthetic text leaks information about real training records.
    Uses n-gram language model scoring on outlier records.

STEP 10 — White-box MIA on fine-tuned model (Carlini et al. LiRA, 2022)
    (unchanged from 07_mia.py)

New in v2:
    - --splits_path: load members/nonmembers from HuggingFace datasets dirs
    - Auto-detects text fields (bhc_text, text, note, content)
    - When splits provided, uses true member/non-member labels (no outlier proxy)
    - Reference LM trained on non-members (proper auxiliary set)
    - Synthetic text field auto-detected (synthetic_bhc or text)

Usage (BioMistral PMC):
    python 07_mia_v2.py --mode blackbox \
        --synthetic outputs/generated/synbench_bio_baseline.jsonl \
        --splits_path outputs/splits_v1_pmc \
        --output outputs/attacks/synbench_bio_baseline \
        --epsilon baseline

Usage (MIMIC — backwards compatible):
    python 07_mia_v2.py --mode blackbox \
        --synthetic ./generated/eps4/synthetic_bhc.jsonl \
        --real ./data/train.jsonl \
        --output ./mia/eps4 \
        --epsilon 4
"""

import os
import sys
import json
import math
import random
import logging
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from scipy import stats as scipy_stats
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, roc_curve


class FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[FlushHandler(sys.stderr)]
)
log = logging.getLogger(__name__)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def gpu_mem():
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            resrv = torch.cuda.memory_reserved() / 1e9
            return f"GPU: {alloc:.1f}GB / {resrv:.1f}GB"
    except ImportError:
        pass
    return "GPU: N/A"


# =========================================================================
# Data loading — supports JSONL and HuggingFace datasets
# =========================================================================

TEXT_FIELDS = ['bhc_text', 'text', 'note', 'content']
ID_FIELDS = ['note_id', 'doc_id', 'id', 'pmcid']
CAT_FIELDS_NESTED = [('control_codes', 'icd_category')]
CAT_FIELDS_FLAT = ['specialty', 'icd_category', 'category']
SYNTH_TEXT_FIELDS = ['synthetic_bhc', 'text']


def get_text(rec):
    for f in TEXT_FIELDS:
        if f in rec and rec[f]:
            return rec[f]
    return ''


def get_id(rec):
    for f in ID_FIELDS:
        if f in rec and rec[f]:
            return str(rec[f])
    return ''


def get_category(rec):
    for parent, child in CAT_FIELDS_NESTED:
        if parent in rec and isinstance(rec[parent], dict) and child in rec[parent]:
            return rec[parent][child]
    for f in CAT_FIELDS_FLAT:
        if f in rec and rec[f]:
            return rec[f]
    return 'Unknown'


def get_synth_text(rec):
    for f in SYNTH_TEXT_FIELDS:
        if f in rec and rec[f]:
            return rec[f]
    return ''


def load_jsonl(path, limit=None):
    records = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            records.append(rec)
    if limit and limit < len(records):
        random.seed(42)
        records = random.sample(records, limit)
    return records


def load_split_dir(split_dir):
    split_dir = Path(split_dir)
    records = []
    for jsonl in sorted(split_dir.glob("*.jsonl")):
        with open(jsonl) as f:
            for line in f:
                records.append(json.loads(line.strip()))
    if records:
        return records
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(split_dir))
        return [dict(row) for row in ds]
    except Exception as e:
        log.warning(f"Could not load {split_dir} as HF dataset: {e}")
    return records


def load_splits(splits_path):
    splits_path = Path(splits_path)
    members = load_split_dir(splits_path / "members")
    nonmembers = load_split_dir(splits_path / "nonmembers")
    clean = load_split_dir(splits_path / "clean_nonmembers")
    canaries = load_split_dir(splits_path / "canaries")
    retain = load_split_dir(splits_path / "retain")
    log.info(f"Splits loaded: members={len(members)}, nonmembers={len(nonmembers)}, "
             f"clean={len(clean)}, canaries={len(canaries)}, retain={len(retain)}")
    return members, nonmembers, clean, canaries, retain


# =========================================================================
# N-gram LM (unchanged from 07_mia.py)
# =========================================================================

class NgramLM:
    def __init__(self, n=2, k=0.01):
        self.n = n
        self.k = k
        self.counts = Counter()
        self.context_counts = Counter()
        self.vocab = set()

    def get_ngrams(self, text):
        words = text.lower().split()
        return [tuple(words[i:i+self.n]) for i in range(len(words) - self.n + 1)]

    def fit(self, texts):
        for text in texts:
            ngrams = self.get_ngrams(text)
            for ng in ngrams:
                self.counts[ng] += 1
                self.context_counts[ng[:-1]] += 1
                self.vocab.update(ng)
        return self

    def log_prob(self, text):
        ngrams = self.get_ngrams(text)
        if not ngrams:
            return float('-inf')
        vocab_size = max(len(self.vocab), 1)
        log_p = 0.0
        for ng in ngrams:
            context = ng[:-1]
            count = self.counts.get(ng, 0)
            ctx_count = self.context_counts.get(context, 0)
            p = (count + self.k) / (ctx_count + self.k * vocab_size)
            log_p += math.log(max(p, 1e-30))
        return log_p / len(ngrams)


# =========================================================================
# Outlier detection (unchanged from 07_mia.py)
# =========================================================================

def detect_outliers_lof(records, n_neighbors=20, top_pct=0.01):
    log.info(f"Detecting outliers (LoF n_neighbors={n_neighbors}, "
             f"top {top_pct*100:.0f}%)...")
    features = []
    for rec in records:
        text = get_text(rec)
        words = text.lower().split()
        n = max(len(words), 1)
        uniq = max(len(set(words)), 1)
        features.append([
            math.log(n),
            math.log(uniq),
            uniq / n,
            text.count('___') / n,
        ])
    X = np.array(features)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_norm = (X - mean) / std

    lof = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(X)-1), contamination='auto')
    lof.fit_predict(X_norm)
    lof_scores = -lof.negative_outlier_factor_

    threshold = np.percentile(lof_scores, (1 - top_pct) * 100)
    outlier_indices = [i for i, s in enumerate(lof_scores) if s >= threshold]

    log.info(f"Outlier threshold (LoF score): {threshold:.4f}")
    log.info(f"Outliers identified: {len(outlier_indices):,}")
    return outlier_indices, lof_scores


# =========================================================================
# STEP 09 — Black-box MIA (SynBench protocol)
# =========================================================================

def run_blackbox_mia_splits(member_records, nonmember_records, syn_records,
                            output_dir, args, auxiliary_records=None):
    """
    Black-box MIA with explicit member/non-member splits.
    SynBench protocol adapted for known membership labels.

    Reference LM is trained on a SEPARATE auxiliary set (not the test
    non-members) to avoid data leakage. If no auxiliary set is provided,
    non-members are split in half: first half for reference, second for testing.

    For each member and non-member record:
      1. Score under synthetic corpus n-gram LM
      2. Score under reference (auxiliary) n-gram LM
      3. ΔP = syn_score - ref_score
    AUC computed with true member/non-member labels.
    """
    log.info("=" * 60)
    log.info("STEP 09: BLACK-BOX MIA (SynBench — splits mode)")
    log.info("=" * 60)

    # Build synthetic n-gram LM
    syn_texts = [get_synth_text(r) for r in syn_records if get_synth_text(r)]
    log.info(f"Building synthetic n-gram LM on {len(syn_texts)} texts...")
    syn_lm = NgramLM(n=args.ngram_size).fit(syn_texts)

    # Build reference n-gram LM on SEPARATE auxiliary data (not test non-members)
    if auxiliary_records:
        aux_texts = [get_text(r) for r in auxiliary_records if get_text(r)]
        test_nonmembers = nonmember_records
        log.info(f"Building reference n-gram LM on {len(aux_texts)} auxiliary texts "
                 f"(separate from {len(test_nonmembers)} test non-members)")
    else:
        random.seed(42)
        shuffled = list(nonmember_records)
        random.shuffle(shuffled)
        mid = len(shuffled) // 2
        ref_nonmembers = shuffled[:mid]
        test_nonmembers = shuffled[mid:]
        aux_texts = [get_text(r) for r in ref_nonmembers if get_text(r)]
        log.info(f"Split non-members: {len(ref_nonmembers)} for reference LM, "
                 f"{len(test_nonmembers)} for testing")
    ref_lm = NgramLM(n=args.ngram_size).fit(aux_texts)

    # Replace nonmember_records with test-only subset for scoring below
    nonmember_records = test_nonmembers

    # Optionally focus on outlier members (worst-case)
    if args.use_outliers:
        outlier_indices, lof_scores = detect_outliers_lof(
            member_records, n_neighbors=args.lof_neighbors,
            top_pct=args.outlier_pct / 100,
        )
        if args.n_outliers and len(outlier_indices) > args.n_outliers:
            random.seed(42)
            outlier_indices = random.sample(outlier_indices, args.n_outliers)
        target_members = [member_records[i] for i in outlier_indices]
        target_lof = [float(lof_scores[i]) for i in outlier_indices]
        log.info(f"Using {len(target_members)} outlier members as targets")
    else:
        target_members = member_records
        target_lof = [0.0] * len(member_records)
        log.info(f"Using all {len(target_members)} members as targets")

    # Score members
    member_results = []
    for i, rec in enumerate(target_members):
        text = get_text(rec)
        if not text.strip():
            continue
        syn_score = syn_lm.log_prob(text)
        ref_score = ref_lm.log_prob(text)
        if math.isinf(syn_score) or math.isinf(ref_score):
            continue
        member_results.append({
            'doc_id': get_id(rec),
            'category': get_category(rec),
            'lof_score': target_lof[i],
            'syn_score': float(syn_score),
            'ref_score': float(ref_score),
            'attack_signal': float(syn_score - ref_score),
            'member': 1,
        })

    # Score non-members
    nonmember_results = []
    n_nonmem_sample = min(len(member_results), len(nonmember_records))
    if n_nonmem_sample < len(nonmember_records):
        random.seed(42)
        nm_sample = random.sample(nonmember_records, n_nonmem_sample)
    else:
        nm_sample = nonmember_records

    for rec in nm_sample:
        text = get_text(rec)
        if not text.strip():
            continue
        syn_score = syn_lm.log_prob(text)
        ref_score = ref_lm.log_prob(text)
        if math.isinf(syn_score) or math.isinf(ref_score):
            continue
        nonmember_results.append({
            'doc_id': get_id(rec),
            'category': get_category(rec),
            'lof_score': 0.0,
            'syn_score': float(syn_score),
            'ref_score': float(ref_score),
            'attack_signal': float(syn_score - ref_score),
            'member': 0,
        })

    all_results = member_results + nonmember_results
    signals = np.array([r['attack_signal'] for r in all_results])
    labels = np.array([r['member'] for r in all_results])

    if len(np.unique(labels)) < 2 or len(signals) < 10:
        log.warning("Insufficient data for AUC — %d records, %d classes",
                     len(signals), len(np.unique(labels)))
        auc = float('nan')
        fpr_arr = tpr_arr = np.array([])
    else:
        fpr_arr, tpr_arr, _ = roc_curve(labels, signals)
        auc = float(roc_auc_score(labels, signals))

    def tpr_at_fpr(fpr_target):
        if len(fpr_arr) == 0:
            return float('nan')
        idx = np.searchsorted(fpr_arr, fpr_target)
        if idx >= len(tpr_arr):
            return float(tpr_arr[-1])
        return float(tpr_arr[idx])

    bb_results = {
        'n_members': int(labels.sum()),
        'n_nonmembers': int((1 - labels).sum()),
        'auc': auc,
        'tpr_at_fpr_01pct': tpr_at_fpr(0.001),
        'tpr_at_fpr_1pct': tpr_at_fpr(0.01),
        'tpr_at_fpr_10pct': tpr_at_fpr(0.10),
        'tpr_at_fpr_20pct': tpr_at_fpr(0.20),
        'attack_signal_member_mean': float(np.mean(signals[labels == 1])) if labels.sum() > 0 else float('nan'),
        'attack_signal_member_std': float(np.std(signals[labels == 1])) if labels.sum() > 0 else float('nan'),
        'attack_signal_nonmember_mean': float(np.mean(signals[labels == 0])) if (1-labels).sum() > 0 else float('nan'),
        'attack_signal_nonmember_std': float(np.std(signals[labels == 0])) if (1-labels).sum() > 0 else float('nan'),
        'roc_curve': {
            'fpr': fpr_arr.tolist() if len(fpr_arr) > 0 else [],
            'tpr': tpr_arr.tolist() if len(tpr_arr) > 0 else [],
        },
        'per_record': all_results,
    }

    log.info(f"Black-box MIA results (splits mode):")
    log.info(f"  Members:          {bb_results['n_members']}")
    log.info(f"  Non-members:      {bb_results['n_nonmembers']}")
    log.info(f"  AUC:              {auc:.4f}")
    log.info(f"  TPR @ 0.1% FPR:  {bb_results['tpr_at_fpr_01pct']:.4f}")
    log.info(f"  TPR @ 1% FPR:    {bb_results['tpr_at_fpr_1pct']:.4f}")
    log.info(f"  TPR @ 10% FPR:   {bb_results['tpr_at_fpr_10pct']:.4f}")
    log.info(f"  TPR @ 20% FPR:   {bb_results['tpr_at_fpr_20pct']:.4f}")
    log.info(f"  Member ΔP mean:     {bb_results['attack_signal_member_mean']:.4f}")
    log.info(f"  Non-member ΔP mean: {bb_results['attack_signal_nonmember_mean']:.4f}")

    bb_path = output_dir / 'blackbox_mia.json'
    save_obj = {k: v for k, v in bb_results.items() if k != 'per_record'}
    save_obj['per_record'] = all_results
    with open(bb_path, 'w') as f:
        json.dump(save_obj, f, indent=2, default=str)
    log.info(f"Saved to {bb_path}")

    return bb_results


def run_blackbox_mia_legacy(real_records, syn_records, output_dir, args):
    """
    Original black-box MIA from 07_mia.py (MIMIC workflow).
    All real records treated as members, outlier detection for targeting.
    """
    log.info("=" * 60)
    log.info("STEP 09: BLACK-BOX MIA (SynBench — legacy mode)")
    log.info("=" * 60)

    outlier_indices, lof_scores = detect_outliers_lof(
        real_records,
        n_neighbors=args.lof_neighbors,
        top_pct=args.outlier_pct / 100,
    )

    if args.n_outliers and len(outlier_indices) > args.n_outliers:
        random.seed(42)
        outlier_indices = random.sample(outlier_indices, args.n_outliers)
    log.info(f"Running attack on {len(outlier_indices):,} target records")

    syn_by_cat = defaultdict(list)
    for rec in syn_records:
        cat = get_category(rec)
        syn_by_cat[cat].append(get_synth_text(rec))

    non_outlier_indices = [i for i in range(len(real_records))
                           if i not in set(outlier_indices)]

    attack_results = []

    for target_idx in outlier_indices:
        target_rec = real_records[target_idx]
        target_text = get_text(target_rec)
        target_cat = get_category(target_rec)

        if not target_text.strip():
            continue

        syn_texts = syn_by_cat.get(target_cat, [])
        if len(syn_texts) < 10:
            syn_texts = [get_synth_text(r) for r in syn_records]

        syn_lm = NgramLM(n=args.ngram_size).fit(syn_texts)
        syn_score = syn_lm.log_prob(target_text)

        ref_scores = []
        for _ in range(args.n_reference):
            ref_sample = random.sample(
                non_outlier_indices,
                min(len(syn_texts), len(non_outlier_indices))
            )
            ref_texts = [get_text(real_records[i]) for i in ref_sample]
            ref_lm = NgramLM(n=args.ngram_size).fit(ref_texts)
            ref_scores.append(ref_lm.log_prob(target_text))

        mean_ref_score = float(np.mean(ref_scores)) if ref_scores else 0.0

        attack_results.append({
            'doc_id': get_id(target_rec),
            'category': target_cat,
            'lof_score': float(lof_scores[target_idx]),
            'syn_score': float(syn_score),
            'ref_score': float(mean_ref_score),
            'attack_signal': float(syn_score - mean_ref_score),
            'member': 1,
        })

    n_nonmember = min(len(attack_results), len(non_outlier_indices))
    nonmember_sample = random.sample(non_outlier_indices, n_nonmember)

    for idx in nonmember_sample:
        rec = real_records[idx]
        text = get_text(rec)
        cat = get_category(rec)

        if not text.strip():
            continue

        syn_texts = syn_by_cat.get(cat, [get_synth_text(r) for r in syn_records])
        syn_lm = NgramLM(n=args.ngram_size).fit(syn_texts)
        syn_score = syn_lm.log_prob(text)

        ref_sample = random.sample(
            [i for i in non_outlier_indices if i != idx],
            min(len(syn_texts), len(non_outlier_indices) - 1)
        )
        ref_texts = [get_text(real_records[i]) for i in ref_sample]
        ref_lm = NgramLM(n=args.ngram_size).fit(ref_texts)
        ref_score = ref_lm.log_prob(text)

        attack_results.append({
            'doc_id': get_id(rec),
            'category': cat,
            'lof_score': float(lof_scores[idx]),
            'syn_score': float(syn_score),
            'ref_score': float(ref_score),
            'attack_signal': float(syn_score - ref_score),
            'member': 0,
        })

    signals = np.array([r['attack_signal'] for r in attack_results])
    labels = np.array([r['member'] for r in attack_results])

    if len(np.unique(labels)) < 2:
        auc = float('nan')
        fpr_arr = tpr_arr = np.array([])
    else:
        fpr_arr, tpr_arr, _ = roc_curve(labels, signals)
        auc = float(roc_auc_score(labels, signals))

    def tpr_at_fpr(fpr_target):
        if len(fpr_arr) == 0:
            return float('nan')
        idx = np.searchsorted(fpr_arr, fpr_target)
        if idx >= len(tpr_arr):
            return float(tpr_arr[-1])
        return float(tpr_arr[idx])

    bb_results = {
        'n_members': int(labels.sum()),
        'n_nonmembers': int((1 - labels).sum()),
        'auc': auc,
        'tpr_at_fpr_01pct': tpr_at_fpr(0.001),
        'tpr_at_fpr_1pct': tpr_at_fpr(0.01),
        'tpr_at_fpr_10pct': tpr_at_fpr(0.10),
        'attack_signal_member_mean': float(np.mean(signals[labels == 1])) if labels.sum() > 0 else float('nan'),
        'attack_signal_nonmember_mean': float(np.mean(signals[labels == 0])) if (1-labels).sum() > 0 else float('nan'),
        'roc_curve': {
            'fpr': fpr_arr.tolist() if len(fpr_arr) > 0 else [],
            'tpr': tpr_arr.tolist() if len(tpr_arr) > 0 else [],
        },
        'per_record': attack_results,
    }

    log.info(f"Black-box MIA results (legacy mode):")
    log.info(f"  AUC:              {auc:.4f}")
    log.info(f"  TPR @ 0.1% FPR:  {bb_results['tpr_at_fpr_01pct']:.4f}")
    log.info(f"  TPR @ 1% FPR:    {bb_results['tpr_at_fpr_1pct']:.4f}")
    log.info(f"  TPR @ 10% FPR:   {bb_results['tpr_at_fpr_10pct']:.4f}")

    bb_path = output_dir / 'blackbox_mia.json'
    save_obj = {k: v for k, v in bb_results.items() if k != 'per_record'}
    save_obj['per_record'] = attack_results
    with open(bb_path, 'w') as f:
        json.dump(save_obj, f, indent=2, default=str)
    log.info(f"Saved to {bb_path}")

    return bb_results


# =========================================================================
# STEP 10 — White-box MIA (unchanged from 07_mia.py)
# =========================================================================

def compute_logit_scaled_loss(text, model, tokenizer, device, max_length=512):
    import torch
    encoding = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    input_ids = encoding['input_ids'].to(device)
    if input_ids.shape[1] <= 1:
        return float('nan')
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        mean_log_p = -outputs.loss.item()
    p = math.exp(max(min(mean_log_p, -0.001), -20.0))
    p = max(min(p, 1 - 1e-7), 1e-7)
    return math.log(p) - math.log(1 - p)


def run_whitebox_mia(real_records, checkpoint_path, base_model_path,
                     output_dir, args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    log.info("=" * 60)
    log.info("STEP 10: WHITE-BOX MIA (Carlini LiRA offline)")
    log.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, local_files_only=True)
    finetuned_model = PeftModel.from_pretrained(base, checkpoint_path)
    finetuned_model = finetuned_model.merge_and_unload().to(device).eval()

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device).eval()

    random.seed(42)
    sample_records = random.sample(real_records, min(args.n_whitebox, len(real_records)))

    results = []
    for i, rec in enumerate(sample_records):
        text = get_text(rec)
        if not text.strip():
            continue
        phi_ft = compute_logit_scaled_loss(text, finetuned_model, tokenizer, device, args.max_length)
        phi_base = compute_logit_scaled_loss(text, base_model, tokenizer, device, args.max_length)
        if math.isnan(phi_ft) or math.isnan(phi_base):
            continue
        results.append({
            'doc_id': get_id(rec),
            'phi_finetuned': phi_ft,
            'phi_base': phi_base,
            'attack_signal': phi_ft - phi_base,
        })
        if (i + 1) % 100 == 0:
            log.info(f"  [{i+1}/{len(sample_records)}]  signal={phi_ft - phi_base:.3f}")

    del finetuned_model, base_model
    torch.cuda.empty_cache()

    signals = np.array([r['attack_signal'] for r in results])
    mu = float(np.mean(signals))
    sigma = float(np.std(signals))

    wb_results = {
        'n_records': len(results),
        'signal_mean': mu,
        'signal_std': sigma,
        'signal_p50': float(np.percentile(signals, 50)),
        'signal_p90': float(np.percentile(signals, 90)),
        'signal_p99': float(np.percentile(signals, 99)),
        'pct_positive': float((signals > 0).mean()),
        'per_record': results,
    }

    log.info(f"White-box MIA: mean={mu:.4f}, std={sigma:.4f}, "
             f"positive={100*(signals > 0).mean():.1f}%")

    wb_path = output_dir / 'whitebox_mia.json'
    with open(wb_path, 'w') as f:
        json.dump(wb_results, f, indent=2, default=str)
    log.info(f"Saved to {wb_path}")
    return wb_results


# =========================================================================
# Main
# =========================================================================

def main(args):
    random.seed(42)
    np.random.seed(42)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Output: {output_dir}")
    log.info(f"Epsilon: {args.epsilon}")
    log.info(f"Mode: {args.mode}")

    use_splits = args.splits_path is not None

    all_results = {'epsilon': args.epsilon, 'mode': args.mode}

    if args.mode in ('blackbox', 'both'):
        syn_records = load_jsonl(args.synthetic)
        log.info(f"Loaded {len(syn_records)} synthetic records")

        if use_splits:
            members, nonmembers, clean, canaries, retain = load_splits(args.splits_path)
            bb_results = run_blackbox_mia_splits(
                members, nonmembers, syn_records, output_dir, args,
                auxiliary_records=None,
            )
        else:
            real_records = load_jsonl(args.real, limit=args.n_real)
            log.info(f"Loaded {len(real_records)} real records")
            bb_results = run_blackbox_mia_legacy(
                real_records, syn_records, output_dir, args
            )

        all_results['blackbox'] = {
            k: v for k, v in bb_results.items()
            if k not in ('per_record', 'roc_curve')
        }

    if args.mode in ('whitebox', 'both'):
        if not args.checkpoint:
            log.error("--checkpoint required for whitebox MIA")
            sys.exit(1)
        if use_splits:
            members, _, _, _, _ = load_splits(args.splits_path)
            real_records = members
        else:
            real_records = load_jsonl(args.real, limit=args.n_real)
        wb_results = run_whitebox_mia(
            real_records, args.checkpoint, args.base_model,
            output_dir, args
        )
        all_results['whitebox'] = {
            k: v for k, v in wb_results.items() if k != 'per_record'
        }

    summary_path = output_dir / 'mia_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode', choices=['blackbox', 'whitebox', 'both'],
                        default='blackbox')

    # Data — either --real (legacy) or --splits_path (v2)
    parser.add_argument('--real', default=None,
                        help='Single JSONL of real records (legacy MIMIC mode)')
    parser.add_argument('--splits_path', default=None,
                        help='Directory with members/nonmembers/clean_nonmembers splits')
    parser.add_argument('--synthetic', default=None,
                        help='JSONL of synthetic records (required for blackbox)')
    parser.add_argument('--n_real', type=int, default=None)

    # Black-box params
    parser.add_argument('--ngram_size', type=int, default=2)
    parser.add_argument('--n_reference', type=int, default=4)
    parser.add_argument('--n_outliers', type=int, default=200)
    parser.add_argument('--outlier_pct', type=float, default=1.0)
    parser.add_argument('--lof_neighbors', type=int, default=20)
    parser.add_argument('--use_outliers', action='store_true',
                        help='Use outlier detection on members (splits mode only)')

    # White-box params
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--base_model', default=None)
    parser.add_argument('--n_whitebox', type=int, default=1000)
    parser.add_argument('--max_length', type=int, default=512)

    # Output
    parser.add_argument('--output', default='./mia/eps4')
    parser.add_argument('--epsilon', default=None)

    args = parser.parse_args()

    if args.mode in ('blackbox', 'both') and not args.synthetic:
        parser.error("--synthetic required for blackbox mode")
    if not args.real and not args.splits_path:
        parser.error("Either --real or --splits_path required")

    main(args)

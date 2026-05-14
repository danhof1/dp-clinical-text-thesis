"""
05_evaluate.py
==============
Phase 4, Steps 09-10 — Evaluate synthetic BHC quality.

Runs automated evaluation metrics on synthetic_bhc.jsonl produced by 03_generate.py.
Compares synthetic BHC against real BHC from train.jsonl.

Tier 1 — Surface Fidelity:
    - Text length KL divergence
    - N-gram frequency distribution comparison (unigram + bigram)
    - MAUVE score (requires mauve-text package)
    - Term overlap (Jaccard on extracted medical terms)
      NOTE: Falls back to simple noun-phrase extraction if QuickUMLS unavailable

Tier 3 — Medical Coherence (automated layers):
    - UniEval coherence/consistency/fluency scoring
      NOTE: Requires UniEval model weights. Falls back to perplexity-based proxy.
    - Per-ICD-category breakdown of all metrics

Does NOT include:
    - Tier 2 TSTR classifier experiment (separate script: 06_tstr.py)
    - MIA audit (separate script)
    - Physician pairwise evaluation (manual process)

Usage:
    python 05_evaluate.py \
        --synthetic ./generated/eps4/synthetic_bhc.jsonl \
        --real ./data/train.jsonl \
        --output ./eval/eps4

    # Quick test on first 200 records
    python 05_evaluate.py \
        --synthetic ./generated/eps4/synthetic_bhc.jsonl \
        --real ./data/train.jsonl \
        --output ./eval/eps4 \
        --sample 200
"""

import os
import sys
import json
import math
import re
import logging
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
from scipy import stats as scipy_stats
from tqdm import tqdm

# =========================================================================
# Logging
# =========================================================================

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


# =========================================================================
# Data loading
# =========================================================================

def load_synthetic(path, sample=None):
    """Load synthetic BHC records from 03_generate.py output."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    if sample and sample < len(records):
        import random
        random.seed(42)
        records = random.sample(records, sample)
    log.info(f"Loaded {len(records):,} synthetic records")
    return records


def load_real(path, sample=None, note_ids=None):
    """
    Load real BHC records from train.jsonl.
    If note_ids provided, load only matching records (for paired comparison).
    Otherwise sample randomly.
    """
    records = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            if note_ids is not None:
                if rec['note_id'] in note_ids:
                    records.append(rec)
            else:
                records.append(rec)

    if note_ids is None and sample and sample < len(records):
        import random
        random.seed(42)
        records = random.sample(records, sample)

    log.info(f"Loaded {len(records):,} real records")
    return records


# =========================================================================
# Tier 1: Surface Fidelity
# =========================================================================

# --- Text length KL divergence ---

def compute_length_kl(syn_texts, real_texts, n_bins=50):
    """
    KL divergence between text length distributions.
    Following Term2Note: KL(real || synthetic).
    Lower = more similar length distributions.
    """
    syn_lens = [len(t.split()) for t in syn_texts]
    real_lens = [len(t.split()) for t in real_texts]

    # Shared bin edges
    all_lens = syn_lens + real_lens
    bins = np.linspace(0, np.percentile(all_lens, 99), n_bins + 1)

    syn_hist, _ = np.histogram(syn_lens, bins=bins, density=True)
    real_hist, _ = np.histogram(real_lens, bins=bins, density=True)

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    syn_hist = syn_hist + eps
    real_hist = real_hist + eps

    # Normalise to proper distributions
    syn_hist = syn_hist / syn_hist.sum()
    real_hist = real_hist / real_hist.sum()

    kl = scipy_stats.entropy(real_hist, syn_hist)

    return {
        'kl_divergence': float(kl),
        'real_length': {
            'mean': float(np.mean(real_lens)),
            'median': float(np.median(real_lens)),
            'std': float(np.std(real_lens)),
        },
        'synthetic_length': {
            'mean': float(np.mean(syn_lens)),
            'median': float(np.median(syn_lens)),
            'std': float(np.std(syn_lens)),
        },
    }


# --- N-gram frequency comparison ---

def get_ngrams(text, n):
    words = text.lower().split()
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]


def compute_ngram_divergence(syn_texts, real_texts, n=1, top_k=1000):
    """
    Compare n-gram frequency distributions between synthetic and real corpora.
    Returns KL divergence over the top-k most frequent n-grams in real data.
    """
    real_counts = Counter()
    syn_counts = Counter()

    for text in real_texts:
        real_counts.update(get_ngrams(text, n))
    for text in syn_texts:
        syn_counts.update(get_ngrams(text, n))

    # Use top-k from real distribution as vocabulary
    vocab = [ng for ng, _ in real_counts.most_common(top_k)]

    real_freq = np.array([real_counts[ng] for ng in vocab], dtype=float)
    syn_freq = np.array([syn_counts[ng] for ng in vocab], dtype=float)

    # Normalise
    eps = 1e-10
    real_freq = real_freq / (real_freq.sum() + eps) + eps
    syn_freq = syn_freq / (syn_freq.sum() + eps) + eps

    real_freq = real_freq / real_freq.sum()
    syn_freq = syn_freq / syn_freq.sum()

    kl = scipy_stats.entropy(real_freq, syn_freq)

    return {
        f'{n}gram_kl_divergence': float(kl),
        f'{n}gram_vocab_real': len(real_counts),
        f'{n}gram_vocab_syn': len(syn_counts),
    }


# --- Term extraction (fallback without QuickUMLS) ---

def extract_terms_simple(text):
    """
    Simple medical term extraction using capitalised multi-word phrases
    and common clinical patterns. Fallback for when QuickUMLS is unavailable.
    """
    terms = set()

    cap_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
    terms.update(p.lower() for p in cap_phrases)

    abbrevs = re.findall(r'\b[A-Z]{2,6}\b', text)
    clinical_abbrevs = {'ICU', 'IV', 'PO', 'BID', 'TID', 'QID', 'PRN',
                        'CHF', 'COPD', 'DM', 'HTN', 'CAD', 'CKD', 'DVT',
                        'PE', 'MI', 'CVA', 'TIA', 'ARDS', 'AKI', 'UTI',
                        'GI', 'CT', 'MRI', 'ECG', 'EKG', 'ABG', 'CBC',
                        'BMP', 'CMP', 'INR', 'PT', 'PTT', 'WBC', 'RBC',
                        'HGB', 'HCT', 'PLT', 'BUN', 'CR', 'AST', 'ALT',
                        'AFIB', 'NSTEMI', 'STEMI', 'PCI', 'CABG', 'EF',
                        'LVEF', 'RVR', 'NSR', 'PICC', 'TPN', 'NGT',
                        'SBP', 'DBP', 'MAP', 'CVP', 'PEEP', 'FIO2'}
    terms.update(a.lower() for a in abbrevs if a in clinical_abbrevs)

    drugs = re.findall(
        r'\b\w+(?:olol|pril|artan|statin|mycin|cillin|azole|pam|lam|pine|done|ide|ine)\b',
        text.lower()
    )
    terms.update(drugs)

    return terms


def compute_term_jaccard(syn_texts, real_texts):
    """
    Jaccard similarity between term sets extracted from synthetic and real corpora.
    """
    real_unary = set()
    syn_unary = set()
    real_binary = set()
    syn_binary = set()

    for text in real_texts:
        terms = extract_terms_simple(text)
        real_unary.update(terms)
        terms_list = sorted(terms)
        for i in range(len(terms_list)):
            for j in range(i + 1, min(i + 10, len(terms_list))):
                real_binary.add((terms_list[i], terms_list[j]))

    for text in syn_texts:
        terms = extract_terms_simple(text)
        syn_unary.update(terms)
        terms_list = sorted(terms)
        for i in range(len(terms_list)):
            for j in range(i + 1, min(i + 10, len(terms_list))):
                syn_binary.add((terms_list[i], terms_list[j]))

    def jaccard(a, b):
        if not a and not b:
            return 1.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0

    return {
        'unary_jaccard': float(jaccard(real_unary, syn_unary)),
        'binary_jaccard': float(jaccard(real_binary, syn_binary)),
        'real_unary_terms': len(real_unary),
        'syn_unary_terms': len(syn_unary),
        'real_binary_pairs': len(real_binary),
        'syn_binary_pairs': len(syn_binary),
        'term_extraction': 'simple (QuickUMLS not available)',
    }


# --- MAUVE feature extraction ---

def _extract_last_token_features(model, tokenizer, texts, device, batch_size=16, max_length=1024):
    """
    Extract sequence-level features from a causal LM by taking the hidden state
    of the last non-padding token.

    For causal (left-to-right) models such as GPT-2, the last token's hidden
    state has attended over the full preceding context and is the natural
    sequence summary. Mean-pooling over a causal model mixes early tokens
    (which have seen little context) with later ones, degrading embedding
    quality — especially for long clinical notes.

    Args:
        model:      AutoModel in eval mode.
        tokenizer:  Matching tokenizer with pad_token set.
        texts:      List of strings.
        device:     torch.device.
        batch_size: Texts per forward pass.
        max_length: Truncation length in tokens.

    Returns:
        np.ndarray of shape (len(texts), hidden_size), dtype float32.
    """
    import torch

    all_features = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
            hidden = out.last_hidden_state          # (B, T, H)

        # attention_mask: 1 = real token, 0 = padding.
        # Sum gives count of real tokens; subtract 1 for 0-based last index.
        attention_mask = enc['attention_mask']      # (B, T)
        last_idx = (attention_mask.sum(dim=1) - 1).clamp(min=0)  # (B,)

        # Gather hidden state at each sequence's last real token position.
        idx_exp = last_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, hidden.size(-1))
        pooled = hidden.gather(dim=1, index=idx_exp).squeeze(1)   # (B, H)

        all_features.append(pooled.cpu().float().numpy())

        if (i // batch_size) % 20 == 0:
            log.info(f"  Featurised {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.concatenate(all_features, axis=0)


def _extract_mean_pool_features(model, tokenizer, texts, device, batch_size=16, max_length=512):
    """
    Extract sequence-level features from a bidirectional encoder via mean pooling.
    Appropriate for BERT-family models where all positions have full context.
    """
    import torch

    all_features = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
            hidden = out.last_hidden_state          # (B, T, H)
            mask = enc['attention_mask'].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        all_features.append(pooled.cpu().float().numpy())

        if (i // batch_size) % 20 == 0:
            log.info(f"  Featurised {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.concatenate(all_features, axis=0)


# --- MAUVE ---

def compute_mauve_score(syn_texts, real_texts, model_name=None, device_id=0):
    """
    Compute MAUVE score between synthetic and real text distributions.

    Follows the paper's algorithm:
      1. Embed texts with an external LM.
         - Causal models (GPT-2, LLaMA, etc.): last non-padding token pooling.
         - Encoder models (BERT, RoBERTa, etc.): masked mean pooling.
      2. Pass raw embeddings directly to mauve.compute_mauve() — NO preprocessing.
         The package performs joint k-means quantisation internally; the cluster
         geometry must reflect natural distances in embedding space.
      3. Return the area under the divergence curve (higher = more similar).

    Fixes vs. original version:
      - Causal models now use last-token pooling (not mean pooling). Mean pooling
        over a causal LM mixes low-quality early-token representations into the
        sequence embedding, which is what caused scores to collapse — not the
        absence of L2 normalisation.
      - L2 normalisation removed. Collapsing all embeddings onto a unit hypersphere
        destroys magnitude information and corrupts k-means cluster geometry,
        causing scores to degenerate regardless of actual text quality.
      - Small-sample warning added: MAUVE is upward-biased and high-variance
        below ~1000 samples (MAUVE paper Appendix D.3, Figure 7).
    """
    try:
        import mauve as mauve_lib
        import torch
        from transformers import AutoTokenizer, AutoModel, AutoConfig
    except ImportError:
        log.warning("mauve-text or transformers not installed — skipping MAUVE score")
        return {'mauve_score': None, 'error': 'mauve-text not installed'}

    if len(syn_texts) < 1000 or len(real_texts) < 1000:
        log.warning(
            f"Small sample size (syn={len(syn_texts)}, real={len(real_texts)}). "
            "MAUVE is upward-biased and high-variance below ~1000 samples "
            "(MAUVE paper Appendix D.3, Figure 7). Treat this result with caution."
        )

    log.info("Computing MAUVE score...")
    try:
        max_texts = min(5000, len(syn_texts), len(real_texts))
        import random
        random.seed(42)
        syn_sample  = random.sample(syn_texts,  max_texts) if len(syn_texts)  > max_texts else syn_texts
        real_sample = random.sample(real_texts, max_texts) if len(real_texts) > max_texts else real_texts

        device = torch.device(
            f'cuda:{device_id}' if device_id >= 0 and torch.cuda.is_available() else 'cpu'
        )
        featurizer_path = model_name or 'gpt2-large'
        log.info(f"Loading featuriser from {featurizer_path} on {device}")

        tokenizer = AutoTokenizer.from_pretrained(featurizer_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        feat_model = AutoModel.from_pretrained(
            featurizer_path,
            torch_dtype=torch.float32,
            local_files_only=True,
        ).to(device)
        feat_model.eval()

        # Determine pooling strategy from model architecture.
        # Causal (decoder-only) models: last-token pooling.
        # Bidirectional encoders: mean pooling.
        cfg = AutoConfig.from_pretrained(featurizer_path, local_files_only=True)
        encoder_types = {
            'bert', 'roberta', 'deberta', 'deberta-v2', 'albert',
            'electra', 'distilbert', 'camembert', 'xlm-roberta',
            'longformer', 'bigbird', 'led',
        }
        model_type = getattr(cfg, 'model_type', '').lower()
        is_encoder = model_type in encoder_types
        pooling_strategy = 'mean (encoder)' if is_encoder else 'last-token (causal)'
        log.info(f"Pooling strategy: {pooling_strategy} (model_type='{model_type}')")

        if is_encoder:
            extract_fn = lambda texts: _extract_mean_pool_features(
                feat_model, tokenizer, texts, device, batch_size=16, max_length=512,
            )
        else:
            extract_fn = lambda texts: _extract_last_token_features(
                feat_model, tokenizer, texts, device, batch_size=16, max_length=1024,
            )

        log.info("Extracting real text features...")
        real_features = extract_fn(real_sample)
        log.info("Extracting synthetic text features...")
        syn_features  = extract_fn(syn_sample)

        del feat_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Pass raw embeddings to MAUVE — no normalisation.
        # MAUVE's internal k-means quantisation relies on natural Euclidean
        # distances in the embedding space. L2 normalisation projects all points
        # onto a unit hypersphere, corrupting those distances and causing k-means
        # to produce degenerate clusters, which collapses MAUVE scores toward 0.
        log.info("Running MAUVE clustering and scoring...")
        result = mauve_lib.compute_mauve(
            p_features=real_features,
            q_features=syn_features,
            verbose=True,
            seed=42,
        )
        score = float(result.mauve)
        log.info(f"MAUVE score: {score:.4f}")

        return {
            'mauve_score':       score,
            'n_synthetic':       len(syn_sample),
            'n_real':            len(real_sample),
            'featuriser':        featurizer_path,
            'pooling_strategy':  pooling_strategy,
            'frontier_integral': float(result.frontier_integral),
        }

    except Exception as e:
        log.error(f"MAUVE computation failed: {type(e).__name__}: {e}")
        return {'mauve_score': None, 'error': str(e)}


# =========================================================================
# Tier 3: Medical Coherence (automated)
# =========================================================================

def compute_perplexity_stats(texts, model_path, device, max_length=512):
    """
    Compute perplexity statistics as a proxy for fluency/coherence.
    Lower perplexity under a clinical model = more fluent clinical text.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    log.info(f"Computing perplexity stats under {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    model.eval()

    ppls = []
    for text in tqdm(texts, desc="PPL scoring"):
        if not text.strip():
            ppls.append(float('inf'))
            continue
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
        input_ids = enc['input_ids'].to(device)
        if input_ids.shape[1] <= 1:
            ppls.append(float('inf'))
            continue
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            ppl = math.exp(min(outputs.loss.item(), 100))
            ppls.append(ppl)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    valid_ppls = [p for p in ppls if p < float('inf')]
    if not valid_ppls:
        return {'error': 'all perplexities were inf'}

    return {
        'mean':    float(np.mean(valid_ppls)),
        'median':  float(np.median(valid_ppls)),
        'std':     float(np.std(valid_ppls)),
        'p10':     float(np.percentile(valid_ppls, 10)),
        'p90':     float(np.percentile(valid_ppls, 90)),
        'n_valid': len(valid_ppls),
        'n_inf':   len(ppls) - len(valid_ppls),
    }


# =========================================================================
# Per-ICD-category breakdown
# =========================================================================

def compute_per_category_metrics(syn_records, real_records):
    """
    Break down length and term metrics by ICD category.
    Reveals which categories degrade most under DP.
    """
    log.info("Computing per-ICD-category metrics...")

    syn_by_cat  = {}
    real_by_cat = {}

    for rec in syn_records:
        cat = rec.get('control_codes', {}).get('icd_category', 'Unknown')
        syn_by_cat.setdefault(cat, []).append(rec['synthetic_bhc'])

    for rec in real_records:
        cat = rec.get('control_codes', {}).get('icd_category', 'Unknown')
        real_by_cat.setdefault(cat, []).append(rec['bhc_text'])

    results = {}
    all_cats = sorted(set(list(syn_by_cat.keys()) + list(real_by_cat.keys())))

    for cat in all_cats:
        syn_texts  = syn_by_cat.get(cat, [])
        real_texts = real_by_cat.get(cat, [])

        entry = {'n_synthetic': len(syn_texts), 'n_real': len(real_texts)}

        if syn_texts and real_texts:
            syn_lens  = [len(t.split()) for t in syn_texts]
            real_lens = [len(t.split()) for t in real_texts]
            entry['syn_length_mean']  = float(np.mean(syn_lens))
            entry['real_length_mean'] = float(np.mean(real_lens))
            entry['length_ratio']     = float(np.mean(syn_lens) / max(np.mean(real_lens), 1))

            term_result = compute_term_jaccard(syn_texts, real_texts)
            entry['unary_jaccard'] = term_result['unary_jaccard']

        results[cat] = entry

    return results


# =========================================================================
# Main evaluation
# =========================================================================

def main(args):
    import torch

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Synthetic: {args.synthetic}")
    log.info(f"Real:      {args.real}")
    log.info(f"Output:    {output_dir}")

    # -- Load data --
    syn_records = load_synthetic(args.synthetic, sample=args.sample)
    syn_texts   = [r['synthetic_bhc'] for r in syn_records]

    syn_note_ids = {r['note_id'] for r in syn_records if r.get('note_id', '').strip()}

    if syn_note_ids:
        real_records = load_real(args.real, note_ids=syn_note_ids)
        if len(real_records) < len(syn_records) * 0.5:
            log.warning(
                f"note_id match returned only {len(real_records):,} records "
                f"(expected ~{len(syn_records):,}) — falling back to random sample"
            )
            real_records = load_real(args.real, sample=len(syn_records))
    else:
        real_records = load_real(args.real, sample=len(syn_records))

    real_texts = [r['bhc_text'] for r in real_records]

    log.info(f"Evaluating {len(syn_texts):,} synthetic vs {len(real_texts):,} real BHC sections")

    all_results = {
        'n_synthetic': len(syn_texts),
        'n_real':      len(real_texts),
        'epsilon':     args.epsilon,
    }

    # =================================================================
    # TIER 1: SURFACE FIDELITY
    # =================================================================
    log.info("=" * 60)
    log.info("TIER 1: SURFACE FIDELITY")
    log.info("=" * 60)

    log.info("Computing text length KL divergence...")
    length_result = compute_length_kl(syn_texts, real_texts)
    all_results['length_kl'] = length_result
    log.info(f"  Length KL divergence: {length_result['kl_divergence']:.4f}")
    log.info(f"  Real mean length:    {length_result['real_length']['mean']:.0f} words")
    log.info(f"  Syn mean length:     {length_result['synthetic_length']['mean']:.0f} words")

    log.info("Computing n-gram frequency divergence...")
    unigram_result = compute_ngram_divergence(syn_texts, real_texts, n=1)
    bigram_result  = compute_ngram_divergence(syn_texts, real_texts, n=2)
    all_results['unigram'] = unigram_result
    all_results['bigram']  = bigram_result
    log.info(f"  Unigram KL: {unigram_result['1gram_kl_divergence']:.4f}")
    log.info(f"  Bigram KL:  {bigram_result['2gram_kl_divergence']:.4f}")

    log.info("Computing term overlap (Jaccard)...")
    term_result = compute_term_jaccard(syn_texts, real_texts)
    all_results['term_overlap'] = term_result
    log.info(f"  Unary Jaccard:  {term_result['unary_jaccard']:.4f}")
    log.info(f"  Binary Jaccard: {term_result['binary_jaccard']:.4f}")

    if not args.skip_mauve:
        mauve_result = compute_mauve_score(
            syn_texts, real_texts,
            model_name=args.mauve_model,
            device_id=args.device_id,
        )
        all_results['mauve'] = mauve_result
    else:
        log.info("Skipping MAUVE (--skip_mauve)")
        all_results['mauve'] = {'mauve_score': None, 'skipped': True}

    # =================================================================
    # TIER 3: MEDICAL COHERENCE (automated)
    # =================================================================
    log.info("=" * 60)
    log.info("TIER 3: MEDICAL COHERENCE (automated)")
    log.info("=" * 60)

    device = torch.device(
        f'cuda:{args.device_id}' if args.device_id >= 0 and torch.cuda.is_available() else 'cpu'
    )

    if args.ppl_model:
        log.info("Computing perplexity-based coherence scores...")
        syn_ppl  = compute_perplexity_stats(syn_texts,  args.ppl_model, device)
        real_ppl = compute_perplexity_stats(real_texts, args.ppl_model, device)
        all_results['coherence_ppl'] = {
            'synthetic': syn_ppl,
            'real':      real_ppl,
            'model':     args.ppl_model,
            'note':      'Lower PPL = more fluent/coherent under the reference model',
        }
        log.info(
            f"  Synthetic PPL: mean={syn_ppl.get('mean', 'N/A'):.1f}  "
            f"median={syn_ppl.get('median', 'N/A'):.1f}"
        )
        log.info(
            f"  Real PPL:      mean={real_ppl.get('mean', 'N/A'):.1f}  "
            f"median={real_ppl.get('median', 'N/A'):.1f}"
        )
    else:
        log.info("Skipping PPL coherence (no --ppl_model specified)")
        all_results['coherence_ppl'] = {'skipped': True}

    # =================================================================
    # PER-CATEGORY BREAKDOWN
    # =================================================================
    log.info("=" * 60)
    log.info("PER-ICD-CATEGORY BREAKDOWN")
    log.info("=" * 60)

    per_cat = compute_per_category_metrics(syn_records, real_records)
    all_results['per_category'] = per_cat

    cats_with_jaccard = [
        (cat, v.get('unary_jaccard', 0))
        for cat, v in per_cat.items()
        if v.get('unary_jaccard') is not None and v['n_synthetic'] >= 10
    ]
    if cats_with_jaccard:
        cats_with_jaccard.sort(key=lambda x: x[1])
        log.info("Lowest term overlap (most degraded by DP):")
        for cat, jac in cats_with_jaccard[:3]:
            log.info(f"  {cat[:50]:50s}  Jaccard={jac:.3f}  n={per_cat[cat]['n_synthetic']}")
        log.info("Highest term overlap (best preserved):")
        for cat, jac in cats_with_jaccard[-3:]:
            log.info(f"  {cat[:50]:50s}  Jaccard={jac:.3f}  n={per_cat[cat]['n_synthetic']}")

    # =================================================================
    # SAVE RESULTS
    # =================================================================
    results_path = output_dir / 'eval_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"Results saved to {results_path}")

    # =================================================================
    # SUMMARY
    # =================================================================
    log.info("=" * 60)
    log.info("EVALUATION SUMMARY")
    log.info("=" * 60)
    log.info(f"  Epsilon:           {args.epsilon}")
    log.info(f"  N synthetic:       {len(syn_texts):,}")
    log.info(f"  N real:            {len(real_texts):,}")
    log.info(f"  Length KL:         {length_result['kl_divergence']:.4f}")
    log.info(f"  Unigram KL:       {unigram_result['1gram_kl_divergence']:.4f}")
    log.info(f"  Bigram KL:        {bigram_result['2gram_kl_divergence']:.4f}")
    log.info(f"  Unary Jaccard:    {term_result['unary_jaccard']:.4f}")
    log.info(f"  Binary Jaccard:   {term_result['binary_jaccard']:.4f}")
    mauve_val = all_results['mauve'].get('mauve_score')
    log.info(f"  MAUVE:            {mauve_val if mauve_val is not None else 'N/A'}")
    log.info("=" * 60)
    log.info("DONE")


# =========================================================================
# Dependency checklist
# =========================================================================

def check_dependencies():
    deps = {}

    try:
        import mauve
        deps['mauve-text'] = 'OK'
    except ImportError:
        deps['mauve-text'] = 'MISSING — pip install mauve-text'

    try:
        from quickumls import QuickUMLS
        deps['QuickUMLS'] = 'OK'
    except ImportError:
        deps['QuickUMLS'] = 'MISSING — using simple term extraction fallback'

    try:
        from sklearn.neighbors import LocalOutlierFactor
        deps['scikit-learn'] = 'OK'
    except ImportError:
        deps['scikit-learn'] = 'MISSING — pip install scikit-learn'

    try:
        import scipy
        deps['scipy'] = 'OK'
    except ImportError:
        deps['scipy'] = 'MISSING — pip install scipy'

    try:
        import torch
        deps['torch'] = f'OK ({torch.__version__})'
    except ImportError:
        deps['torch'] = 'MISSING'

    print("\n=== Evaluation Dependencies ===")
    for dep, status in deps.items():
        marker = "✓" if status.startswith('OK') else "✗"
        print(f"  {marker}  {dep:20s}  {status}")

    print("\n=== Models needed on cluster ===")
    print("  - Base model for PPL coherence (e.g., Llama-3.2-1B-Instruct)")
    print("  - (Optional) BioMistral-7B for MAUVE featurisation")
    print("  - (Optional) UniEval weights for Tier 3 coherence")
    print("  - (Optional) Clinical-Longformer for TSTR (separate script)")
    print("  - (Optional) Asclepius-Llama3-8B for clinical PPL")


# =========================================================================
# CLI
# =========================================================================

if __name__ == '__main__':
    import torch

    parser = argparse.ArgumentParser(
        description='Evaluate synthetic BHC quality',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--synthetic', required=True,
                        help='Path to synthetic_bhc.jsonl from 03_generate.py')
    parser.add_argument('--real', default='./data/train.jsonl',
                        help='Path to train.jsonl (real BHC)')
    parser.add_argument('--output', default='./eval',
                        help='Output directory')
    parser.add_argument('--sample', type=int, default=None,
                        help='Evaluate only N records. '
                             'MAUVE is upward-biased and high-variance below ~1000 samples.')
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Epsilon value (for labeling only)')

    parser.add_argument('--skip_mauve', action='store_true',
                        help='Skip MAUVE computation')
    parser.add_argument('--mauve_model', default=None,
                        help='Path/name of featuriser model (default: gpt2-large). '
                             'Causal models use last-token pooling; encoders use mean pooling.')
    parser.add_argument('--device_id', type=int, default=0,
                        help='CUDA device index (-1 for CPU)')

    parser.add_argument('--ppl_model', default=None,
                        help='Path to causal LM for PPL-based coherence scoring')
    parser.add_argument('--check_deps', action='store_true',
                        help='Check available dependencies and exit')

    args = parser.parse_args()

    if args.check_deps:
        check_dependencies()
        sys.exit(0)

    main(args)
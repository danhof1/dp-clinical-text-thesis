"""
03_generate.py
==============
Phase 4, Step 08 — Generate synthetic BHC sections from a DP-trained checkpoint.

For each control code prefix, generates k candidate BHC completions and selects
the lowest-perplexity candidate under a reference LLM (DP Quality Maximiser).
Post-processing preserves the DP guarantee (Dwork & Roth 2014, Post-Processing Theorem).

Follows Term2Note (Wu et al. 2025) generation protocol:
    - k=4 candidates per prefix
    - Perplexity-based selection using a domain-matched reference LLM
    - Sentence length filter (max 2181 chars, Term2Note Appendix H)
    - temp=0.1, top_p=1.0, repetition_penalty=1.2

PATCH v2 — incremental writes + resume:
    Previous version buffered all output in memory and wrote at the end.
    Jobs killed mid-run lost all progress.

    Fix: output file is opened at job start in append mode. Each record is
    written immediately after generation. On restart, already-completed
    note_ids are loaded and skipped — the job resumes from where it stopped.

    generation_stats.json field naming fix:
        total_generated  = records written to disk (kept)
        failed_filter    = candidates rejected by sentence filter
        Total candidates generated = total_generated * k + failed_filter * k (approx)

Inputs:
    - DP-trained checkpoint directory (contains LoRA adapters + tokenizer)
    - Base model path (Llama-3.2-1B-Instruct)
    - Control code prefixes (extracted from train.jsonl or a separate test set)

Outputs:
    - synthetic_bhc.jsonl — one record per prefix with selected BHC
    - generation_stats.json — summary statistics

Usage:
    # Generate from epsilon=4 checkpoint
    python 03_generate.py \\
        --checkpoint ./models/llama_dp_eps4.0_YYYYMMDD_HHMM/final \\
        --base_model /fs1/shared/model/llm/Llama-3.2-1B-Instruct \\
        --data ./data/train.jsonl \\
        --output ./generated/eps4 \\
        --n_generate 5000 \\
        --k 4

    # Resume an interrupted run (same command — auto-detects existing output)
    python 03_generate.py \\
        --checkpoint ./models/llama_dp_eps4.0_YYYYMMDD_HHMM/final \\
        --base_model /fs1/shared/model/llm/Llama-3.2-1B-Instruct \\
        --data ./data/train.jsonl \\
        --output ./generated/eps4 \\
        --n_generate 5000 \\
        --k 4
"""

import os
import sys
import json
import math
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from peft import PeftModel

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


def gpu_mem():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        resrv = torch.cuda.memory_reserved() / 1e9
        return f"GPU: {alloc:.1f}GB / {resrv:.1f}GB"
    return "GPU: N/A"


# =========================================================================
# Control code extraction
# =========================================================================

def extract_control_prefix(text):
    """
    Given a full training text like:
        "Note Type: Discharge Summary | ICD Category: X | Age Group: Y | Sex: Z:\n[BHC]"
    Returns just the control code prefix (everything up to and including the colon+newline).
    """
    idx = text.find(':\n')
    if idx == -1:
        idx = text.rfind(':')
        if idx == -1:
            return text[:100]
        return text[:idx + 1] + '\n'
    return text[:idx + 1] + '\n'


def load_prefixes(jsonl_path, n_generate=0, n_per_category=0, seed=42):
    """
    Load control code prefixes from train.jsonl.

    Stratified generation (Rosenblatt et al. arXiv 2024):
        If n_per_category > 0, sample exactly n_per_category prefixes per
        ICD category. This corrects the disparate impact of DP-SGD on minority
        classes in synthetic output (Bagdasaryan et al. NeurIPS 2019) by
        ensuring equal representation across all ICD categories.

        Categories with fewer than n_per_category records contribute all
        available records — no oversampling, which would introduce duplicates
        and could weaken DP guarantees.

        If n_per_category == 0, falls back to random sampling of n_generate
        total records (original behaviour).

    Args:
        jsonl_path:      path to train.jsonl
        n_generate:      total prefixes to sample if not using stratified mode
        n_per_category:  prefixes per ICD category (0 = disabled)
        seed:            random seed

    Returns:
        List of dicts with prefix, control_codes, note_id, bhc_text
    """
    log.info(f"Loading prefixes from {jsonl_path}")
    all_records = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            prefix = extract_control_prefix(rec['text'])
            all_records.append({
                'note_id':       rec.get('note_id', ''),
                'prefix':        prefix,
                'control_codes': rec.get('control_codes', {}),
                'bhc_text':      rec.get('bhc_text', ''),
            })

    log.info(f"Loaded {len(all_records):,} total records")
    random.seed(seed)

    if n_per_category > 0:
        # ── Stratified sampling ────────────────────────────────────────────
        log.info(f"Stratified mode: {n_per_category} prefixes per ICD category")
        log.info("(Bagdasaryan et al. NeurIPS 2019; Rosenblatt et al. arXiv 2024)")

        by_cat = {}
        for rec in all_records:
            cat = rec['control_codes'].get('icd_category', 'Unknown')
            by_cat.setdefault(cat, []).append(rec)

        records = []
        n_capped = 0
        for cat in sorted(by_cat.keys()):
            cat_records = by_cat[cat]
            available   = len(cat_records)
            n_sample    = min(n_per_category, available)
            sampled     = random.sample(cat_records, n_sample)
            records.extend(sampled)
            capped = available < n_per_category
            if capped:
                n_capped += 1
            log.info(f"  {cat[:52]:52s}  avail={available:>6,}  "
                     f"sampled={n_sample:>4}{'  ⚠ CAPPED' if capped else ''}")

        log.info(f"Stratified total: {len(records):,} prefixes across "
                 f"{len(by_cat)} categories  ({n_capped} capped due to small size)")

    elif n_generate > 0 and n_generate < len(all_records):
        # ── Random sampling (original behaviour) ──────────────────────────
        records = random.sample(all_records, n_generate)
        log.info(f"Random sample: {n_generate:,} prefixes")

    else:
        records = all_records
        log.info(f"Using all {len(records):,} prefixes")

    # Final distribution summary
    icd_dist = Counter(r['control_codes'].get('icd_category', 'Unknown')
                       for r in records)
    log.info(f"Final ICD distribution ({len(icd_dist)} categories, "
             f"{len(records):,} total):")
    for cat, count in icd_dist.most_common():
        log.info(f"  {cat[:55]:55s}  {count:,}")

    return records


# =========================================================================
# Resume logic
# =========================================================================

def load_completed_ids(output_path):
    """
    Read already-completed note_ids from an existing output file.
    Returns a set of note_id strings.
    """
    if not output_path.exists():
        return set()

    completed = set()
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                nid = rec.get('note_id', '')
                if nid:
                    completed.add(nid)
            except json.JSONDecodeError:
                continue  # skip malformed lines

    log.info(f"Resume: found {len(completed):,} already-completed records in {output_path}")
    return completed


# =========================================================================
# Perplexity computation
# =========================================================================

def compute_perplexity(text, model, tokenizer, device, max_length=2048):
    """
    Compute perplexity of text under the given model.
    Lower = more fluent/likely under the model.
    """
    encodings = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=max_length,
    )
    input_ids = encodings['input_ids'].to(device)

    if input_ids.shape[1] <= 1:
        return float('inf')

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        ppl = math.exp(outputs.loss.item())

    return ppl


def compute_perplexity_batch(texts, model, tokenizer, device, max_length=2048):
    """
    Compute per-sequence perplexity for a batch of texts in one forward pass.
    Uses right-padding so causal mask prevents real tokens from seeing pad tokens,
    producing identical logits to the unbatched case.
    """
    results = [float('inf')] * len(texts)
    valid = [(i, t) for i, t in enumerate(texts) if t.strip()]
    if not valid:
        return results

    valid_indices, valid_texts = zip(*valid)

    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'right'
    encodings = tokenizer(
        list(valid_texts),
        return_tensors='pt',
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    tokenizer.padding_side = orig_padding_side

    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    if input_ids.shape[1] <= 1:
        return results

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().float()

    token_losses = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none',
    ).view(shift_logits.size(0), -1)

    token_losses = token_losses * shift_mask
    seq_lengths = shift_mask.sum(dim=1)
    seq_losses = token_losses.sum(dim=1) / seq_lengths.clamp(min=1)

    for batch_pos, orig_idx in enumerate(valid_indices):
        if seq_lengths[batch_pos] > 0:
            results[orig_idx] = math.exp(seq_losses[batch_pos].item())

    return results


# =========================================================================
# Sentence length filter (Term2Note Appendix H)
# =========================================================================

MAX_SENTENCE_CHARS = 2181

def passes_sentence_filter(text):
    """
    Reject candidates with any sentence longer than 2181 chars.
    Term2Note Appendix H: this threshold aligns best with human quality annotations.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    for sent in sentences:
        if len(sent) > MAX_SENTENCE_CHARS:
            return False
    return True


# =========================================================================
# Generation
# =========================================================================

def generate_candidates(prefix, model, tokenizer, device, k=4,
                        max_new_tokens=1024, temperature=0.1,
                        top_p=1.0, repetition_penalty=1.2):
    """
    Generate k candidate BHC completions for a given control code prefix.
    Parameters follow Term2Note Appendix E.
    Uses num_return_sequences=k for batched generation in a single call.
    """
    input_ids = tokenizer.encode(prefix, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            num_return_sequences=k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prefix_len = input_ids.shape[1]
    candidates = []
    for seq in outputs:
        generated_ids = seq[prefix_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        candidates.append(generated_text.strip())

    return candidates


# =========================================================================
# Main generation pipeline
# =========================================================================

def run_generation(args):
    set_seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'synthetic_bhc.jsonl'
    stats_path  = output_dir / 'generation_stats.json'

    log.info(f"Output: {output_dir}")
    log.info(f"Checkpoint: {args.checkpoint}")
    log.info(f"Base model: {args.base_model}")
    log.info(f"k (candidates per prefix): {args.k}")
    log.info(f"Max new tokens: {args.max_new_tokens}")
    log.info(f"{gpu_mem()}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # -- Load generation model (base + LoRA) --
    log.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    log.info(f"Base model loaded. {gpu_mem()}")

    if args.no_lora:
        # Data 0: base model as-is, no fine-tuning, no LoRA adapters
        log.info("--no_lora set: using base model directly (Data 0 — no fine-tuning)")
        gen_model = base_model.to(device)
        gen_model.eval()
        gen_tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    else:
        log.info(f"Loading LoRA adapters from {args.checkpoint}...")
        gen_model = PeftModel.from_pretrained(base_model, args.checkpoint)
        gen_model = gen_model.merge_and_unload()
        gen_model = gen_model.to(device)
        gen_model.eval()
        gen_tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)

    log.info(f"Generation model ready. {gpu_mem()}")
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

    # -- Load perplexity reference model --
    ppl_model_path = args.ppl_model or args.base_model
    if ppl_model_path == args.base_model:
        log.info("Using base model (pre-LoRA) as perplexity reference")
        ppl_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to(device)
    else:
        log.info(f"Loading perplexity reference model from {ppl_model_path}")
        ppl_model = AutoModelForCausalLM.from_pretrained(
            ppl_model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to(device)
    ppl_model.eval()

    ppl_tokenizer = AutoTokenizer.from_pretrained(ppl_model_path, local_files_only=True)
    if ppl_tokenizer.pad_token is None:
        ppl_tokenizer.pad_token = ppl_tokenizer.eos_token

    log.info(f"Perplexity model ready. {gpu_mem()}")

    # -- Load prefixes (stratified or random) --
    all_prefixes = load_prefixes(
        args.data,
        n_generate=args.n_generate,
        n_per_category=args.n_per_category,
        seed=args.seed,
    )

    # =========================================================================
    # RESUME LOGIC — skip already-completed records
    # =========================================================================
    completed_ids = load_completed_ids(output_path)
    if completed_ids:
        prefixes = [p for p in all_prefixes if p['note_id'] not in completed_ids]
        log.info(f"Resuming: {len(completed_ids):,} done, {len(prefixes):,} remaining "
                 f"(of {len(all_prefixes):,} total)")
    else:
        prefixes = all_prefixes
        log.info(f"Fresh run: {len(prefixes):,} prefixes to generate")

    if not prefixes:
        log.info("All records already generated. Nothing to do.")
        return

    # =========================================================================
    # GENERATE — write each record immediately after completion
    # =========================================================================
    log.info("=" * 60)
    log.info(f"GENERATING {len(prefixes):,} synthetic BHC sections (k={args.k})")
    log.info("=" * 60)

    stats = {
        'n_written':      len(completed_ids),  # already on disk
        'failed_filter':  0,
        'ppl_scores':     [],
    }

    # Open in append mode — safe for both fresh runs and resumes
    with open(output_path, 'a') as out_f:
        for i, rec in enumerate(tqdm(prefixes, desc="Generating")):
            prefix = rec['prefix']

            # Generate k candidates
            candidates = generate_candidates(
                prefix=prefix,
                model=gen_model,
                tokenizer=gen_tokenizer,
                device=device,
                k=args.k,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )

            # Score all candidates' perplexity in one batched forward pass
            ppls = compute_perplexity_batch(candidates, ppl_model, ppl_tokenizer, device)
            scored = list(zip(candidates, ppls))

            # Select lowest perplexity (DP Quality Maximiser)
            scored.sort(key=lambda x: x[1])
            best_text, best_ppl = scored[0]

            # Apply sentence length filter — try next candidates if best fails
            passed = passes_sentence_filter(best_text)
            if not passed:
                stats['failed_filter'] += 1
                for cand_text, cand_ppl in scored[1:]:
                    if passes_sentence_filter(cand_text):
                        best_text = cand_text
                        best_ppl  = cand_ppl
                        passed    = True
                        break

            stats['ppl_scores'].append(best_ppl)

            result = {
                'note_id':       rec['note_id'],
                'control_codes': rec['control_codes'],
                'prefix':        prefix,
                'synthetic_bhc': best_text,
                'perplexity':    best_ppl,
                'passed_filter': passed,
                'original_bhc':  rec['bhc_text'],
                'n_candidates':  len(candidates),
            }

            # 2026-04-28: Save all k candidates when --save_candidates is set.
            # Reason: offline re-ranking (term-aware filter) without regeneration.
            if args.save_candidates:
                result['candidates'] = [
                    {'text': cand, 'ppl': ppl_val}
                    for cand, ppl_val in scored
                ]

            # ── INCREMENTAL WRITE — happens immediately, every record ──────
            out_f.write(json.dumps(result) + '\n')
            out_f.flush()   # flush to OS buffer
            # ─────────────────────────────────────────────────────────────

            stats['n_written'] += 1

            if (i + 1) <= 5 or (i + 1) % 100 == 0:
                log.info(
                    f"  [{i+1}/{len(prefixes)}] total_written={stats['n_written']:,}  "
                    f"ppl={best_ppl:.1f}  "
                    f"len={len(best_text)}  "
                    f"filter={'PASS' if passed else 'FAIL'}  "
                    f"{gpu_mem()}"
                )

    # =========================================================================
    # Stats — count total lines in output file for ground truth
    # =========================================================================
    total_on_disk = sum(1 for _ in open(output_path))

    ppl_arr = [p for p in stats['ppl_scores'] if p < float('inf')]
    if ppl_arr:
        ppl_arr.sort()
        n = len(ppl_arr)
        summary_stats = {
            'total_generated':  total_on_disk,   # records kept on disk (this run + previous)
            'generated_this_run': stats['n_written'] - len(completed_ids),
            'resumed_from':     len(completed_ids),
            'failed_filter':    stats['failed_filter'],  # candidates rejected by sentence filter
            'k':                args.k,
            'epsilon':          args.epsilon,
            'checkpoint':       str(args.checkpoint),
            'ppl_reference':    ppl_model_path,
            'perplexity': {
                'mean':   np.mean(ppl_arr),
                'median': ppl_arr[n // 2],
                'p10':    ppl_arr[int(n * 0.1)],
                'p90':    ppl_arr[int(n * 0.9)],
                'min':    ppl_arr[0],
                'max':    ppl_arr[-1],
            },
            'bhc_length_this_run': {
                'mean':   np.mean([len(r) for r in
                          [json.loads(l)['synthetic_bhc']
                           for l in open(output_path)][-stats['n_written']:]])
                          if stats['n_written'] > 0 else 0,
            },
            'generation_params': {
                'temperature':        args.temperature,
                'top_p':              args.top_p,
                'repetition_penalty': args.repetition_penalty,
                'max_new_tokens':     args.max_new_tokens,
            },
        }
    else:
        summary_stats = {'error': 'all perplexities were inf'}

    with open(stats_path, 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)

    log.info(f"Stats saved to {stats_path}")
    if 'perplexity' in summary_stats:
        log.info(f"  Total on disk:   {total_on_disk:,}")
        log.info(f"  This run:        {summary_stats['generated_this_run']:,}")
        log.info(f"  Resumed from:    {summary_stats['resumed_from']:,}")
        log.info(f"  Failed filter:   {stats['failed_filter']:,}")
        log.info(f"  PPL: mean={summary_stats['perplexity']['mean']:.1f}  "
                 f"median={summary_stats['perplexity']['median']:.1f}  "
                 f"p90={summary_stats['perplexity']['p90']:.1f}")
    log.info("DONE")


# =========================================================================
# CLI
# =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate synthetic BHC from DP-trained checkpoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument('--checkpoint', required=True,
                        help='Path to DP-trained checkpoint dir (LoRA adapters)')
    parser.add_argument('--base_model', required=True,
                        help='Path to base Llama-3.2-1B-Instruct')

    # Data
    parser.add_argument('--data', default='./data/train.jsonl',
                        help='Path to train.jsonl (source of control code prefixes)')
    parser.add_argument('--output', default='./generated',
                        help='Output directory')
    parser.add_argument('--n_generate', type=int, default=5000,
                        help='Number of prefixes to generate for (0 = all). '
                             'Ignored if --n_per_category is set.')
    parser.add_argument('--n_per_category', type=int, default=500,
                        help='Stratified mode: generate this many per ICD category. '
                             'Set to 0 to use --n_generate instead. '
                             'Default 500 gives 10,000 total across 20 categories. '
                             '(Bagdasaryan et al. 2019; Rosenblatt et al. 2024)')

    # Generation params (Term2Note Appendix E defaults)
    parser.add_argument('--k', type=int, default=4,
                        help='Candidates per prefix for quality maximiser')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                        help='Max tokens to generate per candidate')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Nucleus sampling threshold')
    parser.add_argument('--repetition_penalty', type=float, default=1.2,
                        help='Repetition penalty')

    # Perplexity model
    parser.add_argument('--ppl_model', default=None,
                        help='Path to perplexity reference model (default: base model)')

    # Metadata
    parser.add_argument('--no_lora', action='store_true',
                        help='Use base model directly without LoRA adapters. '
                             'Required for Data 0 (base model generation). '
                             'When set, --checkpoint is ignored for the generator '
                             'and --base_model is used as both generator and tokenizer source.')
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Epsilon value (for stats file labeling only)')
    parser.add_argument('--seed', type=int, default=42)

    # 2026-04-28: Added --save_candidates to persist all k candidates per prompt.
    # Reason: enables offline filter experimentation (e.g. term-aware re-ranking)
    # without re-running expensive generation. See 03.3_rerank_candidates.py.
    parser.add_argument('--save_candidates', action='store_true',
                        help='Save all k candidates with PPL scores per record.')

    args = parser.parse_args()
    run_generation(args)
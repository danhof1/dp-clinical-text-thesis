"""
03a_generate_base.py
====================
Generate synthetic BHC sections directly from the raw base model —
NO fine-tuning, NO DP, NO LoRA adapters.
Produces Synthetic Data 0.

Role in the 4-dataset framework:
    Data 0: Base model only, no fine-tuning  <-- THIS SCRIPT
    Data 1: Base + SGD (no DP)               (02b_train_sgd.py -> 03_generate.py)
    Data 2: Base + DP-SGD at eps             (02_train_dp.py   -> 03_generate.py)
    Data 3: Base + Unlearn + DP-SGD          (MRP -> 02_train_dp.py -> 03_generate.py)

Data 0 establishes the floor — what the base model can produce with
just a control code prefix and no domain fine-tuning. Comparing Data 0
vs Data 1 shows the effect of SGD fine-tuning alone. Comparing Data 1
vs Data 2 shows the effect of DP noise alone.

Key difference from 03_generate.py:
    - Loads only the base model (no PeftModel.from_pretrained)
    - No --checkpoint argument required
    - Everything else identical: same stratified prefix selection,
      same PPL-based quality maximiser, same incremental write + resume logic

Usage:
    python 03a_generate_base.py \\
        --base_model /fs1/shared/model/llm/Llama-3.2-1B-Instruct \\
        --data ./data/train.jsonl \\
        --output ./generated/data0_base

    # MTSamples track
    python 03a_generate_base.py \\
        --base_model /fs1/shared/model/llm/Llama-3.2-1B-Instruct \\
        --data ./data/mtsamples/train.jsonl \\
        --output ./generated/mtsamples/data0_base

    # With Asclepius as PPL reference (preferred once installed)
    python 03a_generate_base.py \\
        --base_model /fs1/shared/model/llm/Llama-3.2-1B-Instruct \\
        --ppl_model /fs1/shared/model/llm/Asclepius-Llama3-8B \\
        --data ./data/train.jsonl \\
        --output ./generated/data0_base
"""

import sys
import json
import math
import random
import logging
import argparse
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)


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
# Control code extraction (identical to 03_generate.py)
# =========================================================================

def extract_control_prefix(text):
    idx = text.find(':\n')
    if idx == -1:
        idx = text.rfind(':')
        if idx == -1:
            return text[:100]
        return text[:idx + 1] + '\n'
    return text[:idx + 1] + '\n'


# =========================================================================
# Stratified prefix loading (identical to 03_generate.py)
# =========================================================================

def load_prefixes(jsonl_path, n_per_category=500, n_generate=0, seed=42):
    """
    Load and optionally stratify control code prefixes.
    Identical to 03_generate.py — see that script for full documentation.
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
        log.info(f"Stratified mode: {n_per_category} per ICD category")
        by_cat = {}
        for rec in all_records:
            cat = rec['control_codes'].get('icd_category', 'Unknown')
            by_cat.setdefault(cat, []).append(rec)

        records = []
        n_capped = 0
        for cat in sorted(by_cat.keys()):
            available = len(by_cat[cat])
            n_sample  = min(n_per_category, available)
            records.extend(random.sample(by_cat[cat], n_sample))
            if available < n_per_category:
                n_capped += 1
                log.info(f"  ⚠ CAPPED  {cat[:52]:52s}  "
                         f"avail={available}  sampled={n_sample}")
            else:
                log.info(f"           {cat[:52]:52s}  sampled={n_sample}")

        log.info(f"Stratified total: {len(records):,} prefixes  "
                 f"({n_capped} categories capped)")

    elif n_generate > 0 and n_generate < len(all_records):
        records = random.sample(all_records, n_generate)
        log.info(f"Random sample: {n_generate:,} prefixes")
    else:
        records = all_records
        log.info(f"Using all {len(records):,} prefixes")

    icd_dist = Counter(r['control_codes'].get('icd_category', 'Unknown')
                       for r in records)
    log.info(f"Final ICD distribution ({len(icd_dist)} categories):")
    for cat, count in icd_dist.most_common():
        log.info(f"  {cat[:55]:55s}  {count:,}")

    return records


# =========================================================================
# Resume logic (identical to 03_generate.py)
# =========================================================================

def load_completed_ids(output_path):
    if not output_path.exists():
        return set()
    completed = set()
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                nid = json.loads(line).get('note_id', '')
                if nid:
                    completed.add(nid)
            except json.JSONDecodeError:
                continue
    log.info(f"Resume: {len(completed):,} already-completed records in {output_path}")
    return completed


# =========================================================================
# Perplexity scoring (identical to 03_generate.py)
# =========================================================================

def compute_perplexity(text, model, tokenizer, device, max_length=2048):
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    input_ids = enc['input_ids'].to(device)
    if input_ids.shape[1] <= 1:
        return float('inf')
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        return math.exp(outputs.loss.item())


# =========================================================================
# Sentence length filter (Term2Note Appendix H)
# =========================================================================

MAX_SENTENCE_CHARS = 2181

def passes_sentence_filter(text):
    import re
    for sent in re.split(r'(?<=[.!?])\s+', text.strip()):
        if len(sent) > MAX_SENTENCE_CHARS:
            return False
    return True


# =========================================================================
# Generation
# =========================================================================

def generate_candidates(prefix, model, tokenizer, device, k=4,
                        max_new_tokens=1024, temperature=0.1,
                        top_p=1.0, repetition_penalty=1.2):
    input_ids = tokenizer.encode(prefix, return_tensors='pt').to(device)
    candidates = []
    for _ in range(k):
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids  = output[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        candidates.append(generated_text.strip())
    return candidates


# =========================================================================
# Main
# =========================================================================

def run_generation(args):
    set_seed(args.seed)

    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'synthetic_bhc.jsonl'
    stats_path  = output_dir / 'generation_stats.json'

    log.info(f"Output: {output_dir}")
    log.info(f"Base model: {args.base_model}")
    log.info(f"DATA PATH: Base model only (no fine-tuning) -> Synthetic Data 0")
    log.info(f"k={args.k}  max_new_tokens={args.max_new_tokens}")
    log.info(f"{gpu_mem()}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # -- Load base model (no LoRA, no fine-tuning) --
    log.info("Loading base model (no fine-tuning applied)...")
    gen_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    gen_model.eval()
    log.info(f"Base model loaded. {gpu_mem()}")

    gen_tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True,
    )
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

    # -- Load PPL reference model --
    ppl_model_path = args.ppl_model or args.base_model
    if ppl_model_path == args.base_model:
        log.info("Using base model as PPL reference (same model — no bias concern "
                 "since generation model is also unmodified base)")
        ppl_model     = gen_model
        ppl_tokenizer = gen_tokenizer
    else:
        log.info(f"Loading PPL reference model from {ppl_model_path}")
        ppl_model = AutoModelForCausalLM.from_pretrained(
            ppl_model_path, torch_dtype=torch.bfloat16, local_files_only=True,
        ).to(device)
        ppl_model.eval()
        ppl_tokenizer = AutoTokenizer.from_pretrained(
            ppl_model_path, local_files_only=True,
        )
        if ppl_tokenizer.pad_token is None:
            ppl_tokenizer.pad_token = ppl_tokenizer.eos_token

    log.info(f"PPL model ready. {gpu_mem()}")

    # -- Load prefixes --
    all_prefixes = load_prefixes(
        args.data,
        n_per_category=args.n_per_category,
        n_generate=args.n_generate,
        seed=args.seed,
    )

    # -- Resume --
    completed_ids = load_completed_ids(output_path)
    if completed_ids:
        prefixes = [p for p in all_prefixes if p['note_id'] not in completed_ids]
        log.info(f"Resuming: {len(completed_ids):,} done, {len(prefixes):,} remaining")
    else:
        prefixes = all_prefixes
        log.info(f"Fresh run: {len(prefixes):,} prefixes")

    if not prefixes:
        log.info("All records already generated.")
        return

    # -- Generate with incremental write --
    log.info("=" * 60)
    log.info(f"GENERATING {len(prefixes):,} records from base model (Data 0)")
    log.info("=" * 60)

    stats = {
        'n_written':     len(completed_ids),
        'failed_filter': 0,
        'ppl_scores':    [],
    }

    with open(output_path, 'a') as out_f:
        for i, rec in enumerate(tqdm(prefixes, desc="Generating (base model)")):
            prefix = rec['prefix']

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

            scored = []
            for cand in candidates:
                if not cand.strip():
                    scored.append((cand, float('inf')))
                    continue
                ppl = compute_perplexity(cand, ppl_model, ppl_tokenizer, device)
                scored.append((cand, ppl))

            scored.sort(key=lambda x: x[1])
            best_text, best_ppl = scored[0]

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
                'data_path':     'Data 0: base model, no fine-tuning',
            }

            # Incremental write — no data loss on job kill
            out_f.write(json.dumps(result) + '\n')
            out_f.flush()
            stats['n_written'] += 1

            if (i + 1) <= 5 or (i + 1) % 100 == 0:
                log.info(
                    f"  [{i+1}/{len(prefixes)}] written={stats['n_written']:,}  "
                    f"ppl={best_ppl:.1f}  len={len(best_text)}  "
                    f"filter={'PASS' if passed else 'FAIL'}  {gpu_mem()}"
                )

    # -- Stats --
    total_on_disk = sum(1 for _ in open(output_path))
    ppl_arr = sorted(p for p in stats['ppl_scores'] if p < float('inf'))

    summary = {
        'data_path':            'Data 0: base model, no fine-tuning',
        'total_generated':      total_on_disk,
        'generated_this_run':   stats['n_written'] - len(completed_ids),
        'resumed_from':         len(completed_ids),
        'failed_filter':        stats['failed_filter'],
        'k':                    args.k,
        'epsilon':              'N/A (no training)',
        'base_model':           args.base_model,
        'ppl_reference':        ppl_model_path,
        'perplexity': {
            'mean':   float(np.mean(ppl_arr)) if ppl_arr else None,
            'median': ppl_arr[len(ppl_arr)//2] if ppl_arr else None,
            'p10':    ppl_arr[int(len(ppl_arr)*0.1)] if ppl_arr else None,
            'p90':    ppl_arr[int(len(ppl_arr)*0.9)] if ppl_arr else None,
        } if ppl_arr else {},
        'generation_params': {
            'temperature':        args.temperature,
            'top_p':              args.top_p,
            'repetition_penalty': args.repetition_penalty,
            'max_new_tokens':     args.max_new_tokens,
        },
    }

    with open(stats_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"Stats saved to {stats_path}")
    if ppl_arr:
        log.info(f"  Total on disk:  {total_on_disk:,}")
        log.info(f"  PPL mean:       {summary['perplexity']['mean']:.1f}")
        log.info(f"  PPL median:     {summary['perplexity']['median']:.1f}")
        log.info(f"  Failed filter:  {stats['failed_filter']:,}")
    log.info("DONE")


# =========================================================================
# CLI
# =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate Synthetic Data 0 from raw base model (no fine-tuning)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--base_model', required=True,
                        help='Path to Llama-3.2-1B-Instruct (no LoRA applied)')
    parser.add_argument('--data',    default='./data/train.jsonl',
                        help='Path to train.jsonl (source of control code prefixes)')
    parser.add_argument('--output',  default='./generated/data0_base')

    parser.add_argument('--n_per_category', type=int, default=500,
                        help='Stratified: prefixes per ICD category. '
                             'Set 0 to use --n_generate instead.')
    parser.add_argument('--n_generate',     type=int, default=0,
                        help='Total prefixes if not stratified (0=all)')

    parser.add_argument('--k',                  type=int,   default=4)
    parser.add_argument('--max_new_tokens',     type=int,   default=1024)
    parser.add_argument('--temperature',        type=float, default=0.1)
    parser.add_argument('--top_p',              type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.2)

    parser.add_argument('--ppl_model', default=None,
                        help='PPL reference model path. Default: base model itself.')

    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    run_generation(args)
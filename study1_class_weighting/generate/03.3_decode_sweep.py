#!/usr/bin/env python3
"""
03.3_decode_sweep.py — Decoding strategy sweep for MAUVE improvement.

Tests different sampling strategies on the eps4 checkpoint to determine
if decoding strategy can improve distributional fidelity (MAUVE).

Configs:
  1. current:  temp=0.1, top_p=1.0           (reference — already generated as eps4)
  2. nucleus:  temp=0.3, top_p=0.9
  3. minp:     temp=0.3, min_p=0.1
  4. topk:     temp=0.3, top_k=50

Uses k=1 (no quality maximiser) for speed. n=1000 per config.
After generation, computes MAUVE for each config vs real data.
"""
import os
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
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

PROJ = '/fs1/projects/unlearning_pretraining/Proj_code'
BASE_MODEL = '/fs1/shared/model/llm/Llama-3.2-1B-Instruct'
CHECKPOINT = f'{PROJ}/models/llama_dp_eps4.0_20260318_1813/final'
REAL_DATA = f'{PROJ}/data/train.jsonl'
OUTPUT_ROOT = f'{PROJ}/generated/mimic'
EVAL_DIR = f'{PROJ}/eval'

SEED = 42
N_GENERATE = 1000
MAX_NEW_TOKENS = 1024
REP_PENALTY = 1.2

DECODE_CONFIGS = {
    'eps4_nucleus': {
        'temperature': 0.3, 'top_p': 0.9, 'top_k': 0, 'min_p': 0.0,
    },
    'eps4_minp': {
        'temperature': 0.3, 'top_p': 1.0, 'top_k': 0, 'min_p': 0.1,
    },
    'eps4_topk': {
        'temperature': 0.3, 'top_p': 1.0, 'top_k': 50, 'min_p': 0.0,
    },
}


def extract_control_prefix(text):
    idx = text.find(':\n')
    if idx == -1:
        idx = text.rfind(':')
        if idx == -1:
            return text[:100]
        return text[:idx + 1] + '\n'
    return text[:idx + 1] + '\n'


def load_prefixes(path, n=1000, seed=42):
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            prefix = extract_control_prefix(r['text'])
            records.append({
                'note_id': r.get('note_id', ''),
                'prefix': prefix,
                'control_codes': r.get('control_codes', {}),
                'bhc_text': r.get('bhc_text', ''),
            })
    random.seed(seed)
    if n > 0 and n < len(records):
        records = random.sample(records, n)
    return records


def load_real_texts(path, n=2000, seed=42):
    texts = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            txt = r.get('bhc_text', r.get('text', '')).strip()
            if txt:
                texts.append(txt)
    random.seed(seed)
    if n > 0 and n < len(texts):
        texts = random.sample(texts, n)
    return texts


def generate_one(prefix, model, tokenizer, device, config):
    input_ids = tokenizer.encode(prefix, return_tensors='pt').to(device)

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=config['temperature'],
        repetition_penalty=REP_PENALTY,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if config.get('top_p', 0) > 0 and config.get('top_p', 1.0) < 1.0:
        gen_kwargs['top_p'] = config['top_p']
    if config.get('top_k', 0) > 0:
        gen_kwargs['top_k'] = config['top_k']
    if config.get('min_p', 0) > 0:
        gen_kwargs['min_p'] = config['min_p']

    with torch.no_grad():
        output = model.generate(input_ids=input_ids, **gen_kwargs)

    generated_ids = output[0][input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def compute_mauve(real_texts, syn_texts, device_id=0):
    import mauve
    result = mauve.compute_mauve(
        p_text=real_texts,
        q_text=syn_texts,
        device_id=device_id,
        max_text_length=512,
        verbose=False,
    )
    return result.mauve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default=None,
                        help='Comma-separated config names to run (default: all)')
    parser.add_argument('--skip_eval', action='store_true',
                        help='Skip MAUVE evaluation (just generate)')
    parser.add_argument('--n', type=int, default=N_GENERATE)
    args = parser.parse_args()

    configs_to_run = DECODE_CONFIGS
    if args.configs:
        names = args.configs.split(',')
        configs_to_run = {k: v for k, v in DECODE_CONFIGS.items() if k in names}

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log.info(f'Loading base model from {BASE_MODEL}')
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, local_files_only=True,
    )
    log.info(f'Loading LoRA from {CHECKPOINT}')
    model = PeftModel.from_pretrained(base, CHECKPOINT)
    model = model.merge_and_unload().to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info('Model loaded')

    prefixes = load_prefixes(REAL_DATA, n=args.n, seed=SEED)
    log.info(f'Loaded {len(prefixes)} prefixes')

    for config_name, config in configs_to_run.items():
        log.info(f'\n{"="*60}')
        log.info(f'CONFIG: {config_name}')
        log.info(f'  {config}')
        log.info(f'{"="*60}')

        out_dir = Path(OUTPUT_ROOT) / config_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'synthetic_bhc.jsonl'

        completed = set()
        if out_path.exists():
            with open(out_path) as f:
                for line in f:
                    r = json.loads(line.strip())
                    if r.get('note_id'):
                        completed.add(r['note_id'])
            log.info(f'Resuming: {len(completed)} already done')

        remaining = [p for p in prefixes if p['note_id'] not in completed]
        log.info(f'Generating {len(remaining)} notes')

        with open(out_path, 'a') as out_f:
            for i, rec in enumerate(tqdm(remaining, desc=config_name)):
                text = generate_one(rec['prefix'], model, tokenizer, device, config)
                result = {
                    'note_id': rec['note_id'],
                    'control_codes': rec['control_codes'],
                    'prefix': rec['prefix'],
                    'synthetic_bhc': text,
                }
                out_f.write(json.dumps(result) + '\n')
                out_f.flush()

                if (i + 1) % 100 == 0:
                    log.info(f'  [{i+1}/{len(remaining)}] len={len(text)}')

        stats = {
            'config': config,
            'n_generated': len(remaining) + len(completed),
            'checkpoint': CHECKPOINT,
        }
        with open(out_dir / 'generation_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        log.info(f'Done: {out_path}')

    if args.skip_eval:
        log.info('Skipping MAUVE eval (--skip_eval)')
        return

    log.info('\n' + '=' * 60)
    log.info('MAUVE EVALUATION')
    log.info('=' * 60)

    real_texts = load_real_texts(REAL_DATA, n=2000, seed=SEED)
    log.info(f'Loaded {len(real_texts)} real texts for MAUVE')

    all_configs = list(configs_to_run.keys())
    eps4_orig = Path(OUTPUT_ROOT) / 'eps4' / 'synthetic_bhc.jsonl'
    if eps4_orig.exists():
        all_configs = ['eps4_original'] + all_configs

    results = []
    for config_name in all_configs:
        if config_name == 'eps4_original':
            syn_path = eps4_orig
        else:
            syn_path = Path(OUTPUT_ROOT) / config_name / 'synthetic_bhc.jsonl'

        if not syn_path.exists():
            log.warning(f'Missing: {syn_path}')
            continue

        syn_texts = []
        with open(syn_path) as f:
            for line in f:
                r = json.loads(line.strip())
                txt = r.get('synthetic_bhc', '').strip()
                if txt:
                    syn_texts.append(txt)

        n_eval = min(len(syn_texts), len(real_texts))
        log.info(f'{config_name}: {len(syn_texts)} synthetic, using {n_eval} for MAUVE')

        score = compute_mauve(real_texts[:n_eval], syn_texts[:n_eval])
        log.info(f'  MAUVE = {score:.4f}')
        results.append({'config': config_name, 'mauve': score, 'n_samples': n_eval})

    out_csv = f'{EVAL_DIR}/decode_sweep_mauve.csv'
    if results:
        import csv
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        log.info(f'\nWrote {out_csv}')

    print('\n' + '=' * 60)
    print('DECODING SWEEP RESULTS')
    print('=' * 60)
    for r in results:
        print(f"  {r['config']:25s}  MAUVE={r['mauve']:.4f}  n={r['n_samples']}")


if __name__ == '__main__':
    main()

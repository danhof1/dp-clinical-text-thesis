#!/usr/bin/env python3
"""
07b_tstr_per_category.py — Per-category TSTR to assess fairness of synthetic data.

For each generated run, trains a classifier on synthetic data and evaluates on
real data, reporting F1 per ICD category. This shows whether weighting strategies
help minority categories even if they hurt aggregate TSTR.

Outputs:
  eval/tstr_per_category.csv — one row per run × category
  eval/tstr_fairness.csv — one row per run with disparity metrics
"""
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

PROJ = '/fs1/projects/unlearning_pretraining/Proj_code'
GEN_ROOT = f'{PROJ}/generated/mimic'
REAL_DATA = f'{PROJ}/data/train.jsonl'
EVAL_DIR = f'{PROJ}/eval'
CLF_MODEL = 'distilbert/distilroberta-base'
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def gini(values):
    v = np.sort(np.array(values, dtype=float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2 * (idx * v).sum() / (n * v.sum())) - (n + 1) / n


def load_real(path, test_frac=0.2):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            cat = r.get('control_codes', {}).get('icd_category')
            txt = (r.get('bhc_text') or r.get('text', '')).strip()
            if cat and txt:
                rows.append({'label': cat, 'text': txt})
    random.shuffle(rows)
    split = int(len(rows) * (1 - test_frac))
    return rows[:split], rows[split:]


def load_synthetic(run_name):
    path = Path(GEN_ROOT) / run_name / 'synthetic_bhc.jsonl'
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            cat = r.get('control_codes', {}).get('icd_category')
            txt = r.get('synthetic_bhc', '').strip()
            if cat and txt:
                rows.append({'label': cat, 'text': txt})
    return rows


def discover_runs(requested=None):
    all_runs = sorted(d for d in Path(GEN_ROOT).iterdir()
                      if (Path(GEN_ROOT) / d.name / 'synthetic_bhc.jsonl').exists())
    names = [d.name for d in all_runs]
    if requested:
        return [r for r in names if r in requested]
    return names


class TextDataset(Dataset):
    def __init__(self, records, label2id, tokenizer, max_length=256):
        self.items = []
        for r in records:
            enc = tokenizer(r['text'], truncation=True, max_length=max_length,
                            padding='max_length', return_tensors='pt')
            self.items.append({'input_ids': enc['input_ids'].squeeze(0),
                               'attention_mask': enc['attention_mask'].squeeze(0),
                               'labels': torch.tensor(label2id[r['label']])})
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def train_eval_per_category(train_rows, test_rows, label2id, id2label,
                            epochs=3, batch_size=32, lr=2e-5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(CLF_MODEL)
    train_ds = TextDataset(train_rows, label2id, tok)
    test_ds = TextDataset(test_rows, label2id, tok)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = AutoModelForSequenceClassification.from_pretrained(
        CLF_MODEL, num_labels=len(label2id)
    ).to(device)
    optim = AdamW(model.parameters(), lr=lr)
    total = epochs * len(train_dl)
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * total), total)

    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
            losses.append(out.loss.item())
        print(f'    epoch {epoch+1}/{epochs}  loss={np.mean(losses):.3f}')

    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in test_dl:
            labels = batch.pop('labels').tolist()
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
            gold.extend(labels)

    macro_f1 = float(f1_score(gold, preds, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(gold, preds, average='weighted', zero_division=0))

    per_class_f1 = f1_score(gold, preds, average=None,
                            labels=list(range(len(label2id))), zero_division=0)

    per_category = {}
    test_counts = Counter(r['label'] for r in test_rows)
    train_counts = Counter(r['label'] for r in train_rows)

    for idx, f1_val in enumerate(per_class_f1):
        cat = id2label[idx]
        per_category[cat] = {
            'f1': float(f1_val),
            'n_test': test_counts.get(cat, 0),
            'n_train': train_counts.get(cat, 0),
        }

    return macro_f1, weighted_f1, per_category


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=None)
    ap.add_argument('--epochs', type=int, default=3)
    args = ap.parse_args()

    runs = discover_runs(args.runs.split(',') if args.runs else None)
    print(f'Runs: {runs}')

    print('Loading real MIMIC data...')
    real_train, real_test = load_real(REAL_DATA)
    print(f'  real train={len(real_train)}  test={len(real_test)}')

    all_cats = sorted({r['label'] for r in real_train + real_test})
    label2id = {c: i for i, c in enumerate(all_cats)}
    id2label = {i: c for c, i in label2id.items()}

    print(f'\n[Oracle] train-on-real, test-on-real  ({len(all_cats)} labels)')
    oracle_macro, oracle_weighted, oracle_per_cat = train_eval_per_category(
        real_train, real_test, label2id, id2label, epochs=args.epochs
    )
    print(f'  macro-F1={oracle_macro:.4f}  weighted-F1={oracle_weighted:.4f}')

    per_cat_rows = []
    fairness_rows = []

    for cat, info in sorted(oracle_per_cat.items()):
        per_cat_rows.append({
            'run_id': 'ORACLE_real', 'category': cat,
            'f1': round(info['f1'], 4),
            'n_train': info['n_train'], 'n_test': info['n_test'],
        })

    oracle_f1s = [v['f1'] for v in oracle_per_cat.values()]
    fairness_rows.append({
        'run_id': 'ORACLE_real',
        'macro_f1': round(oracle_macro, 4),
        'weighted_f1': round(oracle_weighted, 4),
        'per_cat_mean': round(np.mean(oracle_f1s), 4),
        'per_cat_std': round(np.std(oracle_f1s), 4),
        'per_cat_min': round(np.min(oracle_f1s), 4),
        'per_cat_max': round(np.max(oracle_f1s), 4),
        'max_min_gap': round(np.max(oracle_f1s) - np.min(oracle_f1s), 4),
        'cv': round(np.std(oracle_f1s) / np.mean(oracle_f1s), 4) if np.mean(oracle_f1s) > 0 else 0,
        'gini': round(gini(oracle_f1s), 4),
        'n_zero_f1': int(np.sum(np.array(oracle_f1s) == 0)),
    })

    for run in runs:
        print(f'\n[TSTR] {run}')
        try:
            syn = load_synthetic(run)
        except Exception as e:
            print(f'  SKIP: {e}')
            continue

        syn_cats = {r['label'] for r in syn}
        test_cats = {r['label'] for r in real_test}
        common = syn_cats & test_cats
        syn_filt = [r for r in syn if r['label'] in common]
        test_filt = [r for r in real_test if r['label'] in common]
        label2id_run = {c: i for i, c in enumerate(sorted(common))}
        id2label_run = {i: c for c, i in label2id_run.items()}

        print(f'  syn={len(syn_filt)}  test={len(test_filt)}  labels={len(common)}')

        macro_f1, weighted_f1, per_cat = train_eval_per_category(
            syn_filt, test_filt, label2id_run, id2label_run, epochs=args.epochs
        )
        print(f'  macro-F1={macro_f1:.4f}  weighted-F1={weighted_f1:.4f}')

        for cat, info in sorted(per_cat.items()):
            per_cat_rows.append({
                'run_id': run, 'category': cat,
                'f1': round(info['f1'], 4),
                'n_train': info['n_train'], 'n_test': info['n_test'],
            })

        cat_f1s = [v['f1'] for v in per_cat.values()]
        fairness_rows.append({
            'run_id': run,
            'macro_f1': round(macro_f1, 4),
            'weighted_f1': round(weighted_f1, 4),
            'per_cat_mean': round(np.mean(cat_f1s), 4),
            'per_cat_std': round(np.std(cat_f1s), 4),
            'per_cat_min': round(np.min(cat_f1s), 4),
            'per_cat_max': round(np.max(cat_f1s), 4),
            'max_min_gap': round(np.max(cat_f1s) - np.min(cat_f1s), 4),
            'cv': round(np.std(cat_f1s) / np.mean(cat_f1s), 4) if np.mean(cat_f1s) > 0 else 0,
            'gini': round(gini(cat_f1s), 4),
            'n_zero_f1': int(np.sum(np.array(cat_f1s) == 0)),
        })

        _save_intermediate(per_cat_rows, fairness_rows)

    _save_intermediate(per_cat_rows, fairness_rows)

    print('\n' + '=' * 70)
    print('TSTR FAIRNESS COMPARISON')
    print('=' * 70)
    for row in fairness_rows:
        print(f"  {row['run_id']:40s}  macro={row['macro_f1']:.3f}  "
              f"cv={row['cv']:.3f}  gini={row['gini']:.3f}  "
              f"gap={row['max_min_gap']:.3f}  zero_cats={row['n_zero_f1']}")


def _save_intermediate(per_cat_rows, fairness_rows):
    out1 = f'{EVAL_DIR}/tstr_per_category.csv'
    if per_cat_rows:
        with open(out1, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=per_cat_rows[0].keys())
            w.writeheader()
            w.writerows(per_cat_rows)

    out2 = f'{EVAL_DIR}/tstr_fairness.csv'
    if fairness_rows:
        with open(out2, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fairness_rows[0].keys())
            w.writeheader()
            w.writerows(fairness_rows)


if __name__ == '__main__':
    main()

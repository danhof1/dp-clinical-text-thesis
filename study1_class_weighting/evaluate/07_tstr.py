from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

PROJ      = '/fs1/projects/unlearning_pretraining/Proj_code'
GEN_ROOT  = f'{PROJ}/generated/mimic'
REAL_DATA = f'{PROJ}/data/train.jsonl'
EVAL_DIR  = f'{PROJ}/eval'
CLF_MODEL = 'distilbert/distilroberta-base'
SEED      = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

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

def train_eval(train_rows, test_rows, label2id, epochs=3, batch_size=32, lr=2e-5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(CLF_MODEL)
    train_ds = TextDataset(train_rows, label2id, tok)
    test_ds  = TextDataset(test_rows,  label2id, tok)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)
    model = AutoModelForSequenceClassification.from_pretrained(CLF_MODEL, num_labels=len(label2id)).to(device)
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
    macro_f1    = float(f1_score(gold, preds, average='macro',    zero_division=0))
    weighted_f1 = float(f1_score(gold, preds, average='weighted', zero_division=0))
    return macro_f1, weighted_f1

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
    all_cats = sorted({r["label"] for r in real_train + real_test})
    label2id = {c: i for i, c in enumerate(all_cats)}
    print(f'\n[Oracle] train-on-real, test-on-real  ({len(all_cats)} labels)')
    oracle_macro, oracle_weighted = train_eval(real_train, real_test, label2id, epochs=args.epochs)
    print(f'  macro-F1={oracle_macro:.4f}  weighted-F1={oracle_weighted:.4f}')
    results = [{'run_id': 'ORACLE_real', 'n_train': len(real_train), 'n_test': len(real_test),
                'n_labels': len(label2id), 'macro_f1': round(oracle_macro, 4),
                'weighted_f1': round(oracle_weighted, 4)}]
    for run in runs:
        print(f'\n[TSTR] {run}')
        try:
            syn = load_synthetic(run)
        except Exception as e:
            print(f'  SKIP: {e}'); continue
        syn_cats  = {r["label"] for r in syn}
        test_cats = {r["label"] for r in real_test}
        common    = syn_cats & test_cats
        syn_filt  = [r for r in syn      if r["label"] in common]
        test_filt = [r for r in real_test if r["label"] in common]
        label2id_run = {c: i for i, c in enumerate(sorted(common))}
        print(f'  syn={len(syn_filt)}  test={len(test_filt)}  labels={len(common)}')
        macro_f1, weighted_f1 = train_eval(syn_filt, test_filt, label2id_run, epochs=args.epochs)
        print(f'  macro-F1={macro_f1:.4f}  weighted-F1={weighted_f1:.4f}')
        results.append({'run_id': run, 'n_train': len(syn_filt), 'n_test': len(test_filt),
                        'n_labels': len(common), 'macro_f1': round(macro_f1, 4),
                        'weighted_f1': round(weighted_f1, 4)})
        out = f'{EVAL_DIR}/tstr_results.csv'
        with open(out, 'w', newline='') as fout:
            w = csv.DictWriter(fout, fieldnames=results[0].keys())
            w.writeheader(); w.writerows(results)
        print(f'  (saved {len(results)} rows to {out})')
    out = f'{EVAL_DIR}/tstr_results.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)
    print(f'\nWrote {out}')

if __name__ == '__main__':
    main()

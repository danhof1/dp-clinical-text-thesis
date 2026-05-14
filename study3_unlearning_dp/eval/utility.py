"""
Downstream-utility evaluation: train-on-synthetic, test-on-real.

The question: if someone is given the synthetic corpus instead of the real
MTSamples, how good a downstream model can they train?

We use specialty classification as a canonical task because MTSamples has
clean specialty labels. Protocol:

  1. Train a classifier on SYNTHETIC (specialty, note) pairs.
  2. Evaluate on REAL held-out MTSamples (the nonmembers split — never
     seen by any fine-tuning).
  3. Compare across conditions (baseline DP vs. unlearn+DP) and report
     macro-F1. The utility-privacy frontier is the main summary plot.

We use a small, fast classifier (distilroberta) because the question is
about the data, not the classifier. Running this for every ε × method
combination should take under an hour per config on a single GPU.

Note: the synthetic notes come out of generate/run.py and carry a "specialty"
field from the prompt. We trust those labels for training.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset, load_from_disk
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("eval.utility")


def load_synthetic(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("specialty") and r.get("text"):
                records.append(r)
    return records


def build_label_space(records: list[dict]) -> dict[str, int]:
    labels = sorted({r["specialty"] for r in records})
    return {lab: i for i, lab in enumerate(labels)}


def prepare_clf_dataset(records, label2id, tokenizer, max_length):
    texts = [r["text"] for r in records]
    labels = [label2id[r["specialty"]] for r in records]
    enc = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
    return Dataset.from_dict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "label": labels,
    })


def collate(tokenizer):
    pad_id = tokenizer.pad_token_id

    def _f(features):
        max_len = max(len(f["input_ids"]) for f in features)
        ids, atn, lbl = [], [], []
        for f in features:
            pad_n = max_len - len(f["input_ids"])
            ids.append(list(f["input_ids"]) + [pad_id] * pad_n)
            atn.append(list(f["attention_mask"]) + [0] * pad_n)
            lbl.append(f["label"])
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(atn),
            "labels": torch.tensor(lbl),
        }
    return _f


def train_classifier(
    train_ds, eval_ds, num_labels: int, cfg: dict,
) -> tuple[float, dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = cfg.get("clf_model", "distilbert/distilroberta-base")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
    ).to(device)

    bs = cfg.get("clf_batch_size", 32)
    loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, collate_fn=collate(tokenizer), num_workers=2,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=bs, shuffle=False, collate_fn=collate(tokenizer), num_workers=2,
    )

    epochs = cfg.get("clf_epochs", 3)
    lr = cfg.get("clf_lr", 2e-5)
    optim = AdamW(model.parameters(), lr=lr)
    total = epochs * len(loader)
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * total), total)

    model.train()
    for epoch in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
        log.info("epoch %d loss=%.3f", epoch, out.loss.item())

    # eval
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in eval_loader:
            batch_gpu = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            out = model(**batch_gpu)
            preds.extend(out.logits.argmax(-1).cpu().tolist())
            gold.extend(batch["labels"].tolist())

    macro_f1 = float(f1_score(gold, preds, average="macro", zero_division=0))
    report = classification_report(gold, preds, zero_division=0, output_dict=True)
    return macro_f1, report


def run(cfg: dict):
    synth = load_synthetic(cfg["synthetic_path"])
    log.info("loaded %d synthetic records", len(synth))

    splits = load_from_disk(cfg["splits_path"])
    # Real test set: held-out MTSamples (nonmembers split). They have the
    # "specialty" field from the MTSamples CSV.
    real_test = [r for r in splits["nonmembers"] if r.get("specialty")]
    log.info("real test set: %d records", len(real_test))

    # Build shared label space. Restrict to labels that appear in BOTH sets
    # so we don't evaluate on specialties the synthetic model never saw.
    synth_labs = {r["specialty"] for r in synth}
    test_labs = {r["specialty"] for r in real_test}
    common = synth_labs & test_labs
    synth = [r for r in synth if r["specialty"] in common]
    real_test = [r for r in real_test if r["specialty"] in common]
    log.info("common specialty labels: %d", len(common))
    label2id = {l: i for i, l in enumerate(sorted(common))}

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.get("clf_model", "distilbert/distilroberta-base")
    )
    max_length = cfg.get("max_length", 512)

    train_ds = prepare_clf_dataset(synth, label2id, tokenizer, max_length)
    eval_ds = prepare_clf_dataset(real_test, label2id, tokenizer, max_length)

    macro_f1, report = train_classifier(train_ds, eval_ds, len(label2id), cfg)
    log.info("macro-F1 on real held-out: %.3f", macro_f1)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "utility_report.json", "w") as f:
        json.dump({
            "synthetic_path": cfg["synthetic_path"],
            "n_train_synthetic": len(synth),
            "n_test_real": len(real_test),
            "n_labels": len(label2id),
            "macro_f1": macro_f1,
            "per_class": report,
        }, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--synthetic_path_override", default=None)
    ap.add_argument("--output_dir_override", default=None)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.synthetic_path_override:
        cfg["synthetic_path"] = args.synthetic_path_override
    if args.output_dir_override:
        cfg["output_dir"] = args.output_dir_override
    run(cfg)


if __name__ == "__main__":
    main()

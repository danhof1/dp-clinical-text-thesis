"""
Shared utilities for the unlearning implementations.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)


log = logging.getLogger(__name__)


@dataclass
class UnlearnConfig:
    # Method selection
    method: str  # "ga" | "ga_gd" | "npo"

    # Paths
    base_model: str
    splits_path: str
    output_dir: str

    # Unlearning hyperparams
    learning_rate: float = 5e-6
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 512
    warmup_steps: int = 20
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # Method-specific
    gd_weight: float = 1.0
    npo_beta: float = 0.1

    # LoRA (reduces AdamW optimizer states from ~56 GB to <300 MB)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # Memory
    gradient_checkpointing: bool = True

    # Eval
    eval_every_steps: int = 50
    save_every_steps: int = 200

    # Precision
    bf16: bool = True

    seed: int = 42


def tokenize_function(tokenizer, max_length):
    def _tok(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        out["labels"] = [list(ids) for ids in out["input_ids"]]
        return out
    return _tok


def collate_fn(tokenizer):
    pad_id = tokenizer.pad_token_id

    def _collate(features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            ids = list(f["input_ids"])
            pad_n = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_n)
            attention_mask.append([1] * len(ids) + [0] * pad_n)
            labels.append(list(f["labels"]) + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    return _collate


def load_model_and_tokenizer(model_name_or_path, bf16=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    return model, tokenizer


def prepare_dataloaders(splits_path, tokenizer, cfg, need_retain):
    splits = load_from_disk(splits_path)
    forget_ds = splits["finetune"]
    tok_fn = tokenize_function(tokenizer, cfg.max_seq_length)

    forget_tok = forget_ds.map(tok_fn, batched=True, remove_columns=forget_ds.column_names)
    forget_loader = DataLoader(
        forget_tok,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn(tokenizer),
        num_workers=0,
    )

    retain_loader = None
    if need_retain:
        retain_ds = splits["retain"]
        retain_tok = retain_ds.map(tok_fn, batched=True, remove_columns=retain_ds.column_names)
        retain_loader = DataLoader(
            retain_tok,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn(tokenizer),
            num_workers=0,
        )

    return forget_loader, retain_loader


def per_token_nll(model, batch):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    return outputs.loss


def per_example_nll(model, batch):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    logits = outputs.logits[..., :-1, :].contiguous()
    labels = batch["labels"][..., 1:].contiguous()
    mask = (labels != -100).float()
    log_probs = F.log_softmax(logits, dim=-1)
    safe_labels = labels.clamp(min=0)
    nll_per_token = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    nll_per_token = nll_per_token * mask
    denom = mask.sum(dim=-1).clamp(min=1.0)
    return nll_per_token.sum(dim=-1) / denom


def save_checkpoint(model, tokenizer, out_dir, step, cfg, metrics, merge_lora=False):
    ckpt_dir = out_dir / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if merge_lora and hasattr(model, "merge_and_unload"):
        # Merge LoRA into base weights so downstream HF loading works without PEFT
        merged = model.merge_and_unload()
        merged.save_pretrained(ckpt_dir)
    else:
        model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    with open(ckpt_dir / "unlearn_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(ckpt_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return ckpt_dir


def cycle(loader):
    while True:
        for batch in loader:
            yield batch

"""
02b_train_sgd.py
================
Standard SGD fine-tuning of Llama-3.2-1B-Instruct — NO differential privacy.
Produces the checkpoint for Synthetic Data 1 (epsilon=inf baseline).

Role in the 4-dataset framework:
    Data 0: Base model, no fine-tuning         (03a_generate_base.py)
    Data 1: Base + standard SGD, no DP  <-- THIS SCRIPT
    Data 2: Base + DP-SGD at eps in {0.5,1,4}  (02_train_dp.py)
    Data 3: Base + MRP unlearn + DP-SGD         (contingent on MTSamples audit)

Data 1 serves two roles:
    1. Upper bound on generation quality — shows what the model can learn
       without the noise constraint. Degradation at each epsilon is then
       attributable purely to DP noise rather than fine-tuning in general.

    2. Class imbalance disentanglement — if conditioning fails for rare ICD
       categories at eps=inf (Data 1), the cause is class imbalance.
       If it only fails under DP (Data 2), the cause is the privacy noise.
       This is the key disentanglement analysis in the thesis.

Identical to 02_train_dp.py except:
    - No Opacus privacy engine
    - No noise multiplier or privacy-motivated gradient clipping
    - Standard AdamW with grad clip=1.0 for numerical stability only
    - Class weighting retained (same intervention, isolates DP effect)
    - Batch size configurable (DP-SGD forces batch=1)
    - Faster: no per-sample gradient computation overhead

Usage:
    # MIMIC (Data 1)
    python 02b_train_sgd.py \\
        --model_path /fs1/shared/model/llm/Llama-3.2-1B-Instruct \\
        --data ./data/train.jsonl \\
        --output ./models

    # MTSamples (Data 1 for MTSamples track)
    python 02b_train_sgd.py \\
        --model_path /fs1/shared/model/llm/Llama-3.2-1B-Instruct \\
        --data ./data/mtsamples/train.jsonl \\
        --output ./models/mtsamples

    # Without class weighting (ablation)
    python 02b_train_sgd.py \\
        --model_path /fs1/shared/model/llm/Llama-3.2-1B-Instruct \\
        --data ./data/train.jsonl \\
        --no_class_weights
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType


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
        return f"GPU: {alloc:.1f}GB alloc / {resrv:.1f}GB reserved"
    return "GPU: not available"


# =========================================================================
# Class weights (identical formula to 02_train_dp.py)
# =========================================================================

def compute_class_weights(categories):
    """
    Inverse frequency weights: w_c = N / (C * n_c), normalised to mean=1.
    Bagdasaryan et al. (NeurIPS 2019); Rosenblatt et al. (arXiv 2024).
    """
    counts = Counter(categories)
    N = len(categories)
    C = len(counts)
    weights = {cat: N / (C * n_c) for cat, n_c in counts.items()}
    mean_w  = np.mean(list(weights.values()))
    weights = {cat: w / mean_w for cat, w in weights.items()}

    stats = {
        'n_categories':    C,
        'n_records':       N,
        'category_counts': dict(counts),
        'category_weights': {
            cat: round(w, 4)
            for cat, w in sorted(weights.items(), key=lambda x: -x[1])
        },
        'weight_range': {
            'min':  round(min(weights.values()), 4),
            'max':  round(max(weights.values()), 4),
            'mean': round(np.mean(list(weights.values())), 4),
        },
    }
    return weights, stats


# =========================================================================
# Dataset
# =========================================================================

class BHCDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.records    = []
        self.categories = []

        log.info(f"Loading dataset from {jsonl_path}")
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line.strip())
                self.records.append(rec['text'])
                self.categories.append(
                    rec.get('control_codes', {}).get('icd_category', 'Unknown')
                )
        log.info(f"Loaded {len(self.records):,} training records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.records[idx],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
        )
        input_ids      = torch.tensor(enc['input_ids'],      dtype=torch.long)
        attention_mask = torch.tensor(enc['attention_mask'], dtype=torch.long)
        labels         = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'labels':         labels,
            'category':       self.categories[idx],
        }


# =========================================================================
# LoRA config (identical to 02_train_dp.py — fair ablation)
# =========================================================================

def get_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=0.05, bias='none',
        target_modules=['q_proj','k_proj','v_proj','o_proj',
                        'gate_proj','up_proj','down_proj'],
    )


# =========================================================================
# Training loop
# =========================================================================

def train(model, tokenizer, dataset, args, output_dir, class_weights=None):
    """
    Standard AdamW training — no Opacus, no noise, no per-sample gradients.

    Key difference from 02_train_dp.py:
        - batch_size > 1 (DP-SGD requires batch=1 for per-sample gradients)
        - grad clipping is for numerical stability only, not privacy
        - no privacy engine, no epsilon tracking
    """
    n           = len(dataset)
    total_steps = args.epochs * max(n // args.batch_size, 1)
    warmup_steps = int(0.05 * total_steps)

    log.info(f"Dataset size:     {n:,}")
    log.info(f"Epochs:           {args.epochs}")
    log.info(f"Batch size:       {args.batch_size}  "
             f"(DP-SGD forces 1; SGD allows larger batches)")
    log.info(f"Total steps:      {total_steps:,}")
    log.info(f"Learning rate:    {args.lr}")
    log.info(f"Grad clip norm:   {args.max_grad_norm}  (stability, not privacy)")
    log.info(f"Class weighting:  {'enabled' if class_weights else 'disabled'}")
    log.info(f"Privacy:          NONE  (epsilon=inf baseline)")
    log.info(f"{gpu_mem()}")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=True,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    log.info(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    log.info(f"Warmup steps: {warmup_steps:,}")

    device      = next(model.parameters()).device
    global_step = 0
    model.train()

    log.info("=" * 60)
    log.info("STARTING STANDARD SGD TRAINING (epsilon=inf, no DP)")
    log.info("=" * 60)

    for epoch in range(args.epochs):
        epoch_loss  = 0.0
        epoch_steps = 0

        for batch in dataloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            categories     = batch['category']

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # Per-batch class weighting
            # Average weights across examples in the batch.
            # (At batch_size=1 this is identical to 02_train_dp.py)
            if class_weights is not None:
                batch_w = torch.tensor(
                    [class_weights.get(cat, 1.0) for cat in categories],
                    dtype=torch.float32, device=device,
                )
                loss = loss * batch_w.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss  += loss.item()
            epoch_steps += 1
            global_step += 1

            should_log = (
                global_step <= 5 or global_step == 20 or
                global_step == 50 or global_step % 200 == 0
            )
            if should_log:
                log.info(
                    f"  e{epoch+1} step={global_step:>7,}/{total_steps:,}  "
                    f"loss={epoch_loss/epoch_steps:.4f}  {gpu_mem()}"
                )

        avg_loss = epoch_loss / epoch_steps
        log.info(f"EPOCH {epoch+1}/{args.epochs} COMPLETE  avg_loss={avg_loss:.4f}")

        ckpt_dir = output_dir / f'checkpoint-epoch{epoch+1}'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = model._module if hasattr(model, '_module') else model
        unwrapped.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        log.info(f"Checkpoint saved -> {ckpt_dir}")

    log.info("Training complete (epsilon=inf, no privacy)")


# =========================================================================
# Main
# =========================================================================

def main(args):
    set_seed(args.seed)

    timestamp  = datetime.now().strftime('%Y%m%d_%H%M')
    run_name   = f"llama_sgd_epsinf_{timestamp}"
    output_dir = Path(args.output) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Run: {run_name}")
    log.info(f"Output: {output_dir}")
    log.info("DATA PATH: Base -> SGD (no DP) -> Synthetic Data 1")

    with open(output_dir / 'run_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = BHCDataset(args.data, tokenizer, args.max_length)

    class_weights = None
    if not args.no_class_weights:
        log.info("=" * 60)
        log.info("COMPUTING INVERSE FREQUENCY CLASS WEIGHTS")
        log.info("=" * 60)
        class_weights, weight_stats = compute_class_weights(dataset.categories)
        log.info(f"  Categories: {weight_stats['n_categories']}  "
                 f"Weight range: {weight_stats['weight_range']['min']} – "
                 f"{weight_stats['weight_range']['max']}")
        log.info("  Highest weighted (rarest categories):")
        for cat, w in list(weight_stats['category_weights'].items())[:3]:
            n = weight_stats['category_counts'][cat]
            log.info(f"    w={w:.3f}  n={n:,}  {cat[:55]}")
        with open(output_dir / 'class_weights.json', 'w') as f:
            json.dump(weight_stats, f, indent=2)
    else:
        log.info("Class weighting disabled")

    log.info(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, local_files_only=True,
    )
    model = get_peft_model(model, get_lora_config())
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)
    log.info(f"Model on {device}. {gpu_mem()}")

    train(model, tokenizer, dataset, args, output_dir, class_weights=class_weights)

    final_dir = output_dir / 'final'
    final_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = model._module if hasattr(model, '_module') else model
    unwrapped.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    report = {
        'data_path':       'Data 1: Base -> SGD (epsilon=inf) -> Synthetic Data 1',
        'epsilon':         'inf',
        'privacy':         'none',
        'n_records':       len(dataset),
        'epochs':          args.epochs,
        'batch_size':      args.batch_size,
        'max_grad_norm':   args.max_grad_norm,
        'lora_r':          8,
        'lora_alpha':      16,
        'max_length':      args.max_length,
        'lr':              args.lr,
        'class_weighting': not args.no_class_weights,
    }
    with open(output_dir / 'training_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    log.info(f"DONE. Final model -> {final_dir}")
    log.info(f"Next step: python 03_generate.py \\")
    log.info(f"    --checkpoint {final_dir} \\")
    log.info(f"    --base_model {args.model_path} \\")
    log.info(f"    --output ./generated/data1_sgd \\")
    log.info(f"    --epsilon inf")


# =========================================================================
# CLI
# =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Standard SGD fine-tuning (no DP) — Data 1 epsilon=inf baseline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model_path',       required=True)
    parser.add_argument('--data',             default='./data/train.jsonl')
    parser.add_argument('--output',           default='./models')
    parser.add_argument('--epochs',           type=int,   default=2)
    parser.add_argument('--batch_size',       type=int,   default=8,
                        help='Can be >1 without DP constraint (DP-SGD forces batch=1)')
    parser.add_argument('--lr',               type=float, default=5e-5)
    parser.add_argument('--max_grad_norm',    type=float, default=1.0,
                        help='Gradient clipping for stability, not privacy')
    parser.add_argument('--max_length',       type=int,   default=512)
    parser.add_argument('--seed',             type=int,   default=42)
    parser.add_argument('--no_class_weights', action='store_true',
                        help='Disable inverse frequency class weighting')
    args = parser.parse_args()
    main(args)
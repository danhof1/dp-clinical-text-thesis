"""
02_train_dp.py
==============
Phase 3 -- DP-SGD fine-tuning of Llama-3.2-1B-Instruct on MIMIC-IV BHC sections.

Method:
    - LoRA adapters (Hu et al., ICLR 2022)
    - DP-SGD via Opacus (Meta) — manual training loop
    - Control code conditioning (Yue et al., ACL 2023)
    - Privacy accounting via Renyi DP moments accountant

CLASS IMBALANCE FIX (Bagdasaryan et al. NeurIPS 2019; Rosenblatt et al. arXiv 2024):
    DP-SGD has disparate impact on underrepresented classes. Minority class
    examples produce larger gradients, which are disproportionately clipped,
    reducing those classes' influence on the model. This worsens existing
    imbalance — "the poor become poorer" under DP.

    Fix: inverse frequency weighted loss (class-weighted ERM).
    Each training example's loss is scaled by w_c = N / (C * n_c) where:
        N  = total training examples
        C  = number of ICD categories
        n_c = examples in this example's ICD category

    This upweights rare ICD categories proportionally, compensating for
    the gradient clipping bias. Implemented as per-example loss scaling
    BEFORE backprop — compatible with Opacus DP-SGD.

    Important: we do NOT use a weighted sampler. DP-SGD privacy accounting
    assumes uniform (Poisson) subsampling. A weighted sampler would change
    the effective sampling rate per class and invalidate the privacy guarantee.
    Per-example loss weighting achieves the same effect without touching
    the sampling distribution.

    Reference:
        Bagdasaryan et al. (2019). Differential Privacy Has Disparate Impact
        on Model Accuracy. NeurIPS 2019.

        Rosenblatt et al. (2024). Differential Privacy Under Class Imbalance:
        Methods and Empirical Insights. arXiv:2411.05733.

Usage:
    python 02_train_dp.py \\
        --model_path /path/to/Llama-3.2-1B-Instruct \\
        --epsilon 4 --epochs 2 --max_length 512

    # Disable class weighting (for ablation / comparison runs):
    python 02_train_dp.py \\
        --model_path /path/to/Llama-3.2-1B-Instruct \\
        --epsilon 4 --no_class_weights
"""

import os
import sys
import json
import math
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
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
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
        return f"GPU: {alloc:.1f}GB alloc / {resrv:.1f}GB reserved"
    return "GPU: not available"


# =========================================================================
# Class weight computation
# =========================================================================

def compute_class_weights(records, category_field='icd_category'):
    """
    Compute inverse frequency class weights for ICD categories.

    w_c = N / (C * n_c)

    where N = total records, C = number of categories, n_c = records in class c.
    This is the standard sklearn-style 'balanced' class weight formula.

    Returns:
        weights: dict mapping icd_category -> float weight
        stats:   dict with distribution info for logging
    """
    categories = [r.get(category_field, 'Unknown') for r in records]
    counts = Counter(categories)
    N = len(records)
    C = len(counts)

    weights = {}
    for cat, n_c in counts.items():
        weights[cat] = N / (C * n_c)

    # Normalise so mean weight = 1.0 (keeps loss scale comparable to unweighted)
    mean_w = np.mean(list(weights.values()))
    weights = {cat: w / mean_w for cat, w in weights.items()}

    stats = {
        'n_categories': C,
        'n_records': N,
        'category_counts': dict(counts),
        'category_weights': {cat: round(w, 4) for cat, w in
                             sorted(weights.items(), key=lambda x: -x[1])},
        'weight_range': {
            'min': round(min(weights.values()), 4),
            'max': round(max(weights.values()), 4),
            'mean': round(np.mean(list(weights.values())), 4),
        },
    }
    return weights, stats


# =========================================================================
# Dataset
# =========================================================================

class BHCDataset(Dataset):
    """
    Loads train.jsonl. Each record's 'text' field is:
        "Note Type: Discharge Summary | ICD Category: ... | Age Group: ... | Sex: M:\n[BHC text]"

    Also stores the ICD category for per-example loss weighting.
    """

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
                # Extract ICD category for loss weighting
                cat = rec.get('control_codes', {}).get('icd_category', 'Unknown')
                self.categories.append(cat)

        log.info(f"Loaded {len(self.records):,} training records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        text = self.records[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
        )
        input_ids      = torch.tensor(encoding['input_ids'],      dtype=torch.long)
        attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'labels':         labels,
            'category':       self.categories[idx],  # for loss weighting
        }


# =========================================================================
# LoRA
# =========================================================================

def get_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias='none',
        target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
        ],
    )


# =========================================================================
# Privacy
# =========================================================================

def compute_delta(n_records):
    """delta = 1 / (N * log(N)), following Term2Note."""
    return 1.0 / (n_records * math.log(n_records))


def compute_noise_multiplier(target_epsilon, delta, sample_rate, steps):
    """
    Binary search for noise multiplier achieving (epsilon, delta)-DP.
    """
    from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent

    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    lo, hi = 0.01, 100.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        rdp = compute_rdp(
            q=sample_rate,
            noise_multiplier=mid,
            steps=steps,
            orders=orders,
        )
        eps, _ = get_privacy_spent(orders=orders, rdp=rdp, delta=delta)
        if eps > target_epsilon:
            lo = mid
        else:
            hi = mid
    return hi


# =========================================================================
# Training loop
# =========================================================================

def find_latest_checkpoint(output_dir, epochs):
    """
    Scan output_dir for completed epoch checkpoints.
    Returns (start_epoch, checkpoint_path) where start_epoch is the next
    epoch to run (0-indexed), and checkpoint_path is the model to load from.

    Epoch checkpoints are saved as checkpoint-epoch1/, checkpoint-epoch2/, etc.
    We only resume from a checkpoint if it contains adapter_config.json,
    indicating the save completed successfully.
    """
    latest_epoch = 0
    latest_path  = None

    for epoch in range(1, epochs + 1):
        ckpt = output_dir / f'checkpoint-epoch{epoch}'
        if (ckpt / 'adapter_config.json').exists():
            latest_epoch = epoch
            latest_path  = ckpt

    if latest_path:
        log.info(f"Found completed checkpoint at epoch {latest_epoch}: {latest_path}")
    else:
        log.info("No existing checkpoints found — starting from scratch")

    return latest_epoch, latest_path


def train(model, tokenizer, dataset, args, output_dir, class_weights=None,
          resume_epoch=0):
    """
    DP-SGD training with Opacus — manual loop with per-example loss weighting.

    class_weights: dict mapping icd_category -> float, or None for unweighted.
    resume_epoch:  number of epochs already completed (0 = fresh start).

    Loss weighting is applied BEFORE backprop so that gradient magnitudes
    reflect the intended class balance. This is compatible with Opacus
    because we are scaling the loss, not the sampling distribution —
    the DP privacy accounting remains valid.

    CHECKPOINT RESUME — DP privacy accounting note:
    Resuming mid-epoch is unsafe for DP accounting because the accountant
    must track exact step counts. We only support epoch-level resume:
    if epoch K is complete, we skip epochs 1..K and start from K+1.
    The privacy accountant is initialised with the steps already taken
    (resume_epoch * n) so the reported epsilon reflects the full run.
    """
    from opacus import PrivacyEngine

    n           = len(dataset)
    delta       = compute_delta(n)
    sample_rate = 1.0 / n
    total_steps = args.epochs * n
    steps_done  = resume_epoch * n  # steps already completed in prior runs

    log.info(f"Dataset size:     {n:,}")
    log.info(f"Delta:            {delta:.2e}")
    log.info(f"Target epsilon:   {args.epsilon}")
    log.info(f"Epochs:           {args.epochs}")
    log.info(f"Resume from:      epoch {resume_epoch} ({steps_done:,} steps already done)")
    log.info(f"Remaining epochs: {args.epochs - resume_epoch}")
    log.info(f"Learning rate:    {args.lr}")
    log.info(f"Max grad norm:    {args.max_grad_norm}")
    log.info(f"Max length:       {args.max_length}")
    log.info(f"Class weighting:  {'enabled' if class_weights else 'disabled'}")
    log.info(f"Sample rate:      {sample_rate:.6e}")
    log.info(f"Total steps:      {total_steps:,}")
    log.info(f"{gpu_mem()}")

    log.info("Computing noise multiplier via binary search...")
    sigma = compute_noise_multiplier(
        target_epsilon=args.epsilon,
        delta=delta,
        sample_rate=sample_rate,
        steps=total_steps,
    )
    log.info(f"Noise multiplier (sigma) = {sigma:.6f}")

    log.info("Creating standard DataLoader (batch_size=1)...")
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    log.info(f"DataLoader: {len(dataloader):,} batches per epoch")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable_params)
    log.info(f"Trainable parameters: {n_params:,}")

    optimizer = AdamW(trainable_params, lr=args.lr)

    warmup_steps = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    log.info(f"Warmup steps: {warmup_steps:,}")

    log.info("=" * 60)
    log.info("ATTACHING OPACUS PRIVACY ENGINE")
    log.info("=" * 60)

    privacy_engine = PrivacyEngine(accountant='rdp')

    try:
        model, optimizer, _ = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=sigma,
            max_grad_norm=args.max_grad_norm,
            poisson_sampling=False,
        )
    except Exception as e:
        log.error(f"make_private FAILED: {type(e).__name__}: {e}")
        raise

    # If resuming, fast-forward the accountant to reflect steps already taken.
    # This ensures epsilon is tracked correctly across the full run.
    if steps_done > 0:
        log.info(f"Fast-forwarding privacy accountant by {steps_done:,} steps "
                 f"(epochs already completed: {resume_epoch})")
        for _ in range(steps_done):
            privacy_engine.accountant.step(
                noise_multiplier=sigma,
                sample_rate=sample_rate,
            )
        eps_so_far = privacy_engine.get_epsilon(delta)
        log.info(f"Epsilon after fast-forward: {eps_so_far:.4f}/{args.epsilon}")

    log.info(f"Opacus attached. sigma={sigma:.6f}  max_grad_norm={args.max_grad_norm}")
    log.info(f"{gpu_mem()}")

    # ── Training loop ──────────────────────────────────────────────────────
    model.train()
    device = next(model.parameters()).device

    log.info("=" * 60)
    log.info("STARTING TRAINING")
    log.info(f"  {args.epochs} epochs total, starting from epoch {resume_epoch + 1}")
    log.info(f"  {args.epochs - resume_epoch} epochs remaining")
    log.info("=" * 60)

    global_step  = steps_done  # start counter from where we left off
    weight_stats = Counter()

    for epoch in range(resume_epoch, args.epochs):  # skip completed epochs
        epoch_loss       = 0.0
        epoch_steps      = 0
        epoch_weighted_loss = 0.0

        for batch in dataloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            category       = batch['category'][0]  # batch_size=1, unwrap list

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # ── Per-example class weight ───────────────────────────────────
            # Scale loss by inverse frequency weight BEFORE backward pass.
            # This increases the gradient magnitude for rare ICD categories,
            # partially compensating for gradient clipping bias (Bagdasaryan 2019).
            # Privacy guarantee is unaffected — we are scaling the objective,
            # not the sampling distribution.
            if class_weights is not None:
                w = class_weights.get(category, 1.0)
                weighted_loss = loss * w
                weight_stats[round(w, 1)] += 1
            else:
                weighted_loss = loss
                w = 1.0
            # ──────────────────────────────────────────────────────────────

            weighted_loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss          += loss.item()
            epoch_weighted_loss += weighted_loss.item()
            epoch_steps         += 1
            global_step         += 1

            should_log = (
                global_step <= 10 or
                global_step == 50 or
                global_step == 100 or
                global_step % 500 == 0
            )

            if should_log:
                eps_spent = privacy_engine.get_epsilon(delta)
                avg_loss  = epoch_loss / epoch_steps
                log.info(
                    f"  e{epoch+1} step={global_step:>8,}/{total_steps:,}  "
                    f"loss={avg_loss:.4f}  "
                    f"eps={eps_spent:.4f}/{args.epsilon}  "
                    f"cat_weight={w:.3f}  "
                    f"{gpu_mem()}"
                )

        # ── End of epoch ───────────────────────────────────────────────────
        eps_spent = privacy_engine.get_epsilon(delta)
        avg_loss  = epoch_loss / epoch_steps
        log.info(f"EPOCH {epoch+1}/{args.epochs} COMPLETE  "
                 f"avg_loss={avg_loss:.4f}  "
                 f"avg_weighted_loss={epoch_weighted_loss/epoch_steps:.4f}  "
                 f"eps_spent={eps_spent:.4f}")

        ckpt_dir = output_dir / f'checkpoint-epoch{epoch+1}'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = model._module if hasattr(model, '_module') else model
        unwrapped.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        log.info(f"Checkpoint saved -> {ckpt_dir}")

    final_eps = privacy_engine.get_epsilon(delta)
    log.info(f"Training complete. Final epsilon = {final_eps:.4f}")

    if class_weights:
        log.info("Weight bucket distribution (weight -> n_examples):")
        for w_bucket, count in sorted(weight_stats.items()):
            log.info(f"  w≈{w_bucket:.1f}  {count:,} examples")

    return final_eps, sigma


# =========================================================================
# Main
# =========================================================================

def main(args):
    set_seed(args.seed)

    # ── Resume detection ───────────────────────────────────────────────────
    # Look for an existing incomplete run for this epsilon in the output dir.
    # If found, resume from the latest completed epoch checkpoint.
    # If not found, start a fresh run with a new timestamped directory.
    existing_runs = sorted(Path(args.output).glob(f"llama_dp_eps{args.epsilon}_*"))
    incomplete    = [r for r in existing_runs
                     if r.is_dir() and not (r / 'final' / 'adapter_config.json').exists()]

    if incomplete:
        # Use the most recent incomplete run
        output_dir = incomplete[-1]
        log.info(f"Found incomplete run: {output_dir}")
        resume_epoch, resume_ckpt = find_latest_checkpoint(output_dir, args.epochs)
        if resume_epoch >= args.epochs:
            log.info("All epochs already complete. Nothing to do.")
            return
    else:
        timestamp  = datetime.now().strftime('%Y%m%d_%H%M')
        run_name   = f"llama_dp_eps{args.epsilon}_{timestamp}"
        output_dir = Path(args.output) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        resume_epoch = 0
        resume_ckpt  = None
        log.info(f"Fresh run: {output_dir}")
    # ──────────────────────────────────────────────────────────────────────

    log.info(f"Output: {output_dir}")
    log.info(f"Resume epoch: {resume_epoch}/{args.epochs}")

    with open(output_dir / 'run_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # -- Tokenizer --
    log.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("Set pad_token = eos_token")

    # -- Dataset --
    dataset = BHCDataset(
        jsonl_path=args.data,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    # -- Class weights (Bagdasaryan 2019 / Rosenblatt 2024) ----------------
    class_weights = None
    if not args.no_class_weights:
        log.info("=" * 60)
        log.info("COMPUTING INVERSE FREQUENCY CLASS WEIGHTS")
        log.info("(Bagdasaryan et al. NeurIPS 2019; Rosenblatt et al. 2024)")
        log.info("=" * 60)

        # Build lightweight record list just for weight computation
        weight_records = [
            {'icd_category': cat}
            for cat in dataset.categories
        ]
        class_weights, weight_stats = compute_class_weights(weight_records)

        log.info(f"  Categories: {weight_stats['n_categories']}")
        log.info(f"  Weight range: min={weight_stats['weight_range']['min']}  "
                 f"max={weight_stats['weight_range']['max']}  "
                 f"mean={weight_stats['weight_range']['mean']}")
        log.info(f"  Top-5 highest weighted (rarest) categories:")
        for cat, w in list(weight_stats['category_weights'].items())[:5]:
            n = weight_stats['category_counts'][cat]
            log.info(f"    w={w:.3f}  n={n:,}  {cat[:55]}")
        log.info(f"  Bottom-5 lowest weighted (most common) categories:")
        for cat, w in list(weight_stats['category_weights'].items())[-5:]:
            n = weight_stats['category_counts'][cat]
            log.info(f"    w={w:.3f}  n={n:,}  {cat[:55]}")

        with open(output_dir / 'class_weights.json', 'w') as f:
            json.dump(weight_stats, f, indent=2)
        log.info(f"  Class weights saved -> {output_dir / 'class_weights.json'}")
    else:
        log.info("Class weighting disabled (--no_class_weights)")

    # -- Model --
    log.info(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    if resume_ckpt:
        log.info(f"Loading LoRA adapters from checkpoint: {resume_ckpt}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, resume_ckpt)
        model = model.merge_and_unload()
        # Re-apply fresh LoRA on top of merged weights for remaining epochs
        model = get_peft_model(model, get_lora_config())
        log.info("Loaded checkpoint and re-applied LoRA for remaining epochs")
    else:
        log.info("Applying LoRA adapters...")
        model = get_peft_model(model, get_lora_config())

    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Moving model to {device}...")
    model = model.to(device)
    log.info(f"Model on {device}. {gpu_mem()}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Train --
    final_eps, sigma = train(
        model, tokenizer, dataset, args, output_dir,
        class_weights=class_weights,
        resume_epoch=resume_epoch,
    )

    # -- Save final --
    final_dir = output_dir / 'final'
    final_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = model._module if hasattr(model, '_module') else model
    unwrapped.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # -- Privacy report --
    report = {
        'epsilon_target':    args.epsilon,
        'epsilon_actual':    final_eps,
        'delta':             compute_delta(len(dataset)),
        'noise_multiplier':  sigma,
        'n_records':         len(dataset),
        'epochs':            args.epochs,
        'max_grad_norm':     args.max_grad_norm,
        'lora_r':            8,
        'lora_alpha':        16,
        'max_length':        args.max_length,
        'lr':                args.lr,
        'class_weighting':   not args.no_class_weights,
    }
    with open(output_dir / 'privacy_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    log.info(f"Privacy report saved -> {output_dir / 'privacy_report.json'}")
    log.info(f"DONE. Final model -> {final_dir}")


# =========================================================================
# CLI
# =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DP fine-tuning of Llama-3.2-1B on MIMIC-IV BHC sections',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data',    default='./data/train.jsonl')
    parser.add_argument('--output',  default='./models')

    parser.add_argument('--epsilon', type=float, default=4.0)
    parser.add_argument('--epochs',  type=int,   default=2)
    parser.add_argument('--lr',      type=float, default=5e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--max_length',    type=int,   default=512)
    parser.add_argument('--seed',          type=int,   default=42)

    parser.add_argument('--no_class_weights', action='store_true',
                        help='Disable inverse frequency class weighting '
                             '(use for ablation runs to isolate imbalance effect)')

    args = parser.parse_args()
    main(args)
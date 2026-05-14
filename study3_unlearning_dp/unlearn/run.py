"""
Run unlearning on a pretrained LLM.

Memory strategy: LoRA adapters on the forget model reduce AdamW optimizer
states from ~56 GB (full 8B) to <300 MB. Reference model (NPO) is plain
frozen base. Gradient checkpointing cuts activation memory further.
Final save merges LoRA into base for clean downstream loading.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from .common import (
    UnlearnConfig,
    cycle,
    load_model_and_tokenizer,
    per_token_nll,
    prepare_dataloaders,
    save_checkpoint,
)
from .methods import (
    compute_unlearn_loss,
    method_needs_reference,
    method_needs_retain,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("unlearn.run")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def quick_ppl(model, loader, device, max_batches=20):
    model.eval()
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = per_token_nll(model, batch)
            ntok = (batch["labels"] != -100).sum().item()
            total_loss += loss.item() * ntok
            total_tok += ntok
    model.train()
    if total_tok == 0:
        return float("nan")
    return math.exp(total_loss / total_tok)


def run(cfg):
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("no CUDA -- exiting")
        raise SystemExit(1)

    log.info("loading model %s", cfg.base_model)
    model, tokenizer = load_model_and_tokenizer(cfg.base_model, bf16=cfg.bf16)
    model.config.use_cache = False

    if cfg.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        peft_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules),
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    if cfg.gradient_checkpointing:
        # enable_input_require_grads needed when embedding is frozen (as with LoRA)
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        log.info("gradient checkpointing enabled")

    model.to(device)
    model.train()

    reference_model = None
    if method_needs_reference(cfg.method):
        log.info("loading frozen reference model for NPO")
        reference_model, _ = load_model_and_tokenizer(cfg.base_model, bf16=cfg.bf16)
        reference_model.to(device)
        reference_model.eval()
        for p in reference_model.parameters():
            p.requires_grad = False

    log.info("preparing dataloaders")
    forget_loader, retain_loader = prepare_dataloaders(
        cfg.splits_path, tokenizer, cfg, need_retain=method_needs_retain(cfg.method),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    log.info("trainable params: %d", sum(p.numel() for p in trainable_params))
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = cfg.num_epochs * len(forget_loader) // cfg.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, cfg.warmup_steps, total_steps)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    retain_iter = cycle(retain_loader) if retain_loader is not None else None
    step, accum = 0, 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg.num_epochs):
        for forget_batch in forget_loader:
            forget_batch = {k: v.to(device) for k, v in forget_batch.items()}
            retain_batch = None
            if retain_iter is not None:
                rb = next(retain_iter)
                retain_batch = {k: v.to(device) for k, v in rb.items()}

            loss = compute_unlearn_loss(
                cfg.method, model, forget_batch, retain_batch, reference_model, cfg,
            )
            (loss / cfg.gradient_accumulation_steps).backward()
            accum += 1

            if accum == cfg.gradient_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                step += 1

                if step % 10 == 0:
                    log.info("epoch=%d step=%d loss=%.4f lr=%.2e",
                             epoch, step, loss.item(), scheduler.get_last_lr()[0])

                if step % cfg.eval_every_steps == 0:
                    retain_ppl = (
                        quick_ppl(model, retain_loader, device)
                        if retain_loader is not None else float("nan")
                    )
                    forget_ppl = quick_ppl(model, forget_loader, device)
                    log.info("eval step=%d retain_ppl=%.2f forget_ppl=%.2f",
                             step, retain_ppl, forget_ppl)

                if step % cfg.save_every_steps == 0:
                    save_checkpoint(model, tokenizer, out_dir, step, cfg,
                                    metrics={"loss": loss.item()}, merge_lora=False)

    # Merge LoRA into base weights for clean downstream loading
    final_dir = save_checkpoint(
        model, tokenizer, out_dir, step, cfg,
        metrics={"final": True, "loss": loss.item()},
        merge_lora=cfg.use_lora,
    )
    with open(out_dir / "final.json", "w") as f:
        json.dump({"final_checkpoint": str(final_dir), "method": cfg.method}, f, indent=2)
    log.info("unlearning complete. final checkpoint at %s", final_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method_override", default=None)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    if args.method_override:
        cfg_dict["method"] = args.method_override
    cfg = UnlearnConfig(**cfg_dict)
    run(cfg)


if __name__ == "__main__":
    main()

"""
RMU: Representation Misdirection for Unlearning
Li et al. (2024) "The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning"
https://arxiv.org/abs/2403.03218

Key idea: instead of gradient ascent (which inverts logits catastrophically), redirect
the internal representations of forget-set inputs toward a random vector in activation
space. A frozen reference model anchors the retain set via KL divergence.
This is far less destructive than GA/NPO, allowing DP re-tuning to recover utility.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import collate_fn, load_model_and_tokenizer, tokenize_function

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("unlearn.rmu")


@dataclass
class RMUConfig:
    base_model: str
    splits_path: str
    output_dir: str

    # RMU hyperparams — from Li et al. 2024 Table 6
    layer_id: int = 7           # transformer layer to misdirect (0-indexed after embeddings)
    alpha: float = 1200.0       # weight on forget misdirection loss
    steering_coeff: float = 20.0  # L2 norm of the random target vector

    num_steps: int = 150        # gradient steps (RMU converges fast; epochs not used)
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    batch_size: int = 4
    max_seq_length: int = 512

    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    bf16: bool = True
    seed: int = 42
    log_every: int = 10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def forget_loss(model, batch: dict, layer_id: int, steering_coeff: float) -> torch.Tensor:
    """
    MSE between hidden states at layer_id and a random unit vector * steering_coeff.

    One shared random direction per batch (broadcast over B and T), matching the
    original RMU implementation. Averaged over non-padding positions only.
    """
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
    )
    # hidden_states[0] = embedding output; hidden_states[i+1] = after transformer layer i
    hidden = outputs.hidden_states[layer_id + 1]  # [B, T, H]

    # Single random direction in H-dim, scaled to steering_coeff norm
    rand_dir = torch.randn(hidden.shape[-1], device=hidden.device, dtype=hidden.dtype)
    rand_dir = rand_dir / (rand_dir.norm() + 1e-8)
    target = rand_dir.view(1, 1, -1) * steering_coeff  # [1, 1, H]

    mask = batch["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
    sq_err = ((hidden - target.detach()) ** 2) * mask
    return sq_err.sum() / (mask.sum() * hidden.shape[-1] + 1e-8)


def retain_loss(model, frozen_model, batch: dict) -> torch.Tensor:
    """
    KL divergence KL(frozen || current) on retain set.
    Keeps the current model close to the original on non-forget data.
    """
    with torch.no_grad():
        ref_logits = frozen_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits  # [B, T, V]

    cur_logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    ).logits  # [B, T, V]

    mask = batch["attention_mask"].bool()  # [B, T]
    log_p_cur = F.log_softmax(cur_logits[mask], dim=-1)
    p_ref = F.softmax(ref_logits[mask], dim=-1)
    return F.kl_div(log_p_cur, p_ref, reduction="batchmean")


def run(cfg: RMUConfig) -> None:
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("no CUDA -- exiting")

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

    # No gradient checkpointing: we need hidden states during forward, and with
    # batch_size=4 at seq_len=512 we're well within 80 GB HBM.
    model.to(device)
    model.train()

    log.info("loading frozen reference model")
    frozen_model, _ = load_model_and_tokenizer(cfg.base_model, bf16=cfg.bf16)
    frozen_model.to(device)
    frozen_model.eval()
    for p in frozen_model.parameters():
        p.requires_grad = False

    log.info("loading data from %s", cfg.splits_path)
    splits = load_from_disk(cfg.splits_path)
    tok_fn = tokenize_function(tokenizer, cfg.max_seq_length)
    col_fn = collate_fn(tokenizer)

    forget_ds = splits["finetune"].map(tok_fn, batched=True, remove_columns=splits["finetune"].column_names)
    retain_ds = splits["retain"].map(tok_fn, batched=True, remove_columns=splits["retain"].column_names)

    forget_loader = DataLoader(
        forget_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=col_fn, num_workers=0, drop_last=True,
    )
    retain_loader = DataLoader(
        retain_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=col_fn, num_workers=0, drop_last=True,
    )

    forget_iter = _cycle(forget_loader)
    retain_iter = _cycle(retain_loader)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    log.info(
        "RMU: %d steps | layer_id=%d | alpha=%.0f | steering_coeff=%.1f",
        cfg.num_steps, cfg.layer_id, cfg.alpha, cfg.steering_coeff,
    )

    for step in range(1, cfg.num_steps + 1):
        f_batch = {k: v.to(device) for k, v in next(forget_iter).items()}
        r_batch = {k: v.to(device) for k, v in next(retain_iter).items()}

        f_loss = forget_loss(model, f_batch, cfg.layer_id, cfg.steering_coeff)
        r_loss = retain_loss(model, frozen_model, r_batch)
        loss = cfg.alpha * f_loss + r_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
        optimizer.step()

        if step % cfg.log_every == 0:
            log.info(
                "step=%d forget_loss=%.4f retain_kl=%.4f total=%.4f",
                step, f_loss.item(), r_loss.item(), loss.item(),
            )

    # Save merged checkpoint at step_final (mirrors step_393 pattern used by other methods)
    ckpt_dir = Path(cfg.output_dir) / "step_final"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log.info("saving checkpoint to %s", ckpt_dir)
    if cfg.use_lora and hasattr(model, "merge_and_unload"):
        merged = model.merge_and_unload()
        merged.save_pretrained(ckpt_dir)
    else:
        model.save_pretrained(ckpt_dir)

    tokenizer.save_pretrained(ckpt_dir)
    with open(ckpt_dir / "unlearn_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    log.info("RMU complete. Checkpoint: %s", ckpt_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    # yaml may parse lora_target_modules as list; convert to tuple for dataclass
    if "lora_target_modules" in cfg_dict:
        cfg_dict["lora_target_modules"] = tuple(cfg_dict["lora_target_modules"])
    cfg = RMUConfig(**cfg_dict)
    run(cfg)


if __name__ == "__main__":
    main()

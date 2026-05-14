"""
RMU-SAM: Representation Misdirection for Unlearning with Sharpness-Aware Minimization

Combines RMU (Li et al. 2024) with SAM (Foret et al. 2021) to create unlearned
states that are resistant to relearning. Standard RMU creates shallow perturbations
that DP-LoRA retraining easily reverses. SAM pushes the model toward flat minima
in the unlearning objective, making the unlearned state robust to subsequent
fine-tuning steps.

Reference:
  - Li et al. (2024) "The WMDP Benchmark" (RMU)
  - Foret et al. (2021) "Sharpness-Aware Minimization for Efficiently Improving Generalization"
  - Fan et al. (2025) "SAM-enhanced unlearning" (ICML 2025, arXiv:2502.05374)
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
log = logging.getLogger("unlearn.rmu_sam")


@dataclass
class RMUSAMConfig:
    base_model: str
    splits_path: str
    output_dir: str

    # RMU hyperparams
    layer_id: int = 7
    alpha: float = 1200.0
    steering_coeff: float = 20.0

    # SAM hyperparams
    sam_rho: float = 0.05           # perturbation radius for SAM ascent step
    sam_adaptive: bool = False      # adaptive SAM (scale rho per-parameter by weight norm)

    num_steps: int = 150
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
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
    )
    hidden = outputs.hidden_states[layer_id + 1]
    rand_dir = torch.randn(hidden.shape[-1], device=hidden.device, dtype=hidden.dtype)
    rand_dir = rand_dir / (rand_dir.norm() + 1e-8)
    target = rand_dir.view(1, 1, -1) * steering_coeff
    mask = batch["attention_mask"].unsqueeze(-1).float()
    sq_err = ((hidden - target.detach()) ** 2) * mask
    return sq_err.sum() / (mask.sum() * hidden.shape[-1] + 1e-8)


def retain_loss(model, frozen_model, batch: dict) -> torch.Tensor:
    with torch.no_grad():
        ref_logits = frozen_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits
    cur_logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    ).logits
    mask = batch["attention_mask"].bool()
    log_p_cur = F.log_softmax(cur_logits[mask], dim=-1)
    p_ref = F.softmax(ref_logits[mask], dim=-1)
    return F.kl_div(log_p_cur, p_ref, reduction="batchmean")


class SAM:
    """
    Sharpness-Aware Minimization wrapper around a base optimizer.

    Two-step process per iteration:
      1. first_step(): perturb weights to w + epsilon (steepest ascent within rho-ball)
      2. second_step(): compute gradients at perturbed point, step from original weights

    This finds parameters where the loss landscape is flat, making the unlearned
    state resistant to small parameter updates during subsequent fine-tuning.
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False):
        self.params = list(params)
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive
        self._backup = {}

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self._backup[i] = p.data.clone()
            e_w = p.grad * scale
            if self.adaptive:
                e_w *= torch.pow(p.data, 2)
            p.data.add_(e_w)

    @torch.no_grad()
    def second_step(self):
        for i, p in enumerate(self.params):
            if i in self._backup:
                p.data.copy_(self._backup[i])
        self._backup = {}
        self.base_optimizer.step()

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.params[0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p.data) if self.adaptive else 1.0) * p.grad).norm(p=2).to(shared_device)
                for p in self.params if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def zero_grad(self):
        self.base_optimizer.zero_grad()


def run(cfg: RMUSAMConfig) -> None:
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
    base_optimizer = AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sam = SAM(trainable, base_optimizer, rho=cfg.sam_rho, adaptive=cfg.sam_adaptive)

    log.info(
        "RMU-SAM: %d steps | layer_id=%d | alpha=%.0f | steering_coeff=%.1f | rho=%.3f",
        cfg.num_steps, cfg.layer_id, cfg.alpha, cfg.steering_coeff, cfg.sam_rho,
    )

    for step in range(1, cfg.num_steps + 1):
        f_batch = {k: v.to(device) for k, v in next(forget_iter).items()}
        r_batch = {k: v.to(device) for k, v in next(retain_iter).items()}

        # --- SAM step 1: compute loss, ascend to sharpest point ---
        f_loss_1 = forget_loss(model, f_batch, cfg.layer_id, cfg.steering_coeff)
        r_loss_1 = retain_loss(model, frozen_model, r_batch)
        loss_1 = cfg.alpha * f_loss_1 + r_loss_1

        sam.zero_grad()
        loss_1.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
        sam.first_step()

        # --- SAM step 2: compute loss at perturbed point, descend from original ---
        f_loss_2 = forget_loss(model, f_batch, cfg.layer_id, cfg.steering_coeff)
        r_loss_2 = retain_loss(model, frozen_model, r_batch)
        loss_2 = cfg.alpha * f_loss_2 + r_loss_2

        sam.zero_grad()
        loss_2.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
        sam.second_step()

        if step % cfg.log_every == 0:
            log.info(
                "step=%d forget=%.4f retain_kl=%.4f total=%.4f (perturbed: %.4f)",
                step, f_loss_1.item(), r_loss_1.item(), loss_1.item(), loss_2.item(),
            )

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

    log.info("RMU-SAM complete. Checkpoint: %s", ckpt_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    if "lora_target_modules" in cfg_dict:
        cfg_dict["lora_target_modules"] = tuple(cfg_dict["lora_target_modules"])
    cfg = RMUSAMConfig(**cfg_dict)
    run(cfg)


if __name__ == "__main__":
    main()

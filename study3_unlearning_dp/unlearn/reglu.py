"""
ReGLU: Representation-Guided Parameter-Efficient LLM Unlearning.
Reverse-engineered from Xiao et al. (arXiv:2604.17396, April 2026).

Algorithm:
  1. Extract hidden states at target layer for forget + retain data
  2. Compute balanced covariance: CovΔ = (1-β)·CovF - β·CovR
  3. Eigendecompose → top-r eigenvectors = optimal forgetting directions
  4. Initialize LoRA A matrices at target layer from these eigenvectors
  5. Train with gradient ascent (forget) + KL (retain) + ROL penalty
     ROL = ||B·A · V_retain||² prevents updates from corrupting retain subspace

Usage:
  python -m unlearn.reglu --config config/reglu.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("unlearn.reglu")


@dataclass
class ReGLUConfig:
    base_model: str
    splits_path: str
    output_dir: str

    layer_id: int = 20

    # ReGLU
    beta: float = 0.5
    rol_weight: float = 0.1
    n_retain_pcs: int = 64
    n_repr_batches: int = 10
    forget_weight: float = 1.0
    retain_weight: float = 1.0

    # Training
    num_steps: int = 150
    learning_rate: float = 5e-5
    batch_size: int = 4
    max_seq_length: int = 512
    max_grad_norm: float = 1.0

    # LoRA
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
    save_merged: bool = False


# ── helpers ──────────────────────────────────────────────────────────

def _cycle(loader):
    while True:
        yield from loader


@torch.no_grad()
def _extract_hidden(model, loader, layer_id, device, n_batches):
    model.eval()
    parts = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
        h = out.hidden_states[layer_id + 1]
        mask = batch["attention_mask"].bool()
        parts.append(h[mask].float().cpu())
    model.train()
    return torch.cat(parts, dim=0)


def _eigen_directions(H_forget, H_retain, beta, r, n_pcs):
    H_F = H_forget - H_forget.mean(0, keepdim=True)
    H_R = H_retain - H_retain.mean(0, keepdim=True)

    CovF = (H_F.T @ H_F) / max(H_F.shape[0] - 1, 1)
    CovR = (H_R.T @ H_R) / max(H_R.shape[0] - 1, 1)
    CovDelta = (1 - beta) * CovF - beta * CovR

    vals_d, vecs_d = torch.linalg.eigh(CovDelta)
    forget_dirs = vecs_d[:, -r:].T.contiguous()

    vals_r, vecs_r = torch.linalg.eigh(CovR)
    retain_pcs = vecs_r[:, -n_pcs:].T.contiguous()

    log.info(
        "CovDelta top eigenvalues: %s",
        [f"{v:.2f}" for v in vals_d[-r:].flip(0).tolist()],
    )
    return forget_dirs, retain_pcs


def _init_lora_eigenvectors(model, layer_id, forget_dirs):
    hidden = forget_dirs.shape[1]
    count = 0
    for name, mod in model.named_modules():
        if f"layers.{layer_id}." not in name:
            continue
        if not hasattr(mod, "lora_A"):
            continue
        A = mod.lora_A["default"].weight
        if A.shape[1] != hidden:
            continue
        r = min(A.shape[0], forget_dirs.shape[0])
        with torch.no_grad():
            A[:r].copy_(forget_dirs[:r].to(A.dtype))
            mod.lora_B["default"].weight.zero_()
        count += 1
        log.info("  eigen-init %s (r=%d)", name, r)
    return count


def _rol_loss(model, layer_id, retain_pcs):
    total = torch.tensor(0.0, device=retain_pcs.device)
    for name, mod in model.named_modules():
        if f"layers.{layer_id}." not in name:
            continue
        if not hasattr(mod, "lora_A"):
            continue
        A = mod.lora_A["default"].weight
        B = mod.lora_B["default"].weight
        if A.shape[1] != retain_pcs.shape[1]:
            continue
        V = retain_pcs.to(dtype=A.dtype)
        AV = A @ V.T
        BAV = B @ AV
        total = total + (BAV ** 2).sum()
    return total


def _retain_kl(model, frozen, batch):
    with torch.no_grad():
        f_logits = frozen(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits
    c_logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    ).logits
    mask = batch["attention_mask"].bool()
    return F.kl_div(
        F.log_softmax(c_logits[mask], dim=-1),
        F.softmax(f_logits[mask], dim=-1),
        reduction="batchmean",
    )


# ── main ─────────────────────────────────────────────────────────────

def run(cfg: ReGLUConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    # model + tokenizer
    log.info("loading model %s", cfg.base_model)
    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # data
    log.info("loading data from %s", cfg.splits_path)
    splits = load_from_disk(cfg.splits_path)

    def tok_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg.max_seq_length,
            padding="max_length",
        )

    forget_ds = splits["finetune"].map(
        tok_fn, batched=True, remove_columns=splits["finetune"].column_names,
    )
    retain_ds = splits["retain"].map(
        tok_fn, batched=True, remove_columns=splits["retain"].column_names,
    )
    forget_ds.set_format("torch")
    retain_ds.set_format("torch")

    forget_loader = DataLoader(
        forget_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
    )
    retain_loader = DataLoader(
        retain_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
    )

    model.to(device)

    # ── step 1: hidden-state covariance ──
    log.info("extracting hidden states at layer %d ...", cfg.layer_id)
    H_F = _extract_hidden(model, forget_loader, cfg.layer_id, device, cfg.n_repr_batches)
    H_R = _extract_hidden(model, retain_loader, cfg.layer_id, device, cfg.n_repr_batches)
    log.info("collected %d forget tokens, %d retain tokens", len(H_F), len(H_R))

    # ── step 2: eigen-directions ──
    forget_dirs, retain_pcs = _eigen_directions(
        H_F, H_R, cfg.beta, cfg.lora_r, cfg.n_retain_pcs,
    )
    del H_F, H_R

    # ── step 3: LoRA with eigen-init ──
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    n_init = _init_lora_eigenvectors(model, cfg.layer_id, forget_dirs)
    log.info("eigen-initialized %d LoRA modules at layer %d", n_init, cfg.layer_id)
    model.print_trainable_parameters()

    # frozen copy for retain KL
    log.info("loading frozen model for retain KL ...")
    frozen = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype)
    frozen.to(device).eval()
    for p in frozen.parameters():
        p.requires_grad = False

    # ── step 4: train ──
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
    )
    forget_iter = _cycle(forget_loader)
    retain_iter = _cycle(retain_loader)
    rp = retain_pcs.to(device=device, dtype=dtype)

    model.train()
    log.info(
        "ReGLU: %d steps | layer=%d | beta=%.2f | rol=%.3f | fw=%.1f | rw=%.1f",
        cfg.num_steps, cfg.layer_id, cfg.beta, cfg.rol_weight,
        cfg.forget_weight, cfg.retain_weight,
    )

    for step in range(1, cfg.num_steps + 1):
        # forget — gradient ascent
        fb = {k: v.to(device) for k, v in next(forget_iter).items()}
        labels = fb["input_ids"].clone()
        labels[~fb["attention_mask"].bool()] = -100
        f_out = model(
            input_ids=fb["input_ids"],
            attention_mask=fb["attention_mask"],
            labels=labels,
        )
        f_loss = -f_out.loss

        # retain — KL
        rb = {k: v.to(device) for k, v in next(retain_iter).items()}
        r_loss = _retain_kl(model, frozen, rb)

        # ROL
        rol = _rol_loss(model, cfg.layer_id, rp)

        loss = cfg.forget_weight * f_loss + cfg.retain_weight * r_loss + cfg.rol_weight * rol
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        if step % cfg.log_every == 0 or step == 1:
            log.info(
                "step=%d forget=%.4f retain_kl=%.4f rol=%.6f total=%.4f",
                step, f_loss.item(), r_loss.item(), rol.item(), loss.item(),
            )

    # ── save ──
    ckpt_dir = Path(cfg.output_dir) / "step_final"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log.info("saving checkpoint to %s", ckpt_dir)
    if cfg.use_lora and hasattr(model, "merge_and_unload"):
        if cfg.save_merged:
            merged = model.merge_and_unload()
            merged.save_pretrained(ckpt_dir)
        else:
            model.save_pretrained(ckpt_dir)
            log.info("saved adapter only (save_merged=False)")
    else:
        model.save_pretrained(ckpt_dir)

    tokenizer.save_pretrained(ckpt_dir)
    with open(ckpt_dir / "unlearn_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    log.info("ReGLU complete. Checkpoint: %s", ckpt_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    if "lora_target_modules" in cfg_dict:
        cfg_dict["lora_target_modules"] = tuple(cfg_dict["lora_target_modules"])

    run(ReGLUConfig(**cfg_dict))

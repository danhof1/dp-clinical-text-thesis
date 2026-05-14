"""
DP-LoRA fine-tuning on MTSamples.

We fine-tune either:
  - the untouched base model (BASELINE: the "unsound" pipeline)
  - an unlearned checkpoint from unlearn/run.py (OURS)

The DP guarantee (ε, δ) applies to the fine-tuning step only. The research
question is whether applying unlearning first meaningfully changes the
empirical privacy of the resulting generator.

Important implementation notes
------------------------------
1. opacus and PEFT have historically had friction: opacus wraps the model
   with GradSampleModule and PEFT also wraps with PeftModel. Order matters.
   The working pattern is: load base -> attach LoRA via PEFT -> freeze base
   params (PEFT does this automatically) -> hand to opacus's PrivacyEngine
   with `make_private_with_epsilon`. Opacus will correctly compute
   per-sample gradients only for trainable LoRA params.

2. Because only LoRA params are trainable, per-sample gradient memory is
   dramatically lower than full-FT DP-SGD, making this tractable on a
   single H100.

3. Gradient accumulation + DP: opacus's BatchMemoryManager handles this,
   so we do NOT use transformers' built-in gradient_accumulation_steps.
   We use a logical batch size and a per-physical-step micro-batch.

4. Privacy accounting is done by opacus's RDP accountant; we specify
   target ε and δ up front and opacus computes the noise multiplier σ.

5. For ε = ∞ (non-private baseline), we skip opacus entirely and run
   standard LoRA fine-tuning. This gives us a clean non-private reference.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("finetune.dp_lora")


@dataclass
class DPLoRAConfig:
    # Inputs
    base_model: str                    # path to base (or unlearned) model
    splits_path: str
    output_dir: str

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # Training
    num_epochs: int = 3
    physical_batch_size: int = 8       # per-GPU micro-batch
    logical_batch_size: int = 64       # effective batch (for DP sampling rate)
    learning_rate: float = 5e-4
    max_seq_length: int = 1024
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0         # DP clipping norm C

    # DP
    epsilon: float = 8.0               # target ε;  set to null/None for non-private
    delta: float = 1e-5                # target δ
    secure_mode: bool = False          # opacus secure RNG, slower

    # Precision
    bf16: bool = True

    # Goldfish Loss (Maini et al. 2024)
    goldfish_ratio: float = 0.0        # fraction of tokens to mask from loss (0 = disabled)

    # Misc
    seed: int = 42
    log_every: int = 10


def _should_apply_dp(cfg: DPLoRAConfig) -> bool:
    return cfg.epsilon is not None and math.isfinite(cfg.epsilon)


def tokenize_and_collate(tokenizer, max_length: int):
    def _tok(examples):
        out = tokenizer(
            examples["text"], truncation=True, max_length=max_length, padding=False,
        )
        out["labels"] = [list(ids) for ids in out["input_ids"]]
        return out

    pad_id = tokenizer.pad_token_id

    def _collate(features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attn, labels = [], [], []
        for f in features:
            ids = list(f["input_ids"])
            pad_n = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_n)
            attn.append([1] * len(ids) + [0] * pad_n)
            labels.append(list(f["labels"]) + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attn),
            "labels": torch.tensor(labels),
        }

    return _tok, _collate



class _TupleDataset(torch.utils.data.Dataset):
    """Wrap HF dataset to return tuples.

    Opacus DPDataLoader does `for x in dataset[0]` to detect dtypes.
    HF datasets return dicts; iterating a dict gives keys (str), not tensors.
    Returning tuples lets Opacus correctly call .dtype on each element.
    """
    KEYS = ("input_ids", "attention_mask", "labels")

    def __init__(self, hf_ds):
        self._ds = hf_ds

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        item = self._ds[idx]
        return tuple(torch.tensor(item[k], dtype=torch.long) for k in self.KEYS)


def _make_tuple_collate(pad_id):
    def _collate(features):
        if len(features) == 0:
            return {
                "input_ids": torch.zeros((0, 1), dtype=torch.long),
                "attention_mask": torch.zeros((0, 1), dtype=torch.long),
                "labels": torch.full((0, 1), -100, dtype=torch.long),
            }
        max_len = max(len(f[0]) for f in features)
        input_ids, attn, labels = [], [], []
        for f in features:
            ids = f[0].tolist()
            pad_n = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_n)
            attn.append([1] * len(ids) + [0] * pad_n)
            lab = f[2].tolist()
            labels.append(lab + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    return _collate


def run(cfg: DPLoRAConfig):
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.error("DP-LoRA requires GPU"); raise SystemExit(1)

    # --- model + tokenizer ---
    log.info("loading base %s", cfg.base_model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=dtype, attn_implementation="sdpa",
    )
    model.config.use_cache = False  # needed for training

    # --- LoRA ---
    peft_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    model.to(device)

    # --- data ---
    splits = load_from_disk(cfg.splits_path)
    ft_ds = splits["finetune"]
    tok_fn, _ = tokenize_and_collate(tokenizer, cfg.max_seq_length)
    ft_tok = ft_ds.map(tok_fn, batched=True, remove_columns=ft_ds.column_names)
    # Wrap in tuple dataset so Opacus can introspect dtypes (HF dict iteration yields keys)
    tuple_ds = _TupleDataset(ft_tok)
    tuple_collate = _make_tuple_collate(tokenizer.pad_token_id)

    loader = DataLoader(
        tuple_ds, batch_size=cfg.physical_batch_size, shuffle=True,
        collate_fn=tuple_collate, num_workers=0, drop_last=True,
    )

    # --- optimizer + scheduler (set up BEFORE opacus wraps) ---
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    total_logical_steps = (
        cfg.num_epochs * len(ft_tok) // cfg.logical_batch_size
    )
    warmup_steps = max(1, int(total_logical_steps * cfg.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_logical_steps)

    # --- opacus wrap (if DP) ---
    privacy_engine = None
    if _should_apply_dp(cfg):
        from opacus import PrivacyEngine
        from opacus.utils.batch_memory_manager import BatchMemoryManager

        privacy_engine = PrivacyEngine(secure_mode=cfg.secure_mode)
        # Epochs are translated into sample rate + #steps internally
        model, optimizer, loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            target_epsilon=cfg.epsilon,
            target_delta=cfg.delta,
            epochs=cfg.num_epochs,
            max_grad_norm=cfg.max_grad_norm,
            poisson_sampling=True,
        )
        log.info(
            "DP engaged: target ε=%.2f δ=%.1e, noise_multiplier=%.4f, C=%.2f",
            cfg.epsilon, cfg.delta,
            optimizer.noise_multiplier, cfg.max_grad_norm,
        )
        batch_mgr_cm = lambda: BatchMemoryManager(
            data_loader=loader,
            max_physical_batch_size=cfg.physical_batch_size,
            optimizer=optimizer,
        )
    else:
        log.info("ε is infinite — running non-private LoRA fine-tuning baseline")
        batch_mgr_cm = None

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- training loop ---
    model.train()
    global_step = 0

    for epoch in range(cfg.num_epochs):
        if batch_mgr_cm is not None:
            from contextlib import ExitStack
            context_stack = ExitStack()
            safe_loader = context_stack.enter_context(batch_mgr_cm())
        else:
            safe_loader = loader
            context_stack = None

        try:
            for step, batch in enumerate(safe_loader):
                if not isinstance(batch, dict) or batch["input_ids"].shape[0] == 0:
                    continue  # empty batch from Poisson sampling
                batch = {k: v.to(device) for k, v in batch.items()}
                if cfg.goldfish_ratio > 0:
                    gf_mask = torch.rand_like(batch["labels"], dtype=torch.float32) < cfg.goldfish_ratio
                    batch["labels"] = batch["labels"].masked_fill(
                        gf_mask & (batch["labels"] != -100), -100
                    )
                out = model(**batch)
                loss = out.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.log_every == 0:
                    msg = f"epoch={epoch} step={global_step} loss={loss.item():.4f}"
                    if privacy_engine is not None:
                        eps_so_far = privacy_engine.get_epsilon(cfg.delta)
                        msg += f" eps_spent={eps_so_far:.3f}"
                    log.info(msg)
        finally:
            if context_stack is not None:
                context_stack.close()

    # --- save ---
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    # PEFT save: save only the adapter weights; user can merge later
    # If opacus wrapped the model, we need to get the underlying module
    underlying = model._module if hasattr(model, "_module") else model
    underlying.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    manifest = {
        "base_model": cfg.base_model,
        "epsilon": cfg.epsilon,
        "delta": cfg.delta,
        "final_epsilon": (
            privacy_engine.get_epsilon(cfg.delta) if privacy_engine is not None else None
        ),
        "config": asdict(cfg),
    }
    with open(final_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    log.info("saved final adapter to %s", final_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base_model_override", default=None,
                    help="override base_model (e.g. to point at an unlearned checkpoint)")
    ap.add_argument("--epsilon_override", default=None, type=float)
    ap.add_argument("--output_dir_override", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    if args.base_model_override:
        cfg_dict["base_model"] = args.base_model_override
    if args.epsilon_override is not None:
        cfg_dict["epsilon"] = args.epsilon_override
    if args.output_dir_override:
        cfg_dict["output_dir"] = args.output_dir_override

    cfg = DPLoRAConfig(**cfg_dict)
    run(cfg)


if __name__ == "__main__":
    main()

"""
Shadow model training for LiRA (Carlini et al. 2022) and U-LiRA (Hayes et al. 2024).

LiRA fits per-example Gaussians over shadow-model losses under the IN
(trained-on) and OUT (held-out) hypotheses. To do that, we need N shadow
models where each example x appears IN roughly half of them and OUT of
the other half.

For U-LiRA (unlearning version), we additionally train an "unlearned"
counterpart of each IN shadow model, by running the same unlearning
procedure used by the target on the same data.

The setup
---------
- Population P: the union of members + nonmembers (and optionally
  clean_nonmembers, though for same-distribution LiRA we stick to MTSamples).
- For each shadow model i in [0, N):
    - Sample a random mask m_i in {0,1}^|P| with P[m_i(x) = 1] = 0.5
    - Train shadow model i on {x : m_i(x) = 1}
    - Record which examples are IN vs OUT for shadow i
- For each target example x, we then have approximately N/2 IN-shadow models
  and N/2 OUT-shadow models. Their loss distributions give us the
  per-example LiRA Gaussians.

Scale
-----
Canonical LiRA at CIFAR uses 256 shadow models. At LLM scale this is
infeasible; we default to 8 shadow models for practical experiments,
which is enough for offline-LiRA with a single "out" population distribution
but too few for strong online-LiRA. We document this as a limitation.

For U-LiRA, we additionally train 8 "unlearned" counterparts, so 16 total
training runs per method × ε configuration. With H100s, LoRA fine-tuning
of Llama-3-8B takes ~1 GPU-hour each, so a full run is ~16 GPU-hours.

All shadow models use LoRA (not full FT) for feasibility. LiRA at LoRA
scale is nonstandard; cite Panda et al. 2024 for precedent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset, concatenate_datasets, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("shadow.train")


@dataclass
class ShadowConfig:
    # What we're shadowing
    base_model: str
    splits_path: str
    output_root: str              # /path/to/shadow_models/

    n_shadows: int = 8            # number of shadow models to train
    sample_prob: float = 0.5      # P(x IN shadow i) for each example
    shadow_seed: int = 12345      # base seed; shadow i uses shadow_seed + i

    # Training (matches finetune/dp_lora.py non-private path)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
    )

    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-4
    max_seq_length: int = 1024
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    bf16: bool = True
    log_every: int = 20

    # For U-LiRA: also train an unlearned version of each shadow
    train_unlearned_counterpart: bool = False
    unlearn_method: str = "ga_gd"
    unlearn_learning_rate: float = 5e-6
    unlearn_epochs: int = 3


def _example_id(row) -> str:
    """Stable ID for membership records — use doc_id if present, else hash the text."""
    if "doc_id" in row and row["doc_id"]:
        return row["doc_id"]
    return hashlib.sha256(row["text"].encode("utf-8")).hexdigest()[:16]


def build_population(splits) -> Dataset:
    """
    Population = members ∪ nonmembers (both from MTSamples distribution).
    This is the set over which we resample IN/OUT masks for shadows.

    We explicitly EXCLUDE clean_nonmembers and canaries from the population
    used for LiRA shadow training — they serve distinct roles:
    - clean_nonmembers: used only at attack-time as a known-never-in reference.
    - canaries: used only for DP audit, not for MIA score calibration.
    """
    m = splits["members"]
    n = splits["nonmembers"]
    # Make sure both have the same columns
    cols = set(m.column_names) & set(n.column_names)
    m = m.remove_columns([c for c in m.column_names if c not in cols])
    n = n.remove_columns([c for c in n.column_names if c not in cols])
    pop = concatenate_datasets([m, n])
    return pop


def sample_membership_mask(population_size: int, sample_prob: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(population_size) < sample_prob).astype(np.int32)


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


def train_one_shadow(
    shadow_idx: int,
    in_dataset: Dataset,
    cfg: ShadowConfig,
    output_dir: Path,
):
    """Train one shadow model via non-private LoRA fine-tuning."""
    device = "cuda"
    torch.manual_seed(cfg.shadow_seed + shadow_idx)

    log.info("shadow %d: %d training examples", shadow_idx, len(in_dataset))

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=dtype, attn_implementation="sdpa",
    )
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    model.to(device)
    model.train()

    tok_fn, collate = tokenize_and_collate(tokenizer, cfg.max_seq_length)
    ds_tok = in_dataset.map(tok_fn, batched=True, remove_columns=in_dataset.column_names)
    loader = DataLoader(
        ds_tok, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate, num_workers=2, drop_last=True,
    )
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    total_steps = cfg.num_epochs * len(loader)
    warmup = max(1, int(total_steps * cfg.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    step = 0
    for epoch in range(cfg.num_epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.log_every == 0:
                log.info("shadow %d step %d loss %.4f", shadow_idx, step, out.loss.item())

    ckpt = output_dir / f"shadow_{shadow_idx:03d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    log.info("shadow %d saved to %s", shadow_idx, ckpt)

    # Free GPU memory before next shadow
    del model, optimizer
    torch.cuda.empty_cache()
    return ckpt


def train_unlearned_counterpart(
    shadow_idx: int,
    in_dataset: Dataset,
    shadow_ckpt: Path,
    cfg: ShadowConfig,
    output_dir: Path,
):
    """
    For U-LiRA: take the trained shadow model and run the same unlearning
    procedure on its IN set. This simulates what happens when a user of the
    target model requests unlearning of their data.
    """
    # We use the already-tested unlearning code to keep behavior identical.
    # Import here to avoid pulling unlearn imports when not needed.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from unlearn.common import UnlearnConfig
    from unlearn.run import run as run_unlearn

    # Persist a tiny DatasetDict for the unlearning run to find
    unl_dir = output_dir / f"shadow_{shadow_idx:03d}_unlearn_splits"
    unl_dir.mkdir(parents=True, exist_ok=True)
    from datasets import DatasetDict
    dd = DatasetDict({
        "finetune": in_dataset,  # unlearning treats "finetune" split as the forget set
        # We borrow the retain set from the original splits
    })
    # We need a retain set too; load it back
    orig_splits = load_from_disk(cfg.splits_path)
    dd = DatasetDict({
        "finetune": in_dataset,
        "retain": orig_splits["retain"],
    })
    dd.save_to_disk(str(unl_dir))

    unlearn_out = output_dir / f"shadow_{shadow_idx:03d}_unlearned"
    ucfg = UnlearnConfig(
        method=cfg.unlearn_method,
        base_model=str(shadow_ckpt),  # start from the trained shadow
        splits_path=str(unl_dir),
        output_dir=str(unlearn_out),
        learning_rate=cfg.unlearn_learning_rate,
        num_epochs=cfg.unlearn_epochs,
        bf16=cfg.bf16,
        seed=cfg.shadow_seed + shadow_idx + 999_999,
    )
    run_unlearn(ucfg)
    return unlearn_out


def run(cfg: ShadowConfig):
    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    splits = load_from_disk(cfg.splits_path)
    population = build_population(splits)
    N = len(population)
    log.info("population size: %d", N)

    # Persist example IDs in a stable order — used at attack time to align
    # membership masks with score vectors
    ids = [_example_id(r) for r in population]
    with open(output_root / "population_ids.json", "w") as f:
        json.dump(ids, f)

    # Persist the population dataset itself so we don't rebuild it at attack time
    population.save_to_disk(str(output_root / "population"))

    membership_matrix = np.zeros((cfg.n_shadows, N), dtype=np.int32)

    for i in range(cfg.n_shadows):
        seed_i = cfg.shadow_seed + i
        mask = sample_membership_mask(N, cfg.sample_prob, seed_i)
        membership_matrix[i] = mask

        in_indices = np.where(mask == 1)[0].tolist()
        in_dataset = population.select(in_indices)

        ckpt = train_one_shadow(i, in_dataset, cfg, output_root)

        if cfg.train_unlearned_counterpart:
            train_unlearned_counterpart(i, in_dataset, ckpt, cfg, output_root)

    # Save the full membership matrix
    np.save(output_root / "membership_matrix.npy", membership_matrix)
    with open(output_root / "shadow_manifest.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)
    log.info("shadow training complete. %d models in %s", cfg.n_shadows, output_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = ShadowConfig(**yaml.safe_load(f))
    run(cfg)


if __name__ == "__main__":
    main()

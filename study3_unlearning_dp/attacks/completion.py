"""
Prefix completion extraction attack.

Given a model and a set of training examples, feed the first N tokens of each
example as a prompt and measure how much of the continuation the model
reproduces verbatim. This is the strongest test of memorization — if the model
can complete a training example from a prefix, it has memorized it.

Reports:
  - exact_match_rate: fraction of continuation tokens reproduced exactly
  - rouge_l: longest common subsequence ratio
  - extraction_rate_at_k: fraction of examples where >=k% of continuation is exact

Reference: Carlini et al. (2021) "Extracting Training Data from Large Language Models"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_from_disk
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("attacks.completion")


def longest_common_subsequence(a: list[int], b: list[int]) -> int:
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    if m > 500 or n > 500:
        return _lcs_approx(a, b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def _lcs_approx(a: list[int], b: list[int]) -> int:
    """Fast LCS approximation via longest common prefix at each offset."""
    b_set = set(b)
    matches = 0
    i = 0
    while i < len(a):
        if a[i] in b_set:
            matches += 1
        i += 1
    return min(matches, len(b))


def longest_common_prefix(a: list[int], b: list[int]) -> int:
    """Count exact token matches from the start."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def load_model(base_model: str, adapter_path: str | None, device: str = "cuda"):
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, attn_implementation="sdpa",
    )
    if adapter_path and (Path(adapter_path) / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        log.info("loaded and merged adapter %s", adapter_path)
    model.eval().to(device)
    return model


@torch.no_grad()
def complete_prefix(
    model, tokenizer, prefix_ids: list[int], max_new_tokens: int, device: str,
) -> list[int]:
    """Greedy completion of a token prefix."""
    input_ids = torch.tensor([prefix_ids], device=device)
    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = out[0][len(prefix_ids):].tolist()
    return generated


def run_completion_attack(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(cfg["base_model"], cfg.get("adapter_path"), device)

    splits = load_from_disk(cfg["splits_path"])
    max_length = cfg.get("max_length", 512)
    prefix_ratio = cfg.get("prefix_ratio", 0.5)
    max_examples = cfg.get("max_examples", 500)

    results_per_example = []

    dataset = splits["members"]
    if len(dataset) > max_examples:
        indices = list(range(len(dataset)))
        np.random.seed(42)
        np.random.shuffle(indices)
        indices = indices[:max_examples]
    else:
        indices = list(range(len(dataset)))

    log.info("running completion attack on %d members (prefix_ratio=%.2f)",
             len(indices), prefix_ratio)

    for idx in tqdm(indices, desc="completion attack"):
        row = dataset[idx]
        tokens = tokenizer(
            row["text"], add_special_tokens=False, truncation=True,
            max_length=max_length,
        )["input_ids"]

        if len(tokens) < 10:
            continue

        prefix_len = max(5, int(len(tokens) * prefix_ratio))
        prefix = tokens[:prefix_len]
        ground_truth = tokens[prefix_len:]

        if len(ground_truth) < 5:
            continue

        generated = complete_prefix(
            model, tokenizer, prefix,
            max_new_tokens=len(ground_truth), device=device,
        )

        gen_len = min(len(generated), len(ground_truth))
        if gen_len == 0:
            continue

        gt_trimmed = ground_truth[:gen_len]
        gen_trimmed = generated[:gen_len]

        exact_prefix_len = longest_common_prefix(gen_trimmed, gt_trimmed)
        exact_match_rate = sum(
            1 for a, b in zip(gen_trimmed, gt_trimmed) if a == b
        ) / gen_len

        lcs_len = longest_common_subsequence(
            gen_trimmed[:200], gt_trimmed[:200]
        )
        rouge_l = lcs_len / max(len(gt_trimmed[:200]), 1)

        results_per_example.append({
            "idx": idx,
            "prefix_len": prefix_len,
            "gt_len": len(ground_truth),
            "gen_len": len(generated),
            "exact_prefix_tokens": exact_prefix_len,
            "exact_match_rate": round(exact_match_rate, 4),
            "rouge_l": round(rouge_l, 4),
        })

    if not results_per_example:
        log.warning("no examples processed")
        return {}

    emrs = [r["exact_match_rate"] for r in results_per_example]
    rls = [r["rouge_l"] for r in results_per_example]
    prefixes = [r["exact_prefix_tokens"] for r in results_per_example]

    summary = {
        "n_examples": len(results_per_example),
        "prefix_ratio": prefix_ratio,
        "mean_exact_match_rate": round(float(np.mean(emrs)), 4),
        "median_exact_match_rate": round(float(np.median(emrs)), 4),
        "p95_exact_match_rate": round(float(np.percentile(emrs, 95)), 4),
        "max_exact_match_rate": round(float(np.max(emrs)), 4),
        "mean_rouge_l": round(float(np.mean(rls)), 4),
        "mean_exact_prefix_tokens": round(float(np.mean(prefixes)), 1),
        "max_exact_prefix_tokens": int(np.max(prefixes)),
        "extraction_rate_10pct": round(
            sum(1 for e in emrs if e >= 0.10) / len(emrs), 4
        ),
        "extraction_rate_25pct": round(
            sum(1 for e in emrs if e >= 0.25) / len(emrs), 4
        ),
        "extraction_rate_50pct": round(
            sum(1 for e in emrs if e >= 0.50) / len(emrs), 4
        ),
    }

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "completion_attack_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "completion_attack_per_example.json", "w") as f:
        json.dump(results_per_example, f, indent=2)

    log.info("\n=== Completion attack summary ===")
    for k, v in summary.items():
        log.info("  %-30s %s", k, v)

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_completion_attack(cfg)


if __name__ == "__main__":
    main()

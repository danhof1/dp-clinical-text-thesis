#!/usr/bin/env python3
"""
Paraphrase PMC case reports at three aggressiveness levels using Llama-3.1-8B-Instruct.

Produces ~100 samples per level for the data-shift diagnostic experiment.
Saves spot-check files (15 examples per level) for clinical fidelity review.

Usage:
  python paraphrase_pmc.py --level light
  python paraphrase_pmc.py --level moderate
  python paraphrase_pmc.py --level aggressive
  python paraphrase_pmc.py --level all
  python paraphrase_pmc.py --level all --n_samples 50
"""
import argparse
import json
import logging
import random
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("paraphrase_pmc")

# ─── Paths ───────────────────────────────────────────────────────
LLAMA = "/fs1/shared/model/llm/Llama-3.1-8B-Instruct"
SPLITS = (
    "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/"
    "New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/"
    "outputs/splits_v1_pmc"
)
OUT_DIR = (
    "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/"
    "New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/"
    "outputs/paraphrased_pmc"
)
CACHE_DIR = "/fs1/projects/unlearning_pretraining/Proj_code/.cache"

# ─── Prompts ─────────────────────────────────────────────────────
SYSTEM_MSG = (
    "You are a medical text rewriting assistant. "
    "Output ONLY the rewritten text. No preamble, commentary, or meta-text."
)

PROMPTS = {
    "light": (
        "Reword the following medical text with minimal changes. "
        "Replace some words and short phrases with synonyms, but keep the "
        "sentence structure, paragraph organization, all medical terminology, "
        "and citation style essentially intact. "
        "Preserve every clinical fact, finding, and outcome exactly.\n\n"
        "Text:\n{text}"
    ),
    "moderate": (
        "Rewrite the following medical text in your own words. "
        "Use different sentence structures and phrasing throughout. "
        "Preserve all medical facts, clinical details, diagnoses, patient "
        "outcomes, and logical flow. Medical terminology should remain "
        "accurate but presentation may vary.\n\n"
        "Text:\n{text}"
    ),
    "aggressive": (
        "Completely rewrite the following medical text as if you are a "
        "different author writing about the same case from scratch. "
        "Substantially change the writing style, restructure the narrative, "
        "and use alternative ways to express medical concepts. "
        "Every clinical fact, diagnosis, finding, and outcome must be "
        "preserved accurately, but the surface form should be as different "
        "as possible from the original.\n\n"
        "Text:\n{text}"
    ),
}

SAMPLE_SEED = 7
MAX_INPUT_CHARS = 4000
SPOT_CHECK_N = 15


def paraphrase_one(model, tokenizer, text, level, device):
    prompt = PROMPTS[level].format(text=text[:MAX_INPUT_CHARS])
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        out[0][input_ids.shape[1]:], skip_special_tokens=True,
    )
    return response.strip()


def write_spot_check(records, level, out_dir):
    path = Path(out_dir) / f"spot_check_{level}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{'=' * 80}\n")
        f.write(f"SPOT CHECK — Level: {level.upper()} — {len(records)} examples\n")
        f.write(
            "Review for: clinical fact preservation, terminology accuracy, "
            "readability, surface-form change\n"
        )
        f.write(f"{'=' * 80}\n\n")
        for i, rec in enumerate(records[:SPOT_CHECK_N]):
            f.write(
                f"--- Example {i+1} "
                f"(pmcid: {rec['pmcid']}, idx: {rec['idx']}) ---\n\n"
            )
            f.write(f"ORIGINAL (first 800 chars):\n{rec['original_text'][:800]}\n\n")
            f.write(
                f"PARAPHRASE (first 800 chars):\n"
                f"{rec['paraphrase_text'][:800]}\n\n"
            )
            f.write(f"Original length:    {rec['original_chars']} chars\n")
            f.write(f"Paraphrase length:  {rec['paraphrase_chars']} chars\n")
            f.write(f"Generation time:    {rec['generation_time_s']}s\n")
            f.write(f"{'─' * 60}\n\n")
    log.info("Spot check written to %s (%d examples)", path, min(len(records), SPOT_CHECK_N))


def run(level, n_samples):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading splits from %s", SPLITS)
    ds = load_from_disk(SPLITS)
    finetune = ds["finetune"]

    random.seed(SAMPLE_SEED)
    indices = sorted(random.sample(range(len(finetune)), min(n_samples, len(finetune))))
    log.info("Selected %d sample indices (seed=%d)", len(indices), SAMPLE_SEED)

    # Save index list for diagnostic to match against
    idx_path = out_dir / "sample_indices.json"
    idx_path.write_text(json.dumps(indices))

    log.info("Loading %s", LLAMA)
    tokenizer = AutoTokenizer.from_pretrained(LLAMA, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=CACHE_DIR,
    )
    model.eval()

    levels = list(PROMPTS.keys()) if level == "all" else [level]

    for lvl in levels:
        log.info("Starting level=%s", lvl)
        jsonl_path = out_dir / f"paraphrased_{lvl}.jsonl"

        # Resume: skip completed indices
        done_indices = set()
        existing_records = []
        if jsonl_path.exists():
            with open(jsonl_path) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        done_indices.add(rec["idx"])
                        existing_records.append(rec)
            log.info("Resuming %s: %d/%d already done", lvl, len(done_indices), len(indices))

        new_records = []
        with open(jsonl_path, "a") as f:
            for count, idx in enumerate(indices):
                if idx in done_indices:
                    continue
                example = finetune[idx]
                text = example["text"]
                pmcid = example.get("pmcid", "unknown")

                t0 = time.time()
                paraphrase = paraphrase_one(model, tokenizer, text, lvl, device)
                elapsed = time.time() - t0

                rec = {
                    "idx": idx,
                    "pmcid": pmcid,
                    "level": lvl,
                    "original_text": text,
                    "paraphrase_text": paraphrase,
                    "original_chars": len(text),
                    "paraphrase_chars": len(paraphrase),
                    "generation_time_s": round(elapsed, 1),
                }
                f.write(json.dumps(rec) + "\n")
                f.flush()
                new_records.append(rec)

                total_done = len(done_indices) + len(new_records)
                if total_done % 10 == 0 or total_done == len(indices):
                    log.info(
                        "[%s] %d/%d done (%.1fs this sample)",
                        lvl, total_done, len(indices), elapsed,
                    )

        all_records = existing_records + new_records
        write_spot_check(all_records, lvl, out_dir)
        log.info(
            "Level=%s complete: %d paraphrases in %s",
            lvl, len(all_records), jsonl_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paraphrase PMC texts at controlled aggressiveness levels",
    )
    parser.add_argument(
        "--level",
        choices=["light", "moderate", "aggressive", "all"],
        required=True,
        help="Aggressiveness regime (or 'all' for all three)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of PMC texts to paraphrase per level (default: 100)",
    )
    args = parser.parse_args()
    run(args.level, args.n_samples)

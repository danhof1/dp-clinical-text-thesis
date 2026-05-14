"""
Corrected generation script for synthetic clinical notes.

Key fix: prompt with actual clinical note openings (section headers)
that match the training data format, NOT "Medical Specialty: X\nDescription: "
which produces job postings.

MTSamples training data starts with:
  PREOPERATIVE DIAGNOSIS: (23%)
  CHIEF COMPLAINT: (8%)
  PREOPERATIVE DIAGNOSES: (8%)
  HISTORY OF PRESENT ILLNESS: (5%)
  SUBJECTIVE: (4%)
  etc.

PMC training data starts with narrative medical prose.

Usage:
    python -m generate.run_v2 \
      --base_model /path/to/model \
      --adapter_path /path/to/adapter \
      --dataset mtsamples \
      --n_samples 200 \
      --output_path outputs/generated/corrected_rmu_eps8.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("generate_v2")

MAX_SENTENCE_CHARS = 2181

# Prompts sampled from actual MTSamples opening patterns, weighted by frequency.
# Each tuple: (prompt_text, weight)
MTS_PROMPTS = [
    ("PREOPERATIVE DIAGNOSIS: ", 351),
    ("CHIEF COMPLAINT:, ", 124),
    ("PREOPERATIVE DIAGNOSES: ", 120),
    ("HISTORY OF PRESENT ILLNESS: ", 69),
    ("SUBJECTIVE:, ", 66),
    ("EXAM:, ", 56),
    ("CC:, ", 51),
    ("PROCEDURE: ", 49),
    ("HISTORY: ", 43),
    ("REASON FOR CONSULTATION: ", 36),
    ("REASON FOR VISIT:, ", 33),
    ("PROCEDURE PERFORMED: ", 22),
    ("TITLE OF OPERATION: ", 21),
    ("DIAGNOSIS: ", 14),
    ("REASON FOR CONSULT:, ", 10),
    ("INDICATIONS: ", 10),
    ("ADMITTING DIAGNOSIS: ", 9),
    ("REASON FOR REFERRAL:, ", 9),
]

# PMC articles start with narrative prose. Use diverse medical openings.
PMC_PROMPTS = [
    ("A ", 30),
    ("The ", 30),
    ("We report ", 10),
    ("This study ", 10),
    ("Background: ", 8),
    ("Introduction: ", 5),
    ("Objective: ", 5),
    ("Case presentation: ", 5),
]


def get_prompt(dataset: str, rng: random.Random) -> tuple[str, str]:
    """Return (prompt_text, prompt_type) matching the training data format."""
    if dataset == "mtsamples":
        prompts, weights = zip(*MTS_PROMPTS)
        choice = rng.choices(prompts, weights=weights, k=1)[0]
        return choice, choice.strip().rstrip(":,").lower().replace(" ", "_")
    else:
        prompts, weights = zip(*PMC_PROMPTS)
        choice = rng.choices(prompts, weights=weights, k=1)[0]
        return choice, choice.strip().rstrip(":").lower().replace(" ", "_")


def passes_sentence_filter(text: str) -> bool:
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        if len(sent) > MAX_SENTENCE_CHARS:
            return False
    return True


def load_generator(base_model: str, adapter_path: str | None):
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        log.info("loaded adapter %s", adapter_path)
    model.eval().cuda()
    return model, tok


def load_ppl_model(base_model: str):
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).eval().cuda()
    log.info("loaded PPL reference model (base, no adapter)")
    return model, tok


@torch.no_grad()
def compute_ppl(text: str, model, tok, max_length: int = 1024) -> float:
    enc = tok(text, return_tensors="pt", truncation=True,
              max_length=max_length).to(model.device)
    if enc["input_ids"].shape[1] <= 1:
        return float("inf")
    loss = model(**enc, labels=enc["input_ids"]).loss
    return math.exp(loss.item())


@torch.no_grad()
def generate_candidates(
    model, tok, prompt: str, k: int,
    max_new_tokens: int, temperature: float,
    top_p: float, repetition_penalty: float,
) -> list[str]:
    enc = tok(prompt, return_tensors="pt", truncation=True,
              max_length=64).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        num_return_sequences=k,
        pad_token_id=tok.pad_token_id,
    )
    prompt_len = enc["input_ids"].shape[1]
    texts = []
    for seq in out:
        decoded = tok.decode(seq[prompt_len:], skip_special_tokens=True).strip()
        # Prepend the prompt so the output is a complete note
        texts.append(prompt + decoded)
    return texts


def load_completed(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    done = set()
    with open(output_path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["idx"])
            except Exception:
                pass
    log.info("resume: %d records already on disk", len(done))
    return done


@torch.no_grad()
def generate(
    model, tok, ppl_model, ppl_tok,
    n_samples: int, output_path: Path,
    dataset: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.9,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    k: int = 4,
    seed: int = 42,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_path)
    rng = random.Random(seed)

    with open(output_path, "a") as fout:
        pbar = tqdm(total=n_samples, initial=len(completed))
        for idx in range(n_samples):
            if idx in completed:
                # Advance RNG state to stay deterministic
                get_prompt(dataset, rng)
                continue

            prompt, prompt_type = get_prompt(dataset, rng)

            candidates = generate_candidates(
                model, tok, prompt, k=k,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            scored = sorted(
                [(c, compute_ppl(c, ppl_model, ppl_tok)) for c in candidates if c],
                key=lambda x: x[1],
            )

            best_text, best_ppl, passed = "", float("inf"), False
            for cand_text, cand_ppl in scored:
                if passes_sentence_filter(cand_text):
                    best_text, best_ppl, passed = cand_text, cand_ppl, True
                    break
            if not passed and scored:
                best_text, best_ppl = scored[0]

            fout.write(json.dumps({
                "idx":           idx,
                "prompt_type":   prompt_type,
                "prompt":        prompt,
                "text":          best_text,
                "perplexity":    round(best_ppl, 3),
                "passed_filter": passed,
                "k":             k,
            }) + "\n")
            fout.flush()
            pbar.update(1)

        pbar.close()

    total = sum(1 for _ in open(output_path))
    log.info("wrote %d total records to %s", total, output_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model",          required=True)
    ap.add_argument("--adapter_path",        default=None)
    ap.add_argument("--dataset",             required=True, choices=["mtsamples", "pmc"])
    ap.add_argument("--n_samples",           type=int,   default=200)
    ap.add_argument("--output_path",         required=True)
    ap.add_argument("--max_new_tokens",      type=int,   default=1024)
    ap.add_argument("--temperature",         type=float, default=0.9)
    ap.add_argument("--top_p",               type=float, default=0.95)
    ap.add_argument("--repetition_penalty",  type=float, default=1.2)
    ap.add_argument("--k",                   type=int,   default=4,
                    help="candidates per sample for PPL-based selection")
    ap.add_argument("--seed",                type=int,   default=42)
    args = ap.parse_args()

    model,     tok     = load_generator(args.base_model, args.adapter_path)
    ppl_model, ppl_tok = load_ppl_model(args.base_model)

    generate(
        model, tok, ppl_model, ppl_tok,
        n_samples=args.n_samples,
        output_path=Path(args.output_path),
        dataset=args.dataset,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        k=args.k,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

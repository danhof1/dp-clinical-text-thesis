"""
Generate synthetic clinical notes from a DP-LoRA fine-tuned model.

Usage:
    python -m generate.run \
      --base_model /path/to/Llama-3.1-8B-Instruct \
      --adapter_path /path/to/dp_lora_eps8_rmu/final \
      --n_samples 2000 \
      --output_path /path/to/outputs/generated/rmu_eps8.jsonl

Enhancements over original:
  - Resume logic: skips already-written records on restart (append mode)
  - k candidates + PPL-based selection (Term2Note DP Quality Maximiser)
  - Sentence length filter (Term2Note Appendix H, max 2181 chars)
  - padding_side=left fix for decoder-only generation
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("generate")

MAX_SENTENCE_CHARS = 2181

SPECIALTY_PROMPTS = [
    "Cardiovascular / Pulmonary",
    "Gastroenterology",
    "Orthopedic",
    "Neurology",
    "Urology",
    "Dermatology",
    "Endocrinology",
    "General Medicine",
    "Radiology",
    "Pediatrics - Neonatal",
    "Psychiatry / Psychology",
    "Obstetrics / Gynecology",
    "Ophthalmology",
    "Hematology - Oncology",
    "Nephrology",
]


def build_prompt(specialty: str) -> str:
    return f"Medical Specialty: {specialty}\nDescription: "


def passes_sentence_filter(text: str) -> bool:
    """Reject text with any sentence longer than 2181 chars (Term2Note Appendix H)."""
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
    """Load base model as PPL reference (no adapter — post-processing theorem safe)."""
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
    """Generate k independent candidates for one prompt."""
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
    return [tok.decode(seq[prompt_len:], skip_special_tokens=True).strip()
            for seq in out]


def load_completed(output_path: Path) -> set[int]:
    """Return set of already-written idx values for resume."""
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
    max_new_tokens: int = 1024,
    temperature: float = 0.9,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    k: int = 4,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_path)

    with open(output_path, "a") as fout:
        pbar = tqdm(total=n_samples, initial=len(completed))
        for idx in range(n_samples):
            if idx in completed:
                continue

            specialty = SPECIALTY_PROMPTS[idx % len(SPECIALTY_PROMPTS)]
            prompt    = build_prompt(specialty)

            candidates = generate_candidates(
                model, tok, prompt, k=k,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            # Score each candidate; pick lowest-PPL that passes sentence filter
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
                "specialty":     specialty,
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
    ap.add_argument("--n_samples",           type=int,   default=2000)
    ap.add_argument("--output_path",         required=True)
    ap.add_argument("--max_new_tokens",      type=int,   default=1024)
    ap.add_argument("--temperature",         type=float, default=0.9)
    ap.add_argument("--top_p",               type=float, default=0.95)
    ap.add_argument("--repetition_penalty",  type=float, default=1.2)
    ap.add_argument("--k",                   type=int,   default=4,
                    help="candidates per sample for PPL-based selection")
    args = ap.parse_args()

    model,     tok     = load_generator(args.base_model, args.adapter_path)
    ppl_model, ppl_tok = load_ppl_model(args.base_model)

    generate(
        model, tok, ppl_model, ppl_tok,
        n_samples=args.n_samples,
        output_path=Path(args.output_path),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        k=args.k,
    )


if __name__ == "__main__":
    main()

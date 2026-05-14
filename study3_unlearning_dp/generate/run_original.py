"""
Generate synthetic clinical notes from a DP-LoRA fine-tuned model.

Usage:
    python -m generate.run \\
      --base_model meta-llama/Meta-Llama-3-8B \\
      --adapter_path /path/to/dp_lora_eps8_base/final \\
      --n_samples 5000 \\
      --output_path /path/to/synth/eps8_base.jsonl

The generator takes short "specialty" prompts and produces full notes.
The prompts themselves are low-information so the output is determined
by the fine-tuned distribution, not the prompt content — this matches
the Yue et al. 2023 setup.

For the synthetic-data vs. training-data extraction analysis, the
n-gram attack (attacks/extraction.py) consumes the JSONL this writes.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("generate")


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
    # Flat, low-information prompt so the note content comes from the fine-tune dist.
    return (
        f"Medical Specialty: {specialty}\n"
        f"Description: "
    )


def load_generator(base_model: str, adapter_path: str | None, bf16: bool = True):
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, attn_implementation="sdpa",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        log.info("loaded adapter %s", adapter_path)
    model.eval().cuda()
    return model, tok


@torch.no_grad()
def generate(
    model, tokenizer, n_samples: int, output_path: Path,
    max_new_tokens: int = 1024,
    temperature: float = 0.9,
    top_p: float = 0.95,
    batch_size: int = 8,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    pbar = tqdm(total=n_samples)

    with open(output_path, "w") as fout:
        while written < n_samples:
            this_batch = min(batch_size, n_samples - written)
            specialties = [
                SPECIALTY_PROMPTS[(written + i) % len(SPECIALTY_PROMPTS)]
                for i in range(this_batch)
            ]
            prompts = [build_prompt(s) for s in specialties]
            enc = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=64,
            ).to(model.device)

            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
            # strip prompt tokens
            prompt_lens = enc["attention_mask"].sum(dim=1)
            for i, (seq, plen, spec) in enumerate(zip(out, prompt_lens, specialties)):
                gen = tokenizer.decode(seq[plen:], skip_special_tokens=True).strip()
                fout.write(json.dumps({
                    "idx": written + i, "specialty": spec, "text": gen,
                }) + "\n")
            written += this_batch
            pbar.update(this_batch)
    pbar.close()
    log.info("wrote %d samples to %s", written, output_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_path", default=None,
                    help="PEFT adapter dir; omit to use base model as-is")
    ap.add_argument("--n_samples", type=int, default=5000)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    model, tok = load_generator(args.base_model, args.adapter_path)
    generate(
        model, tok, args.n_samples, Path(args.output_path),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate synthetic drug reviews for PsyTAR SynBench replication.

Generates drug-review-style text using prompts matched to PsyTAR's domain
(SSRI/SNRI patient experience reports). Uses the same PPL-based quality
selection as the main generate/run.py.

Usage:
    python generate_psytar.py \
      --base_model /path/to/Llama-3.1-8B-Instruct \
      --adapter_path /path/to/dp_lora/final \
      --n_samples 500 \
      --output_path /path/to/outputs/generated/psytar_eps8.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("generate_psytar")

DRUG_PROMPTS = [
    "Write a patient review about their experience taking Lexapro for depression:\n",
    "Write a patient review about their experience taking Zoloft for anxiety:\n",
    "Write a patient review about their experience taking Cymbalta for pain:\n",
    "Write a patient review about their experience taking Effexor XR for depression:\n",
    "Write a patient review about their experience taking Lexapro for anxiety:\n",
    "Write a patient review about their experience taking Zoloft for depression:\n",
    "Write a patient review about their experience taking Cymbalta for depression:\n",
    "Write a patient review about their experience taking Effexor XR for anxiety:\n",
]


def load_model(base_model: str, adapter_path: str | None):
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        log.info("loaded adapter %s", adapter_path)
    model.eval().cuda()
    return model, tok


def compute_ppl(text: str, model, tok) -> float:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    return math.exp(min(out.loss.item(), 20.0))


def generate_candidates(model, tok, prompt, k, max_new_tokens, temperature, top_p):
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.2,
            do_sample=True,
            num_return_sequences=k,
            pad_token_id=tok.pad_token_id,
        )
    prompt_len = enc["input_ids"].shape[1]
    return [tok.decode(seq[prompt_len:], skip_special_tokens=True).strip() for seq in out]


def load_completed(path: Path) -> set[int]:
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["idx"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_path", default=None)
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--k", type=int, default=1, help="Candidates per prompt (1=fast)")
    args = ap.parse_args()

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, tok = load_model(args.base_model, args.adapter_path)
    ppl_model, ppl_tok = model, tok

    completed = load_completed(out_path)
    log.info("resuming from %d completed", len(completed))

    with open(out_path, "a") as fout:
        pbar = tqdm(total=args.n_samples, initial=len(completed))
        for idx in range(args.n_samples):
            if idx in completed:
                continue

            prompt = DRUG_PROMPTS[idx % len(DRUG_PROMPTS)]
            candidates = generate_candidates(
                model, tok, prompt, k=args.k,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            if args.k > 1:
                scored = sorted(
                    [(c, compute_ppl(c, ppl_model, ppl_tok)) for c in candidates if c],
                    key=lambda x: x[1],
                )
                best_text = scored[0][0] if scored else ""
                best_ppl = scored[0][1] if scored else float("inf")
            else:
                best_text = candidates[0] if candidates else ""
                best_ppl = compute_ppl(best_text, ppl_model, ppl_tok) if best_text else float("inf")

            fout.write(json.dumps({
                "idx": idx,
                "prompt": prompt.strip(),
                "text": best_text,
                "perplexity": round(best_ppl, 3),
                "k": args.k,
            }) + "\n")
            fout.flush()
            pbar.update(1)

        pbar.close()

    total = sum(1 for _ in open(out_path))
    log.info("wrote %d total records to %s", total, out_path)


if __name__ == "__main__":
    main()

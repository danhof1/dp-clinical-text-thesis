#!/usr/bin/env python3
"""
Section-wise generation for Term2Note-trained models.

Generates clinical notes section by section, conditioned on SNOMED terms
and all previously generated sections. Uses k=4 quality maximiser
(lowest PPL under reference model) per Term2Note protocol.

Pipeline:
  1. Load term frequency table from training data
  2. For each note to generate:
     a. Sample control codes (or use provided ones)
     b. For each of 6 section groups in order:
        - Sample terms from training distribution for that group
        - Build prompt: prior context + <|section|> + <|terms|> + <|content|>
        - Generate k candidates, select lowest PPL
     c. Assemble full note from selected sections
  3. Write JSONL output

Usage:
  python generate_term2note.py \
    --checkpoint /path/to/dp_lora_adapter \
    --base_model /fs1/shared/model/llm/Llama-3.2-1B-Instruct \
    --term2note_data /path/to/train_term2note.jsonl \
    --output /path/to/output_dir \
    --n_generate 2000
"""
import argparse
import json
import logging
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("generate_term2note")

CACHE_DIR = "/fs1/projects/unlearning_pretraining/Proj_code/.cache"

SECTION_GROUPS = [
    "Patient Information",
    "Clinical Course & History",
    "Examinations & Findings",
    "Laboratory & Imaging Results",
    "Hospital Stay & Treatment",
    "Medications & Discharge Plan",
]

MAX_SENTENCE_CHARS = 2181


# ─── Term distribution ──────────────────────────────────────────

def build_term_table(jsonl_path):
    """Build per-section-group term frequency tables from training data."""
    group_terms = defaultdict(Counter)
    group_term_counts = defaultdict(list)
    control_code_list = []

    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            control_code_list.append(rec.get("control_codes", {}))

            text = rec["text"]
            for section_block in text.split("<|section|>"):
                section_block = section_block.strip()
                if not section_block:
                    continue
                lines = section_block.split("\n")
                group_name = lines[0].strip()
                terms_line = next((l for l in lines if l.startswith("<|terms|>")), None)
                if terms_line and group_name in SECTION_GROUPS:
                    terms_str = terms_line.replace("<|terms|> ", "").strip()
                    if terms_str and terms_str != "none":
                        terms = [t.strip() for t in terms_str.split(",") if t.strip()]
                        for t in terms:
                            group_terms[group_name][t] += 1
                        group_term_counts[group_name].append(len(terms))

    log.info("Built term tables from %d notes", len(control_code_list))
    for g in SECTION_GROUPS:
        n_unique = len(group_terms[g])
        median_count = sorted(group_term_counts.get(g, [0]))[len(group_term_counts.get(g, [0])) // 2]
        log.info("  %-35s %5d unique terms, median %d per note", g, n_unique, median_count)

    return group_terms, group_term_counts, control_code_list


def sample_terms(group_terms, group_term_counts, group_name, rng):
    """Sample terms for a section group from the training distribution."""
    if not group_terms[group_name]:
        return "none"

    counts = group_term_counts.get(group_name, [5])
    n_terms = rng.choice(counts)
    n_terms = max(1, n_terms)

    terms = list(group_terms[group_name].keys())
    weights = list(group_terms[group_name].values())
    total = sum(weights)
    probs = [w / total for w in weights]

    n_terms = min(n_terms, len(terms))
    selected = rng.choices(terms, weights=probs, k=n_terms)
    selected = sorted(set(selected))

    return ", ".join(selected) if selected else "none"


def sample_control_codes(control_code_list, rng):
    """Sample control codes from the training distribution."""
    return rng.choice(control_code_list)


# ─── Generation helpers ─────────────────────────────────────────

def generate_candidates(prompt, model, tokenizer, device, k=4,
                        max_new_tokens=512, temperature=0.1,
                        top_p=1.0, repetition_penalty=1.2):
    input_ids = tokenizer.encode(prompt, return_tensors="pt",
                                 truncation=True, max_length=2048).to(device)
    candidates = []
    for _ in range(k):
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = output[0][input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        # Truncate at next section marker if present
        for marker in ["<|section|>", "<|terms|>"]:
            if marker in gen_text:
                gen_text = gen_text[:gen_text.index(marker)]
        candidates.append(gen_text.strip())
    return candidates


def compute_perplexity(text, model, tokenizer, device, max_length=2048):
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings["input_ids"].to(device)
    if input_ids.shape[1] <= 1:
        return float("inf")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        return math.exp(outputs.loss.item())


def passes_sentence_filter(text):
    import re
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        if len(sent) > MAX_SENTENCE_CHARS:
            return False
    return True


# ─── Main pipeline ──────────────────────────────────────────────

def generate_note(control_codes, group_terms, group_term_counts,
                  model, ppl_model, tokenizer, ppl_tokenizer, device,
                  rng, k=4):
    """Generate a full note section by section."""
    prefix = " | ".join(f"{k}: {v}" for k, v in control_codes.items())
    context = prefix + "\n\n"
    sections_generated = []

    for group_name in SECTION_GROUPS:
        terms_str = sample_terms(group_terms, group_term_counts, group_name, rng)

        prompt = (
            context
            + f"<|section|> {group_name}\n"
            + f"<|terms|> {terms_str}\n"
            + f"<|content|>\n"
        )

        max_tokens = 256 if group_name == "Patient Information" else 512
        candidates = generate_candidates(
            prompt, model, tokenizer, device, k=k, max_new_tokens=max_tokens,
        )

        scored = []
        for cand in candidates:
            if not cand.strip():
                continue
            if not passes_sentence_filter(cand):
                continue
            ppl = compute_perplexity(cand, ppl_model, ppl_tokenizer, device)
            scored.append((cand, ppl))

        if not scored:
            best_text = candidates[0] if candidates else ""
            best_ppl = float("inf")
        else:
            scored.sort(key=lambda x: x[1])
            best_text, best_ppl = scored[0]

        section_block = (
            f"<|section|> {group_name}\n"
            f"<|terms|> {terms_str}\n"
            f"<|content|>\n"
            f"{best_text}"
        )
        context += section_block + "\n\n"
        sections_generated.append({
            "group": group_name,
            "terms": terms_str,
            "text": best_text,
            "ppl": best_ppl,
            "n_candidates": len(candidates),
            "n_passed_filter": len(scored),
        })

    return context, sections_generated


def run(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_term2note.jsonl"
    stats_path = out_dir / "generation_stats.json"

    # Load term tables
    log.info("Building term tables from %s", args.term2note_data)
    group_terms, group_term_counts, control_code_list = build_term_table(args.term2note_data)

    # Load model
    log.info("Loading base model %s", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.checkpoint:
        log.info("Loading LoRA adapter from %s", args.checkpoint)
        model = PeftModel.from_pretrained(model, args.checkpoint)
        model = model.merge_and_unload()

    model.to(device).eval()

    # Load PPL reference model
    ppl_model_path = args.ppl_model or args.base_model
    if ppl_model_path == args.base_model:
        log.info("Using base model as PPL reference")
        ppl_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR,
        ).to(device).eval()
    else:
        log.info("Loading PPL reference model from %s", ppl_model_path)
        ppl_model = AutoModelForCausalLM.from_pretrained(
            ppl_model_path, torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR,
        ).to(device).eval()
    ppl_tokenizer = AutoTokenizer.from_pretrained(ppl_model_path, cache_dir=CACHE_DIR)
    if ppl_tokenizer.pad_token is None:
        ppl_tokenizer.pad_token = ppl_tokenizer.eos_token

    # Resume support
    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["gen_id"])
        log.info("Resuming: %d already done", len(done_ids))

    n_to_generate = args.n_generate - len(done_ids)
    log.info("Generating %d notes (k=%d)", n_to_generate, args.k)

    stats = {"total": 0, "section_ppls": defaultdict(list)}

    with open(out_path, "a") as f_out:
        for i in range(args.n_generate):
            gen_id = f"gen_{i:05d}"
            if gen_id in done_ids:
                continue

            cc = sample_control_codes(control_code_list, rng)
            full_text, sections = generate_note(
                cc, group_terms, group_term_counts,
                model, ppl_model, tokenizer, ppl_tokenizer, device,
                rng, k=args.k,
            )

            # Extract plain text (strip section markers for eval)
            plain_parts = []
            for s in sections:
                if s["text"].strip():
                    plain_parts.append(s["text"])
            plain_text = "\n\n".join(plain_parts)

            record = {
                "gen_id": gen_id,
                "control_codes": cc,
                "text": full_text,
                "plain_text": plain_text,
                "sections": sections,
            }
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()

            stats["total"] += 1
            for s in sections:
                stats["section_ppls"][s["group"]].append(s["ppl"])

            if stats["total"] % 50 == 0:
                avg_ppls = {g: sum(v) / len(v) for g, v in stats["section_ppls"].items() if v}
                log.info(
                    "Generated %d/%d notes. Avg PPL: %s",
                    stats["total"], n_to_generate,
                    ", ".join(f"{g[:15]}={p:.1f}" for g, p in avg_ppls.items()),
                )

    # Write stats
    summary = {
        "total_generated": stats["total"],
        "k": args.k,
        "checkpoint": args.checkpoint or "none",
        "base_model": args.base_model,
        "section_mean_ppl": {
            g: sum(v) / len(v) if v else 0
            for g, v in stats["section_ppls"].items()
        },
    }
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Done. %d notes written to %s", stats["total"], out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Section-wise generation for Term2Note-trained models",
    )
    parser.add_argument("--checkpoint", default=None,
                        help="Path to DP-LoRA adapter directory")
    parser.add_argument("--base_model", required=True,
                        help="Path to base model (e.g. Llama-3.2-1B-Instruct)")
    parser.add_argument("--ppl_model", default=None,
                        help="PPL reference model (default: base model)")
    parser.add_argument("--term2note_data", required=True,
                        help="Path to train_term2note.jsonl")
    parser.add_argument("--output", required=True,
                        help="Output directory")
    parser.add_argument("--n_generate", type=int, default=2000)
    parser.add_argument("--k", type=int, default=4,
                        help="Candidates per section for quality maximiser")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args)

#!/usr/bin/env python3
"""
Section-wise generation for Term2Note-trained models (v2).

Changes from v1:
  - Uses single special tokens instead of multi-token markers
  - ICD-stratified term sampling (real term sets from same-ICD training notes)
  - Generation prompt max_length increased to 4096
  - Post-processing for placeholder name collapse
"""
import argparse
import json
import logging
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList, set_seed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("generate_term2note")

CACHE_DIR = "/fs1/projects/unlearning_pretraining/Proj_code/.cache"

# Single-token markers (mapped to Llama reserved special tokens)
TOK_SECTION = "<|reserved_special_token_0|>"
TOK_TERMS = "<|reserved_special_token_1|>"
TOK_CONTENT = "<|reserved_special_token_2|>"

SECTION_GROUPS = [
    "Patient Information",
    "Clinical Course & History",
    "Examinations & Findings",
    "Laboratory & Imaging Results",
    "Hospital Stay & Treatment",
    "Medications & Discharge Plan",
]

MAX_SENTENCE_CHARS = 2181
MAX_PROMPT_TOKENS = 4096

# Llama-3 chat template token IDs to suppress during generation
CHAT_TEMPLATE_TOKEN_IDS = [128006, 128007]  # <|start_header_id|>, <|end_header_id|>


class FrequencyPenaltyProcessor(LogitsProcessor):
    """Penalise tokens proportionally to how often they appeared in generated text.

    Matches vLLM's frequency_penalty: subtract penalty * count(token) from logits.
    """

    def __init__(self, penalty: float, input_length: int):
        self.penalty = penalty
        self.input_length = input_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated = input_ids[0, self.input_length:]
        if generated.numel() == 0:
            return scores
        token_counts = torch.bincount(generated, minlength=scores.shape[-1])
        if token_counts.shape[0] < scores.shape[-1]:
            token_counts = torch.nn.functional.pad(
                token_counts, (0, scores.shape[-1] - token_counts.shape[0]),
            )
        scores -= self.penalty * token_counts.float().unsqueeze(0).to(scores.device)
        return scores


class EOSLogitBiasProcessor(LogitsProcessor):
    """Add a constant logit bias to the EOS token to encourage early stopping."""

    def __init__(self, eos_token_id: int, bias: float):
        self.eos_token_id = eos_token_id
        self.bias = bias

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.eos_token_id] += self.bias
        return scores

PLACEHOLDER_PATTERNS = [
    (re.compile(r"\bJohn Doe\b", re.IGNORECASE), "___"),
    (re.compile(r"\bJane Smith\b", re.IGNORECASE), "___"),
    (re.compile(r"\bJane Doe\b", re.IGNORECASE), "___"),
    (re.compile(r"\b555-\d{4}\b"), "___"),
    (re.compile(r"\bjohn\.doe@\S+", re.IGNORECASE), "___"),
    (re.compile(r"\bjane\.smith@\S+", re.IGNORECASE), "___"),
    (re.compile(r"\b123 Main St\.?\b", re.IGNORECASE), "___"),
]


def fix_placeholders(text):
    for pat, repl in PLACEHOLDER_PATTERNS:
        text = pat.sub(repl, text)
    return text


# ─── Term distribution (ICD-stratified) ─────────────────────────

def build_term_table(jsonl_path):
    """Build per-ICD, per-section term sets from training data.

    Returns:
        icd_term_sets: {icd_category: [{group: "terms, ..."}]} — one entry per note
        control_code_list: [dict] — all control codes for sampling
    """
    icd_term_sets = defaultdict(list)
    control_code_list = []

    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            cc = rec.get("control_codes", {})
            control_code_list.append(cc)
            icd = cc.get("icd_category", "Unknown")

            text = rec["text"]
            note_terms = {}
            for section_block in text.split(TOK_SECTION):
                section_block = section_block.strip()
                if not section_block:
                    continue
                lines = section_block.split("\n")
                group_name = lines[0].strip()
                terms_line = next((l for l in lines if l.startswith(TOK_TERMS)), None)
                if terms_line and group_name in SECTION_GROUPS:
                    terms_str = terms_line.replace(TOK_TERMS + " ", "").strip()
                    if terms_str and terms_str != "none":
                        note_terms[group_name] = terms_str

            if note_terms:
                icd_term_sets[icd].append(note_terms)

    log.info("Built ICD-stratified term tables from %d notes across %d ICD categories",
             len(control_code_list), len(icd_term_sets))
    for icd, notes in sorted(icd_term_sets.items()):
        log.info("  %-55s %5d notes", icd[:55], len(notes))

    return icd_term_sets, control_code_list


def sample_real_terms(icd_term_sets, icd_category, group_name, rng):
    """Sample terms by picking a real term set from a training note of the same ICD."""
    notes = icd_term_sets.get(icd_category, [])
    if not notes:
        notes_all = [n for ns in icd_term_sets.values() for n in ns]
        if not notes_all:
            return "none"
        note = rng.choice(notes_all)
    else:
        note = rng.choice(notes)

    return note.get(group_name, "none")


def sample_control_codes(control_code_list, rng):
    return rng.choice(control_code_list)


# ─── Generation helpers ─────────────────────────────────────────

def generate_candidates(prompt, model, tokenizer, device, k=4,
                        max_new_tokens=2048, temperature=0.1,
                        top_p=1.0, repetition_penalty=1.2,
                        frequency_penalty=0.4, eos_logit_bias=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors="pt",
                                 truncation=True, max_length=MAX_PROMPT_TOKENS).to(device)
    input_length = input_ids.shape[1]

    logits_processors = LogitsProcessorList([
        FrequencyPenaltyProcessor(frequency_penalty, input_length),
        EOSLogitBiasProcessor(tokenizer.eos_token_id, eos_logit_bias),
    ])

    bad_words_ids = [[tid] for tid in CHAT_TEMPLATE_TOKEN_IDS]

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
                logits_processor=logits_processors,
                bad_words_ids=bad_words_ids,
            )
        gen_ids = output[0][input_length:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        for marker in [TOK_SECTION, TOK_TERMS]:
            if marker in gen_text:
                gen_text = gen_text[:gen_text.index(marker)]
        for tok in [TOK_CONTENT, TOK_TERMS, TOK_SECTION,
                    "<|end_of_text|>", "<|eot_id|>"]:
            gen_text = gen_text.replace(tok, "")
        candidates.append(gen_text.strip())
    return candidates


def compute_perplexity(text, model, tokenizer, device, max_length=4096):
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings["input_ids"].to(device)
    if input_ids.shape[1] <= 1:
        return float("inf")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        return math.exp(outputs.loss.item())


def passes_sentence_filter(text):
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        if len(sent) > MAX_SENTENCE_CHARS:
            return False
    return True


def is_degenerate(text, threshold=0.4):
    """Reject text that is mostly repeated short fragments."""
    if len(text) < 10:
        return True
    words = text.split()
    if not words:
        return True
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < threshold:
        return True
    most_common = max(set(words), key=words.count)
    if words.count(most_common) / len(words) > 0.5:
        return True
    return False


# ─── Main pipeline ──────────────────────────────────────────────

def generate_note(control_codes, icd_term_sets,
                  model, ppl_model, tokenizer, ppl_tokenizer, device,
                  rng, k=4):
    """Generate a full note section by section."""
    prefix = " | ".join(f"{k}: {v}" for k, v in control_codes.items())
    context = prefix + "\n\n"
    sections_generated = []
    icd = control_codes.get("icd_category", "Unknown")

    # Pick one donor note for this entire generation (consistent term sets across sections)
    donor_notes = icd_term_sets.get(icd, [])
    if not donor_notes:
        donor_notes = [n for ns in icd_term_sets.values() for n in ns]
    donor = rng.choice(donor_notes) if donor_notes else {}

    for group_name in SECTION_GROUPS:
        terms_str = donor.get(group_name, "none")

        prompt = (
            context
            + f"{TOK_SECTION} {group_name}\n"
            + f"{TOK_TERMS} {terms_str}\n"
            + f"{TOK_CONTENT}\n"
        )

        candidates = generate_candidates(
            prompt, model, tokenizer, device, k=k,
        )

        scored = []
        for cand in candidates:
            if not cand.strip():
                continue
            if is_degenerate(cand):
                continue
            if not passes_sentence_filter(cand):
                continue
            ppl = compute_perplexity(cand, ppl_model, ppl_tokenizer, device)
            scored.append((cand, ppl))

        if not scored:
            best_text = ""
            best_ppl = float("inf")
        else:
            scored.sort(key=lambda x: x[1])
            best_text, best_ppl = scored[0]

        best_text = fix_placeholders(best_text)

        section_block = (
            f"{TOK_SECTION} {group_name}\n"
            f"{TOK_TERMS} {terms_str}\n"
            f"{TOK_CONTENT}\n"
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

    log.info("Building ICD-stratified term tables from %s", args.term2note_data)
    icd_term_sets, control_code_list = build_term_table(args.term2note_data)

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
                cc, icd_term_sets,
                model, ppl_model, tokenizer, ppl_tokenizer, device,
                rng, k=args.k,
            )

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
        description="Section-wise generation for Term2Note v2 models",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--ppl_model", default=None)
    parser.add_argument("--term2note_data", required=True,
                        help="Path to train_term2note_v2.jsonl (with special tokens)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_generate", type=int, default=2000)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args)

"""
Generation v3: model-aware prompting + quality filtering.

Fixes:
  - Instruction-tuned models (Llama-3.1-8B-Instruct): use chat template
  - Base models (BioMistral-7B): use raw clinical note openings
  - Post-generation quality filter: classify outputs as clinical notes vs junk

Usage:
    python -m generate.run_v3 \
      --base_model /path/to/model \
      --adapter_path /path/to/adapter \
      --dataset mtsamples \
      --model_type instruct \
      --n_samples 200 \
      --output_path outputs/generated/v3/tag.jsonl
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
log = logging.getLogger("generate_v3")

MAX_SENTENCE_CHARS = 2181

# ── Specialty pools ──
MTS_SPECIALTIES = [
    "Surgery", "Consult - History and Phy.", "Cardiovascular / Pulmonary",
    "Orthopedic", "Radiology", "Gastroenterology", "Neurology",
    "General Medicine", "Urology", "Obstetrics / Gynecology",
    "Discharge Summary", "ENT - Otolaryngology", "Hematology - Oncology",
    "Emergency Room Reports", "Nephrology", "Dermatology",
    "Psychiatry / Psychology", "Pediatrics - Neonatal",
]

PMC_SPECIALTIES = [
    "oncology", "cardiology", "neurology", "ent", "infectious disease",
    "gastroenterology", "pulmonary", "surgery", "pediatrics", "endocrine",
]

# ── Section header openings for base models (weighted by MTSamples frequency) ──
MTS_OPENINGS = [
    ("PREOPERATIVE DIAGNOSIS: ", 351),
    ("CHIEF COMPLAINT:,  ", 124),
    ("PREOPERATIVE DIAGNOSES: ", 120),
    ("HISTORY OF PRESENT ILLNESS:,  ", 69),
    ("SUBJECTIVE:,  ", 66),
    ("EXAM:,  ", 56),
    ("CC:,  ", 51),
    ("PROCEDURE: ", 49),
    ("REASON FOR CONSULTATION:,  ", 36),
    ("REASON FOR VISIT:,  ", 33),
    ("DIAGNOSIS: ", 14),
    ("ADMITTING DIAGNOSIS: ", 9),
]


def build_prompt_base(dataset: str, specialty: str, rng: random.Random) -> str:
    """Prompt for base (non-instruct) models: raw text completion."""
    if dataset == "mtsamples":
        openings, weights = zip(*MTS_OPENINGS)
        opening = rng.choices(openings, weights=weights, k=1)[0]
        return opening
    else:
        starters = [
            f"A {rng.randint(25,80)}-year-old {'male' if rng.random() > 0.5 else 'female'} patient ",
            "We report a case of ",
            "This study evaluates ",
            "Background: ",
            "A retrospective analysis of ",
        ]
        return rng.choice(starters)


def build_prompt_instruct(
    dataset: str, specialty: str, tokenizer, rng: random.Random
) -> str:
    """Prompt for instruction-tuned models: chat template."""
    if dataset == "mtsamples":
        openings, weights = zip(*MTS_OPENINGS)
        opening = rng.choices(openings, weights=weights, k=1)[0]
        system_msg = (
            "You are a medical transcriptionist. Write realistic clinical "
            "documentation exactly as it would appear in a patient's medical record. "
            "Do not explain, summarize, or add commentary. Write only the clinical note."
        )
        user_msg = (
            f"Write a clinical note for a {specialty} case. "
            f"Start the note with \"{opening.strip()}\" and continue with the full clinical documentation "
            f"including relevant sections (history, examination, findings, plan as appropriate)."
        )
    else:
        system_msg = (
            "You are a medical researcher writing a clinical case report for a peer-reviewed journal. "
            "Write in formal academic medical prose. Do not add commentary or explanations."
        )
        user_msg = (
            f"Write a clinical case report in the field of {specialty}. "
            f"Begin with the patient presentation and clinical findings."
        )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ── Quality filter ──
CLINICAL_PATTERNS = [
    r'\b\d{1,3}\s*[-]?\s*year\s*[-]?\s*old\b',
    r'\byear old\b',
    r'\byr old\b',
    r'\byo\s+(male|female|man|woman)\b',
    r'\bpatient\b.*\b(present|admit|complain|report|history)\b',
    r'\bchief complaint\b',
    r'\bhistory of present illness\b',
    r'\bphysical exam\b',
    r'\bassessment\b',
    r'\bdiagnosis\b.*\b(:\s*\w|\d)',
    r'\bblood pressure\b',
    r'\bmg/dl\b',
    r'\bmmhg\b',
    r'\badmitted (to|for|with)\b',
    r'\bdischarged?\s*(to|home|on)\b',
    r'\bthe patient (is|was|has|had|present|report)\b',
]

def classify_output(text: str) -> dict:
    """Classify whether generated text is a clinical note vs other content."""
    text_lower = text.lower()
    score = 0
    flags = []

    for pattern in CLINICAL_PATTERNS:
        if re.search(pattern, text_lower):
            score += 1
            flags.append(pattern[:30])

    has_age = bool(re.search(r'\b\d{1,3}\s*[-]?\s*year', text_lower))
    has_patient = 'patient' in text_lower
    has_medical_term = any(t in text_lower for t in [
        'diagnosis', 'surgery', 'examination', 'treatment',
        'symptoms', 'admitted', 'history', 'complaint',
        'procedure', 'anesthesia', 'postoperative',
    ])
    min_length = len(text) >= 100

    is_clinical = (score >= 3) or (has_age and has_patient and has_medical_term)

    # Detect junk patterns
    is_exam_question = any(p in text_lower for p in [
        'which of the following', 'the correct answer',
        'step 1:', '## step', 'multiple choice',
        'what is the best', 'the final answer',
    ])
    is_job_posting = any(p in text_lower for p in [
        'apply now', 'salary range', 'board certified',
        'fellowship position', 'compensation package',
        'requirements:', 'we are seeking',
    ])
    # Strip markdown formatting before checking alpha ratio
    stripped = re.sub(r'[*_#\-\n\r\t|]', '', text)
    alpha_ratio = sum(1 for c in stripped if c.isalpha()) / max(len(stripped), 1)
    has_invented_words = any(
        len(w) > 25 and not any(c in w for c in ':/.-@_')
        for w in text.split()
    )
    is_garbled = len(text) > 50 and (alpha_ratio < 0.4 or has_invented_words)

    label = "clinical_note"
    if is_exam_question:
        label = "exam_question"
    elif is_job_posting:
        label = "job_posting"
    elif is_garbled:
        label = "garbled"
    elif not is_clinical:
        label = "other_medical"

    return {
        "label": label,
        "clinical_score": score,
        "has_age": has_age,
        "has_patient": has_patient,
        "min_length": min_length,
    }


# ── Model loading ──
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
    log.info("loaded PPL reference model")
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
              max_length=512).to(model.device)
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


def passes_sentence_filter(text: str) -> bool:
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        if len(sent) > MAX_SENTENCE_CHARS:
            return False
    return True


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
    dataset: str, model_type: str,
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

    specialties = MTS_SPECIALTIES if dataset == "mtsamples" else PMC_SPECIALTIES

    with open(output_path, "a") as fout:
        pbar = tqdm(total=n_samples, initial=len(completed))
        for idx in range(n_samples):
            specialty = specialties[idx % len(specialties)]

            if idx in completed:
                # Advance RNG to stay deterministic
                if model_type == "instruct":
                    build_prompt_instruct(dataset, specialty, tok, rng)
                else:
                    build_prompt_base(dataset, specialty, rng)
                continue

            if model_type == "instruct":
                prompt = build_prompt_instruct(dataset, specialty, tok, rng)
            else:
                prompt = build_prompt_base(dataset, specialty, rng)

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

            quality = classify_output(best_text)

            fout.write(json.dumps({
                "idx":           idx,
                "specialty":     specialty,
                "prompt_type":   model_type,
                "text":          best_text,
                "perplexity":    round(best_ppl, 3),
                "passed_filter": passed,
                "quality":       quality,
                "k":             k,
            }) + "\n")
            fout.flush()
            pbar.update(1)

        pbar.close()

    # Summary stats
    labels = {}
    with open(output_path) as f:
        for line in f:
            r = json.loads(line)
            lab = r.get("quality", {}).get("label", "unknown")
            labels[lab] = labels.get(lab, 0) + 1
    total = sum(labels.values())
    log.info("Output summary (%d records):", total)
    for lab, count in sorted(labels.items(), key=lambda x: -x[1]):
        log.info("  %s: %d (%.1f%%)", lab, count, 100*count/total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model",          required=True)
    ap.add_argument("--adapter_path",        default=None)
    ap.add_argument("--dataset",             required=True, choices=["mtsamples", "pmc"])
    ap.add_argument("--model_type",          required=True, choices=["base", "instruct"])
    ap.add_argument("--n_samples",           type=int,   default=200)
    ap.add_argument("--output_path",         required=True)
    ap.add_argument("--max_new_tokens",      type=int,   default=1024)
    ap.add_argument("--temperature",         type=float, default=0.9)
    ap.add_argument("--top_p",               type=float, default=0.95)
    ap.add_argument("--repetition_penalty",  type=float, default=1.2)
    ap.add_argument("--k",                   type=int,   default=4)
    ap.add_argument("--seed",                type=int,   default=42)
    args = ap.parse_args()

    model,     tok     = load_generator(args.base_model, args.adapter_path)
    ppl_model, ppl_tok = load_ppl_model(args.base_model)

    generate(
        model, tok, ppl_model, ppl_tok,
        n_samples=args.n_samples,
        output_path=Path(args.output_path),
        dataset=args.dataset,
        model_type=args.model_type,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        k=args.k,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

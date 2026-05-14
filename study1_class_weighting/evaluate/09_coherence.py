"""
08_coherence.py
===============
Per-category conditioning fidelity analysis.

For each synthetic BHC, uses Llama-3.1-8B to extract the primary diagnosis
and maps it to one of the 20 ICD categories. The extracted category is
compared to the control code used to generate the note.

Conditioning fidelity = fraction of synthetic BHCs where the model's
extracted ICD category matches the control code used at generation time.

This is the primary novel contribution metric of the thesis:
    "First physician-validated coherence-vs-epsilon curve for DP clinical text"
    (automated layer — physician evaluation on April 25 provides the manual layer)

Two-stage extraction pipeline:
    Stage 1: Free-text diagnosis extraction
        Prompt Llama-3.1-8B to identify the PRIMARY diagnosis from the BHC text.
        Returns a natural language phrase (e.g., "acute decompensated heart failure").
        Intermediate output is saved for qualitative analysis and thesis examples.

    Stage 2: ICD category classification
        Prompt Llama-3.1-8B to map the extracted diagnosis to one of the 20
        ICD categories from the Term2Note / thesis taxonomy (Wu et al. 2025).
        Constrained to the exact 20 category names — no free generation.

Outputs:
    coherence/<run_dir>/coherence_results.jsonl   — per-record results
    coherence/<run_dir>/coherence_summary.json    — aggregate + per-category metrics

One invocation per synthetic file — run in parallel across GPUs for different
epsilon levels:
    nohup python 08_coherence.py --synthetic ./generated/data2_eps05_weighted/synthetic_bhc.jsonl --epsilon 0.5 --output ./coherence/eps05 > logs/coherence_eps05.log 2>&1 &
    nohup python 08_coherence.py --synthetic ./generated/data2_eps1_weighted/synthetic_bhc.jsonl  --epsilon 1   --output ./coherence/eps1  > logs/coherence_eps1.log  2>&1 &
    nohup python 08_coherence.py --synthetic ./generated/data2_eps4_weighted/synthetic_bhc.jsonl  --epsilon 4   --output ./coherence/eps4  > logs/coherence_eps4.log  2>&1 &
    nohup python 08_coherence.py --synthetic ./generated/data1_sgd/synthetic_bhc.jsonl            --epsilon 999 --output ./coherence/epsinf > logs/coherence_epsinf.log 2>&1 &

After all runs complete, view aggregate results with:
    python results.py --base_dir .
"""

import sys
import json
import re
import logging
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)


# =========================================================================
# Logging
# =========================================================================

class FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[FlushHandler(sys.stderr)]
)
log = logging.getLogger(__name__)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def gpu_mem():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        resrv = torch.cuda.memory_reserved() / 1e9
        return f"GPU: {alloc:.1f}GB / {resrv:.1f}GB"
    return "GPU: N/A"


# =========================================================================
# ICD 20-category taxonomy — Term2Note Appendix G / 01_build_dataset.py
# Must match exactly across all scripts.
# =========================================================================

ICD_CATEGORIES = [
    "Certain Infectious And Parasitic Diseases",
    "Neoplasms",
    "Endocrine, Nutritional And Metabolic Diseases",
    "Diseases Of The Blood And Blood-Forming Organs",
    "Mental And Behavioural Disorders",
    "Diseases Of The Nervous System And Sense Organs",
    "Diseases Of The Circulatory System",
    "Diseases Of The Respiratory System",
    "Diseases Of The Digestive System",
    "Diseases Of The Genitourinary System",
    "Complications Of Pregnancy, Childbirth, And The Puerperium",
    "Diseases Of The Skin And Subcutaneous Tissue",
    "Diseases Of The Musculoskeletal System And Connective Tissue",
    "Congenital Malformations, Deformations And Chromosomal Abnormalities",
    "Certain Conditions Originating In The Perinatal Period",
    "Symptoms, Signs And Abnormal Clinical And Laboratory Findings",
    "Injury, Poisoning And Certain Other Consequences Of External Causes",
    "External Causes Of Morbidity And Mortality",
    "Factors Influencing Health Status And Contact With Health Services",
    "Unknown",
]

# Numbered list string for injection into prompts
ICD_CATEGORY_LIST = "\n".join(
    f"{i+1}. {cat}" for i, cat in enumerate(ICD_CATEGORIES)
)

# Normalised lookup: lowercase stripped -> canonical name
# Used for fuzzy matching of model output
_CAT_LOWER = {cat.lower().strip(): cat for cat in ICD_CATEGORIES}


# =========================================================================
# Prompts
# =========================================================================

EXTRACTION_SYSTEM = (
    "You are a clinical NLP assistant. You extract the primary diagnosis "
    "from hospital discharge notes. Be concise and precise. "
    "Respond with only the primary diagnosis — no explanation, no punctuation."
)

EXTRACTION_PROMPT = (
    "Read the following Brief Hospital Course section from a hospital discharge note.\n\n"
    "Brief Hospital Course:\n{bhc_text}\n\n"
    "What is the PRIMARY diagnosis documented in this note? "
    "State only the primary diagnosis in a few words (e.g., 'acute decompensated heart failure', "
    "'sepsis due to urinary tract infection', 'hip fracture'). "
    "Do not include secondary diagnoses, procedures, or medications."
)

CLASSIFICATION_SYSTEM = (
    "You are a medical coding assistant. You classify diagnoses into ICD disease categories. "
    "You must respond with ONLY the exact category name from the provided list — nothing else."
)

CLASSIFICATION_PROMPT = (
    "Classify the following primary diagnosis into exactly one of the ICD disease categories listed below.\n\n"
    "Primary diagnosis: {diagnosis}\n\n"
    "ICD Disease Categories:\n{category_list}\n\n"
    "Respond with ONLY the exact category name from the list above. "
    "Do not add any explanation, punctuation, or extra words."
)


# =========================================================================
# Model inference
# =========================================================================

def build_chat_prompt(tokenizer, system_msg, user_msg):
    """
    Build a chat-formatted prompt for Llama-3.1-8B-Instruct.
    Falls back to a simple system+user concatenation if the tokenizer
    does not have a chat template.
    """
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for tokenizers without chat template
        prompt = f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"
    return prompt


def generate_response(
    prompt, model, tokenizer, device,
    max_new_tokens=64, temperature=0.1, top_p=1.0,
):
    """
    Generate a short response from the model.
    Low temperature (0.1) for deterministic extraction — same as generation scripts.
    max_new_tokens=64 is sufficient for a diagnosis phrase or category name.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Guard: truncate context if necessary (Llama-3.1-8B supports 128k but
    # we keep inputs reasonable — BHC sections are at most ~1024 tokens)
    if input_ids.shape[1] > args_global.max_length:
        input_ids = input_ids[:, -args_global.max_length:]

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (not the prompt)
    generated_ids = output[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


# =========================================================================
# Stage 1: Free-text diagnosis extraction
# =========================================================================

def extract_diagnosis(bhc_text, model, tokenizer, device):
    """
    Stage 1: Extract the primary diagnosis as free text from a BHC section.

    Returns the extracted diagnosis string (e.g. "acute kidney injury").
    Returns empty string on failure.
    """
    # Truncate very long BHC sections for the prompt — keep first 800 words
    words = bhc_text.split()
    if len(words) > 800:
        bhc_truncated = " ".join(words[:800]) + " [...]"
    else:
        bhc_truncated = bhc_text

    user_msg = EXTRACTION_PROMPT.format(bhc_text=bhc_truncated)
    prompt   = build_chat_prompt(tokenizer, EXTRACTION_SYSTEM, user_msg)

    response = generate_response(prompt, model, tokenizer, device, max_new_tokens=48)

    # Basic cleanup: strip leading "Diagnosis:", "Primary:", etc.
    cleaned = re.sub(r'^(primary\s+diagnosis\s*[:\-]?\s*|diagnosis\s*[:\-]?\s*)', '',
                     response, flags=re.IGNORECASE).strip()
    # Remove surrounding quotes if present
    cleaned = cleaned.strip('"\'')

    return cleaned


# =========================================================================
# Stage 2: ICD category classification
# =========================================================================

def classify_to_icd_category(diagnosis, model, tokenizer, device):
    """
    Stage 2: Map a free-text diagnosis to one of the 20 ICD categories.

    Returns (matched_category, raw_response, match_method) where:
        matched_category: one of ICD_CATEGORIES, or 'Unknown' on failure
        raw_response:     the model's raw text output (for debugging)
        match_method:     'exact' | 'fuzzy' | 'fallback_unknown'
    """
    if not diagnosis or not diagnosis.strip():
        return "Unknown", "", "fallback_unknown"

    user_msg = CLASSIFICATION_PROMPT.format(
        diagnosis=diagnosis,
        category_list=ICD_CATEGORY_LIST,
    )
    prompt = build_chat_prompt(tokenizer, CLASSIFICATION_SYSTEM, user_msg)

    response = generate_response(prompt, model, tokenizer, device, max_new_tokens=32)
    raw = response.strip()

    # Attempt 1: exact match (case-insensitive)
    raw_lower = raw.lower().strip()
    if raw_lower in _CAT_LOWER:
        return _CAT_LOWER[raw_lower], raw, "exact"

    # Attempt 2: exact match after stripping trailing punctuation / numbers
    cleaned = re.sub(r'^[\d]+[\.\)]\s*', '', raw_lower).strip().rstrip('.,:;')
    if cleaned in _CAT_LOWER:
        return _CAT_LOWER[cleaned], raw, "exact"

    # Attempt 3: substring match — model output contains the category name
    for cat_lower, cat_canonical in _CAT_LOWER.items():
        if cat_lower in raw_lower or raw_lower in cat_lower:
            return cat_canonical, raw, "fuzzy"

    # Attempt 4: partial word overlap — find the category with highest token overlap
    raw_tokens = set(re.sub(r'[^a-z\s]', '', raw_lower).split())
    best_cat, best_overlap = None, 0
    for cat_lower, cat_canonical in _CAT_LOWER.items():
        cat_tokens = set(re.sub(r'[^a-z\s]', '', cat_lower).split())
        # Ignore stop words
        stop = {'and', 'of', 'the', 'in', 'or', 'to', 'a', 'an', 'by', 'not',
                'with', 'for', 'on', 'at', 'from', 'as', 'certain', 'other'}
        sig_raw = raw_tokens - stop
        sig_cat = cat_tokens - stop
        if not sig_raw or not sig_cat:
            continue
        overlap = len(sig_raw & sig_cat) / len(sig_cat)  # recall against category
        if overlap > best_overlap:
            best_overlap = overlap
            best_cat = cat_canonical

    if best_cat and best_overlap >= 0.5:
        return best_cat, raw, "fuzzy"

    # Fallback
    return "Unknown", raw, "fallback_unknown"


# =========================================================================
# Resume logic
# =========================================================================

def load_completed_ids(results_path):
    """Load already-processed note_ids from a partial results file."""
    if not results_path.exists():
        return set()
    completed = set()
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                nid = rec.get('note_id', '')
                if nid:
                    completed.add(nid)
            except json.JSONDecodeError:
                continue
    log.info(f"Resume: {len(completed):,} already-processed records in {results_path}")
    return completed


# =========================================================================
# Data loading
# =========================================================================

def load_synthetic(path, sample=None):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if sample and sample < len(records):
        import random
        random.seed(42)
        records = random.sample(records, sample)
    log.info(f"Loaded {len(records):,} synthetic records from {path}")
    return records


# =========================================================================
# Metrics computation
# =========================================================================

def compute_metrics(results):
    """
    Compute aggregate and per-category conditioning fidelity metrics.

    Fidelity = fraction of records where extracted_icd == control_code_icd.
    Per-category breakdown shows which ICD categories are best/worst preserved.
    """
    total      = len(results)
    n_match    = sum(1 for r in results if r['match'])
    n_unknown  = sum(1 for r in results if r['extracted_icd'] == 'Unknown')

    # Per-category breakdown
    by_cat = defaultdict(lambda: {'n': 0, 'n_match': 0, 'n_unknown': 0})
    for r in results:
        cat = r['control_icd']
        by_cat[cat]['n']        += 1
        by_cat[cat]['n_match']  += int(r['match'])
        by_cat[cat]['n_unknown'] += int(r['extracted_icd'] == 'Unknown')

    per_category = {}
    for cat, counts in sorted(by_cat.items()):
        n = counts['n']
        per_category[cat] = {
            'n':            n,
            'n_match':      counts['n_match'],
            'n_unknown':    counts['n_unknown'],
            'fidelity':     round(counts['n_match'] / n, 4) if n > 0 else 0.0,
            'unknown_rate': round(counts['n_unknown'] / n, 4) if n > 0 else 0.0,
        }

    # Match method breakdown
    method_counts = Counter(r['match_method'] for r in results)

    aggregate = {
        'n_total':          total,
        'n_match':          n_match,
        'n_unknown':        n_unknown,
        'overall_fidelity': round(n_match / total, 4) if total > 0 else 0.0,
        'unknown_rate':     round(n_unknown / total, 4) if total > 0 else 0.0,
        'match_method_counts': dict(method_counts),
    }

    return aggregate, per_category


# =========================================================================
# Main
# =========================================================================

# Global args reference needed inside generate_response for max_length
args_global = None


def main(args):
    global args_global
    args_global = args

    set_seed(args.seed)

    output_dir   = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / 'coherence_results.jsonl'
    summary_path = output_dir / 'coherence_summary.json'

    log.info(f"Synthetic data: {args.synthetic}")
    log.info(f"Output:         {output_dir}")
    log.info(f"Epsilon:        {args.epsilon}")
    log.info(f"Model:          {args.model_path}")
    log.info(f"{gpu_mem()}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # -- Load model --
    log.info("Loading Llama-3.1-8B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info(f"Tokenizer loaded. Vocab size: {len(tokenizer):,}")

    log.info(f"Loading Llama-3.1-8B model... {gpu_mem()}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    model.eval()
    log.info(f"Model loaded. {gpu_mem()}")

    # -- Load synthetic records --
    records = load_synthetic(args.synthetic, sample=args.sample)

    # -- Resume --
    completed_ids = load_completed_ids(results_path)
    if completed_ids:
        records = [r for r in records if r.get('note_id', '') not in completed_ids]
        log.info(f"Resuming: {len(completed_ids):,} done, {len(records):,} remaining")
    else:
        log.info(f"Fresh run: {len(records):,} records to process")

    if not records:
        log.info("All records already processed.")
    else:
        # ── Main processing loop ───────────────────────────────────────────
        log.info("=" * 60)
        log.info(f"COHERENCE EXTRACTION: {len(records):,} records  (ε={args.epsilon})")
        log.info("Stage 1: free-text diagnosis extraction")
        log.info("Stage 2: ICD category classification")
        log.info("=" * 60)

        n_match_running = 0
        n_processed     = 0

        with open(results_path, 'a') as out_f:
            for i, rec in enumerate(tqdm(records, desc="Coherence")):
                note_id     = rec.get('note_id', '')
                syn_bhc     = rec.get('synthetic_bhc', '')
                control_icd = rec.get('control_codes', {}).get('icd_category', 'Unknown')

                if not syn_bhc.strip():
                    result = {
                        'note_id':        note_id,
                        'control_icd':    control_icd,
                        'extracted_diag': '',
                        'extracted_icd':  'Unknown',
                        'raw_response':   '',
                        'match_method':   'fallback_unknown',
                        'match':          False,
                        'epsilon':        args.epsilon,
                    }
                    out_f.write(json.dumps(result) + '\n')
                    out_f.flush()
                    n_processed += 1
                    continue

                # Stage 1: extract diagnosis
                extracted_diag = extract_diagnosis(syn_bhc, model, tokenizer, device)

                # Stage 2: classify to ICD category
                extracted_icd, raw_response, match_method = classify_to_icd_category(
                    extracted_diag, model, tokenizer, device
                )

                match = (extracted_icd == control_icd)
                if match:
                    n_match_running += 1
                n_processed += 1

                result = {
                    'note_id':        note_id,
                    'control_icd':    control_icd,
                    'extracted_diag': extracted_diag,
                    'extracted_icd':  extracted_icd,
                    'raw_response':   raw_response,
                    'match_method':   match_method,
                    'match':          match,
                    'epsilon':        args.epsilon,
                }

                # Incremental write — no data loss on job kill
                out_f.write(json.dumps(result) + '\n')
                out_f.flush()

                # Progress logging
                should_log = (
                    (i + 1) <= 5 or
                    (i + 1) == 20 or
                    (i + 1) % 100 == 0
                )
                if should_log:
                    running_fidelity = n_match_running / n_processed
                    log.info(
                        f"  [{i+1}/{len(records)}]  "
                        f"match={'✓' if match else '✗'}  "
                        f"method={match_method}  "
                        f"running_fidelity={running_fidelity:.3f}  "
                        f"control='{control_icd[:30]}'  "
                        f"extracted='{extracted_icd[:30]}'  "
                        f"{gpu_mem()}"
                    )
                    if args.verbose and i < 5:
                        log.info(f"    diag:     '{extracted_diag[:80]}'")
                        log.info(f"    raw_resp: '{raw_response[:60]}'")

    # ── Compute metrics over ALL records (including previously completed) ──
    log.info("Computing final metrics over all records...")
    all_results = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                all_results.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    aggregate, per_category = compute_metrics(all_results)

    summary = {
        'epsilon':      args.epsilon,
        'synthetic':    str(args.synthetic),
        'model':        args.model_path,
        'aggregate':    aggregate,
        'per_category': per_category,
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary saved to {summary_path}")

    # ── Print results table ────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"COHERENCE RESULTS  (ε={args.epsilon})")
    log.info("=" * 60)
    log.info(f"  Total records:      {aggregate['n_total']:,}")
    log.info(f"  Matches:            {aggregate['n_match']:,}")
    log.info(f"  Overall fidelity:   {aggregate['overall_fidelity']:.4f}  "
             f"({aggregate['overall_fidelity']*100:.1f}%)")
    log.info(f"  Unknown rate:       {aggregate['unknown_rate']:.4f}")
    log.info(f"  Match methods:      {aggregate['match_method_counts']}")

    # Per-category sorted by fidelity
    cats_sorted = sorted(per_category.items(), key=lambda x: x[1]['fidelity'])
    log.info("")
    log.info("  Per-category fidelity (lowest → highest):")
    for cat, m in cats_sorted:
        bar = "█" * int(m['fidelity'] * 20)
        log.info(
            f"  {cat[:50]:50s}  "
            f"n={m['n']:>5,}  "
            f"fidelity={m['fidelity']:.3f}  "
            f"|{bar:<20}|"
        )
    log.info("=" * 60)
    log.info("DONE")


# =========================================================================
# CLI
# =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Per-category conditioning fidelity analysis for DP synthetic BHC',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument('--synthetic', required=True,
                        help='Path to synthetic_bhc.jsonl from 03_generate.py')

    # Model
    parser.add_argument('--model_path',
                        default='/fs1/shared/model/llm/Llama-3.1-8B',
                        help='Path to Llama-3.1-8B-Instruct on the cluster')

    # Output
    parser.add_argument('--output', default='./coherence/run',
                        help='Output directory for this epsilon run')
    parser.add_argument('--epsilon', default=None,
                        help='Epsilon label for this run (e.g. 0.5, 1, 4, 999 for inf)')

    # Optional limits
    parser.add_argument('--sample', type=int, default=None,
                        help='Process only N records (for quick testing)')
    parser.add_argument('--max_length', type=int, default=2048,
                        help='Max input token length (prompt truncation threshold)')

    # Debug
    parser.add_argument('--verbose', action='store_true',
                        help='Print extracted diagnosis and raw model response for '
                             'first 5 records (useful for prompt debugging)')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)
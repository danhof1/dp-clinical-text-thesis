"""
Analyze generation quality across all v2/v3 runs.
Applies post-processing: truncates at exam-question transitions.
Classifies and reports viability stats.
"""
import json
import re
import sys
from pathlib import Path
from collections import Counter

# Run this on the cluster
GEN_DIR = Path("/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/"
               "New Experiment/Unlearning/mtsamples_unlearn_dp/"
               "mtsamples_unlearn_dp/outputs/generated")

# Exam/Q&A transition markers — truncate text before these
QA_MARKERS = [
    r'## Step \d',
    r'Based on this information',
    r'What is your (differential )?diagnosis',
    r'Which (of the following|surgical|medication|one)',
    r'Please (choose|provide|select)',
    r'The (correct|final) answer',
    r'\b[A-D]\)\s',  # MCQ answer choices
    r'Question:',
    r'\bWhat do you think\b',
]
QA_PATTERN = re.compile('|'.join(QA_MARKERS), re.IGNORECASE)

CLINICAL_PATTERNS = [
    r'\b\d{1,3}\s*[-]?\s*year\s*[-]?\s*old\b',
    r'\byo\s+(male|female|man|woman)\b',
    r'\bpatient\b.*\b(present|admit|complain|report)\b',
    r'\bchief complaint\b',
    r'\bhistory of present illness\b',
    r'\bphysical exam\b',
    r'\bblood pressure\b',
    r'\bmg/dl\b',
    r'\bmmhg\b',
    r'\badmitted (to|for|with)\b',
    r'\bthe patient (is|was|has|had)\b',
    r'\bvital signs\b',
    r'\bdiagnos(is|es)\b.*:',
    r'\bprocedure\b.*:',
    r'\bmedication\b',
]


def truncate_at_qa(text: str) -> tuple[str, bool]:
    """Truncate text at first exam-question transition. Returns (text, was_truncated)."""
    match = QA_PATTERN.search(text)
    if match:
        truncated = text[:match.start()].rstrip()
        if len(truncated) > 50:
            return truncated, True
    return text, False


def classify(text: str) -> str:
    """Classify text quality after truncation."""
    text_lower = text.lower()

    # Check for garbled text
    stripped = re.sub(r'[*_#\-\n\r\t|]', '', text)
    alpha_ratio = sum(1 for c in stripped if c.isalpha()) / max(len(stripped), 1)
    has_invented = any(
        len(w) > 25 and not any(c in w for c in ':/.-@_')
        for w in text.split()
    )
    if len(text) > 50 and (alpha_ratio < 0.35 or has_invented):
        return "garbled"

    # Clinical note detection
    clinical_score = sum(1 for p in CLINICAL_PATTERNS if re.search(p, text_lower))
    has_age = bool(re.search(r'\b\d{1,3}\s*[-]?\s*year', text_lower))
    has_patient = 'patient' in text_lower
    has_structure = any(h in text_lower for h in [
        'history:', 'exam:', 'findings:', 'plan:', 'assessment:',
        'diagnosis:', 'procedure:', 'medication', 'vital sign',
    ])

    if clinical_score >= 3 or (has_age and has_patient and has_structure):
        return "clinical_note"
    elif clinical_score >= 1:
        return "medical_fragment"
    else:
        return "non_clinical"


def analyze_file(path: Path) -> dict:
    if not path.exists():
        return None

    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        return None

    results = {
        "file": path.name,
        "total": len(records),
        "labels": Counter(),
        "truncated": 0,
        "mean_len": 0,
        "mean_ppl": 0,
        "examples": {},
    }

    total_len = 0
    ppl_sum = 0
    ppl_count = 0

    for r in records:
        text = r["text"]
        text_clean, was_truncated = truncate_at_qa(text)
        if was_truncated:
            results["truncated"] += 1

        label = classify(text_clean)
        results["labels"][label] += 1
        total_len += len(text_clean)

        ppl = r.get("perplexity", float("inf"))
        if ppl < 1e10:
            ppl_sum += ppl
            ppl_count += 1

        if label not in results["examples"]:
            preview = text_clean[:200].replace('\n', ' ')
            results["examples"][label] = preview

    results["mean_len"] = total_len / len(records)
    results["mean_ppl"] = ppl_sum / ppl_count if ppl_count > 0 else float("inf")

    return results


def main():
    print("=" * 80)
    print("GENERATION QUALITY ANALYSIS")
    print("=" * 80)

    # Find all generation files
    for version in ["v2", "v3"]:
        version_dir = GEN_DIR / version
        if not version_dir.exists():
            continue

        print(f"\n{'─' * 40}")
        print(f"  {version.upper()} GENERATION RUNS")
        print(f"{'─' * 40}")

        for jsonl_path in sorted(version_dir.glob("*.jsonl")):
            result = analyze_file(jsonl_path)
            if result is None:
                continue

            print(f"\n  {result['file']} ({result['total']} samples)")
            print(f"    Mean length: {result['mean_len']:.0f} chars | Mean PPL: {result['mean_ppl']:.1f}")
            if result['truncated'] > 0:
                print(f"    Truncated at Q&A: {result['truncated']}/{result['total']}")

            for label, count in result['labels'].most_common():
                pct = 100 * count / result['total']
                print(f"    {label:20s}: {count:3d} ({pct:5.1f}%)")

            for label, example in result['examples'].items():
                if label in ('clinical_note', 'garbled'):
                    print(f"    Example [{label}]: {example}...")

    # Overall summary
    print(f"\n{'=' * 80}")
    print("SUMMARY: Clinical Note Viability by Configuration")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

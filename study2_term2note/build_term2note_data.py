#!/usr/bin/env python3
"""
Build Term2Note training data from raw MIMIC-IV discharge notes.

Pipeline:
  1. Split each discharge note into section groups (Term2Note Table 2)
  2. Extract SNOMED CT terms per section via QuickUMLS
  3. Format as section-wise training examples for DP-SGD fine-tuning

Output: JSONL where each record is one full note, formatted as a sequence of
section-wise generation tasks:

  <|section|> {group_name}
  <|terms|> term1, term2, term3, ...
  <|content|>
  {section text}

The model learns to generate each section conditioned on the group name,
extracted terms, and all preceding sections. At inference, we generate
section by section.

Requires: env_3 Python (QuickUMLS + leveldb + spaCy)

Usage:
  python build_term2note_data.py
  python build_term2note_data.py --max_notes 1000 --workers 4
"""
import argparse
import csv
import json
import logging
import multiprocessing as mp
import os
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("build_term2note_data")

# ─── Paths ───────────────────────────────────────────────────────
DISCHARGE_CSV = "/fs1/projects/unlearning_pretraining/note/discharge.csv"
QUICKUMLS_PATH = "/fs1/projects/unlearning_pretraining/quickumls_data"
TRAIN_JSONL = "/fs1/projects/unlearning_pretraining/Proj_code/data/train.jsonl"
OUT_DIR = "/fs1/projects/unlearning_pretraining/Proj_code/data/term2note"

# ─── Section grouping (Term2Note Table 2) ────────────────────────
SECTION_GROUPS = OrderedDict([
    ("Patient Information", [
        "Name", "Unit No", "Admission Date", "Discharge Date",
        "Date of Birth", "Sex", "Service", "Allergies", "Attending",
    ]),
    ("Clinical Course & History", [
        "Chief Complaint",
        "Major Surgical or Invasive Procedure",
        "History of Present Illness",
        "Review of Systems",
        "Past Medical History",
        "Social History",
        "Family History",
    ]),
    ("Examinations & Findings", [
        "Physical Exam",
        "ADMISSION PHYSICAL EXAM", "ADMISSION EXAM",
        "DISCHARGE PHYSICAL EXAM", "DISCHARGE EXAM",
    ]),
    ("Laboratory & Imaging Results", [
        "Pertinent Results",
        "ADMISSION LABS", "DISCHARGE LABS",
        "IMAGING", "IMPRESSION", "FINDINGS",
    ]),
    ("Hospital Stay & Treatment", [
        "Brief Hospital Course",
        "ACTIVE ISSUES", "CHRONIC ISSUES",
        "TRANSITIONAL ISSUES",
    ]),
    ("Medications & Discharge Plan", [
        "Medications on Admission",
        "Discharge Medications",
        "Discharge Disposition",
        "Discharge Diagnosis",
        "Discharge Condition",
        "Discharge Instructions",
        "Followup Instructions",
        "Facility",
    ]),
])

SECTION_TO_GROUP = {}
for group, sections in SECTION_GROUPS.items():
    for sec in sections:
        SECTION_TO_GROUP[sec.lower()] = group


# ─── Section splitter ────────────────────────────────────────────

# All known section headers (case-insensitive matching)
ALL_HEADERS = set()
for sections in SECTION_GROUPS.values():
    ALL_HEADERS.update(s.lower() for s in sections)

# Regex for section header lines
HEADER_PATTERN = re.compile(
    r"^("
    + "|".join(re.escape(h) for h in sorted(ALL_HEADERS, key=len, reverse=True))
    + r")\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def split_sections(text):
    """Split a discharge note into sections, returning {group_name: text}."""
    sections = []
    last_end = 0
    last_header = None

    for match in HEADER_PATTERN.finditer(text):
        header = match.group(1).strip().lower()
        start = match.start()

        if last_header is not None:
            content = text[last_end:start].strip()
            sections.append((last_header, content))

        last_header = header
        last_end = match.end()

    if last_header is not None:
        content = text[last_end:].strip()
        sections.append((last_header, content))

    grouped = OrderedDict()
    for group in SECTION_GROUPS:
        grouped[group] = ""

    for header, content in sections:
        group = SECTION_TO_GROUP.get(header)
        if group and content:
            if grouped[group]:
                grouped[group] += "\n\n"
            grouped[group] += content

    # Remove empty groups
    return OrderedDict((g, t) for g, t in grouped.items() if t.strip())


# ─── SNOMED term extraction ─────────────────────────────────────

_matcher = None


def get_matcher():
    global _matcher
    if _matcher is None:
        from quickumls import QuickUMLS
        _matcher = QuickUMLS(
            QUICKUMLS_PATH,
            overlapping_criteria="length",
            threshold=0.7,
            window=5,
        )
        log.info("QuickUMLS matcher loaded")
    return _matcher


# SNOMED CT semantic type groups (TUI prefixes for clinical concepts)
SNOMED_SEMTYPES = {
    "T047",  # Disease or Syndrome
    "T048",  # Mental or Behavioral Dysfunction
    "T184",  # Sign or Symptom
    "T121",  # Pharmacologic Substance
    "T200",  # Clinical Drug
    "T061",  # Therapeutic or Preventive Procedure
    "T060",  # Diagnostic Procedure
    "T059",  # Laboratory Procedure
    "T023",  # Body Part, Organ, or Organ Component
    "T033",  # Finding
    "T034",  # Laboratory or Test Result
    "T037",  # Injury or Poisoning
    "T046",  # Pathologic Function
    "T191",  # Neoplastic Process
    "T019",  # Congenital Abnormality
    "T190",  # Anatomical Abnormality
    "T020",  # Acquired Abnormality
    "T049",  # Cell or Molecular Dysfunction
    "T039",  # Physiologic Function
    "T040",  # Organism Function
    "T041",  # Mental Process
    "T201",  # Clinical Attribute
    "T029",  # Body Location or Region
    "T030",  # Body Space or Junction
    "T031",  # Body Substance
    "T074",  # Medical Device
    "T075",  # Research Device
    "T058",  # Health Care Activity
    "T062",  # Research Activity
}


STOPWORDS = {
    "patient", "patients", "present", "presented", "presentation",
    "admission", "admissions", "discharge", "discharged", "discharges",
    "diet", "check up", "exam", "examination", "care", "complete",
    "aware", "art", "contacts", "normal", "negative", "positive",
    "history", "review", "status", "follow up", "follow-up", "treatment",
    "procedure", "test", "result", "results", "finding", "findings",
    "general", "intact", "able", "unable", "likely", "unlikely",
    "stable", "signs", "best", "high", "low", "always", "never",
    "activity", "bid", "tid", "qid", "prn", "daily", "weekly",
    "3 times", "times", "recall", "neck", "lungs", "liver", "heent",
    "nad", "abd", "lad", "breast", "breasts", "skin", "head",
    "diagnostic", "para", "d/c", "brushing", "coherent",
    "m/r", "dullness", "brief periods",
}

CLINICAL_SEMTYPES = {
    "T047",  # Disease or Syndrome
    "T048",  # Mental or Behavioral Dysfunction
    "T184",  # Sign or Symptom
    "T121",  # Pharmacologic Substance
    "T200",  # Clinical Drug
    "T061",  # Therapeutic or Preventive Procedure
    "T060",  # Diagnostic Procedure
    "T059",  # Laboratory Procedure
    "T033",  # Finding
    "T034",  # Laboratory or Test Result
    "T037",  # Injury or Poisoning
    "T046",  # Pathologic Function
    "T191",  # Neoplastic Process
    "T019",  # Congenital Abnormality
    "T190",  # Anatomical Abnormality
    "T020",  # Acquired Abnormality
    "T049",  # Cell or Molecular Dysfunction
}


def extract_snomed_terms(text):
    """Extract SNOMED CT terms from text using QuickUMLS.

    Takes only the best match per phrase span (highest similarity),
    filters to clinical semantic types, and removes generic stopwords.
    """
    if not text or len(text.strip()) < 10:
        return []

    matcher = get_matcher()
    matches = matcher.match(text)

    terms = set()
    for phrase_matches in matches:
        best = max(phrase_matches, key=lambda m: m["similarity"])
        semtypes = best.get("semtypes", set())
        if not (semtypes & CLINICAL_SEMTYPES):
            continue
        if best["similarity"] < 0.85:
            continue
        term = best["term"].lower().strip()
        if len(term) <= 3 or term in STOPWORDS:
            continue
        terms.add(term)

    return sorted(terms)


# ─── Data formatting ─────────────────────────────────────────────

def format_note(sections_with_terms, control_codes=None):
    """
    Format a note as a sequence of section-wise generation tasks.

    Each section block:
      <|section|> Group Name
      <|terms|> term1, term2, ...
      <|content|>
      {section text}
    """
    parts = []

    if control_codes:
        prefix = " | ".join(f"{k}: {v}" for k, v in control_codes.items())
        parts.append(prefix)

    for group_name, content, terms in sections_with_terms:
        term_str = ", ".join(terms) if terms else "none"
        block = (
            f"<|section|> {group_name}\n"
            f"<|terms|> {term_str}\n"
            f"<|content|>\n"
            f"{content}"
        )
        parts.append(block)

    return "\n\n".join(parts)


# ─── Main pipeline ───────────────────────────────────────────────

def load_train_note_ids():
    """Load note IDs from the training set to filter matching notes."""
    note_ids = set()
    if not os.path.exists(TRAIN_JSONL):
        return None
    with open(TRAIN_JSONL) as f:
        for line in f:
            rec = json.loads(line)
            note_ids.add(rec["note_id"])
    log.info("Loaded %d note IDs from training set", len(note_ids))
    return note_ids


def process_note(text, note_id, control_codes=None):
    """Process a single discharge note: split sections, extract terms, format."""
    sections = split_sections(text)
    if not sections:
        return None

    sections_with_terms = []
    for group_name, content in sections.items():
        terms = extract_snomed_terms(content)
        sections_with_terms.append((group_name, content, terms))

    formatted = format_note(sections_with_terms, control_codes)

    return {
        "note_id": note_id,
        "text": formatted,
        "n_sections": len(sections_with_terms),
        "n_terms_total": sum(len(t) for _, _, t in sections_with_terms),
        "section_groups": [g for g, _, _ in sections_with_terms],
        "control_codes": control_codes or {},
    }


def _worker_init():
    """Initialize QuickUMLS matcher in each worker process."""
    get_matcher()


def _worker_fn(args):
    """Process a single note in a worker process."""
    text, note_id, control_codes = args
    return process_note(text, note_id, control_codes)


def run(max_notes, use_train_filter, workers=1):
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = {}
    if use_train_filter and os.path.exists(TRAIN_JSONL):
        log.info("Loading control codes from %s", TRAIN_JSONL)
        with open(TRAIN_JSONL) as f:
            for line in f:
                rec = json.loads(line)
                train_data[rec["note_id"]] = rec.get("control_codes", {})
        log.info("Loaded %d records with control codes", len(train_data))

    valid_note_ids = set(train_data.keys()) if use_train_filter else None

    out_path = out_dir / "train_term2note.jsonl"
    stats_path = out_dir / "build_stats.json"

    log.info("Reading discharge notes from %s", DISCHARGE_CSV)
    csv.field_size_limit(sys.maxsize)

    # Collect work items
    work_items = []
    with open(DISCHARGE_CSV) as f_in:
        reader = csv.DictReader(f_in)
        for i, row in enumerate(reader):
            if max_notes and i >= max_notes:
                break
            note_id = row["note_id"]
            if valid_note_ids and note_id not in valid_note_ids:
                continue
            control_codes = train_data.get(note_id, {})
            work_items.append((row["text"], note_id, control_codes))

    log.info("Collected %d notes to process with %d workers", len(work_items), workers)

    stats = {
        "total_processed": 0,
        "total_written": 0,
        "skipped_no_sections": 0,
        "section_counts": Counter(),
        "terms_per_note": [],
    }

    if workers > 1:
        pool = mp.Pool(workers, initializer=_worker_init)
        results_iter = pool.imap(_worker_fn, work_items, chunksize=50)
    else:
        get_matcher()
        results_iter = map(_worker_fn, work_items)

    with open(out_path, "w") as f_out:
        for result in results_iter:
            stats["total_processed"] += 1

            if result is None:
                stats["skipped_no_sections"] += 1
                continue

            f_out.write(json.dumps(result) + "\n")
            f_out.flush()
            stats["total_written"] += 1
            stats["terms_per_note"].append(result["n_terms_total"])

            for g in result["section_groups"]:
                stats["section_counts"][g] += 1

            if stats["total_written"] % 500 == 0:
                log.info(
                    "Processed %d notes, written %d (%.1f terms/note avg)",
                    stats["total_processed"],
                    stats["total_written"],
                    sum(stats["terms_per_note"]) / len(stats["terms_per_note"]),
                )

    if workers > 1:
        pool.close()
        pool.join()

    # Write stats
    summary = {
        "total_processed": stats["total_processed"],
        "total_written": stats["total_written"],
        "skipped_no_sections": stats["skipped_no_sections"],
        "section_group_counts": dict(stats["section_counts"]),
        "mean_terms_per_note": (
            sum(stats["terms_per_note"]) / len(stats["terms_per_note"])
            if stats["terms_per_note"] else 0
        ),
        "output_path": str(out_path),
    }
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Done. Written %d notes to %s", stats["total_written"], out_path)
    log.info("Stats: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Term2Note formatted training data from MIMIC-IV discharge notes",
    )
    parser.add_argument(
        "--max_notes", type=int, default=0,
        help="Max notes to process from discharge.csv (0 = all)",
    )
    parser.add_argument(
        "--no_train_filter", action="store_true",
        help="Don't filter to training set (process all discharge notes)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (each loads its own QuickUMLS matcher)",
    )
    args = parser.parse_args()
    run(args.max_notes, use_train_filter=not args.no_train_filter, workers=args.workers)

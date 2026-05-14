#!/usr/bin/env python3
"""
Convert Term2Note JSONL to a HuggingFace DatasetDict for dp_lora.py training.

Reads train_term2note.jsonl (output of build_term2note_data.py) and creates
a DatasetDict with a 'finetune' split containing a 'text' column — the format
expected by finetune/dp_lora.py.

Usage:
  python convert_term2note_to_hf.py
  python convert_term2note_to_hf.py --input /path/to/train_term2note.jsonl --output /path/to/splits_term2note
"""
import argparse
import json
import logging
from pathlib import Path

from datasets import Dataset, DatasetDict

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("convert_term2note")

INPUT_DEFAULT = "/fs1/projects/unlearning_pretraining/Proj_code/data/term2note/train_term2note.jsonl"
OUTPUT_DEFAULT = "/fs1/projects/unlearning_pretraining/Proj_code/data/term2note/splits_term2note"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--max_text_len", type=int, default=8000,
                        help="Skip notes with text longer than this (chars)")
    parser.add_argument("--min_sections", type=int, default=3,
                        help="Skip notes with fewer than this many sections")
    args = parser.parse_args()

    records = []
    skipped_long = 0
    skipped_sections = 0

    with open(args.input) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if len(rec["text"]) > args.max_text_len:
                skipped_long += 1
                continue
            if rec["n_sections"] < args.min_sections:
                skipped_sections += 1
                continue
            records.append({"text": rec["text"], "note_id": rec["note_id"]})

    log.info(
        "Loaded %d records (skipped %d too long, %d too few sections)",
        len(records), skipped_long, skipped_sections,
    )

    ds = Dataset.from_list(records)
    dsd = DatasetDict({"finetune": ds})
    dsd.save_to_disk(args.output)
    log.info("Saved DatasetDict to %s (%d examples in finetune split)", args.output, len(ds))

    text_lengths = [len(r["text"]) for r in records]
    log.info(
        "Text length stats: min=%d, median=%d, mean=%d, max=%d",
        min(text_lengths),
        sorted(text_lengths)[len(text_lengths) // 2],
        sum(text_lengths) // len(text_lengths),
        max(text_lengths),
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fix Term2Note training data: replace multi-token markers with single special tokens.

Mapping:
  <|section|>  → <|reserved_special_token_0|>  (token 128002)
  <|terms|>    → <|reserved_special_token_1|>  (token 128003)
  <|content|>  → <|reserved_special_token_2|>  (token 128005)

Reads train_term2note.jsonl, writes train_term2note_v2.jsonl with replacements,
then rebuilds the HuggingFace DatasetDict at splits_term2note_v2/.
"""
import json
import logging
from pathlib import Path
from datasets import Dataset, DatasetDict

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("fix_tokens")

DATA_DIR = Path("/fs1/projects/unlearning_pretraining/Proj_code/data/term2note")
INPUT = DATA_DIR / "train_term2note.jsonl"
OUTPUT = DATA_DIR / "train_term2note_v2.jsonl"
SPLITS_OUT = DATA_DIR / "splits_term2note_v2"

TOKEN_MAP = {
    "<|section|>": "<|reserved_special_token_0|>",
    "<|terms|>": "<|reserved_special_token_1|>",
    "<|content|>": "<|reserved_special_token_2|>",
}

MAX_TEXT_LEN = 8000
MIN_SECTIONS = 3


def replace_tokens(text):
    for old, new in TOKEN_MAP.items():
        text = text.replace(old, new)
    return text


def main():
    records_out = []
    skipped_long = 0
    skipped_sections = 0

    with open(INPUT) as f_in, open(OUTPUT, "w") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            rec = json.loads(line)
            rec["text"] = replace_tokens(rec["text"])

            f_out.write(json.dumps(rec) + "\n")

            if len(rec["text"]) > MAX_TEXT_LEN:
                skipped_long += 1
                continue
            if rec.get("n_sections", 6) < MIN_SECTIONS:
                skipped_sections += 1
                continue
            records_out.append({"text": rec["text"], "note_id": rec["note_id"]})

    log.info("Wrote %d records to %s", len(records_out) + skipped_long + skipped_sections, OUTPUT)
    log.info("For HF dataset: %d kept, %d too long, %d too few sections",
             len(records_out), skipped_long, skipped_sections)

    ds = Dataset.from_list(records_out)
    dsd = DatasetDict({"finetune": ds})
    dsd.save_to_disk(str(SPLITS_OUT))
    log.info("Saved DatasetDict to %s (%d examples)", SPLITS_OUT, len(ds))

    # Verify: check a sample record has the new tokens
    sample = records_out[0]["text"]
    for old in TOKEN_MAP:
        assert old not in sample, f"Old token {old} still present!"
    for new in TOKEN_MAP.values():
        assert new in sample, f"New token {new} not found!"
    log.info("Verification passed: old tokens gone, new tokens present")


if __name__ == "__main__":
    main()

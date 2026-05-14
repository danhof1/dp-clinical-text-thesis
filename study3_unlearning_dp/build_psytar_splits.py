#!/usr/bin/env python3
"""
Build PsyTAR splits for SynBench MIA replication.

Creates a HuggingFace DatasetDict with:
  finetune      — 350 records (= members, used by dp_lora.py)
  members       — 350 records (same data, used by MIA)
  nonmembers    — 200 records (MIA test non-members)
  auxiliary     — ~103 records (SynBench reference LM training data)
"""
import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict

PSYTAR_PATH = "/fs1/projects/unlearning_pretraining/Proj_code/data/psytar/reviews.jsonl"
OUTPUT_DIR  = "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/splits_psytar"

MIN_TOKENS   = 20
N_MEMBERS    = 350
N_NONMEMBERS = 200
SEED = 42


def main():
    records = []
    with open(PSYTAR_PATH) as f:
        for line in f:
            rec = json.loads(line)
            n_tokens = len(rec["text"].split())
            if n_tokens >= MIN_TOKENS:
                records.append(rec)

    print(f"Loaded {len(records)} records with >= {MIN_TOKENS} tokens")

    random.seed(SEED)
    random.shuffle(records)

    members = records[:N_MEMBERS]
    nonmembers = records[N_MEMBERS:N_MEMBERS + N_NONMEMBERS]
    auxiliary = records[N_MEMBERS + N_NONMEMBERS:]

    print(f"  members:    {len(members)}")
    print(f"  nonmembers: {len(nonmembers)}")
    print(f"  auxiliary:  {len(auxiliary)}")

    def to_dataset(recs, split_name):
        rows = {
            "text": [r["text"] for r in recs],
            "record_id": [r["record_id"] for r in recs],
            "drug": [r["drug"] for r in recs],
            "category": [r["category"] for r in recs],
            "doc_id": [f"psytar_{i}" for i in range(len(recs))],
            "split": [split_name] * len(recs),
            "is_canary": [False] * len(recs),
        }
        return Dataset.from_dict(rows)

    dd = DatasetDict({
        "finetune": to_dataset(members, "finetune"),
        "members": to_dataset(members, "members"),
        "nonmembers": to_dataset(nonmembers, "nonmembers"),
        "auxiliary": to_dataset(auxiliary, "auxiliary"),
    })

    out = Path(OUTPUT_DIR)
    dd.save_to_disk(str(out))
    print(f"\nSaved DatasetDict to {out}")
    print(dd)

    stats = {
        "total_filtered": len(records),
        "members": len(members),
        "nonmembers": len(nonmembers),
        "auxiliary": len(auxiliary),
        "min_tokens": MIN_TOKENS,
        "seed": SEED,
    }
    (out / "split_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Split stats: {stats}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run completion attack on nonmembers by swapping the split name."""
import sys
import os
import yaml
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

config_path = sys.argv[1]
with open(config_path) as f:
    cfg = yaml.safe_load(f)

from datasets import load_from_disk, DatasetDict

splits = load_from_disk(cfg["splits_path"])

tmp_path = cfg["splits_path"] + "_nonmem_swap"
swapped = DatasetDict({
    "members": splits["nonmembers"],
    "nonmembers": splits["members"],
})
swapped.save_to_disk(tmp_path)
print(f"Created swapped splits at {tmp_path}")
print(f"  members (actually nonmembers): {len(swapped['members'])}")

cfg["splits_path"] = tmp_path
tmp_config = config_path.replace(".yaml", "_swapped.yaml")
with open(tmp_config, "w") as f:
    yaml.safe_dump(cfg, f)

sys.argv = ["", "--config", tmp_config]
from attacks.completion import main
main()

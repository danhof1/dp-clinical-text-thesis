#!/usr/bin/env python3
"""Run completion attack on a specific split by swapping it into the 'members' position."""
import sys
import os
import yaml
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

config_path = sys.argv[1]
swap_split = sys.argv[2]  # e.g. "nonmembers" or "clean_nonmembers"

with open(config_path) as f:
    cfg = yaml.safe_load(f)

from datasets import load_from_disk, DatasetDict

splits = load_from_disk(cfg["splits_path"])

tmp_path = cfg["splits_path"] + f"_{swap_split}_swap"
swapped = DatasetDict({
    "members": splits[swap_split],
    "nonmembers": splits["members"],
})
swapped.save_to_disk(tmp_path)
print(f"Swapped '{swap_split}' into members position: {len(swapped['members'])} examples")

cfg["splits_path"] = tmp_path
tmp_config = config_path.replace(".yaml", f"_{swap_split}_swapped.yaml")
with open(tmp_config, "w") as f:
    yaml.safe_dump(cfg, f)

sys.argv = ["", "--config", tmp_config]
from attacks.completion import main
main()

#!/usr/bin/env python3
import json, yaml, sys
from pathlib import Path
from datasets import load_from_disk, DatasetDict

config_path = sys.argv[1]
with open(config_path) as f:
    cfg = yaml.safe_load(f)

splits = load_from_disk(cfg["splits_path"])
tmp_path = cfg["splits_path"] + "_synth_swap"
swapped = DatasetDict({
    "members": splits["nonmembers"],
    "nonmembers": splits["members"],
})
swapped.save_to_disk(tmp_path)
print(f"Swapped: members={len(swapped['members'])}")

cfg["splits_path"] = tmp_path
tmp_config = config_path.replace(".yaml", "_swapped.yaml")
with open(tmp_config, "w") as f:
    yaml.safe_dump(cfg, f)

sys.argv = ["", "--config", tmp_config]
from attacks.completion import main
main()

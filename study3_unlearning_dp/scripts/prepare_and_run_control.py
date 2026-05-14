#!/usr/bin/env python3
"""
Cluster-side script: build HF splits from downloaded control texts and submit
the completion attack pipeline.

Run on cluster after SCP'ing data/pretraining_control/:
    python prepare_and_run_control.py

Steps:
  1. Load all_texts.jsonl → create HuggingFace DatasetDict (splits_control)
  2. Submit completion attacks on control splits:
     - baseline model on medical_wiki "members" (positive control)
     - baseline model on synthetic "nonmembers" (negative control)
     - baseline model on gutenberg "retain" (cross-domain control)
  3. Submit MIA tier1 on control splits (baseline model)
"""

from __future__ import annotations

import json
import subprocess
import sys
import yaml
from pathlib import Path
from datasets import Dataset, DatasetDict

REPO   = "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp"
BASE   = "/fs1/shared/model/llm/Llama-3.1-8B-Instruct"
PYTHON = "/fs1/home/h702839428/python_daniel/bin/python"
LOGS   = "/fs1/projects/unlearning_pretraining/Proj_code/logs"
ACCT   = "unlearning_pretraining"
DATA   = f"{REPO}/data/pretraining_control"

submitted = []
failed    = []


def sbatch(script, tag, dependency=None):
    p = Path("/tmp") / f"{tag}.sbatch"
    p.write_text(script)
    cmd = ["sbatch"]
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    cmd.append(str(p))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        jid = r.stdout.strip().split()[-1]
        submitted.append((tag, jid))
        print(f"  {tag} -> job {jid}" + (f" (after {dependency})" if dependency else ""))
        return jid
    else:
        failed.append((tag, r.stderr.strip()))
        print(f"  {tag} FAILED: {r.stderr.strip()}", file=sys.stderr)
        return None


# =========================================================================
# 1. Build HuggingFace splits from JSONL
# =========================================================================
print("=== Building HuggingFace splits from control texts ===")

jsonl_path = Path(DATA) / "all_texts.jsonl"
if not jsonl_path.exists():
    print(f"ERROR: {jsonl_path} not found. SCP data first.")
    sys.exit(1)

texts = []
with open(jsonl_path, encoding="utf-8") as f:
    for line in f:
        texts.append(json.loads(line))

print(f"  Loaded {len(texts)} texts")

medical = [t for t in texts if t["category"] == "medical"]
non_medical = [t for t in texts if t["category"] == "non_medical"]
literature = [t for t in texts if t["category"] == "literature"]
synthetic = [t for t in texts if t["category"] == "random_negative"]

print(f"  medical={len(medical)}, non_medical={len(non_medical)}, literature={len(literature)}, synthetic={len(synthetic)}")

splits_dir = Path(DATA) / "splits_control"

ds = DatasetDict({
    "members": Dataset.from_list([
        {"text": t["text"], "title": t.get("title", ""), "source": t["source"], "category": t["category"]}
        for t in medical
    ]),
    "nonmembers": Dataset.from_list([
        {"text": t["text"], "title": t.get("title", ""), "source": t["source"], "category": t["category"]}
        for t in synthetic
    ]),
    "clean_nonmembers": Dataset.from_list([
        {"text": t["text"], "title": t.get("title", ""), "source": t["source"], "category": t["category"]}
        for t in non_medical + literature
    ]),
    "finetune": Dataset.from_list([
        {"text": t["text"], "title": t.get("title", ""), "source": t["source"], "category": t["category"]}
        for t in medical
    ]),
    "retain": Dataset.from_list([
        {"text": t["text"], "title": t.get("title", ""), "source": t["source"], "category": t["category"]}
        for t in non_medical + literature
    ]),
    "canaries": Dataset.from_list([
        {"text": t["text"], "title": t.get("title", ""), "source": t["source"], "category": t["category"]}
        for t in synthetic[:10]
    ]),
})

ds.save_to_disk(str(splits_dir))
print(f"\n  Saved splits to {splits_dir}")
for name, split in ds.items():
    print(f"    {name}: {len(split)} examples")


# =========================================================================
# 2. Completion attacks on control texts (baseline model)
# =========================================================================
print("\n=== Submitting completion attacks on control splits ===")

# Attack on medical Wikipedia "members" — known-in-training positive control
tag = "compl_control_medical_wiki"
config = {
    "base_model": BASE,
    "splits_path": str(splits_dir),
    "output_dir": f"{REPO}/outputs/attacks/{tag}",
    "prefix_ratio": 0.5,
    "max_length": 512,
    "max_examples": 500,
}
config_path = f"{REPO}/config/{tag}.yaml"
with open(config_path, "w") as f:
    yaml.safe_dump(config, f)

sbatch(f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --account={ACCT}
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output={LOGS}/{tag}_%j.log

cd "{REPO}"
{PYTHON} -m attacks.completion --config "{config_path}"
""", tag)


# Attack on synthetic "nonmembers" — negative control (never in training)
# Need to swap members/nonmembers like submit_nonmember_completion.py
wrapper_script = f"""#!/usr/bin/env python3
import json, yaml, sys
from pathlib import Path
from datasets import load_from_disk, DatasetDict

config_path = sys.argv[1]
with open(config_path) as f:
    cfg = yaml.safe_load(f)

splits = load_from_disk(cfg["splits_path"])
tmp_path = cfg["splits_path"] + "_synth_swap"
swapped = DatasetDict({{
    "members": splits["nonmembers"],
    "nonmembers": splits["members"],
}})
swapped.save_to_disk(tmp_path)
print(f"Swapped: members={{len(swapped['members'])}}")

cfg["splits_path"] = tmp_path
tmp_config = config_path.replace(".yaml", "_swapped.yaml")
with open(tmp_config, "w") as f:
    yaml.safe_dump(cfg, f)

sys.argv = ["", "--config", tmp_config]
from attacks.completion import main
main()
"""
wrapper_path = f"{REPO}/scripts/run_control_synth_completion.py"
Path(wrapper_path).write_text(wrapper_script)

tag = "compl_control_synthetic"
config = {
    "base_model": BASE,
    "splits_path": str(splits_dir),
    "output_dir": f"{REPO}/outputs/attacks/{tag}",
    "prefix_ratio": 0.5,
    "max_length": 512,
    "max_examples": 500,
}
config_path = f"{REPO}/config/{tag}.yaml"
with open(config_path, "w") as f:
    yaml.safe_dump(config, f)

sbatch(f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --account={ACCT}
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output={LOGS}/{tag}_%j.log

cd "{REPO}"
{PYTHON} "{wrapper_path}" "{config_path}"
""", tag)


# Attack on literature/non-medical (clean_nonmembers = gutenberg + general wiki)
wrapper_script2 = f"""#!/usr/bin/env python3
import json, yaml, sys
from pathlib import Path
from datasets import load_from_disk, DatasetDict

config_path = sys.argv[1]
with open(config_path) as f:
    cfg = yaml.safe_load(f)

splits = load_from_disk(cfg["splits_path"])
tmp_path = cfg["splits_path"] + "_lit_swap"
swapped = DatasetDict({{
    "members": splits["clean_nonmembers"],
    "nonmembers": splits["nonmembers"],
}})
swapped.save_to_disk(tmp_path)
print(f"Swapped: members={{len(swapped['members'])}}")

cfg["splits_path"] = tmp_path
tmp_config = config_path.replace(".yaml", "_swapped.yaml")
with open(tmp_config, "w") as f:
    yaml.safe_dump(cfg, f)

sys.argv = ["", "--config", tmp_config]
from attacks.completion import main
main()
"""
wrapper_path2 = f"{REPO}/scripts/run_control_lit_completion.py"
Path(wrapper_path2).write_text(wrapper_script2)

tag = "compl_control_literature"
config = {
    "base_model": BASE,
    "splits_path": str(splits_dir),
    "output_dir": f"{REPO}/outputs/attacks/{tag}",
    "prefix_ratio": 0.5,
    "max_length": 512,
    "max_examples": 500,
}
config_path = f"{REPO}/config/{tag}.yaml"
with open(config_path, "w") as f:
    yaml.safe_dump(config, f)

sbatch(f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --account={ACCT}
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output={LOGS}/{tag}_%j.log

cd "{REPO}"
{PYTHON} "{wrapper_path2}" "{config_path}"
""", tag)


# =========================================================================
# 3. MIA Tier1 on control splits (baseline model)
# =========================================================================
print("\n=== Submitting MIA Tier1 on control splits ===")

tag = "mia_control_baseline"
mia_config = f"""base_model: {BASE}
splits_path: {splits_dir}
reference_model: {BASE}
output_dir: {REPO}/outputs/attacks/control_baseline/tier1
max_length: 1024
"""
config_path = f"{REPO}/config/attacks_tier1_control.yaml"
Path(config_path).write_text(mia_config)

sbatch(f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --account={ACCT}
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output={LOGS}/{tag}_%j.log

cd "{REPO}"
{PYTHON} -m attacks.tier1 --config "{config_path}"
""", tag)


# =========================================================================
# Summary
# =========================================================================
print(f"\n{'='*60}")
print(f"Submitted {len(submitted)}/{len(submitted)+len(failed)} jobs")
if submitted:
    print("\nJob IDs:")
    for t, jid in submitted:
        print(f"  {jid:>8}  {t}")
if failed:
    print("\nFailed:")
    for t, err in failed:
        print(f"  {t}: {err}")

print(f"""
Control Experiment Design:
  compl_control_medical_wiki  = Baseline on medical Wikipedia (KNOWN in training)
  compl_control_synthetic     = Baseline on random synthetic (NOT in training)
  compl_control_literature    = Baseline on Gutenberg + general Wiki (KNOWN in training)
  mia_control_baseline        = MIA on control splits

Key comparisons:
  EMR(medical_wiki) vs EMR(MTSamples members) vs EMR(synthetic)

  If wiki EMR >> MTSamples EMR (0.026): MTSamples is NOT memorized (GOOD)
  If wiki EMR ≈ MTSamples EMR: both are at clinical text predictability floor
  If wiki EMR < synthetic EMR: something is wrong with the control
""")

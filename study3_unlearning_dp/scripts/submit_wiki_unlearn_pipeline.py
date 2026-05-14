#!/usr/bin/env python3
"""
Wikipedia Unlearn Pipeline: definitively test unlearn → synthetic retune → DP re-learn.

Experiment:
  Step 1: Baseline EMR on medical Wikipedia (already have: 0.035 mean, 0.85 max)
  Step 2: RMU unlearn Wikipedia articles → measure EMR (expect big drop)
  Step 3a: DP-LoRA on SAME Wikipedia articles at eps=1,8,inf → measure wiki EMR
  Step 3b: DP-LoRA on MTSamples (unrelated clinical text) at eps=1,8,inf → measure wiki EMR

Key question: Does DP prevent re-memorization even with direct re-exposure (3a)?
             Does clinical utility recover from unrelated text (3b)?

Jobs: 14 total
  Phase 1: RMU unlearn (1 job, ~10 min)
  Phase 2: Post-unlearn completion attack (1 job, ~5 min)
  Phase 3: DP-LoRA retuning (6 jobs, ~30 min each)
  Phase 4: Completion attacks on all DP checkpoints (6 jobs, ~5 min each)
"""

import json
import subprocess
import sys
import yaml
from pathlib import Path
from datasets import Dataset, DatasetDict, load_from_disk

REPO   = "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp"
BASE   = "/fs1/shared/model/llm/Llama-3.1-8B-Instruct"
PYTHON = "/fs1/home/h702839428/python_daniel/bin/python"
LOGS   = "/fs1/projects/unlearning_pretraining/Proj_code/logs"
ACCT   = "unlearning_pretraining"

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
# 0. Build splits for the experiment
# =========================================================================
print("=== Building experiment splits ===")

# Load wiki control texts
wiki_jsonl = Path(f"{REPO}/data/pretraining_control/all_texts.jsonl")
wiki_texts = []
with open(wiki_jsonl, encoding="utf-8") as f:
    for line in f:
        t = json.loads(line)
        if t["category"] == "medical":
            wiki_texts.append(t)

synth_texts = []
nonmed_texts = []
with open(wiki_jsonl, encoding="utf-8") as f:
    for line in f:
        t = json.loads(line)
        if t["category"] == "random_negative":
            synth_texts.append(t)
        elif t["category"] in ("non_medical", "literature"):
            nonmed_texts.append(t)

print(f"  Medical wiki articles: {len(wiki_texts)}")
print(f"  Synthetic texts: {len(synth_texts)}")
print(f"  Non-medical texts: {len(nonmed_texts)}")

# Load MTSamples retain set from splits_v2
splits_v2 = load_from_disk(f"{REPO}/outputs/splits_v2")
retain_texts = [{"text": r["text"]} for r in splits_v2["retain"]]
mts_members = [{"text": r["text"]} for r in splits_v2["members"]]
print(f"  MTSamples retain: {len(retain_texts)}")
print(f"  MTSamples members: {len(mts_members)}")

# Create splits_wiki_forget:
#   finetune = wiki (forget set for RMU, train set for DP-LoRA 4a)
#   retain = MTSamples retain (keep set for RMU)
#   members = wiki (targets for completion attack)
#   nonmembers = synthetic
splits_wiki = DatasetDict({
    "finetune": Dataset.from_list([
        {"text": t["text"], "title": t.get("title", "")} for t in wiki_texts
    ]),
    "retain": Dataset.from_list([
        {"text": t["text"]} for t in retain_texts
    ]),
    "members": Dataset.from_list([
        {"text": t["text"], "title": t.get("title", "")} for t in wiki_texts
    ]),
    "nonmembers": Dataset.from_list([
        {"text": t["text"]} for t in synth_texts
    ]),
    "clean_nonmembers": Dataset.from_list([
        {"text": t["text"]} for t in nonmed_texts
    ]),
    "canaries": Dataset.from_list([
        {"text": t["text"]} for t in synth_texts[:10]
    ]),
})

wiki_splits_path = f"{REPO}/data/pretraining_control/splits_wiki_forget"
splits_wiki.save_to_disk(wiki_splits_path)
print(f"\n  Saved splits_wiki_forget to {wiki_splits_path}")
for name, split in splits_wiki.items():
    print(f"    {name}: {len(split)} examples")

# Create splits_wiki_retune_mts:
#   finetune = MTSamples members (train set for DP-LoRA 4b)
#   retain = same retain
#   members = wiki (still measure wiki EMR)
splits_mts = DatasetDict({
    "finetune": Dataset.from_list([
        {"text": t["text"]} for t in mts_members
    ]),
    "retain": Dataset.from_list([
        {"text": t["text"]} for t in retain_texts
    ]),
    "members": Dataset.from_list([
        {"text": t["text"], "title": t.get("title", "")} for t in wiki_texts
    ]),
    "nonmembers": Dataset.from_list([
        {"text": t["text"]} for t in synth_texts
    ]),
    "clean_nonmembers": Dataset.from_list([
        {"text": t["text"]} for t in nonmed_texts
    ]),
    "canaries": Dataset.from_list([
        {"text": t["text"]} for t in synth_texts[:10]
    ]),
})

mts_splits_path = f"{REPO}/data/pretraining_control/splits_wiki_retune_mts"
splits_mts.save_to_disk(mts_splits_path)
print(f"\n  Saved splits_wiki_retune_mts to {mts_splits_path}")
for name, split in splits_mts.items():
    print(f"    {name}: {len(split)} examples")


# =========================================================================
# 1. RMU unlearn Wikipedia articles
# =========================================================================
print("\n=== Phase 1: RMU unlearn Wikipedia ===")

rmu_out = f"{REPO}/outputs/unlearn_rmu_wiki"
rmu_config = f"""base_model: {BASE}
splits_path: {wiki_splits_path}
output_dir: {rmu_out}
layer_id: 7
alpha: 1200.0
steering_coeff: 20.0
num_steps: 150
learning_rate: 5.0e-05
weight_decay: 0.0
max_grad_norm: 1.0
batch_size: 4
max_seq_length: 512
use_lora: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
- q_proj
- k_proj
- v_proj
- o_proj
- gate_proj
- up_proj
- down_proj
bf16: true
seed: 42
log_every: 10
"""
rmu_config_path = f"{REPO}/config/unlearn_rmu_wiki.yaml"
Path(rmu_config_path).write_text(rmu_config)

rmu_jid = sbatch(f"""#!/bin/bash
#SBATCH --job-name=rmu_wiki
#SBATCH --account={ACCT}
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output={LOGS}/rmu_wiki_%j.log

cd "{REPO}"
{PYTHON} -m unlearn.rmu --config "{rmu_config_path}"
""", "rmu_wiki")


# =========================================================================
# 2. Post-unlearn completion attack (measure wiki EMR after RMU)
# =========================================================================
print("\n=== Phase 2: Post-unlearn completion attack ===")

tag = "compl_post_rmu_wiki"
compl_config = {
    "base_model": f"{rmu_out}/step_final",
    "splits_path": wiki_splits_path,
    "output_dir": f"{REPO}/outputs/attacks/{tag}",
    "prefix_ratio": 0.5,
    "max_length": 512,
    "max_examples": 500,
}
config_path = f"{REPO}/config/{tag}.yaml"
with open(config_path, "w") as f:
    yaml.safe_dump(compl_config, f)

sbatch(f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --account={ACCT}
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output={LOGS}/{tag}_%j.log

cd "{REPO}"
{PYTHON} -m attacks.completion --config "{config_path}"
""", tag, dependency=rmu_jid)


# =========================================================================
# 3a. DP-LoRA on SAME Wikipedia articles (can DP prevent re-memorization?)
# =========================================================================
print("\n=== Phase 3a: DP-LoRA on Wikipedia (re-exposure test) ===")

dp_wiki_jobs = {}

for eps_val, eps_str in [(1.0, "eps1"), (8.0, "eps8"), (None, "epsinf")]:
    tag = f"dp_{eps_str}_wiki_on_wiki"
    out_dir = f"{REPO}/outputs/{tag}"
    eps_line = f"epsilon: {eps_val}" if eps_val is not None else "epsilon: null"

    dp_config = f"""base_model: "{rmu_out}/step_final"
splits_path: "{wiki_splits_path}"
output_dir: "{out_dir}"

lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
- q_proj
- k_proj
- v_proj
- o_proj
- gate_proj
- up_proj
- down_proj

num_epochs: 3
physical_batch_size: 2
logical_batch_size: 64
learning_rate: 0.0005
max_seq_length: 512
max_grad_norm: 1.0
warmup_ratio: 0.03
weight_decay: 0.0

{eps_line}
delta: 0.00001
secure_mode: false

bf16: true
seed: 42
log_every: 10
"""
    config_path = f"{REPO}/config/{tag}.yaml"
    Path(config_path).write_text(dp_config)

    jid = sbatch(f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --account={ACCT}
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output={LOGS}/{tag}_%j.log

cd "{REPO}"
{PYTHON} -m finetune.dp_lora --config "{config_path}"
""", tag, dependency=rmu_jid)

    dp_wiki_jobs[eps_str] = jid


# =========================================================================
# 3b. DP-LoRA on MTSamples (unrelated clinical text recovery)
# =========================================================================
print("\n=== Phase 3b: DP-LoRA on MTSamples (unrelated clinical retune) ===")

dp_mts_jobs = {}

for eps_val, eps_str in [(1.0, "eps1"), (8.0, "eps8"), (None, "epsinf")]:
    tag = f"dp_{eps_str}_wiki_on_mts"
    out_dir = f"{REPO}/outputs/{tag}"
    eps_line = f"epsilon: {eps_val}" if eps_val is not None else "epsilon: null"

    dp_config = f"""base_model: "{rmu_out}/step_final"
splits_path: "{mts_splits_path}"
output_dir: "{out_dir}"

lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
- q_proj
- k_proj
- v_proj
- o_proj
- gate_proj
- up_proj
- down_proj

num_epochs: 3
physical_batch_size: 2
logical_batch_size: 64
learning_rate: 0.0005
max_seq_length: 512
max_grad_norm: 1.0
warmup_ratio: 0.03
weight_decay: 0.0

{eps_line}
delta: 0.00001
secure_mode: false

bf16: true
seed: 42
log_every: 10
"""
    config_path = f"{REPO}/config/{tag}.yaml"
    Path(config_path).write_text(dp_config)

    jid = sbatch(f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --account={ACCT}
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output={LOGS}/{tag}_%j.log

cd "{REPO}"
{PYTHON} -m finetune.dp_lora --config "{config_path}"
""", tag, dependency=rmu_jid)

    dp_mts_jobs[eps_str] = jid


# =========================================================================
# 4. Completion attacks on all DP checkpoints (measure wiki EMR)
# =========================================================================
print("\n=== Phase 4: Completion attacks on all DP checkpoints ===")

# 4a: DP on Wikipedia → measure wiki EMR
for eps_str in ["eps1", "eps8", "epsinf"]:
    adapter = f"{REPO}/outputs/dp_{eps_str}_wiki_on_wiki/final"
    dep = dp_wiki_jobs.get(eps_str)
    tag = f"compl_{eps_str}_wiki_on_wiki"

    compl_config = {
        "base_model": f"{rmu_out}/step_final",
        "adapter_path": adapter,
        "splits_path": wiki_splits_path,
        "output_dir": f"{REPO}/outputs/attacks/{tag}",
        "prefix_ratio": 0.5,
        "max_length": 512,
        "max_examples": 500,
    }
    config_path = f"{REPO}/config/{tag}.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(compl_config, f)

    sbatch(f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --account={ACCT}
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output={LOGS}/{tag}_%j.log

cd "{REPO}"
{PYTHON} -m attacks.completion --config "{config_path}"
""", tag, dependency=dep)


# 4b: DP on MTSamples → measure wiki EMR
for eps_str in ["eps1", "eps8", "epsinf"]:
    adapter = f"{REPO}/outputs/dp_{eps_str}_wiki_on_mts/final"
    dep = dp_mts_jobs.get(eps_str)
    tag = f"compl_{eps_str}_wiki_on_mts"

    compl_config = {
        "base_model": f"{rmu_out}/step_final",
        "adapter_path": adapter,
        "splits_path": wiki_splits_path,
        "output_dir": f"{REPO}/outputs/attacks/{tag}",
        "prefix_ratio": 0.5,
        "max_length": 512,
        "max_examples": 500,
    }
    config_path = f"{REPO}/config/{tag}.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(compl_config, f)

    sbatch(f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --account={ACCT}
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output={LOGS}/{tag}_%j.log

cd "{REPO}"
{PYTHON} -m attacks.completion --config "{config_path}"
""", tag, dependency=dep)


# =========================================================================
# Summary
# =========================================================================
print(f"\n{'='*70}")
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
Pipeline:
  RMU unlearn wiki (1 job, ~10 min)
    -> Post-RMU wiki EMR (1 job, ~5 min)
    -> DP-LoRA on wiki eps=1,8,inf (3 jobs, ~30 min) -> wiki EMR (3 jobs, ~5 min)
    -> DP-LoRA on MTS  eps=1,8,inf (3 jobs, ~30 min) -> wiki EMR (3 jobs, ~5 min)

Expected results:
  Baseline wiki EMR:           0.035 (already measured)
  Post-RMU wiki EMR:           ??? (expect significant drop)
  DP on wiki eps=1 wiki EMR:   ??? (does DP prevent re-memorization?)
  DP on wiki eps=inf wiki EMR: ??? (expect re-memorization without DP)
  DP on MTS eps=1 wiki EMR:    ??? (expect low — unrelated data can't re-memorize)
  DP on MTS eps=inf wiki EMR:  ??? (expect low — even unconstrained, MTS doesn't contain wiki)

Best case for thesis:
  - Post-RMU: wiki EMR drops substantially
  - DP on wiki eps=1,8: wiki EMR stays low (DP prevents re-memorization!)
  - DP on wiki eps=inf: wiki EMR rebounds (proves it's DP doing the work)
  - DP on MTS all eps: wiki EMR stays low (clinical retune doesn't re-introduce memorization)
""")

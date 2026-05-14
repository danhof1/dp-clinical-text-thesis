#!/usr/bin/env python3
"""
Clinical Memorization Proof-of-Concept Pipeline.

Simulates clinical text contamination in pretraining by fine-tuning
Llama-3.1-8B on MTSamples clinical notes WITHOUT DP (inducing memorization),
then applies the unlearn+DP pipeline to erase it.

Pipeline:
  Step 0: Fine-tune on MTSamples members WITHOUT DP (induce memorization)
  Step 1: Completion attack on memorized model (verify EMR is high)
  Step 2: RMU unlearn the memorized clinical text
  Step 3: Completion attack post-RMU (verify EMR drops)
  Step 4a: DP-LoRA retune on PMC (different clinical text) at eps=1,8
  Step 4b: Completion attacks on DP models (verify EMR stays low)
  Step 5: Generation + fidelity eval (verify clinical utility recovered)
"""
import subprocess
import sys
import yaml
from pathlib import Path

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
# Paths for this experiment
# =========================================================================
SPLITS_V2 = f"{REPO}/outputs/splits_v2"
PMC_SPLITS = f"{REPO}/outputs/splits_v1_pmc"

# Output dirs for this experiment
MEMORIZED_MODEL = f"{REPO}/outputs/clinical_poc/memorized_model"
RMU_MODEL       = f"{REPO}/outputs/clinical_poc/unlearn_rmu"
DP_EPS1_MODEL   = f"{REPO}/outputs/clinical_poc/dp_eps1_pmc"
DP_EPS8_MODEL   = f"{REPO}/outputs/clinical_poc/dp_eps8_pmc"
DP_EPSINF_MODEL = f"{REPO}/outputs/clinical_poc/dp_epsinf_pmc"


# =========================================================================
# Step 0: Fine-tune on MTSamples WITHOUT DP to induce memorization
# =========================================================================
print("=== Step 0: Induce memorization (fine-tune without DP, 10 epochs) ===")

memorize_config = f"""base_model: "{BASE}"
splits_path: "{SPLITS_V2}"
output_dir: "{MEMORIZED_MODEL}"

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

num_epochs: 10
physical_batch_size: 4
logical_batch_size: 16
learning_rate: 0.0005
max_seq_length: 512
max_grad_norm: 1.0
warmup_ratio: 0.03
weight_decay: 0.0

epsilon: null
delta: 0.00001
secure_mode: false

bf16: true
seed: 42
log_every: 10
"""

memorize_config_path = f"{REPO}/config/clinical_poc_memorize.yaml"
Path(memorize_config_path).write_text(memorize_config)

memorize_jid = sbatch(f"""#!/bin/bash
#SBATCH --job-name=poc_memorize
#SBATCH --account={ACCT}
#SBATCH --partition=defq
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output={LOGS}/poc_memorize_%j.log

cd "{REPO}"
{PYTHON} -m finetune.dp_lora --config "{memorize_config_path}"
""", "poc_memorize")


# =========================================================================
# Step 1: Completion attack on memorized model
# =========================================================================
print("\n=== Step 1: Verify memorization ===")

tag = "compl_poc_memorized"
compl_config = {
    "base_model": BASE,
    "adapter_path": f"{MEMORIZED_MODEL}/final",
    "splits_path": SPLITS_V2,
    "output_dir": f"{REPO}/outputs/attacks/{tag}",
    "prefix_ratio": 0.5,
    "max_length": 512,
    "max_examples": 500,
}
config_path = f"{REPO}/config/{tag}.yaml"
with open(config_path, "w") as f:
    yaml.safe_dump(compl_config, f)

compl_mem_jid = sbatch(f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --account={ACCT}
#SBATCH --partition=defq
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output={LOGS}/{tag}_%j.log

cd "{REPO}"
{PYTHON} -m attacks.completion --config "{config_path}"
""", tag, dependency=memorize_jid)


# =========================================================================
# Step 2: RMU unlearn the memorized clinical text
# =========================================================================
print("\n=== Step 2: RMU unlearn ===")

rmu_config = f"""base_model: "{MEMORIZED_MODEL}/final_merged"
splits_path: "{SPLITS_V2}"
output_dir: "{RMU_MODEL}"
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
rmu_config_path = f"{REPO}/config/clinical_poc_rmu.yaml"
Path(rmu_config_path).write_text(rmu_config)

# Need to merge LoRA first, then run RMU
merge_and_rmu_jid = sbatch(f"""#!/bin/bash
#SBATCH --job-name=poc_rmu
#SBATCH --account={ACCT}
#SBATCH --partition=defq
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output={LOGS}/poc_rmu_%j.log

cd "{REPO}"

# First merge the memorized LoRA into a standalone model
{PYTHON} -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print('Merging memorized LoRA...')
base = AutoModelForCausalLM.from_pretrained('{BASE}', torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, '{MEMORIZED_MODEL}/final')
merged = model.merge_and_unload()
merged.save_pretrained('{MEMORIZED_MODEL}/final_merged')
tok = AutoTokenizer.from_pretrained('{BASE}')
tok.save_pretrained('{MEMORIZED_MODEL}/final_merged')
print('Merged to {MEMORIZED_MODEL}/final_merged')
"

# Then run RMU
{PYTHON} -m unlearn.rmu --config "{rmu_config_path}"
""", "poc_rmu", dependency=memorize_jid)


# =========================================================================
# Step 3: Completion attack post-RMU
# =========================================================================
print("\n=== Step 3: Post-RMU completion attack ===")

tag = "compl_poc_post_rmu"
compl_config = {
    "base_model": f"{RMU_MODEL}/step_final",
    "splits_path": SPLITS_V2,
    "output_dir": f"{REPO}/outputs/attacks/{tag}",
    "prefix_ratio": 0.5,
    "max_length": 512,
    "max_examples": 500,
}
config_path = f"{REPO}/config/{tag}.yaml"
with open(config_path, "w") as f:
    yaml.safe_dump(compl_config, f)

compl_rmu_jid = sbatch(f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --account={ACCT}
#SBATCH --partition=defq
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output={LOGS}/{tag}_%j.log

cd "{REPO}"
{PYTHON} -m attacks.completion --config "{config_path}"
""", tag, dependency=merge_and_rmu_jid)


# =========================================================================
# Step 4: DP-LoRA retune on PMC (different clinical text)
# =========================================================================
print("\n=== Step 4: DP-LoRA retune on PMC ===")

dp_jobs = {}

for eps_val, eps_str, out_dir in [
    (1.0,  "eps1",   DP_EPS1_MODEL),
    (8.0,  "eps8",   DP_EPS8_MODEL),
    (None, "epsinf", DP_EPSINF_MODEL),
]:
    tag_dp = f"poc_dp_{eps_str}_pmc"
    eps_line = f"epsilon: {eps_val}" if eps_val is not None else "epsilon: null"

    dp_config = f"""base_model: "{RMU_MODEL}/step_final"
splits_path: "{PMC_SPLITS}"
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
    dp_config_path = f"{REPO}/config/{tag_dp}.yaml"
    Path(dp_config_path).write_text(dp_config)

    jid = sbatch(f"""#!/bin/bash
#SBATCH --job-name={tag_dp}
#SBATCH --account={ACCT}
#SBATCH --partition=defq
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output={LOGS}/{tag_dp}_%j.log

cd "{REPO}"
{PYTHON} -m finetune.dp_lora --config "{dp_config_path}"
""", tag_dp, dependency=merge_and_rmu_jid)

    dp_jobs[eps_str] = jid


# =========================================================================
# Step 5: Completion attacks on DP models (measure clinical EMR)
# =========================================================================
print("\n=== Step 5: Completion attacks on DP models ===")

for eps_str, adapter_dir in [
    ("eps1",   DP_EPS1_MODEL),
    ("eps8",   DP_EPS8_MODEL),
    ("epsinf", DP_EPSINF_MODEL),
]:
    tag = f"compl_poc_dp_{eps_str}"
    compl_config = {
        "base_model": f"{RMU_MODEL}/step_final",
        "adapter_path": f"{adapter_dir}/final",
        "splits_path": SPLITS_V2,
        "output_dir": f"{REPO}/outputs/attacks/{tag}",
        "prefix_ratio": 0.5,
        "max_length": 512,
        "max_examples": 500,
    }
    config_path = f"{REPO}/config/{tag}.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(compl_config, f)

    dep = dp_jobs.get(eps_str)
    sbatch(f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --account={ACCT}
#SBATCH --partition=defq
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
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

print("""
Clinical Memorization PoC Pipeline:
  Step 0: Fine-tune on MTSamples (10 epochs, no DP) → induce memorization
  Step 1: Completion attack → verify high EMR
  Step 2: Merge LoRA + RMU unlearn → erase clinical memorization
  Step 3: Completion attack → verify EMR dropped
  Step 4: DP-LoRA on PMC (eps=1,8,inf) → retune on different clinical text
  Step 5: Completion attacks → verify EMR stays low

Expected results:
  Step 1: EMR >> 0.026 baseline (memorization confirmed)
  Step 3: EMR drops to ~0.01 or lower (unlearning works on clinical text)
  Step 5 eps=1,8: EMR stays low (DP prevents re-memorization)
  Step 5 eps=inf: EMR may partially recover (no DP = leakage)

This proves the full pipeline works on clinical text,
not just encyclopedia articles.
""")

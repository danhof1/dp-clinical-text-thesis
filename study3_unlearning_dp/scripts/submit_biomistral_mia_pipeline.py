#!/usr/bin/env python3
"""
BioMistral MIA Pipeline — Retrain deleted adapters + run Tier1 MIA.

Retrains:
  1. RMU unlearn (PMC members, layer=7, alpha=1200, steering=20, 150 steps)
  2. DP-LoRA retune on MTSamples (eps=1, 8, inf)

Then runs Tier1 MIA on:
  - Baseline BioMistral (no unlearning, no DP)
  - Post-RMU checkpoint
  - Post-DP eps=1, 8, inf

All MIA uses splits_v1_pmc with BioMistral-7B as reference model.
"""
import subprocess
import sys
import yaml
from pathlib import Path

REPO   = "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp"
PYTHON = "/fs1/home/h702839428/python_daniel/bin/python"
LOGS   = "/fs1/projects/unlearning_pretraining/Proj_code/logs"
ACCT   = "unlearning_pretraining"

BASE_MODEL  = "/fs1/shared/model/llm/BioMistral-7B"
PMC_SPLITS  = f"{REPO}/outputs/splits_v1_pmc"
MTS_SPLITS  = f"{REPO}/outputs/splits_v2"
OUT_BASE    = f"{REPO}/outputs/clinical_contam/biomistral"

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


def sbatch_header(tag, time="02:00:00", mem="48G", gpus=1):
    return f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --account={ACCT}
#SBATCH --partition=defq
#SBATCH --qos=long
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={LOGS}/{tag}_%j.log

cd "{REPO}"
"""


# =========================================================================
# Phase 0: Tier1 MIA on baseline BioMistral (no adapter, no unlearning)
# =========================================================================
print("--- Phase 0: Tier1 MIA on baseline BioMistral ---")

mia_baseline_config = {
    "base_model": BASE_MODEL,
    "splits_path": PMC_SPLITS,
    "reference_model": BASE_MODEL,
    "output_dir": f"{REPO}/outputs/attacks/mia_biomistral_baseline/tier1",
    "max_length": 1024,
}
cfg_path = f"{REPO}/config/mia_biomistral_baseline.yaml"
Path(cfg_path).write_text(yaml.safe_dump(mia_baseline_config))

baseline_mia_jid = sbatch(
    sbatch_header("mia_bio_base", time="01:00:00") +
    f'{PYTHON} -m attacks.tier1 --config "{cfg_path}"\n',
    "mia_bio_base",
)

# =========================================================================
# Phase 1: RMU unlearn PMC members
# =========================================================================
print("\n--- Phase 1: RMU unlearn (layer=7, alpha=1200, steering=20) ---")

rmu_out = f"{OUT_BASE}/unlearn_rmu"
rmu_config = f"""base_model: "{BASE_MODEL}"
splits_path: "{PMC_SPLITS}"
output_dir: "{rmu_out}"
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
rmu_cfg_path = f"{REPO}/config/mia_biomistral_rmu.yaml"
Path(rmu_cfg_path).write_text(rmu_config)

rmu_jid = sbatch(
    sbatch_header("rmu_bio", time="02:00:00", mem="64G") +
    f'{PYTHON} -m unlearn.rmu --config "{rmu_cfg_path}"\n',
    "rmu_bio",
)

# =========================================================================
# Phase 2: Tier1 MIA on post-RMU checkpoint
# =========================================================================
print("\n--- Phase 2: Tier1 MIA on post-RMU ---")

mia_rmu_config = {
    "base_model": f"{rmu_out}/step_final",
    "splits_path": PMC_SPLITS,
    "reference_model": BASE_MODEL,
    "output_dir": f"{REPO}/outputs/attacks/mia_biomistral_post_rmu/tier1",
    "max_length": 1024,
}
cfg_path = f"{REPO}/config/mia_biomistral_post_rmu.yaml"
Path(cfg_path).write_text(yaml.safe_dump(mia_rmu_config))

rmu_mia_jid = sbatch(
    sbatch_header("mia_bio_rmu", time="01:00:00") +
    f'{PYTHON} -m attacks.tier1 --config "{cfg_path}"\n',
    "mia_bio_rmu",
    dependency=rmu_jid,
)

# =========================================================================
# Phase 3: DP-LoRA retune on MTSamples (eps=1, 8, inf)
# =========================================================================
print("\n--- Phase 3: DP-LoRA retune on MTSamples ---")

dp_jobs = {}
for eps_val, eps_str in [(1.0, "eps1"), (8.0, "eps8"), (None, "epsinf")]:
    dp_tag = f"dp_bio_{eps_str}"
    dp_out = f"{OUT_BASE}/dp_{eps_str}_mts"
    eps_line = f"epsilon: {eps_val}" if eps_val is not None else "epsilon: null"

    dp_config = f"""base_model: "{rmu_out}/step_final"
splits_path: "{MTS_SPLITS}"
output_dir: "{dp_out}"

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
    dp_cfg_path = f"{REPO}/config/mia_biomistral_dp_{eps_str}.yaml"
    Path(dp_cfg_path).write_text(dp_config)

    jid = sbatch(
        sbatch_header(dp_tag, time="04:00:00", mem="64G") +
        f'{PYTHON} -m finetune.dp_lora --config "{dp_cfg_path}"\n',
        dp_tag,
        dependency=rmu_jid,
    )
    dp_jobs[eps_str] = jid

# =========================================================================
# Phase 4: Tier1 MIA on post-DP models
# =========================================================================
print("\n--- Phase 4: Tier1 MIA on post-DP models ---")

for eps_str in ["eps1", "eps8", "epsinf"]:
    dp_out = f"{OUT_BASE}/dp_{eps_str}_mts"
    mia_tag = f"mia_bio_dp_{eps_str}"

    mia_config = {
        "base_model": f"{rmu_out}/step_final",
        "adapter_path": f"{dp_out}/final",
        "splits_path": PMC_SPLITS,
        "reference_model": BASE_MODEL,
        "output_dir": f"{REPO}/outputs/attacks/mia_biomistral_dp_{eps_str}/tier1",
        "max_length": 1024,
    }
    cfg_path = f"{REPO}/config/mia_biomistral_dp_{eps_str}_tier1.yaml"
    Path(cfg_path).write_text(yaml.safe_dump(mia_config))

    sbatch(
        sbatch_header(mia_tag, time="01:00:00") +
        f'{PYTHON} -m attacks.tier1 --config "{cfg_path}"\n',
        mia_tag,
        dependency=dp_jobs.get(eps_str),
    )


# =========================================================================
# Summary
# =========================================================================
print(f"\n{'='*70}")
print(f"Submitted {len(submitted)}/{len(submitted)+len(failed)} jobs")
if submitted:
    print("\nJob chain:")
    for t, jid in submitted:
        print(f"  {jid:>8}  {t}")
if failed:
    print("\nFailed:")
    for t, err in failed:
        print(f"  {t}: {err}")

print("""
Pipeline:
  Phase 0: Tier1 MIA on baseline BioMistral       (immediate)
  Phase 1: RMU unlearn PMC (L7, alpha=1200)        (immediate)
  Phase 2: Tier1 MIA on post-RMU                   (after Phase 1)
  Phase 3: DP-LoRA eps=1,8,inf on MTSamples        (after Phase 1, parallel)
  Phase 4: Tier1 MIA on each DP model              (after respective Phase 3)

Expected MIA progression:
  baseline: elevated AUC (PMC was in training data)
  post-RMU: AUC should drop toward 0.5
  post-DP:  AUC should stay ~0.5 at finite eps, possibly higher at eps=inf
""")

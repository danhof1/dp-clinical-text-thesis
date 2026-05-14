#!/usr/bin/env python3
"""
Clinical Contamination Pipeline — PMC-LLaMA & BioMistral.

Tests memorization of PMC clinical papers in models known to have been
trained on PubMed Central, then runs the full unlearn+DP pipeline on
whichever model shows memorization.

Pipeline per model:
  Phase 1: Completion attack on PMC members (confirm memorization)
  Phase 2: RMU unlearn PMC members
  Phase 3: Completion attack post-RMU (verify EMR drop)
  Phase 4: DP-LoRA retune on MTSamples (different clinical text) eps=1,8,inf
  Phase 5: Completion attacks on DP models (verify EMR stays low)
  Phase 6: Fidelity eval on DP generations
"""
import subprocess
import sys
import yaml
from pathlib import Path

REPO   = "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp"
PYTHON = "/fs1/home/h702839428/python_daniel/bin/python"
LOGS   = "/fs1/projects/unlearning_pretraining/Proj_code/logs"
ACCT   = "unlearning_pretraining"

PMC_SPLITS  = f"{REPO}/outputs/splits_v1_pmc"
MTS_SPLITS  = f"{REPO}/outputs/splits_v2"

MODELS = {
    "pmc_llama": "/fs1/projects/unlearning_pretraining/Proj_code/models/PMC_LLAMA_7B",
    "biomistral": "/fs1/shared/model/llm/BioMistral-7B",
}

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


def sbatch_header(tag, time="02:00:00", mem="48G"):
    return f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --account={ACCT}
#SBATCH --partition=defq
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={LOGS}/{tag}_%j.log

cd "{REPO}"
"""


for model_key, base_model in MODELS.items():
    print(f"\n{'='*70}")
    print(f"MODEL: {model_key} ({base_model})")
    print(f"{'='*70}")

    OUT = f"{REPO}/outputs/clinical_contam/{model_key}"

    # =====================================================================
    # Phase 1: Completion attack — test memorization on PMC members
    # =====================================================================
    print("\n--- Phase 1: Test memorization ---")

    tag = f"compl_{model_key}_pmc_mem"
    compl_config = {
        "base_model": base_model,
        "splits_path": PMC_SPLITS,
        "output_dir": f"{REPO}/outputs/attacks/{tag}",
        "prefix_ratio": 0.5,
        "max_length": 512,
        "max_examples": 500,
    }
    config_path = f"{REPO}/config/{tag}.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(compl_config, f)

    phase1_jid = sbatch(
        sbatch_header(tag, time="01:00:00") +
        f'{PYTHON} -m attacks.completion --config "{config_path}"\n',
        tag,
    )

    # Also test on MTS members (should NOT be memorized — control)
    tag_ctrl = f"compl_{model_key}_mts_mem"
    ctrl_config = {
        "base_model": base_model,
        "splits_path": MTS_SPLITS,
        "output_dir": f"{REPO}/outputs/attacks/{tag_ctrl}",
        "prefix_ratio": 0.5,
        "max_length": 512,
        "max_examples": 500,
    }
    ctrl_config_path = f"{REPO}/config/{tag_ctrl}.yaml"
    with open(ctrl_config_path, "w") as f:
        yaml.safe_dump(ctrl_config, f)

    sbatch(
        sbatch_header(tag_ctrl, time="01:00:00") +
        f'{PYTHON} -m attacks.completion --config "{ctrl_config_path}"\n',
        tag_ctrl,
    )

    # =====================================================================
    # Phase 2: RMU unlearn PMC members
    # =====================================================================
    print("\n--- Phase 2: RMU unlearn ---")

    rmu_tag = f"rmu_{model_key}"
    rmu_out = f"{OUT}/unlearn_rmu"
    rmu_config = f"""base_model: "{base_model}"
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
    rmu_config_path = f"{REPO}/config/{rmu_tag}.yaml"
    Path(rmu_config_path).write_text(rmu_config)

    rmu_jid = sbatch(
        sbatch_header(rmu_tag, time="02:00:00", mem="64G") +
        f'{PYTHON} -m unlearn.rmu --config "{rmu_config_path}"\n',
        rmu_tag,
        dependency=phase1_jid,
    )

    # =====================================================================
    # Phase 3: Completion attack post-RMU
    # =====================================================================
    print("\n--- Phase 3: Post-RMU completion attack ---")

    tag = f"compl_{model_key}_post_rmu"
    compl_config = {
        "base_model": f"{rmu_out}/step_final",
        "splits_path": PMC_SPLITS,
        "output_dir": f"{REPO}/outputs/attacks/{tag}",
        "prefix_ratio": 0.5,
        "max_length": 512,
        "max_examples": 500,
    }
    config_path = f"{REPO}/config/{tag}.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(compl_config, f)

    phase3_jid = sbatch(
        sbatch_header(tag, time="01:00:00") +
        f'{PYTHON} -m attacks.completion --config "{config_path}"\n',
        tag,
        dependency=rmu_jid,
    )

    # =====================================================================
    # Phase 4: DP-LoRA retune on MTSamples (different clinical text)
    # =====================================================================
    print("\n--- Phase 4: DP-LoRA retune on MTSamples ---")

    dp_jobs = {}
    for eps_val, eps_str in [(1.0, "eps1"), (8.0, "eps8"), (None, "epsinf")]:
        dp_tag = f"dp_{model_key}_{eps_str}_mts"
        dp_out = f"{OUT}/dp_{eps_str}_mts"
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
        dp_config_path = f"{REPO}/config/{dp_tag}.yaml"
        Path(dp_config_path).write_text(dp_config)

        jid = sbatch(
            sbatch_header(dp_tag, time="04:00:00", mem="64G") +
            f'{PYTHON} -m finetune.dp_lora --config "{dp_config_path}"\n',
            dp_tag,
            dependency=rmu_jid,
        )
        dp_jobs[eps_str] = jid

    # =====================================================================
    # Phase 5: Completion attacks on DP models (PMC members — re-memorization test)
    # =====================================================================
    print("\n--- Phase 5: Completion attacks on DP models ---")

    for eps_str in ["eps1", "eps8", "epsinf"]:
        tag = f"compl_{model_key}_dp_{eps_str}"
        dp_out = f"{OUT}/dp_{eps_str}_mts"
        compl_config = {
            "base_model": f"{rmu_out}/step_final",
            "adapter_path": f"{dp_out}/final",
            "splits_path": PMC_SPLITS,
            "output_dir": f"{REPO}/outputs/attacks/{tag}",
            "prefix_ratio": 0.5,
            "max_length": 512,
            "max_examples": 500,
        }
        config_path = f"{REPO}/config/{tag}.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(compl_config, f)

        sbatch(
            sbatch_header(tag, time="01:00:00") +
            f'{PYTHON} -m attacks.completion --config "{config_path}"\n',
            tag,
            dependency=dp_jobs.get(eps_str),
        )


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
Clinical Contamination Pipeline (per model):
  Phase 1: Completion attack on PMC members → confirm memorization
           + control on MTSamples members → confirm NOT memorized
  Phase 2: RMU unlearn PMC training data (chained after Phase 1)
  Phase 3: Completion attack post-RMU → verify EMR dropped
  Phase 4: DP-LoRA retune on MTSamples eps=1,8,inf (parallel)
  Phase 5: Completion attacks → verify no re-memorization of PMC

Expected:
  Phase 1: PMC EMR >> MTS EMR (clinical text memorized in these models)
  Phase 3: EMR drops after RMU
  Phase 5: DP eps=1,8 keeps EMR low; eps=inf may leak
""")

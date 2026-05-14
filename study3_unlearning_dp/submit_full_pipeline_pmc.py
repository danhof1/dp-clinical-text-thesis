#!/usr/bin/env python3
"""
Full pipeline comparison: RMU vs ReGLU — retuning on PMC (same data we unlearned).

This is the stronger test: does DP prevent re-memorization when fine-tuning
on the SAME data that was erased? Direct parallel to the Wikipedia experiment.

For each method:
  1. Unlearn PMC memorization (save merged checkpoint)
  2. DP-LoRA eps=8 on PMC (the erased data — the hard test)
  3. Completion attack (does memorization return?)
  4. Generation (50 samples)
  5. Fidelity eval
"""
import subprocess, sys, yaml
from pathlib import Path

REPO   = "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp"
PYTHON = "/fs1/home/h702839428/python_daniel/bin/python"
LOGS   = "/fs1/projects/unlearning_pretraining/Proj_code/logs"
ACCT   = "unlearning_pretraining"
BASE   = "/fs1/shared/model/llm/BioMistral-7B"
PMC_SPLITS = f"{REPO}/outputs/splits_v1_pmc"
OUT    = f"{REPO}/outputs/clinical_contam/biomistral"
CACHE_DIR = "/fs1/projects/unlearning_pretraining/Proj_code/.cache"

submitted = []
failed = []


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


def header(tag, time="02:00:00", mem="80G"):
    return (
        f"#!/bin/bash\n"
        f"#SBATCH --job-name={tag}\n"
        f"#SBATCH --account={ACCT}\n"
        f"#SBATCH --partition=defq\n"
        f"#SBATCH --qos=long\n"
        f"#SBATCH --gres=gpu:1\n"
        f"#SBATCH --cpus-per-task=8\n"
        f"#SBATCH --mem={mem}\n"
        f"#SBATCH --time={time}\n"
        f"#SBATCH --output={LOGS}/{tag}_%j.log\n"
        f"\n"
        f"export HF_HOME={CACHE_DIR}\n"
        f"export HF_DATASETS_CACHE={CACHE_DIR}/datasets\n"
        f"export TRANSFORMERS_CACHE={CACHE_DIR}/transformers\n"
        f"export PYTHONPATH=\"{REPO}:$PYTHONPATH\"\n"
        f"\n"
        f'cd "{REPO}"\n'
    )


# ============================================================
# Pipeline A: RMU alpha=600 s=50, DP retune on PMC
# ============================================================
print("=" * 60)
print("Pipeline A: RMU alpha=600 s=50 -> DP-LoRA eps=8 on PMC")
print("=" * 60)

# Re-use existing merged checkpoint if it exists, otherwise re-run
tag_rmu = "pmc_rmu_a600s50"
rmu_out = f"{OUT}/pipe_rmu_a600s50"

RMU_RUNNER = f'''
import time, os
from pathlib import Path

out_dir = Path("{rmu_out}/step_final")
if out_dir.exists() and (out_dir / "config.json").exists():
    print(f"Reusing existing RMU checkpoint at {{out_dir}}", flush=True)
else:
    from unlearn.rmu import RMUConfig, run as rmu_run
    cfg = RMUConfig(
        base_model="{BASE}",
        splits_path="{PMC_SPLITS}",
        output_dir="{rmu_out}",
        layer_id=20,
        alpha=600,
        steering_coeff=1.0,
        adaptive_steering=True,
        num_steps=50,
        learning_rate=5e-5,
        batch_size=4,
        max_seq_length=512,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bf16=True,
        seed=42,
        log_every=10,
        save_merged=True,
    )
    t0 = time.time()
    rmu_run(cfg)
    print(f"RMU done in {{time.time()-t0:.0f}}s", flush=True)
'''

rmu_runner_path = f"{REPO}/config/pipeline_runners/{tag_rmu}_runner.py"
Path(rmu_runner_path).parent.mkdir(parents=True, exist_ok=True)
with open(rmu_runner_path, "w") as f:
    f.write(RMU_RUNNER)

print("\n--- Step 1: RMU unlearning ---")
rmu_jid = sbatch(
    header(tag_rmu, time="00:30:00", mem="80G")
    + f'{PYTHON} "{rmu_runner_path}"\n',
    tag_rmu,
)

# Step 2: DP-LoRA eps=8 on PMC
tag_dp_rmu = "pmc_dp_rmu"
dp_rmu_out = f"{OUT}/pmc_dp_rmu_eps8"

dp_rmu_config = {
    "base_model": f"{rmu_out}/step_final",
    "splits_path": PMC_SPLITS,
    "output_dir": dp_rmu_out,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "num_epochs": 3,
    "physical_batch_size": 2,
    "logical_batch_size": 64,
    "learning_rate": 0.0005,
    "max_seq_length": 512,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "epsilon": 8.0,
    "delta": 0.00001,
    "secure_mode": False,
    "bf16": True,
    "seed": 42,
    "log_every": 10,
}
dp_rmu_cfg_path = f"{REPO}/config/{tag_dp_rmu}.yaml"
with open(dp_rmu_cfg_path, "w") as f:
    yaml.safe_dump(dp_rmu_config, f)

print("\n--- Step 2: DP-LoRA eps=8 on PMC (RMU) ---")
dp_rmu_jid = sbatch(
    header(tag_dp_rmu, time="04:00:00", mem="64G")
    + f'{PYTHON} -m finetune.dp_lora --config "{dp_rmu_cfg_path}"\n',
    tag_dp_rmu,
    dependency=rmu_jid,
)

# Step 3: Completion attack
tag_atk_rmu = "pmc_catk_rmu"
atk_rmu_out = f"{REPO}/outputs/attacks/pmc_compl_rmu_dp"
atk_rmu_config = {
    "base_model": f"{rmu_out}/step_final",
    "adapter_path": f"{dp_rmu_out}/final",
    "splits_path": PMC_SPLITS,
    "output_dir": atk_rmu_out,
    "prefix_ratio": 0.5,
    "max_length": 512,
    "max_examples": 500,
}
atk_rmu_cfg_path = f"{REPO}/config/{tag_atk_rmu}.yaml"
with open(atk_rmu_cfg_path, "w") as f:
    yaml.safe_dump(atk_rmu_config, f)

print("\n--- Step 3: Completion attack (RMU+DP on PMC) ---")
atk_rmu_jid = sbatch(
    header(tag_atk_rmu, time="01:30:00", mem="48G")
    + f'{PYTHON} -m attacks.completion --config "{atk_rmu_cfg_path}"\n',
    tag_atk_rmu,
    dependency=dp_rmu_jid,
)

# Step 4: Generation
tag_gen_rmu = "pmc_gen_rmu"
gen_rmu_out = f"{REPO}/outputs/generated/pmc_rmu_a600s50_eps8.jsonl"

print("\n--- Step 4: Generation (RMU+DP on PMC) ---")
gen_rmu_jid = sbatch(
    header(tag_gen_rmu, time="02:00:00", mem="48G")
    + f'{PYTHON} -m generate.run'
    + f' --base_model "{rmu_out}/step_final"'
    + f' --adapter_path "{dp_rmu_out}/final"'
    + f' --n_samples 50'
    + f' --output_path "{gen_rmu_out}"\n',
    tag_gen_rmu,
    dependency=dp_rmu_jid,
)

# Step 5: Fidelity eval
tag_fid_rmu = "pmc_fid_rmu"
print("\n--- Step 5: Fidelity eval (RMU+DP on PMC) ---")
sbatch(
    header(tag_fid_rmu, time="01:00:00", mem="48G")
    + f'{PYTHON} scripts/eval_fidelity.py'
    + f' --tag pmc_rmu_a600s50_eps8'
    + f' --generated "{gen_rmu_out}"'
    + f' --dataset pmc\n',
    tag_fid_rmu,
    dependency=gen_rmu_jid,
)


# ============================================================
# Pipeline B: ReGLU beta=0.5, DP retune on PMC
# ============================================================
print("\n" + "=" * 60)
print("Pipeline B: ReGLU beta=0.5 -> DP-LoRA eps=8 on PMC")
print("=" * 60)

tag_reglu = "pmc_reglu_b05"
reglu_out = f"{OUT}/pipe_reglu_b05"

REGLU_RUNNER = f'''
import time
from pathlib import Path

out_dir = Path("{reglu_out}/step_final")
if out_dir.exists() and (out_dir / "config.json").exists():
    print(f"Reusing existing ReGLU checkpoint at {{out_dir}}", flush=True)
else:
    from unlearn.reglu import ReGLUConfig, run as reglu_run
    cfg = ReGLUConfig(
        base_model="{BASE}",
        splits_path="{PMC_SPLITS}",
        output_dir="{reglu_out}",
        layer_id=20,
        beta=0.5,
        rol_weight=0.1,
        n_retain_pcs=64,
        n_repr_batches=10,
        forget_weight=1.0,
        retain_weight=1.0,
        num_steps=150,
        learning_rate=5e-5,
        batch_size=4,
        max_seq_length=512,
        lora_r=16,
        lora_alpha=32,
        bf16=True,
        seed=42,
        log_every=10,
        save_merged=True,
    )
    t0 = time.time()
    reglu_run(cfg)
    print(f"ReGLU done in {{time.time()-t0:.0f}}s", flush=True)
'''

reglu_runner_path = f"{REPO}/config/pipeline_runners/{tag_reglu}_runner.py"
with open(reglu_runner_path, "w") as f:
    f.write(REGLU_RUNNER)

print("\n--- Step 1: ReGLU unlearning ---")
reglu_jid = sbatch(
    header(tag_reglu, time="00:30:00", mem="80G")
    + f'{PYTHON} "{reglu_runner_path}"\n',
    tag_reglu,
)

# Step 2: DP-LoRA eps=8 on PMC
tag_dp_reglu = "pmc_dp_reglu"
dp_reglu_out = f"{OUT}/pmc_dp_reglu_eps8"

dp_reglu_config = {
    "base_model": f"{reglu_out}/step_final",
    "splits_path": PMC_SPLITS,
    "output_dir": dp_reglu_out,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "num_epochs": 3,
    "physical_batch_size": 2,
    "logical_batch_size": 64,
    "learning_rate": 0.0005,
    "max_seq_length": 512,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "epsilon": 8.0,
    "delta": 0.00001,
    "secure_mode": False,
    "bf16": True,
    "seed": 42,
    "log_every": 10,
}
dp_reglu_cfg_path = f"{REPO}/config/{tag_dp_reglu}.yaml"
with open(dp_reglu_cfg_path, "w") as f:
    yaml.safe_dump(dp_reglu_config, f)

print("\n--- Step 2: DP-LoRA eps=8 on PMC (ReGLU) ---")
dp_reglu_jid = sbatch(
    header(tag_dp_reglu, time="04:00:00", mem="64G")
    + f'{PYTHON} -m finetune.dp_lora --config "{dp_reglu_cfg_path}"\n',
    tag_dp_reglu,
    dependency=reglu_jid,
)

# Step 3: Completion attack
tag_atk_reglu = "pmc_catk_reglu"
atk_reglu_out = f"{REPO}/outputs/attacks/pmc_compl_reglu_dp"
atk_reglu_config = {
    "base_model": f"{reglu_out}/step_final",
    "adapter_path": f"{dp_reglu_out}/final",
    "splits_path": PMC_SPLITS,
    "output_dir": atk_reglu_out,
    "prefix_ratio": 0.5,
    "max_length": 512,
    "max_examples": 500,
}
atk_reglu_cfg_path = f"{REPO}/config/{tag_atk_reglu}.yaml"
with open(atk_reglu_cfg_path, "w") as f:
    yaml.safe_dump(atk_reglu_config, f)

print("\n--- Step 3: Completion attack (ReGLU+DP on PMC) ---")
atk_reglu_jid = sbatch(
    header(tag_atk_reglu, time="01:30:00", mem="48G")
    + f'{PYTHON} -m attacks.completion --config "{atk_reglu_cfg_path}"\n',
    tag_atk_reglu,
    dependency=dp_reglu_jid,
)

# Step 4: Generation
tag_gen_reglu = "pmc_gen_reglu"
gen_reglu_out = f"{REPO}/outputs/generated/pmc_reglu_b05_eps8.jsonl"

print("\n--- Step 4: Generation (ReGLU+DP on PMC) ---")
gen_reglu_jid = sbatch(
    header(tag_gen_reglu, time="02:00:00", mem="48G")
    + f'{PYTHON} -m generate.run'
    + f' --base_model "{reglu_out}/step_final"'
    + f' --adapter_path "{dp_reglu_out}/final"'
    + f' --n_samples 50'
    + f' --output_path "{gen_reglu_out}"\n',
    tag_gen_reglu,
    dependency=dp_reglu_jid,
)

# Step 5: Fidelity eval
tag_fid_reglu = "pmc_fid_reglu"
print("\n--- Step 5: Fidelity eval (ReGLU+DP on PMC) ---")
sbatch(
    header(tag_fid_reglu, time="01:00:00", mem="48G")
    + f'{PYTHON} scripts/eval_fidelity.py'
    + f' --tag pmc_reglu_b05_eps8'
    + f' --generated "{gen_reglu_out}"'
    + f' --dataset pmc\n',
    tag_fid_reglu,
    dependency=gen_reglu_jid,
)


# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 60}")
print(f"Submitted {len(submitted)}/{len(submitted)+len(failed)} jobs")
for t, jid in submitted:
    print(f"  {jid:>8}  {t}")
if failed:
    print("Failed:")
    for t, err in failed:
        print(f"  {t}: {err}")

print(f"""
Both pipelines retune on PMC (the SAME data that was unlearned).
This is the hard test: does DP prevent re-memorization of erased content?
Direct parallel to the Wikipedia experiment design.

Pipeline A: RMU alpha=600 s=50 -> DP-LoRA eps=8 on PMC -> attack + gen + fidelity
Pipeline B: ReGLU beta=0.5     -> DP-LoRA eps=8 on PMC -> attack + gen + fidelity
""")

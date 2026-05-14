#!/usr/bin/env python3
"""
PsyTAR SynBench Replication Pipeline.

Replicates SynBench's finding that pre-training contamination leaks through
synthetic data even under DP-SGD, using Llama-3.1-8B + PsyTAR.

Pipeline:
  Phase 0: Build PsyTAR splits (members/nonmembers/auxiliary)
  Phase 1: Extraction test on baseline Llama (confirm contamination)
  Phase 2: DP-LoRA fine-tune on PsyTAR members at eps=1, 8, inf
  Phase 3: Generate 500 synthetic drug reviews per config
  Phase 4: SynBench n-gram MIA on each config
"""
import subprocess
import sys
import yaml
from pathlib import Path

REPO    = "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp"
PYTHON  = "/fs1/home/h702839428/python_daniel/bin/python"
LOGS    = "/fs1/projects/unlearning_pretraining/Proj_code/logs"
ACCT    = "unlearning_pretraining"

BASE_MODEL    = "/fs1/shared/model/llm/Llama-3.1-8B-Instruct"
PSYTAR_SPLITS = f"{REPO}/outputs/splits_psytar"
GEN_DIR       = f"{REPO}/outputs/generated"
ATK_DIR       = f"{REPO}/outputs/attacks"

N_SAMPLES = 500

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


def gpu_header(tag, time="02:00:00", mem="48G"):
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


def cpu_header(tag, time="00:30:00", mem="16G"):
    return f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --account={ACCT}
#SBATCH --partition=defq
#SBATCH --qos=long
#SBATCH --cpus-per-task=4
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={LOGS}/{tag}_%j.log

cd "{REPO}"
"""


# =========================================================================
# Phase 0: Build PsyTAR splits
# =========================================================================
print("--- Phase 0: Build PsyTAR splits ---")
splits_jid = sbatch(
    cpu_header("psytar_splits", time="00:10:00", mem="8G") +
    f'{PYTHON} "{REPO}/build_psytar_splits.py"\n',
    "psytar_splits",
)


# =========================================================================
# Phase 1: Extraction test — confirm Llama memorized PsyTAR
# =========================================================================
print("\n--- Phase 1: Extraction test on baseline Llama ---")
compl_jid = sbatch(
    gpu_header("compl_psytar_base", time="01:00:00") +
    f"""{PYTHON} -m attacks.completion \\
  --base_model "{BASE_MODEL}" \\
  --splits_path "{PSYTAR_SPLITS}" \\
  --output_dir "{ATK_DIR}/compl_psytar_baseline" \\
  --max_samples 200 \\
  --prefix_ratio 0.5 \\
  --max_new_tokens 256
""",
    "compl_psytar_base",
    dependency=splits_jid,
)


# =========================================================================
# Phase 2: DP-LoRA fine-tune on PsyTAR members
# =========================================================================
print("\n--- Phase 2: DP-LoRA fine-tune on PsyTAR members ---")

Path(f"{REPO}/config").mkdir(parents=True, exist_ok=True)

dp_jobs = {}
for eps_val, eps_str in [(1.0, "eps1"), (8.0, "eps8"), (None, "epsinf")]:
    dp_tag = f"dp_psytar_{eps_str}"
    dp_out = f"{REPO}/outputs/dp_lora_{eps_str}_psytar"

    dp_config = {
        "base_model": BASE_MODEL,
        "splits_path": PSYTAR_SPLITS,
        "output_dir": dp_out,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
        "num_epochs": 5,
        "physical_batch_size": 4,
        "logical_batch_size": 32,
        "learning_rate": 0.0005,
        "max_seq_length": 512,
        "max_grad_norm": 1.0,
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "epsilon": eps_val,
        "delta": 0.00001,
        "secure_mode": False,
        "bf16": True,
        "seed": 42,
        "log_every": 10,
    }
    cfg_path = f"{REPO}/config/dp_psytar_{eps_str}.yaml"
    Path(cfg_path).write_text(yaml.safe_dump(dp_config))

    jid = sbatch(
        gpu_header(dp_tag, time="02:00:00", mem="64G") +
        f'{PYTHON} -m finetune.dp_lora --config "{cfg_path}"\n',
        dp_tag,
        dependency=splits_jid,
    )
    dp_jobs[eps_str] = jid


# =========================================================================
# Phase 3: Generate synthetic drug reviews
# =========================================================================
print("\n--- Phase 3: Generate synthetic drug reviews ---")

gen_jobs = {}

# Baseline generation (vanilla Llama, no fine-tuning)
gen_tag = "gen_psytar_base"
gen_jid = sbatch(
    gpu_header(gen_tag, time="02:00:00") +
    f"""{PYTHON} "{REPO}/generate_psytar.py" \\
  --base_model "{BASE_MODEL}" \\
  --n_samples {N_SAMPLES} \\
  --output_path "{GEN_DIR}/psytar_baseline.jsonl" \\
  --k 1 --temperature 0.9 --top_p 0.95
""",
    gen_tag,
    dependency=splits_jid,
)
gen_jobs["baseline"] = gen_jid

# DP-tuned generations
for eps_str in ["eps1", "eps8", "epsinf"]:
    dp_out = f"{REPO}/outputs/dp_lora_{eps_str}_psytar"
    gen_tag = f"gen_psytar_{eps_str}"
    gen_jid = sbatch(
        gpu_header(gen_tag, time="02:00:00") +
        f"""{PYTHON} "{REPO}/generate_psytar.py" \\
  --base_model "{BASE_MODEL}" \\
  --adapter_path "{dp_out}/final" \\
  --n_samples {N_SAMPLES} \\
  --output_path "{GEN_DIR}/psytar_{eps_str}.jsonl" \\
  --k 1 --temperature 0.9 --top_p 0.95
""",
        gen_tag,
        dependency=dp_jobs.get(eps_str),
    )
    gen_jobs[eps_str] = gen_jid


# =========================================================================
# Phase 4: SynBench n-gram MIA
# =========================================================================
print("\n--- Phase 4: SynBench MIA ---")

for name, syn_path, dep_jid in [
    ("baseline", f"{GEN_DIR}/psytar_baseline.jsonl", gen_jobs.get("baseline")),
    ("eps1",     f"{GEN_DIR}/psytar_eps1.jsonl",     gen_jobs.get("eps1")),
    ("eps8",     f"{GEN_DIR}/psytar_eps8.jsonl",     gen_jobs.get("eps8")),
    ("epsinf",   f"{GEN_DIR}/psytar_epsinf.jsonl",   gen_jobs.get("epsinf")),
]:
    mia_tag = f"mia_psytar_{name}"
    sbatch(
        cpu_header(mia_tag, time="00:30:00", mem="16G") +
        f"""{PYTHON} "{REPO}/synbench_mia.py" \\
  --synthetic_path "{syn_path}" \\
  --splits_path "{PSYTAR_SPLITS}" \\
  --output_path "{ATK_DIR}/synbench_psytar_{name}/synbench_results.json" \\
  --n_gram 3 \\
  --auxiliary_split auxiliary
""",
        mia_tag,
        dependency=dep_jid,
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

print(f"""
PsyTAR SynBench Replication Pipeline
=====================================
Phase 0: Build splits (350 members, 200 nonmembers, ~103 auxiliary)
Phase 1: Extraction test on baseline Llama-3.1-8B
Phase 2: DP-LoRA fine-tune at eps=1, 8, inf (5 epochs, batch=32, lr=5e-4)
Phase 3: Generate {N_SAMPLES} synthetic drug reviews per config
Phase 4: SynBench 3-gram delta-P MIA per config

Three-point contamination spectrum:
  Llama + MIMIC   -> no contamination    -> AUC ~0.50 (Track A baseline)
  BioMistral + PMC -> distributional      -> AUC ~0.49 (already done)
  Llama + PsyTAR  -> record-level (this) -> AUC > 0.50 (expected)
""")

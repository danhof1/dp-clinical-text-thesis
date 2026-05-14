#!/usr/bin/env python3
"""
BioMistral SynBench MIA Pipeline.

For each config (baseline, post-RMU, RMU+DP eps=1,8,inf):
  1. Generate 500 synthetic notes
  2. Run SynBench n-gram ΔP MIA

Tests whether pre-training contamination leaks through synthetic data
even when DP-SGD is applied, replicating SynBench's methodology.
"""
import subprocess
from pathlib import Path

REPO   = "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp"
PYTHON = "/fs1/home/h702839428/python_daniel/bin/python"
LOGS   = "/fs1/projects/unlearning_pretraining/Proj_code/logs"
ACCT   = "unlearning_pretraining"

BASE_MODEL = "/fs1/shared/model/llm/BioMistral-7B"
RMU_CKPT   = f"{REPO}/outputs/clinical_contam/biomistral/unlearn_rmu/step_final"
PMC_SPLITS = f"{REPO}/outputs/splits_v1_pmc"
GEN_DIR    = f"{REPO}/outputs/generated"
ATK_DIR    = f"{REPO}/outputs/attacks"
MIA_SCRIPT = f"{REPO}/synbench_mia.py"

N_SAMPLES = 500

submitted = []


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
        dep_str = f" (after {dependency})" if dependency else ""
        print(f"  {tag} -> job {jid}{dep_str}")
        return jid
    else:
        print(f"  {tag} FAILED: {r.stderr.strip()}")
        return None


def header(tag, time="02:00:00", mem="48G"):
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


configs = [
    {
        "name": "bio_baseline",
        "base_model": BASE_MODEL,
        "adapter_path": None,
        "desc": "Baseline BioMistral (contaminated, no intervention)",
    },
    {
        "name": "bio_post_rmu",
        "base_model": RMU_CKPT,
        "adapter_path": None,
        "desc": "Post-RMU (contamination erased)",
    },
    {
        "name": "bio_rmu_dp_eps1",
        "base_model": RMU_CKPT,
        "adapter_path": f"{REPO}/outputs/clinical_contam/biomistral/dp_eps1_pmc/final",
        "desc": "RMU + DP eps=1 on PMC",
    },
    {
        "name": "bio_rmu_dp_eps8",
        "base_model": RMU_CKPT,
        "adapter_path": f"{REPO}/outputs/clinical_contam/biomistral/dp_eps8_pmc/final",
        "desc": "RMU + DP eps=8 on PMC",
    },
    {
        "name": "bio_rmu_dp_epsinf",
        "base_model": RMU_CKPT,
        "adapter_path": f"{REPO}/outputs/clinical_contam/biomistral/dp_epsinf_pmc/final",
        "desc": "RMU + DP eps=inf on PMC (no privacy)",
    },
]

for cfg in configs:
    name = cfg["name"]
    gen_path = f"{GEN_DIR}/synbench_{name}.jsonl"
    atk_path = f"{ATK_DIR}/synbench_{name}/synbench_results.json"

    adapter_arg = f'--adapter_path "{cfg["adapter_path"]}"' if cfg["adapter_path"] else ""

    # Generation job (GPU)
    gen_tag = f"sbgen_{name}"
    gen_jid = sbatch(
        header(gen_tag, time="03:00:00", mem="48G") +
        f"""{PYTHON} -m generate.run \\
  --base_model "{cfg['base_model']}" \\
  {adapter_arg} \\
  --n_samples {N_SAMPLES} \\
  --output_path "{gen_path}" \\
  --k 1 \\
  --temperature 0.9 \\
  --top_p 0.95
""",
        gen_tag,
    )

    # SynBench MIA job (CPU-only, n-gram computation)
    mia_tag = f"sbmia_{name}"
    sbatch(
        cpu_header(mia_tag) +
        f"""{PYTHON} "{MIA_SCRIPT}" \\
  --synthetic_path "{gen_path}" \\
  --splits_path "{PMC_SPLITS}" \\
  --output_path "{atk_path}" \\
  --n_gram 3
""",
        mia_tag,
        dependency=gen_jid,
    )

print(f"\nSubmitted {len(submitted)} jobs:")
for tag, jid in submitted:
    print(f"  {jid:>8}  {tag}")

print(f"""
Pipeline per config:
  1. Generate {N_SAMPLES} synthetic notes (GPU, k=1 for speed)
  2. SynBench n-gram ΔP MIA (CPU, 3-gram)

Configs:
  baseline     — contaminated BioMistral, no intervention
  post_rmu     — after RMU erases PMC knowledge
  rmu+dp eps=1 — RMU + DP retune on PMC (strong privacy)
  rmu+dp eps=8 — RMU + DP retune on PMC (moderate privacy)
  rmu+dp inf   — RMU + retune on PMC (no privacy guarantee)

Expected:
  baseline: elevated AUC (synthetic data reflects memorized PMC)
  post_rmu: AUC ~ 0.5 (PMC knowledge erased)
  eps=1,8:  AUC ~ 0.5 (DP prevents leakage into synthetic data)
  eps=inf:  possibly elevated (no DP protection)
""")

#!/usr/bin/env python3
"""
Submit corrected generation jobs using proper clinical note prompts.

Fix: old prompt was "Medical Specialty: X\nDescription: " which produced
job postings. New prompt uses actual training data openings
(CHIEF COMPLAINT:, PREOPERATIVE DIAGNOSIS:, etc.)

Configs to regenerate:
  1. BioMistral baseline (no unlearn, no DP) — reference quality
  2. BioMistral + RMU + DP eps=8 on MTSamples
  3. BioMistral + ReGLU + DP eps=8 on MTSamples
  4. BioMistral + RMU + DP eps=8 on PMC
  5. BioMistral + ReGLU + DP eps=8 on PMC
  6. Llama-3.1-8B + RMU + DP eps=8 on MTSamples (large-scale)
  7. Llama-3.1-8B + RMU + DP eps=1 on MTSamples
  8. Llama-3.1-8B baseline (no adapter) — reference
"""
import subprocess, sys
from pathlib import Path

REPO   = "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp"
PYTHON = "/fs1/home/h702839428/python_daniel/bin/python"
LOGS   = "/fs1/projects/unlearning_pretraining/Proj_code/logs"
ACCT   = "unlearning_pretraining"
CACHE_DIR = "/fs1/projects/unlearning_pretraining/Proj_code/.cache"

# Base models
BIOMISTRAL = "/fs1/shared/model/llm/BioMistral-7B"
LLAMA8B    = "/fs1/shared/model/llm/Meta-Llama-3.1-8B-Instruct"

# Unlearned checkpoints (merged, full models)
BIOM_RMU   = f"{REPO}/outputs/clinical_contam/biomistral/pipe_rmu_a600s50/step_final"
BIOM_REGLU = f"{REPO}/outputs/clinical_contam/biomistral/pipe_reglu_b05/step_final"
LLAMA_RMU  = f"{REPO}/outputs/unlearn_rmu/step_final"

# DP-LoRA adapters
BIOM_RMU_DP_MTS   = f"{REPO}/outputs/clinical_contam/biomistral/pipe_dp_rmu_eps8/final"
BIOM_REGLU_DP_MTS = f"{REPO}/outputs/clinical_contam/biomistral/pipe_dp_reglu_eps8/final"
BIOM_RMU_DP_PMC   = f"{REPO}/outputs/clinical_contam/biomistral/pmc_dp_rmu_eps8/final"
BIOM_REGLU_DP_PMC = f"{REPO}/outputs/clinical_contam/biomistral/pmc_dp_reglu_eps8/final"
LLAMA_RMU_DP_E8   = f"{REPO}/outputs/dp_lora_eps8_rmu/final"
LLAMA_RMU_DP_E1   = f"{REPO}/outputs/dp_lora_eps1_rmu/final"

GEN_OUT = f"{REPO}/outputs/generated/v2"

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


def header(tag, time="03:00:00", mem="64G"):
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


def gen_cmd(base_model, adapter_path, dataset, n_samples, output_path):
    cmd = (
        f'{PYTHON} -m generate.run_v2'
        f' --base_model "{base_model}"'
        f' --dataset {dataset}'
        f' --n_samples {n_samples}'
        f' --output_path "{output_path}"'
    )
    if adapter_path:
        cmd += f' --adapter_path "{adapter_path}"'
    return cmd


configs = [
    # tag, base_model, adapter_path, dataset, n_samples, time, mem
    ("v2_biom_baseline",       BIOMISTRAL,  None,                "mtsamples", 200, "03:00:00", "48G"),
    ("v2_biom_rmu_dp_mts",     BIOM_RMU,    BIOM_RMU_DP_MTS,    "mtsamples", 200, "03:00:00", "48G"),
    ("v2_biom_reglu_dp_mts",   BIOM_REGLU,  BIOM_REGLU_DP_MTS,  "mtsamples", 200, "03:00:00", "48G"),
    ("v2_biom_rmu_dp_pmc",     BIOM_RMU,    BIOM_RMU_DP_PMC,    "pmc",       200, "03:00:00", "48G"),
    ("v2_biom_reglu_dp_pmc",   BIOM_REGLU,  BIOM_REGLU_DP_PMC,  "pmc",       200, "03:00:00", "48G"),
    ("v2_llama_baseline",      LLAMA8B,     None,                "mtsamples", 200, "04:00:00", "64G"),
    ("v2_llama_rmu_dp_e8",     LLAMA_RMU,   LLAMA_RMU_DP_E8,    "mtsamples", 200, "04:00:00", "64G"),
    ("v2_llama_rmu_dp_e1",     LLAMA_RMU,   LLAMA_RMU_DP_E1,    "mtsamples", 200, "04:00:00", "64G"),
]

print("=" * 60)
print("Corrected Generation v2 — clinical note prompts")
print("=" * 60)

for tag, base, adapter, dataset, n, time, mem in configs:
    out_path = f"{GEN_OUT}/{tag}.jsonl"
    cmd = gen_cmd(base, adapter, dataset, n, out_path)
    print(f"\n--- {tag} ---")
    sbatch(
        header(tag, time=time, mem=mem) + cmd + "\n",
        tag,
    )

print(f"\n{'=' * 60}")
print(f"Submitted {len(submitted)}/{len(submitted)+len(failed)} jobs")
for t, jid in submitted:
    print(f"  {jid:>8}  {t}")
if failed:
    print("Failed:")
    for t, err in failed:
        print(f"  {t}: {err}")

print(f"""
All outputs go to: {GEN_OUT}/
200 samples each (4 candidates per sample, PPL-selected)
Prompts match actual training data format (section headers for MTS, narrative for PMC)
""")

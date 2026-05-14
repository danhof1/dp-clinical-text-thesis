#!/usr/bin/env python3
"""
Submit v3 generation jobs — instruction-formatted for Llama, base for BioMistral.
Also re-runs BioMistral ReGLU with quality classification.

v3 adds:
  - Chat template prompting for instruction-tuned models
  - Quality classifier on each output (clinical_note / exam_question / job_posting / garbled / other)
  - Specialty-conditioned prompts
"""
import subprocess, sys
from pathlib import Path

REPO   = "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp"
PYTHON = "/fs1/home/h702839428/python_daniel/bin/python"
LOGS   = "/fs1/projects/unlearning_pretraining/Proj_code/logs"
ACCT   = "unlearning_pretraining"
CACHE_DIR = "/fs1/projects/unlearning_pretraining/Proj_code/.cache"

BIOMISTRAL = "/fs1/shared/model/llm/BioMistral-7B"
LLAMA8B    = "/fs1/shared/model/llm/Llama-3.1-8B-Instruct"

BIOM_RMU   = f"{REPO}/outputs/clinical_contam/biomistral/pipe_rmu_a600s50/step_final"
BIOM_REGLU = f"{REPO}/outputs/clinical_contam/biomistral/pipe_reglu_b05/step_final"
LLAMA_RMU  = f"{REPO}/outputs/unlearn_rmu/step_final"

BIOM_RMU_DP_MTS   = f"{REPO}/outputs/clinical_contam/biomistral/pipe_dp_rmu_eps8/final"
BIOM_REGLU_DP_MTS = f"{REPO}/outputs/clinical_contam/biomistral/pipe_dp_reglu_eps8/final"
LLAMA_RMU_DP_E8   = f"{REPO}/outputs/dp_lora_eps8_rmu/final"
LLAMA_RMU_DP_E1   = f"{REPO}/outputs/dp_lora_eps1_rmu/final"

GEN_OUT = f"{REPO}/outputs/generated/v3"

submitted = []
failed = []


def sbatch(script, tag):
    p = Path("/tmp") / f"{tag}.sbatch"
    p.write_text(script)
    r = subprocess.run(["sbatch", str(p)], capture_output=True, text=True)
    if r.returncode == 0:
        jid = r.stdout.strip().split()[-1]
        submitted.append((tag, jid))
        print(f"  {tag} -> job {jid}")
        return jid
    else:
        failed.append((tag, r.stderr.strip()))
        print(f"  {tag} FAILED: {r.stderr.strip()}", file=sys.stderr)
        return None


def header(tag, time="04:00:00", mem="64G"):
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


def gen_cmd(base_model, adapter_path, dataset, model_type, n_samples, output_path):
    cmd = (
        f'{PYTHON} -m generate.run_v3'
        f' --base_model "{base_model}"'
        f' --dataset {dataset}'
        f' --model_type {model_type}'
        f' --n_samples {n_samples}'
        f' --output_path "{output_path}"'
    )
    if adapter_path:
        cmd += f' --adapter_path "{adapter_path}"'
    return cmd


configs = [
    # tag, base_model, adapter, dataset, model_type, n_samples, time, mem
    # Llama instruct runs (chat template)
    ("v3_llama_rmu_dp_e8", LLAMA_RMU,  LLAMA_RMU_DP_E8, "mtsamples", "instruct", 200, "04:00:00", "64G"),
    ("v3_llama_rmu_dp_e1", LLAMA_RMU,  LLAMA_RMU_DP_E1, "mtsamples", "instruct", 200, "04:00:00", "64G"),
    ("v3_llama_baseline",  LLAMA8B,    None,             "mtsamples", "instruct", 200, "04:00:00", "64G"),
    # BioMistral base runs (raw completion) with quality classification
    ("v3_biom_reglu_mts",  BIOM_REGLU, BIOM_REGLU_DP_MTS, "mtsamples", "base", 200, "03:00:00", "48G"),
    ("v3_biom_rmu_mts",    BIOM_RMU,   BIOM_RMU_DP_MTS,   "mtsamples", "base", 200, "03:00:00", "48G"),
    ("v3_biom_baseline",   BIOMISTRAL, None,               "mtsamples", "base", 200, "03:00:00", "48G"),
]

print("=" * 60)
print("Generation v3 — instruction prompts + quality classification")
print("=" * 60)

for tag, base, adapter, dataset, mtype, n, time, mem in configs:
    out_path = f"{GEN_OUT}/{tag}.jsonl"
    cmd = gen_cmd(base, adapter, dataset, mtype, n, out_path)
    print(f"\n--- {tag} ({mtype}) ---")
    sbatch(header(tag, time=time, mem=mem) + cmd + "\n", tag)

print(f"\n{'=' * 60}")
print(f"Submitted {len(submitted)}/{len(submitted)+len(failed)} jobs")
for t, jid in submitted:
    print(f"  {jid:>8}  {t}")
if failed:
    print("Failed:")
    for t, err in failed:
        print(f"  {t}: {err}")

print(f"""
Outputs: {GEN_OUT}/
Each record includes quality classification: clinical_note / exam_question / job_posting / garbled / other
Llama runs use chat template; BioMistral uses raw completion.
""")

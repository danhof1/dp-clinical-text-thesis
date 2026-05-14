#!/usr/bin/env python3
"""
Resubmit generation + eval for log and effective weighting experiments.

Fixes from original submit:
  1. --output now points to run-specific subdir (was parent dir — all jobs wrote same file)
  2. Generation walltime: 48h (was 4h — needs ~38h at 15s/note for 9178 notes)
  3. epsinf checkpoints use "epsinf" in dirname, not "eps999.0" — hardcode paths

Training is already complete for:
  - All 4 log configs (eps 0.5, 1, 4, inf)
  - eff eps=0.5
  - eff eps=1, eps=4, eps=inf still training (22031, 22034, 22078)
"""
import subprocess
import sys
from pathlib import Path

PROJ    = "/fs1/projects/unlearning_pretraining/Proj_code"
SCRIPTS = f"{PROJ}/Scripts_2.0"
GEN     = f"{SCRIPTS}/generation_scripts/03.2_generate.py"
EVAL    = f"{SCRIPTS}/05_evaluate.py"
PYTHON  = "/fs1/home/h702839428/python_daniel/bin/python"
LOGS    = f"{PROJ}/logs"
ACCT    = "unlearning_pretraining"
MODEL   = "/fs1/shared/model/llm/Llama-3.2-1B-Instruct"
DATA    = f"{PROJ}/data/train.jsonl"
MODELS  = f"{PROJ}/models"
GEN_DIR = f"{PROJ}/generated/mimic"
EVAL_DIR = f"{PROJ}/eval"

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
        dep_str = f" (after {dependency})" if dependency else ""
        print(f"  {tag} -> job {jid}{dep_str}")
        submitted.append((tag, jid))
        return jid
    else:
        print(f"  {tag} FAILED: {r.stderr.strip()}", file=sys.stderr)
        return None


def gpu_header(tag, time="48:00:00", mem="32G"):
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

"""


# Configs where training is DONE — submit gen+eval immediately
ready_configs = [
    ("eps05_log10_mimic",   f"{MODELS}/llama_dp_eps0.5_log10_mimic_20260425_2124/final"),
    ("eps1_log10_mimic",    f"{MODELS}/llama_dp_eps1.0_log10_mimic_20260425_2124/final"),
    ("eps4_log10_mimic",    f"{MODELS}/llama_dp_eps4.0_log10_mimic_20260425_2124/final"),
    ("epsinf_log10_mimic",  f"{MODELS}/llama_dp_epsinf_log10_mimic_20260425_2124/final"),
    ("eps05_eff0999_mimic", f"{MODELS}/llama_dp_eps0.5_eff0999cap10_mimic_20260425_2124/final"),
]

# Configs still training — gen depends on training job
pending_configs = [
    ("eps1_eff0999_mimic",   "22031"),  # train_eps1_eff0999_mimic
    ("eps4_eff0999_mimic",   "22034"),  # train_eps4_eff0999_mimic
    ("epsinf_eff0999_mimic", "22078"),  # train_epsinf_eff0999
]


def submit_gen_eval(run_tag, ckpt_path, train_dep=None):
    gen_out = f"{GEN_DIR}/{run_tag}"
    gen_tag = f"gen_{run_tag}"

    if ckpt_path.startswith("GLOB:"):
        ckpt_block = f"""CKPT=$(ls -td {ckpt_path[5:]} 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint found for {run_tag}"
    exit 1
fi
echo "Using checkpoint: $CKPT"
"""
        ckpt_var = "$CKPT"
    else:
        ckpt_block = f'echo "Using checkpoint: {ckpt_path}"\n'
        ckpt_var = ckpt_path

    gen_jid = sbatch(
        gpu_header(gen_tag, time="48:00:00") +
        ckpt_block +
        f"""{PYTHON} "{GEN}" \\
  --checkpoint "{ckpt_var}" \\
  --base_model "{MODEL}" \\
  --data "{DATA}" \\
  --output "{gen_out}" \\
  --n_generate 9178 \\
  --n_per_category 500 \\
  --k 4 \\
  --temperature 0.1 \\
  --top_p 1.0 \\
  --repetition_penalty 1.2 \\
  --epsilon 999.0
""",
        gen_tag,
        dependency=train_dep,
    )

    eval_tag = f"eval_{run_tag}"
    sbatch(
        gpu_header(eval_tag, time="02:00:00") +
        f"""{PYTHON} "{EVAL}" \\
  --synthetic "{gen_out}/synthetic_bhc.jsonl" \\
  --real "{DATA}" \\
  --output "{EVAL_DIR}" \\
  --epsilon 999.0
""",
        eval_tag,
        dependency=gen_jid,
    )


print("=== Ready configs (training done) ===")
for run_tag, ckpt in ready_configs:
    submit_gen_eval(run_tag, ckpt)

print("\n=== Pending configs (waiting on training) ===")
for run_tag, train_jid in pending_configs:
    glob_pattern = f'"{MODELS}"/llama_dp_eps*_{run_tag.split("_", 1)[1].replace("_mimic", "")}*_mimic_*/final'
    if "epsinf" in run_tag:
        glob_pattern = f'"{MODELS}"/llama_dp_epsinf_eff0999*_mimic_*/final'
    elif "eps1" in run_tag and "eff" in run_tag:
        glob_pattern = f'"{MODELS}"/llama_dp_eps1.0_eff0999*_mimic_*/final'
    elif "eps4" in run_tag and "eff" in run_tag:
        glob_pattern = f'"{MODELS}"/llama_dp_eps4.0_eff0999*_mimic_*/final'
    submit_gen_eval(run_tag, f"GLOB:{glob_pattern}", train_dep=train_jid)

print(f"\nSubmitted {len(submitted)} jobs total")
for tag, jid in submitted:
    print(f"  {jid:>8}  {tag}")

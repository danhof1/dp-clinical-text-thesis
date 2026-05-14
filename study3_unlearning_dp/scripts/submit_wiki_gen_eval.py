#!/usr/bin/env python3
"""Submit generation + fidelity eval for wiki unlearn pipeline checkpoints."""
import subprocess
import sys
from pathlib import Path

REPO   = "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp"
BASE   = "/fs1/shared/model/llm/Llama-3.1-8B-Instruct"
RMU    = f"{REPO}/outputs/unlearn_rmu_wiki/step_final"
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
# 1. Generate from MTS-retuned checkpoints (eps=1,8,inf)
#    These are the checkpoints where RMU erased wiki, then DP-LoRA
#    retrained on MTSamples clinical text.
# =========================================================================
print("=== Generation from wiki-unlearn + MTS-retune checkpoints ===")

gen_jobs = {}

for eps_str in ["eps1", "eps8", "epsinf"]:
    adapter = f"{REPO}/outputs/dp_{eps_str}_wiki_on_mts/final"
    tag = f"gen_wiki_{eps_str}_mts"
    output = f"{REPO}/outputs/generated/{tag}.jsonl"

    jid = sbatch(f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --account={ACCT}
#SBATCH --partition=defq
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output={LOGS}/{tag}_%j.log

cd "{REPO}"
{PYTHON} -m generate.run \\
  --base_model "{RMU}" \\
  --adapter_path "{adapter}" \\
  --n_samples 500 \\
  --output_path "{output}" \\
  --k 4
""", tag)

    gen_jobs[eps_str] = jid


# =========================================================================
# 2. Also generate from post-RMU (no DP retune) as baseline
# =========================================================================
print("\n=== Generation from post-RMU (no retune) ===")

tag = "gen_wiki_post_rmu"
output = f"{REPO}/outputs/generated/{tag}.jsonl"
rmu_gen_jid = sbatch(f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --account={ACCT}
#SBATCH --partition=defq
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output={LOGS}/{tag}_%j.log

cd "{REPO}"
{PYTHON} -m generate.run \\
  --base_model "{RMU}" \\
  --n_samples 500 \\
  --output_path "{output}" \\
  --k 4
""", tag)


# =========================================================================
# 3. Eval suite on wiki pipeline checkpoints (chained after generation)
# =========================================================================
print("\n=== Eval suite on wiki pipeline checkpoints ===")

eval_configs = [
    ("wiki_post_rmu",    RMU, None, rmu_gen_jid),
    ("wiki_eps1_mts",    RMU, f"{REPO}/outputs/dp_eps1_wiki_on_mts/final",   gen_jobs.get("eps1")),
    ("wiki_eps8_mts",    RMU, f"{REPO}/outputs/dp_eps8_wiki_on_mts/final",   gen_jobs.get("eps8")),
    ("wiki_epsinf_mts",  RMU, f"{REPO}/outputs/dp_epsinf_wiki_on_mts/final", gen_jobs.get("epsinf")),
]

for tag, base, adapter, dep in eval_configs:
    eval_out = f"{REPO}/outputs/eval/{tag}"
    gen_path = f"{REPO}/outputs/generated/gen_{tag}.jsonl"

    adapter_arg = f'--adapter_path "{adapter}"' if adapter else ""

    sbatch(f"""#!/bin/bash
#SBATCH --job-name=eval_{tag}
#SBATCH --account={ACCT}
#SBATCH --partition=defq
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output={LOGS}/eval_{tag}_%j.log

cd "{REPO}"
{PYTHON} -m scripts.eval_suite \\
  --base_model "{base}" \\
  {adapter_arg} \\
  --generated_path "{gen_path}" \\
  --output_dir "{eval_out}" \\
  --splits_path "{REPO}/data/pretraining_control/splits_wiki_retune_mts"
""", f"eval_{tag}", dependency=dep)


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
Pipeline:
  4 generation jobs (500 samples each, ~8h, k=4):
    - post-RMU (no retune) — how bad is generation after unlearning?
    - DP eps=1 MTS retune — can DP+clinical text restore generation quality?
    - DP eps=8 MTS retune — sweet spot?
    - DP eps=inf MTS retune — upper bound without privacy

  4 eval suite jobs (chained after generation, ~30 min each):
    - KS test, relearn attack, MMLU medical
""")

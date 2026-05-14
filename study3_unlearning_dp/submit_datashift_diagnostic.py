#!/usr/bin/env python3
"""
Submit the data-shift diagnostic pipeline to the cluster.

Step 1: Paraphrase 100 PMC texts at three aggressiveness levels (light/moderate/aggressive)
Step 2: Run subspace diagnostic measuring projection onto ReGLU's erased directions

Between steps: user eyeballs spot_check_{level}.txt files for clinical fidelity.
Step 2 can be submitted immediately (chained) or held until spot-check is done.

Usage:
  python submit_datashift_diagnostic.py                   # submit both steps chained
  python submit_datashift_diagnostic.py --paraphrase-only # submit step 1 only, hold step 2
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = (
    "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/"
    "New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp"
)
PYTHON = "/fs1/home/h702839428/python_daniel/bin/python"
LOGS = "/fs1/projects/unlearning_pretraining/Proj_code/logs"
ACCT = "unlearning_pretraining"
CACHE_DIR = "/fs1/projects/unlearning_pretraining/Proj_code/.cache"
THESIS_DIR = "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/thesis_scripts"

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


def header(tag, time="06:00:00", mem="80G", gpus=1):
    return (
        f"#!/bin/bash\n"
        f"#SBATCH --job-name={tag}\n"
        f"#SBATCH --account={ACCT}\n"
        f"#SBATCH --partition=defq\n"
        f"#SBATCH --qos=long\n"
        f"#SBATCH --gres=gpu:{gpus}\n"
        f"#SBATCH --cpus-per-task=8\n"
        f"#SBATCH --mem={mem}\n"
        f"#SBATCH --time={time}\n"
        f"#SBATCH --output={LOGS}/{tag}_%j.log\n"
        f"\n"
        f"export HF_HOME={CACHE_DIR}\n"
        f"export HF_DATASETS_CACHE={CACHE_DIR}/datasets\n"
        f"export TRANSFORMERS_CACHE={CACHE_DIR}/transformers\n"
        f"\n"
        f'cd "{REPO}"\n'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paraphrase-only", action="store_true",
        help="Only submit paraphrase jobs; hold diagnostic until spot-check is done",
    )
    parser.add_argument(
        "--n_samples", type=int, default=100,
        help="Number of PMC texts to paraphrase per level",
    )
    parser.add_argument(
        "--diagnostic-only", action="store_true",
        help="Only submit diagnostic (assumes paraphrases already exist)",
    )
    args = parser.parse_args()

    # Copy scripts to cluster-accessible location
    script_dir = Path(THESIS_DIR)
    print(f"Scripts will be read from: {THESIS_DIR}")
    print("Ensure paraphrase_pmc.py and subspace_diagnostic.py are uploaded there.\n")

    para_script = f"{THESIS_DIR}/paraphrase_pmc.py"
    diag_script = f"{THESIS_DIR}/subspace_diagnostic.py"

    # ── Step 1: Paraphrase at all three levels ───────────────────
    para_jids = []
    if not args.diagnostic_only:
        print("=" * 60)
        print("Step 1: Paraphrase PMC at three aggressiveness levels")
        print("=" * 60)

        for level in ["light", "moderate", "aggressive"]:
            tag = f"para_pmc_{level}"
            # Light is fastest (~30s/sample), aggressive slowest (~60s/sample)
            # 100 samples: light ~1h, moderate ~1.5h, aggressive ~2h
            time_limit = {"light": "02:00:00", "moderate": "03:00:00", "aggressive": "04:00:00"}[level]

            jid = sbatch(
                header(tag, time=time_limit, mem="48G")
                + f'{PYTHON} "{para_script}" --level {level} --n_samples {args.n_samples}\n',
                tag,
            )
            if jid:
                para_jids.append(jid)

    # ── Step 2: Subspace diagnostic ──────────────────────────────
    if not args.paraphrase_only:
        print(f"\n{'=' * 60}")
        print("Step 2: Subspace diagnostic (layers 16, 20, 24)")
        print("=" * 60)

        dep = None
        if para_jids:
            dep = "afterok:" + ":".join(para_jids)
            # Override to use simple afterok:last_jid since all three must finish
            dep = None  # Actually need afterok on ALL three

        tag = "subspace_diag"
        dep_str = None
        if para_jids:
            # Chain after ALL paraphrase jobs complete
            dep_script = (
                f"#!/bin/bash\n"
                f"#SBATCH --job-name={tag}\n"
                f"#SBATCH --account={ACCT}\n"
                f"#SBATCH --partition=defq\n"
                f"#SBATCH --qos=long\n"
                f"#SBATCH --gres=gpu:1\n"
                f"#SBATCH --cpus-per-task=8\n"
                f"#SBATCH --mem=80G\n"
                f"#SBATCH --time=01:30:00\n"
                f"#SBATCH --output={LOGS}/{tag}_%j.log\n"
            )
            for jid in para_jids:
                dep_script += f"#SBATCH --dependency=afterok:{jid}\n"
            dep_script += (
                f"\n"
                f"export HF_HOME={CACHE_DIR}\n"
                f"export HF_DATASETS_CACHE={CACHE_DIR}/datasets\n"
                f"export TRANSFORMERS_CACHE={CACHE_DIR}/transformers\n"
                f"\n"
                f'cd "{REPO}"\n'
                f'{PYTHON} "{diag_script}" --layers 16 20 24 --n_eval 100\n'
            )
            p = Path("/tmp") / f"{tag}.sbatch"
            p.write_text(dep_script)
            r = subprocess.run(["sbatch", str(p)], capture_output=True, text=True)
            if r.returncode == 0:
                jid = r.stdout.strip().split()[-1]
                submitted.append((tag, jid))
                print(f"  {tag} -> job {jid} (after all paraphrase jobs)")
            else:
                failed.append((tag, r.stderr.strip()))
                print(f"  {tag} FAILED: {r.stderr.strip()}", file=sys.stderr)
        else:
            sbatch(
                header(tag, time="01:30:00", mem="80G")
                + f'{PYTHON} "{diag_script}" --layers 16 20 24 --n_eval 100\n',
                tag,
            )

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Submitted {len(submitted)}/{len(submitted)+len(failed)} jobs")
    for t, jid in submitted:
        print(f"  {jid:>8}  {t}")
    if failed:
        print("Failed:")
        for t, err in failed:
            print(f"  {t}: {err}")

    para_dir = (
        f"{REPO}/outputs/paraphrased_pmc"
    )
    diag_dir = (
        f"{REPO}/outputs/subspace_diagnostic"
    )
    print(f"""
Outputs:
  Paraphrases:  {para_dir}/paraphrased_{{light,moderate,aggressive}}.jsonl
  Spot checks:  {para_dir}/spot_check_{{light,moderate,aggressive}}.txt
  Diagnostic:   {diag_dir}/diagnostic_results.json
  Forget dirs:  {diag_dir}/forget_dirs_layer{{16,20,24}}.pt

Workflow:
  1. Wait for paraphrase jobs to finish
  2. Review spot_check_*.txt files for clinical fidelity
  3. If fidelity OK → diagnostic runs automatically (if chained)
     or submit manually: python submit_datashift_diagnostic.py --diagnostic-only
  4. Read diagnostic_results.json for go/no-go on three-stage pipeline
""")


if __name__ == "__main__":
    main()

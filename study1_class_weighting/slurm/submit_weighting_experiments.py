#!/usr/bin/env python3
"""
Submit log + effective number weighting experiments for Track A.

For each strategy × epsilon:
  1. Train DP-SGD with weighting strategy (GPU, ~2-3h)
  2. Generate 9178 synthetic notes (GPU, ~2-3h)
  3. Run fidelity eval + per-category eval (GPU for MAUVE, ~1h)

Strategies:
  log:       w_c = 1 + log(max_n / n_c)           — very compressed
  effective: w_c = (1-β) / (1-β^{n_c}), β=0.999   — Cui et al. CVPR 2019

Epsilon values: 0.5, 1, 4, inf
"""
import subprocess
import sys
from pathlib import Path

PROJ    = "/fs1/projects/unlearning_pretraining/Proj_code"
SCRIPTS = f"{PROJ}/Scripts_2.0"
TRAIN   = f"{PROJ}/Scripts_3.0/Legacy_Train/train_dp_weights_2.py"
GEN     = f"{SCRIPTS}/generation_scripts/03.2_generate.py"
EVAL    = f"{SCRIPTS}/05_evaluate.py"
PERCAT  = f"{SCRIPTS}/per_category_eval.py"

PYTHON  = "/fs1/home/h702839428/python_daniel/bin/python"
LOGS    = f"{PROJ}/logs"
ACCT    = "unlearning_pretraining"

MODEL   = "/fs1/shared/model/llm/Llama-3.2-1B-Instruct"
DATA    = f"{PROJ}/data/train.jsonl"
MODELS  = f"{PROJ}/models"
GEN_DIR = f"{PROJ}/generated/mimic"
EVAL_DIR = f"{PROJ}/eval"

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
        dep_str = f" (after {dependency})" if dependency else ""
        print(f"  {tag} -> job {jid}{dep_str}")
        return jid
    else:
        failed.append((tag, r.stderr.strip()))
        print(f"  {tag} FAILED: {r.stderr.strip()}", file=sys.stderr)
        return None


def gpu_header(tag, time="20:00:00", mem="32G"):
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


def cpu_header(tag, time="01:00:00", mem="16G"):
    return f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --account={ACCT}
#SBATCH --partition=defq
#SBATCH --qos=long
#SBATCH --cpus-per-task=4
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={LOGS}/{tag}_%j.log

"""


# ─── Experiments ──────────────────────────────────────────────────────

strategies = [
    {
        "name": "log",
        "label": "log10",
        "args": "--weight_strategy log --weight_cap 10.0",
    },
    {
        "name": "effective",
        "label": "eff0999",
        "args": "--weight_strategy effective --weight_beta 0.999 --weight_cap 10.0",
    },
]

epsilons = [
    (0.5, "eps05"),
    (1.0, "eps1"),
    (4.0, "eps4"),
    (999.0, "epsinf"),  # 999 → treated as inf by the script
]

for strat in strategies:
    for eps_val, eps_label in epsilons:
        run_tag = f"{eps_label}_{strat['label']}_mimic"

        # ── Phase 1: Train ──
        train_tag = f"train_{run_tag}"
        train_jid = sbatch(
            gpu_header(train_tag, time="20:00:00", mem="32G") +
            f"""{PYTHON} "{TRAIN}" \\
  --model_path "{MODEL}" \\
  --data "{DATA}" \\
  --output "{MODELS}" \\
  --epsilon {eps_val} \\
  --epochs 2 \\
  --lr 5e-5 \\
  --max_grad_norm 1.0 \\
  --max_length 512 \\
  {strat['args']}
""",
            train_tag,
        )

        # ── Phase 2: Generate ──
        # The checkpoint path follows the naming convention from the script.
        # We use a glob pattern in the generation command to find it.
        gen_tag = f"gen_{run_tag}"
        gen_out = f"{GEN_DIR}/{run_tag}"
        gen_jid = sbatch(
            gpu_header(gen_tag, time="04:00:00", mem="32G") +
            f"""# Find the checkpoint directory
CKPT=$(ls -td "{MODELS}"/llama_dp_eps{eps_val}_{strat['label']}*_mimic_*/final 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint found for {run_tag}"
    exit 1
fi
echo "Using checkpoint: $CKPT"

{PYTHON} "{GEN}" \\
  --checkpoint "$CKPT" \\
  --base_model "{MODEL}" \\
  --data "{DATA}" \\
  --output "{GEN_DIR}" \\
  --n_generate 9178 \\
  --n_per_category 500 \\
  --k 4 \\
  --temperature 0.1 \\
  --top_p 1.0 \\
  --repetition_penalty 1.2 \\
  --epsilon {eps_val}
""",
            gen_tag,
            dependency=train_jid,
        )

        # ── Phase 3: Fidelity eval ──
        eval_tag = f"eval_{run_tag}"
        sbatch(
            gpu_header(eval_tag, time="02:00:00", mem="32G") +
            f"""{PYTHON} "{EVAL}" \\
  --synthetic "{gen_out}/synthetic_bhc.jsonl" \\
  --real "{DATA}" \\
  --output "{EVAL_DIR}" \\
  --epsilon {eps_val}
""",
            eval_tag,
            dependency=gen_jid,
        )


# ── Per-category eval (runs on all generated dirs at once, after all gen done) ──
print("\n--- Per-category eval (after all generation completes) ---")
gen_jids = [jid for tag, jid in submitted if tag.startswith("gen_") and jid]
if gen_jids:
    last_gen = gen_jids[-1]
    percat_tag = "percat_new_weights"
    sbatch(
        gpu_header(percat_tag, time="03:00:00", mem="48G") +
        f"""{PYTHON} "{PERCAT}" \\
  --runs "{','.join(s['label'] for s in strategies)}"
""",
        percat_tag,
        dependency=last_gen,
    )


# ─── Summary ──────────────────────────────────────────────────────────
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
Weighting Strategy Experiments
==============================
Strategies: log (w=1+log(N_max/n_c)), effective (Cui et al. 2019, β=0.999)
Epsilons:   0.5, 1, 4, inf
Model:      Llama-3.2-1B-Instruct
Data:       MIMIC-IV BHC ({DATA})

Per config: train (~3h) → generate 9178 notes k=4 (~3h) → fidelity eval (~1h)
Total: {len(strategies) * len(epsilons)} configs × 3 phases = {len(strategies) * len(epsilons) * 3} jobs

After completion, compare against existing results:
  - sqrt cap=10:       eps*_sqrt10_mimic
  - power-law α=0.3:   eps*_power03cap10_mimic
  - unweighted:         eps*
""")

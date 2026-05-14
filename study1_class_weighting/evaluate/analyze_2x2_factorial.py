"""
analyze_2x2_factorial.py
========================
After unweighted-stratified generation + eval, compute the 2x2 decomposition
tables from the experiment plan.

Reads: PROJ/eval/per_category_mauve.csv
Outputs: prints tables + writes analyze_2x2_results.md

Usage:
    python analyze_2x2_factorial.py
    # Or from cluster:
    cd /fs1/projects/unlearning_pretraining/Proj_code
    python /path/to/analyze_2x2_factorial.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJ = Path("/fs1/projects/unlearning_pretraining/Proj_code")
CSV_PATH = PROJ / "eval" / "per_category_mauve.csv"

RUN_PAIRS = {
    "eps05": {
        "unweighted_prop": "eps05",
        "unweighted_strat": "eps05_unweighted_stratified",
        "sqrt_strat": "eps05_sqrt10_mimic",
    },
    "eps1": {
        "unweighted_prop": "eps1",
        "unweighted_strat": "eps1_unweighted_stratified",
        "sqrt_strat": "eps1_sqrt10_mimic",
    },
    "eps4": {
        "unweighted_prop": "eps4",
        "unweighted_strat": "eps4_unweighted_stratified",
        "sqrt_strat": "eps4_sqrt10_mimic",
    },
}


def load_mauve_csv():
    data = defaultdict(dict)
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row["run_id"]
            cat = row["category"]
            mauve = float(row["mauve"])
            data[run_id][cat] = mauve
    return data


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def main():
    data = load_mauve_csv()

    lines = []
    lines.append("# 2×2 Factorial Analysis: Weighting vs. Stratified Sampling\n")
    lines.append(f"**Generated:** from {CSV_PATH}\n")

    # --- Table A: mean MAUVE comparison ---
    lines.append("## Table A: Mean Per-Category MAUVE\n")
    lines.append("| ε | Unweighted-Proportional | Unweighted-Stratified | Sqrt-Stratified | Sampling Effect | Weighting Effect | Total Gap |")
    lines.append("|---|---|---|---|---|---|---|")

    for eps_label in ["eps05", "eps1", "eps4"]:
        runs = RUN_PAIRS[eps_label]
        up = mean(list(data.get(runs["unweighted_prop"], {}).values()))
        us = mean(list(data.get(runs["unweighted_strat"], {}).values()))
        ss = mean(list(data.get(runs["sqrt_strat"], {}).values()))

        if runs["unweighted_strat"] not in data:
            lines.append(f"| {eps_label} | {up:.4f} | **NOT YET RUN** | {ss:.4f} | — | — | {ss - up:.4f} |")
            continue

        sampling_effect = us - up
        weighting_effect = ss - us
        total_gap = ss - up

        lines.append(
            f"| {eps_label} | {up:.4f} | {us:.4f} | {ss:.4f} | "
            f"{sampling_effect:+.4f} | {weighting_effect:+.4f} | {total_gap:+.4f} |"
        )

    lines.append("")

    # --- Table B: per-category breakdown at eps4 ---
    eps4 = RUN_PAIRS["eps4"]
    if eps4["unweighted_strat"] in data:
        lines.append("## Table B: Per-Category Breakdown at ε=4\n")
        lines.append("| Category | Unwtd-Prop | Unwtd-Strat | Sqrt-Strat | Sampling Δ | Weighting Δ |")
        lines.append("|---|---|---|---|---|---|")

        up_data = data.get(eps4["unweighted_prop"], {})
        us_data = data.get(eps4["unweighted_strat"], {})
        ss_data = data.get(eps4["sqrt_strat"], {})

        all_cats = sorted(set(up_data) | set(us_data) | set(ss_data))
        for cat in all_cats:
            up_v = up_data.get(cat)
            us_v = us_data.get(cat)
            ss_v = ss_data.get(cat)

            up_s = f"{up_v:.4f}" if up_v is not None else "—"
            us_s = f"{us_v:.4f}" if us_v is not None else "—"
            ss_s = f"{ss_v:.4f}" if ss_v is not None else "—"

            samp_d = f"{us_v - up_v:+.4f}" if (up_v is not None and us_v is not None) else "—"
            wt_d = f"{ss_v - us_v:+.4f}" if (us_v is not None and ss_v is not None) else "—"

            short_cat = cat[:50]
            lines.append(f"| {short_cat} | {up_s} | {us_s} | {ss_s} | {samp_d} | {wt_d} |")

        # Summary row
        up_vals = [v for v in up_data.values()]
        us_vals = [v for v in us_data.values()]
        ss_vals = [v for v in ss_data.values()]
        lines.append(
            f"| **Mean** | **{mean(up_vals):.4f}** | **{mean(us_vals):.4f}** | "
            f"**{mean(ss_vals):.4f}** | **{mean(us_vals) - mean(up_vals):+.4f}** | "
            f"**{mean(ss_vals) - mean(us_vals):+.4f}** |"
        )
        lines.append("")

    # --- Table C: epsilon sensitivity ---
    lines.append("## Table C: ε Sensitivity\n")
    lines.append("| ε | Unwtd-Prop | Unwtd-Strat | Sqrt-Strat | Sampling % of gap | Weighting % of gap |")
    lines.append("|---|---|---|---|---|---|")

    for eps_label, eps_val in [("eps05", "0.5"), ("eps1", "1"), ("eps4", "4")]:
        runs = RUN_PAIRS[eps_label]
        up = mean(list(data.get(runs["unweighted_prop"], {}).values()))
        us = mean(list(data.get(runs["unweighted_strat"], {}).values()))
        ss = mean(list(data.get(runs["sqrt_strat"], {}).values()))

        if runs["unweighted_strat"] not in data:
            lines.append(f"| {eps_val} | {up:.4f} | — | {ss:.4f} | — | — |")
            continue

        total = ss - up
        samp = us - up
        wt = ss - us
        samp_pct = (samp / total * 100) if total != 0 else 0
        wt_pct = (wt / total * 100) if total != 0 else 0

        lines.append(
            f"| {eps_val} | {up:.4f} | {us:.4f} | {ss:.4f} | "
            f"{samp_pct:.1f}% | {wt_pct:.1f}% |"
        )

    lines.append("")

    # --- Hypothesis check ---
    lines.append("## Hypothesis Evaluation\n")
    for eps_label in ["eps05", "eps1", "eps4"]:
        runs = RUN_PAIRS[eps_label]
        if runs["unweighted_strat"] not in data:
            lines.append(f"- **{eps_label}:** Not yet run\n")
            continue

        up = mean(list(data[runs["unweighted_prop"]].values()))
        us = mean(list(data[runs["unweighted_strat"]].values()))
        ss = mean(list(data[runs["sqrt_strat"]].values()))

        h1 = up < us < ss
        h2 = (ss - us) > (us - up)

        lines.append(f"### {eps_label}")
        lines.append(f"- Unwtd-Prop={up:.4f}, Unwtd-Strat={us:.4f}, Sqrt-Strat={ss:.4f}")
        lines.append(f"- **H1** (between): {'PASS' if h1 else 'FAIL'} — {up:.4f} < {us:.4f} < {ss:.4f}" if h1 else f"- **H1** (between): FAIL — ordering is {up:.4f}, {us:.4f}, {ss:.4f}")
        lines.append(f"- **H2** (weighting > sampling): {'PASS' if h2 else 'FAIL'} — weighting Δ={ss-us:.4f} vs sampling Δ={us-up:.4f}")
        lines.append("")

    output = "\n".join(lines)
    print(output)

    out_path = PROJ / "eval" / "analyze_2x2_results.md"
    with open(out_path, "w") as f:
        f.write(output)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()

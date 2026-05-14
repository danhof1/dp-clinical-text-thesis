"""
09_section_structure.py — BHC Section Structure Rate (Track A, MIMIC)
Output: eval/section_structure_results.csv
"""

from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path

import numpy as np

PROJ     = "/fs1/projects/unlearning_pretraining/Proj_code"
GEN_ROOT = f"{PROJ}/generated/mimic"
REAL_SRC = f"{GEN_ROOT}/data0_base/synthetic_bhc.jsonl"
EVAL_DIR = f"{PROJ}/eval"

SECTIONS = {
    "assessment":          re.compile(r"\bassessment\b", re.I),
    "plan":                re.compile(r"\bplan\b", re.I),
    "medications":         re.compile(r"\bmedications?\b", re.I),
    "discharge_diagnosis": re.compile(r"\bdischarge\s+diagnos", re.I),
    "hospital_course":     re.compile(r"\bhospital\s+course\b", re.I),
    "follow_up":           re.compile(r"\bfollow[\s\-]?up\b", re.I),
    "labs":                re.compile(r"\blabs?\b|\blaboratory\b|\blaboratories\b", re.I),
    "imaging":             re.compile(r"\bimaging\b|\bradiology\b|\bct\s+scan\b|\bmri\b", re.I),
    "physical_exam":       re.compile(r"\bphysical\s+exam\b|\bvital\s+sign", re.I),
    "chief_complaint":     re.compile(r"\bchief\s+complaint\b|\bpresenting\s+complaint\b", re.I),
    "history":             re.compile(r"\bhpi\b|\bhistory\s+of\s+present", re.I),
    "problem_list":        re.compile(r"\bproblem\s+list\b|\bactive\s+problem", re.I),
}


def section_flags(text):
    return {name: int(bool(pat.search(text))) for name, pat in SECTIONS.items()}


def load_texts(jsonl_path, field):
    texts = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            t = r.get(field, "").strip()
            if t:
                texts.append(t)
    return texts


def compute_rates(texts):
    if not texts:
        return {name: 0.0 for name in SECTIONS}
    totals = {name: 0 for name in SECTIONS}
    for text in texts:
        for name, present in section_flags(text).items():
            totals[name] += present
    return {name: totals[name] / len(texts) for name in SECTIONS}


def discover_runs(requested=None):
    all_runs = sorted(
        d.name for d in Path(GEN_ROOT).iterdir()
        if (Path(GEN_ROOT) / d.name / "synthetic_bhc.jsonl").exists()
    )
    if requested:
        return [r for r in all_runs if r in requested]
    return all_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None)
    args = ap.parse_args()

    runs = discover_runs(args.runs.split(",") if args.runs else None)
    print(f"Runs: {runs}\n")

    print("Loading real notes (original_bhc)...")
    real_texts = load_texts(REAL_SRC, "original_bhc")
    print(f"  {len(real_texts)} real notes")
    real_rates = compute_rates(real_texts)

    print("\nReal section rates:")
    for sec, rate in sorted(real_rates.items(), key=lambda x: -x[1]):
        print(f"  {sec:<25}  {rate:.3f}")

    section_names = list(SECTIONS.keys())
    fieldnames = (
        ["run_id", "n_notes"]
        + [f"{s}_rate" for s in section_names]
        + ["fidelity_score", "structural_coverage"]
    )

    results = []
    real_row = {"run_id": "REAL_baseline", "n_notes": len(real_texts)}
    for s in section_names:
        real_row[f"{s}_rate"] = round(real_rates[s], 4)
    real_row["fidelity_score"] = 0.0
    real_row["structural_coverage"] = 1.0
    results.append(real_row)

    for run in runs:
        print(f"\n[{run}]")
        try:
            syn_texts = load_texts(str(Path(GEN_ROOT) / run / "synthetic_bhc.jsonl"), "synthetic_bhc")
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  {len(syn_texts)} synthetic notes")
        syn_rates = compute_rates(syn_texts)

        deviations = [abs(syn_rates[s] - real_rates[s]) for s in section_names]
        fidelity   = float(np.mean(deviations))
        coverages  = [min(syn_rates[s] / real_rates[s], 1.0) for s in section_names if real_rates[s] > 0]
        structural_coverage = float(np.mean(coverages)) if coverages else 0.0

        print(f"  fidelity_score={fidelity:.4f}  structural_coverage={structural_coverage:.4f}")
        for sec in section_names:
            real_r = real_rates[sec]
            syn_r  = syn_rates[sec]
            flag   = " *" if abs(syn_r - real_r) > 0.15 else ""
            print(f"    {sec:<25}  real={real_r:.3f}  syn={syn_r:.3f}{flag}")

        row = {"run_id": run, "n_notes": len(syn_texts)}
        for s in section_names:
            row[f"{s}_rate"] = round(syn_rates[s], 4)
        row["fidelity_score"]      = round(fidelity, 4)
        row["structural_coverage"] = round(structural_coverage, 4)
        results.append(row)

    out = f"{EVAL_DIR}/section_structure_results.csv"
    Path(EVAL_DIR).mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

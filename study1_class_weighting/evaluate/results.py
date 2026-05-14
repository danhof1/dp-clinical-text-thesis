"""
results.py
==========
Print all experiment results to the terminal in formatted tables.

Sections:
  1.  Generated data status (Track A)
  2.  Checkpoint audit (Track A)
  3.  Infini-gram pre-training contamination audit
  4.  Black-box MIA (Track A — SynBench protocol)
  5.  LiRA white-box MIA (Track A)
  6.  Fidelity metrics (Track A)
  7.  ICD category adherence — single-run detail (Track A)
  8.  Aggregate scoring summary (coherence PPL + adherence, all runs)
  9.  Per-category adherence (Track A — all runs × 22 ICD chapters)
  10. Per-category MAUVE (Track A — all runs × 22 ICD chapters)
  11. Track B unlearning MIA (tier-1 attacks on unlearn checkpoints)
  12. Track B DP-LoRA catalog (loss table — key RMU vs GA/NPO finding)
  13. Track B DP-LoRA fidelity (RMU generate → eval metrics)
  14. Track B AI safety eval suite (KS test, re-learn attack, MMLU-Medical)
  15. Track B post-DP MIA (deduped splits_v2, dp_lora_base comparison)
  16. Track B prefix completion extraction attack (Carlini et al. 2021)
  17. Cross-track synthesis & key findings

Usage:
    python Scripts_2.0/results.py --base_dir .
"""

import csv
import json
import argparse
import statistics
from pathlib import Path


# =========================================================================
# Experiment registry
# =========================================================================

MIA_EXPERIMENTS = [
    # MIMIC — unweighted
    ("MIMIC",     "No",  "Base", "mia/mimic/mimic_data0/mia_summary.json"),
    ("MIMIC",     "No",  "0.5",  "mia/mimic/eps05/mia_summary.json"),
    ("MIMIC",     "No",  "1",    "mia/mimic/eps1/mia_summary.json"),
    ("MIMIC",     "No",  "4",    "mia/mimic/eps4/mia_summary.json"),
    ("MIMIC",     "No",  "inf",  "mia/mimic/mimic_epsinf/mia_summary.json"),
    # MIMIC — weighted (thesis primary)
    ("MIMIC",     "Yes", "0.5",  "mia/mimic/mimic_weighted_eps05/mia_summary.json"),
    ("MIMIC",     "Yes", "1",    "mia/mimic/mimic_weighted_eps1/mia_summary.json"),
    ("MIMIC",     "Yes", "inf",  "mia/mimic/mimic_weighted_epsinf/mia_summary.json"),
    # MTSamples
    ("MTSamples", "Yes", "Base", "mia/mtsamples/mtsamples_data0/mia_summary.json"),
    ("MTSamples", "Yes", "0.5",  "mia/mtsamples/mtsamples_eps05/mia_summary.json"),
    ("MTSamples", "Yes", "1",    "mia/mtsamples/mtsamples_eps1/mia_summary.json"),
    ("MTSamples", "Yes", "4",    "mia/mtsamples/mtsamples_eps4/mia_summary.json"),
    ("MTSamples", "Yes", "inf",  "mia/mtsamples/mtsamples_epsinf/mia_summary.json"),
]

LIRA_EXPERIMENTS = [
    ("0.5",   "lira/results/eps0.5/lira_summary.json",
              "models/llama_dp_eps0.5_20260319_1338/final"),
    ("1.0",   "lira/results/eps1.0/lira_summary.json",
              "models/llama_dp_eps1.0_20260329_1639/final"),
    ("4.0",   "lira/results/eps4.0/lira_summary.json",
              "models/llama_dp_eps4.0_20260318_1628/final"),
    ("999.0", "lira/results/eps999.0/lira_summary.json",
              "models/llama_sgd_epsinf_20260328_1324/final"),
]

# (dataset, stratified, epsilon, rel_path, weighting_label)
FIDELITY_EXPERIMENTS = [
    # MIMIC — unweighted baseline
    ("MIMIC", "No",  "Base", "eval/mimic/data0_base/eval_results.json",                "baseline"),
    ("MIMIC", "No",  "0.5",  "eval/mimic/eps05_full/eval_results.json",                "unweighted"),
    ("MIMIC", "No",  "1",    "eval/mimic/eps1_full/eval_results.json",                 "unweighted"),
    ("MIMIC", "No",  "4",    "eval/mimic/eps4_full/eval_results.json",                 "unweighted"),
    ("MIMIC", "No",  "inf",  "eval/mimic/eps_inf/eval_results.json",                   "unweighted"),
    # MIMIC — sqrt_cap10 (Track A primary)
    ("MIMIC", "Yes", "0.5",  "eval/mimic/eps05_sqrt10_mimic/eval_results.json",        "sqrt_cap10"),
    ("MIMIC", "Yes", "1",    "eval/mimic/eps1_sqrt10_mimic/eval_results.json",         "sqrt_cap10"),
    ("MIMIC", "Yes", "4",    "eval/mimic/eps4_sqrt10_mimic/eval_results.json",         "sqrt_cap10"),
    ("MIMIC", "Yes", "inf",  "eval/mimic/epsinf_sqrt10_mimic/eval_results.json",       "sqrt_cap10"),
    # MIMIC — power-law alpha=0.3 cap10
    ("MIMIC", "Yes", "0.5",  "eval/mimic/eps05_power03cap10_mimic/eval_results.json",  "power_a0.3"),
    ("MIMIC", "Yes", "1",    "eval/mimic/eps1_power03cap10_mimic/eval_results.json",   "power_a0.3"),
    ("MIMIC", "Yes", "4",    "eval/mimic/eps4_power03cap10_mimic/eval_results.json",   "power_a0.3"),
    ("MIMIC", "Yes", "inf",  "eval/mimic/epsinf_power03cap10_mimic/eval_results.json", "power_a0.3"),
    # MIMIC — inverse-freq weighted (superseded)
    ("MIMIC", "Yes", "0.5",  "eval/mimic/data2_eps05_weighted/eval_results.json",      "inv-freq"),
    ("MIMIC", "Yes", "1",    "eval/mimic/data2_eps1_weighted/eval_results.json",       "inv-freq"),
    ("MIMIC", "Yes", "4",    "eval/mimic/data2_eps4_weighted/eval_results.json",       "inv-freq"),
    # MTSamples
    ("MTSamples", "Yes", "Base", "eval/mtsamples/data0_base/eval_results.json",        "baseline"),
    ("MTSamples", "Yes", "0.5",  "eval/mtsamples/data2_eps05/eval_results.json",       "unweighted"),
    ("MTSamples", "Yes", "1",    "eval/mtsamples/data2_eps1/eval_results.json",        "unweighted"),
    ("MTSamples", "Yes", "4",    "eval/mtsamples/data2_eps4/eval_results.json",        "unweighted"),
    ("MTSamples", "Yes", "inf",  "eval/mtsamples/data1_sgd/eval_results.json",         "unweighted"),
]

# Fixed: files are in audit/ directly, not audit/sample/
INFIGRAM_FILES = {
    "MIMIC":     "audit/mimic_results_all_ngrams.json",
    "MTSamples": "audit/mtsamples_infini_results.json",
}

ICD_EXPERIMENTS = [
    ("MIMIC", "4", "sqrt_cap10",
     "eval/mimic/eps4_sqrt10_mimic/icd_match/category_accuracy.json"),
]

TRACK_B_REPO = "Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp"

TRACK_B_UNLEARN_DIRS = [
    ("GA",    "MTSamples", "unlearn_ga"),
    ("GA",    "PMC",       "unlearn_ga_pmc"),
    ("GA+GD", "MTSamples", "unlearn_ga_gd"),
    ("GA+GD", "PMC",       "unlearn_ga_gd_pmc"),
    ("NPO",   "MTSamples", "unlearn_npo"),
    ("NPO",   "PMC",       "unlearn_npo_pmc"),
    ("RMU",   "MTSamples", "unlearn_rmu"),
    ("RMU",   "PMC",       "unlearn_rmu_pmc"),
]

TRACK_B_FIDELITY_TAGS = [
    ("RMU", "MTSamples", "1",   "rmu_eps1"),
    ("RMU", "MTSamples", "3",   "rmu_eps3"),
    ("RMU", "MTSamples", "8",   "rmu_eps8"),
    ("RMU", "MTSamples", "inf", "rmu_epsinf"),
    ("RMU", "PMC",       "1",   "rmu_eps1_pmc"),
    ("RMU", "PMC",       "3",   "rmu_eps3_pmc"),
    ("RMU", "PMC",       "8",   "rmu_eps8_pmc"),
    ("RMU", "PMC",       "inf", "rmu_epsinf_pmc"),
]

EPS_ORDER = {"Base": 0, "0.5": 1, "1": 2, "1.0": 2, "3": 3, "4": 4, "4.0": 4,
             "8": 5, "inf": 6, "999.0": 6}

# (stage, dataset, epsilon, eval_dir_name)
# stage: "baseline" | "preDp" | "postDp"
TRACK_B_EVAL_SUITE_TAGS = [
    # MTSamples
    ("baseline", "MTSamples", "—",   "baseline_mtsamples"),
    ("preDp",    "MTSamples", "—",   "unlearn_rmu"),
    ("postDp",   "MTSamples", "1",   "dp_lora_eps1_rmu"),
    ("postDp",   "MTSamples", "3",   "dp_lora_eps3_rmu"),
    ("postDp",   "MTSamples", "8",   "dp_lora_eps8_rmu"),
    ("postDp",   "MTSamples", "inf", "dp_lora_epsinf_rmu"),
    # PMC
    ("baseline", "PMC",       "—",   "baseline_pmc"),
    ("preDp",    "PMC",       "—",   "unlearn_rmu_pmc"),
    ("postDp",   "PMC",       "1",   "dp_lora_eps1_rmu_pmc"),
    ("postDp",   "PMC",       "3",   "dp_lora_eps3_rmu_pmc"),
    ("postDp",   "PMC",       "8",   "dp_lora_eps8_rmu_pmc"),
    ("postDp",   "PMC",       "inf", "dp_lora_epsinf_rmu_pmc"),
]

# (stage, dataset, epsilon, attack_output_dir)
# Post-DP MIA on deduplicated splits_v2  (Section 15)
TRACK_B_POST_DP_MIA_TAGS = [
    # MTSamples -- RMU pipeline
    ("baseline",  "MTSamples", "---",   "v2_baseline_mtsamples"),
    ("preDp",     "MTSamples", "---",   "v2_unlearn_rmu"),
    ("postDp",    "MTSamples", "1",   "v2_dp_lora_eps1_rmu"),
    ("postDp",    "MTSamples", "3",   "v2_dp_lora_eps3_rmu"),
    ("postDp",    "MTSamples", "8",   "v2_dp_lora_eps8_rmu"),
    ("postDp",    "MTSamples", "inf", "v2_dp_lora_epsinf_rmu"),
    # MTSamples -- No-unlearn baseline
    ("noUnlearn", "MTSamples", "1",   "v2_dp_lora_eps1_base"),
    ("noUnlearn", "MTSamples", "3",   "v2_dp_lora_eps3_base"),
    ("noUnlearn", "MTSamples", "8",   "v2_dp_lora_eps8_base"),
    ("noUnlearn", "MTSamples", "inf", "v2_dp_lora_epsinf_base"),
    # PMC -- RMU pipeline
    ("postDp",    "PMC",       "1",   "dp_lora_eps1_rmu_pmc"),
    ("postDp",    "PMC",       "3",   "dp_lora_eps3_rmu_pmc"),
    ("postDp",    "PMC",       "8",   "dp_lora_eps8_rmu_pmc"),
    ("postDp",    "PMC",       "inf", "dp_lora_epsinf_rmu_pmc"),
    # PMC -- No-unlearn baseline
    ("noUnlearn", "PMC",       "1",   "dp_lora_eps1_base_pmc"),
    ("noUnlearn", "PMC",       "3",   "dp_lora_eps3_base_pmc"),
    ("noUnlearn", "PMC",       "8",   "dp_lora_eps8_base_pmc"),
    ("noUnlearn", "PMC",       "inf", "dp_lora_epsinf_base_pmc"),
]

# Completion attack configs -- (stage, dataset, epsilon, output_dir_name)
TRACK_B_COMPLETION_TAGS = [
    ("baseline", "MTSamples", "---",   "compl_baseline_mts"),
    ("preDp",    "MTSamples", "---",   "compl_rmu_mts"),
    ("postDp",   "MTSamples", "1",   "compl_eps1_rmu_mts"),
    ("postDp",   "MTSamples", "3",   "compl_eps3_rmu_mts"),
    ("postDp",   "MTSamples", "8",   "compl_eps8_rmu_mts"),
    ("postDp",   "MTSamples", "inf", "compl_epsinf_rmu_mts"),
    ("baseline", "PMC",       "---",   "compl_baseline_pmc"),
    ("preDp",    "PMC",       "---",   "compl_rmu_pmc"),
    ("postDp",   "PMC",       "1",   "compl_eps1_rmu_pmc"),
    ("postDp",   "PMC",       "3",   "compl_eps3_rmu_pmc"),
    ("postDp",   "PMC",       "8",   "compl_eps8_rmu_pmc"),
    ("postDp",   "PMC",       "inf", "compl_epsinf_rmu_pmc"),
]




# =========================================================================
# Loaders
# =========================================================================

def load_mia(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        bb = d.get("blackbox", d)
        return {
            "auc":    bb.get("auc"),
            "tpr_1":  bb.get("tpr_at_fpr_1pct"),
            "tpr_10": bb.get("tpr_at_fpr_10pct"),
        }
    except Exception:
        return None


def load_lira(path, expected_checkpoint):
    p = Path(path)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        sig = d.get("attack_signal", {})
        phi = d.get("phi_target", {})
        actual_ckpt = d.get("target_checkpoint", "")
        ckpt_ok = expected_checkpoint.rstrip("/") in str(actual_ckpt).rstrip("/")
        return {
            "pct_above_out": sig.get("pct_above_out"),
            "score_mean":    sig.get("mean"),
            "phi_mean":      phi.get("mean"),
            "phi_std":       phi.get("std"),
            "global_sigma":  d.get("global_sigma_out"),
            "n_shadows":     d.get("n_shadows"),
            "checkpoint_ok": ckpt_ok,
        }
    except Exception:
        return None


def load_fidelity(path):
    """Load Track A eval_results.json (nested format from 05_evaluate.py)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        text = p.read_text()
        # Handle Python-serialized Infinity/NaN (not valid JSON)
        text = text.replace(": Infinity", ": null").replace(": -Infinity", ": null").replace(": NaN", ": null")
        d = json.loads(text)
        mauve_raw = d.get("mauve")
        if isinstance(mauve_raw, dict):
            mauve_val = mauve_raw.get("mauve_score")
        elif isinstance(mauve_raw, (int, float)):
            mauve_val = float(mauve_raw)
        else:
            mauve_val = None
        ppl_raw = d.get("coherence_ppl")
        ppl_val = ppl_raw.get("mean") if isinstance(ppl_raw, dict) else ppl_raw
        return {
            "length_kl":  d.get("length_kl", {}).get("kl_divergence"),
            "unigram_kl": d.get("unigram", {}).get("1gram_kl_divergence"),
            "bigram_kl":  d.get("bigram", {}).get("2gram_kl_divergence"),
            "unary_jac":  d.get("term_overlap", {}).get("unary_jaccard"),
            "binary_jac": d.get("term_overlap", {}).get("binary_jaccard"),
            "mauve":      mauve_val,
            "ppl":        ppl_val,
        }
    except Exception:
        return None


def load_track_b_fidelity(path):
    """Load Track B fidelity JSON (flat format from scripts/eval_fidelity.py)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        text = p.read_text()
        text = text.replace(": Infinity", ": null").replace(": -Infinity", ": null").replace(": NaN", ": null")
        d = json.loads(text)
        mauve_raw = d.get("mauve")
        if isinstance(mauve_raw, dict):
            mauve_val = mauve_raw.get("mauve_score")
        elif isinstance(mauve_raw, (int, float)):
            mauve_val = float(mauve_raw)
        else:
            mauve_val = None
        return {
            "length_kl":  d.get("length_kl"),
            "unigram_kl": d.get("unigram_kl"),
            "bigram_kl":  d.get("bigram_kl"),
            "unary_jac":  d.get("unary_jaccard"),
            "binary_jac": None,          # not computed by eval_fidelity.py
            "ppl":        d.get("ppl_mean"),
            "mauve":      mauve_val,
            "n_synthetic": d.get("n_synthetic"),
        }
    except Exception:
        return None


def load_track_b_tier1(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        records = json.loads(p.read_text())
        if not isinstance(records, list):
            return None
        return {r["attack"]: r for r in records}
    except Exception:
        return None


def load_icd(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def load_infigram(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        result = {
            "dataset":   d.get("dataset", "unknown"),
            "index":     d.get("index", "unknown"),
            "n_records": d.get("n_records", 0),
            "ngrams":    {},
        }
        for ngram_size in d.get("ngram_sizes", []):
            key = str(ngram_size)
            raw_stats = d["per_ngram_size"][key]
            all_recs  = [r for r in d["records"] if str(r["ngram_size"]) == key]
            errors    = [r for r in all_recs if r["n_errors"] == 50 and r["n_queried"] == 0]
            valid     = [r for r in all_recs if not (r["n_errors"] == 50 and r["n_queried"] == 0)]
            overlaps  = [r["overlap_fraction"] for r in valid]
            result["ngrams"][ngram_size] = {
                "n_total":      len(all_recs),
                "n_api_errors": len(errors),
                "n_valid":      len(valid),
                "raw_mean":     raw_stats["mean_overlap"],
                "raw_median":   raw_stats["median_overlap"],
                "raw_p90":      raw_stats["p90_overlap"],
                "raw_pct_any":  raw_stats["pct_records_any_overlap"],
                "corr_mean":    statistics.mean(overlaps)   if overlaps else 0.0,
                "corr_median":  statistics.median(overlaps) if overlaps else 0.0,
                "corr_pct_any": (sum(1 for o in overlaps if o > 0) / len(valid) * 100
                                 if valid else 0.0),
            }
        return result
    except Exception:
        return None


def load_eval_suite(path):
    """Load outputs/eval/{tag}/eval_suite_results.json for Track B eval suite."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        ks = d.get("ks_test", {})
        mmlu = d.get("mmlu_medical", {})
        meta = mmlu.get("_meta", {})
        rl = d.get("relearn_attack", {})
        return {
            "ks_stat":       ks.get("ks_statistic"),
            "ks_p":          ks.get("p_value"),
            "ks_verdict":    ks.get("verdict"),
            "forget_nll":    ks.get("forget_nll_mean"),
            "control_nll":   ks.get("control_nll_mean"),
            "mmlu_mean":     meta.get("mean_accuracy"),
            "mmlu_ck":       mmlu.get("clinical_knowledge", {}).get("accuracy"),
            "mmlu_pm":       mmlu.get("professional_medicine", {}).get("accuracy"),
            "mmlu_an":       mmlu.get("anatomy", {}).get("accuracy"),
            "rl_ppl_before": rl.get("ppl_before_relearn"),
            "rl_ppl_after":  rl.get("ppl_after_relearn"),
            "rl_recovery":   rl.get("recovery_pct"),
            "rl_verdict":    rl.get("verdict"),
            "rl_error":      rl.get("error"),
        }
    except Exception:
        return None


def load_csv_rows(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return list(csv.DictReader(f))
    except Exception:
        return None


def load_track_b_catalog(base):
    p = Path(base) / TRACK_B_REPO / "results" / "experiment_catalog.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# =========================================================================
# Formatting helpers
# =========================================================================


def load_completion_attack(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def fmt(val, width=8, decimals=4):
    if val is None:
        return "—".ljust(width)
    return f"{float(val):.{decimals}f}".ljust(width)


def pct(val, width=8):
    if val is None:
        return "—".ljust(width)
    return f"{val*100:.1f}%".ljust(width)


def auc_flag(auc):
    if auc is None:
        return ""
    if auc >= 0.60:
        return " HIGH"
    if auc >= 0.55:
        return " ELEV"
    return ""


def lira_flag(p):
    if p is None:
        return ""
    if p > 0.55:
        return " MEMORISED"
    if p > 0.50:
        return " WEAK"
    return " OK"


def eps_label(epsilon):
    if epsilon == "Base":
        return "Base    "
    if epsilon in ("inf", "999.0"):
        return "e=inf   "
    return f"e={epsilon}     "[:8]


def section(title):
    print()
    print("  " + "=" * 92)
    print(f"  {title}")
    print("  " + "=" * 92)


def divider(width=92):
    print("  " + "-" * width)


def group_header(label):
    print(f"\n  -- {label}")


def pending(label=""):
    return f"  {'PENDING':>8}  {label}"


# =========================================================================
# 1. Generation status
# =========================================================================

def print_generation_stats(base):
    section("GENERATED DATA STATUS  (Track A)")

    datasets = [
        ("MIMIC Base (no FT)",      "Base", "generated/mimic/data0_base/synthetic_bhc.jsonl",               9178),
        ("MIMIC SGD (ε=inf)",       "inf",  "generated/mimic/data1_sgd/synthetic_bhc.jsonl",                7700),
        ("MIMIC unweighted",        "0.5",  "generated/mimic/eps05/synthetic_bhc.jsonl",                    5000),
        ("MIMIC unweighted",        "1",    "generated/mimic/eps1/synthetic_bhc.jsonl",                     5000),
        ("MIMIC unweighted",        "4",    "generated/mimic/eps4/synthetic_bhc.jsonl",                     5000),
        ("MIMIC unweighted",        "inf",  "generated/mimic/eps_inf/synthetic_bhc.jsonl",                  5000),
        ("MIMIC sqrt_cap10",        "0.5",  "generated/mimic/eps05_sqrt10_mimic/synthetic_bhc.jsonl",       9178),
        ("MIMIC sqrt_cap10",        "1",    "generated/mimic/eps1_sqrt10_mimic/synthetic_bhc.jsonl",        9178),
        ("MIMIC sqrt_cap10",        "4",    "generated/mimic/eps4_sqrt10_mimic/synthetic_bhc.jsonl",        9178),
        ("MIMIC sqrt_cap10",        "inf",  "generated/mimic/epsinf_sqrt10_mimic/synthetic_bhc.jsonl",      9178),
        ("MIMIC power_a0.3",        "0.5",  "generated/mimic/eps05_power03cap10_mimic/synthetic_bhc.jsonl", 9178),
        ("MIMIC power_a0.3",        "1",    "generated/mimic/eps1_power03cap10_mimic/synthetic_bhc.jsonl",  9178),
        ("MIMIC power_a0.3",        "4",    "generated/mimic/eps4_power03cap10_mimic/synthetic_bhc.jsonl",  9178),
        ("MIMIC power_a0.3",        "inf",  "generated/mimic/epsinf_power03cap10_mimic/synthetic_bhc.jsonl",9178),
        ("MIMIC inv-freq (old)",    "0.5",  "generated/mimic/data2_eps05_weighted/synthetic_bhc.jsonl",     9178),
        ("MIMIC inv-freq (old)",    "1",    "generated/mimic/data2_eps1_weighted/synthetic_bhc.jsonl",      9178),
        ("MIMIC inv-freq (old)",    "4",    "generated/mimic/data2_eps4_weighted/synthetic_bhc.jsonl",      9178),
        ("MTSamples Base",          "Base", "generated/mtsamples/data0_base/synthetic_bhc.jsonl",           3594),
        ("MTSamples SGD",           "inf",  "generated/mtsamples/data1_sgd/synthetic_bhc.jsonl",            3594),
        ("MTSamples unweighted",    "0.5",  "generated/mtsamples/data2_eps05/synthetic_bhc.jsonl",          3594),
        ("MTSamples unweighted",    "1",    "generated/mtsamples/data2_eps1/synthetic_bhc.jsonl",           3594),
        ("MTSamples unweighted",    "4",    "generated/mtsamples/data2_eps4/synthetic_bhc.jsonl",           3594),
    ]

    print(f"  {'Dataset':<28}  {'Epsilon':<8}  {'Lines':>7}  {'Target':>7}  Status")
    divider()
    for (label, epsilon, relpath, target) in datasets:
        p = Path(base) / relpath
        if p.exists():
            lines = sum(1 for _ in open(p))
            status = "DONE" if lines >= target else f"PARTIAL {lines/target*100:.0f}%"
        else:
            lines, status = 0, "PENDING"
        eps_str = f"e={epsilon}" if epsilon not in ("Base", "inf") else epsilon
        print(f"  {label:<28}  {eps_str:<8}  {lines:>7,}  {target:>7,}  {status}")
    divider()


# =========================================================================
# 2. Checkpoint audit
# =========================================================================

def print_checkpoint_audit(base):
    section("CHECKPOINT AUDIT  (April 5, 2026)")
    models = [
        ("eps=0.5 weighted",   "models/llama_dp_eps0.5_20260319_1338",  True),
        ("eps=0.5 unweighted", "models/llama_dp_eps0.5_20260319_1349",  False),
        ("eps=1.0 unweighted", "models/llama_dp_eps1.0_20260318_1831",  False),
        ("eps=1.0 weighted",   "models/llama_dp_eps1.0_20260329_1639",  True),
        ("eps=4.0 weighted",   "models/llama_dp_eps4.0_20260318_1628",  True),
        ("eps=4.0 unweighted", "models/llama_dp_eps4.0_20260318_1813",  False),
        ("epsinf weighted",    "models/llama_sgd_epsinf_20260328_1324", True),
    ]
    print(f"  {'Label':<22}  {'Path':<45}  {'Weighted':<10}  Use for thesis")
    divider()
    for (label, path, weighted) in models:
        use = "YES" if weighted else "no (superseded)"
        print(f"  {label:<22}  {path:<45}  {'YES' if weighted else 'no':<10}  {use}")
    divider()
    print()
    print("  NOTE: sqrt_cap10 and power_a0.3 runs use checkpoints trained in Scripts_3.0/")
    print()


# =========================================================================
# 3. Infini-gram contamination audit
# =========================================================================

def print_infigram_table(base):
    section("INFINI-GRAM PRE-TRAINING CONTAMINATION AUDIT  (Dolma v1.7 / Llama index)")

    mimic = load_infigram(Path(base) / INFIGRAM_FILES["MIMIC"])
    mts   = load_infigram(Path(base) / INFIGRAM_FILES["MTSamples"])

    if mimic is None and mts is None:
        print("  No Infini-gram audit files found at audit/")
        return

    print(f"  {'Dataset':<12}  {'n-gram':<8}  {'n':>5}  {'Errors':>7}  "
          f"{'Raw mean':>9}  {'Raw %any':>9}  "
          f"{'Corr mean':>10}  {'Corr %any':>10}  Interpretation")
    divider()

    contamination_flags = {}
    for label, result in [("MIMIC", mimic), ("MTSamples", mts)]:
        if result is None:
            print(f"  {label:<12}  -- FILE NOT FOUND")
            continue
        group_header(f"{label}  (n={result['n_records']} records, index: {result['index']})")
        for ngram_size in sorted(result["ngrams"].keys()):
            s = result["ngrams"][ngram_size]
            n_err = s["n_api_errors"]
            err_note = f" ({n_err} API err excl)" if n_err > 0 else ""
            if ngram_size == 13:
                c = s["corr_pct_any"]
                contamination_flags[label] = c
                if c >= 50:   interp = f"CONTAMINATED ({c:.0f}%)"
                elif c >= 10: interp = f"PARTIAL ({c:.0f}%)"
                else:         interp = f"CLEAN ({c:.0f}%)"
            else:
                interp = ""
            print(f"  {label:<12}  {ngram_size}-gram   "
                  f"{s['n_valid']:>5}  {n_err:>7}  "
                  f"{s['raw_mean']:>9.3f}  {s['raw_pct_any']:>8.1f}%  "
                  f"{s['corr_mean']:>10.3f}  {s['corr_pct_any']:>9.1f}%  "
                  f"{interp}{err_note}")

    divider()
    mimic_13 = contamination_flags.get("MIMIC")
    mts_13   = contamination_flags.get("MTSamples")
    if mimic_13 is not None and mts_13 is not None:
        print(f"\n  MIMIC-IV:   {mimic_13:.1f}% of records share >=1 verbatim 13-gram with Dolma v1.7")
        print(f"  MTSamples:  {mts_13:.1f}% of records share >=1 verbatim 13-gram with Dolma v1.7")
        if mimic_13 < 10 and mts_13 >= 50:
            print("\n  CLEAN SEPARATION confirmed:")
            print("    MIMIC absent from Llama pre-training -> DP covers fine-tuning phase only.")
            print("    MTSamples contamination explains epsilon-invariant TPR@1%FPR (Tramer ICML 2024).")
    print("\n  Index: v4_dolma-v1_7_llama  |  Citation: Liu et al. (2024). arXiv:2401.17377.")


# =========================================================================
# 4. Black-box MIA — Track A
# =========================================================================

def print_mia_table(base):
    section("BLACK-BOX MIA  (Track A -- SynBench protocol, Sun et al. 2025)")
    print(f"  {'Dataset':<12}  {'Weighted':<9}  {'Epsilon':<10}  "
          f"{'AUC':<8}  {'TPR@1%FPR':<11}  {'TPR@10%FPR':<12}  Status")
    divider()

    sorted_exp = sorted(
        MIA_EXPERIMENTS,
        key=lambda r: (0 if r[0]=="MIMIC" else 1, 0 if r[1]=="No" else 1, EPS_ORDER.get(r[2], 99)),
    )
    current_group = None
    for (dataset, strat, epsilon, relpath) in sorted_exp:
        group = (dataset, strat)
        if group != current_group:
            group_header(f"{dataset} -- {'weighted' if strat=='Yes' else 'unweighted'}")
            current_group = group
        result = load_mia(Path(base) / relpath)
        eps_str = eps_label(epsilon).strip()
        if result is None:
            print(f"  {dataset:<12}  {strat:<9}  {eps_str:<10}  "
                  f"{'--':<8}  {'--':<11}  {'--':<12}  PENDING")
        else:
            flag = auc_flag(result["auc"])
            print(f"  {dataset:<12}  {strat:<9}  {eps_str:<10}  "
                  f"{result['auc']:<8.4f}  {result['tpr_1']:<11.3f}  "
                  f"{result['tpr_10']:<12.3f}  DONE{flag}")

    divider()
    print()
    print("  HIGH = AUC>=0.60  ELEV = AUC>=0.55  (no flag) = near random")
    print("  NOTE: Near-zero MIMIC AUC is NOT evidence of privacy -- MIMIC absent from Llama pre-training.")
    print("        MTSamples epsilon-invariant signal = pre-training contamination (Tramer 2024).")
    print("  NOTE: mimic_weighted_eps4 MIA not available (empty results directory).")


# =========================================================================
# 5. LiRA white-box MIA — Track A
# =========================================================================

def print_lira_table(base):
    section("LiRA WHITE-BOX MIA  (Track A -- Carlini et al. S&P 2022, offline variant)")
    print(f"  {'Epsilon':<10}  {'pct>OUT':<10}  {'ScoreMean':<11}  "
          f"{'phi_mean':<10}  {'phi_std':<9}  {'sigma_out':<10}  "
          f"{'Shadows':<9}  {'Ckpt OK':<9}  Interp")
    divider()

    group_header("WEIGHTED models -- thesis primary results")
    all_ok = True
    for (epsilon, relpath, expected_ckpt) in LIRA_EXPERIMENTS:
        result = load_lira(Path(base) / relpath, expected_ckpt)
        eps_str = eps_label(epsilon).strip()
        if result is None:
            print(f"  {eps_str:<10}  {'--':<10}  {'--':<11}  "
                  f"{'--':<10}  {'--':<9}  {'--':<10}  {'--':<9}  {'--':<9}  PENDING")
            continue
        ckpt_flag = "OK" if result["checkpoint_ok"] else "WRONG"
        if not result["checkpoint_ok"]:
            all_ok = False
        pct_str = f"{result['pct_above_out']*100:.1f}%" if result['pct_above_out'] is not None else "--"
        print(f"  {eps_str:<10}  {pct_str:<10}  "
              f"{result['score_mean']:<11.4f}  "
              f"{result['phi_mean']:<10.4f}  "
              f"{result['phi_std']:<9.4f}  "
              f"{result['global_sigma']:<10.4f}  "
              f"{result['n_shadows']:<9}  "
              f"{ckpt_flag:<9}  "
              f"{lira_flag(result['pct_above_out'])}")

    divider()
    if not all_ok:
        print("  WARNING: One or more LiRA runs used wrong checkpoint.")
    print("\n  pct>OUT >55% = memorisation signal  |  ~50% = random baseline")


# =========================================================================
# 6. Fidelity metrics — Track A
# =========================================================================

def print_fidelity_table(base):
    section("FIDELITY METRICS  (Track A -- lower KL better, higher Jaccard/MAUVE better)")
    print(f"  {'Weighting':<12}  {'Eps':<8}  "
          f"{'LenKL':<7}  {'UniJac':<8}  {'BiJac':<8}  "
          f"{'PPL':<7}  {'MAUVE':<8}  Status")
    divider()

    group_order = {"baseline": 0, "unweighted": 1, "sqrt_cap10": 2, "power_a0.3": 3, "inv-freq": 4}
    sorted_exp = sorted(
        FIDELITY_EXPERIMENTS,
        key=lambda r: (
            0 if r[0]=="MIMIC" else 1,
            group_order.get(r[4], 99),
            EPS_ORDER.get(r[2], 99),
        ),
    )

    current_group = None
    for (dataset, strat, epsilon, relpath, weighting) in sorted_exp:
        group = (dataset, weighting)
        if group != current_group:
            note = " (superseded)" if weighting in ("inv-freq",) else ""
            group_header(f"{dataset} -- {weighting}{note}")
            current_group = group

        result = load_fidelity(Path(base) / relpath)
        eps_str = eps_label(epsilon).strip()
        if result is None:
            print(f"  {'':12}  {eps_str:<8}  "
                  f"{'--':<7}  {'--':<8}  {'--':<8}  {'--':<7}  {'--':<8}  PENDING")
        else:
            mauve_str = f"{result['mauve']:.4f}" if result['mauve'] is not None else "--"
            ppl_str   = f"{result['ppl']:.1f}"   if result['ppl']   is not None else "--"
            bj_str    = f"{result['binary_jac']:.4f}" if result['binary_jac'] is not None else "--"
            print(f"  {'':12}  {eps_str:<8}  "
                  f"{result['length_kl']:<7.3f}  "
                  f"{result['unary_jac']:<8.4f}  "
                  f"{bj_str:<8}  "
                  f"{ppl_str:<7}  "
                  f"{mauve_str:<8}  DONE")

    divider()
    print()
    print("  LenKL  = KL divergence of word-count distributions (Term2Note protocol)")
    print("  UniJac = Unary term Jaccard (medical vocab overlap)")
    print("  BiJac  = Binary term Jaccard (co-occurring term pair overlap)")
    print("  PPL    = Coherence perplexity (Asclepius-Llama3-8B, lower = better clinical fluency)")
    print("  MAUVE  = Distributional similarity in Asclepius embedding space (higher = better)")


# =========================================================================
# 7. ICD adherence — single-run detail
# =========================================================================

def print_icd_adherence_table(base):
    section("ICD CATEGORY ADHERENCE DETAIL  (Track A -- Asclepius-Llama3-8B judge)")

    for (dataset, epsilon, weighting, relpath) in ICD_EXPERIMENTS:
        d = load_icd(Path(base) / relpath)
        label = f"{dataset} e={epsilon} {weighting}"
        if d is None:
            print(f"  {label}: FILE NOT FOUND ({relpath})")
            print("  NOTE: Full per-category adherence across all runs in Section 9.")
            continue

        overall = d.get("overall", {})
        print(f"\n  {label}")
        print(f"  Judge:  {d.get('judge_model', 'unknown')}")
        print(f"  Source: {d.get('synthetic_source', 'unknown')}")
        print()
        print(f"  Overall  n={overall.get('n_total', 0):,}  "
              f"strict_acc={overall.get('strict_accuracy', 0):.4f}  "
              f"decisive_acc={overall.get('decisive_accuracy', 0):.4f}")
        print(f"           YES={overall.get('n_yes', 0):,}  "
              f"NO={overall.get('n_no', 0):,}  "
              f"UNCERTAIN={overall.get('n_uncertain', 0):,}  "
              f"PARSE_ERR={overall.get('n_parse_error', 0):,}")
        print()
        print(f"  {'Category':<64}  {'N':>5}  {'YES':>5}  {'Strict':>7}  {'Decisive':>9}")
        divider(90)

        per_cat = d.get("per_category", {})
        for cat, stats in sorted(per_cat.items(), key=lambda x: x[1].get("strict_accuracy", 0), reverse=True):
            n    = stats.get("n_total", 0)
            yes  = stats.get("n_yes", 0)
            sacc = stats.get("strict_accuracy", 0)
            dacc = stats.get("decisive_accuracy", 0)
            flag = " <" if sacc >= 0.25 else ""
            print(f"  {cat[:63]:<64}  {n:>5}  {yes:>5}  {sacc:>7.4f}  {dacc:>9.4f}{flag}")

        divider(90)

    print()
    print("  strict_accuracy  = n_yes / n_total")
    print("  decisive_accuracy = n_yes / (n_yes + n_no)")
    print("  NOTE: Full per-category adherence (all 17 runs × 22 categories) in Section 9.")


# =========================================================================
# 8. Aggregate scoring summary
# =========================================================================

def print_scoring_summary(base):
    section("AGGREGATE SCORING SUMMARY  (Track A -- coherence PPL + ICD adherence, all runs)")

    rows = load_csv_rows(Path(base) / "generated/scoring_summary.csv")
    if rows is None:
        print("  generated/scoring_summary.csv not found.")
        return

    # Sort by weighting group then epsilon
    def sort_key(r):
        w = r.get("weighting", "")
        e = str(r.get("epsilon", ""))
        order = {"none": 0, "sqrt_cap10": 1, "power03_cap10": 2, "inv_freq": 3, "unknown": 4}
        return (0 if r.get("source") == "mimic" else 1,
                order.get(w, 9), EPS_ORDER.get(e, 99))

    rows_sorted = sorted(rows, key=sort_key)

    print(f"  {'Run ID':<32}  {'ε':>6}  {'Weighting':<16}  "
          f"{'n':>5}  {'Coh PPL':>8}  {'Adherence':>10}  Flags")
    divider()

    for r in rows_sorted:
        rid    = r.get("run_id", "")
        eps    = str(r.get("epsilon", "--"))
        wt     = str(r.get("weighting", "--"))
        n      = r.get("n_scored", "--")
        ppl    = r.get("mean_coherence_ppl", "")
        adh    = r.get("mean_adherence", "")
        flags  = r.get("flagged", "")
        ppl_s  = f"{float(ppl):.1f}" if ppl else "--"
        adh_s  = f"{float(adh):.3f}" if adh else "--"
        flag_s = f"  ← {flags}" if flags else ""
        print(f"  {rid:<32}  {eps:>6}  {wt:<16}  "
              f"{n:>5}  {ppl_s:>8}  {adh_s:>10}{flag_s}")

    divider()
    print()
    print("  Coh PPL   = perplexity of synthetic note under Asclepius-Llama3-8B (lower = better)")
    print("  Adherence = Asclepius judge 0–1 score (does note match ICD-10 label?)")
    print("  NOTE: n≈43-50/run (stratified sample). Full per-category breakdown in Section 9.")


# =========================================================================
# 9. Per-category adherence — all runs
# =========================================================================

def print_per_category_adherence(base):
    section("PER-CATEGORY ADHERENCE  (Track A -- 50 samples/category × 17 runs × 22 ICD chapters)")

    rows = load_csv_rows(Path(base) / "eval/per_category_adherence.csv")
    if rows is None:
        print("  eval/per_category_adherence.csv not found.")
        print("  Job per_cat_eval (21220) is running — results will appear here when complete.")
        return

    # Pivot: run × category → mean_adherence
    runs = sorted({r["run_id"] for r in rows})
    cats = sorted({r["category"] for r in rows})
    data = {(r["run_id"], r["category"]): r for r in rows}

    # Print heatmap-style table: categories as rows, runs as columns (abbreviated)
    run_abbr = {r: r[:18] for r in runs}
    print(f"\n  {'Category':<52}  " + "  ".join(f"{run_abbr[r]:>7}" for r in runs))
    divider(52 + 9 * len(runs))

    for cat in cats:
        row_vals = []
        for run in runs:
            cell = data.get((run, cat))
            if cell:
                v = float(cell["mean_adherence"])
                row_vals.append(f"{v:>7.3f}")
            else:
                row_vals.append(f"{'--':>7}")
        print(f"  {cat[:51]:<52}  " + "  ".join(row_vals))

    divider(52 + 9 * len(runs))
    print()
    print("  Values: mean adherence 0.0–1.0 (Asclepius judge, 50 notes/cell)")
    print("  0.6+ = match  |  0.3–0.6 = partial  |  <0.3 = mismatch")

    # Summary: best/worst categories per run group
    print()
    print("  Worst-performing categories (mean_adherence across all runs):")
    cat_means = {}
    for cat in cats:
        vals = [float(data[(r, cat)]["mean_adherence"])
                for r in runs if (r, cat) in data]
        if vals:
            cat_means[cat] = statistics.mean(vals)
    for cat, mean in sorted(cat_means.items(), key=lambda x: x[1])[:5]:
        print(f"    {cat:<55}  avg={mean:.3f}")


# =========================================================================
# 10. Per-category MAUVE — all runs
# =========================================================================

def print_per_category_mauve(base):
    section("PER-CATEGORY MAUVE  (Track A -- Asclepius featurizer × 17 runs × 21 ICD chapters)")

    rows = load_csv_rows(Path(base) / "eval/per_category_mauve.csv")
    if rows is None:
        print("  eval/per_category_mauve.csv not found.")
        print("  Job per_cat_eval (21220) is running — results will appear here when complete.")
        return

    runs = sorted({r["run_id"] for r in rows})
    cats = sorted({r["category"] for r in rows})
    data = {(r["run_id"], r["category"]): r for r in rows}

    run_abbr = {r: r[:18] for r in runs}
    print(f"\n  {'Category':<52}  " + "  ".join(f"{run_abbr[r]:>7}" for r in runs))
    divider(52 + 9 * len(runs))

    for cat in cats:
        row_vals = []
        for run in runs:
            cell = data.get((run, cat))
            if cell and cell.get("mauve_score") not in (None, "", "None"):
                v = float(cell["mauve_score"])
                row_vals.append(f"{v:>7.4f}")
            else:
                row_vals.append(f"{'--':>7}")
        print(f"  {cat[:51]:<52}  " + "  ".join(row_vals))

    divider(52 + 9 * len(runs))
    print()
    print("  MAUVE: higher = synthetic distribution closer to real MIMIC for that ICD chapter")
    print("  'External Causes of Morbidity' excluded (only 1 note — below min_mauve_n=30)")

    # Worst categories
    print()
    print("  Lowest MAUVE categories (hardest to replicate, averaged across runs):")
    cat_means = {}
    for cat in cats:
        vals = [float(data[(r, cat)]["mauve_score"])
                for r in runs
                if (r, cat) in data and data[(r, cat)].get("mauve_score") not in (None, "", "None")]
        if vals:
            cat_means[cat] = statistics.mean(vals)
    for cat, mean in sorted(cat_means.items(), key=lambda x: x[1])[:5]:
        print(f"    {cat:<55}  avg={mean:.4f}")


# =========================================================================
# 11. Track B — unlearning MIA
# =========================================================================

def print_track_b_unlearn_mia_table(base):
    section("TRACK B -- UNLEARNING MIA  (tier-1 attacks on unlearn checkpoints)")

    tb_base = Path(base) / TRACK_B_REPO
    if not tb_base.exists():
        print(f"  Track B repo not found: {tb_base}")
        return

    PRIMARY_ATTACKS = [
        ("loss_vs_samedist_nonmem",        "loss/same"),
        ("loss_vs_clean_nonmem",           "loss/clean"),
        ("min_k_20_vs_samedist_nonmem",    "MinK20/same"),
        ("min_k_pp_20_vs_samedist_nonmem", "MinK++/same"),
        ("ref_ratio_vs_samedist_nonmem",   "RefRatio/same"),
    ]

    header_atk = "  ".join(f"{'AUC':>6} {'T@1%':>5}" for _ in PRIMARY_ATTACKS)
    label_atk  = "  ".join(f"{lbl:<12}" for (_, lbl) in PRIMARY_ATTACKS)
    print(f"  {'Method':<8}  {'Data':<10}  {header_atk}  Status")
    print(f"  {'':8}  {'':10}  {label_atk}")
    divider()

    for (method, dataset, dir_name) in TRACK_B_UNLEARN_DIRS:
        tier1_path = tb_base / "outputs" / "attacks" / dir_name / "tier1" / "tier1_results.json"
        attacks = load_track_b_tier1(tier1_path)

        row_parts = []
        for (atk_key, _) in PRIMARY_ATTACKS:
            if attacks is None:
                row_parts.append(f"{'--':>6} {'--':>5}")
            else:
                r = attacks.get(atk_key)
                if r is None:
                    row_parts.append(f"{'--':>6} {'--':>5}")
                else:
                    auc = r.get("auc", 0)
                    tpr = r.get("tpr_at_1pct_fpr", 0)
                    flag = "*" if auc >= 0.55 else " "
                    row_parts.append(f"{auc:>6.4f} {tpr:>5.3f}{flag}")

        status = "DONE" if attacks is not None else "PENDING"
        print(f"  {method:<8}  {dataset:<10}  {'  '.join(row_parts)}  {status}")

    divider()
    print()
    print("  AUC near 0.50 = attacker at chance (unlearning effective for MIA)")
    print("  AUC >> 0.50   = membership still detectable (unlearning incomplete)")
    print("  * = AUC >= 0.55 (elevated signal)")
    print("  Cohorts: members vs. same-dist-nonmembers / vs. clean-nonmembers")


# =========================================================================
# 12. Track B — DP-LoRA catalog (loss table)
# =========================================================================

def print_track_b_catalog(base):
    section("TRACK B -- DP-LoRA CATALOG  (final training loss — key finding)")

    catalog = load_track_b_catalog(base)
    if catalog is None:
        print("  results/experiment_catalog.json not found.")
        return

    rows = catalog.get("dp_lora", [])
    if not rows:
        print("  No dp_lora rows in catalog.")
        return

    print(f"  {'Tag':<35}  {'ε':>5}  {'Method':<8}  {'Dataset':<10}  "
          f"{'First loss':>10}  {'Final loss':>10}  {'ε spent':>8}  Note")
    divider()

    # Group by method for readability
    for method_filter in ["base", "ga", "ga_gd", "npo", "rmu"]:
        method_rows = [r for r in rows
                       if not r.get("parse_error") and r.get("method") == method_filter]
        if not method_rows:
            continue
        group_header(f"Method: {method_filter.upper()}")
        for r in sorted(method_rows,
                        key=lambda x: (x.get("dataset",""), EPS_ORDER.get(str(x.get("epsilon_str","")), 99))):
            fl = f"{r['first_loss']:.2f}" if r.get("first_loss") is not None else "N/A"
            ll = f"{r['final_loss']:.2f}" if r.get("final_loss") is not None else "N/A"
            es = f"{r['final_eps_spent']:.3f}" if r.get("final_eps_spent") is not None else "N/A"
            # Flag stuck models
            note = ""
            if r.get("final_loss") is not None and r.get("epsilon_str") != "inf":
                if r["final_loss"] > 20:
                    note = "STUCK"
                elif r["final_loss"] < 8:
                    note = "recovered"
            print(f"  {r['tag']:<35}  {str(r.get('epsilon_str','')):>5}  "
                  f"{r.get('method',''):<8}  {r.get('dataset',''):<10}  "
                  f"{fl:>10}  {ll:>10}  {es:>8}  {note}")

    divider()
    print()
    print("  KEY FINDING: Only RMU achieves final_loss < 8 at finite ε.")
    print("  GA/NPO/GA+GD final_loss 65–94 at finite ε = DP clipping cannot repair gradient ascent.")
    print("  All methods recover (loss ~1.4–2.1) at ε=∞ as expected.")


# =========================================================================
# 13. Track B — DP-LoRA fidelity
# =========================================================================

def print_track_b_fidelity_table(base):
    section("TRACK B -- DP-LoRA FIDELITY  (RMU unlearn -> DP re-tune -> generate -> eval)")

    tb_base  = Path(base) / TRACK_B_REPO
    fid_dir  = tb_base / "outputs" / "fidelity"

    print(f"  {'Method':<8}  {'Data':<10}  {'Eps':<6}  "
          f"{'LenKL':<7}  {'UniJac':<8}  {'PPL':<7}  {'MAUVE':<8}  Status")
    divider()

    for (method, dataset, epsilon, tag) in TRACK_B_FIDELITY_TAGS:
        fid_path = fid_dir / f"{tag}_fidelity.json"
        result   = load_track_b_fidelity(fid_path)
        eps_str  = f"e={epsilon}" if epsilon != "inf" else "e=inf"
        if result is None:
            print(f"  {method:<8}  {dataset:<10}  {eps_str:<6}  "
                  f"{'--':<7}  {'--':<8}  {'--':<7}  {'--':<8}  PENDING")
        else:
            mauve_str = f"{result['mauve']:.4f}" if result['mauve'] is not None else "--"
            ppl_str   = f"{result['ppl']:.1f}"   if result['ppl']   is not None else "--"
            print(f"  {method:<8}  {dataset:<10}  {eps_str:<6}  "
                  f"{result['length_kl']:<7.3f}  "
                  f"{result['unary_jac']:<8.4f}  "
                  f"{ppl_str:<7}  "
                  f"{mauve_str:<8}  DONE")

    divider()
    print()
    print("  NOTE: GA/NPO/GA+GD stuck at final_loss 65–94 at finite ε — fidelity not meaningful.")
    print("        RMU recovers to loss ~5.7 at all finite ε — fidelity eval is valid.")
    print("        Generation jobs: 21202-21209  |  Fidelity eval jobs: 21210-21211, 21221-21226")




# =========================================================================
# 14. Track B — AI safety eval suite
# =========================================================================

def print_track_b_eval_suite(base):
    section("TRACK B -- AI SAFETY EVAL SUITE  (KS test + re-learn attack + MMLU-Medical)")

    tb_base = Path(base) / TRACK_B_REPO
    eval_dir = tb_base / "outputs" / "eval"

    # Header
    print(f"  {'Stage':<10}  {'Data':<10}  {'ε':>5}  │  "
          f"{'KS p':>8}  {'Verdict':<6}  │  "
          f"{'FgtNLL':>7}  {'CtlNLL':>7}  │  "
          f"{'PPLpre':>7}  {'PPLpost':>8}  {'Rcvy%':>7}  {'RL Verdict':<10}  │  "
          f"{'MMLU':>6}  {'CK':>5}  {'PM':>5}  {'AN':>5}")
    divider(130)

    for dataset_label in ("MTSamples", "PMC"):
        group_header(f"Forget set: {dataset_label}")
        for (stage, dataset, epsilon, tag) in TRACK_B_EVAL_SUITE_TAGS:
            if dataset != dataset_label:
                continue
            r = load_eval_suite(eval_dir / tag / "eval_suite_results.json")
            eps_str = epsilon if epsilon != "—" else "—"

            if r is None:
                print(f"  {stage:<10}  {dataset:<10}  {eps_str:>5}  │  "
                      f"{'—':>8}  {'—':<6}  │  "
                      f"{'—':>7}  {'—':>7}  │  "
                      f"{'—':>7}  {'—':>8}  {'—':>7}  {'—':<10}  │  "
                      f"{'—':>6}  {'—':>5}  {'—':>5}  {'—':>5}  PENDING")
                continue

            # KS
            ks_p = f"{r['ks_p']:.4f}" if r['ks_p'] is not None else "—"
            ks_v = (r['ks_verdict'] or "—")[:6]
            fgt  = f"{r['forget_nll']:.2f}" if r['forget_nll'] is not None else "—"
            ctl  = f"{r['control_nll']:.2f}" if r['control_nll'] is not None else "—"

            # Re-learn
            if r['rl_error']:
                rl_pre = f"{r['rl_ppl_before']:.1f}" if r['rl_ppl_before'] else "—"
                rl_post = "ERROR"
                rl_rcv  = "—"
                rl_v    = "ERROR"
            elif r['rl_ppl_before'] is not None:
                rl_pre  = f"{r['rl_ppl_before']:.1f}"
                rl_post = f"{r['rl_ppl_after']:.1f}"
                rl_rcv  = f"{r['rl_recovery']:.0f}%"
                rl_v    = r['rl_verdict'] or "—"
                if stage == "preDp" and rl_v == "ROBUST":
                    rl_v = "ROBUST*"
            else:
                rl_pre = rl_post = rl_rcv = rl_v = "—"

            # MMLU
            mmlu = f"{r['mmlu_mean']:.3f}" if r['mmlu_mean'] is not None else "—"
            ck   = f"{r['mmlu_ck']:.2f}" if r['mmlu_ck'] is not None else "—"
            pm   = f"{r['mmlu_pm']:.2f}" if r['mmlu_pm'] is not None else "—"
            an   = f"{r['mmlu_an']:.2f}" if r['mmlu_an'] is not None else "—"

            print(f"  {stage:<10}  {dataset:<10}  {eps_str:>5}  │  "
                  f"{ks_p:>8}  {ks_v:<6}  │  "
                  f"{fgt:>7}  {ctl:>7}  │  "
                  f"{rl_pre:>7}  {rl_post:>8}  {rl_rcv:>7}  {rl_v:<10}  │  "
                  f"{mmlu:>6}  {ck:>5}  {pm:>5}  {an:>5}")

    divider(130)
    print()
    print("  KS test:  Two-sample Kolmogorov-Smirnov on per-example NLL (forget vs control, n=300 each)")
    print("            SCAR_DETECTED at baseline → pre-existing distributional gap (not caused by training)")
    print("  Re-learn: 50-step fine-tune on 5% forget set; negative recovery% = model resists re-memorisation")
    print("            ROBUST* = RMU pre-DP: technically robust, but only because model collapsed (PPL 73→1540)")
    print("  MMLU:     Zero-shot MCQ on clinical_knowledge (CK), professional_medicine (PM), anatomy (AN)")
    print("            ~535 total questions from cais/mmlu; baseline Llama-3.1-8B-Instruct ≈ 0.734")
    print("  ERROR:    dp_lora_epsinf_rmu_pmc re-learn failed (CUDA OOM during 50-step fine-tune)")


# =========================================================================


# =========================================================================
# 15. Track B -- post-DP MIA on deduped splits_v2
# =========================================================================

def print_track_b_post_dp_mia_table(base):
    section("TRACK B -- POST-DP MIA  (deduped splits_v2, tier-1 attacks)")

    tb_base = Path(base) / TRACK_B_REPO

    PRIMARY_ATTACKS = [
        ("loss_vs_samedist_nonmem",        "loss/same"),
        ("loss_vs_clean_nonmem",           "loss/clean"),
        ("min_k_20_vs_samedist_nonmem",    "MinK20/same"),
        ("min_k_pp_20_vs_samedist_nonmem", "MinK++/same"),
    ]

    header_atk = "  ".join(f"{'AUC':>6} {'T@1%':>5}" for _ in PRIMARY_ATTACKS)
    label_atk  = "  ".join(f"{lbl:<12}" for (_, lbl) in PRIMARY_ATTACKS)
    print(f"  {'Stage':<10}  {'Data':<10}  {'eps':>5}  {header_atk}  Status")
    print(f"  {'':10}  {'':10}  {'':>5}  {label_atk}")
    divider(120)

    for dataset_label in ("MTSamples", "PMC"):
        group_header(f"Forget set: {dataset_label}")
        for (stage, dataset, epsilon, tag) in TRACK_B_POST_DP_MIA_TAGS:
            if dataset != dataset_label:
                continue
            tier1_path = tb_base / "outputs" / "attacks" / tag / "tier1" / "tier1_results.json"
            attacks = load_track_b_tier1(tier1_path)
            eps_str = epsilon

            row_parts = []
            for (atk_key, _) in PRIMARY_ATTACKS:
                if attacks is None:
                    row_parts.append(f"{'--':>6} {'--':>5}")
                else:
                    r = attacks.get(atk_key)
                    if r is None:
                        row_parts.append(f"{'--':>6} {'--':>5}")
                    else:
                        auc = r.get("auc", 0)
                        tpr = r.get("tpr_at_1pct_fpr", 0)
                        flag = "*" if auc >= 0.55 else " "
                        row_parts.append(f"{auc:>6.4f} {tpr:>5.3f}{flag}")

            status = "DONE" if attacks is not None else "PENDING"
            print(f"  {stage:<10}  {dataset:<10}  {eps_str:>5}  {'  '.join(row_parts)}  {status}")

        divider(120)

    print()
    print("  Splits v2: deduplicated (1524 members, 439 nonmembers, zero overlap)")
    print("  noUnlearn = dp_lora directly on base model (no RMU step)")
    print("  Key finding: same-dist AUC ~0.52 at all finite epsilon (attacker at chance)")
    print("               noUnlearn matches RMU pipeline => unlearning redundant for same-dist MIA")
    print("               epsilon=inf elevates clean AUC to ~0.98 (DP noise IS the protective mechanism)")
    print("  * = AUC >= 0.55 (elevated signal)")


# =========================================================================
# 16. Track B -- prefix completion extraction attack
# =========================================================================

def print_track_b_completion_attack(base):
    section("TRACK B -- COMPLETION EXTRACTION ATTACK  (prefix=50%, greedy decode)")

    tb_base = Path(base) / TRACK_B_REPO
    atk_dir = tb_base / "outputs" / "attacks"

    print(f"  {'Stage':<10}  {'Data':<10}  {'eps':>5}  |  "
          f"{'EMR':>6}  {'p95EMR':>6}  {'maxEMR':>6}  |  "
          f"{'ROUGE-L':>7}  |  "
          f"{'ext10%':>6}  {'ext25%':>6}  {'ext50%':>6}  |  "
          f"{'n':>5}")
    divider(110)

    for dataset_label in ("MTSamples", "PMC"):
        group_header(f"Forget set: {dataset_label}")
        for (stage, dataset, epsilon, tag) in TRACK_B_COMPLETION_TAGS:
            if dataset != dataset_label:
                continue
            r = load_completion_attack(atk_dir / tag / "completion_attack_results.json")
            eps_str = epsilon

            if r is None:
                print(f"  {stage:<10}  {dataset:<10}  {eps_str:>5}  |  "
                      f"{'--':>6}  {'--':>6}  {'--':>6}  |  "
                      f"{'--':>7}  |  "
                      f"{'--':>6}  {'--':>6}  {'--':>6}  |  "
                      f"{'--':>5}  PENDING")
                continue

            emr     = f"{r['mean_exact_match_rate']:.4f}"
            p95     = f"{r['p95_exact_match_rate']:.4f}"
            mx      = f"{r['max_exact_match_rate']:.4f}"
            rl      = f"{r['mean_rouge_l']:.4f}"
            e10     = f"{r['extraction_rate_10pct']:.4f}"
            e25     = f"{r['extraction_rate_25pct']:.4f}"
            e50     = f"{r['extraction_rate_50pct']:.4f}"
            n       = str(r['n_examples'])

            print(f"  {stage:<10}  {dataset:<10}  {eps_str:>5}  |  "
                  f"{emr:>6}  {p95:>6}  {mx:>6}  |  "
                  f"{rl:>7}  |  "
                  f"{e10:>6}  {e25:>6}  {e50:>6}  |  "
                  f"{n:>5}")

        divider(110)

    print()
    print("  EMR = mean exact match rate (fraction of continuation tokens reproduced)")
    print("  ROUGE-L = longest common subsequence ratio (token-level)")
    print("  extK% = fraction of examples where >= K% of continuation is exact")
    print("  Carlini et al. (2021) protocol: greedy decode from 50% prefix")
    print("  Key finding: RMU reduces extraction ~56% (EMR 0.026->0.011), but DP re-learning")
    print("               restores baseline extraction rates (EMR ~0.026 at all epsilon)")


# =========================================================================
# 17. Cross-track synthesis & key findings
# =========================================================================

def print_synthesis(base):
    section("CROSS-TRACK SYNTHESIS & KEY FINDINGS")

    print("""
  ┌─────────────────────────────────────────────────────────────────────────────────────────┐
  │  PIPELINE: base LLM → RMU unlearning → DP-LoRA re-tuning → generation → evaluation    │
  │  Model:    Llama-3.1-8B-Instruct                                                       │
  │  Forget:   MTSamples (medical transcriptions) and PMC (biomedical abstracts)            │
  └─────────────────────────────────────────────────────────────────────────────────────────┘

  FINDING 1 — Gradient-clipping incompatibility (Section 12)
  ──────────────────────────────────────────────────────────
    GA, GA+GD, NPO inflate per-example loss to 65–94 after unlearning.
    DP-SGD clips per-example gradients to max_grad_norm, so the enormous gradients
    from loss-inflated checkpoints are clipped to near-zero effective signal.
    Result: final_loss stays 65–94 at all finite ε (training cannot converge).
    Only RMU (representation misdirection) preserves loss near baseline (~5.7),
    allowing DP-LoRA to converge (final_loss ≈ 5.7 at ε=1,3,8).

  FINDING 2 — Full baseline recovery after RMU + DP-LoRA (Section 14)
  ────────────────────────────────────────────────────────────────────
    After DP-LoRA re-tuning at any finite ε:
      • Forget-set NLL returns to baseline (2.09 vs 2.09 baseline on MTSamples)
      • MMLU-Medical returns to 0.731–0.739 (baseline = 0.734)
      • Re-learn attack verdict = ROBUST across all finite ε
    This means the DP-LoRA step fully repairs the RMU disruption while operating
    under formal (ε,δ)-DP guarantees for the re-tuning phase.

  FINDING 3 — DP noise as regularisation (Section 14)
  ───────────────────────────────────────────────────
    ε=∞ (no DP noise) produces WORSE MMLU than finite ε:
      MTSamples: ε=∞ → MMLU 0.601  vs  ε=1 → MMLU 0.737
    Hypothesis: DP noise prevents overfitting during the post-unlearning rebuild,
    acting as implicit regularisation. This is counterintuitive — more noise = better
    general reasoning — and may be the strongest novel contribution.

  FINDING 4 — MIA at chance after unlearning (Section 11)
  ──────────────────────────────────────────────────────
    All 8 unlearn checkpoints (GA, GA+GD, NPO, RMU × 2 datasets):
      AUC = 0.48–0.51 across loss, Min-K%20, Min-K%10, zlib, ref_ratio, Min-K%++
      Attacker at chance → membership signal erased by all four unlearning methods.
    Canary detection AUC = 0.83–1.00 (sanity check: planted canaries remain distinguishable).

  FINDING 5 — Pre-existing KS scar in baseline (Section 14)
  ─────────────────────────────────────────────────────────
    Baseline model (never fine-tuned) shows SCAR_DETECTED (KS p=0.0002 on MTSamples,
    p=0.016 on PMC). This means the forget/control splits have inherent distributional
    differences. All KS verdicts must be interpreted RELATIVE to baseline, not in absolute
    terms. Post-DP models match baseline KS statistics (p=0.0003–0.027).

  CAVEAT — Pipeline is NOT end-to-end (ε,δ)-DP
  ─────────────────────────────────────────────
    RMU unlearning is a heuristic (not a DP mechanism). The formal ε guarantee applies
    ONLY to the DP-LoRA re-tuning phase. The overall pipeline provides:
      (1) Heuristic membership signal erasure (MIA AUC ≈ 0.50 post-unlearning)
      (2) Formal DP guarantee for the adaptation phase
      (3) No formal composition of (1) and (2)
    Do not overclaim end-to-end DP protection.

    FINDING 6 -- Same-dist MIA at chance throughout entire pipeline (Section 15)
  ---------------------------------------------------------------------------
    Post-DP MIA on deduped splits (v2, zero member/nonmember overlap):
      All same-dist AUC = 0.51-0.52 at finite eps (attacker at chance).
      dp_lora_base (no unlearning) = dp_lora_rmu at every finite eps.
      -> Unlearning is redundant for same-distribution MIA defense.
      -> DP-LoRA alone provides full same-dist protection.
    Only eps=inf elevates signal: clean AUC rises to 0.975-0.980.

  FINDING 7 -- RMU extraction reduction undone by DP re-learning (Section 16)
  ---------------------------------------------------------------------------
    Prefix completion attack (Carlini et al. 2021 protocol):
      Baseline EMR=0.026 -> RMU cuts to 0.011 (-56%)
      But after DP-LoRA: EMR returns to ~0.026 at ALL epsilon (1, 8, inf).
      RMU genuinely reduces memorisation, but DP re-tuning restores it.
    This is the critical negative result: the unlearn->retune pipeline does NOT
    reduce verbatim extraction. The model re-memorises the same training data
    during DP-LoRA, regardless of privacy budget.

  STATUS -- Still pending:
    * U-LiRA at eps=8 on RMU checkpoints (shadow training required)
    * Fidelity metrics on generated text (MAUVE, PPL, n-gram overlap)
    * TSTR utility evaluation
    * N-gram extraction (clinical headline metric)
    * Goldfish loss training + completion attacks (running)
    * PMC completion attacks (running)
""")

# =========================================================================
# Main
# =========================================================================

def main(args):
    base = args.base_dir
    print()
    print("  +" + "=" * 90 + "+")
    print("  |  THESIS RESULTS SUMMARY — Differentially Private Synthetic Clinical Text Generation   |")
    print("  |  Daniel Doyon — Hofstra University M.S. Data Science                                  |")
    print("  +" + "=" * 90 + "+")

    print_generation_stats(base)
    print_checkpoint_audit(base)
    print_infigram_table(base)
    print_mia_table(base)
    print_lira_table(base)
    print_fidelity_table(base)
    print_icd_adherence_table(base)
    print_scoring_summary(base)
    print_per_category_adherence(base)
    print_per_category_mauve(base)
    print_track_b_unlearn_mia_table(base)
    print_track_b_catalog(base)
    print_track_b_fidelity_table(base)
    print_track_b_eval_suite(base)
    print_track_b_post_dp_mia_table(base)
    print_track_b_completion_attack(base)
    print_synthesis(base)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print all thesis experiment results to terminal",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base_dir", default=".",
        help="Project root (containing mia/, eval/, generated/, models/, audit/, Scripts_3.0/)"
    )
    args = parser.parse_args()
    main(args)

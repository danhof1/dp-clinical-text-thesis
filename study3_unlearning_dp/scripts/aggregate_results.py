"""
Aggregate all experimental results into one table and emit the plots
that go into the paper.

Consumes:
  - $ATTACK_ROOT/*/tier1/tier1_results.json
  - $ATTACK_ROOT/*/extraction/extraction_n*.json
  - $ATTACK_ROOT/*/lira_online/lira_online.json   (if present)
  - $ATTACK_ROOT/*/ulira/lira_online_ulira.json   (if present)
  - $FIDELITY_ROOT/*/fidelity_report.json
  - $UTILITY_ROOT/*/utility_report.json

Produces:
  - results_table.csv     (one row per (source, ε) cell)
  - privacy_utility.png   (tradeoff plot)
  - extraction_by_eps.png (extraction rate vs ε, by source condition)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("aggregate")


TAG_PATTERN = re.compile(r"^dp_lora_eps(inf|\d+)_(.+)$")  # fixed: eps=inf|digits; rest=full method name


def parse_tag(tag: str) -> tuple[str, str]:
    """dp_lora_eps8_unl_ga_gd -> (eps='8', source='unl_ga_gd')"""
    m = TAG_PATTERN.match(tag)
    if not m:
        return (tag, "unknown")
    eps, source = m.group(1), m.group(2)
    return eps, source


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def row_for_tag(
    tag: str,
    attack_root: Path,
    fidelity_root: Path,
    utility_root: Path,
) -> dict | None:
    eps, source = parse_tag(tag)
    row = {"tag": tag, "epsilon": eps, "source": source}

    tier1 = attack_root / tag / "tier1" / "tier1_results.json"
    if not tier1.exists():
        return None
    tier1_res = load_json(tier1)
    for r in tier1_res:
        # collapse "{attack}_vs_{cohort}" into two columns
        name = r["attack"]
        row[f"{name}_auc"] = r["auc"]
        row[f"{name}_tpr@1"] = r["tpr_at_1pct_fpr"]
        row[f"{name}_tpr@0.1"] = r["tpr_at_0p1pct_fpr"]

    # Extraction (one summary per n)
    ext_dir = attack_root / tag / "extraction"
    if ext_dir.exists():
        for f in ext_dir.glob("extraction_n*.json"):
            ext = load_json(f)
            n = ext["n"]
            row[f"extract_n{n}_frac_samples_with_hits"] = ext["frac_samples_with_hits"]
            row[f"extract_n{n}_mean_hits"] = ext["mean_hits_per_sample"]

    # LiRA (optional)
    lira_online = attack_root / tag / "lira_online" / "lira_online.json"
    if lira_online.exists():
        lira = load_json(lira_online)
        for r in lira:
            row[f"lira_{r['attack']}_auc"] = r["auc"]
            row[f"lira_{r['attack']}_tpr@1"] = r["tpr_at_1pct_fpr"]

    # U-LiRA (optional)
    ulira = attack_root / tag / "ulira" / "lira_online_ulira.json"
    if ulira.exists():
        ulr = load_json(ulira)
        for r in ulr:
            row[f"ulira_{r['attack']}_auc"] = r["auc"]
            row[f"ulira_{r['attack']}_tpr@1"] = r["tpr_at_1pct_fpr"]

    # Fidelity
    fid_path = fidelity_root / tag / "fidelity_report.json"
    if fid_path.exists():
        fid = load_json(fid_path)
        if "mauve" in fid:
            row["mauve"] = fid["mauve"]["mauve"]
        if "ner_jsd" in fid:
            row["ner_jsd"] = fid["ner_jsd"]
        slen = fid.get("synthetic_length_stats", {})
        row["synth_mean_tokens"] = slen.get("mean_tokens")
        row["synth_ttr"] = slen.get("type_token_ratio_first_1k")

    # Utility
    util_path = utility_root / tag / "utility_report.json"
    if util_path.exists():
        util = load_json(util_path)
        row["downstream_macro_f1"] = util.get("macro_f1")

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", required=True, help="experiment root (EXP_ROOT)")
    ap.add_argument("--output", default=None, help="CSV output; defaults to exp_root/results_table.csv")
    args = ap.parse_args()

    exp = Path(args.exp_root)
    attack_root = exp / "attacks"
    fidelity_root = exp / "fidelity"
    utility_root = exp / "utility"

    tags = sorted([p.name for p in attack_root.iterdir() if p.is_dir() and p.name != "base_model"])
    log.info("found %d tags", len(tags))

    rows = []
    for tag in tags:
        r = row_for_tag(tag, attack_root, fidelity_root, utility_root)
        if r is not None:
            rows.append(r)

    df = pd.DataFrame(rows)
    out = Path(args.output or (exp / "results_table.csv"))
    df.to_csv(out, index=False)
    log.info("wrote %d rows to %s", len(df), out)

    # Try plotting if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Privacy-utility tradeoff: downstream F1 vs extraction rate, colored by source
        if "downstream_macro_f1" in df.columns and "extract_n50_frac_samples_with_hits" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 5))
            for src, sub in df.groupby("source"):
                ax.scatter(
                    sub["extract_n50_frac_samples_with_hits"],
                    sub["downstream_macro_f1"],
                    label=src, s=60,
                )
                for _, row in sub.iterrows():
                    ax.annotate(f"ε={row['epsilon']}",
                                (row["extract_n50_frac_samples_with_hits"],
                                 row["downstream_macro_f1"]),
                                fontsize=8, xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel("fraction of synthetic notes containing ≥1 50-gram leak")
            ax.set_ylabel("downstream macro-F1 on real held-out")
            ax.legend()
            ax.set_title("Privacy–utility tradeoff")
            fig.tight_layout()
            fig.savefig(exp / "privacy_utility.png", dpi=150)
            log.info("wrote %s", exp / "privacy_utility.png")

        # Extraction vs ε line plot
        if "extract_n50_frac_samples_with_hits" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 5))
            # sort ε as if numeric ("inf" -> large)
            def eps_sort_key(e):
                if e == "inf":
                    return 1e9
                try:
                    return float(e)
                except ValueError:
                    return -1.0
            df2 = df.copy()
            df2["_eps_sort"] = df2["epsilon"].map(eps_sort_key)
            df2 = df2.sort_values("_eps_sort")
            for src, sub in df2.groupby("source"):
                ax.plot(sub["epsilon"], sub["extract_n50_frac_samples_with_hits"],
                        marker="o", label=src)
            ax.set_xlabel("ε")
            ax.set_ylabel("50-gram leak rate in synthetic output")
            ax.set_title("Verbatim leakage by ε and source condition")
            ax.legend()
            fig.tight_layout()
            fig.savefig(exp / "extraction_by_eps.png", dpi=150)
            log.info("wrote %s", exp / "extraction_by_eps.png")

    except ImportError:
        log.warning("matplotlib not available; skipping plots")


if __name__ == "__main__":
    main()

"""
catalog_experiments.py  —  lightweight status catalog for Track B experiments.

Scans outputs/ and logs/ to report every completed dp_lora adapter and
unlearn checkpoint, including the final training loss extracted from SLURM logs.

Produces:  results/experiment_catalog.json
           results/experiment_catalog.csv

Does NOT require attack/fidelity/utility results (those come later via
aggregate_results.py once MIA attacks have run).

Usage:
    python3 -m scripts.catalog_experiments --repo_root <REPO>
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import glob
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("catalog")

# Fixed pattern: eps is EITHER "inf" OR digits only (not greedy \w+)
TAG_PATTERN = re.compile(r"^dp_lora_eps(inf|\d+)_(.+)$")
EPS_DISPLAY = {"1": 1.0, "3": 3.0, "8": 8.0, "inf": float("inf")}

LOG_DIR = Path("/fs1/projects/unlearning_pretraining/Proj_code/logs")
LOSS_RE = re.compile(r"step=\d+ loss=([\d.]+)")
EPS_SPENT_RE = re.compile(r"eps_spent=([\d.]+)")


def parse_tag(tag: str) -> dict:
    m = TAG_PATTERN.match(tag)
    if not m:
        return {"tag": tag, "epsilon_str": None, "epsilon": None, "method": None,
                "dataset": None, "parse_error": True}
    eps_str, rest = m.group(1), m.group(2)
    # Split off _pmc suffix if present
    if rest.endswith("_pmc"):
        method = rest[:-4]
        dataset = "pmc"
    else:
        method = rest
        dataset = "mtsamples"
    return {
        "tag": tag,
        "epsilon_str": eps_str,
        "epsilon": EPS_DISPLAY.get(eps_str, eps_str),
        "method": method,
        "dataset": dataset,
        "parse_error": False,
    }


def find_log(tag: str) -> Path | None:
    """Find SLURM log for a dp_lora job.

    Logs are named dplora_{JOBID}_dplora_{eps}_{method}.log
    e.g. tag=dp_lora_eps1_ga_gd -> dplora_*_dplora_eps1_ga_gd.log
    """
    # strip "dp_lora_" prefix to get the job-name suffix
    suffix = tag.removeprefix("dp_lora_")
    pattern = str(LOG_DIR / f"dplora_*_dplora_{suffix}.log")
    hits = sorted(glob.glob(pattern))
    return Path(hits[-1]) if hits else None


def extract_loss_from_log(log_path: Path) -> dict:
    """Return first_loss, final_loss, final_eps_spent from a dp_lora log."""
    result = {"first_loss": None, "final_loss": None, "final_eps_spent": None,
              "total_steps": 0}
    try:
        with open(log_path) as f:
            lines = [l for l in f if "step=" in l and "loss=" in l]
        if not lines:
            return result
        m0 = LOSS_RE.search(lines[0])
        ml = LOSS_RE.search(lines[-1])
        me = EPS_SPENT_RE.search(lines[-1])
        if m0:
            result["first_loss"] = float(m0.group(1))
        if ml:
            result["final_loss"] = float(ml.group(1))
        if me:
            result["final_eps_spent"] = float(me.group(1))
        result["total_steps"] = len(lines)
    except Exception:
        pass
    return result


def catalog_dp_lora(repo: Path) -> list[dict]:
    outputs = repo / "outputs"
    rows = []
    for adapter_dir in sorted(outputs.glob("dp_lora_*/final")):
        tag = adapter_dir.parent.name
        meta = parse_tag(tag)

        meta["adapter_path"] = str(adapter_dir)
        meta["adapter_exists"] = (adapter_dir / "adapter_model.safetensors").exists()

        log_path = find_log(tag)
        meta["log_path"] = str(log_path) if log_path else None
        if log_path and log_path.exists():
            meta.update(extract_loss_from_log(log_path))
        else:
            meta.update({"first_loss": None, "final_loss": None,
                         "final_eps_spent": None, "total_steps": 0})

        rows.append(meta)
    return rows


def catalog_unlearn(repo: Path) -> list[dict]:
    outputs = repo / "outputs"
    rows = []
    # Pattern: outputs/unlearn_{method}/ containing step_* or step_final
    for unlearn_dir in sorted(outputs.glob("unlearn_*")):
        name = unlearn_dir.name
        # Determine method and dataset
        stem = name.removeprefix("unlearn_")
        if stem.endswith("_pmc"):
            method = stem[:-4]
            dataset = "pmc"
        else:
            method = stem
            dataset = "mtsamples"

        # Find checkpoint dir (step_393, step_final, etc.)
        ckpt_dirs = sorted(unlearn_dir.glob("step_*"))
        final_ckpt = ckpt_dirs[-1] if ckpt_dirs else None

        # Check if it's a valid merged model (has safetensors shards)
        shards = list(final_ckpt.glob("*.safetensors")) if final_ckpt else []
        has_valid_ckpt = len(shards) >= 4  # merged 8B = 4 shards

        # Check config.json is the HF model config (not UnlearnConfig)
        config_ok = False
        if final_ckpt and (final_ckpt / "config.json").exists():
            try:
                with open(final_ckpt / "config.json") as f:
                    d = json.load(f)
                config_ok = "architectures" in d
            except Exception:
                pass

        rows.append({
            "name": name,
            "method": method,
            "dataset": dataset,
            "checkpoint_dir": str(final_ckpt) if final_ckpt else None,
            "checkpoint_step": final_ckpt.name if final_ckpt else None,
            "num_shards": len(shards),
            "has_valid_checkpoint": has_valid_ckpt,
            "config_json_ok": config_ok,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root)
    results_dir = repo / "results"
    results_dir.mkdir(exist_ok=True)

    # DP-LoRA catalog
    dp_rows = catalog_dp_lora(repo)
    log.info("dp_lora: %d adapters found", len(dp_rows))
    parse_errors = [r for r in dp_rows if r.get("parse_error")]
    if parse_errors:
        log.warning("TAG_PATTERN parse errors: %s", [r["tag"] for r in parse_errors])

    # Unlearn catalog
    unlearn_rows = catalog_unlearn(repo)
    log.info("unlearn: %d checkpoints found", len(unlearn_rows))

    # Summary stats
    methods = sorted(set(r["method"] for r in dp_rows if not r.get("parse_error")))
    datasets = sorted(set(r["dataset"] for r in dp_rows if not r.get("parse_error")))
    eps_vals = sorted(set(r["epsilon_str"] for r in dp_rows if not r.get("parse_error")),
                      key=lambda x: float("inf") if x == "inf" else float(x))

    catalog = {
        "summary": {
            "dp_lora_adapters_total": len(dp_rows),
            "dp_lora_adapters_valid": sum(1 for r in dp_rows if r.get("adapter_exists")),
            "unlearn_checkpoints_total": len(unlearn_rows),
            "unlearn_checkpoints_valid": sum(1 for r in unlearn_rows if r["has_valid_checkpoint"]),
            "methods": methods,
            "datasets": datasets,
            "epsilon_values": eps_vals,
            "parse_errors": [r["tag"] for r in parse_errors],
        },
        "dp_lora": dp_rows,
        "unlearn": unlearn_rows,
    }

    out_json = results_dir / "experiment_catalog.json"
    with open(out_json, "w") as f:
        json.dump(catalog, f, indent=2)
    log.info("wrote %s", out_json)

    # Also write a flat CSV
    try:
        import csv
        out_csv = results_dir / "experiment_catalog.csv"
        if dp_rows:
            fieldnames = list(dp_rows[0].keys())
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(dp_rows)
            log.info("wrote %s", out_csv)
    except Exception as e:
        log.warning("CSV write failed: %s", e)

    # Print loss table
    print("\n=== DP-LoRA Final Loss by Method × ε ===")
    print(f"{'tag':<35} {'eps':>5} {'method':<12} {'dataset':<10} {'first_loss':>10} {'final_loss':>10} {'eps_spent':>9}")
    print("-" * 100)
    for r in dp_rows:
        if r.get("parse_error"):
            continue
        fl = f"{r['first_loss']:.2f}" if r['first_loss'] is not None else "N/A"
        ll = f"{r['final_loss']:.2f}" if r['final_loss'] is not None else "N/A"
        es = f"{r['final_eps_spent']:.3f}" if r['final_eps_spent'] is not None else "N/A"
        print(f"{r['tag']:<35} {str(r['epsilon_str']):>5} {r['method']:<12} {r['dataset']:<10} {fl:>10} {ll:>10} {es:>9}")

    print("\n=== Unlearn Checkpoints ===")
    for r in unlearn_rows:
        ok = "✓" if r["has_valid_checkpoint"] else "✗"
        cfg = "✓" if r["config_json_ok"] else "✗ BAD config.json"
        print(f"  {r['name']:<25} step={r['checkpoint_step'] or 'MISSING':<12} shards={r['num_shards']} {ok} cfg={cfg}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
10_fairness.py — Compute fairness metrics across ICD categories for Track A.

Reads per_category_mauve.csv and per_category_adherence.csv, computes:
  - Max-min gap per run
  - Coefficient of variation (std/mean) per run
  - Gini coefficient per run
  - Number of categories below threshold per run

Outputs: eval/fairness_results.csv (one row per run × metric)
         eval/fairness_summary.csv (one row per run, all metrics)
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJ = '/fs1/projects/unlearning_pretraining/Proj_code'
EVAL = f'{PROJ}/eval'


def gini(values):
    v = np.sort(np.array(values, dtype=float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2 * (idx * v).sum() / (n * v.sum())) - (n + 1) / n


def load_per_category_csv(path, value_col):
    data = defaultdict(dict)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            run = row['run_id']
            cat = row['category']
            val = float(row[value_col])
            data[run][cat] = val
    return data


def compute_fairness(per_cat_values):
    vals = list(per_cat_values.values())
    if not vals:
        return {}
    arr = np.array(vals)
    return {
        'n_categories': len(vals),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'max_min_gap': float(np.max(arr) - np.min(arr)),
        'cv': float(np.std(arr) / np.mean(arr)) if np.mean(arr) > 0 else float('inf'),
        'gini': float(gini(arr)),
        'below_0.01': int(np.sum(arr < 0.01)),
        'below_0.05': int(np.sum(arr < 0.05)),
    }


def main():
    mauve_data = load_per_category_csv(f'{EVAL}/per_category_mauve.csv', 'mauve_score')
    adher_data = load_per_category_csv(f'{EVAL}/per_category_adherence.csv', 'mean_adherence')

    all_runs = sorted(set(list(mauve_data.keys()) + list(adher_data.keys())))

    summary_rows = []
    detail_rows = []

    for run in all_runs:
        row = {'run_id': run}

        if run in mauve_data:
            m = compute_fairness(mauve_data[run])
            row['mauve_mean'] = m['mean']
            row['mauve_std'] = m['std']
            row['mauve_max_min_gap'] = m['max_min_gap']
            row['mauve_cv'] = m['cv']
            row['mauve_gini'] = m['gini']
            row['mauve_below_0.01'] = m['below_0.01']
            row['mauve_n_categories'] = m['n_categories']

            for cat, val in sorted(mauve_data[run].items()):
                detail_rows.append({
                    'run_id': run, 'metric': 'mauve', 'category': cat, 'value': val
                })

        if run in adher_data:
            a = compute_fairness(adher_data[run])
            row['adherence_mean'] = a['mean']
            row['adherence_std'] = a['std']
            row['adherence_max_min_gap'] = a['max_min_gap']
            row['adherence_cv'] = a['cv']
            row['adherence_gini'] = a['gini']
            row['adherence_n_categories'] = a['n_categories']

            for cat, val in sorted(adher_data[run].items()):
                detail_rows.append({
                    'run_id': run, 'metric': 'adherence', 'category': cat, 'value': val
                })

        summary_rows.append(row)

    out_summary = f'{EVAL}/fairness_summary.csv'
    if summary_rows:
        with open(out_summary, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            w.writeheader()
            w.writerows(summary_rows)
        print(f'Wrote {out_summary} ({len(summary_rows)} runs)')

    out_detail = f'{EVAL}/fairness_detail.csv'
    if detail_rows:
        with open(out_detail, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=detail_rows[0].keys())
            w.writeheader()
            w.writerows(detail_rows)
        print(f'Wrote {out_detail} ({len(detail_rows)} rows)')

    print('\n' + '=' * 80)
    print('FAIRNESS SUMMARY')
    print('=' * 80)

    run_groups = {
        'Standard DP-SGD': ['eps05', 'eps1', 'eps4', 'eps_inf'],
        'Sqrt-cap weighting': ['eps05_sqrt10_mimic', 'eps1_sqrt10_mimic',
                               'eps4_sqrt10_mimic', 'epsinf_sqrt10_mimic'],
        'Power-law weighting': ['eps05_power03cap10_mimic', 'eps1_power03cap10_mimic',
                                'eps4_power03cap10_mimic', 'epsinf_power03cap10_mimic'],
        'Epoch-weighted': ['data2_eps05_weighted', 'data2_eps1_weighted',
                           'data2_eps4_weighted'],
        'Baselines': ['data0_base', 'data1_sgd'],
    }

    for group_name, runs in run_groups.items():
        print(f'\n--- {group_name} ---')
        for run in runs:
            if run not in mauve_data:
                continue
            m = compute_fairness(mauve_data[run])
            a = compute_fairness(adher_data[run]) if run in adher_data else {}
            print(f'  {run:40s}  MAUVE: cv={m["cv"]:.3f} gini={m["gini"]:.3f} '
                  f'gap={m["max_min_gap"]:.4f}  '
                  f'Adh: cv={a.get("cv", 0):.3f} gini={a.get("gini", 0):.3f} '
                  f'gap={a.get("max_min_gap", 0):.3f}')

    print('\n--- KEY COMPARISON: Standard vs Weighting at eps=4 ---')
    for run in ['eps4', 'eps4_sqrt10_mimic', 'eps4_power03cap10_mimic']:
        if run in mauve_data:
            m = compute_fairness(mauve_data[run])
            a = compute_fairness(adher_data[run]) if run in adher_data else {}
            print(f'  {run:40s}')
            print(f'    MAUVE — mean={m["mean"]:.4f}  cv={m["cv"]:.3f}  '
                  f'gini={m["gini"]:.3f}  min={m["min"]:.4f}  max={m["max"]:.4f}')
            if a:
                print(f'    Adher — mean={a["mean"]:.3f}  cv={a["cv"]:.3f}  '
                      f'gini={a["gini"]:.3f}  min={a["min"]:.3f}  max={a["max"]:.3f}')


if __name__ == '__main__':
    main()

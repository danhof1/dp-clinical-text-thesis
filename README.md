# Design-Space Analysis of Quality Interventions for Differentially Private Clinical Text Generation

Research code for an MS thesis investigating quality-improvement interventions for
differentially private (DP) synthetic clinical text generation. The work spans three
studies:

- **Study 1 — Class Weighting** (`study1_class_weighting/`): six DP-SGD class-weighting
  strategies (unweighted, inverse-frequency, sqrt cap=10, power-law α=0.3 cap=10,
  log cap=10, effective β=0.999 cap=10) on MIMIC-IV Brief Hospital Course notes,
  fine-tuning Llama-3.2-1B-Instruct with LoRA under Opacus DP-SGD.
- **Study 2 — Term2Note** (`study2_term2note/`): SNOMED-conditioned, section-wise
  clinical-note generation under DP, compared against unconditioned Track A generation.
- **Study 3 — Unlearning + DP** (`study3_unlearning_dp/`): machine unlearning
  (RMU / Gradient Ascent / NPO / ReGLU) composed with DP-LoRA fine-tuning, on
  Llama-3.1-8B-Instruct and BioMistral-7B, evaluated against pretraining-derived
  extraction with completion and Tier-1 membership-inference attacks.

## ⚠️ Data availability

**No data is included in this repository.** MIMIC-IV is credentialed-access under a
PhysioNet Data Use Agreement — no patient text, no generated synthetic notes, no model
checkpoints, and no evaluation result files (`.csv` / `.json` / `.jsonl`) are committed
here. The PMC and MTSamples corpora are referenced but not redistributed. This
repository contains **code and experiment configs only**.

## Repository structure

```
study1_class_weighting/      # Study 1 — DP-SGD class-weighting battery (Track A)
  train/                     # DP-LoRA + non-DP SGD training
  generate/                  # quality-maximizer generation, decoding sweep, reranking
  evaluate/                  # MAUVE, TSTR, n-grams, section structure, fairness,
                             #   diversity, coherence
  mia/                       # LiRA whitebox membership inference
  utils/                     # sampling, dataset audit, pipeline orchestration
  slurm/                     # SLURM submission scripts
study2_term2note/            # Study 2 — Term2Note SNOMED-conditioned generation
  slurm/
study3_unlearning_dp/        # Study 3 — unlearning + DP (Track B). Python package;
  unlearn/                   #   internal package layout preserved (cross-module
  finetune/                  #   imports + __init__.py files).
  generate/
  attacks/                   # completion extraction, Tier-1 MIA, LiRA
  eval/                      # eval suite (MMLU / KS / relearn)
  data/
  shadow_training/
  config/                    # ~350 YAML experiment configs (+ runner subdirs)
  scripts/                   # pipeline submission + result aggregation
  contamination/             # PMC paraphrase + subspace diagnostics
```

## Script → thesis artifact map

| Script | Thesis artifact |
|---|---|
| `study1/train/02.2_train_dp.py` | §3.3.2 — all Study 1 DP-LoRA checkpoints |
| `study1/train/02b_train_sgd.py` | §3.3.2 — non-DP SGD baseline |
| `study1/generate/03.2_generate.py` | §3.3 — Study 1 synthetic-note generation |
| `study1/generate/03.3_decode_sweep.py` | Table 4.6 — decoding ablation |
| `study1/generate/03.3_rerank_candidates.py` | Table 4.7 — candidate reranking |
| `study1/evaluate/05_evaluate.py` | §3.6.1 — corpus-level MAUVE (Tables 4.1, 4.10, 4.21) |
| `study1/evaluate/per_category_eval.py` | Per-category MAUVE + adherence (Tables 4.1, 4.3, Appendix B.1/B.2) |
| `study1/evaluate/07b_tstr_per_category.py` | Table 4.2 — per-category TSTR |
| `study1/evaluate/08_clinical_ngrams.py` | Tables 4.4, 4.11 — clinical n-gram coverage |
| `study1/evaluate/09_section_structure.py` | Table 4.9 — section detection |
| `study1/evaluate/10_fairness.py` | Table 4.5 — per-category fairness |
| `study1/evaluate/11_distinct_n.py`, `diversity_analysis_full.py` | Table 4.7, Appendix D, §5.6 — diversity sweep |
| `study1/mia/04_lira.py` | Table 4.8b — LiRA whitebox MIA |
| `study2/build_term2note_data.py`, `generate_term2note.py` | §3.4 — Study 2 (Tables 4.9–4.12) |
| `study3/unlearn/rmu.py`, `unlearn/methods.py` | §3.5 — unlearning interventions |
| `study3/finetune/dp_lora.py` | §3.5 — Track B DP-LoRA (Tables 4.14, B.3) |
| `study3/attacks/completion.py` | Completion extraction (Tables 4.15–4.20, B.4–B.7) |
| `study3/attacks/tier1.py` | Table 4.13 — Tier-1 MIA |
| `study3/eval/evaluator.py` | Eval suite — MMLU / KS / relearn |

## Environment

Experiments were run on a SLURM HPC cluster (STAR-HPC). Core Python dependencies:
`torch`, `transformers`, `opacus`, `peft`, `mauve-text`, `nltk`, `datasets`, `numpy`,
`pyyaml`. Exact pinned versions live in the cluster environment and are not reproduced
here.

## Caveats

- **Hardcoded paths.** Scripts contain absolute cluster paths
  (`/fs1/projects/unlearning_pretraining/Proj_code/...`) and reference the SLURM account
  `unlearning_pretraining`. These must be adapted to run in any other environment.
- **Source of truth.** These scripts were copied from the cluster project directory;
  this repository is the organized, published snapshot, not the live working tree.
- **Study 3 package layout.** `study3_unlearning_dp/` preserves an internal Python
  package structure (relative imports, `__init__.py` files) — do not flatten it.

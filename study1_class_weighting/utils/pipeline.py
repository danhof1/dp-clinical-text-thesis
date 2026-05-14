#!/usr/bin/env python3
"""
Pipeline visualization for: Differentially Private Synthetic Clinical Text Generation
Daniel Doyon — Hofstra University M.S. Data Science

Run:  python pipeline.py [--track a|b|both] [--detail]

Prints an annotated pipeline diagram showing how each stage feeds into the next,
what scripts/modules implement each component, and what artifacts are produced.
"""

import argparse
import sys

# ── ANSI colors (graceful fallback if piped) ─────────────────────────────

try:
    _color = sys.stdout.isatty()
except Exception:
    _color = False

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _color else text

def bold(t):    return _c("1", t)
def dim(t):     return _c("2", t)
def cyan(t):    return _c("36", t)
def green(t):   return _c("32", t)
def yellow(t):  return _c("33", t)
def red(t):     return _c("31", t)
def magenta(t): return _c("35", t)
def blue(t):    return _c("34", t)
def white(t):   return _c("97", t)


W = 100  # box width

def box_top(title="", width=W):
    if title:
        pad = width - len(title) - 4
        return bold(f"  +{'=' * 2} {title} {'=' * max(pad, 0)}+")
    return bold(f"  +{'=' * (width - 2)}+")

def box_bot(width=W):
    return bold(f"  +{'=' * (width - 2)}+")

def box_line(text, width=W):
    pad = width - 4 - len(text.replace('\033[0m', '').replace('\033[1m', '')
                              .replace('\033[2m', '').replace('\033[36m', '')
                              .replace('\033[32m', '').replace('\033[33m', '')
                              .replace('\033[31m', '').replace('\033[35m', '')
                              .replace('\033[34m', '').replace('\033[97m', ''))
    return f"  | {text}{' ' * max(pad, 0)} |"

def arrow(label="", width=W):
    mid = width // 2
    if label:
        return dim(f"  {' ' * (mid - 1)}|") + f"\n  {' ' * (mid - len(label)//2 - 1)}{dim(label)}\n" + dim(f"  {' ' * (mid - 1)}V")
    return dim(f"  {' ' * (mid - 1)}|\n  {' ' * (mid - 1)}V")

def side_arrow(left_label, right_label, width=W):
    return f"  {left_label:>{width//2 - 2}}  -->  {right_label}"

def separator():
    return ""

def section_header(title):
    return f"\n{'=' * (W + 2)}\n  {bold(title)}\n{'=' * (W + 2)}"


# =========================================================================
# TRACK B PIPELINE
# =========================================================================

def print_track_b_overview():
    print(section_header("TRACK B PIPELINE OVERVIEW"))
    print(f"""
  {bold('Model:')}     Llama-3.1-8B-Instruct (8B params, bfloat16)
  {bold('Datasets:')}  MTSamples (medical transcriptions) + PMC (biomedical abstracts)
  {bold('Goal:')}      Unlearn sensitive training data, then DP-retune for safe generation
  {bold('Repo:')}      Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/

  {bold('Pipeline flow:')}

    {cyan('Raw Data')}  -->  {cyan('Split')}  -->  {green('Unlearn')}  -->  {yellow('DP-LoRA')}  -->  {magenta('Generate')}  -->  {blue('Evaluate')}
                              |             |
                              v             v
                         {red('MIA Attack')}    {red('Completion')}
                         {red('(post-unlearn)')} {red('Attack')}
""")


def print_track_b_data():
    print(box_top("STAGE 0: DATA PREPARATION"))
    print(box_line(bold("Data Splits") + " (created once, reused by all downstream stages)"))
    print(box_line(""))
    print(box_line(cyan("splits_v1/") + "  Original splits for training"))
    print(box_line("  finetune     Forget set — text the model was trained on"))
    print(box_line("  retain       Retention set — general text to preserve capability"))
    print(box_line("  members      Same as finetune (for MIA evaluation)"))
    print(box_line("  nonmembers   Text NOT in finetune (same distribution)"))
    print(box_line("  canaries     Planted memorization markers (100 samples)"))
    print(box_line(""))
    print(box_line(cyan("splits_v2/") + "  Deduplicated splits (fixes 455 dupes in v1)"))
    print(box_line("  members=1524  nonmembers=439  clean_nonmembers=1000"))
    print(box_line("  Zero overlap verified. All MIA on MTSamples uses v2."))
    print(box_line(""))
    print(box_line(cyan("splits_v1_pmc/") + "  PMC dataset variant (same structure)"))
    print(box_line(""))
    print(box_line(dim("Script: rebuild_splits_and_submit.py (one-time)")))
    print(box_bot())


def print_track_b_unlearn(detail=False):
    print(arrow("splits_v1 (finetune + retain)"))
    print(box_top("STAGE 1: UNLEARNING"))
    print(box_line(bold("Goal:") + " Erase membership signal for forget set from the base model"))
    print(box_line(""))
    print(box_line(green("Method A: RMU") + " (Representation Misdirection for Unlearning)"))
    print(box_line("  Module:  " + cyan("unlearn.rmu") + "  (unlearn/rmu.py)"))
    print(box_line("  Config:  config/unlearn_rmu_cluster.yaml"))
    print(box_line("  Key idea: redirect hidden representations at layer 7 toward"))
    print(box_line("  random vectors, while anchoring retain set via KL divergence"))
    print(box_line("  Optimizer: AdamW, 150 steps, lr=5e-5"))
    print(box_line("  Output: " + cyan("outputs/unlearn_rmu/step_final/") + " (merged checkpoint)"))
    print(box_line(""))
    print(box_line(green("Method B: RMU-SAM") + " (RMU + Sharpness-Aware Minimization)"))
    print(box_line("  Module:  " + cyan("unlearn.rmu_sam") + "  (unlearn/rmu_sam.py)"))
    print(box_line("  Key idea: SAM finds flat minima in the unlearning objective,"))
    print(box_line("  making the unlearned state resistant to subsequent fine-tuning"))
    print(box_line("  Two-step per iteration:"))
    print(box_line("    1. Perturb weights to sharpest point within rho-ball (ascent)"))
    print(box_line("    2. Compute gradients at perturbed point, step original (descent)"))
    print(box_line("  Extra params: sam_rho=0.05 (perturbation radius)"))
    print(box_line("  Output: " + cyan("outputs/unlearn_rmu_sam/step_final/")))
    print(box_line(""))
    print(box_line(green("Other methods tested:") + " GA, GA+GD, NPO (all failed at finite eps)"))
    print(box_line("  GA/GA+GD/NPO inflate loss to 65-94, DP clipping can't recover"))
    print(box_line("  Only RMU preserves loss near baseline (~5.7), enabling DP-LoRA"))
    print(box_line(""))

    if detail:
        print(box_line(dim("-" * (W - 4))))
        print(box_line(bold("RMU Loss Functions:")))
        print(box_line("  forget_loss = MSE(hidden[layer_id], random_dir * steering_coeff)"))
        print(box_line("    hidden = model(forget_batch, output_hidden_states=True)[layer+1]"))
        print(box_line("    random_dir = unit vector in R^4096, scaled to norm=20"))
        print(box_line("    Averaged over non-padding positions"))
        print(box_line(""))
        print(box_line("  retain_loss = KL(frozen_model || current_model) on retain batch"))
        print(box_line("    Keeps model close to original on non-forget data"))
        print(box_line(""))
        print(box_line("  total_loss = alpha(1200) * forget_loss + retain_loss"))
        print(box_line(""))
        print(box_line(bold("Key hyperparameters:")))
        print(box_line("  layer_id=7  alpha=1200  steering_coeff=20  num_steps=150"))
        print(box_line("  LoRA: r=16 alpha=32 targets=[q,k,v,o,gate,up,down]_proj"))
        print(box_line(""))
        print(box_line(bold("SAM inner loop (rmu_sam.py):")))
        print(box_line("  for each step:"))
        print(box_line("    loss_1 = alpha * forget_loss(w) + retain_loss(w)"))
        print(box_line("    epsilon = rho * grad(loss_1) / ||grad(loss_1)||"))
        print(box_line("    loss_2 = alpha * forget_loss(w + epsilon) + retain_loss(w + epsilon)"))
        print(box_line("    w = w - lr * grad(loss_2)  # descent from original weights"))
        print(box_line(""))

    print(box_bot())


def print_track_b_mia_post_unlearn():
    print(arrow("unlearned checkpoint"))
    print(box_top("STAGE 1b: MIA ATTACK (post-unlearn verification)"))
    print(box_line(bold("Goal:") + " Verify unlearning erased membership signal"))
    print(box_line(""))
    print(box_line("  Module:  " + cyan("attacks.tier1")))
    print(box_line("  Input:   unlearned checkpoint + splits_v2 (members vs nonmembers)"))
    print(box_line("  Output:  " + cyan("outputs/attacks/unlearn_{method}/tier1/tier1_results.json")))
    print(box_line(""))
    print(box_line("  Attacks run (6 total):"))
    print(box_line("    loss              Per-example NLL (lower = more likely member)"))
    print(box_line("    Min-K%20          Avg log-prob of 20% lowest-probability tokens"))
    print(box_line("    Min-K%10          Same, 10% threshold"))
    print(box_line("    zlib              NLL normalized by zlib compression ratio"))
    print(box_line("    ref_ratio         log(p_target / p_reference) per example"))
    print(box_line("    Min-K%++          Normalized Min-K variant (Carlini++ style)"))
    print(box_line(""))
    print(box_line("  Three-way cohort decomposition (mandatory):"))
    print(box_line("    same-dist:    members vs same-distribution nonmembers"))
    print(box_line("    clean:        members vs clean nonmembers (different distribution)"))
    print(box_line("    canaries:     canaries vs clean nonmembers"))
    print(box_line(""))
    print(box_line(yellow("  Result: AUC ~0.48-0.51 (at chance) for all methods")))
    print(box_line(yellow("  Canary AUC=0.83-1.00 (sanity check: canaries still detectable)")))
    print(box_bot())


def print_track_b_dp_lora(detail=False):
    print(arrow("unlearned checkpoint (base_model for DP-LoRA)"))
    print(box_top("STAGE 2: DP-LoRA FINE-TUNING"))
    print(box_line(bold("Goal:") + " Re-tune model on forget set WITH formal (eps,delta)-DP guarantee"))
    print(box_line(""))
    print(box_line("  Module:  " + cyan("finetune.dp_lora") + "  (finetune/dp_lora.py)"))
    print(box_line("  Config:  config/dp_lora_{eps}_{method}_cluster.yaml"))
    print(box_line("  Input:   unlearned checkpoint + splits (finetune set)"))
    print(box_line("  Output:  " + cyan("outputs/dp_lora_{eps}_{method}/final/") + " (LoRA adapter)"))
    print(box_line(""))
    print(box_line("  Privacy budgets tested: " + bold("eps = {1, 3, 8, inf}")))
    print(box_line("  delta = 1e-5 (fixed)"))
    print(box_line(""))
    print(box_line("  DP mechanism: Opacus (per-example gradient clipping + Gaussian noise)"))
    print(box_line("    1. Compute per-example gradients (via Opacus hooks)"))
    print(box_line("    2. Clip each to max_grad_norm=1.0"))
    print(box_line("    3. Average + add Gaussian noise calibrated to (eps, delta)"))
    print(box_line("    4. Update LoRA adapter weights only (base frozen)"))
    print(box_line(""))
    print(box_line("  " + bold("Goldfish Loss variant:") + " (finetune/dp_lora.py, goldfish_ratio param)"))
    print(box_line("  Randomly masks fraction of tokens from loss (Maini et al. 2024)"))
    print(box_line("  Prevents model from memorizing any single token sequence fully"))
    print(box_line("  goldfish_ratio=0.2 means 20% of non-padding tokens masked per step"))
    print(box_line(""))

    if detail:
        print(box_line(dim("-" * (W - 4))))
        print(box_line(bold("Training configuration:")))
        print(box_line("  physical_batch_size=2  logical_batch_size=64  (Poisson sampling)"))
        print(box_line("  max_seq_length=512  num_epochs=3  lr=0.0005"))
        print(box_line("  LoRA: r=16 alpha=32 targets=[q,k,v,o,gate,up,down]_proj"))
        print(box_line("  warmup_ratio=0.03  weight_decay=0.0  secure_mode=false"))
        print(box_line(""))
        print(box_line(bold("Key finding:")))
        print(box_line("  RMU: final_loss ~5.7 at all finite eps (near baseline ~5.1)"))
        print(box_line("  GA/NPO: final_loss 65-94 at finite eps (stuck, can't recover)"))
        print(box_line("  All methods: loss ~1.4-2.1 at eps=inf (converges without noise)"))
        print(box_line(""))

    print(box_bot())


def print_track_b_attacks_post_dp(detail=False):
    print(arrow("DP-LoRA adapter + base model"))
    print(box_top("STAGE 3: PRIVACY ATTACKS (post-DP-LoRA)"))
    print(box_line(bold("Goal:") + " Measure residual privacy leakage after the full pipeline"))
    print(box_line(""))
    print(box_line(red("Attack A: MIA Tier 1") + " (same as Stage 1b, on DP model)"))
    print(box_line("  Module:  " + cyan("attacks.tier1")))
    print(box_line("  Input:   base_model + adapter_path + splits_v2"))
    print(box_line("  Output:  " + cyan("outputs/attacks/v2_dp_lora_{eps}_{method}/tier1/")))
    print(box_line("  Tests both RMU pipeline and no-unlearn baseline (dp_lora_base)"))
    print(box_line(""))
    print(box_line(red("Attack B: Prefix Completion Extraction") + " (Carlini et al. 2021)"))
    print(box_line("  Module:  " + cyan("attacks.completion") + "  (attacks/completion.py)"))
    print(box_line("  Config:  config/compl_{eps}_{method}_{dataset}.yaml"))
    print(box_line("  Input:   model (merged adapter) + splits (members)"))
    print(box_line("  Output:  " + cyan("outputs/attacks/compl_*/completion_attack_results.json")))
    print(box_line(""))
    print(box_line("  Protocol:"))
    print(box_line("    1. Take first 50% of tokens from each training example (prefix)"))
    print(box_line("    2. Greedy-decode continuation for len(remaining tokens)"))
    print(box_line("    3. Compare generated continuation to ground truth"))
    print(box_line(""))
    print(box_line("  Metrics reported:"))
    print(box_line("    EMR          Mean exact match rate (fraction of tokens reproduced)"))
    print(box_line("    p95 EMR      95th percentile EMR (worst-case leakage)"))
    print(box_line("    ROUGE-L      Longest common subsequence ratio"))
    print(box_line("    ext@10/25/50 Fraction of examples with >=K% exact match"))
    print(box_line(""))

    if detail:
        print(box_line(dim("-" * (W - 4))))
        print(box_line(bold("Key finding — MIA:")))
        print(box_line("  Same-dist AUC ~0.52 at all finite eps (attacker at chance)"))
        print(box_line("  dp_lora_base = dp_lora_rmu at every eps (unlearning redundant)"))
        print(box_line("  Only eps=inf elevates signal (clean AUC ~0.98)"))
        print(box_line(""))
        print(box_line(bold("Key finding — Extraction:")))
        print(box_line("  Baseline EMR=0.026 -> RMU cuts to 0.011 (-56%)"))
        print(box_line("  But DP-LoRA restores EMR to ~0.026 at ALL epsilon"))
        print(box_line("  RMU-SAM aims to make unlearning resist this restoration"))
        print(box_line(""))

    print(box_bot())


def print_track_b_eval_suite(detail=False):
    print(arrow("checkpoint (any stage)"))
    print(box_top("STAGE 4: EVALUATION SUITE"))
    print(box_line(bold("Goal:") + " Measure utility preservation across the pipeline"))
    print(box_line(""))
    print(box_line("  Module:  " + cyan("eval.suite") + "  (eval/evaluator.py)"))
    print(box_line("  Config:  config/eval_{method}_{dataset}.yaml"))
    print(box_line("  Output:  " + cyan("outputs/eval/{tag}/eval_suite_results.json")))
    print(box_line(""))
    print(box_line(blue("  Metric 1: KS Test") + " (distributional shift detection)"))
    print(box_line("    Two-sample Kolmogorov-Smirnov on per-example NLL"))
    print(box_line("    forget set (n=300) vs control set (n=300)"))
    print(box_line("    SCAR_DETECTED if p < 0.05 (NLL distributions differ)"))
    print(box_line(""))
    print(box_line(blue("  Metric 2: Re-learn Attack") + " (unlearning robustness)"))
    print(box_line("    50-step AdamW fine-tune on 5% of forget set"))
    print(box_line("    Measures PPL recovery: can model easily re-memorize?"))
    print(box_line("    ROBUST = PPL does not decrease toward pre-unlearn levels"))
    print(box_line(""))
    print(box_line(blue("  Metric 3: MMLU-Medical") + " (general medical reasoning)"))
    print(box_line("    Zero-shot MCQ on 3 MMLU subsets (~535 questions):"))
    print(box_line("      clinical_knowledge, professional_medicine, anatomy"))
    print(box_line("    Baseline Llama-3.1-8B-Instruct: ~0.734"))
    print(box_line(""))

    if detail:
        print(box_line(dim("-" * (W - 4))))
        print(box_line(bold("Key findings:")))
        print(box_line("  RMU degrades MMLU: 0.73 -> 0.44 (MTS) / 0.24 (PMC)"))
        print(box_line("  DP-LoRA fully restores: 0.73-0.74 at all finite eps"))
        print(box_line("  eps=inf MMLU drops to 0.60 (overfitting, no DP regularization)"))
        print(box_line("  RMU KS=1.0 (complete NLL shift), DP restores KS~0.17"))
        print(box_line(""))

    print(box_bot())


def print_track_b_generation():
    print(arrow("DP-LoRA adapter + base model"))
    print(box_top("STAGE 5: SYNTHETIC TEXT GENERATION"))
    print(box_line(bold("Goal:") + " Generate synthetic clinical notes for downstream use"))
    print(box_line(""))
    print(box_line("  Module:  " + cyan("generate.run")))
    print(box_line("  Command: python -m generate.run --base_model {ckpt}"))
    print(box_line("             --adapter_path {adapter} --n_samples 2000"))
    print(box_line("             --output_path outputs/generated/{tag}.jsonl --k 4"))
    print(box_line("  Output:  JSONL with fields: text, specialty, perplexity, passed_filter"))
    print(box_line(""))
    print(box_line("  Generates 2000 clinical notes per configuration"))
    print(box_line("  Best-of-k sampling (k=4): generate 4 candidates, pick lowest PPL"))
    print(box_line("  Filter: discard incoherent/degenerate outputs"))
    print(box_line(""))
    print(box_line("  Configurations: eps={1,3,8,inf} x {MTSamples, PMC} = 8 jobs"))
    print(box_line("  Runtime: ~15h per job (H100 GPU)"))
    print(box_bot())


def print_track_b_fidelity():
    print(arrow("generated JSONL + reference splits"))
    print(box_top("STAGE 6: FIDELITY EVALUATION"))
    print(box_line(bold("Goal:") + " Verify synthetic text quality (generation != collapse)"))
    print(box_line(""))
    print(box_line("  Script:  " + cyan("scripts/eval_fidelity.py")))
    print(box_line("  Output:  " + cyan("outputs/fidelity/{tag}_fidelity.json")))
    print(box_line(""))
    print(box_line("  Metrics:"))
    print(box_line("    MAUVE        Distribution similarity (neural featurizer)"))
    print(box_line("    PPL          Perplexity under reference model"))
    print(box_line("    length_kl    KL divergence on word-count distributions"))
    print(box_line("    unigram_kl   Unigram distribution divergence"))
    print(box_line("    bigram_kl    Bigram distribution divergence"))
    print(box_line("    jaccard      Vocabulary overlap ratio"))
    print(box_line(""))
    print(box_line(yellow("  Rule: every privacy claim must be paired with MAUVE + fidelity")))
    print(box_line(yellow("  Collapse != privacy. Low MAUVE means useless, not private.")))
    print(box_bot())


def print_track_b_full(detail=False):
    print_track_b_overview()
    print_track_b_data()
    print_track_b_unlearn(detail)
    print_track_b_mia_post_unlearn()
    print_track_b_dp_lora(detail)
    print_track_b_attacks_post_dp(detail)
    print_track_b_eval_suite(detail)
    print_track_b_generation()
    print_track_b_fidelity()
    print_track_b_comparison_table()
    print_track_b_file_map()


def print_track_b_comparison_table():
    print()
    print(box_top("EXPERIMENTAL CONDITIONS MATRIX"))
    print(box_line(bold("Unlearning methods tested:")))
    print(box_line("  GA       Gradient Ascent (maximize loss on forget set)"))
    print(box_line("  GA+GD    Gradient Ascent + Gradient Descent (ascend forget, descend retain)"))
    print(box_line("  NPO      Negative Preference Optimization"))
    print(box_line("  RMU      Representation Misdirection for Unlearning  <-- best"))
    print(box_line("  RMU-SAM  RMU + Sharpness-Aware Minimization          <-- experimental"))
    print(box_line(""))
    print(box_line(bold("DP-LoRA privacy budgets:")))
    print(box_line("  eps=1    Strong privacy (most noise)"))
    print(box_line("  eps=3    Moderate privacy"))
    print(box_line("  eps=8    Relaxed privacy"))
    print(box_line("  eps=inf  No DP noise (ablation baseline)"))
    print(box_line(""))
    print(box_line(bold("Datasets:")))
    print(box_line("  MTSamples   Medical transcriptions (diverse specialties)"))
    print(box_line("  PMC         PubMed Central biomedical abstracts"))
    print(box_line(""))
    print(box_line(bold("Training-time interventions:")))
    print(box_line("  Standard      DP-LoRA with standard cross-entropy loss"))
    print(box_line("  Goldfish      DP-LoRA + goldfish_ratio=0.2 (random token masking)"))
    print(box_line(""))
    print(box_line(bold("Full condition count:") + " 4 unlearn x 4 eps x 2 datasets x 2 loss = 64"))
    print(box_line(bold("Actually run:")         + " 40 DP-LoRA + 8 goldfish + 6 SAM = 54 adapters"))
    print(box_bot())


def print_track_b_file_map():
    print()
    print(box_top("FILE MAP — Track B Repo"))
    print(box_line(dim("mtsamples_unlearn_dp/mtsamples_unlearn_dp/")))
    print(box_line(""))
    print(box_line(cyan("unlearn/") + "                Unlearning implementations"))
    print(box_line("  common.py              Shared loaders, tokenizers, collate fns"))
    print(box_line("  rmu.py                 RMU: representation misdirection"))
    print(box_line("  rmu_sam.py             RMU-SAM: sharpness-aware RMU"))
    print(box_line("  methods.py             GA, GA+GD, NPO implementations"))
    print(box_line("  run.py                 CLI dispatcher for unlearn methods"))
    print(box_line(""))
    print(box_line(cyan("finetune/") + "               DP fine-tuning"))
    print(box_line("  dp_lora.py             DP-LoRA with Opacus (+ goldfish loss patch)"))
    print(box_line(""))
    print(box_line(cyan("attacks/") + "                Privacy attack implementations"))
    print(box_line("  tier1.py               MIA: loss, Min-K%, zlib, ref_ratio, Min-K%++"))
    print(box_line("  completion.py          Prefix completion extraction (Carlini 2021)"))
    print(box_line(""))
    print(box_line(cyan("eval/") + "                   Evaluation suite"))
    print(box_line("  evaluator.py           KS test, re-learn attack, MMLU-Medical"))
    print(box_line("  suite.py               CLI entry point for eval"))
    print(box_line(""))
    print(box_line(cyan("generate/") + "               Text generation"))
    print(box_line("  run.py                 Best-of-k sampling with quality filter"))
    print(box_line(""))
    print(box_line(cyan("scripts/") + "                Utility scripts"))
    print(box_line("  eval_fidelity.py       MAUVE, PPL, KL divergence metrics"))
    print(box_line("  catalog_experiments.py Scan outputs/ and build experiment catalog"))
    print(box_line("  aggregate_results.py   Consolidate all results into one table"))
    print(box_line(""))
    print(box_line(cyan("config/") + "                 YAML configuration files"))
    print(box_line("  unlearn_rmu_cluster.yaml          RMU unlearning config"))
    print(box_line("  dp_lora_{eps}_{method}.yaml       DP-LoRA training configs"))
    print(box_line("  attacks_tier1_{label}.yaml         MIA attack configs"))
    print(box_line("  compl_{eps}_{method}_{ds}.yaml     Completion attack configs"))
    print(box_line("  eval_{method}_{ds}.yaml            Eval suite configs"))
    print(box_line(""))
    print(box_line(cyan("outputs/") + "                All artifacts"))
    print(box_line("  splits_v1/                  Original train/test splits"))
    print(box_line("  splits_v2/                  Deduplicated splits"))
    print(box_line("  splits_v1_pmc/              PMC dataset splits"))
    print(box_line("  unlearn_{method}/step_final/    Unlearned checkpoints"))
    print(box_line("  dp_lora_{eps}_{method}/final/   DP-LoRA adapters"))
    print(box_line("  attacks/{tag}/tier1/             MIA results (JSON)"))
    print(box_line("  attacks/compl_*/                 Completion attack results"))
    print(box_line("  eval/{tag}/                      Eval suite results"))
    print(box_line("  generated/{tag}.jsonl            Synthetic text output"))
    print(box_line("  fidelity/{tag}_fidelity.json     Fidelity metrics"))
    print(box_line("  results/experiment_catalog.json  40-entry experiment tracker"))
    print(box_bot())


# =========================================================================
# TRACK A PIPELINE
# =========================================================================

def print_track_a_full(detail=False):
    print(section_header("TRACK A PIPELINE OVERVIEW"))
    print(f"""
  {bold('Model:')}     Llama-3.2-1B-Instruct (1B params)
  {bold('Dataset:')}   MIMIC-IV Brief Hospital Course (56K+ real clinical notes)
  {bold('Goal:')}      Compare DP-SGD weighting strategies for synthetic note generation
  {bold('Repo:')}      Scripts_2.0/

  {bold('Pipeline flow:')}

    {cyan('MIMIC-IV')}  -->  {yellow('DP-SGD Fine-tune')}  -->  {magenta('Generate')}  -->  {blue('Evaluate')}
    (ICD codes)     (weighted loss)       (2K notes)      (fidelity + MIA)
""")
    print(box_top("WEIGHTING STRATEGIES"))
    print(box_line("  unweighted        Standard cross-entropy (baseline)"))
    print(box_line("  inverse-freq      Weight by 1/class_count (collapsed — not useful)"))
    print(box_line("  sqrt cap=10       Weight by sqrt(count), capped at 10"))
    print(box_line("  power-law a=0.3   Weight by count^0.3 (sub-linear scaling)"))
    print(box_line(""))
    print(box_line("  Each strategy tested at eps = {0.5, 1, 4, inf}"))
    print(box_line("  Total: 17 generation directories in generated/mimic/"))
    print(box_bot())
    print()
    print(box_top("EVALUATION METRICS"))
    print(box_line("  MAUVE             Distribution similarity (neural features)"))
    print(box_line("  PPL               Perplexity under reference model"))
    print(box_line("  KL divergence     Unigram/bigram distribution distance"))
    print(box_line("  ICD adherence     Does generated text match prompted ICD category?"))
    print(box_line("  Clinical n-grams  Overlap of clinical terminology"))
    print(box_line("  Section structure  Presence of expected clinical sections"))
    print(box_line("  TSTR              Train-on-Synthetic, Test-on-Real classifier"))
    print(box_line("  MIA               Membership inference (LiRA + Tier 1)"))
    print(box_line("  Per-category      MAUVE and adherence broken down by ICD tier"))
    print(box_line(""))
    print(box_line(bold("Scripts:")))
    print(box_line("  Scripts_2.0/05_evaluate.py          MAUVE, PPL, KL"))
    print(box_line("  Scripts_2.0/07_tstr.py              TSTR (distilroberta classifier)"))
    print(box_line("  Scripts_2.0/08_clinical_ngrams.py   Clinical n-gram overlap"))
    print(box_line("  Scripts_2.0/09_section_structure.py  Section structure rates"))
    print(box_line("  Scripts_2.0/per_category_eval.py     Per-category MAUVE + adherence"))
    print(box_line("  Scripts_2.0/results.py              Consolidated results display"))
    print(box_bot())


# =========================================================================
# CROSS-TRACK SYNTHESIS
# =========================================================================

def print_synthesis():
    print(section_header("CROSS-TRACK SYNTHESIS"))
    print()
    print(box_top("KEY FINDINGS"))
    print(box_line(""))
    print(box_line(green("1.") + " GA/GA+GD/NPO are incompatible with DP gradient clipping"))
    print(box_line("   Loss inflated to 65-94 -> clipped gradients carry zero signal"))
    print(box_line("   Only RMU (loss ~5.7) allows DP-LoRA convergence"))
    print(box_line(""))
    print(box_line(green("2.") + " DP-LoRA fully restores utility after RMU"))
    print(box_line("   MMLU: 0.73 (baseline) -> 0.44 (RMU) -> 0.74 (DP-LoRA, any finite eps)"))
    print(box_line("   KS test: 1.0 (RMU) -> 0.17 (DP-LoRA)"))
    print(box_line(""))
    print(box_line(green("3.") + " DP noise acts as regularization (counterintuitive)"))
    print(box_line("   eps=inf: MMLU drops to 0.60, KS elevated, extraction elevated"))
    print(box_line("   More noise = better generalization (prevents overfitting during rebuild)"))
    print(box_line(""))
    print(box_line(green("4.") + " Same-dist MIA at chance throughout (attacker fails)"))
    print(box_line("   AUC ~0.52 at all finite eps, with or without unlearning"))
    print(box_line("   dp_lora_base = dp_lora_rmu (unlearning redundant for MIA)"))
    print(box_line(""))
    print(box_line(green("5.") + " BUT extraction persists despite MIA defense"))
    print(box_line("   RMU cuts extraction 56%, but DP re-learning restores it"))
    print(box_line("   EMR ~0.026 at all eps (same as baseline)"))
    print(box_line("   MIA-safe != extraction-safe (different threat vectors)"))
    print(box_line(""))
    print(box_line(green("6.") + " Goldfish loss + SAM-enhanced RMU = experimental interventions"))
    print(box_line("   Goldfish: mask tokens during DP-LoRA to prevent re-memorization"))
    print(box_line("   SAM-RMU: flat-minima unlearning that resists retuning reversal"))
    print(box_line("   Results pending — these target the extraction gap specifically"))
    print(box_line(""))
    print(box_line(red("CAVEAT:") + " Pipeline is NOT end-to-end (eps,delta)-DP"))
    print(box_line("   RMU is heuristic. Formal DP applies ONLY to the DP-LoRA phase."))
    print(box_line("   Do not overclaim end-to-end DP protection."))
    print(box_bot())


# =========================================================================
# MAIN
# =========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Pipeline diagram — DP Synthetic Clinical Text Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--track", choices=["a", "b", "both"], default="both",
                    help="Which track to display (default: both)")
    ap.add_argument("--detail", action="store_true",
                    help="Show detailed hyperparameters and loss functions")
    args = ap.parse_args()

    print()
    print(bold("  " + "=" * W))
    print(bold("  PIPELINE: Differentially Private Synthetic Clinical Text Generation"))
    print(bold("  Daniel Doyon — Hofstra University M.S. Data Science"))
    print(bold("  " + "=" * W))

    if args.track in ("b", "both"):
        print_track_b_full(args.detail)

    if args.track in ("a", "both"):
        print_track_a_full(args.detail)

    if args.track == "both":
        print_synthesis()

    print()


if __name__ == "__main__":
    main()

"""
04_lira.py
==========
Likelihood Ratio Attack (LiRA) — white-box membership inference audit.
Carlini et al., "Membership Inference Attacks From First Principles," IEEE S&P 2022.

WHY THIS IS NEEDED
------------------
The black-box MIA (07_mia.py) found near-zero TPR on MIMIC across all epsilon
values. This is NOT evidence of privacy — it is evidence that the black-box
n-gram attack has no signal on MIMIC because the base model (Llama-3.2-1B) was
never pre-trained on MIMIC. The attack cannot distinguish members from
non-members because both look equally "foreign" to the n-gram model.

LiRA directly probes the target model's loss on individual training records,
comparing it to the distribution of losses from shadow models trained on
subsets that excluded each record. This is the gold-standard privacy audit and
is expected to find memorisation signal even on MIMIC.

DESIGN: OFFLINE LiRA (Carlini et al. Algorithm 1 minus lines 5,6,10,12)
------------------------------------------------------------------------
Online LiRA trains shadow models both WITH and WITHOUT each target record
(IN and OUT models). Offline LiRA trains OUT models only — shadow models trained
on random 50% subsets, where each target record is absent from approximately
half the shadows by construction. This halves GPU requirements while losing
only ~20% TPR at 0.1% FPR (Carlini et al. Section V-B).

The attack signal for each target record x:
    phi(x)               = logit-scaled loss on x under target model
    mu_out, sigma_out    = Gaussian fit to phi(x) across all OUT shadow models
    signal               = 1 - Phi((phi_target - mu_out) / sigma_out)
where Phi is the standard normal CDF. Higher signal = likely member.

LOGIT SCALING (Carlini et al. Section VI-A)
-------------------------------------------
Raw cross-entropy loss is not approximately Gaussian. The stable logit-scaled
variant is: phi(p) = log(p) - log(1-p) for p = exp(-loss).
This maps to approximately normal distributions suitable for Gaussian LRT.

SHADOW MODEL TRAINING
---------------------
Each shadow model is trained with FULL DP-SGD at the same epsilon as the
target model. Using SGD shadows for a DP target would produce systematically
different loss distributions, miscalibrating the Gaussian fit.

JOB STRUCTURE
-------------
The script has three modes:

  --mode train_shadow:
    Trains ONE shadow model on a random 50% subset of real data.
    Run once per shadow. Launch 4 in parallel per session (4 GPUs per session).

    Shadow checkpoints:  lira/shadows/eps{E}/shadow_{N}/final/

  --mode score:
    Loads the target model + all completed shadow checkpoints for one epsilon.
    Computes logit-scaled loss per record per model.
    Fits Gaussians and computes LiRA attack scores.
    Output: lira/results/eps{E}/lira_summary.json

  --mode status:
    Prints how many shadows are complete for a given epsilon.

USAGE
-----
Step 1: Train shadow models (run in parallel, 4 per session):

    export TRANSFORMERS_CACHE=/fs1/projects/unlearning_pretraining/.cache
    export HF_HOME=/fs1/projects/unlearning_pretraining/.cache
    cd /fs1/projects/unlearning_pretraining/Proj_code

    # Launch 4 at once (one per GPU per session), change shadow_id each job
    for ID in 0 1 2 3; do
        nohup python 04_lira.py --mode train_shadow \\
            --shadow_id $ID --epsilon 1 \\
            > logs/lira_shadow_eps1_s${ID}.log 2>&1 &
    done
    # Repeat for IDs 4-7 in a second session, and for other epsilon values

Step 2: Check status:
    python 04_lira.py --mode status --epsilon 1

Step 3: Score (run after all 8 shadows complete for this epsilon):
    nohup python 04_lira.py --mode score \\
        --target_checkpoint ./models/llama_dp_eps1.0_20260329_1639/final \\
        --epsilon 1 \\
        > logs/lira_score_eps1.log 2>&1 &

EPSILON VALUES AND TARGET CHECKPOINTS
--------------------------------------
    epsilon=0.5  → models/llama_dp_eps0.5_20260319_1349/final
    epsilon=1    → models/llama_dp_eps1.0_20260329_1639/final
    epsilon=4    → models/llama_dp_eps4.0_20260318_1628/final
    epsilon=999  → models/llama_sgd_epsinf_20260328_1324/final  (add --no_dp for shadows)
"""

import sys
import json
import math
import random
import logging
import argparse
from pathlib import Path

import torch
import numpy as np
from scipy import stats as scipy_stats
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType


# =========================================================================
# Logging
# =========================================================================

class FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[FlushHandler(sys.stderr)]
)
log = logging.getLogger(__name__)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def gpu_mem():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        resrv = torch.cuda.memory_reserved() / 1e9
        return f"GPU: {alloc:.1f}GB / {resrv:.1f}GB"
    return "GPU: N/A"


# =========================================================================
# Dataset — identical to 02.2_train_dp.py for matched training procedure
# =========================================================================

class BHCDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.records    = []
        self.note_ids   = []
        self.categories = []

        log.info(f"Loading dataset from {jsonl_path}")
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line.strip())
                self.records.append(rec['text'])
                self.note_ids.append(rec.get('note_id', ''))
                self.categories.append(
                    rec.get('control_codes', {}).get('icd_category', 'Unknown')
                )
        log.info(f"Loaded {len(self.records):,} records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.records[idx],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
        )
        input_ids      = torch.tensor(enc['input_ids'],      dtype=torch.long)
        attention_mask = torch.tensor(enc['attention_mask'], dtype=torch.long)
        labels         = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'labels':         labels,
            'category':       self.categories[idx],
        }


# =========================================================================
# LoRA config — identical to 02.2_train_dp.py
# =========================================================================

def get_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=0.05, bias='none',
        target_modules=['q_proj','k_proj','v_proj','o_proj',
                        'gate_proj','up_proj','down_proj'],
    )


# =========================================================================
# Privacy accounting — identical to 02.2_train_dp.py
# =========================================================================

def compute_delta(n):
    return 1.0 / (n * math.log(n))


def compute_noise_multiplier(target_epsilon, delta, sample_rate, steps):
    from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    lo, hi = 0.01, 100.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        rdp = compute_rdp(q=sample_rate, noise_multiplier=mid,
                          steps=steps, orders=orders)
        eps, _ = get_privacy_spent(orders=orders, rdp=rdp, delta=delta)
        if eps > target_epsilon:
            lo = mid
        else:
            hi = mid
    return hi


# =========================================================================
# Class weights — identical to 02.2_train_dp.py
# =========================================================================

def compute_class_weights(categories):
    from collections import Counter
    counts  = Counter(categories)
    N, C    = len(categories), len(counts)
    weights = {cat: N / (C * n) for cat, n in counts.items()}
    mean_w  = np.mean(list(weights.values()))
    return {cat: w / mean_w for cat, w in weights.items()}


# =========================================================================
# LOGIT-SCALED LOSS
# Carlini et al. (2022) Section VI-A, stable variant.
# =========================================================================

def compute_logit_scaled_loss(text, model, tokenizer, device, max_length=512):
    """
    Compute phi(x) = logit-scaled mean token confidence.

    phi(p) = log(p / (1-p))  where  p = exp(-cross_entropy_loss)

    This maps the loss to an approximately Gaussian distribution,
    required for the Gaussian likelihood ratio test in LiRA.
    Returns float (phi value), or NaN on degenerate inputs.
    """
    enc = tokenizer(
        text, return_tensors='pt',
        truncation=True, max_length=max_length,
    )
    input_ids = enc['input_ids'].to(device)
    if input_ids.shape[1] <= 1:
        return float('nan')

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        # outputs.loss = mean CE = mean(-log p_token)
        # mean_log_p   = -loss
        mean_log_p = -outputs.loss.item()

    # Clamp to avoid overflow/underflow
    mean_log_p = max(min(mean_log_p, -1e-6), -20.0)
    p   = math.exp(mean_log_p)
    p   = max(min(p, 1 - 1e-7), 1e-7)
    phi = math.log(p) - math.log(1.0 - p)
    return phi


# =========================================================================
# MODE: train_shadow
# =========================================================================

def train_shadow(args):
    """
    Train one shadow model on a random n_shadow_data fraction of train.jsonl.
    shadow_id is used as the random seed for subset selection, ensuring each
    shadow sees a different independent subset.

    Training procedure is IDENTICAL to 02.2_train_dp.py (same LoRA config,
    same class weighting, same DP-SGD setup) to match the target model's
    loss distribution. Mismatched training (e.g. SGD shadows for DP target)
    would miscalibrate the Gaussian fit in the LiRA scoring step.
    """
    set_seed(args.shadow_id)

    shadow_dir = (Path(args.shadow_dir)
                  / f"eps{args.epsilon}"
                  / f"shadow_{args.shadow_id}")
    shadow_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already complete
    if (shadow_dir / 'final' / 'adapter_config.json').exists():
        log.info(f"Shadow {args.shadow_id} already complete → skipping")
        return

    log.info(f"Training shadow {args.shadow_id}  (epsilon={args.epsilon}  "
             f"no_dp={args.no_dp})")
    log.info(f"Output: {shadow_dir}")

    with open(shadow_dir / 'shadow_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)

    # -- Tokenizer --
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- Full dataset + random 50% subset --
    full_dataset = BHCDataset(args.data, tokenizer, args.max_length)
    N_full       = len(full_dataset)

    rng        = random.Random(args.shadow_id)
    n_shadow   = int(N_full * args.n_shadow_data)
    shadow_idx = rng.sample(range(N_full), n_shadow)
    shadow_ds  = Subset(full_dataset, shadow_idx)

    # Persist which note_ids this shadow was trained on — needed by scorer
    shadow_note_ids = [full_dataset.note_ids[i] for i in shadow_idx]
    with open(shadow_dir / 'shadow_note_ids.json', 'w') as f:
        json.dump(shadow_note_ids, f)

    log.info(f"Shadow subset: {n_shadow:,} / {N_full:,} "
             f"(seed={args.shadow_id}, fraction={args.n_shadow_data})")

    shadow_cats  = [full_dataset.categories[i] for i in shadow_idx]
    class_weights = compute_class_weights(shadow_cats)

    # -- Model --
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, local_files_only=True,
    )
    model = get_peft_model(model, get_lora_config())
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)
    log.info(f"Model ready.  {gpu_mem()}")
    model.print_trainable_parameters()

    n           = len(shadow_ds)
    total_steps = args.epochs * n
    sample_rate = 1.0 / n
    dataloader  = DataLoader(shadow_ds, batch_size=1, shuffle=True, num_workers=0)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr)
    warmup    = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    if args.no_dp:
        log.info("SGD shadow (--no_dp): no privacy engine attached")
        final_eps = float('inf')
        sigma     = 0.0
    else:
        from opacus import PrivacyEngine
        delta = compute_delta(n)
        sigma = compute_noise_multiplier(
            args.epsilon, delta, sample_rate, total_steps,
        )
        log.info(f"DP-SGD shadow: target_eps={args.epsilon}  "
                 f"sigma={sigma:.4f}  delta={delta:.2e}")
        privacy_engine = PrivacyEngine(accountant='rdp')
        model, optimizer, _ = privacy_engine.make_private(
            module=model, optimizer=optimizer, data_loader=dataloader,
            noise_multiplier=sigma, max_grad_norm=args.max_grad_norm,
            poisson_sampling=False,
        )

    # -- Training loop --
    model.train()
    global_step = 0
    log.info("=" * 60)
    log.info(f"SHADOW {args.shadow_id}  eps={args.epsilon}  "
             f"n={n:,}  steps={total_steps:,}")
    log.info("=" * 60)

    for epoch in range(args.epochs):
        epoch_loss, epoch_steps = 0.0, 0

        for batch in dataloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            category       = batch['category'][0]

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels)
            loss    = outputs.loss * class_weights.get(category, 1.0)

            loss.backward()
            if args.no_dp:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss  += outputs.loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step <= 5 or global_step % 500 == 0:
                info = (f"eps={privacy_engine.get_epsilon(delta):.3f}/{args.epsilon}  "
                        if not args.no_dp else "")
                log.info(f"  e{epoch+1} step={global_step:>8,}/{total_steps:,}  "
                         f"loss={epoch_loss/epoch_steps:.4f}  {info}{gpu_mem()}")

        avg = epoch_loss / epoch_steps
        log.info(f"EPOCH {epoch+1}/{args.epochs}  avg_loss={avg:.4f}")

        ckpt = shadow_dir / f'checkpoint-epoch{epoch+1}'
        ckpt.mkdir(parents=True, exist_ok=True)
        unwrapped = model._module if hasattr(model, '_module') else model
        unwrapped.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        log.info(f"Checkpoint → {ckpt}")

    if not args.no_dp:
        final_eps = privacy_engine.get_epsilon(delta)

    final_dir = shadow_dir / 'final'
    final_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = model._module if hasattr(model, '_module') else model
    unwrapped.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    with open(shadow_dir / 'shadow_report.json', 'w') as f:
        json.dump({
            'shadow_id':      args.shadow_id,
            'epsilon':        args.epsilon,
            'epsilon_actual': final_eps if not args.no_dp else 'inf',
            'sigma':          sigma,
            'n_records':      n,
            'n_full':         N_full,
            'fraction':       args.n_shadow_data,
            'epochs':         args.epochs,
            'no_dp':          args.no_dp,
        }, f, indent=2)

    log.info(f"Shadow {args.shadow_id} complete → {final_dir}")
    if not args.no_dp:
        log.info(f"Final epsilon: {final_eps:.4f}")


# =========================================================================
# MODE: score
# =========================================================================

def score_lira(args):
    """
    Offline LiRA scoring.

    For each target record x:
      1. phi_target  = logit_scaled_loss(x, target_model)
      2. phi_out[s]  = logit_scaled_loss(x, shadow_s)  for each OUT shadow s
         (shadow s is OUT for x if x not in shadow_s's training set)
      3. mu_out      = mean(phi_out)          — per-example
         sigma_out   = std(all out pairs)     — global (Carlini §VI-B)
      4. signal      = 1 - Phi((phi_target - mu_out) / sigma_out)
         High signal = target model much more confident on x than OUT shadows
                     = x is likely a training member (memorised)

    Global variance (Carlini et al. Section VI-B):
      With < 64 shadow models, per-example variance estimates are noisy.
      Estimating sigma_out globally across all (record, shadow) OUT pairs
      is recommended and nearly matches the full attack.
    """
    set_seed(42)

    output_dir = Path(args.output) / f"eps{args.epsilon}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info(f"LiRA SCORING  epsilon={args.epsilon}")
    log.info(f"Target:     {args.target_checkpoint}")
    log.info(f"Shadow dir: {args.shadow_dir}/eps{args.epsilon}/")
    log.info(f"Output:     {output_dir}")
    log.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- Discover completed shadow checkpoints --
    shadow_base = Path(args.shadow_dir) / f"eps{args.epsilon}"
    shadow_finals = sorted(shadow_base.glob("shadow_*/final"))
    shadow_finals = [d for d in shadow_finals
                     if (d / 'adapter_config.json').exists()]

    if not shadow_finals:
        log.error(f"No completed shadows found in {shadow_base}")
        log.error("Run --mode train_shadow first.")
        sys.exit(1)

    n_shadows = len(shadow_finals)
    log.info(f"Found {n_shadows} completed shadow checkpoints")

    # Load which note_ids each shadow was trained on
    shadow_member_sets = []
    for sf in shadow_finals:
        nid_path = sf.parent / 'shadow_note_ids.json'
        if nid_path.exists():
            with open(nid_path) as f:
                shadow_member_sets.append(set(json.load(f)))
        else:
            log.warning(f"No shadow_note_ids.json for {sf.parent} — "
                        f"treating all records as OUT")
            shadow_member_sets.append(set())

    # -- Load dataset --
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    full_dataset = BHCDataset(args.data, tokenizer, args.max_length)
    N = len(full_dataset)

    rng = random.Random(42)
    n_score = args.n_score if args.n_score and args.n_score < N else N
    score_indices  = rng.sample(range(N), n_score) if n_score < N else list(range(N))
    score_note_ids = [full_dataset.note_ids[i] for i in score_indices]
    score_texts    = [full_dataset.records[i]  for i in score_indices]
    log.info(f"Scoring {len(score_indices):,} records")

    # -- Shadow model losses --
    # phi_shadow[r][s] = logit-scaled loss of record r under shadow s
    #                    NaN if record r was IN shadow s's training set
    phi_shadow = np.full((n_score, n_shadows), np.nan, dtype=np.float64)

    for s_idx, shadow_final in enumerate(shadow_finals):
        log.info(f"Shadow {s_idx+1}/{n_shadows}: {shadow_final}  {gpu_mem()}")
        base         = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, local_files_only=True,
        )
        shadow_model = PeftModel.from_pretrained(base, str(shadow_final))
        shadow_model = shadow_model.merge_and_unload().to(device)
        shadow_model.eval()

        member_set = shadow_member_sets[s_idx]

        for r_idx, (note_id, text) in enumerate(zip(score_note_ids, score_texts)):
            if note_id in member_set:
                continue  # This shadow trained on x — skip (OUT requirement)
            phi_shadow[r_idx, s_idx] = compute_logit_scaled_loss(
                text, shadow_model, tokenizer, device, args.max_length,
            )
            if (r_idx + 1) % 500 == 0:
                log.info(f"  shadow {s_idx}: {r_idx+1}/{n_score}  {gpu_mem()}")

        del shadow_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    np.save(output_dir / 'phi_shadow.npy', phi_shadow)
    with open(output_dir / 'score_note_ids.json', 'w') as f:
        json.dump(score_note_ids, f)
    log.info(f"Shadow losses saved → {output_dir}/phi_shadow.npy")

    # -- Target model losses --
    log.info(f"Loading target model from {args.target_checkpoint}...")
    base         = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, local_files_only=True,
    )
    target_model = PeftModel.from_pretrained(base, args.target_checkpoint)
    target_model = target_model.merge_and_unload().to(device)
    target_model.eval()

    phi_target = np.full(n_score, np.nan, dtype=np.float64)
    for r_idx, text in enumerate(score_texts):
        phi_target[r_idx] = compute_logit_scaled_loss(
            text, target_model, tokenizer, device, args.max_length,
        )
        if (r_idx + 1) % 500 == 0:
            log.info(f"  target: {r_idx+1}/{n_score}  {gpu_mem()}")

    del target_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    np.save(output_dir / 'phi_target.npy', phi_target)

    # -- Gaussian fitting + LiRA scores --
    log.info("Fitting Gaussians (global sigma, per-example mu)...")

    # Global sigma_out across all valid (record, shadow) OUT pairs
    all_out_phis  = phi_shadow[~np.isnan(phi_shadow)]
    global_sigma  = float(np.std(all_out_phis)) if len(all_out_phis) > 1 else 1.0
    log.info(f"Global OUT sigma: {global_sigma:.4f}  "
             f"({len(all_out_phis):,} valid pairs from {n_shadows} shadows)")

    attack_scores = []
    mu_outs       = []

    for r_idx in range(n_score):
        phi_t = phi_target[r_idx]
        if np.isnan(phi_t):
            attack_scores.append(np.nan)
            mu_outs.append(np.nan)
            continue

        out_phis = phi_shadow[r_idx, ~np.isnan(phi_shadow[r_idx, :])]
        mu_out   = (float(np.mean(out_phis)) if len(out_phis) >= 1
                    else float(np.mean(all_out_phis)))
        mu_outs.append(mu_out)

        if global_sigma > 0:
            z     = (phi_t - mu_out) / global_sigma
            score = 1.0 - float(scipy_stats.norm.cdf(z))
        else:
            score = 0.5
        attack_scores.append(score)

    attack_scores = np.array(attack_scores)
    mu_outs       = np.array(mu_outs)
    phi_t_valid   = phi_target[~np.isnan(phi_target)]
    mu_out_valid  = mu_outs[~np.isnan(mu_outs)]
    scores_valid  = attack_scores[~np.isnan(attack_scores)]

    # Primary memorisation signal: fraction of records where target > OUT mean
    pct_above = float(np.mean(phi_t_valid > mu_out_valid[:len(phi_t_valid)]))

    log.info(f"Valid scores: {len(scores_valid):,}")
    log.info(f"% target > OUT mean: {pct_above*100:.1f}%  "
             f"(random baseline = 50%; higher = memorisation signal)")

    # -- Save --
    per_record = [
        {
            'note_id':       score_note_ids[r],
            'phi_target':    float(phi_target[r]) if not np.isnan(phi_target[r]) else None,
            'mu_out':        float(mu_outs[r])    if not np.isnan(mu_outs[r])    else None,
            'attack_score':  float(attack_scores[r]) if not np.isnan(attack_scores[r]) else None,
            'n_out_shadows': int(np.sum(~np.isnan(phi_shadow[r, :]))),
        }
        for r in range(n_score)
    ]

    summary = {
        'epsilon':          args.epsilon,
        'target_checkpoint': str(args.target_checkpoint),
        'n_shadows':        n_shadows,
        'n_scored':         n_score,
        'n_valid':          int(len(scores_valid)),
        'global_sigma_out': float(global_sigma),
        'attack_signal': {
            'mean':          float(np.mean(scores_valid)),
            'std':           float(np.std(scores_valid)),
            'median':        float(np.median(scores_valid)),
            'p90':           float(np.percentile(scores_valid, 90)),
            'p99':           float(np.percentile(scores_valid, 99)),
            'pct_above_out': float(pct_above),
        },
        'phi_target': {
            'mean':   float(np.mean(phi_t_valid)),
            'std':    float(np.std(phi_t_valid)),
            'median': float(np.median(phi_t_valid)),
        },
        'interpretation': (
            'pct_above_out > 0.55: strong memorisation signal. '
            'pct_above_out ~0.50: no signal (consistent with effective DP or attack limitation). '
            'NOTE: All scored records are training members. For true TPR/FPR analysis, '
            'score held-out non-members and combine with these results.'
        ),
        'per_record_sample': per_record[:200],
    }

    summary_path = output_dir / 'lira_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Summary → {summary_path}")

    # -- Print table row --
    log.info("=" * 60)
    log.info(f"LiRA RESULTS  epsilon={args.epsilon}")
    log.info("=" * 60)
    log.info(f"  Shadows:              {n_shadows}")
    log.info(f"  Records scored:       {len(scores_valid):,}")
    log.info(f"  Global sigma_out:     {global_sigma:.4f}")
    log.info(f"  Attack score mean:    {np.mean(scores_valid):.4f}")
    log.info(f"  Attack score std:     {np.std(scores_valid):.4f}")
    log.info(f"  % target > OUT mean:  {pct_above*100:.1f}%")
    log.info("")
    if pct_above > 0.55:
        log.info("  → STRONG memorisation signal detected.")
    elif pct_above > 0.50:
        log.info("  → WEAK signal (slight above chance).")
    else:
        log.info("  → NO signal. Consistent with effective DP or attack limitation.")
    log.info("=" * 60)
    log.info("DONE")


# =========================================================================
# MODE: status
# =========================================================================

def check_status(args):
    """Print shadow training progress for a given epsilon."""
    shadow_base = Path(args.shadow_dir) / f"eps{args.epsilon}"
    if not shadow_base.exists():
        log.info(f"No shadow dir found: {shadow_base}")
        return

    log.info(f"Shadow status — epsilon={args.epsilon}  ({shadow_base})")
    n_complete = 0
    for shadow_dir in sorted(shadow_base.glob("shadow_*")):
        final = shadow_dir / 'final' / 'adapter_config.json'
        if final.exists():
            report_path = shadow_dir / 'shadow_report.json'
            n_records   = '?'
            eps_actual  = '?'
            if report_path.exists():
                with open(report_path) as f:
                    r = json.load(f)
                n_records  = f"{r.get('n_records', '?'):,}"
                eps_actual = r.get('epsilon_actual', '?')
            log.info(f"  {shadow_dir.name}: COMPLETE  "
                     f"n={n_records}  eps_actual={eps_actual}")
            n_complete += 1
        else:
            partial = sorted(shadow_dir.glob("checkpoint-epoch*"))
            if partial:
                log.info(f"  {shadow_dir.name}: IN PROGRESS  "
                         f"(last: {partial[-1].name})")
            else:
                log.info(f"  {shadow_dir.name}: NOT STARTED")

    log.info(f"  Total complete: {n_complete}")

    result = Path(args.output) / f"eps{args.epsilon}" / 'lira_summary.json'
    if result.exists():
        with open(result) as f:
            r = json.load(f)
        pct = r['attack_signal']['pct_above_out'] * 100
        log.info(f"  Scoring: COMPLETE  pct_above_out={pct:.1f}%")
    else:
        log.info("  Scoring: NOT YET RUN")


# =========================================================================
# Main
# =========================================================================

def main(args):
    if args.mode == 'train_shadow':
        train_shadow(args)
    elif args.mode == 'score':
        if not args.target_checkpoint:
            log.error("--target_checkpoint required for score mode")
            sys.exit(1)
        score_lira(args)
    elif args.mode == 'status':
        check_status(args)


# =========================================================================
# CLI
# =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='LiRA white-box MIA — offline variant (Carlini et al. 2022)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--mode', required=True,
                        choices=['train_shadow', 'score', 'status'])

    # Shared paths
    parser.add_argument('--model_path',
                        default='/fs1/shared/model/llm/Llama-3.2-1B-Instruct')
    parser.add_argument('--data',
                        default='./data/train.jsonl')
    parser.add_argument('--shadow_dir',
                        default='./lira/shadows')
    parser.add_argument('--output',
                        default='./lira/results')
    parser.add_argument('--epsilon', type=float, default=1.0)

    # train_shadow params
    parser.add_argument('--shadow_id', type=int, default=0,
                        help='Index of this shadow (0-based); used as random seed')
    parser.add_argument('--n_shadow_data', type=float, default=0.5,
                        help='Fraction of training data per shadow (default: 0.5)')
    parser.add_argument('--no_dp', action='store_true',
                        help='Train shadow without DP — use for epsilon=inf baseline')
    parser.add_argument('--epochs',        type=int,   default=2)
    parser.add_argument('--lr',            type=float, default=5e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--max_length',    type=int,   default=512)

    # score params
    parser.add_argument('--target_checkpoint', default=None,
                        help='Path to target model /final directory')
    parser.add_argument('--n_score', type=int, default=2000,
                        help='Records to score (0 = all). 2000 is fast and stable.')

    args = parser.parse_args()
    main(args)
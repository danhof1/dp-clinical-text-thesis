"""
score_all_generations.py
Scans all generated/ subdirectories and scores each run with Asclepius-Llama3-8B:
  coherence_ppl  — perplexity of the synthetic note under Asclepius (lower = clinical)
  adherence_score — zero-shot judge, 0.0–1.0 normalized (does text match ICD label?)
  adherence_label — "match" / "partial" / "mismatch"

Outputs:
  generated/scoring_results.jsonl   one record per note
  generated/scoring_summary.csv     one row per run with aggregated stats + flags
"""

import csv
import json
import os
import re
import random
import datetime
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor

PROJ       = "/fs1/projects/unlearning_pretraining/Proj_code"
GEN_ROOT   = f"{PROJ}/generated"
LOGS_DIR   = f"{PROJ}/logs"
OUT_JSONL  = f"{GEN_ROOT}/scoring_results.jsonl"
OUT_CSV    = f"{GEN_ROOT}/scoring_summary.csv"
JUDGE_PATH = "/fs1/shared/model/llm/Asclepius-Llama3-8B"
SAMPLE_N   = 50
MIN_TOKENS = 10     # skip degenerate notes shorter than this
SEED       = 42
random.seed(SEED)

# ── Hyperparameter inference ───────────────────────────────────────────────────

HARDCODED = {
    "data0_base":              {"epsilon": None,         "weighting": "none",     "source": "base_no_ft"},
    "data2_eps0.5_weighted":   {"epsilon": 0.5,          "weighting": "inv_freq", "source": "mimic"},
    "data2_eps05_weighted":    {"epsilon": 0.5,          "weighting": "inv_freq", "source": "mimic"},
    "data2_eps1_weighted":     {"epsilon": 1.0,          "weighting": "inv_freq", "source": "mimic"},
    "data2_eps4_weighted":     {"epsilon": 4.0,          "weighting": "inv_freq", "source": "mimic"},
    "data2_eps1.0_weighted":   {"epsilon": 1.0,          "weighting": "inv_freq", "source": "mimic"},
    "data2_eps4.0_weighted":   {"epsilon": 4.0,          "weighting": "inv_freq", "source": "mimic"},
    "eps0.5":                  {"epsilon": 0.5,          "weighting": "none",     "source": "mimic"},
    "eps1":                    {"epsilon": 1.0,          "weighting": "none",     "source": "mimic"},
    "eps4":                    {"epsilon": 4.0,          "weighting": "none",     "source": "mimic"},
    "eps_inf":                 {"epsilon": float("inf"), "weighting": "none",     "source": "mimic"},
}

def parse_eps(s: str):
    """Convert '05', '0.5', '4', 'inf', '999' to float."""
    s = s.lower().strip("_")
    if s in ("inf", "999", "infinity"):
        return float("inf")
    if s == "05":           # eps05 → ε=0.5 (project shorthand)
        return 0.5
    try:
        return float(s)
    except ValueError:
        return None

def infer_meta(dirname: str) -> dict:
    """Return {epsilon, weighting, source} for a generation directory name."""
    if dirname in HARDCODED:
        return dict(HARDCODED[dirname])

    d = dirname.lower()

    # sqrt runs: eps4_sqrt10_mimic, epsinf_sqrt10_mimic, eps05_sqrt10_mimic
    m = re.match(r"^eps([0-9inf_.]+)_sqrt(\d+)_(.+?)(?:_unlearn[-_](.+))?$", d)
    if m and "sqrt" in d:
        return {
            "epsilon":  parse_eps(m.group(1)),
            "weighting": f"sqrt_cap{m.group(2)}",
            "source":   m.group(3),
            "unlearn":  m.group(4) or None,
        }

    # power-law runs: eps4_power03cap10_mimic, epsinf_power03cap10_mimic
    m = re.match(r"^eps([0-9inf_.]+)_power(\d+)cap(\d+)_(.+?)(?:_unlearn[-_](.+))?$", d)
    if m and "power" in d:
        return {
            "epsilon":  parse_eps(m.group(1)),
            "weighting": f"power{m.group(2)}_cap{m.group(3)}",
            "source":   m.group(4),
            "unlearn":  m.group(5) or None,
        }

    # generic inv_freq: data2_eps4_weighted, data2_eps0.5_weighted
    m = re.match(r"^data\d+_eps([0-9inf_.]+)_weighted$", d)
    if m:
        return {"epsilon": parse_eps(m.group(1)), "weighting": "inv_freq", "source": "mimic"}

    # generic unweighted: eps4, eps0.5, eps_inf
    m = re.match(r"^eps([0-9inf_.]+)$", d)
    if m:
        return {"epsilon": parse_eps(m.group(1)), "weighting": "none", "source": "mimic"}

    return {"epsilon": None, "weighting": "unknown", "source": "unknown"}


def detective_resolve(dirpath: str, meta: dict) -> dict:
    """
    If meta still has unknowns, look inside the directory, training logs,
    and checkpoint configs to infer missing fields.
    """
    if meta.get("epsilon") is not None and meta.get("weighting") != "unknown":
        return meta

    # 1. Check for a meta.json dropped by the training script
    for fname in ("meta.json", "config.json", "training_args.json"):
        candidate = os.path.join(dirpath, fname)
        if os.path.exists(candidate):
            try:
                with open(candidate) as f:
                    saved = json.load(f)
                for key in ("epsilon", "weighting", "source"):
                    if key in saved and (meta.get(key) in (None, "unknown")):
                        meta[key] = saved[key]
            except Exception:
                pass

    # 2. Scan logs/ for a log whose filename mentions our dir basename
    dirname = os.path.basename(dirpath)
    if os.path.isdir(LOGS_DIR):
        for log_name in sorted(os.listdir(LOGS_DIR)):
            if dirname in log_name or any(p in log_name for p in dirname.split("_") if len(p) > 3):
                log_path = os.path.join(LOGS_DIR, log_name)
                try:
                    with open(log_path, errors="replace") as f:
                        header = f.read(3000)
                    if meta.get("epsilon") is None:
                        m = re.search(r"epsilon[\"'\s:=]+([0-9.inf]+)", header, re.I)
                        if m:
                            meta["epsilon"] = parse_eps(m.group(1))
                    if meta.get("weighting") == "unknown":
                        m = re.search(r"weight(?:ing)?[_\s]*strat(?:egy)?[\"'\s:=]+(\w+)", header, re.I)
                        if m:
                            meta["weighting"] = m.group(1).lower()
                except Exception:
                    pass

    return meta


# ── Scoring utilities ─────────────────────────────────────────────────────────

class ForceDigit(LogitsProcessor):
    """Restrict the next token to bare digits 1–5."""
    def __init__(self, digit_ids):
        self.digit_ids = digit_ids

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        for tid in self.digit_ids:
            mask[:, tid] = 0.0
        return scores + mask


def get_digit_ids(tokenizer):
    ids = []
    for d in "12345":
        for variant in (d, f" {d}"):
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if len(toks) == 1:
                ids.append(toks[0])
    return list(set(ids))


def compute_ppl(text: str, model, tokenizer, max_length=512) -> float:
    enc = tokenizer(
        text, return_tensors="pt", max_length=max_length, truncation=True
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        loss = model(**enc, labels=enc["input_ids"]).loss
    return round(torch.exp(loss).item(), 2)


def score_adherence(text: str, category: str, model, tokenizer,
                    force_proc: ForceDigit, snippet_tokens=300) -> float:
    """Return adherence normalized to 0.0–1.0, or -1.0 on parse failure."""
    toks    = tokenizer.encode(text, add_special_tokens=False)
    snippet = tokenizer.decode(toks[:snippet_tokens], skip_special_tokens=True)
    prompt  = (
        f"ICD-10 category: {category}\n\n"
        "Does the following hospital course note reflect the above ICD-10 category?\n"
        "1 = no match at all\n"
        "2 = weak match\n"
        "3 = moderate match\n"
        "4 = strong match\n"
        "5 = clearly matches\n"
        "Reply with a single digit (1-5) only.\n\n"
        f"Note:\n{snippet}\n\nScore:"
    )
    inp = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=[force_proc],
        )
    raw = tokenizer.decode(
        out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()
    for ch in raw:
        if ch in "12345":
            return round((int(ch) - 1) / 4, 3)   # map 1–5 → 0.0–1.0
    return -1.0


def adherence_label(score_norm: float) -> str:
    if score_norm < 0:      return "parse_error"
    if score_norm >= 0.6:   return "match"
    if score_norm >= 0.3:   return "partial"
    return "mismatch"


# ── Discover runs ─────────────────────────────────────────────────────────────

runs = []
for name in sorted(os.listdir(GEN_ROOT)):
    path  = os.path.join(GEN_ROOT, name)
    jsonl = os.path.join(path, "synthetic_bhc.jsonl")
    if os.path.isdir(path) and os.path.exists(jsonl):
        meta = infer_meta(name)
        meta = detective_resolve(path, meta)
        runs.append((name, path, jsonl, meta))

print(f"\nFound {len(runs)} generation runs:\n")
print(f"  {'Directory':45s}  {'ε':>6}  {'weighting':18s}  source")
print("  " + "─" * 80)
for name, _, _, meta in runs:
    eps = "base" if meta.get("epsilon") is None else str(meta["epsilon"])
    print(f"  {name:45s}  {eps:>6}  {str(meta.get('weighting', '?'))[:18]:18s}  {meta.get('source', '?')}")

# ── Load judge model ──────────────────────────────────────────────────────────

print(f"\nLoading judge: {JUDGE_PATH} ...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(JUDGE_PATH)
model = AutoModelForCausalLM.from_pretrained(
    JUDGE_PATH, torch_dtype=torch.float16, device_map="auto"
)
model.eval()
digit_ids   = get_digit_ids(tokenizer)
force_digit = ForceDigit(digit_ids)
print("  Judge loaded.\n", flush=True)

# ── Score all runs ────────────────────────────────────────────────────────────

all_note_records = []
summary_rows     = []

for run_name, run_path, jsonl_path, meta in runs:
    print(f"\n{'═'*72}")
    print(f"  Run: {run_name}")
    eps_str = "base" if meta.get("epsilon") is None else str(meta["epsilon"])
    print(f"  ε={eps_str}  w={meta.get('weighting')}  src={meta.get('source')}", flush=True)

    # Load + group by category
    cat_records = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            cat = r.get("control_codes", {}).get("icd_category", "Unknown")
            cat_records[cat].append(r)

    total = sum(len(v) for v in cat_records.values())
    print(f"  {total:,} records, {len(cat_records)} categories", flush=True)

    # Stratified sample: spread SAMPLE_N notes across all categories
    n_cats  = max(len(cat_records), 1)
    per_cat = max(1, SAMPLE_N // n_cats)
    sample  = []
    for cat, recs in cat_records.items():
        sample.extend(random.sample(recs, min(per_cat, len(recs))))
    if len(sample) > SAMPLE_N:
        sample = random.sample(sample, SAMPLE_N)

    # Score
    ppls, adh_scores = [], []
    run_note_records = []

    for i, r in enumerate(sample, 1):
        cat     = r.get("control_codes", {}).get("icd_category", "Unknown")
        text    = r.get("synthetic_bhc", "").strip()
        note_id = r.get("note_id", f"idx{i}")

        # Skip degenerate notes
        tok_count = len(tokenizer.encode(text, add_special_tokens=False))
        if tok_count < MIN_TOKENS:
            continue

        ppl     = compute_ppl(text, model, tokenizer)
        adh     = score_adherence(text, cat, model, tokenizer, force_digit)
        label   = adherence_label(adh)

        ppls.append(ppl)
        if adh >= 0:
            adh_scores.append(adh)

        rec = {
            "run_id":          run_name,
            "epsilon":         meta.get("epsilon"),
            "weighting":       meta.get("weighting"),
            "source":          meta.get("source"),
            "note_id":         note_id,
            "category":        cat,
            "coherence_ppl":   ppl,
            "adherence_score": adh,
            "adherence_label": label,
        }
        run_note_records.append(rec)

        if i % 10 == 0 or i == len(sample):
            print(
                f"  [{i:3d}/{len(sample)}]  ppl={ppl:8.1f}  adh={adh:+.3f}  [{label:11s}]  {cat[:35]}",
                flush=True,
            )

    all_note_records.extend(run_note_records)

    mean_ppl = round(sum(ppls) / len(ppls), 2) if ppls else None
    mean_adh = round(sum(adh_scores) / len(adh_scores), 3) if adh_scores else None

    flags = []
    if mean_ppl is not None and mean_ppl > 200:
        flags.append(f"HIGH_PPL={mean_ppl:.0f}")
    if mean_adh is not None and mean_adh < 0.3:
        flags.append(f"LOW_ADH={mean_adh:.3f}")
    flag_str = "; ".join(flags) if flags else ""

    row = {
        "run_id":             run_name,
        "epsilon":            meta.get("epsilon"),
        "weighting":          meta.get("weighting"),
        "source":             meta.get("source"),
        "n_scored":           len(run_note_records),
        "mean_coherence_ppl": mean_ppl,
        "mean_adherence":     mean_adh,
        "flagged":            flag_str,
    }
    summary_rows.append(row)

    print(
        f"\n  → n_scored={len(run_note_records)}  mean_ppl={mean_ppl}  "
        f"mean_adh={mean_adh}  flags=[{flag_str or 'none'}]",
        flush=True,
    )

# ── Write outputs ─────────────────────────────────────────────────────────────

with open(OUT_JSONL, "w") as f:
    for rec in all_note_records:
        f.write(json.dumps(rec) + "\n")

CSV_FIELDS = ["run_id", "epsilon", "weighting", "source", "n_scored",
              "mean_coherence_ppl", "mean_adherence", "flagged"]
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(summary_rows)

# ── Final summary table ───────────────────────────────────────────────────────

print(f"\n\n{'═'*100}")
print("FINAL SUMMARY")
print(f"{'═'*100}")
print(
    f"\n{'Run':45s}  {'ε':>6}  {'weight':18s}  {'n':>4}  {'mean_ppl':>10}  {'mean_adh':>9}  flags"
)
print("─" * 100)

for s in summary_rows:
    eps = "base" if s["epsilon"] is None else str(s["epsilon"])
    flag_marker = " ← FLAGGED" if s["flagged"] else ""
    print(
        f"{s['run_id'][:45]:45s}  {eps:>6}  {str(s['weighting'])[:18]:18s}  "
        f"{s['n_scored']:>4}  {str(s['mean_coherence_ppl'] or '?'):>10}  "
        f"{str(s['mean_adherence'] or '?'):>9}  {s['flagged']}{flag_marker}"
    )

print("─" * 100)
print(f"\nWrote {len(all_note_records)} note records → {OUT_JSONL}")
print(f"Wrote {len(summary_rows)} run rows     → {OUT_CSV}")
print(f"\nDone at {datetime.datetime.now().isoformat()}")

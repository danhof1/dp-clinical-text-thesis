"""
sample_by_class.py  (v2 — per-note LLM scoring via Asclepius-Llama3-8B)
Samples representative MIMIC-IV BHC notes per ICD-10 tier, scores each note
for token length, clinical coherence, and ICD-category adherence.
Appends results to generated/class_samples.jsonl with run_id + timestamp.
"""

import json, random, os, datetime, uuid
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJ       = "/fs1/projects/unlearning_pretraining/Proj_code"
DATA       = f"{PROJ}/data/train.jsonl"
OUT        = f"{PROJ}/generated/class_samples.jsonl"
JUDGE_PATH = "/fs1/shared/model/llm/Asclepius-Llama3-8B"
SEED       = 42
random.seed(SEED)

RANKING = """
╔══════════════════════════════════════════════════════════════════╗
║  MODEL RANKING — clinical text evaluation (on-cluster)          ║
╠══════════════════════════════════════════════════════════════════╣
║  1. Asclepius-Llama3-8B   ← CHOSEN                              ║
║     Only clinical-domain fine-tune available; 8B instruct;      ║
║     fast for 36 inferences (~3 min); knows clinical ontology.   ║
║  2. Llama-3.1-8B-Instruct — same size/speed, no clinical spec.  ║
║  3. Qwen3.5-27B-Claude    — stronger general judge, 3× slower,  ║
║     no clinical domain; overkill for coherence scoring.         ║
║  4. llama70b / Llama-4-*  — most capable but very slow here.    ║
║  ✗  Clinical-Longformer   — encoder-only, NOT a generative judge║
║  ✗  Llama-3.2-1B          — too small for reliable scoring.     ║
╠══════════════════════════════════════════════════════════════════╣
║  Off-cluster gap: Meditron-70B or OpenBioLLM-70B would give     ║
║  marginally better clinical coherence scoring; BioMistral and   ║
║  PubMedBERT are confirmed unavailable. Asclepius is adequate.   ║
╚══════════════════════════════════════════════════════════════════╝
"""
print(RANKING, flush=True)

# ── Load & group by ICD category ──────────────────────────────────────────────
print("Loading data/train.jsonl ...", flush=True)
records = defaultdict(list)
with open(DATA) as f:
    for line in f:
        r = json.loads(line)
        cat = r.get("control_codes", {}).get("icd_category", "Unknown")
        records[cat].append(r)
counts = {c: len(v) for c, v in records.items()}
print(f"  {sum(counts.values()):,} records, {len(counts)} categories", flush=True)

def tier_of(n):
    if n > 10_000: return "MAJORITY"
    if n >= 500:   return "MEDIUM"
    return "MINORITY"

by_tier = defaultdict(list)
for c, n in counts.items():
    by_tier[tier_of(n)].append((c, n))
for t in by_tier:
    by_tier[t].sort(key=lambda x: x[1])   # ascending by count

# ── Span-representative sampling: pick 2 categories that span each tier ───────
def span_pick(lst, k=2):
    """Pick k items evenly spaced across a sorted list (avoids top/bottom only)."""
    if len(lst) <= k:
        return lst
    step = len(lst) / k
    return [lst[min(int(i * step + step / 2), len(lst) - 1)] for i in range(k)]

to_score = []
for tier_name in ["MAJORITY", "MEDIUM", "MINORITY"]:
    for cat, n in span_pick(by_tier[tier_name], k=2):
        pool = records[cat]
        for rec in random.sample(pool, min(2, len(pool))):
            to_score.append({
                "tier":         tier_name,
                "icd_category": cat,
                "n_category":   n,
                "note_id":      rec.get("note_id", "?"),
                "bhc_text":     rec.get("bhc_text", ""),
            })

print(f"\n  Selected {len(to_score)} notes for scoring:", flush=True)
for s in to_score:
    print(f"    [{s['tier']}] {s['icd_category'][:55]:55s}  n={s['n_category']:>6,}", flush=True)

# ── Load Asclepius judge ──────────────────────────────────────────────────────
print(f"\nLoading judge: {JUDGE_PATH} ...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(JUDGE_PATH)
model = AutoModelForCausalLM.from_pretrained(
    JUDGE_PATH, torch_dtype=torch.float16, device_map="auto"
)
model.eval()
print("  Judge loaded.\n", flush=True)

def ask_digit(prompt: str) -> int:
    """Force generation to one of tokens '1'–'5' via logit bias; fallback text scan."""
    inp = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Find token IDs for bare digits "1"–"5" (handles tokenizers that add a leading space)
    digit_ids = []
    for d in "12345":
        for variant in (d, f" {d}"):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                digit_ids.append(ids[0])
    digit_ids = list(set(digit_ids))

    # Build logit_bias dict to strongly favour digit tokens
    # HF doesn't have a native logit_bias param, so we use a LogitsProcessor
    import torch as _torch
    from transformers import LogitsProcessor

    class ForceDigit(LogitsProcessor):
        def __call__(self, input_ids, scores):
            mask = _torch.full_like(scores, float("-inf"))
            for tid in digit_ids:
                mask[:, tid] = 0.0
            return scores + mask

    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=[ForceDigit()],
        )
    raw = tokenizer.decode(
        out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()
    for ch in raw:
        if ch in "12345":
            return int(ch)
    return -1

# ── Score each note ───────────────────────────────────────────────────────────
run_id    = uuid.uuid4().hex[:8]
timestamp = datetime.datetime.now().isoformat()
results   = []

print(f"Scoring {len(to_score)} notes (run_id={run_id}) ...", flush=True)
for i, s in enumerate(to_score, 1):
    toks    = tokenizer.encode(s["bhc_text"], add_special_tokens=False)
    tok_len = len(toks)
    snippet = tokenizer.decode(toks[:400], skip_special_tokens=True)

    coh = ask_digit(
        "Rate the clinical coherence of this hospital course note.\n"
        "1 = completely incoherent or non-clinical text\n"
        "5 = fluent, clinically coherent narrative\n"
        "Reply with a single digit (1-5) only.\n\n"
        f"Note:\n{snippet}\n\nScore:"
    )
    adh = ask_digit(
        f"ICD-10 category label: {s['icd_category']}\n\n"
        "Does this hospital course note reflect the above ICD-10 category?\n"
        "1 = no match at all\n"
        "5 = clearly matches the stated category\n"
        "Reply with a single digit (1-5) only.\n\n"
        f"Note:\n{snippet}\n\nScore:"
    )

    result = {
        "run_id":          run_id,
        "timestamp":       timestamp,
        "tier":            s["tier"],
        "icd_category":    s["icd_category"],
        "n_category":      s["n_category"],
        "note_id":         s["note_id"],
        "token_length":    tok_len,
        "coherence_score": coh,
        "adherence_score": adh,
        "bhc_text":        s["bhc_text"],
    }
    results.append(result)
    print(
        f"  [{i:2d}/{len(to_score)}] [{s['tier']}] "
        f"{s['icd_category'][:45]:45s}  n={s['n_category']:>6,}  "
        f"tok={tok_len:5d}  coh={coh}  adh={adh}",
        flush=True,
    )

# ── Append to output file ─────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "a") as fh:
    for r in results:
        fh.write(json.dumps(r) + "\n")
print(f"\nAppended {len(results)} records → {OUT}  (run_id={run_id})", flush=True)

# ── Summary table ─────────────────────────────────────────────────────────────
ORDER = ["MAJORITY", "MEDIUM", "MINORITY"]
W = 99
print(f"\n{'─'*W}")
print(f"{'Tier':<10} {'Category':<52} {'n':>7}  {'tok':>5}  {'coh':>4}  {'adh':>4}  note_id")
print(f"{'─'*W}")
for r in sorted(results, key=lambda x: (ORDER.index(x["tier"]), -x["n_category"])):
    print(
        f"{r['tier']:<10} {r['icd_category'][:52]:<52} {r['n_category']:>7,}  "
        f"{r['token_length']:>5}  {r['coherence_score']:>4}  {r['adherence_score']:>4}  {r['note_id']}"
    )
print(f"{'─'*W}")

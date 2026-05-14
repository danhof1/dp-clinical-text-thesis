#!/usr/bin/env python3
"""MTSamples + PMC data audit for Track B unlearning experiment."""
import csv, json, hashlib, random, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

MTSAMPLES = "/fs1/projects/unlearning_pretraining/mtsamples/mtsamples.csv"
PMC       = "/fs1/projects/unlearning_pretraining/PMC/pmc_case_reports.jsonl"
TOKENIZER = "/fs1/shared/model/llm/Llama-3.1-8B-Instruct"

MEDICAL_TERMS = {
    "patient","diagnosis","treatment","surgery","medication","symptoms","procedure",
    "physician","hospital","clinical","medical","disease","condition","therapy",
    "history","pain","blood","heart","lung","liver","kidney","cancer","tumor",
    "infection","acute","chronic","bilateral","anterior","posterior","examination"
}

def is_clinical(t):
    tl = t.lower()
    return sum(1 for w in MEDICAL_TERMS if w in tl) >= 3

print("Loading tokenizer...", flush=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(TOKENIZER)
print("Tokenizer loaded.\n", flush=True)

def ntoks(text):
    return len(tok.encode(text, add_special_tokens=False))

def stats(vals):
    a = np.array(vals, dtype=float)
    return (f"min={int(a.min())} p5={int(np.percentile(a,5))} "
            f"med={int(np.median(a))} p95={int(np.percentile(a,95))} max={int(a.max())}")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — MTSamples
# ─────────────────────────────────────────────────────────────────────────────
print("="*66)
print("TASK 1 — MTSamples Audit")
print("="*66, flush=True)

rows = []
with open(MTSAMPLES, newline="", encoding="utf-8", errors="replace") as f:
    for row in csv.DictReader(f):
        rows.append(row)
print(f"Rows in CSV (incl. blanks): {len(rows)}")

n_empty = n_dup = n_short = n_nonclin = 0
seen = set()
spec_recs = defaultdict(list)

for r in rows:
    text = (r.get("transcription") or "").strip()
    spec = (r.get("medical_specialty") or "Unknown").strip()
    if not text:
        n_empty += 1
        continue
    h = hashlib.md5(text.encode()).hexdigest()
    if h in seen:
        n_dup += 1
        continue
    seen.add(h)
    spec_recs[spec].append({
        "text": text,
        "name": (r.get("sample_name") or "").strip(),
        "desc": (r.get("description") or "").strip(),
        "spec": spec,
    })

usable = sum(len(v) for v in spec_recs.values())
print(f"Usable (non-empty, non-dup): {usable}")
print(f"  Empty/missing:  {n_empty}")
print(f"  Exact dupes:    {n_dup}")

all_texts = [r["text"] for v in spec_recs.values() for r in v]
char_lens = [len(t) for t in all_texts]

random.seed(42)
tok_sample = random.sample(all_texts, min(600, len(all_texts)))
print(f"\nComputing token lengths on {len(tok_sample)}-record sample...", flush=True)
tok_lens_sample = [ntoks(t) for t in tok_sample]

print(f"\nChar lengths (all {usable}): {stats(char_lens)}")
print(f"Token lengths (sample {len(tok_sample)}): {stats(tok_lens_sample)}")

for t in all_texts:
    approx = len(t) // 4
    if approx < 50:
        n_short += 1
    if not is_clinical(t):
        n_nonclin += 1

print(f"\nQuality flags:")
print(f"  < 50 tokens (approx):    {n_short}")
print(f"  Possibly non-clinical:   {n_nonclin}")

print(f"\nSpecialty breakdown ({len(spec_recs)} specialties):")
for spec, recs in sorted(spec_recs.items(), key=lambda x: -len(x[1])):
    print(f"  {spec:<55} n={len(recs):>4}  ({100*len(recs)/usable:5.1f}%)")

# Forget-set split: prepare.py uses min_note_tokens=128 → ~512 chars
eligible = [r for v in spec_recs.values() for r in v if len(r["text"]) >= 512]
n_members  = min(2000, len(eligible))
n_nonmembers = min(1000, len(eligible) - n_members)
print(f"\nForget-set feasibility (>= ~128 tok / 512 chars):")
print(f"  Eligible: {len(eligible)} / {usable}")
print(f"  -> members (forget): {n_members}   nonmembers: {n_nonmembers}")
spec_elig = Counter(r["spec"] for r in eligible)
print(f"  Coverage: {len(spec_elig)} specialties  "
      f"min={min(spec_elig.values())}  max={max(spec_elig.values())}")

print(f"\n--- SPOT CHECK (5 per specialty, top 6) ---")
for spec, recs in sorted(spec_recs.items(), key=lambda x: -len(x[1]))[:6]:
    print(f"\n[{spec}] (n={len(recs)})")
    for r in recs[:5]:
        preview = r["text"][:220].replace("\n", " ")
        print(f"  [{r['name'][:35]:35}] {preview}...")

print(f"\n{'='*66}")
print("MTSamples VERDICT: ", end="")
issues = []
if n_empty > 10:
    issues.append(f"{n_empty} empty records")
if n_dup > 5:
    issues.append(f"{n_dup} duplicates")
if n_nonclin > 200:
    issues.append(f"{n_nonclin} possibly non-clinical")
if len(eligible) < 3100:
    issues.append(f"only {len(eligible)} records pass length filter")
print("NEEDS FILTERING — " + "; ".join(issues) if issues else "GOOD — ready for unlearning as-is")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — PMC Case Reports
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*66)
print("TASK 2 — PMC Case Reports Audit")
print("="*66, flush=True)

pmc_raw = []
with open(PMC) as f:
    for line in f:
        line = line.strip()
        if line:
            pmc_raw.append(json.loads(line))
print(f"Total PMC records: {len(pmc_raw)}")
print(f"Keys: {list(pmc_raw[0].keys())}")

pmc_empty = pmc_dup = pmc_short = pmc_nonclin = 0
seen_pmc = set()
pmc_spec = defaultdict(list)

for r in pmc_raw:
    text = (r.get("text") or "").strip()
    spec = (r.get("specialty") or "unknown").strip()
    if not text:
        pmc_empty += 1
        continue
    h = hashlib.md5(text.encode()).hexdigest()
    if h in seen_pmc:
        pmc_dup += 1
        continue
    seen_pmc.add(h)
    pmc_spec[spec].append({
        "text": text,
        "pmcid": r.get("pmcid", ""),
        "title": (r.get("title") or "")[:60],
        "spec": spec,
    })

pmc_usable = sum(len(v) for v in pmc_spec.values())
pmc_texts  = [r["text"] for v in pmc_spec.values() for r in v]
pmc_chars  = [len(t) for t in pmc_texts]

print(f"\nUsable: {pmc_usable}  Empty: {pmc_empty}  Dupes: {pmc_dup}")
print(f"Char lengths: {stats(pmc_chars)}")

print(f"\nComputing token lengths (all {pmc_usable} records)...", flush=True)
pmc_tok = [ntoks(t) for t in pmc_texts]
print(f"Token lengths: {stats(pmc_tok)}")

p95 = int(np.percentile(pmc_tok, 95))
rec_maxlen = 512 if p95 <= 512 else (768 if p95 <= 768 else 1024)
print(f"\np95 = {p95} tokens  ->  recommended max_length: {rec_maxlen}")

for t in pmc_texts:
    if len(t) // 4 < 50:
        pmc_short += 1
    if not is_clinical(t):
        pmc_nonclin += 1

print(f"Quality: short={pmc_short}  non-clinical={pmc_nonclin}")

after_min  = sum(1 for l in pmc_tok if l >= 128)
after_1024 = sum(1 for l in pmc_tok if 128 <= l <= 1024)
after_512  = sum(1 for l in pmc_tok if 128 <= l <= 512)
print(f"\nLength filter impact (min 128 tokens):")
print(f"  Pass min: {after_min}   cap@512: {after_512}   cap@1024: {after_1024}")

print(f"\nSpecialty breakdown ({len(pmc_spec)} specialties):")
for spec, recs in sorted(pmc_spec.items(), key=lambda x: -len(x[1])):
    print(f"  {spec:<50} n={len(recs):>4}  ({100*len(recs)/pmc_usable:5.1f}%)")

print(f"\n--- PMC SPOT CHECK (3 per specialty, top 8) ---")
for spec, recs in sorted(pmc_spec.items(), key=lambda x: -len(x[1]))[:8]:
    print(f"\n[{spec}] (n={len(recs)})")
    for r in recs[:3]:
        preview = r["text"][:280].replace("\n", " ")
        print(f"  [{r['pmcid']} {r['title'][:40]:40}] {preview}...")

print(f"\n{'='*66}")
print("PMC VERDICT: ", end="")
pmc_issues = []
if pmc_empty > 20:
    pmc_issues.append(f"{pmc_empty} empty")
if pmc_dup > 10:
    pmc_issues.append(f"{pmc_dup} dupes")
if pmc_nonclin > 500:
    pmc_issues.append(f"{pmc_nonclin} possibly non-clinical")
if after_min < 7000:
    pmc_issues.append(f"only {after_min} pass length filter")
print("NEEDS FILTERING — " + "; ".join(pmc_issues) if pmc_issues else "GOOD — ready as-is")
print(f"Recommended max_length={rec_maxlen}")
print("\nAudit complete.", flush=True)

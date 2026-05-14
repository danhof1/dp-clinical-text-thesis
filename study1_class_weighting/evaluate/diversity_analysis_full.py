"""
diversity_analysis_full.py — Full-coverage diversity sweep
Track A: real_bhc + data0_base + data1_sgd + 6 strategies x 4 epsilons (minus inv-freq at eps=inf).
Output: eval/diversity_analysis_full.json
"""

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJ = Path("/fs1/projects/unlearning_pretraining/Proj_code")
SUBSAMPLE = 1000
SELF_BLEU_SAMPLE = 500
SEED = 42

GEN = PROJ / "generated/mimic"

RUNS = {
    "real_bhc": ("real", None),
    "data0_base": ("synthetic_bhc", GEN / "data0_base/synthetic_bhc.jsonl"),
    "data1_sgd": ("synthetic_bhc", GEN / "data1_sgd/synthetic_bhc.jsonl"),

    "eps05_unweighted":        ("synthetic_bhc", GEN / "eps05/synthetic_bhc.jsonl"),
    "eps05_sqrt10":            ("synthetic_bhc", GEN / "eps05_sqrt10_mimic/synthetic_bhc.jsonl"),
    "eps05_log10":             ("synthetic_bhc", GEN / "eps05_log10_mimic/synthetic_bhc.jsonl"),
    "eps05_power03":           ("synthetic_bhc", GEN / "eps05_power03cap10_mimic/synthetic_bhc.jsonl"),
    "eps05_eff0999":           ("synthetic_bhc", GEN / "eps05_eff0999_mimic/synthetic_bhc.jsonl"),
    "eps05_invfreq":           ("synthetic_bhc", GEN / "data2_eps05_weighted/synthetic_bhc.jsonl"),

    "eps1_unweighted":         ("synthetic_bhc", GEN / "eps1/synthetic_bhc.jsonl"),
    "eps1_sqrt10":             ("synthetic_bhc", GEN / "eps1_sqrt10_mimic/synthetic_bhc.jsonl"),
    "eps1_log10":              ("synthetic_bhc", GEN / "eps1_log10_mimic/synthetic_bhc.jsonl"),
    "eps1_power03":            ("synthetic_bhc", GEN / "eps1_power03cap10_mimic/synthetic_bhc.jsonl"),
    "eps1_eff0999":            ("synthetic_bhc", GEN / "eps1_eff0999_mimic/synthetic_bhc.jsonl"),
    "eps1_invfreq":            ("synthetic_bhc", GEN / "data2_eps1_weighted/synthetic_bhc.jsonl"),

    "eps4_unweighted":         ("synthetic_bhc", GEN / "eps4/synthetic_bhc.jsonl"),
    "eps4_sqrt10":             ("synthetic_bhc", GEN / "eps4_sqrt10_mimic/synthetic_bhc.jsonl"),
    "eps4_log10":              ("synthetic_bhc", GEN / "eps4_log10_mimic/synthetic_bhc.jsonl"),
    "eps4_power03":            ("synthetic_bhc", GEN / "eps4_power03cap10_mimic/synthetic_bhc.jsonl"),
    "eps4_eff0999":            ("synthetic_bhc", GEN / "eps4_eff0999_mimic/synthetic_bhc.jsonl"),
    "eps4_invfreq":            ("synthetic_bhc", GEN / "data2_eps4_weighted/synthetic_bhc.jsonl"),

    "epsinf_unweighted":       ("synthetic_bhc", GEN / "eps_inf/synthetic_bhc.jsonl"),
    "epsinf_sqrt10":           ("synthetic_bhc", GEN / "epsinf_sqrt10_mimic/synthetic_bhc.jsonl"),
    "epsinf_log10":            ("synthetic_bhc", GEN / "epsinf_log10_mimic/synthetic_bhc.jsonl"),
    "epsinf_power03":          ("synthetic_bhc", GEN / "epsinf_power03cap10_mimic/synthetic_bhc.jsonl"),
    "epsinf_eff0999":          ("synthetic_bhc", GEN / "epsinf_eff0999_mimic/synthetic_bhc.jsonl"),
}

FEATURIZER = "/fs1/shared/model/llm/Asclepius-Llama3-8B"
OUT_PATH = PROJ / "eval/diversity_analysis_full.json"
CHECKPOINT_EVERY = 1


def load_texts_real(max_n=None):
    texts = []
    with open(PROJ / "data/train.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            t = rec.get("bhc_text") or rec.get("text") or ""
            if t and len(t) > 50:
                texts.append(t)
    if max_n and len(texts) > max_n:
        random.seed(SEED)
        texts = random.sample(texts, max_n)
    return texts


def load_texts_syn(path, field, max_n=None):
    texts = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            t = rec.get(field, rec.get("text", rec.get("generated_bhc", "")))
            if t and len(t) > 50:
                texts.append(t)
    if max_n and len(texts) > max_n:
        random.seed(SEED)
        texts = random.sample(texts, max_n)
    return texts


def tokenize_simple(text):
    return text.lower().split()


def distinct_n(texts, n):
    total = 0
    unique = set()
    for text in texts:
        tokens = tokenize_simple(text)
        for i in range(len(tokens) - n + 1):
            unique.add(tuple(tokens[i:i + n]))
            total += 1
    return len(unique) / total if total > 0 else 0


def bleu_single(hyp, references, max_n=4):
    hyp_tokens = tokenize_simple(hyp)
    if len(hyp_tokens) == 0:
        return 0.0
    clipped, total = 0, 0
    for n in range(1, max_n + 1):
        hyp_ngrams = Counter()
        for i in range(len(hyp_tokens) - n + 1):
            hyp_ngrams[tuple(hyp_tokens[i:i + n])] += 1
        max_ref = Counter()
        for ref in references:
            rt = tokenize_simple(ref)
            r_ng = Counter()
            for i in range(len(rt) - n + 1):
                r_ng[tuple(rt[i:i + n])] += 1
            for ng, c in r_ng.items():
                if c > max_ref[ng]:
                    max_ref[ng] = c
        for ng, c in hyp_ngrams.items():
            clipped += min(c, max_ref.get(ng, 0))
            total += c
    return clipped / total if total > 0 else 0


def self_bleu(texts, sample_n=500):
    if len(texts) > sample_n:
        random.seed(SEED)
        idx = random.sample(range(len(texts)), sample_n)
    else:
        idx = list(range(len(texts)))
    scores = []
    for i in tqdm(idx, desc="Self-BLEU"):
        refs = [texts[j] for j in range(len(texts)) if j != i]
        if len(refs) > 200:
            random.seed(SEED + i)
            refs = random.sample(refs, 200)
        scores.append(bleu_single(texts[i], refs))
    return float(np.mean(scores)) if scores else 0.0


def embed_texts(texts, model, tokenizer, batch_size=8):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            last = out.hidden_states[-1]
            lengths = inputs["attention_mask"].sum(dim=1) - 1
            last_tok = last[torch.arange(len(batch)), lengths]
            embs.append(last_tok.cpu().float().numpy())
    return np.concatenate(embs, axis=0)


def intra_set_cosine(embeddings, sample_n=1000):
    if len(embeddings) > sample_n:
        random.seed(SEED)
        idx = random.sample(range(len(embeddings)), sample_n)
        embeddings = embeddings[idx]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = embeddings / norms
    n = len(normed)
    cos = normed @ normed.T
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pair = cos[mask]
    dist = 1 - pair
    return float(np.mean(dist)), float(np.std(dist))


def save_checkpoint(results):
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    random.seed(SEED)

    results = {}
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing entries from {OUT_PATH}")

    # Phase 1: load all texts
    all_texts = {}
    for name, (field, path) in RUNS.items():
        if path is None:
            texts = load_texts_real(max_n=SUBSAMPLE)
        else:
            if not path.exists():
                print(f"SKIP {name}: {path} not found")
                continue
            texts = load_texts_syn(path, field=field, max_n=SUBSAMPLE)
        print(f"{name}: loaded {len(texts)} texts")
        all_texts[name] = texts

    # Phase 2: CPU metrics (distinct-n + self-BLEU). Resume-safe.
    for name, texts in all_texts.items():
        if name in results and "self_bleu" in results[name]:
            print(f"  [skip CPU metrics] {name} — already done")
            continue
        print(f"\n{'='*60}\nCPU: {name} (n={len(texts)})")
        d1 = distinct_n(texts, 1)
        d2 = distinct_n(texts, 2)
        d3 = distinct_n(texts, 3)
        sb = self_bleu(texts, sample_n=SELF_BLEU_SAMPLE)
        print(f"  d1={d1:.4f} d2={d2:.4f} d3={d3:.4f} sbleu={sb:.4f}")
        results[name] = {
            "n": len(texts),
            "distinct_1": round(d1, 4),
            "distinct_2": round(d2, 4),
            "distinct_3": round(d3, 4),
            "self_bleu": round(sb, 4),
        }
        save_checkpoint(results)

    # Phase 3: GPU embedding cosine
    needs_embed = [n for n in all_texts if "mean_cosine_dist" not in results.get(n, {})]
    if not needs_embed:
        print("\nAll cosine distances already computed; skipping embedding phase.")
        return
    print(f"\nLoading embedding model for {len(needs_embed)} runs...")
    tok = AutoTokenizer.from_pretrained(FEATURIZER)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        FEATURIZER, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="auto",
    )
    model.eval()

    for name in needs_embed:
        print(f"\nEmbedding: {name} (n={len(all_texts[name])})")
        embs = embed_texts(all_texts[name], model, tok)
        m, s = intra_set_cosine(embs)
        print(f"  mean_cos={m:.4f}  std_cos={s:.4f}")
        results[name]["mean_cosine_dist"] = round(m, 4)
        results[name]["std_cosine_dist"] = round(s, 4)
        save_checkpoint(results)

    print(f"\nDone. {len(results)} entries -> {OUT_PATH}")


if __name__ == "__main__":
    main()

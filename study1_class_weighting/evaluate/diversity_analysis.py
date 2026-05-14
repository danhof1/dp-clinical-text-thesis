"""
diversity_analysis.py — Section 6 of Study 1 advisor response
Computes Self-BLEU, Distinct-n, and embedding-based intra-set distance
for 6 synthetic runs + real BHCs at eps=4.

Run on cluster:
    cd /fs1/projects/unlearning_pretraining/Proj_code
    CUDA_VISIBLE_DEVICES=0 python /path/to/diversity_analysis.py

Output: eval/diversity_analysis.json
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

RUNS = {
    "real_bhc": None,
    "data1_sgd": PROJ / "generated/mimic/data1_sgd/synthetic_bhc.jsonl",
    "eps4_unweighted": PROJ / "generated/mimic/eps4/synthetic_bhc.jsonl",
    "eps4_sqrt10": PROJ / "generated/mimic/eps4_sqrt10_mimic/synthetic_bhc.jsonl",
    "eps4_log10": PROJ / "generated/mimic/eps4_log10_mimic/synthetic_bhc.jsonl",
    "eps4_power03": PROJ / "generated/mimic/eps4_power03cap10_mimic/synthetic_bhc.jsonl",
    "eps4_invfreq": PROJ / "generated/mimic/data2_eps4_weighted/synthetic_bhc.jsonl",
}

FEATURIZER = "/fs1/shared/model/llm/Asclepius-Llama3-8B"


def load_texts(path, field="bhc_text", max_n=None):
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
            ngram = tuple(tokens[i:i + n])
            unique.add(ngram)
            total += 1
    return len(unique) / total if total > 0 else 0


def bleu_single(hypothesis, references, max_n=4):
    from collections import Counter
    import math

    hyp_tokens = tokenize_simple(hypothesis)
    if len(hyp_tokens) == 0:
        return 0.0

    clipped_counts = 0
    total_counts = 0
    for n in range(1, max_n + 1):
        hyp_ngrams = Counter()
        for i in range(len(hyp_tokens) - n + 1):
            hyp_ngrams[tuple(hyp_tokens[i:i + n])] += 1

        max_ref_ngrams = Counter()
        for ref in references:
            ref_tokens = tokenize_simple(ref)
            ref_ngrams = Counter()
            for i in range(len(ref_tokens) - n + 1):
                ref_ngrams[tuple(ref_tokens[i:i + n])] += 1
            for ngram, count in ref_ngrams.items():
                max_ref_ngrams[ngram] = max(max_ref_ngrams[ngram], count)

        for ngram, count in hyp_ngrams.items():
            clipped_counts += min(count, max_ref_ngrams.get(ngram, 0))
            total_counts += count

    return clipped_counts / total_counts if total_counts > 0 else 0


def self_bleu(texts, sample_n=500):
    if len(texts) > sample_n:
        random.seed(SEED)
        indices = random.sample(range(len(texts)), sample_n)
    else:
        indices = list(range(len(texts)))

    scores = []
    for i in tqdm(indices, desc="Self-BLEU"):
        refs = [texts[j] for j in range(len(texts)) if j != i]
        if len(refs) > 200:
            random.seed(SEED + i)
            refs = random.sample(refs, 200)
        scores.append(bleu_single(texts[i], refs))
    return np.mean(scores)


def embed_texts(texts, model, tokenizer, batch_size=8):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            mask = inputs["attention_mask"]
            lengths = mask.sum(dim=1) - 1
            last_tok = last_hidden[torch.arange(len(batch)), lengths]
            embeddings.append(last_tok.cpu().float().numpy())
    return np.concatenate(embeddings, axis=0)


def intra_set_cosine(embeddings, sample_n=1000):
    if len(embeddings) > sample_n:
        random.seed(SEED)
        idx = random.sample(range(len(embeddings)), sample_n)
        embeddings = embeddings[idx]

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = embeddings / norms

    n = len(normed)
    cos_sims = normed @ normed.T
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pairwise = cos_sims[mask]

    distances = 1 - pairwise
    return float(np.mean(distances)), float(np.std(distances))


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    random.seed(SEED)

    all_texts = {}
    for name, path in RUNS.items():
        if name == "real_bhc":
            texts = load_texts(PROJ / "data/train.jsonl", field="bhc_text", max_n=SUBSAMPLE)
        else:
            texts = load_texts(path, field="synthetic_bhc", max_n=SUBSAMPLE)
        print(f"{name}: loaded {len(texts)} texts")
        all_texts[name] = texts

    results = {}
    for name, texts in all_texts.items():
        print(f"\n{'='*60}")
        print(f"Processing: {name} (n={len(texts)})")

        d1 = distinct_n(texts, 1)
        d2 = distinct_n(texts, 2)
        d3 = distinct_n(texts, 3)
        print(f"  Distinct-1={d1:.4f}, Distinct-2={d2:.4f}, Distinct-3={d3:.4f}")

        sb = self_bleu(texts, sample_n=SELF_BLEU_SAMPLE)
        print(f"  Self-BLEU={sb:.4f}")

        results[name] = {
            "n": len(texts),
            "distinct_1": round(d1, 4),
            "distinct_2": round(d2, 4),
            "distinct_3": round(d3, 4),
            "self_bleu": round(sb, 4),
        }

    print(f"\n{'='*60}")
    print("Loading embedding model...")
    tokenizer = AutoTokenizer.from_pretrained(FEATURIZER)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        FEATURIZER, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="auto",
    )
    model.eval()

    for name, texts in all_texts.items():
        print(f"\nEmbedding: {name} (n={len(texts)})")
        embs = embed_texts(texts, model, tokenizer)
        mean_dist, std_dist = intra_set_cosine(embs)
        print(f"  Mean cosine distance={mean_dist:.4f}, SD={std_dist:.4f}")
        results[name]["mean_cosine_dist"] = round(mean_dist, 4)
        results[name]["std_cosine_dist"] = round(std_dist, 4)

    out_path = PROJ / "eval/diversity_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")

    print("\n=== SUMMARY TABLE ===")
    print(f"{'Run':<25} | Self-BLEU↓ | D-1 ↑  | D-2 ↑  | D-3 ↑  | Cos-Dist ↑")
    print("-" * 85)
    for name in RUNS:
        r = results[name]
        print(f"{name:<25} | {r['self_bleu']:.4f}    | {r['distinct_1']:.4f} | {r['distinct_2']:.4f} | {r['distinct_3']:.4f} | {r['mean_cosine_dist']:.4f}")


if __name__ == "__main__":
    main()

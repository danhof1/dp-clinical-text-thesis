"""
mauve_sensitivity.py
Systematic test: same data, vary N and num_buckets to reproduce the 8x MAUVE gap.

Key difference:
  - per_cat_eval / 05_evaluate: N=5000, num_buckets auto or explicit 500
  - mauve_sanity.py: N=500, num_buckets=250

MAUVE 'auto' rule: num_buckets = N / 10
  N=500  -> auto = 50 buckets
  N=5000 -> auto = 500 buckets

We test N in [500, 1000, 2000, 5000] and num_buckets in [auto, 50, 100, 250, 500, 1000].
"""

import json, random, time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJ     = "/fs1/projects/unlearning_pretraining/Proj_code"
GEN_ROOT = f"{PROJ}/generated/mimic"
JUDGE    = "/fs1/shared/model/llm/Asclepius-Llama3-8B"
SEED     = 42
random.seed(SEED)
np.random.seed(SEED)


def load_texts(jsonl_path, field, n=None):
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            t = r.get(field, "").strip()
            if t:
                rows.append(t)
    if n is not None:
        rows = random.sample(rows, min(n, len(rows)))
    return rows


@torch.no_grad()
def embed(texts, model, tok, max_length=512, batch_size=8):
    vecs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        enc = tok(chunk, return_tensors="pt", truncation=True,
                  max_length=max_length, padding=True).to(model.device)
        out = model(**enc, output_hidden_states=True)
        h   = out.hidden_states[-1]
        lens = enc["attention_mask"].sum(dim=1) - 1
        for j, l in enumerate(lens):
            vecs.append(h[j, l].cpu().float().numpy())
    return np.stack(vecs)


def run_mauve(p_feats, q_feats, num_buckets, device_id=0):
    import mauve as mauve_lib
    t0 = time.time()
    try:
        result = mauve_lib.compute_mauve(
            p_features=p_feats, q_features=q_feats,
            device_id=device_id, verbose=False,
            num_buckets=num_buckets,
        )
        score = float(result.mauve)
        actual_buckets = result.num_buckets
    except Exception as e:
        score = None
        actual_buckets = num_buckets
        print(f"  MAUVE error: {e}")
    dt = time.time() - t0
    return score, actual_buckets, dt


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = 0 if device == "cuda" else -1

    base_jsonl  = f"{GEN_ROOT}/data0_base/synthetic_bhc.jsonl"
    sqrt4_jsonl = f"{GEN_ROOT}/eps4_sqrt10_mimic/synthetic_bhc.jsonl"

    print(f"Loading Asclepius from {JUDGE}...")
    tok = AutoTokenizer.from_pretrained(JUDGE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="auto",
    )
    model.eval()
    print("  Loaded.\n")

    print("Loading texts...")
    MAX_N = 5000
    all_real = load_texts(base_jsonl, "original_bhc")
    all_syn  = load_texts(sqrt4_jsonl, "synthetic_bhc")
    print(f"  Available: {len(all_real)} real, {len(all_syn)} synthetic")

    print(f"\nEmbedding {MAX_N} real texts...")
    real_feats_full = embed(random.sample(all_real, MAX_N), model, tok)
    print(f"Embedding {MAX_N} synthetic texts...")
    syn_feats_full  = embed(random.sample(all_syn, MAX_N), model, tok)
    print("  Done.\n")

    Ns = [500, 1000, 2000, 5000]
    bucket_specs = ['auto', 50, 100, 250, 500, 1000]

    print("=" * 90)
    print(f"{'N':>6}  {'buckets_req':>12}  {'buckets_actual':>14}  {'MAUVE':>8}  {'seconds':>8}")
    print("-" * 90)

    results = []
    for n in Ns:
        p = real_feats_full[:n]
        q = syn_feats_full[:n]
        for bspec in bucket_specs:
            if bspec == 'auto':
                nb = 'auto'
                expected_auto = max(2, round(n / 10))
                label = f"auto(={expected_auto})"
            else:
                nb = bspec
                label = str(bspec)
            score, actual_b, dt = run_mauve(p, q, nb, device_id)
            score_str = f"{score:.4f}" if score is not None else "FAILED"
            print(f"{n:>6}  {label:>12}  {actual_b:>14}  {score_str:>8}  {dt:>8.1f}s")
            results.append({
                "N": n,
                "buckets_requested": bspec if isinstance(bspec, int) else "auto",
                "buckets_actual": actual_b,
                "mauve": score,
                "seconds": round(dt, 1),
            })

    print("=" * 90)
    print()
    print("KEY COMPARISON:")
    print("  mauve_sanity.py (N=500, buckets=250):         reported 0.050")
    print("  per_cat_eval    (N=5000, buckets=500):        reported 0.0063")
    print()

    out = f"{PROJ}/eval/mauve_sensitivity_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

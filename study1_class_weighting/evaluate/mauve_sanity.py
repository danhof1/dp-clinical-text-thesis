"""
mauve_sanity.py

Sanity-checks the MAUVE scores we're seeing (~0.005) by computing:

  Test 1 — real vs real (upper bound):
    Split original_bhc into two halves. MAUVE between them should be high
    (~0.5-1.0) if the featurizer + MAUVE pipeline is working correctly.

  Test 2 — real vs synthetic (baseline):
    original_bhc vs synthetic_bhc from data0_base (no DP, unconstrained).
    This is what our existing pipeline reports as ~0.0054.

  Test 3 — real vs synthetic (sqrt_cap10 eps=4):
    The primary thesis condition, reported as ~0.0063.

  Test 4 — real A vs real B shuffled (repeatability):
    Same as Test 1 but with a different random seed. If Test 1 >> Test 2,
    featurizer is fine and 0.005 means genuine distributional gap.
    If Test 1 ≈ 0.005, featurizer/MAUVE combination has a ceiling issue.

Prints results to stdout and writes mauve_sanity_results.json to EVAL_DIR.
"""

import json
import random
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJ     = "/fs1/projects/unlearning_pretraining/Proj_code"
GEN_ROOT = f"{PROJ}/generated/mimic"
EVAL_DIR = f"{PROJ}/eval"
JUDGE    = "/fs1/shared/model/llm/Asclepius-Llama3-8B"
N        = 500   # samples per side — matches what 05_evaluate.py uses
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


def run_mauve(p_feats, q_feats, label, device_id=0):
    import mauve as mauve_lib
    n_buckets = min(500, len(p_feats) // 2)
    try:
        result = mauve_lib.compute_mauve(
            p_features=p_feats, q_features=q_feats,
            device_id=device_id, verbose=False,
            num_buckets=n_buckets,
        )
        score = float(result.mauve)
    except Exception as e:
        score = None
        print(f"  MAUVE error for {label}: {e}")
    print(f"  {label:<55}  n={len(p_feats)}  buckets={n_buckets}  MAUVE={score:.4f}" if score else
          f"  {label:<55}  FAILED")
    return score


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = 0 if device == "cuda" else -1

    base_jsonl     = f"{GEN_ROOT}/data0_base/synthetic_bhc.jsonl"
    sqrt4_jsonl    = f"{GEN_ROOT}/eps4_sqrt10_mimic/synthetic_bhc.jsonl"

    print("\n" + "="*70)
    print("MAUVE SANITY CHECK")
    print("="*70)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\nLoading Asclepius from {JUDGE}...")
    tok = AutoTokenizer.from_pretrained(JUDGE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="auto",
    )
    model.eval()
    print("  Loaded.\n")

    # ── Load texts ────────────────────────────────────────────────────────
    print(f"Loading texts (N={N} per side)...")
    all_real = load_texts(base_jsonl, "original_bhc")
    print(f"  Total real notes available: {len(all_real)}")

    # Split real into two halves for test 1 & 4
    random.shuffle(all_real)
    half = len(all_real) // 2
    real_A = all_real[:half]
    real_B = all_real[half:]
    real_A_sample = random.sample(real_A, min(N, len(real_A)))
    real_B_sample = random.sample(real_B, min(N, len(real_B)))

    syn_base   = load_texts(base_jsonl,  "synthetic_bhc", N)
    syn_sqrt4  = load_texts(sqrt4_jsonl, "synthetic_bhc", N) if Path(sqrt4_jsonl).exists() else []

    # ── Embed ─────────────────────────────────────────────────────────────
    print("\nEmbedding texts with Asclepius...")
    feats = {}
    for name, texts in [
        ("real_A",    real_A_sample),
        ("real_B",    real_B_sample),
        ("syn_base",  syn_base),
        ("syn_sqrt4", syn_sqrt4),
    ]:
        if texts:
            print(f"  Embedding {name} ({len(texts)} texts)...")
            feats[name] = embed(texts, model, tok)

    # ── Run MAUVE tests ───────────────────────────────────────────────────
    print("\n" + "-"*70)
    print("RESULTS")
    print("-"*70)
    results = {}

    # Test 1: real vs real (expected: high, ~0.5-1.0)
    print("\n[Test 1] real_A vs real_B  (UPPER BOUND — should be high)")
    results["real_vs_real"] = run_mauve(feats["real_A"], feats["real_B"],
                                         "real_A vs real_B", device_id)

    # Test 2: real vs synthetic baseline
    print("\n[Test 2] real vs synthetic (data0_base, no DP)  (pipeline reports ~0.0054)")
    results["real_vs_syn_base"] = run_mauve(feats["real_A"], feats["syn_base"],
                                             "real vs syn_base", device_id)

    # Test 3: real vs synthetic sqrt_cap10 eps=4
    if "syn_sqrt4" in feats:
        print("\n[Test 3] real vs synthetic (sqrt_cap10 eps=4)  (pipeline reports ~0.0063)")
        results["real_vs_syn_sqrt4"] = run_mauve(feats["real_A"], feats["syn_sqrt4"],
                                                  "real vs syn_sqrt4_eps4", device_id)

    # Test 4: real vs real with different seed (repeatability check)
    print("\n[Test 4] real_B vs real_A  (same as Test 1, reversed — repeatability)")
    results["real_vs_real_rev"] = run_mauve(feats["real_B"], feats["real_A"],
                                             "real_B vs real_A (reversed)", device_id)

    # ── Diagnosis ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    r_r   = results.get("real_vs_real")
    r_syn = results.get("real_vs_syn_base")
    if r_r is not None and r_syn is not None:
        ratio = r_r / r_syn if r_syn > 0 else float("inf")
        print(f"\n  real-vs-real MAUVE:        {r_r:.4f}")
        print(f"  real-vs-synthetic MAUVE:   {r_syn:.4f}")
        print(f"  Ratio (r/r ÷ r/syn):       {ratio:.1f}x")
        print()
        if r_r < 0.05:
            print("  VERDICT: Featurizer/MAUVE pipeline has a ceiling issue.")
            print("           Even real-vs-real scores near zero.")
            print("           The 0.005 scores are artifactual — do not interpret as quality signal.")
            print("           Recommendation: use a different featurizer (e.g. GPT-2) or")
            print("           report MAUVE with text input rather than Asclepius embeddings.")
        elif ratio < 5:
            print("  VERDICT: MAUVE is working but synthetic notes are genuinely far from real.")
            print("           The 0.005 scores reflect real distributional gap, not artifact.")
            print("           Per-category MAUVE may be higher (mixing effect).")
        else:
            print("  VERDICT: MAUVE is working correctly.")
            print(f"           real-vs-real={r_r:.3f} >> real-vs-synthetic={r_syn:.4f}.")
            print("           The 0.005 scores are meaningful — synthetic distribution")
            print("           is genuinely far from real in Asclepius embedding space.")
            print("           Per-category MAUVE will be the key diagnostic.")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = f"{EVAL_DIR}/mauve_sanity_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_per_side": N,
            "featurizer": JUDGE,
            "results": results,
        }, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

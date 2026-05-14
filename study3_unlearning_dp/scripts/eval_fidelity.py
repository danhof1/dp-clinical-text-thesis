"""
eval_fidelity.py — Track B fidelity metrics on RMU dp_lora generations.

Computes the same metrics as Track A (aggregate_summary.csv):
  - MAUVE score (Asclepius-Llama3-8B featurizer)
  - PPL_mean under Asclepius-Llama3-8B
  - Length KL divergence
  - Unigram KL divergence
  - Bigram KL divergence
  - Unary Jaccard overlap

Real reference: splits_v1/nonmembers (MTSamples held-out) or
                splits_v1_pmc/nonmembers (PMC held-out)

Usage:
    python eval_fidelity.py --tag rmu_eps1 --dataset mtsamples
    python eval_fidelity.py --tag rmu_eps1_pmc --dataset pmc

Output: outputs/fidelity/{tag}_fidelity.json
"""

from __future__ import annotations
import argparse
import json
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("eval.fidelity")

REPO = "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp"
ASCLEPIUS = "/fs1/shared/model/llm/Asclepius-Llama3-8B"


def load_jsonl(path: str) -> list[str]:
    texts = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("text"):
                texts.append(r["text"])
    return texts


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def kl_divergence(p_counts: Counter, q_counts: Counter) -> float:
    """KL(P || Q) in nats. Smoothed with add-1."""
    vocab = set(p_counts) | set(q_counts)
    total_p = sum(p_counts.values()) + len(vocab)
    total_q = sum(q_counts.values()) + len(vocab)
    kl = 0.0
    for w in vocab:
        p = (p_counts.get(w, 0) + 1) / total_p
        q = (q_counts.get(w, 0) + 1) / total_q
        kl += p * math.log(p / q)
    return kl


def ngram_counts(texts: list[str], n: int) -> Counter:
    counts = Counter()
    for t in texts:
        words = tokenize_words(t)
        for i in range(len(words) - n + 1):
            counts[tuple(words[i:i+n])] += 1
    return counts


def length_kl(syn_texts: list[str], real_texts: list[str]) -> float:
    """KL on word-count distributions (binned into 50-word buckets)."""
    def lengths(texts):
        return [len(tokenize_words(t)) // 50 for t in texts]
    syn_l = Counter(lengths(syn_texts))
    real_l = Counter(lengths(real_texts))
    return kl_divergence(syn_l, real_l)


def unary_jaccard(syn_texts: list[str], real_texts: list[str]) -> float:
    """Unary Jaccard: |vocab overlap| / |vocab union|."""
    syn_vocab = set(w for t in syn_texts for w in tokenize_words(t))
    real_vocab = set(w for t in real_texts for w in tokenize_words(t))
    if not syn_vocab and not real_vocab:
        return 0.0
    return len(syn_vocab & real_vocab) / len(syn_vocab | real_vocab)


@torch.no_grad()
def compute_ppl(texts: list[str], model, tokenizer, max_length: int = 512,
                device: str = "cuda", batch: int = 4) -> float:
    ppls = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        enc = tokenizer(chunk, return_tensors="pt", truncation=True,
                        max_length=max_length, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, labels=enc["input_ids"])
        # per-example loss via summing token losses
        logits = out.logits  # [B, T, V]
        labels = enc["input_ids"]  # [B, T]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        attn = enc["attention_mask"][:, 1:].contiguous()
        import torch.nn.functional as F
        loss_flat = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.shape)
        for j in range(len(chunk)):
            n_tok = attn[j].sum().item()
            if n_tok > 0:
                nll = (loss_flat[j] * attn[j]).sum().item() / n_tok
                ppls.append(math.exp(min(nll, 20.0)))
    return float(np.mean(ppls)) if ppls else float("inf")


def compute_mauve(syn_texts: list[str], real_texts: list[str],
                  featurizer_path: str, device: str) -> dict:
    try:
        import mauve as mauve_lib
        from transformers import AutoModel
        log.info("Computing MAUVE with featurizer %s", featurizer_path)
        tokenizer = AutoTokenizer.from_pretrained(featurizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(
            featurizer_path, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device)
        model.eval()

        def embed(texts):
            vecs = []
            for i in range(0, len(texts), 4):
                chunk = texts[i:i+4]
                enc = tokenizer(chunk, return_tensors="pt", truncation=True,
                                max_length=512, padding=True).to(device)
                with torch.no_grad():
                    out = model(**enc)
                    # last-token pooling (causal LM convention)
                    seq_lens = enc["attention_mask"].sum(dim=1) - 1
                    hidden = out.last_hidden_state
                    for j, sl in enumerate(seq_lens):
                        vecs.append(hidden[j, sl].cpu().float().numpy())
            return np.stack(vecs)

        p_feats = embed(real_texts[:2000])
        q_feats = embed(syn_texts[:2000])
        del model

        result = mauve_lib.compute_mauve(
            p_features=p_feats,
            q_features=q_feats,
            device_id=0 if device == "cuda" else -1,
            verbose=False,
        )
        return {
            "mauve_score": float(result.mauve),
            "n_real": len(p_feats),
            "n_synthetic": len(q_feats),
            "featuriser": featurizer_path,
        }
    except ImportError:
        log.warning("mauve-text not installed — skipping MAUVE")
        return {"mauve_score": None, "skipped": True}
    except Exception as e:
        log.error("MAUVE failed: %s", e)
        return {"mauve_score": None, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="e.g. rmu_eps1, rmu_epsinf_pmc")
    ap.add_argument("--dataset", choices=["mtsamples", "pmc"], default="mtsamples")
    ap.add_argument("--generated", default=None, help="Override generated JSONL path")
    ap.add_argument("--no_mauve", action="store_true")
    ap.add_argument("--n_real", type=int, default=1000, help="Real texts to use")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load generated texts
    gen_path = args.generated or f"{REPO}/outputs/generated/{args.tag}.jsonl"
    syn_texts = load_jsonl(gen_path)
    log.info("Loaded %d synthetic texts from %s", len(syn_texts), gen_path)

    # Load real reference texts (nonmembers = held-out, never fine-tuned on)
    splits_name = "splits_v1_pmc" if args.dataset == "pmc" else "splits_v1"
    splits = load_from_disk(f"{REPO}/outputs/{splits_name}")
    real_texts = [r["text"] for r in splits["nonmembers"]][:args.n_real]
    log.info("Loaded %d real texts from %s/nonmembers", len(real_texts), splits_name)

    results = {
        "tag": args.tag,
        "dataset": args.dataset,
        "n_synthetic": len(syn_texts),
        "n_real": len(real_texts),
    }

    # PPL under Asclepius
    log.info("Loading Asclepius for PPL...")
    tok = AutoTokenizer.from_pretrained(ASCLEPIUS)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        ASCLEPIUS, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    results["ppl_mean"] = compute_ppl(syn_texts[:500], mdl, tok, device=device)
    log.info("PPL_mean = %.2f", results["ppl_mean"])

    # MAUVE (reuse Asclepius as featurizer)
    if not args.no_mauve:
        results["mauve"] = compute_mauve(syn_texts, real_texts, ASCLEPIUS, device)
    del mdl

    # n-gram metrics (CPU, fast)
    syn_uni = ngram_counts(syn_texts, 1)
    real_uni = ngram_counts(real_texts, 1)
    syn_bi  = ngram_counts(syn_texts, 2)
    real_bi  = ngram_counts(real_texts, 2)

    results["length_kl"]   = length_kl(syn_texts, real_texts)
    results["unigram_kl"]  = kl_divergence(syn_uni, real_uni)
    results["bigram_kl"]   = kl_divergence(syn_bi, real_bi)
    results["unary_jaccard"] = unary_jaccard(syn_texts, real_texts)

    log.info("length_kl=%.4f unigram_kl=%.4f bigram_kl=%.4f jaccard=%.4f",
             results["length_kl"], results["unigram_kl"],
             results["bigram_kl"], results["unary_jaccard"])

    out_dir = Path(f"{REPO}/outputs/fidelity")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}_fidelity.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()

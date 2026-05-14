"""
Shared utilities for membership inference attacks.

All attacks return per-example SCORES where HIGHER = more likely to be a MEMBER.
We then compute ROC and report AUC and TPR at low FPR (1%, 0.1%) per
Carlini et al. 2022 "Membership Inference Attacks from First Principles".
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class AttackResult:
    name: str
    auc: float
    tpr_at_1pct_fpr: float
    tpr_at_0p1pct_fpr: float
    n_members: int
    n_nonmembers: int

    def to_dict(self):
        return {
            "attack": self.name,
            "auc": float(self.auc),
            "tpr_at_1pct_fpr": float(self.tpr_at_1pct_fpr),
            "tpr_at_0p1pct_fpr": float(self.tpr_at_0p1pct_fpr),
            "n_members": int(self.n_members),
            "n_nonmembers": int(self.n_nonmembers),
        }


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """
    labels: 1 = member, 0 = non-member
    scores: higher = more likely member
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    # first TPR value at or below the target FPR threshold
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    if idx < 0:
        return 0.0
    return float(tpr[idx])


def summarize(name: str, member_scores: np.ndarray, nonmember_scores: np.ndarray) -> AttackResult:
    labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    scores = np.concatenate([member_scores, nonmember_scores])
    auc = roc_auc_score(labels, scores)
    return AttackResult(
        name=name,
        auc=auc,
        tpr_at_1pct_fpr=tpr_at_fpr(labels, scores, 0.01),
        tpr_at_0p1pct_fpr=tpr_at_fpr(labels, scores, 0.001),
        n_members=len(member_scores),
        n_nonmembers=len(nonmember_scores),
    )


# ------- Core scoring primitives -------

@torch.no_grad()
def compute_token_log_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    max_length: int = 1024,
    device: str = "cuda",
) -> np.ndarray:
    """
    Return a 1-D numpy array of log-probabilities assigned by the model
    to each actual next-token in `text`.
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc.input_ids.to(device)
    if input_ids.shape[1] < 2:
        return np.array([], dtype=np.float32)
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, :-1, :]          # [T-1, V]
    targets = input_ids[0, 1:]                  # [T-1]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    tok_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return tok_lp.detach().cpu().numpy().astype(np.float32)


def loss_from_log_probs(tok_lp: np.ndarray) -> float:
    """Mean NLL (= cross-entropy loss per token)."""
    if tok_lp.size == 0:
        return float("inf")
    return float(-tok_lp.mean())


def min_k_percent_score(tok_lp: np.ndarray, k_pct: float = 20.0) -> float:
    """
    Min-K% Prob (Shi et al. 2024):
    Mean log-prob of the k% of tokens with LOWEST log-probability.
    Higher = more likely a training member.

    For MIA, we return this mean (high value = member), but note some
    implementations return the *negative* to get a "score to MIA threshold"
    form. We stick with the convention: higher = member.
    """
    if tok_lp.size == 0:
        return -float("inf")
    k = max(1, int(len(tok_lp) * k_pct / 100.0))
    lowest = np.sort(tok_lp)[:k]
    return float(lowest.mean())


def zlib_ratio_score(text: str, loss: float) -> float:
    """
    zlib entropy / model loss ratio (Carlini et al. 2021 extraction paper).
    Kept for backward-compat; weaker than Min-K% on modern LLMs.
    """
    if loss <= 0 or not text:
        return float("inf")
    zlib_entropy = len(zlib.compress(text.encode("utf-8")))
    return zlib_entropy / loss


def save_results(results: list[AttackResult], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)


def iter_texts(dataset) -> Iterable[str]:
    for row in dataset:
        yield row["text"]

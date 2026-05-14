"""
Unlearning algorithms.

Three methods, one interface. Each takes:
    model, forget_batch, [retain_batch], cfg
and returns a scalar loss to .backward().

References
----------
- GA:    Jang et al. ACL 2023, "Knowledge Unlearning for Mitigating Privacy Risks"
- GA+GD: Yao et al.  ACL 2024, "Machine Unlearning of Pre-trained LLMs"
         (their most robust method; gradient ascent on forget set +
          gradient descent on in-distribution retain set)
- NPO:   Zhang et al. 2024, "Negative Preference Optimization"
         (DPO-style objective that addresses the unbounded-loss problem
          of vanilla GA; uses the original frozen model as a reference)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from .common import UnlearnConfig, per_example_nll, per_token_nll


def ga_loss(
    model: PreTrainedModel,
    forget_batch: dict[str, torch.Tensor],
    cfg: UnlearnConfig,
) -> torch.Tensor:
    """
    Gradient Ascent (Jang et al. 2023).

    We MAXIMIZE the NLL on the forget set, i.e. minimize -NLL.
    This is unbounded (loss can go to -∞), so must be used with tight
    step counts / early stopping; see Yao et al. and NPO for why this
    motivates GA+GD or NPO.
    """
    loss = per_token_nll(model, forget_batch)
    return -loss


def ga_gd_loss(
    model: PreTrainedModel,
    forget_batch: dict[str, torch.Tensor],
    retain_batch: dict[str, torch.Tensor],
    cfg: UnlearnConfig,
) -> torch.Tensor:
    """
    Gradient Ascent + Gradient Descent on in-distribution data (Yao et al. 2024).

    L = -NLL(forget) + λ · NLL(retain)

    The retain term anchors the model to in-distribution text so that
    gradient ascent doesn't destroy general language modeling.
    Yao et al. found this is the most hyperparameter-robust of the
    seven methods they benchmarked on pretraining unlearning.
    """
    forget_loss = per_token_nll(model, forget_batch)
    retain_loss = per_token_nll(model, retain_batch)
    return -forget_loss + cfg.gd_weight * retain_loss


def npo_loss(
    model: PreTrainedModel,
    reference_model: PreTrainedModel,
    forget_batch: dict[str, torch.Tensor],
    cfg: UnlearnConfig,
) -> torch.Tensor:
    """
    Negative Preference Optimization (Zhang et al. 2024).

    Borrows DPO's pairwise preference loss but uses only "dispreferred"
    (forget) samples, with the original frozen model as the reference.

    L = -(2 / β) · log σ(-β · (log π(x) - log π_ref(x)))

    where π is the current model and π_ref is the original pretrained
    model (frozen). Unlike GA, this is bounded and stable.
    """
    # per-example NLLs (log π = -NLL, note sign)
    nll_current = per_example_nll(model, forget_batch)        # [B]
    with torch.no_grad():
        nll_reference = per_example_nll(reference_model, forget_batch)  # [B]

    # log π - log π_ref = (-nll_current) - (-nll_reference) = nll_ref - nll_cur
    log_ratio = nll_reference - nll_current                   # [B]

    # L = - (2/β) * log σ(-β * log_ratio)
    # maximize loss(current) relative to reference -> current becomes WORSE on forget
    loss = -(2.0 / cfg.npo_beta) * F.logsigmoid(-cfg.npo_beta * log_ratio)
    return loss.mean()


def compute_unlearn_loss(
    method: str,
    model: PreTrainedModel,
    forget_batch: dict[str, torch.Tensor],
    retain_batch: dict[str, torch.Tensor] | None,
    reference_model: PreTrainedModel | None,
    cfg: UnlearnConfig,
) -> torch.Tensor:
    if method == "ga":
        return ga_loss(model, forget_batch, cfg)
    elif method == "ga_gd":
        assert retain_batch is not None, "GA+GD requires a retain batch"
        return ga_gd_loss(model, forget_batch, retain_batch, cfg)
    elif method == "npo":
        assert reference_model is not None, "NPO requires a reference model"
        return npo_loss(model, reference_model, forget_batch, cfg)
    else:
        raise ValueError(f"unknown unlearning method: {method}")


def method_needs_retain(method: str) -> bool:
    return method == "ga_gd"


def method_needs_reference(method: str) -> bool:
    return method == "npo"

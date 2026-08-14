"""Detached, decision-level diagnostics for the independent policy's Blank class."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping, Sequence

import numpy as np
import torch


_TOLERANCE = 1.0e-6


def extract_failed_transmission_bytes(
    info: Mapping[str, object],
    *,
    expected_n_ue: int,
) -> dict[str, float]:
    """Return exact scheduled, goodput, and failed-attempt bytes for one TTI."""
    required = ("actual_ue_served_bytes", "actual_ue_goodput_bytes")
    missing = [key for key in required if key not in info]
    if missing:
        raise ValueError(f"missing protocol byte arrays: {', '.join(missing)}")

    arrays: dict[str, np.ndarray] = {}
    for key in required:
        try:
            values = np.asarray(info[key], dtype=np.float64).reshape((expected_n_ue,))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must reshape to ({expected_n_ue},)") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"{key} must contain only finite values")
        if np.any(values < 0.0):
            raise ValueError(f"{key} must contain only non-negative values")
        arrays[key] = values

    scheduled_bytes = np.sum(arrays["actual_ue_served_bytes"], dtype=np.float64)
    goodput_bytes = np.sum(arrays["actual_ue_goodput_bytes"], dtype=np.float64)
    if goodput_bytes > scheduled_bytes + _TOLERANCE:
        raise ValueError("goodput bytes must not exceed scheduled bytes")
    return {
        "scheduled_bytes": float(scheduled_bytes),
        "goodput_bytes": float(goodput_bytes),
        "failed_bytes": float(max(scheduled_bytes - goodput_bytes, 0.0)),
    }


@dataclass(frozen=True)
class IndependentPolicyDiagnostics:
    valid_ue_count: torch.Tensor
    p_blank: torch.Tensor
    p_service: torch.Tensor
    p_best_ue: torch.Tensor
    blank_margin: torch.Tensor
    margin_valid: torch.Tensor
    selected_blank: torch.Tensor
    greedy_blank: torch.Tensor
    forced_blank: torch.Tensor
    learned_blank: torch.Tensor
    fragmentation_blank: torch.Tensor
    conservative_blank: torch.Tensor


def _detached_vector(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().reshape(-1).cpu()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_inputs(
    adjusted_logits: torch.Tensor,
    probabilities: torch.Tensor,
    valid_main: torch.Tensor,
    selected: torch.Tensor,
    blank_class_index: int,
) -> None:
    _require(adjusted_logits.dim() == 2, "adjusted_logits must have shape [N,C]")
    _require(probabilities.shape == adjusted_logits.shape, "probabilities shape must match adjusted_logits")
    _require(
        valid_main.shape == (adjusted_logits.shape[0], adjusted_logits.shape[1] - 1),
        "valid_main must have shape [N,C-1]",
    )
    _require(selected.shape == (adjusted_logits.shape[0],), "selected must have shape [N]")
    _require(blank_class_index == adjusted_logits.shape[1] - 1, "blank_class_index must be the final class")
    _require(torch.isfinite(adjusted_logits).all().item(), "adjusted_logits must be finite")
    _require(torch.isfinite(probabilities).all().item(), "probabilities must be finite")
    _require(((probabilities >= 0.0) & (probabilities <= 1.0)).all().item(), "probabilities must be in [0,1]")
    _require(
        ((selected >= 0) & (selected < adjusted_logits.shape[1])).all().item(),
        "selected contains an invalid class index",
    )


def compute_independent_policy_diagnostics(
    *,
    adjusted_logits: torch.Tensor,
    probabilities: torch.Tensor,
    valid_main: torch.Tensor,
    selected: torch.Tensor,
    blank_class_index: int,
) -> IndependentPolicyDiagnostics:
    """Classify the exact already-masked distribution without changing policy behavior."""
    _validate_inputs(adjusted_logits, probabilities, valid_main, selected, blank_class_index)

    valid_ue_count = valid_main.sum(dim=-1)
    margin_valid = valid_ue_count > 0
    p_blank = probabilities[:, blank_class_index]
    p_service = probabilities[:, :blank_class_index].masked_fill(~valid_main, 0.0).sum(dim=-1)
    p_best_ue = probabilities[:, :blank_class_index].masked_fill(~valid_main, 0.0).max(dim=-1).values
    best_valid_logit = adjusted_logits[:, :blank_class_index].masked_fill(~valid_main, -torch.inf).max(dim=-1).values
    blank_margin = torch.where(
        margin_valid,
        adjusted_logits[:, blank_class_index] - best_valid_logit,
        torch.zeros_like(p_blank),
    )

    greedy_blank = adjusted_logits.argmax(dim=-1) == blank_class_index
    selected_blank = selected == blank_class_index
    forced_blank = greedy_blank & ~margin_valid
    learned_blank = greedy_blank & margin_valid
    fragmentation_blank = learned_blank & (p_service > p_blank)
    conservative_blank = learned_blank & (p_blank >= p_service)

    _require(torch.isfinite(p_service).all().item(), "p_service must be finite")
    _require(torch.isfinite(p_best_ue).all().item(), "p_best_ue must be finite")
    _require(torch.isfinite(blank_margin).all().item(), "blank_margin must be finite")
    _require(((p_blank >= 0.0) & (p_blank <= 1.0)).all().item(), "p_blank must be in [0,1]")
    _require(((p_service >= 0.0) & (p_service <= 1.0)).all().item(), "p_service must be in [0,1]")
    _require(((p_best_ue >= 0.0) & (p_best_ue <= 1.0)).all().item(), "p_best_ue must be in [0,1]")
    _require(torch.allclose(p_blank + p_service, torch.ones_like(p_blank), atol=_TOLERANCE, rtol=0.0), "p_blank + p_service must equal one")
    _require((p_best_ue <= p_service + _TOLERANCE).all().item(), "p_best_ue must not exceed p_service")
    _require(torch.equal(learned_blank, fragmentation_blank | conservative_blank), "learned Blank must partition")
    _require(not (fragmentation_blank & conservative_blank).any().item(), "Blank classes must be disjoint")
    _require(
        (p_blank[forced_blank] >= 1.0 - _TOLERANCE).all().item(),
        "forced Blank decisions must have probability one",
    )

    return IndependentPolicyDiagnostics(
        valid_ue_count=_detached_vector(valid_ue_count),
        p_blank=_detached_vector(p_blank),
        p_service=_detached_vector(p_service),
        p_best_ue=_detached_vector(p_best_ue),
        blank_margin=_detached_vector(blank_margin),
        margin_valid=_detached_vector(margin_valid),
        selected_blank=_detached_vector(selected_blank),
        greedy_blank=_detached_vector(greedy_blank),
        forced_blank=_detached_vector(forced_blank),
        learned_blank=_detached_vector(learned_blank),
        fragmentation_blank=_detached_vector(fragmentation_blank),
        conservative_blank=_detached_vector(conservative_blank),
    )


def concatenate_independent_policy_diagnostics(
    rows: Sequence[IndependentPolicyDiagnostics],
) -> IndependentPolicyDiagnostics:
    """Concatenate detached per-decision records collected across policy calls."""
    _require(bool(rows), "at least one diagnostics row is required")
    values: dict[str, torch.Tensor] = {}
    for field in fields(IndependentPolicyDiagnostics):
        tensors = [getattr(row, field.name) for row in rows]
        _require(all(tensor.dim() == 1 for tensor in tensors), f"{field.name} must be one-dimensional")
        values[field.name] = torch.cat(tensors)
    lengths = {tensor.numel() for tensor in values.values()}
    _require(len(lengths) == 1, "diagnostic fields must have equal lengths")
    return IndependentPolicyDiagnostics(**values)


def _distribution_summary(values: torch.Tensor, name: str, summary: dict[str, float]) -> None:
    summary[f"{name}_mean"] = float(values.mean().item())
    for percentile in (10, 50, 90):
        summary[f"{name}_p{percentile}"] = float(torch.quantile(values, percentile / 100.0).item())


def summarize_independent_policy_diagnostics(
    diagnostics: IndependentPolicyDiagnostics,
    *,
    classification_prefix: str,
) -> dict[str, float]:
    """Produce rollout/evaluation scalar metrics from collected decision records."""
    count = diagnostics.p_blank.numel()
    _require(count > 0, "cannot summarize empty diagnostics")
    for field in fields(IndependentPolicyDiagnostics):
        _require(getattr(diagnostics, field.name).shape == (count,), "diagnostic fields must have equal lengths")

    denominator = float(count)
    learned_count = int(diagnostics.learned_blank.sum().item())
    valid_margins = diagnostics.blank_margin[diagnostics.margin_valid]
    summary = {
        "decision_count": denominator,
        "selected_blank_ratio": float(diagnostics.selected_blank.sum().item()) / denominator,
        f"{classification_prefix}greedy_blank_ratio": float(diagnostics.greedy_blank.sum().item()) / denominator,
        f"{classification_prefix}forced_blank_ratio": float(diagnostics.forced_blank.sum().item()) / denominator,
        f"{classification_prefix}learned_blank_ratio": float(learned_count) / denominator,
        f"{classification_prefix}fragmentation_blank_ratio": float(diagnostics.fragmentation_blank.sum().item()) / denominator,
        f"{classification_prefix}conservative_blank_ratio": float(diagnostics.conservative_blank.sum().item()) / denominator,
        f"{classification_prefix}fragmentation_share_among_learned_blank": (
            float(diagnostics.fragmentation_blank.sum().item()) / learned_count if learned_count else 0.0
        ),
        f"{classification_prefix}learned_blank_count": float(learned_count),
        "mean_valid_ue_count": float(diagnostics.valid_ue_count.float().mean().item()),
        "valid_margin_count": float(valid_margins.numel()),
    }
    _distribution_summary(diagnostics.p_blank, "p_blank", summary)
    _distribution_summary(diagnostics.p_service, "p_service", summary)
    _distribution_summary(diagnostics.p_best_ue, "p_best_ue", summary)
    if valid_margins.numel() == 0:
        for suffix in ("mean", "p10", "p50", "p90"):
            summary[f"blank_margin_{suffix}"] = 0.0
    else:
        _distribution_summary(valid_margins, "blank_margin", summary)
    return summary

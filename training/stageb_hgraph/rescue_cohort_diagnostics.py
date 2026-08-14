"""Detached selected-candidate evidence for independent-policy rescue decisions."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Sequence

import torch

from .graph_ops import PackedPrgCandidates
from .independent_blank_diagnostics import IndependentPolicyDiagnostics
from .types import SparseEntityGraph


@dataclass(frozen=True)
class RescueDecisionDiagnostics:
    selected_ue: torch.Tensor
    cell_index: torch.Tensor
    local_prg_index: torch.Tensor
    selected_nonblank: torch.Tensor
    fragmentation_rescued: torch.Tensor
    selected_forced_blank: torch.Tensor
    selected_conservative_blank: torch.Tensor
    estimated_service_bytes: torch.Tensor
    quality_db: torch.Tensor
    quality_gap_db: torch.Tensor
    prg_risk: torch.Tensor
    ue_tb_err_last: torch.Tensor
    remaining_demand_ratio: torch.Tensor
    p_blank: torch.Tensor
    p_service: torch.Tensor
    p_best_ue: torch.Tensor
    success_proxy: torch.Tensor


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _detached_vector(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().reshape(-1).cpu()


def _validate_policy_row(policy_diagnostics: IndependentPolicyDiagnostics, count: int) -> None:
    for field in fields(IndependentPolicyDiagnostics):
        value = getattr(policy_diagnostics, field.name)
        _require(value.shape == (count,), f"policy diagnostics {field.name} shape mismatch")
        if value.dtype is not torch.bool:
            _require(torch.isfinite(value).all().item(), f"policy diagnostics {field.name} must be finite")


def build_rescue_decision_diagnostics(
    *,
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    token_indices: torch.Tensor,
    local_prg: int,
    selected: torch.Tensor,
    service_bytes: torch.Tensor,
    sequential_features: torch.Tensor,
    policy_diagnostics: IndependentPolicyDiagnostics,
) -> RescueDecisionDiagnostics:
    """Gather selected-class evidence for one reuse group without affecting actions."""
    meta = graph.snapshot.static_meta
    tokens = torch.as_tensor(token_indices, device=packed.ue_ids.device, dtype=torch.long).reshape(-1)
    selected = torch.as_tensor(selected, device=packed.ue_ids.device, dtype=torch.long).reshape(-1)
    count = int(tokens.numel())
    _require(count == meta.n_cell, "token_indices must contain one token per cell")
    _require(0 <= local_prg < meta.n_prg, "local_prg is out of range")
    _require(selected.shape == (count,), "selected shape mismatch")
    _require(((tokens >= 0) & (tokens < packed.num_prg_tokens)).all().item(), "token index is out of range")
    _require(
        torch.equal(
            meta.prg_local_index.to(device=tokens.device, dtype=torch.long).index_select(0, tokens),
            torch.full_like(tokens, local_prg),
        ),
        "token local PRG mismatch",
    )
    _require(packed.edge_attr is not None and packed.edge_attr.dim() == 3, "packed base edge attributes are required")
    _require(packed.edge_attr.shape[2] >= 5, "packed base edge width must be at least five")
    _require(graph.snapshot.ue_features.values.shape[1] > 5, "UE features require index five")
    _require(service_bytes.shape == packed.ue_ids.shape, "service_bytes shape mismatch")
    _require(
        sequential_features.shape[:2] == (count, packed.max_candidates)
        and sequential_features.dim() == 3
        and sequential_features.shape[2] > 0,
        "sequential_features shape mismatch",
    )
    _require(((selected >= 0) & (selected <= packed.blank_class_index)).all().item(), "selected class is out of range")
    _validate_policy_row(policy_diagnostics, count)

    selected_nonblank = selected < packed.blank_class_index
    selected_ue = torch.full((count,), -1, dtype=torch.long, device=tokens.device)
    zeros = torch.zeros((count,), dtype=service_bytes.dtype, device=tokens.device)
    if packed.max_candidates > 0:
        selected_class = selected.clamp(max=packed.max_candidates - 1)
        candidate_mask = packed.mask.to(device=tokens.device).index_select(0, tokens)
        _require(
            bool(candidate_mask.gather(1, selected_class.unsqueeze(-1)).squeeze(-1)[selected_nonblank].all().item()),
            "selected nonblank class is not a candidate",
        )
        candidate_ids = packed.ue_ids.to(device=tokens.device).index_select(0, tokens)
        gathered_ue = candidate_ids.gather(1, selected_class.unsqueeze(-1)).squeeze(-1)
        _require(
            bool(((gathered_ue[selected_nonblank] >= 0) & (gathered_ue[selected_nonblank] < meta.n_ue)).all().item()),
            "selected UE is out of range",
        )
        selected_ue[selected_nonblank] = gathered_ue[selected_nonblank]
        edge_values = packed.edge_attr.to(device=tokens.device).index_select(0, tokens).gather(
            1,
            selected_class[:, None, None].expand(-1, 1, packed.edge_attr.shape[2]),
        ).squeeze(1)
        gathered_service = service_bytes.to(device=tokens.device).index_select(0, tokens).gather(
            1, selected_class.unsqueeze(-1)
        ).squeeze(-1)
        gathered_remaining = sequential_features.to(device=tokens.device).gather(
            1, selected_class[:, None, None].expand(-1, 1, sequential_features.shape[2])
        ).squeeze(1)[:, 0]
        safe_ue = gathered_ue.clamp(min=0, max=max(0, meta.n_ue - 1))
        gathered_tb_err = graph.snapshot.ue_features.values.to(device=tokens.device)[safe_ue, 5]
        _require(
            all(
                torch.isfinite(value[selected_nonblank]).all().item()
                for value in (
                    edge_values[:, 0], edge_values[:, 1], edge_values[:, 4], gathered_service,
                    gathered_remaining, gathered_tb_err,
                )
            ),
            "gathered selected-candidate values must be finite",
        )
        estimated_service_bytes = torch.where(selected_nonblank, gathered_service, zeros)
        quality_db = torch.where(selected_nonblank, edge_values[:, 0], zeros)
        quality_gap_db = torch.where(selected_nonblank, edge_values[:, 1], zeros)
        prg_risk = torch.where(selected_nonblank, edge_values[:, 4], zeros)
        ue_tb_err_last = torch.where(selected_nonblank, gathered_tb_err, zeros)
        remaining_demand_ratio = torch.where(selected_nonblank, gathered_remaining, zeros)
    else:
        _require(not bool(selected_nonblank.any().item()), "nonblank selection requires candidates")
        estimated_service_bytes = zeros
        quality_db = zeros
        quality_gap_db = zeros
        prg_risk = zeros
        ue_tb_err_last = zeros
        remaining_demand_ratio = zeros

    success_proxy = torch.where(
        selected_nonblank,
        torch.sigmoid((quality_db - 0.35 * quality_gap_db - 7.5 * prg_risk - 6.0 * ue_tb_err_last) / 4.0),
        torch.zeros_like(quality_db),
    )
    values = {
        "selected_ue": selected_ue,
        "cell_index": meta.prg_owner_cell.to(device=tokens.device, dtype=torch.long).index_select(0, tokens),
        "local_prg_index": torch.full_like(tokens, local_prg),
        "selected_nonblank": selected_nonblank,
        "fragmentation_rescued": policy_diagnostics.fragmentation_rescued.to(tokens.device),
        "selected_forced_blank": policy_diagnostics.selected_forced_blank.to(tokens.device),
        "selected_conservative_blank": policy_diagnostics.selected_conservative_blank.to(tokens.device),
        "estimated_service_bytes": estimated_service_bytes,
        "quality_db": quality_db,
        "quality_gap_db": quality_gap_db,
        "prg_risk": prg_risk,
        "ue_tb_err_last": ue_tb_err_last,
        "remaining_demand_ratio": remaining_demand_ratio,
        "p_blank": policy_diagnostics.p_blank.to(tokens.device),
        "p_service": policy_diagnostics.p_service.to(tokens.device),
        "p_best_ue": policy_diagnostics.p_best_ue.to(tokens.device),
        "success_proxy": success_proxy,
    }
    return RescueDecisionDiagnostics(**{name: _detached_vector(value) for name, value in values.items()})


def concatenate_rescue_decision_diagnostics(
    rows: Sequence[RescueDecisionDiagnostics],
) -> RescueDecisionDiagnostics:
    """Concatenate fixed-schema detached decision rows in policy execution order."""
    _require(bool(rows), "at least one rescue diagnostics row is required")
    values: dict[str, torch.Tensor] = {}
    for field in fields(RescueDecisionDiagnostics):
        tensors = [getattr(row, field.name) for row in rows]
        _require(all(tensor.dim() == 1 for tensor in tensors), f"{field.name} must be one-dimensional")
        values[field.name] = _detached_vector(torch.cat(tensors))
    lengths = {tensor.numel() for tensor in values.values()}
    _require(len(lengths) == 1, "rescue diagnostics fields must have equal lengths")
    return RescueDecisionDiagnostics(**values)

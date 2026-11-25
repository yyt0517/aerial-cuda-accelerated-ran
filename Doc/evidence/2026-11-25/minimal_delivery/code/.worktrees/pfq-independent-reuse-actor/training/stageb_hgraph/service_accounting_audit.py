"""Read-only audit for independent-actor service headroom accounting."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping, Sequence

import numpy as np
import torch

from .graph_ops import PackedPrgCandidates


NEW_TRANSMISSION_QUEUE_DEQUEUE = "new_transmission_queue_dequeue"
_TARGET_RATIO = 2.0 / 3.0
_GLOBAL_TOLERANCE = 0.05
_GROUP_TOLERANCE = 0.10
_MIN_GROUP_UE_TTIS = 50
_MIN_GROUP_ACTUAL_PRGS = 200


@dataclass(frozen=True)
class ServiceAccountingDecisionDiagnostics:
    selected_ue: torch.Tensor
    cell_index: torch.Tensor
    local_prg_index: torch.Tensor
    selected_nonblank: torch.Tensor
    raw_service_bytes: torch.Tensor
    effective_service_bytes: torch.Tensor
    initial_demand_bytes: torch.Tensor
    remaining_demand_before_bytes: torch.Tensor
    quality_db: torch.Tensor


@dataclass(frozen=True)
class ServiceAccountingUeTtiRow:
    tti: int
    ue: int
    cell: int
    planned_prg_ids: tuple[int, ...]
    accepted_prg_ids: tuple[int, ...]
    actual_prg_ids: tuple[int, ...]
    raw_predicted_service_bytes: float
    scheduled_bytes: float
    successful_bytes: float
    initial_backlog_bytes: float
    minimum_predecision_margin_bytes: float
    quality_db_mean: float
    eligible: bool
    ineligible_reason: str | None


def _detached_vector(value: torch.Tensor) -> torch.Tensor:
    return value.detach().reshape(-1).cpu()


def build_service_accounting_decision_diagnostics(
    *,
    packed: PackedPrgCandidates,
    token_indices: torch.Tensor,
    selected: torch.Tensor,
    raw_service_bytes: torch.Tensor,
    effective_service_bytes: torch.Tensor,
    initial_demand_bytes: torch.Tensor,
    remaining_demand_before_bytes: torch.Tensor,
) -> ServiceAccountingDecisionDiagnostics:
    """Capture selected-class service state before the sequential update."""
    tokens = torch.as_tensor(token_indices, device=packed.ue_ids.device, dtype=torch.long).reshape(-1)
    selected = torch.as_tensor(selected, device=tokens.device, dtype=torch.long).reshape(-1)
    count = int(tokens.numel())
    _require(selected.shape == (count,), "selected shape mismatch")
    _require(bool(((tokens >= 0) & (tokens < packed.num_prg_tokens)).all().item()), "token index out of range")
    _require(raw_service_bytes.shape == packed.ue_ids.shape, "raw service shape mismatch")
    _require(effective_service_bytes.shape == packed.ue_ids.shape, "effective service shape mismatch")
    _require(packed.edge_attr is not None and packed.edge_attr.shape[:2] == packed.ue_ids.shape, "packed edge attributes are required")
    _require(packed.edge_attr.dim() == 3 and packed.edge_attr.shape[2] > 0, "packed quality feature is missing")

    selected_nonblank = selected < packed.blank_class_index
    selected_ue = torch.full((count,), -1, dtype=torch.long, device=tokens.device)
    zeros = torch.zeros((count,), dtype=raw_service_bytes.dtype, device=tokens.device)
    if packed.max_candidates > 0:
        selected_class = selected.clamp(max=packed.max_candidates - 1)
        candidates = packed.ue_ids.to(device=tokens.device).index_select(0, tokens)
        gathered_ue = candidates.gather(1, selected_class.unsqueeze(-1)).squeeze(-1)
        selected_ue[selected_nonblank] = gathered_ue[selected_nonblank]
        safe_ue = gathered_ue.clamp(min=0, max=max(0, initial_demand_bytes.numel() - 1))

        def gather_candidate(matrix: torch.Tensor) -> torch.Tensor:
            return matrix.to(device=tokens.device).index_select(0, tokens).gather(
                1, selected_class.unsqueeze(-1)
            ).squeeze(-1)

        gathered_raw = gather_candidate(raw_service_bytes)
        gathered_effective = gather_candidate(effective_service_bytes)
        edge = packed.edge_attr.to(device=tokens.device).index_select(0, tokens).gather(
            1,
            selected_class[:, None, None].expand(-1, 1, packed.edge_attr.shape[2]),
        ).squeeze(1)
        gathered_initial = initial_demand_bytes.to(device=tokens.device).gather(0, safe_ue)
        gathered_remaining = remaining_demand_before_bytes.to(device=tokens.device).gather(0, safe_ue)
        raw = torch.where(selected_nonblank, gathered_raw, zeros)
        effective = torch.where(selected_nonblank, gathered_effective, zeros)
        initial = torch.where(selected_nonblank, gathered_initial, zeros)
        remaining = torch.where(selected_nonblank, gathered_remaining, zeros)
        quality = torch.where(selected_nonblank, edge[:, 0], zeros)
    else:
        _require(not bool(selected_nonblank.any().item()), "nonblank selection requires candidates")
        raw = effective = initial = remaining = quality = zeros
    values = {
        "selected_ue": selected_ue,
        "cell_index": torch.div(tokens, packed.num_prg_tokens // count, rounding_mode="floor"),
        "local_prg_index": torch.remainder(tokens, packed.num_prg_tokens // count),
        "selected_nonblank": selected_nonblank,
        "raw_service_bytes": raw,
        "effective_service_bytes": effective,
        "initial_demand_bytes": initial,
        "remaining_demand_before_bytes": remaining,
        "quality_db": quality,
    }
    return ServiceAccountingDecisionDiagnostics(
        **{name: _detached_vector(value) for name, value in values.items()}
    )


def concatenate_service_accounting_decision_diagnostics(
    rows: Sequence[ServiceAccountingDecisionDiagnostics],
) -> ServiceAccountingDecisionDiagnostics:
    _require(bool(rows), "at least one service accounting diagnostic row is required")
    values = {
        field.name: _detached_vector(
            torch.cat([getattr(row, field.name) for row in rows])
        )
        for field in fields(ServiceAccountingDecisionDiagnostics)
    }
    _require(len({value.numel() for value in values.values()}) == 1, "service diagnostic lengths differ")
    return ServiceAccountingDecisionDiagnostics(**values)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _vector(value: torch.Tensor, *, name: str, expected: int) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().reshape(-1).cpu()
    _require(tensor.shape == (expected,), f"{name} shape mismatch")
    if tensor.dtype is not torch.bool:
        _require(bool(torch.isfinite(tensor.to(dtype=torch.float64)).all().item()), f"{name} must be finite")
    return tensor


def _mask(info: Mapping[str, object], name: str, *, n_ue: int, n_prg: int) -> np.ndarray:
    _require(name in info, f"missing required {name}")
    value = np.asarray(info[name])
    _require(value.shape == (n_ue, n_prg), f"{name} shape mismatch")
    _require(np.issubdtype(value.dtype, np.number) or value.dtype == np.bool_, f"{name} must be numeric")
    _require(np.isfinite(value).all(), f"{name} must be finite")
    _require(np.logical_or(value == 0, value == 1).all(), f"{name} must be binary")
    return value.astype(np.bool_, copy=False)


def _bytes(info: Mapping[str, object], name: str, *, n_ue: int) -> np.ndarray:
    _require(name in info, f"missing required {name}")
    value = np.asarray(info[name], dtype=np.float64)
    _require(value.shape == (n_ue,), f"{name} shape mismatch")
    _require(np.isfinite(value).all(), f"{name} must be finite")
    _require((value >= 0.0).all(), f"{name} must be non-negative")
    return value


def join_service_accounting_tti(
    decisions: ServiceAccountingDecisionDiagnostics,
    *,
    info: Mapping[str, object],
    n_ue: int,
    n_cell: int,
    n_prg: int,
    tti: int,
) -> list[ServiceAccountingUeTtiRow]:
    """Join planned PRG IDs to exact pre-PDSCH and realized simulator PRG IDs."""
    _require(n_ue >= 0 and n_cell > 0 and n_prg > 0, "invalid service audit dimensions")
    _require("harq_enabled" in info, "service audit requires runtime HARQ state")
    _require(int(info["harq_enabled"]) == 0, "service audit requires a HARQ-disabled new-transmission queue")
    expected = n_cell * n_prg
    values = {
        field.name: _vector(getattr(decisions, field.name), name=field.name, expected=expected)
        for field in fields(decisions)
    }
    selected = values["selected_nonblank"].to(dtype=torch.bool)
    selected_ue = values["selected_ue"].to(dtype=torch.long)
    cells = values["cell_index"].to(dtype=torch.long)
    local_prgs = values["local_prg_index"].to(dtype=torch.long)
    _require(bool(((cells >= 0) & (cells < n_cell)).all().item()), "cell index out of range")
    _require(bool(((local_prgs >= 0) & (local_prgs < n_prg)).all().item()), "local PRG index out of range")
    tokens = cells * n_prg + local_prgs
    _require(torch.equal(torch.sort(tokens).values, torch.arange(expected)), "decision tokens do not exactly cover the TTI")
    _require(bool(((selected_ue[selected] >= 0) & (selected_ue[selected] < n_ue)).all().item()), "selected UE out of range")
    _require(bool((selected_ue[~selected] == -1).all().item()), "Blank decision must use UE -1")

    raw = values["raw_service_bytes"].to(dtype=torch.float64)
    effective = values["effective_service_bytes"].to(dtype=torch.float64)
    initial = values["initial_demand_bytes"].to(dtype=torch.float64)
    remaining = values["remaining_demand_before_bytes"].to(dtype=torch.float64)
    quality = values["quality_db"].to(dtype=torch.float64)
    for name, tensor in (
        ("raw service", raw),
        ("effective service", effective),
        ("initial demand", initial),
        ("remaining demand", remaining),
    ):
        _require(bool((tensor[selected] >= 0.0).all().item()), f"{name} must be non-negative")
    accepted = _mask(info, "accepted_pre_pdsch_ue_prg_mask", n_ue=n_ue, n_prg=n_prg)
    actual = _mask(info, "actual_ue_prg_mask", n_ue=n_ue, n_prg=n_prg)
    scheduled = _bytes(info, "actual_ue_served_bytes", n_ue=n_ue)
    successful = _bytes(info, "actual_ue_goodput_bytes", n_ue=n_ue)
    _require(bool((successful <= scheduled).all()), "successful bytes exceed scheduled bytes")

    planned_ues = set(int(value) for value in selected_ue[selected].tolist())
    for ue in range(n_ue):
        if ue not in planned_ues:
            _require(not bool(accepted[ue].any() or actual[ue].any() or scheduled[ue] or successful[ue]), "unplanned UE has simulator evidence")

    rows: list[ServiceAccountingUeTtiRow] = []
    for ue in sorted(planned_ues):
        ue_mask = selected & (selected_ue == ue)
        ue_cells = torch.unique(cells[ue_mask])
        _require(ue_cells.numel() == 1, "selected UE appears in multiple cells")
        planned_ids = tuple(sorted(int(value) for value in local_prgs[ue_mask].tolist()))
        _require(len(planned_ids) == len(set(planned_ids)), "selected UE has duplicate local PRG")
        accepted_ids = tuple(int(value) for value in np.flatnonzero(accepted[ue]).tolist())
        actual_ids = tuple(int(value) for value in np.flatnonzero(actual[ue]).tolist())
        raw_values = raw[ue_mask]
        initial_values = initial[ue_mask]
        remaining_values = remaining[ue_mask]
        _require(bool(torch.equal(initial_values, initial_values[:1].expand_as(initial_values))), "initial demand changed within a TTI")
        raw_sum = float(raw_values.sum().item())
        initial_backlog = float(initial_values[0].item())
        minimum_margin = float((remaining_values - raw_values).min().item())
        reason: str | None = None
        if accepted_ids != planned_ids:
            reason = "accepted_prg_set_mismatch"
        elif actual_ids != planned_ids:
            reason = "actual_prg_set_mismatch"
        elif initial_backlog < raw_sum:
            reason = "initial_demand_cap"
        elif minimum_margin < 0.0:
            reason = "predecision_demand_cap"
        rows.append(
            ServiceAccountingUeTtiRow(
                tti=int(tti),
                ue=ue,
                cell=int(ue_cells.item()),
                planned_prg_ids=planned_ids,
                accepted_prg_ids=accepted_ids,
                actual_prg_ids=actual_ids,
                raw_predicted_service_bytes=raw_sum,
                scheduled_bytes=float(scheduled[ue]),
                successful_bytes=float(successful[ue]),
                initial_backlog_bytes=initial_backlog,
                minimum_predecision_margin_bytes=minimum_margin,
                quality_db_mean=float(quality[ue_mask].mean().item()),
                eligible=reason is None,
                ineligible_reason=reason,
            )
        )
    return rows


def _summarize_group(
    rows: Sequence[ServiceAccountingUeTtiRow],
    *,
    global_ratio: float,
) -> dict[str, float | int | bool | None]:
    predicted = float(sum(row.raw_predicted_service_bytes for row in rows))
    scheduled = float(sum(row.scheduled_bytes for row in rows))
    actual_prgs = int(sum(len(row.actual_prg_ids) for row in rows))
    ratio = None if predicted == 0.0 else scheduled / predicted
    is_gate = len(rows) >= _MIN_GROUP_UE_TTIS and actual_prgs >= _MIN_GROUP_ACTUAL_PRGS
    return {
        "sample_count": len(rows),
        "actual_prg_count": actual_prgs,
        "raw_predicted_service_bytes": predicted,
        "scheduled_bytes": scheduled,
        "scheduled_to_raw_predicted_ratio": ratio,
        "is_gate_group": is_gate,
        "gate_passed": None if not is_gate or ratio is None else abs(ratio - global_ratio) <= _GROUP_TOLERANCE,
    }


def summarize_service_accounting_audit(
    rows: Sequence[ServiceAccountingUeTtiRow],
    *,
    scheduled_bytes_semantics: str,
    per_prg_scheduled_bytes_available: bool,
) -> dict[str, object]:
    """Apply the prospectively frozen global, cell, and equal-count quality gates."""
    _require(
        scheduled_bytes_semantics == NEW_TRANSMISSION_QUEUE_DEQUEUE,
        "scheduled bytes do not have new-transmission queue-dequeue semantics",
    )
    eligible = [row for row in rows if row.eligible]
    predicted = float(sum(row.raw_predicted_service_bytes for row in eligible))
    scheduled = float(sum(row.scheduled_bytes for row in eligible))
    global_ratio = None if predicted == 0.0 else scheduled / predicted
    global_passed = global_ratio is not None and abs(global_ratio - _TARGET_RATIO) <= _GLOBAL_TOLERANCE

    cell_groups: list[dict[str, object]] = []
    for cell in sorted({row.cell for row in eligible}):
        summary = _summarize_group(
            [row for row in eligible if row.cell == cell], global_ratio=float(global_ratio or 0.0)
        )
        cell_groups.append({"cell": cell, **summary})

    ordered = sorted(eligible, key=lambda row: (row.quality_db_mean, row.tti, row.ue))
    quality_groups: list[dict[str, object]] = []
    for index, indices in enumerate(np.array_split(np.arange(len(ordered)), 3)):
        group_rows = [ordered[int(position)] for position in indices.tolist()]
        summary = _summarize_group(group_rows, global_ratio=float(global_ratio or 0.0))
        quality_groups.append(
            {
                "quality_group": ("low", "middle", "high")[index],
                "quality_min_db": None if not group_rows else min(row.quality_db_mean for row in group_rows),
                "quality_max_db": None if not group_rows else max(row.quality_db_mean for row in group_rows),
                **summary,
            }
        )

    formal_groups = [
        group
        for group in (*cell_groups, *quality_groups)
        if bool(group["is_gate_group"])
    ]
    group_passed = all(group["gate_passed"] is True for group in formal_groups)
    ineligible_reasons: dict[str, int] = {}
    for row in rows:
        if row.eligible:
            continue
        key = str(row.ineligible_reason)
        ineligible_reasons[key] = ineligible_reasons.get(key, 0) + 1
    return {
        "schema_version": 1,
        "protocol_id": "seed42-service-accounting-read-only-audit-v1",
        "status": "PASS" if global_passed and group_passed else "FAIL",
        "scheduled_bytes_semantics": scheduled_bytes_semantics,
        "eligible_ue_tti_count": len(eligible),
        "eligible_actual_prg_count": sum(len(row.actual_prg_ids) for row in eligible),
        "ineligible_ue_tti_count": len(rows) - len(eligible),
        "ineligible_reason_counts": ineligible_reasons,
        "global": {
            "target_ratio": _TARGET_RATIO,
            "absolute_tolerance": _GLOBAL_TOLERANCE,
            "raw_predicted_service_bytes": predicted,
            "scheduled_bytes": scheduled,
            "successful_bytes": float(sum(row.successful_bytes for row in eligible)),
            "scheduled_to_raw_predicted_ratio": global_ratio,
            "gate_passed": global_passed,
        },
        "group_gate": {
            "minimum_ue_tti_count": _MIN_GROUP_UE_TTIS,
            "minimum_actual_prg_count": _MIN_GROUP_ACTUAL_PRGS,
            "maximum_absolute_delta_from_global": _GROUP_TOLERANCE,
            "formal_group_count": len(formal_groups),
            "all_formal_groups_passed": group_passed,
        },
        "cell_groups": cell_groups,
        "quality_binning": "decision_pre_post_eq_quality_equal_sample_count_tertiles",
        "quality_groups": quality_groups,
        "per_prg_scheduled_bytes_available": bool(per_prg_scheduled_bytes_available),
        "prg_segment_service_ratio_gate_enabled": bool(per_prg_scheduled_bytes_available),
        "prg_segment_note": (
            "No UE-TTI scheduled bytes were apportioned to PRGs; PRG segments are not a service-ratio gate."
            if not per_prg_scheduled_bytes_available
            else "True per-PRG scheduled bytes are available."
        ),
    }

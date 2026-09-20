"""Frozen blank-only CTBS non-starvation selector and detached diagnostics."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Mapping, Sequence

import torch

from .ctbs_headroom import CtbsHeadroomState


LOG1P_DRAIN_PRESSURE_INDEX = 12
SCHEDULED_DEQUEUE_ADEQUACY_INDEX = 13
TTL_NORMALIZED_SERVICE_AGE_INDEX = 14
MINIMUM_LOG1P_DRAIN_PRESSURE = math.log(2.0)
MAXIMUM_SCHEDULED_DEQUEUE_ADEQUACY = 1.0
MINIMUM_TTL_NORMALIZED_SERVICE_AGE = 0.25
MAXIMUM_OVERRIDES_PER_CELL_PER_TTI = 1

_FEATURE_NAMES = (
    "log1p_drain_pressure",
    "scheduled_dequeue_adequacy",
    "ttl_normalized_service_age",
)


@dataclass(frozen=True)
class CtbsNonStarvationGuardConfig:
    enabled: bool = False
    protocol_id: str = field(default="ctbs-blank-v1", init=False)


class GuardReason(IntEnum):
    ACTOR_NONBLANK = 0
    CELL_BUDGET_USED = 1
    NO_PRESSURE_CANDIDATE = 2
    NO_POSITIVE_CTBS_HEADROOM = 3
    OVERRIDDEN = 4


@dataclass(frozen=True)
class GuardStepDiagnostics:
    cell_index: torch.Tensor
    local_prg_index: torch.Tensor
    blank_class_index: torch.Tensor
    actor_choice: torch.Tensor
    executed_choice: torch.Tensor
    guard_override: torch.Tensor
    selected_class: torch.Tensor
    selected_ue: torch.Tensor
    candidate_was_valid: torch.Tensor
    log1p_drain_pressure: torch.Tensor
    scheduled_dequeue_adequacy: torch.Tensor
    ttl_normalized_service_age: torch.Tensor
    ctbs_capacity_before_bytes: torch.Tensor
    ctbs_capacity_after_bytes: torch.Tensor
    reason_code: torch.Tensor


@dataclass(frozen=True)
class GuardStepResult:
    executed_choice: torch.Tensor
    override_used_by_cell: torch.Tensor
    diagnostics: GuardStepDiagnostics


def _diagnostics_for_disabled(
    *, actor_choice: torch.Tensor, blank_class_index: int, local_prg: int
) -> GuardStepDiagnostics:
    n_cell = int(actor_choice.numel())
    device = actor_choice.device
    blank = actor_choice == int(blank_class_index)
    reason = torch.where(
        blank,
        torch.full((n_cell,), int(GuardReason.NO_PRESSURE_CANDIDATE), device=device),
        torch.full((n_cell,), int(GuardReason.ACTOR_NONBLANK), device=device),
    ).to(dtype=torch.long)
    zeros_float = torch.zeros((n_cell,), dtype=torch.float32, device=device)
    zeros_int = torch.zeros((n_cell,), dtype=torch.int64, device=device)
    return GuardStepDiagnostics(
        cell_index=torch.arange(n_cell, dtype=torch.long, device=device),
        local_prg_index=torch.full((n_cell,), int(local_prg), dtype=torch.long, device=device),
        blank_class_index=torch.full((n_cell,), int(blank_class_index), dtype=torch.long, device=device),
        actor_choice=actor_choice.clone(),
        executed_choice=actor_choice.clone(),
        guard_override=torch.zeros((n_cell,), dtype=torch.bool, device=device),
        selected_class=torch.full((n_cell,), -1, dtype=torch.long, device=device),
        selected_ue=torch.full((n_cell,), -1, dtype=torch.long, device=device),
        candidate_was_valid=torch.zeros((n_cell,), dtype=torch.bool, device=device),
        log1p_drain_pressure=zeros_float.clone(),
        scheduled_dequeue_adequacy=zeros_float.clone(),
        ttl_normalized_service_age=zeros_float.clone(),
        ctbs_capacity_before_bytes=zeros_int.clone(),
        ctbs_capacity_after_bytes=zeros_int.clone(),
        reason_code=reason,
    )


def _validate_enabled_inputs(
    *,
    blank_class_index: int,
    local_prg: int,
    actor_choice: torch.Tensor,
    candidate_ue_ids: torch.Tensor,
    valid_main: torch.Tensor,
    remaining_demand_bytes: torch.Tensor,
    ue_features: torch.Tensor,
    ue_feature_names: Sequence[str],
    ctbs_states: Sequence[CtbsHeadroomState] | None,
    prg_layer_cap: Sequence[Sequence[int]] | None,
    override_used_by_cell: torch.Tensor,
) -> None:
    if int(blank_class_index) < 0 or int(local_prg) < 0:
        raise ValueError("blank class index and local PRG must be non-negative")
    if actor_choice.ndim != 1:
        raise ValueError("actor choices must be rank one")
    if candidate_ue_ids.ndim != 2 or valid_main.ndim != 2:
        raise ValueError("candidate IDs and valid-main mask must be rank two")
    if candidate_ue_ids.shape != valid_main.shape:
        raise ValueError("candidate IDs and valid-main mask dimensions disagree")
    if candidate_ue_ids.shape[0] != actor_choice.numel():
        raise ValueError("candidate rows and actor choices dimensions disagree")
    if remaining_demand_bytes.ndim != 1 or ue_features.ndim != 2:
        raise ValueError("remaining demand must be rank one and features rank two")
    if remaining_demand_bytes.numel() != ue_features.shape[0]:
        raise ValueError("remaining demand and feature UE dimensions disagree")
    if override_used_by_cell.ndim != 1 or override_used_by_cell.numel() != actor_choice.numel():
        raise ValueError("per-cell override budget dimensions disagree")
    tensors = (
        candidate_ue_ids,
        valid_main,
        remaining_demand_bytes,
        ue_features,
        override_used_by_cell,
    )
    if any(tensor.device != actor_choice.device for tensor in tensors):
        raise ValueError("guard tensors must use one device")
    if valid_main.dtype != torch.bool or override_used_by_cell.dtype != torch.bool:
        raise ValueError("valid-main and override budget must be boolean")
    if ue_features.shape[1] <= TTL_NORMALIZED_SERVICE_AGE_INDEX:
        raise ValueError("guard requires at least 15 UE features")
    if tuple(ue_feature_names[LOG1P_DRAIN_PRESSURE_INDEX:TTL_NORMALIZED_SERVICE_AGE_INDEX + 1]) != _FEATURE_NAMES:
        raise ValueError("guard feature names at slots 12--14 disagree")
    if ctbs_states is None or prg_layer_cap is None:
        raise ValueError("enabled guard requires CTBS states and layer caps")
    n_ue = int(ue_features.shape[0])
    if len(ctbs_states) != n_ue or len(prg_layer_cap) != n_ue:
        raise ValueError("CTBS state and layer-cap UE dimensions disagree")
    if any(int(ue) < 0 or int(ue) >= n_ue for ue in candidate_ue_ids[valid_main].tolist()):
        raise ValueError("valid candidate UE is out of range")
    for state, caps in zip(ctbs_states, prg_layer_cap):
        if not isinstance(state, CtbsHeadroomState):
            raise ValueError("guard requires CtbsHeadroomState values")
        if int(local_prg) >= len(caps):
            raise ValueError("layer caps do not cover the local PRG")


def apply_ctbs_non_starvation_guard_step(
    *,
    config: CtbsNonStarvationGuardConfig,
    blank_class_index: int,
    local_prg: int,
    actor_choice: torch.Tensor,
    candidate_ue_ids: torch.Tensor,
    valid_main: torch.Tensor,
    remaining_demand_bytes: torch.Tensor,
    ue_features: torch.Tensor,
    ue_feature_names: Sequence[str],
    ctbs_states: Sequence[CtbsHeadroomState] | None,
    prg_layer_cap: Sequence[Sequence[int]] | None,
    override_used_by_cell: torch.Tensor,
) -> GuardStepResult:
    """Apply at most one legal blank replacement per cell for one local PRG."""
    if not config.enabled:
        return GuardStepResult(
            executed_choice=actor_choice.clone(),
            override_used_by_cell=override_used_by_cell.clone(),
            diagnostics=_diagnostics_for_disabled(
                actor_choice=actor_choice,
                blank_class_index=blank_class_index,
                local_prg=local_prg,
            ),
        )

    _validate_enabled_inputs(
        blank_class_index=blank_class_index,
        local_prg=local_prg,
        actor_choice=actor_choice,
        candidate_ue_ids=candidate_ue_ids,
        valid_main=valid_main,
        remaining_demand_bytes=remaining_demand_bytes,
        ue_features=ue_features,
        ue_feature_names=ue_feature_names,
        ctbs_states=ctbs_states,
        prg_layer_cap=prg_layer_cap,
        override_used_by_cell=override_used_by_cell,
    )
    assert ctbs_states is not None
    assert prg_layer_cap is not None

    n_cell = int(actor_choice.numel())
    device = actor_choice.device
    executed = actor_choice.clone()
    used = override_used_by_cell.clone()
    selected_class = torch.full((n_cell,), -1, dtype=torch.long, device=device)
    selected_ue = torch.full((n_cell,), -1, dtype=torch.long, device=device)
    guard_override = torch.zeros((n_cell,), dtype=torch.bool, device=device)
    candidate_was_valid = torch.zeros((n_cell,), dtype=torch.bool, device=device)
    pressure_out = torch.zeros((n_cell,), dtype=ue_features.dtype, device=device)
    adequacy_out = torch.zeros((n_cell,), dtype=ue_features.dtype, device=device)
    age_out = torch.zeros((n_cell,), dtype=ue_features.dtype, device=device)
    before_out = torch.zeros((n_cell,), dtype=torch.int64, device=device)
    after_out = torch.zeros((n_cell,), dtype=torch.int64, device=device)
    reason = torch.empty((n_cell,), dtype=torch.long, device=device)

    for cell in range(n_cell):
        if int(actor_choice[cell]) != int(blank_class_index):
            reason[cell] = int(GuardReason.ACTOR_NONBLANK)
            continue
        if bool(used[cell]):
            reason[cell] = int(GuardReason.CELL_BUDGET_USED)
            continue

        pressure_candidates: list[tuple[int, int, float, float, float]] = []
        for class_index in range(int(candidate_ue_ids.shape[1])):
            if not bool(valid_main[cell, class_index]):
                continue
            ue = int(candidate_ue_ids[cell, class_index])
            pressure_tensor = ue_features[ue, LOG1P_DRAIN_PRESSURE_INDEX]
            adequacy_tensor = ue_features[ue, SCHEDULED_DEQUEUE_ADEQUACY_INDEX]
            age_tensor = ue_features[ue, TTL_NORMALIZED_SERVICE_AGE_INDEX]
            if (
                float(remaining_demand_bytes[ue]) > 0.0
                and bool(pressure_tensor > MINIMUM_LOG1P_DRAIN_PRESSURE)
                and bool(adequacy_tensor < MAXIMUM_SCHEDULED_DEQUEUE_ADEQUACY)
                and bool(age_tensor >= MINIMUM_TTL_NORMALIZED_SERVICE_AGE)
            ):
                pressure_candidates.append(
                    (
                        class_index,
                        ue,
                        float(pressure_tensor),
                        float(adequacy_tensor),
                        float(age_tensor),
                    )
                )
        if not pressure_candidates:
            reason[cell] = int(GuardReason.NO_PRESSURE_CANDIDATE)
            continue

        ctbs_candidates: list[tuple[tuple[float, float, int], int, int, float, float, float, int, int]] = []
        for class_index, ue, pressure, adequacy, age in pressure_candidates:
            before = int(ctbs_states[ue].cumulative_tbs_bytes)
            after = int(
                ctbs_states[ue]
                .with_selected_prg(int(local_prg), layer_cap=int(prg_layer_cap[ue][int(local_prg)]))
                .cumulative_tbs_bytes
            )
            if after > before:
                ctbs_candidates.append(
                    ((age, pressure, -ue), class_index, ue, pressure, adequacy, age, before, after)
                )
        if not ctbs_candidates:
            reason[cell] = int(GuardReason.NO_POSITIVE_CTBS_HEADROOM)
            continue

        _, class_index, ue, pressure, adequacy, age, before, after = max(ctbs_candidates)
        executed[cell] = class_index
        used[cell] = True
        selected_class[cell] = class_index
        selected_ue[cell] = ue
        guard_override[cell] = True
        candidate_was_valid[cell] = True
        pressure_out[cell] = pressure
        adequacy_out[cell] = adequacy
        age_out[cell] = age
        before_out[cell] = before
        after_out[cell] = after
        reason[cell] = int(GuardReason.OVERRIDDEN)

    return GuardStepResult(
        executed_choice=executed,
        override_used_by_cell=used,
        diagnostics=GuardStepDiagnostics(
            cell_index=torch.arange(n_cell, dtype=torch.long, device=device),
            local_prg_index=torch.full((n_cell,), int(local_prg), dtype=torch.long, device=device),
            blank_class_index=torch.full((n_cell,), int(blank_class_index), dtype=torch.long, device=device),
            actor_choice=actor_choice.clone(),
            executed_choice=executed.clone(),
            guard_override=guard_override,
            selected_class=selected_class,
            selected_ue=selected_ue,
            candidate_was_valid=candidate_was_valid,
            log1p_drain_pressure=pressure_out,
            scheduled_dequeue_adequacy=adequacy_out,
            ttl_normalized_service_age=age_out,
            ctbs_capacity_before_bytes=before_out,
            ctbs_capacity_after_bytes=after_out,
            reason_code=reason,
        ),
    )


def concatenate_guard_diagnostics(rows: Sequence[GuardStepDiagnostics]) -> GuardStepDiagnostics:
    if not rows:
        raise ValueError("cannot concatenate empty guard diagnostics")
    return GuardStepDiagnostics(
        **{
            name: torch.cat([getattr(row, name) for row in rows], dim=0)
            for name in GuardStepDiagnostics.__dataclass_fields__
        }
    )


def guard_diagnostics_to_rows(*, tti: int, diagnostics: GuardStepDiagnostics) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(int(diagnostics.cell_index.numel())):
        reason = GuardReason(int(diagnostics.reason_code[index])).name.lower()
        rows.append(
            {
                "tti": int(tti),
                "cell_index": int(diagnostics.cell_index[index]),
                "local_prg_index": int(diagnostics.local_prg_index[index]),
                "blank_class_index": int(diagnostics.blank_class_index[index]),
                "actor_choice": int(diagnostics.actor_choice[index]),
                "executed_choice": int(diagnostics.executed_choice[index]),
                "guard_override": bool(diagnostics.guard_override[index]),
                "selected_class": int(diagnostics.selected_class[index]),
                "selected_ue": int(diagnostics.selected_ue[index]),
                "candidate_was_valid": bool(diagnostics.candidate_was_valid[index]),
                "log1p_drain_pressure": float(diagnostics.log1p_drain_pressure[index]),
                "scheduled_dequeue_adequacy": float(diagnostics.scheduled_dequeue_adequacy[index]),
                "ttl_normalized_service_age": float(diagnostics.ttl_normalized_service_age[index]),
                "ctbs_capacity_before_bytes": int(diagnostics.ctbs_capacity_before_bytes[index]),
                "ctbs_capacity_after_bytes": int(diagnostics.ctbs_capacity_after_bytes[index]),
                "reason": reason,
            }
        )
    return rows


def summarize_guard_diagnostics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    overrides = [row for row in rows if bool(row["guard_override"])]
    reasons = Counter(str(row["reason"]) for row in rows)
    per_cell = Counter(int(row["cell_index"]) for row in overrides)
    per_ue = Counter(int(row["selected_ue"]) for row in overrides)
    per_cell_tti = Counter((int(row["tti"]), int(row["cell_index"])) for row in overrides)
    return {
        "total_override_count": len(overrides),
        "nonblank_override_count": sum(
            int(row["actor_choice"]) != int(row["blank_class_index"])
            for row in overrides
        ),
        "override_count_by_cell": dict(sorted(per_cell.items())),
        "override_count_by_ue": dict(sorted(per_ue.items())),
        "actor_nonblank_count": reasons[GuardReason.ACTOR_NONBLANK.name.lower()],
        "cell_budget_used_count": reasons[GuardReason.CELL_BUDGET_USED.name.lower()],
        "no_pressure_candidate_count": reasons[GuardReason.NO_PRESSURE_CANDIDATE.name.lower()],
        "no_positive_ctbs_headroom_count": reasons[GuardReason.NO_POSITIVE_CTBS_HEADROOM.name.lower()],
        "maximum_overrides_per_cell_per_tti": max(per_cell_tti.values(), default=0),
    }


__all__ = [
    "LOG1P_DRAIN_PRESSURE_INDEX",
    "SCHEDULED_DEQUEUE_ADEQUACY_INDEX",
    "TTL_NORMALIZED_SERVICE_AGE_INDEX",
    "MINIMUM_LOG1P_DRAIN_PRESSURE",
    "MAXIMUM_SCHEDULED_DEQUEUE_ADEQUACY",
    "MINIMUM_TTL_NORMALIZED_SERVICE_AGE",
    "MAXIMUM_OVERRIDES_PER_CELL_PER_TTI",
    "CtbsNonStarvationGuardConfig",
    "GuardReason",
    "GuardStepDiagnostics",
    "GuardStepResult",
    "apply_ctbs_non_starvation_guard_step",
    "concatenate_guard_diagnostics",
    "guard_diagnostics_to_rows",
    "summarize_guard_diagnostics",
]

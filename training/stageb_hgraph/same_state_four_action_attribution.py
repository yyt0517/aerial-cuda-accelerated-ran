#!/usr/bin/env python3
"""Minimal four-action attribution helpers for same-state PFQ/DPA spikes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


_ACTION_KEYS = (
    "pfq",
    "ue_reallocation_only",
    "topology_only",
    "dpa",
)


def deadline_state_by_ue(
    ue_features: Sequence[Sequence[float]],
    feature_names: Sequence[str],
) -> dict[str, list[float]]:
    """Extract the frozen deadline-service state needed by the TTI visual."""
    features = np.asarray(ue_features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(feature_names):
        raise ValueError("UE feature names and dimensions disagree")
    source_to_report = {
        "buffer_bytes": "backlog_bytes_by_ue",
        "ttl_slack_ms": "ttl_slack_ms_by_ue",
        "log1p_drain_pressure": "log1p_drain_pressure_by_ue",
        "scheduled_dequeue_adequacy": "scheduled_dequeue_adequacy_by_ue",
        "ttl_normalized_service_age": "ttl_normalized_service_age_by_ue",
    }
    missing = [name for name in source_to_report if name not in feature_names]
    if missing:
        raise ValueError(f"deadline state is missing UE features: {missing}")
    return {
        report_name: features[:, list(feature_names).index(source_name)].tolist()
        for source_name, report_name in source_to_report.items()
    }


def build_four_actions(
    pfq_chosen_ue_per_prg: Sequence[int],
    dpa_chosen_ue_per_prg: Sequence[int],
) -> dict[str, np.ndarray]:
    """Split DPA changes into disjoint both-active UE and topology patches."""
    pfq = np.asarray(pfq_chosen_ue_per_prg, dtype=np.int64).reshape(-1)
    dpa = np.asarray(dpa_chosen_ue_per_prg, dtype=np.int64).reshape(-1)
    if pfq.shape != dpa.shape:
        raise ValueError("PFQ and DPA action dimensions disagree")

    both_active = (pfq >= 0) & (dpa >= 0)
    topology_changed = (pfq >= 0) != (dpa >= 0)
    ue_only = pfq.copy()
    ue_only[both_active] = dpa[both_active]
    topology_only = pfq.copy()
    topology_only[topology_changed] = dpa[topology_changed]
    return {
        "pfq": pfq.copy(),
        "ue_reallocation_only": ue_only,
        "topology_only": topology_only,
        "dpa": dpa.copy(),
    }


def action_structure_metrics(
    chosen_ue_per_prg: Sequence[int],
    *,
    n_cell: int,
    n_prg: int,
) -> dict[str, int]:
    """Count resource use and same-PRG cross-cell coactivity."""
    chosen = np.asarray(chosen_ue_per_prg, dtype=np.int64).reshape(-1)
    if chosen.size != int(n_cell) * int(n_prg):
        raise ValueError("action structure dimensions disagree")
    active_per_prg = (chosen.reshape(int(n_cell), int(n_prg)) >= 0).sum(axis=0)
    return {
        "scheduled_cell_prg_count": int(active_per_prg.sum()),
        "coactive_cell_pair_count": int(
            np.sum(active_per_prg * (active_per_prg - 1) // 2)
        ),
        "full_reuse_prg_count": int(np.count_nonzero(active_per_prg == int(n_cell))),
        "two_cell_prg_count": int(np.count_nonzero(active_per_prg == 2)),
        "one_cell_prg_count": int(np.count_nonzero(active_per_prg == 1)),
        "all_blank_prg_count": int(np.count_nonzero(active_per_prg == 0)),
    }


def shapley_two_patch_contributions(values: Mapping[str, float]) -> dict[str, float]:
    """Return exact two-factor Shapley attribution for one scalar metric."""
    missing = [key for key in _ACTION_KEYS if key not in values]
    if missing:
        raise ValueError(f"four-action metric is missing keys: {missing}")
    y0 = float(values["pfq"])
    yu = float(values["ue_reallocation_only"])
    ym = float(values["topology_only"])
    yd = float(values["dpa"])
    ue_contribution = 0.5 * ((yu - y0) + (yd - ym))
    topology_contribution = 0.5 * ((ym - y0) + (yd - yu))
    total_delta = yd - y0
    return {
        "ue_reallocation_contribution": float(ue_contribution),
        "topology_contribution": float(topology_contribution),
        "total_delta": float(total_delta),
        "additivity_residual": float(
            total_delta - ue_contribution - topology_contribution
        ),
    }


def select_representative_tti(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select the resource-matched positive case nearest median mechanism gains."""
    qualified = [
        row
        for row in rows
        if abs(int(row["scheduled_cell_prg_delta_ai_minus_pfq"])) <= 1
        and float(row["dpa_minus_pfq_demand_capped_service_bytes"]) > 0.0
        and float(row["dpa_minus_pfq_phi_delta"]) > 0.0
    ]
    if not qualified:
        raise ValueError("no representative TTI satisfies the frozen selection rule")
    service = np.asarray(
        [float(row["dpa_minus_pfq_demand_capped_service_bytes"]) for row in qualified],
        dtype=np.float64,
    )
    pressure = np.asarray(
        [float(row["dpa_minus_pfq_phi_delta"]) for row in qualified],
        dtype=np.float64,
    )
    service_scale = max(float(np.ptp(service)), 1.0)
    pressure_scale = max(float(np.ptp(pressure)), 1.0e-12)
    service_mid = float(np.median(service))
    pressure_mid = float(np.median(pressure))
    ranked = sorted(
        qualified,
        key=lambda row: (
            (
                (float(row["dpa_minus_pfq_demand_capped_service_bytes"]) - service_mid)
                / service_scale
            )
            ** 2
            + (
                (float(row["dpa_minus_pfq_phi_delta"]) - pressure_mid)
                / pressure_scale
            )
            ** 2,
            int(row["tti"]),
        ),
    )
    return ranked[0]


def _upper_tail_cvar(values: np.ndarray, tail_fraction: float) -> float:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    count = max(1, int(np.ceil(float(tail_fraction) * finite.size)))
    return float(np.sort(finite)[-count:].mean())


def infer_recent_scheduled_dequeue(
    backlog_bytes: Sequence[float],
    ttl_slack_ms: Sequence[float],
    log1p_drain_pressure: Sequence[float],
    *,
    slot_duration_ms: float = 0.5,
) -> np.ndarray:
    """Invert the exported dimensionless drain-pressure feature."""
    if float(slot_duration_ms) <= 0.0:
        raise ValueError("slot_duration_ms must be positive")
    backlog = np.maximum(np.asarray(backlog_bytes, dtype=np.float64).reshape(-1), 0.0)
    ttl_ms = np.asarray(ttl_slack_ms, dtype=np.float64).reshape(-1)
    log_pressure = np.maximum(
        np.asarray(log1p_drain_pressure, dtype=np.float64).reshape(-1), 0.0
    )
    if not (backlog.shape == ttl_ms.shape == log_pressure.shape):
        raise ValueError("deadline feature dimensions disagree")
    pressure = np.expm1(log_pressure)
    slack_tti = np.maximum(ttl_ms / float(slot_duration_ms), 1.0)
    valid = (backlog > 0.0) & (ttl_ms >= 0.0) & (pressure > 0.0)
    recent = np.zeros_like(backlog)
    recent[valid] = np.maximum(
        (backlog[valid] / pressure[valid] - 1.0) / slack_tti[valid],
        0.0,
    )
    return recent


def counterfactual_deadline_pressure(
    backlog_bytes: Sequence[float],
    ttl_slack_ms: Sequence[float],
    recent_scheduled_dequeue_bytes: Sequence[float],
    scheduled_dequeue_bytes: Sequence[float],
    *,
    slot_duration_ms: float = 0.5,
    ewma_alpha: float = 0.05,
    tail_fraction: float = 0.10,
) -> dict[str, Any]:
    """Evaluate a no-arrival one-step pressure proxy using scheduled dequeue."""
    if float(slot_duration_ms) <= 0.0:
        raise ValueError("slot_duration_ms must be positive")
    if not 0.0 <= float(ewma_alpha) <= 1.0:
        raise ValueError("ewma_alpha must be in [0, 1]")
    if not 0.0 < float(tail_fraction) <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    backlog = np.maximum(np.asarray(backlog_bytes, dtype=np.float64).reshape(-1), 0.0)
    ttl_ms = np.asarray(ttl_slack_ms, dtype=np.float64).reshape(-1)
    recent = np.maximum(
        np.asarray(recent_scheduled_dequeue_bytes, dtype=np.float64).reshape(-1), 0.0
    )
    scheduled = np.maximum(
        np.asarray(scheduled_dequeue_bytes, dtype=np.float64).reshape(-1), 0.0
    )
    if not (backlog.shape == ttl_ms.shape == recent.shape == scheduled.shape):
        raise ValueError("counterfactual pressure dimensions disagree")

    active = (backlog > 0.0) & (ttl_ms >= 0.0)
    slack_before = np.maximum(ttl_ms / float(slot_duration_ms), 1.0)
    pressure_before = np.zeros_like(backlog)
    pressure_before[active] = backlog[active] / (
        recent[active] * slack_before[active] + 1.0
    )
    log_before = np.log1p(pressure_before)

    recent_after = (1.0 - float(ewma_alpha)) * recent + float(ewma_alpha) * scheduled
    backlog_after = np.maximum(backlog - scheduled, 0.0)
    ttl_after = np.maximum(ttl_ms / float(slot_duration_ms) - 1.0, 0.0)
    slack_after = np.maximum(ttl_after, 1.0)
    pressure_after = np.zeros_like(backlog)
    after_active = active & (backlog_after > 0.0)
    pressure_after[after_active] = backlog_after[after_active] / (
        recent_after[after_active] * slack_after[after_active] + 1.0
    )
    log_after = np.log1p(pressure_after)
    phi_before = _upper_tail_cvar(log_before[active], tail_fraction)
    phi_after = _upper_tail_cvar(log_after[active], tail_fraction)
    return {
        "active_ue_ids": np.flatnonzero(active).astype(int).tolist(),
        "recent_scheduled_dequeue_after": recent_after,
        "backlog_after": backlog_after,
        "ttl_slack_tti_after": ttl_after,
        "log_pressure_before": log_before,
        "log_pressure_after": log_after,
        "phi_before": float(phi_before),
        "phi_after": float(phi_after),
        "phi_delta": float(phi_before - phi_after),
    }

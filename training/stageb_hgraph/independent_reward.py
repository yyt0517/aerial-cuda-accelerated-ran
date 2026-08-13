from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

from .independent_objectives import ExpiryAwareUeWindowTracker, compute_delay_debt_potential
from .reward_utils import extract_reward_metrics
from .types import GraphFeatureSnapshot, SparseEntityGraph


@dataclass(frozen=True)
class IndependentRewardConfig:
    treatment: Literal["A", "B", "C", "D"]
    offered_bytes_per_tti: float = 108_000.0
    slot_duration_s: float = 0.0005
    gamma: float = 0.99
    ue_window_steps: int = 400
    delay_scale_p95: float = 0.0
    ue_utility_scale_p95: float = 0.0
    goodput_weight: float = 1.0
    failed_weight: float = 1.0
    expired_weight: float = 2.0
    delay_weight: float = 0.25
    ue_utility_weight: float = 0.25
    delay_topk: int = 8

    def __post_init__(self) -> None:
        if self.treatment not in ("A", "B", "C", "D"):
            raise ValueError(f"unknown independent treatment: {self.treatment}")
        if float(self.offered_bytes_per_tti) <= 0.0:
            raise ValueError("offered_bytes_per_tti must be positive")
        if self.treatment in ("C", "D") and float(self.delay_scale_p95) <= 0.0:
            raise ValueError("treatment C/D requires positive frozen delay_scale_p95")
        if self.treatment == "D" and float(self.ue_utility_scale_p95) <= 0.0:
            raise ValueError("treatment D requires positive frozen ue_utility_scale_p95")


def _snapshot(value: SparseEntityGraph | GraphFeatureSnapshot) -> GraphFeatureSnapshot:
    return value.snapshot if isinstance(value, SparseEntityGraph) else value


def _ue_array(info: Mapping[str, object], key: str, n_ue: int) -> np.ndarray:
    if key not in info or info[key] is None:
        raise ValueError(f"expiry-aware UE diagnostics missing {key}")
    try:
        value = np.asarray(info[key], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid expiry-aware UE diagnostic {key}") from exc
    if value.shape != (n_ue,):
        raise ValueError(
            f"expiry-aware UE diagnostic {key} shape mismatch: "
            f"got={value.shape} expected={(n_ue,)}"
        )
    return value


def _demand_mask(graph: SparseEntityGraph | GraphFeatureSnapshot) -> np.ndarray:
    return (
        _snapshot(graph).ue_features.values[:, 0].detach().cpu().numpy().reshape(-1) > 0.0
    )


def compute_independent_reward(
    before_graph: SparseEntityGraph | GraphFeatureSnapshot,
    after_graph: SparseEntityGraph | GraphFeatureSnapshot,
    info: Mapping[str, object],
    tracker: ExpiryAwareUeWindowTracker | None,
    cfg: IndependentRewardConfig,
) -> dict[str, float]:
    """Compute the locked G-F-2E reward and optional frozen-scale auxiliaries."""
    # Revalidate because tests and checkpoint loaders may construct configs without
    # going through a CLI preflight.
    cfg.__post_init__()
    offered = float(cfg.offered_bytes_per_tti)
    before_n_ue = int(_snapshot(before_graph).static_meta.n_ue)
    after_n_ue = int(_snapshot(after_graph).static_meta.n_ue)
    if before_n_ue != after_n_ue:
        raise ValueError(f"UE count changed across reward transition: {before_n_ue} -> {after_n_ue}")

    goodput_by_ue = _ue_array(info, "actual_ue_goodput_bytes", before_n_ue)
    scheduled_by_ue = _ue_array(info, "actual_ue_served_bytes", before_n_ue)
    goodput_bytes = float(np.maximum(goodput_by_ue, 0.0).sum())
    scheduled_bytes = float(np.maximum(scheduled_by_ue, 0.0).sum())
    failed_bytes = max(0.0, scheduled_bytes - goodput_bytes)
    expired_bytes = max(0.0, float(extract_reward_metrics(info)["expired_bytes"]))

    goodput_raw = goodput_bytes / offered
    failed_raw = failed_bytes / offered
    expired_raw = expired_bytes / offered
    goodput_weighted = float(cfg.goodput_weight) * goodput_raw
    failed_weighted = -float(cfg.failed_weight) * failed_raw
    expired_weighted = -float(cfg.expired_weight) * expired_raw
    base = goodput_weighted + failed_weighted + expired_weighted

    delay_enabled = cfg.treatment in ("C", "D")
    delay_before = 0.0
    delay_after = 0.0
    delay_delta = 0.0
    delay_scaled = 0.0
    delay_clipped = 0.0
    delay_weighted = 0.0
    if delay_enabled:
        delay_before = compute_delay_debt_potential(before_graph, topk=int(cfg.delay_topk))
        delay_after = compute_delay_debt_potential(after_graph, topk=int(cfg.delay_topk))
        delay_delta = delay_before - float(cfg.gamma) * delay_after
        delay_scaled = delay_delta / float(cfg.delay_scale_p95)
        delay_clipped = float(np.clip(delay_scaled, -1.0, 1.0))
        delay_weighted = float(cfg.delay_weight) * delay_clipped

    utility_enabled = cfg.treatment == "D"
    utility_before = 0.0
    utility_after = 0.0
    utility_delta = 0.0
    utility_scaled = 0.0
    utility_clipped = 0.0
    utility_weighted = 0.0
    if utility_enabled:
        if tracker is None or int(tracker.n_ue) != before_n_ue:
            raise ValueError("treatment D requires a matching expiry-aware UE window tracker")
        if not bool(info.get("actual_ue_packet_expiry_aware_diagnostics_available", False)):
            raise ValueError("treatment D requires expiry-aware UE diagnostics")
        before_demand = _demand_mask(before_graph)
        after_demand = _demand_mask(after_graph)
        utility_before = tracker.utility(before_demand)
        utility_after = tracker.update(
            after_demand,
            goodput_by_ue,
            _ue_array(info, "actual_ue_packet_system_time_ms", before_n_ue),
            _ue_array(info, "actual_ue_packet_expired_system_time_ms", before_n_ue),
        )
        utility_delta = float(cfg.gamma) * utility_after - utility_before
        utility_scaled = utility_delta / float(cfg.ue_utility_scale_p95)
        utility_clipped = float(np.clip(utility_scaled, -1.0, 1.0))
        utility_weighted = float(cfg.ue_utility_weight) * utility_clipped

    total = base + delay_weighted + utility_weighted
    return {
        "reward_total": float(total),
        "reward_base": float(base),
        "reward_goodput_bytes": goodput_bytes,
        "reward_scheduled_bytes": scheduled_bytes,
        "reward_failed_bytes": failed_bytes,
        "reward_expired_bytes": expired_bytes,
        "reward_goodput_raw": goodput_raw,
        "reward_failed_raw": failed_raw,
        "reward_expired_raw": expired_raw,
        "reward_goodput_weighted": goodput_weighted,
        "reward_failed_weighted": failed_weighted,
        "reward_expired_weighted": expired_weighted,
        "reward_delay_enabled": float(delay_enabled),
        "reward_delay_before_raw": delay_before,
        "reward_delay_after_raw": delay_after,
        "reward_delay_delta_raw": delay_delta,
        "reward_delay_delta_scaled": delay_scaled,
        "reward_delay_delta_clipped": delay_clipped,
        "reward_delay_weighted": delay_weighted,
        "reward_ue_utility_enabled": float(utility_enabled),
        "reward_ue_utility_before_raw": utility_before,
        "reward_ue_utility_after_raw": utility_after,
        "reward_ue_utility_delta_raw": utility_delta,
        "reward_ue_utility_delta_scaled": utility_scaled,
        "reward_ue_utility_delta_clipped": utility_clipped,
        "reward_ue_utility_weighted": utility_weighted,
        "reward_delay_scale_p95": float(cfg.delay_scale_p95),
        "reward_ue_utility_scale_p95": float(cfg.ue_utility_scale_p95),
    }


__all__ = ["IndependentRewardConfig", "compute_independent_reward"]

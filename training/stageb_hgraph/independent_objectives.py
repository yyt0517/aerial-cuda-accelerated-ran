from __future__ import annotations

from collections import deque
from typing import Mapping, Sequence

import numpy as np
import torch

from .reward_utils import compute_queue_pressure_potential
from .types import GraphFeatureSnapshot, SparseEntityGraph


def _snapshot(value: SparseEntityGraph | GraphFeatureSnapshot) -> GraphFeatureSnapshot:
    return value.snapshot if isinstance(value, SparseEntityGraph) else value


def compute_delay_debt_potential(
    graph: SparseEntityGraph | GraphFeatureSnapshot,
    *,
    topk: int = 8,
) -> float:
    snapshot = _snapshot(graph)
    demand_mask = snapshot.ue_features.values[:, 0] > 0.0
    value = compute_queue_pressure_potential(
        snapshot.ue_features.values,
        demand_mask=demand_mask,
        topk=topk,
    )
    return float(value.detach().cpu().item())


def compute_ue_expiry_aware_utility(
    demand_mask: np.ndarray,
    successful_goodput_bytes: np.ndarray,
    outcome_system_time_ms: np.ndarray,
    *,
    epsilon: float = 1.0e-6,
) -> float:
    active = np.asarray(demand_mask, dtype=bool).reshape(-1)
    goodput = np.asarray(successful_goodput_bytes, dtype=np.float64).reshape(-1)
    outcome_time = np.asarray(outcome_system_time_ms, dtype=np.float64).reshape(-1)
    if goodput.shape != active.shape or outcome_time.shape != active.shape:
        raise ValueError(
            "UE utility shape mismatch: "
            f"active={active.shape} goodput={goodput.shape} outcome_time={outcome_time.shape}"
        )
    if not np.any(active):
        return 0.0
    rate_mbps = np.zeros_like(goodput)
    positive_time = outcome_time > 0.0
    rate_mbps[positive_time] = (
        np.maximum(goodput[positive_time], 0.0) * 8.0
        / (outcome_time[positive_time] * 1000.0)
    )
    return float(np.log(float(epsilon) + rate_mbps[active]).mean())


def compute_ue_expiry_aware_rate_tail_utility(
    demand_mask: np.ndarray,
    successful_goodput_bytes: np.ndarray,
    outcome_system_time_ms: np.ndarray,
    *,
    tail_fraction: float = 0.10,
    epsilon_mbps: float = 1.0e-6,
) -> float:
    """Return the mean unit-normalized log rate of the worst active-UE tail.

    ``outcome_system_time_ms`` is supplied by the expiry-aware rolling
    tracker and already contains both delivered and expired packet residence
    time.  UEs with no outcome in the window remain at zero rate, so they are
    retained in the tail rather than disappearing from the objective.  Rates
    are in Mbps and use 1 Mbps as the fixed unit scale via ``log1p``; this
    avoids a large first-window bonus from ``log(epsilon)``.
    """
    active = np.asarray(demand_mask, dtype=bool).reshape(-1)
    goodput = np.asarray(successful_goodput_bytes, dtype=np.float64).reshape(-1)
    outcome_time = np.asarray(outcome_system_time_ms, dtype=np.float64).reshape(-1)
    if goodput.shape != active.shape or outcome_time.shape != active.shape:
        raise ValueError(
            "UE expiry-aware rate-tail shape mismatch: "
            f"active={active.shape} goodput={goodput.shape} outcome_time={outcome_time.shape}"
        )
    if not 0.0 < float(tail_fraction) <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    if float(epsilon_mbps) <= 0.0:
        raise ValueError("epsilon_mbps must be positive")
    if not np.any(active):
        return 0.0
    rate_mbps = np.zeros_like(goodput)
    positive_time = outcome_time > 0.0
    rate_mbps[positive_time] = (
        np.maximum(goodput[positive_time], 0.0) * 8.0
        / (outcome_time[positive_time] * 1000.0)
    )
    log_rates = np.log1p(
        np.maximum(rate_mbps[active], 0.0) + float(epsilon_mbps)
    )
    tail_count = max(1, int(np.ceil(float(tail_fraction) * log_rates.size)))
    return float(np.partition(log_rates, tail_count - 1)[:tail_count].mean())


class ExpiryAwareUeWindowTracker:
    def __init__(self, n_ue: int, window_steps: int = 400):
        if int(n_ue) <= 0:
            raise ValueError("n_ue must be positive")
        if int(window_steps) <= 0:
            raise ValueError("window_steps must be positive")
        self.n_ue = int(n_ue)
        self.window_steps = int(window_steps)
        self.reset()

    def reset(self) -> None:
        self._goodput_steps: deque[np.ndarray] = deque()
        self._outcome_time_steps: deque[np.ndarray] = deque()
        self._last_delivered_time_ms: np.ndarray | None = None
        self._last_expired_time_ms: np.ndarray | None = None
        self.window_successful_goodput_bytes = np.zeros((self.n_ue,), dtype=np.float64)
        self.window_outcome_system_time_ms = np.zeros((self.n_ue,), dtype=np.float64)

    def _array(self, value: np.ndarray, name: str, dtype) -> np.ndarray:
        out = np.asarray(value, dtype=dtype).reshape(-1)
        if out.shape != (self.n_ue,):
            raise ValueError(f"{name} shape mismatch: got={out.shape} expected={(self.n_ue,)}")
        return out

    def update(
        self,
        demand_mask: np.ndarray,
        successful_goodput_bytes: np.ndarray,
        cumulative_delivered_time_ms: np.ndarray,
        cumulative_expired_time_ms: np.ndarray,
    ) -> float:
        active = self._array(demand_mask, "demand_mask", bool)
        goodput = np.maximum(
            self._array(successful_goodput_bytes, "successful_goodput_bytes", np.float64),
            0.0,
        )
        delivered = self._array(
            cumulative_delivered_time_ms,
            "cumulative_delivered_time_ms",
            np.float64,
        )
        expired = self._array(
            cumulative_expired_time_ms,
            "cumulative_expired_time_ms",
            np.float64,
        )
        if self._last_delivered_time_ms is None or self._last_expired_time_ms is None:
            delivered_delta = np.maximum(delivered, 0.0)
            expired_delta = np.maximum(expired, 0.0)
        else:
            delivered_delta = delivered - self._last_delivered_time_ms
            expired_delta = expired - self._last_expired_time_ms
            if np.any(delivered_delta < -1.0e-6) or np.any(expired_delta < -1.0e-6):
                self.reset()
                delivered_delta = np.maximum(delivered, 0.0)
                expired_delta = np.maximum(expired, 0.0)
        self._last_delivered_time_ms = delivered.copy()
        self._last_expired_time_ms = expired.copy()
        outcome_delta = np.maximum(delivered_delta, 0.0) + np.maximum(expired_delta, 0.0)

        self._goodput_steps.append(goodput.copy())
        self._outcome_time_steps.append(outcome_delta.copy())
        self.window_successful_goodput_bytes += goodput
        self.window_outcome_system_time_ms += outcome_delta
        while len(self._goodput_steps) > self.window_steps:
            self.window_successful_goodput_bytes -= self._goodput_steps.popleft()
            self.window_outcome_system_time_ms -= self._outcome_time_steps.popleft()

        return compute_ue_expiry_aware_utility(
            active,
            self.window_successful_goodput_bytes,
            self.window_outcome_system_time_ms,
        )

    def utility(self, demand_mask: np.ndarray) -> float:
        """Return the current window utility without mutating its counters."""
        active = self._array(demand_mask, "demand_mask", bool)
        return compute_ue_expiry_aware_utility(
            active,
            self.window_successful_goodput_bytes,
            self.window_outcome_system_time_ms,
        )

    def rate_tail_utility(
        self,
        demand_mask: np.ndarray,
        *,
        tail_fraction: float = 0.10,
        epsilon_mbps: float = 1.0e-6,
    ) -> float:
        """Return the current worst-tail log rate without mutating counters."""
        return compute_ue_expiry_aware_rate_tail_utility(
            demand_mask,
            self.window_successful_goodput_bytes,
            self.window_outcome_system_time_ms,
            tail_fraction=tail_fraction,
            epsilon_mbps=epsilon_mbps,
        )


class DeadlineTailPressureTracker:
    """Rolling scheduled-service state for the deadline-tail potential.

    The pressure is intentionally computed from scheduled dequeue bytes, not
    successful bytes.  Backlog and TTL are read from the pre/post graph
    snapshots, while the rolling service-rate estimate is updated exactly once
    per environment transition.
    """

    def __init__(
        self,
        n_ue: int,
        *,
        window_steps: int = 100,
        slot_duration_s: float = 0.0005,
        epsilon_bytes: float = 1.0,
        tail_fraction: float = 0.10,
    ) -> None:
        if int(n_ue) <= 0:
            raise ValueError("n_ue must be positive")
        if int(window_steps) <= 0:
            raise ValueError("window_steps must be positive")
        if float(slot_duration_s) <= 0.0:
            raise ValueError("slot_duration_s must be positive")
        if float(epsilon_bytes) <= 0.0:
            raise ValueError("epsilon_bytes must be positive")
        if not 0.0 < float(tail_fraction) <= 1.0:
            raise ValueError("tail_fraction must be in (0, 1]")
        self.n_ue = int(n_ue)
        self.window_steps = int(window_steps)
        self.slot_duration_s = float(slot_duration_s)
        self.epsilon_bytes = float(epsilon_bytes)
        self.tail_fraction = float(tail_fraction)
        self._expiry_rate_tracker = ExpiryAwareUeWindowTracker(
            self.n_ue, window_steps=self.window_steps
        )
        self.reset()

    def reset(self) -> None:
        self._scheduled_history: deque[np.ndarray] = deque(maxlen=self.window_steps)
        self._expiry_rate_tracker.reset()

    @property
    def expiry_rate_tracker(self) -> ExpiryAwareUeWindowTracker:
        return self._expiry_rate_tracker

    @property
    def history_steps(self) -> int:
        return len(self._scheduled_history)

    def _service_rate_bps(self) -> np.ndarray:
        if not self._scheduled_history:
            return np.zeros((self.n_ue,), dtype=np.float64)
        scheduled = np.stack(tuple(self._scheduled_history), axis=0)
        return scheduled.sum(axis=0) / (
            float(len(self._scheduled_history)) * self.slot_duration_s
        )

    def potential(self, graph: SparseEntityGraph | GraphFeatureSnapshot) -> float:
        features = _snapshot(graph).ue_features.values.detach().cpu().numpy()
        if features.ndim != 2 or features.shape[0] != self.n_ue or features.shape[1] <= 9:
            raise ValueError(
                "deadline-tail pressure requires UE features [U,>=10], "
                f"got={features.shape} expected U={self.n_ue}"
            )
        backlog = np.maximum(features[:, 0].astype(np.float64, copy=False), 0.0)
        ttl_ms = features[:, 9].astype(np.float64, copy=False)
        active = (backlog > 0.0) & (ttl_ms >= 0.0)
        if not np.any(active):
            return 0.0
        ttl_s = np.maximum(ttl_ms, 0.0) * 1.0e-3
        pressure = backlog / (self._service_rate_bps() * ttl_s + self.epsilon_bytes)
        values = np.log1p(np.maximum(pressure[active], 0.0))
        tail_count = max(1, int(np.ceil(self.tail_fraction * values.size)))
        return float(np.sort(values)[-tail_count:].mean())

    def transition(
        self,
        before_graph: SparseEntityGraph | GraphFeatureSnapshot,
        after_graph: SparseEntityGraph | GraphFeatureSnapshot,
        scheduled_bytes: np.ndarray,
    ) -> dict[str, float]:
        before = self.potential(before_graph)
        scheduled = np.asarray(scheduled_bytes, dtype=np.float64).reshape(-1)
        if scheduled.shape != (self.n_ue,):
            raise ValueError(
                "scheduled_bytes shape mismatch: "
                f"got={scheduled.shape} expected={(self.n_ue,)}"
            )
        self._scheduled_history.append(np.maximum(scheduled, 0.0))
        after = self.potential(after_graph)
        return {
            "phi_before": float(before),
            "phi_after": float(after),
            "phi_delta": float(before - after),
            "scheduled_history_steps": float(self.history_steps),
        }


def _measured_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    warmup_steps: int,
    measure_steps: int,
) -> list[Mapping[str, object]]:
    lower = int(warmup_steps)
    upper = lower + int(measure_steps)
    measured = [row for row in rows if lower < int(row["tti"]) <= upper]
    if len(measured) != int(measure_steps):
        raise ValueError(
            f"measured row count mismatch: got={len(measured)} expected={int(measure_steps)}"
        )
    return measured


def compute_calibration_scales(
    rows: Sequence[Mapping[str, object]],
    *,
    warmup_steps: int,
    measure_steps: int,
) -> dict[str, float | int]:
    measured = _measured_rows(rows, warmup_steps=warmup_steps, measure_steps=measure_steps)
    delta_d = np.abs(np.asarray([float(row["raw_delta_d"]) for row in measured], dtype=np.float64))
    delta_u = np.abs(np.asarray([float(row["raw_delta_u"]) for row in measured], dtype=np.float64))
    return {
        "delay_scale_p95": float(np.percentile(delta_d, 95)),
        "ue_utility_scale_p95": float(np.percentile(delta_u, 95)),
        "warmup_steps": int(warmup_steps),
        "measure_steps": int(measure_steps),
    }


__all__ = [
    "DeadlineTailPressureTracker",
    "ExpiryAwareUeWindowTracker",
    "compute_calibration_scales",
    "compute_delay_debt_potential",
    "compute_ue_expiry_aware_rate_tail_utility",
    "compute_ue_expiry_aware_utility",
]

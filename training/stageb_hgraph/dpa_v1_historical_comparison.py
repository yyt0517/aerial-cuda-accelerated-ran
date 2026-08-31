"""Metrics for the DPA-v1 historical-comparison experiment.

The aggregator deliberately keeps evaluator-summary metrics separate from
metrics reconstructed from the per-TTI service and packet-outcome traces.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def _pct(values: Sequence[float], percentile: float) -> float:
    return (
        float(np.percentile(np.asarray(values, dtype=np.float64), percentile))
        if values
        else 0.0
    )


def _jain(values: Sequence[float]) -> float:
    total = float(sum(values))
    squares = float(sum(value * value for value in values))
    return 0.0 if squares <= 0.0 else total * total / (len(values) * squares)


@dataclass(frozen=True)
class WindowSpec:
    label: str
    first_tti: int
    last_tti: int

    @property
    def tti_count(self) -> int:
        return self.last_tti - self.first_tti + 1


def summarize_dpa_window(
    *,
    eval_summary: Mapping[str, object],
    service_rows: Sequence[Mapping[str, object]],
    outcome_rows: Sequence[Mapping[str, object]],
    window: WindowSpec,
    slot_duration_s: float = 0.0005,
) -> dict[str, object]:
    if window.first_tti <= 0 or window.last_tti < window.first_tti:
        raise ValueError("invalid TTI window")
    if window.label not in {"full_4000", "steady_3000"}:
        raise ValueError(f"unsupported window label: {window.label}")

    prefix = "" if window.label == "full_4000" else "stats_"
    selected_service = [
        row
        for row in service_rows
        if window.first_tti <= int(row["tti"]) <= window.last_tti
    ]
    selected_outcomes = [
        row
        for row in outcome_rows
        if window.first_tti <= int(row["terminal_tti"]) <= window.last_tti
    ]

    by_ue: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in selected_service:
        by_ue[int(row["ue_id"])].append(row)
    if not by_ue:
        raise ValueError("service window is empty")
    expected_ttis = set(range(window.first_tti, window.last_tti + 1))
    for ue_id, rows in by_ue.items():
        observed = {int(row["tti"]) for row in rows}
        if observed != expected_ttis:
            raise ValueError(f"incomplete service coverage for UE {ue_id}")

    duration_s = window.tti_count * slot_duration_s
    ue_goodput: list[float] = []
    scheduled_ratio: list[float] = []
    positive_dequeue_ratio: list[float] = []
    zero_service_ids: list[int] = []
    max_streak = 0
    expired_bytes = 0

    for ue_id in sorted(by_ue):
        rows = sorted(by_ue[ue_id], key=lambda row: int(row["tti"]))
        successful = sum(int(row["successful_bytes"]) for row in rows)
        scheduled = sum(int(row["scheduled_dequeue_bytes"]) for row in rows)
        expired_bytes += sum(int(row["expired_bytes"]) for row in rows)
        ue_goodput.append(successful * 8.0 / duration_s / 1.0e6)
        scheduled_ratio.append(
            sum(int(row["actual_prg_count"]) > 0 for row in rows) / window.tti_count
        )
        positive_dequeue_ratio.append(
            sum(int(row["scheduled_dequeue_bytes"]) > 0 for row in rows)
            / window.tti_count
        )

        had_backlog = any(int(row["pre_queue_bytes"]) > 0 for row in rows)
        if had_backlog and scheduled == 0:
            zero_service_ids.append(ue_id)
        current_streak = 0
        for row in rows:
            if int(row["pre_queue_bytes"]) > 0 and int(row["scheduled_dequeue_bytes"]) == 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

    # Delays intentionally include every terminal outcome, including expired
    # packets. This keeps the delay distribution boundary-consistent.
    delays = [float(row["system_time_ms"]) for row in selected_outcomes]
    outcomes_by_ue: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in selected_outcomes:
        outcomes_by_ue[int(row["ue_id"])].append(row)

    ue_rates: list[float] = []
    for rows in outcomes_by_ue.values():
        terminal_rows = [
            row
            for row in rows
            if str(row["outcome"]) in {"delivered", "expired"}
        ]
        outcome_time_ms = sum(float(row["system_time_ms"]) for row in terminal_rows)
        delivered_bytes = sum(
            int(row["packet_size_bytes"])
            for row in terminal_rows
            if str(row["outcome"]) == "delivered"
        )
        if outcome_time_ms > 0.0:
            # bytes * 8 / milliseconds / 1000 = megabits/second
            ue_rates.append(delivered_bytes * 8.0 / (outcome_time_ms * 1000.0))

    return {
        "window": window.label,
        "first_tti": window.first_tti,
        "last_tti": window.last_tti,
        "tti_count": window.tti_count,
        "goodput_mbps": float(eval_summary[f"{prefix}mean_goodput_mbps"]),
        "bler": float(eval_summary[f"{prefix}mean_tb_err_rate"]),
        "expiry_drop_rate": float(eval_summary[f"{prefix}mean_expiry_drop_rate"]),
        "prg_utilization_ratio": float(
            eval_summary[f"{prefix}mean_prg_utilization_ratio"]
        ),
        "blank_ratio": float(eval_summary[f"{prefix}mean_blank_ratio"]),
        "expired_bytes": expired_bytes,
        "outcome_count": len(delays),
        "outcome_delay_mean_ms": float(np.mean(delays)) if delays else 0.0,
        "outcome_delay_p50_ms": _pct(delays, 50),
        "outcome_delay_p90_ms": _pct(delays, 90),
        "outcome_delay_p95_ms": _pct(delays, 95),
        "outcome_delay_max_ms": max(delays, default=0.0),
        "ue_goodput_mean_mbps": float(np.mean(ue_goodput)),
        "ue_goodput_p5_mbps": _pct(ue_goodput, 5),
        "ue_goodput_p10_mbps": _pct(ue_goodput, 10),
        "ue_goodput_min_mbps": min(ue_goodput),
        "ue_goodput_jain": _jain(ue_goodput),
        "scheduled_ratio_mean": float(np.mean(scheduled_ratio)),
        "scheduled_ratio_p5": _pct(scheduled_ratio, 5),
        "scheduled_ratio_jain": _jain(scheduled_ratio),
        "positive_dequeue_ratio_mean": float(np.mean(positive_dequeue_ratio)),
        "ue_expiry_aware_rate_mean_mbps": float(np.mean(ue_rates)) if ue_rates else 0.0,
        "ue_expiry_aware_rate_p10_mbps": _pct(ue_rates, 10),
        "ue_expiry_aware_rate_min_mbps": min(ue_rates, default=0.0),
        "zero_scheduled_dequeue_ue_ids": zero_service_ids,
        "max_backlogged_zero_dequeue_streak_tti": max_streak,
        "metric_units": {
            "goodput_mbps": "Mbps",
            "bler": "ratio",
            "expiry_drop_rate": "ratio",
            "outcome_delay_mean_ms": "ms",
            "ue_expiry_aware_rate_mean_mbps": "Mbps",
        },
        "source_semantics": {
            "system_metrics": (
                "evaluator summary; mean_* for full_4000 and stats_mean_* for steady_3000"
            ),
            "expired_bytes": "sum of service-balance expired_bytes over the inclusive TTI window",
            "outcome_delay": (
                "all packet outcomes selected by inclusive terminal_tti, including delivered and expired"
            ),
            "ue_goodput": "successful_bytes aggregated per UE over the inclusive service window",
            "ue_expiry_aware_rate": (
                "delivered packet bytes*8 divided by that UE's delivered-plus-expired outcome time"
            ),
            "starvation": "positive-backlog rows with zero scheduled dequeue",
        },
    }

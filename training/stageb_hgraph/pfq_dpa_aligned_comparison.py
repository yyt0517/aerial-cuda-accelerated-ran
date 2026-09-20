"""Strictly aligned PFQ/DPA expiry-aware comparison metrics.

The module deliberately reconstructs both arms from the same service and
packet-outcome trace schemas.  Legacy served-only packet delay and packet-rate
metrics are not part of this contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .dpa_v1_historical_comparison import WindowSpec


SEEDS = tuple(range(41, 51))
FULL_WINDOW = WindowSpec("full_4000", 1, 4000)
STEADY_WINDOW = WindowSpec("steady_3000", 1001, 4000)

HEADLINE_METRICS = (
    "goodput_mbps",
    "served_mbps",
    "tb_bler",
    "expiry_drop_rate",
    "expired_bytes",
    "outcome_delay_mean_ms",
    "outcome_delay_p50_ms",
    "outcome_delay_p90_ms",
    "outcome_delay_p95_ms",
    "outcome_delay_max_ms",
    "ue_expiry_aware_rate_mean_mbps",
    "ue_expiry_aware_rate_p10_mbps",
    "ue_expiry_aware_rate_min_mbps",
    "ue_goodput_mean_mbps",
    "ue_goodput_p5_mbps",
    "ue_goodput_p10_mbps",
    "ue_goodput_min_mbps",
    "ue_goodput_jain",
    "prg_utilization_ratio",
    "blank_ratio",
    "prg_reuse_ratio",
    "unique_ue_mean",
    "end_backlog_bytes",
    "backlog_free_ue_ratio",
    "max_backlogged_zero_dequeue_streak_tti",
)

METRIC_DIRECTIONS = {
    "goodput_mbps": "higher_better",
    "served_mbps": "higher_better",
    "tb_bler": "lower_better",
    "expiry_drop_rate": "lower_better",
    "expired_bytes": "lower_better",
    "outcome_delay_mean_ms": "lower_better",
    "outcome_delay_p50_ms": "lower_better",
    "outcome_delay_p90_ms": "lower_better",
    "outcome_delay_p95_ms": "lower_better",
    "outcome_delay_max_ms": "lower_better",
    "ue_expiry_aware_rate_mean_mbps": "higher_better",
    "ue_expiry_aware_rate_p10_mbps": "higher_better",
    "ue_expiry_aware_rate_min_mbps": "higher_better",
    "ue_goodput_mean_mbps": "higher_better",
    "ue_goodput_p5_mbps": "higher_better",
    "ue_goodput_p10_mbps": "higher_better",
    "ue_goodput_min_mbps": "higher_better",
    "ue_goodput_jain": "higher_better",
    "prg_utilization_ratio": "context",
    "blank_ratio": "context",
    "prg_reuse_ratio": "context",
    "unique_ue_mean": "context",
    "end_backlog_bytes": "lower_better",
    "backlog_free_ue_ratio": "higher_better",
    "max_backlogged_zero_dequeue_streak_tti": "lower_better",
}

OUTCOME_FIELDS = {
    "ue_id",
    "packet_id",
    "terminal_tti",
    "outcome",
    "packet_size_bytes",
    "system_time_ms",
}


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


def summarize_aligned_window(
    *,
    service_rows: Sequence[Mapping[str, object]],
    outcome_rows: Sequence[Mapping[str, object]],
    decision_rows: Sequence[Mapping[str, object]],
    window: WindowSpec,
    slot_duration_s: float = 0.0005,
    n_cell: int = 3,
    n_prg: int = 17,
) -> dict[str, object]:
    """Compute one canonical window identically for PFQ and DPA."""
    if window.first_tti <= 0 or window.last_tti < window.first_tti:
        raise ValueError("invalid TTI window")
    if slot_duration_s <= 0.0 or n_cell <= 0 or n_prg <= 0:
        raise ValueError("invalid aligned metric dimensions")

    selected_service = [
        row
        for row in service_rows
        if window.first_tti <= int(row["tti"]) <= window.last_tti
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

    terminal_outcomes = [
        row
        for row in outcome_rows
        if window.first_tti <= int(row["terminal_tti"]) <= window.last_tti
        and str(row["outcome"]) in {"delivered", "expired"}
    ]
    for row in terminal_outcomes:
        for field in ("ue_id", "packet_size_bytes", "system_time_ms"):
            if field not in row:
                raise ValueError(f"packet outcome is missing {field}")

    duration_s = window.tti_count * slot_duration_s
    total_successful_bytes = sum(
        int(row["successful_bytes"]) for row in selected_service
    )
    total_scheduled_bytes = sum(
        int(row["scheduled_dequeue_bytes"]) for row in selected_service
    )
    total_expired_bytes = sum(int(row["expired_bytes"]) for row in selected_service)
    total_offered_bytes = sum(int(row.get("offered_bytes", 0)) for row in selected_service)
    total_tb_tx = sum(int(row.get("tb_tx_count", 0)) for row in selected_service)
    total_tb_err = sum(int(row.get("tb_err_count", 0)) for row in selected_service)
    total_prg = sum(int(row["actual_prg_count"]) for row in selected_service)

    service_by_tti: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in selected_service:
        service_by_tti[int(row["tti"])].append(row)
    unique_ue_by_tti = [
        sum(int(row["actual_prg_count"]) > 0 for row in service_by_tti[tti])
        for tti in range(window.first_tti, window.last_tti + 1)
    ]

    outcome_by_ue: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in terminal_outcomes:
        outcome_by_ue[int(row["ue_id"])].append(row)
    ue_rates: list[float] = []
    no_outcome_ue_count = 0
    for ue_id in sorted(by_ue):
        rows = outcome_by_ue.get(ue_id, [])
        outcome_time_ms = sum(float(row["system_time_ms"]) for row in rows)
        delivered_bytes = sum(
            int(row["packet_size_bytes"])
            for row in rows
            if str(row["outcome"]) == "delivered"
        )
        if outcome_time_ms > 0.0:
            ue_rates.append(delivered_bytes * 8.0 / (outcome_time_ms * 1000.0))
        else:
            ue_rates.append(0.0)
            no_outcome_ue_count += 1

    ue_goodput: list[float] = []
    scheduled_ratios: list[float] = []
    positive_dequeue_ratios: list[float] = []
    zero_service_ids: list[int] = []
    max_zero_dequeue_streak = 0
    end_backlog_by_ue: dict[str, int] = {}
    for ue_id in sorted(by_ue):
        rows = sorted(by_ue[ue_id], key=lambda row: int(row["tti"]))
        ue_success = sum(int(row["successful_bytes"]) for row in rows)
        ue_scheduled = sum(int(row["scheduled_dequeue_bytes"]) for row in rows)
        ue_goodput.append(ue_success * 8.0 / duration_s / 1.0e6)
        scheduled_ratios.append(
            sum(int(row["actual_prg_count"]) > 0 for row in rows) / window.tti_count
        )
        positive_dequeue_ratios.append(
            sum(int(row["scheduled_dequeue_bytes"]) > 0 for row in rows)
            / window.tti_count
        )
        if any(int(row["pre_queue_bytes"]) > 0 for row in rows) and ue_scheduled == 0:
            zero_service_ids.append(ue_id)
        streak = 0
        for row in rows:
            if int(row["pre_queue_bytes"]) > 0 and int(row["scheduled_dequeue_bytes"]) == 0:
                streak += 1
                max_zero_dequeue_streak = max(max_zero_dequeue_streak, streak)
            else:
                streak = 0
        end_backlog_by_ue[str(ue_id)] = int(rows[-1]["post_queue_bytes"])

    selected_decisions = [
        row
        for row in decision_rows
        if window.first_tti <= int(row["tti"]) <= window.last_tti
    ]
    if selected_decisions and len(selected_decisions) != window.tti_count:
        raise ValueError("decision trace does not cover the complete TTI window")
    prg_reuse_values = [
        float(row["prg_reuse_ratio"])
        for row in selected_decisions
        if row.get("prg_reuse_ratio") is not None
    ]

    delays = [float(row["system_time_ms"]) for row in terminal_outcomes]
    end_backlogs = list(end_backlog_by_ue.values())
    utilization = total_prg / float(window.tti_count * n_cell * n_prg)
    return {
        "window": window.label,
        "first_tti": window.first_tti,
        "last_tti": window.last_tti,
        "tti_count": window.tti_count,
        "active_ue_count": len(by_ue),
        "goodput_mbps": total_successful_bytes * 8.0 / duration_s / 1.0e6,
        "served_mbps": total_scheduled_bytes * 8.0 / duration_s / 1.0e6,
        "tb_bler": total_tb_err / total_tb_tx if total_tb_tx else 0.0,
        "tb_tx_count": total_tb_tx,
        "tb_err_count": total_tb_err,
        "expired_bytes": total_expired_bytes,
        "expiry_drop_rate": (
            total_expired_bytes / total_offered_bytes if total_offered_bytes else 0.0
        ),
        "terminal_outcome_count": len(terminal_outcomes),
        "delivered_outcome_count": sum(
            str(row["outcome"]) == "delivered" for row in terminal_outcomes
        ),
        "expired_outcome_count": sum(
            str(row["outcome"]) == "expired" for row in terminal_outcomes
        ),
        "outcome_delay_mean_ms": float(np.mean(delays)) if delays else 0.0,
        "outcome_delay_p50_ms": _pct(delays, 50),
        "outcome_delay_p90_ms": _pct(delays, 90),
        "outcome_delay_p95_ms": _pct(delays, 95),
        "outcome_delay_max_ms": max(delays, default=0.0),
        "ue_expiry_aware_rate_mean_mbps": float(np.mean(ue_rates)),
        "ue_expiry_aware_rate_p10_mbps": _pct(ue_rates, 10),
        "ue_expiry_aware_rate_min_mbps": min(ue_rates, default=0.0),
        "ue_expiry_aware_rate_ue_count": len(ue_rates),
        "ue_expiry_aware_rate_no_outcome_ue_count": no_outcome_ue_count,
        "ue_goodput_mean_mbps": float(np.mean(ue_goodput)),
        "ue_goodput_p5_mbps": _pct(ue_goodput, 5),
        "ue_goodput_p10_mbps": _pct(ue_goodput, 10),
        "ue_goodput_min_mbps": min(ue_goodput, default=0.0),
        "ue_goodput_jain": _jain(ue_goodput),
        "scheduled_ratio_mean": float(np.mean(scheduled_ratios)),
        "scheduled_ratio_p5": _pct(scheduled_ratios, 5),
        "scheduled_ratio_jain": _jain(scheduled_ratios),
        "positive_dequeue_ratio_mean": float(np.mean(positive_dequeue_ratios)),
        "prg_utilization_ratio": utilization,
        "blank_ratio": 1.0 - utilization,
        "prg_reuse_ratio": (
            float(np.mean(prg_reuse_values)) if prg_reuse_values else None
        ),
        "unique_ue_mean": float(np.mean(unique_ue_by_tti)),
        "end_backlog_bytes": sum(end_backlogs),
        "backlog_free_ue_ratio": (
            sum(value == 0 for value in end_backlogs) / len(end_backlogs)
        ),
        "zero_scheduled_dequeue_ue_ids": zero_service_ids,
        "max_backlogged_zero_dequeue_streak_tti": max_zero_dequeue_streak,
        "end_backlog_bytes_by_ue": end_backlog_by_ue,
        "source_semantics": {
            "delay": "delivered and expired terminal outcomes",
            "ue_effective_service_rate": (
                "delivered packet bits divided by delivered-plus-expired outcome system time; "
                "all active UEs included"
            ),
            "pending": "right-censored; represented by end backlog rather than fabricated delay",
        },
    }


def _read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object: {path}")
            rows.append(value)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_fields(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return set(csv.DictReader(handle).fieldnames or [])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _trace_has_exact_ttis(rows: Sequence[Mapping[str, object]], count: int) -> bool:
    return [int(row["tti"]) for row in rows] == list(range(1, count + 1))


def _service_has_exact_grid(
    rows: Sequence[Mapping[str, object]], *, tti_count: int, ue_count: int
) -> bool:
    return (
        len(rows) == tti_count * ue_count
        and sorted({int(row["tti"]) for row in rows}) == list(range(1, tti_count + 1))
        and sorted({int(row["ue_id"]) for row in rows}) == list(range(ue_count))
    )


def validate_seed_artifacts(
    *,
    seed: int,
    dpa_dir: Path,
    pfq_dir: Path,
    expected_sim_sha256: str,
    expected_tti_count: int = 4000,
    expected_ue_count: int = 36,
) -> dict[str, object]:
    """Reject any PFQ/DPA pair that is not structurally and causally aligned."""
    dpa_dir = Path(dpa_dir).resolve()
    pfq_dir = Path(pfq_dir).resolve()
    required_dpa = (
        "eval_summary.json",
        "eval_steps.jsonl",
        "service_balance_trace.csv",
        "packet_outcome_samples.csv",
        "traffic_offered_packets.csv",
        "ue_topology.csv",
    )
    required_pfq = (
        "reference_manifest.json",
        "tti_trace.jsonl",
        "service_balance_trace.csv",
        "packet_outcome_samples.csv",
        "traffic_offered_packets.csv",
        "ue_topology.csv",
    )
    for name in required_dpa:
        if not (dpa_dir / name).is_file():
            raise ValueError(f"missing DPA artifact: {dpa_dir / name}")
    for name in required_pfq:
        if not (pfq_dir / name).is_file():
            raise ValueError(f"missing PFQ artifact: {pfq_dir / name}")

    dpa_summary = _require_mapping(_read_json(dpa_dir / "eval_summary.json"), "DPA summary")
    pfq_manifest = _require_mapping(
        _read_json(pfq_dir / "reference_manifest.json"), "PFQ manifest"
    )
    scenario = _require_mapping(pfq_manifest.get("scenario"), "PFQ scenario")
    dpa_steps = _read_jsonl(dpa_dir / "eval_steps.jsonl")
    pfq_steps = _read_jsonl(pfq_dir / "tti_trace.jsonl")
    dpa_service = _read_csv(dpa_dir / "service_balance_trace.csv")
    pfq_service = _read_csv(pfq_dir / "service_balance_trace.csv")
    dpa_outcome_fields = _csv_fields(dpa_dir / "packet_outcome_samples.csv")
    pfq_outcome_fields = _csv_fields(pfq_dir / "packet_outcome_samples.csv")
    offered_dpa = _sha256(dpa_dir / "traffic_offered_packets.csv")
    offered_pfq = _sha256(pfq_dir / "traffic_offered_packets.csv")
    topology_dpa = _sha256(dpa_dir / "ue_topology.csv")
    topology_pfq = _sha256(pfq_dir / "ue_topology.csv")

    checks = {
        "dpa_simulator_hash": dpa_summary.get("sim_binary_sha256")
        == expected_sim_sha256,
        "pfq_simulator_hash": pfq_manifest.get("binary_sha256")
        == expected_sim_sha256,
        "dpa_completed_steps": int(dpa_summary.get("completed_steps", -1))
        == expected_tti_count,
        "pfq_zero_extra_warmup": int(pfq_manifest.get("warmup_steps", -1)) == 0,
        "pfq_measure_steps": int(pfq_manifest.get("measure_steps", -1))
        == expected_tti_count,
        "topology_seed": int(scenario.get("topology_seed", -1)) == int(seed),
        "traffic_seed": int(scenario.get("traffic_seed", -1)) == int(seed),
        "dpa_decision_tti_coverage": _trace_has_exact_ttis(
            dpa_steps, expected_tti_count
        ),
        "pfq_decision_tti_coverage": _trace_has_exact_ttis(
            pfq_steps, expected_tti_count
        ),
        "dpa_service_grid": _service_has_exact_grid(
            dpa_service, tti_count=expected_tti_count, ue_count=expected_ue_count
        ),
        "pfq_service_grid": _service_has_exact_grid(
            pfq_service, tti_count=expected_tti_count, ue_count=expected_ue_count
        ),
        "dpa_outcome_schema": OUTCOME_FIELDS.issubset(dpa_outcome_fields),
        "pfq_outcome_schema": OUTCOME_FIELDS.issubset(pfq_outcome_fields),
        "offered_stream_hash": offered_dpa == offered_pfq,
        "topology_hash": topology_dpa == topology_pfq,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"seed {seed} alignment failed: {', '.join(failed)}")
    return {
        "aligned": True,
        "checks": checks,
        "hashes": {
            "simulator": expected_sim_sha256,
            "offered_stream": offered_dpa,
            "topology": topology_dpa,
            "dpa_outcomes": _sha256(dpa_dir / "packet_outcome_samples.csv"),
            "pfq_outcomes": _sha256(pfq_dir / "packet_outcome_samples.csv"),
        },
    }


def _describe(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _winner(dpa: float, pfq: float, direction: str) -> str:
    if np.isclose(dpa, pfq, rtol=1.0e-12, atol=1.0e-12):
        return "tie"
    if direction == "higher_better":
        return "dpa_v1" if dpa > pfq else "pfq"
    if direction == "lower_better":
        return "dpa_v1" if dpa < pfq else "pfq"
    return "context"


def aggregate_paired_results(
    seed_payloads: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not seed_payloads:
        raise ValueError("paired seed payloads are empty")
    seeds = [int(payload["seed"]) for payload in seed_payloads]
    if len(set(seeds)) != len(seeds):
        raise ValueError("paired seed payloads contain duplicate seeds")
    windows = sorted(
        {
            str(window)
            for payload in seed_payloads
            for window in _require_mapping(payload["windows"], "windows")
        }
    )
    aggregate: dict[str, dict[str, object]] = {}
    per_seed: list[dict[str, object]] = []
    for window in windows:
        aggregate[window] = {}
        for metric in HEADLINE_METRICS:
            values: list[tuple[int, float, float]] = []
            for payload in seed_payloads:
                window_payload = _require_mapping(payload["windows"], "windows").get(window)
                if not isinstance(window_payload, Mapping):
                    continue
                dpa_metrics = _require_mapping(window_payload.get("dpa_v1"), "DPA metrics")
                pfq_metrics = _require_mapping(window_payload.get("pfq"), "PFQ metrics")
                if dpa_metrics.get(metric) is None or pfq_metrics.get(metric) is None:
                    continue
                values.append(
                    (
                        int(payload["seed"]),
                        float(dpa_metrics[metric]),
                        float(pfq_metrics[metric]),
                    )
                )
            if not values:
                continue
            dpa_values = [value[1] for value in values]
            pfq_values = [value[2] for value in values]
            deltas = [dpa - pfq for _, dpa, pfq in values]
            direction = METRIC_DIRECTIONS[metric]
            winners = [_winner(dpa, pfq, direction) for _, dpa, pfq in values]
            aggregate[window][metric] = {
                "direction": direction,
                "paired_seed_count": len(values),
                "dpa_v1": _describe(dpa_values),
                "pfq": _describe(pfq_values),
                "paired_delta_dpa_minus_pfq": _describe(deltas),
                "wins": {
                    "dpa_v1": winners.count("dpa_v1"),
                    "tie": winners.count("tie"),
                    "pfq": winners.count("pfq"),
                },
            }
            for (seed, dpa, pfq), winner in zip(values, winners):
                per_seed.append(
                    {
                        "seed": seed,
                        "window": window,
                        "metric": metric,
                        "direction": direction,
                        "dpa_v1": dpa,
                        "pfq": pfq,
                        "delta_dpa_minus_pfq": dpa - pfq,
                        "winner": winner,
                    }
                )
    return {
        "schema_version": 1,
        "seeds": sorted(seeds),
        "headline_metrics": list(HEADLINE_METRICS),
        "metric_semantics": {
            "outcome_delay": "delivered plus expired terminal packet outcomes",
            "ue_effective_service_rate": (
                "delivered packet bits / delivered-plus-expired outcome system time; "
                "all active UEs included"
            ),
            "pending": "right-censored and represented by end backlog",
        },
        "aggregate": aggregate,
        "per_seed_comparison": sorted(
            per_seed, key=lambda row: (int(row["seed"]), str(row["window"]), str(row["metric"]))
        ),
    }


def extract_seed_payload(
    *,
    seed: int,
    dpa_dir: Path,
    pfq_dir: Path,
    expected_sim_sha256: str,
) -> dict[str, object]:
    alignment = validate_seed_artifacts(
        seed=seed,
        dpa_dir=dpa_dir,
        pfq_dir=pfq_dir,
        expected_sim_sha256=expected_sim_sha256,
    )
    dpa_service = _read_csv(Path(dpa_dir) / "service_balance_trace.csv")
    pfq_service = _read_csv(Path(pfq_dir) / "service_balance_trace.csv")
    dpa_outcomes = _read_csv(Path(dpa_dir) / "packet_outcome_samples.csv")
    pfq_outcomes = _read_csv(Path(pfq_dir) / "packet_outcome_samples.csv")
    dpa_decisions = _read_jsonl(Path(dpa_dir) / "eval_steps.jsonl")
    pfq_decisions = _read_jsonl(Path(pfq_dir) / "tti_trace.jsonl")
    windows: dict[str, object] = {}
    for window in (FULL_WINDOW, STEADY_WINDOW):
        windows[window.label] = {
            "dpa_v1": summarize_aligned_window(
                service_rows=dpa_service,
                outcome_rows=dpa_outcomes,
                decision_rows=dpa_decisions,
                window=window,
            ),
            "pfq": summarize_aligned_window(
                service_rows=pfq_service,
                outcome_rows=pfq_outcomes,
                decision_rows=pfq_decisions,
                window=window,
            ),
        }
    return {
        "schema_version": 1,
        "seed": seed,
        "aligned": True,
        "alignment": alignment,
        "windows": windows,
        "sources": {
            "dpa_dir": str(Path(dpa_dir).resolve()),
            "pfq_dir": str(Path(pfq_dir).resolve()),
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sim_hash_from_dpa_manifest(payload: Mapping[str, object]) -> str:
    hashes = _require_mapping(payload.get("hashes"), "DPA manifest hashes")
    simulator = _require_mapping(hashes.get("simulator"), "DPA simulator hash")
    value = str(simulator.get("sha256", ""))
    if not value:
        raise ValueError("DPA simulator SHA-256 is missing")
    return value


def run_preflight(args: argparse.Namespace) -> None:
    dpa_root = Path(args.dpa_root).resolve()
    dpa_manifest_path = dpa_root / "campaign_manifest.json"
    dpa_manifest = _require_mapping(_read_json(dpa_manifest_path), "DPA campaign manifest")
    if [int(seed) for seed in dpa_manifest.get("seeds", [])] != list(SEEDS):
        raise ValueError("DPA campaign does not contain frozen seeds 41--50")
    sim_hash = _sim_hash_from_dpa_manifest(dpa_manifest)
    sim_bin = Path(args.sim_bin).resolve()
    if _sha256(sim_bin) != sim_hash:
        raise ValueError("simulator binary hash differs from DPA campaign")
    payload = {
        "schema_version": 1,
        "protocol_id": "dpa-v1-pfq-seed41-50-aligned-expiry-aware-v1",
        "seeds": list(SEEDS),
        "windows": {
            FULL_WINDOW.label: vars(FULL_WINDOW),
            STEADY_WINDOW.label: vars(STEADY_WINDOW),
        },
        "simulator": {"path": str(sim_bin), "sha256": sim_hash},
        "dpa_root": str(dpa_root),
        "metric_semantics": {
            "delay": "delivered and expired outcomes",
            "ue_effective_service_rate": (
                "delivered packet bits / delivered-plus-expired outcome time; 36 active UEs"
            ),
        },
    }
    _write_json(Path(args.out).resolve(), payload)


def run_extract_seed(args: argparse.Namespace) -> None:
    manifest = _require_mapping(
        _read_json(Path(args.campaign_manifest).resolve()), "campaign manifest"
    )
    sim_hash = str(_require_mapping(manifest["simulator"], "simulator")["sha256"])
    payload = extract_seed_payload(
        seed=int(args.seed),
        dpa_dir=Path(args.dpa_dir),
        pfq_dir=Path(args.pfq_dir),
        expected_sim_sha256=sim_hash,
    )
    _write_json(Path(args.out).resolve(), payload)


def _write_report_csvs(out_root: Path, report: Mapping[str, object]) -> None:
    per_seed = report["per_seed_comparison"]
    with (out_root / "per_seed_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = (
            "seed",
            "window",
            "metric",
            "direction",
            "dpa_v1",
            "pfq",
            "delta_dpa_minus_pfq",
            "winner",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_seed)
    with (out_root / "aggregate_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = (
            "window",
            "metric",
            "direction",
            "arm",
            "mean",
            "std",
            "median",
            "min",
            "max",
            "dpa_wins",
            "ties",
            "pfq_wins",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        aggregate = _require_mapping(report["aggregate"], "aggregate")
        for window, metrics_raw in aggregate.items():
            metrics = _require_mapping(metrics_raw, "aggregate window")
            for metric, record_raw in metrics.items():
                record = _require_mapping(record_raw, "aggregate metric")
                wins = _require_mapping(record["wins"], "wins")
                for arm in ("dpa_v1", "pfq", "paired_delta_dpa_minus_pfq"):
                    stats = _require_mapping(record[arm], "statistics")
                    writer.writerow(
                        {
                            "window": window,
                            "metric": metric,
                            "direction": record["direction"],
                            "arm": arm,
                            **stats,
                            "dpa_wins": wins["dpa_v1"],
                            "ties": wins["tie"],
                            "pfq_wins": wins["pfq"],
                        }
                    )


def run_aggregate(args: argparse.Namespace) -> None:
    campaign_root = Path(args.campaign_root).resolve()
    payloads = [
        _require_mapping(
            _read_json(campaign_root / f"seed{seed}" / "seed_metrics.json"),
            f"seed{seed} metrics",
        )
        for seed in SEEDS
    ]
    report = aggregate_paired_results(payloads)
    report["protocol_id"] = "dpa-v1-pfq-seed41-50-aligned-expiry-aware-v1"
    report["all_seeds_aligned"] = all(bool(payload.get("aligned")) for payload in payloads)
    report["stress_seeds"] = {
        str(seed): next(payload for payload in payloads if int(payload["seed"]) == seed)
        for seed in (45, 48)
    }
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    _write_json(out_root / "final_report.json", report)
    _write_json(out_root / "per_seed_comparison.json", report["per_seed_comparison"])
    _write_json(out_root / "aggregate_comparison.json", report["aggregate"])
    _write_report_csvs(out_root, report)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly aligned DPA-v1 versus PFQ expiry-aware comparison"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--dpa-root", required=True)
    preflight.add_argument("--sim-bin", required=True)
    preflight.add_argument("--out", required=True)
    preflight.set_defaults(func=run_preflight)
    extract = subparsers.add_parser("extract-seed")
    extract.add_argument("--campaign-manifest", required=True)
    extract.add_argument("--seed", type=int, choices=SEEDS, required=True)
    extract.add_argument("--dpa-dir", required=True)
    extract.add_argument("--pfq-dir", required=True)
    extract.add_argument("--out", required=True)
    extract.set_defaults(func=run_extract_seed)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--campaign-root", required=True)
    aggregate.add_argument("--out-root", required=True)
    aggregate.set_defaults(func=run_aggregate)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

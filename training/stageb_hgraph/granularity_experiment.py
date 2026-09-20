"""Minimal fixed-load packet-granularity experiment metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .dpa_v1_historical_comparison import WindowSpec
from .pfq_dpa_aligned_comparison import FULL_WINDOW, STEADY_WINDOW, summarize_aligned_window


_COHORT_METRICS = (
    "active_ue_count",
    "goodput_mbps",
    "served_mbps",
    "tb_bler",
    "tb_tx_count",
    "tb_err_count",
    "expired_bytes",
    "expiry_drop_rate",
    "terminal_outcome_count",
    "delivered_outcome_count",
    "expired_outcome_count",
    "outcome_delay_mean_ms",
    "outcome_delay_p50_ms",
    "outcome_delay_p90_ms",
    "outcome_delay_p95_ms",
    "outcome_delay_max_ms",
    "ue_expiry_aware_rate_mean_mbps",
    "ue_expiry_aware_rate_p10_mbps",
    "ue_expiry_aware_rate_min_mbps",
    "end_backlog_bytes",
    "backlog_free_ue_ratio",
    "max_backlogged_zero_dequeue_streak_tti",
)

_T_CRITICAL_95 = {2: 4.3026527299, 9: 2.2621571629}


def traffic_class_by_ue(profile: str, n_ue: int) -> dict[int, str]:
    if profile == "homogeneous":
        return {ue: "homogeneous" for ue in range(n_ue)}
    if profile == "mixed_granularity_v1":
        if n_ue % 2:
            raise ValueError("mixed_granularity_v1 requires an even UE count")
        return {
            ue: "high_frequency_short" if ue % 2 == 0 else "low_frequency_large"
            for ue in range(n_ue)
        }
    raise ValueError(f"unknown traffic profile: {profile}")


def _cohort_metrics(
    *,
    service_rows: Sequence[Mapping[str, object]],
    outcome_rows: Sequence[Mapping[str, object]],
    ue_ids: set[int],
    window: WindowSpec,
) -> dict[str, object]:
    service = [row for row in service_rows if int(row["ue_id"]) in ue_ids]
    outcomes = [row for row in outcome_rows if int(row["ue_id"]) in ue_ids]
    metrics = summarize_aligned_window(
        service_rows=service,
        outcome_rows=outcomes,
        decision_rows=[],
        window=window,
    )
    selected = {key: metrics[key] for key in _COHORT_METRICS}
    offered_bytes = sum(
        int(row.get("offered_bytes", 0))
        for row in service
        if window.first_tti <= int(row["tti"]) <= window.last_tti
    )
    selected["offered_bytes"] = offered_bytes
    selected["offered_bytes_per_tti"] = offered_bytes / window.tti_count
    return selected


def build_granularity_summary(
    *,
    service_rows: Sequence[Mapping[str, object]],
    outcome_rows: Sequence[Mapping[str, object]],
    decision_rows: Sequence[Mapping[str, object]],
    profile: str,
    window: WindowSpec,
) -> dict[str, object]:
    ue_ids = sorted({int(row["ue_id"]) for row in service_rows})
    if ue_ids != list(range(len(ue_ids))):
        raise ValueError("service trace UE IDs must be contiguous from zero")
    classes = traffic_class_by_ue(profile, len(ue_ids))

    overall = summarize_aligned_window(
        service_rows=service_rows,
        outcome_rows=outcome_rows,
        decision_rows=decision_rows,
        window=window,
    )
    selected_service = [
        row
        for row in service_rows
        if window.first_tti <= int(row["tti"]) <= window.last_tti
    ]
    offered_bytes = sum(int(row.get("offered_bytes", 0)) for row in selected_service)
    overall["offered_bytes"] = offered_bytes
    overall["offered_bytes_per_tti"] = offered_bytes / window.tti_count

    labels = sorted(set(classes.values()))
    traffic_classes = {
        label: _cohort_metrics(
            service_rows=service_rows,
            outcome_rows=outcome_rows,
            ue_ids={ue for ue, value in classes.items() if value == label},
            window=window,
        )
        for label in labels
    }
    per_ue = {
        str(ue): {
            "traffic_class": classes[ue],
            **_cohort_metrics(
                service_rows=service_rows,
                outcome_rows=outcome_rows,
                ue_ids={ue},
                window=window,
            ),
        }
        for ue in ue_ids
    }
    return {
        "profile": profile,
        "window": window.label,
        "overall": overall,
        "cell_radio_efficiency": cell_radio_efficiency_summary(
            service_rows=service_rows,
            window=window,
        ),
        "traffic_classes": traffic_classes,
        "per_ue": per_ue,
    }


def _numeric_cell_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    total = float(array.sum())
    squares = float(np.square(array).sum())
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "jain": 0.0 if squares <= 0.0 else total * total / (len(array) * squares),
    }


def cell_radio_efficiency_summary(
    *,
    service_rows: Sequence[Mapping[str, object]],
    window: WindowSpec,
    n_cell: int = 3,
    n_prg: int = 17,
    cell_bandwidth_hz: float = 97_920_000.0,
    slot_duration_s: float = 0.0005,
) -> dict[str, object]:
    if n_cell <= 0 or n_prg <= 0 or cell_bandwidth_hz <= 0.0 or slot_duration_s <= 0.0:
        raise ValueError("invalid cell radio-efficiency dimensions")
    selected = [
        row
        for row in service_rows
        if window.first_tti <= int(row["tti"]) <= window.last_tti
    ]
    observed_cells = {int(row["serving_cell"]) for row in selected}
    if observed_cells != set(range(n_cell)):
        raise ValueError("service trace does not cover the expected serving cells")
    duration_s = window.tti_count * slot_duration_s
    per_cell: dict[str, dict[str, float | int]] = {}
    for cell in range(n_cell):
        rows = [row for row in selected if int(row["serving_cell"]) == cell]
        successful_bytes = sum(int(row["successful_bytes"]) for row in rows)
        served_bytes = sum(int(row["scheduled_dequeue_bytes"]) for row in rows)
        active_prgs = sum(int(row["actual_prg_count"]) for row in rows)
        tb_tx = sum(int(row.get("tb_tx_count", 0)) for row in rows)
        tb_err = sum(int(row.get("tb_err_count", 0)) for row in rows)
        mcs_weighted = sum(
            int(row["mcs"]) * int(row.get("tb_tx_count", 0))
            for row in rows
            if int(row["mcs"]) >= 0 and int(row.get("tb_tx_count", 0)) > 0
        )
        layer_weighted = sum(
            int(row["layer"]) * int(row.get("tb_tx_count", 0))
            for row in rows
            if int(row["layer"]) >= 0 and int(row.get("tb_tx_count", 0)) > 0
        )
        goodput_bps = successful_bytes * 8.0 / duration_s
        served_bps = served_bytes * 8.0 / duration_s
        per_cell[str(cell)] = {
            "goodput_mbps": goodput_bps / 1.0e6,
            "served_mbps": served_bps / 1.0e6,
            "goodput_spectral_efficiency_bps_hz": goodput_bps / cell_bandwidth_hz,
            "served_spectral_efficiency_bps_hz": served_bps / cell_bandwidth_hz,
            "mcs_tx_weighted_mean": mcs_weighted / tb_tx if tb_tx else 0.0,
            "layer_tx_weighted_mean": layer_weighted / tb_tx if tb_tx else 0.0,
            "prg_utilization_ratio": active_prgs / (window.tti_count * n_prg),
            "successful_bytes_per_active_prg": (
                successful_bytes / active_prgs if active_prgs else 0.0
            ),
            "tb_bler": tb_err / tb_tx if tb_tx else 0.0,
            "tb_tx_count": tb_tx,
            "tb_err_count": tb_err,
        }
    across_cells = {
        metric: _numeric_cell_summary(
            [float(per_cell[str(cell)][metric]) for cell in range(n_cell)]
        )
        for metric in (
            "goodput_mbps",
            "served_mbps",
            "goodput_spectral_efficiency_bps_hz",
            "served_spectral_efficiency_bps_hz",
            "mcs_tx_weighted_mean",
            "layer_tx_weighted_mean",
            "prg_utilization_ratio",
            "successful_bytes_per_active_prg",
            "tb_bler",
        )
    }
    return {
        "cell_bandwidth_hz": cell_bandwidth_hz,
        "per_cell": per_cell,
        "across_cells": across_cells,
    }


def paired_seed_statistics(
    pairs: Sequence[tuple[int, float, float]],
) -> dict[str, object]:
    if not pairs:
        raise ValueError("paired seed values are empty")
    seeds = [int(seed) for seed, _, _ in pairs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("paired seed values contain duplicate seeds")
    treatment = np.asarray([value for _, value, _ in pairs], dtype=np.float64)
    control = np.asarray([value for _, _, value in pairs], dtype=np.float64)
    delta = treatment - control
    delta_std = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
    if len(delta) == 1:
        ci_low = ci_high = float(delta[0])
    else:
        critical = _T_CRITICAL_95.get(len(delta) - 1)
        if critical is None:
            raise ValueError("paired 95% CI supports only the frozen 3- or 10-seed design")
        margin = critical * delta_std / math.sqrt(len(delta))
        ci_low = float(delta.mean() - margin)
        ci_high = float(delta.mean() + margin)
    return {
        "seeds": seeds,
        "treatment_mean": float(treatment.mean()),
        "treatment_std": float(treatment.std(ddof=1)) if len(treatment) > 1 else 0.0,
        "treatment_median": float(np.median(treatment)),
        "control_mean": float(control.mean()),
        "control_std": float(control.std(ddof=1)) if len(control) > 1 else 0.0,
        "control_median": float(np.median(control)),
        "delta_mean": float(delta.mean()),
        "delta_std": delta_std,
        "delta_median": float(np.median(delta)),
        "delta_ci95_low": ci_low,
        "delta_ci95_high": ci_high,
        "wins": {
            "treatment": int(np.sum(delta > 0.0)),
            "tie": int(np.sum(delta == 0.0)),
            "control": int(np.sum(delta < 0.0)),
        },
        "per_seed": [
            {
                "seed": seed,
                "treatment": treatment_value,
                "control": control_value,
                "delta": treatment_value - control_value,
            }
            for seed, treatment_value, control_value in pairs
        ],
    }


def evaluate_stage1_stop(seed_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(seed_rows) != 3:
        raise ValueError("stage-1 stop gate requires exactly three seeds")
    expiry_seeds = [
        int(row["seed"])
        for row in seed_rows
        if float(row["dpa_mixed_expiry_drop_rate"]) > 0.0
    ]
    efficiency_drop_seeds = [
        int(row["seed"])
        for row in seed_rows
        if float(row["dpa_mixed_goodput_efficiency"])
        / float(row["dpa_homogeneous_goodput_efficiency"])
        - 1.0
        < -0.01
    ]
    loses_pfq_seeds = [
        int(row["seed"])
        for row in seed_rows
        if float(row["dpa_mixed_goodput_mbps"])
        < float(row["pfq_mixed_goodput_mbps"])
    ]
    backlog_pressure_seeds = [
        int(row["seed"])
        for row in seed_rows
        if float(row["dpa_large_backlog_pressure"]) > 0.05
    ]
    triggers = {
        "dpa_mixed_expiry_seed_ids": expiry_seeds,
        "paired_goodput_efficiency_drop_over_1pct_seed_ids": efficiency_drop_seeds,
        "paired_goodput_efficiency_drop_over_1pct_count": len(efficiency_drop_seeds),
        "dpa_mixed_loses_pfq_seed_ids": loses_pfq_seeds,
        "dpa_mixed_loses_pfq_count": len(loses_pfq_seeds),
        "dpa_large_backlog_pressure_over_5pct_seed_ids": backlog_pressure_seeds,
        "dpa_large_backlog_pressure_over_5pct_count": len(backlog_pressure_seeds),
    }
    expand = bool(expiry_seeds) or any(
        len(seeds) >= 2
        for seeds in (efficiency_drop_seeds, loses_pfq_seeds, backlog_pressure_seeds)
    )
    return {
        "expand_to_ten_seeds": expand,
        "triggers": triggers,
    }


_CAMPAIGN_METRIC_DIRECTIONS = {
    "goodput_mbps": "higher_better",
    "goodput_efficiency": "higher_better",
    "tb_bler": "lower_better",
    "expiry_drop_rate": "lower_better",
    "outcome_delay_mean_ms": "lower_better",
    "outcome_delay_p95_ms": "lower_better",
    "end_backlog_bytes": "lower_better",
    "max_backlogged_zero_dequeue_streak_tti": "lower_better",
    "goodput_spectral_efficiency_mean_bps_hz": "higher_better",
    "goodput_spectral_efficiency_min_bps_hz": "higher_better",
    "cell_goodput_jain": "higher_better",
    "mcs_tx_weighted_mean": "context",
    "layer_tx_weighted_mean": "context",
    "prg_utilization_mean": "context",
    "successful_bytes_per_active_prg_mean": "higher_better",
    "cell_tb_bler_mean": "lower_better",
}


def _campaign_scalar_metrics(summary: Mapping[str, object]) -> dict[str, float]:
    overall = summary["overall"]
    cells = summary["cell_radio_efficiency"]["across_cells"]
    offered_mbps = float(overall["offered_bytes_per_tti"]) * 0.016
    return {
        "goodput_mbps": float(overall["goodput_mbps"]),
        "goodput_efficiency": float(overall["goodput_mbps"]) / offered_mbps,
        "tb_bler": float(overall["tb_bler"]),
        "expiry_drop_rate": float(overall["expiry_drop_rate"]),
        "outcome_delay_mean_ms": float(overall["outcome_delay_mean_ms"]),
        "outcome_delay_p95_ms": float(overall["outcome_delay_p95_ms"]),
        "end_backlog_bytes": float(overall["end_backlog_bytes"]),
        "max_backlogged_zero_dequeue_streak_tti": float(
            overall["max_backlogged_zero_dequeue_streak_tti"]
        ),
        "goodput_spectral_efficiency_mean_bps_hz": float(
            cells["goodput_spectral_efficiency_bps_hz"]["mean"]
        ),
        "goodput_spectral_efficiency_min_bps_hz": float(
            cells["goodput_spectral_efficiency_bps_hz"]["min"]
        ),
        "cell_goodput_jain": float(cells["goodput_mbps"]["jain"]),
        "mcs_tx_weighted_mean": float(cells["mcs_tx_weighted_mean"]["mean"]),
        "layer_tx_weighted_mean": float(cells["layer_tx_weighted_mean"]["mean"]),
        "prg_utilization_mean": float(cells["prg_utilization_ratio"]["mean"]),
        "successful_bytes_per_active_prg_mean": float(
            cells["successful_bytes_per_active_prg"]["mean"]
        ),
        "cell_tb_bler_mean": float(cells["tb_bler"]["mean"]),
    }


def _directed_paired_statistics(
    pairs: Sequence[tuple[int, float, float]], direction: str
) -> dict[str, object]:
    stats = paired_seed_statistics(pairs)
    stats["direction"] = direction
    if direction == "lower_better":
        stats["wins"] = {
            "treatment": sum(treatment < control for _, treatment, control in pairs),
            "tie": sum(treatment == control for _, treatment, control in pairs),
            "control": sum(treatment > control for _, treatment, control in pairs),
        }
    elif direction == "context":
        stats["wins"] = None
    return stats


def aggregate_campaign_reports(
    seed_reports: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    seeds = sorted(int(seed) for seed in seed_reports)
    if len(seeds) not in (3, 10):
        raise ValueError("campaign aggregation requires the frozen 3- or 10-seed cohort")
    scalar_by_seed: dict[int, dict[str, dict[str, dict[str, float]]]] = {}
    for seed in seeds:
        profiles = seed_reports[seed]["profiles"]
        scalar_by_seed[seed] = {
            profile: {
                policy: _campaign_scalar_metrics(
                    profiles[profile]["policies"][policy][STEADY_WINDOW.label]
                )
                for policy in ("dpa_v1", "pfq", "rrq")
            }
            for profile in ("homogeneous", "mixed_granularity_v1")
        }

    paired: dict[str, object] = {"mixed_minus_homogeneous": {}}
    for policy in ("dpa_v1", "pfq", "rrq"):
        paired["mixed_minus_homogeneous"][policy] = {
            metric: _directed_paired_statistics(
                [
                    (
                        seed,
                        scalar_by_seed[seed]["mixed_granularity_v1"][policy][metric],
                        scalar_by_seed[seed]["homogeneous"][policy][metric],
                    )
                    for seed in seeds
                ],
                direction,
            )
            for metric, direction in _CAMPAIGN_METRIC_DIRECTIONS.items()
        }
    for baseline in ("pfq", "rrq"):
        label = f"mixed_dpa_minus_{baseline}"
        paired[label] = {
            metric: _directed_paired_statistics(
                [
                    (
                        seed,
                        scalar_by_seed[seed]["mixed_granularity_v1"]["dpa_v1"][metric],
                        scalar_by_seed[seed]["mixed_granularity_v1"][baseline][metric],
                    )
                    for seed in seeds
                ],
                direction,
            )
            for metric, direction in _CAMPAIGN_METRIC_DIRECTIONS.items()
        }

    stage1_rows = []
    for seed in seeds:
        profiles = seed_reports[seed]["profiles"]
        dpa_hom = profiles["homogeneous"]["policies"]["dpa_v1"][STEADY_WINDOW.label]
        dpa_mixed = profiles["mixed_granularity_v1"]["policies"]["dpa_v1"][
            STEADY_WINDOW.label
        ]
        pfq_mixed = profiles["mixed_granularity_v1"]["policies"]["pfq"][
            STEADY_WINDOW.label
        ]
        dpa_hom_scalar = scalar_by_seed[seed]["homogeneous"]["dpa_v1"]
        dpa_mixed_scalar = scalar_by_seed[seed]["mixed_granularity_v1"]["dpa_v1"]
        large = dpa_mixed["traffic_classes"]["low_frequency_large"]
        stage1_rows.append(
            {
                "seed": seed,
                "dpa_mixed_expiry_drop_rate": dpa_mixed_scalar["expiry_drop_rate"],
                "dpa_mixed_goodput_efficiency": dpa_mixed_scalar["goodput_efficiency"],
                "dpa_homogeneous_goodput_efficiency": dpa_hom_scalar[
                    "goodput_efficiency"
                ],
                "dpa_mixed_goodput_mbps": dpa_mixed_scalar["goodput_mbps"],
                "pfq_mixed_goodput_mbps": float(pfq_mixed["overall"]["goodput_mbps"]),
                "dpa_large_backlog_pressure": float(large["end_backlog_bytes"])
                / (float(large["offered_bytes_per_tti"]) * 400.0),
            }
        )
    return {
        "schema_version": 1,
        "seeds": seeds,
        "metric_directions": _CAMPAIGN_METRIC_DIRECTIONS,
        "per_seed_scalar_metrics": scalar_by_seed,
        "paired": paired,
        "stage1_stop": evaluate_stage1_stop(stage1_rows) if len(seeds) == 3 else None,
        "stage1_stop_inputs": stage1_rows,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_scenario_alignment(
    run_dirs: Mapping[str, Path],
) -> dict[str, object]:
    required = {"dpa_v1", "pfq", "rrq"}
    if set(run_dirs) != required:
        raise ValueError(f"policies must be exactly {sorted(required)}")
    offered_hashes = {
        policy: _sha256(Path(run_dir) / "traffic_offered_packets.csv")
        for policy, run_dir in run_dirs.items()
    }
    if len(set(offered_hashes.values())) != 1:
        raise ValueError("offered stream differs across policies")
    topology_hashes = {
        policy: _sha256(Path(run_dir) / "ue_topology.csv")
        for policy, run_dir in run_dirs.items()
    }
    if len(set(topology_hashes.values())) != 1:
        raise ValueError("topology differs across policies")
    return {
        "aligned": True,
        "offered_stream_sha256": next(iter(offered_hashes.values())),
        "topology_sha256": next(iter(topology_hashes.values())),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def aggregate_experiment(root: Path, sim_bin: Path) -> dict[str, object]:
    root = Path(root).resolve()
    policies = ("dpa_v1", "pfq", "rrq")
    profiles = ("homogeneous", "mixed_granularity_v1")
    simulator_hash = _sha256(Path(sim_bin).resolve())
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment": "fixed-load-packet-granularity-seed48-v1",
        "seed": 48,
        "slot_duration_ms": 0.5,
        "packet_ttl_ms": 200.0,
        "expected_offered_bytes_per_tti": 108000.0,
        "simulator_sha256": simulator_hash,
        "profiles": {},
    }
    profile_reports: dict[str, object] = {}
    for profile in profiles:
        run_dirs = {policy: root / profile / policy for policy in policies}
        alignment = validate_scenario_alignment(run_dirs)
        policy_reports: dict[str, object] = {}
        for policy, run_dir in run_dirs.items():
            if policy == "dpa_v1":
                manifest = json.loads((run_dir / "eval_summary.json").read_text(encoding="utf-8"))
                recorded_hash = str(manifest["sim_binary_sha256"])
                decision_path = run_dir / "eval_steps.jsonl"
            else:
                manifest = json.loads(
                    (run_dir / "reference_manifest.json").read_text(encoding="utf-8")
                )
                recorded_hash = str(manifest["binary_sha256"])
                decision_path = run_dir / "tti_trace.jsonl"
            if recorded_hash != simulator_hash:
                raise ValueError(f"{profile}/{policy} simulator hash mismatch")
            service = _read_csv(run_dir / "service_balance_trace.csv")
            outcomes = _read_csv(run_dir / "packet_outcome_samples.csv")
            decisions = _read_jsonl(decision_path)
            policy_reports[policy] = {
                window.label: build_granularity_summary(
                    service_rows=service,
                    outcome_rows=outcomes,
                    decision_rows=decisions,
                    profile=profile,
                    window=window,
                )
                for window in (FULL_WINDOW, STEADY_WINDOW)
            }
        profile_reports[profile] = {
            "alignment": alignment,
            "policies": policy_reports,
        }
    report["profiles"] = profile_reports
    return report


_CSV_METRICS = (
    "offered_bytes_per_tti",
    "goodput_mbps",
    "served_mbps",
    "tb_bler",
    "expiry_drop_rate",
    "outcome_delay_mean_ms",
    "outcome_delay_p95_ms",
    "ue_expiry_aware_rate_mean_mbps",
    "ue_expiry_aware_rate_p10_mbps",
    "end_backlog_bytes",
    "backlog_free_ue_ratio",
    "max_backlogged_zero_dequeue_streak_tti",
)


def write_report_files(root: Path, report: Mapping[str, object]) -> None:
    root = Path(root).resolve()
    (root / "granularity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    profiles = report["profiles"]
    headline_rows: list[dict[str, object]] = []
    per_ue_rows: list[dict[str, object]] = []
    for profile, profile_payload in profiles.items():
        for policy, windows in profile_payload["policies"].items():
            steady = windows[STEADY_WINDOW.label]
            cohorts = {"overall": steady["overall"], **steady["traffic_classes"]}
            for cohort, metrics in cohorts.items():
                headline_rows.append(
                    {
                        "profile": profile,
                        "policy": policy,
                        "cohort": cohort,
                        **{metric: metrics[metric] for metric in _CSV_METRICS},
                    }
                )
            for ue_id, metrics in steady["per_ue"].items():
                per_ue_rows.append(
                    {
                        "profile": profile,
                        "policy": policy,
                        "ue_id": ue_id,
                        "traffic_class": metrics["traffic_class"],
                        **{metric: metrics[metric] for metric in _CSV_METRICS},
                    }
                )
    for name, rows in (
        ("headline_steady.csv", headline_rows),
        ("per_ue_steady.csv", per_ue_rows),
    ):
        with (root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--sim-bin", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    report = aggregate_experiment(root, Path(args.sim_bin))
    write_report_files(root, report)


if __name__ == "__main__":
    main()

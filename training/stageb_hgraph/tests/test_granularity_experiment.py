from __future__ import annotations

from pathlib import Path

import pytest

from training.stageb_hgraph.dpa_v1_historical_comparison import WindowSpec
from training.stageb_hgraph.granularity_experiment import (
    aggregate_campaign_reports,
    build_granularity_summary,
    cell_radio_efficiency_summary,
    evaluate_stage1_stop,
    paired_seed_statistics,
    traffic_class_by_ue,
    validate_scenario_alignment,
)


def _service(tti: int, ue_id: int) -> dict[str, int]:
    successful = 1500 if ue_id % 2 == 0 else 12000
    return {
        "tti": tti,
        "ue_id": ue_id,
        "serving_cell": ue_id % 3,
        "offered_bytes": successful,
        "pre_queue_bytes": successful,
        "scheduled_dequeue_bytes": successful,
        "post_queue_bytes": 0,
        "successful_bytes": successful,
        "failed_bytes": 0,
        "expired_bytes": 0,
        "actual_prg_count": 1,
        "mcs": 0,
        "layer": 1,
        "tb_tx_count": 1,
        "tb_err_count": 0,
    }


def test_mixed_profile_is_interleaved_and_load_balanced() -> None:
    classes = traffic_class_by_ue("mixed_granularity_v1", 36)

    assert [ue for ue, label in classes.items() if label == "high_frequency_short"] == list(
        range(0, 36, 2)
    )
    assert [ue for ue, label in classes.items() if label == "low_frequency_large"] == list(
        range(1, 36, 2)
    )


def test_summary_contains_overall_class_and_per_ue_metrics() -> None:
    service = [_service(tti, ue) for tti in (1, 2) for ue in range(4)]
    outcomes = [
        {
            "terminal_tti": 1,
            "ue_id": ue,
            "outcome": "delivered",
            "packet_size_bytes": 1500 if ue % 2 == 0 else 12000,
            "system_time_ms": 1.0,
        }
        for ue in range(4)
    ]

    report = build_granularity_summary(
        service_rows=service,
        outcome_rows=outcomes,
        decision_rows=[{"tti": 1}, {"tti": 2}],
        profile="mixed_granularity_v1",
        window=WindowSpec("test", 1, 2),
    )

    assert report["overall"]["offered_bytes_per_tti"] == 27000.0
    assert report["traffic_classes"]["high_frequency_short"]["goodput_mbps"] == 48.0
    assert report["traffic_classes"]["low_frequency_large"]["goodput_mbps"] == 384.0
    assert report["per_ue"]["0"]["goodput_mbps"] == 24.0
    assert report["per_ue"]["1"]["goodput_mbps"] == 192.0
    assert report["per_ue"]["0"]["max_backlogged_zero_dequeue_streak_tti"] == 0


def test_alignment_requires_identical_stream_and_topology_for_all_policies(
    tmp_path: Path,
) -> None:
    run_dirs = {}
    for policy in ("dpa_v1", "pfq", "rrq"):
        run_dir = tmp_path / policy
        run_dir.mkdir()
        (run_dir / "traffic_offered_packets.csv").write_text("same\n", encoding="utf-8")
        (run_dir / "ue_topology.csv").write_text("same-topology\n", encoding="utf-8")
        run_dirs[policy] = run_dir

    report = validate_scenario_alignment(run_dirs)
    assert report["aligned"] is True

    (run_dirs["rrq"] / "traffic_offered_packets.csv").write_text(
        "different\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="offered stream"):
        validate_scenario_alignment(run_dirs)


def test_cell_radio_efficiency_uses_cell_bandwidth_and_tx_weighted_mcs() -> None:
    rows = [
        {
            **_service(1, 0),
            "serving_cell": 0,
            "successful_bytes": 100,
            "scheduled_dequeue_bytes": 120,
            "actual_prg_count": 1,
            "mcs": 10,
            "layer": 2,
            "tb_tx_count": 1,
            "tb_err_count": 0,
        },
        {
            **_service(2, 0),
            "serving_cell": 0,
            "successful_bytes": 50,
            "scheduled_dequeue_bytes": 100,
            "actual_prg_count": 1,
            "mcs": 20,
            "layer": 4,
            "tb_tx_count": 2,
            "tb_err_count": 1,
        },
        {
            **_service(1, 1),
            "serving_cell": 1,
            "successful_bytes": 200,
            "scheduled_dequeue_bytes": 200,
            "actual_prg_count": 2,
            "mcs": 5,
            "layer": 1,
            "tb_tx_count": 1,
            "tb_err_count": 0,
        },
        {
            **_service(2, 1),
            "serving_cell": 1,
            "successful_bytes": 0,
            "scheduled_dequeue_bytes": 0,
            "actual_prg_count": 0,
            "mcs": -1,
            "layer": -1,
            "tb_tx_count": 0,
            "tb_err_count": 0,
        },
    ]

    report = cell_radio_efficiency_summary(
        service_rows=rows,
        window=WindowSpec("test", 1, 2),
        n_cell=2,
        n_prg=2,
        cell_bandwidth_hz=100.0,
        slot_duration_s=1.0,
    )

    cell0 = report["per_cell"]["0"]
    assert cell0["goodput_spectral_efficiency_bps_hz"] == pytest.approx(6.0)
    assert cell0["served_spectral_efficiency_bps_hz"] == pytest.approx(8.8)
    assert cell0["mcs_tx_weighted_mean"] == pytest.approx(50.0 / 3.0)
    assert cell0["layer_tx_weighted_mean"] == pytest.approx(10.0 / 3.0)
    assert cell0["prg_utilization_ratio"] == 0.5
    assert cell0["successful_bytes_per_active_prg"] == 75.0
    assert cell0["tb_bler"] == pytest.approx(1.0 / 3.0)
    assert report["across_cells"]["goodput_mbps"]["jain"] == pytest.approx(0.98)


def test_paired_seed_statistics_use_paired_deltas_and_student_t_ci() -> None:
    report = paired_seed_statistics(
        [(101, 11.0, 10.0), (102, 14.0, 12.0), (103, 13.0, 10.0)]
    )

    assert report["treatment_mean"] == pytest.approx(38.0 / 3.0)
    assert report["treatment_std"] == pytest.approx((7.0 / 3.0) ** 0.5)
    assert report["treatment_median"] == 13.0
    assert report["control_mean"] == pytest.approx(32.0 / 3.0)
    assert report["control_std"] == pytest.approx(2.0 / 3.0**0.5)
    assert report["control_median"] == 10.0
    assert report["delta_mean"] == 2.0
    assert report["delta_std"] == 1.0
    assert report["delta_median"] == 2.0
    assert report["delta_ci95_low"] == pytest.approx(2.0 - 4.3026527299 / 3**0.5)
    assert report["delta_ci95_high"] == pytest.approx(2.0 + 4.3026527299 / 3**0.5)
    assert report["wins"] == {"treatment": 3, "tie": 0, "control": 0}


def test_stage1_stop_expands_only_when_a_frozen_trigger_fires() -> None:
    clean = [
        {
            "seed": seed,
            "dpa_mixed_expiry_drop_rate": 0.0,
            "dpa_mixed_goodput_efficiency": 0.90,
            "dpa_homogeneous_goodput_efficiency": 0.905,
            "dpa_mixed_goodput_mbps": 101.0,
            "pfq_mixed_goodput_mbps": 100.0,
            "dpa_large_backlog_pressure": 0.04,
        }
        for seed in (101, 102, 103)
    ]
    assert evaluate_stage1_stop(clean)["expand_to_ten_seeds"] is False

    triggered = [dict(row) for row in clean]
    triggered[0]["dpa_mixed_goodput_efficiency"] = 0.89
    triggered[1]["dpa_mixed_goodput_efficiency"] = 0.89
    report = evaluate_stage1_stop(triggered)
    assert report["expand_to_ten_seeds"] is True
    assert report["triggers"]["paired_goodput_efficiency_drop_over_1pct_count"] == 2


def _campaign_summary(goodput: float, *, low_backlog: float = 0.0) -> dict[str, object]:
    cell_metrics = {
        "goodput_mbps": {"mean": goodput / 3.0, "min": 30.0, "max": 40.0, "jain": 0.99},
        "goodput_spectral_efficiency_bps_hz": {
            "mean": 5.0,
            "min": 4.0,
            "max": 6.0,
            "jain": 0.98,
        },
        "mcs_tx_weighted_mean": {"mean": 10.0, "min": 8.0, "max": 12.0, "jain": 0.97},
        "layer_tx_weighted_mean": {"mean": 2.0, "min": 1.5, "max": 2.5, "jain": 0.96},
        "prg_utilization_ratio": {"mean": 0.8, "min": 0.7, "max": 0.9, "jain": 0.99},
        "successful_bytes_per_active_prg": {
            "mean": 2000.0,
            "min": 1800.0,
            "max": 2200.0,
            "jain": 0.99,
        },
        "tb_bler": {"mean": 0.1, "min": 0.08, "max": 0.12, "jain": 0.98},
    }
    return {
        "overall": {
            "offered_bytes_per_tti": 108000.0,
            "goodput_mbps": goodput,
            "tb_bler": 0.1,
            "expiry_drop_rate": 0.0,
            "outcome_delay_mean_ms": 2.0,
            "outcome_delay_p95_ms": 4.0,
            "end_backlog_bytes": 1000.0,
            "max_backlogged_zero_dequeue_streak_tti": 10.0,
        },
        "cell_radio_efficiency": {"across_cells": cell_metrics},
        "traffic_classes": {
            "low_frequency_large": {
                "offered_bytes_per_tti": 54000.0,
                "end_backlog_bytes": low_backlog,
            }
        },
    }


def test_campaign_aggregation_is_seed_paired_and_includes_cell_metrics() -> None:
    reports = {}
    for seed in (101, 102, 103):
        reports[seed] = {
            "profiles": {
                "homogeneous": {
                    "policies": {
                        policy: {"steady_3000": _campaign_summary(value)}
                        for policy, value in (("dpa_v1", 100.0), ("pfq", 99.0), ("rrq", 95.0))
                    }
                },
                "mixed_granularity_v1": {
                    "policies": {
                        policy: {
                            "steady_3000": _campaign_summary(
                                value,
                                low_backlog=1000.0 if policy == "dpa_v1" else 2000.0,
                            )
                        }
                        for policy, value in (("dpa_v1", 101.0), ("pfq", 99.0), ("rrq", 95.0))
                    }
                },
            }
        }

    result = aggregate_campaign_reports(reports)

    assert result["seeds"] == [101, 102, 103]
    goodput = result["paired"]["mixed_minus_homogeneous"]["dpa_v1"]["goodput_mbps"]
    assert goodput["delta_mean"] == 1.0
    assert goodput["wins"] == {"treatment": 3, "tie": 0, "control": 0}
    assert (
        result["paired"]["mixed_dpa_minus_pfq"]["goodput_spectral_efficiency_mean_bps_hz"][
            "delta_mean"
        ]
        == 0.0
    )
    assert result["stage1_stop"]["expand_to_ten_seeds"] is False

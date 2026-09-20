from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("policy", ["pf", "pfq", "rr", "rrq"])
def test_native_four_policy_cli_accepts_explicit_policy(monkeypatch, policy):
    from training.stageb_hgraph.pfq_reference_calibration import _parse_args
    monkeypatch.setattr("sys.argv", ["calibration", "--sim-bin", "/unused",
        "--out-dir", "/unused", "--reference-policy", policy])
    assert _parse_args().reference_policy == policy


def test_explicit_pressure_ttl_overrides_stale_tti_env():
    from training.stageb_hgraph import pfq_reference_calibration as calibration
    assert hasattr(calibration, "packet_ttl_env"), "pressure TTL must be explicitly supported"
    assert calibration.packet_ttl_env(12.5) == {
        "CUMAC_PACKET_TTL_MS": "12.5", "CUMAC_PACKET_TTL_TTI": "0"}
    with pytest.raises(ValueError):
        calibration.packet_ttl_env(0)

from training.stageb_hgraph.independent_objectives import (
    ExpiryAwareUeWindowTracker,
    compute_calibration_scales,
)
from training.stageb_hgraph.pfq_reference_calibration import (
    MEASURED_TERMINAL_TTI_EXCLUSIVE,
    build_service_balance_audit_env,
    build_reference_summary,
    filter_outcome_samples,
    pfq_schedule_structure_metrics,
    traffic_profile_metadata,
    validate_observed_scenario,
    validate_reference_manifest,
)
from training.stageb_hgraph.online_protocol import VERSION


def _trace_rows() -> list[dict[str, float | int]]:
    rows = []
    for tti in range(1, 5001):
        rows.append(
            {
                "tti": tti,
                "goodput_bytes": float(tti),
                "raw_delta_d": float(tti - 3000),
                "raw_delta_u": float(2 * (tti - 3000)),
            }
        )
    return rows


def test_warmup_is_excluded_and_absolute_p95_scales_are_shared() -> None:
    rows = _trace_rows()

    summary = build_reference_summary(rows, warmup_steps=1000, measure_steps=4000)
    scales = compute_calibration_scales(rows, warmup_steps=1000, measure_steps=4000)

    measured = np.arange(1001.0, 5001.0)
    assert summary["measured_row_count"] == 4000
    assert summary["measured_tti_first"] == 1001
    assert summary["measured_tti_last"] == 5000
    assert summary["goodput_bytes_mean"] == pytest.approx(float(measured.mean()))
    assert summary["full_row_count"] == 5000
    assert summary["full_goodput_bytes_mean"] == pytest.approx(float(np.arange(1.0, 5001.0).mean()))
    assert scales["delay_scale_p95"] == pytest.approx(
        float(np.percentile(np.abs(measured - 3000.0), 95))
    )
    assert scales["ue_utility_scale_p95"] == pytest.approx(
        float(np.percentile(np.abs(2.0 * (measured - 3000.0)), 95))
    )


def test_zero_scale_is_recorded_without_calibration_failure() -> None:
    rows = [
        {"tti": tti, "raw_delta_d": 0.0, "raw_delta_u": float(tti % 3)}
        for tti in range(1, 5001)
    ]

    scales = compute_calibration_scales(rows, warmup_steps=1000, measure_steps=4000)

    assert scales["delay_scale_p95"] == 0.0
    assert scales["ue_utility_scale_p95"] > 0.0


def test_expiry_aware_tracker_counts_expiry_time_but_zero_success_bits() -> None:
    tracker = ExpiryAwareUeWindowTracker(n_ue=2, window_steps=4)
    active = np.array([True, True])

    utility_before_expiry = tracker.update(
        active,
        successful_goodput_bytes=np.array([3000.0, 3000.0]),
        cumulative_delivered_time_ms=np.array([0.5, 0.5]),
        cumulative_expired_time_ms=np.array([0.0, 0.0]),
    )
    utility_after_expiry = tracker.update(
        active,
        successful_goodput_bytes=np.array([0.0, 3000.0]),
        cumulative_delivered_time_ms=np.array([0.5, 1.0]),
        cumulative_expired_time_ms=np.array([200.0, 0.0]),
    )

    assert utility_after_expiry < utility_before_expiry
    np.testing.assert_allclose(tracker.window_successful_goodput_bytes, [3000.0, 6000.0])
    np.testing.assert_allclose(tracker.window_outcome_system_time_ms, [200.5, 1.0])


def test_manifest_requires_binary_source_scenario_and_protocol() -> None:
    manifest = {
        "binary_sha256": "a" * 64,
        "source_revision": "deadbeef",
        "protocol_version": VERSION,
        "scenario": {
            "name": "coop_boundary",
            "topology_seed": 41,
            "traffic_seed": 41,
            "n_cell": 3,
            "n_ue": 36,
            "n_prg": 17,
            "ue_placement": "coop_boundary",
            "cell_association": "strongest",
            "cell_radius_m": 500.0,
            "prbs_per_prg": 16,
            "allocation_type": 0,
            "precoding": "svd",
            "packet_size_bytes": 3000,
            "packets_per_ue_tti": 1.0,
            "packet_ttl_ms": 200.0,
        },
    }

    validate_reference_manifest(manifest)
    for key in ("binary_sha256", "source_revision", "protocol_version", "scenario"):
        broken = dict(manifest)
        broken.pop(key)
        with pytest.raises(ValueError, match=key):
            validate_reference_manifest(broken)
    symbolic = dict(manifest, source_revision="ref: refs/heads/research/example")
    with pytest.raises(ValueError, match="source_revision"):
        validate_reference_manifest(symbolic)


def test_observed_scenario_dimensions_are_not_assumed() -> None:
    validate_observed_scenario({"n_cell": 3, "n_active_ue": 36, "n_prg": 17})
    with pytest.raises(RuntimeError, match="canonical scenario mismatch"):
        validate_observed_scenario({"n_cell": 3, "n_active_ue": 35, "n_prg": 17})


def test_outcome_window_excludes_terminal_1000_and_includes_1001(tmp_path: Path) -> None:
    path = tmp_path / "packet_outcome_samples.csv"
    path.write_text(
        "ue_id,packet_id,terminal_tti,outcome,system_time_ms\n"
        "0,1,1000,expired,200.0\n"
        "0,2,1001,expired,200.0\n"
        "1,3,5000,delivered,1.0\n"
        "1,4,5001,delivered,1.0\n",
        encoding="utf-8",
    )

    rows = filter_outcome_samples(
        path,
        terminal_tti_exclusive=MEASURED_TERMINAL_TTI_EXCLUSIVE,
        terminal_tti_inclusive=5000,
    )

    assert [row["terminal_tti"] for row in rows] == [1001, 5000]


def test_pfq_schedule_structure_metrics_record_opportunity_without_gating() -> None:
    action_prg = np.asarray(
        [0, 0, 0, 0, 1, 2, 0, -1, -1, -1, -1, -1],
        dtype=np.int16,
    )
    reward_terms = np.zeros((23,), dtype=np.float64)
    reward_terms[6] = 7.0 / 12.0
    reward_terms[8] = 0.25

    metrics = pfq_schedule_structure_metrics(
        action_prg,
        {"reward_terms": reward_terms, "n_cell": 3, "n_prg": 4},
    )

    assert metrics == {
        "blank_ratio": pytest.approx(5.0 / 12.0),
        "prg_opportunity_count": 7.0,
        "prg_utilization_ratio": pytest.approx(7.0 / 12.0),
        "prg_reuse_ratio": pytest.approx(0.25),
        "local_prg_00_opportunity_count": 3.0,
        "local_prg_01_opportunity_count": 3.0,
        "local_prg_02_opportunity_count": 1.0,
        "local_prg_03_opportunity_count": 0.0,
    }


def test_service_balance_audit_env_is_explicit_and_frozen(tmp_path: Path) -> None:
    env = build_service_balance_audit_env(tmp_path, trace_first_tti=602)

    assert env == {
        "CUMAC_ONLINE_EXPORT_PHY_TBS_AUDIT": "1",
        "CUMAC_SERVICE_BALANCE_TRACE_CSV": str(tmp_path / "service_balance_trace.csv"),
        "CUMAC_SERVICE_BALANCE_TRACE_FIRST_TTI": "602",
        "CUMAC_TRAFFIC_OFFERED_SAMPLES_CSV": str(tmp_path / "traffic_offered_packets.csv"),
    }
    with pytest.raises(ValueError, match="positive"):
        build_service_balance_audit_env(tmp_path, trace_first_tti=0)


def test_mixed_granularity_metadata_preserves_expected_load() -> None:
    homogeneous = traffic_profile_metadata("homogeneous")
    mixed = traffic_profile_metadata("mixed_granularity_v1")

    assert homogeneous["expected_offered_bytes_per_tti"] == 108000.0
    assert mixed["expected_offered_bytes_per_tti"] == 108000.0
    assert mixed["flow_classes"] == {
        "high_frequency_short": {
            "ue_ids": list(range(0, 36, 2)),
            "packet_size_bytes": 1500,
            "arrival_rate_packets_per_tti": 2.0,
        },
        "low_frequency_large": {
            "ue_ids": list(range(1, 36, 2)),
            "packet_size_bytes": 12000,
            "arrival_rate_packets_per_tti": 0.25,
        },
    }

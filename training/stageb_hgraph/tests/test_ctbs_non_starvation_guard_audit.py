from __future__ import annotations

import csv
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import pytest

from training.stageb_hgraph import ctbs_non_starvation_guard_audit as audit_module
from training.stageb_hgraph.online_protocol import VERSION
from training.stageb_hgraph.ctbs_non_starvation_guard_audit import (
    ACTOR_STATE_DICT_SHA256,
    CHECKPOINT_FILE_SHA256,
    CTBS_CONFIG_SHA256,
    FEATURE_SOURCE_SHA256,
    FROZEN_OFFERED_FINGERPRINT,
    GUARD_TRACE_FIELDS,
    IMPLEMENTATION_SOURCE_PATHS,
    SIMULATOR_SHA256,
    UE_FEATURE_NAMES,
    audit_guard_trace,
    build_phase_a_report,
    load_guard_trace,
    main,
    summarize_outcome_delays,
)


REFERENCE_ARTIFACT_ROLES = (
    "pfq_metrics",
    "pfq_starvation",
    "pfq_outcomes",
    "pfq_offered_packets",
    "unguarded_starvation",
)
MANDATORY_GUARD_CHECKS = (
    "complete_decision_coverage",
    "unique_decision_coordinates",
    "canonical_decision_coordinates",
    "measured_decision_coverage",
    "nonblank_override_count_zero",
    "max_one_override_per_cell_tti",
    "pressure_strictly_above_log2",
    "adequacy_strictly_below_one",
    "age_at_least_quarter_ttl",
    "override_candidate_valid",
    "positive_ctbs_capacity_delta",
    "reason_codes_known",
    "unchanged_without_override",
    "override_executes_nonblank",
    "override_reason_consistent",
)


def _guard_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for tti in (1, 2):
        for cell in range(3):
            for local_prg in range(2):
                rows.append(
                    {
                        "tti": tti,
                        "cell_index": cell,
                        "local_prg_index": local_prg,
                        "blank_class_index": 2,
                        "actor_choice": 0,
                        "executed_choice": 0,
                        "guard_override": False,
                        "selected_class": -1,
                        "selected_ue": -1,
                        "candidate_was_valid": False,
                        "log1p_drain_pressure": 0.0,
                        "scheduled_dequeue_adequacy": 0.0,
                        "ttl_normalized_service_age": 0.0,
                        "ctbs_capacity_before_bytes": 0,
                        "ctbs_capacity_after_bytes": 0,
                        "reason": "actor_nonblank",
                    }
                )
    rows[0].update(
        actor_choice=2,
        executed_choice=0,
        guard_override=True,
        selected_class=0,
        selected_ue=4,
        candidate_was_valid=True,
        log1p_drain_pressure=math.log(2.0) + 0.01,
        scheduled_dequeue_adequacy=0.99,
        ttl_normalized_service_age=0.25,
        ctbs_capacity_before_bytes=100,
        ctbs_capacity_after_bytes=101,
        reason="overridden",
    )
    return rows


def _audit(rows: list[dict[str, object]]) -> dict[str, object]:
    return audit_guard_trace(
        rows,
        measured_tti_first=1,
        measured_tti_last=2,
        expected_tti_first=1,
        expected_tti_last=2,
        expected_n_cell=3,
        expected_n_prg=2,
    )


def test_guard_trace_audit_accepts_one_exact_legal_override() -> None:
    audit = _audit(_guard_rows())

    assert audit["row_count"] == 12
    assert audit["override_count"] == 1
    assert audit["measured_override_count"] == 1
    assert audit["override_count_by_cell"] == {"0": 1}
    assert audit["override_count_by_ue"] == {"4": 1}
    assert all(audit["checks"].values())


@pytest.mark.parametrize(
    ("check", "mutate"),
    [
        ("nonblank_override_count_zero", lambda rows: rows[0].update(actor_choice=0)),
        (
            "max_one_override_per_cell_tti",
            lambda rows: rows[1].update(
                actor_choice=2,
                executed_choice=0,
                guard_override=True,
                selected_class=0,
                selected_ue=4,
                candidate_was_valid=True,
                log1p_drain_pressure=1.0,
                scheduled_dequeue_adequacy=0.5,
                ttl_normalized_service_age=0.5,
                ctbs_capacity_before_bytes=100,
                ctbs_capacity_after_bytes=101,
                reason="overridden",
            ),
        ),
        (
            "pressure_strictly_above_log2",
            lambda rows: rows[0].update(log1p_drain_pressure=math.log(2.0)),
        ),
        (
            "adequacy_strictly_below_one",
            lambda rows: rows[0].update(scheduled_dequeue_adequacy=1.0),
        ),
        (
            "age_at_least_quarter_ttl",
            lambda rows: rows[0].update(ttl_normalized_service_age=0.249999),
        ),
        (
            "override_candidate_valid",
            lambda rows: rows[0].update(candidate_was_valid=False),
        ),
        (
            "positive_ctbs_capacity_delta",
            lambda rows: rows[0].update(ctbs_capacity_after_bytes=100),
        ),
    ],
)
def test_guard_trace_audit_exposes_each_illegal_override(
    check: str, mutate
) -> None:
    rows = _guard_rows()
    mutate(rows)

    assert _audit(rows)["checks"][check] is False


def _write_guard_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GUARD_TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_load_guard_trace_requires_exact_schema_types_and_full_coverage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "guard.csv"
    _write_guard_csv(path, _guard_rows())

    loaded = load_guard_trace(
        path,
        expected_tti_first=1,
        expected_tti_last=2,
        expected_n_cell=3,
        expected_n_prg=2,
    )
    expected = sorted(
        _guard_rows(),
        key=lambda row: (row["tti"], row["local_prg_index"], row["cell_index"]),
    )
    assert loaded == expected

    malformed = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    malformed[0]["ctbs_capacity_before_bytes"] = "100.0"
    _write_guard_csv(path, malformed)
    with pytest.raises(ValueError, match="ctbs_capacity_before_bytes"):
        load_guard_trace(
            path,
            expected_tti_first=1,
            expected_tti_last=2,
            expected_n_cell=3,
            expected_n_prg=2,
        )


def test_load_guard_trace_rejects_duplicate_and_noncanonical_coordinates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "guard.csv"
    rows = _guard_rows()
    rows[-1] = deepcopy(rows[0])
    _write_guard_csv(path, rows)

    with pytest.raises(ValueError, match="duplicate"):
        load_guard_trace(
            path,
            expected_tti_first=1,
            expected_tti_last=2,
            expected_n_cell=3,
            expected_n_prg=2,
        )


def test_summarize_outcome_delays_filters_the_measured_window_and_reports_tail(
    tmp_path: Path,
) -> None:
    path = tmp_path / "outcomes.csv"
    path.write_text(
        "terminal_tti,system_time_ms\n1,1\n2,2\n3,3\n4,4\n5,100\n",
        encoding="utf-8",
    )

    report = summarize_outcome_delays(path, 2, 5)

    assert report["sample_count"] == 4
    assert report["mean_ms"] == pytest.approx(27.25)
    assert report["p50_ms"] == pytest.approx(3.5)
    assert report["p90_ms"] == pytest.approx(71.2)
    assert report["p95_ms"] == pytest.approx(85.6)
    assert report["max_ms"] == 100.0


@pytest.mark.parametrize(
    "contents,match",
    [
        ("terminal_tti,system_time_ms\n0,1\n", "terminal_tti"),
        ("terminal_tti,system_time_ms\n1,nan\n", "system_time_ms"),
        ("terminal_tti,system_time_ms\n1,-0.5\n", "non-negative"),
        ("terminal_tti,system_time_ms\n1,1\n", "empty measured"),
    ],
)
def test_summarize_outcome_delays_fails_closed(
    tmp_path: Path, contents: str, match: str
) -> None:
    path = tmp_path / "outcomes.csv"
    path.write_text(contents, encoding="utf-8")
    first, last = (2, 2) if "empty" in match else (1, 1)
    with pytest.raises(ValueError, match=match):
        summarize_outcome_delays(path, first, last)


def _metrics(**updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "goodput_mbps": 100.0,
        "tb_bler": 0.1,
        "expiry_rate": 0.01,
        "outcome_delay_p95_ms": 20.0,
        "ue_macro_ea_rate_mbps": 4.0,
        "ue_p10_ea_rate_mbps": 1.0,
        "action_replay_mismatch_count": 0.0,
        "illegal_action_count": 0.0,
        "independent_deletion_equal": True,
        "ea_metrics_available": True,
        "protocol_version": VERSION,
        "measured_tti_first": 1001,
        "measured_tti_last": 5000,
        "prg_utilization_ratio_mean": 0.8,
        "blank_ratio_mean": 0.2,
        "scheduled_bytes_mean": 1000.0,
        "failed_bytes_mean": 10.0,
    }
    values.update(updates)
    return values


def _actor_summary(**updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "protocol_version": VERSION,
        "completed_steps": 5000,
        "stats_count": 4000,
        "measured_tti_first": 1001,
        "measured_tti_last": 5000,
        "association_candidate_counts": [8, 17, 11],
        "topology_candidate_width": 17,
        "runtime_candidate_width": 17,
        "candidate_full_coverage": True,
        "candidate_truncation_count": 0,
        "stats_sum_candidate_truncation_count": 0.0,
        "stats_sum_action_replay_mismatch_count": 0.0,
        "stats_sum_illegal_action_count": 0.0,
        "independent_deletion_equal": True,
        "actor_state_dict_sha256": ACTOR_STATE_DICT_SHA256,
        "checkpoint_sha256": CHECKPOINT_FILE_SHA256,
        "sim_binary_sha256": SIMULATOR_SHA256,
        "runtime_harq_enabled": 0,
        "independent_headroom_accounting": "ctbs-v1",
        "ctbs_pre_action_handshake_enabled": True,
        "ctbs_config_sha256": CTBS_CONFIG_SHA256,
        "ctbs_snapshot_count": 5000,
        "ctbs_snapshot_id_first": 1,
        "ctbs_snapshot_id_last": 5000,
        "ctbs_mcs_mismatch_count": 0,
        "ctbs_layer_mismatch_count": 0,
        "ctbs_tbs_mismatch_count": 0,
        "ctbs_reference_label_deletion_equal": True,
        "ctbs_mask_reactivation_count": 0,
        "ctbs_mask_trigger_count_by_local_prg": [0] * 17,
        "ctbs_selected_count_by_local_prg": [1] * 17,
        "ctbs_valid_candidate_count_by_local_prg": [1] * 17,
        "decode_mode": "argmax",
        "independent_non_starvation_guard": "ctbs-blank-v1",
        "non_starvation_guard_config": {
            "minimum_log1p_drain_pressure": math.log(2.0),
            "maximum_scheduled_dequeue_adequacy": 1.0,
            "minimum_ttl_normalized_service_age": 0.25,
            "maximum_overrides_per_cell_per_tti": 1,
            "override_nonblank": False,
            "require_positive_ctbs_capacity_delta": True,
        },
        "candidate_policy_mode": "independent",
        "slot_repair_max_per_cell": 0,
        "safe_fill_max_per_cell": 0,
        "candidate_blank_budget_max_per_cell": -1,
        "candidate_budgeted_select_mode": "off",
        "actor_config": {"ue_feat_dim": 16},
        "stats_mean_prg_utilization_ratio": 0.75,
        "stats_mean_blank_ratio": 0.25,
        "stats_sum_scheduled_bytes": 12345.0,
        "stats_sum_failed_bytes": 123.0,
        "ue_macro_expiry_aware_rate_mbps": 3.5,
        "ue_p10_expiry_aware_rate_mbps": 0.8,
        "ue_min_expiry_aware_rate_mbps": 0.2,
    }
    values.update(updates)
    return values


def _ue(
    ue_id: int,
    *,
    scheduled: int,
    rate: float,
    expired: int,
    streak: int,
) -> dict[str, object]:
    return {
        "ue_id": ue_id,
        "backlogged_tti_count": 4000,
        "scheduled_dequeue_bytes": scheduled,
        "expiry_aware_rate_mbps": rate,
        "expired_bytes": expired,
        "failed_bytes": 10,
        "max_backlogged_zero_dequeue_streak_tti": streak,
    }


def _starvation(*, guarded: bool) -> dict[str, object]:
    per_ue = [
        _ue(ue_id, scheduled=500, rate=0.5, expired=30, streak=20)
        for ue_id in range(36)
    ]
    per_ue[4] = _ue(
        4,
        scheduled=200 if guarded else 100,
        rate=0.2 if guarded else 0.1,
        expired=80 if guarded else 100,
        streak=100 if guarded else 307,
    )
    per_ue[6] = _ue(
        6,
        scheduled=300 if guarded else 200,
        rate=0.3 if guarded else 0.2,
        expired=70 if guarded else 90,
        streak=90 if guarded else 272,
    )
    serving_cell_by_ue = {
        str(ue_id): 0 if ue_id < 8 else (1 if ue_id < 25 else 2)
        for ue_id in range(36)
    }
    return {
        "protocol_id": "dimensionless-service-starvation-audit-v2",
        "seed": 51,
        "per_ue": per_ue,
        "streak_threshold_tti": 400,
        "trace_coverage": {
            "row_count": 144000,
            "tti_first": 1001,
            "tti_last": 5000,
            "ue_count": 36,
            "association_counts": [8, 17, 11],
            "serving_cell_by_ue": serving_cell_by_ue,
        },
    }


def _manifest() -> dict[str, object]:
    return {
        "protocol_id": "iter8-ctbs-legal-non-starvation-guard-seed51-v1",
        "seed": 51,
        "warmup_tti": 1000,
        "measured_tti": 4000,
        "measured_tti_first": 1001,
        "measured_tti_last": 5000,
        "packet_ttl_tti": 400,
        "packet_ttl_ms": 200,
        "decode_mode": "argmax",
        "independent_headroom_accounting": "ctbs-v1",
        "guard_mode": "ctbs-blank-v1",
        "runtime_candidate_width": 17,
        "association_candidate_counts": [8, 17, 11],
        "training_or_adaptation_allowed": False,
        "additional_seed_allowed": False,
        "checkpoint_file_sha256": CHECKPOINT_FILE_SHA256,
        "actor_state_dict_sha256": ACTOR_STATE_DICT_SHA256,
        "simulator_sha256": SIMULATOR_SHA256,
        "ctbs_config_sha256": CTBS_CONFIG_SHA256,
        "performance_thresholds": {
            "actor_over_pfq_goodput_min": 0.99,
            "actor_minus_pfq_bler_max": 0.002,
            "actor_minus_pfq_expiry_max": 0.0005,
        },
        "feature_source_sha256": dict(FEATURE_SOURCE_SHA256),
        "implementation_source_sha256": {
            path: "d" * 64 for path in IMPLEMENTATION_SOURCE_PATHS
        },
        "ue_feature_names": list(UE_FEATURE_NAMES),
        "ue_feature_dimension": 16,
        "candidate_policy_mode": "independent",
        "slot_repair_max_per_cell": 0,
        "safe_fill_max_per_cell": 0,
        "candidate_blank_budget_max_per_cell": -1,
        "candidate_budgeted_select_mode": "off",
        "guard_config": {
            "minimum_log1p_drain_pressure": math.log(2.0),
            "maximum_scheduled_dequeue_adequacy": 1.0,
            "minimum_ttl_normalized_service_age": 0.25,
            "maximum_overrides_per_cell_per_tti": 1,
            "override_nonblank": False,
            "require_positive_ctbs_capacity_delta": True,
        },
        "reference_artifact_paths": {
            role: f"references/{role}.artifact" for role in REFERENCE_ARTIFACT_ROLES
        },
        "reference_artifact_sha256": {
            role: str(index + 1) * 64
            for index, role in enumerate(REFERENCE_ARTIFACT_ROLES)
        },
    }


def _guard_audit() -> dict[str, object]:
    return {
        "row_count": 255000,
        "measured_row_count": 204000,
        "override_count": 1,
        "measured_override_count": 1,
        "cell_budget_used_count": 0,
        "measured_cell_budget_used_count": 0,
        "no_positive_ctbs_headroom_count": 0,
        "measured_no_positive_ctbs_headroom_count": 0,
        "reason_counts": {"overridden": 1},
        "measured_reason_counts": {"overridden": 1},
        "checks": {key: True for key in MANDATORY_GUARD_CHECKS},
        "valid": True,
        "passed": True,
    }


def _offered() -> dict[str, object]:
    return {
        "exact": True,
        "frozen_expected": dict(FROZEN_OFFERED_FINGERPRINT),
        "pfq": dict(FROZEN_OFFERED_FINGERPRINT),
        "actor": dict(FROZEN_OFFERED_FINGERPRINT),
    }


def _phase_inputs() -> dict[str, object]:
    manifest = _manifest()
    observed = {
        "checkpoint": CHECKPOINT_FILE_SHA256,
        "actor_state": ACTOR_STATE_DICT_SHA256,
        "simulator": SIMULATOR_SHA256,
        **FEATURE_SOURCE_SHA256,
        **{path: "d" * 64 for path in IMPLEMENTATION_SOURCE_PATHS},
    }
    for role in REFERENCE_ARTIFACT_ROLES:
        relative = manifest["reference_artifact_paths"][role]
        canonical = str(Path("/worktree", relative))
        observed[f"reference_manifest_relative_path:{role}"] = relative
        observed[f"reference_manifest_path:{role}"] = canonical
        observed[f"reference_actual_path:{role}"] = canonical
        observed[f"reference_sha256:{role}"] = manifest[
            "reference_artifact_sha256"
        ][role]
    return {
        "frozen_manifest": manifest,
        "pfq_metrics": _metrics(),
        "actor_metrics": _metrics(
            goodput_mbps=99.0, tb_bler=0.102, expiry_rate=0.0105
        ),
        "actor_summary": _actor_summary(),
        "guard_audit": _guard_audit(),
        "offered_stream_comparison": _offered(),
        "pfq_starvation": _starvation(guarded=True),
        "unguarded_starvation": _starvation(guarded=False),
        "actor_starvation": _starvation(guarded=True),
        "pfq_outcome_delays": {
            "sample_count": 4,
            "mean_ms": 20.0,
            "p50_ms": 19.0,
            "p90_ms": 30.0,
            "p95_ms": 35.0,
            "max_ms": 40.0,
        },
        "actor_outcome_delays": {
            "sample_count": 4,
            "mean_ms": 18.0,
            "p50_ms": 17.0,
            "p90_ms": 25.0,
            "p95_ms": 30.0,
            "max_ms": 35.0,
        },
        "observed_file_sha256": observed,
    }


def test_phase_a_report_passes_all_four_gates_and_records_target_directions() -> None:
    report = build_phase_a_report(**_phase_inputs())

    assert report["passed"] is True
    assert report["classification"] == "ctbs_blank_v1_seed51_pass"
    assert report["provenance_or_invariants_gate"]["passed"] is True
    assert report["business_gate"]["passed"] is True
    assert report["target_ue_directional_gate"]["passed"] is True
    assert report["all_backlogged_ue_gate"]["passed"] is True
    for ue_id in ("4", "6"):
        assert all(report["target_ue_comparison"][ue_id]["checks"].values())


def test_phase_a_rejects_a_partial_all_true_guard_audit() -> None:
    values = _phase_inputs()
    values["guard_audit"].update(valid=True, passed=True, checks={"legal": True})

    report = build_phase_a_report(**values)

    assert report["classification"] == "invalid_experiment"
    assert report["provenance_or_invariants_gate"]["guard_invariants_pass"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda values: values["observed_file_sha256"].update(actor_state="bad"),
        lambda values: values["frozen_manifest"]["performance_thresholds"].update(
            actor_minus_pfq_expiry_max=0.5
        ),
    ],
)
def test_phase_a_provenance_rejects_recomputed_state_or_threshold_drift(mutate) -> None:
    values = _phase_inputs()
    mutate(values)

    report = build_phase_a_report(**values)

    assert report["classification"] == "invalid_experiment"


def test_phase_a_provenance_rejects_an_omitted_backlogged_ue() -> None:
    values = _phase_inputs()
    values["actor_starvation"]["per_ue"].pop(10)

    report = build_phase_a_report(**values)

    assert report["classification"] == "invalid_experiment"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda values: values["observed_file_sha256"].update(
            **{"reference_sha256:pfq_metrics": "f" * 64}
        ),
        lambda values: values["observed_file_sha256"].update(
            **{"reference_actual_path:pfq_outcomes": "/worktree/stale/outcomes.csv"}
        ),
        lambda values: values["frozen_manifest"]["reference_artifact_paths"].update(
            pfq_metrics="references/stale-pfq-metrics.artifact"
        ),
        lambda values: values["pfq_starvation"].update(seed=50),
        lambda values: values["unguarded_starvation"].update(protocol_id="stale-v1"),
    ],
)
def test_phase_a_rejects_stale_or_conflicting_reference_artifacts(mutate) -> None:
    values = _phase_inputs()
    mutate(values)

    report = build_phase_a_report(**values)

    assert report["classification"] == "invalid_experiment"


@pytest.mark.parametrize(
    "path",
    [
        "/worktree/references/pfq_metrics.artifact",
        "references/../pfq_metrics.artifact",
        "references/./pfq_metrics.artifact",
    ],
)
def test_phase_a_rejects_noncanonical_reference_manifest_paths(path: str) -> None:
    values = _phase_inputs()
    values["frozen_manifest"]["reference_artifact_paths"]["pfq_metrics"] = path
    values["observed_file_sha256"][
        "reference_manifest_relative_path:pfq_metrics"
    ] = path

    report = build_phase_a_report(**values)

    assert report["classification"] == "invalid_experiment"


def test_phase_a_rejects_offered_stream_mismatch() -> None:
    values = _phase_inputs()
    values["offered_stream_comparison"]["exact"] = False

    report = build_phase_a_report(**values)

    assert report["classification"] == "invalid_experiment"


def test_non_target_backlogged_ue_failure_is_a_mechanism_failure() -> None:
    values = _phase_inputs()
    values["actor_starvation"]["per_ue"][10].update(
        scheduled_dequeue_bytes=0,
        expiry_aware_rate_mbps=0.0,
        max_backlogged_zero_dequeue_streak_tti=400,
    )

    report = build_phase_a_report(**values)

    assert report["target_ue_directional_gate"]["passed"] is True
    assert report["all_backlogged_ue_gate"]["passed"] is False
    assert report["classification"] == "insufficient_blank_opportunities"


@pytest.mark.parametrize(
    ("field", "value"),
    [("tb_bler", 0.103), ("expiry_rate", 0.0106)],
)
def test_bler_and_expiry_business_failures_keep_business_precedence(
    field: str, value: float
) -> None:
    values = _phase_inputs()
    values["actor_metrics"][field] = value

    report = build_phase_a_report(**values)

    assert report["provenance_or_invariants_gate"]["passed"] is True
    assert report["business_gate"]["passed"] is False
    assert report["classification"] == "business_metric_cost"


def test_metric_table_retains_exact_pfq_actor_values_and_legitimate_zero() -> None:
    values = _phase_inputs()
    values["actor_summary"].update(
        ue_macro_expiry_aware_rate_mbps=0.0,
        ue_p10_expiry_aware_rate_mbps=0.0,
        ue_min_expiry_aware_rate_mbps=0.0,
    )

    report = build_phase_a_report(**values)

    assert report["metrics"]["pfq"] == {
        "outcome_delay_ms": {
            "mean_ms": 20.0,
            "p50_ms": 19.0,
            "p90_ms": 30.0,
            "p95_ms": 35.0,
            "max_ms": 40.0,
        },
        "ue_macro_expiry_aware_rate_mbps": 4.0,
        "ue_p10_expiry_aware_rate_mbps": 1.0,
        "ue_min_expiry_aware_rate_mbps": 0.2,
        "goodput_mbps": 100.0,
        "tb_bler": 0.1,
        "expiry_rate": 0.01,
        "prg_utilization_ratio": 0.8,
        "blank_ratio": 0.2,
        "scheduled_dequeue_bytes": 17500,
        "failed_bytes": 360,
        "per_ue_expired_bytes": {
            **{str(ue): 30 for ue in range(36)},
            "4": 80,
            "6": 70,
        },
    }
    assert report["metrics"]["actor"] == {
        "outcome_delay_ms": {
            "mean_ms": 18.0,
            "p50_ms": 17.0,
            "p90_ms": 25.0,
            "p95_ms": 30.0,
            "max_ms": 35.0,
        },
        "ue_macro_expiry_aware_rate_mbps": 0.0,
        "ue_p10_expiry_aware_rate_mbps": 0.0,
        "ue_min_expiry_aware_rate_mbps": 0.0,
        "goodput_mbps": 99.0,
        "tb_bler": 0.102,
        "expiry_rate": 0.0105,
        "prg_utilization_ratio": 0.75,
        "blank_ratio": 0.25,
        "scheduled_dequeue_bytes": 17500,
        "failed_bytes": 360,
        "per_ue_expired_bytes": {
            **{str(ue): 30 for ue in range(36)},
            "4": 80,
            "6": 70,
        },
    }


@pytest.mark.parametrize(
    ("family", "field", "value", "policy", "reported_field"),
    [
        (
            "pfq_metrics",
            "ue_macro_ea_rate_mbps",
            float("nan"),
            "pfq",
            "ue_macro_expiry_aware_rate_mbps",
        ),
        (
            "pfq_metrics",
            "ue_p10_ea_rate_mbps",
            "malformed",
            "pfq",
            "ue_p10_expiry_aware_rate_mbps",
        ),
        (
            "actor_summary",
            "ue_macro_expiry_aware_rate_mbps",
            float("inf"),
            "actor",
            "ue_macro_expiry_aware_rate_mbps",
        ),
        (
            "actor_summary",
            "ue_p10_expiry_aware_rate_mbps",
            "malformed",
            "actor",
            "ue_p10_expiry_aware_rate_mbps",
        ),
    ],
)
def test_supplied_invalid_macro_or_p10_is_invalid_not_derived(
    family: str,
    field: str,
    value: object,
    policy: str,
    reported_field: str,
) -> None:
    values = _phase_inputs()
    values[family][field] = value

    report = build_phase_a_report(**values)

    assert report["classification"] == "invalid_experiment"
    assert report["provenance_or_invariants_gate"]["provenance_checks"][
        "policy_metric_values_finite_if_present"
    ] is False
    assert report["metrics"][policy][reported_field] is None


def test_absent_optional_actor_rate_summary_values_still_use_documented_fallbacks(
) -> None:
    values = _phase_inputs()
    del values["actor_summary"]["ue_macro_expiry_aware_rate_mbps"]
    del values["actor_summary"]["ue_p10_expiry_aware_rate_mbps"]
    del values["actor_summary"]["ue_min_expiry_aware_rate_mbps"]

    report = build_phase_a_report(**values)

    assert report["classification"] == "ctbs_blank_v1_seed51_pass"
    assert report["metrics"]["actor"][
        "ue_macro_expiry_aware_rate_mbps"
    ] == pytest.approx(0.4861111111111111)
    assert report["metrics"]["actor"][
        "ue_p10_expiry_aware_rate_mbps"
    ] == pytest.approx(0.5)
    assert report["metrics"]["actor"]["ue_min_expiry_aware_rate_mbps"] is None


@pytest.mark.parametrize(
    ("classification", "mutate"),
    [
        (
            "invalid_experiment",
            lambda values: values["frozen_manifest"].update(
                simulator_sha256="bad"
            ),
        ),
        (
            "no_positive_ctbs_headroom",
            lambda values: values["guard_audit"].update(
                measured_override_count=0,
                measured_no_positive_ctbs_headroom_count=2,
            ),
        ),
        (
            "insufficient_blank_opportunities",
            lambda values: values["guard_audit"].update(
                measured_override_count=0,
                measured_no_positive_ctbs_headroom_count=0,
            ),
        ),
        (
            "one_prg_budget_insufficient",
            lambda values: (
                values["actor_starvation"]["per_ue"][4].update(
                    scheduled_dequeue_bytes=50
                ),
                values["guard_audit"].update(measured_cell_budget_used_count=1),
            ),
        ),
        (
            "no_positive_ctbs_headroom",
            lambda values: (
                values["actor_starvation"]["per_ue"][4].update(
                    scheduled_dequeue_bytes=50
                ),
                values["guard_audit"].update(
                    measured_no_positive_ctbs_headroom_count=1
                ),
            ),
        ),
        (
            "insufficient_blank_opportunities",
            lambda values: values["actor_starvation"]["per_ue"][4].update(
                scheduled_dequeue_bytes=50
            ),
        ),
        (
            "business_metric_cost",
            lambda values: values["actor_metrics"].update(goodput_mbps=98.9),
        ),
    ],
)
def test_phase_a_classification_precedence(classification: str, mutate) -> None:
    values = _phase_inputs()
    mutate(values)

    report = build_phase_a_report(**values)

    assert report["passed"] is False
    assert report["classification"] == classification


def test_valid_business_failure_still_reports_every_metric_family() -> None:
    values = _phase_inputs()
    values["actor_metrics"]["expiry_rate"] = 0.02

    report = build_phase_a_report(**values)

    assert report["classification"] == "business_metric_cost"
    assert set(report["metrics"]["actor"]["outcome_delay_ms"]) >= {
        "mean_ms",
        "p50_ms",
        "p90_ms",
        "p95_ms",
        "max_ms",
    }
    assert set(report["metrics"]["actor"]) >= {
        "ue_macro_expiry_aware_rate_mbps",
        "ue_p10_expiry_aware_rate_mbps",
        "ue_min_expiry_aware_rate_mbps",
        "goodput_mbps",
        "tb_bler",
        "expiry_rate",
        "prg_utilization_ratio",
        "blank_ratio",
        "scheduled_dequeue_bytes",
        "failed_bytes",
        "per_ue_expired_bytes",
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_observed_hashes_recompute_each_reference_from_the_cli_file(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    cli_paths: dict[str, str] = {}
    for role in REFERENCE_ARTIFACT_ROLES:
        path = tmp_path / f"{role}.artifact"
        contents = f"frozen-{role}\n".encode()
        path.write_bytes(contents)
        manifest["reference_artifact_paths"][role] = path.name
        manifest["reference_artifact_sha256"][role] = hashlib.sha256(
            contents
        ).hexdigest()
        cli_paths[role] = str(path)

    observed, _ = audit_module._observed_hashes(
        tmp_path.resolve(), manifest, cli_paths
    )

    for role in REFERENCE_ARTIFACT_ROLES:
        assert observed[f"reference_manifest_relative_path:{role}"] == manifest[
            "reference_artifact_paths"
        ][role]
        assert observed[f"reference_manifest_path:{role}"] == str(
            Path(cli_paths[role]).resolve()
        )
        assert observed[f"reference_actual_path:{role}"] == str(
            Path(cli_paths[role]).resolve()
        )
        assert observed[f"reference_sha256:{role}"] == manifest[
            "reference_artifact_sha256"
        ][role]

    Path(cli_paths["pfq_metrics"]).write_text("edited\n", encoding="utf-8")
    edited, _ = audit_module._observed_hashes(
        tmp_path.resolve(), manifest, cli_paths
    )
    assert edited["reference_sha256:pfq_metrics"] != manifest[
        "reference_artifact_sha256"
    ]["pfq_metrics"]


def _cli_args(tmp_path: Path) -> tuple[list[str], dict[str, Path]]:
    outputs = {
        "guard": tmp_path / "guard_audit.json",
        "offered": tmp_path / "offered_comparison.json",
        "report": tmp_path / "final_report.json",
    }
    args = [
        "--worktree-root",
        str(tmp_path),
        "--frozen-manifest",
        str(tmp_path / "manifest.json"),
        "--pfq-metrics",
        str(tmp_path / "pfq_metrics.json"),
        "--pfq-starvation",
        str(tmp_path / "pfq_starvation.json"),
        "--pfq-outcomes",
        str(tmp_path / "pfq_outcomes.csv"),
        "--pfq-offered-packets",
        str(tmp_path / "pfq_offered.csv"),
        "--unguarded-starvation",
        str(tmp_path / "unguarded.json"),
        "--actor-summary",
        str(tmp_path / "actor_summary.json"),
        "--actor-metrics",
        str(tmp_path / "actor_metrics.json"),
        "--actor-starvation",
        str(tmp_path / "actor_starvation.json"),
        "--actor-outcomes",
        str(tmp_path / "actor_outcomes.csv"),
        "--actor-offered-packets",
        str(tmp_path / "actor_offered.csv"),
        "--guard-trace",
        str(tmp_path / "guard.csv"),
        "--guard-audit-out",
        str(outputs["guard"]),
        "--offered-comparison-out",
        str(outputs["offered"]),
        "--final-report-out",
        str(outputs["report"]),
    ]
    return args, outputs


@pytest.mark.parametrize(
    ("actor_goodput", "expected_rc", "classification"),
    [
        (99.0, 0, "ctbs_blank_v1_seed51_pass"),
        (98.9, 2, "business_metric_cost"),
    ],
)
def test_cli_writes_all_reports_for_pass_and_valid_negative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    actor_goodput: float,
    expected_rc: int,
    classification: str,
) -> None:
    inputs = _phase_inputs()
    inputs["actor_metrics"]["goodput_mbps"] = actor_goodput
    manifest_paths = {
        "pfq_metrics": "pfq_metrics.json",
        "pfq_starvation": "pfq_starvation.json",
        "pfq_outcomes": "pfq_outcomes.csv",
        "pfq_offered_packets": "pfq_offered.csv",
        "unguarded_starvation": "unguarded.json",
    }
    inputs["frozen_manifest"]["reference_artifact_paths"] = manifest_paths
    observed = dict(inputs["observed_file_sha256"])
    for role, relative in manifest_paths.items():
        canonical = str((tmp_path / relative).resolve())
        observed[f"reference_manifest_relative_path:{role}"] = relative
        observed[f"reference_manifest_path:{role}"] = canonical
        observed[f"reference_actual_path:{role}"] = canonical

    payload_paths = {
        "frozen_manifest": "manifest.json",
        "pfq_metrics": "pfq_metrics.json",
        "pfq_starvation": "pfq_starvation.json",
        "unguarded_starvation": "unguarded.json",
        "actor_summary": "actor_summary.json",
        "actor_metrics": "actor_metrics.json",
        "actor_starvation": "actor_starvation.json",
    }
    for name, relative in payload_paths.items():
        _write_json(tmp_path / relative, inputs[name])

    monkeypatch.setattr(audit_module, "load_guard_trace", lambda path: [])
    monkeypatch.setattr(
        audit_module, "audit_guard_trace", lambda rows, **kwargs: _guard_audit()
    )
    monkeypatch.setattr(
        audit_module, "_offered_comparison", lambda pfq, actor: _offered()
    )
    monkeypatch.setattr(
        audit_module,
        "summarize_outcome_delays",
        lambda path, first, last: {
            "sample_count": 4,
            "mean_ms": 18.0,
            "p50_ms": 17.0,
            "p90_ms": 25.0,
            "p95_ms": 30.0,
            "max_ms": 35.0,
        },
    )
    monkeypatch.setattr(
        audit_module,
        "_observed_hashes",
        lambda root, manifest, references: (observed, []),
    )

    args, outputs = _cli_args(tmp_path)
    rc = main(args)

    assert rc == expected_rc
    assert json.loads(outputs["guard"].read_text())["passed"] is True
    assert json.loads(outputs["offered"].read_text())["exact"] is True
    report = json.loads(outputs["report"].read_text())
    assert report["classification"] == classification
    assert report["passed"] is (expected_rc == 0)


def test_cli_writes_all_reports_before_returning_invalid_exit(
    tmp_path: Path,
) -> None:
    inputs = _phase_inputs()
    manifest = tmp_path / "manifest.json"
    pfq_metrics = tmp_path / "pfq_metrics.json"
    pfq_starvation = tmp_path / "pfq_starvation.json"
    unguarded = tmp_path / "unguarded.json"
    actor_summary = tmp_path / "actor_summary.json"
    actor_metrics = tmp_path / "actor_metrics.json"
    actor_starvation = tmp_path / "actor_starvation.json"
    for path, payload in (
        (manifest, inputs["frozen_manifest"]),
        (pfq_metrics, inputs["pfq_metrics"]),
        (pfq_starvation, inputs["pfq_starvation"]),
        (unguarded, inputs["unguarded_starvation"]),
        (actor_summary, inputs["actor_summary"]),
        (actor_metrics, inputs["actor_metrics"]),
        (actor_starvation, inputs["actor_starvation"]),
    ):
        _write_json(path, payload)

    outcomes = tmp_path / "outcomes.csv"
    outcomes.write_text("terminal_tti,system_time_ms\n1001,1\n", encoding="utf-8")
    offered = tmp_path / "offered.csv"
    offered.write_text(
        "ue_id,packet_id,arrival_tti,packet_size_bytes,ttl_tti\n0,1,1,3000,400\n",
        encoding="utf-8",
    )
    guard = tmp_path / "guard.csv"
    guard.write_text("not,the,guard,schema\n", encoding="utf-8")
    guard_out = tmp_path / "guard_audit.json"
    offered_out = tmp_path / "offered_comparison.json"
    report_out = tmp_path / "final_report.json"

    rc = main(
        [
            "--worktree-root",
            str(tmp_path),
            "--frozen-manifest",
            str(manifest),
            "--pfq-metrics",
            str(pfq_metrics),
            "--pfq-starvation",
            str(pfq_starvation),
            "--pfq-outcomes",
            str(outcomes),
            "--pfq-offered-packets",
            str(offered),
            "--unguarded-starvation",
            str(unguarded),
            "--actor-summary",
            str(actor_summary),
            "--actor-metrics",
            str(actor_metrics),
            "--actor-starvation",
            str(actor_starvation),
            "--actor-outcomes",
            str(outcomes),
            "--actor-offered-packets",
            str(offered),
            "--guard-trace",
            str(guard),
            "--guard-audit-out",
            str(guard_out),
            "--offered-comparison-out",
            str(offered_out),
            "--final-report-out",
            str(report_out),
        ]
    )

    assert rc == 3
    assert json.loads(guard_out.read_text())["valid"] is False
    assert "exact" in json.loads(offered_out.read_text())
    assert json.loads(report_out.read_text())["classification"] == "invalid_experiment"

from __future__ import annotations

import csv

import pytest

from training.stageb_hgraph.independent_gate import (
    architecture_gate,
    compare_manifests,
    compare_treatments,
    load_outcome_delay_p95,
    normalize_gate_metrics,
    score_checkpoint,
    select_treatment_checkpoints,
)
from training.stageb_hgraph.online_protocol import VERSION


def _manifest(*, binary: str = "binary-a", source: str = "source-a", seed: int = 41):
    return {
        "binary_sha256": binary,
        "source_revision": source,
        "protocol_version": VERSION,
        "scenario": {
            "name": "coop_boundary",
            "topology_seed": seed,
            "traffic_seed": seed,
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


def _metrics(**updates):
    values = {
        "goodput_mbps": 100.0,
        "tb_bler": 0.10,
        "expiry_rate": 0.0100,
        "outcome_delay_p95_ms": 20.0,
        "ue_macro_ea_rate_mbps": 4.0,
        "ue_p10_ea_rate_mbps": 1.0,
        "action_replay_mismatch_count": 0.0,
        "illegal_action_count": 0.0,
        "independent_deletion_equal": True,
        "ea_metrics_available": True,
        "measured_tti_first": 1001,
        "measured_tti_last": 5000,
    }
    values.update(updates)
    return values


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"binary_sha256": "binary-b"}, "binary_sha256"),
        ({"source_revision": "source-b"}, "source_revision"),
        ({"protocol_version": VERSION - 1}, "protocol_version"),
        ({"scenario": {**_manifest()["scenario"], "traffic_seed": 42}}, "scenario"),
    ],
)
def test_manifest_mismatch_fails_before_metric_comparison(updates, match):
    treatment = {**_manifest(), **updates}
    with pytest.raises(ValueError, match=match):
        compare_manifests(_manifest(), treatment)


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"ea_metrics_available": False}, "expiry-aware"),
        ({"outcome_delay_p95_ms": None}, "outcome_delay_p95_ms"),
        ({"ue_macro_ea_rate_mbps": None}, "ue_macro_ea_rate_mbps"),
        ({"ue_p10_ea_rate_mbps": None}, "ue_p10_ea_rate_mbps"),
    ],
)
def test_architecture_gate_fails_closed_on_missing_ea_metrics(updates, match):
    with pytest.raises(ValueError, match=match):
        architecture_gate(_metrics(), _metrics(**updates))


def test_seed41_architecture_gate_uses_locked_thresholds_and_full_evidence():
    pfq = _metrics()
    actor = _metrics(
        goodput_mbps=99.0,
        tb_bler=0.102,
        expiry_rate=0.0105,
    )
    report = architecture_gate(pfq, actor)

    assert report["passed"] is True
    assert report["checks"]["goodput"]["numerator"] == 99.0
    assert report["checks"]["goodput"]["denominator"] == 100.0
    assert report["checks"]["goodput"]["relative_delta"] == pytest.approx(-0.01)
    assert report["checks"]["goodput"]["threshold"] == 0.99
    assert report["checks"]["tb_bler"]["absolute_delta"] == pytest.approx(0.002)
    assert report["checks"]["expiry_rate"]["absolute_delta"] == pytest.approx(0.0005)

    assert architecture_gate(pfq, _metrics(goodput_mbps=98.999))["passed"] is False
    assert architecture_gate(pfq, _metrics(tb_bler=0.10201))["passed"] is False
    assert architecture_gate(pfq, _metrics(expiry_rate=0.010501))["passed"] is False
    assert architecture_gate(pfq, _metrics(action_replay_mismatch_count=1))["passed"] is False
    assert architecture_gate(pfq, _metrics(illegal_action_count=1))["passed"] is False
    assert architecture_gate(pfq, _metrics(independent_deletion_equal=False))["passed"] is False


def test_checkpoint_score_is_pfq_relative_and_penalizes_worse_delay():
    pfq = _metrics()
    fast = _metrics(goodput_mbps=101.0, ue_macro_ea_rate_mbps=4.4, outcome_delay_p95_ms=20.0)
    slow = _metrics(goodput_mbps=101.0, ue_macro_ea_rate_mbps=4.4, outcome_delay_p95_ms=24.0)

    fast_report = score_checkpoint(pfq, fast)
    slow_report = score_checkpoint(pfq, slow)

    assert fast_report["components"]["relative_goodput_delta"] == pytest.approx(0.01)
    assert fast_report["components"]["relative_ue_ea_rate_delta"] == pytest.approx(0.10)
    assert slow_report["components"]["worse_outcome_delay_relative"] == pytest.approx(0.20)
    assert fast_report["score"] > slow_report["score"]


def test_checkpoint_selection_only_scores_cadence_candidates_that_pass_hard_gates():
    candidates = {
        4: _metrics(goodput_mbps=98.0),
        8: _metrics(goodput_mbps=99.0),
        10: _metrics(goodput_mbps=110.0),
        12: _metrics(goodput_mbps=100.0, action_replay_mismatch_count=1),
        16: _metrics(goodput_mbps=100.5),
        20: _metrics(goodput_mbps=100.2),
    }
    report = select_treatment_checkpoints(_metrics(), candidates)

    assert set(report["candidates"]) == {"4", "8", "12", "16", "20"}
    assert report["candidates"]["4"]["scored"] is False
    assert report["candidates"]["12"]["scored"] is False
    assert report["selected_iteration"] == 16
    assert report["first_gate_iteration"] == 8
    assert report["first_gate_measured_ppo_ttis"] == 4096


def test_treatment_preference_and_h2_time_to_gate_are_fail_closed():
    pfq = _metrics()
    selections = {
        "A": select_treatment_checkpoints(pfq, {4: _metrics(goodput_mbps=98), 8: _metrics()}),
        "B": select_treatment_checkpoints(pfq, {4: _metrics()}),
        "C": select_treatment_checkpoints(
            pfq,
            {4: _metrics(outcome_delay_p95_ms=19, ue_macro_ea_rate_mbps=4.1, ue_p10_ea_rate_mbps=1.1)},
        ),
        "D": select_treatment_checkpoints(
            pfq,
            {4: _metrics(outcome_delay_p95_ms=21, ue_macro_ea_rate_mbps=4.2, ue_p10_ea_rate_mbps=1.2)},
        ),
    }
    comparison = compare_treatments(pfq, selections)

    assert comparison["h2"]["supported"] is True
    assert comparison["h2"]["a_first_gate_measured_ppo_ttis"] == 4096
    assert comparison["h2"]["b_first_gate_measured_ppo_ttis"] == 2048
    assert comparison["treatments"]["C"]["auxiliary_preference_eligible"] is True
    assert comparison["treatments"]["D"]["auxiliary_preference_eligible"] is False
    assert comparison["selected_treatment"] == "C"
    assert comparison["launch_paired_seeds_41_43"] is True

    same_time = compare_treatments(
        pfq,
        {
            "A": select_treatment_checkpoints(pfq, {4: _metrics()}),
            "B": select_treatment_checkpoints(pfq, {4: _metrics()}),
        },
    )
    assert same_time["h2"]["supported"] is False
    assert "2048" in same_time["h2"]["reason"]

    failed = compare_treatments(
        pfq,
        {"A": select_treatment_checkpoints(pfq, {4: _metrics(goodput_mbps=90)})},
    )
    assert failed["selected_treatment"] is None
    assert failed["launch_paired_seeds_41_43"] is False


def test_outcome_delay_p95_uses_only_matching_terminal_tti_interval(tmp_path):
    path = tmp_path / "outcomes.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ue_id", "packet_id", "terminal_tti", "outcome", "system_time_ms"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"ue_id": 0, "packet_id": 0, "terminal_tti": 1000, "outcome": "delivered", "system_time_ms": 999},
                {"ue_id": 0, "packet_id": 1, "terminal_tti": 1001, "outcome": "delivered", "system_time_ms": 1},
                {"ue_id": 0, "packet_id": 2, "terminal_tti": 5000, "outcome": "expired", "system_time_ms": 200},
                {"ue_id": 0, "packet_id": 3, "terminal_tti": 5001, "outcome": "delivered", "system_time_ms": 999},
            ]
        )

    report = load_outcome_delay_p95(path, measured_tti_first=1001, measured_tti_last=5000)
    assert report["sample_count"] == 2
    assert report["terminal_tti_first"] == 1001
    assert report["terminal_tti_last"] == 5000
    assert report["outcome_delay_p95_ms"] == pytest.approx(190.05)


def test_evaluator_summary_is_normalized_to_gate_metrics_without_defaults():
    normalized = normalize_gate_metrics(
        {
            "protocol_version": VERSION,
            "stats_mean_goodput_mbps": 99.0,
            "measured_tb_bler": 0.11,
            "stats_count": 4000,
            "stats_sum_expired_bytes": 8_640_000,
            "packet_outcome_delay_p95_ms": 100.0,
            "packet_outcome_measured_count": 8,
            "ue_macro_expiry_aware_rate_mbps": 3.0,
            "ue_p10_expiry_aware_rate_mbps": 0.4,
            "ue_expiry_aware_rate_active_count": 36,
            "stats_sum_action_replay_mismatch_count": 0.0,
            "stats_sum_illegal_action_count": 0.0,
            "independent_deletion_equal": True,
            "measured_tti_first": 1001,
            "measured_tti_last": 5000,
        }
    )
    assert normalized == _metrics(
        goodput_mbps=99.0,
        tb_bler=0.11,
        expiry_rate=0.02,
        outcome_delay_p95_ms=100.0,
        ue_macro_ea_rate_mbps=3.0,
        ue_p10_ea_rate_mbps=0.4,
    )

    with pytest.raises(ValueError, match="expiry-aware"):
        normalize_gate_metrics(
            {
                "protocol_version": VERSION,
                "stats_mean_goodput_mbps": 99.0,
                "measured_tb_bler": 0.11,
                "stats_count": 4000,
                "stats_sum_expired_bytes": 8_640_000,
                "packet_outcome_delay_p95_ms": None,
            }
        )

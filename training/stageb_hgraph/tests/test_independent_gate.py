from __future__ import annotations

import csv

import pytest

from training.stageb_hgraph.independent_gate import (
    architecture_gate,
    compare_manifests,
    compare_treatments,
    ctbs_formal_evaluation_gate,
    ctbs_seed42_formal_pair_gate,
    ctbs_seed42_smoke_gate,
    long_multiseed_seed42_gate,
    short_debt_treatment_gate,
    formal_seed42_pair_gate,
    load_outcome_delay_p95,
    normalize_gate_metrics,
    score_checkpoint,
    select_treatment_checkpoints,
    staged_multiseed_safety_prefix_gate,
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


def _w0_v2_inputs():
    static = {
        "passed": True,
        "actor_state_dict_sha256": "tensor-content-hash",
        "state_tensor_names_equal": True,
        "state_tensor_values_bitwise_equal": True,
        "state_dict_hash_match_after_strict_load": True,
    }
    seed41 = {
        "completed_steps": 128,
        "candidate_mode": "all_demand_ues",
        "candidate_width_policy": "runtime_topology",
        "candidate_truncation": "forbidden",
        "candidate_full_coverage": True,
        "candidate_truncation_count": 0,
        "stats_sum_action_replay_mismatch_count": 0.0,
        "stats_sum_illegal_action_count": 0.0,
        "independent_deletion_equal": True,
        "actor_state_dict_sha256": "tensor-content-hash",
        "training_observed_max_candidates": 13,
        "topology_candidate_width": 13,
        "runtime_candidate_width": 14,
        "candidate_width_equivalence_steps": 128,
        "candidate_width_equivalence_protocol": "fp32-w0-v2",
        "candidate_width_equivalence": {
            "candidate_choice_equal": True,
            "mask_equal": True,
            "type0_action_equal": True,
            "extra_padding_invalid": True,
            "runtime_replay_mismatch_count": 0,
            "valid_logits_atol": 1.0e-5,
            "chosen_logprob_atol": 1.0e-6,
            "valid_logits_max_abs_diff": 9.54e-7,
            "chosen_logprob_max_abs_diff": 5.96e-8,
            "nan_or_inf_count": 0,
            "passed": True,
        },
    }
    seed42 = {
        "completed_steps": 128,
        "candidate_mode": "all_demand_ues",
        "candidate_width_policy": "runtime_topology",
        "candidate_truncation": "forbidden",
        "candidate_full_coverage": True,
        "candidate_truncation_count": 0,
        "stats_sum_action_replay_mismatch_count": 0.0,
        "stats_sum_illegal_action_count": 0.0,
        "independent_deletion_equal": True,
        "actor_state_dict_sha256": "tensor-content-hash",
        "training_observed_max_candidates": 13,
        "association_candidate_counts": [11, 11, 14],
        "topology_candidate_width": 14,
        "runtime_candidate_width": 14,
    }
    return static, seed41, seed42


def _formal_seed42_actor_summary(**updates):
    summary = {
        "completed_steps": 5000,
        "stats_count": 4000,
        "measured_tti_first": 1001,
        "measured_tti_last": 5000,
        "sim_binary_sha256": "binary-a",
        "candidate_mode": "all_demand_ues",
        "candidate_width_policy": "runtime_topology",
        "candidate_truncation": "forbidden",
        "candidate_full_coverage": True,
        "candidate_truncation_count": 0,
        "stats_sum_candidate_truncation_count": 0.0,
        "stats_sum_action_replay_mismatch_count": 0.0,
        "stats_sum_illegal_action_count": 0.0,
        "independent_deletion_equal": True,
        "actor_state_dict_sha256": "frozen-b4",
        "training_observed_max_candidates": 13,
        "association_candidate_counts": [11, 11, 14],
        "topology_candidate_width": 14,
        "runtime_candidate_width": 14,
        "stats_mean_blank_ratio": 0.25,
        "stats_mean_fill_ratio": 0.75,
        "stats_mean_prg_utilization_ratio": 0.75,
        "stats_mean_prg_reuse_ratio": 0.5,
    }
    summary.update(updates)
    return summary


def _formal_seed42_manifest(*, source_state: str = "frozen-source"):
    return {
        **_manifest(seed=42),
        "source_state_sha256": source_state,
        "checkpoint_state_dict_sha256": "frozen-b4",
        "warmup_steps": 1000,
        "measure_steps": 4000,
    }


def _ctbs_actor_summary(*, completed_steps: int, stats_count: int, **updates):
    summary = _formal_seed42_actor_summary(
        completed_steps=completed_steps,
        stats_count=stats_count,
        measured_tti_first=1 if stats_count == completed_steps else 1001,
        measured_tti_last=completed_steps,
        protocol_version=VERSION,
        runtime_harq_enabled=0,
        independent_headroom_accounting="ctbs-v1",
        ctbs_pre_action_handshake_enabled=True,
        ctbs_config_sha256="c" * 64,
        ctbs_snapshot_count=completed_steps,
        ctbs_snapshot_id_first=1,
        ctbs_snapshot_id_last=completed_steps,
        ctbs_mcs_mismatch_count=0,
        ctbs_layer_mismatch_count=0,
        ctbs_tbs_mismatch_count=0,
        ctbs_reference_label_deletion_equal=True,
        ctbs_mask_reactivation_count=0,
        ctbs_mask_trigger_count_by_local_prg=[0] * 17,
        ctbs_selected_count_by_local_prg=[1] * 17,
        ctbs_valid_candidate_count_by_local_prg=[1] * 17,
    )
    summary.update(updates)
    return summary


def test_ctbs_seed42_smoke_requires_causal_exact_and_replay_gates() -> None:
    passed = ctbs_seed42_smoke_gate(
        _ctbs_actor_summary(completed_steps=128, stats_count=128),
        frozen_state_dict_sha256="frozen-b4",
        frozen_binary_sha256="binary-a",
    )
    failed = ctbs_seed42_smoke_gate(
        _ctbs_actor_summary(
            completed_steps=128,
            stats_count=128,
            ctbs_tbs_mismatch_count=1,
        ),
        frozen_state_dict_sha256="frozen-b4",
        frozen_binary_sha256="binary-a",
    )

    assert passed["passed"] is True
    assert passed["classification"] == "b4_ctbs_seed42_smoke_pass"
    assert failed["passed"] is False
    assert failed["checks"]["exact_tbs_parity"] is False


def test_ctbs_seed42_formal_gate_keeps_mechanism_diagnostics_record_only() -> None:
    report = ctbs_seed42_formal_pair_gate(
        _metrics(),
        _metrics(goodput_mbps=99.0, tb_bler=0.102, expiry_rate=0.0105),
        _ctbs_actor_summary(completed_steps=5000, stats_count=4000),
        _formal_seed42_manifest(),
        _formal_seed42_manifest(),
        frozen_state_dict_sha256="frozen-b4",
    )

    assert report["passed"] is True
    assert report["classification"] == "seed42_ctbs_formal_gate_pass"
    assert report["structural_gate"]["checks"]["ctbs_exact_tbs_parity"] is True
    assert report["recorded_diagnostics"]["ctbs"]["gate_role"] == "record_only"


def _short_treatment_service_report(prg_ratio: float) -> dict[str, object]:
    pfq_target = 1000.0
    pfq_cell = 10_000.0
    actor_cell = 10_000.0
    actor_target = pfq_target * prg_ratio
    return {
        "deficits": {
            "actual_prg_count": {
                "ctbs_over_pfq": prg_ratio,
                "ctbs_overall": actor_target,
                "pfq_overall": pfq_target,
            }
        },
        "population_association_only": {
            "ctbs_cell2": {"actual_prg_count": actor_cell},
            "pfq_cell2": {"actual_prg_count": pfq_cell},
        },
    }


def test_short_debt_treatment_requires_performance_and_clear_ue2_recovery() -> None:
    summary = _ctbs_actor_summary(completed_steps=5000, stats_count=4000)
    smoke = {"passed": True}
    promising = short_debt_treatment_gate(
        _metrics(),
        _metrics(goodput_mbps=99.0, tb_bler=0.102, expiry_rate=0.0105),
        summary,
        smoke,
        _short_treatment_service_report(0.91),
        actor_state_dict_sha256="frozen-b4",
        binary_sha256="binary-a",
    )
    boundary_failure = short_debt_treatment_gate(
        _metrics(),
        _metrics(goodput_mbps=99.0, tb_bler=0.102, expiry_rate=0.0105),
        summary,
        smoke,
        _short_treatment_service_report(0.90),
        actor_state_dict_sha256="frozen-b4",
        binary_sha256="binary-a",
    )

    assert promising["classification"] == "short_debt_treatment_promising"
    assert promising["passed"] is True
    assert boundary_failure["classification"] == "short_debt_treatment_failure"
    assert boundary_failure["ue2_allocation_gate"]["passed"] is False


def test_generic_ctbs_formal_evaluation_accepts_seed41_topology() -> None:
    report = ctbs_formal_evaluation_gate(
        _metrics(),
        _metrics(goodput_mbps=99.0, tb_bler=0.102, expiry_rate=0.0105),
        _ctbs_actor_summary(
            completed_steps=5000,
            stats_count=4000,
            association_candidate_counts=[11, 12, 13],
            topology_candidate_width=13,
            runtime_candidate_width=13,
            actor_state_dict_sha256="trained-long",
        ),
        actor_state_dict_sha256="trained-long",
        binary_sha256="binary-a",
        expected_association_counts=(11, 12, 13),
        expected_runtime_width=13,
        protocol_id="seed41-long-evaluation",
    )

    assert report["passed"] is True
    assert report["structural_gate"]["passed"] is True
    assert report["performance_gate"]["passed"] is True


def test_long_multiseed_seed42_requires_performance_and_ue2_allocation() -> None:
    summary = _ctbs_actor_summary(
        completed_steps=5000,
        stats_count=4000,
        actor_state_dict_sha256="trained-long",
    )
    passed = long_multiseed_seed42_gate(
        _metrics(),
        _metrics(goodput_mbps=99.0, tb_bler=0.102, expiry_rate=0.0105),
        summary,
        _short_treatment_service_report(0.91),
        actor_state_dict_sha256="trained-long",
        binary_sha256="binary-a",
    )
    failed = long_multiseed_seed42_gate(
        _metrics(),
        _metrics(goodput_mbps=99.0, tb_bler=0.102, expiry_rate=0.0105),
        summary,
        _short_treatment_service_report(0.90),
        actor_state_dict_sha256="trained-long",
        binary_sha256="binary-a",
    )

    assert passed["classification"] == "long_multiseed_seed42_pass"
    assert passed["passed"] is True
    assert failed["classification"] == "long_multiseed_seed42_failure"
    assert failed["ue2_allocation_gate"]["passed"] is False


def test_formal_seed42_pair_passes_only_structural_then_performance_gates() -> None:
    report = formal_seed42_pair_gate(
        _metrics(),
        _metrics(goodput_mbps=99.0, tb_bler=0.102, expiry_rate=0.0105),
        _formal_seed42_actor_summary(),
        _formal_seed42_manifest(),
        _formal_seed42_manifest(),
        frozen_state_dict_sha256="frozen-b4",
    )

    assert report["protocol_id"] == "seed42-pfq-b4-formal-v1"
    assert report["structural_gate"]["status"] == "PASS"
    assert report["performance_gate"]["status"] == "PASS"
    assert report["classification"] == "seed42_formal_gate_pass"
    assert report["passed"] is True
    assert report["status"] == (
        "Seed42 formal PFQ-B4 pair PASS: frozen B4 passes the first "
        "unseen-seed performance gate."
    )


def test_formal_seed42_pair_structural_failure_precedes_performance() -> None:
    report = formal_seed42_pair_gate(
        _metrics(),
        _metrics(goodput_mbps=110.0),
        _formal_seed42_actor_summary(candidate_truncation_count=1),
        _formal_seed42_manifest(),
        _formal_seed42_manifest(),
        frozen_state_dict_sha256="frozen-b4",
    )

    assert report["structural_gate"]["status"] == "FAIL"
    assert report["performance_gate"]["status"] == "NOT_EVALUATED"
    assert report["classification"] == "runtime_compatibility_path_failure"
    assert report["passed"] is False


def test_formal_seed42_pair_reports_performance_failure_after_structural_pass() -> None:
    report = formal_seed42_pair_gate(
        _metrics(),
        _metrics(goodput_mbps=98.9),
        _formal_seed42_actor_summary(),
        _formal_seed42_manifest(),
        _formal_seed42_manifest(),
        frozen_state_dict_sha256="frozen-b4",
    )

    assert report["structural_gate"]["status"] == "PASS"
    assert report["performance_gate"]["status"] == "FAIL"
    assert report["classification"] == "seed42_performance_gate_failure"
    assert report["passed"] is False


def test_formal_seed42_diagnostics_are_recorded_but_never_promoted_to_gates() -> None:
    pfq = _metrics(outcome_delay_p95_ms=999.0, ue_p10_ea_rate_mbps=0.001)
    pfq.update(
        blank_ratio_mean=0.99,
        prg_opportunity_count_mean=0.01,
        prg_utilization_ratio_mean=0.01,
        prg_reuse_ratio_mean=0.01,
    )
    actor = _metrics(
        goodput_mbps=99.0,
        tb_bler=0.102,
        expiry_rate=0.0105,
        outcome_delay_p95_ms=9999.0,
        ue_p10_ea_rate_mbps=0.00001,
    )

    report = formal_seed42_pair_gate(
        pfq,
        actor,
        _formal_seed42_actor_summary(
            stats_mean_blank_ratio=1.0,
            stats_mean_fill_ratio=0.0,
        ),
        _formal_seed42_manifest(),
        _formal_seed42_manifest(),
        frozen_state_dict_sha256="frozen-b4",
    )

    assert report["passed"] is True
    assert report["recorded_diagnostics"]["gate_role"] == "record_only"
    assert report["recorded_diagnostics"]["actor"]["prg_opportunity_count_mean"] == 0.0
    assert "outcome_delay_p95_ms" not in report["performance_gate"]["gate"]["checks"]


def test_w0_v2_reports_seed41_and_seed42_conclusions_separately() -> None:
    from training.stageb_hgraph import independent_gate

    gate = getattr(independent_gate, "candidate_width_w0_v2_gate", None)
    assert callable(gate), "W0-v2 summary gate is missing"
    static, seed41, seed42 = _w0_v2_inputs()

    report = gate(static, seed41, seed42, expected_steps=128)

    assert report["protocol_id"] == "W0-v2"
    assert report["seed41_numerical_compatibility"]["status"] == "PASS"
    assert report["seed42_structural_smoke"]["status"] == "PASS"
    assert report["passed"] is True
    assert report["status"] == (
        "W0-v2 PASS: behaviorally identical under frozen FP32 tolerance "
        "and structurally compatible with seed42 topology."
    )


def test_w0_v2_never_hides_a_seed_specific_failure() -> None:
    from training.stageb_hgraph import independent_gate

    gate = getattr(independent_gate, "candidate_width_w0_v2_gate", None)
    assert callable(gate), "W0-v2 summary gate is missing"
    static, seed41, seed42 = _w0_v2_inputs()
    seed42["candidate_truncation_count"] = 1

    report = gate(static, seed41, seed42, expected_steps=128)

    assert report["seed41_numerical_compatibility"]["status"] == "PASS"
    assert report["seed42_structural_smoke"]["status"] == "FAIL"
    assert report["passed"] is False
    assert not report["status"].startswith("W0-v2 PASS")


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


def _safe_multiseed_training_row(iteration: int) -> dict[str, object]:
    return {
        "iter": float(iteration),
        "topology_seed": float(41 if iteration % 2 == 1 else 42),
        "rollout_steps": 512.0,
        "rollout_reward_total_mean": 0.8,
        "policy_loss": -0.01,
        "value_loss": 0.2,
        "entropy_mean": 1.0,
        "approx_kl_mean": 0.001,
        "rollout_ctbs_pre_action_handshake_enabled": True,
        "rollout_ctbs_mcs_mismatch_count": 0.0,
        "rollout_ctbs_layer_mismatch_count": 0.0,
        "rollout_ctbs_tbs_mismatch_count": 0.0,
        "rollout_ctbs_reference_label_deletion_equal": True,
        "rollout_action_replay_mismatch_count_mean": 0.0,
        "rollout_native_reject_diagnostics_available_mean": 1.0,
        "rollout_python_preflight_diagnostics_available_mean": 1.0,
        "rollout_native_reject_wrong_cell_slot_count_mean": 0.0,
        "rollout_native_reject_empty_slot_count_mean": 0.0,
        "rollout_native_reject_prg_mask_count_mean": 0.0,
        "rollout_native_reject_oob_slot_count_mean": 0.0,
        "rollout_native_reject_duplicate_ue_count_mean": 0.0,
        "rollout_python_preflight_wrong_cell_slot_count_mean": 0.0,
        "rollout_python_preflight_empty_slot_count_mean": 0.0,
        "rollout_python_preflight_prg_mask_count_mean": 0.0,
        "rollout_python_preflight_oob_slot_count_mean": 0.0,
        "rollout_python_preflight_duplicate_ue_count_mean": 0.0,
    }


def test_staged_multiseed_safety_prefix_passes_at_eight_alternating_rollouts():
    report = staged_multiseed_safety_prefix_gate(
        [_safe_multiseed_training_row(iteration) for iteration in range(1, 9)]
    )

    assert report["status"] == "PASS"
    assert report["prefix_complete"] is True
    assert report["measured_ttis"] == 4096
    assert report["observed_topology_seeds"] == [41, 42, 41, 42, 41, 42, 41, 42]
    assert all(report["checks"].values())


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ({"rollout_reward_total_mean": float("nan")}, "all_numeric_metrics_finite"),
        ({"topology_seed": 41.0}, "seed_cycle_exact"),
        ({"rollout_action_replay_mismatch_count_mean": 1.0}, "zero_replay_mismatch"),
        ({"rollout_native_reject_prg_mask_count_mean": 1.0}, "zero_illegal_action"),
        ({"rollout_ctbs_tbs_mismatch_count": 1.0}, "ctbs_exact_parity"),
    ],
)
def test_staged_multiseed_safety_prefix_fails_closed(
    mutation: dict[str, object], failed_check: str
):
    rows = [_safe_multiseed_training_row(iteration) for iteration in range(1, 9)]
    rows[1].update(mutation)

    report = staged_multiseed_safety_prefix_gate(rows)

    assert report["status"] == "FAIL"
    assert report["safe_so_far"] is False
    assert report["checks"][failed_check] is False

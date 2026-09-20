"""Fail-closed research gates for the PFQ-independent boundary experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .online_protocol import VERSION
from .independent_contract import (
    W0_V2_CHOSEN_LOGPROB_ATOL,
    W0_V2_VALID_LOGITS_ATOL,
)


CHECKPOINT_ITERATIONS = (4, 8, 12, 16, 20)
PPO_TTIS_PER_ITERATION = 512
GOODPUT_RATIO_MIN = 0.99
BLER_ABSOLUTE_DELTA_MAX = 0.002
EXPIRY_ABSOLUTE_DELTA_MAX = 0.0005
SCORE_UE_WEIGHT = 0.25
SCORE_DELAY_WEIGHT = 0.25

_METRIC_FIELDS = (
    "goodput_mbps",
    "tb_bler",
    "expiry_rate",
    "outcome_delay_p95_ms",
    "ue_macro_ea_rate_mbps",
    "ue_p10_ea_rate_mbps",
    "action_replay_mismatch_count",
    "illegal_action_count",
    "measured_tti_first",
    "measured_tti_last",
)

_TRAINING_ILLEGAL_COUNT_FIELDS = (
    "rollout_native_reject_wrong_cell_slot_count_mean",
    "rollout_native_reject_empty_slot_count_mean",
    "rollout_native_reject_prg_mask_count_mean",
    "rollout_native_reject_oob_slot_count_mean",
    "rollout_native_reject_duplicate_ue_count_mean",
    "rollout_python_preflight_wrong_cell_slot_count_mean",
    "rollout_python_preflight_empty_slot_count_mean",
    "rollout_python_preflight_prg_mask_count_mean",
    "rollout_python_preflight_oob_slot_count_mean",
    "rollout_python_preflight_duplicate_ue_count_mean",
)


def staged_multiseed_safety_prefix_gate(
    history: Sequence[Mapping[str, object]],
    *,
    prefix_iterations: int = 8,
    expected_seed_cycle: Sequence[int] = (41, 42),
) -> dict[str, object]:
    """Fail closed on numerical or action-path faults in a training prefix."""
    if int(prefix_iterations) <= 0:
        raise ValueError("prefix_iterations must be positive")
    if not expected_seed_cycle:
        raise ValueError("expected_seed_cycle must not be empty")
    rows = list(history[: int(prefix_iterations)])
    seeds = [int(round(float(row.get("topology_seed", -1)))) for row in rows]
    expected = [
        int(expected_seed_cycle[index % len(expected_seed_cycle)])
        for index in range(len(rows))
    ]

    def numeric_metrics_finite(row: Mapping[str, object]) -> bool:
        return all(
            math.isfinite(float(value))
            for value in row.values()
            if isinstance(value, (int, float, np.number))
        )

    checks = {
        "all_numeric_metrics_finite": all(numeric_metrics_finite(row) for row in rows),
        "seed_cycle_exact": seeds == expected,
        "fixed_512_tti_rollouts": all(
            float(row.get("rollout_steps", -1.0)) == 512.0 for row in rows
        ),
        "zero_replay_mismatch": all(
            float(row.get("rollout_action_replay_mismatch_count_mean", float("inf")))
            == 0.0
            for row in rows
        ),
        "zero_illegal_action": all(
            float(row.get("rollout_native_reject_diagnostics_available_mean", 0.0))
            == 1.0
            and float(row.get("rollout_python_preflight_diagnostics_available_mean", 0.0))
            == 1.0
            and all(float(row.get(field, float("inf"))) == 0.0 for field in _TRAINING_ILLEGAL_COUNT_FIELDS)
            for row in rows
        ),
        "ctbs_exact_parity": all(
            row.get("rollout_ctbs_pre_action_handshake_enabled") is True
            and float(row.get("rollout_ctbs_mcs_mismatch_count", float("inf"))) == 0.0
            and float(row.get("rollout_ctbs_layer_mismatch_count", float("inf"))) == 0.0
            and float(row.get("rollout_ctbs_tbs_mismatch_count", float("inf"))) == 0.0
            and row.get("rollout_ctbs_reference_label_deletion_equal") is True
            for row in rows
        ),
    }
    safe = all(checks.values())
    complete = len(rows) == int(prefix_iterations)
    status = "FAIL" if not safe else ("PASS" if complete else "IN_PROGRESS")
    return {
        "protocol_id": "b4-ctbs-c1-multiseed-long-v1-safety-prefix",
        "status": status,
        "safe_so_far": safe,
        "prefix_complete": complete,
        "required_iterations": int(prefix_iterations),
        "observed_iterations": len(rows),
        "measured_ttis": sum(int(float(row.get("rollout_steps", 0))) for row in rows),
        "expected_seed_cycle": [int(seed) for seed in expected_seed_cycle],
        "observed_topology_seeds": seeds,
        "checks": checks,
    }


def _candidate_width_common_checks(
    summary: Mapping[str, object],
    *,
    expected_steps: int,
    actor_state_dict_sha256: object,
) -> dict[str, bool]:
    return {
        "completed_steps": summary.get("completed_steps") == expected_steps,
        "candidate_mode": summary.get("candidate_mode") == "all_demand_ues",
        "candidate_width_policy": summary.get("candidate_width_policy") == "runtime_topology",
        "candidate_truncation_forbidden": summary.get("candidate_truncation") == "forbidden",
        "candidate_full_coverage": summary.get("candidate_full_coverage") is True,
        "zero_truncation": summary.get("candidate_truncation_count") == 0,
        "zero_replay_mismatch": summary.get("stats_sum_action_replay_mismatch_count") == 0.0,
        "zero_illegal_action": summary.get("stats_sum_illegal_action_count") == 0.0,
        "pfq_deletion_equal": summary.get("independent_deletion_equal") is True,
        "state_dict_frozen": summary.get("actor_state_dict_sha256")
        == actor_state_dict_sha256,
        "training_width_is_metadata": summary.get("training_observed_max_candidates") == 13,
    }


def candidate_width_w0_v2_gate(
    static: Mapping[str, object],
    seed41: Mapping[str, object],
    seed42: Mapping[str, object],
    *,
    expected_steps: int,
) -> dict[str, object]:
    """Summarize numerical compatibility and structural smoke independently."""
    state_hash = static.get("actor_state_dict_sha256")
    static_checks = {
        "static_gate_passed": static.get("passed") is True,
        "state_tensor_names_equal": static.get("state_tensor_names_equal") is True,
        "state_tensor_values_bitwise_equal": static.get(
            "state_tensor_values_bitwise_equal"
        )
        is True,
        "state_dict_hash_match_after_strict_load": static.get(
            "state_dict_hash_match_after_strict_load"
        )
        is True,
    }
    raw_equivalence = seed41.get("candidate_width_equivalence")
    equivalence = raw_equivalence if isinstance(raw_equivalence, Mapping) else {}
    seed41_checks = {
        **_candidate_width_common_checks(
            seed41,
            expected_steps=expected_steps,
            actor_state_dict_sha256=state_hash,
        ),
        "topology_width": seed41.get("topology_candidate_width") == 13,
        "runtime_width": seed41.get("runtime_candidate_width") == 14,
        "equivalence_steps": seed41.get("candidate_width_equivalence_steps")
        == expected_steps,
        "protocol": seed41.get("candidate_width_equivalence_protocol") == "fp32-w0-v2",
        "choice_bitwise_equal": equivalence.get("candidate_choice_equal") is True,
        "mask_bitwise_equal": equivalence.get("mask_equal") is True,
        "type0_action_bitwise_equal": equivalence.get("type0_action_equal") is True,
        "extra_padding_invalid": equivalence.get("extra_padding_invalid") is True,
        "runtime_replay_mismatch_zero": equivalence.get(
            "runtime_replay_mismatch_count"
        )
        == 0,
        "valid_logits_tolerance_frozen": equivalence.get("valid_logits_atol")
        == W0_V2_VALID_LOGITS_ATOL,
        "chosen_logprob_tolerance_frozen": equivalence.get("chosen_logprob_atol")
        == W0_V2_CHOSEN_LOGPROB_ATOL,
        "valid_logits_within_tolerance": float(
            equivalence.get("valid_logits_max_abs_diff", float("inf"))
        )
        <= W0_V2_VALID_LOGITS_ATOL,
        "chosen_logprob_within_tolerance": float(
            equivalence.get("chosen_logprob_max_abs_diff", float("inf"))
        )
        <= W0_V2_CHOSEN_LOGPROB_ATOL,
        "zero_nan_or_inf": equivalence.get("nan_or_inf_count") == 0,
        "per_step_gate_passed": equivalence.get("passed") is True,
    }
    seed42_checks = {
        **_candidate_width_common_checks(
            seed42,
            expected_steps=expected_steps,
            actor_state_dict_sha256=state_hash,
        ),
        "association_counts": seed42.get("association_candidate_counts")
        == [11, 11, 14],
        "topology_width": seed42.get("topology_candidate_width") == 14,
        "runtime_width": seed42.get("runtime_candidate_width") == 14,
    }
    seed41_passed = all(static_checks.values()) and all(seed41_checks.values())
    seed42_passed = all(seed42_checks.values())
    passed = seed41_passed and seed42_passed
    return {
        "protocol_id": "W0-v2",
        "frozen_fp32_tolerances": {
            "valid_logits_max_abs_diff": W0_V2_VALID_LOGITS_ATOL,
            "chosen_logprob_max_abs_diff": W0_V2_CHOSEN_LOGPROB_ATOL,
            "nan_or_inf_count": 0,
        },
        "static_state_dict": {
            "status": "PASS" if all(static_checks.values()) else "FAIL",
            "passed": all(static_checks.values()),
            "checks": static_checks,
        },
        "seed41_numerical_compatibility": {
            "status": "PASS" if seed41_passed else "FAIL",
            "passed": seed41_passed,
            "checks": seed41_checks,
            "observed": dict(equivalence),
        },
        "seed42_structural_smoke": {
            "status": "PASS" if seed42_passed else "FAIL",
            "passed": seed42_passed,
            "checks": seed42_checks,
        },
        "passed": passed,
        "status": (
            "W0-v2 PASS: behaviorally identical under frozen FP32 tolerance "
            "and structurally compatible with seed42 topology."
            if passed
            else "W0-v2 FAIL: seed41 numerical compatibility and/or seed42 structural smoke failed."
        ),
    }


def _finite(metrics: Mapping[str, object], field: str) -> float:
    if field not in metrics or metrics[field] is None:
        raise ValueError(f"missing required metric {field}")
    value = float(metrics[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite required metric {field}")
    return value


def _validated_metrics(metrics: Mapping[str, object]) -> dict[str, float | bool]:
    if not bool(metrics.get("ea_metrics_available", False)):
        raise ValueError("expiry-aware metrics are unavailable or pre-correction")
    values: dict[str, float | bool] = {
        field: _finite(metrics, field) for field in _METRIC_FIELDS
    }
    if "independent_deletion_equal" not in metrics:
        raise ValueError("missing required metric independent_deletion_equal")
    values["independent_deletion_equal"] = bool(metrics["independent_deletion_equal"])
    values["ea_metrics_available"] = True
    if values["goodput_mbps"] <= 0.0:
        raise ValueError("goodput_mbps comparator must be positive")
    if values["ue_macro_ea_rate_mbps"] <= 0.0:
        raise ValueError("ue_macro_ea_rate_mbps comparator must be positive")
    if values["measured_tti_last"] < values["measured_tti_first"]:
        raise ValueError("invalid measured TTI interval")
    return values


def compare_manifests(
    reference: Mapping[str, object], treatment: Mapping[str, object]
) -> dict[str, object]:
    """Require source, binary, protocol, and canonical scenario equality."""

    checks: dict[str, bool] = {}
    for field in ("binary_sha256", "source_revision", "protocol_version"):
        if field not in reference or field not in treatment:
            raise ValueError(f"manifest missing {field}")
        checks[field] = reference[field] == treatment[field]
        if not checks[field]:
            raise ValueError(
                f"manifest {field} mismatch: reference={reference[field]!r} "
                f"treatment={treatment[field]!r}"
            )
    if int(reference["protocol_version"]) != VERSION:
        raise ValueError(
            f"manifest protocol_version is pre-correction: "
            f"got={reference['protocol_version']} required={VERSION}"
        )
    if not isinstance(reference.get("scenario"), Mapping) or not isinstance(
        treatment.get("scenario"), Mapping
    ):
        raise ValueError("manifest scenario must be a mapping")
    checks["scenario"] = dict(reference["scenario"]) == dict(treatment["scenario"])
    if not checks["scenario"]:
        raise ValueError("manifest scenario mismatch")
    return {"passed": True, "checks": checks}


def _comparison(
    *,
    numerator: float,
    denominator: float,
    threshold: float,
    passed: bool,
    reason: str,
) -> dict[str, object]:
    absolute_delta = numerator - denominator
    relative_delta = absolute_delta / denominator if denominator != 0.0 else None
    return {
        "numerator": numerator,
        "denominator": denominator,
        "absolute_delta": absolute_delta,
        "relative_delta": relative_delta,
        "threshold": threshold,
        "passed": bool(passed),
        "reason": reason,
    }


def architecture_gate(
    pfq_metrics: Mapping[str, object], actor_metrics: Mapping[str, object]
) -> dict[str, object]:
    pfq = _validated_metrics(pfq_metrics)
    actor = _validated_metrics(actor_metrics)
    if (
        int(pfq["measured_tti_first"]) != int(actor["measured_tti_first"])
        or int(pfq["measured_tti_last"]) != int(actor["measured_tti_last"])
    ):
        raise ValueError("PFQ and actor measured TTI intervals do not match")

    goodput_ratio = float(actor["goodput_mbps"]) / float(pfq["goodput_mbps"])
    bler_delta = float(actor["tb_bler"]) - float(pfq["tb_bler"])
    expiry_delta = float(actor["expiry_rate"]) - float(pfq["expiry_rate"])
    checks = {
        "goodput": _comparison(
            numerator=float(actor["goodput_mbps"]),
            denominator=float(pfq["goodput_mbps"]),
            threshold=GOODPUT_RATIO_MIN,
            passed=goodput_ratio >= GOODPUT_RATIO_MIN,
            reason=f"actor/PFQ goodput={goodput_ratio:.9g} must be >= {GOODPUT_RATIO_MIN}",
        ),
        "tb_bler": _comparison(
            numerator=float(actor["tb_bler"]),
            denominator=float(pfq["tb_bler"]),
            threshold=BLER_ABSOLUTE_DELTA_MAX,
            passed=bler_delta <= BLER_ABSOLUTE_DELTA_MAX + 1.0e-12,
            reason=f"actor-PFQ BLER={bler_delta:.9g} must be <= {BLER_ABSOLUTE_DELTA_MAX}",
        ),
        "expiry_rate": _comparison(
            numerator=float(actor["expiry_rate"]),
            denominator=float(pfq["expiry_rate"]),
            threshold=EXPIRY_ABSOLUTE_DELTA_MAX,
            passed=expiry_delta <= EXPIRY_ABSOLUTE_DELTA_MAX + 1.0e-12,
            reason=f"actor-PFQ expiry={expiry_delta:.9g} must be <= {EXPIRY_ABSOLUTE_DELTA_MAX}",
        ),
        "action_replay": _comparison(
            numerator=float(actor["action_replay_mismatch_count"]),
            denominator=0.0,
            threshold=0.0,
            passed=float(actor["action_replay_mismatch_count"]) == 0.0,
            reason="action replay mismatch count must equal zero",
        ),
        "legality": _comparison(
            numerator=float(actor["illegal_action_count"]),
            denominator=0.0,
            threshold=0.0,
            passed=float(actor["illegal_action_count"]) == 0.0,
            reason="illegal action count must equal zero",
        ),
        "pfq_deletion": {
            "numerator": int(bool(actor["independent_deletion_equal"])),
            "denominator": 1,
            "absolute_delta": int(bool(actor["independent_deletion_equal"])) - 1,
            "relative_delta": 0.0 if bool(actor["independent_deletion_equal"]) else -1.0,
            "threshold": "exact_true",
            "passed": bool(actor["independent_deletion_equal"]),
            "reason": "PFQ-field deletion must preserve graph, logits, trace, and Type-0 action exactly",
        },
    }
    return {
        "passed": all(bool(check["passed"]) for check in checks.values()),
        "checks": checks,
        "measured_tti_first": int(actor["measured_tti_first"]),
        "measured_tti_last": int(actor["measured_tti_last"]),
    }


def _optional_float(mapping: Mapping[str, object], key: str) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def formal_seed42_pair_gate(
    pfq_metrics: Mapping[str, object],
    actor_metrics: Mapping[str, object],
    actor_summary: Mapping[str, object],
    pfq_manifest: Mapping[str, object],
    actor_manifest: Mapping[str, object],
    *,
    frozen_state_dict_sha256: str,
) -> dict[str, object]:
    """Apply the frozen seed42 structural gate before its performance gate."""
    manifest_error: str | None = None
    try:
        manifest_gate = compare_manifests(pfq_manifest, actor_manifest)
    except ValueError as exc:
        manifest_gate = {"passed": False, "checks": {}}
        manifest_error = str(exc)

    pfq_scenario = pfq_manifest.get("scenario")
    actor_scenario = actor_manifest.get("scenario")
    source_state = str(pfq_manifest.get("source_state_sha256", ""))
    structural_checks = {
        "manifest_equivalence": manifest_gate["passed"] is True,
        "source_state_frozen": bool(source_state)
        and actor_manifest.get("source_state_sha256") == source_state,
        "seed42_topology_and_traffic": isinstance(pfq_scenario, Mapping)
        and isinstance(actor_scenario, Mapping)
        and pfq_scenario.get("topology_seed") == 42
        and pfq_scenario.get("traffic_seed") == 42
        and actor_scenario.get("topology_seed") == 42
        and actor_scenario.get("traffic_seed") == 42,
        "formal_horizon": pfq_manifest.get("warmup_steps") == 1000
        and pfq_manifest.get("measure_steps") == 4000
        and actor_manifest.get("warmup_steps") == 1000
        and actor_manifest.get("measure_steps") == 4000,
        "actor_binary_hash": actor_summary.get("sim_binary_sha256")
        == actor_manifest.get("binary_sha256"),
        "completed_steps": actor_summary.get("completed_steps") == 5000,
        "measured_steps": actor_summary.get("stats_count") == 4000,
        "measured_interval": actor_summary.get("measured_tti_first") == 1001
        and actor_summary.get("measured_tti_last") == 5000,
        "association_counts": actor_summary.get("association_candidate_counts")
        == [11, 11, 14],
        "topology_candidate_width": actor_summary.get("topology_candidate_width")
        == 14,
        "runtime_candidate_width": actor_summary.get("runtime_candidate_width")
        == 14,
        "candidate_mode": actor_summary.get("candidate_mode")
        == "all_demand_ues",
        "candidate_width_policy": actor_summary.get("candidate_width_policy")
        == "runtime_topology",
        "candidate_truncation_forbidden": actor_summary.get(
            "candidate_truncation"
        )
        == "forbidden",
        "full_candidate_coverage": actor_summary.get("candidate_full_coverage")
        is True,
        "zero_candidate_truncation": actor_summary.get(
            "candidate_truncation_count"
        )
        == 0
        and actor_summary.get("stats_sum_candidate_truncation_count") == 0.0,
        "zero_replay_mismatch": actor_summary.get(
            "stats_sum_action_replay_mismatch_count"
        )
        == 0.0,
        "zero_illegal_action": actor_summary.get(
            "stats_sum_illegal_action_count"
        )
        == 0.0,
        "pfq_field_deletion_equality": actor_summary.get(
            "independent_deletion_equal"
        )
        is True,
        "state_dict_hash_frozen": actor_summary.get("actor_state_dict_sha256")
        == frozen_state_dict_sha256
        and actor_manifest.get("checkpoint_state_dict_sha256")
        == frozen_state_dict_sha256,
        "training_width_is_metadata": actor_summary.get(
            "training_observed_max_candidates"
        )
        == 13,
    }
    structural_passed = all(structural_checks.values())
    structural_gate = {
        "status": "PASS" if structural_passed else "FAIL",
        "passed": structural_passed,
        "checks": structural_checks,
        "manifest_checks": manifest_gate.get("checks", {}),
        "manifest_error": manifest_error,
    }

    performance: dict[str, object] | None = None
    if structural_passed:
        performance = architecture_gate(pfq_metrics, actor_metrics)
        performance_gate = {
            "status": "PASS" if performance["passed"] else "FAIL",
            "passed": bool(performance["passed"]),
            "gate": performance,
        }
    else:
        performance_gate = {
            "status": "NOT_EVALUATED",
            "passed": False,
            "reason": "structural gate failed",
            "gate": None,
        }

    actor_fill = _optional_float(actor_summary, "stats_mean_fill_ratio")
    diagnostics = {
        "gate_role": "record_only",
        "pfq": {
            "outcome_delay_p95_ms": _optional_float(
                pfq_metrics, "outcome_delay_p95_ms"
            ),
            "ue_p10_ea_rate_mbps": _optional_float(
                pfq_metrics, "ue_p10_ea_rate_mbps"
            ),
            "blank_ratio_mean": _optional_float(pfq_metrics, "blank_ratio_mean"),
            "prg_opportunity_count_mean": _optional_float(
                pfq_metrics, "prg_opportunity_count_mean"
            ),
            "prg_utilization_ratio_mean": _optional_float(
                pfq_metrics, "prg_utilization_ratio_mean"
            ),
            "prg_reuse_ratio_mean": _optional_float(
                pfq_metrics, "prg_reuse_ratio_mean"
            ),
        },
        "actor": {
            "outcome_delay_p95_ms": _optional_float(
                actor_metrics, "outcome_delay_p95_ms"
            ),
            "ue_p10_ea_rate_mbps": _optional_float(
                actor_metrics, "ue_p10_ea_rate_mbps"
            ),
            "blank_ratio_mean": _optional_float(
                actor_summary, "stats_mean_blank_ratio"
            ),
            "prg_opportunity_count_mean": (
                actor_fill * 51.0 if actor_fill is not None else None
            ),
            "prg_utilization_ratio_mean": _optional_float(
                actor_summary, "stats_mean_prg_utilization_ratio"
            ),
            "prg_reuse_ratio_mean": _optional_float(
                actor_summary, "stats_mean_prg_reuse_ratio"
            ),
        },
    }

    if not structural_passed:
        classification = "runtime_compatibility_path_failure"
        status = (
            "Seed42 formal PFQ-B4 pair FAIL: runtime compatibility path failed."
        )
        passed = False
    elif not bool(performance and performance["passed"]):
        classification = "seed42_performance_gate_failure"
        status = (
            "Seed42 formal PFQ-B4 pair FAIL: structural gates pass but one or "
            "more frozen performance gates fail."
        )
        passed = False
    else:
        classification = "seed42_formal_gate_pass"
        status = (
            "Seed42 formal PFQ-B4 pair PASS: frozen B4 passes the first "
            "unseen-seed performance gate."
        )
        passed = True

    return {
        "protocol_id": "seed42-pfq-b4-formal-v1",
        "frozen_state_dict_sha256": frozen_state_dict_sha256,
        "structural_gate": structural_gate,
        "performance_gate": performance_gate,
        "recorded_diagnostics": diagnostics,
        "classification": classification,
        "passed": passed,
        "status": status,
    }


def _ctbs_structural_checks(
    actor_summary: Mapping[str, object],
    *,
    expected_completed_steps: int,
    expected_measured_steps: int,
    expected_measured_first: int,
    expected_measured_last: int,
    frozen_state_dict_sha256: str,
    frozen_binary_sha256: str | None = None,
    expected_association_counts: Sequence[int] | None = (11, 11, 14),
    expected_runtime_width: int | None = 14,
) -> dict[str, bool]:
    first_snapshot = int(actor_summary.get("ctbs_snapshot_id_first", -1))
    last_snapshot = int(actor_summary.get("ctbs_snapshot_id_last", -1))
    snapshot_count = int(actor_summary.get("ctbs_snapshot_count", -1))
    config_hash = actor_summary.get("ctbs_config_sha256")
    vector_fields = (
        "ctbs_mask_trigger_count_by_local_prg",
        "ctbs_selected_count_by_local_prg",
        "ctbs_valid_candidate_count_by_local_prg",
    )
    raw_association_counts = actor_summary.get("association_candidate_counts")
    association_counts = (
        [int(value) for value in raw_association_counts]
        if isinstance(raw_association_counts, (list, tuple))
        else []
    )
    association_ok = (
        association_counts == [int(value) for value in expected_association_counts]
        if expected_association_counts is not None
        else len(association_counts) == 3
        and sum(association_counts) == 36
        and all(value > 0 for value in association_counts)
    )
    resolved_width = (
        int(expected_runtime_width)
        if expected_runtime_width is not None
        else max(association_counts, default=-1)
    )
    checks = {
        "protocol_version": actor_summary.get("protocol_version") == VERSION,
        "completed_steps": actor_summary.get("completed_steps")
        == expected_completed_steps,
        "measured_steps": actor_summary.get("stats_count")
        == expected_measured_steps,
        "measured_interval": actor_summary.get("measured_tti_first")
        == expected_measured_first
        and actor_summary.get("measured_tti_last") == expected_measured_last,
        "association_counts": association_ok,
        "topology_width": actor_summary.get("topology_candidate_width") == resolved_width,
        "runtime_width": actor_summary.get("runtime_candidate_width") == resolved_width,
        "full_candidate_coverage": actor_summary.get("candidate_full_coverage")
        is True,
        "zero_candidate_truncation": actor_summary.get(
            "candidate_truncation_count"
        )
        == 0
        and actor_summary.get("stats_sum_candidate_truncation_count") == 0.0,
        "zero_replay_mismatch": actor_summary.get(
            "stats_sum_action_replay_mismatch_count"
        )
        == 0.0,
        "zero_illegal_action": actor_summary.get(
            "stats_sum_illegal_action_count"
        )
        == 0.0,
        "pfq_field_deletion_equality": actor_summary.get(
            "independent_deletion_equal"
        )
        is True,
        "state_dict_hash_frozen": actor_summary.get("actor_state_dict_sha256")
        == frozen_state_dict_sha256,
        "harq_disabled": actor_summary.get("runtime_harq_enabled") == 0,
        "ctbs_accounting_selected": actor_summary.get(
            "independent_headroom_accounting"
        )
        == "ctbs-v1",
        "pre_action_handshake_enabled": actor_summary.get(
            "ctbs_pre_action_handshake_enabled"
        )
        is True,
        "config_hash_recorded": isinstance(config_hash, str)
        and len(config_hash) == 64,
        "one_snapshot_per_tti": snapshot_count == expected_completed_steps,
        "snapshot_ids_consecutive": first_snapshot > 0
        and last_snapshot - first_snapshot + 1 == snapshot_count,
        "exact_mcs_parity": actor_summary.get("ctbs_mcs_mismatch_count") == 0,
        "exact_layer_parity": actor_summary.get("ctbs_layer_mismatch_count") == 0,
        "exact_tbs_parity": actor_summary.get("ctbs_tbs_mismatch_count") == 0,
        "reference_label_deletion_equality": actor_summary.get(
            "ctbs_reference_label_deletion_equal"
        )
        is True,
        "zero_mask_reactivation": actor_summary.get(
            "ctbs_mask_reactivation_count"
        )
        == 0,
        "per_prg_mechanism_vectors_complete": all(
            isinstance(actor_summary.get(field), list)
            and len(actor_summary[field]) == 17
            for field in vector_fields
        ),
    }
    if frozen_binary_sha256 is not None:
        checks["binary_hash_frozen"] = (
            actor_summary.get("sim_binary_sha256") == frozen_binary_sha256
        )
    return checks


def ctbs_formal_evaluation_gate(
    pfq_metrics: Mapping[str, object],
    actor_metrics: Mapping[str, object],
    actor_summary: Mapping[str, object],
    *,
    actor_state_dict_sha256: str,
    binary_sha256: str,
    expected_association_counts: Sequence[int] | None,
    expected_runtime_width: int | None,
    protocol_id: str,
) -> dict[str, object]:
    """Apply generic causal CTBS and architecture gates to one formal seed."""
    structural_checks = _ctbs_structural_checks(
        actor_summary,
        expected_completed_steps=5000,
        expected_measured_steps=4000,
        expected_measured_first=1001,
        expected_measured_last=5000,
        frozen_state_dict_sha256=actor_state_dict_sha256,
        frozen_binary_sha256=binary_sha256,
        expected_association_counts=expected_association_counts,
        expected_runtime_width=expected_runtime_width,
    )
    structural_passed = all(structural_checks.values())
    performance = architecture_gate(pfq_metrics, actor_metrics) if structural_passed else None
    performance_passed = bool(performance and performance["passed"])
    passed = structural_passed and performance_passed
    return {
        "protocol_id": protocol_id,
        "structural_gate": {
            "status": "PASS" if structural_passed else "FAIL",
            "passed": structural_passed,
            "checks": structural_checks,
        },
        "performance_gate": (
            {"status": "PASS" if performance_passed else "FAIL", **performance}
            if performance is not None
            else {"status": "NOT_EVALUATED", "passed": False}
        ),
        "passed": passed,
        "classification": (
            f"{protocol_id}_pass" if passed else f"{protocol_id}_failure"
        ),
    }


def long_multiseed_seed42_gate(
    pfq_metrics: Mapping[str, object],
    actor_metrics: Mapping[str, object],
    actor_summary: Mapping[str, object],
    service_balance_report: Mapping[str, object],
    *,
    actor_state_dict_sha256: str,
    binary_sha256: str,
) -> dict[str, object]:
    """Gate seed42 before any seed41 or untouched-seed43 evaluation."""
    report = ctbs_formal_evaluation_gate(
        pfq_metrics,
        actor_metrics,
        actor_summary,
        actor_state_dict_sha256=actor_state_dict_sha256,
        binary_sha256=binary_sha256,
        expected_association_counts=(11, 11, 14),
        expected_runtime_width=14,
        protocol_id="long_multiseed_seed42",
    )
    deficits = service_balance_report.get("deficits", {})
    population = service_balance_report.get("population_association_only", {})
    try:
        prg_ratio = float(deficits["actual_prg_count"]["ctbs_over_pfq"])
        actor_target_prgs = float(deficits["actual_prg_count"]["ctbs_overall"])
        pfq_target_prgs = float(deficits["actual_prg_count"]["pfq_overall"])
        actor_cell_prgs = float(population["ctbs_cell2"]["actual_prg_count"])
        pfq_cell_prgs = float(population["pfq_cell2"]["actual_prg_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("service-balance report is missing UE2 PRG allocation fields") from exc
    if min(actor_target_prgs, pfq_target_prgs, actor_cell_prgs, pfq_cell_prgs) <= 0.0:
        raise ValueError("service-balance PRG allocation fields must be positive")
    actor_share = actor_target_prgs / actor_cell_prgs
    pfq_share = pfq_target_prgs / pfq_cell_prgs
    share_ratio = actor_share / pfq_share
    allocation_checks = {
        "ue2_actual_prg_over_pfq_gt_0_90": prg_ratio > 0.90,
        "ue2_cell_share_over_pfq_gt_0_90": share_ratio > 0.90,
    }
    allocation_passed = all(allocation_checks.values())
    report["ue2_allocation_gate"] = {
        "status": "PASS" if allocation_passed else "FAIL",
        "passed": allocation_passed,
        "checks": allocation_checks,
        "observed": {
            "actual_prg_over_pfq": prg_ratio,
            "actor_cell2_prg_share": actor_share,
            "pfq_cell2_prg_share": pfq_share,
            "cell2_prg_share_over_pfq": share_ratio,
        },
    }
    report["passed"] = bool(report["passed"] and allocation_passed)
    report["classification"] = (
        "long_multiseed_seed42_pass"
        if report["passed"]
        else "long_multiseed_seed42_failure"
    )
    return report


def ctbs_seed42_smoke_gate(
    actor_summary: Mapping[str, object],
    *,
    frozen_state_dict_sha256: str,
    frozen_binary_sha256: str,
) -> dict[str, object]:
    """Gate the approved 128-TTI causal B4-CTBS seed42 smoke."""
    checks = _ctbs_structural_checks(
        actor_summary,
        expected_completed_steps=128,
        expected_measured_steps=128,
        expected_measured_first=1,
        expected_measured_last=128,
        frozen_state_dict_sha256=frozen_state_dict_sha256,
        frozen_binary_sha256=frozen_binary_sha256,
    )
    passed = all(checks.values())
    return {
        "protocol_id": "seed42-b4-ctbs-v1-causal-smoke-v3",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "classification": (
            "b4_ctbs_seed42_smoke_pass"
            if passed
            else "b4_ctbs_seed42_smoke_structural_or_causal_failure"
        ),
        "checks": checks,
        "recorded_mechanism_diagnostics": {
            key: actor_summary.get(key)
            for key in (
                "ctbs_raw_selected_service_bytes_sum",
                "ctbs_capacity_decrease_count",
                "ctbs_remaining_rebound_count",
                "ctbs_mask_trigger_count_by_local_prg",
                "ctbs_selected_count_by_local_prg",
                "ctbs_valid_candidate_count_by_local_prg",
            )
        },
    }


def ctbs_seed42_formal_pair_gate(
    pfq_metrics: Mapping[str, object],
    actor_metrics: Mapping[str, object],
    actor_summary: Mapping[str, object],
    pfq_manifest: Mapping[str, object],
    actor_manifest: Mapping[str, object],
    *,
    frozen_state_dict_sha256: str,
) -> dict[str, object]:
    """Apply causal CTBS gates before the unchanged seed42 performance gate."""
    base = formal_seed42_pair_gate(
        pfq_metrics,
        actor_metrics,
        actor_summary,
        pfq_manifest,
        actor_manifest,
        frozen_state_dict_sha256=frozen_state_dict_sha256,
    )
    ctbs_checks = _ctbs_structural_checks(
        actor_summary,
        expected_completed_steps=5000,
        expected_measured_steps=4000,
        expected_measured_first=1001,
        expected_measured_last=5000,
        frozen_state_dict_sha256=frozen_state_dict_sha256,
        frozen_binary_sha256=str(actor_manifest.get("binary_sha256", "")),
    )
    base_checks = dict(base["structural_gate"]["checks"])
    combined_checks = {
        **base_checks,
        **{f"ctbs_{name}": value for name, value in ctbs_checks.items()},
    }
    structural_passed = all(combined_checks.values())
    structural_gate = {
        **base["structural_gate"],
        "status": "PASS" if structural_passed else "FAIL",
        "passed": structural_passed,
        "checks": combined_checks,
    }
    if structural_passed:
        performance_gate = base["performance_gate"]
    else:
        performance_gate = {
            "status": "NOT_EVALUATED",
            "passed": False,
            "reason": "CTBS structural or causal gate failed",
            "gate": None,
        }
    performance_passed = bool(performance_gate.get("passed", False))
    if not structural_passed:
        classification = "ctbs_runtime_or_causal_path_failure"
        status = "Seed42 PFQ-B4-CTBS-v1 pair FAIL: structural or causal CTBS gate failed."
    elif not performance_passed:
        classification = "seed42_ctbs_performance_gate_failure"
        status = "Seed42 PFQ-B4-CTBS-v1 pair FAIL: frozen performance gate failed."
    else:
        classification = "seed42_ctbs_formal_gate_pass"
        status = "Seed42 PFQ-B4-CTBS-v1 pair PASS: the bounded CTBS treatment passes the frozen gate."
    recorded = dict(base["recorded_diagnostics"])
    recorded["ctbs"] = {
        "gate_role": "record_only",
        **{
            key: actor_summary.get(key)
            for key in (
                "ctbs_raw_selected_service_bytes_sum",
                "ctbs_capacity_decrease_count",
                "ctbs_remaining_rebound_count",
                "ctbs_mask_trigger_count_by_local_prg",
                "ctbs_selected_count_by_local_prg",
                "ctbs_valid_candidate_count_by_local_prg",
            )
        },
    }
    return {
        "protocol_id": "seed42-pfq-b4-ctbs-v1-formal-v1",
        "frozen_state_dict_sha256": frozen_state_dict_sha256,
        "structural_gate": structural_gate,
        "performance_gate": performance_gate,
        "recorded_diagnostics": recorded,
        "classification": classification,
        "passed": structural_passed and performance_passed,
        "status": status,
    }


def short_debt_treatment_gate(
    pfq_metrics: Mapping[str, object],
    actor_metrics: Mapping[str, object],
    actor_summary: Mapping[str, object],
    smoke_report: Mapping[str, object],
    service_balance_report: Mapping[str, object],
    *,
    actor_state_dict_sha256: str,
    binary_sha256: str,
) -> dict[str, object]:
    """Gate the one-shot B4-CTBS-C1 short treatment without C0 attribution."""
    structural_checks = {
        "seed42_smoke_passed": smoke_report.get("passed") is True,
        **_ctbs_structural_checks(
            actor_summary,
            expected_completed_steps=5000,
            expected_measured_steps=4000,
            expected_measured_first=1001,
            expected_measured_last=5000,
            frozen_state_dict_sha256=actor_state_dict_sha256,
            frozen_binary_sha256=binary_sha256,
        ),
    }
    structural_passed = all(structural_checks.values())
    performance = architecture_gate(pfq_metrics, actor_metrics) if structural_passed else None

    deficits = service_balance_report.get("deficits", {})
    population = service_balance_report.get("population_association_only", {})
    try:
        prg_ratio = float(deficits["actual_prg_count"]["ctbs_over_pfq"])
        actor_target_prgs = float(deficits["actual_prg_count"]["ctbs_overall"])
        pfq_target_prgs = float(deficits["actual_prg_count"]["pfq_overall"])
        actor_cell_prgs = float(population["ctbs_cell2"]["actual_prg_count"])
        pfq_cell_prgs = float(population["pfq_cell2"]["actual_prg_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("service-balance report is missing UE2 PRG allocation fields") from exc
    if min(actor_target_prgs, pfq_target_prgs, actor_cell_prgs, pfq_cell_prgs) <= 0.0:
        raise ValueError("service-balance PRG allocation fields must be positive")
    actor_share = actor_target_prgs / actor_cell_prgs
    pfq_share = pfq_target_prgs / pfq_cell_prgs
    share_ratio = actor_share / pfq_share
    allocation_checks = {
        "ue2_actual_prg_over_pfq_gt_0_90": prg_ratio > 0.90,
        "ue2_cell_share_over_pfq_gt_0_90": share_ratio > 0.90,
    }
    allocation_recovered = all(allocation_checks.values())
    performance_passed = bool(performance and performance["passed"])
    passed = structural_passed and performance_passed and allocation_recovered
    return {
        "protocol_id": "b4-ctbs-c1-short-treatment-v1",
        "structural_gate": {
            "status": "PASS" if structural_passed else "FAIL",
            "passed": structural_passed,
            "checks": structural_checks,
        },
        "performance_gate": (
            {"status": "PASS" if performance_passed else "FAIL", **performance}
            if performance is not None
            else {"status": "NOT_EVALUATED", "passed": False}
        ),
        "ue2_allocation_gate": {
            "status": "PASS" if allocation_recovered else "FAIL",
            "passed": allocation_recovered,
            "checks": allocation_checks,
            "observed": {
                "actual_prg_over_pfq": prg_ratio,
                "actor_cell2_prg_share": actor_share,
                "pfq_cell2_prg_share": pfq_share,
                "cell2_prg_share_over_pfq": share_ratio,
            },
        },
        "passed": passed,
        "classification": (
            "short_debt_treatment_promising"
            if passed
            else "short_debt_treatment_failure"
        ),
        "interpretation_scope": (
            "The complete C1 treatment package is promising; debt-reward-specific "
            "attribution is not established without a later C0 control."
            if passed
            else "The fixed-budget C1 treatment failed; no extra training or coefficient scan is authorized."
        ),
    }


def score_checkpoint(
    pfq_metrics: Mapping[str, object], actor_metrics: Mapping[str, object]
) -> dict[str, object]:
    pfq = _validated_metrics(pfq_metrics)
    actor = _validated_metrics(actor_metrics)
    goodput_delta = (
        float(actor["goodput_mbps"]) - float(pfq["goodput_mbps"])
    ) / float(pfq["goodput_mbps"])
    ue_delta = (
        float(actor["ue_macro_ea_rate_mbps"])
        - float(pfq["ue_macro_ea_rate_mbps"])
    ) / float(pfq["ue_macro_ea_rate_mbps"])
    worse_delay = max(
        0.0,
        (
            float(actor["outcome_delay_p95_ms"])
            - float(pfq["outcome_delay_p95_ms"])
        )
        / max(float(pfq["outcome_delay_p95_ms"]), 1.0e-12),
    )
    score = goodput_delta + SCORE_UE_WEIGHT * ue_delta - SCORE_DELAY_WEIGHT * worse_delay
    return {
        "score": score,
        "components": {
            "relative_goodput_delta": goodput_delta,
            "relative_ue_ea_rate_delta": ue_delta,
            "worse_outcome_delay_relative": worse_delay,
        },
        "weights": {
            "ue_ea_rate": SCORE_UE_WEIGHT,
            "worse_outcome_delay": SCORE_DELAY_WEIGHT,
        },
    }


def select_treatment_checkpoints(
    pfq_metrics: Mapping[str, object],
    candidates: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    _validated_metrics(pfq_metrics)
    reports: dict[str, object] = {}
    selected_iteration: int | None = None
    selected_score = -math.inf
    first_gate_iteration: int | None = None
    for iteration in CHECKPOINT_ITERATIONS:
        metrics = candidates.get(iteration)
        if metrics is None:
            reports[str(iteration)] = {
                "present": False,
                "scored": False,
                "reason": "candidate evaluation is missing",
            }
            continue
        gate = architecture_gate(pfq_metrics, metrics)
        entry: dict[str, object] = {
            "present": True,
            "scored": False,
            "architecture_gate": gate,
            "metrics": dict(metrics),
        }
        if gate["passed"]:
            if first_gate_iteration is None:
                first_gate_iteration = iteration
            scored = score_checkpoint(pfq_metrics, metrics)
            entry.update(scored)
            entry["scored"] = True
            if float(scored["score"]) > selected_score:
                selected_score = float(scored["score"])
                selected_iteration = iteration
        else:
            entry["reason"] = "hard architecture/replay/legality gate failed"
        reports[str(iteration)] = entry
    selected_metrics = (
        dict(candidates[selected_iteration]) if selected_iteration is not None else None
    )
    return {
        "checkpoint_cadence_iterations": list(CHECKPOINT_ITERATIONS),
        "measured_ppo_ttis_per_iteration": PPO_TTIS_PER_ITERATION,
        "candidates": reports,
        "selected_iteration": selected_iteration,
        "selected_score": selected_score if selected_iteration is not None else None,
        "selected_metrics": selected_metrics,
        "first_gate_iteration": first_gate_iteration,
        "first_gate_measured_ppo_ttis": (
            first_gate_iteration * PPO_TTIS_PER_ITERATION
            if first_gate_iteration is not None
            else None
        ),
    }


def _auxiliary_preference(
    treatment: str,
    pfq_metrics: Mapping[str, object],
    selection: Mapping[str, object],
) -> dict[str, object]:
    metrics = selection.get("selected_metrics")
    if metrics is None:
        return {"passed": False, "reason": "no architecture-gate-passing checkpoint"}
    gate = architecture_gate(pfq_metrics, metrics)
    pfq = _validated_metrics(pfq_metrics)
    actor = _validated_metrics(metrics)
    checks = {
        "architecture_gate": bool(gate["passed"]),
        "outcome_delay_p95_no_worse": float(actor["outcome_delay_p95_ms"])
        <= float(pfq["outcome_delay_p95_ms"]),
        "ue_macro_ea_rate_no_worse": float(actor["ue_macro_ea_rate_mbps"])
        >= float(pfq["ue_macro_ea_rate_mbps"]),
        "ue_p10_ea_rate_no_worse": float(actor["ue_p10_ea_rate_mbps"])
        >= float(pfq["ue_p10_ea_rate_mbps"]),
    }
    return {
        "passed": treatment in ("C", "D") and all(checks.values()),
        "checks": checks,
        "reason": (
            "auxiliary treatment meets PFQ secondary KPIs"
            if treatment in ("C", "D") and all(checks.values())
            else "auxiliary treatment is absent or fails at least one PFQ secondary KPI"
        ),
    }


def compare_treatments(
    pfq_metrics: Mapping[str, object],
    selections: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    _validated_metrics(pfq_metrics)
    treatment_reports: dict[str, object] = {}
    eligible: list[tuple[float, int, str]] = []
    simplicity = {"B": 0, "A": 1, "C": 2, "D": 3}
    for treatment in ("A", "B", "C", "D"):
        if treatment not in selections:
            continue
        selection = selections[treatment]
        aux = _auxiliary_preference(treatment, pfq_metrics, selection)
        has_gate = selection.get("selected_iteration") is not None
        allowed = has_gate and (treatment in ("A", "B") or bool(aux["passed"]))
        treatment_reports[treatment] = {
            "selected_iteration": selection.get("selected_iteration"),
            "selected_score": selection.get("selected_score"),
            "first_gate_measured_ppo_ttis": selection.get(
                "first_gate_measured_ppo_ttis"
            ),
            "architecture_gate_passed": has_gate,
            "auxiliary_preference_eligible": bool(aux["passed"]),
            "auxiliary_preference": aux,
            "winner_eligible": allowed,
        }
        if allowed:
            eligible.append(
                (
                    float(selection["selected_score"]),
                    -simplicity[treatment],
                    treatment,
                )
            )

    a_ttis = selections.get("A", {}).get("first_gate_measured_ppo_ttis")
    b_ttis = selections.get("B", {}).get("first_gate_measured_ppo_ttis")
    h2_supported = b_ttis is not None and (a_ttis is None or int(b_ttis) < int(a_ttis))
    if b_ttis is not None and a_ttis is not None and int(b_ttis) == int(a_ttis):
        h2_reason = (
            "A and B first pass at the same checkpoint; no demonstrated benefit "
            "at the available 2048-TTI resolution"
        )
    elif h2_supported:
        h2_reason = "B reaches the H1 gate earlier than A, or A does not reach it within budget"
    else:
        h2_reason = "PFQ BC did not demonstrate a lower measured PPO interaction count to H1"

    selected_treatment = max(eligible)[2] if eligible else None
    return {
        "treatments": treatment_reports,
        "selected_treatment": selected_treatment,
        "launch_paired_seeds_41_43": selected_treatment is not None,
        "h1": {
            "supported": selected_treatment is not None,
            "reason": (
                "at least one independent treatment passes the seed-41 architecture gate"
                if selected_treatment is not None
                else "no independent treatment passes the seed-41 architecture gate"
            ),
        },
        "h2": {
            "supported": h2_supported,
            "a_first_gate_measured_ppo_ttis": a_ttis,
            "b_first_gate_measured_ppo_ttis": b_ttis,
            "reason": h2_reason,
        },
    }


def load_outcome_delay_p95(
    path: Path, *, measured_tti_first: int, measured_tti_last: int
) -> dict[str, object]:
    delays: list[float] = []
    terminal_ttis: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            terminal_tti = int(row["terminal_tti"])
            if int(measured_tti_first) <= terminal_tti <= int(measured_tti_last):
                delays.append(float(row["system_time_ms"]))
                terminal_ttis.append(terminal_tti)
    if not delays:
        raise ValueError("no packet terminal samples in the measured TTI interval")
    return {
        "sample_count": len(delays),
        "terminal_tti_first": min(terminal_ttis),
        "terminal_tti_last": max(terminal_ttis),
        "measured_tti_first": int(measured_tti_first),
        "measured_tti_last": int(measured_tti_last),
        "outcome_delay_p95_ms": float(np.percentile(delays, 95)),
    }


def normalize_gate_metrics(summary: Mapping[str, object]) -> dict[str, object]:
    """Convert a deterministic evaluator summary to the locked gate schema."""

    if int(summary.get("protocol_version", -1)) != VERSION:
        raise ValueError("evaluator summary is pre-correction or has the wrong protocol_version")
    ea_ready = (
        summary.get("packet_outcome_delay_p95_ms") is not None
        and float(summary.get("packet_outcome_measured_count", 0.0)) > 0.0
        and float(summary.get("ue_expiry_aware_rate_active_count", 0.0)) > 0.0
    )
    if not ea_ready:
        raise ValueError("expiry-aware evaluator metrics are missing or empty")
    measured_count = _finite(summary, "stats_count")
    if measured_count <= 0.0:
        raise ValueError("evaluator measured window is empty")
    mapping = {
        "goodput_mbps": "stats_mean_goodput_mbps",
        "tb_bler": "measured_tb_bler",
        "outcome_delay_p95_ms": "packet_outcome_delay_p95_ms",
        "ue_macro_ea_rate_mbps": "ue_macro_expiry_aware_rate_mbps",
        "ue_p10_ea_rate_mbps": "ue_p10_expiry_aware_rate_mbps",
        "action_replay_mismatch_count": "stats_sum_action_replay_mismatch_count",
        "illegal_action_count": "stats_sum_illegal_action_count",
        "measured_tti_first": "measured_tti_first",
        "measured_tti_last": "measured_tti_last",
    }
    normalized: dict[str, object] = {
        target: _finite(summary, source) for target, source in mapping.items()
    }
    expired_bytes = _finite(summary, "stats_sum_expired_bytes")
    normalized["expiry_rate"] = expired_bytes / (measured_count * 108_000.0)
    normalized["measured_tti_first"] = int(normalized["measured_tti_first"])
    normalized["measured_tti_last"] = int(normalized["measured_tti_last"])
    normalized["independent_deletion_equal"] = bool(
        summary.get("independent_deletion_equal", False)
    )
    normalized["ea_metrics_available"] = True
    _validated_metrics(normalized)
    return normalized


def _load_json(path: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_named_paths(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        parsed[name] = path
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select-checkpoints")
    select.add_argument("--pfq-metrics", required=True)
    select.add_argument("--pfq-manifest", required=True)
    select.add_argument("--treatment-manifest", required=True)
    select.add_argument("--candidate", action="append", default=[], help="ITER=eval_summary.json")
    select.add_argument("--out", required=True)

    compare = subparsers.add_parser("compare-treatments")
    compare.add_argument("--pfq-metrics", required=True)
    compare.add_argument("--selection", action="append", default=[], help="TREATMENT=selection.json")
    compare.add_argument("--out-dir", required=True)
    normalize = subparsers.add_parser("normalize-eval")
    normalize.add_argument("--summary", required=True)
    normalize.add_argument("--out", required=True)
    width = subparsers.add_parser("candidate-width-w0-v2")
    width.add_argument("--static", required=True)
    width.add_argument("--seed41", required=True)
    width.add_argument("--seed42", required=True)
    width.add_argument("--expected-steps", type=int, required=True)
    width.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "normalize-eval":
        _write_json(Path(args.out), normalize_gate_metrics(_load_json(args.summary)))
        return 0
    if args.command == "candidate-width-w0-v2":
        report = candidate_width_w0_v2_gate(
            _load_json(args.static),
            _load_json(args.seed41),
            _load_json(args.seed42),
            expected_steps=int(args.expected_steps),
        )
        _write_json(Path(args.out), report)
        return 0 if bool(report["passed"]) else 2
    if args.command == "select-checkpoints":
        compare_manifests(_load_json(args.pfq_manifest), _load_json(args.treatment_manifest))
        candidate_paths = _parse_named_paths(args.candidate)
        candidates = {int(key): _load_json(value) for key, value in candidate_paths.items()}
        report = select_treatment_checkpoints(_load_json(args.pfq_metrics), candidates)
        _write_json(Path(args.out), report)
        return 0
    selections = {
        key: _load_json(value) for key, value in _parse_named_paths(args.selection).items()
    }
    pfq_metrics = _load_json(args.pfq_metrics)
    report = compare_treatments(pfq_metrics, selections)
    out_dir = Path(args.out_dir)
    _write_json(out_dir / "checkpoint_selection.json", selections)
    _write_json(out_dir / "treatment_comparison.json", report)
    selected = report["selected_treatment"]
    winner = selections[str(selected)] if selected is not None else None
    gate = (
        architecture_gate(pfq_metrics, winner["selected_metrics"])
        if winner is not None
        else {"passed": False, "reason": "no treatment selected"}
    )
    _write_json(out_dir / "seed41_gate.json", gate)
    return 0 if bool(gate["passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())

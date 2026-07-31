"""Fail-closed gate for the Stage-B PFQ-residual and tail-credit smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


_INITIAL_RESIDUAL_TOLERANCE = 1e-8
_PRG_TOKENS_PER_TTI = 3 * 17
_PYTHON_REJECT_FIELDS = (
    "rollout_python_preflight_wrong_cell_slot_count_mean",
    "rollout_python_preflight_empty_slot_count_mean",
    "rollout_python_preflight_prg_mask_count_mean",
    "rollout_python_preflight_oob_slot_count_mean",
    "rollout_python_preflight_duplicate_ue_count_mean",
)
_NATIVE_REJECT_FIELDS = (
    "rollout_native_reject_wrong_cell_slot_count_mean",
    "rollout_native_reject_empty_slot_count_mean",
    "rollout_native_reject_prg_mask_count_mean",
    "rollout_native_reject_oob_slot_count_mean",
    "rollout_native_reject_duplicate_ue_count_mean",
)
_QUEUE_FIELDS = (
    "rollout_queue_pressure_before_mean",
    "rollout_queue_pressure_after_mean",
    "rollout_queue_pressure_delta_mean",
    "rollout_queue_pressure_reward_mean",
)


def _finite(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value: {value!r}")
    return number


def _nonnegative(value: object) -> float:
    number = _finite(value)
    if number < 0.0:
        raise ValueError(f"negative value: {value!r}")
    return number


def _exact_int(value: object) -> int:
    number = _finite(value)
    if not number.is_integer():
        raise ValueError(f"expected an integer, got {value!r}")
    return int(number)


def _all_rows_have(
    rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> bool:
    return bool(rows) and all(
        field in row and str(row[field]).strip() != ""
        for row in rows
        for field in fields
    )


def _config_matches(summary: Mapping[str, object]) -> bool:
    try:
        return (
            bool(_exact_int(summary["pfq_residual_init"]))
            and _exact_int(summary["ue_feat_dim"]) == 16
            and _finite(summary["candidate_pfq_anchor_logit_coef"]) == 1.0
            and _finite(summary["candidate_blank_logit_bias"]) == -4.0
            and _finite(summary["candidate_success_logit_scale"]) == 0.0
            and _finite(summary["candidate_blank_success_scale"]) == 0.0
            and _finite(summary["queue_pressure_weight"]) == 0.25
            and _exact_int(summary["queue_pressure_topk"]) == 8
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def evaluate_residual_tail_gate(
    summary: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Evaluate residual initialization, dense credit, and runtime alignment."""

    errors: list[str] = []
    evidence: dict[str, object] = {
        "history_rows": len(rows),
        "initial_residual_tolerance": _INITIAL_RESIDUAL_TOLERANCE,
    }

    try:
        iterations = [_exact_int(row["iter"]) for row in rows]
        two_iterations = (
            _exact_int(summary["iterations"]) == 2
            and _exact_int(summary["completed_iterations"]) == 2
            and iterations == [1, 2]
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        iterations = []
        two_iterations = False
        errors.append(f"invalid iteration provenance: {exc}")
    evidence["history_iterations"] = iterations

    initial_residual = math.nan
    initial_pfq_base = math.nan
    residual_updated = False
    try:
        initial_residual = _nonnegative(
            rows[0]["rollout_candidate_residual_logits_abs_mean"]
        )
        initial_pfq_base = _nonnegative(
            rows[0]["rollout_candidate_pfq_base_logits_abs_mean"]
        )
        later_residuals = [
            _nonnegative(row["rollout_candidate_residual_logits_abs_mean"])
            for row in rows[1:]
        ]
        residual_updated = any(
            value > _INITIAL_RESIDUAL_TOLERANCE for value in later_residuals
        )
        evidence["later_residual_abs_mean"] = later_residuals
    except (IndexError, KeyError, TypeError, ValueError, OverflowError) as exc:
        errors.append(f"invalid residual evidence: {exc}")
    evidence["initial_residual_abs_mean"] = (
        initial_residual if math.isfinite(initial_residual) else None
    )
    evidence["initial_pfq_base_abs_mean"] = (
        initial_pfq_base if math.isfinite(initial_pfq_base) else None
    )

    gradients_positive = False
    entropy_positive = False
    try:
        gradients = [_nonnegative(row["candidate_policy_grad_norm"]) for row in rows]
        entropies = [_nonnegative(row["rollout_entropy_mean"]) for row in rows]
        gradients_positive = bool(gradients) and all(value > 0.0 for value in gradients)
        entropy_positive = bool(entropies) and all(value > 0.0 for value in entropies)
        evidence["candidate_policy_grad_norms"] = gradients
        evidence["candidate_policy_entropies"] = entropies
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        errors.append(f"invalid optimizer evidence: {exc}")

    queue_metrics_finite = False
    queue_reward_consistent = False
    queue_credit_observed = False
    try:
        if not _all_rows_have(rows, _QUEUE_FIELDS):
            raise ValueError("queue-pressure fields are missing")
        queue_values = [
            {field: _finite(row[field]) for field in _QUEUE_FIELDS} for row in rows
        ]
        weight = _finite(summary["queue_pressure_weight"])
        queue_metrics_finite = True
        queue_reward_consistent = all(
            math.isclose(
                values["rollout_queue_pressure_reward_mean"],
                weight * values["rollout_queue_pressure_delta_mean"],
                rel_tol=1e-5,
                abs_tol=1e-9,
            )
            for values in queue_values
        )
        queue_credit_observed = any(
            values["rollout_queue_pressure_reward_mean"] > 0.0
            for values in queue_values
        )
        evidence["queue_pressure"] = queue_values
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        errors.append(f"invalid queue-pressure evidence: {exc}")

    direct_only = False
    override_below_limit = False
    try:
        direct_ratios = [
            _nonnegative(row["rollout_candidate_policy_direct_ratio"]) for row in rows
        ]
        direct_only = bool(direct_ratios) and all(value == 1.0 for value in direct_ratios)
        weights = [_nonnegative(row["rollout_steps"]) for row in rows]
        if not weights or sum(weights) <= 0.0:
            raise ValueError("rollout_steps must sum to a positive value")
        overrides = [
            _nonnegative(row["rollout_non_hard_override_count_mean"]) for row in rows
        ]
        override_ratio = sum(
            count * weight for count, weight in zip(overrides, weights)
        ) / (sum(weights) * _PRG_TOKENS_PER_TTI)
        override_below_limit = override_ratio < 0.05
        evidence["candidate_policy_direct_ratios"] = direct_ratios
        evidence["non_hard_override_ratio"] = override_ratio
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        errors.append(f"invalid direct-policy evidence: {exc}")

    python_reject_zero = False
    native_reject_zero = False
    replay_exact = False
    diagnostics_available = False
    try:
        required = (
            *_PYTHON_REJECT_FIELDS,
            *_NATIVE_REJECT_FIELDS,
            "rollout_python_preflight_diagnostics_available_mean",
            "rollout_native_reject_diagnostics_available_mean",
            "rollout_action_replay_mismatch_count_mean",
        )
        if not _all_rows_have(rows, required):
            raise ValueError("reject/replay fields are missing")
        python_rejects = sum(
            _nonnegative(row[field]) for row in rows for field in _PYTHON_REJECT_FIELDS
        )
        native_rejects = sum(
            _nonnegative(row[field]) for row in rows for field in _NATIVE_REJECT_FIELDS
        )
        replay_mismatches = sum(
            _nonnegative(row["rollout_action_replay_mismatch_count_mean"])
            for row in rows
        )
        diagnostics_available = all(
            _finite(row["rollout_python_preflight_diagnostics_available_mean"]) == 1.0
            and _finite(row["rollout_native_reject_diagnostics_available_mean"]) == 1.0
            for row in rows
        )
        python_reject_zero = python_rejects == 0.0
        native_reject_zero = native_rejects == 0.0
        replay_exact = replay_mismatches == 0.0
        evidence["python_preflight_reject_sum"] = python_rejects
        evidence["native_reject_sum"] = native_rejects
        evidence["action_replay_mismatch_sum"] = replay_mismatches
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        errors.append(f"invalid reject/replay evidence: {exc}")

    checks = {
        "two_iteration_smoke_complete": two_iterations,
        "candidate_policy_mode_direct": summary.get("candidate_policy_mode") == "direct",
        "residual_config_exact": _config_matches(summary),
        "initial_residual_near_zero": (
            math.isfinite(initial_residual)
            and initial_residual <= _INITIAL_RESIDUAL_TOLERANCE
        ),
        "initial_pfq_base_positive": (
            math.isfinite(initial_pfq_base) and initial_pfq_base > 0.0
        ),
        "residual_updated_after_first_iteration": residual_updated,
        "candidate_gradient_positive": gradients_positive,
        "candidate_entropy_positive": entropy_positive,
        "queue_metrics_finite": queue_metrics_finite,
        "queue_reward_matches_pressure_delta": queue_reward_consistent,
        "positive_queue_credit_observed": queue_credit_observed,
        "candidate_policy_direct_only": direct_only,
        "non_hard_override_below_5pct": override_below_limit,
        "diagnostics_fully_available": diagnostics_available,
        "python_preflight_zero": python_reject_zero,
        "native_reject_zero": native_reject_zero,
        "action_replay_exact": replay_exact,
    }
    return {
        "checks": checks,
        "errors": errors,
        "evidence": evidence,
        "run_config": {
            key: summary.get(key)
            for key in (
                "candidate_policy_mode",
                "pfq_residual_init",
                "ue_feat_dim",
                "candidate_pfq_anchor_logit_coef",
                "candidate_blank_logit_bias",
                "candidate_success_logit_scale",
                "candidate_blank_success_scale",
                "queue_pressure_weight",
                "queue_pressure_topk",
            )
        },
        "passed": not errors and all(checks.values()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_report(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        with args.summary.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if not isinstance(summary, dict):
            raise ValueError("summary JSON must contain an object")
        with args.history.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        report = evaluate_residual_tail_gate(summary, rows)
    except (OSError, csv.Error, json.JSONDecodeError, ValueError) as exc:
        report = {
            "checks": {},
            "errors": [str(exc)],
            "evidence": {},
            "run_config": {},
            "passed": False,
        }
    try:
        report["provenance"] = {
            "history_path": str(args.history),
            "history_sha256": _sha256(args.history),
            "source_revision": args.source_revision,
            "summary_path": str(args.summary),
            "summary_sha256": _sha256(args.summary),
        }
    except OSError as exc:
        report["errors"].append(str(exc))
        report["passed"] = False
    _write_report(args.out, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

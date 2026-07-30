"""Fail-closed causal action-alignment gate for Stage-B online PPO runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import fmean
from typing import Mapping, Sequence


_EVIDENCE_FIELDS = (
    "candidate_policy_grad_norm",
    "candidate_policy_entropy",
    "non_hard_override_ratio",
    "python_preflight_reject_count",
    "native_reject_count",
    "action_replay_mismatch_count",
)
_PYTHON_REJECT_MEAN_FIELDS = (
    "rollout_python_preflight_wrong_cell_slot_count_mean",
    "rollout_python_preflight_empty_slot_count_mean",
    "rollout_python_preflight_prg_mask_count_mean",
    "rollout_python_preflight_oob_slot_count_mean",
    "rollout_python_preflight_duplicate_ue_count_mean",
)
_NATIVE_REJECT_MEAN_FIELDS = (
    "rollout_native_reject_wrong_cell_slot_count_mean",
    "rollout_native_reject_empty_slot_count_mean",
    "rollout_native_reject_prg_mask_count_mean",
    "rollout_native_reject_oob_slot_count_mean",
    "rollout_native_reject_duplicate_ue_count_mean",
)
_PRG_TOKENS_PER_TTI = 3 * 17
_RUN_CONFIG_FIELDS = (
    "iterations",
    "completed_iterations",
    "rollout_steps",
    "candidate_policy_mode",
    "action_head_mode",
    "candidate_mode",
    "max_prg_candidates",
    "candidate_budgeted_select_mode",
    "candidate_budgeted_hard_cap",
    "device",
    "actor_last",
    "actor_best",
    "actor_absolute_best",
    "actor_threshold_best",
)


def _finite_float(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value: {value!r}")
    return number


def _nonnegative_float(value: object) -> float:
    number = _finite_float(value)
    if number < 0.0:
        raise ValueError(f"negative value: {value!r}")
    return number


def _positive_float(value: object) -> float:
    number = _finite_float(value)
    if number <= 0.0:
        raise ValueError(f"non-positive value: {value!r}")
    return number


def _finite_decimal(value: object) -> Decimal:
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"invalid decimal value: {value!r}") from exc
    if not number.is_finite():
        raise ValueError(f"non-finite value: {value!r}")
    return number


def _nonnegative_decimal(value: object) -> Decimal:
    number = _finite_decimal(value)
    if number < 0:
        raise ValueError(f"negative value: {value!r}")
    return number


def evaluate_causal_gate(summary: Mapping[str, float]) -> dict[str, object]:
    """Evaluate the six causal alignment checks without defaulting evidence."""

    values: dict[str, float] = {}
    evidence: dict[str, float | None] = {}
    missing_evidence: list[str] = []
    invalid_evidence: list[str] = []
    for field in _EVIDENCE_FIELDS:
        if field not in summary:
            missing_evidence.append(field)
            evidence[field] = None
            values[field] = math.nan
            continue
        try:
            values[field] = _nonnegative_float(summary[field])
            evidence[field] = values[field]
        except (TypeError, ValueError, OverflowError):
            invalid_evidence.append(field)
            evidence[field] = None
            values[field] = math.nan

    grad = values["candidate_policy_grad_norm"]
    entropy = values["candidate_policy_entropy"]
    override_ratio = values["non_hard_override_ratio"]
    python_reject = values["python_preflight_reject_count"]
    native_reject = values["native_reject_count"]
    replay_mismatch = values["action_replay_mismatch_count"]
    checks = {
        "candidate_gradient_nonzero": math.isfinite(grad) and grad > 0.0,
        "training_entropy_positive": math.isfinite(entropy) and entropy > 0.0,
        "non_hard_override_below_5pct": (
            math.isfinite(override_ratio) and override_ratio < 0.05
        ),
        "python_preflight_zero": math.isfinite(python_reject)
        and python_reject == 0.0,
        "native_reject_zero": math.isfinite(native_reject)
        and native_reject == 0.0,
        "action_replay_exact": math.isfinite(replay_mismatch)
        and replay_mismatch == 0.0,
    }
    return {
        "checks": checks,
        "evidence": evidence,
        "invalid_evidence": invalid_evidence,
        "missing_evidence": missing_evidence,
        "passed": not missing_evidence
        and not invalid_evidence
        and all(checks.values()),
    }


def _row_weights(rows: Sequence[Mapping[str, object]]) -> list[float]:
    return [_positive_float(row["rollout_steps"]) for row in rows]


def _mean_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return fmean(_nonnegative_float(row[field]) for row in rows)


def _weighted_mean_field(
    rows: Sequence[Mapping[str, object]], field: str, weights: Sequence[float]
) -> float:
    denominator = sum(weights)
    if denominator <= 0.0:
        raise ValueError("rollout_steps must sum to a positive value")
    return (
        sum(
            _nonnegative_float(row[field]) * weight
            for row, weight in zip(rows, weights)
        )
        / denominator
    )


def _weighted_count_from_means(
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
    weights: Sequence[float],
) -> float:
    return sum(
        sum(_nonnegative_float(row[field]) for field in fields) * weight
        for row, weight in zip(rows, weights)
    )


def _all_rows_have(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bool:
    return bool(rows) and all(
        field in row and str(row[field]).strip() != ""
        for row in rows
        for field in fields
    )


def _aggregate_history(
    _summary: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, float], list[str]]:
    evidence: dict[str, float] = {}
    errors: list[str] = []
    if not rows:
        return evidence, ["history contains no metric rows"]

    try:
        weights = _row_weights(rows)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        weights = []
        errors.append(f"invalid rollout_steps: {exc}")

    if _all_rows_have(rows, ("candidate_policy_grad_norm",)):
        try:
            evidence["candidate_policy_grad_norm"] = _mean_field(
                rows, "candidate_policy_grad_norm"
            )
        except (TypeError, ValueError, OverflowError) as exc:
            evidence["candidate_policy_grad_norm"] = math.nan
            errors.append(f"invalid candidate_policy_grad_norm: {exc}")

    if _all_rows_have(rows, ("rollout_entropy_mean",)):
        if weights:
            try:
                evidence["candidate_policy_entropy"] = _weighted_mean_field(
                    rows, "rollout_entropy_mean", weights
                )
            except (TypeError, ValueError, OverflowError) as exc:
                evidence["candidate_policy_entropy"] = math.nan
                errors.append(f"invalid rollout_entropy_mean: {exc}")
    elif _all_rows_have(rows, ("entropy_mean",)):
        try:
            evidence["candidate_policy_entropy"] = _mean_field(
                rows, "entropy_mean"
            )
        except (TypeError, ValueError, OverflowError) as exc:
            evidence["candidate_policy_entropy"] = math.nan
            errors.append(f"invalid entropy_mean: {exc}")

    if weights:
        if _all_rows_have(rows, ("rollout_non_hard_override_count_mean",)):
            try:
                decimal_weights = [_finite_decimal(weight) for weight in weights]
                override_count = sum(
                    _nonnegative_decimal(
                        row["rollout_non_hard_override_count_mean"]
                    )
                    * weight
                    for row, weight in zip(rows, decimal_weights)
                )
                opportunities = sum(decimal_weights) * Decimal(_PRG_TOKENS_PER_TTI)
                evidence["non_hard_override_ratio"] = float(
                    override_count / opportunities
                )
            except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
                evidence["non_hard_override_ratio"] = math.nan
                errors.append(f"invalid non-hard override evidence: {exc}")

        count_sources = (
            ("python_preflight_reject_count", _PYTHON_REJECT_MEAN_FIELDS),
            ("native_reject_count", _NATIVE_REJECT_MEAN_FIELDS),
            (
                "action_replay_mismatch_count",
                ("rollout_action_replay_mismatch_count_mean",),
            ),
        )
        for target, actual_mean_fields in count_sources:
            if not _all_rows_have(rows, actual_mean_fields):
                continue
            try:
                evidence[target] = _weighted_count_from_means(
                    rows, actual_mean_fields, weights
                )
            except (TypeError, ValueError, OverflowError) as exc:
                evidence[target] = math.nan
                errors.append(f"invalid {target}: {exc}")

    availability_field = "rollout_native_reject_diagnostics_available_mean"
    if "native_reject_count" in evidence:
        if not _all_rows_have(rows, (availability_field,)):
            evidence["native_reject_count"] = math.nan
            errors.append("native reject diagnostic availability is missing")
        else:
            try:
                availability = [
                    _finite_float(row[availability_field]) for row in rows
                ]
                if any(value != 1.0 for value in availability):
                    evidence["native_reject_count"] = math.nan
                    errors.append(
                        "native reject diagnostics were unavailable for part of the run"
                    )
            except (TypeError, ValueError, OverflowError) as exc:
                evidence["native_reject_count"] = math.nan
                errors.append(f"invalid native reject diagnostic availability: {exc}")

    availability_field = "rollout_python_preflight_diagnostics_available"
    if "python_preflight_reject_count" in evidence:
        if not _all_rows_have(rows, (availability_field,)):
            evidence["python_preflight_reject_count"] = math.nan
            errors.append("Python preflight diagnostic availability is missing")
        else:
            try:
                availability = [
                    _finite_float(row[availability_field]) for row in rows
                ]
                if any(value != 1.0 for value in availability):
                    evidence["python_preflight_reject_count"] = math.nan
                    errors.append(
                        "Python preflight diagnostics were unavailable for part of the run"
                    )
            except (TypeError, ValueError, OverflowError) as exc:
                evidence["python_preflight_reject_count"] = math.nan
                errors.append(
                    f"invalid Python preflight diagnostic availability: {exc}"
                )

    return evidence, errors


def _exact_int(value: object) -> int:
    number = _finite_float(value)
    if not number.is_integer():
        raise ValueError(f"expected an integer, got {value!r}")
    return int(number)


def _run_provenance(
    summary: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, bool], list[str]]:
    run_config = {field: summary.get(field) for field in _RUN_CONFIG_FIELDS}
    errors: list[str] = []
    try:
        history_iterations = [_exact_int(row["iter"]) for row in rows]
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        history_iterations = []
        errors.append(f"invalid history iteration provenance: {exc}")
    try:
        topology_seeds = sorted(
            {_exact_int(row["topology_seed"]) for row in rows}
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        topology_seeds = []
    run_config["history_iterations"] = history_iterations
    run_config["history_rows"] = len(rows)
    run_config["topology_seeds"] = topology_seeds

    mode_direct = summary.get("candidate_policy_mode") == "direct"
    if not mode_direct:
        errors.append("candidate_policy_mode must be direct")
    try:
        two_iterations_complete = (
            _exact_int(summary["iterations"]) == 2
            and _exact_int(summary["completed_iterations"]) == 2
            and history_iterations == [1, 2]
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        two_iterations_complete = False
        errors.append(f"invalid two-iteration provenance: {exc}")
    if not two_iterations_complete and not any(
        error.startswith("invalid two-iteration provenance") for error in errors
    ):
        errors.append(
            "run must request and complete exactly two iterations with history [1, 2]"
        )
    checks = {
        "candidate_policy_mode_direct": mode_direct,
        "two_iterations_complete": two_iterations_complete,
    }
    return run_config, checks, errors


def _read_history(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_report(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    input_errors: list[str] = []
    evidence: dict[str, float] = {}
    summary: dict[str, object] = {}
    rows: list[dict[str, str]] = []
    try:
        with args.summary.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if not isinstance(summary, dict):
            raise ValueError("summary JSON must contain an object")
        rows = _read_history(args.history)
        evidence, input_errors = _aggregate_history(summary, rows)
    except (OSError, csv.Error, json.JSONDecodeError, ValueError) as exc:
        input_errors = [str(exc)]

    report = evaluate_causal_gate(evidence)
    report["aggregation_errors"] = input_errors
    run_config, provenance_checks, provenance_errors = _run_provenance(
        summary, rows
    )
    report["provenance_checks"] = provenance_checks
    report["provenance_errors"] = provenance_errors
    report["run_config"] = run_config
    report["passed"] = bool(
        report["passed"] and not input_errors and all(provenance_checks.values())
    )
    _write_report(args.out, report)
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

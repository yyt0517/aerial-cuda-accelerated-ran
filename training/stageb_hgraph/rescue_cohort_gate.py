"""Fail-closed scientific gate for the PFQ-independent rescue cohort diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


_CHECKPOINT_SHA256 = "c3f2399f94e4dc8930f9b17313acff0727d3a8bdc916ba3cc31e88665a1d92cc"
_SIM_BINARY_SHA256 = "3ca2aeb195430e9851a3f7be483b28832ad666c12981354e8e7e4280c00bf15c"
_SCENARIO_ID = "coop_boundary-3c36ue-17prg-type0-svd-p3000-load1-ttl200-seed41"
_DECISION_COUNT = 26112.0
_POSITIVE_STRATA = ("1_2", "3_5", "ge_6")
_RAW_KEYS = (
    "ue_tti_count",
    "tti_count",
    "planned_prg_count",
    "rescued_prg_count",
    "accepted_prg_count",
    "actual_prg_count",
    "predicted_service_bytes",
    "scheduled_bytes",
    "goodput_bytes",
    "failed_bytes",
    "tb_tx_count",
    "tb_err_count",
    "early_rescue_count",
    "middle_rescue_count",
    "late_rescue_count",
)
_DELTA_METRICS = (
    "predicted_service_bytes_per_planned_prg",
    "scheduled_bytes_per_actual_prg",
    "goodput_bytes_per_actual_prg",
    "failed_byte_ratio",
    "tb_error_ratio",
    "planned_to_accepted_ratio",
    "accepted_to_actual_ratio",
    "quality_db_mean",
    "quality_gap_db_mean",
    "prg_risk_mean",
    "ue_tb_err_last_mean",
    "remaining_demand_ratio_mean",
    "success_proxy_mean",
    "p_blank_mean",
    "p_service_mean",
    "p_best_ue_mean",
)
_STRATUM_METRICS = (
    "predicted_service_bytes_per_planned_prg",
    "scheduled_bytes_per_actual_prg",
    "failed_byte_ratio",
    "tb_error_ratio",
)
_PER_TTI_METRICS = (
    "predicted_service_bytes_per_planned_prg",
    "scheduled_bytes_per_actual_prg",
    "goodput_bytes_per_actual_prg",
    "failed_byte_ratio",
    "tb_error_ratio",
    "planned_to_accepted_ratio",
    "accepted_to_actual_ratio",
)
_FROZEN_BEHAVIOR_KEYS = (
    "stats_sum_scheduled_bytes",
    "stats_sum_ue_goodput_bytes",
    "stats_sum_failed_bytes",
    "measured_tb_tx_count",
    "measured_tb_err_count",
    "diagnostic_selected_forced_blank_count",
    "diagnostic_selected_fragmentation_blank_count",
    "diagnostic_selected_conservative_blank_count",
    "diagnostic_fragmentation_rescue_count",
)
_NEXT_EXPERIMENTS = {
    "reliability_supported": "test reliability-weighted aggregate Serve mass with the frozen actor",
    "service_value_supported": "test service-value-weighted aggregate Serve mass with the frozen actor",
    "acceptance_supported": "inspect the Type-0/PDSCH acceptance path before any policy change",
    "inconclusive": "consider per-PRG simulator tracing under a separate approved design",
}
_EVIDENCE_BOUNDARY = (
    "Descriptive association at UE x TTI resolution for one checkpoint, scenario, and seed; "
    "UE-level bytes and TB outcomes are not causal evidence for an individual rescued PRG."
)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _finite(values: Mapping[str, object], key: str, label: str) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} {key} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} {key} must be finite")
    return result


def _nonnegative(values: Mapping[str, object], key: str, label: str) -> float:
    result = _finite(values, key, label)
    if result < 0.0:
        raise ValueError(f"{label} {key} must be non-negative")
    return result


def _equal(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        raise ValueError(message)


def _validate_identity(
    summary: Mapping[str, object], manifest: Mapping[str, object], frozen: Mapping[str, object]
) -> dict[str, object]:
    _equal(summary.get("decode_mode"), "hierarchical_argmax", "decode must equal hierarchical_argmax")
    if summary.get("requires_pfq_features") is not False:
        raise ValueError("summary must be PFQ-independent")
    if summary.get("independent_deletion_equal") is not True:
        raise ValueError("summary PFQ deletion identity failed")
    _equal(summary.get("completed_steps"), 1512, "completed_steps must equal 1512")
    _equal(summary.get("stats_warmup_steps"), 1000, "summary warmup must equal 1000")
    _equal(summary.get("stats_count"), 512, "stats_count must equal 512")
    _equal(
        (summary.get("measured_tti_first"), summary.get("measured_tti_last")),
        (1001, 1512),
        "measured TTI window must equal 1001..1512",
    )
    if _finite(summary, "diagnostic_decision_count", "summary") != _DECISION_COUNT:
        raise ValueError(f"diagnostic decision count must equal {_DECISION_COUNT}")
    for key, description in (
        ("stats_sum_action_replay_mismatch_count", "replay"),
        ("stats_sum_illegal_action_count", "illegal"),
    ):
        if _finite(summary, key, "summary") != 0.0:
            raise ValueError(f"{description} count must equal zero")

    _equal(manifest.get("scenario_id"), _SCENARIO_ID, "scenario does not match frozen treatment")
    _equal(manifest.get("seed"), 41, "seed must equal 41")
    _equal(manifest.get("warmup_ttis"), 1000, "manifest warmup must equal 1000")
    _equal(manifest.get("measure_ttis"), 512, "manifest measure must equal 512")
    _equal(manifest.get("a8_iteration"), 8, "A8 iteration must equal 8")
    for key, expected, description in (
        ("checkpoint_sha256", _CHECKPOINT_SHA256, "checkpoint"),
        ("sim_binary_sha256", _SIM_BINARY_SHA256, "simulator"),
    ):
        _equal(summary.get(key), expected, f"summary {description} hash does not match frozen value")
        _equal(manifest.get(key), expected, f"manifest {description} hash does not match frozen value")
        _equal(frozen.get(key), expected, f"reference {description} hash does not match frozen value")

    scheduled = _finite(summary, "stats_sum_scheduled_bytes", "summary")
    goodput = _finite(summary, "stats_sum_ue_goodput_bytes", "summary")
    failed = _finite(summary, "stats_sum_failed_bytes", "summary")
    if scheduled != goodput + failed:
        raise ValueError("summary byte identity failed")
    for key in _FROZEN_BEHAVIOR_KEYS:
        value = _finite(summary, key, "summary")
        reference_value = _finite(frozen, key, "frozen reference")
        if value != reference_value:
            raise ValueError(f"frozen behavior total changed: {key}")

    return {
        "checkpoint_sha256": _CHECKPOINT_SHA256,
        "sim_binary_sha256": _SIM_BINARY_SHA256,
        "scenario_id": _SCENARIO_ID,
        "seed": 41,
        "warmup_ttis": 1000,
        "measure_ttis": 512,
        "measured_tti_first": 1001,
        "measured_tti_last": 1512,
        "diagnostic_decision_count": _DECISION_COUNT,
    }


def _validate_cohort(row: Mapping[str, object], label: str, *, positive: bool) -> None:
    for key in _RAW_KEYS:
        _nonnegative(row, key, label)
    if positive and _finite(row, "ue_tti_count", label) > 0.0:
        for key in _STRATUM_METRICS:
            _finite(row, key, label)
    for key in _DELTA_METRICS:
        if not positive and key not in row:
            raise ValueError(f"{label} missing {key}")


def _direction_counts(
    strata: Mapping[str, object], metric: str
) -> dict[str, int]:
    counts = {"populated": 0, "rescue_higher": 0, "rescue_lower": 0, "equal": 0}
    for name in _POSITIVE_STRATA:
        pair = _mapping(strata.get(name), f"stratum {name}")
        rescue = _mapping(pair.get("rescue_exposed"), f"stratum {name} rescue_exposed")
        ordinary = _mapping(pair.get("ordinary_service"), f"stratum {name} ordinary_service")
        populated = (
            _nonnegative(rescue, "ue_tti_count", f"stratum {name} rescue_exposed") > 0
            and _nonnegative(ordinary, "ue_tti_count", f"stratum {name} ordinary_service") > 0
        )
        if not populated:
            continue
        counts["populated"] += 1
        rescue_value = _finite(rescue, metric, f"stratum {name} rescue_exposed")
        ordinary_value = _finite(ordinary, metric, f"stratum {name} ordinary_service")
        if rescue_value > ordinary_value:
            counts["rescue_higher"] += 1
        elif rescue_value < ordinary_value:
            counts["rescue_lower"] += 1
        else:
            counts["equal"] += 1
    return counts


def _validate_diagnostics(summary: Mapping[str, object]) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
]:
    _equal(summary.get("rescue_cohort_diagnostics_enabled"), 1, "rescue cohort diagnostics not enabled")
    diagnostics = _mapping(summary.get("rescue_cohort_diagnostics"), "rescue_cohort_diagnostics")
    _equal(diagnostics.get("schema_version"), 1, "rescue cohort schema_version must equal 1")
    _equal(diagnostics.get("unit"), "ue_tti", "rescue cohort unit must equal ue_tti")
    cohorts = _mapping(diagnostics.get("cohorts"), "cohorts")
    _equal(set(cohorts), {"rescue_exposed", "ordinary_service"}, "cohort names mismatch")
    rescue = _mapping(cohorts.get("rescue_exposed"), "rescue_exposed")
    ordinary = _mapping(cohorts.get("ordinary_service"), "ordinary_service")
    _validate_cohort(rescue, "rescue_exposed", positive=False)
    _validate_cohort(ordinary, "ordinary_service", positive=False)
    for key in _DELTA_METRICS:
        _finite(rescue, key, "rescue_exposed")
        _finite(ordinary, key, "ordinary_service")

    strata = _mapping(diagnostics.get("actual_prg_strata"), "actual_prg_strata")
    _equal(set(strata), {"zero", *_POSITIVE_STRATA}, "actual PRG strata mismatch")
    zero_pair = _mapping(strata.get("zero"), "stratum zero")
    _equal(set(zero_pair), {"rescue_exposed", "ordinary_service"}, "stratum zero cohort names mismatch")
    for cohort_name in ("rescue_exposed", "ordinary_service"):
        zero_row = _mapping(zero_pair.get(cohort_name), f"stratum zero {cohort_name}")
        for key in _RAW_KEYS:
            _nonnegative(zero_row, key, f"stratum zero {cohort_name}")
    for name in _POSITIVE_STRATA:
        pair = _mapping(strata.get(name), f"stratum {name}")
        _equal(set(pair), {"rescue_exposed", "ordinary_service"}, f"stratum {name} cohort names mismatch")
        rescue_row = _mapping(pair.get("rescue_exposed"), f"stratum {name} rescue_exposed")
        ordinary_row = _mapping(pair.get("ordinary_service"), f"stratum {name} ordinary_service")
        _validate_cohort(rescue_row, f"stratum {name} rescue_exposed", positive=True)
        _validate_cohort(ordinary_row, f"stratum {name} ordinary_service", positive=True)

    per_tti = _mapping(diagnostics.get("per_tti_comparisons"), "per_tti_comparisons")
    for metric in _PER_TTI_METRICS:
        item = _mapping(per_tti.get(metric), f"per-TTI {metric}")
        paired = _nonnegative(item, "paired_tti_count", f"per-TTI {metric}")
        if paired == 0.0:
            if item.get("mean_delta_rescue_minus_ordinary") is not None or item.get("direction_agreement_fraction") is not None:
                raise ValueError(f"per-TTI {metric} empty comparison must use null values")
            continue
        _finite(item, "mean_delta_rescue_minus_ordinary", f"per-TTI {metric}")
        agreement = _finite(item, "direction_agreement_fraction", f"per-TTI {metric}")
        if not 0.0 <= agreement <= 1.0:
            raise ValueError(f"per-TTI {metric} direction_agreement_fraction is invalid")

    depletion = _mapping(diagnostics.get("depletion"), "depletion")
    for key in (
        "cell_tti_count",
        "rescue_count",
        "decisions_before_or_without_prior_rescue",
        "decisions_after_prior_rescue",
        "forced_before_or_without_prior_rescue",
        "forced_after_prior_rescue",
        "conservative_before_or_without_prior_rescue",
        "conservative_after_prior_rescue",
    ):
        _nonnegative(depletion, key, "depletion")

    rescued_sum = _nonnegative(rescue, "rescued_prg_count", "rescue_exposed") + _nonnegative(
        ordinary, "rescued_prg_count", "ordinary_service"
    )
    if rescued_sum != _finite(summary, "diagnostic_fragmentation_rescue_count", "summary"):
        raise ValueError("cohort rescued PRG sum does not match diagnostic rescue count")
    return diagnostics, rescue, ordinary, strata, per_tti


def _classify(
    rescue: Mapping[str, object],
    ordinary: Mapping[str, object],
    counts: Mapping[str, Mapping[str, int]],
    per_tti: Mapping[str, object],
    strata: Mapping[str, object],
) -> str:
    lower_success = _finite(rescue, "success_proxy_mean", "rescue_exposed") < _finite(
        ordinary, "success_proxy_mean", "ordinary_service"
    )
    reliability = lower_success and any(
        _finite(rescue, metric, "rescue_exposed") > _finite(ordinary, metric, "ordinary_service")
        and counts[metric]["rescue_higher"] >= 2
        for metric in ("failed_byte_ratio", "tb_error_ratio")
    )
    if reliability:
        return "reliability_supported"

    service_metrics = (
        "predicted_service_bytes_per_planned_prg",
        "scheduled_bytes_per_actual_prg",
    )
    lower_service_overall = all(
        _finite(rescue, metric, "rescue_exposed") < _finite(ordinary, metric, "ordinary_service")
        for metric in service_metrics
    )
    jointly_lower = 0
    if lower_service_overall:
        for name in _POSITIVE_STRATA:
            pair = _mapping(strata.get(name), name)
            rescue_row = _mapping(pair.get("rescue_exposed"), f"stratum {name} rescue_exposed")
            ordinary_row = _mapping(pair.get("ordinary_service"), f"stratum {name} ordinary_service")
            if _finite(rescue_row, "ue_tti_count", name) <= 0 or _finite(ordinary_row, "ue_tti_count", name) <= 0:
                continue
            if all(_finite(rescue_row, metric, name) < _finite(ordinary_row, metric, name) for metric in service_metrics):
                jointly_lower += 1
    if lower_service_overall and jointly_lower >= 2:
        return "service_value_supported"

    for metric in ("planned_to_accepted_ratio", "accepted_to_actual_ratio"):
        item = _mapping(per_tti.get(metric), f"per-TTI {metric}")
        if (
            _finite(rescue, metric, "rescue_exposed") < _finite(ordinary, metric, "ordinary_service")
            and _finite(item, "direction_agreement_fraction", f"per-TTI {metric}") > 0.5
        ):
            return "acceptance_supported"
    return "inconclusive"


def build_rescue_cohort_report(
    summary: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    frozen_reference: Mapping[str, object],
) -> dict[str, object]:
    """Validate one formal diagnostic and classify its rescue-quality mechanism."""
    identity = _validate_identity(summary, manifest, frozen_reference)
    diagnostics, rescue, ordinary, strata, per_tti = _validate_diagnostics(summary)
    direction_counts = {
        metric: _direction_counts(strata, metric) for metric in _STRATUM_METRICS
    }
    classification = _classify(rescue, ordinary, direction_counts, per_tti, strata)
    deltas = {
        metric: _finite(rescue, metric, "rescue_exposed")
        - _finite(ordinary, metric, "ordinary_service")
        for metric in _DELTA_METRICS
    }
    return {
        "valid": True,
        "identity": identity,
        "frozen_behavior_totals": {
            key: _finite(summary, key, "summary") for key in _FROZEN_BEHAVIOR_KEYS
        },
        "cohorts": {
            "rescue_exposed": dict(rescue),
            "ordinary_service": dict(ordinary),
        },
        "cohort_deltas": deltas,
        "positive_prg_strata": {
            name: dict(_mapping(strata[name], f"stratum {name}")) for name in _POSITIVE_STRATA
        },
        "populated_stratum_direction_counts": direction_counts,
        "per_tti_consistency": dict(per_tti),
        "depletion": dict(_mapping(diagnostics.get("depletion"), "depletion")),
        "classification": classification,
        "hypothesis_supported": classification
        in {"reliability_supported", "service_value_supported"},
        "evidence_boundary": _EVIDENCE_BOUNDARY,
        "next_experiment": _NEXT_EXPERIMENTS[classification],
    }


def _markdown(report: Mapping[str, object]) -> str:
    identity = _mapping(report.get("identity"), "report identity")
    cohorts = _mapping(report.get("cohorts"), "report cohorts")
    deltas = _mapping(report.get("cohort_deltas"), "report cohort_deltas")
    strata = _mapping(report.get("positive_prg_strata"), "report positive_prg_strata")
    per_tti = _mapping(report.get("per_tti_consistency"), "report per_tti_consistency")
    depletion = _mapping(report.get("depletion"), "report depletion")
    metrics = (
        "predicted_service_bytes_per_planned_prg",
        "scheduled_bytes_per_actual_prg",
        "failed_byte_ratio",
        "tb_error_ratio",
        "planned_to_accepted_ratio",
        "accepted_to_actual_ratio",
        "success_proxy_mean",
    )
    lines = [
        "# Rescue Cohort Diagnostic Report",
        "",
        "## Validity and identity",
        "",
        f"- Valid evidence: `{report['valid']}`",
        f"- Checkpoint SHA-256: `{identity['checkpoint_sha256']}`",
        f"- Simulator SHA-256: `{identity['sim_binary_sha256']}`",
        f"- Scenario/seed: `{identity['scenario_id']}` / `{identity['seed']}`",
        f"- Window: warmup `{identity['warmup_ttis']}`, measured `{identity['measure_ttis']}` "
        f"(TTI `{identity['measured_tti_first']}..{identity['measured_tti_last']}`)",
        "",
        "## Overall cohorts",
        "",
        "| Cohort | UE x TTI | Planned/rescued/accepted/actual PRGs | Scheduled/goodput/failed bytes | TB tx/error |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for name in ("rescue_exposed", "ordinary_service"):
        row = _mapping(cohorts.get(name), name)
        lines.append(
            f"| {name} | {row['ue_tti_count']} | {row['planned_prg_count']} / {row['rescued_prg_count']} / "
            f"{row['accepted_prg_count']} / {row['actual_prg_count']} | {row['scheduled_bytes']} / "
            f"{row['goodput_bytes']} / {row['failed_bytes']} | {row['tb_tx_count']} / {row['tb_err_count']} |"
        )
    lines.extend(["", "Overall rescue-minus-ordinary deltas:", ""])
    for metric in metrics:
        lines.append(f"- `{metric}`: `{deltas[metric]}`")

    lines.extend(
        [
            "",
            "## Positive-PRG strata",
            "",
            "| Stratum | Cohort | UE x TTI | Predicted/planned | Scheduled/actual | Failed ratio | TB-error ratio |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in _POSITIVE_STRATA:
        pair = _mapping(strata.get(name), name)
        for cohort_name in ("rescue_exposed", "ordinary_service"):
            row = _mapping(pair.get(cohort_name), f"{name} {cohort_name}")
            lines.append(
                f"| {name} | {cohort_name} | {row['ue_tti_count']} | "
                f"{row['predicted_service_bytes_per_planned_prg']} | {row['scheduled_bytes_per_actual_prg']} | "
                f"{row['failed_byte_ratio']} | {row['tb_error_ratio']} |"
            )

    lines.extend(
        [
            "",
            "## Per-TTI direction consistency",
            "",
            "| Metric | Mean delta | Paired TTIs | Direction agreement |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for metric in _PER_TTI_METRICS:
        item = _mapping(per_tti.get(metric), metric)
        lines.append(
            f"| {metric} | {item['mean_delta_rescue_minus_ordinary']} | "
            f"{item['paired_tti_count']} | {item['direction_agreement_fraction']} |"
        )

    lines.extend(["", "## Demand depletion", ""])
    for key, value in depletion.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Classification",
            "",
            f"- Classification: `{report['classification']}`",
            f"- Hypothesis supported: `{report['hypothesis_supported']}`",
            f"- Evidence limitation: {report['evidence_boundary']}",
            f"- Next experiment: {report['next_experiment']}",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, str(path))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--frozen-reference", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--markdown-out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_rescue_cohort_report(
        _read_json(args.summary),
        manifest=_read_json(args.manifest),
        frozen_reference=_read_json(args.frozen_reference),
    )
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown_out.write_text(_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

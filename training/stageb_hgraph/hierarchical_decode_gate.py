"""Fail-closed evidence gate for the A8 hierarchical decode experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping


_SCENARIO_ID = "coop_boundary-3c36ue-17prg-type0-svd-p3000-load1-ttl200-seed41"
_DECISION_COUNT = 26112.0
_FROZEN_CONTROL_GOODPUT_MBPS = 1547.837


def _finite_number(values: Mapping[str, object], key: str, label: str) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} {key} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} {key} must be finite")
    return result


def _nonempty_string(values: Mapping[str, object], key: str, label: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} {key} must be nonempty")
    return value


def _reported_number(values: Mapping[str, object], key: str, label: str) -> float | None:
    """Keep non-gate evidence visible without turning its absence into a verdict."""
    try:
        return _finite_number(values, key, label)
    except ValueError:
        return None


def _require_equal(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        raise ValueError(message)


def _manifest_value(manifest: Mapping[str, object], key: str) -> object:
    if key in manifest:
        return manifest[key]
    subjects = manifest.get("subjects")
    if isinstance(subjects, Mapping):
        a8 = subjects.get("A8")
        if isinstance(a8, Mapping):
            if key == "a8_iteration":
                return a8.get("iteration")
            return a8.get(key)
    return None


def _validate_manifest(
    manifest: Mapping[str, object], summary: Mapping[str, object], label: str
) -> None:
    _require_equal(
        manifest.get("scenario_id"), _SCENARIO_ID, f"{label} scenario does not match A8"
    )
    _require_equal(manifest.get("seed"), 41, f"{label} seed must equal 41")
    _require_equal(manifest.get("warmup_ttis"), 1000, f"{label} warmup must equal 1000")
    _require_equal(manifest.get("measure_ttis"), 512, f"{label} measure must equal 512")
    _require_equal(_manifest_value(manifest, "a8_iteration"), 8, f"{label} A8 iteration must equal 8")

    for key, description in (
        ("checkpoint_sha256", "checkpoint"),
        ("sim_binary_sha256", "simulator"),
    ):
        manifest_hash = _manifest_value(manifest, key)
        summary_hash = _nonempty_string(summary, key, label)
        if manifest_hash != summary_hash:
            raise ValueError(f"{label} manifest {description} hash does not match summary")


def _validate_summary(summary: Mapping[str, object], label: str, decode_mode: str) -> None:
    _require_equal(summary.get("decode_mode"), decode_mode, f"{label} decode must equal {decode_mode}")
    if summary.get("requires_pfq_features") is not False:
        raise ValueError(f"{label} must be PFQ-independent")
    if summary.get("independent_deletion_equal") is not True:
        raise ValueError(f"{label} PFQ deletion identity failed")
    _require_equal(summary.get("completed_steps"), 1512, f"{label} completed_steps must equal 1512")
    _require_equal(summary.get("stats_warmup_steps"), 1000, f"{label} warmup must equal 1000")
    _require_equal(summary.get("stats_count"), 512, f"{label} stats_count must equal 512")
    _require_equal(
        (summary.get("measured_tti_first"), summary.get("measured_tti_last")),
        (1001, 1512),
        f"{label} measured TTI window must equal 1001..1512",
    )
    if _finite_number(summary, "diagnostic_decision_count", label) != _DECISION_COUNT:
        raise ValueError(f"{label} diagnostic decision count must equal {_DECISION_COUNT}")
    for key, description in (
        ("stats_sum_action_replay_mismatch_count", "replay"),
        ("stats_sum_illegal_action_count", "illegal"),
    ):
        if _finite_number(summary, key, label) != 0.0:
            raise ValueError(f"{label} {description} count must equal zero")

    scheduled = _finite_number(summary, "stats_sum_scheduled_bytes", label)
    goodput = _finite_number(summary, "stats_sum_ue_goodput_bytes", label)
    failed = _finite_number(summary, "stats_sum_failed_bytes", label)
    if scheduled != goodput + failed:
        raise ValueError(f"{label} byte identity failed")


def _subject_report(summary: Mapping[str, object], label: str) -> dict[str, object]:
    decisions = _finite_number(summary, "diagnostic_decision_count", label)
    def ratio_and_count(ratio_key: str, count_key: str | None = None) -> dict[str, float | None]:
        ratio = _reported_number(summary, ratio_key, label)
        count = _reported_number(summary, count_key, label) if count_key else (
            None if ratio is None else ratio * decisions
        )
        return {"count": count, "ratio": ratio}

    return {
        "authoritative_goodput_mbps": _finite_number(summary, "stats_mean_ue_goodput_mbps", label),
        "scheduled_mbps": _reported_number(summary, "stats_mean_scheduled_mbps", label),
        "failed_mbps": _reported_number(summary, "stats_failed_mbps", label),
        "measured_tb": {
            "tx_count": _reported_number(summary, "measured_tb_tx_count", label),
            "error_count": _reported_number(summary, "measured_tb_err_count", label),
            "bler": _reported_number(summary, "measured_tb_bler", label),
        },
        "fragmentation": {
            "selected": ratio_and_count("diagnostic_selected_blank_ratio"),
            "forced": ratio_and_count("diagnostic_forced_blank_ratio"),
            "conservative": ratio_and_count("diagnostic_conservative_blank_ratio"),
            "counterfactual": ratio_and_count(
                "diagnostic_fragmentation_blank_ratio", "diagnostic_fragmentation_blank_count"
            ),
            "rescue": ratio_and_count(
                "diagnostic_fragmentation_rescue_ratio", "diagnostic_fragmentation_rescue_count"
            ),
            "changed_action": ratio_and_count(
                "diagnostic_changed_from_greedy_ratio", "diagnostic_changed_from_greedy_count"
            ),
            "selected_fragmentation_blank_count": _reported_number(
                summary, "diagnostic_selected_fragmentation_blank_count", label
            ),
        },
        "expiry": {
            "expired_packet_count": _reported_number(summary, "stats_sum_packet_expired_count", label),
            "expired_bytes": _reported_number(summary, "stats_sum_expired_bytes", label),
            "expired_system_time_ms": _reported_number(
                summary, "stats_sum_packet_expired_system_time_ms", label
            ),
            "packet_system_time_ms": _reported_number(summary, "stats_sum_packet_system_time_ms", label),
            "expiry_aware_delay_mean_ms": _reported_number(
                summary, "stats_weighted_packet_delay_mean_ms", label
            ),
            "expiry_aware_outcome_delay_mean_ms": _reported_number(
                summary, "stats_weighted_packet_outcome_delay_mean_ms", label
            ),
            "expiry_aware_rate_mbps": _reported_number(
                summary, "stats_mean_packet_expiry_aware_rate_mbps", label
            ),
            "expiry_aware_rate_per_packet_mean_mbps": _reported_number(
                summary, "stats_mean_packet_expiry_aware_rate_per_packet_mean_mbps", label
            ),
            "ue_min_rate_mbps": _reported_number(summary, "ue_min_expiry_aware_rate_mbps", label),
            "ue_p10_rate_mbps": _reported_number(summary, "ue_p10_expiry_aware_rate_mbps", label),
            "ue_macro_rate_mbps": _reported_number(summary, "ue_macro_expiry_aware_rate_mbps", label),
        },
    }


def build_hierarchical_decode_report(
    control: Mapping[str, object],
    treatment: Mapping[str, object],
    *,
    control_manifest: Mapping[str, object],
    treatment_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Validate matched A8 evidence and state the one-goodput hypothesis verdict."""
    _validate_summary(control, "control", "argmax")
    _validate_summary(treatment, "treatment", "hierarchical_argmax")
    _validate_manifest(control_manifest, control, "control")
    _validate_manifest(treatment_manifest, treatment, "treatment")

    control_checkpoint = _nonempty_string(control, "checkpoint_sha256", "control")
    treatment_checkpoint = _nonempty_string(treatment, "checkpoint_sha256", "treatment")
    if control_checkpoint != treatment_checkpoint:
        raise ValueError("control and treatment checkpoint hashes must match")
    control_simulator = _nonempty_string(control, "sim_binary_sha256", "control")
    treatment_simulator = _nonempty_string(treatment, "sim_binary_sha256", "treatment")
    if control_simulator != treatment_simulator:
        raise ValueError("control and treatment simulator hashes must match")

    control_goodput = _finite_number(control, "stats_mean_ue_goodput_mbps", "control")
    treatment_goodput = _finite_number(treatment, "stats_mean_ue_goodput_mbps", "treatment")
    if abs(control_goodput - _FROZEN_CONTROL_GOODPUT_MBPS) > 1.0e-6:
        raise ValueError("frozen control goodput does not match 1547.837 Mbps")

    selected_fragmentation = _finite_number(
        treatment, "diagnostic_selected_fragmentation_blank_count", "treatment"
    )
    if selected_fragmentation != 0.0:
        raise ValueError("treatment selected fragmentation Blank count must equal zero")
    counterfactual = _finite_number(
        treatment, "diagnostic_fragmentation_blank_count", "treatment"
    )
    rescue = _finite_number(treatment, "diagnostic_fragmentation_rescue_count", "treatment")
    if rescue != counterfactual:
        raise ValueError("treatment fragmentation rescue count must equal counterfactual count")

    control_subject = _subject_report(control, "control")
    treatment_subject = _subject_report(treatment, "treatment")
    delta = treatment_goodput - control_goodput
    return {
        "valid": True,
        "hypothesis_supported": treatment_goodput >= control_goodput,
        "authoritative_goodput_mbps": {
            "control": control_goodput,
            "treatment": treatment_goodput,
            "delta": delta,
        },
        "control": control_subject,
        "treatment": treatment_subject,
        "identity": {
            "checkpoint_sha256": control_checkpoint,
            "sim_binary_sha256": control_simulator,
            "seed": 41,
            "warmup_ttis": 1000,
            "measure_ttis": 512,
            "pfq_independence_checks": {
                "control_requires_pfq_features": control.get("requires_pfq_features"),
                "treatment_requires_pfq_features": treatment.get("requires_pfq_features"),
                "control_independent_deletion_equal": control.get("independent_deletion_equal"),
                "treatment_independent_deletion_equal": treatment.get("independent_deletion_equal"),
            },
        },
        "limitation": (
            "Rescued PRGs are counted, but aggregate TB bytes are not attributed to individual rescues."
        ),
    }


def _markdown(report: Mapping[str, object]) -> str:
    goodput = report["authoritative_goodput_mbps"]
    identity = report["identity"]
    assert isinstance(goodput, Mapping) and isinstance(identity, Mapping)
    lines = [
        "# Hierarchical Decode Report",
        "",
        f"- Valid evidence: `{report['valid']}`",
        f"- Hypothesis supported: `{report['hypothesis_supported']}`",
        f"- Authoritative goodput Mbps: control `{goodput['control']}`, treatment `{goodput['treatment']}`, delta `{goodput['delta']}`",
        f"- Checkpoint SHA-256: `{identity['checkpoint_sha256']}`",
        f"- Simulator SHA-256: `{identity['sim_binary_sha256']}`",
        f"- Seed/window: `{identity['seed']}` / warmup `{identity['warmup_ttis']}` / measure `{identity['measure_ttis']}`",
        "",
        "## Subjects",
        "",
        "| Subject | Scheduled Mbps | Goodput Mbps | Failed Mbps | TB tx/error/BLER |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for label in ("control", "treatment"):
        subject = report[label]
        assert isinstance(subject, Mapping)
        tb = subject["measured_tb"]
        assert isinstance(tb, Mapping)
        lines.append(
            f"| {label} | {subject['scheduled_mbps']} | {subject['authoritative_goodput_mbps']} | "
            f"{subject['failed_mbps']} | {tb['tx_count']} / {tb['error_count']} / {tb['bler']} |"
        )
    lines.extend(
        [
            "",
            "## Fragmentation",
            "",
            "| Subject | Selected count/ratio | Forced count/ratio | Conservative count/ratio | Counterfactual count/ratio | Rescue count/ratio | Changed-action count/ratio |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for label in ("control", "treatment"):
        subject = report[label]
        assert isinstance(subject, Mapping)
        fragmentation = subject["fragmentation"]
        assert isinstance(fragmentation, Mapping)
        columns = []
        for name in ("selected", "forced", "conservative", "counterfactual", "rescue", "changed_action"):
            metric = fragmentation[name]
            assert isinstance(metric, Mapping)
            columns.append(f"{metric['count']} / {metric['ratio']}")
        lines.append(f"| {label} | " + " | ".join(columns) + " |")
    lines.extend(
        [
            "",
            "## Expiry-aware metrics",
            "",
            "| Subject | Expired packets/bytes/system time | Delay/outcome delay ms | UE min/P10/macro Mbps |",
            "| --- | --- | --- | --- |",
        ]
    )
    for label in ("control", "treatment"):
        subject = report[label]
        assert isinstance(subject, Mapping)
        expiry = subject["expiry"]
        assert isinstance(expiry, Mapping)
        lines.append(
            f"| {label} | {expiry['expired_packet_count']} / {expiry['expired_bytes']} / "
            f"{expiry['expired_system_time_ms']} | {expiry['expiry_aware_delay_mean_ms']} / "
            f"{expiry['expiry_aware_outcome_delay_mean_ms']} | {expiry['ue_min_rate_mbps']} / "
            f"{expiry['ue_p10_rate_mbps']} / {expiry['ue_macro_rate_mbps']} |"
        )
    pfq = identity["pfq_independence_checks"]
    assert isinstance(pfq, Mapping)
    lines.extend(
        [
            "",
            "## PFQ-independence checks",
            "",
            f"- Control/treatment requires PFQ features: `{pfq['control_requires_pfq_features']}` / `{pfq['treatment_requires_pfq_features']}`",
            f"- Control/treatment deletion identity: `{pfq['control_independent_deletion_equal']}` / `{pfq['treatment_independent_deletion_equal']}`",
            "",
            "## Limitation",
            "",
            str(report["limitation"]),
            "",
        ]
    )
    return "\n".join(lines)


def _load_mapping(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--control-manifest", type=Path, required=True)
    parser.add_argument("--treatment-manifest", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_hierarchical_decode_report(
        _load_mapping(args.control),
        _load_mapping(args.treatment),
        control_manifest=_load_mapping(args.control_manifest),
        treatment_manifest=_load_mapping(args.treatment_manifest),
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_out.write_text(_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

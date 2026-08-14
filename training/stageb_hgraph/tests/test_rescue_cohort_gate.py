from __future__ import annotations

import json
import subprocess
import sys

import pytest

from training.stageb_hgraph.rescue_cohort_gate import (
    _markdown,
    build_rescue_cohort_report,
)


CHECKPOINT = "c3f2399f94e4dc8930f9b17313acff0727d3a8bdc916ba3cc31e88665a1d92cc"
SIMULATOR = "3ca2aeb195430e9851a3f7be483b28832ad666c12981354e8e7e4280c00bf15c"
SCENARIO = "coop_boundary-3c36ue-17prg-type0-svd-p3000-load1-ttl200-seed41"


def _cohort(
    *,
    ue_tti_count: int,
    planned: int,
    rescued: int,
    accepted: int,
    actual: int,
    predicted_per_planned: float,
    scheduled_per_actual: float,
    failed_ratio: float,
    tb_error_ratio: float,
    success_proxy: float,
) -> dict[str, object]:
    scheduled = scheduled_per_actual * actual
    failed = scheduled * failed_ratio
    tb_tx = 100
    return {
        "ue_tti_count": ue_tti_count,
        "tti_count": 10,
        "planned_prg_count": planned,
        "rescued_prg_count": rescued,
        "accepted_prg_count": accepted,
        "actual_prg_count": actual,
        "predicted_service_bytes": predicted_per_planned * planned,
        "scheduled_bytes": scheduled,
        "goodput_bytes": scheduled - failed,
        "failed_bytes": failed,
        "tb_tx_count": tb_tx,
        "tb_err_count": tb_error_ratio * tb_tx,
        "early_rescue_count": rescued,
        "middle_rescue_count": 0,
        "late_rescue_count": 0,
        "predicted_service_bytes_per_planned_prg": predicted_per_planned,
        "scheduled_bytes_per_actual_prg": scheduled_per_actual,
        "goodput_bytes_per_actual_prg": None if actual == 0 else (scheduled - failed) / actual,
        "failed_byte_ratio": failed_ratio,
        "tb_error_ratio": tb_error_ratio,
        "planned_to_accepted_ratio": accepted / planned,
        "accepted_to_actual_ratio": None if accepted == 0 else actual / accepted,
        "quality_db_mean": 8.0,
        "quality_gap_db_mean": 2.0,
        "prg_risk_mean": 0.2,
        "ue_tb_err_last_mean": 0.1,
        "remaining_demand_ratio_mean": 0.5,
        "success_proxy_mean": success_proxy,
        "p_blank_mean": 0.2,
        "p_service_mean": 0.8,
        "p_best_ue_mean": 0.6,
    }


def _per_tti() -> dict[str, dict[str, object]]:
    return {
        metric: {
            "mean_delta_rescue_minus_ordinary": delta,
            "paired_tti_count": 8,
            "direction_agreement_fraction": agreement,
        }
        for metric, delta, agreement in (
            ("predicted_service_bytes_per_planned_prg", -20.0, 0.75),
            ("scheduled_bytes_per_actual_prg", -10.0, 0.75),
            ("goodput_bytes_per_actual_prg", -12.0, 0.75),
            ("failed_byte_ratio", 0.1, 0.75),
            ("tb_error_ratio", 0.1, 0.75),
            ("planned_to_accepted_ratio", -0.1, 0.75),
            ("accepted_to_actual_ratio", -0.1, 0.75),
        )
    }


def _summary() -> dict[str, object]:
    rescue = _cohort(
        ue_tti_count=20,
        planned=6000,
        rescued=4728,
        accepted=4800,
        actual=4000,
        predicted_per_planned=80.0,
        scheduled_per_actual=80.0,
        failed_ratio=0.2,
        tb_error_ratio=0.2,
        success_proxy=0.4,
    )
    ordinary = _cohort(
        ue_tti_count=40,
        planned=12000,
        rescued=0,
        accepted=10800,
        actual=9000,
        predicted_per_planned=100.0,
        scheduled_per_actual=100.0,
        failed_ratio=0.1,
        tb_error_ratio=0.1,
        success_proxy=0.7,
    )
    strata: dict[str, object] = {
        "zero": {
            "rescue_exposed": _cohort(
                ue_tti_count=2, planned=4, rescued=2, accepted=0, actual=0,
                predicted_per_planned=60.0, scheduled_per_actual=0.0,
                failed_ratio=0.0, tb_error_ratio=0.0, success_proxy=0.3,
            ),
            "ordinary_service": _cohort(
                ue_tti_count=1, planned=2, rescued=0, accepted=0, actual=0,
                predicted_per_planned=70.0, scheduled_per_actual=0.0,
                failed_ratio=0.0, tb_error_ratio=0.0, success_proxy=0.5,
            ),
        }
    }
    for index, name in enumerate(("1_2", "3_5", "ge_6")):
        strata[name] = {
            "rescue_exposed": _cohort(
                ue_tti_count=5, planned=20, rescued=10, accepted=18, actual=10,
                predicted_per_planned=70.0 + index,
                scheduled_per_actual=70.0 + index,
                failed_ratio=0.2 if index < 2 else 0.05,
                tb_error_ratio=0.2 if index < 2 else 0.05,
                success_proxy=0.4,
            ),
            "ordinary_service": _cohort(
                ue_tti_count=6, planned=24, rescued=0, accepted=22, actual=12,
                predicted_per_planned=90.0,
                scheduled_per_actual=90.0,
                failed_ratio=0.1,
                tb_error_ratio=0.1,
                success_proxy=0.7,
            ),
        }
    return {
        "checkpoint_sha256": CHECKPOINT,
        "sim_binary_sha256": SIMULATOR,
        "decode_mode": "hierarchical_argmax",
        "requires_pfq_features": False,
        "independent_deletion_equal": True,
        "completed_steps": 1512,
        "stats_warmup_steps": 1000,
        "stats_count": 512,
        "measured_tti_first": 1001,
        "measured_tti_last": 1512,
        "diagnostic_decision_count": 26112.0,
        "stats_sum_action_replay_mismatch_count": 0.0,
        "stats_sum_illegal_action_count": 0.0,
        "stats_sum_scheduled_bytes": 54960288.0,
        "stats_sum_ue_goodput_bytes": 49104376.0,
        "stats_sum_failed_bytes": 5855912.0,
        "measured_tb_tx_count": 11324.0,
        "measured_tb_err_count": 1251.0,
        "diagnostic_fragmentation_rescue_count": 4728.0,
        "diagnostic_selected_forced_blank_count": 916.0,
        "diagnostic_selected_fragmentation_blank_count": 0.0,
        "diagnostic_selected_conservative_blank_count": 792.0,
        "rescue_cohort_diagnostics_enabled": 1,
        "rescue_cohort_diagnostics": {
            "schema_version": 1,
            "unit": "ue_tti",
            "cohorts": {"rescue_exposed": rescue, "ordinary_service": ordinary},
            "actual_prg_strata": strata,
            "per_tti_comparisons": _per_tti(),
            "depletion": {
                "cell_tti_count": 30,
                "rescue_count": 4728,
                "decisions_before_or_without_prior_rescue": 12000,
                "decisions_after_prior_rescue": 14112,
                "forced_before_or_without_prior_rescue": 500,
                "forced_after_prior_rescue": 416,
                "conservative_before_or_without_prior_rescue": 400,
                "conservative_after_prior_rescue": 392,
            },
        },
    }


def _manifest() -> dict[str, object]:
    return {
        "phase": "rescue-cohort-seed41",
        "scenario_id": SCENARIO,
        "seed": 41,
        "warmup_ttis": 1000,
        "measure_ttis": 512,
        "a8_iteration": 8,
        "checkpoint_sha256": CHECKPOINT,
        "sim_binary_sha256": SIMULATOR,
    }


def _reference() -> dict[str, object]:
    summary = _summary()
    summary.pop("rescue_cohort_diagnostics")
    summary.pop("rescue_cohort_diagnostics_enabled")
    return summary


@pytest.mark.parametrize(
    ("target", "key", "value", "match"),
    [
        ("summary", "decode_mode", "argmax", "decode"),
        ("summary", "checkpoint_sha256", "wrong", "checkpoint"),
        ("summary", "sim_binary_sha256", "wrong", "simulator"),
        ("manifest", "seed", 42, "seed"),
        ("manifest", "warmup_ttis", 999, "warmup"),
        ("manifest", "measure_ttis", 511, "measure"),
        ("summary", "diagnostic_decision_count", 26111.0, "decision"),
        ("summary", "independent_deletion_equal", False, "deletion"),
        ("summary", "stats_sum_action_replay_mismatch_count", 1.0, "replay"),
        ("summary", "stats_sum_illegal_action_count", 1.0, "illegal"),
    ],
)
def test_report_rejects_wrong_identity_window_or_action_invariance(
    target: str, key: str, value: object, match: str
) -> None:
    """Catches formal evidence escaping its frozen execution contract."""
    inputs = {"summary": _summary(), "manifest": _manifest()}
    inputs[target][key] = value

    with pytest.raises(ValueError, match=match):
        build_rescue_cohort_report(
            inputs["summary"], manifest=inputs["manifest"], frozen_reference=_reference()
        )


def test_report_rejects_failed_byte_identity() -> None:
    """Catches transmission bytes that do not account exactly."""
    summary = _summary()
    summary["stats_sum_failed_bytes"] = 5855911.0
    reference = _reference()
    reference["stats_sum_failed_bytes"] = 5855911.0

    with pytest.raises(ValueError, match="byte identity"):
        build_rescue_cohort_report(summary, manifest=_manifest(), frozen_reference=reference)


@pytest.mark.parametrize(
    "key",
    [
        "stats_sum_scheduled_bytes",
        "stats_sum_ue_goodput_bytes",
        "stats_sum_failed_bytes",
        "measured_tb_tx_count",
        "measured_tb_err_count",
        "diagnostic_selected_forced_blank_count",
        "diagnostic_selected_fragmentation_blank_count",
        "diagnostic_selected_conservative_blank_count",
        "diagnostic_fragmentation_rescue_count",
    ],
)
def test_report_rejects_changed_frozen_behavior_totals(key: str) -> None:
    """Catches interpreting a behavior-changing diagnostic as mechanism evidence."""
    summary = _summary()
    summary[key] = float(summary[key]) + 1.0
    if key == "stats_sum_scheduled_bytes":
        summary["stats_sum_failed_bytes"] = float(summary["stats_sum_failed_bytes"]) + 1.0
    elif key in {"stats_sum_ue_goodput_bytes", "stats_sum_failed_bytes"}:
        summary["stats_sum_scheduled_bytes"] = float(summary["stats_sum_scheduled_bytes"]) + 1.0

    with pytest.raises(ValueError, match="frozen behavior"):
        build_rescue_cohort_report(summary, manifest=_manifest(), frozen_reference=_reference())


def test_report_rejects_cohort_rescue_sum_mismatch() -> None:
    """Catches a nested cohort that loses a rescued planned PRG."""
    summary = _summary()
    diagnostics = summary["rescue_cohort_diagnostics"]
    diagnostics["cohorts"]["rescue_exposed"]["rescued_prg_count"] = 4727

    with pytest.raises(ValueError, match="rescued PRG"):
        build_rescue_cohort_report(summary, manifest=_manifest(), frozen_reference=_reference())


@pytest.mark.parametrize(
    ("path", "match"),
    [
        (("cohorts", "rescue_exposed", "success_proxy_mean"), "success_proxy_mean"),
        (("actual_prg_strata", "1_2", "rescue_exposed", "failed_byte_ratio"), "failed_byte_ratio"),
        (("per_tti_comparisons", "planned_to_accepted_ratio", "direction_agreement_fraction"), "direction_agreement"),
    ],
)
def test_report_rejects_missing_or_null_populated_mechanism_denominator(
    path: tuple[str, ...], match: str
) -> None:
    """Catches null evidence being silently converted into a supported mechanism."""
    summary = _summary()
    value = summary["rescue_cohort_diagnostics"]
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = None

    with pytest.raises(ValueError, match=match):
        build_rescue_cohort_report(summary, manifest=_manifest(), frozen_reference=_reference())


def _set_overall_pair(summary: dict[str, object], metric: str, rescue: float, ordinary: float) -> None:
    cohorts = summary["rescue_cohort_diagnostics"]["cohorts"]
    cohorts["rescue_exposed"][metric] = rescue
    cohorts["ordinary_service"][metric] = ordinary


def _set_stratum_pair(
    summary: dict[str, object], stratum: str, metric: str, rescue: float, ordinary: float
) -> None:
    pair = summary["rescue_cohort_diagnostics"]["actual_prg_strata"][stratum]
    pair["rescue_exposed"][metric] = rescue
    pair["ordinary_service"][metric] = ordinary


def test_ordered_reliability_classification_and_report_payload() -> None:
    """Catches the highest-priority reliability mechanism being skipped."""
    report = build_rescue_cohort_report(
        _summary(), manifest=_manifest(), frozen_reference=_reference()
    )

    assert report["valid"] is True
    assert report["classification"] == "reliability_supported"
    assert report["hypothesis_supported"] is True
    assert report["next_experiment"] == (
        "test reliability-weighted aggregate Serve mass with the frozen actor"
    )
    assert report["identity"] == {
        "checkpoint_sha256": CHECKPOINT,
        "sim_binary_sha256": SIMULATOR,
        "scenario_id": SCENARIO,
        "seed": 41,
        "warmup_ttis": 1000,
        "measure_ttis": 512,
        "measured_tti_first": 1001,
        "measured_tti_last": 1512,
        "diagnostic_decision_count": 26112.0,
    }
    assert report["populated_stratum_direction_counts"]["failed_byte_ratio"] == {
        "populated": 3,
        "rescue_higher": 2,
        "rescue_lower": 1,
        "equal": 0,
    }
    assert report["cohort_deltas"]["success_proxy_mean"] == pytest.approx(-0.3)
    assert report["cohorts"]["rescue_exposed"]["planned_prg_count"] == 6000
    assert report["per_tti_consistency"]["failed_byte_ratio"]["paired_tti_count"] == 8
    assert report["depletion"]["rescue_count"] == 4728
    assert "UE x TTI" in report["evidence_boundary"]


def test_ordered_service_value_classification() -> None:
    """Catches lower service value being hidden behind comparable reliability."""
    summary = _summary()
    _set_overall_pair(summary, "success_proxy_mean", 0.7, 0.7)
    _set_overall_pair(summary, "failed_byte_ratio", 0.1, 0.1)
    _set_overall_pair(summary, "tb_error_ratio", 0.1, 0.1)

    report = build_rescue_cohort_report(summary, manifest=_manifest(), frozen_reference=_reference())

    assert report["classification"] == "service_value_supported"
    assert report["hypothesis_supported"] is True
    assert report["next_experiment"] == (
        "test service-value-weighted aggregate Serve mass with the frozen actor"
    )


def test_ordered_acceptance_classification_is_not_hypothesis_support() -> None:
    """Catches a pipeline-retention loss being mislabeled as low service value."""
    summary = _summary()
    for metric in ("success_proxy_mean", "failed_byte_ratio", "tb_error_ratio"):
        _set_overall_pair(summary, metric, 0.5, 0.5)
    for metric in (
        "predicted_service_bytes_per_planned_prg",
        "scheduled_bytes_per_actual_prg",
    ):
        _set_overall_pair(summary, metric, 100.0, 100.0)

    report = build_rescue_cohort_report(summary, manifest=_manifest(), frozen_reference=_reference())

    assert report["classification"] == "acceptance_supported"
    assert report["hypothesis_supported"] is False
    assert report["next_experiment"] == (
        "inspect the Type-0/PDSCH acceptance path before any policy change"
    )


def test_conflicting_directions_are_inconclusive() -> None:
    """Catches weak or conflicting evidence being promoted to a mechanism."""
    summary = _summary()
    for metric in ("success_proxy_mean", "failed_byte_ratio", "tb_error_ratio"):
        _set_overall_pair(summary, metric, 0.5, 0.5)
    _set_overall_pair(summary, "predicted_service_bytes_per_planned_prg", 90.0, 100.0)
    _set_overall_pair(summary, "scheduled_bytes_per_actual_prg", 110.0, 100.0)
    _set_overall_pair(summary, "planned_to_accepted_ratio", 0.9, 0.9)
    _set_overall_pair(summary, "accepted_to_actual_ratio", 0.8, 0.8)

    report = build_rescue_cohort_report(summary, manifest=_manifest(), frozen_reference=_reference())

    assert report["classification"] == "inconclusive"
    assert report["hypothesis_supported"] is False
    assert report["next_experiment"] == (
        "consider per-PRG simulator tracing under a separate approved design"
    )


def test_service_value_requires_two_strata_where_both_metrics_are_lower() -> None:
    """Catches separate one-metric stratum wins being combined into false support."""
    summary = _summary()
    for metric in ("success_proxy_mean", "failed_byte_ratio", "tb_error_ratio"):
        _set_overall_pair(summary, metric, 0.5, 0.5)
    for stratum in ("1_2", "3_5", "ge_6"):
        _set_stratum_pair(
            summary, stratum, "predicted_service_bytes_per_planned_prg",
            80.0 if stratum == "1_2" else 110.0, 100.0,
        )
        _set_stratum_pair(
            summary, stratum, "scheduled_bytes_per_actual_prg",
            80.0 if stratum == "3_5" else 110.0, 100.0,
        )
    _set_overall_pair(summary, "planned_to_accepted_ratio", 0.9, 0.9)
    _set_overall_pair(summary, "accepted_to_actual_ratio", 0.8, 0.8)

    report = build_rescue_cohort_report(summary, manifest=_manifest(), frozen_reference=_reference())

    assert report["classification"] == "inconclusive"


def test_unpopulated_positive_stratum_keeps_null_ratios_out_of_evidence() -> None:
    """Catches an empty stratum being rejected or counted as mechanism evidence."""
    summary = _summary()
    pair = summary["rescue_cohort_diagnostics"]["actual_prg_strata"]["ge_6"]
    for row in pair.values():
        for raw in (
            "ue_tti_count", "tti_count", "planned_prg_count", "rescued_prg_count",
            "accepted_prg_count", "actual_prg_count", "predicted_service_bytes",
            "scheduled_bytes", "goodput_bytes", "failed_bytes", "tb_tx_count",
            "tb_err_count", "early_rescue_count", "middle_rescue_count", "late_rescue_count",
        ):
            row[raw] = 0
        for metric in (
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
        ):
            row[metric] = None

    report = build_rescue_cohort_report(
        summary, manifest=_manifest(), frozen_reference=_reference()
    )

    assert report["classification"] == "reliability_supported"
    assert report["populated_stratum_direction_counts"]["failed_byte_ratio"]["populated"] == 2


def test_markdown_and_module_cli_render_identical_compact_report(tmp_path) -> None:
    """Catches the CLI diverging from the validated library report."""
    summary = _summary()
    manifest = _manifest()
    reference = _reference()
    summary_path = tmp_path / "summary.json"
    manifest_path = tmp_path / "manifest.json"
    reference_path = tmp_path / "reference.json"
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    reference_path.write_text(json.dumps(reference), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "training.stageb_hgraph.rescue_cohort_gate",
            "--summary",
            str(summary_path),
            "--manifest",
            str(manifest_path),
            "--frozen-reference",
            str(reference_path),
            "--json-out",
            str(json_path),
            "--markdown-out",
            str(markdown_path),
        ],
        check=True,
    )
    expected = build_rescue_cohort_report(
        summary, manifest=manifest, frozen_reference=reference
    )

    assert json.loads(json_path.read_text(encoding="utf-8")) == expected
    assert markdown_path.read_text(encoding="utf-8") == _markdown(expected)
    markdown = markdown_path.read_text(encoding="utf-8")
    for section in (
        "## Validity and identity",
        "## Overall cohorts",
        "## Positive-PRG strata",
        "## Per-TTI direction consistency",
        "## Demand depletion",
        "## Classification",
    ):
        assert section in markdown

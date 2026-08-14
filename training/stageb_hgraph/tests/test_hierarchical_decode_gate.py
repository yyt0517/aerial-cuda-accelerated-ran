from __future__ import annotations

import pytest

from training.stageb_hgraph.hierarchical_decode_gate import (
    _markdown,
    build_hierarchical_decode_report,
)


def _summary(*, decode_mode: str, goodput_mbps: float) -> dict[str, object]:
    return {
        "checkpoint_sha256": "checkpoint-a8-sha256",
        "sim_binary_sha256": "simulator-sha256",
        "decode_mode": decode_mode,
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
        "stats_sum_scheduled_bytes": 100000.0,
        "stats_sum_ue_goodput_bytes": 99000.0,
        "stats_sum_failed_bytes": 1000.0,
        "stats_mean_scheduled_mbps": 3.125,
        "stats_mean_ue_goodput_mbps": goodput_mbps,
        "stats_failed_mbps": 0.03125,
        "measured_tb_tx_count": 1000.0,
        "measured_tb_err_count": 10.0,
        "measured_tb_bler": 0.01,
        "diagnostic_selected_blank_ratio": 0.02,
        "diagnostic_selected_blank_count": 522.24,
        "diagnostic_forced_blank_ratio": 0.005,
        "diagnostic_fragmentation_blank_count": 5.0,
        "diagnostic_fragmentation_blank_ratio": 5.0 / 26112.0,
        "diagnostic_conservative_blank_ratio": 0.015,
        "diagnostic_fragmentation_rescue_count": 5.0,
        "diagnostic_fragmentation_rescue_ratio": 5.0 / 26112.0,
        "diagnostic_changed_from_greedy_count": 5.0,
        "diagnostic_changed_from_greedy_ratio": 5.0 / 26112.0,
        "diagnostic_selected_fragmentation_blank_count": 0.0,
        "stats_sum_packet_expired_count": 7.0,
        "stats_sum_expired_bytes": 21000.0,
        "stats_sum_packet_expired_system_time_ms": 31.5,
        "stats_sum_packet_system_time_ms": 84.0,
        "stats_weighted_packet_delay_mean_ms": 1.4,
        "stats_weighted_packet_outcome_delay_mean_ms": 1.6,
        "stats_mean_packet_expiry_aware_rate_mbps": 1.2,
        "stats_mean_packet_expiry_aware_rate_per_packet_mean_mbps": 1.1,
        "ue_min_expiry_aware_rate_mbps": 0.1,
        "ue_p10_expiry_aware_rate_mbps": 0.2,
        "ue_macro_expiry_aware_rate_mbps": 1.0,
    }


def _manifest() -> dict[str, object]:
    return {
        "phase": "hierarchical-decode-seed41",
        "scenario_id": "coop_boundary-3c36ue-17prg-type0-svd-p3000-load1-ttl200-seed41",
        "seed": 41,
        "warmup_ttis": 1000,
        "measure_ttis": 512,
        "a8_iteration": 8,
        "checkpoint_sha256": "checkpoint-a8-sha256",
        "sim_binary_sha256": "simulator-sha256",
    }


def test_report_supports_valid_treatment_at_or_above_control_goodput() -> None:
    """Catches a valid hierarchical result being denied despite matched goodput."""
    control = _summary(decode_mode="argmax", goodput_mbps=1547.837)
    treatment = _summary(decode_mode="hierarchical_argmax", goodput_mbps=1547.837)

    report = build_hierarchical_decode_report(
        control,
        treatment,
        control_manifest=_manifest(),
        treatment_manifest=_manifest(),
    )

    assert report["valid"] is True
    assert report["hypothesis_supported"] is True
    assert report["authoritative_goodput_mbps"] == {
        "control": 1547.837,
        "treatment": 1547.837,
        "delta": 0.0,
    }
    assert report["control_goodput_mbps"] == 1547.837
    assert report["treatment_goodput_mbps"] == 1547.837
    assert report["delta_goodput_mbps"] == 0.0
    assert report["treatment_selected_fragmentation_blank_count"] == 0.0
    assert report["treatment_fragmentation_rescue_count"] == 5.0
    assert report["treatment_counterfactual_fragmentation_blank_count"] == 5.0
    assert report["treatment"]["fragmentation"]["rescue"]["count"] == 5.0
    assert "not attributed" in str(report["limitation"])
    markdown = _markdown(report)
    assert "## Fragmentation" in markdown
    assert "## Expiry-aware metrics" in markdown
    assert "PFQ-independence checks" in markdown


def test_report_valid_but_falsified_below_control_goodput() -> None:
    """Catches treating a scientific falsification as an invalid run."""
    control = _summary(decode_mode="argmax", goodput_mbps=1547.837)
    treatment = _summary(decode_mode="hierarchical_argmax", goodput_mbps=1547.836)
    for summary in (control, treatment):
        for key in (
            "stats_mean_scheduled_mbps",
            "stats_failed_mbps",
            "measured_tb_tx_count",
            "measured_tb_err_count",
            "measured_tb_bler",
            "diagnostic_selected_blank_ratio",
            "diagnostic_forced_blank_ratio",
            "diagnostic_fragmentation_blank_ratio",
            "diagnostic_conservative_blank_ratio",
            "diagnostic_fragmentation_rescue_ratio",
            "diagnostic_changed_from_greedy_count",
            "diagnostic_changed_from_greedy_ratio",
            "stats_sum_packet_expired_count",
            "stats_sum_expired_bytes",
            "stats_sum_packet_expired_system_time_ms",
            "stats_sum_packet_system_time_ms",
            "stats_weighted_packet_delay_mean_ms",
            "stats_weighted_packet_outcome_delay_mean_ms",
            "stats_mean_packet_expiry_aware_rate_mbps",
            "stats_mean_packet_expiry_aware_rate_per_packet_mean_mbps",
            "ue_min_expiry_aware_rate_mbps",
            "ue_p10_expiry_aware_rate_mbps",
            "ue_macro_expiry_aware_rate_mbps",
        ):
            summary.pop(key)
    report = build_hierarchical_decode_report(
        control,
        treatment,
        control_manifest=_manifest(),
        treatment_manifest=_manifest(),
    )

    assert report["valid"] is True
    assert report["hypothesis_supported"] is False
    assert report["authoritative_goodput_mbps"]["delta"] == pytest.approx(-0.001)


@pytest.mark.parametrize(
    ("target", "key", "value", "match"),
    [
        ("control", "decode_mode", "sample", "argmax"),
        ("treatment", "decode_mode", "argmax", "hierarchical_argmax"),
        ("treatment", "checkpoint_sha256", "other-checkpoint", "checkpoint"),
        ("treatment", "sim_binary_sha256", "other-simulator", "simulator"),
        ("control_manifest", "scenario_id", "other", "scenario"),
        ("control_manifest", "seed", 42, "seed"),
        ("control_manifest", "warmup_ttis", 999, "warmup"),
        ("control_manifest", "measure_ttis", 511, "measure"),
        ("control_manifest", "a8_iteration", 4, "A8"),
        ("treatment", "requires_pfq_features", True, "PFQ"),
        ("treatment", "independent_deletion_equal", False, "deletion"),
        ("treatment", "completed_steps", 1511, "completed"),
        ("treatment", "stats_warmup_steps", 999, "warmup"),
        ("treatment", "stats_count", 511, "stats_count"),
        ("treatment", "measured_tti_first", 1000, "TTI"),
        ("treatment", "measured_tti_last", 1511, "TTI"),
        ("treatment", "diagnostic_decision_count", 26111.0, "decision"),
        ("treatment", "stats_sum_action_replay_mismatch_count", 1.0, "replay"),
        ("treatment", "stats_sum_illegal_action_count", 1.0, "illegal"),
        ("treatment", "stats_sum_failed_bytes", 999.0, "byte identity"),
        ("treatment", "stats_sum_failed_bytes", 999.9999995, "byte identity"),
        ("control", "stats_mean_ue_goodput_mbps", 1547.836998, "frozen control"),
    ],
)
def test_report_rejects_identity_window_or_accounting_mismatch(
    target: str, key: str, value: object, match: str
) -> None:
    """Catches any identity, fixed-window, or byte-accounting gate being relaxed."""
    inputs: dict[str, dict[str, object]] = {
        "control": _summary(decode_mode="argmax", goodput_mbps=1547.837),
        "treatment": _summary(decode_mode="hierarchical_argmax", goodput_mbps=1547.837),
        "control_manifest": _manifest(),
        "treatment_manifest": _manifest(),
    }
    inputs[target][key] = value

    with pytest.raises(ValueError, match=match):
        build_hierarchical_decode_report(
            inputs["control"],
            inputs["treatment"],
            control_manifest=inputs["control_manifest"],
            treatment_manifest=inputs["treatment_manifest"],
        )


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("diagnostic_selected_fragmentation_blank_count", 1.0, "selected fragmentation"),
        ("diagnostic_fragmentation_rescue_count", 4.0, "rescue"),
    ],
)
def test_report_rejects_unrescued_or_selected_fragmentation_blank(
    key: str, value: float, match: str
) -> None:
    """Catches hierarchical decoding leaving a counterfactual fragmentation Blank."""
    treatment = _summary(decode_mode="hierarchical_argmax", goodput_mbps=1547.837)
    treatment[key] = value

    with pytest.raises(ValueError, match=match):
        build_hierarchical_decode_report(
            _summary(decode_mode="argmax", goodput_mbps=1547.837),
            treatment,
            control_manifest=_manifest(),
            treatment_manifest=_manifest(),
        )

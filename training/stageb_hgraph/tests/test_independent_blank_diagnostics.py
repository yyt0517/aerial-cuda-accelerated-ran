from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from training.stageb_hgraph import independent_blank_diagnostics as blank_diagnostics
from training.stageb_hgraph.independent_blank_diagnostics import (
    compute_independent_policy_diagnostics,
    extract_failed_transmission_bytes,
    summarize_independent_policy_diagnostics,
)


def test_extract_failed_transmission_bytes_is_exact() -> None:
    metrics = extract_failed_transmission_bytes(
        {
            "actual_ue_served_bytes": np.array([3000.0, 6000.0]),
            "actual_ue_goodput_bytes": np.array([3000.0, 3000.0]),
        },
        expected_n_ue=2,
    )
    assert metrics == {
        "scheduled_bytes": 9000.0,
        "goodput_bytes": 6000.0,
        "failed_bytes": 3000.0,
    }


@pytest.mark.parametrize(
    "info",
    [
        {"actual_ue_goodput_bytes": np.array([1.0, 1.0])},
        {"actual_ue_served_bytes": np.array([1.0, 1.0])},
        {
            "actual_ue_served_bytes": np.array([1.0]),
            "actual_ue_goodput_bytes": np.array([1.0, 1.0]),
        },
        {
            "actual_ue_served_bytes": np.array([1.0, -1.0]),
            "actual_ue_goodput_bytes": np.array([1.0, 0.0]),
        },
        {
            "actual_ue_served_bytes": np.array([1.0, 1.0]),
            "actual_ue_goodput_bytes": np.array([1.0, np.inf]),
        },
        {
            "actual_ue_served_bytes": np.array([1.0, 1.0]),
            "actual_ue_goodput_bytes": np.array([1.0, 1.0000011]),
        },
    ],
    ids=[
        "missing-scheduled",
        "missing-goodput",
        "wrong-shape",
        "negative",
        "non-finite",
        "goodput-exceeds-scheduled",
    ],
)
def test_extract_failed_transmission_bytes_rejects_invalid_protocol_evidence(
    info: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        extract_failed_transmission_bytes(info, expected_n_ue=2)


@pytest.mark.parametrize(
    ("derived_goodput_mbps", "existing_goodput_mbps"),
    [
        (np.nan, 1.0),
        (np.inf, 1.0),
        (-np.inf, 1.0),
        (1.0, np.nan),
        (1.0, np.inf),
        (1.0, -np.inf),
    ],
)
def test_validate_measured_window_goodput_rejects_nonfinite_values(
    derived_goodput_mbps: float,
    existing_goodput_mbps: float,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        blank_diagnostics.validate_measured_window_goodput_mbps(
            derived_goodput_mbps=derived_goodput_mbps,
            existing_goodput_mbps=existing_goodput_mbps,
        )


def test_validate_measured_window_goodput_accepts_live_float32_rounding() -> None:
    blank_diagnostics.validate_measured_window_goodput_mbps(
        derived_goodput_mbps=840.0,
        existing_goodput_mbps=839.9999694824219,
    )


def test_validate_measured_window_goodput_accepts_at_most_one_float32_ulp() -> None:
    blank_diagnostics.validate_measured_window_goodput_mbps(
        derived_goodput_mbps=1000.00006103515625,
        existing_goodput_mbps=1000.0,
    )
    with pytest.raises(ValueError, match="does not match"):
        blank_diagnostics.validate_measured_window_goodput_mbps(
            derived_goodput_mbps=1000.0001220703125,
            existing_goodput_mbps=1000.0,
        )


def test_validate_measured_window_goodput_retains_absolute_floor() -> None:
    blank_diagnostics.validate_measured_window_goodput_mbps(
        derived_goodput_mbps=0.0000005,
        existing_goodput_mbps=0.0,
    )
    with pytest.raises(ValueError, match="does not match"):
        blank_diagnostics.validate_measured_window_goodput_mbps(
            derived_goodput_mbps=0.000002,
            existing_goodput_mbps=0.0,
        )


@pytest.mark.parametrize(
    ("derived_goodput_mbps", "existing_goodput_mbps"),
    [
        (1.0e40, 1.000000002e40),
        (1.0e40, 1.0),
        (1.0, 1.0e40),
    ],
    ids=["review-neighborhood", "derived-out-of-range", "existing-out-of-range"],
)
def test_validate_measured_window_goodput_rejects_values_outside_float32_range(
    derived_goodput_mbps: float,
    existing_goodput_mbps: float,
) -> None:
    with pytest.raises(ValueError, match="within finite float32 range"):
        blank_diagnostics.validate_measured_window_goodput_mbps(
            derived_goodput_mbps=derived_goodput_mbps,
            existing_goodput_mbps=existing_goodput_mbps,
        )


def test_validate_measured_window_goodput_float32_max_tolerance_is_finite() -> None:
    with pytest.raises(ValueError, match="does not match"):
        blank_diagnostics.validate_measured_window_goodput_mbps(
            derived_goodput_mbps=3.4028234663852886e38,
            existing_goodput_mbps=0.0,
        )


def test_classifies_fragmentation_conservatism_forced_and_service() -> None:
    """Catches incorrectly deriving Blank causes from sampled choices or UE classes."""
    adjusted = torch.tensor([
        [0.0, 0.0, 0.1],
        [-2.0, -2.0, 0.0],
        [-1.0e9, -1.0e9, 0.0],
        [2.0, 0.0, -1.0],
    ])
    valid_main = torch.tensor([
        [True, True],
        [True, True],
        [False, False],
        [True, True],
    ])
    probabilities = torch.softmax(adjusted, dim=-1)
    selected = adjusted.argmax(dim=-1)

    diag = compute_independent_policy_diagnostics(
        adjusted_logits=adjusted,
        probabilities=probabilities,
        valid_main=valid_main,
        selected=selected,
        blank_class_index=2,
    )

    assert diag.greedy_blank.tolist() == [True, True, True, False]
    assert diag.forced_blank.tolist() == [False, False, True, False]
    assert diag.learned_blank.tolist() == [True, True, False, False]
    assert diag.fragmentation_blank.tolist() == [True, False, False, False]
    assert diag.conservative_blank.tolist() == [False, True, False, False]
    assert diag.margin_valid.tolist() == [True, True, False, True]
    assert diag.blank_margin[2].item() == 0.0
    assert torch.allclose(diag.p_blank + diag.p_service, torch.ones(4))
    assert bool((diag.p_best_ue <= diag.p_service + 1.0e-7).all())
    assert torch.equal(diag.learned_blank, diag.fragmentation_blank | diag.conservative_blank)
    assert not bool((diag.fragmentation_blank & diag.conservative_blank).any())


def test_accepts_one_float32_epsilon_service_reduction_overshoot() -> None:
    """A valid float32 softmax may sum the valid service classes just above one."""
    adjusted = torch.tensor([[
        4.342998504638672,
        -4.97344446182251,
        -7.302927494049072,
        6.091166019439697,
        -2.5201714038848877,
        12.424532890319824,
        8.097794532775879,
        -4.578446388244629,
        -2.7105801105499268,
        2.073176622390747,
        -10.496687889099121,
        -4.6770243644714355,
        8.363064765930176,
        -7.6468353271484375,
    ]], dtype=torch.float32)
    probabilities = torch.log_softmax(adjusted, dim=-1).exp()
    valid_main = torch.ones((1, 13), dtype=torch.bool)
    raw_p_service = probabilities[:, :-1].sum(dim=-1)
    assert raw_p_service.item() == 1.0 + torch.finfo(torch.float32).eps

    diag = compute_independent_policy_diagnostics(
        adjusted_logits=adjusted,
        probabilities=probabilities,
        valid_main=valid_main,
        selected=adjusted.argmax(dim=-1),
        blank_class_index=13,
    )

    assert 0.0 <= diag.p_service.item() <= 1.0
    assert diag.p_service.item() == 1.0
    assert torch.allclose(diag.p_blank + diag.p_service, torch.ones(1), atol=1.0e-6, rtol=0.0)
    assert diag.greedy_blank.tolist() == [False]
    assert diag.learned_blank.tolist() == [False]
    assert diag.fragmentation_blank.tolist() == [False]
    assert diag.conservative_blank.tolist() == [False]


def test_rejects_material_service_reduction_overshoot() -> None:
    adjusted = torch.zeros((1, 4), dtype=torch.float32)
    probabilities = torch.tensor([[1.0, 2.0e-6, 0.0, 0.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="p_service.*tolerance"):
        compute_independent_policy_diagnostics(
            adjusted_logits=adjusted,
            probabilities=probabilities,
            valid_main=torch.ones((1, 3), dtype=torch.bool),
            selected=torch.tensor([0]),
            blank_class_index=3,
        )


def test_rejects_material_blank_service_mass_mismatch() -> None:
    adjusted = torch.zeros((1, 3), dtype=torch.float32)
    probabilities = torch.tensor([[0.8, 0.0, 0.1]], dtype=torch.float32)

    with pytest.raises(ValueError, match=r"p_blank \+ p_service must equal one"):
        compute_independent_policy_diagnostics(
            adjusted_logits=adjusted,
            probabilities=probabilities,
            valid_main=torch.ones((1, 2), dtype=torch.bool),
            selected=torch.tensor([0]),
            blank_class_index=2,
        )


@pytest.mark.parametrize("invalid_probability", [math.nan, math.inf])
def test_rejects_nonfinite_diagnostic_probability(invalid_probability: float) -> None:
    probabilities = torch.tensor([[invalid_probability, 0.0, 1.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="probabilities must be finite"):
        compute_independent_policy_diagnostics(
            adjusted_logits=torch.zeros((1, 3), dtype=torch.float32),
            probabilities=probabilities,
            valid_main=torch.ones((1, 2), dtype=torch.bool),
            selected=torch.tensor([2]),
            blank_class_index=2,
        )


def test_summary_uses_all_decisions_and_excludes_forced_blank_margins() -> None:
    """Catches incorrect denominators, NaN empty shares, or forced-row margin quantiles."""
    adjusted = torch.tensor([
        [-1.0e9, -1.0e9, 0.0],
        [2.0, 0.0, -1.0],
        [0.0, 0.0, 0.0],
    ])
    valid_main = torch.tensor([
        [False, False],
        [True, True],
        [True, True],
    ])
    diag = compute_independent_policy_diagnostics(
        adjusted_logits=adjusted,
        probabilities=torch.softmax(adjusted, dim=-1),
        valid_main=valid_main,
        selected=torch.tensor([2, 2, 0]),
        blank_class_index=2,
    )

    summary = summarize_independent_policy_diagnostics(
        diag,
        classification_prefix="shadow_",
    )

    assert summary["decision_count"] == 3.0
    assert summary["valid_margin_count"] == 2.0
    assert summary["selected_blank_ratio"] == 2.0 / 3.0
    assert summary["shadow_greedy_blank_ratio"] == 1.0 / 3.0
    assert summary["shadow_forced_blank_ratio"] == 1.0 / 3.0
    assert summary["shadow_learned_blank_ratio"] == 0.0
    assert summary["shadow_fragmentation_share_among_learned_blank"] == 0.0
    assert not math.isnan(summary["shadow_fragmentation_share_among_learned_blank"])
    assert summary["blank_margin_p10"] == pytest.approx(-2.7)
    assert summary["blank_margin_p50"] == pytest.approx(-1.5)
    assert summary["blank_margin_p90"] == pytest.approx(-0.3)


def test_choose_b_subject_prefers_formal_selection() -> None:
    selection = {"selected_iteration": 12, "candidates": {}}
    assert blank_diagnostics.choose_b_diagnostic_iteration(selection) == 12


def test_choose_b_subject_falls_back_to_best_valid_goodput() -> None:
    selection = {
        "selected_iteration": None,
        "candidates": {
            "4": {"present": True, "metrics": {
                "goodput_mbps": 1500.0,
                "action_replay_mismatch_count": 0.0,
                "illegal_action_count": 0.0,
                "independent_deletion_equal": True,
            }},
            "8": {"present": True, "metrics": {
                "goodput_mbps": 1520.0,
                "action_replay_mismatch_count": 0.0,
                "illegal_action_count": 0.0,
                "independent_deletion_equal": True,
            }},
            "12": {"present": True, "metrics": {
                "goodput_mbps": 1600.0,
                "action_replay_mismatch_count": 1.0,
                "illegal_action_count": 0.0,
                "independent_deletion_equal": True,
            }},
        },
    }
    assert blank_diagnostics.choose_b_diagnostic_iteration(selection) == 8


def test_choose_b_subject_breaks_goodput_ties_by_lower_iteration() -> None:
    selection = {
        "selected_iteration": None,
        "candidates": {
            str(iteration): {
                "present": True,
                "metrics": {
                    "goodput_mbps": 1520.0,
                    "action_replay_mismatch_count": 0.0,
                    "illegal_action_count": 0.0,
                    "independent_deletion_equal": True,
                },
            }
            for iteration in (8, 4)
        },
    }
    assert blank_diagnostics.choose_b_diagnostic_iteration(selection) == 4


def test_choose_b_subject_ignores_valid_off_cadence_candidate() -> None:
    valid_metrics = {
        "action_replay_mismatch_count": 0.0,
        "illegal_action_count": 0.0,
        "independent_deletion_equal": True,
    }
    selection = {
        "selected_iteration": None,
        "candidates": {
            "5": {
                "present": True,
                "metrics": {**valid_metrics, "goodput_mbps": 9999.0},
            },
            "8": {
                "present": True,
                "metrics": {**valid_metrics, "goodput_mbps": 1520.0},
            },
        },
    }
    assert blank_diagnostics.choose_b_diagnostic_iteration(selection) == 8


def test_choose_b_subject_rejects_off_cadence_only_fallback() -> None:
    selection = {
        "selected_iteration": None,
        "candidates": {
            "5": {
                "present": True,
                "metrics": {
                    "goodput_mbps": 9999.0,
                    "action_replay_mismatch_count": 0.0,
                    "illegal_action_count": 0.0,
                    "independent_deletion_equal": True,
                },
            },
        },
    }
    with pytest.raises(ValueError, match="no valid fallback"):
        blank_diagnostics.choose_b_diagnostic_iteration(selection)


@pytest.mark.parametrize(
    "selection",
    [
        {},
        {"selected_iteration": None},
        {"selected_iteration": None, "candidates": {}},
        {
            "selected_iteration": None,
            "candidates": {
                "4": {
                    "present": True,
                    "metrics": {
                        "goodput_mbps": 1520.0,
                        "action_replay_mismatch_count": 0.0,
                        "illegal_action_count": 1.0,
                        "independent_deletion_equal": True,
                    },
                },
            },
        },
    ],
    ids=("missing-selection", "missing-candidates", "empty-candidates", "no-valid-candidate"),
)
def test_choose_b_subject_rejects_missing_or_invalid_candidates(
    selection: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        blank_diagnostics.choose_b_diagnostic_iteration(selection)


def _diagnostic_summary(
    checkpoint_hash: str,
    *,
    fragmentation_share: float = 0.6,
    learned_blank_count: float = 4000.0,
    forced_blank_ratio: float = 1000.0 / 26112.0,
) -> dict[str, object]:
    learned_blank_ratio = learned_blank_count / 26112.0
    fragmentation_blank_ratio = learned_blank_ratio * fragmentation_share
    conservative_blank_ratio = learned_blank_ratio - fragmentation_blank_ratio
    greedy_blank_ratio = forced_blank_ratio + learned_blank_ratio
    return {
        "checkpoint_sha256": checkpoint_hash,
        "sim_binary_sha256": "sim-hash",
        "actor_family": "pfq_independent_reuse_v1",
        "decode_mode": "argmax",
        "requires_pfq_features": False,
        "independent_deletion_equal": True,
        "independent_policy_diagnostics_enabled": 1,
        "stats_count": 512,
        "measured_tti_first": 1001,
        "measured_tti_last": 1512,
        "stats_sum_action_replay_mismatch_count": 0.0,
        "stats_sum_illegal_action_count": 0.0,
        "diagnostic_decision_count": 26112.0,
        "diagnostic_selected_blank_ratio": greedy_blank_ratio,
        "diagnostic_greedy_blank_ratio": greedy_blank_ratio,
        "diagnostic_forced_blank_ratio": forced_blank_ratio,
        "diagnostic_learned_blank_ratio": learned_blank_ratio,
        "diagnostic_fragmentation_blank_ratio": fragmentation_blank_ratio,
        "diagnostic_conservative_blank_ratio": conservative_blank_ratio,
        "diagnostic_fragmentation_share_among_learned_blank": fragmentation_share,
        "diagnostic_learned_blank_count": learned_blank_count,
        "diagnostic_mean_valid_ue_count": 2.0,
        "diagnostic_valid_margin_count": 25112.0,
        "diagnostic_p_blank_mean": 0.25,
        "diagnostic_p_blank_p10": 0.05,
        "diagnostic_p_blank_p50": 0.20,
        "diagnostic_p_blank_p90": 0.55,
        "diagnostic_p_service_mean": 0.75,
        "diagnostic_p_service_p10": 0.45,
        "diagnostic_p_service_p50": 0.80,
        "diagnostic_p_service_p90": 0.95,
        "diagnostic_p_best_ue_mean": 0.45,
        "diagnostic_p_best_ue_p10": 0.20,
        "diagnostic_p_best_ue_p50": 0.45,
        "diagnostic_p_best_ue_p90": 0.70,
        "diagnostic_blank_margin_mean": -0.5,
        "diagnostic_blank_margin_p10": -2.0,
        "diagnostic_blank_margin_p50": -0.4,
        "diagnostic_blank_margin_p90": 0.8,
        "stats_sum_scheduled_bytes": 100000.0,
        "stats_sum_ue_goodput_bytes": 80000.0,
        "stats_sum_failed_bytes": 20000.0,
        "stats_mean_scheduled_mbps": 3.125,
        "stats_mean_ue_goodput_mbps": 2.5,
        "stats_failed_mbps": 0.625,
        "measured_tb_tx_count": 1000.0,
        "measured_tb_err_count": 100.0,
        "measured_tb_bler": 0.1,
    }


def _diagnostic_summaries(**a8_overrides: float) -> dict[str, dict[str, object]]:
    return {
        "A4": _diagnostic_summary("a4-hash", fragmentation_share=0.7),
        "A8": _diagnostic_summary("a8-hash", **a8_overrides),
        "B": _diagnostic_summary("b-hash", fragmentation_share=0.4),
    }


@pytest.mark.parametrize(
    ("fragmentation_share", "learned_blank_count", "supported", "reason"),
    [
        (0.500001, 4000.0, True, None),
        (0.5, 4000.0, False, None),
        (0.49, 4000.0, False, None),
        (0.0, 0.0, None, "no learned Blank decisions"),
    ],
)
def test_blank_report_evaluates_strict_a8_fragmentation_hypothesis(
    fragmentation_share: float,
    learned_blank_count: float,
    supported: bool | None,
    reason: str | None,
) -> None:
    report = blank_diagnostics.build_blank_diagnostic_report(
        summaries=_diagnostic_summaries(
            fragmentation_share=fragmentation_share,
            learned_blank_count=learned_blank_count,
        ),
        b_iteration=4,
        b_formal_selection={"selected_iteration": 4},
    )

    hypothesis = report["hypothesis"]
    assert hypothesis["name"] == "H_frag"
    assert hypothesis["subject"] == "A8"
    assert hypothesis["supported"] is supported
    assert hypothesis.get("reason") == reason
    assert report["b_selection"]["formal_gate_promoted"] is True
    assert report["subjects"]["A4"]["role"] == "positive_control"


def test_blank_report_marks_fallback_b_as_not_formally_promoted() -> None:
    report = blank_diagnostics.build_blank_diagnostic_report(
        summaries=_diagnostic_summaries(),
        b_iteration=8,
        b_formal_selection={"selected_iteration": None},
    )
    assert report["b_selection"] == {
        "iteration": 8,
        "formal_selected_iteration": None,
        "formal_gate_promoted": False,
    }


def test_blank_report_checks_forced_blank_before_policy_parameterization() -> None:
    report = blank_diagnostics.build_blank_diagnostic_report(
        summaries=_diagnostic_summaries(
            fragmentation_share=0.8,
            learned_blank_count=1000.0,
            forced_blank_ratio=0.2,
        ),
        b_iteration=4,
        b_formal_selection={"selected_iteration": 4},
    )
    assert report["hypothesis"]["supported"] is True
    assert report["next_experiment"] == (
        "Inspect candidate validity and sequential demand estimation before any policy change."
    )


def test_blank_report_does_not_treat_all_zero_components_as_forced_dominance() -> None:
    report = blank_diagnostics.build_blank_diagnostic_report(
        summaries=_diagnostic_summaries(
            fragmentation_share=0.0,
            learned_blank_count=0.0,
            forced_blank_ratio=0.0,
        ),
        b_iteration=4,
        b_formal_selection={"selected_iteration": 4},
    )
    assert report["next_experiment"] == (
        "Retain the flat categorical and test learned conservatism evidence "
        "(reward/BC calibration) without forced fill."
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"decode_mode": "sample"}, "argmax"),
        ({"actor_family": "legacy_stageb_hgraph"}, "independent actor"),
        ({"requires_pfq_features": True}, "PFQ-independent"),
        ({"independent_deletion_equal": False}, "deletion"),
        ({"independent_policy_diagnostics_enabled": 0}, "diagnostics"),
        ({"stats_count": 511}, "512"),
        ({"measured_tti_last": 1511}, "1001--1512"),
        ({"diagnostic_decision_count": 26111.0}, "26112"),
        ({"stats_sum_action_replay_mismatch_count": 1.0}, "replay"),
        ({"stats_sum_illegal_action_count": 1.0}, "illegal"),
        ({"diagnostic_p_blank_mean": math.nan}, "finite"),
        ({"stats_sum_failed_bytes": 19999.0}, "byte identity"),
        ({"measured_tb_bler": 0.2}, "TB identity"),
    ],
)
def test_blank_report_fails_closed_on_invalid_evidence(
    mutation: dict[str, object],
    message: str,
) -> None:
    summaries = _diagnostic_summaries()
    summaries["A8"].update(mutation)
    with pytest.raises(ValueError, match=message):
        blank_diagnostics.build_blank_diagnostic_report(
            summaries=summaries,
            b_iteration=4,
            b_formal_selection={"selected_iteration": 4},
        )


def test_blank_report_requires_exact_labels_and_common_nonempty_simulator_hash() -> None:
    summaries = _diagnostic_summaries()
    summaries["A8"]["sim_binary_sha256"] = "different-sim"
    with pytest.raises(ValueError, match="simulator"):
        blank_diagnostics.build_blank_diagnostic_report(
            summaries=summaries,
            b_iteration=4,
            b_formal_selection={"selected_iteration": 4},
        )

    summaries = _diagnostic_summaries()
    summaries["extra"] = summaries.pop("B")
    with pytest.raises(ValueError, match="labels"):
        blank_diagnostics.build_blank_diagnostic_report(
            summaries=summaries,
            b_iteration=4,
            b_formal_selection={"selected_iteration": 4},
        )


def test_blank_diagnostic_cli_selects_b_and_writes_both_reports(tmp_path: Path) -> None:
    selection_path = tmp_path / "checkpoint_selection.json"
    selection_path.write_text(
        json.dumps({"selected_iteration": 4, "candidates": {}}),
        encoding="utf-8",
    )
    selected = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.stageb_hgraph.independent_blank_diagnostics",
            "select-b",
            "--selection",
            str(selection_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert selected.returncode == 0
    assert selected.stdout == "4\n"
    assert selected.stderr == ""

    summaries = _diagnostic_summaries()
    summary_paths: dict[str, Path] = {}
    for label, summary in summaries.items():
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        summary_paths[label] = path
    json_out = tmp_path / "blank_diagnostic_report.json"
    markdown_out = tmp_path / "blank_diagnostic_report.md"
    generated = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.stageb_hgraph.independent_blank_diagnostics",
            "report",
            "--a4",
            str(summary_paths["A4"]),
            "--a8",
            str(summary_paths["A8"]),
            "--b",
            str(summary_paths["B"]),
            "--b-iteration",
            "4",
            "--b-selection",
            str(selection_path),
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(markdown_out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert generated.returncode == 0
    assert generated.stdout == ""
    assert generated.stderr == ""
    assert json.loads(json_out.read_text(encoding="utf-8"))["valid"] is True
    markdown = markdown_out.read_text(encoding="utf-8")
    assert "# Independent Blank Diagnostic Report" in markdown
    assert "H_frag on A8" in markdown

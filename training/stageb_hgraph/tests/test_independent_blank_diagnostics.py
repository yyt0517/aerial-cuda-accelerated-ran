from __future__ import annotations

import math

import pytest
import torch

from training.stageb_hgraph.independent_blank_diagnostics import (
    compute_independent_policy_diagnostics,
    summarize_independent_policy_diagnostics,
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

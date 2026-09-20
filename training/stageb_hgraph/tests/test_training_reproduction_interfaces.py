from __future__ import annotations

import sys

import pytest
import torch


def test_ppo_candidate_iteration_list_is_strict_and_unioned_with_interval() -> None:
    """Catches silently dropping explicit post-update candidate iterations."""
    from training.stageb_hgraph.online_ppo import (
        _candidate_checkpoint_due,
        _parse_candidate_checkpoint_iters,
    )

    explicit = _parse_candidate_checkpoint_iters("21,8,21")
    assert explicit == frozenset({8, 21})
    assert _candidate_checkpoint_due(8, 4, explicit, best_min_iter=12) is True
    assert _candidate_checkpoint_due(12, 4, explicit, best_min_iter=12) is True
    assert _candidate_checkpoint_due(21, 4, explicit, best_min_iter=12) is True
    assert _candidate_checkpoint_due(22, 4, explicit, best_min_iter=12) is False
    with pytest.raises(ValueError, match="positive"):
        _parse_candidate_checkpoint_iters("0,4")
    with pytest.raises(ValueError, match="comma-separated"):
        _parse_candidate_checkpoint_iters("4,nope")


def test_raw_training_scale_metrics_are_computed_before_advantage_normalization() -> None:
    """Catches reporting normalized advantages as the raw optimization scale."""
    from training.stageb_hgraph.online_ppo import _raw_training_scale_metrics

    metrics = _raw_training_scale_metrics(
        torch.tensor([1.0, 2.0, 9.0]),
        torch.tensor([10.0, 20.0, 30.0]),
        torch.tensor([8.0, 20.0, 32.0]),
    )

    assert metrics["adv_raw_mean"] == pytest.approx(4.0)
    assert metrics["adv_raw_std"] == pytest.approx((38.0 / 3.0) ** 0.5)
    assert metrics["return_raw_mean"] == pytest.approx(20.0)
    assert metrics["value_raw_std"] == pytest.approx(96.0**0.5)
    assert metrics["explained_variance_raw"] == pytest.approx(0.96)
    assert metrics["adv_raw_p05"] < metrics["adv_raw_p95"]


def test_new_cli_defaults_and_explicit_values(monkeypatch) -> None:
    """Catches runner-facing options being absent or changing legacy defaults."""
    from training.stageb_hgraph import online_imitation, online_ppo

    monkeypatch.setattr(sys, "argv", ["online_ppo.py", "--sim-bin", "sim"])
    ppo_default = online_ppo.parse_args()
    assert ppo_default.independent_remove_deadline_reward == 0
    assert ppo_default.candidate_checkpoint_iters == ""

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "online_imitation.py",
            "--independent-bc-dataset-in",
            "archive",
            "--independent-headroom-accounting",
            "ctbs-v1",
            "--independent-ablation",
            "no_reuse_attention",
        ],
    )
    bc = online_imitation.parse_args()
    assert bc.sim_bin == ""
    assert bc.independent_bc_dataset_in == "archive"
    assert bc.independent_bc_dataset_out == ""
    assert bc.independent_headroom_accounting == "ctbs-v1"
    assert bc.independent_ablation == "no_reuse_attention"
    assert bc.independent_bc_collect_only == 0

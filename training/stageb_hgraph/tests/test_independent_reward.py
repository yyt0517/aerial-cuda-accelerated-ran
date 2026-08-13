from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from training.stageb_hgraph.edge_generator import EdgeGeneratorConfig, SparseEdgeGenerator
from training.stageb_hgraph.feature_snapshot import FeatureSnapshotBuilder
from training.stageb_hgraph.independent_objectives import (
    ExpiryAwareUeWindowTracker,
    compute_ue_expiry_aware_utility,
)
from training.stageb_hgraph.independent_reward import (
    IndependentRewardConfig,
    compute_independent_reward,
)
from training.stageb_hgraph.tests.graph_fixtures import make_candidate_observation


def _graph_with_debt(value: float):
    obs = make_candidate_observation(n_prg=2, ue_feat_dim=16)
    graph = SparseEdgeGenerator(
        EdgeGeneratorConfig(candidate_mode="all_demand_ues", ue_prg_topk=2)
    ).build_graph(FeatureSnapshotBuilder().build(obs))
    graph.snapshot.ue_features.values[:, 12:16] = value
    return graph


def _info(
    *,
    goodput_bytes: float = 108_000.0,
    scheduled_bytes: float = 108_000.0,
    expired_bytes: float = 0.0,
    expiry_aware: bool = True,
) -> dict[str, object]:
    n_ue = 6
    terms = np.zeros((23,), dtype=np.float64)
    terms[9] = expired_bytes
    return {
        "reward_terms": terms,
        "actual_ue_goodput_bytes": np.full((n_ue,), goodput_bytes / n_ue),
        "actual_ue_served_bytes": np.full((n_ue,), scheduled_bytes / n_ue),
        "actual_ue_packet_system_time_ms": np.full((n_ue,), 10.0),
        "actual_ue_packet_expired_system_time_ms": np.zeros((n_ue,)),
        "actual_ue_packet_expiry_aware_diagnostics_available": expiry_aware,
    }


def _cfg(treatment: str = "A", **changes) -> IndependentRewardConfig:
    base = IndependentRewardConfig(
        treatment=treatment,
        delay_scale_p95=1.0,
        ue_utility_scale_p95=1.0,
    )
    return replace(base, **changes)


def test_base_reward_is_capacity_normalized_goodput_minus_failures_and_expiry() -> None:
    graph = _graph_with_debt(0.0)

    perfect = compute_independent_reward(graph, graph, _info(), None, _cfg("A"))
    failed = compute_independent_reward(
        graph,
        graph,
        _info(goodput_bytes=105_000.0, scheduled_bytes=108_000.0),
        None,
        _cfg("A"),
    )
    expired = compute_independent_reward(
        graph,
        graph,
        _info(expired_bytes=3000.0),
        None,
        _cfg("A"),
    )

    assert perfect["reward_total"] == pytest.approx(1.0)
    assert failed["reward_failed_bytes"] == pytest.approx(3000.0)
    assert failed["reward_total"] == pytest.approx((105_000.0 - 3000.0) / 108_000.0)
    assert expired["reward_expired_weighted"] == pytest.approx(-2.0 * 3000.0 / 108_000.0)
    assert expired["reward_total"] == pytest.approx(1.0 - 2.0 * 3000.0 / 108_000.0)


def test_delay_debt_reduction_is_positive_and_clipped_by_frozen_scale() -> None:
    before = _graph_with_debt(6000.0)
    after = _graph_with_debt(0.0)
    result = compute_independent_reward(
        before,
        after,
        _info(),
        None,
        _cfg("C", delay_scale_p95=0.01),
    )

    assert result["reward_delay_delta_raw"] > 0.0
    assert result["reward_delay_delta_clipped"] == pytest.approx(1.0)
    assert result["reward_delay_weighted"] == pytest.approx(0.25)
    assert result["reward_ue_utility_weighted"] == pytest.approx(0.0)


def test_expiry_adds_time_without_bits_and_lowers_window_utility() -> None:
    tracker = ExpiryAwareUeWindowTracker(n_ue=1, window_steps=400)
    delivered = tracker.update(
        np.array([True]),
        np.array([3000.0]),
        np.array([10.0]),
        np.array([0.0]),
    )
    expired = tracker.update(
        np.array([True]),
        np.array([0.0]),
        np.array([10.0]),
        np.array([200.0]),
    )

    assert tracker.window_successful_goodput_bytes[0] == pytest.approx(3000.0)
    assert tracker.window_outcome_system_time_ms[0] == pytest.approx(210.0)
    assert expired < delivered


def test_log_utility_rewards_same_bits_more_for_low_rate_ue() -> None:
    active = np.array([True, True])
    time_ms = np.array([100.0, 100.0])
    baseline = np.array([3000.0, 30_000.0])
    base_u = compute_ue_expiry_aware_utility(active, baseline, time_ms)
    low_u = compute_ue_expiry_aware_utility(active, baseline + [3000.0, 0.0], time_ms)
    high_u = compute_ue_expiry_aware_utility(active, baseline + [0.0, 3000.0], time_ms)

    assert low_u - base_u > high_u - base_u


@pytest.mark.parametrize(
    ("treatment", "delay_enabled", "utility_enabled"),
    [("A", 0.0, 0.0), ("B", 0.0, 0.0), ("C", 1.0, 0.0), ("D", 1.0, 1.0)],
)
def test_treatment_ablation_enables_only_locked_auxiliaries(
    treatment: str,
    delay_enabled: float,
    utility_enabled: float,
) -> None:
    tracker = ExpiryAwareUeWindowTracker(n_ue=6)
    result = compute_independent_reward(
        _graph_with_debt(6000.0),
        _graph_with_debt(0.0),
        _info(),
        tracker,
        _cfg(treatment),
    )

    assert result["reward_delay_enabled"] == delay_enabled
    assert result["reward_ue_utility_enabled"] == utility_enabled
    assert (result["reward_delay_weighted"] != 0.0) is bool(delay_enabled)
    assert (result["reward_ue_utility_weighted"] != 0.0) is bool(utility_enabled)


def test_relevant_treatments_require_positive_frozen_scales() -> None:
    graph = _graph_with_debt(0.0)
    with pytest.raises(ValueError, match="delay_scale_p95"):
        compute_independent_reward(graph, graph, _info(), None, _cfg("C", delay_scale_p95=0.0))
    with pytest.raises(ValueError, match="ue_utility_scale_p95"):
        compute_independent_reward(
            graph,
            graph,
            _info(),
            ExpiryAwareUeWindowTracker(6),
            _cfg("D", ue_utility_scale_p95=0.0),
        )


def test_delivered_only_diagnostics_cannot_satisfy_treatment_d() -> None:
    graph = _graph_with_debt(0.0)
    info = _info(expiry_aware=False)
    info.pop("actual_ue_packet_expired_system_time_ms")

    with pytest.raises(ValueError, match="expiry-aware"):
        compute_independent_reward(
            graph,
            graph,
            info,
            ExpiryAwareUeWindowTracker(6),
            _cfg("D"),
        )


def test_counter_reset_clears_window_instead_of_creating_negative_outcome_time() -> None:
    tracker = ExpiryAwareUeWindowTracker(1)
    tracker.update(np.array([True]), np.array([3000.0]), np.array([50.0]), np.array([200.0]))
    tracker.update(np.array([True]), np.array([0.0]), np.array([1.0]), np.array([0.0]))

    assert tracker.window_outcome_system_time_ms[0] == pytest.approx(1.0)
    assert tracker.window_successful_goodput_bytes[0] == pytest.approx(0.0)


def test_ppo_loader_uses_frozen_seed41_scales_and_rejects_unacknowledged_load(
    tmp_path,
) -> None:
    from training.stageb_hgraph.online_ppo import _load_independent_reward_scales

    scales_path = tmp_path / "reward_scales.json"
    scales_path.write_text(
        json.dumps({"delay_scale_p95": 2.0, "ue_utility_scale_p95": 3.0}),
        encoding="utf-8",
    )
    (tmp_path / "reference_manifest.json").write_text(
        json.dumps({"scenario": {"topology_seed": 41}}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        reward_mode="pfq_independent",
        independent_offered_bytes_per_tti=108_000.0,
        independent_allow_noncanonical_offered_load=0,
        independent_reward_scales=str(scales_path),
        independent_treatment="D",
        gamma=0.99,
    )

    assert _load_independent_reward_scales(args) == {
        "delay_scale_p95": 2.0,
        "ue_utility_scale_p95": 3.0,
    }
    args.independent_offered_bytes_per_tti = 1.0
    with pytest.raises(ValueError, match="noncanonical"):
        _load_independent_reward_scales(args)

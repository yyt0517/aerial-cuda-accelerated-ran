from __future__ import annotations

import numpy as np
import pytest
import torch

from training.stageb_hgraph.edge_generator import EdgeGeneratorConfig, SparseEdgeGenerator
from training.stageb_hgraph.feature_snapshot import FeatureSnapshotBuilder
from training.stageb_hgraph.independent_actor import (
    IndependentActorConfig,
    PFQIndependentReuseActor,
)
from training.stageb_hgraph.independent_policy import (
    IndependentPolicyTrace,
    replay_independent_trace,
)
from training.stageb_hgraph.online_ppo import (
    GraphValueHead,
    _evaluate_graph,
    _evaluate_graph_batch,
    _record_submitted_action_replay_mismatch,
    _sample_joint_action,
)
from training.stageb_hgraph.tests.graph_fixtures import make_candidate_observation


def _graph():
    obs = make_candidate_observation(n_prg=3, ue_feat_dim=16)
    return SparseEdgeGenerator(
        EdgeGeneratorConfig(
            candidate_mode="all_demand_ues",
            include_pfq_score_anchor=False,
            include_candidate_risk_features=False,
        )
    ).build_graph(FeatureSnapshotBuilder().build(obs))


def _actor():
    return PFQIndependentReuseActor(
        IndependentActorConfig(
            hidden_dim=16,
            message_layers=1,
            max_candidates=2,
            up_edge_attr_dim=5,
        )
    )


def test_independent_graph_and_batch_evaluation_reuse_existing_critic_pooling() -> None:
    actor = _actor()
    critic = GraphValueHead(hidden_dim=16)
    graph = _graph()

    one, one_value = _evaluate_graph(actor, critic, graph)
    batch, batch_value = _evaluate_graph_batch(actor, critic, [graph, graph])

    assert one["reuse_embeddings"].shape == (3, 3, 16)
    assert one_value.shape == ()
    assert batch["reuse_embeddings"].shape == (2, 3, 3, 16)
    assert batch_value.shape == (2,)


def test_rollout_action_stores_trace_exact_arrays_and_replays_without_mismatch() -> None:
    torch.manual_seed(11)
    actor = _actor()
    critic = GraphValueHead(hidden_dim=16)
    graph = _graph()
    actor_out, _ = _evaluate_graph(actor, critic, graph)

    torch.manual_seed(23)
    sampled_off = _sample_joint_action(
        actor,
        actor_out,
        graph,
        independent_policy_diagnostics=False,
    )
    torch.manual_seed(23)
    sampled = _sample_joint_action(
        actor,
        actor_out,
        graph,
        independent_policy_diagnostics=True,
    )
    _record_submitted_action_replay_mismatch(graph, sampled)

    assert isinstance(sampled["independent_policy_trace"], IndependentPolicyTrace)
    assert sampled_off["independent_policy_trace"].diagnostics is None
    assert sampled["independent_policy_trace"].diagnostics is not None
    assert np.array_equal(sampled_off["action_ue_select"], sampled["action_ue_select"])
    assert np.array_equal(sampled_off["action_prg_alloc"], sampled["action_prg_alloc"])
    assert sampled["candidate_policy_mode"] == "independent"
    assert sampled["action_replay_mismatch_count"] == 0.0
    assert np.asarray(sampled["action_ue_select"]).shape == (6,)
    assert np.asarray(sampled["action_prg_alloc"]).shape == (9,)


def test_independent_replay_policy_gradient_reaches_attention_and_ue_scorer() -> None:
    torch.manual_seed(17)
    actor = _actor()
    critic = GraphValueHead(hidden_dim=16)
    graph = _graph()
    actor_out, _ = _evaluate_graph(actor, critic, graph)
    sampled = _sample_joint_action(actor, actor_out, graph)
    trace = sampled["independent_policy_trace"]

    logp, _entropy, decoded = replay_independent_trace(
        actor,
        graph,
        sampled["candidate_packed"],
        trace,
    )
    (-logp).backward()

    assert torch.equal(
        decoded.action_ue_select.cpu(),
        torch.as_tensor(sampled["action_ue_select"], dtype=torch.int32),
    )
    assert torch.equal(
        decoded.action_prg_alloc.cpu(),
        torch.as_tensor(sampled["action_prg_alloc"], dtype=torch.int16),
    )
    assert actor.reuse_attention.in_proj_weight.grad is not None
    assert actor.reuse_attention.in_proj_weight.grad.abs().sum() > 0.0
    scorer_grads = [p.grad for p in actor.ue_scorer.parameters()]
    assert any(grad is not None and grad.abs().sum() > 0.0 for grad in scorer_grads)
    forbidden = ("pfq", "slot", "safe_fill", "repair", "teacher")
    assert not any(any(word in name.lower() for word in forbidden) for name, _ in actor.named_parameters())


def test_independent_checkpoint_loader_requires_explicit_family_and_no_pfq(tmp_path) -> None:
    from training.stageb_hgraph.eval_checkpoint_online import _load_actor_checkpoint

    actor = _actor()
    valid = {
        "actor_state": actor.state_dict(),
        "actor_config": actor.model_config_dict(),
        "graph_config": {
            "candidate_mode": "all_demand_ues",
            "include_pfq_score_anchor": False,
            "include_candidate_risk_features": False,
        },
    }
    valid_path = tmp_path / "valid.pt"
    torch.save(valid, valid_path)
    loaded, _ = _load_actor_checkpoint(valid_path, torch.device("cpu"))
    assert isinstance(loaded, PFQIndependentReuseActor)

    invalid = dict(valid)
    invalid["actor_config"] = dict(valid["actor_config"], requires_pfq_features=True)
    invalid_path = tmp_path / "invalid.pt"
    torch.save(invalid, invalid_path)
    with pytest.raises(ValueError, match="PFQ|pfq"):
        _load_actor_checkpoint(invalid_path, torch.device("cpu"))

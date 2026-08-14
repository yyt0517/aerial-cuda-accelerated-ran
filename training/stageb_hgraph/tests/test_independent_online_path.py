from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from training.stageb_hgraph.edge_generator import EdgeGeneratorConfig, SparseEdgeGenerator
from training.stageb_hgraph.feature_snapshot import FeatureSnapshotBuilder
from training.stageb_hgraph.independent_actor import (
    IndependentActorConfig,
    PFQIndependentReuseActor,
)
from training.stageb_hgraph.independent_blank_diagnostics import (
    summarize_independent_policy_diagnostics,
)
from training.stageb_hgraph.independent_policy import (
    IndependentPolicyTrace,
    replay_independent_trace,
)
from training.stageb_hgraph.graph_ops import type0_action_preflight_reject_counts
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
    rng_off = torch.random.get_rng_state()
    torch.manual_seed(23)
    sampled = _sample_joint_action(
        actor,
        actor_out,
        graph,
        independent_policy_diagnostics=True,
    )
    rng_on = torch.random.get_rng_state()
    _record_submitted_action_replay_mismatch(graph, sampled)

    trace_off = sampled_off["independent_policy_trace"]
    trace = sampled["independent_policy_trace"]
    assert isinstance(trace, IndependentPolicyTrace)
    assert trace_off.diagnostics is None
    assert trace.diagnostics is not None
    assert trace.diagnostics.valid_ue_count.numel() == 9
    assert np.array_equal(sampled_off["action_ue_select"], sampled["action_ue_select"])
    assert np.array_equal(sampled_off["action_prg_alloc"], sampled["action_prg_alloc"])
    assert torch.equal(trace_off.choice, trace.choice)
    assert torch.equal(trace_off.logp, trace.logp)
    assert torch.equal(trace_off.entropy, trace.entropy)
    assert torch.equal(rng_off, rng_on)
    assert sampled["candidate_policy_mode"] == "independent"
    assert sampled["action_replay_mismatch_count"] == 0.0
    assert np.asarray(sampled["action_ue_select"]).shape == (6,)
    assert np.asarray(sampled["action_prg_alloc"]).shape == (9,)

    summary = summarize_independent_policy_diagnostics(
        trace.diagnostics,
        classification_prefix="shadow_",
    )
    assert "shadow_greedy_blank_ratio" in summary
    assert "shadow_forced_blank_ratio" in summary
    assert "shadow_learned_blank_ratio" in summary
    assert "shadow_fragmentation_blank_ratio" in summary
    assert "shadow_conservative_blank_ratio" in summary

    _, _, replayed = replay_independent_trace(
        actor,
        graph,
        sampled["candidate_packed"],
        trace,
    )
    assert np.array_equal(
        replayed.action_ue_select.cpu().numpy(),
        sampled["action_ue_select"],
    )
    assert np.array_equal(
        replayed.action_prg_alloc.cpu().numpy(),
        sampled["action_prg_alloc"],
    )


def test_deterministic_evaluator_collects_independent_policy_diagnostics() -> None:
    from training.stageb_hgraph.eval_checkpoint_online import _select_joint_action

    selected = _select_joint_action(
        _actor(),
        _graph(),
        decode_mode="argmax",
        candidate_policy_mode="independent",
        independent_policy_diagnostics=True,
    )

    trace = selected["independent_policy_trace"]
    assert trace.diagnostics is not None
    assert trace.diagnostics.valid_ue_count.numel() == 9


@pytest.mark.parametrize(
    ("decode_mode", "expected_internal_mode"),
    [("argmax", "greedy"), ("hierarchical_argmax", "hierarchical_greedy")],
)
def test_independent_deterministic_evaluator_routes_requested_decode_mode(
    monkeypatch,
    decode_mode: str,
    expected_internal_mode: str,
) -> None:
    """Catches evaluation silently using flat argmax for hierarchical requests."""
    from training.stageb_hgraph import eval_checkpoint_online

    original = eval_checkpoint_online.select_independent_action
    observed_modes: list[str] = []

    def select_with_mode_spy(*args, **kwargs):
        observed_modes.append(kwargs["mode"])
        return original(*args, **kwargs)

    monkeypatch.setattr(eval_checkpoint_online, "select_independent_action", select_with_mode_spy)
    graph = _graph()
    selected = eval_checkpoint_online._select_joint_action(
        _actor(),
        graph,
        decode_mode=decode_mode,
        candidate_policy_mode="independent",
    )

    assert observed_modes == [expected_internal_mode]
    assert selected["action_replay_mismatch_count"] == 0.0
    assert all(
        int(reject.sum().item()) == 0
        for reject in type0_action_preflight_reject_counts(
            graph,
            torch.as_tensor(selected["action_ue_select"]),
            torch.as_tensor(selected["action_prg_alloc"]),
        ).values()
    )


def test_evaluator_cli_accepts_hierarchical_argmax(monkeypatch) -> None:
    from training.stageb_hgraph import eval_checkpoint_online

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_checkpoint_online.py",
            "--checkpoint",
            "actor.pt",
            "--socket-path",
            "bridge.sock",
            "--decode-mode",
            "hierarchical_argmax",
        ],
    )

    assert eval_checkpoint_online.parse_args().decode_mode == "hierarchical_argmax"


def test_nonindependent_hierarchical_argmax_rejects_before_simulator_start(
    monkeypatch,
    tmp_path,
) -> None:
    from training.stageb_hgraph import eval_checkpoint_online

    class SimulatorMustNotStart:
        def __init__(self, *_args, **_kwargs) -> None:
            pytest.fail("simulator startup was reached before decode validation")

    monkeypatch.setattr(eval_checkpoint_online, "BridgeEnvClient", SimulatorMustNotStart)
    monkeypatch.setattr(eval_checkpoint_online, "_load_actor_checkpoint", lambda *_args: (object(), {}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_checkpoint_online.py",
            "--checkpoint",
            str(tmp_path / "candidate.pt"),
            "--socket-path",
            "simulator-must-not-start.sock",
            "--decode-mode",
            "hierarchical_argmax",
            "--device",
            "cpu",
        ],
    )

    with pytest.raises(ValueError, match="hierarchical_argmax.*independent"):
        eval_checkpoint_online.main()


def test_nonindependent_decode_path_rejects_hierarchical_argmax() -> None:
    from training.stageb_hgraph.eval_checkpoint_online import _select_joint_action

    with pytest.raises(ValueError, match="hierarchical_argmax.*independent"):
        _select_joint_action(object(), object(), decode_mode="hierarchical_argmax")


def test_independent_sample_evaluation_remains_rejected() -> None:
    from training.stageb_hgraph.eval_checkpoint_online import _select_joint_action

    with pytest.raises(ValueError, match="deterministic"):
        _select_joint_action(
            _actor(),
            _graph(),
            decode_mode="sample",
            candidate_policy_mode="independent",
        )


def test_pfq_deletion_equality_uses_hierarchical_treatment_mode(monkeypatch) -> None:
    """Catches the deletion audit defaulting to greedy under a hierarchical request."""
    from training.stageb_hgraph import eval_checkpoint_online

    original = eval_checkpoint_online.select_independent_action
    observed_modes: list[str] = []

    def select_with_mode_spy(*args, **kwargs):
        observed_modes.append(kwargs["mode"])
        return original(*args, **kwargs)

    monkeypatch.setattr(eval_checkpoint_online, "select_independent_action", select_with_mode_spy)
    observation = make_candidate_observation(n_prg=3, ue_feat_dim=16)
    edge_generator = SparseEdgeGenerator(
        EdgeGeneratorConfig(
            candidate_mode="all_demand_ues",
            include_pfq_score_anchor=False,
            include_candidate_risk_features=False,
        )
    )

    assert eval_checkpoint_online._independent_deletion_equal(
        _actor().eval(),
        observation,
        edge_generator,
        eval_checkpoint_online.SnapshotBuilderConfig(),
        torch.device("cpu"),
        mode="hierarchical_greedy",
    )
    assert observed_modes == ["hierarchical_greedy", "hierarchical_greedy"]


def test_deterministic_evaluator_rejects_diagnostics_for_non_independent_actor() -> None:
    from training.stageb_hgraph.eval_checkpoint_online import _select_joint_action

    with pytest.raises(ValueError, match="independent actor"):
        _select_joint_action(
            object(),
            object(),
            decode_mode="argmax",
            independent_policy_diagnostics=True,
        )


def test_online_clis_accept_independent_policy_diagnostics_flag(monkeypatch) -> None:
    from training.stageb_hgraph import eval_checkpoint_online, online_ppo

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "online_ppo.py",
            "--sim-bin",
            "simulator",
            "--independent-policy-diagnostics",
            "1",
        ],
    )
    assert online_ppo.parse_args().independent_policy_diagnostics == 1

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_checkpoint_online.py",
            "--checkpoint",
            "actor.pt",
            "--socket-path",
            "bridge.sock",
            "--independent-policy-diagnostics",
            "1",
        ],
    )
    assert eval_checkpoint_online.parse_args().independent_policy_diagnostics == 1


def test_online_ppo_rejects_diagnostics_for_non_independent_actor_before_simulator_start(
    monkeypatch,
    tmp_path,
) -> None:
    from training.stageb_hgraph import online_ppo

    class SimulatorMustNotStart:
        def __init__(self, _args) -> None:
            pytest.fail("simulator startup was reached before diagnostics validation")

    monkeypatch.setattr(online_ppo, "OnlineEpisodeRunner", SimulatorMustNotStart)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "online_ppo.py",
            "--sim-bin",
            "simulator-must-not-start",
            "--action-head-mode",
            "candidate",
            "--independent-policy-diagnostics",
            "1",
            "--device",
            "cpu",
            "--out-dir",
            str(tmp_path / "run"),
        ],
    )

    with pytest.raises(
        ValueError,
        match="--independent-policy-diagnostics requires --action-head-mode independent",
    ):
        online_ppo.main()


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

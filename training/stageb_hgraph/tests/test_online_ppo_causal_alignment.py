import argparse
import sys
from dataclasses import MISSING, fields

import pytest
import torch

from training.stageb_hgraph.candidate_policy import (
    CandidatePolicyConfig,
    replay_candidate_policy_trace,
    select_candidate_policy_action,
)
from training.stageb_hgraph.graph_ops import (
    materialize_candidate_choices_to_type0,
    pack_prg_candidates,
)
from training.stageb_hgraph.model import HGraphModelConfig, StageBHGraphPolicy
from training.stageb_hgraph.online_ppo import (
    GraphValueHead,
    RolloutStep,
    _direct_candidate_logp_entropy,
    _direct_candidate_logp_entropy_batch,
    _ppo_update,
    _resolve_checkpoint_candidate_policy_mode,
    _sample_joint_action,
    _validate_candidate_policy_mode,
    parse_args,
)
from training.stageb_hgraph.tests.graph_fixtures import make_candidate_graph


def test_direct_mode_excludes_ue_selector_logp_and_updates_candidate_head():
    torch.manual_seed(11)
    graph = make_candidate_graph(n_prg=2)
    actor = StageBHGraphPolicy(
        HGraphModelConfig(action_head_mode="candidate", max_prg_candidates=2)
    )
    actor_out = actor(graph)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    trace = select_candidate_policy_action(
        graph,
        packed,
        packed.mask,
        actor_out["candidate_prg_logits"],
        CandidatePolicyConfig(mode="sample"),
        generator=torch.Generator().manual_seed(11),
    )

    new_logp, entropy = _direct_candidate_logp_entropy(
        actor_out,
        graph,
        trace,
        CandidatePolicyConfig(mode="sample"),
    )
    loss = -new_logp
    loss.backward()

    grad = actor.action_head.candidate_pair_score[-1].weight.grad
    assert torch.isfinite(loss)
    assert grad is not None and grad.abs().sum().item() > 0.0
    assert entropy.item() > 0.0


def test_direct_rollout_returns_exact_trace_materialization_and_metrics(monkeypatch):
    torch.manual_seed(17)
    graph = make_candidate_graph(n_prg=2)
    actor = StageBHGraphPolicy(
        HGraphModelConfig(action_head_mode="candidate", max_prg_candidates=2)
    )
    actor_out = actor(graph)
    monkeypatch.setattr(
        actor,
        "forward_graph",
        lambda *args, **kwargs: pytest.fail("direct mode sampled the legacy UE selector"),
    )

    sampled = _sample_joint_action(
        actor,
        actor_out,
        graph,
        candidate_policy_mode="direct",
        candidate_budgeted_select_mode="sample",
        candidate_decode_demand_slack_bytes=0.0,
    )

    trace = sampled["candidate_policy_trace"]
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    expected = materialize_candidate_choices_to_type0(graph, packed, trace.choice)
    assert torch.equal(sampled["prg_action_class"], trace.choice.cpu())
    assert torch.equal(torch.from_numpy(sampled["action_ue_select"]), expected.action_ue_select.cpu())
    assert torch.equal(torch.from_numpy(sampled["action_prg_alloc"]), expected.action_prg_alloc.cpu())
    assert torch.equal(sampled["logp"], trace.logp.cpu())
    assert sampled["coverage_repair_count"] == 0.0
    assert sampled["safe_fill_count"] == 0.0
    assert sampled["blank_budget_repair_count"] == 0.0
    assert sampled["non_hard_override_count"] == 0.0


def _direct_args(**overrides):
    values = {
        "candidate_policy_mode": "direct",
        "action_head_mode": "candidate",
        "candidate_budgeted_select_mode": "sample",
        "candidate_blank_budget_max_per_cell": -1,
        "slot_repair_max_per_cell": 0,
        "safe_fill_max_per_cell": 0,
        "candidate_budgeted_tail_bias_coef": 0.0,
        "candidate_budgeted_rescue_bias_coef": 0.0,
        "candidate_budgeted_completion_bonus": 0.0,
        "candidate_budgeted_tail_quota_enable": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action_head_mode", "slot"),
        ("candidate_budgeted_select_mode", "greedy"),
        ("candidate_blank_budget_max_per_cell", 0),
        ("slot_repair_max_per_cell", 1),
        ("safe_fill_max_per_cell", 1),
        ("candidate_budgeted_tail_bias_coef", 0.1),
        ("candidate_budgeted_rescue_bias_coef", 0.1),
        ("candidate_budgeted_completion_bonus", 0.1),
        ("candidate_budgeted_tail_quota_enable", 1),
    ],
)
def test_direct_mode_rejects_incompatible_configuration(field, value):
    with pytest.raises(ValueError, match="direct"):
        _validate_candidate_policy_mode(_direct_args(**{field: value}))


def test_ppo_direct_batch_replays_each_stored_trace():
    torch.manual_seed(23)
    graphs = [make_candidate_graph(n_prg=2), make_candidate_graph(n_prg=2)]
    actor = StageBHGraphPolicy(
        HGraphModelConfig(action_head_mode="candidate", max_prg_candidates=2)
    )
    outputs = [actor(graph) for graph in graphs]
    cfg = CandidatePolicyConfig(mode="sample")
    traces = []
    for graph, actor_out in zip(graphs, outputs):
        packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
        traces.append(
            select_candidate_policy_action(
                graph,
                packed,
                packed.mask,
                actor_out["candidate_prg_logits"],
                cfg,
            )
        )
    batched_out = {
        "candidate_prg_logits": torch.stack(
            [out["candidate_prg_logits"] for out in outputs],
            dim=0,
        ),
        "ue_logits": torch.full((2, 1), float("nan")),
    }

    logp, entropy = _direct_candidate_logp_entropy_batch(
        batched_out,
        graphs,
        traces,
        cfg,
    )

    expected = [
        replay_candidate_policy_trace(
            graph,
            pack_prg_candidates(graph, max_candidates=2, fixed_width=True),
            out["candidate_prg_logits"],
            trace,
            cfg,
        )
        for graph, out, trace in zip(graphs, outputs, traces)
    ]
    assert torch.allclose(logp, torch.stack([item[0] for item in expected]))
    assert torch.allclose(entropy, torch.stack([item[1] for item in expected]))


def test_legacy_checkpoint_and_default_mode_remain_legacy(monkeypatch):
    assert _resolve_checkpoint_candidate_policy_mode({"args": {"action_head_mode": "candidate"}}) == "legacy"
    assert (
        _resolve_checkpoint_candidate_policy_mode(
            {"args": {"candidate_policy_mode": "direct"}}
        )
        == "direct"
    )
    monkeypatch.setattr(sys, "argv", ["online_ppo.py", "--sim-bin", "unused"])
    assert parse_args().candidate_policy_mode == "legacy"


def test_direct_ppo_update_reports_candidate_ownership_metrics():
    torch.manual_seed(29)
    graph = make_candidate_graph(n_prg=2)
    actor = StageBHGraphPolicy(
        HGraphModelConfig(action_head_mode="candidate", max_prg_candidates=2)
    )
    sampled = _sample_joint_action(
        actor,
        actor(graph),
        graph,
        candidate_policy_mode="direct",
        candidate_budgeted_select_mode="sample",
        candidate_decode_demand_slack_bytes=0.0,
    )
    required = {
        field.name: 0.0
        for field in fields(RolloutStep)
        if field.default is MISSING and field.default_factory is MISSING
    }
    required.update(
        graph=graph,
        ue_action_class=sampled["ue_action_class"],
        prg_action_class=sampled["prg_action_class"],
        old_logp=sampled["logp"],
        candidate_policy_trace=sampled["candidate_policy_trace"],
        candidate_policy_mode="direct",
    )
    rollout = [RolloutStep(**required)]
    critic = GraphValueHead(hidden_dim=128)

    metrics = _ppo_update(
        actor=actor,
        critic=critic,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=1.0e-3),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1.0e-3),
        rollout=rollout,
        advantages=torch.ones(1),
        returns=torch.zeros(1),
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        update_epochs=1,
        minibatch_size=1,
        iter_idx=1,
        update_log_every_minibatches=0,
        target_kl=0.0,
        candidate_policy_mode="direct",
        candidate_budgeted_select_mode="sample",
        candidate_decode_demand_slack_bytes=0.0,
    )

    assert metrics["ue_selector_policy_logp_included"] == 0.0
    assert metrics["candidate_head_grad_norm"] > 0.0

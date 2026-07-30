import torch

from training.stageb_hgraph.candidate_policy import (
    CandidatePolicyConfig,
    replay_candidate_policy_trace,
    select_candidate_policy_action,
)
from training.stageb_hgraph.graph_ops import pack_prg_candidates
from training.stageb_hgraph.tests.graph_fixtures import make_candidate_graph


def fixture_policy_inputs():
    graph = make_candidate_graph(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    return graph, packed, packed.mask.clone()


def test_candidate_fixture_has_three_cells_and_all_demand_ues():
    graph = make_candidate_graph(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    assert graph.snapshot.static_meta.n_cell == 3
    assert graph.snapshot.static_meta.n_prg == 2
    assert packed.ue_ids.shape == (6, 2)
    assert packed.mask.all()


def test_sample_trace_replays_exact_logp_and_has_entropy():
    graph, packed, valid = fixture_policy_inputs()
    logits = torch.zeros((packed.num_prg_tokens, packed.max_candidates + 1), requires_grad=True)
    generator = torch.Generator().manual_seed(7)
    trace = select_candidate_policy_action(
        graph,
        packed,
        valid,
        logits,
        CandidatePolicyConfig(mode="sample", demand_cap_slack_bytes=0.0),
        generator=generator,
    )
    replay_logp, replay_entropy = replay_candidate_policy_trace(
        graph,
        packed,
        logits,
        trace,
        CandidatePolicyConfig(mode="sample", demand_cap_slack_bytes=0.0),
    )
    assert torch.equal(trace.choice, trace.choice.clamp(0, packed.blank_class_index))
    assert torch.allclose(replay_logp, trace.logp, atol=1e-6)
    assert replay_entropy.item() > 0.0


def test_blank_is_selectable_when_its_logit_is_largest():
    graph, packed, valid = fixture_policy_inputs()
    logits = torch.zeros((packed.num_prg_tokens, packed.max_candidates + 1))
    logits[:, packed.blank_class_index] = 20.0
    trace = select_candidate_policy_action(
        graph,
        packed,
        valid,
        logits,
        CandidatePolicyConfig(mode="greedy", demand_cap_slack_bytes=0.0),
    )
    assert torch.equal(
        trace.choice,
        torch.full_like(trace.choice, packed.blank_class_index),
    )


def test_replayed_logp_backpropagates_to_selected_candidate_logits():
    graph, packed, valid = fixture_policy_inputs()
    logits = torch.randn(
        packed.num_prg_tokens,
        packed.max_candidates + 1,
        generator=torch.Generator().manual_seed(3),
        requires_grad=True,
    )
    trace = select_candidate_policy_action(
        graph,
        packed,
        valid,
        logits,
        CandidatePolicyConfig(mode="sample"),
        generator=torch.Generator().manual_seed(5),
    )
    logp, _ = replay_candidate_policy_trace(
        graph,
        packed,
        logits,
        trace,
        CandidatePolicyConfig(mode="sample"),
    )
    (-logp).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0.0

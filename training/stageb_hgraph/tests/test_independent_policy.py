from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from training.stageb_hgraph.graph_ops import (
    pack_prg_candidates,
    type0_action_preflight_reject_counts,
)
from training.stageb_hgraph.independent_policy import (
    _select_hierarchical_greedy,
    estimate_candidate_service_bytes,
    replay_independent_trace,
    resolve_full_candidate_width,
    select_independent_action,
)
from training.stageb_hgraph.tests.graph_fixtures import make_candidate_graph


class RecordingActor(nn.Module):
    def __init__(self, n_prg: int, max_candidates: int = 2):
        super().__init__()
        self.cfg = SimpleNamespace(max_candidates=max_candidates)
        self.raw_logits = nn.Parameter(torch.zeros((n_prg, 3, max_candidates + 1)))
        with torch.no_grad():
            self.raw_logits[..., 0] = 4.0
            self.raw_logits[..., 1] = 2.0
            self.raw_logits[..., 2] = -1.0
        self.sequential_calls: list[torch.Tensor] = []

    def encode_graph(self, graph):
        return {}

    def score_reuse_group(
        self,
        encoded,
        graph,
        packed,
        local_prg: int,
        sequential_features: torch.Tensor,
    ) -> torch.Tensor:
        self.sequential_calls.append(sequential_features.detach().clone())
        return self.raw_logits[local_prg]


def _small_demand_graph(n_prg: int = 3):
    return make_candidate_graph(
        n_prg=n_prg,
        backlog_bytes=(100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
    )


def test_hierarchical_greedy_rescues_fragmentation_and_selects_best_valid_ue() -> None:
    adjusted = torch.log(torch.tensor([[0.30, 0.30, 0.40]]))
    valid_main = torch.tensor([[True, True]])

    selected = _select_hierarchical_greedy(adjusted, valid_main, blank_class_index=2)

    assert adjusted.argmax(dim=-1).item() == 2
    assert selected.tolist() == [0]


def test_hierarchical_greedy_keeps_conservative_blank_and_tie_blank() -> None:
    adjusted = torch.log(torch.tensor([[0.20, 0.20, 0.60], [0.25, 0.25, 0.50]]))
    valid_main = torch.tensor([[True, True], [True, True]])

    selected = _select_hierarchical_greedy(adjusted, valid_main, blank_class_index=2)

    assert selected.tolist() == [2, 2]


def test_hierarchical_greedy_forces_blank_without_valid_ue() -> None:
    adjusted = torch.tensor([[8.0, 4.0, 0.0]])
    valid_main = torch.tensor([[False, False]])

    selected = _select_hierarchical_greedy(adjusted, valid_main, blank_class_index=2)

    assert selected.tolist() == [2]


def test_hierarchical_greedy_excludes_invalid_high_logit() -> None:
    adjusted = torch.tensor([[8.0, 2.0, 1.0]])
    valid_main = torch.tensor([[False, True]])

    selected = _select_hierarchical_greedy(adjusted, valid_main, blank_class_index=2)

    assert selected.tolist() == [1]


def test_hierarchical_factorization_matches_flat_logp_entropy_and_trace() -> None:
    adjusted = torch.log(torch.tensor([[0.25, 0.35, 0.40]]))
    valid_main = torch.tensor([[True, True]])
    selected_ue = _select_hierarchical_greedy(adjusted, valid_main, blank_class_index=2)
    flat_logp = torch.log_softmax(adjusted, dim=-1)
    flat_prob = flat_logp.exp()
    p_service = flat_prob[:, :2].masked_fill(~valid_main, 0).sum(-1)
    conditional = flat_prob[:, :2] / p_service.unsqueeze(-1)
    hierarchical_group_entropy = -(
        p_service * p_service.log() + flat_prob[:, 2] * flat_logp[:, 2]
    )
    conditional_entropy = -(conditional * conditional.log()).sum(-1)

    assert torch.allclose(
        p_service.log() + conditional.gather(1, selected_ue.unsqueeze(-1)).squeeze(-1).log(),
        flat_logp.gather(1, selected_ue.unsqueeze(-1)).squeeze(-1),
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        hierarchical_group_entropy + p_service * conditional_entropy,
        -(flat_prob * flat_logp).sum(-1),
        atol=1e-6,
        rtol=0.0,
    )

    graph = _small_demand_graph(n_prg=1)
    actor = RecordingActor(n_prg=1)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    with torch.no_grad():
        actor.raw_logits[0].copy_(adjusted.expand_as(actor.raw_logits[0]))
    trace, action = select_independent_action(actor, graph, packed, mode="hierarchical_greedy")

    assert torch.allclose(trace.logp, flat_logp.gather(1, selected_ue.unsqueeze(-1)).squeeze(-1).mean())
    assert torch.allclose(trace.entropy, (-(flat_prob * flat_logp).sum(-1)).mean())
    assert all(
        int(reject.sum().item()) == 0
        for reject in type0_action_preflight_reject_counts(
            graph,
            action.action_ue_select,
            action.action_prg_alloc,
        ).values()
    )


def test_fixed_local_order_and_preupdate_three_cell_state() -> None:
    graph = _small_demand_graph(n_prg=17)
    actor = RecordingActor(n_prg=17)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)

    trace, _ = select_independent_action(actor, graph, packed, mode="greedy")

    assert trace.local_prg_order.tolist() == list(range(17))
    assert trace.valid_at_step.shape == (17, 3, 3)
    assert len(actor.sequential_calls) == 17
    assert torch.equal(actor.sequential_calls[0], torch.tensor([[[1.0, 0.0, 0.0]] * 2] * 3))
    # All cells at p=1 see the same pre-update convention: selected class 0 is
    # exhausted, while class 1 is untouched.
    assert torch.equal(actor.sequential_calls[1][:, 0, 0], torch.zeros(3))
    assert torch.equal(actor.sequential_calls[1][:, 1, 0], torch.ones(3))
    assert torch.allclose(
        actor.sequential_calls[1][:, 0, 1],
        torch.full((3,), 1.0 / 17.0),
    )
    assert torch.equal(trace.valid_at_step[1, :, 0], torch.zeros(3, dtype=torch.bool))
    assert torch.equal(trace.valid_at_step[2, :, :2], torch.zeros((3, 2), dtype=torch.bool))
    assert trace.valid_at_step[:, :, -1].all()


def test_locked_service_formula_and_candidate_width() -> None:
    graph = make_candidate_graph(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)

    service = estimate_candidate_service_bytes(graph, packed)

    expected = torch.log2(torch.tensor(11.0)) * (16 * 12 * 30_000) * (11 / 14) * 0.0005 / 8
    assert service.shape == (6, 2)
    assert torch.allclose(service[packed.mask], torch.full_like(service[packed.mask], expected))
    assert bool((service >= 1.0).all().item())
    assert resolve_full_candidate_width(graph) == 2


@pytest.mark.parametrize("mode", ["sample", "greedy"])
def test_rollout_replay_exact_trace_and_action(mode: str) -> None:
    graph = _small_demand_graph(n_prg=3)
    actor = RecordingActor(n_prg=3)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    generator = torch.Generator().manual_seed(17) if mode == "sample" else None

    trace, action = select_independent_action(
        actor,
        graph,
        packed,
        mode=mode,
        generator=generator,
    )
    replay_logp, replay_entropy, replay_action = replay_independent_trace(
        actor,
        graph,
        packed,
        trace,
    )

    assert torch.allclose(replay_logp, trace.logp, atol=1.0e-7)
    assert torch.allclose(replay_entropy, trace.entropy, atol=1.0e-7)
    assert torch.equal(replay_action.action_ue_select, action.action_ue_select)
    assert torch.equal(replay_action.action_prg_alloc, action.action_prg_alloc)
    (-replay_logp).backward()
    assert actor.raw_logits.grad is not None
    assert bool(torch.isfinite(actor.raw_logits.grad).all().item())


@pytest.mark.parametrize("mode", ["sample", "greedy"])
def test_diagnostic_collection_is_action_and_rng_invariant(mode: str) -> None:
    graph = _small_demand_graph(n_prg=3)
    actor = RecordingActor(n_prg=3)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    generator_off = torch.Generator().manual_seed(29) if mode == "sample" else None
    generator_on = torch.Generator().manual_seed(29) if mode == "sample" else None

    trace_off, action_off = select_independent_action(
        actor,
        graph,
        packed,
        mode=mode,
        generator=generator_off,
        collect_diagnostics=False,
    )
    trace_on, action_on = select_independent_action(
        actor,
        graph,
        packed,
        mode=mode,
        generator=generator_on,
        collect_diagnostics=True,
    )

    assert trace_off.diagnostics is None
    assert trace_on.diagnostics is not None
    assert trace_on.diagnostics.valid_ue_count.numel() == 3 * 3
    assert torch.equal(trace_off.choice, trace_on.choice)
    assert torch.equal(trace_off.valid_at_step, trace_on.valid_at_step)
    assert torch.equal(trace_off.logp, trace_on.logp)
    assert torch.equal(trace_off.entropy, trace_on.entropy)
    assert torch.equal(action_off.action_ue_select, action_on.action_ue_select)
    assert torch.equal(action_off.action_prg_alloc, action_on.action_prg_alloc)
    if mode == "sample":
        assert torch.equal(generator_off.get_state(), generator_on.get_state())

    replay_logp, replay_entropy, replay_action = replay_independent_trace(
        actor,
        graph,
        packed,
        trace_on,
    )
    assert torch.equal(replay_logp, trace_on.logp)
    assert torch.equal(replay_entropy, trace_on.entropy)
    assert torch.equal(replay_action.action_ue_select, action_on.action_ue_select)
    assert torch.equal(replay_action.action_prg_alloc, action_on.action_prg_alloc)


def test_replay_rejects_stored_mask_or_action_mismatch() -> None:
    graph = _small_demand_graph(n_prg=3)
    actor = RecordingActor(n_prg=3)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    trace, _ = select_independent_action(actor, graph, packed, mode="greedy")

    bad_mask = trace.valid_at_step.clone()
    bad_mask[0, 0, 0] = False
    with pytest.raises(ValueError, match="stored mask"):
        replay_independent_trace(
            actor,
            graph,
            packed,
            SimpleNamespace(**{**trace.__dict__, "valid_at_step": bad_mask}),
        )

    bad_choice = trace.choice.clone()
    bad_choice[0] = packed.blank_class_index + 1
    with pytest.raises(ValueError, match="choice"):
        replay_independent_trace(
            actor,
            graph,
            packed,
            SimpleNamespace(**{**trace.__dict__, "choice": bad_choice}),
        )

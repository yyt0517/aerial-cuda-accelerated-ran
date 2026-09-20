from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from training.stageb_hgraph.edge_generator import EdgeGeneratorConfig, SparseEdgeGenerator
from training.stageb_hgraph.feature_snapshot import FeatureSnapshotBuilder
from training.stageb_hgraph.graph_ops import (
    pack_prg_candidates,
    type0_action_preflight_reject_counts,
)
from training.stageb_hgraph.independent_actor import (
    IndependentActorConfig,
    PFQIndependentReuseActor,
)
from training.stageb_hgraph.ctbs_non_starvation_guard import (
    CtbsNonStarvationGuardConfig,
)
from training.stageb_hgraph import independent_policy as independent_policy_module
from training.stageb_hgraph.independent_policy import (
    _select_hierarchical_greedy,
    estimate_candidate_service_bytes,
    replay_independent_trace,
    resolve_full_candidate_width,
    select_independent_action,
)
from training.stageb_hgraph.tests.graph_fixtures import (
    make_candidate_graph,
    make_candidate_observation,
)


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


class WidthPerturbedActor(nn.Module):
    """Same discrete policy with a controlled FP32-width perturbation."""

    def __init__(self, perturbation: float):
        super().__init__()
        self.cfg = SimpleNamespace(max_candidates=2)
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.perturbation = float(perturbation)

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
        logits = torch.full(
            (3, packed.max_candidates + 1),
            -4.0,
            dtype=self.anchor.dtype,
            device=self.anchor.device,
        )
        logits[:, 0] = 4.0
        logits[:, 1] = 2.0
        logits[:, -1] = -1.0
        if packed.max_candidates == 3:
            logits[:, 0] += self.perturbation
        return logits + self.anchor * 0.0


def _small_demand_graph(n_prg: int = 3):
    return make_candidate_graph(
        n_prg=n_prg,
        backlog_bytes=(100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
    )


def _guard_graph(n_prg: int = 2):
    observation = make_candidate_observation(
        n_prg=n_prg,
        ue_feat_dim=16,
        backlog_bytes=(12000.0,) * 6,
    )
    observation["obs_ue_features"][:, 12] = math.log(3.0)
    observation["obs_ue_features"][:, 13] = 0.5
    observation["obs_ue_features"][:, 14] = 0.5
    return SparseEdgeGenerator(
        EdgeGeneratorConfig(
            candidate_mode="all_demand_ues",
            include_pfq_score_anchor=False,
            include_candidate_risk_features=False,
        )
    ).build_graph(FeatureSnapshotBuilder().build(observation))


def _blank_recording_actor(n_prg: int = 2) -> RecordingActor:
    actor = RecordingActor(n_prg=n_prg)
    with torch.no_grad():
        actor.raw_logits[..., 0] = 2.0
        actor.raw_logits[..., 1] = -1.0
        actor.raw_logits[..., 2] = 4.0
    return actor


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


def test_service_accounting_diagnostics_capture_absolute_predecision_headroom() -> None:
    """Catches auditing demand-cap bias from normalized or post-update headroom."""
    graph = make_candidate_graph(
        n_prg=2,
        backlog_bytes=(10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0),
    )
    actor = RecordingActor(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)

    trace, _ = select_independent_action(
        actor,
        graph,
        packed,
        mode="greedy",
        collect_diagnostics=True,
    )

    audit = trace.service_accounting_diagnostics
    assert audit is not None
    assert audit.selected_nonblank.tolist() == [True] * 6
    assert torch.equal(audit.raw_service_bytes, audit.effective_service_bytes)
    assert audit.initial_demand_bytes.tolist() == [10000.0] * 6
    first_service = audit.raw_service_bytes[:3]
    assert audit.remaining_demand_before_bytes[:3].tolist() == [10000.0] * 3
    assert torch.allclose(
        audit.remaining_demand_before_bytes[3:],
        torch.full((3,), 10000.0) - first_service,
    )


def test_extra_padding_preserves_independent_policy_semantics() -> None:
    comparator = getattr(
        independent_policy_module,
        "compare_candidate_padding_semantics",
        None,
    )
    assert callable(comparator), "candidate padding equivalence gate is missing"
    torch.manual_seed(19)
    observation = make_candidate_observation(
        n_prg=3,
        ue_feat_dim=16,
        backlog_bytes=(100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
    )
    graph = SparseEdgeGenerator(
        EdgeGeneratorConfig(
            candidate_mode="all_demand_ues",
            include_pfq_score_anchor=False,
            include_candidate_risk_features=False,
        )
    ).build_graph(FeatureSnapshotBuilder().build(observation))
    actor = PFQIndependentReuseActor(
        IndependentActorConfig(
            hidden_dim=16,
            message_layers=1,
            max_candidates=2,
        )
    ).eval()

    report = comparator(
        actor,
        graph,
        reference_width=2,
        padded_width=3,
        mode="greedy",
    )

    assert report == {
        "candidate_choice_equal": True,
        "mask_equal": True,
        "logprob_equal": True,
        "replay_equal": True,
        "type0_action_equal": True,
        "extra_padding_invalid": True,
    }


def test_fp32_padding_gate_accepts_only_bounded_numerical_differences() -> None:
    comparator = getattr(
        independent_policy_module,
        "compare_candidate_padding_fp32_equivalence",
        None,
    )
    assert callable(comparator), "W0-v2 FP32 equivalence gate is missing"
    graph = _small_demand_graph(n_prg=3)

    accepted = comparator(
        WidthPerturbedActor(5.0e-6),
        graph,
        reference_width=2,
        padded_width=3,
        mode="greedy",
        valid_logits_atol=1.0e-5,
        chosen_logprob_atol=1.0e-6,
    )
    rejected = comparator(
        WidthPerturbedActor(2.0e-5),
        graph,
        reference_width=2,
        padded_width=3,
        mode="greedy",
        valid_logits_atol=1.0e-5,
        chosen_logprob_atol=1.0e-6,
    )

    for report in (accepted, rejected):
        assert report["candidate_choice_equal"] is True
        assert report["mask_equal"] is True
        assert report["type0_action_equal"] is True
        assert report["extra_padding_invalid"] is True
        assert report["runtime_replay_mismatch_count"] == 0
        assert report["nan_or_inf_count"] == 0
    assert 0.0 < accepted["valid_logits_max_abs_diff"] <= 1.0e-5
    assert 0.0 < accepted["chosen_logprob_max_abs_diff"] <= 1.0e-6
    assert accepted["passed"] is True
    assert rejected["valid_logits_max_abs_diff"] > 1.0e-5
    assert rejected["passed"] is False


def test_disabled_non_starvation_guard_is_bitwise_identity() -> None:
    from training.stageb_hgraph.tests.test_ctbs_headroom import _inputs

    graph = _small_demand_graph(n_prg=3)
    actor = RecordingActor(n_prg=3)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    ctbs_inputs = _inputs(n_prg=3)

    default_trace, default_action = select_independent_action(
        actor,
        graph,
        packed,
        mode="greedy",
        ctbs_headroom_inputs=ctbs_inputs,
    )
    disabled_trace, disabled_action = select_independent_action(
        actor,
        graph,
        packed,
        mode="greedy",
        ctbs_headroom_inputs=ctbs_inputs,
        non_starvation_guard=CtbsNonStarvationGuardConfig(enabled=False),
    )

    assert torch.equal(default_trace.actor_choice, disabled_trace.actor_choice)
    assert torch.equal(default_trace.choice, disabled_trace.choice)
    assert torch.equal(
        default_trace.guard_override,
        torch.zeros_like(default_trace.choice, dtype=torch.bool),
    )
    assert torch.equal(default_trace.choice, disabled_trace.choice)
    assert torch.equal(default_trace.logp, disabled_trace.logp)
    assert torch.equal(default_trace.entropy, disabled_trace.entropy)
    assert torch.equal(
        default_action.action_ue_select,
        disabled_action.action_ue_select,
    )
    assert torch.equal(
        default_action.action_prg_alloc,
        disabled_action.action_prg_alloc,
    )


def test_guarded_blank_updates_sequential_ctbs_state_once_per_cell_and_replays() -> None:
    from training.stageb_hgraph.tests.test_ctbs_headroom import _inputs

    graph = _guard_graph(n_prg=2)
    actor = _blank_recording_actor(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)

    trace, action = select_independent_action(
        actor,
        graph,
        packed,
        mode="greedy",
        collect_diagnostics=True,
        ctbs_headroom_inputs=_inputs(n_prg=2),
        non_starvation_guard=CtbsNonStarvationGuardConfig(enabled=True),
    )

    blank = packed.blank_class_index
    assert trace.actor_choice.tolist() == [blank] * 6
    assert trace.choice.tolist() == [0, blank, 0, blank, 0, blank]
    assert trace.guard_override.tolist() == [True, False, True, False, True, False]
    assert trace.guard_diagnostics.guard_override.tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert trace.guard_diagnostics.ctbs_capacity_before_bytes.dtype == torch.int64
    assert trace.guard_diagnostics.ctbs_capacity_after_bytes.dtype == torch.int64
    assert trace.guard_diagnostics.ctbs_capacity_after_bytes[:3].tolist() == [
        253,
        253,
        253,
    ]
    expected_remaining_ratio = torch.full((3,), (12000.0 - 253.0) / 12000.0)
    assert torch.allclose(
        actor.sequential_calls[1][:, 0, 0],
        expected_remaining_ratio,
    )
    assert torch.equal(
        actor.sequential_calls[1][:, 0, 1],
        torch.full((3,), 1.0 / 17.0),
    )
    assert torch.allclose(
        actor.sequential_calls[1][:, 0, 2],
        torch.full((3,), 253.0 / 12000.0),
    )
    expected_logp = torch.cat(
        [
            torch.log_softmax(actor.raw_logits[local_prg], dim=-1)[:, blank]
            for local_prg in range(2)
        ]
    ).mean()
    assert torch.equal(trace.logp, expected_logp)
    assert action.chosen_ue_per_prg.tolist() == [0, -1, 2, -1, 4, -1]
    assert all(
        int(reject.sum().item()) == 0
        for reject in type0_action_preflight_reject_counts(
            graph,
            action.action_ue_select,
            action.action_prg_alloc,
        ).values()
    )

    replay_logp, replay_entropy, replay_action = replay_independent_trace(
        actor,
        graph,
        packed,
        trace,
    )
    assert torch.equal(replay_logp, trace.logp)
    assert torch.equal(replay_entropy, trace.entropy)
    assert torch.equal(replay_action.action_ue_select, action.action_ue_select)
    assert torch.equal(replay_action.action_prg_alloc, action.action_prg_alloc)


def test_guarded_replay_rejects_inconsistent_raw_executed_trace() -> None:
    from training.stageb_hgraph.tests.test_ctbs_headroom import _inputs

    graph = _guard_graph(n_prg=2)
    actor = _blank_recording_actor(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    trace, _ = select_independent_action(
        actor,
        graph,
        packed,
        mode="greedy",
        ctbs_headroom_inputs=_inputs(n_prg=2),
        non_starvation_guard=CtbsNonStarvationGuardConfig(enabled=True),
    )

    inconsistent_override = trace.guard_override.clone()
    inconsistent_override[0] = False
    with pytest.raises(ValueError, match="guard override"):
        replay_independent_trace(
            actor,
            graph,
            packed,
            replace(trace, guard_override=inconsistent_override),
        )

    inconsistent_actor_diagnostics = trace.guard_diagnostics.actor_choice.clone()
    inconsistent_actor_diagnostics[0] = 0
    with pytest.raises(ValueError, match="guard diagnostics"):
        replay_independent_trace(
            actor,
            graph,
            packed,
            replace(
                trace,
                guard_diagnostics=replace(
                    trace.guard_diagnostics,
                    actor_choice=inconsistent_actor_diagnostics,
                ),
            ),
        )


@pytest.mark.parametrize("mode", ["sample", "hierarchical_greedy"])
def test_enabled_guard_rejects_non_greedy_modes(mode: str) -> None:
    from training.stageb_hgraph.tests.test_ctbs_headroom import _inputs

    graph = _guard_graph(n_prg=2)
    actor = _blank_recording_actor(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)

    with pytest.raises(ValueError, match="greedy"):
        select_independent_action(
            actor,
            graph,
            packed,
            mode=mode,
            ctbs_headroom_inputs=_inputs(n_prg=2),
            non_starvation_guard=CtbsNonStarvationGuardConfig(enabled=True),
        )


def test_enabled_guard_fails_closed_for_missing_ctbs_teacher_or_feature_contract() -> None:
    from training.stageb_hgraph.tests.test_ctbs_headroom import _inputs

    graph = _guard_graph(n_prg=2)
    actor = _blank_recording_actor(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    enabled = CtbsNonStarvationGuardConfig(enabled=True)

    with pytest.raises(ValueError, match="CTBS"):
        select_independent_action(
            actor,
            graph,
            packed,
            mode="greedy",
            non_starvation_guard=enabled,
        )

    with pytest.raises(ValueError, match="teacher"):
        independent_policy_module._run_policy(
            actor,
            graph,
            packed,
            mode="greedy",
            teacher_actions=(torch.full((6,), -1), torch.full((6,), -1)),
            ctbs_headroom_inputs=_inputs(n_prg=2),
            non_starvation_guard=enabled,
        )

    bad_names = graph.snapshot.ue_features.names[:12] + (
        "wrong_pressure_name",
    ) + graph.snapshot.ue_features.names[13:]
    bad_graph = replace(
        graph,
        snapshot=replace(
            graph.snapshot,
            ue_features=replace(graph.snapshot.ue_features, names=bad_names),
        ),
    )
    with pytest.raises(ValueError, match="feature names"):
        select_independent_action(
            actor,
            bad_graph,
            packed,
            mode="greedy",
            ctbs_headroom_inputs=_inputs(n_prg=2),
            non_starvation_guard=enabled,
        )


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


@pytest.mark.parametrize("mode", ["sample", "greedy", "hierarchical_greedy"])
def test_diagnostic_invariance_preserves_action_and_rng(mode: str) -> None:
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
    assert trace_off.rescue_diagnostics is None
    assert trace_on.rescue_diagnostics is not None
    assert trace_on.diagnostics.valid_ue_count.numel() == 3 * 3
    assert trace_on.rescue_diagnostics.selected_ue.numel() == graph.snapshot.n_cell * graph.snapshot.n_prg
    assert torch.equal(
        trace_on.rescue_diagnostics.fragmentation_rescued,
        trace_on.diagnostics.fragmentation_rescued,
    )
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


def test_blank_expiry_instrumentation_is_read_only_and_post_mask_normalized() -> None:
    graph = _small_demand_graph(n_prg=3)
    actor = RecordingActor(n_prg=3)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)

    trace_off, action_off = select_independent_action(
        actor,
        graph,
        packed,
        mode="greedy",
    )
    trace_on, action_on = select_independent_action(
        actor,
        graph,
        packed,
        mode="greedy",
        collect_blank_expiry_attribution=True,
    )

    assert trace_off.blank_expiry_diagnostics is None
    diagnostics = trace_on.blank_expiry_diagnostics
    assert diagnostics is not None
    assert torch.equal(trace_on.choice, trace_off.choice)
    assert torch.equal(trace_on.valid_at_step, trace_off.valid_at_step)
    assert torch.equal(action_on.action_ue_select, action_off.action_ue_select)
    assert torch.equal(action_on.action_prg_alloc, action_off.action_prg_alloc)
    assert diagnostics.candidate_ue_ids.shape == (9, 2)
    assert diagnostics.candidate_valid.shape == (9, 2)
    assert diagnostics.candidate_probability.shape == (9, 2)
    assert diagnostics.blank_probability.shape == (9,)
    assert torch.equal(
        diagnostics.candidate_probability.masked_select(~diagnostics.candidate_valid),
        torch.zeros_like(
            diagnostics.candidate_probability.masked_select(~diagnostics.candidate_valid)
        ),
    )
    torch.testing.assert_close(
        diagnostics.candidate_probability.sum(dim=1)
        + diagnostics.blank_probability,
        torch.ones((9,)),
    )
    replay_logp, replay_entropy, replay_action = replay_independent_trace(
        actor,
        graph,
        packed,
        trace_on,
    )
    torch.testing.assert_close(replay_logp, trace_on.logp)
    torch.testing.assert_close(replay_entropy, trace_on.entropy)
    assert torch.equal(replay_action.action_ue_select, action_on.action_ue_select)
    assert torch.equal(replay_action.action_prg_alloc, action_on.action_prg_alloc)

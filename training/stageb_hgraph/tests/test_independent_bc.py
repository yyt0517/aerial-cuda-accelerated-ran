from __future__ import annotations

import pytest
import torch

from training.stageb_hgraph.graph_ops import PackedPrgCandidates, pack_prg_candidates
from training.stageb_hgraph.independent_actor import (
    IndependentActorConfig,
    PFQIndependentReuseActor,
)
from training.stageb_hgraph.independent_policy import (
    independent_teacher_targets_from_type0,
    independent_teacher_trace_from_type0,
)
from training.stageb_hgraph.tests.graph_fixtures import make_candidate_graph


def _teacher_actions():
    # Native slots deliberately reverse the candidate-row UE order in each cell.
    action_ue = torch.tensor([1, 0, 3, 2, 5, 4])
    # native layout is local_prg-major, then cell.
    action_prg = torch.tensor([0, 2, 4, -1, 3, 5])
    return action_ue, action_prg


def _replace_packed(packed, **changes):
    return PackedPrgCandidates(
        ue_ids=changes.get("ue_ids", packed.ue_ids),
        mask=changes.get("mask", packed.mask),
        counts=changes.get("counts", packed.counts),
        edge_attr=changes.get("edge_attr", packed.edge_attr),
    )


def test_direct_teacher_conversion_follows_slot_to_ue_to_candidate() -> None:
    graph = make_candidate_graph(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    action_ue, action_prg = _teacher_actions()
    valid = torch.ones((2, 3, 3), dtype=torch.bool)

    targets = independent_teacher_targets_from_type0(
        graph,
        packed,
        action_ue,
        action_prg,
        valid,
    )

    # Cell-major token rows: [cell0 p0,p1, cell1 p0,p1, cell2 p0,p1].
    assert targets.tolist() == [1, 2, 1, 0, 1, 0]
    assert targets[1].item() == packed.blank_class_index


def test_direct_teacher_conversion_rejects_missing_invalid_or_wrong_cell() -> None:
    graph = make_candidate_graph(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    action_ue, action_prg = _teacher_actions()
    valid = torch.ones((2, 3, 3), dtype=torch.bool)

    missing_mask = packed.mask.clone()
    missing_mask[0, 1] = False
    with pytest.raises(ValueError, match="missing|duplicated"):
        independent_teacher_targets_from_type0(
            graph,
            _replace_packed(packed, mask=missing_mask),
            action_ue,
            action_prg,
            valid,
        )

    invalid_slot = action_prg.clone()
    invalid_slot[0] = 99
    with pytest.raises(ValueError, match="slot out of range"):
        independent_teacher_targets_from_type0(
            graph, packed, action_ue, invalid_slot, valid
        )

    wrong_cell_ue = action_ue.clone()
    wrong_cell_ue[0] = 2
    with pytest.raises(ValueError, match="wrong cell"):
        independent_teacher_targets_from_type0(
            graph, packed, wrong_cell_ue, action_prg, valid
        )

    invalid = valid.clone()
    invalid[0, 0, 1] = False
    with pytest.raises(ValueError, match="headroom-invalid"):
        independent_teacher_targets_from_type0(
            graph, packed, action_ue, action_prg, invalid
        )


def test_teacher_forced_trace_uses_rollout_headroom_and_rejects_repeat_after_exhaustion() -> None:
    graph = make_candidate_graph(
        n_prg=2,
        backlog_bytes=(100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
    )
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    actor = PFQIndependentReuseActor(
        IndependentActorConfig(
            hidden_dim=16,
            message_layers=1,
            max_candidates=2,
            up_edge_attr_dim=6,
        )
    )
    action_ue = torch.tensor([0, 1, 2, 3, 4, 5])
    repeated_same_ue = torch.tensor([0, 2, 4, 0, 2, 4])

    with pytest.raises(ValueError, match="invalid"):
        independent_teacher_trace_from_type0(
            actor,
            graph,
            packed,
            action_ue,
            repeated_same_ue,
        )

    second_prg_blank = torch.tensor([0, 2, 4, -1, -1, -1])
    trace = independent_teacher_trace_from_type0(
        actor,
        graph,
        packed,
        action_ue,
        second_prg_blank,
    )
    assert trace.choice.tolist() == [0, 2, 0, 2, 0, 2]
    assert trace.valid_at_step[1, :, 0].tolist() == [False, False, False]
    assert torch.isfinite(trace.logp)

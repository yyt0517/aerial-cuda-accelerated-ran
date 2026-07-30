from __future__ import annotations

from dataclasses import replace

import torch

from training.stageb_hgraph.graph_ops import (
    PackedPrgCandidates,
    materialize_candidate_choices_to_type0,
    pack_prg_candidates,
    type0_action_preflight_reject_counts,
)
from training.stageb_hgraph.tests.graph_fixtures import make_candidate_graph


def _replace_static_meta(graph, **changes):
    static_meta = replace(graph.snapshot.static_meta, **changes)
    return replace(graph, snapshot=replace(graph.snapshot, static_meta=static_meta))


def test_materialization_uses_unique_slots_and_preserves_chosen_ue():
    graph = make_candidate_graph(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    choice = torch.tensor([0, 0, 1, 1, 0, packed.blank_class_index])

    decoded = materialize_candidate_choices_to_type0(graph, packed, choice)

    assert decoded.chosen_ue_per_prg.tolist() == [0, 0, 3, 3, 4, -1]
    assert decoded.prg_chosen_mask.tolist() == [True, True, True, True, True, False]
    assert decoded.action_ue_select.tolist() == [0, -1, 3, -1, 4, -1]
    assert decoded.action_prg_alloc.tolist() == [0, 2, 4, 0, 2, -1]
    assert int(decoded.prg_chosen_mask.sum()) == 5
    assert decoded.final_blank_count == 1
    selected = decoded.action_ue_select[decoded.action_ue_select >= 0]
    assert selected.tolist() == [0, 3, 4]
    assert selected.numel() == torch.unique(selected).numel()
    assert decoded.non_hard_override_count == 0
    for token, ue in enumerate(decoded.chosen_ue_per_prg.tolist()):
        if ue < 0:
            continue
        native_idx = int(graph.snapshot.static_meta.prg_local_index[token]) * 3
        native_idx += int(graph.snapshot.static_meta.prg_owner_cell[token])
        slot = int(decoded.action_prg_alloc[native_idx])
        assert int(decoded.action_ue_select[slot]) == ue


def test_materialized_action_passes_type0_preflight():
    graph = make_candidate_graph(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    choice = torch.zeros((packed.num_prg_tokens,), dtype=torch.long)

    decoded = materialize_candidate_choices_to_type0(graph, packed, choice)

    assert decoded.chosen_ue_per_prg.tolist() == [0, 0, 2, 2, 4, 4]
    assert decoded.action_ue_select.tolist() == [0, -1, 2, -1, 4, -1]
    assert decoded.action_prg_alloc.tolist() == [0, 2, 4, 0, 2, 4]
    assert int(decoded.prg_chosen_mask.sum()) == 6
    reject = type0_action_preflight_reject_counts(
        graph, decoded.action_ue_select, decoded.action_prg_alloc
    )
    assert sum(float(value.sum()) for value in reject.values()) == 0.0


def test_first_local_prg_occurrence_determines_slot_order_when_tokens_are_shuffled():
    graph = make_candidate_graph(n_prg=3)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    permutation = torch.tensor([1, 0, 2, 4, 3, 5, 7, 6, 8])
    graph = _replace_static_meta(
        graph,
        prg_owner_cell=graph.snapshot.static_meta.prg_owner_cell[permutation],
        prg_local_index=graph.snapshot.static_meta.prg_local_index[permutation],
    )
    packed = PackedPrgCandidates(
        ue_ids=packed.ue_ids[permutation],
        mask=packed.mask[permutation],
        counts=packed.counts[permutation],
        edge_attr=packed.edge_attr[permutation] if packed.edge_attr is not None else None,
    )
    blank = packed.blank_class_index
    choice = torch.tensor([0, 1, 1, blank, blank, blank, blank, blank, blank])

    decoded = materialize_candidate_choices_to_type0(graph, packed, choice)

    assert decoded.chosen_ue_per_prg.tolist() == [0, 1, 1, -1, -1, -1, -1, -1, -1]
    assert decoded.action_ue_select.tolist() == [1, 0, -1, -1, -1, -1]
    assert decoded.action_prg_alloc.tolist() == [0, -1, -1, 1, -1, -1, 0, -1, -1]
    assert decoded.prg_chosen_mask.tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert decoded.final_blank_count == 6
    assert decoded.illegal_choice_count == 0
    assert decoded.slot_capacity_drop_count == 0


def test_global_ue_mask_rejects_exact_choice_without_fallback():
    graph = make_candidate_graph(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    action_mask_ue = graph.snapshot.static_meta.action_mask_ue.clone()
    action_mask_ue[0] = False
    graph = _replace_static_meta(graph, action_mask_ue=action_mask_ue)
    choice = torch.tensor([0, 2, 2, 2, 2, 2])

    decoded = materialize_candidate_choices_to_type0(graph, packed, choice)

    assert decoded.chosen_ue_per_prg.tolist() == [-1, -1, -1, -1, -1, -1]
    assert decoded.action_ue_select.tolist() == [-1, -1, -1, -1, -1, -1]
    assert decoded.action_prg_alloc.tolist() == [-1, -1, -1, -1, -1, -1]
    assert decoded.final_blank_count == 6
    assert decoded.illegal_choice_count == 1
    assert decoded.slot_capacity_drop_count == 0
    assert decoded.non_hard_override_count == 0


def test_cell_ue_mask_rejects_exact_choice_without_fallback():
    graph = make_candidate_graph(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    action_mask_cell_ue = graph.snapshot.static_meta.action_mask_cell_ue.clone()
    action_mask_cell_ue[0, 0] = False
    graph = _replace_static_meta(graph, action_mask_cell_ue=action_mask_cell_ue)
    choice = torch.tensor([0, 2, 2, 2, 2, 2])

    decoded = materialize_candidate_choices_to_type0(graph, packed, choice)

    assert decoded.chosen_ue_per_prg.tolist() == [-1, -1, -1, -1, -1, -1]
    assert decoded.action_ue_select.tolist() == [-1, -1, -1, -1, -1, -1]
    assert decoded.action_prg_alloc.tolist() == [-1, -1, -1, -1, -1, -1]
    assert decoded.final_blank_count == 6
    assert decoded.illegal_choice_count == 1
    assert decoded.slot_capacity_drop_count == 0
    assert decoded.non_hard_override_count == 0


def test_prg_cell_mask_rejects_exact_choice_without_fallback():
    graph = make_candidate_graph(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    action_mask_prg_cell = graph.snapshot.static_meta.action_mask_prg_cell.clone()
    action_mask_prg_cell[0, 0] = False
    graph = _replace_static_meta(graph, action_mask_prg_cell=action_mask_prg_cell)
    choice = torch.tensor([0, 2, 2, 2, 2, 2])

    decoded = materialize_candidate_choices_to_type0(graph, packed, choice)

    assert decoded.chosen_ue_per_prg.tolist() == [-1, -1, -1, -1, -1, -1]
    assert decoded.action_ue_select.tolist() == [-1, -1, -1, -1, -1, -1]
    assert decoded.action_prg_alloc.tolist() == [-1, -1, -1, -1, -1, -1]
    assert decoded.final_blank_count == 6
    assert decoded.illegal_choice_count == 1
    assert decoded.slot_capacity_drop_count == 0
    assert decoded.non_hard_override_count == 0


def test_invalid_candidate_class_is_illegal_not_blank_or_fallback():
    graph = make_candidate_graph(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    choice = torch.tensor([3, 2, 2, 2, 2, 2])

    decoded = materialize_candidate_choices_to_type0(graph, packed, choice)

    assert decoded.chosen_ue_per_prg.tolist() == [-1, -1, -1, -1, -1, -1]
    assert decoded.action_ue_select.tolist() == [-1, -1, -1, -1, -1, -1]
    assert decoded.action_prg_alloc.tolist() == [-1, -1, -1, -1, -1, -1]
    assert decoded.final_blank_count == 6
    assert decoded.illegal_choice_count == 1
    assert decoded.slot_capacity_drop_count == 0
    assert decoded.non_hard_override_count == 0


def test_slot_capacity_overflow_drops_exact_choice_without_fallback():
    graph = make_candidate_graph(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    graph = _replace_static_meta(graph, n_sched_ue=3)
    choice = torch.tensor([2, 2, 0, 1, 2, 2])

    decoded = materialize_candidate_choices_to_type0(graph, packed, choice)

    assert decoded.chosen_ue_per_prg.tolist() == [-1, -1, 2, -1, -1, -1]
    assert decoded.action_ue_select.tolist() == [-1, -1, 2]
    assert decoded.action_prg_alloc.tolist() == [-1, 2, -1, -1, -1, -1]
    assert decoded.prg_chosen_mask.tolist() == [False, False, True, False, False, False]
    assert decoded.final_blank_count == 5
    assert decoded.illegal_choice_count == 0
    assert decoded.slot_capacity_drop_count == 1
    assert decoded.non_hard_override_count == 0

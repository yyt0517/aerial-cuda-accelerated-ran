from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from training.stageb_hgraph.graph_ops import (
    PackedPrgCandidates,
    pack_prg_candidates,
    type0_action_preflight_reject_counts,
)
from training.stageb_hgraph.independent_policy import materialize_independent_choices_to_type0
from training.stageb_hgraph.tests.graph_fixtures import make_candidate_graph


def _replace_meta(graph, **changes):
    return replace(
        graph,
        snapshot=replace(
            graph.snapshot,
            static_meta=replace(graph.snapshot.static_meta, **changes),
        ),
    )


def _replace_packed(packed, **changes):
    return PackedPrgCandidates(
        ue_ids=changes.get("ue_ids", packed.ue_ids),
        mask=changes.get("mask", packed.mask),
        counts=changes.get("counts", packed.counts),
        edge_attr=changes.get("edge_attr", packed.edge_attr),
    )


def test_identity_only_materializer_sorts_slots_and_passes_preflight() -> None:
    graph = make_candidate_graph(n_prg=3)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    blank = packed.blank_class_index
    # Cell-major token order: choose UE 1 then UE 0 in cell 0, UE 3 then UE 2
    # in cell 1, and UE 5 in cell 2. Slot order must still be ascending UE ID.
    choice = torch.tensor([1, 0, blank, 1, 0, blank, 1, blank, blank])

    decoded = materialize_independent_choices_to_type0(graph, packed, choice)

    assert decoded.action_ue_select.tolist() == [0, 1, 2, 3, 5, -1]
    assert decoded.action_prg_alloc.tolist() == [1, 3, 4, 0, 2, -1, -1, -1, -1]
    assert decoded.chosen_ue_per_prg.tolist() == [1, 0, -1, 3, 2, -1, 5, -1, -1]
    assert decoded.prg_chosen_mask.tolist() == [True, True, False, True, True, False, True, False, False]
    assert decoded.final_blank_count == 4
    rejects = type0_action_preflight_reject_counts(
        graph,
        decoded.action_ue_select,
        decoded.action_prg_alloc,
    )
    assert all(int(value.sum().item()) == 0 for value in rejects.values())
    for field in (
        "cap_drop_count",
        "fallback_success_count",
        "safe_fill_count",
        "non_hard_override_count",
        "illegal_choice_count",
        "slot_capacity_drop_count",
    ):
        assert getattr(decoded, field) == 0


def test_materializer_rejects_wrong_cell_duplicate_invalid_mask_and_bad_class() -> None:
    graph = make_candidate_graph(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    blank = packed.blank_class_index

    wrong_ids = packed.ue_ids.clone()
    wrong_ids[0, 0] = 2
    with pytest.raises(ValueError, match="wrong-cell"):
        materialize_independent_choices_to_type0(
            graph,
            _replace_packed(packed, ue_ids=wrong_ids),
            torch.tensor([0, blank, blank, blank, blank, blank]),
        )

    duplicate_ids = packed.ue_ids.clone()
    duplicate_ids[0, 1] = duplicate_ids[0, 0]
    with pytest.raises(ValueError, match="duplicate"):
        materialize_independent_choices_to_type0(
            graph,
            _replace_packed(packed, ue_ids=duplicate_ids),
            torch.full((6,), blank),
        )

    prg_mask = graph.snapshot.static_meta.action_mask_prg_cell.clone()
    prg_mask[0, 0] = False
    with pytest.raises(ValueError, match="PRG mask"):
        materialize_independent_choices_to_type0(
            _replace_meta(graph, action_mask_prg_cell=prg_mask),
            packed,
            torch.tensor([0, blank, blank, blank, blank, blank]),
        )

    with pytest.raises(ValueError, match="choice"):
        materialize_independent_choices_to_type0(
            graph,
            packed,
            torch.tensor([blank + 1, blank, blank, blank, blank, blank]),
        )


def test_materializer_rejects_missing_or_overflowed_slots() -> None:
    graph = make_candidate_graph(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    blank = packed.blank_class_index

    with pytest.raises(ValueError, match="slot"):
        materialize_independent_choices_to_type0(
            _replace_meta(graph, n_sched_ue=3),
            packed,
            torch.tensor([blank, blank, 0, 1, blank, blank]),
        )

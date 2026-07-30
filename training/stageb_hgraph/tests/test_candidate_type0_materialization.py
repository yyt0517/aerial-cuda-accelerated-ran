from __future__ import annotations

import torch

from training.stageb_hgraph.graph_ops import (
    materialize_candidate_choices_to_type0,
    pack_prg_candidates,
    type0_action_preflight_reject_counts,
)
from training.stageb_hgraph.tests.graph_fixtures import make_candidate_graph


def test_materialization_uses_unique_slots_and_preserves_chosen_ue():
    graph = make_candidate_graph(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    choice = torch.tensor([0, 0, 1, 1, 0, packed.blank_class_index])

    decoded = materialize_candidate_choices_to_type0(graph, packed, choice)

    selected = decoded.action_ue_select[decoded.action_ue_select >= 0]
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

    reject = type0_action_preflight_reject_counts(
        graph, decoded.action_ue_select, decoded.action_prg_alloc
    )
    assert sum(float(value.sum()) for value in reject.values()) == 0.0

from __future__ import annotations

from dataclasses import fields

import torch

from training.stageb_hgraph.graph_ops import PackedPrgCandidates
from training.stageb_hgraph.independent_blank_diagnostics import IndependentPolicyDiagnostics
from training.stageb_hgraph.rescue_cohort_diagnostics import (
    build_rescue_decision_diagnostics,
)
from training.stageb_hgraph.types import (
    GraphFeatureSnapshot,
    NamedFeatureBlock,
    SparseEntityGraph,
    StaticMeta,
)


def _two_cell_graph() -> SparseEntityGraph:
    meta = StaticMeta(
        n_cell=2,
        n_ue=3,
        n_sched_ue=3,
        n_prg=1,
        topology="test",
        action_mask_cell_ue=torch.tensor([[True, True, False], [False, False, True]]),
        action_mask_prg_cell=torch.ones((2, 1), dtype=torch.bool),
        cell_adjacency=torch.empty((0, 2), dtype=torch.long),
        cell_edge_attr=None,
        ue_serving_cell=torch.tensor([0, 0, 1]),
        prg_owner_cell=torch.tensor([0, 1]),
        prg_local_index=torch.tensor([0, 0]),
    )
    snapshot = GraphFeatureSnapshot(
        cell_features=NamedFeatureBlock(torch.zeros((2, 1)), ("cell",)),
        ue_features=NamedFeatureBlock(
            torch.tensor([
                [100.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                [200.0, 0.0, 0.0, 0.0, 0.0, 0.7],
                [300.0, 0.0, 0.0, 0.0, 0.0, 0.9],
            ]),
            ("f0", "f1", "f2", "f3", "f4", "tb_err_last"),
        ),
        prg_features=NamedFeatureBlock(torch.zeros((2, 1)), ("prg",)),
        static_meta=meta,
    )
    return SparseEntityGraph(snapshot=snapshot, edges={})


def _policy_diagnostics() -> IndependentPolicyDiagnostics:
    return IndependentPolicyDiagnostics(
        valid_ue_count=torch.tensor([2, 1]),
        p_blank=torch.tensor([0.2, 0.8]),
        p_service=torch.tensor([0.8, 0.2]),
        p_best_ue=torch.tensor([0.6, 0.2]),
        blank_margin=torch.tensor([-1.0, 1.0]),
        margin_valid=torch.tensor([True, True]),
        selected_blank=torch.tensor([False, True]),
        greedy_blank=torch.tensor([False, True]),
        changed_from_greedy=torch.tensor([False, False]),
        forced_blank=torch.tensor([False, False]),
        learned_blank=torch.tensor([False, True]),
        fragmentation_blank=torch.tensor([False, False]),
        fragmentation_rescued=torch.tensor([True, False]),
        conservative_blank=torch.tensor([False, True]),
        selected_forced_blank=torch.tensor([False, False]),
        selected_fragmentation_blank=torch.tensor([False, False]),
        selected_conservative_blank=torch.tensor([False, True]),
    )


def test_build_rescue_decision_diagnostics_gathers_selected_candidate_and_masks_blank() -> None:
    """Catches gathering a wrong class or retaining selected-candidate values for Blank."""
    graph = _two_cell_graph()
    packed = PackedPrgCandidates(
        ue_ids=torch.tensor([[0, 1], [2, -1]]),
        mask=torch.tensor([[True, True], [True, False]]),
        counts=torch.tensor([2, 1]),
        edge_attr=torch.tensor([
            [[10.0, 1.0, 0.0, 0.0, 0.1], [11.0, 2.0, 0.0, 0.0, 0.2]],
            [[20.0, 3.0, 0.0, 0.0, 0.3], [0.0, 0.0, 0.0, 0.0, 0.0]],
        ]),
    )
    result = build_rescue_decision_diagnostics(
        graph=graph,
        packed=packed,
        token_indices=torch.tensor([0, 1]),
        local_prg=0,
        selected=torch.tensor([1, 2]),
        service_bytes=torch.tensor([[100.0, 200.0], [300.0, 400.0]]),
        sequential_features=torch.tensor([
            [[0.9, 0.0, 0.0], [0.4, 0.0, 0.0]],
            [[0.8, 0.0, 0.0], [0.2, 0.0, 0.0]],
        ]),
        policy_diagnostics=_policy_diagnostics(),
    )

    assert result.selected_ue.tolist() == [1, -1]
    assert result.cell_index.tolist() == [0, 1]
    assert result.local_prg_index.tolist() == [0, 0]
    assert result.selected_nonblank.tolist() == [True, False]
    assert result.fragmentation_rescued.tolist() == [True, False]
    assert result.selected_forced_blank.tolist() == [False, False]
    assert result.selected_conservative_blank.tolist() == [False, True]
    assert result.estimated_service_bytes.tolist() == [200.0, 0.0]
    assert result.quality_db.tolist() == [11.0, 0.0]
    assert result.quality_gap_db.tolist() == [2.0, 0.0]
    assert torch.allclose(result.prg_risk, torch.tensor([0.2, 0.0]))
    assert torch.allclose(result.ue_tb_err_last, torch.tensor([0.7, 0.0]))
    assert torch.allclose(result.remaining_demand_ratio, torch.tensor([0.4, 0.0]))
    assert torch.allclose(result.p_blank, torch.tensor([0.2, 0.8]))
    assert torch.allclose(result.p_service, torch.tensor([0.8, 0.2]))
    assert torch.allclose(result.p_best_ue, torch.tensor([0.6, 0.2]))

    selected = result.selected_nonblank
    expected_success = torch.sigmoid(
        (
            result.quality_db[selected]
            - 0.35 * result.quality_gap_db[selected]
            - 7.5 * result.prg_risk[selected]
            - 6.0 * result.ue_tb_err_last[selected]
        )
        / 4.0
    )
    assert torch.allclose(result.success_proxy[selected], expected_success)
    assert result.success_proxy[~selected].tolist() == [0.0]
    lengths = {getattr(result, field.name).numel() for field in fields(result)}
    assert lengths == {2}
    for field in fields(result):
        value = getattr(result, field.name)
        assert value.device.type == "cpu"
        assert value.dim() == 1
        assert not value.requires_grad

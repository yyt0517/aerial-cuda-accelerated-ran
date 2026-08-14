from __future__ import annotations

from dataclasses import fields

import pytest
import numpy as np

import torch

from training.stageb_hgraph.graph_ops import PackedPrgCandidates
from training.stageb_hgraph.independent_blank_diagnostics import IndependentPolicyDiagnostics
from training.stageb_hgraph.rescue_cohort_diagnostics import (
    RescueCellTtiRow,
    RescueDecisionDiagnostics,
    RescueUeTtiRow,
    build_rescue_decision_diagnostics,
    join_rescue_cohort_tti,
    summarize_rescue_cohorts,
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


def _rescue_decisions() -> RescueDecisionDiagnostics:
    """Cell-major decisions for two cells and two local PRGs."""
    return RescueDecisionDiagnostics(
        selected_ue=torch.tensor([0, 1, 0, -1]),
        cell_index=torch.tensor([0, 1, 0, 1]),
        local_prg_index=torch.tensor([0, 0, 1, 1]),
        selected_nonblank=torch.tensor([True, True, True, False]),
        fragmentation_rescued=torch.tensor([True, False, False, False]),
        selected_forced_blank=torch.tensor([False, False, False, True]),
        selected_conservative_blank=torch.tensor([False, False, False, False]),
        estimated_service_bytes=torch.tensor([7.0, 10.0, 8.0, 0.0]),
        quality_db=torch.tensor([1.0, 3.0, 2.0, 0.0]),
        quality_gap_db=torch.tensor([0.1, 0.3, 0.2, 0.0]),
        prg_risk=torch.tensor([0.01, 0.03, 0.02, 0.0]),
        ue_tb_err_last=torch.tensor([0.1, 0.3, 0.2, 0.0]),
        remaining_demand_ratio=torch.tensor([0.9, 0.7, 0.8, 0.0]),
        p_blank=torch.tensor([0.1, 0.2, 0.3, 0.4]),
        p_service=torch.tensor([0.9, 0.8, 0.7, 0.6]),
        p_best_ue=torch.tensor([0.6, 0.5, 0.4, 0.3]),
        success_proxy=torch.tensor([0.8, 0.7, 0.6, 0.0]),
    )


def _outcome_info() -> dict[str, object]:
    return {
        "accepted_pre_pdsch_ue_prg_count": np.array([2, 1]),
        "actual_ue_prg_count": np.array([0, 1]),
        "actual_ue_served_bytes": np.array([0.0, 20.0]),
        "actual_ue_goodput_bytes": np.array([0.0, 15.0]),
        "actual_ue_tb_tx_count": np.array([0, 1]),
        "actual_ue_tb_err_count": np.array([0, 0]),
    }


def test_join_rescue_cohort_keeps_zero_actual_rescue_ue_at_ue_tti_resolution() -> None:
    """Catches PRG-level outcome attribution or dropping an accepted zero-actual UE."""
    decisions = _rescue_decisions()
    ue_rows, cell_rows = join_rescue_cohort_tti(
        decisions,
        chosen_ue_per_prg=torch.tensor([0, 0, 1, -1]),
        info=_outcome_info(),
        n_ue=2,
        n_cell=2,
        n_prg=2,
        tti=17,
    )

    assert [(row.ue, row.rescue_exposed) for row in ue_rows] == [(0, True), (1, False)]
    rescue_row = ue_rows[0]
    assert rescue_row.planned_prg_count == 2
    assert rescue_row.rescued_prg_count == 1
    assert rescue_row.accepted_prg_count == 2
    assert rescue_row.actual_prg_count == 0
    assert rescue_row.scheduled_bytes == 0.0
    assert rescue_row.goodput_bytes == 0.0
    assert rescue_row.predicted_service_bytes == 15.0
    assert rescue_row.early_rescue_count == 1
    assert rescue_row.middle_rescue_count == 0
    assert rescue_row.late_rescue_count == 0
    assert [(row.cell, row.rescue_count, row.first_rescue_local_prg) for row in cell_rows] == [
        (0, 1, 0),
        (1, 0, -1),
    ]
    assert cell_rows[1].forced_before_or_without_prior_rescue == 1
    reconstructed = torch.full((4,), -1, dtype=torch.long)
    for row in ue_rows:
        for decision_index, selected_ue in enumerate(decisions.selected_ue.tolist()):
            if selected_ue == row.ue:
                token = int(decisions.cell_index[decision_index].item()) * 2 + int(
                    decisions.local_prg_index[decision_index].item()
                )
                reconstructed[token] = row.ue
    assert torch.equal(reconstructed, torch.tensor([0, 0, 1, -1]))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda info: info.update(actual_ue_prg_count=np.array([0, 1, 2])), "shape"),
        (lambda info: info.update(actual_ue_served_bytes=np.array([-1.0, 20.0])), "non-negative"),
        (lambda info: info.update(actual_ue_goodput_bytes=np.array([0.0, np.inf])), "finite"),
        (lambda info: info.update(actual_ue_goodput_bytes=np.array([0.0, 21.0])), "goodput"),
        (lambda info: info.update(actual_ue_tb_err_count=np.array([0, 2])), "tb_err"),
        (lambda info: info.update(accepted_pre_pdsch_ue_prg_count=np.array([3, 1])), "accepted"),
        (lambda info: info.update(actual_ue_prg_count=np.array([0, 2])), "actual"),
    ],
)
def test_join_rescue_cohort_rejects_invalid_outcome_evidence(change, message: str) -> None:
    """Catches treating malformed protocol outcomes as zero-valued cohort evidence."""
    info = _outcome_info()
    change(info)
    with pytest.raises(ValueError, match=message):
        join_rescue_cohort_tti(
            _rescue_decisions(),
            chosen_ue_per_prg=torch.tensor([0, 0, 1, -1]),
            info=info,
            n_ue=2,
            n_cell=2,
            n_prg=2,
            tti=17,
        )


def test_join_rescue_cohort_rejects_nonzero_actual_evidence_for_unplanned_ue() -> None:
    """Catches silently omitting a UE that received simulator evidence without a plan."""
    info = _outcome_info()
    for key, values in tuple(info.items()):
        info[key] = np.append(np.asarray(values), 0.0 if "bytes" in key else 0)
    info["actual_ue_prg_count"] = np.array([0, 1, 1])
    info["actual_ue_served_bytes"] = np.array([0.0, 20.0, 3.0])
    info["actual_ue_goodput_bytes"] = np.array([0.0, 15.0, 2.0])
    with pytest.raises(ValueError, match="unplanned"):
        join_rescue_cohort_tti(
            _rescue_decisions(),
            chosen_ue_per_prg=torch.tensor([0, 0, 1, -1]),
            info=info,
            n_ue=3,
            n_cell=2,
            n_prg=2,
            tti=17,
        )


def _row(*, tti: int, ue: int, rescue: bool, actual: int, planned: int, accepted: int,
         predicted: float, scheduled: float, goodput: float, tx: int, err: int) -> RescueUeTtiRow:
    return RescueUeTtiRow(
        tti=tti, ue=ue, rescue_exposed=rescue, planned_prg_count=planned,
        rescued_prg_count=1 if rescue else 0, accepted_prg_count=accepted,
        actual_prg_count=actual, predicted_service_bytes=predicted,
        scheduled_bytes=scheduled, goodput_bytes=goodput, tb_tx_count=tx,
        tb_err_count=err, quality_db_sum=float(planned * 2),
        quality_gap_db_sum=float(planned), prg_risk_sum=float(planned) / 10.0,
        ue_tb_err_last_sum=float(planned) / 5.0,
        remaining_demand_ratio_sum=float(planned) / 2.0,
        success_proxy_sum=float(planned) * 0.8, p_blank_sum=float(planned) * 0.1,
        p_service_sum=float(planned) * 0.9, p_best_ue_sum=float(planned) * 0.6,
        early_rescue_count=1 if rescue else 0, middle_rescue_count=0,
        late_rescue_count=0,
    )


def test_summarize_rescue_cohorts_uses_total_denominators_and_disjoint_actual_strata() -> None:
    """Catches averaging row ratios or assigning zero-actual rows a fabricated service rate."""
    ue_rows = [
        _row(tti=10, ue=0, rescue=True, actual=0, planned=2, accepted=2, predicted=15, scheduled=0, goodput=0, tx=0, err=0),
        _row(tti=10, ue=1, rescue=False, actual=2, planned=1, accepted=1, predicted=10, scheduled=20, goodput=10, tx=2, err=1),
        _row(tti=11, ue=0, rescue=True, actual=2, planned=1, accepted=1, predicted=10, scheduled=20, goodput=18, tx=2, err=0),
        _row(tti=11, ue=1, rescue=False, actual=6, planned=3, accepted=2, predicted=30, scheduled=60, goodput=45, tx=6, err=2),
    ]
    cell_rows = [
        RescueCellTtiRow(tti=10, cell=0, rescue_count=1, first_rescue_local_prg=0,
                          decisions_before_or_without_prior_rescue=2, decisions_after_prior_rescue=1,
                          forced_before_or_without_prior_rescue=1, forced_after_prior_rescue=0,
                          conservative_before_or_without_prior_rescue=0, conservative_after_prior_rescue=1),
    ]
    summary = summarize_rescue_cohorts(ue_rows, cell_rows)

    assert summary["schema_version"] == 1
    assert summary["unit"] == "ue_tti"
    assert set(summary["cohorts"]) == {"rescue_exposed", "ordinary_service"}
    assert set(summary["actual_prg_strata"]) == {"zero", "1_2", "3_5", "ge_6"}
    raw_keys = {
        "ue_tti_count", "tti_count", "planned_prg_count", "rescued_prg_count",
        "accepted_prg_count", "actual_prg_count", "predicted_service_bytes",
        "scheduled_bytes", "goodput_bytes", "failed_bytes", "tb_tx_count",
        "tb_err_count", "early_rescue_count", "middle_rescue_count", "late_rescue_count",
    }
    derived_keys = {
        "predicted_service_bytes_per_planned_prg", "scheduled_bytes_per_actual_prg",
        "goodput_bytes_per_actual_prg", "failed_byte_ratio", "tb_error_ratio",
        "planned_to_accepted_ratio", "accepted_to_actual_ratio", "quality_db_mean",
        "quality_gap_db_mean", "prg_risk_mean", "ue_tb_err_last_mean",
        "remaining_demand_ratio_mean", "success_proxy_mean", "p_blank_mean",
        "p_service_mean", "p_best_ue_mean",
    }
    ordinary = summary["cohorts"]["ordinary_service"]
    assert raw_keys <= set(ordinary)
    assert derived_keys <= set(ordinary)
    assert ordinary["scheduled_bytes_per_actual_prg"] == 10.0
    assert ordinary["goodput_bytes_per_actual_prg"] == 55.0 / 8.0
    assert ordinary["failed_byte_ratio"] == 25.0 / 80.0
    assert ordinary["planned_to_accepted_ratio"] == 3.0 / 4.0
    assert ordinary["accepted_to_actual_ratio"] == 8.0 / 3.0
    zero = summary["actual_prg_strata"]["zero"]["rescue_exposed"]
    assert zero["ue_tti_count"] == 1
    assert zero["scheduled_bytes_per_actual_prg"] is None
    assert zero["goodput_bytes_per_actual_prg"] is None
    assert summary["actual_prg_strata"]["1_2"]["ordinary_service"]["ue_tti_count"] == 1
    assert summary["actual_prg_strata"]["3_5"]["ordinary_service"]["ue_tti_count"] == 0
    assert summary["actual_prg_strata"]["ge_6"]["ordinary_service"]["ue_tti_count"] == 1
    comparison = summary["per_tti_comparisons"]["goodput_bytes_per_actual_prg"]
    assert set(comparison) == {
        "mean_delta_rescue_minus_ordinary", "paired_tti_count", "direction_agreement_fraction"
    }

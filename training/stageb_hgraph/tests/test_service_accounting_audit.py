from __future__ import annotations

import numpy as np
import pytest
import torch

from training.stageb_hgraph.service_accounting_audit import (
    NEW_TRANSMISSION_QUEUE_DEQUEUE,
    ServiceAccountingDecisionDiagnostics,
    ServiceAccountingUeTtiRow,
    join_service_accounting_tti,
    summarize_service_accounting_audit,
)


def _decisions() -> ServiceAccountingDecisionDiagnostics:
    """One UE is planned on local PRGs {0, 2}; the other decisions are Blank."""
    return ServiceAccountingDecisionDiagnostics(
        selected_ue=torch.tensor([0, -1, -1, -1, 0, -1]),
        cell_index=torch.tensor([0, 1, 0, 1, 0, 1]),
        local_prg_index=torch.tensor([0, 0, 1, 1, 2, 2]),
        selected_nonblank=torch.tensor([True, False, False, False, True, False]),
        raw_service_bytes=torch.tensor([300.0, 0.0, 0.0, 0.0, 300.0, 0.0]),
        effective_service_bytes=torch.tensor([300.0, 0.0, 0.0, 0.0, 300.0, 0.0]),
        initial_demand_bytes=torch.tensor([1000.0, 0.0, 0.0, 0.0, 1000.0, 0.0]),
        remaining_demand_before_bytes=torch.tensor(
            [1000.0, 0.0, 0.0, 0.0, 700.0, 0.0]
        ),
        quality_db=torch.tensor([3.0, 0.0, 0.0, 0.0, 6.0, 0.0]),
    )


def _info(*, actual_mask: np.ndarray | None = None) -> dict[str, object]:
    accepted = np.array([[1, 0, 1], [0, 0, 0]], dtype=np.uint8)
    return {
        "harq_enabled": 0,
        "accepted_pre_pdsch_ue_prg_mask": accepted,
        "actual_ue_prg_mask": accepted.copy() if actual_mask is None else actual_mask,
        "actual_ue_served_bytes": np.array([400.0, 0.0]),
        "actual_ue_goodput_bytes": np.array([350.0, 0.0]),
    }


def test_join_requires_exact_local_prg_sets_not_equal_counts() -> None:
    """Catches accepting planned/actual PRG sets that have equal cardinality but different IDs."""
    rows = join_service_accounting_tti(
        _decisions(),
        info=_info(actual_mask=np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)),
        n_ue=2,
        n_cell=2,
        n_prg=3,
        tti=9,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.planned_prg_ids == (0, 2)
    assert row.accepted_prg_ids == (0, 2)
    assert row.actual_prg_ids == (0, 1)
    assert row.eligible is False
    assert row.ineligible_reason == "actual_prg_set_mismatch"


def test_join_marks_only_uncapped_new_data_queue_rows_eligible() -> None:
    """Catches demand-clipped predictions entering the calibration denominator."""
    rows = join_service_accounting_tti(
        _decisions(),
        info=_info(),
        n_ue=2,
        n_cell=2,
        n_prg=3,
        tti=9,
    )

    row = rows[0]
    assert row.eligible is True
    assert row.ineligible_reason is None
    assert row.raw_predicted_service_bytes == 600.0
    assert row.scheduled_bytes == 400.0
    assert row.successful_bytes == 350.0
    assert row.initial_backlog_bytes == 1000.0
    assert row.minimum_predecision_margin_bytes == 400.0

    clipped = _decisions()
    clipped = ServiceAccountingDecisionDiagnostics(
        **{
            **clipped.__dict__,
            "remaining_demand_before_bytes": torch.tensor(
                [1000.0, 0.0, 0.0, 0.0, 250.0, 0.0]
            ),
        }
    )
    clipped_row = join_service_accounting_tti(
        clipped,
        info=_info(),
        n_ue=2,
        n_cell=2,
        n_prg=3,
        tti=10,
    )[0]
    assert clipped_row.eligible is False
    assert clipped_row.ineligible_reason == "predecision_demand_cap"


def test_join_rejects_scheduled_bytes_when_harq_is_enabled() -> None:
    """Catches calibrating new-data headroom with bytes that may include retransmissions."""
    info = _info()
    info["harq_enabled"] = 1

    with pytest.raises(ValueError, match="HARQ-disabled"):
        join_service_accounting_tti(
            _decisions(), info=info, n_ue=2, n_cell=2, n_prg=3, tti=9
        )


def _audit_row(
    *,
    tti: int,
    ue: int,
    cell: int,
    quality: float,
    ratio: float,
    prgs: tuple[int, ...] = (0, 1, 2, 3),
) -> ServiceAccountingUeTtiRow:
    predicted = 600.0
    return ServiceAccountingUeTtiRow(
        tti=tti,
        ue=ue,
        cell=cell,
        planned_prg_ids=prgs,
        accepted_prg_ids=prgs,
        actual_prg_ids=prgs,
        raw_predicted_service_bytes=predicted,
        scheduled_bytes=predicted * ratio,
        successful_bytes=predicted * ratio * 0.9,
        initial_backlog_bytes=1000.0,
        minimum_predecision_margin_bytes=400.0,
        quality_db_mean=quality,
        eligible=True,
        ineligible_reason=None,
    )


def test_summary_passes_frozen_global_cell_and_equal_count_quality_gates() -> None:
    """Catches changing the frozen 2/3 tolerance or computing bins after ratios."""
    rows = [
        _audit_row(
            tti=index,
            ue=index,
            cell=index % 3,
            quality=float(index),
            ratio=2.0 / 3.0,
        )
        for index in range(150)
    ]

    report = summarize_service_accounting_audit(
        rows,
        scheduled_bytes_semantics=NEW_TRANSMISSION_QUEUE_DEQUEUE,
        per_prg_scheduled_bytes_available=False,
    )

    assert report["status"] == "PASS"
    assert report["eligible_ue_tti_count"] == 150
    assert report["eligible_actual_prg_count"] == 600
    assert report["global"]["scheduled_to_raw_predicted_ratio"] == pytest.approx(2.0 / 3.0)
    assert report["global"]["gate_passed"] is True
    assert [group["sample_count"] for group in report["quality_groups"]] == [50, 50, 50]
    assert all(group["is_gate_group"] for group in report["quality_groups"])
    assert all(group["gate_passed"] for group in report["quality_groups"])
    assert report["prg_segment_service_ratio_gate_enabled"] is False


def test_summary_stops_when_a_populated_cell_is_not_globally_stable() -> None:
    """Catches allowing a global average to hide a cell-dependent service bias."""
    ratios = {0: 0.56, 1: 0.72, 2: 0.72}
    rows = [
        _audit_row(
            tti=index,
            ue=index,
            cell=index % 3,
            quality=float((index * 17) % 150),
            ratio=ratios[index % 3],
        )
        for index in range(150)
    ]

    report = summarize_service_accounting_audit(
        rows,
        scheduled_bytes_semantics=NEW_TRANSMISSION_QUEUE_DEQUEUE,
        per_prg_scheduled_bytes_available=False,
    )

    assert report["global"]["gate_passed"] is True
    assert report["status"] == "FAIL"
    cell0 = next(group for group in report["cell_groups"] if group["cell"] == 0)
    assert cell0["sample_count"] == 50
    assert cell0["actual_prg_count"] == 200
    assert cell0["is_gate_group"] is True
    assert cell0["gate_passed"] is False


def test_summary_reports_small_groups_without_gating_them() -> None:
    """Catches treating sparse strata as formal evidence against a global constant."""
    rows = [
        _audit_row(tti=index, ue=index, cell=0, quality=float(index), ratio=2.0 / 3.0)
        for index in range(100)
    ]
    rows.append(_audit_row(tti=100, ue=100, cell=1, quality=100.0, ratio=0.2))

    report = summarize_service_accounting_audit(
        rows,
        scheduled_bytes_semantics=NEW_TRANSMISSION_QUEUE_DEQUEUE,
        per_prg_scheduled_bytes_available=False,
    )

    cell1 = next(group for group in report["cell_groups"] if group["cell"] == 1)
    assert cell1["is_gate_group"] is False
    assert cell1["gate_passed"] is None


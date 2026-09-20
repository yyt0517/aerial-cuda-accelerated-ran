from __future__ import annotations

import pytest
import torch

from training.stageb_hgraph.blank_expiry_attribution_audit import (
    HARD_MASK,
    MIXED_INCONCLUSIVE,
    SELECTED_PHY_FAILED,
    SELECTED_SUCCESS_INSUFFICIENT,
    VALID_BUT_BLANK,
    VALID_BUT_OTHER_UE,
    V2_FULL_HORIZON_PROTOCOL_ID,
    AtomicExposure,
    BlankExpiryDecisionDiagnostics,
    BlankExpiryTtiTrace,
    ExpiredPacket,
    build_atomic_exposures,
    build_blank_expiry_attribution_report,
    canonical_compact_cpu_fingerprints,
    canonical_evaluator_progress_fingerprint,
    canonical_packet_outcome_fingerprint,
    canonical_type0_action_fingerprints,
    classify_exposure,
    classify_v2_full_horizon_result,
    combine_view_conclusions,
    dominant_category,
    post_mask_probabilities,
    read_formal_expired_packets,
)


@pytest.mark.parametrize(
    ("valid", "selected_ue", "target_ue", "tb_tx", "tb_err", "expected"),
    [
        (False, -1, 7, 0, 0, HARD_MASK),
        (True, -1, 7, 0, 0, VALID_BUT_BLANK),
        (True, 8, 7, 1, 0, VALID_BUT_OTHER_UE),
        (True, 7, 7, 1, 1, SELECTED_PHY_FAILED),
        (True, 7, 7, 1, 0, SELECTED_SUCCESS_INSUFFICIENT),
    ],
)
def test_classify_exposure_partitions_all_five_primary_categories(
    valid: bool,
    selected_ue: int,
    target_ue: int,
    tb_tx: int,
    tb_err: int,
    expected: str,
) -> None:
    assert classify_exposure(
        target_valid=valid,
        selected_ue=selected_ue,
        target_ue=target_ue,
        tb_tx_count=tb_tx,
        tb_err_count=tb_err,
    ) == expected


def test_classify_exposure_rejects_selected_target_without_tb_outcome() -> None:
    with pytest.raises(ValueError, match="selected target UE has no TB outcome"):
        classify_exposure(
            target_valid=True,
            selected_ue=7,
            target_ue=7,
            tb_tx_count=0,
            tb_err_count=0,
        )


def test_dominance_accepts_exact_50_percent_and_10_point_lead() -> None:
    result = dominant_category(
        {HARD_MASK: 50.0, VALID_BUT_BLANK: 40.0, VALID_BUT_OTHER_UE: 10.0}
    )
    assert result.category == HARD_MASK
    assert result.share == pytest.approx(0.5)
    assert result.lead == pytest.approx(0.1)


@pytest.mark.parametrize(
    "counts",
    [
        {HARD_MASK: 49.0, VALID_BUT_BLANK: 39.0, VALID_BUT_OTHER_UE: 12.0},
        {HARD_MASK: 50.0, VALID_BUT_BLANK: 40.0001, VALID_BUT_OTHER_UE: 9.9999},
    ],
)
def test_dominance_rejects_below_either_frozen_boundary(
    counts: dict[str, float],
) -> None:
    assert dominant_category(counts).category == MIXED_INCONCLUSIVE


def test_cross_view_conflict_is_mixed_inconclusive() -> None:
    atomic = dominant_category({HARD_MASK: 60.0, VALID_BUT_BLANK: 40.0})
    ue_tti = dominant_category({VALID_BUT_BLANK: 70.0, HARD_MASK: 30.0})
    assert combine_view_conclusions(atomic, ue_tti) == MIXED_INCONCLUSIVE


def test_post_mask_probability_normalizes_only_valid_actions() -> None:
    probabilities = post_mask_probabilities(
        torch.tensor([2.0, 100.0, 1.0, 0.0]),
        torch.tensor([True, False, True, True]),
    )
    assert probabilities[1].item() == 0.0
    assert probabilities.sum().item() == pytest.approx(1.0)
    expected = torch.softmax(torch.tensor([2.0, 1.0, 0.0]), dim=0)
    torch.testing.assert_close(probabilities[[0, 2, 3]], expected)


def _atom(
    *,
    tti: int,
    local_prg: int,
    category: str,
    packet_id: int = 1,
    ue: int = 4,
    weight: int = 1200,
) -> AtomicExposure:
    return AtomicExposure(
        packet_id=packet_id,
        ue=ue,
        tti=tti,
        serving_cell=1,
        local_prg=local_prg,
        category=category,
        blank_subcategory="conservative" if category == VALID_BUT_BLANK else None,
        expiry_byte_weight=weight,
    )


def test_report_uses_expiry_byte_weight_and_ue_tti_modal_view() -> None:
    packet = ExpiredPacket(
        packet_id=1,
        ue=4,
        arrival_tti=2,
        terminal_tti=3,
        packet_size_bytes=3000,
        remaining_bytes_at_expiry=1200,
    )
    atoms = [
        _atom(tti=tti, local_prg=prg, category=category)
        for tti, categories in (
            (2, [HARD_MASK, HARD_MASK, VALID_BUT_BLANK]),
            (3, [VALID_BUT_BLANK, VALID_BUT_BLANK, VALID_BUT_BLANK]),
        )
        for prg, category in enumerate(categories)
    ]
    report = build_blank_expiry_attribution_report(
        packets=[packet],
        exposures=atoms,
        measured_tti_first=3,
        measured_tti_last=3,
        trace_first_tti=2,
        n_prg=3,
    )
    assert report["coverage"]["expected_atomic_count"] == 6
    assert report["coverage"]["unclassified_count"] == 0
    assert report["atomic_view"]["weighted_counts"][HARD_MASK] == 2400
    assert report["atomic_view"]["weighted_counts"][VALID_BUT_BLANK] == 4800
    assert report["ue_tti_view"]["unweighted_counts"][HARD_MASK] == 1
    assert report["ue_tti_view"]["unweighted_counts"][VALID_BUT_BLANK] == 1


def test_report_fails_closed_when_any_lifecycle_decision_is_missing() -> None:
    packet = ExpiredPacket(
        packet_id=1,
        ue=4,
        arrival_tti=2,
        terminal_tti=3,
        packet_size_bytes=3000,
        remaining_bytes_at_expiry=1200,
    )
    atoms = [
        _atom(tti=2, local_prg=0, category=HARD_MASK),
        _atom(tti=2, local_prg=1, category=HARD_MASK),
        _atom(tti=3, local_prg=0, category=VALID_BUT_BLANK),
        _atom(tti=3, local_prg=1, category=VALID_BUT_BLANK),
    ]
    with pytest.raises(ValueError, match="lifecycle coverage mismatch"):
        build_blank_expiry_attribution_report(
            packets=[packet],
            exposures=atoms,
            measured_tti_first=3,
            measured_tti_last=3,
            trace_first_tti=2,
            n_prg=3,
        )


def test_report_fails_closed_when_formal_packet_arrived_before_trace() -> None:
    packet = ExpiredPacket(
        packet_id=1,
        ue=4,
        arrival_tti=1,
        terminal_tti=3,
        packet_size_bytes=3000,
        remaining_bytes_at_expiry=1200,
    )
    with pytest.raises(ValueError, match="arrival_tti precedes trace_first_tti"):
        build_blank_expiry_attribution_report(
            packets=[packet],
            exposures=[],
            measured_tti_first=3,
            measured_tti_last=3,
            trace_first_tti=2,
            n_prg=3,
        )


def _decision_diagnostics() -> BlankExpiryDecisionDiagnostics:
    return BlankExpiryDecisionDiagnostics(
        cell_index=torch.tensor([0, 1]),
        local_prg_index=torch.tensor([0, 0]),
        candidate_ue_ids=torch.tensor([[4, -1], [8, -1]]),
        candidate_present=torch.tensor([[True, False], [True, False]]),
        candidate_valid=torch.tensor([[True, False], [True, False]]),
        candidate_logit=torch.tensor([[1.0, -4.0], [2.0, -4.0]]),
        candidate_probability=torch.tensor([[0.4, 0.0], [0.8, 0.0]]),
        initial_backlog_bytes=torch.tensor([[3000.0, 0.0], [3000.0, 0.0]]),
        remaining_demand_before_bytes=torch.tensor([[3000.0, 0.0], [3000.0, 0.0]]),
        cumulative_capacity_before_bytes=torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        hol_delay_ms=torch.tensor([[10.0, 0.0], [10.0, 0.0]]),
        ttl_slack_ms=torch.tensor([[190.0, -1.0], [190.0, -1.0]]),
        selected_ue=torch.tensor([-1, 8]),
        selected_class=torch.tensor([2, 0]),
        selected_rank=torch.tensor([1, 1]),
        valid_nonblank_count=torch.tensor([1, 1]),
        blank_logit=torch.tensor([1.5, 0.0]),
        blank_probability=torch.tensor([0.6, 0.2]),
        best_valid_ue_logit=torch.tensor([1.0, 2.0]),
        service_probability=torch.tensor([0.4, 0.8]),
        blank_subcategory_code=torch.tensor([2, 0]),
    )


def test_build_atomic_exposures_joins_target_choice_and_tb_outcome() -> None:
    packet = ExpiredPacket(1, 4, 2, 2, 3000, 1200)
    trace = BlankExpiryTtiTrace(
        tti=2,
        decisions=_decision_diagnostics(),
        actual_ue_served_bytes=(0.0,) * 9,
        actual_ue_goodput_bytes=(0.0,) * 9,
        actual_ue_tb_tx_count=(0,) * 9,
        actual_ue_tb_err_count=(0,) * 9,
    )

    atoms = build_atomic_exposures(
        packets=[packet],
        traces=[trace],
        serving_cell_by_ue=(0, 0, 0, 0, 0, 1, 1, 1, 1),
        n_cell=2,
        n_prg=1,
    )

    assert len(atoms) == 1
    assert atoms[0].category == VALID_BUT_BLANK
    assert atoms[0].blank_subcategory == "conservative"


def test_read_formal_expired_packets_requires_enriched_lifecycle_columns(
    tmp_path,
) -> None:
    path = tmp_path / "packet_outcome_samples.csv"
    path.write_text(
        "ue_id,packet_id,arrival_tti,packet_size_bytes,remaining_bytes_at_terminal,"
        "terminal_tti,outcome,system_time_ms\n"
        "4,1,602,3000,1200,1001,expired,200.0\n"
        "4,2,603,3000,0,1002,delivered,1.0\n",
        encoding="utf-8",
    )
    packets = read_formal_expired_packets(
        path,
        measured_tti_first=1001,
        measured_tti_last=1512,
        trace_first_tti=602,
    )
    assert packets == [ExpiredPacket(1, 4, 602, 1001, 3000, 1200)]


def test_read_formal_expired_packets_fails_closed_without_lifecycle_columns(
    tmp_path,
) -> None:
    path = tmp_path / "packet_outcome_samples.csv"
    path.write_text(
        "ue_id,packet_id,terminal_tti,outcome,system_time_ms\n"
        "4,1,1001,expired,200.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="required lifecycle columns"):
        read_formal_expired_packets(
            path,
            measured_tti_first=1001,
            measured_tti_last=1512,
            trace_first_tti=602,
        )


def test_v2_full_horizon_result_prioritizes_trajectory_nonreproduction() -> None:
    assert classify_v2_full_horizon_result(
        trajectory_reproduced=False,
        expired_packet_count=0,
        attribution_conclusion=MIXED_INCONCLUSIVE,
    ) == "formal_trajectory_nonreproduction"


@pytest.mark.parametrize("expired_packet_count", [0, 49])
def test_v2_full_horizon_result_requires_50_expired_packets(
    expired_packet_count: int,
) -> None:
    assert classify_v2_full_horizon_result(
        trajectory_reproduced=True,
        expired_packet_count=expired_packet_count,
        attribution_conclusion=VALID_BUT_BLANK,
    ) == "insufficient_event_support"


def test_v2_full_horizon_result_applies_attribution_at_50_events() -> None:
    assert classify_v2_full_horizon_result(
        trajectory_reproduced=True,
        expired_packet_count=50,
        attribution_conclusion=VALID_BUT_BLANK,
    ) == VALID_BUT_BLANK


def test_report_records_prospective_v2_protocol() -> None:
    report = build_blank_expiry_attribution_report(
        packets=[],
        exposures=[],
        measured_tti_first=1001,
        measured_tti_last=5000,
        trace_first_tti=602,
        n_prg=17,
        protocol_id=V2_FULL_HORIZON_PROTOCOL_ID,
    )
    assert report["protocol_id"] == V2_FULL_HORIZON_PROTOCOL_ID


def test_type0_action_fingerprint_is_order_independent_and_complete() -> None:
    rows = [
        {"tti": 1001, "serving_cell": cell, "local_prg": prg, "selected_ue": 10 * cell + prg}
        for prg in range(2)
        for cell in range(2)
    ]
    forward = canonical_type0_action_fingerprints(
        rows,
        tti_first=1001,
        tti_last=1001,
        n_cell=2,
        n_prg=2,
    )
    reverse = canonical_type0_action_fingerprints(
        list(reversed(rows)),
        tti_first=1001,
        tti_last=1001,
        n_cell=2,
        n_prg=2,
    )
    assert forward == reverse
    assert forward["tti_count"] == 1
    assert len(forward["global_sha256"]) == 64
    assert list(forward["per_tti_sha256"]) == ["1001"]

    with pytest.raises(ValueError, match="complete cell-PRG coverage"):
        canonical_type0_action_fingerprints(
            rows[:-1],
            tti_first=1001,
            tti_last=1001,
            n_cell=2,
            n_prg=2,
        )


def test_packet_outcome_fingerprint_matches_legacy_and_enriched_csv(tmp_path) -> None:
    legacy = tmp_path / "legacy.csv"
    enriched = tmp_path / "enriched.csv"
    legacy.write_text(
        "ue_id,packet_id,terminal_tti,outcome,system_time_ms\n"
        "2,7,1001,delivered,0.5\n2,8,1002,expired,200.0\n",
        encoding="utf-8",
    )
    enriched.write_text(
        "ue_id,packet_id,arrival_tti,packet_size_bytes,remaining_bytes_at_terminal,"
        "terminal_tti,outcome,system_time_ms\n"
        "2,7,1001,3000,0,1001,delivered,0.500000\n"
        "2,8,603,3000,1200,1002,expired,200.000000\n",
        encoding="utf-8",
    )
    reference = canonical_packet_outcome_fingerprint(
        legacy, tti_first=1001, tti_last=1002
    )
    observed = canonical_packet_outcome_fingerprint(
        enriched, tti_first=1001, tti_last=1002
    )
    assert reference == observed
    assert reference["delivered_count"] == 1
    assert reference["expired_count"] == 1


def test_compact_cpu_fingerprint_resolves_ring_slots_by_tti(tmp_path) -> None:
    path = tmp_path / "output_short.txt"
    path.write_text(
        "mcType0SumCellThrRecordsCpu = [50 20 30 40];\n"
        "mcType0SumCellPfRecordsCpu = [5 2 3 4];\n"
        "mcType0SumInsThrRecordsCpu = [500 200 300 400];\n"
        "mcType0SumCellThrRecordsGpu = [0 0 0 0];\n",
        encoding="utf-8",
    )
    result = canonical_compact_cpu_fingerprints(
        path,
        tti_first=2,
        tti_last=3,
        completed_steps=5,
        ring_horizon=4,
    )
    assert result["tti_count"] == 2
    assert result["series"]["mcType0SumCellThrRecordsCpu"]["values"] == ["20", "30"]
    assert "mcType0SumCellThrRecordsGpu" not in result["series"]


def test_evaluator_progress_fingerprint_ignores_instrumentation_summary_only(
    tmp_path,
) -> None:
    reference = tmp_path / "reference.log"
    observed = tmp_path / "observed.log"
    progress = (
        "[hgraph-eval] step=500/5000 measured=0 goodput=0.0\n"
        "[hgraph-eval] step=1000/5000 measured=0 goodput=0.0\n"
    )
    reference.write_text(
        progress + '{"completed_steps":5000,"audit_enabled":false}\n',
        encoding="utf-8",
    )
    observed.write_text(
        progress + '{"completed_steps":5000,"audit_enabled":true,"trace_count":4399}\n',
        encoding="utf-8",
    )

    assert canonical_evaluator_progress_fingerprint(reference) == (
        canonical_evaluator_progress_fingerprint(observed)
    )

    observed.write_text(
        progress.replace("goodput=0.0", "goodput=1.0", 1)
        + '{"completed_steps":5000,"audit_enabled":true}\n',
        encoding="utf-8",
    )
    assert canonical_evaluator_progress_fingerprint(reference) != (
        canonical_evaluator_progress_fingerprint(observed)
    )

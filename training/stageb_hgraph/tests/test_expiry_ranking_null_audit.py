from __future__ import annotations

import pytest

from training.stageb_hgraph.expiry_ranking_null_audit import (
    PacketRecord,
    SelectionOpportunity,
    active_ue_tti_union,
    classify_urgency_response,
    classify_underselection,
    derive_runtime_association,
    extract_target_opportunity,
    match_delivered_controls,
    normalized_ue_tti_byte_efficiency,
    summarize_selection_opportunities,
    ttl_slack_bin,
)


def packet(packet_id: int, arrival: int, terminal: int, outcome: str) -> PacketRecord:
    return PacketRecord(
        packet_id=packet_id,
        ue=2,
        arrival_tti=arrival,
        terminal_tti=terminal,
        outcome=outcome,
        packet_size_bytes=3000,
        remaining_bytes_at_terminal=744 if outcome == "expired" else 0,
    )


def test_active_union_deduplicates_overlapping_packets_for_the_same_ue() -> None:
    active = active_ue_tti_union(
        [packet(10, 100, 102, "expired"), packet(11, 102, 104, "expired")]
    )
    assert active == {(2, 100), (2, 101), (2, 102), (2, 103), (2, 104)}


def test_runtime_association_comes_from_actor_candidate_rows() -> None:
    rows = [
        {
            "serving_cell": 0,
            "candidate_ue_ids": [8, -1],
            "candidate_present": [True, False],
        },
        {
            "serving_cell": 2,
            "candidate_ue_ids": [2, 3],
            "candidate_present": [True, True],
        },
    ]
    assert derive_runtime_association(rows) == {2: 2, 3: 2, 8: 0}


def test_runtime_association_rejects_a_ue_seen_in_multiple_cells() -> None:
    rows = [
        {"serving_cell": 0, "candidate_ue_ids": [2], "candidate_present": [True]},
        {"serving_cell": 2, "candidate_ue_ids": [2], "candidate_present": [True]},
    ]
    with pytest.raises(ValueError, match="multiple runtime serving cells"):
        derive_runtime_association(rows)


def test_matching_breaks_equal_arrival_distance_by_packet_identity() -> None:
    expired = [packet(50, 100, 110, "expired")]
    delivered = [
        packet(9, 102, 102, "delivered"),
        packet(7, 98, 98, "delivered"),
    ]
    result = match_delivered_controls(expired, delivered)
    assert [(left.packet_id, right.packet_id) for left, right in result.pairs] == [(50, 7)]
    assert result.coverage == 1.0


def test_matching_is_without_replacement_and_reports_coverage() -> None:
    result = match_delivered_controls(
        [packet(50, 100, 110, "expired"), packet(51, 101, 111, "expired")],
        [packet(7, 100, 100, "delivered")],
    )
    assert len(result.pairs) == 1
    assert result.coverage == 0.5


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (100.0001, "gt_100_ms"),
        (100.0, "gt_50_le_100_ms"),
        (50.0001, "gt_50_le_100_ms"),
        (50.0, "gt_20_le_50_ms"),
        (20.0001, "gt_20_le_50_ms"),
        (20.0, "ge_0_le_20_ms"),
        (0.0, "ge_0_le_20_ms"),
        (-0.001, "negative"),
        (None, "missing"),
    ],
)
def test_ttl_slack_bins_use_frozen_actor_time_boundaries(value, expected) -> None:
    assert ttl_slack_bin(value) == expected


def test_zero_selected_prg_is_missing_efficiency_not_zero() -> None:
    assert normalized_ue_tti_byte_efficiency(
        scheduled_bytes=0.0,
        successful_bytes=0.0,
        selected_prg_count=0,
    ) is None
    assert normalized_ue_tti_byte_efficiency(
        scheduled_bytes=900.0,
        successful_bytes=600.0,
        selected_prg_count=3,
    ) == {"scheduled_bytes_per_selected_prg": 300.0, "successful_bytes_per_selected_prg": 200.0}


def test_null_ratio_excludes_blank_and_uses_post_mask_nonblank_count() -> None:
    rows = [
        SelectionOpportunity(2, 100, 0, 14, 2, 1, 0.0, 80.0, 120.0),
        SelectionOpportunity(2, 100, 1, 7, 9, 3, 1.5, 80.0, 120.0),
        SelectionOpportunity(2, 100, 2, 5, -1, 2, None, 80.0, 120.0),
    ]
    result = summarize_selection_opportunities(rows)
    assert result["nonblank_opportunity_count"] == 2
    assert result["blank_excluded_count"] == 1
    assert result["selected_target_count"] == 1
    assert result["uniform_expected_selected_count"] == pytest.approx(1 / 14 + 1 / 7)
    assert result["selection_ratio_to_uniform"] == pytest.approx(1 / (1 / 14 + 1 / 7))
    assert result["post_mask_valid_nonblank_ue_count"]["mean"] == 10.5


def test_null_ratio_reports_zero_denominator_without_dividing() -> None:
    result = summarize_selection_opportunities([])
    assert result["uniform_expected_selected_count"] == 0.0
    assert result["selection_ratio_to_uniform"] is None
    assert result["status"] == "zero_denominator"


def test_target_rank_and_margin_use_only_post_mask_valid_nonblank_ues() -> None:
    decision = {
        "tti": 1001,
        "serving_cell": 2,
        "local_prg": 4,
        "candidate_ue_ids": [2, 5, 7, -1],
        "candidate_present": [True, True, True, False],
        "candidate_valid": [True, False, True, False],
        "candidate_logits": [1.0, 99.0, 3.5, -10.0],
        "ttl_slack_ms": [20.0, 80.0, 40.0, -1.0],
        "hol_delay_ms": [180.0, 120.0, 160.0, 0.0],
        "selected_ue": 7,
    }
    row = extract_target_opportunity(decision, target_ue=2)
    assert row is not None
    assert row.valid_nonblank_count == 2
    assert row.target_rank == 2
    assert row.chosen_target_logit_margin == 2.5
    assert row.ttl_slack_ms == 20.0


def test_selected_target_has_zero_margin() -> None:
    decision = {
        "tti": 1001,
        "serving_cell": 2,
        "local_prg": 4,
        "candidate_ue_ids": [2],
        "candidate_present": [True],
        "candidate_valid": [True],
        "candidate_logits": [1.0],
        "ttl_slack_ms": [20.0],
        "hol_delay_ms": [180.0],
        "selected_ue": 2,
    }
    assert extract_target_opportunity(decision, target_ue=2).chosen_target_logit_margin == 0.0


@pytest.mark.parametrize(
    ("expired", "control", "full", "raw", "disagreement", "formal"),
    [
        (0.80, 1.00, 1.10, "material_underselection", False, "material_underselection"),
        (0.95, 1.00, 1.10, "weak_directional_underselection", False, "weak_directional_underselection"),
        (1.00, 1.20, 1.30, "underselection_not_supported", False, "underselection_not_supported"),
        (0.80, 1.00, 0.70, "material_underselection", True, "sensitivity_disagreement"),
    ],
)
def test_three_level_underselection_and_sensitivity_override(
    expired, control, full, raw, disagreement, formal
) -> None:
    result = classify_underselection(
        expired_ratio=expired,
        matched_control_ratio=control,
        full_delivered_ratio=full,
    )
    assert result["matched_control_classification"] == raw
    assert result["sensitivity_disagreement"] is disagreement
    assert result["formal_conclusion"] == formal


def test_low_match_coverage_overrides_control_conclusion() -> None:
    result = classify_underselection(
        expired_ratio=0.5,
        matched_control_ratio=1.0,
        full_delivered_ratio=1.1,
        match_coverage=0.899,
    )
    assert result["formal_conclusion"] == "insufficient_match_coverage"


def test_zero_selection_denominator_is_not_reported_as_no_underselection() -> None:
    result = classify_underselection(
        expired_ratio=None,
        matched_control_ratio=1.0,
        full_delivered_ratio=1.0,
    )
    assert result["formal_conclusion"] == "insufficient_opportunity_support"


@pytest.mark.parametrize(
    ("means", "expected"),
    [
        ([8.0, 7.0, 6.0], "monotonic_rank_improvement"),
        ([6.0, 7.0, 8.0], "rank_not_improved_as_ttl_declines"),
        ([8.0, 6.0, 7.0], "mixed_nonmonotonic_urgency_response"),
        ([7.0, 7.0, 7.0], "rank_not_improved_as_ttl_declines"),
    ],
)
def test_urgency_response_distinguishes_monotonic_mixed_and_no_improvement(
    means, expected
) -> None:
    assert classify_urgency_response(means) == expected


def test_urgency_response_requires_two_nonempty_ttl_bins() -> None:
    assert classify_urgency_response([None, 7.0, None]) == "insufficient_ttl_bin_support"

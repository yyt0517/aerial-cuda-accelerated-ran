from __future__ import annotations

import csv
from pathlib import Path

import pytest

from training.stageb_hgraph.service_balance_audit import (
    canonical_offered_packet_stream,
    classify_service_balance,
    load_service_balance_trace,
    load_terminal_outcomes,
    material_deficit,
    summarize_fixed_windows,
    validate_trace_coverage,
)


SERVICE_FIELDS = (
    "tti",
    "ue_id",
    "serving_cell",
    "offered_bytes",
    "accepted_bytes",
    "rejected_bytes",
    "expired_bytes",
    "previous_post_queue_bytes",
    "pre_queue_bytes",
    "scheduled_dequeue_bytes",
    "post_queue_bytes",
    "successful_bytes",
    "failed_bytes",
    "demand_valid_prg_count",
    "actual_prg_count",
    "actual_prg_mask",
    "tbs_capacity_bytes",
    "mcs",
    "layer",
    "tb_tx_count",
    "tb_err_count",
    "hol_delay_ms",
    "ttl_slack_ms",
)


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _service_row(tti: int, **updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "tti": tti,
        "ue_id": 2,
        "serving_cell": 2,
        "offered_bytes": 3000,
        "accepted_bytes": 3000,
        "rejected_bytes": 0,
        "expired_bytes": 0,
        "previous_post_queue_bytes": 1000,
        "pre_queue_bytes": 4000,
        "scheduled_dequeue_bytes": 1500,
        "post_queue_bytes": 2500,
        "successful_bytes": 1200,
        "failed_bytes": 300,
        "demand_valid_prg_count": 17,
        "actual_prg_count": 2,
        "actual_prg_mask": "0x00003",
        "tbs_capacity_bytes": 1500,
        "mcs": 4,
        "layer": 2,
        "tb_tx_count": 1,
        "tb_err_count": 0,
        "hol_delay_ms": 1.5,
        "ttl_slack_ms": 198.5,
    }
    row.update(updates)
    return row


def test_offered_packet_hash_covers_identity_arrival_size_and_ttl(tmp_path: Path) -> None:
    fields = ("ue_id", "packet_id", "arrival_tti", "packet_size_bytes", "ttl_tti")
    rows = [
        {"ue_id": 2, "packet_id": 11, "arrival_tti": 7, "packet_size_bytes": 3000, "ttl_tti": 400},
        {"ue_id": 1, "packet_id": 10, "arrival_tti": 7, "packet_size_bytes": 2500, "ttl_tti": 400},
    ]
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    changed = tmp_path / "changed.csv"
    _write_csv(left, fields, rows)
    _write_csv(right, fields, list(reversed(rows)))
    changed_rows = [dict(row) for row in rows]
    changed_rows[0]["ttl_tti"] = 399
    _write_csv(changed, fields, changed_rows)

    left_fingerprint = canonical_offered_packet_stream(left)
    right_fingerprint = canonical_offered_packet_stream(right)
    changed_fingerprint = canonical_offered_packet_stream(changed)

    assert left_fingerprint == right_fingerprint
    assert left_fingerprint["packet_count"] == 2
    assert left_fingerprint["sha256"] != changed_fingerprint["sha256"]


def test_service_trace_enforces_event_order_and_all_byte_identities(tmp_path: Path) -> None:
    valid = tmp_path / "valid.csv"
    _write_csv(valid, SERVICE_FIELDS, [_service_row(602)])
    rows = load_service_balance_trace(valid, trace_first_tti=602, trace_last_tti=602)
    assert rows[0]["post_queue_bytes"] == 2500

    bad_queue = tmp_path / "bad_queue.csv"
    _write_csv(bad_queue, SERVICE_FIELDS, [_service_row(602, pre_queue_bytes=3999)])
    with pytest.raises(ValueError, match="pre-queue conservation"):
        load_service_balance_trace(bad_queue, trace_first_tti=602, trace_last_tti=602)

    bad_phy = tmp_path / "bad_phy.csv"
    _write_csv(bad_phy, SERVICE_FIELDS, [_service_row(602, failed_bytes=299)])
    with pytest.raises(ValueError, match="successful.*failed"):
        load_service_balance_trace(bad_phy, trace_first_tti=602, trace_last_tti=602)


def test_fixed_windows_are_exactly_tti_1001_to_5000_in_100_tti_blocks() -> None:
    rows = []
    previous = 0
    for tti in range(1001, 5001):
        row = _service_row(
            tti,
            offered_bytes=1,
            accepted_bytes=1,
            previous_post_queue_bytes=previous,
            pre_queue_bytes=previous + 1,
            scheduled_dequeue_bytes=1,
            post_queue_bytes=previous,
            successful_bytes=1,
            failed_bytes=0,
            actual_prg_count=1,
            tbs_capacity_bytes=1,
        )
        rows.append(row)
    windows = summarize_fixed_windows(rows, ue_id=2)
    assert len(windows) == 40
    assert (windows[0]["tti_first"], windows[0]["tti_last"]) == (1001, 1100)
    assert (windows[-1]["tti_first"], windows[-1]["tti_last"]) == (4901, 5000)
    assert all(window["accepted_bytes"] == 100 for window in windows)


def test_trace_coverage_requires_every_ue_at_every_tti_and_fixed_association() -> None:
    rows = [
        _service_row(tti, ue_id=ue, serving_cell=ue // 2)
        for tti in (602, 603)
        for ue in range(4)
    ]
    result = validate_trace_coverage(
        rows,
        expected_ue_ids=range(4),
        trace_first_tti=602,
        trace_last_tti=603,
    )
    assert result["row_count"] == 8
    assert result["association_counts"] == [2, 2]

    with pytest.raises(ValueError, match="coverage"):
        validate_trace_coverage(
            rows[:-1],
            expected_ue_ids=range(4),
            trace_first_tti=602,
            trace_last_tti=603,
        )
    changed = [dict(row) for row in rows]
    changed[-1]["serving_cell"] = 0
    with pytest.raises(ValueError, match="association"):
        validate_trace_coverage(
            changed,
            expected_ue_ids=range(4),
            trace_first_tti=602,
            trace_last_tti=603,
        )


def test_deadline_urgency_uses_decision_time_ttl_slack() -> None:
    rows = []
    previous = 0
    for tti in range(1001, 5001):
        ttl = 20.0 if tti == 1001 else (50.0 if tti == 1002 else 101.0)
        rows.append(
            _service_row(
                tti,
                offered_bytes=1,
                accepted_bytes=1,
                previous_post_queue_bytes=previous,
                pre_queue_bytes=previous + 1,
                scheduled_dequeue_bytes=1,
                post_queue_bytes=previous,
                successful_bytes=1,
                failed_bytes=0,
                actual_prg_count=1,
                tbs_capacity_bytes=1,
                ttl_slack_ms=ttl,
            )
        )
    first = summarize_fixed_windows(rows, ue_id=2)[0]
    assert first["critical_20ms_scheduled_dequeue_bytes"] == 1
    assert first["urgent_50ms_scheduled_dequeue_bytes"] == 2


def test_terminal_outcomes_only_include_formal_terminal_interval(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.csv"
    path.write_text(
        "ue_id,packet_id,arrival_tti,packet_size_bytes,remaining_bytes_at_terminal,"
        "terminal_tti,outcome,system_time_ms\n"
        "2,1,601,3000,500,1000,expired,200.0\n"
        "2,2,602,3000,400,1001,expired,200.0\n"
        "2,3,1001,3000,0,1002,delivered,1.0\n"
        "2,4,4601,3000,3000,5000,expired,200.0\n"
        "2,5,4602,3000,3000,5001,expired,200.0\n",
        encoding="utf-8",
    )

    result = load_terminal_outcomes(path, ue_id=2)

    assert result["terminal_packet_count"] == 3
    assert result["expired_packet_count"] == 2
    assert result["expired_bytes"] == 3400
    assert result["terminal_tti_first"] == 1001
    assert result["terminal_tti_last"] == 5000


def test_material_deficit_freezes_ratio_and_30_of_40_boundaries() -> None:
    pfq = [{"scheduled_dequeue_bytes": 100.0} for _ in range(40)]
    exactly_thirty = [
        {"scheduled_dequeue_bytes": 80.0 if index < 30 else 120.0}
        for index in range(40)
    ]
    only_twenty_nine = [
        {"scheduled_dequeue_bytes": 80.0 if index < 29 else 120.0}
        for index in range(40)
    ]
    ratio_too_high = [
        {"scheduled_dequeue_bytes": 90.0001}
        for _ in range(40)
    ]

    assert material_deficit(pfq, exactly_thirty, "scheduled_dequeue_bytes")["material"] is True
    assert material_deficit(pfq, only_twenty_nine, "scheduled_dequeue_bytes")["material"] is False
    assert material_deficit(pfq, ratio_too_high, "scheduled_dequeue_bytes")["material"] is False


@pytest.mark.parametrize(
    ("accepted", "scheduled", "allocation", "conditional_tbs", "deadline", "expected"),
    [
        (True, True, True, False, False, "admission_difference"),
        (False, True, True, False, False, "allocation_deficit"),
        (False, True, False, True, False, "conditional_tbs_deficit"),
        (False, False, False, False, True, "deadline_timing_deficit"),
        (False, True, False, False, False, "inconclusive"),
    ],
)
def test_classification_requires_the_frozen_mechanism_evidence(
    accepted: bool,
    scheduled: bool,
    allocation: bool,
    conditional_tbs: bool,
    deadline: bool,
    expected: str,
) -> None:
    assert classify_service_balance(
        accepted_arrival_deficit=accepted,
        scheduled_service_deficit=scheduled,
        allocation_deficit=allocation,
        conditional_tbs_deficit=conditional_tbs,
        deadline_timing_deficit=deadline,
    ) == expected

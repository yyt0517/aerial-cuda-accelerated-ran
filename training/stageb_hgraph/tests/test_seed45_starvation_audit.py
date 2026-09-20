from training.stageb_hgraph.seed45_starvation_audit import (
    protocol_identity,
    summarize_starvation,
)


def test_protocol_identity_records_the_frozen_evaluation_seed():
    assert protocol_identity(48) == {
        "protocol_id": "dimensionless-service-starvation-audit-v2",
        "seed": 48,
    }


def _row(tti: int, ue_id: int, backlog: int, scheduled: int, expired: int = 0):
    return {
        "tti": tti,
        "ue_id": ue_id,
        "pre_queue_bytes": backlog,
        "scheduled_dequeue_bytes": scheduled,
        "expired_bytes": expired,
        "actual_prg_count": 1 if scheduled else 0,
        "successful_bytes": scheduled,
        "failed_bytes": 0,
    }


def test_starvation_audit_uses_scheduled_dequeue_and_frozen_streak_threshold():
    rows = []
    for tti in range(1001, 1401):
        rows.extend(
            (
                _row(tti, 0, 12000, 0, expired=3000 if tti == 1400 else 0),
                _row(tti, 1, 12000, 3000 if tti % 2 == 0 else 0),
                _row(tti, 2, 0, 0),
            )
        )

    outcomes = [
        {
            "ue_id": 0,
            "packet_size_bytes": 3000,
            "remaining_bytes_at_terminal": 3000,
            "outcome": "expired",
            "system_time_ms": 200.0,
        },
        {
            "ue_id": 1,
            "packet_size_bytes": 3000,
            "remaining_bytes_at_terminal": 0,
            "outcome": "delivered",
            "system_time_ms": 1.0,
        },
    ]
    report = summarize_starvation(
        rows,
        outcome_rows=outcomes,
        streak_threshold_tti=400,
    )

    assert report["passed"] is False
    assert report["guard_required"] is True
    assert report["starved_ue_ids"] == [0]
    ue0, ue1, ue2 = report["per_ue"]
    assert ue0["scheduled_dequeue_bytes"] == 0
    assert ue0["max_backlogged_zero_dequeue_streak_tti"] == 400
    assert ue0["expired_bytes"] == 3000
    assert ue0["starved_zero_total_scheduled"] is True
    assert ue0["starved_long_streak"] is True
    assert ue1["scheduled_dequeue_bytes"] == 600000
    assert ue1["actual_prg_count"] == 200
    assert ue1["delivered_packet_count"] == 1
    assert ue1["expired_packet_count"] == 0
    assert ue1["outcome_delay_mean_ms"] == 1.0
    assert ue1["expiry_aware_rate_mbps"] == 24.0
    assert ue1["max_backlogged_zero_dequeue_streak_tti"] == 1
    assert ue1["starved"] is False
    assert ue2["backlogged_tti_count"] == 0
    assert ue2["starved"] is False

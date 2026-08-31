import math

from training.stageb_hgraph.dpa_v1_historical_comparison import (
    WindowSpec,
    summarize_dpa_window,
)


def _service(tti, ue_id, backlog, scheduled, successful, expired=0, prgs=0):
    return {
        "tti": tti,
        "ue_id": ue_id,
        "pre_queue_bytes": backlog,
        "scheduled_dequeue_bytes": scheduled,
        "successful_bytes": successful,
        "failed_bytes": scheduled - successful,
        "expired_bytes": expired,
        "actual_prg_count": prgs,
    }


def test_steady_window_includes_expired_outcomes_and_detects_zero_service():
    rows = [
        _service(1, 0, 3000, 3000, 3000, prgs=1),
        _service(1, 1, 3000, 0, 0),
        _service(2, 0, 3000, 3000, 3000, prgs=1),
        _service(2, 1, 3000, 0, 0),
        _service(3, 0, 3000, 0, 0),
        _service(3, 1, 6000, 0, 0, expired=3000),
        _service(4, 0, 3000, 3000, 3000, prgs=1),
        _service(4, 1, 6000, 0, 0),
    ]
    outcomes = [
        {"terminal_tti": 2, "ue_id": 0, "outcome": "delivered", "packet_size_bytes": 3000, "system_time_ms": 1.0},
        {"terminal_tti": 3, "ue_id": 0, "outcome": "expired", "packet_size_bytes": 3000, "system_time_ms": 200.0},
        {"terminal_tti": 3, "ue_id": 1, "outcome": "expired", "packet_size_bytes": 3000, "system_time_ms": 200.0},
    ]
    summary = {
        "mean_goodput_mbps": 100.0,
        "mean_tb_err_rate": 0.20,
        "mean_expiry_drop_rate": 0.10,
        "mean_prg_utilization_ratio": 0.80,
        "mean_blank_ratio": 0.20,
        "stats_mean_goodput_mbps": 123.0,
        "stats_mean_tb_err_rate": 0.15,
        "stats_mean_expiry_drop_rate": 0.08,
        "stats_mean_prg_utilization_ratio": 0.75,
        "stats_mean_blank_ratio": 0.25,
    }

    result = summarize_dpa_window(
        eval_summary=summary,
        service_rows=rows,
        outcome_rows=outcomes,
        window=WindowSpec("steady_3000", 2, 4),
        slot_duration_s=0.0005,
    )

    assert result["tti_count"] == 3
    assert result["goodput_mbps"] == 123.0
    assert result["expired_bytes"] == 3000
    assert result["outcome_count"] == 3
    assert math.isclose(result["outcome_delay_mean_ms"], 401.0 / 3.0)
    assert result["outcome_delay_p50_ms"] == 200.0
    assert result["outcome_delay_p95_ms"] == 200.0
    assert result["outcome_delay_max_ms"] == 200.0
    assert result["zero_scheduled_dequeue_ue_ids"] == [1]
    assert result["max_backlogged_zero_dequeue_streak_tti"] == 3
    assert result["ue_goodput_min_mbps"] == 0.0
    assert result["ue_goodput_mean_mbps"] == 16.0
    assert result["ue_goodput_jain"] == 0.5
    assert result["ue_expiry_aware_rate_min_mbps"] == 0.0

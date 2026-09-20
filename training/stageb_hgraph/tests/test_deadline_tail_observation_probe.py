from __future__ import annotations

import numpy as np

from training.stageb_hgraph import deadline_tail_observation_probe as probe


def test_target_service_diagnostics_reports_drain_deficit_and_allocations():
    """Catch target diagnostics silently falling back to the old hard-coded UE2."""
    result = probe._target_service_diagnostics(
        target_ue_id=1,
        feature_names=(
            "buffer_bytes",
            "ttl_slack_ms",
            "recent_scheduled_ratio",
            "recent_goodput_deficit_norm",
            "service_gap_tti",
        ),
        ue_features=np.asarray(
            [
                [1000.0, 10.0, 0.5, 0.2, 3.0],
                [4000.0, 1.0, 0.0, 1.0, 500.0],
                [300.0, 100.0, 0.8, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        scheduled_service_rate_bps=np.asarray([1.0e6, 0.0, 1.0e6]),
        active_ue_ids=[0, 1, 2],
        actor_chosen_ue=np.asarray([[0, 2], [2, -1]]),
        pfq_chosen_ue=np.asarray([[1, 2], [1, -1]]),
        slot_duration_s=0.0005,
    )

    assert result["ue_id"] == 1
    assert result["required_drain_bytes_per_tti"] == 2000.0
    assert result["recent_scheduled_bytes_per_tti"] == 0.0
    assert result["drain_deficit_bytes_per_tti"] == 2000.0
    assert result["drain_deficit_rank_desc"] == 1
    assert result["actor_prg_count"] == 0
    assert result["pfq_prg_count"] == 2
    assert result["features"]["service_gap_tti"] == 500.0


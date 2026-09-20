import numpy as np
import pytest

from training.stageb_hgraph.same_state_four_action_attribution import (
    action_structure_metrics,
    build_four_actions,
    counterfactual_deadline_pressure,
    deadline_state_by_ue,
    infer_recent_scheduled_dequeue,
    select_representative_tti,
    shapley_two_patch_contributions,
)


def test_build_four_actions_separates_ue_and_topology_patches() -> None:
    # Cell-major, two PRGs per cell.  Tokens 1 and 4 are both-active UE
    # replacements; tokens 2 and 3 change the active/blank topology.
    pfq = np.asarray([0, 1, 2, -1, 4, 5], dtype=np.int64)
    dpa = np.asarray([0, 6, -1, 3, 7, 5], dtype=np.int64)

    actions = build_four_actions(pfq, dpa)

    assert actions["pfq"].tolist() == [0, 1, 2, -1, 4, 5]
    assert actions["ue_reallocation_only"].tolist() == [0, 6, 2, -1, 7, 5]
    assert actions["topology_only"].tolist() == [0, 1, -1, 3, 4, 5]
    assert actions["dpa"].tolist() == [0, 6, -1, 3, 7, 5]


def test_shapley_two_patch_contributions_are_exactly_additive() -> None:
    result = shapley_two_patch_contributions(
        {
            "pfq": 10.0,
            "ue_reallocation_only": 16.0,
            "topology_only": 12.0,
            "dpa": 21.0,
        }
    )

    assert result["ue_reallocation_contribution"] == pytest.approx(7.5)
    assert result["topology_contribution"] == pytest.approx(3.5)
    assert result["total_delta"] == pytest.approx(11.0)
    assert result["additivity_residual"] == pytest.approx(0.0)


def test_deadline_pressure_uses_scheduled_dequeue_and_ttl_slack() -> None:
    backlog = np.asarray([100.0], dtype=np.float64)
    ttl_ms = np.asarray([5.0], dtype=np.float64)  # 10 TTIs at 0.5 ms.
    recent_service = np.asarray([5.0], dtype=np.float64)
    before_log_pressure = np.asarray([1.0854542], dtype=np.float64)

    inferred = infer_recent_scheduled_dequeue(
        backlog,
        ttl_ms,
        before_log_pressure,
        slot_duration_ms=0.5,
    )
    result = counterfactual_deadline_pressure(
        backlog,
        ttl_ms,
        recent_service,
        np.asarray([20.0], dtype=np.float64),
        slot_duration_ms=0.5,
        ewma_alpha=0.05,
    )

    assert inferred.tolist() == pytest.approx([5.0], abs=1.0e-5)
    assert result["recent_scheduled_dequeue_after"].tolist() == pytest.approx([5.75])
    assert result["backlog_after"].tolist() == pytest.approx([80.0])
    assert result["ttl_slack_tti_after"].tolist() == pytest.approx([9.0])
    assert result["phi_after"] < result["phi_before"]
    assert result["phi_delta"] > 0.0


def test_action_structure_counts_cross_cell_coactivity() -> None:
    action = np.asarray([0, 1, 2, -1, 4, 5], dtype=np.int64)

    result = action_structure_metrics(action, n_cell=3, n_prg=2)

    assert result == {
        "scheduled_cell_prg_count": 5,
        "coactive_cell_pair_count": 4,
        "full_reuse_prg_count": 1,
        "two_cell_prg_count": 1,
        "one_cell_prg_count": 0,
        "all_blank_prg_count": 0,
    }


def test_representative_tti_is_median_mechanism_case_not_maximum() -> None:
    rows = [
        {
            "tti": 1001,
            "scheduled_cell_prg_delta_ai_minus_pfq": 0,
            "dpa_minus_pfq_demand_capped_service_bytes": 10.0,
            "dpa_minus_pfq_phi_delta": 1.0,
        },
        {
            "tti": 1002,
            "scheduled_cell_prg_delta_ai_minus_pfq": 1,
            "dpa_minus_pfq_demand_capped_service_bytes": 20.0,
            "dpa_minus_pfq_phi_delta": 2.0,
        },
        {
            "tti": 1003,
            "scheduled_cell_prg_delta_ai_minus_pfq": -1,
            "dpa_minus_pfq_demand_capped_service_bytes": 30.0,
            "dpa_minus_pfq_phi_delta": 3.0,
        },
        {
            "tti": 1004,
            "scheduled_cell_prg_delta_ai_minus_pfq": -3,
            "dpa_minus_pfq_demand_capped_service_bytes": 20.0,
            "dpa_minus_pfq_phi_delta": 2.0,
        },
    ]

    selected = select_representative_tti(rows)

    assert selected["tti"] == 1002


def test_deadline_state_extracts_reportable_pre_action_fields() -> None:
    names = [
        "buffer_bytes",
        "ttl_slack_ms",
        "log1p_drain_pressure",
        "scheduled_dequeue_adequacy",
        "ttl_normalized_service_age",
    ]
    features = np.asarray(
        [[100.0, 2.0, 1.5, 0.25, 0.75], [50.0, 1.0, 2.5, 0.50, 0.90]],
        dtype=np.float64,
    )

    result = deadline_state_by_ue(features, names)

    assert result == {
        "backlog_bytes_by_ue": [100.0, 50.0],
        "ttl_slack_ms_by_ue": [2.0, 1.0],
        "log1p_drain_pressure_by_ue": [1.5, 2.5],
        "scheduled_dequeue_adequacy_by_ue": [0.25, 0.5],
        "ttl_normalized_service_age_by_ue": [0.75, 0.9],
    }

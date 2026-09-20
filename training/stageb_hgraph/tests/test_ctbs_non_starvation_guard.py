from __future__ import annotations

import math

import pytest
import torch

from training.stageb_hgraph.ctbs_headroom import CtbsHeadroomState
from training.stageb_hgraph.ctbs_non_starvation_guard import (
    CtbsNonStarvationGuardConfig,
    GuardReason,
    apply_ctbs_non_starvation_guard_step,
    concatenate_guard_diagnostics,
    guard_diagnostics_to_rows,
    summarize_guard_diagnostics,
)
from training.stageb_hgraph.phy_tbs_alignment_audit import PhyTbsConfig


FEATURE_NAMES = (
    "buffer_bytes",
    "avg_rate_mbps",
    "wb_sinr_lin",
    "cqi",
    "ri",
    "tb_err_last",
    "new_data_flag",
    "stale_slots",
    "hol_delay_ms",
    "ttl_slack_ms",
    "recent_scheduled_ratio",
    "recent_goodput_deficit_norm",
    "log1p_drain_pressure",
    "scheduled_dequeue_adequacy",
    "ttl_normalized_service_age",
    "reserved_zero",
)


def _phy_config() -> PhyTbsConfig:
    return PhyTbsConfig(
        mcs_lut_type=0,
        mcs_sinr_cap_db=25.99,
        layer_sinval_threshold=0.1,
        prbs_per_prg=16,
        pdsch_data_symbols=11,
        effective_layer_cap=4,
    )


def _states() -> tuple[CtbsHeadroomState, ...]:
    config = _phy_config()
    return tuple(
        CtbsHeadroomState(
            initial_demand_bytes=12_000.0,
            reconstructed_mcs=10,
            config=config,
        )
        for _ in range(6)
    )


def _features() -> torch.Tensor:
    features = torch.zeros((6, len(FEATURE_NAMES)), dtype=torch.float32)
    features[0, 12] = math.log(3.0)
    features[0, 13] = 0.5
    features[0, 14] = 0.5
    return features


def _run(
    *,
    features: torch.Tensor | None = None,
    states: tuple[CtbsHeadroomState, ...] | None = None,
    actor_choice: torch.Tensor | None = None,
    override_used_by_cell: torch.Tensor | None = None,
    local_prg: int = 0,
    prg_layer_cap: tuple[tuple[int, ...], ...] = ((4,) * 5,) * 6,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    use_default_states: bool = True,
) -> object:
    return apply_ctbs_non_starvation_guard_step(
        config=CtbsNonStarvationGuardConfig(enabled=True),
        blank_class_index=2,
        local_prg=local_prg,
        actor_choice=torch.tensor([2, 0, 0]) if actor_choice is None else actor_choice,
        candidate_ue_ids=torch.tensor([[0, 1], [2, 3], [4, 5]]),
        valid_main=torch.ones((3, 2), dtype=torch.bool),
        remaining_demand_bytes=torch.full((6,), 12000.0),
        ue_features=_features() if features is None else features,
        ue_feature_names=feature_names,
        ctbs_states=_states() if states is None and use_default_states else states,
        prg_layer_cap=prg_layer_cap,
        override_used_by_cell=(
            torch.zeros(3, dtype=torch.bool)
            if override_used_by_cell is None
            else override_used_by_cell
        ),
    )


def test_eligible_blank_uses_unique_urgent_ctbs_positive_candidate() -> None:
    """Catches a blank action failing to select its only eligible CTBS-positive UE."""
    features = _features()
    states = _states()

    result = apply_ctbs_non_starvation_guard_step(
        config=CtbsNonStarvationGuardConfig(enabled=True),
        blank_class_index=2,
        local_prg=0,
        actor_choice=torch.tensor([2, 0, 2]),
        candidate_ue_ids=torch.tensor([[0, 1], [2, 3], [4, 5]]),
        valid_main=torch.ones((3, 2), dtype=torch.bool),
        remaining_demand_bytes=torch.full((6,), 12000.0),
        ue_features=features,
        ue_feature_names=FEATURE_NAMES,
        ctbs_states=states,
        prg_layer_cap=((4,) * 3,) * 6,
        override_used_by_cell=torch.zeros(3, dtype=torch.bool),
    )

    assert result.executed_choice.tolist() == [0, 0, 2]
    assert result.override_used_by_cell.tolist() == [True, False, False]
    assert result.diagnostics.guard_override.tolist() == [True, False, False]
    assert result.diagnostics.selected_ue.tolist() == [0, -1, -1]
    assert (
        result.diagnostics.ctbs_capacity_after_bytes[0]
        > result.diagnostics.ctbs_capacity_before_bytes[0]
    )


@pytest.mark.parametrize(
    ("pressure", "adequacy", "age", "expected_override"),
    [
        (math.log(2.0), 0.5, 0.50, False),
        (math.log(2.0) + 1e-6, 1.0, 0.50, False),
        (math.log(2.0) + 1e-6, 0.5, 0.25 - 1e-6, False),
        (math.log(2.0) + 1e-6, 0.5, 0.25, True),
    ],
)
def test_pressure_eligibility_boundaries(
    pressure: float, adequacy: float, age: float, expected_override: bool
) -> None:
    """Catches changing strict pressure/adequacy or inclusive age thresholds."""
    features = _features()
    features[0, 12:15] = torch.tensor([pressure, adequacy, age])

    result = _run(features=features)

    assert bool(result.diagnostics.guard_override[0]) is expected_override


def test_nonblank_actor_choice_is_preserved() -> None:
    """Catches the guard replacing a legal nonblank actor selection."""
    result = _run(actor_choice=torch.tensor([1, 0, 0]))

    assert result.executed_choice.tolist() == [1, 0, 0]
    assert result.diagnostics.reason_code.tolist()[0] == GuardReason.ACTOR_NONBLANK


def test_used_cell_budget_blocks_second_replacement() -> None:
    """Catches more than one replacement in a cell during the same TTI."""
    result = _run(override_used_by_cell=torch.tensor([True, False, False]))

    assert result.executed_choice.tolist() == [2, 0, 0]
    assert result.override_used_by_cell.tolist() == [True, False, False]
    assert result.diagnostics.reason_code.tolist()[0] == GuardReason.CELL_BUDGET_USED


def test_nonpositive_cumulative_ctbs_delta_is_rejected() -> None:
    """Catches accepting a candidate whose exact cumulative TBS falls."""
    state = _states()[0]
    for prg in range(4):
        state = state.with_selected_prg(prg, layer_cap=4)
    states = (state,) + _states()[1:]

    result = _run(
        states=states,
        local_prg=4,
        prg_layer_cap=((4, 4, 4, 4, 3),) + ((4,) * 5,) * 5,
    )

    assert result.executed_choice.tolist() == [2, 0, 0]
    assert result.diagnostics.reason_code.tolist()[0] == GuardReason.NO_POSITIVE_CTBS_HEADROOM


@pytest.mark.parametrize(
    ("ue0", "ue1", "expected_ue"),
    [
        ((math.log(3.0), 0.5, 0.75), (math.log(9.0), 0.5, 0.5), 0),
        ((math.log(3.0), 0.5, 0.5), (math.log(9.0), 0.5, 0.5), 1),
        ((math.log(3.0), 0.5, 0.5), (math.log(3.0), 0.5, 0.5), 0),
    ],
)
def test_candidate_ranking_uses_age_then_pressure_then_smaller_ue_id(
    ue0: tuple[float, float, float], ue1: tuple[float, float, float], expected_ue: int
) -> None:
    """Catches ranking by pressure first or leaving final ties nondeterministic."""
    features = _features()
    features[0, 12:15] = torch.tensor(ue0)
    features[1, 12:15] = torch.tensor(ue1)

    result = _run(features=features)

    assert result.executed_choice.tolist()[0] == expected_ue
    assert result.diagnostics.selected_ue.tolist()[0] == expected_ue


def test_disabled_mode_is_identity_without_ctbs_or_feature_access() -> None:
    """Catches disabled evaluation requiring CTBS state or changing an action bit."""
    actor_choice = torch.tensor([2, 1, 0])
    budget = torch.tensor([True, False, True])
    result = apply_ctbs_non_starvation_guard_step(
        config=CtbsNonStarvationGuardConfig(enabled=False),
        blank_class_index=2,
        local_prg=0,
        actor_choice=actor_choice,
        candidate_ue_ids=torch.empty((0,)),
        valid_main=torch.empty((0,)),
        remaining_demand_bytes=torch.empty((0,)),
        ue_features=torch.empty((0, 0)),
        ue_feature_names=(),
        ctbs_states=None,
        prg_layer_cap=None,
        override_used_by_cell=budget,
    )

    assert torch.equal(result.executed_choice, actor_choice)
    assert result.executed_choice.data_ptr() != actor_choice.data_ptr()
    assert torch.equal(result.override_used_by_cell, budget)
    assert not result.diagnostics.guard_override.any()


@pytest.mark.parametrize(
    "states, features, feature_names",
    [
        (None, _features(), FEATURE_NAMES),
        (_states(), torch.zeros((6, 14)), FEATURE_NAMES),
        (_states(), _features(), FEATURE_NAMES[:12] + ("wrong",) + FEATURE_NAMES[13:]),
    ],
)
def test_enabled_mode_fails_closed_for_missing_ctbs_or_feature_contract(
    states: tuple[CtbsHeadroomState, ...] | None,
    features: torch.Tensor,
    feature_names: tuple[str, ...],
) -> None:
    """Catches an enabled guard silently using incomplete causal inputs."""
    with pytest.raises(ValueError):
        _run(
            states=states,
            features=features,
            feature_names=feature_names,
            use_default_states=False,
        )


def test_diagnostics_rows_and_summary_preserve_decision_trace() -> None:
    """Catches trace conversion dropping raw action, local PRG, or no-action reasons."""
    first = _run()
    second_features = _features()
    second_features[2, 12:15] = torch.tensor([math.log(3.0), 0.5, 0.5])
    second = _run(
        features=second_features,
        actor_choice=torch.tensor([2, 2, 2]),
        override_used_by_cell=torch.tensor([True, False, False]),
        local_prg=1,
    )
    combined = concatenate_guard_diagnostics((first.diagnostics, second.diagnostics))
    rows = guard_diagnostics_to_rows(tti=7, diagnostics=first.diagnostics)
    summary = summarize_guard_diagnostics(
        rows
        + guard_diagnostics_to_rows(tti=8, diagnostics=second.diagnostics)
        + [
            {
                **rows[1],
                "tti": 9,
                "reason": "no_positive_ctbs_headroom",
                "guard_override": False,
            }
        ]
    )

    assert combined.local_prg_index.tolist() == [0, 0, 0, 1, 1, 1]
    assert len(rows) == 3
    assert rows[0]["tti"] == 7
    assert rows[0]["cell_index"] == 0
    assert rows[0]["local_prg_index"] == 0
    assert rows[0]["actor_choice"] == 2
    assert rows[0]["executed_choice"] == 0
    assert rows[0]["reason"] == "overridden"
    assert rows[0]["log1p_drain_pressure"] == pytest.approx(math.log(3.0))
    assert rows[0]["ctbs_capacity_after_bytes"] > rows[0]["ctbs_capacity_before_bytes"]
    assert summary["total_override_count"] == 2
    assert summary["nonblank_override_count"] == 0
    assert summary["override_count_by_cell"] == {0: 1, 1: 1}
    assert summary["override_count_by_ue"] == {0: 1, 2: 1}
    assert summary["actor_nonblank_count"] == 2
    assert summary["cell_budget_used_count"] == 1
    assert summary["no_pressure_candidate_count"] == 1
    assert summary["no_positive_ctbs_headroom_count"] == 1
    assert summary["maximum_overrides_per_cell_per_tti"] == 1


def test_ctbs_capacity_diagnostics_and_trace_rows_are_exact_integers() -> None:
    """Catches CTBS byte capacities being rounded through a feature-float tensor."""
    enabled = _run()
    disabled = apply_ctbs_non_starvation_guard_step(
        config=CtbsNonStarvationGuardConfig(enabled=False),
        blank_class_index=2,
        local_prg=0,
        actor_choice=torch.tensor([2, 1, 0]),
        candidate_ue_ids=torch.empty((0,)),
        valid_main=torch.empty((0,)),
        remaining_demand_bytes=torch.empty((0,)),
        ue_features=torch.empty((0, 0)),
        ue_feature_names=(),
        ctbs_states=None,
        prg_layer_cap=None,
        override_used_by_cell=torch.tensor([True, False, True]),
    )
    rows = guard_diagnostics_to_rows(tti=7, diagnostics=enabled.diagnostics)

    assert enabled.diagnostics.ctbs_capacity_before_bytes.dtype == torch.int64
    assert enabled.diagnostics.ctbs_capacity_after_bytes.dtype == torch.int64
    assert disabled.diagnostics.ctbs_capacity_before_bytes.dtype == torch.int64
    assert disabled.diagnostics.ctbs_capacity_after_bytes.dtype == torch.int64
    assert isinstance(rows[0]["ctbs_capacity_before_bytes"], int)
    assert isinstance(rows[0]["ctbs_capacity_after_bytes"], int)

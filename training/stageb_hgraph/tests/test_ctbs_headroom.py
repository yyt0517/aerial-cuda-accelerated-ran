from __future__ import annotations

import numpy as np
import torch

from training.stageb_hgraph.ctbs_headroom import (
    CtbsHeadroomInputs,
    CtbsHeadroomState,
    compare_ctbs_reference_labels,
)
from training.stageb_hgraph.graph_ops import pack_prg_candidates
from training.stageb_hgraph.independent_policy import (
    replay_independent_trace,
    select_independent_action,
)
from training.stageb_hgraph.phy_tbs_alignment_audit import PhyTbsConfig
from training.stageb_hgraph.tests.graph_fixtures import make_candidate_graph
from training.stageb_hgraph.tests.test_independent_policy import RecordingActor


def _config() -> PhyTbsConfig:
    return PhyTbsConfig(
        mcs_lut_type=0,
        mcs_sinr_cap_db=25.99,
        layer_sinval_threshold=0.1,
        prbs_per_prg=16,
        pdsch_data_symbols=11,
        effective_layer_cap=4,
    )


def test_exact_cumulative_capacity_can_fall_and_remaining_demand_can_rise() -> None:
    """Catches clipping a negative cumulative-TBS change or using a running maximum."""
    state = CtbsHeadroomState(
        initial_demand_bytes=1250.0,
        reconstructed_mcs=0,
        config=_config(),
    )

    for local_prg in range(4):
        state = state.with_selected_prg(local_prg, layer_cap=4)
    assert state.cumulative_tbs_bytes == 993
    assert state.remaining_demand_bytes == 257.0

    state = state.with_selected_prg(4, layer_cap=3)
    assert state.cumulative_tbs_bytes == 929
    assert state.remaining_demand_bytes == 321.0

    state = state.with_selected_prg(5, layer_cap=3)
    assert state.cumulative_tbs_bytes == 1122
    assert state.remaining_demand_bytes == 128.0


def _inputs(n_prg: int) -> CtbsHeadroomInputs:
    layer_caps = np.full((6, n_prg), 4, dtype=np.uint8)
    layer_caps[:, 4:] = 3
    return CtbsHeadroomInputs.from_arrays(
        tti=0,
        snapshot_id=41,
        config_sha256="12" * 32,
        config=_config(),
        ue_ids=np.arange(6, dtype=np.int32),
        serving_cell=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32),
        wb_sinr_first_layer_db=np.full((6,), -4.0, dtype=np.float32),
        olla_delta=np.zeros((6,), dtype=np.float32),
        olla_delta_up=np.ones((6,), dtype=np.float32),
        olla_delta_down=np.full((6,), 0.1, dtype=np.float32),
        prior_tb_err=np.full((6,), -1, dtype=np.int32),
        prg_layer_cap=layer_caps,
    )


def test_ctbs_rollout_replay_share_frozen_tuple_and_mask_on_exact_capacity() -> None:
    """Catches replay re-reading PHY inputs or reverting to additive raw service."""
    graph = make_candidate_graph(
        n_prg=7,
        backlog_bytes=(1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0),
    )
    actor = RecordingActor(n_prg=7)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    inputs = _inputs(n_prg=7)

    trace, action = select_independent_action(
        actor,
        graph,
        packed,
        mode="greedy",
        collect_diagnostics=True,
        ctbs_headroom_inputs=inputs,
    )
    frozen_inputs = trace.ctbs_headroom_inputs
    replay_logp, replay_entropy, replay_action = replay_independent_trace(
        actor,
        graph,
        packed,
        trace,
    )

    assert trace.ctbs_headroom_inputs == inputs
    assert trace.ctbs_headroom_inputs is frozen_inputs
    assert trace.valid_at_step[:6, :, 0].all()
    assert not trace.valid_at_step[6, :, 0].any()
    assert trace.ctbs_diagnostics is not None
    ue0 = trace.ctbs_diagnostics.selected_ue == 0
    assert trace.ctbs_diagnostics.raw_service_bytes[ue0].numel() == 6
    assert trace.ctbs_diagnostics.cumulative_tbs_after_bytes[ue0].tolist() == [
        253.0,
        497.0,
        737.0,
        993.0,
        929.0,
        1122.0,
    ]
    assert trace.ctbs_diagnostics.remaining_after_bytes[ue0].tolist() == [
        747.0,
        503.0,
        263.0,
        7.0,
        71.0,
        0.0,
    ]
    assert trace.ctbs_diagnostics.layer_after[ue0].tolist() == [4, 4, 4, 4, 3, 3]
    assert torch.equal(replay_logp, trace.logp)
    assert torch.equal(replay_entropy, trace.entropy)
    assert torch.equal(replay_action.action_ue_select, action.action_ue_select)
    assert torch.equal(replay_action.action_prg_alloc, action.action_prg_alloc)


def test_ctbs_reconstruction_is_independent_of_post_action_reference_labels() -> None:
    inputs = _inputs(n_prg=6)
    chosen = torch.tensor(
        [0] * 6 + [2] * 6 + [4] * 6,
        dtype=torch.long,
    )
    matching = compare_ctbs_reference_labels(
        inputs,
        chosen_ue_per_prg=chosen,
        actual_mcs=np.asarray([0, -1, 0, -1, 0, -1], dtype=np.int32),
        actual_layer=np.asarray([3, 0, 3, 0, 3, 0], dtype=np.int32),
        actual_tbs_capacity_bytes=np.asarray([1122, 0, 1122, 0, 1122, 0], dtype=np.int32),
        n_cell=3,
        n_prg=6,
    )
    altered_labels = compare_ctbs_reference_labels(
        inputs,
        chosen_ue_per_prg=chosen,
        actual_mcs=np.asarray([1, -1, 1, -1, 1, -1], dtype=np.int32),
        actual_layer=np.asarray([2, 0, 2, 0, 2, 0], dtype=np.int32),
        actual_tbs_capacity_bytes=np.asarray([1, 0, 1, 0, 1, 0], dtype=np.int32),
        n_cell=3,
        n_prg=6,
    )

    assert matching.all_exact is True
    assert matching.reference_label_deletion_equal is True
    assert matching.reconstructed_records == altered_labels.reconstructed_records
    assert altered_labels.all_exact is False

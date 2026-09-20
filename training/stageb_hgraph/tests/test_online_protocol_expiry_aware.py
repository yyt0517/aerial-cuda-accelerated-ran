from __future__ import annotations

import hashlib
import struct
from dataclasses import replace

import numpy as np

from training.stageb_hgraph.bridge_env import (
    BridgeEnvClient,
    BridgeEnvConfig,
    PreActionPhySnapshot,
)
from training.stageb_hgraph.online_protocol import (
    MSG_PRE_ACTION_REQ,
    MSG_PRE_ACTION_RSP,
    PRE_ACTION_RSP_HEAD_STRUCT,
    RESET_FLAG_ENABLE_CTBS,
    PHY_TBS_CAPTURE_PHASE_PRE_ACTION_PRE_MCS,
    PHY_TBS_OLLA_SOURCE_CPU,
    REWARD_TERM_COUNT,
    STATE_FLAG_HAS_CTBS_CONFIG,
    STATE_FLAG_HAS_ACTUAL_UE_PACKET_DIAGNOSTICS,
    STATE_FLAG_HAS_PHY_TBS_AUDIT_DIAGNOSTICS,
    STATE_FLAG_HAS_UE_PRG_SET_DIAGNOSTICS,
    STATE_HEAD_STRUCT,
    STEP_REQ_HEAD_STRUCT,
    VERSION,
    EnvDims,
    pack_step_req,
    unpack_state_header,
)
from training.stageb_hgraph.reward_utils import extract_reward_metrics


def _f32(values) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


def _i32(values) -> bytes:
    return np.asarray(values, dtype="<i4").tobytes()


def _u8(values) -> bytes:
    return np.asarray(values, dtype="u1").tobytes()


def _packet_diagnostic_payload() -> tuple[bytes, tuple[float, ...]]:
    reward_terms = tuple(float(i) + 0.25 for i in range(23))
    dims = (
        1,  # n_cell
        2,  # n_active_ue
        2,  # n_sched_ue
        1,  # n_tot_cell
        1,  # n_prg
        0,  # n_edges
        0,  # alloc_type
        1,  # action_alloc_len
        1,  # cell_feat_dim
        1,  # ue_feat_dim
        1,  # edge_feat_dim
        1,  # prg_feat_dim
        0,  # post_eq_layer_dim
        0,  # harq_enabled
    )
    head = STATE_HEAD_STRUCT.pack(
        7,
        0,
        3.5,
        *reward_terms,
        *dims,
        STATE_FLAG_HAS_ACTUAL_UE_PACKET_DIAGNOSTICS,
    )
    body = b"".join(
        (
            _f32([1.0]),
            _f32([2.0, 3.0]),
            _f32([4.0]),
            _u8([1, 1]),
            _u8([1, 1]),
            _u8([1]),
            _i32([5, 6]),  # delivered packets
            _i32([7, 8]),  # pending packets
            _f32([120000.0, 144000.0]),  # delivered bits
            _f32([2.5, 3.0]),  # delivered packet time
            _f32([48.0, 48.0]),  # delivered-only aggregate rate
            _f32([48.0, 48.0]),  # delivered-only per-packet mean
            _i32([1, 2]),  # expired packets
            _f32([200.0, 400.0]),  # expired packet time
            _f32([202.5, 403.0]),  # outcome packet time
            _f32([0.5925926, 0.3573201]),  # EA aggregate rate
            _f32([40.0, 36.0]),  # EA per-packet mean
        )
    )
    return head + body, reward_terms


def test_protocol_v16_preserves_reward_indices_and_adds_harq_state() -> None:
    payload, reward_terms = _packet_diagnostic_payload()

    assert VERSION == 16
    assert REWARD_TERM_COUNT == 23
    header, _ = unpack_state_header(payload)
    assert header.reward_terms[:17] == reward_terms[:17]
    assert header.reward_terms[17:] == reward_terms[17:]
    assert header.dims.harq_enabled == 0


def test_reset_state_carries_ctbs_configuration_once() -> None:
    reward_terms = tuple(0.0 for _ in range(REWARD_TERM_COUNT))
    dims = (1, 2, 2, 1, 3, 0, 0, 3, 1, 1, 1, 1, 0, 0)
    head = STATE_HEAD_STRUCT.pack(
        0,
        0,
        0.0,
        *reward_terms,
        *dims,
        STATE_FLAG_HAS_CTBS_CONFIG,
    )
    config = [0.0, 25.99, 0.1, 16.0, 11.0, 4.0]
    body = b"".join(
        (
            _f32([1.0]),
            _f32([2.0, 3.0]),
            _f32([4.0, 5.0, 6.0]),
            _u8([1, 1]),
            _u8([1, 1]),
            _u8([1, 1, 1]),
            _f32(config),
        )
    )

    client = BridgeEnvClient(BridgeEnvConfig(socket_path="unused"))
    _, _, _, info = client._parse_state_payload(head + body)

    np.testing.assert_allclose(info["ctbs_config"], config)
    assert info["ctbs_config_sha256"] == hashlib.sha256(_f32(config)).hexdigest()
    assert info["ctbs_config_available"] is True


def test_pre_action_payload_contains_only_causal_fields() -> None:
    n_ue = 2
    n_prg = 3
    config_hash = bytes(reversed(range(32)))
    payload = b"".join(
        (
            PRE_ACTION_RSP_HEAD_STRUCT.pack(
                7,
                99,
                n_ue,
                n_prg,
                PHY_TBS_OLLA_SOURCE_CPU,
                PHY_TBS_CAPTURE_PHASE_PRE_ACTION_PRE_MCS,
                config_hash,
            ),
            _i32([0, 1]),
            _i32([0, 2]),
            _f32([7.5, 11.25]),
            _f32([0.2, -0.1]),
            _f32([1.0, 1.0]),
            _f32([0.1, 0.2]),
            _i32([0, 1]),
            _u8([[2, 1, 1], [4, 3, 2]]),
        )
    )
    client = BridgeEnvClient(BridgeEnvConfig(socket_path="unused"))

    snapshot = client._parse_pre_action_payload(payload)

    assert MSG_PRE_ACTION_REQ == 8
    assert MSG_PRE_ACTION_RSP == 9
    assert snapshot.tti == 7
    assert snapshot.snapshot_id == 99
    assert snapshot.config_sha256 == config_hash.hex()
    np.testing.assert_array_equal(snapshot.ue_ids, [0, 1])
    np.testing.assert_array_equal(snapshot.serving_cell, [0, 2])
    np.testing.assert_allclose(snapshot.wb_sinr_first_layer_db, [7.5, 11.25])
    np.testing.assert_allclose(snapshot.olla_delta, [0.2, -0.1])
    np.testing.assert_allclose(snapshot.olla_delta_up, [1.0, 1.0])
    np.testing.assert_allclose(snapshot.olla_delta_down, [0.1, 0.2])
    np.testing.assert_array_equal(snapshot.prior_tb_err, [0, 1])
    np.testing.assert_array_equal(snapshot.prg_layer_cap, [[2, 1, 1], [4, 3, 2]])
    for forbidden in (
        "actual_mcs",
        "actual_layer",
        "actual_tbs_capacity_bytes",
        "scheduled_bytes",
        "successful_bytes",
    ):
        assert not hasattr(snapshot, forbidden)


def _pre_action_snapshot(*, tti: int = 7, snapshot_id: int = 99) -> PreActionPhySnapshot:
    return PreActionPhySnapshot(
        tti=tti,
        snapshot_id=snapshot_id,
        config_sha256="ab" * 32,
        olla_source_code=PHY_TBS_OLLA_SOURCE_CPU,
        capture_phase_code=PHY_TBS_CAPTURE_PHASE_PRE_ACTION_PRE_MCS,
        ue_ids=np.asarray([0, 1], dtype=np.int32),
        serving_cell=np.asarray([0, 2], dtype=np.int32),
        wb_sinr_first_layer_db=np.asarray([7.5, 11.25], dtype=np.float32),
        olla_delta=np.asarray([0.2, -0.1], dtype=np.float32),
        olla_delta_up=np.asarray([1.0, 1.0], dtype=np.float32),
        olla_delta_down=np.asarray([0.1, 0.2], dtype=np.float32),
        prior_tb_err=np.asarray([0, 1], dtype=np.int32),
        prg_layer_cap=np.asarray([[2, 1, 1], [4, 3, 2]], dtype=np.uint8),
    )


def test_pre_action_snapshot_binding_fails_closed_on_mismatch_stale_or_duplicate() -> None:
    client = BridgeEnvClient(
        BridgeEnvConfig(socket_path="unused", enable_ctbs=True)
    )
    client._dims = EnvDims(
        n_cell=3,
        n_active_ue=2,
        n_sched_ue=2,
        n_tot_cell=3,
        n_prg=3,
        n_edges=0,
        alloc_type=0,
        action_alloc_len=9,
        cell_feat_dim=1,
        ue_feat_dim=1,
        edge_feat_dim=1,
        prg_feat_dim=1,
        post_eq_layer_dim=0,
        harq_enabled=0,
    )
    client._current_tti = 7
    client._ctbs_config_sha256 = "ab" * 32
    valid = _pre_action_snapshot()

    for bad, match in (
        (replace(valid, tti=8), "TTI"),
        (replace(valid, config_sha256="cd" * 32), "config hash"),
        (replace(valid, snapshot_id=0), "snapshot ID"),
    ):
        with np.testing.assert_raises_regex(RuntimeError, match):
            client._accept_pre_action_snapshot(bad)

    client._accept_pre_action_snapshot(valid)
    with np.testing.assert_raises_regex(RuntimeError, "pending snapshot"):
        client._accept_pre_action_snapshot(replace(valid, snapshot_id=100))
    assert client._consume_pre_action_snapshot(tti=7) == 99
    with np.testing.assert_raises_regex(RuntimeError, "missing pre-action snapshot"):
        client._consume_pre_action_snapshot(tti=7)
    with np.testing.assert_raises_regex(RuntimeError, "stale or duplicate"):
        client._accept_pre_action_snapshot(valid)


def test_step_request_binds_action_to_snapshot_id() -> None:
    payload = pack_step_req(
        step_idx=7,
        snapshot_id=99,
        action_ue=_i32([1, -1]),
        action_prg=np.asarray([0, -1, 1], dtype="<i2").tobytes(),
        ue_len=2,
        prg_len=3,
    )

    assert RESET_FLAG_ENABLE_CTBS == 1
    assert STEP_REQ_HEAD_STRUCT.unpack_from(payload, 0) == (7, 99, 2, 3)


def test_protocol_exposes_exact_accepted_and_actual_ue_local_prg_sets() -> None:
    reward_terms = tuple(0.0 for _ in range(REWARD_TERM_COUNT))
    dims = (
        1,  # n_cell
        2,  # n_active_ue
        2,  # n_sched_ue
        1,  # n_tot_cell
        3,  # n_prg
        0,  # n_edges
        0,  # alloc_type
        3,  # action_alloc_len
        1,  # cell_feat_dim
        1,  # ue_feat_dim
        1,  # edge_feat_dim
        1,  # prg_feat_dim
        0,  # post_eq_layer_dim
        0,  # harq_enabled
    )
    head = STATE_HEAD_STRUCT.pack(
        8,
        0,
        0.0,
        *reward_terms,
        *dims,
        STATE_FLAG_HAS_UE_PRG_SET_DIAGNOSTICS,
    )
    body = b"".join(
        (
            _f32([1.0]),
            _f32([2.0, 3.0]),
            _f32([4.0, 5.0, 6.0]),
            _u8([1, 1]),
            _u8([1, 1]),
            _u8([1, 1, 1]),
            _u8([[1, 0, 1], [0, 1, 0]]),
            _u8([[1, 0, 1], [0, 0, 0]]),
        )
    )

    client = BridgeEnvClient(BridgeEnvConfig(socket_path="unused"))
    _, _, _, info = client._parse_state_payload(head + body)

    np.testing.assert_array_equal(
        info["accepted_pre_pdsch_ue_prg_mask"],
        [[1, 0, 1], [0, 1, 0]],
    )
    np.testing.assert_array_equal(
        info["actual_ue_prg_mask"],
        [[1, 0, 1], [0, 0, 0]],
    )
    assert info["ue_prg_set_diagnostics_available"] is True
    assert info["harq_enabled"] == 0


def test_protocol_v15_separates_cpu_causal_phy_inputs_from_reference_labels() -> None:
    reward_terms = tuple(0.0 for _ in range(REWARD_TERM_COUNT))
    n_ue = 2
    n_prg = 3
    dims = (
        1,  # n_cell
        n_ue,
        2,  # n_sched_ue
        1,  # n_tot_cell
        n_prg,
        0,  # n_edges
        0,  # alloc_type
        n_prg,  # action_alloc_len
        1,  # cell_feat_dim
        1,  # ue_feat_dim
        1,  # edge_feat_dim
        1,  # prg_feat_dim
        0,  # post_eq_layer_dim
        0,  # harq_enabled
    )
    head = STATE_HEAD_STRUCT.pack(
        9,
        0,
        0.0,
        *reward_terms,
        *dims,
        STATE_FLAG_HAS_PHY_TBS_AUDIT_DIAGNOSTICS,
    )
    body = b"".join(
        (
            _f32([1.0]),  # cell features
            _f32([2.0, 3.0]),  # UE features
            _f32([4.0, 5.0, 6.0]),  # PRG features
            _u8([1, 1]),
            _u8([1, 1]),
            _u8([1, 1, 1]),
            _f32([7.5, 11.25]),  # decision-pre first-layer WB SINR dB
            _f32([0.2, -0.1]),  # decision-pre OLLA delta
            _f32([1.0, 1.0]),  # OLLA delta-up
            _f32([0.1, 0.2]),  # OLLA delta-down
            _i32([0, 1]),  # prior TB outcome applied by current MCS selection
            _u8([[2, 1, 1], [4, 3, 2]]),  # causal per-PRG layer capability
            _f32([0.0, 25.99, 0.1, 16.0, 11.0, 4.0]),  # fixed PHY/TBS config
            _i32(
                [
                    PHY_TBS_OLLA_SOURCE_CPU,
                    PHY_TBS_CAPTURE_PHASE_PRE_ACTION_PRE_MCS,
                ]
            ),
            _i32([10, 20]),  # reference actual MCS
            _i32([1, 2]),  # reference actual layer
            _i32([3824, 0]),  # reference pre-dequeue TBS capacity
        )
    )

    client = BridgeEnvClient(BridgeEnvConfig(socket_path="unused"))
    _, _, _, info = client._parse_state_payload(head + body)

    np.testing.assert_allclose(info["decision_pre_wb_sinr_first_layer_db"], [7.5, 11.25])
    np.testing.assert_allclose(info["decision_pre_olla_delta"], [0.2, -0.1])
    np.testing.assert_allclose(info["decision_pre_olla_delta_up"], [1.0, 1.0])
    np.testing.assert_allclose(info["decision_pre_olla_delta_down"], [0.1, 0.2])
    np.testing.assert_array_equal(info["decision_pre_tb_err"], [0, 1])
    np.testing.assert_array_equal(
        info["decision_pre_prg_layer_cap"],
        [[2, 1, 1], [4, 3, 2]],
    )
    np.testing.assert_allclose(info["phy_tbs_config"], [0.0, 25.99, 0.1, 16.0, 11.0, 4.0])
    assert info["phy_tbs_olla_source_code"] == PHY_TBS_OLLA_SOURCE_CPU
    assert (
        info["phy_tbs_capture_phase_code"]
        == PHY_TBS_CAPTURE_PHASE_PRE_ACTION_PRE_MCS
    )
    np.testing.assert_array_equal(info["actual_ue_mcs"], [10, 20])
    np.testing.assert_array_equal(info["actual_ue_layer"], [1, 2])
    np.testing.assert_array_equal(info["actual_ue_tbs_capacity_bytes"], [3824, 0])
    assert info["phy_tbs_audit_diagnostics_available"] is True


def test_expanded_packet_payload_is_consumed_exactly() -> None:
    payload, _ = _packet_diagnostic_payload()
    client = BridgeEnvClient(BridgeEnvConfig(socket_path="unused"))

    _, _, _, info = client._parse_state_payload(payload)

    np.testing.assert_array_equal(info["actual_ue_packet_expired_packets"], [1, 2])
    np.testing.assert_allclose(info["actual_ue_packet_expired_system_time_ms"], [200.0, 400.0])
    np.testing.assert_allclose(info["actual_ue_packet_outcome_system_time_ms"], [202.5, 403.0])
    np.testing.assert_allclose(
        info["actual_ue_packet_expiry_aware_rate_mbps"],
        [0.5925926, 0.3573201],
    )
    np.testing.assert_allclose(
        info["actual_ue_packet_expiry_aware_rate_per_packet_mean_mbps"],
        [40.0, 36.0],
    )
    assert info["actual_ue_packet_expiry_aware_diagnostics_available"] is True


def test_reward_extraction_exposes_all_six_appended_terms() -> None:
    terms = tuple(float(i) for i in range(23))

    metrics = extract_reward_metrics({"reward_terms": terms})

    assert metrics["packet_expired_count"] == 17.0
    assert metrics["packet_expired_system_time_ms"] == 18.0
    assert metrics["packet_outcome_count"] == 19.0
    assert metrics["packet_outcome_system_time_ms"] == 20.0
    assert metrics["packet_expiry_aware_rate_mbps"] == 21.0
    assert metrics["packet_expiry_aware_rate_per_packet_mean_mbps"] == 22.0

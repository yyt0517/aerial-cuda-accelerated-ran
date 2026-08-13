from __future__ import annotations

import struct

import numpy as np

from training.stageb_hgraph.bridge_env import BridgeEnvClient, BridgeEnvConfig
from training.stageb_hgraph.online_protocol import (
    REWARD_TERM_COUNT,
    STATE_FLAG_HAS_ACTUAL_UE_PACKET_DIAGNOSTICS,
    STATE_HEAD_STRUCT,
    VERSION,
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


def test_protocol_v12_preserves_old_indices_and_appends_six_terms() -> None:
    payload, reward_terms = _packet_diagnostic_payload()

    assert VERSION == 12
    assert REWARD_TERM_COUNT == 23
    header, _ = unpack_state_header(payload)
    assert header.reward_terms[:17] == reward_terms[:17]
    assert header.reward_terms[17:] == reward_terms[17:]


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

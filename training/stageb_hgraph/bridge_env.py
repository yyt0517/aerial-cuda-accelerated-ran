#!/usr/bin/env python3
"""Minimal online bridge client for the Stage-B sparse-graph stack."""

from __future__ import annotations

import hashlib
import os
import socket
import time
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import numpy as np

from .online_protocol import (
    HEADER_STRUCT,
    MSG_CLOSE_REQ,
    MSG_CLOSE_RSP,
    MSG_ERROR_RSP,
    MSG_PRE_ACTION_REQ,
    MSG_PRE_ACTION_RSP,
    MSG_RESET_REQ,
    MSG_RESET_RSP,
    MSG_STEP_REQ,
    MSG_STEP_RSP,
    PHY_TBS_CAPTURE_PHASE_PRE_ACTION_PRE_MCS,
    PHY_TBS_OLLA_SOURCE_CPU,
    PRE_ACTION_RSP_HEAD_STRUCT,
    RESET_FLAG_ENABLE_CTBS,
    STATE_FLAG_HAS_ACCEPTED_PRE_PDSCH_UE_DIAGNOSTICS,
    STATE_FLAG_HAS_ACTUAL_UE_DIAGNOSTICS,
    STATE_FLAG_HAS_ACTUAL_UE_PACKET_DIAGNOSTICS,
    STATE_FLAG_HAS_NATIVE_REJECT_DIAGNOSTICS,
    STATE_FLAG_HAS_NATIVE_SLOT_LAYOUT_DIAGNOSTICS,
    STATE_FLAG_HAS_POST_EQ_SINR,
    STATE_FLAG_HAS_PHY_TBS_AUDIT_DIAGNOSTICS,
    STATE_FLAG_HAS_CTBS_CONFIG,
    STATE_FLAG_HAS_TEACHER_ACTION,
    STATE_FLAG_HAS_UE_PRG_SET_DIAGNOSTICS,
    pack_header,
    pack_pre_action_req,
    pack_reset_req,
    pack_step_req,
    unpack_error_payload,
    unpack_header,
    unpack_state_header,
)

_I32 = np.dtype("<i4")
_I16 = np.dtype("<i2")
_F32 = np.dtype("<f4")
_U8 = np.dtype("u1")


@dataclass(frozen=True)
class BridgeEnvConfig:
    socket_path: str
    connect_timeout_s: float = 20.0
    enable_ctbs: bool = False


@dataclass(frozen=True)
class PreActionPhySnapshot:
    tti: int
    snapshot_id: int
    config_sha256: str
    olla_source_code: int
    capture_phase_code: int
    ue_ids: np.ndarray
    serving_cell: np.ndarray
    wb_sinr_first_layer_db: np.ndarray
    olla_delta: np.ndarray
    olla_delta_up: np.ndarray
    olla_delta_down: np.ndarray
    prior_tb_err: np.ndarray
    prg_layer_cap: np.ndarray


class BridgeEnvClient:
    def __init__(self, cfg: BridgeEnvConfig):
        self.cfg = cfg
        self._sock: Optional[socket.socket] = None
        self._dims = None
        self._step_idx = 0
        self._closed = False
        self._current_tti: int | None = None
        self._ctbs_config_sha256: str | None = None
        self._pending_pre_action_snapshot: PreActionPhySnapshot | None = None
        self._last_consumed_snapshot_id = 0

    @property
    def dims(self):
        return self._dims

    def connect(self) -> None:
        if self._sock is not None:
            return
        deadline = time.time() + max(0.1, self.cfg.connect_timeout_s)
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            if not os.path.exists(self.cfg.socket_path):
                time.sleep(0.05)
                continue
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(self.cfg.socket_path)
                self._sock = sock
                return
            except OSError as exc:
                last_err = exc
                sock.close()
                time.sleep(0.05)
        raise RuntimeError(f"failed to connect socket={self.cfg.socket_path}, err={last_err}")

    def _recv_exact(self, n: int) -> bytes:
        assert self._sock is not None
        data = bytearray()
        while len(data) < n:
            chunk = self._sock.recv(n - len(data))
            if not chunk:
                raise RuntimeError("socket closed while receiving")
            data.extend(chunk)
        return bytes(data)

    def _send(self, msg_type: int, payload: bytes) -> None:
        assert self._sock is not None
        self._sock.sendall(pack_header(msg_type, len(payload)) + payload)

    def _recv(self, expected_type: int) -> bytes:
        blob = self._recv_exact(HEADER_STRUCT.size)
        msg_type, payload_len = unpack_header(blob)
        payload = self._recv_exact(payload_len)
        if msg_type == MSG_ERROR_RSP:
            code, msg = unpack_error_payload(payload)
            raise RuntimeError(f"bridge error code={code}, msg={msg}")
        if msg_type != expected_type:
            raise RuntimeError(f"unexpected message type={msg_type}, expected={expected_type}")
        return payload

    def _parse_state_payload(self, payload: bytes) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        header, off = unpack_state_header(payload)
        dims = header.dims

        def take(dtype: np.dtype, count: int, shape):
            nonlocal off
            nbytes = int(dtype.itemsize) * int(count)
            if off + nbytes > len(payload):
                raise RuntimeError("state payload truncated")
            arr = np.frombuffer(payload, dtype=dtype, count=count, offset=off).copy().reshape(shape)
            off += nbytes
            return arr

        obs_cell = take(_F32, dims.n_cell * dims.cell_feat_dim, (dims.n_cell, dims.cell_feat_dim))
        obs_ue = take(_F32, dims.n_active_ue * dims.ue_feat_dim, (dims.n_active_ue, dims.ue_feat_dim))
        obs_prg = take(_F32, dims.n_cell * dims.n_prg * dims.prg_feat_dim, (dims.n_cell, dims.n_prg, dims.prg_feat_dim))
        obs_edge_index = take(_I16, dims.n_edges * 2, (dims.n_edges, 2))
        obs_edge_attr = take(_F32, dims.n_edges * dims.edge_feat_dim, (dims.n_edges, dims.edge_feat_dim))
        action_mask_ue = take(_U8, dims.n_active_ue, (dims.n_active_ue,))
        action_mask_cell_ue = take(_U8, dims.n_cell * dims.n_active_ue, (dims.n_cell, dims.n_active_ue))
        action_mask_prg_cell = take(_U8, dims.n_cell * dims.n_prg, (dims.n_cell, dims.n_prg))

        post_eq_sinr = None
        if header.state_flags & STATE_FLAG_HAS_POST_EQ_SINR:
            if dims.post_eq_layer_dim <= 0:
                raise RuntimeError("state payload flags mark post_eq_sinr present, but post_eq_layer_dim <= 0")
            count = dims.n_active_ue * dims.n_prg * dims.post_eq_layer_dim
            post_eq_sinr = take(_F32, count, (dims.n_active_ue, dims.n_prg, dims.post_eq_layer_dim))

        teacher_action_ue = None
        teacher_action_prg = None
        if header.state_flags & STATE_FLAG_HAS_TEACHER_ACTION:
            teacher_action_ue = take(_I32, dims.n_sched_ue, (dims.n_sched_ue,))
            teacher_action_prg = take(_I16, dims.action_alloc_len, (dims.action_alloc_len,))

        accepted_pre_pdsch_ue_prg_count = None
        if header.state_flags & STATE_FLAG_HAS_ACCEPTED_PRE_PDSCH_UE_DIAGNOSTICS:
            accepted_pre_pdsch_ue_prg_count = take(_I32, dims.n_active_ue, (dims.n_active_ue,))

        native_reject_wrong_cell_slot_count = None
        native_reject_empty_slot_count = None
        native_reject_prg_mask_count = None
        native_reject_oob_slot_count = None
        native_reject_duplicate_ue_count = None
        if header.state_flags & STATE_FLAG_HAS_NATIVE_REJECT_DIAGNOSTICS:
            native_reject_wrong_cell_slot_count = take(_I32, dims.n_cell, (dims.n_cell,))
            native_reject_empty_slot_count = take(_I32, dims.n_cell, (dims.n_cell,))
            native_reject_prg_mask_count = take(_I32, dims.n_cell, (dims.n_cell,))
            native_reject_oob_slot_count = take(_I32, dims.n_cell, (dims.n_cell,))
            native_reject_duplicate_ue_count = take(_I32, dims.n_cell, (dims.n_cell,))

        native_slot_count_per_cell = None
        native_cell_slot_start = None
        native_slot_to_cell = None
        native_slot_valid = None
        if header.state_flags & STATE_FLAG_HAS_NATIVE_SLOT_LAYOUT_DIAGNOSTICS:
            native_slot_count_per_cell = take(_I32, dims.n_cell, (dims.n_cell,))
            native_cell_slot_start = take(_I32, dims.n_cell, (dims.n_cell,))
            native_slot_to_cell = take(_I32, dims.n_sched_ue, (dims.n_sched_ue,))
            native_slot_valid = take(_I32, dims.n_sched_ue, (dims.n_sched_ue,))

        actual_ue_prg_count = None
        actual_ue_served_bytes = None
        actual_ue_goodput_bytes = None
        actual_ue_tb_tx_count = None
        actual_ue_tb_err_count = None
        if header.state_flags & STATE_FLAG_HAS_ACTUAL_UE_DIAGNOSTICS:
            actual_ue_prg_count = take(_I32, dims.n_active_ue, (dims.n_active_ue,))
            actual_ue_served_bytes = take(_F32, dims.n_active_ue, (dims.n_active_ue,))
            actual_ue_goodput_bytes = take(_F32, dims.n_active_ue, (dims.n_active_ue,))
            actual_ue_tb_tx_count = take(_I32, dims.n_active_ue, (dims.n_active_ue,))
            actual_ue_tb_err_count = take(_I32, dims.n_active_ue, (dims.n_active_ue,))

        actual_ue_packet_delivered_packets = None
        actual_ue_packet_pending_packets = None
        actual_ue_packet_delivered_bits = None
        actual_ue_packet_system_time_ms = None
        actual_ue_packet_service_rate_mbps = None
        actual_ue_packet_service_rate_per_packet_mean_mbps = None
        actual_ue_packet_expired_packets = None
        actual_ue_packet_expired_system_time_ms = None
        actual_ue_packet_outcome_system_time_ms = None
        actual_ue_packet_expiry_aware_rate_mbps = None
        actual_ue_packet_expiry_aware_rate_per_packet_mean_mbps = None
        if header.state_flags & STATE_FLAG_HAS_ACTUAL_UE_PACKET_DIAGNOSTICS:
            actual_ue_packet_delivered_packets = take(_I32, dims.n_active_ue, (dims.n_active_ue,))
            actual_ue_packet_pending_packets = take(_I32, dims.n_active_ue, (dims.n_active_ue,))
            actual_ue_packet_delivered_bits = take(_F32, dims.n_active_ue, (dims.n_active_ue,))
            actual_ue_packet_system_time_ms = take(_F32, dims.n_active_ue, (dims.n_active_ue,))
            actual_ue_packet_service_rate_mbps = take(_F32, dims.n_active_ue, (dims.n_active_ue,))
            actual_ue_packet_service_rate_per_packet_mean_mbps = take(
                _F32,
                dims.n_active_ue,
                (dims.n_active_ue,),
            )
            actual_ue_packet_expired_packets = take(_I32, dims.n_active_ue, (dims.n_active_ue,))
            actual_ue_packet_expired_system_time_ms = take(_F32, dims.n_active_ue, (dims.n_active_ue,))
            actual_ue_packet_outcome_system_time_ms = take(_F32, dims.n_active_ue, (dims.n_active_ue,))
            actual_ue_packet_expiry_aware_rate_mbps = take(_F32, dims.n_active_ue, (dims.n_active_ue,))
            actual_ue_packet_expiry_aware_rate_per_packet_mean_mbps = take(
                _F32,
                dims.n_active_ue,
                (dims.n_active_ue,),
            )

        accepted_pre_pdsch_ue_prg_mask = None
        actual_ue_prg_mask = None
        if header.state_flags & STATE_FLAG_HAS_UE_PRG_SET_DIAGNOSTICS:
            shape = (dims.n_active_ue, dims.n_prg)
            count = dims.n_active_ue * dims.n_prg
            accepted_pre_pdsch_ue_prg_mask = take(_U8, count, shape)
            actual_ue_prg_mask = take(_U8, count, shape)

        decision_pre_wb_sinr_first_layer_db = None
        decision_pre_olla_delta = None
        decision_pre_olla_delta_up = None
        decision_pre_olla_delta_down = None
        decision_pre_tb_err = None
        decision_pre_prg_layer_cap = None
        phy_tbs_config = None
        phy_tbs_olla_source_code = None
        phy_tbs_capture_phase_code = None
        actual_ue_mcs = None
        actual_ue_layer = None
        actual_ue_tbs_capacity_bytes = None
        if header.state_flags & STATE_FLAG_HAS_PHY_TBS_AUDIT_DIAGNOSTICS:
            ue_shape = (dims.n_active_ue,)
            decision_pre_wb_sinr_first_layer_db = take(_F32, dims.n_active_ue, ue_shape)
            decision_pre_olla_delta = take(_F32, dims.n_active_ue, ue_shape)
            decision_pre_olla_delta_up = take(_F32, dims.n_active_ue, ue_shape)
            decision_pre_olla_delta_down = take(_F32, dims.n_active_ue, ue_shape)
            decision_pre_tb_err = take(_I32, dims.n_active_ue, ue_shape)
            decision_pre_prg_layer_cap = take(
                _U8,
                dims.n_active_ue * dims.n_prg,
                (dims.n_active_ue, dims.n_prg),
            )
            phy_tbs_config = take(_F32, 6, (6,))
            phy_tbs_provenance = take(_I32, 2, (2,))
            phy_tbs_olla_source_code = int(phy_tbs_provenance[0])
            phy_tbs_capture_phase_code = int(phy_tbs_provenance[1])
            actual_ue_mcs = take(_I32, dims.n_active_ue, ue_shape)
            actual_ue_layer = take(_I32, dims.n_active_ue, ue_shape)
            actual_ue_tbs_capacity_bytes = take(_I32, dims.n_active_ue, ue_shape)

        ctbs_config = None
        ctbs_config_sha256 = None
        if header.state_flags & STATE_FLAG_HAS_CTBS_CONFIG:
            ctbs_config = take(_F32, 6, (6,))
            ctbs_config_sha256 = hashlib.sha256(
                ctbs_config.astype(_F32, copy=False).tobytes(order="C")
            ).hexdigest()

        if off != len(payload):
            raise RuntimeError(f"state payload parse mismatch: parsed={off} total={len(payload)}")

        self._dims = dims
        obs = {
            "obs_cell_features": obs_cell,
            "obs_ue_features": obs_ue,
            "obs_prg_features": obs_prg,
            "obs_edge_index": obs_edge_index.astype(np.int64, copy=False),
            "obs_edge_attr": obs_edge_attr,
            "action_mask_ue": action_mask_ue.astype(np.bool_, copy=False),
            "action_mask_cell_ue": action_mask_cell_ue.astype(np.bool_, copy=False),
            "action_mask_prg_cell": action_mask_prg_cell.astype(np.bool_, copy=False),
            "n_sched_ue": np.int32(dims.n_sched_ue),
        }
        if post_eq_sinr is not None:
            obs["post_eq_sinr"] = post_eq_sinr
        if teacher_action_ue is not None and teacher_action_prg is not None:
            obs["teacher_action_ue_select"] = teacher_action_ue.astype(np.int32, copy=False)
            obs["teacher_action_prg_alloc"] = teacher_action_prg.astype(np.int16, copy=False)
        info = {
            "tti": header.tti,
            "reward_terms": header.reward_terms,
            "n_cell": dims.n_cell,
            "n_active_ue": dims.n_active_ue,
            "n_sched_ue": dims.n_sched_ue,
            "n_tot_cell": dims.n_tot_cell,
            "n_prg": dims.n_prg,
            "alloc_type": dims.alloc_type,
            "action_alloc_len": dims.action_alloc_len,
            "cell_feat_dim": dims.cell_feat_dim,
            "ue_feat_dim": dims.ue_feat_dim,
            "edge_feat_dim": dims.edge_feat_dim,
            "prg_feat_dim": dims.prg_feat_dim,
            "post_eq_layer_dim": int(dims.post_eq_layer_dim),
            "harq_enabled": int(dims.harq_enabled),
            "teacher_available": bool(header.state_flags & STATE_FLAG_HAS_TEACHER_ACTION),
            "accepted_pre_pdsch_ue_diagnostics_available": bool(
                header.state_flags & STATE_FLAG_HAS_ACCEPTED_PRE_PDSCH_UE_DIAGNOSTICS
            ),
            "native_reject_diagnostics_available": bool(
                header.state_flags & STATE_FLAG_HAS_NATIVE_REJECT_DIAGNOSTICS
            ),
            "native_slot_layout_diagnostics_available": bool(
                header.state_flags & STATE_FLAG_HAS_NATIVE_SLOT_LAYOUT_DIAGNOSTICS
            ),
            "actual_ue_diagnostics_available": bool(
                header.state_flags & STATE_FLAG_HAS_ACTUAL_UE_DIAGNOSTICS
            ),
            "actual_ue_packet_diagnostics_available": bool(
                header.state_flags & STATE_FLAG_HAS_ACTUAL_UE_PACKET_DIAGNOSTICS
            ),
            "actual_ue_packet_expiry_aware_diagnostics_available": bool(
                header.state_flags & STATE_FLAG_HAS_ACTUAL_UE_PACKET_DIAGNOSTICS
            ),
            "ue_prg_set_diagnostics_available": bool(
                header.state_flags & STATE_FLAG_HAS_UE_PRG_SET_DIAGNOSTICS
            ),
            "phy_tbs_audit_diagnostics_available": bool(
                header.state_flags & STATE_FLAG_HAS_PHY_TBS_AUDIT_DIAGNOSTICS
            ),
            "ctbs_config_available": bool(
                header.state_flags & STATE_FLAG_HAS_CTBS_CONFIG
            ),
        }
        if accepted_pre_pdsch_ue_prg_count is not None:
            info["accepted_pre_pdsch_ue_prg_count"] = accepted_pre_pdsch_ue_prg_count.astype(
                np.int32,
                copy=False,
            )
        if native_reject_wrong_cell_slot_count is not None:
            info.update(
                {
                    "native_reject_wrong_cell_slot_count": native_reject_wrong_cell_slot_count.astype(
                        np.int32,
                        copy=False,
                    ),
                    "native_reject_empty_slot_count": native_reject_empty_slot_count.astype(
                        np.int32,
                        copy=False,
                    ),
                    "native_reject_prg_mask_count": native_reject_prg_mask_count.astype(
                        np.int32,
                        copy=False,
                    ),
                    "native_reject_oob_slot_count": native_reject_oob_slot_count.astype(
                        np.int32,
                        copy=False,
                    ),
                    "native_reject_duplicate_ue_count": native_reject_duplicate_ue_count.astype(
                        np.int32,
                        copy=False,
                    ),
                }
            )
        if native_slot_count_per_cell is not None:
            info.update(
                {
                    "native_slot_count_per_cell": native_slot_count_per_cell.astype(np.int32, copy=False),
                    "native_cell_slot_start": native_cell_slot_start.astype(np.int32, copy=False),
                    "native_slot_to_cell": native_slot_to_cell.astype(np.int32, copy=False),
                    "native_slot_valid": native_slot_valid.astype(np.int32, copy=False),
                }
            )
        if actual_ue_prg_count is not None:
            info.update(
                {
                    "actual_ue_prg_count": actual_ue_prg_count.astype(np.int32, copy=False),
                    "actual_ue_served_bytes": actual_ue_served_bytes.astype(np.float32, copy=False),
                    "actual_ue_goodput_bytes": actual_ue_goodput_bytes.astype(np.float32, copy=False),
                    "actual_ue_tb_tx_count": actual_ue_tb_tx_count.astype(np.int32, copy=False),
                    "actual_ue_tb_err_count": actual_ue_tb_err_count.astype(np.int32, copy=False),
                }
            )
        if actual_ue_packet_delivered_packets is not None:
            info.update(
                {
                    "actual_ue_packet_delivered_packets": actual_ue_packet_delivered_packets.astype(
                        np.int32,
                        copy=False,
                    ),
                    "actual_ue_packet_pending_packets": actual_ue_packet_pending_packets.astype(
                        np.int32,
                        copy=False,
                    ),
                    "actual_ue_packet_delivered_bits": actual_ue_packet_delivered_bits.astype(
                        np.float32,
                        copy=False,
                    ),
                    "actual_ue_packet_system_time_ms": actual_ue_packet_system_time_ms.astype(
                        np.float32,
                        copy=False,
                    ),
                    "actual_ue_packet_service_rate_mbps": actual_ue_packet_service_rate_mbps.astype(
                        np.float32,
                        copy=False,
                    ),
                    "actual_ue_packet_service_rate_per_packet_mean_mbps": (
                        actual_ue_packet_service_rate_per_packet_mean_mbps.astype(
                            np.float32,
                            copy=False,
                        )
                    ),
                    "actual_ue_packet_expired_packets": actual_ue_packet_expired_packets.astype(
                        np.int32,
                        copy=False,
                    ),
                    "actual_ue_packet_expired_system_time_ms": (
                        actual_ue_packet_expired_system_time_ms.astype(np.float32, copy=False)
                    ),
                    "actual_ue_packet_outcome_system_time_ms": (
                        actual_ue_packet_outcome_system_time_ms.astype(np.float32, copy=False)
                    ),
                    "actual_ue_packet_expiry_aware_rate_mbps": (
                        actual_ue_packet_expiry_aware_rate_mbps.astype(np.float32, copy=False)
                    ),
                    "actual_ue_packet_expiry_aware_rate_per_packet_mean_mbps": (
                        actual_ue_packet_expiry_aware_rate_per_packet_mean_mbps.astype(
                            np.float32,
                            copy=False,
                        )
                    ),
                }
            )
        if accepted_pre_pdsch_ue_prg_mask is not None:
            info.update(
                {
                    "accepted_pre_pdsch_ue_prg_mask": accepted_pre_pdsch_ue_prg_mask.astype(
                        np.uint8, copy=False
                    ),
                    "actual_ue_prg_mask": actual_ue_prg_mask.astype(np.uint8, copy=False),
                }
            )
        if decision_pre_wb_sinr_first_layer_db is not None:
            info.update(
                {
                    "decision_pre_wb_sinr_first_layer_db": decision_pre_wb_sinr_first_layer_db,
                    "decision_pre_olla_delta": decision_pre_olla_delta,
                    "decision_pre_olla_delta_up": decision_pre_olla_delta_up,
                    "decision_pre_olla_delta_down": decision_pre_olla_delta_down,
                    "decision_pre_tb_err": decision_pre_tb_err.astype(np.int32, copy=False),
                    "decision_pre_prg_layer_cap": decision_pre_prg_layer_cap.astype(
                        np.uint8, copy=False
                    ),
                    "phy_tbs_config": phy_tbs_config,
                    "phy_tbs_olla_source_code": phy_tbs_olla_source_code,
                    "phy_tbs_capture_phase_code": phy_tbs_capture_phase_code,
                    "actual_ue_mcs": actual_ue_mcs.astype(np.int32, copy=False),
                    "actual_ue_layer": actual_ue_layer.astype(np.int32, copy=False),
                    "actual_ue_tbs_capacity_bytes": actual_ue_tbs_capacity_bytes.astype(
                        np.int32, copy=False
                    ),
                }
            )
        if ctbs_config is not None:
            info.update(
                {
                    "ctbs_config": ctbs_config,
                    "ctbs_config_sha256": ctbs_config_sha256,
                }
            )
        return obs, float(header.reward_scalar), bool(header.done), info

    def _parse_pre_action_payload(self, payload: bytes) -> PreActionPhySnapshot:
        if len(payload) < PRE_ACTION_RSP_HEAD_STRUCT.size:
            raise RuntimeError("pre-action payload too small")
        (
            tti,
            snapshot_id,
            n_ue,
            n_prg,
            olla_source_code,
            capture_phase_code,
            config_hash,
        ) = PRE_ACTION_RSP_HEAD_STRUCT.unpack_from(payload, 0)
        off = PRE_ACTION_RSP_HEAD_STRUCT.size

        def take(dtype: np.dtype, count: int, shape):
            nonlocal off
            nbytes = int(dtype.itemsize) * int(count)
            if off + nbytes > len(payload):
                raise RuntimeError("pre-action payload truncated")
            arr = np.frombuffer(payload, dtype=dtype, count=count, offset=off).copy().reshape(shape)
            off += nbytes
            return arr

        ue_ids = take(_I32, n_ue, (n_ue,))
        serving_cell = take(_I32, n_ue, (n_ue,))
        wb_sinr = take(_F32, n_ue, (n_ue,))
        olla_delta = take(_F32, n_ue, (n_ue,))
        olla_delta_up = take(_F32, n_ue, (n_ue,))
        olla_delta_down = take(_F32, n_ue, (n_ue,))
        prior_tb_err = take(_I32, n_ue, (n_ue,))
        layer_cap = take(_U8, n_ue * n_prg, (n_ue, n_prg))
        if off != len(payload):
            raise RuntimeError(f"pre-action payload parse mismatch: parsed={off} total={len(payload)}")
        return PreActionPhySnapshot(
            tti=int(tti),
            snapshot_id=int(snapshot_id),
            config_sha256=bytes(config_hash).hex(),
            olla_source_code=int(olla_source_code),
            capture_phase_code=int(capture_phase_code),
            ue_ids=ue_ids,
            serving_cell=serving_cell,
            wb_sinr_first_layer_db=wb_sinr,
            olla_delta=olla_delta,
            olla_delta_up=olla_delta_up,
            olla_delta_down=olla_delta_down,
            prior_tb_err=prior_tb_err,
            prg_layer_cap=layer_cap,
        )

    def _accept_pre_action_snapshot(
        self,
        snapshot: PreActionPhySnapshot,
    ) -> PreActionPhySnapshot:
        if not self.cfg.enable_ctbs:
            raise RuntimeError("pre-action snapshot received while CTBS handshake is disabled")
        if self._dims is None or self._current_tti is None or self._ctbs_config_sha256 is None:
            raise RuntimeError("CTBS reset configuration is unavailable")
        if self._pending_pre_action_snapshot is not None:
            raise RuntimeError("cannot replace a pending snapshot")
        if int(snapshot.tti) != int(self._current_tti):
            raise RuntimeError(
                f"pre-action snapshot TTI mismatch: got={snapshot.tti} expected={self._current_tti}"
            )
        if int(snapshot.snapshot_id) <= 0:
            raise RuntimeError("pre-action snapshot ID must be positive")
        if int(snapshot.snapshot_id) <= int(self._last_consumed_snapshot_id):
            raise RuntimeError("pre-action snapshot is stale or duplicate")
        if snapshot.config_sha256 != self._ctbs_config_sha256:
            raise RuntimeError("pre-action CTBS config hash mismatch")
        if snapshot.olla_source_code != PHY_TBS_OLLA_SOURCE_CPU:
            raise RuntimeError("pre-action snapshot is not sourced from CPU OLLA")
        if snapshot.capture_phase_code != PHY_TBS_CAPTURE_PHASE_PRE_ACTION_PRE_MCS:
            raise RuntimeError("pre-action snapshot was not captured before action and MCS")

        n_ue = int(self._dims.n_active_ue)
        n_prg = int(self._dims.n_prg)
        vector_fields = (
            snapshot.ue_ids,
            snapshot.serving_cell,
            snapshot.wb_sinr_first_layer_db,
            snapshot.olla_delta,
            snapshot.olla_delta_up,
            snapshot.olla_delta_down,
            snapshot.prior_tb_err,
        )
        if any(np.asarray(value).shape != (n_ue,) for value in vector_fields):
            raise RuntimeError("pre-action UE vector dimensions do not match reset")
        if np.asarray(snapshot.prg_layer_cap).shape != (n_ue, n_prg):
            raise RuntimeError("pre-action layer-cap dimensions do not match reset")
        if not np.array_equal(snapshot.ue_ids, np.arange(n_ue, dtype=np.int32)):
            raise RuntimeError("pre-action UE identity vector is not canonical")
        if bool(
            ((snapshot.serving_cell < 0) | (snapshot.serving_cell >= self._dims.n_cell)).any()
        ):
            raise RuntimeError("pre-action serving-cell identity is out of range")
        for values in (
            snapshot.wb_sinr_first_layer_db,
            snapshot.olla_delta,
            snapshot.olla_delta_up,
            snapshot.olla_delta_down,
        ):
            if not bool(np.isfinite(values).all()):
                raise RuntimeError("pre-action PHY tuple contains NaN or Inf")
        if not bool(np.isin(snapshot.prior_tb_err, (-1, 0, 1)).all()):
            raise RuntimeError("pre-action prior TB error is invalid")
        if bool((snapshot.prg_layer_cap < 1).any()):
            raise RuntimeError("pre-action layer capability must be positive")

        def frozen(values: np.ndarray) -> np.ndarray:
            copied = np.asarray(values).copy()
            copied.setflags(write=False)
            return copied

        accepted = replace(
            snapshot,
            ue_ids=frozen(snapshot.ue_ids),
            serving_cell=frozen(snapshot.serving_cell),
            wb_sinr_first_layer_db=frozen(snapshot.wb_sinr_first_layer_db),
            olla_delta=frozen(snapshot.olla_delta),
            olla_delta_up=frozen(snapshot.olla_delta_up),
            olla_delta_down=frozen(snapshot.olla_delta_down),
            prior_tb_err=frozen(snapshot.prior_tb_err),
            prg_layer_cap=frozen(snapshot.prg_layer_cap),
        )
        self._pending_pre_action_snapshot = accepted
        return accepted

    def _consume_pre_action_snapshot(self, *, tti: int) -> int:
        snapshot = self._pending_pre_action_snapshot
        if snapshot is None:
            raise RuntimeError("missing pre-action snapshot for CTBS action")
        if int(snapshot.tti) != int(tti):
            raise RuntimeError(
                f"pre-action snapshot TTI mismatch at consume: got={snapshot.tti} expected={tti}"
            )
        self._pending_pre_action_snapshot = None
        self._last_consumed_snapshot_id = int(snapshot.snapshot_id)
        return int(snapshot.snapshot_id)

    def request_pre_action_snapshot(self, *, expected_tti: int) -> PreActionPhySnapshot:
        if not self.cfg.enable_ctbs:
            raise RuntimeError("CTBS pre-action handshake is disabled")
        if self._ctbs_config_sha256 is None:
            raise RuntimeError("CTBS config hash is unavailable")
        self._send(
            MSG_PRE_ACTION_REQ,
            pack_pre_action_req(expected_tti, self._ctbs_config_sha256),
        )
        snapshot = self._parse_pre_action_payload(self._recv(MSG_PRE_ACTION_RSP))
        return self._accept_pre_action_snapshot(snapshot)

    def reset(self, seed: int = 0, episode_horizon: int = 400):
        self.connect()
        self._step_idx = 0
        self._current_tti = None
        self._ctbs_config_sha256 = None
        self._pending_pre_action_snapshot = None
        self._last_consumed_snapshot_id = 0
        reset_flags = RESET_FLAG_ENABLE_CTBS if self.cfg.enable_ctbs else 0
        self._send(
            MSG_RESET_REQ,
            pack_reset_req(
                seed=seed,
                episode_horizon=episode_horizon,
                flags=reset_flags,
            ),
        )
        rsp = self._recv(MSG_RESET_RSP)
        obs, reward, done, info = self._parse_state_payload(rsp)
        if done:
            raise RuntimeError("env returned done=true on reset")
        self._current_tti = int(info["tti"])
        if self.cfg.enable_ctbs:
            if not bool(info.get("ctbs_config_available", False)):
                raise RuntimeError("CTBS reset response is missing the TBS configuration")
            self._ctbs_config_sha256 = str(info["ctbs_config_sha256"])
        return obs, reward, info

    def step(self, action_ue_select: np.ndarray, action_prg_alloc: np.ndarray):
        if self._dims is None:
            raise RuntimeError("reset must be called before step")
        ue = np.asarray(action_ue_select, dtype=np.int32).reshape(-1)
        prg = np.asarray(action_prg_alloc, dtype=np.int16).reshape(-1)
        if ue.size != self._dims.n_sched_ue:
            raise ValueError(f"action_ue_select size mismatch: got={ue.size}, expected={self._dims.n_sched_ue}")
        if prg.size != self._dims.action_alloc_len:
            raise ValueError(f"action_prg_alloc size mismatch: got={prg.size}, expected={self._dims.action_alloc_len}")
        snapshot_id = 0
        step_tti = self._step_idx
        if self.cfg.enable_ctbs:
            if self._current_tti is None:
                raise RuntimeError("CTBS current TTI is unavailable")
            step_tti = int(self._current_tti)
            snapshot_id = self._consume_pre_action_snapshot(tti=step_tti)
        payload = pack_step_req(
            step_idx=step_tti,
            snapshot_id=snapshot_id,
            action_ue=ue.astype(_I32, copy=False).tobytes(order="C"),
            action_prg=prg.astype(_I16, copy=False).tobytes(order="C"),
            ue_len=int(ue.size),
            prg_len=int(prg.size),
        )
        self._send(MSG_STEP_REQ, payload)
        rsp = self._recv(MSG_STEP_RSP)
        obs, reward, done, info = self._parse_state_payload(rsp)
        self._step_idx += 1
        self._current_tti = int(info["tti"])
        return obs, reward, done, info

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._sock is not None:
            try:
                self._send(MSG_CLOSE_REQ, b"")
                _ = self._recv(MSG_CLOSE_RSP)
            except Exception:
                pass
            try:
                self._sock.close()
            finally:
                self._sock = None


__all__ = ["BridgeEnvClient", "BridgeEnvConfig", "PreActionPhySnapshot"]

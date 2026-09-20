"""Read-only reconstruction for the causal PHY/TBS alignment audit."""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from .service_accounting_audit import ServiceAccountingDecisionDiagnostics
from .online_protocol import (
    PHY_TBS_CAPTURE_PHASE_PRE_ACTION_PRE_MCS,
    PHY_TBS_OLLA_SOURCE_CPU,
)


PHY_TBS_AUDIT_V1_PROTOCOL_ID = "seed42-phy-tbs-alignment-read-only-audit-v1"
PHY_TBS_AUDIT_V2_CPU_OLLA_PROTOCOL_ID = (
    "seed42-phy-tbs-alignment-read-only-audit-v2-cpu-olla"
)


_MCS_LUT_0 = (
    -4.57, -3.02, -0.60, 1.26, 2.87, 4.89, 5.90, 6.67, 7.72, 8.41,
    9.36, 10.56, 11.65, 12.64, 13.23, 14.25, 15.23, 16.14, 16.72, 17.79,
    19.19, 19.65, 20.95, 21.75, 22.60, 23.91, 24.38, 25.99,
)
_MCS_LUT_1 = (
    -4.30, -2.44, -0.46, 1.24, 2.94, 4.89, 5.72, 6.84, 7.67, 8.67,
    9.23, 10.48, 11.49, 12.37, 13.42, 14.43, 15.04, 16.00, 16.98, 18.03,
    19.24, 19.88, 20.74, 21.71, 22.94, 23.50, 24.98, 26.08,
)
MCS_CODE_RATES = (
    120.0, 193.0, 308.0, 449.0, 602.0, 378.0, 434.0, 490.0, 553.0, 616.0,
    658.0, 466.0, 517.0, 567.0, 616.0, 666.0, 719.0, 772.0, 822.0, 873.0,
    682.5, 711.0, 754.0, 797.0, 841.0, 885.0, 916.5, 948.0,
)
MCS_QAM_ORDERS = (
    2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8,
)
_SMALL_TBS_TABLE = (
    24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144,
    152, 160, 168, 176, 184, 192, 208, 224, 240, 256, 272, 288, 304, 320,
    336, 352, 368, 384, 408, 432, 456, 480, 504, 528, 552, 576, 608, 640,
    672, 704, 736, 768, 808, 848, 888, 928, 984, 1032, 1064, 1128, 1160,
    1192, 1224, 1256, 1288, 1320, 1352, 1416, 1480, 1544, 1608, 1672, 1736,
    1800, 1864, 1928, 2024, 2088, 2152, 2216, 2280, 2408, 2472, 2536, 2600,
    2664, 2728, 2792, 2856, 2976, 3104, 3240, 3368, 3496, 3624, 3752, 3824,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class PhyTbsConfig:
    mcs_lut_type: int
    mcs_sinr_cap_db: float
    layer_sinval_threshold: float
    prbs_per_prg: int
    pdsch_data_symbols: int
    effective_layer_cap: int

    @classmethod
    def from_vector(cls, values: Sequence[float]) -> "PhyTbsConfig":
        _require(len(values) == 6, "PHY/TBS config must contain six values")
        config = cls(
            mcs_lut_type=int(values[0]),
            mcs_sinr_cap_db=float(values[1]),
            layer_sinval_threshold=float(values[2]),
            prbs_per_prg=int(values[3]),
            pdsch_data_symbols=int(values[4]),
            effective_layer_cap=int(values[5]),
        )
        _require(config.mcs_lut_type in (0, 1), "unsupported MCS LUT type")
        _require(config.prbs_per_prg > 0, "PRBs per PRG must be positive")
        _require(config.pdsch_data_symbols > 0, "PDSCH data symbols must be positive")
        _require(config.effective_layer_cap > 0, "effective layer cap must be positive")
        return config

    def as_vector(self) -> tuple[float, ...]:
        return (
            float(self.mcs_lut_type),
            float(self.mcs_sinr_cap_db),
            float(self.layer_sinval_threshold),
            float(self.prbs_per_prg),
            float(self.pdsch_data_symbols),
            float(self.effective_layer_cap),
        )


@dataclass(frozen=True)
class ReconstructedPhyTbs:
    reconstructed_mcs: int
    prefix_local_prgs: tuple[tuple[int, ...], ...]
    prefix_layers: tuple[int, ...]
    prefix_tbs_bytes: tuple[int, ...]
    marginal_tbs_bytes: tuple[int, ...]

    @property
    def reconstructed_layer(self) -> int:
        return self.prefix_layers[-1]

    @property
    def reconstructed_tbs_bytes(self) -> int:
        return self.prefix_tbs_bytes[-1]

    @property
    def zero_marginal_count(self) -> int:
        return sum(value == 0 for value in self.marginal_tbs_bytes)

    @property
    def negative_marginal_count(self) -> int:
        return sum(value < 0 for value in self.marginal_tbs_bytes)


@dataclass(frozen=True)
class ReferenceParity:
    mcs_exact: bool
    layer_exact: bool
    tbs_exact: bool

    @property
    def all_exact(self) -> bool:
        return self.mcs_exact and self.layer_exact and self.tbs_exact


@dataclass(frozen=True)
class PhyTbsAuditUeTtiRow:
    tti: int
    ue: int
    cell: int
    planned_prg_ids: tuple[int, ...]
    accepted_prg_ids: tuple[int, ...]
    actual_prg_ids: tuple[int, ...]
    initial_backlog_bytes: float
    raw_predicted_service_bytes: float
    scheduled_bytes: float
    successful_bytes: float
    quality_db_mean: float
    config: PhyTbsConfig
    reconstructed: ReconstructedPhyTbs
    reference: ReferenceParity
    actual_mcs: int
    actual_layer: int
    actual_tbs_capacity_bytes: int
    reference_label_deletion_equal: bool
    eligible: bool
    ineligible_reason: str | None
    protocol_id: str
    olla_source_code: int | None
    capture_phase_code: int | None

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        tti: int,
        ue: int,
        cell: int,
        planned_prg_ids: Sequence[int],
        accepted_prg_ids: Sequence[int],
        initial_backlog_bytes: float,
        raw_predicted_service_bytes: float,
        scheduled_bytes: float,
        successful_bytes: float,
        quality_db_mean: float,
        protocol_id: str = PHY_TBS_AUDIT_V1_PROTOCOL_ID,
    ) -> "PhyTbsAuditUeTtiRow":
        validate_phy_tbs_audit_provenance(record, protocol_id=protocol_id)
        planned = tuple(int(value) for value in planned_prg_ids)
        accepted = tuple(int(value) for value in accepted_prg_ids)
        actual = tuple(int(value) for value in record["actual_prg_ids"])
        _require(planned == tuple(sorted(set(planned))), "planned PRGs must be unique and ordered")
        _require(accepted == tuple(sorted(set(accepted))), "accepted PRGs must be unique and ordered")
        values = (
            initial_backlog_bytes,
            raw_predicted_service_bytes,
            scheduled_bytes,
            successful_bytes,
            quality_db_mean,
        )
        _require(all(math.isfinite(float(value)) for value in values), "audit row values must be finite")
        _require(initial_backlog_bytes >= 0.0, "initial backlog must be non-negative")
        _require(raw_predicted_service_bytes >= 0.0, "raw prediction must be non-negative")
        _require(scheduled_bytes >= 0.0, "scheduled bytes must be non-negative")
        _require(successful_bytes >= 0.0, "successful bytes must be non-negative")
        _require(successful_bytes <= scheduled_bytes, "successful bytes exceed scheduled bytes")

        reconstructed = reconstruct_from_audit_record(record)
        without_labels = {
            key: value
            for key, value in record.items()
            if key
            not in {
                "actual_ue_mcs",
                "actual_ue_layer",
                "actual_ue_tbs_capacity_bytes",
            }
        }
        deletion_reconstructed = reconstruct_from_audit_record(without_labels)
        deletion_equal = reconstructed == deletion_reconstructed
        reference = compare_reference_labels(reconstructed, record)
        actual_tbs = int(record["actual_ue_tbs_capacity_bytes"])
        reason: str | None = None
        if accepted != planned:
            reason = "accepted_prg_set_mismatch"
        elif actual != planned:
            reason = "actual_prg_set_mismatch"
        elif float(initial_backlog_bytes) < float(actual_tbs):
            reason = "pre_schedule_buffer_below_tbs"
        return cls(
            tti=int(tti),
            ue=int(ue),
            cell=int(cell),
            planned_prg_ids=planned,
            accepted_prg_ids=accepted,
            actual_prg_ids=actual,
            initial_backlog_bytes=float(initial_backlog_bytes),
            raw_predicted_service_bytes=float(raw_predicted_service_bytes),
            scheduled_bytes=float(scheduled_bytes),
            successful_bytes=float(successful_bytes),
            quality_db_mean=float(quality_db_mean),
            config=PhyTbsConfig.from_vector(record["phy_tbs_config"]),
            reconstructed=reconstructed,
            reference=reference,
            actual_mcs=int(record["actual_ue_mcs"]),
            actual_layer=int(record["actual_ue_layer"]),
            actual_tbs_capacity_bytes=actual_tbs,
            reference_label_deletion_equal=deletion_equal,
            eligible=reason is None,
            ineligible_reason=reason,
            protocol_id=protocol_id,
            olla_source_code=(
                None
                if "phy_tbs_olla_source_code" not in record
                else int(record["phy_tbs_olla_source_code"])
            ),
            capture_phase_code=(
                None
                if "phy_tbs_capture_phase_code" not in record
                else int(record["phy_tbs_capture_phase_code"])
            ),
        )


def _cpp_round_positive(value: float) -> int:
    return int(math.floor(value + 0.5))


def determine_tbs_bytes(
    rb_size: int,
    data_symbols: int,
    layers: int,
    code_rate: float,
    qam_order: int,
) -> int:
    """Compute TBS in bits, then convert once to physical queue bytes."""
    _require(rb_size > 0 and data_symbols > 0 and layers > 0, "TBS dimensions must be positive")
    _require(code_rate > 0.0 and qam_order > 0, "TBS modulation parameters must be positive")
    n_info = float(
        np.float32(rb_size * 12 * data_symbols * layers)
        * np.float32(code_rate)
        * np.float32(qam_order)
    )
    if n_info <= 3824.0:
        n = max(math.floor(math.log2(n_info)) - 6, 3)
        quantum = 2**n
        n_info_prime = max(int(quantum * math.floor(n_info / quantum)), 24)
        for tbs in _SMALL_TBS_TABLE:
            if tbs >= n_info_prime:
                return tbs // 8
        raise ValueError("small TBS lookup did not find a value")

    n = math.floor(math.log2(n_info - 24.0)) - 5
    quantum = 2**n
    rounded = _cpp_round_positive((n_info - 24.0) / quantum)
    n_info_prime = float(max(int(quantum * rounded), 3840))
    if code_rate < 0.25:
        c = math.ceil((n_info + 24.0) / 3816.0)
    elif n_info_prime > 8424.0:
        c = math.ceil((n_info_prime + 24.0) / 8424.0)
    else:
        c = 1
    return int(8 * c * math.ceil((n_info_prime + 24.0) / (8.0 * c)) - 24) // 8


def reconstruct_mcs(
    *,
    wb_sinr_first_layer_db: float,
    olla_delta: float,
    olla_delta_up: float,
    olla_delta_down: float,
    prior_tb_err: int,
    config: PhyTbsConfig,
) -> int:
    values = (wb_sinr_first_layer_db, olla_delta, olla_delta_up, olla_delta_down)
    _require(all(math.isfinite(value) for value in values), "MCS inputs must be finite")
    _require(prior_tb_err in (-1, 0, 1), "prior TB error must be -1, 0, or 1")
    updated_delta = np.float32(olla_delta)
    if prior_tb_err == 1:
        updated_delta = np.float32(updated_delta + np.float32(olla_delta_up))
    elif prior_tb_err == 0:
        updated_delta = np.float32(updated_delta - np.float32(olla_delta_down))
    capped_sinr = np.float32(min(np.float32(wb_sinr_first_layer_db), np.float32(config.mcs_sinr_cap_db)))
    effective_sinr = np.float32(capped_sinr - updated_delta)
    lut = _MCS_LUT_0 if config.mcs_lut_type == 0 else _MCS_LUT_1
    for mcs in range(27, -1, -1):
        if effective_sinr >= np.float32(lut[mcs]):
            return mcs
    return 0


def reconstruct_from_audit_record(record: Mapping[str, object]) -> ReconstructedPhyTbs:
    config = PhyTbsConfig.from_vector(record["phy_tbs_config"])
    prgs = tuple(int(value) for value in record["actual_prg_ids"])
    _require(bool(prgs), "at least one actual PRG is required")
    _require(prgs == tuple(sorted(set(prgs))), "actual PRGs must follow strict actor order")
    layer_caps = tuple(int(value) for value in record["decision_pre_prg_layer_cap"])
    _require(prgs[-1] < len(layer_caps), "actual PRG exceeds layer-cap vector")
    _require(
        all(1 <= layer_caps[prg] <= config.effective_layer_cap for prg in prgs),
        "selected PRG layer capability is out of range",
    )
    mcs = reconstruct_mcs(
        wb_sinr_first_layer_db=float(record["decision_pre_wb_sinr_first_layer_db"]),
        olla_delta=float(record["decision_pre_olla_delta"]),
        olla_delta_up=float(record["decision_pre_olla_delta_up"]),
        olla_delta_down=float(record["decision_pre_olla_delta_down"]),
        prior_tb_err=int(record["decision_pre_tb_err"]),
        config=config,
    )
    prefix_prgs: list[tuple[int, ...]] = []
    prefix_layers: list[int] = []
    prefix_tbs: list[int] = []
    for count in range(1, len(prgs) + 1):
        prefix = prgs[:count]
        layers = min(layer_caps[prg] for prg in prefix)
        tbs = determine_tbs_bytes(
            count * config.prbs_per_prg,
            config.pdsch_data_symbols,
            layers,
            MCS_CODE_RATES[mcs] / 1024.0,
            MCS_QAM_ORDERS[mcs],
        )
        prefix_prgs.append(prefix)
        prefix_layers.append(layers)
        prefix_tbs.append(tbs)
    previous = 0
    marginals: list[int] = []
    for tbs in prefix_tbs:
        marginals.append(tbs - previous)
        previous = tbs
    return ReconstructedPhyTbs(
        reconstructed_mcs=mcs,
        prefix_local_prgs=tuple(prefix_prgs),
        prefix_layers=tuple(prefix_layers),
        prefix_tbs_bytes=tuple(prefix_tbs),
        marginal_tbs_bytes=tuple(marginals),
    )


def compare_reference_labels(
    reconstructed: ReconstructedPhyTbs,
    reference: Mapping[str, object],
) -> ReferenceParity:
    return ReferenceParity(
        mcs_exact=reconstructed.reconstructed_mcs == int(reference["actual_ue_mcs"]),
        layer_exact=reconstructed.reconstructed_layer == int(reference["actual_ue_layer"]),
        tbs_exact=(
            reconstructed.reconstructed_tbs_bytes
            == int(reference["actual_ue_tbs_capacity_bytes"])
        ),
    )


def validate_phy_tbs_audit_provenance(
    record: Mapping[str, object],
    *,
    protocol_id: str,
) -> None:
    """Fail closed when v2 is not sourced from the causal CPU MCS path."""
    _require(
        protocol_id in (
            PHY_TBS_AUDIT_V1_PROTOCOL_ID,
            PHY_TBS_AUDIT_V2_CPU_OLLA_PROTOCOL_ID,
        ),
        "unsupported PHY/TBS audit protocol",
    )
    if protocol_id == PHY_TBS_AUDIT_V1_PROTOCOL_ID:
        return
    _require(
        int(record.get("phy_tbs_olla_source_code", -1))
        == PHY_TBS_OLLA_SOURCE_CPU,
        "v2 requires the CPU OLLA state used by mcsSelSol",
    )
    _require(
        int(record.get("phy_tbs_capture_phase_code", -1))
        == PHY_TBS_CAPTURE_PHASE_PRE_ACTION_PRE_MCS,
        "v2 requires a pre-action, pre-MCS OLLA snapshot",
    )


def phy_tbs_config_sha256(
    config: PhyTbsConfig,
    *,
    protocol_id: str = PHY_TBS_AUDIT_V1_PROTOCOL_ID,
) -> str:
    """Hash the frozen LUTs and reconstruction semantics, not a serialized artifact."""
    payload = {
        "protocol_id": protocol_id,
        "reconstruction_function": "cumac_4t4r_mcs_lut_and_determine_tbs_pdsch_v1",
        "config_vector": config.as_vector(),
        "mcs_lut_0": _MCS_LUT_0,
        "mcs_lut_1": _MCS_LUT_1,
        "mcs_code_rates": MCS_CODE_RATES,
        "mcs_qam_orders": MCS_QAM_ORDERS,
        "small_tbs_table": _SMALL_TBS_TABLE,
        "tbs_units": "physical_bytes_tbs_bits_div8_v2",
        "prefix_definition": "first_k_actual_prgs_in_actor_local_prg_order_0_to_16",
        "olla_semantics": (
            "cpu_mcs_selection_lut_pre_action_snapshot_then_prior_tb_update_fp32"
            if protocol_id == PHY_TBS_AUDIT_V2_CPU_OLLA_PROTOCOL_ID
            else "v1_mixed_gpu_snapshot_cpu_reference"
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def join_phy_tbs_alignment_tti(
    decisions: ServiceAccountingDecisionDiagnostics,
    *,
    info: Mapping[str, object],
    n_ue: int,
    n_cell: int,
    n_prg: int,
    tti: int,
    protocol_id: str = PHY_TBS_AUDIT_V1_PROTOCOL_ID,
) -> list[PhyTbsAuditUeTtiRow]:
    """Join one actor trace to causal PHY inputs and post-decision reference labels."""
    _require(n_ue > 0 and n_cell > 0 and n_prg > 0, "invalid PHY/TBS audit dimensions")
    _require(int(info.get("harq_enabled", -1)) == 0, "PHY/TBS audit requires HARQ disabled")
    validate_phy_tbs_audit_provenance(info, protocol_id=protocol_id)
    expected = n_cell * n_prg

    def decision_vector(name: str, *, dtype: torch.dtype) -> torch.Tensor:
        value = torch.as_tensor(getattr(decisions, name), dtype=dtype).detach().reshape(-1).cpu()
        _require(value.shape == (expected,), f"{name} shape mismatch")
        if dtype.is_floating_point:
            _require(bool(torch.isfinite(value).all().item()), f"{name} must be finite")
        return value

    selected_ue = decision_vector("selected_ue", dtype=torch.long)
    cells = decision_vector("cell_index", dtype=torch.long)
    local_prgs = decision_vector("local_prg_index", dtype=torch.long)
    selected = decision_vector("selected_nonblank", dtype=torch.bool)
    raw = decision_vector("raw_service_bytes", dtype=torch.float64)
    initial = decision_vector("initial_demand_bytes", dtype=torch.float64)
    quality = decision_vector("quality_db", dtype=torch.float64)
    _require(bool(((cells >= 0) & (cells < n_cell)).all().item()), "cell index out of range")
    _require(bool(((local_prgs >= 0) & (local_prgs < n_prg)).all().item()), "PRG index out of range")
    tokens = cells * n_prg + local_prgs
    _require(torch.equal(torch.sort(tokens).values, torch.arange(expected)), "decision tokens do not cover the TTI")
    _require(bool(((selected_ue[selected] >= 0) & (selected_ue[selected] < n_ue)).all().item()), "selected UE out of range")
    _require(bool((selected_ue[~selected] == -1).all().item()), "Blank must use UE -1")

    def array(name: str, shape: tuple[int, ...], *, dtype: np.dtype) -> np.ndarray:
        _require(name in info, f"missing required {name}")
        value = np.asarray(info[name], dtype=dtype)
        _require(value.shape == shape, f"{name} shape mismatch")
        if np.issubdtype(value.dtype, np.floating):
            _require(bool(np.isfinite(value).all()), f"{name} must be finite")
        return value

    accepted_mask = array(
        "accepted_pre_pdsch_ue_prg_mask", (n_ue, n_prg), dtype=np.uint8
    )
    actual_mask = array("actual_ue_prg_mask", (n_ue, n_prg), dtype=np.uint8)
    _require(bool(np.isin(accepted_mask, (0, 1)).all()), "accepted PRG mask must be binary")
    _require(bool(np.isin(actual_mask, (0, 1)).all()), "actual PRG mask must be binary")
    scheduled = array("actual_ue_served_bytes", (n_ue,), dtype=np.float64)
    successful = array("actual_ue_goodput_bytes", (n_ue,), dtype=np.float64)
    _require(bool((scheduled >= 0.0).all()), "scheduled bytes must be non-negative")
    _require(bool((successful >= 0.0).all()), "successful bytes must be non-negative")
    _require(bool((successful <= scheduled).all()), "successful bytes exceed scheduled bytes")
    wb_sinr = array("decision_pre_wb_sinr_first_layer_db", (n_ue,), dtype=np.float64)
    olla_delta = array("decision_pre_olla_delta", (n_ue,), dtype=np.float64)
    olla_up = array("decision_pre_olla_delta_up", (n_ue,), dtype=np.float64)
    olla_down = array("decision_pre_olla_delta_down", (n_ue,), dtype=np.float64)
    prior_tb_err = array("decision_pre_tb_err", (n_ue,), dtype=np.int32)
    layer_caps = array("decision_pre_prg_layer_cap", (n_ue, n_prg), dtype=np.uint8)
    config_values = array("phy_tbs_config", (6,), dtype=np.float64)
    actual_mcs = array("actual_ue_mcs", (n_ue,), dtype=np.int32)
    actual_layer = array("actual_ue_layer", (n_ue,), dtype=np.int32)
    actual_tbs = array("actual_ue_tbs_capacity_bytes", (n_ue,), dtype=np.int32)

    planned_ues = sorted(set(int(value) for value in selected_ue[selected].tolist()))
    rows: list[PhyTbsAuditUeTtiRow] = []
    for ue in planned_ues:
        ue_selection = selected & (selected_ue == ue)
        ue_cells = torch.unique(cells[ue_selection])
        _require(ue_cells.numel() == 1, "selected UE appears in multiple cells")
        planned_ids = tuple(int(value) for value in local_prgs[ue_selection].tolist())
        _require(planned_ids == tuple(sorted(set(planned_ids))), "planned PRGs do not follow actor order")
        accepted_ids = tuple(int(value) for value in np.flatnonzero(accepted_mask[ue]))
        actual_ids = tuple(int(value) for value in np.flatnonzero(actual_mask[ue]))
        _require(bool(actual_ids), "planned UE has no actual PRGs")
        initial_values = initial[ue_selection]
        _require(
            bool(torch.equal(initial_values, initial_values[:1].expand_as(initial_values))),
            "initial demand changed within a TTI",
        )
        record: dict[str, object] = {
            "actual_prg_ids": actual_ids,
            "decision_pre_wb_sinr_first_layer_db": float(wb_sinr[ue]),
            "decision_pre_olla_delta": float(olla_delta[ue]),
            "decision_pre_olla_delta_up": float(olla_up[ue]),
            "decision_pre_olla_delta_down": float(olla_down[ue]),
            "decision_pre_tb_err": int(prior_tb_err[ue]),
            "decision_pre_prg_layer_cap": tuple(int(value) for value in layer_caps[ue]),
            "phy_tbs_config": tuple(float(value) for value in config_values),
            "actual_ue_mcs": int(actual_mcs[ue]),
            "actual_ue_layer": int(actual_layer[ue]),
            "actual_ue_tbs_capacity_bytes": int(actual_tbs[ue]),
        }
        if protocol_id == PHY_TBS_AUDIT_V2_CPU_OLLA_PROTOCOL_ID:
            record["phy_tbs_olla_source_code"] = int(
                info["phy_tbs_olla_source_code"]
            )
            record["phy_tbs_capture_phase_code"] = int(
                info["phy_tbs_capture_phase_code"]
            )
        rows.append(
            PhyTbsAuditUeTtiRow.from_record(
                record,
                tti=tti,
                ue=ue,
                cell=int(ue_cells.item()),
                planned_prg_ids=planned_ids,
                accepted_prg_ids=accepted_ids,
                initial_backlog_bytes=float(initial_values[0]),
                raw_predicted_service_bytes=float(raw[ue_selection].sum()),
                scheduled_bytes=float(scheduled[ue]),
                successful_bytes=float(successful[ue]),
                quality_db_mean=float(quality[ue_selection].mean()),
                protocol_id=protocol_id,
            )
        )
    return rows


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else numerator / denominator


def _quality_tertiles(
    rows: Sequence[PhyTbsAuditUeTtiRow],
) -> list[tuple[str, list[PhyTbsAuditUeTtiRow]]]:
    ordered = sorted(rows, key=lambda row: (row.quality_db_mean, row.tti, row.ue))
    return [
        (name, [ordered[int(position)] for position in indices.tolist()])
        for name, indices in zip(
            ("low", "middle", "high"),
            np.array_split(np.arange(len(ordered)), 3),
            strict=True,
        )
    ]


def _alignment_group(
    rows: Sequence[PhyTbsAuditUeTtiRow],
    *,
    minimum_ue_ttis: int,
    minimum_actual_prgs: int,
    reference_ratio: float,
    tolerance: float,
) -> dict[str, object]:
    predicted = float(sum(row.reconstructed.reconstructed_tbs_bytes for row in rows))
    scheduled = float(sum(row.scheduled_bytes for row in rows))
    actual_prgs = int(sum(len(row.actual_prg_ids) for row in rows))
    ratio = _ratio(scheduled, predicted)
    formal = len(rows) >= minimum_ue_ttis and actual_prgs >= minimum_actual_prgs
    return {
        "ue_tti_count": len(rows),
        "actual_prg_count": actual_prgs,
        "predicted_tbs_bytes": predicted,
        "scheduled_bytes": scheduled,
        "scheduled_to_predicted_tbs_ratio": ratio,
        "is_gate_group": formal,
        "gate_passed": (
            None
            if not formal or ratio is None
            else abs(ratio - reference_ratio) <= tolerance
        ),
    }


def summarize_phy_tbs_alignment_audit(
    rows: Sequence[PhyTbsAuditUeTtiRow],
    *,
    protocol_id: str = PHY_TBS_AUDIT_V1_PROTOCOL_ID,
) -> dict[str, object]:
    """Apply exact-label, prefix-additivity, and conditional alignment gates."""
    eligible = [row for row in rows if row.eligible]
    _require(
        all(row.protocol_id == protocol_id for row in rows),
        "PHY/TBS audit row protocol changed during the audit",
    )
    configs = {
        phy_tbs_config_sha256(row.config, protocol_id=protocol_id) for row in rows
    }
    _require(len(configs) <= 1, "PHY/TBS configuration changed during the audit")

    mcs_mismatches = sum(not row.reference.mcs_exact for row in eligible)
    layer_mismatches = sum(not row.reference.layer_exact for row in eligible)
    tbs_mismatches = sum(not row.reference.tbs_exact for row in eligible)
    deletion_equal = all(row.reference_label_deletion_equal for row in rows)
    olla_sources = sorted(
        {row.olla_source_code for row in rows if row.olla_source_code is not None}
    )
    capture_phases = sorted(
        {row.capture_phase_code for row in rows if row.capture_phase_code is not None}
    )
    parity_passed = (
        bool(eligible)
        and mcs_mismatches == 0
        and layer_mismatches == 0
        and tbs_mismatches == 0
        and deletion_equal
    )
    parity_mismatch_examples = [
        {
            "tti": row.tti,
            "ue": row.ue,
            "cell": row.cell,
            "actual_prg_ids": row.actual_prg_ids,
            "reconstructed_mcs": row.reconstructed.reconstructed_mcs,
            "actual_mcs": row.actual_mcs,
            "reconstructed_layer": row.reconstructed.reconstructed_layer,
            "actual_layer": row.actual_layer,
            "reconstructed_tbs_capacity_bytes": row.reconstructed.reconstructed_tbs_bytes,
            "actual_tbs_capacity_bytes": row.actual_tbs_capacity_bytes,
        }
        for row in eligible
        if not row.reference.all_exact
    ][:20]

    marginal_values = [
        value
        for row in eligible
        for value in row.reconstructed.marginal_tbs_bytes
    ]
    negative_marginals = sum(value < 0 for value in marginal_values)
    zero_marginals = sum(value == 0 for value in marginal_values)
    monotone_violations = sum(
        any(right < left for left, right in zip((0, *row.reconstructed.prefix_tbs_bytes[:-1]), row.reconstructed.prefix_tbs_bytes))
        for row in eligible
    )
    nan_or_inf_count = 0
    sequential_passed = (
        bool(eligible)
        and negative_marginals == 0
        and monotone_violations == 0
        and nan_or_inf_count == 0
    )
    negative_marginal_examples = [
        {
            "tti": row.tti,
            "ue": row.ue,
            "cell": row.cell,
            "actual_prg_ids": row.actual_prg_ids,
            "prefix_layers": row.reconstructed.prefix_layers,
            "prefix_tbs_bytes": row.reconstructed.prefix_tbs_bytes,
            "marginal_tbs_bytes": row.reconstructed.marginal_tbs_bytes,
        }
        for row in eligible
        if row.reconstructed.negative_marginal_count > 0
    ][:20]

    predicted = float(sum(row.reconstructed.reconstructed_tbs_bytes for row in eligible))
    scheduled = float(sum(row.scheduled_bytes for row in eligible))
    raw = float(sum(row.raw_predicted_service_bytes for row in eligible))
    scheduled_to_predicted = _ratio(scheduled, predicted)
    tbs_wape = _ratio(
        float(
            sum(
                abs(row.scheduled_bytes - row.reconstructed.reconstructed_tbs_bytes)
                for row in eligible
            )
        ),
        scheduled,
    )
    raw_wape = _ratio(
        float(
            sum(
                abs(row.scheduled_bytes - row.raw_predicted_service_bytes)
                for row in eligible
            )
        ),
        scheduled,
    )
    improvement = (
        None
        if raw_wape in (None, 0.0) or tbs_wape is None
        else (raw_wape - tbs_wape) / raw_wape
    )
    global_passed = (
        scheduled_to_predicted is not None
        and 0.9 <= scheduled_to_predicted <= 1.1
        and improvement is not None
        and improvement >= 0.25
    )

    quality_groups = _quality_tertiles(eligible)
    quality_membership = {
        (row.tti, row.ue): name
        for name, group_rows in quality_groups
        for row in group_rows
    }
    strata: dict[tuple[str, int], list[PhyTbsAuditUeTtiRow]] = {}
    for row in eligible:
        key = (quality_membership[(row.tti, row.ue)], len(row.actual_prg_ids))
        strata.setdefault(key, []).append(row)
    quality_prg_groups: list[dict[str, object]] = []
    for (quality_group, prg_count), group_rows in sorted(strata.items()):
        quality_prg_groups.append(
            {
                "quality_group": quality_group,
                "actual_prg_count_per_ue_tti": prg_count,
                **_alignment_group(
                    group_rows,
                    minimum_ue_ttis=50,
                    minimum_actual_prgs=200,
                    reference_ratio=1.0,
                    tolerance=0.15,
                ),
            }
        )
    formal_quality_prg = [group for group in quality_prg_groups if group["is_gate_group"]]
    quality_prg_passed = bool(formal_quality_prg) and all(
        group["gate_passed"] is True for group in formal_quality_prg
    )

    cell_groups: list[dict[str, object]] = []
    for (quality_group, prg_count), pooled_rows in sorted(strata.items()):
        pooled_predicted = float(
            sum(row.reconstructed.reconstructed_tbs_bytes for row in pooled_rows)
        )
        pooled_scheduled = float(sum(row.scheduled_bytes for row in pooled_rows))
        pooled_ratio = _ratio(pooled_scheduled, pooled_predicted)
        for cell in sorted({row.cell for row in pooled_rows}):
            group_rows = [row for row in pooled_rows if row.cell == cell]
            group = _alignment_group(
                group_rows,
                minimum_ue_ttis=30,
                minimum_actual_prgs=100,
                reference_ratio=float(pooled_ratio or 0.0),
                tolerance=0.10,
            )
            cell_groups.append(
                {
                    "quality_group": quality_group,
                    "actual_prg_count_per_ue_tti": prg_count,
                    "cell": cell,
                    "pooled_stratum_ratio": pooled_ratio,
                    **group,
                }
            )
    formal_cell_groups = [group for group in cell_groups if group["is_gate_group"]]
    cell_residual_passed = bool(formal_cell_groups) and all(
        group["gate_passed"] is True for group in formal_cell_groups
    )
    statistical_passed = global_passed and quality_prg_passed and cell_residual_passed

    statistical_is_formal = protocol_id == PHY_TBS_AUDIT_V1_PROTOCOL_ID
    if not parity_passed:
        classification = "decision_time_phy_path_not_exact"
    elif not sequential_passed:
        classification = "final_tbs_reconstructable_but_not_sequentially_additive"
    elif statistical_is_formal and not statistical_passed:
        classification = "conditional_alignment_gate_failure"
    else:
        classification = "phy_tbs_alignment_compatible"
    ineligible_reasons: dict[str, int] = {}
    for row in rows:
        if row.eligible:
            continue
        reason = str(row.ineligible_reason)
        ineligible_reasons[reason] = ineligible_reasons.get(reason, 0) + 1
    return {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "status": "PASS" if classification == "phy_tbs_alignment_compatible" else "FAIL",
        "classification": classification,
        "eligible_ue_tti_count": len(eligible),
        "eligible_actual_prg_count": sum(len(row.actual_prg_ids) for row in eligible),
        "ineligible_ue_tti_count": len(rows) - len(eligible),
        "ineligible_reason_counts": ineligible_reasons,
        "phy_tbs_config_sha256": next(iter(configs), None),
        "phy_tbs_config": None if not rows else rows[0].config.as_vector(),
        "reference_label_deletion_equal": deletion_equal,
        "causal_provenance": {
            "olla_source_codes": olla_sources,
            "capture_phase_codes": capture_phases,
            "cpu_olla_source_exact": (
                olla_sources == [PHY_TBS_OLLA_SOURCE_CPU]
                if protocol_id == PHY_TBS_AUDIT_V2_CPU_OLLA_PROTOCOL_ID
                else None
            ),
            "pre_action_pre_mcs_capture_exact": (
                capture_phases == [PHY_TBS_CAPTURE_PHASE_PRE_ACTION_PRE_MCS]
                if protocol_id == PHY_TBS_AUDIT_V2_CPU_OLLA_PROTOCOL_ID
                else None
            ),
        },
        "record_level_parity": {
            "mcs_mismatch_count": mcs_mismatches,
            "layer_mismatch_count": layer_mismatches,
            "tbs_mismatch_count": tbs_mismatches,
            "all_exact": parity_passed,
            "mismatch_examples": parity_mismatch_examples,
        },
        "sequential_usability": {
            "prefix_definition": "first_k_actual_prgs_in_actor_local_prg_order_0_to_16",
            "prefix_count": len(marginal_values),
            "negative_marginal_count": negative_marginals,
            "zero_marginal_count": zero_marginals,
            "zero_marginal_fraction": _ratio(float(zero_marginals), float(len(marginal_values))),
            "monotonicity_violation_count": monotone_violations,
            "nan_or_inf_count": nan_or_inf_count,
            "formal_gate_evaluated": parity_passed,
            "gate_passed": sequential_passed if parity_passed else None,
            "negative_marginal_examples": negative_marginal_examples,
        },
        "global_alignment": {
            "scheduled_bytes": scheduled,
            "predicted_tbs_bytes": predicted,
            "raw_predicted_service_bytes": raw,
            "scheduled_to_predicted_tbs_ratio": scheduled_to_predicted,
            "allowed_ratio_interval": [0.9, 1.1],
            "tbs_wape": tbs_wape,
            "raw_wape": raw_wape,
            "raw_wape_improvement_fraction": improvement,
            "minimum_wape_improvement_fraction": 0.25,
            "gate_passed": global_passed,
        },
        "statistical_alignment": {
            "formal_gate": statistical_is_formal,
            "gate_passed": statistical_passed if statistical_is_formal else None,
            "role": (
                "formal_v1_conditional_alignment_gate"
                if statistical_is_formal
                else "record_only_v2_completeness_and_conditional_stability_evidence"
            ),
        },
        "quality_binning": "decision_pre_quality_equal_sample_count_tertiles_frozen_before_ratios",
        "quality_prg_count_gate": {
            "minimum_ue_tti_count": 50,
            "minimum_actual_prg_count": 200,
            "maximum_absolute_delta_from_one": 0.15,
            "formal_group_count": len(formal_quality_prg),
            "all_formal_groups_passed": quality_prg_passed,
            "groups": quality_prg_groups,
        },
        "cell_residual_gate": {
            "cell_id_used_as_reconstruction_input": False,
            "minimum_ue_tti_count": 30,
            "minimum_actual_prg_count": 100,
            "maximum_absolute_delta_from_pooled_stratum": 0.10,
            "formal_group_count": len(formal_cell_groups),
            "all_formal_groups_passed": cell_residual_passed,
            "groups": cell_groups,
        },
    }


__all__ = [
    "MCS_CODE_RATES",
    "MCS_QAM_ORDERS",
    "PHY_TBS_AUDIT_V1_PROTOCOL_ID",
    "PHY_TBS_AUDIT_V2_CPU_OLLA_PROTOCOL_ID",
    "PhyTbsAuditUeTtiRow",
    "PhyTbsConfig",
    "ReconstructedPhyTbs",
    "ReferenceParity",
    "compare_reference_labels",
    "determine_tbs_bytes",
    "join_phy_tbs_alignment_tti",
    "phy_tbs_config_sha256",
    "reconstruct_from_audit_record",
    "reconstruct_mcs",
    "summarize_phy_tbs_alignment_audit",
    "validate_phy_tbs_audit_provenance",
]

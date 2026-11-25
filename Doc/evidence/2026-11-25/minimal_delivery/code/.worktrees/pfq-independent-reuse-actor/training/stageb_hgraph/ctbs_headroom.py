"""Exact cumulative-TBS headroom state for the B4-CTBS-v1 treatment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import torch

from .phy_tbs_alignment_audit import (
    MCS_CODE_RATES,
    MCS_QAM_ORDERS,
    PhyTbsConfig,
    determine_tbs_bytes,
    reconstruct_mcs,
)


@dataclass(frozen=True)
class CtbsHeadroomInputs:
    tti: int
    snapshot_id: int
    config_sha256: str
    config: PhyTbsConfig
    ue_ids: tuple[int, ...]
    serving_cell: tuple[int, ...]
    wb_sinr_first_layer_db: tuple[float, ...]
    olla_delta: tuple[float, ...]
    olla_delta_up: tuple[float, ...]
    olla_delta_down: tuple[float, ...]
    prior_tb_err: tuple[int, ...]
    prg_layer_cap: tuple[tuple[int, ...], ...]

    @classmethod
    def from_arrays(
        cls,
        *,
        tti: int,
        snapshot_id: int,
        config_sha256: str,
        config: PhyTbsConfig,
        ue_ids: Sequence[int],
        serving_cell: Sequence[int],
        wb_sinr_first_layer_db: Sequence[float],
        olla_delta: Sequence[float],
        olla_delta_up: Sequence[float],
        olla_delta_down: Sequence[float],
        prior_tb_err: Sequence[int],
        prg_layer_cap: Sequence[Sequence[int]],
    ) -> "CtbsHeadroomInputs":
        ids = tuple(int(value) for value in ue_ids)
        n_ue = len(ids)
        vector_lengths = (
            len(serving_cell),
            len(wb_sinr_first_layer_db),
            len(olla_delta),
            len(olla_delta_up),
            len(olla_delta_down),
            len(prior_tb_err),
            len(prg_layer_cap),
        )
        if n_ue <= 0 or any(length != n_ue for length in vector_lengths):
            raise ValueError("CTBS pre-action UE vector dimensions disagree")
        layer_rows = tuple(tuple(int(value) for value in row) for row in prg_layer_cap)
        n_prg = len(layer_rows[0])
        if n_prg <= 0 or any(len(row) != n_prg for row in layer_rows):
            raise ValueError("CTBS pre-action layer-cap rows disagree")
        if ids != tuple(range(n_ue)):
            raise ValueError("CTBS UE identities must be canonical")
        if int(snapshot_id) <= 0:
            raise ValueError("CTBS snapshot ID must be positive")
        if len(str(config_sha256)) != 64:
            raise ValueError("CTBS config hash must contain 64 hexadecimal characters")
        try:
            bytes.fromhex(str(config_sha256))
        except ValueError as exc:
            raise ValueError("CTBS config hash must be hexadecimal") from exc
        float_vectors = (
            wb_sinr_first_layer_db,
            olla_delta,
            olla_delta_up,
            olla_delta_down,
        )
        if not all(np.isfinite(np.asarray(values, dtype=np.float64)).all() for values in float_vectors):
            raise ValueError("CTBS pre-action tuple contains NaN or Inf")
        if any(value not in (-1, 0, 1) for value in prior_tb_err):
            raise ValueError("CTBS prior TB errors must be -1, 0, or 1")
        if any(
            value < 1 or value > config.effective_layer_cap
            for row in layer_rows
            for value in row
        ):
            raise ValueError("CTBS layer capability is out of range")
        return cls(
            tti=int(tti),
            snapshot_id=int(snapshot_id),
            config_sha256=str(config_sha256),
            config=config,
            ue_ids=ids,
            serving_cell=tuple(int(value) for value in serving_cell),
            wb_sinr_first_layer_db=tuple(float(value) for value in wb_sinr_first_layer_db),
            olla_delta=tuple(float(value) for value in olla_delta),
            olla_delta_up=tuple(float(value) for value in olla_delta_up),
            olla_delta_down=tuple(float(value) for value in olla_delta_down),
            prior_tb_err=tuple(int(value) for value in prior_tb_err),
            prg_layer_cap=layer_rows,
        )

    def reconstructed_mcs_for_ue(self, ue: int) -> int:
        ue = int(ue)
        if ue < 0 or ue >= len(self.ue_ids):
            raise ValueError("CTBS UE index is out of range")
        return reconstruct_mcs(
            wb_sinr_first_layer_db=self.wb_sinr_first_layer_db[ue],
            olla_delta=self.olla_delta[ue],
            olla_delta_up=self.olla_delta_up[ue],
            olla_delta_down=self.olla_delta_down[ue],
            prior_tb_err=self.prior_tb_err[ue],
            config=self.config,
        )


@dataclass(frozen=True)
class CtbsHeadroomState:
    initial_demand_bytes: float
    reconstructed_mcs: int
    config: PhyTbsConfig
    selected_local_prgs: tuple[int, ...] = ()
    selected_layer_caps: tuple[int, ...] = ()
    cumulative_tbs_bytes: int = 0

    @property
    def remaining_demand_bytes(self) -> float:
        return max(0.0, float(self.initial_demand_bytes) - float(self.cumulative_tbs_bytes))

    def with_selected_prg(self, local_prg: int, *, layer_cap: int) -> "CtbsHeadroomState":
        local_prg = int(local_prg)
        layer_cap = int(layer_cap)
        if local_prg < 0:
            raise ValueError("local PRG must be non-negative")
        if self.selected_local_prgs and local_prg <= self.selected_local_prgs[-1]:
            raise ValueError("selected PRGs must follow strict actor order")
        if not 1 <= layer_cap <= self.config.effective_layer_cap:
            raise ValueError("selected PRG layer capability is out of range")
        prgs = self.selected_local_prgs + (local_prg,)
        caps = self.selected_layer_caps + (layer_cap,)
        layers = min(caps)
        tbs_bytes = determine_tbs_bytes(
            len(prgs) * self.config.prbs_per_prg,
            self.config.pdsch_data_symbols,
            layers,
            MCS_CODE_RATES[self.reconstructed_mcs] / 1024.0,
            MCS_QAM_ORDERS[self.reconstructed_mcs],
        )
        return replace(
            self,
            selected_local_prgs=prgs,
            selected_layer_caps=caps,
            cumulative_tbs_bytes=tbs_bytes,
        )


@dataclass(frozen=True)
class CtbsReconstructedRecord:
    ue: int
    local_prgs: tuple[int, ...]
    reconstructed_mcs: int
    reconstructed_layer: int
    reconstructed_tbs_capacity_bytes: int


@dataclass(frozen=True)
class CtbsReferenceComparison:
    reconstructed_records: tuple[CtbsReconstructedRecord, ...]
    mcs_mismatch_count: int
    layer_mismatch_count: int
    tbs_mismatch_count: int
    reference_label_deletion_equal: bool = True

    @property
    def all_exact(self) -> bool:
        return (
            self.reference_label_deletion_equal
            and self.mcs_mismatch_count == 0
            and self.layer_mismatch_count == 0
            and self.tbs_mismatch_count == 0
        )


def compare_ctbs_reference_labels(
    inputs: CtbsHeadroomInputs,
    *,
    chosen_ue_per_prg: Sequence[int],
    actual_mcs: Sequence[int],
    actual_layer: Sequence[int],
    actual_tbs_capacity_bytes: Sequence[int],
    n_cell: int,
    n_prg: int,
) -> CtbsReferenceComparison:
    """Reconstruct from causal inputs first, then compare post-action labels."""
    n_ue = len(inputs.ue_ids)
    chosen = tuple(int(value) for value in chosen_ue_per_prg)
    if len(chosen) != int(n_cell) * int(n_prg):
        raise ValueError("CTBS chosen-UE vector dimensions disagree")
    references = tuple(tuple(int(value) for value in values) for values in (
        actual_mcs,
        actual_layer,
        actual_tbs_capacity_bytes,
    ))
    if any(len(values) != n_ue for values in references):
        raise ValueError("CTBS reference-label dimensions disagree")

    selected_prgs: list[list[int]] = [[] for _ in range(n_ue)]
    for token, ue in enumerate(chosen):
        if ue < 0:
            continue
        if ue >= n_ue:
            raise ValueError("CTBS chosen UE is out of range")
        cell = token // int(n_prg)
        local_prg = token % int(n_prg)
        if inputs.serving_cell[ue] != cell:
            raise ValueError("CTBS chosen UE belongs to a different cell")
        selected_prgs[ue].append(local_prg)

    records: list[CtbsReconstructedRecord] = []
    for ue, prgs in enumerate(selected_prgs):
        if not prgs:
            continue
        state = CtbsHeadroomState(
            initial_demand_bytes=0.0,
            reconstructed_mcs=inputs.reconstructed_mcs_for_ue(ue),
            config=inputs.config,
        )
        for local_prg in prgs:
            state = state.with_selected_prg(
                local_prg,
                layer_cap=inputs.prg_layer_cap[ue][local_prg],
            )
        records.append(
            CtbsReconstructedRecord(
                ue=ue,
                local_prgs=tuple(prgs),
                reconstructed_mcs=state.reconstructed_mcs,
                reconstructed_layer=min(state.selected_layer_caps),
                reconstructed_tbs_capacity_bytes=state.cumulative_tbs_bytes,
            )
        )

    actual_mcs_values, actual_layer_values, actual_tbs_values = references
    return CtbsReferenceComparison(
        reconstructed_records=tuple(records),
        mcs_mismatch_count=sum(
            record.reconstructed_mcs != actual_mcs_values[record.ue]
            for record in records
        ),
        layer_mismatch_count=sum(
            record.reconstructed_layer != actual_layer_values[record.ue]
            for record in records
        ),
        tbs_mismatch_count=sum(
            record.reconstructed_tbs_capacity_bytes != actual_tbs_values[record.ue]
            for record in records
        ),
    )


@dataclass(frozen=True)
class CtbsDecisionDiagnostics:
    selected_ue: torch.Tensor
    cell_index: torch.Tensor
    local_prg_index: torch.Tensor
    selected_nonblank: torch.Tensor
    raw_service_bytes: torch.Tensor
    cumulative_tbs_before_bytes: torch.Tensor
    cumulative_tbs_after_bytes: torch.Tensor
    remaining_before_bytes: torch.Tensor
    remaining_after_bytes: torch.Tensor
    layer_before: torch.Tensor
    layer_after: torch.Tensor


def concatenate_ctbs_diagnostics(
    rows: Sequence[CtbsDecisionDiagnostics],
) -> CtbsDecisionDiagnostics:
    if not rows:
        raise ValueError("cannot concatenate empty CTBS diagnostics")
    return CtbsDecisionDiagnostics(
        **{
            field: torch.cat([getattr(row, field) for row in rows], dim=0)
            for field in CtbsDecisionDiagnostics.__dataclass_fields__
        }
    )


__all__ = [
    "CtbsHeadroomInputs",
    "CtbsHeadroomState",
    "CtbsReconstructedRecord",
    "CtbsReferenceComparison",
    "CtbsDecisionDiagnostics",
    "compare_ctbs_reference_labels",
    "concatenate_ctbs_diagnostics",
]

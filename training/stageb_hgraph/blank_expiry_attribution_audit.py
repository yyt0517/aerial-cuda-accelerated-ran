"""Read-only attribution of expired-packet scheduling exposure."""

from __future__ import annotations

import math
import csv
import hashlib
import re
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping, Sequence

import torch


HARD_MASK = "hard_mask"
VALID_BUT_BLANK = "valid_but_blank"
VALID_BUT_OTHER_UE = "valid_but_other_ue"
SELECTED_PHY_FAILED = "selected_phy_failed"
SELECTED_SUCCESS_INSUFFICIENT = "selected_success_insufficient"
MIXED_INCONCLUSIVE = "mixed/inconclusive"
PRIMARY_CATEGORIES = (
    HARD_MASK,
    VALID_BUT_BLANK,
    VALID_BUT_OTHER_UE,
    SELECTED_PHY_FAILED,
    SELECTED_SUCCESS_INSUFFICIENT,
)
BLANK_SUBCATEGORIES = ("conservative", "fragmentation", "other")
FROZEN_B4_STATE_DICT_SHA256 = (
    "4ce1704303d9f38a6846511630d790a55532acbee777083330bd6445648b6571"
)
V1_PROTOCOL_ID = "seed42-ctbs-blank-expiry-attribution-read-only-audit-v1"
V2_FULL_HORIZON_PROTOCOL_ID = (
    "seed42-ctbs-blank-expiry-attribution-read-only-audit-v2-full-horizon"
)
V2_MIN_EXPIRED_PACKETS = 50


@dataclass(frozen=True)
class DominanceResult:
    category: str
    share: float
    lead: float


@dataclass(frozen=True)
class ExpiredPacket:
    packet_id: int
    ue: int
    arrival_tti: int
    terminal_tti: int
    packet_size_bytes: int
    remaining_bytes_at_expiry: int


@dataclass(frozen=True)
class AtomicExposure:
    packet_id: int
    ue: int
    tti: int
    serving_cell: int
    local_prg: int
    category: str
    blank_subcategory: str | None
    expiry_byte_weight: int


@dataclass(frozen=True)
class BlankExpiryDecisionDiagnostics:
    """Detached audit-only state for every cell-local-PRG decision."""

    cell_index: torch.Tensor
    local_prg_index: torch.Tensor
    candidate_ue_ids: torch.Tensor
    candidate_present: torch.Tensor
    candidate_valid: torch.Tensor
    candidate_logit: torch.Tensor
    candidate_probability: torch.Tensor
    initial_backlog_bytes: torch.Tensor
    remaining_demand_before_bytes: torch.Tensor
    cumulative_capacity_before_bytes: torch.Tensor
    hol_delay_ms: torch.Tensor
    ttl_slack_ms: torch.Tensor
    selected_ue: torch.Tensor
    selected_class: torch.Tensor
    selected_rank: torch.Tensor
    valid_nonblank_count: torch.Tensor
    blank_logit: torch.Tensor
    blank_probability: torch.Tensor
    best_valid_ue_logit: torch.Tensor
    service_probability: torch.Tensor
    blank_subcategory_code: torch.Tensor


@dataclass(frozen=True)
class BlankExpiryTtiTrace:
    tti: int
    decisions: BlankExpiryDecisionDiagnostics
    actual_ue_served_bytes: tuple[float, ...]
    actual_ue_goodput_bytes: tuple[float, ...]
    actual_ue_tb_tx_count: tuple[int, ...]
    actual_ue_tb_err_count: tuple[int, ...]


def classify_v2_full_horizon_result(
    *,
    trajectory_reproduced: bool,
    expired_packet_count: int,
    attribution_conclusion: str,
) -> str:
    """Apply the prospective v2 trajectory and event-support gates in order."""
    if expired_packet_count < 0:
        raise ValueError("expired_packet_count must be nonnegative")
    if not trajectory_reproduced:
        return "formal_trajectory_nonreproduction"
    if expired_packet_count < V2_MIN_EXPIRED_PACKETS:
        return "insufficient_event_support"
    return str(attribution_conclusion)


def canonical_type0_action_fingerprints(
    rows: Sequence[Mapping[str, object]],
    *,
    tti_first: int,
    tti_last: int,
    n_cell: int,
    n_prg: int,
) -> dict[str, object]:
    """Hash the deterministic Type-0 schedule identity selected per cell-PRG."""
    if tti_first <= 0 or tti_last < tti_first or n_cell <= 0 or n_prg <= 0:
        raise ValueError("Type-0 fingerprint dimensions are invalid")
    selected_by_key: dict[tuple[int, int, int], int] = {}
    for row in rows:
        tti = int(row["tti"])
        if not tti_first <= tti <= tti_last:
            continue
        cell = int(row["serving_cell"])
        local_prg = int(row["local_prg"])
        selected_ue = int(row["selected_ue"])
        if not 0 <= cell < n_cell or not 0 <= local_prg < n_prg or selected_ue < -1:
            raise ValueError("Type-0 fingerprint row is invalid")
        key = (tti, cell, local_prg)
        if key in selected_by_key:
            raise ValueError("Type-0 fingerprint rows must be unique")
        selected_by_key[key] = selected_ue

    global_hash = hashlib.sha256()
    per_tti: dict[str, str] = {}
    for tti in range(tti_first, tti_last + 1):
        expected = [
            (tti, cell, local_prg)
            for local_prg in range(n_prg)
            for cell in range(n_cell)
        ]
        if any(key not in selected_by_key for key in expected):
            raise ValueError("Type-0 fingerprint requires complete cell-PRG coverage")
        payload = struct.pack("<q", tti) + b"".join(
            struct.pack("<i", selected_by_key[key]) for key in expected
        )
        digest = hashlib.sha256(payload).hexdigest()
        per_tti[str(tti)] = digest
        global_hash.update(payload)
    return {
        "tti_first": tti_first,
        "tti_last": tti_last,
        "tti_count": tti_last - tti_first + 1,
        "identity": "deterministic_type0_selected_ue_per_cell_local_prg",
        "global_sha256": global_hash.hexdigest(),
        "per_tti_sha256": per_tti,
    }


def canonical_packet_outcome_fingerprint(
    path: Path,
    *,
    tti_first: int,
    tti_last: int,
) -> dict[str, object]:
    """Hash the legacy packet terminal fields shared by old and enriched CSVs."""
    required = {"ue_id", "packet_id", "terminal_tti", "outcome", "system_time_ms"}
    rows: list[tuple[int, int, int, str, float]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("packet outcome CSV is missing shared trajectory columns")
        for row in reader:
            terminal_tti = int(row["terminal_tti"])
            if tti_first <= terminal_tti <= tti_last:
                rows.append(
                    (
                        int(row["ue_id"]),
                        int(row["packet_id"]),
                        terminal_tti,
                        str(row["outcome"]),
                        float(row["system_time_ms"]),
                    )
                )
    digest = hashlib.sha256()
    for ue, packet_id, terminal_tti, outcome, system_time_ms in rows:
        encoded_outcome = outcome.encode("utf-8")
        digest.update(struct.pack("<qqqI", ue, packet_id, terminal_tti, len(encoded_outcome)))
        digest.update(encoded_outcome)
        digest.update(struct.pack("<d", system_time_ms))
    return {
        "tti_first": tti_first,
        "tti_last": tti_last,
        "row_count": len(rows),
        "delivered_count": sum(outcome == "delivered" for *_, outcome, _ in rows),
        "expired_count": sum(outcome == "expired" for *_, outcome, _ in rows),
        "sha256": digest.hexdigest(),
    }


def canonical_compact_cpu_fingerprints(
    path: Path,
    *,
    tti_first: int,
    tti_last: int,
    completed_steps: int,
    ring_horizon: int = 4000,
) -> dict[str, object]:
    """Resolve compact CPU record ring slots to chronological TTI fingerprints."""
    earliest_available = max(1, int(completed_steps) - int(ring_horizon) + 1)
    if (
        tti_first < earliest_available
        or tti_last > completed_steps
        or tti_last < tti_first
        or ring_horizon <= 0
    ):
        raise ValueError("requested compact trajectory interval is unavailable")
    pattern = re.compile(r"^(\w+RecordsCpu) = \[(.*)\];$")
    raw_series: dict[str, list[str]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            values = match.group(2).split()
            if len(values) != ring_horizon:
                raise ValueError("compact CPU record does not match ring horizon")
            raw_series[match.group(1)] = values
    required = {
        "mcType0SumCellThrRecordsCpu",
        "mcType0SumCellPfRecordsCpu",
        "mcType0SumInsThrRecordsCpu",
    }
    if set(raw_series) != required:
        raise ValueError("compact CPU trajectory series are incomplete")
    output: dict[str, dict[str, object]] = {}
    for name in sorted(raw_series):
        values = [
            raw_series[name][(tti - 1) % ring_horizon]
            for tti in range(tti_first, tti_last + 1)
        ]
        output[name] = {
            "sha256": hashlib.sha256(" ".join(values).encode("utf-8")).hexdigest(),
            "values": values,
        }
    return {
        "tti_first": tti_first,
        "tti_last": tti_last,
        "tti_count": tti_last - tti_first + 1,
        "series": output,
    }


def canonical_evaluator_progress_fingerprint(path: Path) -> dict[str, object]:
    """Hash evaluator progress snapshots without audit-only final metadata."""
    lines = [
        line
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.startswith("[hgraph-eval]")
    ]
    if not lines:
        raise ValueError("evaluator log has no progress snapshots")
    payload = "\n".join(lines).encode("utf-8") + b"\n"
    return {
        "line_count": len(lines),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "first": lines[0],
        "last": lines[-1],
    }


def build_blank_expiry_decision_diagnostics(
    *,
    cell_index: torch.Tensor,
    local_prg: int,
    candidate_ue_ids: torch.Tensor,
    candidate_present: torch.Tensor,
    candidate_valid: torch.Tensor,
    logits: torch.Tensor,
    selected_class: torch.Tensor,
    initial_backlog_bytes: torch.Tensor,
    remaining_demand_before_bytes: torch.Tensor,
    cumulative_capacity_before_bytes: torch.Tensor,
    hol_delay_ms: torch.Tensor,
    ttl_slack_ms: torch.Tensor,
) -> BlankExpiryDecisionDiagnostics:
    """Capture one reuse group's attribution state without touching policy tensors."""
    ids = torch.as_tensor(candidate_ue_ids).detach()
    present = torch.as_tensor(candidate_present, device=ids.device, dtype=torch.bool).detach()
    valid = torch.as_tensor(candidate_valid, device=ids.device, dtype=torch.bool).detach()
    raw_logits = torch.as_tensor(logits, device=ids.device).detach()
    selected = torch.as_tensor(selected_class, device=ids.device, dtype=torch.long).detach()
    if ids.dim() != 2 or present.shape != ids.shape or valid.shape != ids.shape:
        raise ValueError("candidate audit tensors must have matching rank-two shapes")
    count, width = ids.shape
    if raw_logits.shape != (count, width + 1) or selected.shape != (count,):
        raise ValueError("candidate logits or selected class shape mismatch")
    if bool((valid & ~present).any().item()):
        raise ValueError("valid candidate must be present")
    full_valid = torch.cat(
        (valid, torch.ones((count, 1), dtype=torch.bool, device=ids.device)), dim=1
    )
    probabilities = torch.stack(
        [post_mask_probabilities(raw_logits[row], full_valid[row]) for row in range(count)]
    )
    if not bool(full_valid.gather(1, selected[:, None]).all().item()):
        raise ValueError("selected class is invalid after the hard mask")
    selected_ue = torch.full((count,), -1, dtype=torch.long, device=ids.device)
    nonblank = selected < width
    if bool(nonblank.any().item()):
        gathered = ids.gather(1, selected.clamp(max=max(0, width - 1))[:, None]).squeeze(1)
        selected_ue[nonblank] = gathered[nonblank]
    masked_logits = raw_logits.masked_fill(~full_valid, -torch.inf)
    sorted_classes = torch.argsort(masked_logits, dim=1, descending=True, stable=True)
    selected_rank = (
        torch.nonzero(sorted_classes == selected[:, None], as_tuple=False)[:, 1] + 1
    )
    valid_nonblank_count = valid.sum(dim=1)
    service_probability = probabilities[:, :width].sum(dim=1)
    blank_probability = probabilities[:, width]
    best_valid_ue_logit = masked_logits[:, :width].max(dim=1).values
    best_valid_ue_logit = torch.where(
        valid_nonblank_count > 0,
        best_valid_ue_logit,
        torch.full_like(best_valid_ue_logit, -torch.inf),
    )
    # 0=not Blank, 1=forced, 2=conservative, 3=fragmentation, 4=other.
    blank_code = torch.zeros((count,), dtype=torch.long, device=ids.device)
    selected_blank = ~nonblank
    blank_code[selected_blank & (valid_nonblank_count == 0)] = 1
    learned_blank = selected_blank & (valid_nonblank_count > 0)
    blank_code[learned_blank & (blank_probability >= service_probability)] = 2
    blank_code[learned_blank & (service_probability > blank_probability)] = 3
    unresolved = learned_blank & (blank_code == 0)
    blank_code[unresolved] = 4

    def matrix(value: torch.Tensor, name: str) -> torch.Tensor:
        result = torch.as_tensor(value, device=ids.device).detach()
        if result.shape != ids.shape or not bool(torch.isfinite(result).all().item()):
            raise ValueError(f"{name} must be a finite candidate matrix")
        return result.cpu()

    result = BlankExpiryDecisionDiagnostics(
        cell_index=torch.as_tensor(cell_index, device=ids.device, dtype=torch.long).detach().reshape(-1).cpu(),
        local_prg_index=torch.full((count,), int(local_prg), dtype=torch.long),
        candidate_ue_ids=ids.cpu(),
        candidate_present=present.cpu(),
        candidate_valid=valid.cpu(),
        candidate_logit=raw_logits[:, :width].cpu(),
        candidate_probability=probabilities[:, :width].cpu(),
        initial_backlog_bytes=matrix(initial_backlog_bytes, "initial_backlog_bytes"),
        remaining_demand_before_bytes=matrix(
            remaining_demand_before_bytes, "remaining_demand_before_bytes"
        ),
        cumulative_capacity_before_bytes=matrix(
            cumulative_capacity_before_bytes, "cumulative_capacity_before_bytes"
        ),
        hol_delay_ms=matrix(hol_delay_ms, "hol_delay_ms"),
        ttl_slack_ms=matrix(ttl_slack_ms, "ttl_slack_ms"),
        selected_ue=selected_ue.cpu(),
        selected_class=selected.cpu(),
        selected_rank=selected_rank.cpu(),
        valid_nonblank_count=valid_nonblank_count.cpu(),
        blank_logit=raw_logits[:, width].cpu(),
        blank_probability=blank_probability.cpu(),
        best_valid_ue_logit=best_valid_ue_logit.cpu(),
        service_probability=service_probability.cpu(),
        blank_subcategory_code=blank_code.cpu(),
    )
    if result.cell_index.shape != (count,):
        raise ValueError("cell_index must contain one value per decision")
    return result


def concatenate_blank_expiry_decision_diagnostics(
    rows: Sequence[BlankExpiryDecisionDiagnostics],
) -> BlankExpiryDecisionDiagnostics:
    if not rows:
        raise ValueError("at least one Blank-expiry decision row is required")
    values: dict[str, torch.Tensor] = {}
    for field in fields(BlankExpiryDecisionDiagnostics):
        tensors = [getattr(row, field.name) for row in rows]
        trailing = tensors[0].shape[1:]
        if any(tensor.shape[1:] != trailing for tensor in tensors):
            raise ValueError(f"{field.name} trailing shape mismatch")
        values[field.name] = torch.cat(tensors, dim=0).detach().cpu()
    return BlankExpiryDecisionDiagnostics(**values)


def read_formal_expired_packets(
    path: Path,
    *,
    measured_tti_first: int,
    measured_tti_last: int,
    trace_first_tti: int,
) -> list[ExpiredPacket]:
    required = {
        "ue_id",
        "packet_id",
        "arrival_tti",
        "packet_size_bytes",
        "remaining_bytes_at_terminal",
        "terminal_tti",
        "outcome",
    }
    packets: list[ExpiredPacket] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                "packet outcome CSV is missing required lifecycle columns: "
                f"{sorted(required)}"
            )
        for row in reader:
            terminal_tti = int(row["terminal_tti"])
            if not measured_tti_first <= terminal_tti <= measured_tti_last:
                continue
            if row["outcome"] != "expired":
                continue
            packet = ExpiredPacket(
                packet_id=int(row["packet_id"]),
                ue=int(row["ue_id"]),
                arrival_tti=int(row["arrival_tti"]),
                terminal_tti=terminal_tti,
                packet_size_bytes=int(row["packet_size_bytes"]),
                remaining_bytes_at_expiry=int(row["remaining_bytes_at_terminal"]),
            )
            if packet.arrival_tti < trace_first_tti:
                raise ValueError(
                    "formal expired packet arrival_tti precedes trace_first_tti"
                )
            if (
                packet.packet_id <= 0
                or packet.ue < 0
                or packet.packet_size_bytes <= 0
                or packet.remaining_bytes_at_expiry <= 0
                or packet.remaining_bytes_at_expiry > packet.packet_size_bytes
                or packet.terminal_tti < packet.arrival_tti
            ):
                raise ValueError("formal expired packet lifecycle values are invalid")
            packets.append(packet)
    packet_ids = [packet.packet_id for packet in packets]
    if len(packet_ids) != len(set(packet_ids)):
        raise ValueError("formal expired packet IDs must be unique")
    return packets


def build_atomic_exposures(
    *,
    packets: Sequence[ExpiredPacket],
    traces: Sequence[BlankExpiryTtiTrace],
    serving_cell_by_ue: Sequence[int],
    n_cell: int,
    n_prg: int,
) -> list[AtomicExposure]:
    if n_cell <= 0 or n_prg <= 0:
        raise ValueError("trace cell and PRG dimensions must be positive")
    trace_by_tti: dict[int, BlankExpiryTtiTrace] = {}
    expected_decisions = n_cell * n_prg
    for trace in traces:
        if trace.tti in trace_by_tti:
            raise ValueError("Blank-expiry trace TTIs must be unique")
        decisions = trace.decisions
        if decisions.cell_index.shape != (expected_decisions,):
            raise ValueError("Blank-expiry trace does not cover all cell-PRG decisions")
        expected_cell = torch.arange(n_cell).repeat(n_prg)
        expected_prg = torch.arange(n_prg).repeat_interleave(n_cell)
        if not torch.equal(decisions.cell_index.to(torch.long), expected_cell):
            raise ValueError("Blank-expiry decision cell order mismatch")
        if not torch.equal(decisions.local_prg_index.to(torch.long), expected_prg):
            raise ValueError("Blank-expiry decision PRG order mismatch")
        n_ue = len(serving_cell_by_ue)
        outcome_lengths = {
            len(trace.actual_ue_served_bytes),
            len(trace.actual_ue_goodput_bytes),
            len(trace.actual_ue_tb_tx_count),
            len(trace.actual_ue_tb_err_count),
        }
        if outcome_lengths != {n_ue}:
            raise ValueError("Blank-expiry UE outcome vector shape mismatch")
        if any(
            err < 0 or tx < 0 or err > tx
            for tx, err in zip(trace.actual_ue_tb_tx_count, trace.actual_ue_tb_err_count)
        ):
            raise ValueError("Blank-expiry TB outcome values are invalid")
        trace_by_tti[trace.tti] = trace

    atoms: list[AtomicExposure] = []
    blank_names = {1: "forced", 2: "conservative", 3: "fragmentation", 4: "other"}
    for packet in packets:
        if not 0 <= packet.ue < len(serving_cell_by_ue):
            raise ValueError("formal expired packet UE is out of range")
        serving_cell = int(serving_cell_by_ue[packet.ue])
        if not 0 <= serving_cell < n_cell:
            raise ValueError("formal expired packet serving cell is out of range")
        for tti in range(packet.arrival_tti, packet.terminal_tti + 1):
            trace = trace_by_tti.get(tti)
            if trace is None:
                raise ValueError(f"lifecycle coverage mismatch: missing trace TTI {tti}")
            for local_prg in range(n_prg):
                row = local_prg * n_cell + serving_cell
                decisions = trace.decisions
                ids = decisions.candidate_ue_ids[row]
                present = decisions.candidate_present[row]
                matches = torch.nonzero(present & (ids == packet.ue), as_tuple=False).flatten()
                if matches.numel() > 1:
                    raise ValueError("target UE is duplicated in a candidate row")
                target_valid = False
                if matches.numel() == 1:
                    target_valid = bool(
                        decisions.candidate_valid[row, int(matches[0].item())].item()
                    )
                selected_ue = int(decisions.selected_ue[row].item())
                category = classify_exposure(
                    target_valid=target_valid,
                    selected_ue=selected_ue,
                    target_ue=packet.ue,
                    tb_tx_count=int(trace.actual_ue_tb_tx_count[packet.ue]),
                    tb_err_count=int(trace.actual_ue_tb_err_count[packet.ue]),
                )
                blank_subcategory = None
                if category == VALID_BUT_BLANK:
                    code = int(decisions.blank_subcategory_code[row].item())
                    blank_subcategory = blank_names.get(code)
                    if blank_subcategory not in BLANK_SUBCATEGORIES:
                        raise ValueError("selected Blank lacks a valid internal attribution")
                atoms.append(
                    AtomicExposure(
                        packet_id=packet.packet_id,
                        ue=packet.ue,
                        tti=tti,
                        serving_cell=serving_cell,
                        local_prg=local_prg,
                        category=category,
                        blank_subcategory=blank_subcategory,
                        expiry_byte_weight=packet.remaining_bytes_at_expiry,
                    )
                )
    return atoms


def decision_diagnostics_to_rows(
    *,
    tti: int,
    diagnostics: BlankExpiryDecisionDiagnostics,
) -> list[dict[str, object]]:
    count = int(diagnostics.cell_index.numel())
    width = int(diagnostics.candidate_ue_ids.shape[1])
    matrix_fields = (
        "candidate_ue_ids",
        "candidate_present",
        "candidate_valid",
        "candidate_logit",
        "candidate_probability",
        "initial_backlog_bytes",
        "remaining_demand_before_bytes",
        "cumulative_capacity_before_bytes",
        "hol_delay_ms",
        "ttl_slack_ms",
    )
    if any(getattr(diagnostics, name).shape != (count, width) for name in matrix_fields):
        raise ValueError("Blank-expiry decision diagnostic matrix shape mismatch")
    vector_fields = (
        "cell_index",
        "local_prg_index",
        "selected_ue",
        "selected_class",
        "selected_rank",
        "valid_nonblank_count",
        "blank_logit",
        "blank_probability",
        "best_valid_ue_logit",
        "service_probability",
        "blank_subcategory_code",
    )
    if any(getattr(diagnostics, name).shape != (count,) for name in vector_fields):
        raise ValueError("Blank-expiry decision diagnostic vector shape mismatch")
    blank_names = {0: None, 1: "forced", 2: "conservative", 3: "fragmentation", 4: "other"}
    rows: list[dict[str, object]] = []
    for row in range(count):
        present = diagnostics.candidate_present[row].to(torch.bool)
        valid = diagnostics.candidate_valid[row].to(torch.bool)
        mask_reason = [
            "valid" if bool(valid[col].item()) else (
                "headroom_exhausted" if bool(present[col].item()) else "not_candidate_padding"
            )
            for col in range(width)
        ]
        code = int(diagnostics.blank_subcategory_code[row].item())
        if code not in blank_names:
            raise ValueError("Blank-expiry subcategory code is out of range")
        probability_sum = float(
            diagnostics.candidate_probability[row].sum().item()
            + diagnostics.blank_probability[row].item()
        )
        if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError("Blank-expiry post-mask probabilities do not sum to one")
        rows.append(
            {
                "tti": int(tti),
                "serving_cell": int(diagnostics.cell_index[row].item()),
                "local_prg": int(diagnostics.local_prg_index[row].item()),
                "valid_nonblank_count": int(diagnostics.valid_nonblank_count[row].item()),
                "candidate_ue_ids": [int(value) for value in diagnostics.candidate_ue_ids[row].tolist()],
                "candidate_present": [bool(value) for value in present.tolist()],
                "candidate_valid": [bool(value) for value in valid.tolist()],
                "candidate_mask_reason": mask_reason,
                "candidate_logits": [float(value) for value in diagnostics.candidate_logit[row].tolist()],
                "candidate_probabilities": [float(value) for value in diagnostics.candidate_probability[row].tolist()],
                "initial_backlog_bytes": [float(value) for value in diagnostics.initial_backlog_bytes[row].tolist()],
                "remaining_demand_before_bytes": [float(value) for value in diagnostics.remaining_demand_before_bytes[row].tolist()],
                "ctbs_capacity_before_bytes": [float(value) for value in diagnostics.cumulative_capacity_before_bytes[row].tolist()],
                "hol_delay_ms": [float(value) for value in diagnostics.hol_delay_ms[row].tolist()],
                "ttl_slack_ms": [float(value) for value in diagnostics.ttl_slack_ms[row].tolist()],
                "blank_logit": float(diagnostics.blank_logit[row].item()),
                "blank_probability": float(diagnostics.blank_probability[row].item()),
                "best_valid_ue_logit": (
                    float(diagnostics.best_valid_ue_logit[row].item())
                    if math.isfinite(float(diagnostics.best_valid_ue_logit[row].item()))
                    else None
                ),
                "service_probability": float(diagnostics.service_probability[row].item()),
                "selected_ue": int(diagnostics.selected_ue[row].item()),
                "selected_class": int(diagnostics.selected_class[row].item()),
                "selected_rank": int(diagnostics.selected_rank[row].item()),
                "blank_subcategory": blank_names[code],
                "post_mask_probability_sum": probability_sum,
            }
        )
    return rows


def packet_lifecycle_rows(
    *,
    packets: Sequence[ExpiredPacket],
    traces: Sequence[BlankExpiryTtiTrace],
    serving_cell_by_ue: Sequence[int],
    n_cell: int,
    n_prg: int,
) -> list[dict[str, object]]:
    trace_by_tti = {trace.tti: trace for trace in traces}
    rows: list[dict[str, object]] = []
    for packet in packets:
        serving_cell = int(serving_cell_by_ue[packet.ue])
        for tti in range(packet.arrival_tti, packet.terminal_tti + 1):
            trace = trace_by_tti.get(tti)
            if trace is None:
                raise ValueError(f"lifecycle coverage mismatch: missing trace TTI {tti}")
            selected_prgs: list[int] = []
            remaining_by_prg: list[float | None] = []
            capacity_by_prg: list[float | None] = []
            hol_by_prg: list[float | None] = []
            ttl_by_prg: list[float | None] = []
            initial_backlog: float | None = None
            for local_prg in range(n_prg):
                decision_row = local_prg * n_cell + serving_cell
                diagnostics = trace.decisions
                if int(diagnostics.selected_ue[decision_row].item()) == packet.ue:
                    selected_prgs.append(local_prg)
                matches = torch.nonzero(
                    diagnostics.candidate_present[decision_row]
                    & (diagnostics.candidate_ue_ids[decision_row] == packet.ue),
                    as_tuple=False,
                ).flatten()
                if matches.numel() > 1:
                    raise ValueError("target UE is duplicated in a candidate row")
                if matches.numel() == 0:
                    remaining_by_prg.append(None)
                    capacity_by_prg.append(None)
                    hol_by_prg.append(None)
                    ttl_by_prg.append(None)
                    continue
                candidate = int(matches[0].item())
                initial_backlog = float(
                    diagnostics.initial_backlog_bytes[decision_row, candidate].item()
                )
                remaining_by_prg.append(float(
                    diagnostics.remaining_demand_before_bytes[decision_row, candidate].item()
                ))
                capacity_by_prg.append(float(
                    diagnostics.cumulative_capacity_before_bytes[decision_row, candidate].item()
                ))
                hol_by_prg.append(float(diagnostics.hol_delay_ms[decision_row, candidate].item()))
                ttl_by_prg.append(float(diagnostics.ttl_slack_ms[decision_row, candidate].item()))
            rows.append(
                {
                    "packet_id": packet.packet_id,
                    "ue": packet.ue,
                    "serving_cell": serving_cell,
                    "arrival_tti": packet.arrival_tti,
                    "terminal_tti": packet.terminal_tti,
                    "tti": tti,
                    "packet_size_bytes": packet.packet_size_bytes,
                    "remaining_bytes_at_expiry": packet.remaining_bytes_at_expiry,
                    "expired_bytes_this_tti": (
                        packet.remaining_bytes_at_expiry if tti == packet.terminal_tti else 0
                    ),
                    "initial_backlog_bytes": initial_backlog,
                    "selected_local_prgs": selected_prgs,
                    "scheduled_bytes": float(trace.actual_ue_served_bytes[packet.ue]),
                    "successful_bytes": float(trace.actual_ue_goodput_bytes[packet.ue]),
                    "tb_tx_count": int(trace.actual_ue_tb_tx_count[packet.ue]),
                    "tb_err_count": int(trace.actual_ue_tb_err_count[packet.ue]),
                    "remaining_demand_before_by_local_prg": remaining_by_prg,
                    "ctbs_capacity_before_by_local_prg": capacity_by_prg,
                    "hol_delay_ms_by_local_prg": hol_by_prg,
                    "ttl_slack_ms_by_local_prg": ttl_by_prg,
                }
            )
    return rows


def classify_exposure(
    *,
    target_valid: bool,
    selected_ue: int,
    target_ue: int,
    tb_tx_count: int,
    tb_err_count: int,
) -> str:
    """Assign one target-UE decision exposure to exactly one primary class."""
    if target_ue < 0 or selected_ue < -1:
        raise ValueError("UE identities must use non-negative values or -1 for Blank")
    if tb_tx_count < 0 or tb_err_count < 0 or tb_err_count > tb_tx_count:
        raise ValueError("TB outcome counts are invalid")
    if not target_valid:
        if selected_ue == target_ue:
            raise ValueError("hard-masked target UE cannot be selected")
        return HARD_MASK
    if selected_ue == -1:
        return VALID_BUT_BLANK
    if selected_ue != target_ue:
        return VALID_BUT_OTHER_UE
    if tb_tx_count <= 0:
        raise ValueError("selected target UE has no TB outcome")
    return SELECTED_PHY_FAILED if tb_err_count > 0 else SELECTED_SUCCESS_INSUFFICIENT


def dominant_category(counts: Mapping[str, float]) -> DominanceResult:
    """Apply the prospective >=50% and >=10 percentage-point dominance rule."""
    values: list[tuple[str, float]] = []
    for category, raw in counts.items():
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("dominance counts must be finite and non-negative")
        values.append((str(category), value))
    total = sum(value for _, value in values)
    if total <= 0.0:
        return DominanceResult(MIXED_INCONCLUSIVE, 0.0, 0.0)
    values.sort(key=lambda item: (-item[1], item[0]))
    leader, leader_count = values[0]
    runner_up_count = values[1][1] if len(values) > 1 else 0.0
    share = leader_count / total
    lead = (leader_count - runner_up_count) / total
    category = leader if share >= 0.5 and lead >= 0.1 else MIXED_INCONCLUSIVE
    return DominanceResult(category, share, lead)


def combine_view_conclusions(
    atomic: DominanceResult,
    ue_tti: DominanceResult,
) -> str:
    if (
        atomic.category != MIXED_INCONCLUSIVE
        and atomic.category == ue_tti.category
    ):
        return atomic.category
    return MIXED_INCONCLUSIVE


def post_mask_probabilities(
    logits: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    logits = torch.as_tensor(logits)
    valid = torch.as_tensor(valid, device=logits.device, dtype=torch.bool)
    if logits.shape != valid.shape or logits.dim() != 1:
        raise ValueError("logits and valid must have matching one-dimensional shapes")
    if not bool(torch.isfinite(logits[valid]).all().item()):
        raise ValueError("valid logits must be finite")
    if not bool(valid.any().item()):
        raise ValueError("at least one post-mask action must be valid")
    adjusted = logits.masked_fill(~valid, -torch.inf)
    probabilities = torch.softmax(adjusted, dim=0)
    if not bool(torch.isfinite(probabilities).all().item()):
        raise ValueError("post-mask probabilities must be finite")
    return probabilities


def _shares(counts: Mapping[str, float]) -> dict[str, float]:
    total = float(sum(counts.values()))
    return {
        category: (float(value) / total if total > 0.0 else 0.0)
        for category, value in counts.items()
    }


def _mode(values: Sequence[str]) -> str:
    if not values:
        raise ValueError("UE-TTI modal view cannot be empty")
    counts = Counter(values)
    highest = max(counts.values())
    leaders = [category for category, count in counts.items() if count == highest]
    return leaders[0] if len(leaders) == 1 else MIXED_INCONCLUSIVE


def _dominance_payload(result: DominanceResult) -> dict[str, object]:
    return {
        "category": result.category,
        "share": result.share,
        "lead_over_runner_up": result.lead,
        "minimum_share": 0.5,
        "minimum_lead": 0.1,
    }


def build_blank_expiry_attribution_report(
    *,
    packets: Sequence[ExpiredPacket],
    exposures: Sequence[AtomicExposure],
    measured_tti_first: int,
    measured_tti_last: int,
    trace_first_tti: int,
    n_prg: int,
    protocol_id: str = V1_PROTOCOL_ID,
) -> dict[str, object]:
    if measured_tti_first <= 0 or measured_tti_last < measured_tti_first:
        raise ValueError("measured TTI interval is invalid")
    if trace_first_tti <= 0 or n_prg <= 0:
        raise ValueError("trace_first_tti and n_prg must be positive")
    packet_by_id: dict[int, ExpiredPacket] = {}
    for packet in packets:
        if packet.packet_id in packet_by_id:
            raise ValueError("expired packet IDs must be unique")
        if not measured_tti_first <= packet.terminal_tti <= measured_tti_last:
            raise ValueError("formal expired packet terminal TTI is outside measured interval")
        if packet.arrival_tti < trace_first_tti:
            raise ValueError("expired packet arrival_tti precedes trace_first_tti")
        if (
            packet.packet_id <= 0
            or packet.ue < 0
            or packet.arrival_tti <= 0
            or packet.terminal_tti < packet.arrival_tti
            or packet.packet_size_bytes <= 0
            or packet.remaining_bytes_at_expiry <= 0
            or packet.remaining_bytes_at_expiry > packet.packet_size_bytes
        ):
            raise ValueError("expired packet lifecycle values are invalid")
        packet_by_id[packet.packet_id] = packet

    exposure_by_key: dict[tuple[int, int, int, int], AtomicExposure] = {}
    cells_by_packet: dict[int, set[int]] = defaultdict(set)
    for exposure in exposures:
        packet = packet_by_id.get(exposure.packet_id)
        if packet is None:
            raise ValueError("exposure references a non-formal expired packet")
        if (
            exposure.ue != packet.ue
            or exposure.expiry_byte_weight != packet.remaining_bytes_at_expiry
            or not packet.arrival_tti <= exposure.tti <= packet.terminal_tti
            or exposure.serving_cell < 0
            or not 0 <= exposure.local_prg < n_prg
            or exposure.category not in PRIMARY_CATEGORIES
        ):
            raise ValueError("atomic exposure values are inconsistent with packet lifecycle")
        if exposure.category == VALID_BUT_BLANK:
            if exposure.blank_subcategory not in BLANK_SUBCATEGORIES:
                raise ValueError("valid_but_blank exposure requires a valid subcategory")
        elif exposure.blank_subcategory is not None:
            raise ValueError("Blank subcategory is only valid for valid_but_blank")
        key = (
            exposure.packet_id,
            exposure.tti,
            exposure.serving_cell,
            exposure.local_prg,
        )
        if key in exposure_by_key:
            raise ValueError("atomic exposure keys must be unique")
        exposure_by_key[key] = exposure
        cells_by_packet[exposure.packet_id].add(exposure.serving_cell)

    expected_atomic_count = 0
    expected_keys: set[tuple[int, int, int, int]] = set()
    for packet in packets:
        cells = cells_by_packet.get(packet.packet_id, set())
        if len(cells) != 1:
            raise ValueError("lifecycle coverage mismatch: serving cell is missing or unstable")
        serving_cell = next(iter(cells))
        for tti in range(packet.arrival_tti, packet.terminal_tti + 1):
            for local_prg in range(n_prg):
                expected_keys.add((packet.packet_id, tti, serving_cell, local_prg))
                expected_atomic_count += 1
    if set(exposure_by_key) != expected_keys:
        raise ValueError(
            "lifecycle coverage mismatch: "
            f"actual={len(exposure_by_key)} expected={expected_atomic_count}"
        )

    unweighted = {category: 0 for category in PRIMARY_CATEGORIES}
    weighted = {category: 0 for category in PRIMARY_CATEGORIES}
    blank_unweighted = {category: 0 for category in BLANK_SUBCATEGORIES}
    blank_weighted = {category: 0 for category in BLANK_SUBCATEGORIES}
    cell_prg: dict[tuple[int, int], dict[str, int]] = defaultdict(
        lambda: {category: 0 for category in PRIMARY_CATEGORIES}
    )
    base_category: dict[tuple[int, int, int], str] = {}
    active_packet_weights: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
    for exposure in exposures:
        unweighted[exposure.category] += 1
        weighted[exposure.category] += exposure.expiry_byte_weight
        cell_prg[(exposure.serving_cell, exposure.local_prg)][exposure.category] += 1
        if exposure.category == VALID_BUT_BLANK:
            assert exposure.blank_subcategory is not None
            blank_unweighted[exposure.blank_subcategory] += 1
            blank_weighted[exposure.blank_subcategory] += exposure.expiry_byte_weight
        base_key = (exposure.ue, exposure.tti, exposure.local_prg)
        previous = base_category.setdefault(base_key, exposure.category)
        if previous != exposure.category:
            raise ValueError("overlapping packet lifecycles disagree on UE decision category")
        active_packet_weights[(exposure.ue, exposure.tti)][exposure.packet_id] = (
            exposure.expiry_byte_weight
        )

    ue_tti_unweighted = {category: 0 for category in (*PRIMARY_CATEGORIES, MIXED_INCONCLUSIVE)}
    ue_tti_weighted = {category: 0 for category in (*PRIMARY_CATEGORIES, MIXED_INCONCLUSIVE)}
    for (ue, tti), weights_by_packet in active_packet_weights.items():
        categories = [base_category[(ue, tti, local_prg)] for local_prg in range(n_prg)]
        category = _mode(categories)
        ue_tti_unweighted[category] += 1
        ue_tti_weighted[category] += sum(weights_by_packet.values())

    atomic_dominance = dominant_category(weighted)
    ue_tti_dominance = dominant_category(ue_tti_weighted)
    conclusion = combine_view_conclusions(atomic_dominance, ue_tti_dominance)
    cell_prg_rows = [
        {
            "serving_cell": cell,
            "local_prg": local_prg,
            "unweighted_counts": counts,
            "unweighted_shares": _shares(counts),
        }
        for (cell, local_prg), counts in sorted(cell_prg.items())
    ]
    return {
        "protocol_id": str(protocol_id),
        "classification": conclusion,
        "coverage": {
            "trace_first_tti": trace_first_tti,
            "measured_tti_first": measured_tti_first,
            "measured_tti_last": measured_tti_last,
            "formal_expired_packet_count": len(packets),
            "expected_atomic_count": expected_atomic_count,
            "actual_atomic_count": len(exposures),
            "unclassified_count": 0,
            "complete": True,
        },
        "atomic_view": {
            "unit": "expired_packet_x_tti_x_serving_cell_x_local_prg",
            "weight": "remaining_bytes_at_expiry",
            "unweighted_counts": unweighted,
            "unweighted_shares": _shares(unweighted),
            "weighted_counts": weighted,
            "weighted_shares": _shares(weighted),
            "dominance": _dominance_payload(atomic_dominance),
            "valid_but_blank_internal": {
                "unweighted_counts": blank_unweighted,
                "unweighted_shares": _shares(blank_unweighted),
                "weighted_counts": blank_weighted,
                "weighted_shares": _shares(blank_weighted),
            },
        },
        "ue_tti_view": {
            "unit": "deduplicated_ue_tti_modal_over_17_local_prgs",
            "tie_policy": MIXED_INCONCLUSIVE,
            "weight": "sum_remaining_bytes_at_expiry_of_active_formal_packets",
            "unweighted_counts": ue_tti_unweighted,
            "unweighted_shares": _shares(ue_tti_unweighted),
            "weighted_counts": ue_tti_weighted,
            "weighted_shares": _shares(ue_tti_weighted),
            "dominance": _dominance_payload(ue_tti_dominance),
        },
        "cell_prg_exposure": {
            "interpretation": "exposure attribution only; not a per-PRG causal contribution",
            "rows": cell_prg_rows,
        },
        "formal_conclusion": conclusion,
    }

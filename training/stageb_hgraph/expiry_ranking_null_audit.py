"""Offline candidate-cardinality null audit for CTBS expiry exposure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Mapping, Sequence


PROTOCOL_ID = "seed42-ctbs-expiry-ranking-null-audit-v1"
TRACE_FIRST_TTI = 602
MEASURED_TTI_FIRST = 1001
MEASURED_TTI_LAST = 5000
MIN_MATCH_COVERAGE = 0.90


@dataclass(frozen=True)
class PacketRecord:
    packet_id: int
    ue: int
    arrival_tti: int
    terminal_tti: int
    outcome: str
    packet_size_bytes: int
    remaining_bytes_at_terminal: int


@dataclass(frozen=True)
class MatchResult:
    pairs: tuple[tuple[PacketRecord, PacketRecord], ...]
    coverage: float


@dataclass(frozen=True)
class SelectionOpportunity:
    target_ue: int
    tti: int
    local_prg: int
    valid_nonblank_count: int
    selected_ue: int
    target_rank: int
    chosen_target_logit_margin: float | None
    hol_delay_ms: float | None
    ttl_slack_ms: float | None


def active_ue_tti_union(packets: Sequence[PacketRecord]) -> set[tuple[int, int]]:
    """Return the packet-lifecycle union, deduplicated by target UE and TTI."""
    active: set[tuple[int, int]] = set()
    for packet in packets:
        if packet.terminal_tti < packet.arrival_tti:
            raise ValueError("packet terminal TTI precedes arrival")
        active.update(
            (packet.ue, tti)
            for tti in range(packet.arrival_tti, packet.terminal_tti + 1)
        )
    return active


def match_delivered_controls(
    expired: Sequence[PacketRecord],
    delivered: Sequence[PacketRecord],
) -> MatchResult:
    """Greedy same-UE nearest-arrival matching without replacement."""
    available = list(delivered)
    pairs: list[tuple[PacketRecord, PacketRecord]] = []
    for target in sorted(expired, key=lambda row: (row.ue, row.arrival_tti, row.packet_id)):
        candidates = [row for row in available if row.ue == target.ue]
        if not candidates:
            continue
        control = min(
            candidates,
            key=lambda row: (abs(row.arrival_tti - target.arrival_tti), row.packet_id),
        )
        available.remove(control)
        pairs.append((target, control))
    coverage = 1.0 if not expired else len(pairs) / len(expired)
    return MatchResult(pairs=tuple(pairs), coverage=float(coverage))


def ttl_slack_bin(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "missing"
    value = float(value)
    if value < 0.0:
        return "negative"
    if value <= 20.0:
        return "ge_0_le_20_ms"
    if value <= 50.0:
        return "gt_20_le_50_ms"
    if value <= 100.0:
        return "gt_50_le_100_ms"
    return "gt_100_ms"


def normalized_ue_tti_byte_efficiency(
    *,
    scheduled_bytes: float,
    successful_bytes: float,
    selected_prg_count: int,
) -> dict[str, float] | None:
    if selected_prg_count < 0:
        raise ValueError("selected_prg_count must be nonnegative")
    if selected_prg_count == 0:
        return None
    if scheduled_bytes < 0.0 or successful_bytes < 0.0:
        raise ValueError("byte counts must be nonnegative")
    return {
        "scheduled_bytes_per_selected_prg": float(scheduled_bytes / selected_prg_count),
        "successful_bytes_per_selected_prg": float(successful_bytes / selected_prg_count),
    }


def _candidate_index(decision: Mapping[str, object], target_ue: int) -> int | None:
    ue_ids = [int(value) for value in decision["candidate_ue_ids"]]
    present = [bool(value) for value in decision["candidate_present"]]
    matches = [
        index for index, (ue, is_present) in enumerate(zip(ue_ids, present))
        if is_present and ue == target_ue
    ]
    if len(matches) > 1:
        raise ValueError("target UE is duplicated in a candidate row")
    return None if not matches else matches[0]


def extract_target_opportunity(
    decision: Mapping[str, object],
    *,
    target_ue: int,
) -> SelectionOpportunity | None:
    """Extract one target's post-mask nonblank opportunity from a decision row."""
    target_index = _candidate_index(decision, target_ue)
    if target_index is None:
        return None
    present = [bool(value) for value in decision["candidate_present"]]
    valid = [bool(value) for value in decision["candidate_valid"]]
    logits = [float(value) for value in decision["candidate_logits"]]
    ue_ids = [int(value) for value in decision["candidate_ue_ids"]]
    if not (len(present) == len(valid) == len(logits) == len(ue_ids)):
        raise ValueError("candidate arrays have inconsistent lengths")
    if not valid[target_index]:
        return None
    valid_indices = [
        index for index, (is_present, is_valid) in enumerate(zip(present, valid))
        if is_present and is_valid
    ]
    if target_index not in valid_indices or not valid_indices:
        raise ValueError("valid target is absent from the post-mask nonblank set")
    target_logit = logits[target_index]
    target_rank = 1 + sum(logits[index] > target_logit for index in valid_indices)
    selected_ue = int(decision["selected_ue"])
    margin: float | None = None
    if selected_ue >= 0:
        selected_matches = [
            index for index in valid_indices if ue_ids[index] == selected_ue
        ]
        if len(selected_matches) != 1:
            raise ValueError("selected nonblank UE is absent from the valid candidate set")
        margin = float(logits[selected_matches[0]] - target_logit)
    ttl_values = decision.get("ttl_slack_ms")
    hol_values = decision.get("hol_delay_ms")
    ttl = None if ttl_values is None else float(ttl_values[target_index])
    hol = None if hol_values is None else float(hol_values[target_index])
    return SelectionOpportunity(
        target_ue=target_ue,
        tti=int(decision["tti"]),
        local_prg=int(decision["local_prg"]),
        valid_nonblank_count=len(valid_indices),
        selected_ue=selected_ue,
        target_rank=int(target_rank),
        chosen_target_logit_margin=margin,
        hol_delay_ms=hol,
        ttl_slack_ms=ttl,
    )


def _numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    data = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(data),
        "mean": None if not data else float(mean(data)),
        "median": None if not data else float(median(data)),
        "min": None if not data else float(min(data)),
        "max": None if not data else float(max(data)),
    }


def summarize_selection_opportunities(
    rows: Sequence[SelectionOpportunity],
) -> dict[str, object]:
    nonblank = [row for row in rows if row.selected_ue >= 0]
    expected = sum(1.0 / row.valid_nonblank_count for row in nonblank)
    selected_target = sum(row.selected_ue == row.target_ue for row in nonblank)
    ratio = None if expected == 0.0 else float(selected_target / expected)
    ttl_groups: dict[str, list[SelectionOpportunity]] = defaultdict(list)
    for row in rows:
        ttl_groups[ttl_slack_bin(row.ttl_slack_ms)].append(row)
    bin_order = (
        "gt_100_ms",
        "gt_50_le_100_ms",
        "gt_20_le_50_ms",
        "ge_0_le_20_ms",
        "negative",
        "missing",
    )
    ttl_summary: dict[str, object] = {}
    for name in bin_order:
        group = ttl_groups.get(name, [])
        group_nonblank = [row for row in group if row.selected_ue >= 0]
        group_expected = sum(1.0 / row.valid_nonblank_count for row in group_nonblank)
        group_selected = sum(row.selected_ue == row.target_ue for row in group_nonblank)
        ttl_summary[name] = {
            "target_valid_decision_count": len(group),
            "nonblank_opportunity_count": len(group_nonblank),
            "blank_excluded_count": len(group) - len(group_nonblank),
            "selected_target_count": group_selected,
            "uniform_expected_selected_count": float(group_expected),
            "selection_ratio_to_uniform": (
                None if group_expected == 0.0 else float(group_selected / group_expected)
            ),
            "target_rank": _numeric_summary(row.target_rank for row in group),
        }
    return {
        "status": "zero_denominator" if expected == 0.0 else "ok",
        "unit": "deduplicated_target_ue_x_tti_x_local_prg",
        "valid_count_definition": (
            "post-mask valid nonblank UE count; every included record contains the target UE"
        ),
        "target_valid_decision_count": len(rows),
        "nonblank_opportunity_count": len(nonblank),
        "blank_excluded_count": len(rows) - len(nonblank),
        "selected_target_count": selected_target,
        "uniform_expected_selected_count": float(expected),
        "selection_ratio_to_uniform": ratio,
        "post_mask_valid_nonblank_ue_count": _numeric_summary(
            row.valid_nonblank_count for row in nonblank
        ),
        "target_rank": _numeric_summary(row.target_rank for row in rows),
        "chosen_minus_target_logit_margin": _numeric_summary(
            row.chosen_target_logit_margin
            for row in nonblank
            if row.chosen_target_logit_margin is not None
        ),
        "hol_delay_ms": _numeric_summary(
            row.hol_delay_ms for row in rows if row.hol_delay_ms is not None
        ),
        "ttl_slack_bins": ttl_summary,
    }


def classify_underselection(
    *,
    expired_ratio: float | None,
    matched_control_ratio: float | None,
    full_delivered_ratio: float | None,
    match_coverage: float = 1.0,
) -> dict[str, object]:
    if match_coverage < MIN_MATCH_COVERAGE:
        return {
            "matched_control_classification": "not_evaluated",
            "sensitivity_disagreement": False,
            "formal_conclusion": "insufficient_match_coverage",
            "expired_to_matched_control_ratio": None,
        }
    if expired_ratio is None or matched_control_ratio is None:
        return {
            "matched_control_classification": "not_evaluated",
            "sensitivity_disagreement": False,
            "formal_conclusion": "insufficient_opportunity_support",
            "expired_to_matched_control_ratio": None,
        }
    direction = (
        matched_control_ratio > 0.0
        and expired_ratio < 1.0
        and expired_ratio < matched_control_ratio
    )
    relative = (
        None
        if expired_ratio is None or matched_control_ratio in (None, 0.0)
        else float(expired_ratio / matched_control_ratio)
    )
    if direction and relative is not None and relative <= 0.90:
        raw = "material_underselection"
    elif direction:
        raw = "weak_directional_underselection"
    else:
        raw = "underselection_not_supported"
    disagreement = bool(
        raw != "underselection_not_supported"
        and full_delivered_ratio is not None
        and expired_ratio is not None
        and expired_ratio > full_delivered_ratio
    )
    return {
        "matched_control_classification": raw,
        "sensitivity_disagreement": disagreement,
        "formal_conclusion": "sensitivity_disagreement" if disagreement else raw,
        "expired_to_matched_control_ratio": relative,
    }


def classify_urgency_response(rank_means: Sequence[float | None]) -> str:
    """Classify rank response as actor-time TTL slack declines."""
    observed = [float(value) for value in rank_means if value is not None]
    if len(observed) < 2:
        return "insufficient_ttl_bin_support"
    deltas = [right - left for left, right in zip(observed, observed[1:])]
    if all(delta <= 0.0 for delta in deltas) and any(delta < 0.0 for delta in deltas):
        return "monotonic_rank_improvement"
    if all(delta >= 0.0 for delta in deltas):
        return "rank_not_improved_as_ttl_declines"
    return "mixed_nonmonotonic_urgency_response"


def _read_packets(path: Path) -> list[PacketRecord]:
    required = {
        "ue_id",
        "packet_id",
        "arrival_tti",
        "packet_size_bytes",
        "remaining_bytes_at_terminal",
        "terminal_tti",
        "outcome",
    }
    packets: list[PacketRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("packet outcome CSV lacks lifecycle fields")
        for row in reader:
            packets.append(
                PacketRecord(
                    packet_id=int(row["packet_id"]),
                    ue=int(row["ue_id"]),
                    arrival_tti=int(row["arrival_tti"]),
                    terminal_tti=int(row["terminal_tti"]),
                    outcome=str(row["outcome"]),
                    packet_size_bytes=int(row["packet_size_bytes"]),
                    remaining_bytes_at_terminal=int(row["remaining_bytes_at_terminal"]),
                )
            )
    return packets


def _read_topology(path: Path) -> dict[int, int]:
    serving: dict[int, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["node_type"] == "ue":
                serving[int(row["ue_id"])] = int(row["serving_cell_id"])
    if not serving:
        raise ValueError("topology has no UE rows")
    return serving


def derive_runtime_association(
    decisions: Iterable[Mapping[str, object]],
) -> dict[int, int]:
    """Recover the actor's runtime association from candidate-row ownership."""
    cells_by_ue: dict[int, set[int]] = defaultdict(set)
    for decision in decisions:
        cell = int(decision["serving_cell"])
        ue_ids = [int(value) for value in decision["candidate_ue_ids"]]
        present = [bool(value) for value in decision["candidate_present"]]
        if len(ue_ids) != len(present):
            raise ValueError("candidate identity/presence lengths differ")
        for ue, is_present in zip(ue_ids, present):
            if is_present:
                if ue < 0:
                    raise ValueError("present runtime candidate has a negative UE identity")
                cells_by_ue[ue].add(cell)
    multiple = {ue: sorted(cells) for ue, cells in cells_by_ue.items() if len(cells) != 1}
    if multiple:
        raise ValueError(f"UE appears in multiple runtime serving cells: {multiple}")
    return {ue: next(iter(cells)) for ue, cells in sorted(cells_by_ue.items())}


def _by_tti(active: set[tuple[int, int]]) -> dict[int, set[int]]:
    output: dict[int, set[int]] = defaultdict(set)
    for ue, tti in active:
        output[tti].add(ue)
    return output


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _service_efficiency_summary(
    active: set[tuple[int, int]],
    service_by_ue_tti: Mapping[tuple[int, int], tuple[int, float, float]],
) -> dict[str, object]:
    scheduled_efficiency: list[float] = []
    successful_efficiency: list[float] = []
    selected_counts: list[int] = []
    zero_selected = 0
    missing = 0
    for key in sorted(active):
        service = service_by_ue_tti.get(key)
        if service is None:
            missing += 1
            continue
        selected_count, scheduled, successful = service
        normalized = normalized_ue_tti_byte_efficiency(
            scheduled_bytes=scheduled,
            successful_bytes=successful,
            selected_prg_count=selected_count,
        )
        if normalized is None:
            zero_selected += 1
            continue
        selected_counts.append(selected_count)
        scheduled_efficiency.append(normalized["scheduled_bytes_per_selected_prg"])
        successful_efficiency.append(normalized["successful_bytes_per_selected_prg"])
    return {
        "name": "normalized UE-TTI byte efficiency",
        "unit": "UE-TTI TB-level bytes divided by selected PRG count",
        "active_ue_tti_count": len(active),
        "source_available_count": len(active) - missing,
        "source_missing_count": missing,
        "zero_selected_prg_ue_tti_count": zero_selected,
        "efficiency_sample_count": len(selected_counts),
        "selected_prg_count": _numeric_summary(selected_counts),
        "scheduled_bytes_per_selected_prg": _numeric_summary(scheduled_efficiency),
        "successful_bytes_per_selected_prg": _numeric_summary(successful_efficiency),
        "limitation": (
            "The simulator reports UE-TTI/TB bytes, not true per-PRG bytes; division by selected "
            "PRG count is a normalization and does not attribute bytes to individual PRGs."
        ),
    }


def run_offline_audit(*, source_dir: Path, out_dir: Path) -> dict[str, object]:
    source_dir = source_dir.resolve()
    out_dir = out_dir.resolve()
    actor_dir = source_dir / "actor"
    required_paths = {
        "source_gate": source_dir / "audit_report_gate_v2.json",
        "decisions": actor_dir / "blank_expiry_decision_rows.jsonl",
        "lifecycles": actor_dir / "blank_expiry_packet_lifecycle_rows.jsonl",
        "packets": actor_dir / "packet_outcome_samples.csv",
        "topology": actor_dir / "ue_topology.csv",
        "eval_summary": actor_dir / "eval_summary.json",
    }
    for name, path in required_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    source_gate = json.loads(required_paths["source_gate"].read_text(encoding="utf-8"))
    if source_gate.get("status") != "PASS":
        raise ValueError("source full-horizon attribution Gate did not pass")
    eval_summary = json.loads(required_paths["eval_summary"].read_text(encoding="utf-8"))
    topology_generator_cell = _read_topology(required_paths["topology"])
    serving_cell = derive_runtime_association(_read_jsonl(required_paths["decisions"]))
    if set(serving_cell) != set(topology_generator_cell):
        raise ValueError("runtime association does not cover the complete topology UE set")
    packets = _read_packets(required_paths["packets"])
    formal_packets = [
        row for row in packets
        if MEASURED_TTI_FIRST <= row.terminal_tti <= MEASURED_TTI_LAST
    ]
    expired = [row for row in formal_packets if row.outcome == "expired"]
    if not expired:
        raise ValueError("null audit requires formal expired packets")
    if any(row.arrival_tti < TRACE_FIRST_TTI for row in expired):
        raise ValueError("expired packet lifecycle precedes the frozen trace")
    target_ues = sorted({row.ue for row in expired})
    same_ue_delivered = [
        row for row in formal_packets
        if row.outcome == "delivered"
        and row.ue in target_ues
        and row.arrival_tti >= TRACE_FIRST_TTI
    ]
    matches: list[tuple[PacketRecord, PacketRecord]] = []
    for ue in target_ues:
        result = match_delivered_controls(
            [row for row in expired if row.ue == ue],
            [row for row in same_ue_delivered if row.ue == ue],
        )
        matches.extend(result.pairs)
    match_coverage = len(matches) / len(expired)
    cohorts = {
        "all_expired": expired,
        "matched_expired": [left for left, _ in matches],
        "matched_delivered": [right for _, right in matches],
        "full_same_ue_delivered": same_ue_delivered,
    }
    active = {name: active_ue_tti_union(rows) for name, rows in cohorts.items()}
    active_by_tti = {name: _by_tti(keys) for name, keys in active.items()}
    opportunities: dict[str, list[SelectionOpportunity]] = {
        name: [] for name in cohorts
    }
    candidate_coverage = {
        name: {"target_not_present_count": 0, "target_hard_masked_count": 0}
        for name in cohorts
    }
    expected_association_counts = eval_summary.get("association_candidate_counts")
    if not isinstance(expected_association_counts, list) or not expected_association_counts:
        raise ValueError("evaluation summary lacks runtime association counts")
    n_cell = len(expected_association_counts)
    observed_association_counts = [
        sum(owner == cell for owner in serving_cell.values()) for cell in range(n_cell)
    ]
    if observed_association_counts != expected_association_counts:
        raise ValueError(
            "runtime association reconstructed from candidates disagrees with summary: "
            f"observed={observed_association_counts} expected={expected_association_counts}"
        )
    cell_rows = [
        {
            "decision_count": 0,
            "selected_nonblank_count": 0,
            "blank_count": 0,
            "forced_blank_count": 0,
            "valid_nonblank_sum": 0,
            "backlog_by_tti": [],
        }
        for _ in range(n_cell)
    ]
    for decision in _read_jsonl(required_paths["decisions"]):
        tti = int(decision["tti"])
        cell = int(decision["serving_cell"])
        local_prg = int(decision["local_prg"])
        present = [bool(value) for value in decision["candidate_present"]]
        valid = [bool(value) for value in decision["candidate_valid"]]
        valid_count = sum(a and b for a, b in zip(present, valid))
        if MEASURED_TTI_FIRST <= tti <= MEASURED_TTI_LAST:
            stats = cell_rows[cell]
            stats["decision_count"] += 1
            stats["valid_nonblank_sum"] += valid_count
            if int(decision["selected_ue"]) < 0:
                stats["blank_count"] += 1
                if valid_count == 0:
                    stats["forced_blank_count"] += 1
            else:
                stats["selected_nonblank_count"] += 1
            if local_prg == 0:
                backlog = sum(
                    float(value)
                    for value, is_present in zip(
                        decision["initial_backlog_bytes"], present
                    )
                    if is_present
                )
                stats["backlog_by_tti"].append(backlog)
        for name in cohorts:
            for target_ue in active_by_tti[name].get(tti, ()):
                if serving_cell[target_ue] != cell:
                    continue
                target_index = _candidate_index(decision, target_ue)
                if target_index is None:
                    candidate_coverage[name]["target_not_present_count"] += 1
                    continue
                if not valid[target_index]:
                    candidate_coverage[name]["target_hard_masked_count"] += 1
                    continue
                opportunity = extract_target_opportunity(
                    decision, target_ue=target_ue
                )
                if opportunity is None:
                    raise RuntimeError("valid target opportunity was not extracted")
                opportunities[name].append(opportunity)
    selection = {
        name: {
            **summarize_selection_opportunities(rows),
            "packet_count": len(cohorts[name]),
            "active_ue_tti_union_count": len(active[name]),
            **candidate_coverage[name],
        }
        for name, rows in opportunities.items()
    }
    underselection = classify_underselection(
        expired_ratio=selection["matched_expired"]["selection_ratio_to_uniform"],
        matched_control_ratio=selection["matched_delivered"]["selection_ratio_to_uniform"],
        full_delivered_ratio=selection["full_same_ue_delivered"]["selection_ratio_to_uniform"],
        match_coverage=match_coverage,
    )
    rank_bin_order = (
        "gt_100_ms",
        "gt_50_le_100_ms",
        "gt_20_le_50_ms",
        "ge_0_le_20_ms",
    )
    rank_by_ttl_bin = {
        name: selection["all_expired"]["ttl_slack_bins"][name]["target_rank"]["mean"]
        for name in rank_bin_order
    }
    early_rank = rank_by_ttl_bin["gt_100_ms"]
    urgent_rank = rank_by_ttl_bin["ge_0_le_20_ms"]
    urgency = {
        "bin_order_from_less_to_more_urgent": list(rank_bin_order),
        "target_rank_mean_by_ttl_bin": rank_by_ttl_bin,
        "early_target_rank_mean": early_rank,
        "urgent_target_rank_mean": urgent_rank,
        "urgent_minus_early_rank_mean": (
            None if early_rank is None or urgent_rank is None else float(urgent_rank - early_rank)
        ),
        "classification": classify_urgency_response(list(rank_by_ttl_bin.values())),
    }
    service_by_ue_tti: dict[tuple[int, int], tuple[int, float, float]] = {}
    for row in _read_jsonl(required_paths["lifecycles"]):
        key = (int(row["ue"]), int(row["tti"]))
        value = (
            len(row["selected_local_prgs"]),
            float(row["scheduled_bytes"]),
            float(row["successful_bytes"]),
        )
        prior = service_by_ue_tti.get(key)
        if prior is not None and prior != value:
            raise ValueError("overlapping packet lifecycle rows disagree on UE-TTI service")
        service_by_ue_tti[key] = value
    service_efficiency = {
        name: _service_efficiency_summary(keys, service_by_ue_tti)
        for name, keys in active.items()
    }
    matched_overlap = active["matched_expired"] & active["matched_delivered"]
    matched_union = active["matched_expired"] | active["matched_delivered"]
    expired_by_ue: dict[int, dict[str, int]] = defaultdict(
        lambda: {"packet_count": 0, "remaining_bytes_at_expiry": 0}
    )
    for row in expired:
        expired_by_ue[row.ue]["packet_count"] += 1
        expired_by_ue[row.ue]["remaining_bytes_at_expiry"] += row.remaining_bytes_at_terminal
    association_count = {
        cell: sum(owner == cell for owner in serving_cell.values())
        for cell in range(n_cell)
    }
    packet_terminal_by_cell = {
        cell: {
            "delivered_packet_count": 0,
            "delivered_packet_bytes_mac_terminal_proxy": 0,
            "expired_packet_count": 0,
            "remaining_bytes_at_expiry": 0,
        }
        for cell in range(n_cell)
    }
    for row in formal_packets:
        cell = serving_cell[row.ue]
        stats = packet_terminal_by_cell[cell]
        if row.outcome == "delivered":
            stats["delivered_packet_count"] += 1
            stats["delivered_packet_bytes_mac_terminal_proxy"] += row.packet_size_bytes
        elif row.outcome == "expired":
            stats["expired_packet_count"] += 1
            stats["remaining_bytes_at_expiry"] += row.remaining_bytes_at_terminal
    cell_summary: list[dict[str, object]] = []
    for cell, stats in enumerate(cell_rows):
        backlog = stats.pop("backlog_by_tti")
        decision_count = int(stats["decision_count"])
        first = backlog[:100]
        last = backlog[-100:]
        cell_summary.append(
            {
                "cell": cell,
                "association_ue_count": association_count[cell],
                "theoretical_offered_bytes": association_count[cell] * 3000 * 4000,
                "offered_bytes_source": (
                    "frozen scenario: 3000-byte packet per UE per TTI over 4000 measured TTIs"
                ),
                **stats,
                "prg_utilization_ratio": (
                    None if decision_count == 0 else stats["selected_nonblank_count"] / decision_count
                ),
                "blank_ratio": None if decision_count == 0 else stats["blank_count"] / decision_count,
                "mean_post_mask_valid_nonblank_ues": (
                    None if decision_count == 0 else stats["valid_nonblank_sum"] / decision_count
                ),
                "mean_initial_backlog_bytes": None if not backlog else float(mean(backlog)),
                "first_100_tti_mean_initial_backlog_bytes": None if not first else float(mean(first)),
                "last_100_tti_mean_initial_backlog_bytes": None if not last else float(mean(last)),
                "last_minus_first_100_tti_backlog_bytes": (
                    None if not first or not last else float(mean(last) - mean(first))
                ),
                **packet_terminal_by_cell[cell],
                "phy_successful_bytes": None,
                "mean_tbs_bytes": None,
            }
        )
    expiry_rows = [
        {
            "ue": ue,
            "serving_cell": serving_cell[ue],
            **expired_by_ue[ue],
        }
        for ue in sorted(expired_by_ue)
    ]
    expired_ue_count_by_cell = {
        str(cell): sum(row["serving_cell"] == cell for row in expiry_rows)
        for cell in range(n_cell)
    }
    report = {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "source_protocol_id": source_gate.get("protocol_id"),
        "source_gate_status": source_gate.get("status"),
        "source_dir": str(source_dir),
        "simulation_executed": False,
        "actor_modified": False,
        "measured_terminal_window": [MEASURED_TTI_FIRST, MEASURED_TTI_LAST],
        "decision_trace_window": [TRACE_FIRST_TTI, MEASURED_TTI_LAST],
        "runtime_association": {
            "source": "actor candidate-row ownership across the complete decision trace",
            "counts": observed_association_counts,
            "ue_to_cell": {str(ue): cell for ue, cell in sorted(serving_cell.items())},
            "topology_generator_cell_mismatch_count": sum(
                topology_generator_cell[ue] != cell for ue, cell in serving_cell.items()
            ),
            "topology_generator_cell_is_not_used_for_runtime_association": True,
        },
        "expiry_distribution": {
            "expired_packet_count": len(expired),
            "expired_ue_count": len(expiry_rows),
            "target_ues": target_ues,
            "expired_ue_count_by_cell": expired_ue_count_by_cell,
            "per_ue": expiry_rows,
        },
        "matching": {
            "policy": "same UE; nearest arrival TTI; ties by packet identity; no replacement",
            "expired_packet_count": len(expired),
            "matched_pair_count": len(matches),
            "coverage": float(match_coverage),
            "minimum_coverage": MIN_MATCH_COVERAGE,
            "coverage_passed": match_coverage >= MIN_MATCH_COVERAGE,
            "active_ue_tti_overlap": {
                "intersection_count": len(matched_overlap),
                "union_count": len(matched_union),
                "jaccard": None if not matched_union else len(matched_overlap) / len(matched_union),
                "interpretation": (
                    "High overlap means expired and delivered packet cohorts reuse many of the same "
                    "target-UE scheduling decisions and are not independent trajectory controls."
                ),
            },
        },
        "selection_null": selection,
        "underselection": underselection,
        "urgency_response": urgency,
        "service_efficiency": service_efficiency,
        "cell_load_backlog_expiry_association": {
            "interpretation_scope": "load/backlog/expiry association only",
            "cells": cell_summary,
            "capacity_classification_forbidden": True,
        },
        "data_limitations": [
            "Blank decisions are excluded from the primary selection null and reported separately.",
            "UE-TTI scheduled/successful bytes are TB-level totals; normalized efficiency is not true per-PRG attribution.",
            "Complete per-cell PHY successful bytes and TBS were not persisted; PHY capacity or cell capacity failure cannot be classified.",
            "MAC packet terminal delivered bytes are not ACK-based PHY goodput.",
            "Packet lifecycle cohorts overlap in time; the primary unit is the deduplicated target UE x TTI x PRG union.",
            "Matched expired and delivered packet lifecycles may share target-UE TTIs; overlap is reported and limits control discrimination.",
            "Measured-window theoretical offered bytes and terminal-window MAC packet bytes have different boundary semantics because terminal packets may have arrived during warmup.",
        ],
        "formal_conclusion": underselection["formal_conclusion"],
        "input_provenance": {
            name: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            for name, path in required_paths.items()
        },
        "frozen_checkpoint_state_dict_sha256": eval_summary.get("actor_state_dict_sha256"),
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (out_dir / "matched_packet_pairs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("ue", "expired_packet_id", "expired_arrival_tti", "delivered_packet_id", "delivered_arrival_tti", "absolute_arrival_tti_distance"))
        for left, right in matches:
            writer.writerow((left.ue, left.packet_id, left.arrival_tti, right.packet_id, right.arrival_tti, abs(left.arrival_tti - right.arrival_tti)))
    with (out_dir / "expiry_by_ue.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("ue", "serving_cell", "packet_count", "remaining_bytes_at_expiry"))
        writer.writeheader()
        writer.writerows(expiry_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_offline_audit(source_dir=args.source_dir, out_dir=args.out_dir)
    print(json.dumps({
        "status": report["status"],
        "formal_conclusion": report["formal_conclusion"],
        "output": str((args.out_dir / "audit_report.json").resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

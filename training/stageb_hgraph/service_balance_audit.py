from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Mapping, Sequence


PROTOCOL_ID = "seed42-cell2-ue2-service-balance-read-only-audit-v1"
TRACE_FIRST_TTI = 602
TRACE_LAST_TTI = 5000
MEASURED_FIRST_TTI = 1001
MEASURED_LAST_TTI = 5000
WINDOW_TTI = 100
MATERIAL_RATIO_MAX = 0.90
MATERIAL_WINDOW_COUNT = 30

_INT_FIELDS = (
    "tti",
    "ue_id",
    "serving_cell",
    "offered_bytes",
    "accepted_bytes",
    "rejected_bytes",
    "expired_bytes",
    "previous_post_queue_bytes",
    "pre_queue_bytes",
    "scheduled_dequeue_bytes",
    "post_queue_bytes",
    "successful_bytes",
    "failed_bytes",
    "demand_valid_prg_count",
    "actual_prg_count",
    "tbs_capacity_bytes",
    "mcs",
    "layer",
    "tb_tx_count",
    "tb_err_count",
)
_FLOAT_FIELDS = ("hol_delay_ms", "ttl_slack_ms")
_SERVICE_FIELDS = set(_INT_FIELDS + _FLOAT_FIELDS + ("actual_prg_mask",))


def _integral(raw: str, name: str) -> int:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or value != math.floor(value):
        raise ValueError(f"{name} must be a finite integer")
    return int(value)


def canonical_offered_packet_stream(path: Path | str) -> dict[str, object]:
    source = Path(path)
    required = ("ue_id", "packet_id", "arrival_tti", "packet_size_bytes", "ttl_tti")
    rows: list[tuple[int, int, int, int, int]] = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(required).issubset(reader.fieldnames):
            raise ValueError("offered packet stream schema is incomplete")
        for raw in reader:
            row = tuple(_integral(raw[name], name) for name in required)
            if row[0] < 0 or row[1] <= 0 or row[2] <= 0 or row[3] < 0 or row[4] <= 0:
                raise ValueError("offered packet stream contains invalid values")
            rows.append(row)
    packet_ids = [row[1] for row in rows]
    if len(packet_ids) != len(set(packet_ids)):
        raise ValueError("offered packet IDs must be unique")
    rows.sort(key=lambda row: (row[1], row[0], row[2], row[3], row[4]))
    digest = hashlib.sha256()
    for row in rows:
        digest.update(struct.pack("<qqqqq", *row))
    return {
        "packet_count": len(rows),
        "offered_bytes": sum(row[3] for row in rows),
        "sha256": digest.hexdigest(),
        "hash_contract": "sorted (packet_id,ue_id,arrival_tti,size_bytes,ttl_tti) signed-int64 tuples",
    }


def load_service_balance_trace(
    path: Path | str,
    *,
    trace_first_tti: int = TRACE_FIRST_TTI,
    trace_last_tti: int = TRACE_LAST_TTI,
) -> list[dict[str, object]]:
    source = Path(path)
    rows: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not _SERVICE_FIELDS.issubset(reader.fieldnames):
            missing = sorted(_SERVICE_FIELDS.difference(reader.fieldnames or ()))
            raise ValueError(f"service-balance trace schema is incomplete: {missing}")
        for raw in reader:
            row: dict[str, object] = {name: _integral(raw[name], name) for name in _INT_FIELDS}
            for name in _FLOAT_FIELDS:
                value = float(raw[name])
                if not math.isfinite(value):
                    raise ValueError(f"{name} must be finite")
                row[name] = value
            row["actual_prg_mask"] = str(raw["actual_prg_mask"])
            tti = int(row["tti"])
            ue = int(row["ue_id"])
            if not int(trace_first_tti) <= tti <= int(trace_last_tti):
                raise ValueError(f"service-balance TTI outside frozen trace interval: {tti}")
            identity = (tti, ue)
            if identity in seen:
                raise ValueError(f"duplicate service-balance row: {identity}")
            seen.add(identity)
            nonnegative = (
                "offered_bytes",
                "accepted_bytes",
                "rejected_bytes",
                "expired_bytes",
                "previous_post_queue_bytes",
                "pre_queue_bytes",
                "scheduled_dequeue_bytes",
                "post_queue_bytes",
                "successful_bytes",
                "failed_bytes",
                "demand_valid_prg_count",
                "actual_prg_count",
                "tbs_capacity_bytes",
                "tb_tx_count",
                "tb_err_count",
            )
            if any(int(row[name]) < 0 for name in nonnegative):
                raise ValueError("service-balance byte/count fields must be non-negative")
            if int(row["offered_bytes"]) != int(row["accepted_bytes"]) + int(row["rejected_bytes"]):
                raise ValueError("offered bytes must equal accepted plus rejected bytes")
            if int(row["pre_queue_bytes"]) != (
                int(row["previous_post_queue_bytes"])
                + int(row["accepted_bytes"])
                - int(row["expired_bytes"])
            ):
                raise ValueError("pre-queue conservation failed")
            if int(row["post_queue_bytes"]) != (
                int(row["pre_queue_bytes"]) - int(row["scheduled_dequeue_bytes"])
            ):
                raise ValueError("post-queue conservation failed")
            if int(row["successful_bytes"]) + int(row["failed_bytes"]) != int(
                row["scheduled_dequeue_bytes"]
            ):
                raise ValueError("successful plus failed bytes must equal scheduled dequeue")
            rows.append(row)
    rows.sort(key=lambda row: (int(row["tti"]), int(row["ue_id"])))
    return rows


def validate_trace_coverage(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_ue_ids: Sequence[int] | range,
    trace_first_tti: int = TRACE_FIRST_TTI,
    trace_last_tti: int = TRACE_LAST_TTI,
) -> dict[str, object]:
    expected_ues = {int(ue) for ue in expected_ue_ids}
    expected = {
        (tti, ue)
        for tti in range(int(trace_first_tti), int(trace_last_tti) + 1)
        for ue in expected_ues
    }
    actual = {(int(row["tti"]), int(row["ue_id"])) for row in rows}
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise ValueError(
            "service-balance trace coverage failed: "
            f"first_missing={missing[:1]} first_extra={extra[:1]}"
        )
    serving_cell_by_ue: dict[int, int] = {}
    for row in rows:
        ue = int(row["ue_id"])
        cell = int(row["serving_cell"])
        previous = serving_cell_by_ue.setdefault(ue, cell)
        if previous != cell:
            raise ValueError(f"serving-cell association changed for UE {ue}")
    cell_ids = sorted(set(serving_cell_by_ue.values()))
    if cell_ids != list(range(len(cell_ids))):
        raise ValueError("serving-cell association IDs are not contiguous from zero")
    return {
        "row_count": len(rows),
        "tti_first": int(trace_first_tti),
        "tti_last": int(trace_last_tti),
        "ue_count": len(expected_ues),
        "association_counts": [
            sum(cell == cell_id for cell in serving_cell_by_ue.values()) for cell_id in cell_ids
        ],
        "serving_cell_by_ue": {
            str(ue): serving_cell_by_ue[ue] for ue in sorted(serving_cell_by_ue)
        },
    }


def load_terminal_outcomes(
    path: Path | str,
    *,
    ue_id: int,
    measured_first_tti: int = MEASURED_FIRST_TTI,
    measured_last_tti: int = MEASURED_LAST_TTI,
    trace_first_tti: int = TRACE_FIRST_TTI,
) -> dict[str, object]:
    required = (
        "ue_id",
        "packet_id",
        "arrival_tti",
        "packet_size_bytes",
        "remaining_bytes_at_terminal",
        "terminal_tti",
        "outcome",
        "system_time_ms",
    )
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(required).issubset(reader.fieldnames):
            raise ValueError("packet outcome schema is incomplete")
        for raw in reader:
            if _integral(raw["ue_id"], "ue_id") != int(ue_id):
                continue
            terminal_tti = _integral(raw["terminal_tti"], "terminal_tti")
            if not int(measured_first_tti) <= terminal_tti <= int(measured_last_tti):
                continue
            arrival_tti = _integral(raw["arrival_tti"], "arrival_tti")
            if arrival_tti < int(trace_first_tti):
                raise ValueError(
                    "formal terminal packet lifecycle predates service-balance trace"
                )
            outcome = str(raw["outcome"])
            if outcome not in {"delivered", "expired"}:
                raise ValueError(f"unknown packet outcome: {outcome}")
            system_time_ms = float(raw["system_time_ms"])
            if not math.isfinite(system_time_ms) or system_time_ms < 0.0:
                raise ValueError("packet outcome system time must be finite and non-negative")
            rows.append(
                {
                    "packet_id": _integral(raw["packet_id"], "packet_id"),
                    "arrival_tti": arrival_tti,
                    "packet_size_bytes": _integral(raw["packet_size_bytes"], "packet_size_bytes"),
                    "remaining_bytes_at_terminal": _integral(
                        raw["remaining_bytes_at_terminal"], "remaining_bytes_at_terminal"
                    ),
                    "terminal_tti": terminal_tti,
                    "outcome": outcome,
                    "system_time_ms": system_time_ms,
                }
            )
    packet_ids = [int(row["packet_id"]) for row in rows]
    if len(packet_ids) != len(set(packet_ids)):
        raise ValueError("formal packet outcomes contain duplicate packet IDs")
    rows.sort(key=lambda row: (int(row["terminal_tti"]), int(row["packet_id"])))
    expired = [row for row in rows if row["outcome"] == "expired"]
    delivered = [row for row in rows if row["outcome"] == "delivered"]
    return {
        "terminal_packet_count": len(rows),
        "delivered_packet_count": len(delivered),
        "expired_packet_count": len(expired),
        "delivered_packet_bytes": sum(int(row["packet_size_bytes"]) for row in delivered),
        "expired_bytes": sum(int(row["remaining_bytes_at_terminal"]) for row in expired),
        "terminal_tti_first": min((int(row["terminal_tti"]) for row in rows), default=None),
        "terminal_tti_last": max((int(row["terminal_tti"]) for row in rows), default=None),
        "expired_packet_ids": [int(row["packet_id"]) for row in expired],
    }


def summarize_fixed_windows(
    rows: Sequence[Mapping[str, object]],
    *,
    ue_id: int,
    measured_first_tti: int = MEASURED_FIRST_TTI,
    measured_last_tti: int = MEASURED_LAST_TTI,
    window_tti: int = WINDOW_TTI,
) -> list[dict[str, float | int]]:
    span = int(measured_last_tti) - int(measured_first_tti) + 1
    if span <= 0 or span % int(window_tti) != 0:
        raise ValueError("measured interval must contain complete fixed windows")
    by_tti = {
        int(row["tti"]): row
        for row in rows
        if int(row["ue_id"]) == int(ue_id)
        and int(measured_first_tti) <= int(row["tti"]) <= int(measured_last_tti)
    }
    expected = set(range(int(measured_first_tti), int(measured_last_tti) + 1))
    if set(by_tti) != expected:
        missing = sorted(expected.difference(by_tti))
        raise ValueError(f"target UE measured trace is incomplete: first_missing={missing[:1]}")
    sum_fields = (
        "offered_bytes",
        "accepted_bytes",
        "rejected_bytes",
        "expired_bytes",
        "scheduled_dequeue_bytes",
        "successful_bytes",
        "failed_bytes",
        "demand_valid_prg_count",
        "actual_prg_count",
        "tbs_capacity_bytes",
        "tb_tx_count",
        "tb_err_count",
    )
    windows: list[dict[str, float | int]] = []
    for first in range(int(measured_first_tti), int(measured_last_tti) + 1, int(window_tti)):
        last = first + int(window_tti) - 1
        block = [by_tti[tti] for tti in range(first, last + 1)]
        window: dict[str, float | int] = {
            "tti_first": first,
            "tti_last": last,
            **{name: sum(int(row[name]) for row in block) for name in sum_fields},
        }
        window["urgent_50ms_scheduled_dequeue_bytes"] = sum(
            int(row["scheduled_dequeue_bytes"])
            for row in block
            if 0.0 <= float(row["ttl_slack_ms"]) <= 50.0
        )
        window["critical_20ms_scheduled_dequeue_bytes"] = sum(
            int(row["scheduled_dequeue_bytes"])
            for row in block
            if 0.0 <= float(row["ttl_slack_ms"]) <= 20.0
        )
        actual_prgs = int(window["actual_prg_count"])
        scheduled = int(window["scheduled_dequeue_bytes"])
        window["conditional_phy_capacity_bytes_per_prg"] = (
            float(window["tbs_capacity_bytes"]) / actual_prgs if actual_prgs > 0 else 0.0
        )
        window["reliability_successful_per_scheduled"] = (
            float(window["successful_bytes"]) / scheduled if scheduled > 0 else 0.0
        )
        window["arrival_service_gap_bytes"] = int(window["accepted_bytes"]) - scheduled
        window["queue_drift_bytes"] = (
            int(block[-1]["post_queue_bytes"])
            - int(block[0]["previous_post_queue_bytes"])
        )
        if int(window["queue_drift_bytes"]) != (
            int(window["accepted_bytes"])
            - int(window["expired_bytes"])
            - scheduled
        ):
            raise ValueError("window queue conservation failed")
        windows.append(window)
    return windows


def _overall_value(windows: Sequence[Mapping[str, float | int]], metric: str) -> float:
    if metric == "conditional_phy_capacity_bytes_per_prg":
        prgs = sum(float(window["actual_prg_count"]) for window in windows)
        return (
            sum(float(window["tbs_capacity_bytes"]) for window in windows) / prgs
            if prgs > 0.0
            else 0.0
        )
    if metric == "reliability_successful_per_scheduled":
        scheduled = sum(float(window["scheduled_dequeue_bytes"]) for window in windows)
        return (
            sum(float(window["successful_bytes"]) for window in windows) / scheduled
            if scheduled > 0.0
            else 0.0
        )
    return sum(float(window[metric]) for window in windows)


def material_deficit(
    pfq_windows: Sequence[Mapping[str, float | int]],
    ctbs_windows: Sequence[Mapping[str, float | int]],
    metric: str,
) -> dict[str, object]:
    if len(pfq_windows) != 40 or len(ctbs_windows) != 40:
        raise ValueError("material deficit requires exactly forty paired windows")
    pfq_total = _overall_value(pfq_windows, metric)
    ctbs_total = _overall_value(ctbs_windows, metric)
    ratio = ctbs_total / pfq_total if pfq_total > 0.0 else None
    lower_windows = sum(
        float(ctbs[metric]) < float(pfq[metric])
        for pfq, ctbs in zip(pfq_windows, ctbs_windows)
    )
    return {
        "metric": metric,
        "pfq_overall": pfq_total,
        "ctbs_overall": ctbs_total,
        "ctbs_over_pfq": ratio,
        "ctbs_lower_window_count": lower_windows,
        "window_count": 40,
        "material": (
            ratio is not None
            and ratio <= MATERIAL_RATIO_MAX
            and lower_windows >= MATERIAL_WINDOW_COUNT
        ),
        "ratio_threshold": MATERIAL_RATIO_MAX,
        "direction_window_threshold": MATERIAL_WINDOW_COUNT,
    }


def classify_service_balance(
    *,
    accepted_arrival_deficit: bool,
    scheduled_service_deficit: bool,
    allocation_deficit: bool,
    conditional_tbs_deficit: bool,
    deadline_timing_deficit: bool,
) -> str:
    if accepted_arrival_deficit:
        return "admission_difference"
    if scheduled_service_deficit and allocation_deficit:
        return "allocation_deficit"
    if scheduled_service_deficit and conditional_tbs_deficit:
        return "conditional_tbs_deficit"
    if not scheduled_service_deficit and deadline_timing_deficit:
        return "deadline_timing_deficit"
    return "inconclusive"


def _population_summary(
    rows: Sequence[Mapping[str, object]], *, serving_cell: int | None = None
) -> dict[str, object]:
    selected = [
        row
        for row in rows
        if MEASURED_FIRST_TTI <= int(row["tti"]) <= MEASURED_LAST_TTI
        and (serving_cell is None or int(row["serving_cell"]) == int(serving_cell))
    ]
    totals = {
        name: sum(int(row[name]) for row in selected)
        for name in (
            "offered_bytes",
            "accepted_bytes",
            "rejected_bytes",
            "expired_bytes",
            "scheduled_dequeue_bytes",
            "successful_bytes",
            "failed_bytes",
            "demand_valid_prg_count",
            "actual_prg_count",
            "tbs_capacity_bytes",
            "tb_tx_count",
            "tb_err_count",
        )
    }
    actual_prgs = totals["actual_prg_count"]
    scheduled = totals["scheduled_dequeue_bytes"]
    return {
        "serving_cell": serving_cell,
        "row_count": len(selected),
        **totals,
        "conditional_phy_capacity_bytes_per_prg": (
            totals["tbs_capacity_bytes"] / actual_prgs if actual_prgs else None
        ),
        "reliability_successful_per_scheduled": (
            totals["successful_bytes"] / scheduled if scheduled else None
        ),
        "final_post_queue_bytes": sum(
            int(row["post_queue_bytes"])
            for row in selected
            if int(row["tti"]) == MEASURED_LAST_TTI
        ),
    }


def _load_target_policy_validity(
    path: Path | str, *, ue_id: int
) -> dict[str, object]:
    valid = 0
    selected = 0
    row_count = 0
    by_tti: dict[int, dict[str, int]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            tti = int(raw["tti"])
            if not MEASURED_FIRST_TTI <= tti <= MEASURED_LAST_TTI:
                continue
            if int(raw["serving_cell"]) != 2:
                continue
            ids = [int(value) for value in raw["candidate_ue_ids"]]
            present = [bool(value) for value in raw["candidate_present"]]
            candidate_valid = [bool(value) for value in raw["candidate_valid"]]
            target_columns = [
                index
                for index, candidate in enumerate(ids)
                if candidate == int(ue_id) and present[index]
            ]
            if len(target_columns) > 1:
                raise ValueError(
                    f"target UE candidate multiplicity exceeds one at TTI {tti}: {target_columns}"
                )
            is_valid = bool(target_columns) and candidate_valid[target_columns[0]]
            is_selected = int(raw["selected_ue"]) == int(ue_id)
            if is_selected and not is_valid:
                raise ValueError(f"target UE selected while invalid at TTI {tti}")
            valid += int(is_valid)
            selected += int(is_selected)
            row_count += 1
            tti_row = by_tti.setdefault(tti, {"valid_prgs": 0, "selected_prgs": 0})
            tti_row["valid_prgs"] += int(is_valid)
            tti_row["selected_prgs"] += int(is_selected)
    if row_count != 4000 * 17 or len(by_tti) != 4000:
        raise ValueError(
            "target UE policy-validity trace is incomplete: "
            f"rows={row_count} ttis={len(by_tti)}"
        )
    return {
        "post_mask_valid_prg_exposures": valid,
        "selected_prg_exposures": selected,
        "validity_rate": valid / row_count,
        "selection_given_valid_rate": selected / valid if valid else None,
        "per_tti": [
            {"tti": tti, **by_tti[tti]} for tti in sorted(by_tti)
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired seed42 Cell2/UE2 service-balance audit")
    parser.add_argument("--pfq-service-trace", required=True)
    parser.add_argument("--ctbs-service-trace", required=True)
    parser.add_argument("--pfq-offered-packets", required=True)
    parser.add_argument("--ctbs-offered-packets", required=True)
    parser.add_argument("--pfq-outcomes", required=True)
    parser.add_argument("--ctbs-outcomes", required=True)
    parser.add_argument("--ctbs-decision-rows", default="")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pfq_offer = canonical_offered_packet_stream(args.pfq_offered_packets)
    ctbs_offer = canonical_offered_packet_stream(args.ctbs_offered_packets)
    pfq_rows = load_service_balance_trace(args.pfq_service_trace)
    ctbs_rows = load_service_balance_trace(args.ctbs_service_trace)
    pfq_coverage = validate_trace_coverage(pfq_rows, expected_ue_ids=range(36))
    ctbs_coverage = validate_trace_coverage(ctbs_rows, expected_ue_ids=range(36))
    expected_association = [11, 11, 14]
    if pfq_coverage["association_counts"] != expected_association:
        raise ValueError(f"PFQ association mismatch: {pfq_coverage['association_counts']}")
    if ctbs_coverage["association_counts"] != expected_association:
        raise ValueError(f"CTBS association mismatch: {ctbs_coverage['association_counts']}")
    if pfq_coverage["serving_cell_by_ue"] != ctbs_coverage["serving_cell_by_ue"]:
        raise ValueError("paired serving-cell association differs")
    pfq_windows = summarize_fixed_windows(pfq_rows, ue_id=2)
    ctbs_windows = summarize_fixed_windows(ctbs_rows, ue_id=2)
    deficits = {
        metric: material_deficit(pfq_windows, ctbs_windows, metric)
        for metric in (
            "accepted_bytes",
            "scheduled_dequeue_bytes",
            "actual_prg_count",
            "tbs_capacity_bytes",
            "conditional_phy_capacity_bytes_per_prg",
            "reliability_successful_per_scheduled",
            "urgent_50ms_scheduled_dequeue_bytes",
            "critical_20ms_scheduled_dequeue_bytes",
        )
    }
    pfq_outcomes = load_terminal_outcomes(args.pfq_outcomes, ue_id=2)
    ctbs_outcomes = load_terminal_outcomes(args.ctbs_outcomes, ue_id=2)
    deadline_timing_deficit = bool(
        not deficits["scheduled_dequeue_bytes"]["material"]
        and ctbs_outcomes["expired_bytes"] > pfq_outcomes["expired_bytes"]
        and deficits["urgent_50ms_scheduled_dequeue_bytes"]["material"]
    )
    classification = classify_service_balance(
        accepted_arrival_deficit=bool(deficits["accepted_bytes"]["material"]),
        scheduled_service_deficit=bool(deficits["scheduled_dequeue_bytes"]["material"]),
        allocation_deficit=bool(deficits["actual_prg_count"]["material"]),
        conditional_tbs_deficit=bool(
            deficits["conditional_phy_capacity_bytes_per_prg"]["material"]
        ),
        deadline_timing_deficit=deadline_timing_deficit,
    )
    paired = pfq_offer == ctbs_offer
    queue_pressure = {
        "pfq_arrival_service_gap_bytes": sum(
            int(window["arrival_service_gap_bytes"]) for window in pfq_windows
        ),
        "ctbs_arrival_service_gap_bytes": sum(
            int(window["arrival_service_gap_bytes"]) for window in ctbs_windows
        ),
        "ctbs_higher_gap_window_count": sum(
            int(ctbs["arrival_service_gap_bytes"]) > int(pfq["arrival_service_gap_bytes"])
            for pfq, ctbs in zip(pfq_windows, ctbs_windows)
        ),
    }
    policy_validity = (
        _load_target_policy_validity(args.ctbs_decision_rows, ue_id=2)
        if args.ctbs_decision_rows
        else None
    )
    runtime_gates = {
        "offered_packet_stream_exact": paired,
        "trace_coverage_exact": True,
        "association_counts_exact": True,
        "queue_conservation_exact": True,
        "scheduled_reliability_conservation_exact": True,
    }
    payload = {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if all(runtime_gates.values()) else "FAIL",
        "classification": classification if paired else "offered_stream_mismatch",
        "current_research_state": (
            "B4-CTBS causal and goodput feasible, but the seed42 tail-reliability mechanism "
            "is not yet closed."
        ),
        "runtime_gates": runtime_gates,
        "pairing": {"exact": paired, "pfq": pfq_offer, "ctbs": ctbs_offer},
        "coverage": {"pfq": pfq_coverage, "ctbs": ctbs_coverage},
        "deficits": deficits,
        "queue_pressure": queue_pressure,
        "terminal_outcomes_ue2": {"pfq": pfq_outcomes, "ctbs": ctbs_outcomes},
        "ctbs_policy_validity_ue2": policy_validity,
        "population_association_only": {
            "scope_note": (
                "Cell summaries establish load/backlog/expiry association only; they do not "
                "establish PHY-capacity or cell-capacity failure."
            ),
            "pfq_cluster": _population_summary(pfq_rows),
            "ctbs_cluster": _population_summary(ctbs_rows),
            "pfq_cell2": _population_summary(pfq_rows, serving_cell=2),
            "ctbs_cell2": _population_summary(ctbs_rows, serving_cell=2),
        },
        "pfq_windows": pfq_windows,
        "ctbs_windows": ctbs_windows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not paired:
        raise SystemExit("offered packet streams differ")


if __name__ == "__main__":
    main()

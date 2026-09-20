"""Fail-closed Phase-A audit for the frozen seed51 CTBS blank guard."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .ctbs_non_starvation_guard import GuardReason
from .independent_contract import state_dict_sha256
from .independent_gate import ctbs_formal_evaluation_gate
from .online_protocol import VERSION
from .service_balance_audit import canonical_offered_packet_stream


PROTOCOL_ID = "iter8-ctbs-legal-non-starvation-guard-seed51-v1"
CHECKPOINT_FILE_SHA256 = (
    "94a2e40f0c0a56632e0c0ca63af4fa54d894d35aea38c2dd34afc0cfeadf2d7a"
)
ACTOR_STATE_DICT_SHA256 = (
    "6039c59e92ecda5ed75731fef86b71f567fe0c227142d9ed267434934b1e56be"
)
SIMULATOR_SHA256 = (
    "4bdfee54c3203a808f42fd85367f6c46457bbf20dd1314e0c5855d2f07400a10"
)
CTBS_CONFIG_SHA256 = (
    "be277403fbfff0eb6359b94e3b000344deac00bfe5acbbd91004060dd3336631"
)

CHECKPOINT_RELATIVE_PATH = (
    "output/pfq_independent_reuse/formal/"
    "deadline_dimensionless_features_seed45_v1/train_attempt2/ppo_actor_last.pt"
)
SIMULATOR_RELATIVE_PATH = (
    "build.reuse_actor/cuMAC/examples/multiCellSchedulerUeSelection/"
    "multiCellSchedulerUeSelection"
)
FEATURE_SOURCE_SHA256 = {
    "cuMAC/examples/onlineTrainBridge/DeadlineServiceFeatures.h": (
        "4a353601a8378888d089e34d3f04225770ae757ac07d5bc4108533a1a384975a"
    ),
    "cuMAC/examples/onlineTrainBridge/OnlineObservationExtrasBuilder.h": (
        "ccf58c6c07e5ab65da373870d7d615b8048611508e4972910c4a4871744f876b"
    ),
    "training/stageb_hgraph/feature_snapshot.py": (
        "e86976c6208ff6e23e350d1480beaac011d694870d4a68e4e6fb9f7077aa7f7f"
    ),
}
IMPLEMENTATION_SOURCE_PATHS = (
    "training/stageb_hgraph/ctbs_non_starvation_guard.py",
    "training/stageb_hgraph/independent_policy.py",
    "training/stageb_hgraph/eval_checkpoint_online.py",
    "training/stageb_hgraph/ctbs_non_starvation_guard_audit.py",
)
UE_FEATURE_NAMES = (
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
FROZEN_OFFERED_FINGERPRINT = {
    "packet_count": 180016,
    "offered_bytes": 540048000,
    "sha256": "4abe2d4ac1606b3a635cc4eaf3db506055b04fc2c8e56dc4542782995037a404",
}
FROZEN_GUARD_CONFIG = {
    "minimum_log1p_drain_pressure": math.log(2.0),
    "maximum_scheduled_dequeue_adequacy": 1.0,
    "minimum_ttl_normalized_service_age": 0.25,
    "maximum_overrides_per_cell_per_tti": 1,
    "override_nonblank": False,
    "require_positive_ctbs_capacity_delta": True,
}
FROZEN_PERFORMANCE_THRESHOLDS = {
    "actor_over_pfq_goodput_min": 0.99,
    "actor_minus_pfq_bler_max": 0.002,
    "actor_minus_pfq_expiry_max": 0.0005,
}

GUARD_TRACE_FIELDS = (
    "tti",
    "cell_index",
    "local_prg_index",
    "blank_class_index",
    "actor_choice",
    "executed_choice",
    "guard_override",
    "selected_class",
    "selected_ue",
    "candidate_was_valid",
    "log1p_drain_pressure",
    "scheduled_dequeue_adequacy",
    "ttl_normalized_service_age",
    "ctbs_capacity_before_bytes",
    "ctbs_capacity_after_bytes",
    "reason",
)
_INT_FIELDS = (
    "tti",
    "cell_index",
    "local_prg_index",
    "blank_class_index",
    "actor_choice",
    "executed_choice",
    "selected_class",
    "selected_ue",
    "ctbs_capacity_before_bytes",
    "ctbs_capacity_after_bytes",
)
_FLOAT_FIELDS = (
    "log1p_drain_pressure",
    "scheduled_dequeue_adequacy",
    "ttl_normalized_service_age",
)
_BOOL_FIELDS = ("guard_override", "candidate_was_valid")
KNOWN_REASON_NAMES = frozenset(reason.name.lower() for reason in GuardReason)
MANDATORY_GUARD_INVARIANT_CHECKS = frozenset(
    {
        "complete_decision_coverage",
        "unique_decision_coordinates",
        "canonical_decision_coordinates",
        "measured_decision_coverage",
        "nonblank_override_count_zero",
        "max_one_override_per_cell_tti",
        "pressure_strictly_above_log2",
        "adequacy_strictly_below_one",
        "age_at_least_quarter_ttl",
        "override_candidate_valid",
        "positive_ctbs_capacity_delta",
        "reason_codes_known",
        "unchanged_without_override",
        "override_executes_nonblank",
        "override_reason_consistent",
    }
)
REFERENCE_ARTIFACT_ROLES = (
    "pfq_metrics",
    "pfq_starvation",
    "pfq_outcomes",
    "pfq_offered_packets",
    "unguarded_starvation",
)
STARVATION_AUDIT_PROTOCOL_ID = "dimensionless-service-starvation-audit-v2"
_INTEGER_TEXT = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_SHA256_TEXT = re.compile(r"[0-9a-f]{64}\Z")


def _strict_integer(raw: object, field: str) -> int:
    if not isinstance(raw, str) or _INTEGER_TEXT.fullmatch(raw) is None:
        raise ValueError(f"{field} must be an exact base-10 integer")
    return int(raw)


def _strict_boolean(raw: object, field: str) -> bool:
    if raw == "True":
        return True
    if raw == "False":
        return False
    raise ValueError(f"{field} must be exactly True or False")


def _strict_float(raw: object, field: str) -> float:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{field} must be a finite floating-point value")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{field} must be a finite floating-point value") from exc
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _validate_expected_shape(
    expected_tti_first: int,
    expected_tti_last: int,
    expected_n_cell: int,
    expected_n_prg: int,
) -> None:
    if expected_tti_first <= 0 or expected_tti_last < expected_tti_first:
        raise ValueError("expected TTI interval is invalid")
    if expected_n_cell <= 0 or expected_n_prg <= 0:
        raise ValueError("expected cell and PRG counts must be positive")


def load_guard_trace(
    path: Path | str,
    *,
    expected_tti_first: int = 1,
    expected_tti_last: int = 5000,
    expected_n_cell: int = 3,
    expected_n_prg: int = 17,
) -> list[dict[str, object]]:
    """Load the exact Task-3 schema and require canonical full coverage."""
    _validate_expected_shape(
        expected_tti_first,
        expected_tti_last,
        expected_n_cell,
        expected_n_prg,
    )
    source = Path(path)
    parsed: list[dict[str, object]] = []
    seen: set[tuple[int, int, int]] = set()
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != GUARD_TRACE_FIELDS:
            raise ValueError(
                "guard trace schema must equal the fixed 16-column contract"
            )
        for line_number, raw in enumerate(reader, start=2):
            if None in raw or set(raw) != set(GUARD_TRACE_FIELDS):
                raise ValueError(f"guard trace row {line_number} has extra or missing fields")
            row: dict[str, object] = {
                field: _strict_integer(raw[field], field) for field in _INT_FIELDS
            }
            row.update(
                {field: _strict_boolean(raw[field], field) for field in _BOOL_FIELDS}
            )
            row.update({field: _strict_float(raw[field], field) for field in _FLOAT_FIELDS})
            reason = raw["reason"]
            if not isinstance(reason, str) or not reason:
                raise ValueError("reason must be a non-empty string")
            row["reason"] = reason

            tti = int(row["tti"])
            cell = int(row["cell_index"])
            local_prg = int(row["local_prg_index"])
            if not expected_tti_first <= tti <= expected_tti_last:
                raise ValueError(f"guard trace TTI is outside the expected horizon: {tti}")
            if not 0 <= cell < expected_n_cell:
                raise ValueError(f"guard trace cell index is noncanonical: {cell}")
            if not 0 <= local_prg < expected_n_prg:
                raise ValueError(f"guard trace local PRG index is noncanonical: {local_prg}")
            if min(
                int(row["blank_class_index"]),
                int(row["actor_choice"]),
                int(row["executed_choice"]),
                int(row["ctbs_capacity_before_bytes"]),
                int(row["ctbs_capacity_after_bytes"]),
            ) < 0:
                raise ValueError("guard trace class/capacity fields must be non-negative")
            if int(row["selected_class"]) < -1 or int(row["selected_ue"]) < -1:
                raise ValueError("guard trace selected identifiers must be at least -1")
            key = (tti, cell, local_prg)
            if key in seen:
                raise ValueError(f"duplicate guard trace decision coordinate: {key}")
            seen.add(key)
            parsed.append(row)

    expected_count = (
        (expected_tti_last - expected_tti_first + 1)
        * expected_n_cell
        * expected_n_prg
    )
    if len(parsed) != expected_count:
        raise ValueError(
            "guard trace canonical full coverage failed: "
            f"rows={len(parsed)} expected={expected_count}"
        )
    parsed.sort(
        key=lambda row: (
            int(row["tti"]),
            int(row["local_prg_index"]),
            int(row["cell_index"]),
        )
    )
    return parsed


def _typed_audit_row(raw: Mapping[str, object]) -> dict[str, object]:
    if set(raw) != set(GUARD_TRACE_FIELDS):
        raise ValueError("guard audit row does not have the fixed 16 fields")
    row: dict[str, object] = {}
    for field in _INT_FIELDS:
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must be an integer in parsed guard rows")
        row[field] = value
    for field in _BOOL_FIELDS:
        value = raw[field]
        if not isinstance(value, bool):
            raise ValueError(f"{field} must be boolean in parsed guard rows")
        row[field] = value
    for field in _FLOAT_FIELDS:
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be numeric in parsed guard rows")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{field} must be finite in parsed guard rows")
        row[field] = converted
    if not isinstance(raw["reason"], str):
        raise ValueError("reason must be a string in parsed guard rows")
    row["reason"] = raw["reason"]
    return row


def _string_counter(counter: Counter[int] | Counter[str]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=str)}


def audit_guard_trace(
    rows: Sequence[Mapping[str, object]],
    *,
    measured_tti_first: int,
    measured_tti_last: int,
    expected_tti_first: int,
    expected_tti_last: int,
    expected_n_cell: int,
    expected_n_prg: int,
) -> dict[str, object]:
    """Calculate structural and legal-override invariants directly from rows."""
    _validate_expected_shape(
        expected_tti_first,
        expected_tti_last,
        expected_n_cell,
        expected_n_prg,
    )
    if not expected_tti_first <= measured_tti_first <= measured_tti_last <= expected_tti_last:
        raise ValueError("measured TTI interval must lie inside the expected horizon")
    parsed = [_typed_audit_row(row) for row in rows]
    expected_count = (
        (expected_tti_last - expected_tti_first + 1)
        * expected_n_cell
        * expected_n_prg
    )
    expected_measured_count = (
        (measured_tti_last - measured_tti_first + 1)
        * expected_n_cell
        * expected_n_prg
    )
    keys = [
        (int(row["tti"]), int(row["cell_index"]), int(row["local_prg_index"]))
        for row in parsed
    ]
    canonical = all(
        expected_tti_first <= tti <= expected_tti_last
        and 0 <= cell < expected_n_cell
        and 0 <= local_prg < expected_n_prg
        for tti, cell, local_prg in keys
    )
    measured = [
        row
        for row in parsed
        if measured_tti_first <= int(row["tti"]) <= measured_tti_last
    ]
    overrides = [row for row in parsed if bool(row["guard_override"])]
    measured_overrides = [row for row in measured if bool(row["guard_override"])]
    overrides_per_cell_tti = Counter(
        (int(row["tti"]), int(row["cell_index"])) for row in overrides
    )
    reasons = Counter(str(row["reason"]) for row in parsed)
    measured_reasons = Counter(str(row["reason"]) for row in measured)

    checks = {
        "complete_decision_coverage": len(parsed) == expected_count,
        "unique_decision_coordinates": len(keys) == len(set(keys)),
        "canonical_decision_coordinates": canonical,
        "measured_decision_coverage": len(measured) == expected_measured_count,
        "nonblank_override_count_zero": all(
            not bool(row["guard_override"])
            or int(row["actor_choice"]) == int(row["blank_class_index"])
            for row in parsed
        ),
        "max_one_override_per_cell_tti": max(
            overrides_per_cell_tti.values(), default=0
        )
        <= 1,
        "pressure_strictly_above_log2": all(
            not bool(row["guard_override"])
            or float(row["log1p_drain_pressure"]) > math.log(2.0)
            for row in parsed
        ),
        "adequacy_strictly_below_one": all(
            not bool(row["guard_override"])
            or float(row["scheduled_dequeue_adequacy"]) < 1.0
            for row in parsed
        ),
        "age_at_least_quarter_ttl": all(
            not bool(row["guard_override"])
            or float(row["ttl_normalized_service_age"]) >= 0.25
            for row in parsed
        ),
        "override_candidate_valid": all(
            not bool(row["guard_override"]) or bool(row["candidate_was_valid"])
            for row in parsed
        ),
        "positive_ctbs_capacity_delta": all(
            not bool(row["guard_override"])
            or int(row["ctbs_capacity_after_bytes"])
            > int(row["ctbs_capacity_before_bytes"])
            for row in parsed
        ),
        "reason_codes_known": all(str(row["reason"]) in KNOWN_REASON_NAMES for row in parsed),
        "unchanged_without_override": all(
            bool(row["guard_override"])
            or int(row["actor_choice"]) == int(row["executed_choice"])
            for row in parsed
        ),
        "override_executes_nonblank": all(
            not bool(row["guard_override"])
            or (
                int(row["executed_choice"]) != int(row["blank_class_index"])
                and int(row["executed_choice"]) == int(row["selected_class"])
                and int(row["selected_ue"]) >= 0
            )
            for row in parsed
        ),
        "override_reason_consistent": all(
            bool(row["guard_override"]) == (str(row["reason"]) == "overridden")
            for row in parsed
        ),
    }

    per_cell = Counter(int(row["cell_index"]) for row in overrides)
    per_ue = Counter(int(row["selected_ue"]) for row in overrides)
    measured_per_cell = Counter(int(row["cell_index"]) for row in measured_overrides)
    measured_per_ue = Counter(int(row["selected_ue"]) for row in measured_overrides)
    result: dict[str, object] = {
        "valid": True,
        "row_count": len(parsed),
        "measured_row_count": len(measured),
        "expected_row_count": expected_count,
        "expected_measured_row_count": expected_measured_count,
        "override_count": len(overrides),
        "measured_override_count": len(measured_overrides),
        "override_count_by_cell": _string_counter(per_cell),
        "override_count_by_ue": _string_counter(per_ue),
        "measured_override_count_by_cell": _string_counter(measured_per_cell),
        "measured_override_count_by_ue": _string_counter(measured_per_ue),
        "reason_counts": _string_counter(reasons),
        "measured_reason_counts": _string_counter(measured_reasons),
        "maximum_overrides_per_cell_per_tti": max(
            overrides_per_cell_tti.values(), default=0
        ),
        "checks": checks,
    }
    for reason in (
        "actor_nonblank",
        "cell_budget_used",
        "no_pressure_candidate",
        "no_positive_ctbs_headroom",
    ):
        result[f"{reason}_count"] = int(reasons[reason])
        result[f"measured_{reason}_count"] = int(measured_reasons[reason])
    result["passed"] = all(checks.values())
    return result


def summarize_outcome_delays(
    path: Path | str, measured_tti_first: int, measured_tti_last: int
) -> dict[str, object]:
    """Summarize all terminal outcomes in the inclusive measured window."""
    if measured_tti_first <= 0 or measured_tti_last < measured_tti_first:
        raise ValueError("measured TTI interval is invalid")
    delays: list[float] = []
    terminal_ttis: list[int] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"terminal_tti", "system_time_ms"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("packet outcome schema is missing terminal_tti/system_time_ms")
        for raw in reader:
            terminal_tti = _strict_integer(raw["terminal_tti"], "terminal_tti")
            if terminal_tti <= 0:
                raise ValueError("terminal_tti must be a positive integer")
            delay = _strict_float(raw["system_time_ms"], "system_time_ms")
            if delay < 0.0:
                raise ValueError("system_time_ms must be non-negative")
            if measured_tti_first <= terminal_tti <= measured_tti_last:
                terminal_ttis.append(terminal_tti)
                delays.append(delay)
    if not delays:
        raise ValueError("empty measured outcome-delay window")
    values = np.asarray(delays, dtype=np.float64)
    return {
        "sample_count": len(delays),
        "terminal_tti_first": min(terminal_ttis),
        "terminal_tti_last": max(terminal_ttis),
        "measured_tti_first": int(measured_tti_first),
        "measured_tti_last": int(measured_tti_last),
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
    }


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _starvation_audits_complete_and_paired(
    audits: Sequence[Mapping[str, object]],
) -> bool:
    serving_maps: list[dict[str, object]] = []
    for audit in audits:
        coverage = _mapping(audit.get("trace_coverage"))
        raw_rows = audit.get("per_ue")
        if not isinstance(raw_rows, list):
            return False
        try:
            ue_ids = [int(row["ue_id"]) for row in raw_rows if isinstance(row, Mapping)]
        except (KeyError, TypeError, ValueError):
            return False
        serving = coverage.get("serving_cell_by_ue")
        if (
            len(ue_ids) != 36
            or set(ue_ids) != set(range(36))
            or audit.get("protocol_id") != STARVATION_AUDIT_PROTOCOL_ID
            or audit.get("seed") != 51
            or coverage.get("row_count") != 144000
            or coverage.get("tti_first") != 1001
            or coverage.get("tti_last") != 5000
            or coverage.get("ue_count") != 36
            or coverage.get("association_counts") != [8, 17, 11]
            or not isinstance(serving, Mapping)
            or set(serving) != {str(ue) for ue in range(36)}
        ):
            return False
        serving_maps.append(dict(serving))
    return len(serving_maps) == 3 and serving_maps[0] == serving_maps[1] == serving_maps[2]


def _per_ue(starvation: Mapping[str, object], name: str) -> dict[int, Mapping[str, object]]:
    raw_rows = starvation.get("per_ue")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"{name} starvation audit has no per_ue rows")
    rows: dict[int, Mapping[str, object]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{name} starvation per_ue row is not an object")
        ue_id = int(raw["ue_id"])
        if ue_id < 0 or ue_id in rows:
            raise ValueError(f"{name} starvation UE identities are invalid")
        for field in (
            "backlogged_tti_count",
            "scheduled_dequeue_bytes",
            "expired_bytes",
            "failed_bytes",
            "max_backlogged_zero_dequeue_streak_tti",
        ):
            value = int(raw[field])
            if value < 0:
                raise ValueError(f"{name} starvation {field} must be non-negative")
        rate = raw.get("expiry_aware_rate_mbps")
        if rate is not None and not math.isfinite(float(rate)):
            raise ValueError(f"{name} starvation rate must be finite")
        rows[ue_id] = raw
    return rows


def _target_comparison(
    guarded: Mapping[str, object], unguarded: Mapping[str, object]
) -> dict[str, object]:
    guarded_values = {
        "scheduled_dequeue_bytes": int(guarded["scheduled_dequeue_bytes"]),
        "expiry_aware_rate_mbps": float(guarded["expiry_aware_rate_mbps"]),
        "expired_bytes": int(guarded["expired_bytes"]),
        "max_backlogged_zero_dequeue_streak_tti": int(
            guarded["max_backlogged_zero_dequeue_streak_tti"]
        ),
    }
    unguarded_values = {
        "scheduled_dequeue_bytes": int(unguarded["scheduled_dequeue_bytes"]),
        "expiry_aware_rate_mbps": float(unguarded["expiry_aware_rate_mbps"]),
        "expired_bytes": int(unguarded["expired_bytes"]),
        "max_backlogged_zero_dequeue_streak_tti": int(
            unguarded["max_backlogged_zero_dequeue_streak_tti"]
        ),
    }
    checks = {
        "scheduled_dequeue_bytes_strictly_increased": guarded_values[
            "scheduled_dequeue_bytes"
        ]
        > unguarded_values["scheduled_dequeue_bytes"],
        "expiry_aware_rate_strictly_increased": guarded_values[
            "expiry_aware_rate_mbps"
        ]
        > unguarded_values["expiry_aware_rate_mbps"],
        "expired_bytes_strictly_decreased": guarded_values["expired_bytes"]
        < unguarded_values["expired_bytes"],
        "zero_dequeue_streak_strictly_decreased": guarded_values[
            "max_backlogged_zero_dequeue_streak_tti"
        ]
        < unguarded_values["max_backlogged_zero_dequeue_streak_tti"],
    }
    return {
        "guarded": guarded_values,
        "unguarded": unguarded_values,
        "delta_guarded_minus_unguarded": {
            key: guarded_values[key] - unguarded_values[key] for key in guarded_values
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _present_values_are_finite(
    values: Mapping[str, object], fields: Sequence[str]
) -> bool:
    return all(
        field not in values
        or values[field] is None
        or _finite_or_none(values[field]) is not None
        for field in fields
    )


def _rate_summary(rows: Mapping[int, Mapping[str, object]]) -> dict[str, float | None]:
    rates = [
        float(row["expiry_aware_rate_mbps"])
        for row in rows.values()
        if row.get("expiry_aware_rate_mbps") is not None
    ]
    if not rates:
        return {"macro": None, "p10": None, "minimum": None}
    values = np.asarray(rates, dtype=np.float64)
    return {
        "macro": float(np.mean(values)),
        "p10": float(np.percentile(values, 10)),
        "minimum": float(np.min(values)),
    }


def _policy_metrics(
    metrics: Mapping[str, object],
    summary: Mapping[str, object],
    starvation_rows: Mapping[int, Mapping[str, object]],
    delays: Mapping[str, object] | None,
    *,
    actor: bool,
) -> dict[str, object]:
    rates = _rate_summary(starvation_rows)
    macro_source = (
        summary.get("ue_macro_expiry_aware_rate_mbps")
        if actor
        else metrics.get("ue_macro_ea_rate_mbps")
    )
    p10_source = (
        summary.get("ue_p10_expiry_aware_rate_mbps")
        if actor
        else metrics.get("ue_p10_ea_rate_mbps")
    )
    macro_rate = _finite_or_none(macro_source)
    p10_rate = _finite_or_none(p10_source)
    return {
        "outcome_delay_ms": {
            key: _finite_or_none(_mapping(delays).get(key))
            for key in ("mean_ms", "p50_ms", "p90_ms", "p95_ms", "max_ms")
        },
        "ue_macro_expiry_aware_rate_mbps": (
            rates["macro"] if macro_source is None else macro_rate
        ),
        "ue_p10_expiry_aware_rate_mbps": (
            rates["p10"] if p10_source is None else p10_rate
        ),
        "ue_min_expiry_aware_rate_mbps": _finite_or_none(
            summary.get("ue_min_expiry_aware_rate_mbps")
        )
        if actor
        else rates["minimum"],
        "goodput_mbps": _finite_or_none(metrics.get("goodput_mbps")),
        "tb_bler": _finite_or_none(metrics.get("tb_bler")),
        "expiry_rate": _finite_or_none(metrics.get("expiry_rate")),
        "prg_utilization_ratio": _finite_or_none(
            summary.get("stats_mean_prg_utilization_ratio")
            if actor
            else metrics.get("prg_utilization_ratio_mean")
        ),
        "blank_ratio": _finite_or_none(
            summary.get("stats_mean_blank_ratio")
            if actor
            else metrics.get("blank_ratio_mean")
        ),
        "scheduled_dequeue_bytes": sum(
            int(row["scheduled_dequeue_bytes"]) for row in starvation_rows.values()
        ),
        "failed_bytes": sum(int(row["failed_bytes"]) for row in starvation_rows.values()),
        "per_ue_expired_bytes": {
            str(ue): int(starvation_rows[ue]["expired_bytes"])
            for ue in sorted(starvation_rows)
        },
    }


def _manifest_file_hashes_match(
    manifest: Mapping[str, object], observed: Mapping[str, str]
) -> bool:
    implementation = manifest.get("implementation_source_sha256")
    if not isinstance(implementation, Mapping) or set(implementation) != set(
        IMPLEMENTATION_SOURCE_PATHS
    ):
        return False
    return all(observed.get(str(path)) == digest for path, digest in implementation.items())


def _reference_artifacts_bound(
    manifest: Mapping[str, object], observed: Mapping[str, str]
) -> bool:
    paths = manifest.get("reference_artifact_paths")
    hashes = manifest.get("reference_artifact_sha256")
    expected_roles = set(REFERENCE_ARTIFACT_ROLES)
    if (
        not isinstance(paths, Mapping)
        or not isinstance(hashes, Mapping)
        or set(paths) != expected_roles
        or set(hashes) != expected_roles
    ):
        return False
    for role in REFERENCE_ARTIFACT_ROLES:
        relative = paths[role]
        digest = hashes[role]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or Path(relative).as_posix() != relative
            or any(part in {"", ".", ".."} for part in Path(relative).parts)
            or not isinstance(digest, str)
            or _SHA256_TEXT.fullmatch(digest) is None
            or observed.get(f"reference_manifest_relative_path:{role}")
            != relative
            or observed.get(f"reference_manifest_path:{role}")
            != observed.get(f"reference_actual_path:{role}")
            or observed.get(f"reference_sha256:{role}") != digest
        ):
            return False
    return True


def build_phase_a_report(
    *,
    frozen_manifest: Mapping[str, object],
    pfq_metrics: Mapping[str, object],
    actor_metrics: Mapping[str, object],
    actor_summary: Mapping[str, object],
    guard_audit: Mapping[str, object],
    offered_stream_comparison: Mapping[str, object],
    pfq_starvation: Mapping[str, object],
    unguarded_starvation: Mapping[str, object],
    actor_starvation: Mapping[str, object],
    pfq_outcome_delays: Mapping[str, object],
    actor_outcome_delays: Mapping[str, object],
    observed_file_sha256: Mapping[str, str],
    input_errors: Sequence[str] = (),
) -> dict[str, object]:
    """Combine frozen CTBS, provenance, business, and service gates."""
    policy_metric_values_finite = all(
        (
            _present_values_are_finite(
                pfq_metrics,
                (
                    "goodput_mbps",
                    "tb_bler",
                    "expiry_rate",
                    "ue_macro_ea_rate_mbps",
                    "ue_p10_ea_rate_mbps",
                    "prg_utilization_ratio_mean",
                    "blank_ratio_mean",
                ),
            ),
            _present_values_are_finite(
                actor_metrics, ("goodput_mbps", "tb_bler", "expiry_rate")
            ),
            _present_values_are_finite(
                actor_summary,
                (
                    "ue_macro_expiry_aware_rate_mbps",
                    "ue_p10_expiry_aware_rate_mbps",
                    "ue_min_expiry_aware_rate_mbps",
                    "stats_mean_prg_utilization_ratio",
                    "stats_mean_blank_ratio",
                ),
            ),
            _present_values_are_finite(
                pfq_outcome_delays,
                ("mean_ms", "p50_ms", "p90_ms", "p95_ms", "max_ms"),
            ),
            _present_values_are_finite(
                actor_outcome_delays,
                ("mean_ms", "p50_ms", "p90_ms", "p95_ms", "max_ms"),
            ),
        )
    )
    try:
        formal_gate = ctbs_formal_evaluation_gate(
            pfq_metrics,
            actor_metrics,
            actor_summary,
            actor_state_dict_sha256=ACTOR_STATE_DICT_SHA256,
            binary_sha256=SIMULATOR_SHA256,
            expected_association_counts=(8, 17, 11),
            expected_runtime_width=17,
            protocol_id="iter8_ctbs_blank_v1_seed51",
        )
    except (TypeError, ValueError, OverflowError) as exc:
        formal_gate = {
            "structural_gate": {
                "status": "FAIL",
                "passed": False,
                "checks": {},
                "error": f"{type(exc).__name__}: {exc}",
            },
            "performance_gate": {
                "status": "NOT_EVALUATED",
                "passed": False,
            },
        }

    feature_manifest = frozen_manifest.get("feature_source_sha256")
    feature_sources_exact = (
        isinstance(feature_manifest, Mapping)
        and dict(feature_manifest) == FEATURE_SOURCE_SHA256
        and all(
            observed_file_sha256.get(path) == digest
            for path, digest in FEATURE_SOURCE_SHA256.items()
        )
    )
    actor_config = _mapping(actor_summary.get("actor_config"))
    provenance_checks = {
        "no_input_errors": not input_errors,
        "policy_metric_values_finite_if_present": policy_metric_values_finite,
        "protocol_exact": frozen_manifest.get("protocol_id") == PROTOCOL_ID,
        "seed51_exact": frozen_manifest.get("seed") == 51,
        "formal_horizon_exact": frozen_manifest.get("warmup_tti") == 1000
        and frozen_manifest.get("measured_tti") == 4000
        and frozen_manifest.get("measured_tti_first") == 1001
        and frozen_manifest.get("measured_tti_last") == 5000,
        "ttl_exact": frozen_manifest.get("packet_ttl_tti") == 400
        and frozen_manifest.get("packet_ttl_ms") == 200,
        "checkpoint_file_hash_exact": frozen_manifest.get(
            "checkpoint_file_sha256"
        )
        == CHECKPOINT_FILE_SHA256
        and actor_summary.get("checkpoint_sha256") == CHECKPOINT_FILE_SHA256
        and observed_file_sha256.get("checkpoint") == CHECKPOINT_FILE_SHA256,
        "actor_state_hash_exact": frozen_manifest.get("actor_state_dict_sha256")
        == ACTOR_STATE_DICT_SHA256
        and actor_summary.get("actor_state_dict_sha256") == ACTOR_STATE_DICT_SHA256
        and observed_file_sha256.get("actor_state") == ACTOR_STATE_DICT_SHA256,
        "simulator_hash_exact": frozen_manifest.get("simulator_sha256")
        == SIMULATOR_SHA256
        and actor_summary.get("sim_binary_sha256") == SIMULATOR_SHA256
        and observed_file_sha256.get("simulator") == SIMULATOR_SHA256,
        "ctbs_config_hash_exact": frozen_manifest.get("ctbs_config_sha256")
        == CTBS_CONFIG_SHA256
        and actor_summary.get("ctbs_config_sha256") == CTBS_CONFIG_SHA256,
        "feature_source_hashes_exact": feature_sources_exact,
        "implementation_source_hashes_exact": _manifest_file_hashes_match(
            frozen_manifest, observed_file_sha256
        ),
        "feature_names_exact": frozen_manifest.get("ue_feature_names")
        == list(UE_FEATURE_NAMES),
        "feature_dimension_exact": frozen_manifest.get("ue_feature_dimension") == 16
        and actor_config.get("ue_feat_dim") == 16,
        "decode_mode_argmax": frozen_manifest.get("decode_mode") == "argmax"
        and actor_summary.get("decode_mode") == "argmax",
        "ctbs_v1_selected": frozen_manifest.get("independent_headroom_accounting")
        == "ctbs-v1"
        and actor_summary.get("independent_headroom_accounting") == "ctbs-v1",
        "guard_mode_exact": frozen_manifest.get("guard_mode") == "ctbs-blank-v1"
        and actor_summary.get("independent_non_starvation_guard")
        == "ctbs-blank-v1",
        "guard_config_exact": frozen_manifest.get("guard_config")
        == FROZEN_GUARD_CONFIG
        and actor_summary.get("non_starvation_guard_config") == FROZEN_GUARD_CONFIG,
        "runtime_topology_exact": frozen_manifest.get("runtime_candidate_width") == 17
        and frozen_manifest.get("association_candidate_counts") == [8, 17, 11],
        "repair_fill_budget_disabled": all(
            frozen_manifest.get(key) == expected
            and actor_summary.get(key) == expected
            for key, expected in (
                ("candidate_policy_mode", "independent"),
                ("slot_repair_max_per_cell", 0),
                ("safe_fill_max_per_cell", 0),
                ("candidate_blank_budget_max_per_cell", -1),
                ("candidate_budgeted_select_mode", "off"),
            )
        ),
        "training_and_additional_seed_forbidden": frozen_manifest.get(
            "training_or_adaptation_allowed"
        )
        is False
        and frozen_manifest.get("additional_seed_allowed") is False,
        "performance_thresholds_exact": frozen_manifest.get(
            "performance_thresholds"
        )
        == FROZEN_PERFORMANCE_THRESHOLDS,
        "reference_artifact_paths_and_hashes_exact": _reference_artifacts_bound(
            frozen_manifest, observed_file_sha256
        ),
        "pfq_metric_protocol_exact": pfq_metrics.get("protocol_version") == VERSION,
        "canonical_offered_stream_exact": offered_stream_comparison.get("exact")
        is True
        and offered_stream_comparison.get("pfq") == FROZEN_OFFERED_FINGERPRINT
        and offered_stream_comparison.get("actor") == FROZEN_OFFERED_FINGERPRINT,
        "starvation_audits_complete_and_paired": _starvation_audits_complete_and_paired(
            (pfq_starvation, unguarded_starvation, actor_starvation)
        ),
    }

    guard_checks = guard_audit.get("checks")
    guard_invariants_pass = (
        guard_audit.get("valid") is True
        and guard_audit.get("passed") is True
        and isinstance(guard_checks, Mapping)
        and set(guard_checks) == MANDATORY_GUARD_INVARIANT_CHECKS
        and all(guard_checks[key] is True for key in MANDATORY_GUARD_INVARIANT_CHECKS)
    )
    provenance_or_invariants_pass = (
        bool(formal_gate["structural_gate"]["passed"])
        and all(provenance_checks.values())
        and guard_invariants_pass
    )

    pfq_ues = _per_ue(pfq_starvation, "PFQ")
    unguarded_ues = _per_ue(unguarded_starvation, "unguarded actor")
    guarded_ues = _per_ue(actor_starvation, "guarded actor")
    if not {4, 6}.issubset(unguarded_ues) or not {4, 6}.issubset(guarded_ues):
        raise ValueError("UE4 and UE6 must be present in both actor starvation audits")
    target_comparison = {
        str(ue): _target_comparison(guarded_ues[ue], unguarded_ues[ue])
        for ue in (4, 6)
    }
    target_directional_pass = all(
        bool(comparison["passed"]) for comparison in target_comparison.values()
    )

    all_backlogged_checks: dict[str, dict[str, bool]] = {}
    for ue, row in sorted(guarded_ues.items()):
        if int(row["backlogged_tti_count"]) <= 0:
            continue
        rate = row.get("expiry_aware_rate_mbps")
        all_backlogged_checks[str(ue)] = {
            "positive_scheduled_dequeue": int(row["scheduled_dequeue_bytes"]) > 0,
            "positive_expiry_aware_rate": rate is not None and float(rate) > 0.0,
            "streak_below_400": int(
                row["max_backlogged_zero_dequeue_streak_tti"]
            )
            < 400,
        }
    all_backlogged_pass = bool(all_backlogged_checks) and all(
        all(checks.values()) for checks in all_backlogged_checks.values()
    )
    target_ue_gate_pass = target_directional_pass and all_backlogged_pass
    business_gate = formal_gate["performance_gate"]
    business_gate_pass = bool(business_gate.get("passed", False))

    measured_override_count = int(guard_audit.get("measured_override_count", 0))
    cell_budget_used_count = int(
        guard_audit.get(
            "measured_cell_budget_used_count",
            guard_audit.get("cell_budget_used_count", 0),
        )
    )
    no_positive_ctbs_headroom_count = int(
        guard_audit.get(
            "measured_no_positive_ctbs_headroom_count",
            guard_audit.get("no_positive_ctbs_headroom_count", 0),
        )
    )
    if not provenance_or_invariants_pass:
        classification = "invalid_experiment"
    elif measured_override_count == 0:
        classification = (
            "no_positive_ctbs_headroom"
            if no_positive_ctbs_headroom_count > 0
            else "insufficient_blank_opportunities"
        )
    elif not target_ue_gate_pass:
        classification = (
            "one_prg_budget_insufficient"
            if cell_budget_used_count > 0
            else (
                "no_positive_ctbs_headroom"
                if no_positive_ctbs_headroom_count > 0
                else "insufficient_blank_opportunities"
            )
        )
    elif not business_gate_pass:
        classification = "business_metric_cost"
    else:
        classification = "ctbs_blank_v1_seed51_pass"

    metrics = {
        "pfq": _policy_metrics(
            pfq_metrics,
            {},
            pfq_ues,
            pfq_outcome_delays,
            actor=False,
        ),
        "actor": _policy_metrics(
            actor_metrics,
            actor_summary,
            guarded_ues,
            actor_outcome_delays,
            actor=True,
        ),
        "unguarded_actor_service": _policy_metrics(
            {}, {}, unguarded_ues, None, actor=False
        ),
    }
    passed = (
        provenance_or_invariants_pass
        and measured_override_count > 0
        and business_gate_pass
        and target_directional_pass
        and all_backlogged_pass
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "classification": classification,
        "input_errors": list(input_errors),
        "provenance_or_invariants_gate": {
            "passed": provenance_or_invariants_pass,
            "provenance_checks": provenance_checks,
            "guard_invariants_pass": guard_invariants_pass,
            "formal_structural_gate": formal_gate["structural_gate"],
        },
        "business_gate": business_gate,
        "target_ue_directional_gate": {
            "passed": target_directional_pass,
            "required_ue_ids": [4, 6],
        },
        "all_backlogged_ue_gate": {
            "passed": all_backlogged_pass,
            "checks_by_ue": all_backlogged_checks,
        },
        "target_ue_gate": {"passed": target_ue_gate_pass},
        "target_ue_comparison": target_comparison,
        "metrics": metrics,
        "guard_summary": {
            key: guard_audit.get(key)
            for key in (
                "override_count",
                "measured_override_count",
                "override_count_by_cell",
                "override_count_by_ue",
                "reason_counts",
                "measured_reason_counts",
                "actor_nonblank_count",
                "cell_budget_used_count",
                "no_pressure_candidate_count",
                "no_positive_ctbs_headroom_count",
            )
        },
        "offered_stream_comparison": dict(offered_stream_comparison),
        "frozen_manifest": dict(frozen_manifest),
        "observed_file_sha256": dict(observed_file_sha256),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _actor_state_sha256(path: Path) -> str:
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    state = checkpoint.get("actor_state")
    if state is None:
        state = checkpoint.get("model_state", checkpoint)
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint actor state must be a mapping")
    return state_dict_sha256(state)


def _load_json(path: Path | str) -> dict[str, object]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return payload


def _write_json(path: Path | str, payload: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _inside_root(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ValueError(f"manifest file path must be relative: {relative}")
    if relative_path.as_posix() != relative or any(
        part in {"", ".", ".."} for part in relative_path.parts
    ):
        raise ValueError(f"manifest file path must be canonical: {relative}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"manifest file path escapes worktree root: {relative}") from exc
    return path


def _observed_hashes(
    root: Path,
    manifest: Mapping[str, object],
    reference_cli_paths: Mapping[str, str],
) -> tuple[dict[str, str], list[str]]:
    observed: dict[str, str] = {}
    errors: list[str] = []
    fixed = {
        "checkpoint": CHECKPOINT_RELATIVE_PATH,
        "simulator": SIMULATOR_RELATIVE_PATH,
        **{path: path for path in FEATURE_SOURCE_SHA256},
    }
    implementation = manifest.get("implementation_source_sha256")
    if isinstance(implementation, Mapping):
        fixed.update({str(path): str(path) for path in implementation})
    for key, relative in fixed.items():
        try:
            observed[key] = _sha256(_inside_root(root, relative))
        except (OSError, ValueError) as exc:
            errors.append(f"file hash {key}: {exc}")
    try:
        observed["actor_state"] = _actor_state_sha256(
            _inside_root(root, CHECKPOINT_RELATIVE_PATH)
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        errors.append(f"actor state hash: {exc}")
    manifest_reference_paths = manifest.get("reference_artifact_paths")
    manifest_reference_hashes = manifest.get("reference_artifact_sha256")
    if (
        not isinstance(manifest_reference_paths, Mapping)
        or not isinstance(manifest_reference_hashes, Mapping)
        or set(manifest_reference_paths) != set(REFERENCE_ARTIFACT_ROLES)
        or set(manifest_reference_hashes) != set(REFERENCE_ARTIFACT_ROLES)
        or set(reference_cli_paths) != set(REFERENCE_ARTIFACT_ROLES)
    ):
        errors.append("reference artifact manifest/CLI role schema is incomplete")
        return observed, errors
    for role in REFERENCE_ARTIFACT_ROLES:
        try:
            manifest_path = _inside_root(root, str(manifest_reference_paths[role]))
            actual_path = Path(reference_cli_paths[role]).resolve()
            actual_path.relative_to(root)
            observed[f"reference_manifest_relative_path:{role}"] = str(
                manifest_reference_paths[role]
            )
            observed[f"reference_manifest_path:{role}"] = str(manifest_path)
            observed[f"reference_actual_path:{role}"] = str(actual_path)
            observed[f"reference_sha256:{role}"] = _sha256(actual_path)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"reference artifact {role}: {exc}")
    return observed, errors


def _offered_comparison(
    pfq_path: Path | str, actor_path: Path | str
) -> dict[str, object]:
    pfq = canonical_offered_packet_stream(pfq_path)
    actor = canonical_offered_packet_stream(actor_path)
    pfq_core = {key: pfq[key] for key in FROZEN_OFFERED_FINGERPRINT}
    actor_core = {key: actor[key] for key in FROZEN_OFFERED_FINGERPRINT}
    return {
        "exact": pfq_core == actor_core == FROZEN_OFFERED_FINGERPRINT,
        "paired_streams_equal": pfq_core == actor_core,
        "frozen_expected": dict(FROZEN_OFFERED_FINGERPRINT),
        "pfq": pfq_core,
        "actor": actor_core,
        "pfq_hash_contract": pfq.get("hash_contract"),
        "actor_hash_contract": actor.get("hash_contract"),
    }


def _invalid_guard_audit(error: str) -> dict[str, object]:
    return {
        "valid": False,
        "passed": False,
        "error": error,
        "checks": {"trace_loaded_and_audited": False},
        "override_count": 0,
        "measured_override_count": 0,
        "reason_counts": {},
        "measured_reason_counts": {},
    }


def _invalid_offered(error: str) -> dict[str, object]:
    return {
        "exact": False,
        "error": error,
        "frozen_expected": dict(FROZEN_OFFERED_FINGERPRINT),
        "pfq": None,
        "actor": None,
    }


def _invalid_final_report(
    errors: Sequence[str],
    *,
    guard_audit: Mapping[str, object],
    offered: Mapping[str, object],
) -> dict[str, object]:
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "INVALID",
        "passed": False,
        "classification": "invalid_experiment",
        "input_errors": list(errors),
        "provenance_or_invariants_gate": {"passed": False},
        "business_gate": {"passed": False, "status": "NOT_EVALUATED"},
        "target_ue_directional_gate": {"passed": False},
        "all_backlogged_ue_gate": {"passed": False},
        "metrics": {
            "pfq": {},
            "actor": {},
            "unguarded_actor_service": {},
        },
        "guard_summary": dict(guard_audit),
        "offered_stream_comparison": dict(offered),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "worktree-root",
        "frozen-manifest",
        "pfq-metrics",
        "pfq-starvation",
        "pfq-outcomes",
        "pfq-offered-packets",
        "unguarded-starvation",
        "actor-summary",
        "actor-metrics",
        "actor-starvation",
        "actor-outcomes",
        "actor-offered-packets",
        "guard-trace",
        "guard-audit-out",
        "offered-comparison-out",
        "final-report-out",
    ):
        parser.add_argument(f"--{name}", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    errors: list[str] = []

    try:
        guard_rows = load_guard_trace(args.guard_trace)
        guard_audit = audit_guard_trace(
            guard_rows,
            measured_tti_first=1001,
            measured_tti_last=5000,
            expected_tti_first=1,
            expected_tti_last=5000,
            expected_n_cell=3,
            expected_n_prg=17,
        )
    except Exception as exc:  # preserve artifacts for every invalid trace failure
        errors.append(f"guard trace: {type(exc).__name__}: {exc}")
        guard_audit = _invalid_guard_audit(errors[-1])

    try:
        offered = _offered_comparison(
            args.pfq_offered_packets, args.actor_offered_packets
        )
    except Exception as exc:  # preserve artifacts for every invalid pairing failure
        errors.append(f"offered streams: {type(exc).__name__}: {exc}")
        offered = _invalid_offered(errors[-1])

    payloads: dict[str, dict[str, object]] = {}
    for name, path in (
        ("frozen_manifest", args.frozen_manifest),
        ("pfq_metrics", args.pfq_metrics),
        ("pfq_starvation", args.pfq_starvation),
        ("unguarded_starvation", args.unguarded_starvation),
        ("actor_summary", args.actor_summary),
        ("actor_metrics", args.actor_metrics),
        ("actor_starvation", args.actor_starvation),
    ):
        try:
            payloads[name] = _load_json(path)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            payloads[name] = {}

    delays: dict[str, dict[str, object]] = {}
    for name, path in (
        ("pfq", args.pfq_outcomes),
        ("actor", args.actor_outcomes),
    ):
        try:
            delays[name] = summarize_outcome_delays(path, 1001, 5000)
        except Exception as exc:
            errors.append(f"{name} outcomes: {type(exc).__name__}: {exc}")
            delays[name] = {}

    root = Path(args.worktree_root).resolve()
    observed, hash_errors = _observed_hashes(
        root,
        payloads["frozen_manifest"],
        {
            "pfq_metrics": args.pfq_metrics,
            "pfq_starvation": args.pfq_starvation,
            "pfq_outcomes": args.pfq_outcomes,
            "pfq_offered_packets": args.pfq_offered_packets,
            "unguarded_starvation": args.unguarded_starvation,
        },
    )
    errors.extend(hash_errors)
    try:
        report = build_phase_a_report(
            frozen_manifest=payloads["frozen_manifest"],
            pfq_metrics=payloads["pfq_metrics"],
            actor_metrics=payloads["actor_metrics"],
            actor_summary=payloads["actor_summary"],
            guard_audit=guard_audit,
            offered_stream_comparison=offered,
            pfq_starvation=payloads["pfq_starvation"],
            unguarded_starvation=payloads["unguarded_starvation"],
            actor_starvation=payloads["actor_starvation"],
            pfq_outcome_delays=delays["pfq"],
            actor_outcome_delays=delays["actor"],
            observed_file_sha256=observed,
            input_errors=errors,
        )
    except Exception as exc:
        errors.append(f"combined report: {type(exc).__name__}: {exc}")
        report = _invalid_final_report(
            errors, guard_audit=guard_audit, offered=offered
        )

    _write_json(args.guard_audit_out, guard_audit)
    _write_json(args.offered_comparison_out, offered)
    _write_json(args.final_report_out, report)
    if report.get("classification") == "invalid_experiment":
        return 3
    return 0 if report.get("passed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTOR_STATE_DICT_SHA256",
    "CHECKPOINT_FILE_SHA256",
    "CTBS_CONFIG_SHA256",
    "FEATURE_SOURCE_SHA256",
    "FROZEN_OFFERED_FINGERPRINT",
    "GUARD_TRACE_FIELDS",
    "IMPLEMENTATION_SOURCE_PATHS",
    "KNOWN_REASON_NAMES",
    "MANDATORY_GUARD_INVARIANT_CHECKS",
    "REFERENCE_ARTIFACT_ROLES",
    "SIMULATOR_SHA256",
    "UE_FEATURE_NAMES",
    "audit_guard_trace",
    "build_phase_a_report",
    "load_guard_trace",
    "main",
    "summarize_outcome_delays",
]

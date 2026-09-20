from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from training.stageb_hgraph.service_balance_audit import (
    load_service_balance_trace,
    validate_trace_coverage,
)


PROTOCOL_ID = "dimensionless-service-starvation-audit-v2"


def protocol_identity(seed: int) -> dict[str, object]:
    if seed <= 0:
        raise ValueError("seed must be positive")
    return {"protocol_id": PROTOCOL_ID, "seed": int(seed)}


def summarize_starvation(
    rows: Sequence[Mapping[str, object]],
    *,
    outcome_rows: Sequence[Mapping[str, object]] = (),
    streak_threshold_tti: int = 400,
) -> dict[str, object]:
    if streak_threshold_tti <= 0:
        raise ValueError("streak_threshold_tti must be positive")

    by_ue: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_ue[int(row["ue_id"])].append(row)
    outcomes_by_ue: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in outcome_rows:
        outcomes_by_ue[int(row["ue_id"])].append(row)

    per_ue: list[dict[str, object]] = []
    for ue_id in sorted(by_ue):
        ue_rows = sorted(by_ue[ue_id], key=lambda row: int(row["tti"]))
        backlogged_tti_count = 0
        positive_service_tti_count = 0
        scheduled_dequeue_bytes = 0
        actual_prg_count = 0
        successful_bytes = 0
        failed_bytes = 0
        expired_bytes = 0
        current_streak = 0
        max_streak = 0
        for row in ue_rows:
            backlog = int(row["pre_queue_bytes"])
            scheduled = int(row["scheduled_dequeue_bytes"])
            expired = int(row["expired_bytes"])
            scheduled_dequeue_bytes += scheduled
            actual_prg_count += int(row.get("actual_prg_count", 0))
            successful_bytes += int(row.get("successful_bytes", 0))
            failed_bytes += int(row.get("failed_bytes", 0))
            expired_bytes += expired
            if backlog > 0:
                backlogged_tti_count += 1
                if scheduled > 0:
                    positive_service_tti_count += 1
                    current_streak = 0
                else:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        zero_total = backlogged_tti_count > 0 and scheduled_dequeue_bytes == 0
        long_streak = max_streak >= streak_threshold_tti
        ue_outcomes = outcomes_by_ue.get(ue_id, [])
        delivered = [row for row in ue_outcomes if str(row["outcome"]) == "delivered"]
        expired = [row for row in ue_outcomes if str(row["outcome"]) == "expired"]
        outcome_delays = [float(row["system_time_ms"]) for row in ue_outcomes]
        delivered_packet_bytes = sum(int(row["packet_size_bytes"]) for row in delivered)
        outcome_time_ms = sum(outcome_delays)
        per_ue.append(
            {
                "ue_id": ue_id,
                "backlogged_tti_count": backlogged_tti_count,
                "positive_service_tti_count": positive_service_tti_count,
                "scheduled_dequeue_bytes": scheduled_dequeue_bytes,
                "actual_prg_count": actual_prg_count,
                "successful_bytes": successful_bytes,
                "failed_bytes": failed_bytes,
                "expired_bytes": expired_bytes,
                "delivered_packet_count": len(delivered),
                "expired_packet_count": len(expired),
                "expired_packet_remaining_bytes": sum(
                    int(row["remaining_bytes_at_terminal"]) for row in expired
                ),
                "outcome_delay_mean_ms": (
                    outcome_time_ms / len(outcome_delays) if outcome_delays else None
                ),
                "expiry_aware_rate_mbps": (
                    delivered_packet_bytes * 8.0 / (outcome_time_ms * 1000.0)
                    if outcome_time_ms > 0.0
                    else None
                ),
                "max_backlogged_zero_dequeue_streak_tti": max_streak,
                "starved_zero_total_scheduled": zero_total,
                "starved_long_streak": long_streak,
                "starved": zero_total or long_streak,
            }
        )

    starved_ue_ids = [int(row["ue_id"]) for row in per_ue if bool(row["starved"])]
    return {
        "streak_threshold_tti": streak_threshold_tti,
        "starved_ue_ids": starved_ue_ids,
        "guard_required": bool(starved_ue_ids),
        "passed": not starved_ue_ids,
        "per_ue": per_ue,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit measured per-UE starvation")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--service-trace", required=True)
    parser.add_argument("--measured-first-tti", type=int, required=True)
    parser.add_argument("--measured-last-tti", type=int, required=True)
    parser.add_argument("--expected-ue-count", type=int, default=36)
    parser.add_argument("--streak-threshold-tti", type=int, default=400)
    parser.add_argument("--eval-summary", default="")
    parser.add_argument("--outcome-samples", default="")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    trace_path = Path(args.service_trace).resolve()
    rows = load_service_balance_trace(
        trace_path,
        trace_first_tti=args.measured_first_tti,
        trace_last_tti=args.measured_last_tti,
    )
    coverage = validate_trace_coverage(
        rows,
        expected_ue_ids=range(args.expected_ue_count),
        trace_first_tti=args.measured_first_tti,
        trace_last_tti=args.measured_last_tti,
    )
    outcome_rows: list[dict[str, object]] = []
    if args.outcome_samples:
        with Path(args.outcome_samples).resolve().open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                terminal_tti = int(row["terminal_tti"])
                if args.measured_first_tti <= terminal_tti <= args.measured_last_tti:
                    outcome_rows.append(row)
    payload = {
        **protocol_identity(args.seed),
        "service_trace": str(trace_path),
        "service_trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        "measured_tti_first": args.measured_first_tti,
        "measured_tti_last": args.measured_last_tti,
        "trace_coverage": coverage,
        **summarize_starvation(
            rows,
            outcome_rows=outcome_rows,
            streak_threshold_tti=args.streak_threshold_tti,
        ),
    }
    if args.eval_summary:
        summary_path = Path(args.eval_summary).resolve()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        keys = (
            "stats_mean_goodput_mbps",
            "stats_mean_tb_err_rate",
            "stats_mean_expiry_drop_rate",
            "stats_mean_blank_ratio",
            "stats_weighted_packet_outcome_delay_mean_ms",
            "ue_macro_expiry_aware_rate_mbps",
            "ue_p10_expiry_aware_rate_mbps",
            "ue_min_expiry_aware_rate_mbps",
            "stats_sum_action_replay_mismatch_count",
            "stats_sum_illegal_action_count",
            "ctbs_mcs_mismatch_count",
            "ctbs_layer_mismatch_count",
            "ctbs_tbs_mismatch_count",
            "ctbs_reference_label_deletion_equal",
        )
        payload["eval_summary"] = str(summary_path)
        payload["eval_metrics"] = {key: summary.get(key) for key in keys}

    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

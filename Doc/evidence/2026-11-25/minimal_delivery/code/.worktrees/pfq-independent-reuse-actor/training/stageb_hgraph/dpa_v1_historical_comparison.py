"""Metrics for the DPA-v1 historical-comparison experiment.

The aggregator deliberately keeps evaluator-summary metrics separate from
metrics reconstructed from the per-TTI service and packet-outcome traces.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SEEDS = tuple(range(41, 51))
MODEL_NAME = "DPA-v1 (Deadline-Pressure Actor v1)"

FROZEN_HASHES = {
    "checkpoint_file": "94a2e40f0c0a56632e0c0ca63af4fa54d894d35aea38c2dd34afc0cfeadf2d7a",
    "actor_state": "6039c59e92ecda5ed75731fef86b71f567fe0c227142d9ed267434934b1e56be",
    "simulator": "4bdfee54c3203a808f42fd85367f6c46457bbf20dd1314e0c5855d2f07400a10",
    "deadline_service_features": "4a353601a8378888d089e34d3f04225770ae757ac07d5bc4108533a1a384975a",
    "online_observation_extras_builder": "ccf58c6c07e5ab65da373870d7d615b8048611508e4972910c4a4871744f876b",
    "feature_snapshot": "e86976c6208ff6e23e350d1480beaac011d694870d4a68e4e6fb9f7077aa7f7f",
    "starvation_audit": "22b3ec97f0ef5737f518d7cd0b9766b42eb53e314136ef1120e383ac40a9a775",
    "v9c_checkpoint": "7e1243960798ba3be7287235538b318a42ad62004e5f5a83f5a115234499bfb3",
    "v9c_aggregate_manifest": "e1aba7d5b98d3169dbf2b1971cce9324bb74abae00226d3be3a399ca6788f6ee",
    "v9c_comparison_json": "6c72b39e8733bb2c2c5fa1d654e9fe9a0985184a5843b783069385824e43c03b",
    "pfq_aggregate_manifest": "0df7d8eb94c9c3a4ce40ed9fd73d903eeec32e5c6fa8a1c474a1fe7206bb6b47",
    "pfq_mean_json": "9c00fee4affc295762a16fe6616676a0cf270559d76e99dbef3385ffc45d04cb",
}

V9C_BLOCK_RE = re.compile(
    r"tti=(?P<tti>\d+)\s+goodput=(?P<goodput>[0-9.]+)\s+"
    r"tb_err=(?P<tb_err>[0-9.]+)\s+blank=(?P<blank>[0-9.]+)\s+"
    r"fill=(?P<fill>[0-9.]+).*?uniq_ue=(?P<uniq_ue>[0-9.]+)"
)

ROW_FIELDS = (
    "arm",
    "seed",
    "window",
    "metric",
    "value",
    "unit",
    "direction",
    "source_artifact",
    "comparison_scope",
)

METRIC_META: dict[str, tuple[str, str]] = {
    "goodput_mbps": ("Mbps", "higher_better"),
    "served_mbps": ("Mbps", "higher_better"),
    "bler": ("ratio", "lower_better"),
    "success_rate": ("ratio", "higher_better"),
    "expiry_drop_rate": ("ratio", "lower_better"),
    "expired_bytes": ("bytes", "lower_better"),
    "expired_packets": ("packets", "lower_better"),
    "prg_utilization_ratio": ("ratio", "context"),
    "prg_reuse_ratio": ("ratio", "context"),
    "blank_ratio": ("ratio", "context"),
    "fill_ratio": ("ratio", "context"),
    "unique_ue_mean": ("UE", "context"),
    "outcome_delay_mean_ms": ("ms", "lower_better"),
    "outcome_delay_p50_ms": ("ms", "lower_better"),
    "outcome_delay_p90_ms": ("ms", "lower_better"),
    "outcome_delay_p95_ms": ("ms", "lower_better"),
    "outcome_delay_max_ms": ("ms", "lower_better"),
    "packet_delay_mean_ms": ("ms", "lower_better"),
    "packet_delay_p50_ms": ("ms", "lower_better"),
    "packet_delay_p90_ms": ("ms", "lower_better"),
    "packet_delay_p95_ms": ("ms", "lower_better"),
    "packet_delay_max_ms": ("ms", "lower_better"),
    "ue_goodput_mean_mbps": ("Mbps", "higher_better"),
    "ue_goodput_p5_mbps": ("Mbps", "higher_better"),
    "ue_goodput_p10_mbps": ("Mbps", "higher_better"),
    "ue_goodput_min_mbps": ("Mbps", "higher_better"),
    "ue_goodput_jain": ("ratio", "higher_better"),
    "scheduled_ratio_mean": ("ratio", "higher_better"),
    "scheduled_ratio_p5": ("ratio", "higher_better"),
    "scheduled_ratio_jain": ("ratio", "higher_better"),
    "positive_dequeue_ratio_mean": ("ratio", "higher_better"),
    "ue_expiry_aware_rate_mean_mbps": ("Mbps", "higher_better"),
    "ue_expiry_aware_rate_p10_mbps": ("Mbps", "higher_better"),
    "ue_expiry_aware_rate_min_mbps": ("Mbps", "higher_better"),
    "max_backlogged_zero_dequeue_streak_tti": ("TTI", "lower_better"),
    "backlog_free_ue_ratio": ("ratio", "higher_better"),
    "residual_buffer_ratio": ("ratio", "lower_better"),
    "packet_effective_service_rate_mbps": ("Mbps", "higher_better"),
    "ue_macro_packet_effective_service_rate_mbps": ("Mbps", "higher_better"),
}


def _pct(values: Sequence[float], percentile: float) -> float:
    return (
        float(np.percentile(np.asarray(values, dtype=np.float64), percentile))
        if values
        else 0.0
    )


def _jain(values: Sequence[float]) -> float:
    total = float(sum(values))
    squares = float(sum(value * value for value in values))
    return 0.0 if squares <= 0.0 else total * total / (len(values) * squares)


@dataclass(frozen=True)
class WindowSpec:
    label: str
    first_tti: int
    last_tti: int

    @property
    def tti_count(self) -> int:
        return self.last_tti - self.first_tti + 1


FULL_WINDOW = WindowSpec("full_4000", 1, 4000)
STEADY_WINDOW = WindowSpec("steady_3000", 1001, 4000)


def resolve_recorded_path(recorded: str, *, legacy_root: Path) -> Path:
    source = Path(recorded)
    if source.is_file():
        return source
    prefix = Path("/opt/nvidia/cuBB")
    try:
        relative = source.relative_to(prefix)
    except ValueError as exc:
        raise ValueError(f"unsupported historical path: {recorded}") from exc
    resolved = legacy_root / relative
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def parse_v9c_steady_blocks(
    path: Path,
    *,
    first_tti: int = 1001,
    last_tti: int = 4000,
) -> dict[str, object]:
    rows: list[dict[str, float | int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = V9C_BLOCK_RE.search(line)
        if match is None:
            continue
        row = {
            "tti": int(match.group("tti")),
            "goodput": float(match.group("goodput")),
            "tb_err": float(match.group("tb_err")),
            "blank": float(match.group("blank")),
            "fill": float(match.group("fill")),
            "uniq_ue": float(match.group("uniq_ue")),
        }
        if int(row["tti"]) - 499 >= first_tti and int(row["tti"]) <= last_tti:
            rows.append(row)
    expected_ends = [
        tti
        for tti in range(500, last_tti + 1, 500)
        if tti - 499 >= first_tti
    ]
    observed_ends = [int(row["tti"]) for row in rows]
    if observed_ends != expected_ends:
        raise ValueError(
            f"incomplete v9c steady blocks: observed={observed_ends} expected={expected_ends}"
        )
    return {
        "block_count": len(rows),
        "source_block_end_ttis": observed_ends,
        "goodput_mbps": float(
            statistics.mean(float(row["goodput"]) for row in rows)
        ),
        "tb_err_rate": float(
            statistics.mean(float(row["tb_err"]) for row in rows)
        ),
        "blank_ratio": float(
            statistics.mean(float(row["blank"]) for row in rows)
        ),
        "prg_utilization_ratio": float(
            statistics.mean(float(row["fill"]) for row in rows)
        ),
        "unique_ue_mean": float(
            statistics.mean(float(row["uniq_ue"]) for row in rows)
        ),
    }


def classify_scorecard(report: Mapping[str, object]) -> str:
    if not bool(report.get("structural_valid", False)):
        return "invalid_experiment"
    service = report.get("dpa_service_health", {})
    if not isinstance(service, Mapping):
        raise ValueError("dpa_service_health must be a mapping")
    if service.get("zero_service_seed_ues") or service.get("ttl_streak_seed_ues"):
        return "cross_layout_service_failed_starvation"

    dpa = report.get("dpa_full_mean", {})
    pfq = report.get("pfq_full_mean", {})
    v9c = report.get("v9c_full_mean", {})
    if not all(isinstance(item, Mapping) for item in (dpa, pfq, v9c)):
        raise ValueError("scorecard mean fields must be mappings")
    try:
        goodput_improved = float(dpa["goodput_mbps"]) >= float(
            pfq["goodput_mbps"]
        ) and float(dpa["goodput_mbps"]) >= float(v9c["goodput_mbps"])
        bler_not_worse = float(dpa["bler"]) <= float(pfq["bler"])
        expiry_not_worse = float(dpa["expiry_drop_rate"]) <= float(
            pfq["expiry_drop_rate"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("scorecard is missing required headline means") from exc
    if goodput_improved and bler_not_worse and expiry_not_worse:
        return "cross_layout_service_improved_with_no_starvation"
    if goodput_improved and not expiry_not_worse:
        return "aggregate_efficiency_improved_but_expiry_tail_failed"
    return "no_starvation_but_aggregate_efficiency_not_improved"


def validate_seed_summary(
    summary: Mapping[str, object],
    *,
    independent_runtime_candidate_width_request: int,
    expected_hashes: Mapping[str, str],
) -> dict[str, bool]:
    prg_vectors = (
        summary.get("ctbs_mask_trigger_count_by_local_prg"),
        summary.get("ctbs_selected_count_by_local_prg"),
        summary.get("ctbs_valid_candidate_count_by_local_prg"),
    )
    checks = {
        "completed_steps": int(summary.get("completed_steps", -1)) == 4000,
        "stats_warmup_steps": int(summary.get("stats_warmup_steps", -1)) == 1000,
        "stats_count": int(summary.get("stats_count", -1)) == 3000,
        "checkpoint_hash": summary.get("checkpoint_sha256")
        == expected_hashes["checkpoint_file"],
        "actor_state_hash": summary.get("actor_state_dict_sha256")
        == expected_hashes["actor_state"],
        "simulator_hash": summary.get("sim_binary_sha256")
        == expected_hashes["simulator"],
        "argmax_decode": summary.get("decode_mode") == "argmax",
        "ctbs_v1": summary.get("independent_headroom_accounting") == "ctbs-v1",
        "guard_off": summary.get("independent_non_starvation_guard") == "off",
        "candidate_truncation_forbidden": summary.get("candidate_truncation")
        == "forbidden",
        "candidate_width_request_zero": int(
            independent_runtime_candidate_width_request
        )
        == 0,
        "candidate_width_policy_runtime_topology": summary.get(
            "candidate_width_policy"
        )
        == "runtime_topology",
        "candidate_width_topology_resolved": int(
            summary.get("runtime_candidate_width", -1)
        )
        == int(summary.get("topology_candidate_width", -2)),
        "prg_decision_count_17": all(
            isinstance(vector, list) and len(vector) == 17 for vector in prg_vectors
        ),
        "candidate_truncation_zero": int(
            summary.get("candidate_truncation_count", -1)
        )
        == 0
        and float(summary.get("stats_sum_candidate_truncation_count", -1.0))
        == 0.0,
        "replay_mismatch_zero": float(
            summary.get("stats_sum_action_replay_mismatch_count", -1.0)
        )
        == 0.0,
        "illegal_action_zero": float(
            summary.get("stats_sum_illegal_action_count", -1.0)
        )
        == 0.0,
        "mask_reactivation_zero": int(
            summary.get("ctbs_mask_reactivation_count", -1)
        )
        == 0,
        "ctbs_mcs_parity": int(summary.get("ctbs_mcs_mismatch_count", -1)) == 0,
        "ctbs_layer_parity": int(summary.get("ctbs_layer_mismatch_count", -1))
        == 0,
        "ctbs_tbs_parity": int(summary.get("ctbs_tbs_mismatch_count", -1)) == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"structural seed validation failed: {', '.join(failed)}")
    return checks


def summarize_dpa_window(
    *,
    eval_summary: Mapping[str, object],
    service_rows: Sequence[Mapping[str, object]],
    outcome_rows: Sequence[Mapping[str, object]],
    window: WindowSpec,
    slot_duration_s: float = 0.0005,
) -> dict[str, object]:
    if window.first_tti <= 0 or window.last_tti < window.first_tti:
        raise ValueError("invalid TTI window")
    if window.label not in {"full_4000", "steady_3000"}:
        raise ValueError(f"unsupported window label: {window.label}")

    prefix = "" if window.label == "full_4000" else "stats_"
    selected_service = [
        row
        for row in service_rows
        if window.first_tti <= int(row["tti"]) <= window.last_tti
    ]
    selected_outcomes = [
        row
        for row in outcome_rows
        if window.first_tti <= int(row["terminal_tti"]) <= window.last_tti
    ]

    by_ue: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in selected_service:
        by_ue[int(row["ue_id"])].append(row)
    if not by_ue:
        raise ValueError("service window is empty")
    expected_ttis = set(range(window.first_tti, window.last_tti + 1))
    for ue_id, rows in by_ue.items():
        observed = {int(row["tti"]) for row in rows}
        if observed != expected_ttis:
            raise ValueError(f"incomplete service coverage for UE {ue_id}")

    duration_s = window.tti_count * slot_duration_s
    ue_goodput: list[float] = []
    scheduled_ratio: list[float] = []
    positive_dequeue_ratio: list[float] = []
    zero_service_ids: list[int] = []
    max_streak = 0
    expired_bytes = 0

    for ue_id in sorted(by_ue):
        rows = sorted(by_ue[ue_id], key=lambda row: int(row["tti"]))
        successful = sum(int(row["successful_bytes"]) for row in rows)
        scheduled = sum(int(row["scheduled_dequeue_bytes"]) for row in rows)
        expired_bytes += sum(int(row["expired_bytes"]) for row in rows)
        ue_goodput.append(successful * 8.0 / duration_s / 1.0e6)
        scheduled_ratio.append(
            sum(int(row["actual_prg_count"]) > 0 for row in rows) / window.tti_count
        )
        positive_dequeue_ratio.append(
            sum(int(row["scheduled_dequeue_bytes"]) > 0 for row in rows)
            / window.tti_count
        )

        had_backlog = any(int(row["pre_queue_bytes"]) > 0 for row in rows)
        if had_backlog and scheduled == 0:
            zero_service_ids.append(ue_id)
        current_streak = 0
        for row in rows:
            if int(row["pre_queue_bytes"]) > 0 and int(row["scheduled_dequeue_bytes"]) == 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

    # Delays intentionally include every terminal outcome, including expired
    # packets. This keeps the delay distribution boundary-consistent.
    delays = [float(row["system_time_ms"]) for row in selected_outcomes]
    outcomes_by_ue: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in selected_outcomes:
        outcomes_by_ue[int(row["ue_id"])].append(row)

    ue_rates: list[float] = []
    no_outcome_ue_count = 0
    for ue_id in sorted(by_ue):
        rows = outcomes_by_ue.get(ue_id, [])
        terminal_rows = [
            row
            for row in rows
            if str(row["outcome"]) in {"delivered", "expired"}
        ]
        outcome_time_ms = sum(float(row["system_time_ms"]) for row in terminal_rows)
        delivered_bytes = sum(
            int(row["packet_size_bytes"])
            for row in terminal_rows
            if str(row["outcome"]) == "delivered"
        )
        if outcome_time_ms > 0.0:
            # bytes * 8 / milliseconds / 1000 = megabits/second
            ue_rates.append(delivered_bytes * 8.0 / (outcome_time_ms * 1000.0))
        else:
            ue_rates.append(0.0)
            no_outcome_ue_count += 1

    return {
        "window": window.label,
        "first_tti": window.first_tti,
        "last_tti": window.last_tti,
        "tti_count": window.tti_count,
        "goodput_mbps": float(eval_summary[f"{prefix}mean_goodput_mbps"]),
        "bler": float(eval_summary[f"{prefix}mean_tb_err_rate"]),
        "expiry_drop_rate": float(eval_summary[f"{prefix}mean_expiry_drop_rate"]),
        "prg_utilization_ratio": float(
            eval_summary[f"{prefix}mean_prg_utilization_ratio"]
        ),
        "blank_ratio": float(eval_summary[f"{prefix}mean_blank_ratio"]),
        "expired_bytes": expired_bytes,
        "outcome_count": len(delays),
        "outcome_delay_mean_ms": float(np.mean(delays)) if delays else 0.0,
        "outcome_delay_p50_ms": _pct(delays, 50),
        "outcome_delay_p90_ms": _pct(delays, 90),
        "outcome_delay_p95_ms": _pct(delays, 95),
        "outcome_delay_max_ms": max(delays, default=0.0),
        "ue_goodput_mean_mbps": float(np.mean(ue_goodput)),
        "ue_goodput_p5_mbps": _pct(ue_goodput, 5),
        "ue_goodput_p10_mbps": _pct(ue_goodput, 10),
        "ue_goodput_min_mbps": min(ue_goodput),
        "ue_goodput_jain": _jain(ue_goodput),
        "scheduled_ratio_mean": float(np.mean(scheduled_ratio)),
        "scheduled_ratio_p5": _pct(scheduled_ratio, 5),
        "scheduled_ratio_jain": _jain(scheduled_ratio),
        "positive_dequeue_ratio_mean": float(np.mean(positive_dequeue_ratio)),
        "ue_expiry_aware_rate_mean_mbps": float(np.mean(ue_rates)) if ue_rates else 0.0,
        "ue_expiry_aware_rate_p10_mbps": _pct(ue_rates, 10),
        "ue_expiry_aware_rate_min_mbps": min(ue_rates, default=0.0),
        "ue_expiry_aware_rate_ue_count": len(ue_rates),
        "ue_expiry_aware_rate_no_outcome_ue_count": no_outcome_ue_count,
        "zero_scheduled_dequeue_ue_ids": zero_service_ids,
        "max_backlogged_zero_dequeue_streak_tti": max_streak,
        "metric_units": {
            "goodput_mbps": "Mbps",
            "bler": "ratio",
            "expiry_drop_rate": "ratio",
            "outcome_delay_mean_ms": "ms",
            "ue_expiry_aware_rate_mean_mbps": "Mbps",
        },
        "source_semantics": {
            "system_metrics": (
                "evaluator summary; mean_* for full_4000 and stats_mean_* for steady_3000"
            ),
            "expired_bytes": "sum of service-balance expired_bytes over the inclusive TTI window",
            "outcome_delay": (
                "all packet outcomes selected by inclusive terminal_tti, including delivered and expired"
            ),
            "ue_goodput": "successful_bytes aggregated per UE over the inclusive service window",
            "ue_expiry_aware_rate": (
                "delivered packet bytes*8 divided by that UE's delivered-plus-expired outcome time"
            ),
            "starvation": "positive-backlog rows with zero scheduled dequeue",
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _find_frozen_file(root: Path, pattern: str, expected_sha256: str) -> Path:
    candidates = sorted(path for path in root.glob(pattern) if path.is_file())
    matches = [path for path in candidates if _sha256(path) == expected_sha256]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one frozen artifact under {root} matching {pattern}; "
            f"hash_matches={[str(path) for path in matches]}"
        )
    return matches[0]


def _hash_record(path: Path, expected_sha256: str) -> dict[str, str]:
    resolved = path.resolve()
    observed = _sha256(resolved)
    if observed != expected_sha256:
        raise ValueError(
            f"frozen hash mismatch for {resolved}: "
            f"observed={observed} expected={expected_sha256}"
        )
    return {
        "path": str(resolved),
        "sha256": observed,
        "expected_sha256": expected_sha256,
    }


def _frozen_hash_values(manifest_hashes: object) -> dict[str, str]:
    if not isinstance(manifest_hashes, Mapping):
        raise ValueError("campaign manifest is missing hashes")
    observed: dict[str, str] = {}
    for name, expected in FROZEN_HASHES.items():
        record = manifest_hashes.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"campaign manifest is missing hash record: {name}")
        value = str(record.get("sha256", ""))
        if value != expected:
            raise ValueError(
                f"campaign manifest frozen hash mismatch for {name}: "
                f"observed={value} expected={expected}"
            )
        observed[name] = value
    return observed


def _checkpoint_actor_state_sha256(path: Path) -> str:
    import torch

    from training.stageb_hgraph.independent_contract import state_dict_sha256

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    actor_state = checkpoint.get("actor_state")
    if not isinstance(actor_state, Mapping):
        raise ValueError("checkpoint is missing actor_state")
    return state_dict_sha256(actor_state)


def run_preflight(args: argparse.Namespace) -> None:
    worktree_root = Path(args.worktree_root).resolve()
    legacy_root = Path(args.legacy_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    simulator = Path(args.sim_bin).resolve()
    v9c_root = Path(args.v9c_root).resolve()
    pfq_root = Path(args.pfq_root).resolve()

    v9c_manifest = v9c_root / "aggregate_manifest.csv"
    v9c_comparison = _find_frozen_file(
        v9c_root,
        "RAYLEIGH/*compare_mean.json",
        FROZEN_HASHES["v9c_comparison_json"],
    )
    pfq_manifest = pfq_root / "aggregate_manifest.csv"
    pfq_mean = pfq_root / "RAYLEIGH" / "rrq_vs_pfq_compare_mean.json"
    comparison = _read_json(v9c_comparison)
    if not isinstance(comparison, Mapping):
        raise ValueError("v9c comparison JSON must be a mapping")
    source_runs = comparison.get("source_runs")
    if not isinstance(source_runs, list) or len(source_runs) != len(SEEDS):
        raise ValueError("v9c comparison must contain ten source runs")
    if [int(run["seed"]) for run in source_runs] != list(SEEDS):
        raise ValueError("v9c comparison seed order is not 41--50")
    v9c_checkpoint = resolve_recorded_path(
        str(source_runs[0]["model_path"]), legacy_root=legacy_root
    )
    if any(str(run.get("model_path")) != str(source_runs[0]["model_path"]) for run in source_runs):
        raise ValueError("v9c source runs do not share one frozen checkpoint")

    paths = {
        "checkpoint_file": checkpoint,
        "simulator": simulator,
        "deadline_service_features": worktree_root
        / "cuMAC/examples/onlineTrainBridge/DeadlineServiceFeatures.h",
        "online_observation_extras_builder": worktree_root
        / "cuMAC/examples/onlineTrainBridge/OnlineObservationExtrasBuilder.h",
        "feature_snapshot": worktree_root
        / "training/stageb_hgraph/feature_snapshot.py",
        "starvation_audit": worktree_root
        / "training/stageb_hgraph/seed45_starvation_audit.py",
        "v9c_checkpoint": v9c_checkpoint,
        "v9c_aggregate_manifest": v9c_manifest,
        "v9c_comparison_json": v9c_comparison,
        "pfq_aggregate_manifest": pfq_manifest,
        "pfq_mean_json": pfq_mean,
    }
    hashes = {
        name: _hash_record(path, FROZEN_HASHES[name])
        for name, path in paths.items()
    }
    observed_actor_state = _checkpoint_actor_state_sha256(checkpoint)
    if observed_actor_state != FROZEN_HASHES["actor_state"]:
        raise ValueError(
            "frozen actor-state mismatch: "
            f"observed={observed_actor_state} expected={FROZEN_HASHES['actor_state']}"
        )
    hashes["actor_state"] = {
        "path": f"{checkpoint.resolve()}::actor_state",
        "sha256": observed_actor_state,
        "expected_sha256": FROZEN_HASHES["actor_state"],
    }

    manifest = {
        "schema_version": 1,
        "experiment": "dpa_v1_seed41_50_historical_comparison_v1",
        "model_name": MODEL_NAME,
        "model_provenance": "feature-only iter8 checkpoint",
        "seeds": list(SEEDS),
        "windows": {
            "full_4000": {"first_tti": 1, "last_tti": 4000, "tti_count": 4000},
            "steady_3000": {
                "first_tti": 1001,
                "last_tti": 4000,
                "tti_count": 3000,
            },
        },
        "decode_mode": "argmax",
        "headroom_accounting": "ctbs-v1",
        "independent_runtime_candidate_width_request": 0,
        "guard_mode": "off",
        "guard_enabled": False,
        "hashes": hashes,
        "historical_sources": {
            "legacy_root": str(legacy_root),
            "v9c_root": str(v9c_root),
            "v9c_comparison_json": str(v9c_comparison.resolve()),
            "pfq_root": str(pfq_root),
            "pfq_mean_json": str(pfq_mean.resolve()),
        },
        "historical_comparison": {
            "strict_paired_superiority_claim_allowed": False,
            "limitation": (
                "PFQ and v9c are preserved historical executions from an older "
                "simulator binary without a complete offered-stream fingerprint; "
                "cohort differences may be described but strict paired or "
                "binary-identical superiority may not be claimed."
            ),
        },
    }
    out = Path(args.out).resolve()
    if args.verify_only:
        if not out.is_file():
            raise ValueError(f"verify-only manifest is missing or not a file: {out}")
        try:
            recorded_manifest = _read_json(out)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"verify-only manifest has malformed JSON: {out}") from exc
        if not isinstance(recorded_manifest, Mapping):
            raise ValueError(f"verify-only manifest is not a JSON mapping: {out}")
        if recorded_manifest != manifest:
            raise ValueError("verify-only manifest does not match recomputed manifest")
        return
    _write_json(out, manifest)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _load_outcome_rows(path: Path) -> list[dict[str, str]]:
    required = {
        "ue_id",
        "packet_size_bytes",
        "terminal_tti",
        "outcome",
        "system_time_ms",
    }
    rows = _load_csv_rows(path)
    if not rows:
        raise ValueError("packet outcome trace is empty")
    if not required.issubset(rows[0]):
        raise ValueError(f"packet outcome trace schema is incomplete: {path}")
    return rows


def _validate_exit_codes(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.strip()] = int(value.strip())
    if values.get("eval_rc") != 0 or values.get("sim_rc") != 0:
        raise ValueError(f"nonzero evaluator/simulator exit code: {values}")
    return values


def _per_ue_service(
    service_rows: Sequence[Mapping[str, object]],
    *,
    window: WindowSpec,
) -> list[dict[str, object]]:
    selected = [
        row
        for row in service_rows
        if window.first_tti <= int(row["tti"]) <= window.last_tti
    ]
    by_ue: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in selected:
        by_ue[int(row["ue_id"])].append(row)
    result: list[dict[str, object]] = []
    duration_s = window.tti_count * 0.0005
    for ue_id in sorted(by_ue):
        rows = sorted(by_ue[ue_id], key=lambda row: int(row["tti"]))
        scheduled = sum(int(row["scheduled_dequeue_bytes"]) for row in rows)
        successful = sum(int(row["successful_bytes"]) for row in rows)
        current_streak = 0
        max_streak = 0
        for row in rows:
            if int(row["pre_queue_bytes"]) > 0 and int(row["scheduled_dequeue_bytes"]) == 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        result.append(
            {
                "ue_id": ue_id,
                "scheduled_dequeue_bytes": scheduled,
                "successful_bytes": successful,
                "expired_bytes": sum(int(row["expired_bytes"]) for row in rows),
                "goodput_mbps": successful * 8.0 / duration_s / 1.0e6,
                "scheduled_ratio": sum(int(row["actual_prg_count"]) > 0 for row in rows)
                / window.tti_count,
                "positive_dequeue_ratio": sum(
                    int(row["scheduled_dequeue_bytes"]) > 0 for row in rows
                )
                / window.tti_count,
                "had_positive_backlog": any(
                    int(row["pre_queue_bytes"]) > 0 for row in rows
                ),
                "max_backlogged_zero_dequeue_streak_tti": max_streak,
            }
        )
    return result


def _add_dpa_source_metrics(
    summary: dict[str, object],
    *,
    eval_summary: Mapping[str, object],
    prefix: str,
    kpi_summary: Mapping[str, object] | None,
) -> None:
    eval_fields = {
        "prg_reuse_ratio": f"{prefix}mean_prg_reuse_ratio",
        "fill_ratio": f"{prefix}mean_fill_ratio",
        "unique_ue_mean": f"{prefix}mean_uniq_ue",
    }
    for metric, source_key in eval_fields.items():
        if eval_summary.get(source_key) is not None:
            summary[metric] = float(eval_summary[source_key])
    if kpi_summary is None:
        return
    kpi_fields = {
        ("traffic", "served_mbps_est"): "served_mbps",
        ("traffic", "expired_packets"): "expired_packets",
        ("traffic", "packet_effective_service_rate_mbps"): "packet_effective_service_rate_mbps",
        ("traffic", "ue_macro_packet_effective_service_rate_mbps"): "ue_macro_packet_effective_service_rate_mbps",
        ("traffic", "packet_delay_mean_ms"): "packet_delay_mean_ms",
        ("traffic", "packet_delay_p50_ms"): "packet_delay_p50_ms",
        ("traffic", "packet_delay_p90_ms"): "packet_delay_p90_ms",
        ("traffic", "packet_delay_p95_ms"): "packet_delay_p95_ms",
        ("traffic", "packet_delay_max_ms"): "packet_delay_max_ms",
        ("global_kpi", "residual_buffer_ratio"): "residual_buffer_ratio",
        ("global_kpi", "backlog_free_ue_ratio"): "backlog_free_ue_ratio",
    }
    for (section, source_key), metric in kpi_fields.items():
        section_payload = kpi_summary.get(section, {})
        if isinstance(section_payload, Mapping) and section_payload.get(source_key) is not None:
            summary[metric] = section_payload[source_key]


def run_extract_seed(args: argparse.Namespace) -> None:
    manifest_path = Path(args.campaign_manifest).resolve()
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError("campaign manifest must be a mapping")
    seed = int(args.seed)
    if seed not in SEEDS or seed not in [int(value) for value in manifest.get("seeds", [])]:
        raise ValueError(f"seed is outside frozen cohort: {seed}")
    actor_dir = Path(args.actor_dir).resolve()
    eval_summary_path = actor_dir / "eval_summary.json"
    eval_summary = _read_json(eval_summary_path)
    if not isinstance(eval_summary, Mapping):
        raise ValueError("eval_summary must be a mapping")
    manifest_hashes = manifest.get("hashes")
    frozen_hashes = _frozen_hash_values(manifest_hashes)
    expected_hashes = {
        name: frozen_hashes[name]
        for name in ("checkpoint_file", "actor_state", "simulator")
    }
    structural_checks = validate_seed_summary(
        eval_summary,
        independent_runtime_candidate_width_request=int(
            manifest.get("independent_runtime_candidate_width_request", -1)
        ),
        expected_hashes=expected_hashes,
    )
    exit_codes = _validate_exit_codes(actor_dir / "exit_codes.txt")
    eval_steps_path = actor_dir / "eval_steps.jsonl"
    step_rows: list[Mapping[str, object]] = []
    with eval_steps_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError("eval step row must be a mapping")
            step_rows.append(row)
    eval_step_count = len(step_rows)
    if eval_step_count != 4000:
        raise ValueError(f"eval step trace count is {eval_step_count}, expected 4000")
    trace_zero_fields = (
        "candidate_truncation_count",
        "action_replay_mismatch_count",
        "illegal_action_count",
    )
    for field in trace_zero_fields:
        total = sum(float(row.get(field, 0.0)) for row in step_rows)
        if total != 0.0:
            raise ValueError(f"full eval trace has nonzero {field}: {total}")

    service_path = actor_dir / "service_balance_trace.csv"
    service_rows = _load_csv_rows(service_path)
    observed_ues = sorted({int(row["ue_id"]) for row in service_rows})
    observed_ttis = sorted({int(row["tti"]) for row in service_rows})
    if observed_ues != list(range(36)) or observed_ttis != list(range(1, 4001)):
        raise ValueError("service trace does not cover UEs 0--35 and TTIs 1--4000")
    if len(service_rows) != 36 * 4000:
        raise ValueError("service trace does not contain exactly 144000 rows")
    outcome_path = actor_dir / "packet_outcome_samples.csv"
    outcome_rows = _load_outcome_rows(outcome_path)
    kpi_summary_path = actor_dir / "kpi_summary.json"
    kpi_summary = _read_json(kpi_summary_path)
    if not isinstance(kpi_summary, Mapping):
        raise ValueError("kpi_summary must be a mapping")

    windows: dict[str, dict[str, object]] = {}
    for window in (FULL_WINDOW, STEADY_WINDOW):
        summary = summarize_dpa_window(
            eval_summary=eval_summary,
            service_rows=service_rows,
            outcome_rows=outcome_rows,
            window=window,
        )
        _add_dpa_source_metrics(
            summary,
            eval_summary=eval_summary,
            prefix="" if window.label == "full_4000" else "stats_",
            kpi_summary=kpi_summary if window.label == "full_4000" else None,
        )
        summary["per_ue_service"] = _per_ue_service(service_rows, window=window)
        windows[window.label] = summary

    audits: dict[str, object] = {}
    for label in ("full_4000", "steady_3000"):
        audit_path = actor_dir / f"starvation_{label}.json"
        audit = _read_json(audit_path)
        if not isinstance(audit, Mapping):
            raise ValueError(f"starvation audit must be a mapping: {audit_path}")
        if int(audit.get("seed", -1)) != seed:
            raise ValueError(f"starvation audit seed mismatch: {audit_path}")
        audits[label] = audit

    payload = {
        "schema_version": 1,
        "model_name": MODEL_NAME,
        "seed": seed,
        "structural_valid": True,
        "structural_checks": structural_checks,
        "exit_codes": exit_codes,
        "eval_step_count": eval_step_count,
        "windows": windows,
        "service_audits": audits,
        "source_artifacts": {
            "campaign_manifest": str(manifest_path),
            "eval_summary": str(eval_summary_path),
            "eval_steps": str(eval_steps_path),
            "service_balance_trace": str(service_path),
            "packet_outcomes": str(outcome_path),
            "kpi_summary": str(kpi_summary_path),
            "starvation_full_4000": str(actor_dir / "starvation_full_4000.json"),
            "starvation_steady_3000": str(actor_dir / "starvation_steady_3000.json"),
        },
    }
    _write_json(Path(args.out).resolve(), payload)


PFQ_ALIASES = {
    "traffic.served_mbps_est": "served_mbps",
    "traffic.goodput_mbps": "goodput_mbps",
    "traffic.expired_bytes": "expired_bytes",
    "traffic.expired_packets": "expired_packets",
    "traffic.expiry_drop_rate": "expiry_drop_rate",
    "global_kpi.average_ue_goodput_mbps": "ue_goodput_mean_mbps",
    "global_kpi.ue_goodput_jain": "ue_goodput_jain",
    "global_kpi.ue_goodput_p5_mbps": "ue_goodput_p5_mbps",
    "global_kpi.ue_goodput_p10_mbps": "ue_goodput_p10_mbps",
    "global_kpi.global_tb_bler": "bler",
    "global_kpi.global_tx_success_rate": "success_rate",
    "global_kpi.residual_buffer_ratio": "residual_buffer_ratio",
    "global_kpi.backlog_free_ue_ratio": "backlog_free_ue_ratio",
    "global_kpi.prg_utilization_ratio": "prg_utilization_ratio",
    "traffic.packet_effective_service_rate_mbps": "packet_effective_service_rate_mbps",
    "traffic.ue_macro_packet_effective_service_rate_mbps": "ue_macro_packet_effective_service_rate_mbps",
    "traffic.packet_delay_mean_ms": "packet_delay_mean_ms",
    "traffic.packet_delay_p50_ms": "packet_delay_p50_ms",
    "traffic.packet_delay_p90_ms": "packet_delay_p90_ms",
    "traffic.packet_delay_p95_ms": "packet_delay_p95_ms",
    "global_kpi.scheduled_ratio_mean": "scheduled_ratio_mean",
    "global_kpi.scheduled_ratio_jain": "scheduled_ratio_jain",
    "global_kpi.scheduled_ratio_p5": "scheduled_ratio_p5",
}


def _normalized_row(
    *,
    arm: str,
    seed: int,
    window: str,
    metric: str,
    value: object,
    unit: str | None,
    direction: str | None,
    source_artifact: Path | str,
    comparison_scope: str,
) -> dict[str, object]:
    normalized_value: float | int | None
    if value is None or value == "":
        normalized_value = None
    elif isinstance(value, (int, float)):
        normalized_value = value
    else:
        normalized_value = float(value)
    return {
        "arm": arm,
        "seed": seed,
        "window": window,
        "metric": metric,
        "value": normalized_value,
        "unit": unit,
        "direction": direction,
        "source_artifact": str(source_artifact),
        "comparison_scope": comparison_scope,
    }


def materialize_historical_grid(
    rows: Sequence[Mapping[str, object]],
    *,
    arm: str,
    seeds: Sequence[int],
    source_notes: Mapping[str, str],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for row in rows:
        normalized = dict(row)
        window = str(normalized["window"])
        if str(normalized["arm"]) == arm and normalized.get("value") is None:
            if window not in source_notes:
                raise ValueError(f"missing unavailable source note for {arm} {window}")
            normalized["source_artifact"] = source_notes[window]
            normalized["comparison_scope"] = "unavailable_historical_metric"
        result.append(normalized)
    existing = {
        (str(row["arm"]), int(row["seed"]), str(row["window"]), str(row["metric"]))
        for row in result
    }
    for seed in seeds:
        for window in ("full_4000", "steady_3000"):
            if window not in source_notes:
                raise ValueError(f"missing unavailable source note for {arm} {window}")
            for metric, (unit, direction) in METRIC_META.items():
                key = (arm, int(seed), window, metric)
                if key in existing:
                    continue
                result.append(
                    _normalized_row(
                        arm=arm,
                        seed=int(seed),
                        window=window,
                        metric=metric,
                        value=None,
                        unit=unit,
                        direction=direction,
                        source_artifact=source_notes[window],
                        comparison_scope="unavailable_historical_metric",
                    )
                )
                existing.add(key)
    return result


def _dpa_rows(seed_metrics: Mapping[str, object], source: Path) -> list[dict[str, object]]:
    seed = int(seed_metrics["seed"])
    windows = seed_metrics["windows"]
    artifacts = seed_metrics.get("source_artifacts", {})
    eval_metrics = {
        "goodput_mbps",
        "bler",
        "expiry_drop_rate",
        "prg_utilization_ratio",
        "prg_reuse_ratio",
        "blank_ratio",
        "fill_ratio",
        "unique_ue_mean",
    }
    kpi_metrics = {
        "served_mbps",
        "expired_packets",
        "packet_effective_service_rate_mbps",
        "ue_macro_packet_effective_service_rate_mbps",
        "packet_delay_mean_ms",
        "packet_delay_p50_ms",
        "packet_delay_p90_ms",
        "packet_delay_p95_ms",
        "packet_delay_max_ms",
        "residual_buffer_ratio",
        "backlog_free_ue_ratio",
    }
    outcome_metrics = {
        "outcome_delay_mean_ms",
        "outcome_delay_p50_ms",
        "outcome_delay_p90_ms",
        "outcome_delay_p95_ms",
        "outcome_delay_max_ms",
        "ue_expiry_aware_rate_mean_mbps",
        "ue_expiry_aware_rate_p10_mbps",
        "ue_expiry_aware_rate_min_mbps",
    }
    rows: list[dict[str, object]] = []
    for window_label in ("full_4000", "steady_3000"):
        metrics = windows[window_label]
        for metric, (unit, direction) in METRIC_META.items():
            if metric not in metrics:
                continue
            if metric in eval_metrics:
                source_artifact = artifacts.get("eval_summary", source)
            elif metric in kpi_metrics:
                source_artifact = artifacts.get("kpi_summary", source)
            elif metric in outcome_metrics:
                source_artifact = artifacts.get("packet_outcomes", source)
            else:
                source_artifact = artifacts.get("service_balance_trace", source)
            rows.append(
                _normalized_row(
                    arm="dpa_v1",
                    seed=seed,
                    window=window_label,
                    metric=metric,
                    value=metrics[metric],
                    unit=unit,
                    direction=direction,
                    source_artifact=source_artifact,
                    comparison_scope="new_dpa_rollout",
                )
            )
    return rows


def _pfq_rows(pfq_root: Path, seeds: Sequence[int]) -> tuple[list[dict[str, object]], dict[int, Path]]:
    rows: list[dict[str, object]] = []
    sources: dict[int, Path] = {}
    for seed in seeds:
        path = pfq_root / "per_seed" / f"s{seed}" / "RAYLEIGH" / "rrq_vs_pfq_compare.json"
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"PFQ comparison must be a mapping: {path}")
        values = payload.get("rr_vs_pf_metrics")
        if not isinstance(values, list):
            raise ValueError(f"PFQ comparison is missing rr_vs_pf_metrics: {path}")
        sources[seed] = path.resolve()
        for entry in values:
            original_name = str(entry["name"])
            metric = PFQ_ALIASES.get(original_name, original_name)
            unit = entry.get("unit")
            if unit == "%":
                unit = "ratio"
            rows.append(
                _normalized_row(
                    arm="pfq_historical",
                    seed=seed,
                    window="full_4000",
                    metric=metric,
                    value=entry.get("pf"),
                    unit=str(unit) if unit is not None else None,
                    direction=str(entry.get("direction")) if entry.get("direction") else None,
                    source_artifact=path.resolve(),
                    comparison_scope="historical_same_seed_cohort_full_4000",
                )
            )
    return (
        materialize_historical_grid(
            rows,
            arm="pfq_historical",
            seeds=seeds,
            source_notes={
                "full_4000": (
                    "unavailable: metric is not present in the per-seed PFQ comparison JSON"
                ),
                "steady_3000": (
                    "unavailable: PFQ has no preserved equivalent post-1000 per-TTI trace"
                ),
            },
        ),
        sources,
    )


V9C_EVAL_FIELDS = {
    "mean_goodput_mbps": "goodput_mbps",
    "mean_tb_err_rate": "bler",
    "mean_expiry_drop_rate": "expiry_drop_rate",
    "mean_prg_utilization_ratio": "prg_utilization_ratio",
    "mean_prg_reuse_ratio": "prg_reuse_ratio",
    "mean_blank_ratio": "blank_ratio",
    "mean_fill_ratio": "fill_ratio",
    "mean_uniq_ue": "unique_ue_mean",
}

V9C_KPI_FIELDS = {
    ("traffic", "served_mbps_est"): "served_mbps",
    ("traffic", "expired_bytes"): "expired_bytes",
    ("traffic", "expired_packets"): "expired_packets",
    ("traffic", "packet_effective_service_rate_mbps"): "packet_effective_service_rate_mbps",
    ("traffic", "ue_macro_packet_effective_service_rate_mbps"): "ue_macro_packet_effective_service_rate_mbps",
    ("traffic", "packet_delay_mean_ms"): "packet_delay_mean_ms",
    ("traffic", "packet_delay_p50_ms"): "packet_delay_p50_ms",
    ("traffic", "packet_delay_p90_ms"): "packet_delay_p90_ms",
    ("traffic", "packet_delay_p95_ms"): "packet_delay_p95_ms",
    ("traffic", "packet_delay_max_ms"): "packet_delay_max_ms",
    ("global_kpi", "average_ue_goodput_mbps"): "ue_goodput_mean_mbps",
    ("global_kpi", "ue_goodput_p5_mbps"): "ue_goodput_p5_mbps",
    ("global_kpi", "ue_goodput_p10_mbps"): "ue_goodput_p10_mbps",
    ("global_kpi", "ue_goodput_jain"): "ue_goodput_jain",
    ("global_kpi", "scheduled_ratio_mean"): "scheduled_ratio_mean",
    ("global_kpi", "scheduled_ratio_p5"): "scheduled_ratio_p5",
    ("global_kpi", "scheduled_ratio_jain"): "scheduled_ratio_jain",
    ("global_kpi", "residual_buffer_ratio"): "residual_buffer_ratio",
    ("global_kpi", "backlog_free_ue_ratio"): "backlog_free_ue_ratio",
}


def _v9c_rows(
    comparison_path: Path,
    *,
    legacy_root: Path,
) -> tuple[list[dict[str, object]], dict[int, dict[str, Path]]]:
    comparison = _read_json(comparison_path)
    if not isinstance(comparison, Mapping) or not isinstance(comparison.get("source_runs"), list):
        raise ValueError("v9c comparison source_runs are missing")
    rows: list[dict[str, object]] = []
    sources: dict[int, dict[str, Path]] = {}
    for run in comparison["source_runs"]:
        seed = int(run["seed"])
        eval_path = resolve_recorded_path(str(run["eval_summary_json"]), legacy_root=legacy_root)
        kpi_path = resolve_recorded_path(str(run["kpi_summary_json"]), legacy_root=legacy_root)
        log_path = comparison_path.parents[1] / "per_seed_eval" / f"s{seed}" / "checkpoint_eval.log"
        eval_summary = _read_json(eval_path)
        kpi = _read_json(kpi_path)
        if not isinstance(eval_summary, Mapping) or not isinstance(kpi, Mapping):
            raise ValueError(f"invalid v9c source JSON for seed {seed}")
        sources[seed] = {
            "eval_summary": eval_path.resolve(),
            "kpi_summary": kpi_path.resolve(),
            "checkpoint_eval_log": log_path.resolve(),
        }
        for source_key, metric in V9C_EVAL_FIELDS.items():
            unit, direction = METRIC_META[metric]
            rows.append(
                _normalized_row(
                    arm="v9c_historical",
                    seed=seed,
                    window="full_4000",
                    metric=metric,
                    value=eval_summary.get(source_key),
                    unit=unit,
                    direction=direction,
                    source_artifact=eval_path.resolve(),
                    comparison_scope="historical_same_seed_cohort_full_4000",
                )
            )
        for (section, source_key), metric in V9C_KPI_FIELDS.items():
            section_payload = kpi.get(section, {})
            value = section_payload.get(source_key) if isinstance(section_payload, Mapping) else None
            unit, direction = METRIC_META[metric]
            rows.append(
                _normalized_row(
                    arm="v9c_historical",
                    seed=seed,
                    window="full_4000",
                    metric=metric,
                    value=value,
                    unit=unit,
                    direction=direction,
                    source_artifact=kpi_path.resolve(),
                    comparison_scope="historical_same_seed_cohort_full_4000",
                )
            )
        steady = parse_v9c_steady_blocks(log_path, first_tti=1001, last_tti=4000)
        for source_key, metric in (
            ("goodput_mbps", "goodput_mbps"),
            ("tb_err_rate", "bler"),
            ("blank_ratio", "blank_ratio"),
            ("prg_utilization_ratio", "prg_utilization_ratio"),
            ("unique_ue_mean", "unique_ue_mean"),
        ):
            unit, direction = METRIC_META[metric]
            rows.append(
                _normalized_row(
                    arm="v9c_historical",
                    seed=seed,
                    window="steady_3000",
                    metric=metric,
                    value=steady[source_key],
                    unit=unit,
                    direction=direction,
                    source_artifact=log_path.resolve(),
                    comparison_scope=(
                        "historical_same_seed_cohort_steady_3000_six_equal_500_tti_blocks"
                    ),
                )
            )
    if sorted(sources) != list(SEEDS):
        raise ValueError("v9c sources do not cover seeds 41--50")
    return (
        materialize_historical_grid(
            rows,
            arm="v9c_historical",
            seeds=SEEDS,
            source_notes={
                "full_4000": (
                    "unavailable: metric is not present in the recorded v9c evaluator/KPI artifacts"
                ),
                "steady_3000": (
                    "unavailable: metric is not emitted by the preserved v9c 500-TTI block logs"
                ),
            },
        ),
        sources,
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["arm"]), str(row["window"]), str(row["metric"]))].append(row)
    aggregate: list[dict[str, object]] = []
    for key in sorted(groups):
        arm, window, metric = key
        group = groups[key]
        numeric = [row for row in group if row["value"] is not None]
        values = [float(row["value"]) for row in numeric]
        direction = group[0].get("direction")
        worst: Mapping[str, object] | None = None
        if numeric:
            if direction == "lower_better":
                worst = max(numeric, key=lambda row: float(row["value"]))
            else:
                worst = min(numeric, key=lambda row: float(row["value"]))
        aggregate.append(
            {
                "arm": arm,
                "window": window,
                "metric": metric,
                "count": len(values),
                "mean": statistics.mean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
                "median": statistics.median(values) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "worst_seed": int(worst["seed"]) if worst is not None else None,
                "worst_value": worst["value"] if worst is not None else None,
                "unit": group[0].get("unit"),
                "direction": direction,
                "comparison_scope": group[0].get("comparison_scope"),
            }
        )
    return aggregate


def _row_index(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, int, str, str], Mapping[str, object]]:
    index: dict[tuple[str, int, str, str], Mapping[str, object]] = {}
    for row in rows:
        key = (str(row["arm"]), int(row["seed"]), str(row["window"]), str(row["metric"]))
        if key in index:
            continue
        index[key] = row
    return index


def _mean_metrics(
    aggregate: Sequence[Mapping[str, object]], arm: str, window: str
) -> dict[str, object]:
    return {
        str(row["metric"]): row["mean"]
        for row in aggregate
        if row["arm"] == arm and row["window"] == window
    }


def _win_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    index = _row_index(rows)
    result: dict[str, object] = {}
    for historical in ("pfq_historical", "v9c_historical"):
        arm_result: dict[str, object] = {}
        for window in ("full_4000", "steady_3000"):
            metric_result: dict[str, object] = {}
            for metric, (_, direction) in METRIC_META.items():
                if direction not in {"higher_better", "lower_better"}:
                    continue
                wins = losses = ties = compared = 0
                for seed in SEEDS:
                    dpa = index.get(("dpa_v1", seed, window, metric))
                    baseline = index.get((historical, seed, window, metric))
                    if dpa is None or baseline is None or dpa["value"] is None or baseline["value"] is None:
                        continue
                    compared += 1
                    left = float(dpa["value"])
                    right = float(baseline["value"])
                    if left == right:
                        ties += 1
                    elif (left > right) == (direction == "higher_better"):
                        wins += 1
                    else:
                        losses += 1
                if compared:
                    metric_result[metric] = {
                        "compared": compared,
                        "dpa_wins": wins,
                        "dpa_losses": losses,
                        "ties": ties,
                    }
            if metric_result:
                arm_result[window] = metric_result
        result[historical] = arm_result
    return result


def _per_seed_deltas(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    index = _row_index(rows)
    deltas: list[dict[str, object]] = []
    for historical in ("pfq_historical", "v9c_historical"):
        for seed in SEEDS:
            for window in ("full_4000", "steady_3000"):
                for metric, (unit, direction) in METRIC_META.items():
                    dpa = index.get(("dpa_v1", seed, window, metric))
                    baseline = index.get((historical, seed, window, metric))
                    if (
                        dpa is None
                        or baseline is None
                        or dpa["value"] is None
                        or baseline["value"] is None
                    ):
                        continue
                    dpa_value = float(dpa["value"])
                    historical_value = float(baseline["value"])
                    delta = dpa_value - historical_value
                    if direction == "higher_better":
                        improved: bool | None = delta > 0.0
                    elif direction == "lower_better":
                        improved = delta < 0.0
                    else:
                        improved = None
                    deltas.append(
                        {
                            "historical_arm": historical,
                            "seed": seed,
                            "window": window,
                            "metric": metric,
                            "dpa_value": dpa_value,
                            "historical_value": historical_value,
                            "dpa_minus_historical": delta,
                            "dpa_over_historical_ratio": (
                                dpa_value / historical_value
                                if historical_value != 0.0
                                else None
                            ),
                            "dpa_improved": improved,
                            "unit": unit,
                            "direction": direction,
                        }
                    )
    return deltas


def _source_check(
    *, source: Path, source_field: str, source_value: object, report_value: object
) -> dict[str, object]:
    return {
        "source_path": str(source.resolve()),
        "source_field": source_field,
        "source_value": source_value,
        "report_value": report_value,
        "exact_match": source_value == report_value,
    }


def _spot_checks(
    *,
    rows: Sequence[Mapping[str, object]],
    dpa_metrics: Mapping[int, Mapping[str, object]],
    pfq_sources: Mapping[int, Path],
    v9c_sources: Mapping[int, Mapping[str, Path]],
) -> list[dict[str, object]]:
    index = _row_index(rows)
    checks: list[dict[str, object]] = []
    for seed in (41, 45, 48):
        dpa_source = Path(dpa_metrics[seed]["source_artifacts"]["eval_summary"])
        dpa_eval = _read_json(dpa_source)
        pfq_source = pfq_sources[seed]
        pfq_payload = _read_json(pfq_source)
        pf_goodput = next(
            entry["pf"]
            for entry in pfq_payload["rr_vs_pf_metrics"]
            if entry["name"] == "traffic.goodput_mbps"
        )
        v9_eval_source = v9c_sources[seed]["eval_summary"]
        v9_eval = _read_json(v9_eval_source)
        v9_log_source = v9c_sources[seed]["checkpoint_eval_log"]
        v9_steady = parse_v9c_steady_blocks(v9_log_source)
        seed_checks = {
            "dpa_full_goodput": _source_check(
                source=dpa_source,
                source_field="mean_goodput_mbps",
                source_value=dpa_eval["mean_goodput_mbps"],
                report_value=index[("dpa_v1", seed, "full_4000", "goodput_mbps")]["value"],
            ),
            "dpa_steady_goodput": _source_check(
                source=dpa_source,
                source_field="stats_mean_goodput_mbps",
                source_value=dpa_eval["stats_mean_goodput_mbps"],
                report_value=index[("dpa_v1", seed, "steady_3000", "goodput_mbps")]["value"],
            ),
            "v9c_full_goodput": _source_check(
                source=v9_eval_source,
                source_field="mean_goodput_mbps",
                source_value=v9_eval["mean_goodput_mbps"],
                report_value=index[("v9c_historical", seed, "full_4000", "goodput_mbps")]["value"],
            ),
            "pfq_full_goodput": _source_check(
                source=pfq_source,
                source_field="rr_vs_pf_metrics[name=traffic.goodput_mbps].pf",
                source_value=pf_goodput,
                report_value=index[("pfq_historical", seed, "full_4000", "goodput_mbps")]["value"],
            ),
            "v9c_steady_goodput": _source_check(
                source=v9_log_source,
                source_field="arithmetic mean of complete blocks ending 1500,2000,2500,3000,3500,4000",
                source_value=v9_steady["goodput_mbps"],
                report_value=index[("v9c_historical", seed, "steady_3000", "goodput_mbps")]["value"],
            ),
        }
        if not all(bool(check["exact_match"]) for check in seed_checks.values()):
            raise ValueError(f"aggregate source spot check failed for seed {seed}")
        checks.append({"seed": seed, "checks": seed_checks})
    return checks


def run_aggregate(args: argparse.Namespace) -> None:
    manifest_path = Path(args.campaign_manifest).resolve()
    campaign_root = Path(args.campaign_root).resolve()
    legacy_root = Path(args.legacy_root).resolve()
    v9c_root = Path(args.v9c_root).resolve()
    pfq_root = Path(args.pfq_root).resolve()
    out_root = Path(args.out_root).resolve()
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError("campaign manifest must be a mapping")
    if [int(seed) for seed in manifest.get("seeds", [])] != list(SEEDS):
        raise ValueError("campaign manifest does not freeze seeds 41--50")
    manifest_hashes = manifest.get("hashes", {})
    frozen_hashes = _frozen_hash_values(manifest_hashes)
    for name in FROZEN_HASHES:
        if name == "actor_state":
            continue
        record = manifest_hashes[name]
        _hash_record(Path(str(record["path"])), frozen_hashes[name])
    actor_checkpoint = Path(str(manifest_hashes["checkpoint_file"]["path"]))
    if _checkpoint_actor_state_sha256(actor_checkpoint) != frozen_hashes["actor_state"]:
        raise ValueError("campaign checkpoint actor state no longer matches the frozen hash")
    for name, path in (
        ("v9c_aggregate_manifest", v9c_root / "aggregate_manifest.csv"),
        ("pfq_aggregate_manifest", pfq_root / "aggregate_manifest.csv"),
        ("pfq_mean_json", pfq_root / "RAYLEIGH" / "rrq_vs_pfq_compare_mean.json"),
    ):
        expected = frozen_hashes[name]
        _hash_record(path, expected)
    v9c_comparison = _find_frozen_file(
        v9c_root,
        "RAYLEIGH/*compare_mean.json",
        frozen_hashes["v9c_comparison_json"],
    )

    dpa_metrics: dict[int, Mapping[str, object]] = {}
    per_seed_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        source = campaign_root / f"seed{seed}" / "seed_metrics.json"
        payload = _read_json(source)
        if not isinstance(payload, Mapping) or payload.get("structural_valid") is not True:
            raise ValueError(f"DPA seed metrics are structurally invalid: {source}")
        if int(payload.get("seed", -1)) != seed:
            raise ValueError(f"DPA seed identity mismatch: {source}")
        dpa_metrics[seed] = payload
        per_seed_rows.extend(_dpa_rows(payload, source.resolve()))
    pfq_rows, pfq_sources = _pfq_rows(pfq_root, SEEDS)
    v9c_rows, v9c_sources = _v9c_rows(v9c_comparison, legacy_root=legacy_root)
    per_seed_rows.extend(pfq_rows)
    per_seed_rows.extend(v9c_rows)
    per_seed_rows.sort(
        key=lambda row: (
            str(row["arm"]),
            int(row["seed"]),
            str(row["window"]),
            str(row["metric"]),
        )
    )
    aggregate = _aggregate_rows(per_seed_rows)

    _write_json(out_root / "per_seed_metrics.json", per_seed_rows)
    _write_csv(out_root / "per_seed_metrics.csv", per_seed_rows, ROW_FIELDS)
    _write_json(out_root / "aggregate_metrics.json", aggregate)
    aggregate_fields = (
        "arm",
        "window",
        "metric",
        "count",
        "mean",
        "std",
        "median",
        "min",
        "max",
        "worst_seed",
        "worst_value",
        "unit",
        "direction",
        "comparison_scope",
    )
    _write_csv(out_root / "aggregate_metrics.csv", aggregate, aggregate_fields)

    comparability = {
        "strict_paired_superiority_claim_allowed": False,
        "full_4000": {
            "dpa_v1_vs_pfq": "same seed cohort and horizon; historical binary comparison",
            "dpa_v1_vs_v9c": "same seed cohort and horizon; historical binary comparison",
        },
        "steady_3000": {
            "dpa_v1_vs_v9c": "same TTI 1001--4000 window; v9c uses six preserved equal 500-TTI blocks",
            "pfq": "not available; PFQ full-4000 values are context only and are not paired with steady results",
        },
        "limitations": [
            "historical PFQ/v9c used an older simulator binary",
            "historical executions lack a complete offered-stream fingerprint",
            "missing historical fields remain null and are not reconstructed from proxies",
        ],
        "source_hashes": {
            name: manifest_hashes[name]
            for name in (
                "v9c_checkpoint",
                "v9c_aggregate_manifest",
                "v9c_comparison_json",
                "pfq_aggregate_manifest",
                "pfq_mean_json",
            )
        },
    }
    _write_json(out_root / "historical_comparability.json", comparability)

    index = _row_index(per_seed_rows)
    service_health = {"zero_service_seed_ues": [], "ttl_streak_seed_ues": []}
    for seed, payload in dpa_metrics.items():
        per_ue = payload["windows"]["full_4000"]["per_ue_service"]
        for ue in per_ue:
            if bool(ue["had_positive_backlog"]) and int(ue["scheduled_dequeue_bytes"]) == 0:
                service_health["zero_service_seed_ues"].append(
                    {"seed": seed, "ue_id": int(ue["ue_id"])}
                )
            if int(ue["max_backlogged_zero_dequeue_streak_tti"]) >= 400:
                service_health["ttl_streak_seed_ues"].append(
                    {
                        "seed": seed,
                        "ue_id": int(ue["ue_id"]),
                        "streak_tti": int(ue["max_backlogged_zero_dequeue_streak_tti"]),
                    }
                )
    dpa_full_mean = _mean_metrics(aggregate, "dpa_v1", "full_4000")
    pfq_full_mean = _mean_metrics(aggregate, "pfq_historical", "full_4000")
    v9c_full_mean = _mean_metrics(aggregate, "v9c_historical", "full_4000")
    headline_metrics = (
        "goodput_mbps",
        "bler",
        "expiry_drop_rate",
        "prg_utilization_ratio",
        "blank_ratio",
        "outcome_delay_mean_ms",
        "outcome_delay_p95_ms",
        "packet_delay_mean_ms",
        "packet_delay_p95_ms",
        "ue_goodput_p10_mbps",
        "ue_goodput_jain",
        "ue_expiry_aware_rate_p10_mbps",
        "max_backlogged_zero_dequeue_streak_tti",
    )
    headline_by_seed: dict[str, object] = {}
    for seed in SEEDS:
        seed_payload: dict[str, object] = {}
        for arm in ("dpa_v1", "pfq_historical", "v9c_historical"):
            arm_payload: dict[str, object] = {}
            for window in ("full_4000", "steady_3000"):
                values = {
                    metric: (
                        index[(arm, seed, window, metric)]["value"]
                        if (arm, seed, window, metric) in index
                        else None
                    )
                    for metric in headline_metrics
                }
                arm_payload[window] = values
            seed_payload[arm] = arm_payload
        headline_by_seed[str(seed)] = seed_payload
    worst_goodput = next(
        row
        for row in aggregate
        if row["arm"] == "dpa_v1"
        and row["window"] == "full_4000"
        and row["metric"] == "goodput_mbps"
    )
    worst_seed = int(worst_goodput["worst_seed"])
    worst_per_ue = dpa_metrics[worst_seed]["windows"]["full_4000"]["per_ue_service"]
    worst_has_starvation = any(
        (bool(ue["had_positive_backlog"]) and int(ue["scheduled_dequeue_bytes"]) == 0)
        or int(ue["max_backlogged_zero_dequeue_streak_tti"]) >= 400
        for ue in worst_per_ue
    )
    worst_mechanism = (
        "starvation"
        if worst_has_starvation
        else (
            "expiry_tail"
            if float(index[("dpa_v1", worst_seed, "full_4000", "expiry_drop_rate")]["value"]) > 0.0
            else "throughput_or_reliability"
        )
    )
    report: dict[str, object] = {
        "model_name": MODEL_NAME,
        "structural_valid": True,
        "strict_paired_superiority_claim_allowed": False,
        "dpa_full_mean": dpa_full_mean,
        "pfq_full_mean": pfq_full_mean,
        "v9c_full_mean": v9c_full_mean,
        "dpa_service_health": service_health,
        "headline_by_seed": headline_by_seed,
        "aggregate_metrics": aggregate,
        "per_seed_deltas": _per_seed_deltas(per_seed_rows),
        "win_counts": _win_counts(per_seed_rows),
        "stress_seed_callouts": {
            str(seed): headline_by_seed[str(seed)] for seed in (43, 45, 48)
        },
        "worst_dpa_topology": {
            "seed": worst_seed,
            "goodput_mbps": worst_goodput["worst_value"],
            "dominant_mechanism": worst_mechanism,
            "headline": headline_by_seed[str(worst_seed)]["dpa_v1"],
        },
        "spot_check_provenance": _spot_checks(
            rows=per_seed_rows,
            dpa_metrics=dpa_metrics,
            pfq_sources=pfq_sources,
            v9c_sources=v9c_sources,
        ),
        "historical_comparability": comparability,
    }
    report["classification"] = classify_scorecard(report)
    _write_json(out_root / "final_report.json", report)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen DPA-v1 seed41--50 historical comparison"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight", help="validate frozen provenance and write campaign manifest"
    )
    preflight.add_argument("--worktree-root", required=True)
    preflight.add_argument("--legacy-root", required=True)
    preflight.add_argument("--checkpoint", required=True)
    preflight.add_argument("--sim-bin", required=True)
    preflight.add_argument("--v9c-root", required=True)
    preflight.add_argument("--pfq-root", required=True)
    preflight.add_argument("--out", required=True)
    preflight.add_argument(
        "--verify-only",
        action="store_true",
        help="recompute frozen provenance and require an unchanged matching manifest",
    )
    preflight.set_defaults(func=run_preflight)

    extract = subparsers.add_parser(
        "extract-seed", help="validate and normalize one DPA seed result"
    )
    extract.add_argument("--campaign-manifest", required=True)
    extract.add_argument("--seed", type=int, required=True)
    extract.add_argument("--actor-dir", required=True)
    extract.add_argument("--out", required=True)
    extract.set_defaults(func=run_extract_seed)

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="assemble normalized historical scorecard reports"
    )
    aggregate_parser.add_argument("--campaign-manifest", required=True)
    aggregate_parser.add_argument("--campaign-root", required=True)
    aggregate_parser.add_argument("--legacy-root", required=True)
    aggregate_parser.add_argument("--v9c-root", required=True)
    aggregate_parser.add_argument("--pfq-root", required=True)
    aggregate_parser.add_argument("--out-root", required=True)
    aggregate_parser.set_defaults(func=run_aggregate)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

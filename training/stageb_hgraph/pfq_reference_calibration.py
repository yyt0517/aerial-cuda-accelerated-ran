from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .bridge_env import BridgeEnvClient, BridgeEnvConfig
from .feature_snapshot import FeatureSnapshotBuilder
from .independent_objectives import (
    ExpiryAwareUeWindowTracker,
    compute_calibration_scales,
    compute_delay_debt_potential,
)
from .online_protocol import VERSION

MEASURED_TERMINAL_TTI_EXCLUSIVE = 1000
FORMAL_WARMUP_STEPS = 1000
FORMAL_MEASURE_STEPS = 4000

_SCENARIO_KEYS = {
    "name",
    "topology_seed",
    "traffic_seed",
    "n_cell",
    "n_ue",
    "n_prg",
    "ue_placement",
    "cell_association",
    "cell_radius_m",
    "prbs_per_prg",
    "allocation_type",
    "precoding",
    "packet_size_bytes",
    "packets_per_ue_tti",
    "packet_ttl_ms",
}

_CANONICAL_DIMS = {"n_cell": 3, "n_active_ue": 36, "n_prg": 17}


def validate_reference_manifest(manifest: Mapping[str, object]) -> None:
    for key in ("binary_sha256", "source_revision", "protocol_version", "scenario"):
        if key not in manifest:
            raise ValueError(f"reference manifest missing {key}")
    if int(manifest["protocol_version"]) != VERSION:
        raise ValueError(f"protocol_version must be {VERSION}")
    revision = str(manifest["source_revision"]).strip()
    if not revision or revision.startswith("ref:"):
        raise ValueError("source_revision must be a resolved commit fingerprint")
    scenario = manifest["scenario"]
    if not isinstance(scenario, Mapping):
        raise ValueError("scenario must be a mapping")
    missing = sorted(_SCENARIO_KEYS.difference(scenario))
    if missing:
        raise ValueError(f"scenario missing keys: {missing}")


def validate_observed_scenario(info: Mapping[str, object]) -> None:
    observed = {key: int(info[key]) for key in _CANONICAL_DIMS}
    if observed != _CANONICAL_DIMS:
        raise RuntimeError(
            f"canonical scenario mismatch: observed={observed} expected={_CANONICAL_DIMS}"
        )


def build_reference_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    warmup_steps: int,
    measure_steps: int,
) -> dict[str, float | int]:
    lower = int(warmup_steps)
    upper = lower + int(measure_steps)
    measured = [row for row in rows if lower < int(row["tti"]) <= upper]
    if len(measured) != int(measure_steps):
        raise ValueError(
            f"measured row count mismatch: got={len(measured)} expected={int(measure_steps)}"
        )
    summary: dict[str, float | int] = {
        "full_row_count": len(rows),
        "warmup_steps": lower,
        "measured_row_count": len(measured),
        "measured_tti_first": int(measured[0]["tti"]),
        "measured_tti_last": int(measured[-1]["tti"]),
    }
    numeric_keys = sorted(
        set.intersection(
            *(set(row.keys()) for row in measured),
        ).difference({"tti"})
    )
    for key in numeric_keys:
        try:
            values = np.asarray([float(row[key]) for row in measured], dtype=np.float64)
        except (TypeError, ValueError):
            continue
        summary[f"{key}_mean"] = float(values.mean())
    full_numeric_keys = sorted(
        set.intersection(
            *(set(row.keys()) for row in rows),
        ).difference({"tti"})
    )
    for key in full_numeric_keys:
        try:
            values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        except (TypeError, ValueError):
            continue
        summary[f"full_{key}_mean"] = float(values.mean())
    return summary


def filter_outcome_samples(
    path: Path,
    *,
    terminal_tti_exclusive: int,
    terminal_tti_inclusive: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            terminal_tti = int(row["terminal_tti"])
            if int(terminal_tti_exclusive) < terminal_tti <= int(terminal_tti_inclusive):
                rows.append(
                    {
                        "ue_id": int(row["ue_id"]),
                        "packet_id": int(row["packet_id"]),
                        "terminal_tti": terminal_tti,
                        "outcome": row["outcome"],
                        "system_time_ms": float(row["system_time_ms"]),
                    }
                )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(repo_root: Path) -> str:
    override = os.environ.get("CUMAC_SOURCE_REVISION", "").strip()
    if override:
        return override
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        # A host-created linked worktree can have an absolute .git pointer that is
        # invalid inside the container mount. Its admin HEAD remains available in
        # the mounted main repository.
        git_pointer = repo_root / ".git"
        common_root = repo_root.parent.parent
        admin_head = common_root / ".git" / "worktrees" / repo_root.name / "HEAD"
        if git_pointer.is_file() and admin_head.is_file():
            revision = admin_head.read_text(encoding="utf-8").strip()
            if revision.startswith("ref:"):
                ref_name = revision[4:].strip()
                ref_path = common_root / ".git" / ref_name
                if ref_path.is_file():
                    revision = ref_path.read_text(encoding="utf-8").strip()
                else:
                    packed_refs = common_root / ".git" / "packed-refs"
                    if packed_refs.is_file():
                        for line in packed_refs.read_text(encoding="utf-8").splitlines():
                            if line and not line.startswith(("#", "^")):
                                commit, packed_ref = line.split(" ", 1)
                                if packed_ref == ref_name:
                                    revision = commit
                                    break
            if revision and not revision.startswith("ref:"):
                return revision
        raise


def _extend_ld_library_path(env: dict[str, str], sim_bin: Path) -> None:
    build_cumac_dir = sim_bin.parents[2]
    lib_dirs = [
        str(path)
        for path in (build_cumac_dir / "src", build_cumac_dir / "lib" / "cumac_msg")
        if path.is_dir()
    ]
    if lib_dirs:
        current = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([current] if current else []))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate PFQ-independent objectives on PFQ pass-through")
    parser.add_argument("--sim-bin", required=True)
    parser.add_argument("--sim-args", default="-d 1 -b 2 -f 0 -x 0 -g 100 -r 3000")
    parser.add_argument("--sim-cwd", default="")
    parser.add_argument("--topology-seed", type=int, default=41)
    parser.add_argument("--traffic-seed", type=int, default=41)
    parser.add_argument("--warmup-steps", type=int, default=FORMAL_WARMUP_STEPS)
    parser.add_argument("--measure-steps", type=int, default=FORMAL_MEASURE_STEPS)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--formal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.formal and (
        int(args.warmup_steps) != FORMAL_WARMUP_STEPS
        or int(args.measure_steps) != FORMAL_MEASURE_STEPS
    ):
        raise ValueError("formal calibration requires exactly 1000 warmup + 4000 measured TTIs")
    total_steps = int(args.warmup_steps) + int(args.measure_steps)
    if total_steps <= 0:
        raise ValueError("calibration horizon must be positive")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_bin = Path(args.sim_bin).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    socket_path = str(Path(tempfile.gettempdir()) / f"cumac_pfq_cal_{os.getpid()}.sock")
    outcome_csv = out_dir / "packet_outcome_samples.csv"
    sim_log = out_dir / "sim.log"
    env = os.environ.copy()
    env.update(
        {
            "CUMAC_ONLINE_BRIDGE": "1",
            "CUMAC_ONLINE_SOCKET": socket_path,
            "CUMAC_ONLINE_PERSISTENT": "1",
            "CUMAC_ONLINE_EXPORT_TEACHER_ACTION": "1",
            "CUMAC_ONLINE_EXPORT_POST_EQ_SINR": "1",
            "CUMAC_PACKET_DELAY_SAMPLES": "1",
            "CUMAC_PACKET_OUTCOME_SAMPLES_CSV": str(outcome_csv),
            "CUMAC_TOPOLOGY_SEED": str(args.topology_seed),
            "CUMAC_TRAFFIC_SEED": str(args.traffic_seed),
            "CUMAC_CELL_ASSOC_MODE": "strongest",
            "CUMAC_UE_PLACEMENT_MODE": "coop_boundary",
            "CUMAC_UE_VORONOI_CLIP": "1",
            "CUMAC_BS_TX_PATTERN": "omni",
            "CUMAC_PACKET_TTL_MS": "200",
            "CUMAC_PACKET_TTL_TTI": "0",
            "CUMAC_TRAFFIC_ARRIVAL_RATE": "1.0",
            "CUMAC_COMPACT_TTI_LOG": "1",
            "CUMAC_PROGRESS_TTI_INTERVAL": "0",
            "CUMAC_COMPARE_TTI_INTERVAL": "0",
            "CUMAC_EXEC_MODE": "both",
            "CUMAC_ONLINE_REWARD_MODE": "goodput_only",
        }
    )
    _extend_ld_library_path(env, sim_bin)
    command = [str(sim_bin), *shlex.split(args.sim_args)]
    rows: list[dict[str, object]] = []
    snapshot_builder = FeatureSnapshotBuilder()
    tracker: ExpiryAwareUeWindowTracker | None = None
    with open(sim_log, "w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            command,
            cwd=args.sim_cwd or str(repo_root),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        client = BridgeEnvClient(BridgeEnvConfig(socket_path=socket_path, connect_timeout_s=60.0))
        try:
            obs, _, info = client.reset(seed=args.traffic_seed, episode_horizon=total_steps)
            if "teacher_action_ue_select" not in obs or "teacher_action_prg_alloc" not in obs:
                raise RuntimeError("PFQ calibration requires teacher actions")
            if "post_eq_sinr" not in obs:
                raise RuntimeError("PFQ calibration requires post-EQ SINR")
            validate_observed_scenario(info)
            tracker = ExpiryAwareUeWindowTracker(int(info["n_active_ue"]), window_steps=400)
            previous_utility = 0.0
            for step in range(total_steps):
                action_ue = np.asarray(obs["teacher_action_ue_select"], dtype=np.int32)
                action_prg = np.asarray(obs["teacher_action_prg_alloc"], dtype=np.int16)
                before = compute_delay_debt_potential(snapshot_builder.build(obs))
                obs, _, done, info = client.step(action_ue, action_prg)
                if not info.get("actual_ue_packet_expiry_aware_diagnostics_available", False):
                    raise RuntimeError("PFQ calibration requires expiry-aware per-UE diagnostics")
                after = compute_delay_debt_potential(snapshot_builder.build(obs))
                demand = np.asarray(obs["obs_ue_features"][:, 0] > 0.0, dtype=bool)
                utility = tracker.update(
                    demand,
                    np.asarray(info["actual_ue_goodput_bytes"], dtype=np.float64),
                    np.asarray(info["actual_ue_packet_system_time_ms"], dtype=np.float64),
                    np.asarray(info["actual_ue_packet_expired_system_time_ms"], dtype=np.float64),
                )
                terms = info["reward_terms"]
                rows.append(
                    {
                        "tti": step + 1,
                        "goodput_bytes": float(np.asarray(info["actual_ue_goodput_bytes"]).sum()),
                        "scheduled_bytes": float(np.asarray(info["actual_ue_served_bytes"]).sum()),
                        "failed_bytes": float(
                            max(
                                0.0,
                                np.asarray(info["actual_ue_served_bytes"]).sum()
                                - np.asarray(info["actual_ue_goodput_bytes"]).sum(),
                            )
                        ),
                        "tb_err_rate": float(terms[2]),
                        "tb_tx_count": float(
                            np.asarray(info["actual_ue_tb_tx_count"], dtype=np.float64).sum()
                        ),
                        "tb_err_count": float(
                            np.asarray(info["actual_ue_tb_err_count"], dtype=np.float64).sum()
                        ),
                        "expired_bytes": float(terms[9]),
                        "raw_delta_d": float(before - 0.99 * after),
                        "raw_delta_u": float(0.99 * utility - previous_utility),
                        "ue_goodput_bytes": np.asarray(
                            info["actual_ue_goodput_bytes"], dtype=np.float64
                        ).tolist(),
                        "ue_delivered_system_time_ms_cumulative": np.asarray(
                            info["actual_ue_packet_system_time_ms"], dtype=np.float64
                        ).tolist(),
                        "ue_expired_system_time_ms_cumulative": np.asarray(
                            info["actual_ue_packet_expired_system_time_ms"], dtype=np.float64
                        ).tolist(),
                        "ue_outcome_system_time_ms_cumulative": np.asarray(
                            info["actual_ue_packet_outcome_system_time_ms"], dtype=np.float64
                        ).tolist(),
                        "ue_expiry_aware_rate_mbps_cumulative": np.asarray(
                            info["actual_ue_packet_expiry_aware_rate_mbps"], dtype=np.float64
                        ).tolist(),
                        "ue_window_successful_goodput_bytes": (
                            tracker.window_successful_goodput_bytes.tolist()
                        ),
                        "ue_window_outcome_system_time_ms": (
                            tracker.window_outcome_system_time_ms.tolist()
                        ),
                    }
                )
                previous_utility = utility
                if done and step + 1 != total_steps:
                    raise RuntimeError("simulator ended before requested calibration horizon")
            client.close()
            return_code = proc.wait(timeout=60)
            if return_code != 0:
                raise RuntimeError(f"simulator exited with code {return_code}")
        finally:
            client.close()
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)

    trace_path = out_dir / "tti_trace.jsonl"
    trace_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    scales = compute_calibration_scales(
        rows,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
    )
    (out_dir / "reward_scales.json").write_text(json.dumps(scales, indent=2), encoding="utf-8")
    summary = build_reference_summary(
        rows,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
    )
    measured_rows = rows[int(args.warmup_steps) : int(args.warmup_steps) + int(args.measure_steps)]
    measured_ue_goodput = np.asarray(
        [row["ue_goodput_bytes"] for row in measured_rows], dtype=np.float64
    ).sum(axis=0)
    delivered_time_start = np.asarray(
        rows[int(args.warmup_steps) - 1]["ue_delivered_system_time_ms_cumulative"]
        if int(args.warmup_steps) > 0
        else np.zeros_like(measured_ue_goodput),
        dtype=np.float64,
    )
    expired_time_start = np.asarray(
        rows[int(args.warmup_steps) - 1]["ue_expired_system_time_ms_cumulative"]
        if int(args.warmup_steps) > 0
        else np.zeros_like(measured_ue_goodput),
        dtype=np.float64,
    )
    delivered_time_end = np.asarray(
        measured_rows[-1]["ue_delivered_system_time_ms_cumulative"], dtype=np.float64
    )
    expired_time_end = np.asarray(
        measured_rows[-1]["ue_expired_system_time_ms_cumulative"], dtype=np.float64
    )
    measured_outcome_time = np.maximum(
        0.0,
        (delivered_time_end - delivered_time_start)
        + (expired_time_end - expired_time_start),
    )
    active_ue = measured_outcome_time > 0.0
    ue_ea_rates = np.zeros_like(measured_ue_goodput)
    ue_ea_rates[active_ue] = (
        measured_ue_goodput[active_ue] * 8.0
        / (measured_outcome_time[active_ue] * 1000.0)
    )
    summary["protocol_version"] = VERSION
    summary["goodput_mbps"] = float(summary["goodput_bytes_mean"]) * 0.016
    measured_tb_tx_count = float(sum(row["tb_tx_count"] for row in measured_rows))
    measured_tb_err_count = float(sum(row["tb_err_count"] for row in measured_rows))
    summary["measured_tb_tx_count"] = measured_tb_tx_count
    summary["measured_tb_err_count"] = measured_tb_err_count
    summary["tb_bler"] = (
        measured_tb_err_count / measured_tb_tx_count
        if measured_tb_tx_count > 0.0
        else None
    )
    summary["expiry_rate"] = float(
        summary["expired_bytes_mean"] / (36.0 * 3000.0)
    )
    summary["ue_macro_ea_rate_mbps"] = (
        float(ue_ea_rates[active_ue].mean()) if np.any(active_ue) else 0.0
    )
    summary["ue_p10_ea_rate_mbps"] = (
        float(np.percentile(ue_ea_rates[active_ue], 10)) if np.any(active_ue) else 0.0
    )
    summary["ue_ea_rate_active_count"] = int(active_ue.sum())
    summary["action_replay_mismatch_count"] = 0.0
    summary["illegal_action_count"] = 0.0
    summary["independent_deletion_equal"] = True
    if outcome_csv.exists():
        measured_outcomes = filter_outcome_samples(
            outcome_csv,
            terminal_tti_exclusive=args.warmup_steps,
            terminal_tti_inclusive=total_steps,
        )
        delays = [float(row["system_time_ms"]) for row in measured_outcomes]
        summary["packet_outcome_measured_count"] = len(delays)
        summary["packet_outcome_delay_p95_ms"] = float(np.percentile(delays, 95)) if delays else 0.0
    summary["outcome_delay_p95_ms"] = summary.get("packet_outcome_delay_p95_ms")
    summary["ea_metrics_available"] = bool(
        summary.get("packet_outcome_measured_count", 0) > 0
        and summary["ue_ea_rate_active_count"] > 0
    )
    (out_dir / "reference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "binary_sha256": _sha256(sim_bin),
        "source_revision": _git_revision(repo_root),
        "protocol_version": VERSION,
        "scenario": {
            "name": "coop_boundary",
            "topology_seed": int(args.topology_seed),
            "traffic_seed": int(args.traffic_seed),
            "n_cell": 3,
            "n_ue": 36,
            "n_prg": 17,
            "ue_placement": "coop_boundary",
            "cell_association": "strongest",
            "cell_radius_m": 500.0,
            "prbs_per_prg": 16,
            "allocation_type": 0,
            "precoding": "svd",
            "packet_size_bytes": 3000,
            "packets_per_ue_tti": 1.0,
            "packet_ttl_ms": 200.0,
        },
        "command": command,
        "warmup_steps": int(args.warmup_steps),
        "measure_steps": int(args.measure_steps),
    }
    validate_reference_manifest(manifest)
    (out_dir / "reference_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

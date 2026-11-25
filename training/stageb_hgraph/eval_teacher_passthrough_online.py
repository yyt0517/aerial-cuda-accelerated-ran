#!/usr/bin/env python3
"""Drive Stage-B online bridge by echoing exported native teacher actions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

from training.stageb_hgraph.bridge_env import BridgeEnvClient, BridgeEnvConfig
from training.stageb_hgraph.reward_utils import extract_reward_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--episode-horizon", type=int, default=4000)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--connect-timeout-s", type=float, default=20.0)
    parser.add_argument("--print-every-steps", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_steps < 1 or args.episode_horizon < 1:
        raise ValueError("episode horizon and max steps must be positive")
    if args.warmup_steps < 0 or args.warmup_steps >= args.max_steps:
        raise ValueError("warmup steps must satisfy 0 <= warmup < max steps")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = BridgeEnvClient(
        BridgeEnvConfig(socket_path=args.socket_path, connect_timeout_s=args.connect_timeout_s)
    )
    rows: list[dict[str, float | int | str]] = []
    try:
        obs, _reward0, _info0 = env.reset(seed=args.seed, episode_horizon=args.episode_horizon)
        for step in range(args.max_steps):
            if "teacher_action_ue_select" not in obs or "teacher_action_prg_alloc" not in obs:
                raise RuntimeError("bridge observation does not contain exported teacher actions")
            action_ue = np.asarray(obs["teacher_action_ue_select"], dtype=np.int32).copy()
            action_prg = np.asarray(obs["teacher_action_prg_alloc"], dtype=np.int16).copy()
            action_hash = hashlib.sha256(action_ue.tobytes() + action_prg.tobytes()).hexdigest()
            next_obs, raw_reward, done, info = env.step(action_ue, action_prg)
            metrics = extract_reward_metrics(info)
            rows.append(
                {
                    "step": step,
                    "tti": int(info.get("tti", step + 1)) - 1,
                    "goodput_mbps": float(metrics.get("goodput_mbps", 0.0)),
                    "tb_err_rate": float(metrics.get("tb_err_rate", 0.0)),
                    "prg_utilization_ratio": float(metrics.get("prg_utilization_ratio", 0.0)),
                    "prg_reuse_ratio": float(metrics.get("prg_reuse_ratio", 0.0)),
                    "action_hash": action_hash,
                }
            )
            if args.print_every_steps > 0 and (step + 1) % args.print_every_steps == 0:
                print(
                    f"[teacher-eval] steps={step + 1} "
                    f"goodput={rows[-1]['goodput_mbps']:.4f} "
                    f"prg_util={rows[-1]['prg_utilization_ratio']:.4f}",
                    flush=True,
                )
            obs = next_obs
            if done:
                break
    finally:
        env.close()

    if len(rows) != args.max_steps:
        raise RuntimeError(f"teacher evaluator completed {len(rows)} steps, expected {args.max_steps}")
    with (args.out_dir / "teacher_eval_steps.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected = rows[args.warmup_steps :]
    summary = {
        "seed": args.seed,
        "completed_steps": len(rows),
        "warmup_steps": args.warmup_steps,
        "stats_count": len(selected),
        "mean_goodput_mbps": statistics.fmean(float(row["goodput_mbps"]) for row in selected),
        "mean_tb_err_rate": statistics.fmean(float(row["tb_err_rate"]) for row in selected),
        "mean_prg_utilization_ratio": statistics.fmean(
            float(row["prg_utilization_ratio"]) for row in selected
        ),
        "mean_prg_reuse_ratio": statistics.fmean(float(row["prg_reuse_ratio"]) for row in selected),
    }
    (args.out_dir / "teacher_eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

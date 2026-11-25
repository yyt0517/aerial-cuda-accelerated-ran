#!/usr/bin/env python3
"""Compare post-action interference logs from paired PFQ and intelligent runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


SEED_RE = re.compile(r"(?:^|[/_])s(?:eed)?(\d+)(?=$|[/_])", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pfq-root", type=Path, required=True)
    parser.add_argument("--ai-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--filename", default="actual_interference_tti.csv")
    parser.add_argument("--warmup-tti", type=int, default=0)
    return parser.parse_args()


def seed_from_path(path: Path) -> int:
    matches = list(SEED_RE.finditer(path.as_posix()))
    if not matches:
        raise ValueError(f"cannot infer seed from path: {path}")
    return int(matches[-1].group(1))


def discover(root: Path, filename: str) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in sorted(root.rglob(filename)):
        seed = seed_from_path(path)
        if seed in found:
            raise ValueError(f"duplicate seed {seed}: {found[seed]} and {path}")
        found[seed] = path
    if not found:
        raise ValueError(f"no {filename} found under {root}")
    return found


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def inr_to_db(inr: float) -> float:
    return 10.0 * math.log10(inr) if inr > 0.0 else float("-inf")


def load_run(path: Path, warmup_tti: int) -> dict[str, float | int | str]:
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["tti"]) >= warmup_tti]
    if not rows:
        raise ValueError(f"no rows remain after warmup={warmup_tti}: {path}")

    tti_means = [float(row["inr_mean_all"]) for row in rows]
    tti_link_p95 = [float(row["inr_p95"]) for row in rows]
    tti_goodput_mbps = [float(row["goodput_mbps"]) for row in rows]
    inr_sum = sum(float(row["inr_sum"]) for row in rows)
    scheduled_links = sum(int(row["scheduled_link_count"]) for row in rows)
    interfered_links = sum(int(row["interfered_link_count"]) for row in rows)
    interfered_inr_sum = sum(
        float(row["inr_mean_interfered"]) * int(row["interfered_link_count"]) for row in rows
    )
    tti_mean_inr = statistics.fmean(tti_means)
    pooled_link_mean_inr = inr_sum / scheduled_links if scheduled_links else 0.0
    return {
        "path": str(path),
        "tti_count": len(rows),
        "scheduled_link_count": scheduled_links,
        "interfered_link_count": interfered_links,
        "interfered_link_ratio": interfered_links / scheduled_links if scheduled_links else 0.0,
        "tti_mean_inr": tti_mean_inr,
        "tti_mean_inr_db": inr_to_db(tti_mean_inr),
        "pooled_link_mean_inr": pooled_link_mean_inr,
        "pooled_link_mean_inr_db": inr_to_db(pooled_link_mean_inr),
        "pooled_interfered_link_mean_inr": interfered_inr_sum / interfered_links if interfered_links else 0.0,
        "pooled_interfered_link_mean_inr_db": inr_to_db(interfered_inr_sum / interfered_links)
        if interfered_links
        else float("-inf"),
        "tti_mean_inr_p50": quantile(tti_means, 0.50),
        "tti_mean_inr_p90": quantile(tti_means, 0.90),
        "tti_mean_inr_p95": quantile(tti_means, 0.95),
        "mean_tti_link_p95_inr": statistics.fmean(tti_link_p95),
        "mean_goodput_mbps": statistics.fmean(tti_goodput_mbps),
    }


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    if args.warmup_tti < 0:
        raise ValueError("--warmup-tti must be non-negative")
    pfq_paths = discover(args.pfq_root, args.filename)
    ai_paths = discover(args.ai_root, args.filename)
    seeds = sorted(set(pfq_paths) & set(ai_paths))
    if not seeds:
        raise ValueError("PFQ and AI roots have no paired seeds")

    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        pfq = load_run(pfq_paths[seed], args.warmup_tti)
        ai = load_run(ai_paths[seed], args.warmup_tti)
        row: dict[str, float | int | str] = {"seed": seed}
        for prefix, metrics in (("pfq", pfq), ("ai", ai)):
            row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        for metric in (
            "tti_mean_inr",
            "tti_mean_inr_db",
            "pooled_link_mean_inr",
            "pooled_link_mean_inr_db",
            "pooled_interfered_link_mean_inr",
            "pooled_interfered_link_mean_inr_db",
            "interfered_link_ratio",
            "tti_mean_inr_p95",
            "mean_tti_link_p95_inr",
            "mean_goodput_mbps",
        ):
            row[f"delta_{metric}"] = float(ai[metric]) - float(pfq[metric])
        rows.append(row)

    metric_names = (
        "tti_mean_inr",
        "tti_mean_inr_db",
        "pooled_link_mean_inr",
        "pooled_link_mean_inr_db",
        "pooled_interfered_link_mean_inr",
        "pooled_interfered_link_mean_inr_db",
        "interfered_link_ratio",
        "tti_mean_inr_p50",
        "tti_mean_inr_p90",
        "tti_mean_inr_p95",
        "mean_tti_link_p95_inr",
        "mean_goodput_mbps",
    )
    summary: dict[str, object] = {
        "paired_seeds": seeds,
        "warmup_tti": args.warmup_tti,
        "pfq_root": str(args.pfq_root),
        "ai_root": str(args.ai_root),
        "metrics": {},
    }
    metrics_out = summary["metrics"]
    assert isinstance(metrics_out, dict)
    for metric in metric_names:
        pfq_values = [float(row[f"pfq_{metric}"]) for row in rows]
        ai_values = [float(row[f"ai_{metric}"]) for row in rows]
        metrics_out[metric] = {
            "pfq": mean_std(pfq_values),
            "ai": mean_std(ai_values),
            "paired_delta_ai_minus_pfq": mean_std([a - p for p, a in zip(pfq_values, ai_values)]),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "actual_interference_paired_seeds.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.output_dir / "actual_interference_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    tti = metrics_out["tti_mean_inr"]
    pooled = metrics_out["pooled_link_mean_inr"]
    pooled_interfered = metrics_out["pooled_interfered_link_mean_inr"]
    ratio = metrics_out["interfered_link_ratio"]
    goodput = metrics_out["mean_goodput_mbps"]
    assert isinstance(tti, dict) and isinstance(pooled, dict) and isinstance(pooled_interfered, dict) and isinstance(ratio, dict) and isinstance(goodput, dict)
    md = [
        "# Stage-B actual post-action interference comparison",
        "",
        f"Paired seeds: {', '.join(map(str, seeds))}; warm-up excluded: {args.warmup_tti} TTI.",
        "",
        "| metric | PFQ mean | AI mean | paired AI - PFQ |",
        "|---|---:|---:|---:|",
        f"| TTI mean INR | {tti['pfq']['mean']:.6f} | {tti['ai']['mean']:.6f} | {tti['paired_delta_ai_minus_pfq']['mean']:.6f} |",
        f"| TTI mean interference (dB over noise) | {10 * math.log10(tti['pfq']['mean']):.6f} | {10 * math.log10(tti['ai']['mean']):.6f} | {10 * math.log10(tti['ai']['mean'] / tti['pfq']['mean']):.6f} |",
        f"| Pooled scheduled-link mean INR | {pooled['pfq']['mean']:.6f} | {pooled['ai']['mean']:.6f} | {pooled['paired_delta_ai_minus_pfq']['mean']:.6f} |",
        f"| Pooled interfered-link mean INR | {pooled_interfered['pfq']['mean']:.6f} | {pooled_interfered['ai']['mean']:.6f} | {pooled_interfered['paired_delta_ai_minus_pfq']['mean']:.6f} |",
        f"| Interfered scheduled-link ratio | {ratio['pfq']['mean']:.6%} | {ratio['ai']['mean']:.6%} | {ratio['paired_delta_ai_minus_pfq']['mean']:.6%} |",
        f"| Goodput (Mbps) | {goodput['pfq']['mean']:.6f} | {goodput['ai']['mean']:.6f} | {goodput['paired_delta_ai_minus_pfq']['mean']:.6f} |",
        "",
        "INR is computed after the scheduling action from cross-cell channel matrices, the actually selected interferer UE precoder, PRG occupancy, and selected layer count.",
    ]
    (args.output_dir / "actual_interference_summary.md").write_text("\n".join(md) + "\n")
    print(json_path)


if __name__ == "__main__":
    main()

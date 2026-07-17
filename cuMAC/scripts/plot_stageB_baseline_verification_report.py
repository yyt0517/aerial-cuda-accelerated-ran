#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASELINES = ("rr", "pf", "rrq", "pfq")
COLORS = {"rr": "#2563EB", "pf": "#D97706", "rrq": "#059669", "pfq": "#7C3AED"}
SVG_HASHSALT = "stageB-baseline-verification-v1"
SAVEFIG_METADATA = {"Creator": "Stage-B baseline verification plotter", "Date": None}
matplotlib.rcParams["svg.hashsalt"] = SVG_HASHSALT
REQUIRED_METRICS = (
    "traffic.goodput_mbps",
    "traffic.packet_delay_p95_ms",
    "traffic.expiry_drop_rate",
    "global_kpi.ue_goodput_jain",
    "global_kpi.global_tb_bler",
    "global_kpi.prg_utilization_ratio",
    "global_kpi.ue_goodput_p5_mbps",
)


def _parse_seed_list(raw: str, context: str) -> tuple[int, ...]:
    tokens = raw.split("|")
    if not raw or any(not token or token.strip() != token for token in tokens):
        raise ValueError(f"{context} must contain pipe-delimited integer seeds")
    try:
        seeds = tuple(int(token) for token in tokens)
    except ValueError as error:
        raise ValueError(f"{context} must contain pipe-delimited integer seeds") from error
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{context} seeds must be unique")
    return seeds


def load_metric_table(
    path: Path,
    expected_baselines: tuple[str, ...] = BASELINES,
    expected_seed_count: int = 10,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing aggregate metrics CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"metric", "unit", "direction", "common_seed_count", "common_seeds"}
        for baseline in expected_baselines:
            required_columns.update(
                {
                    f"{baseline}_mean",
                    f"{baseline}_std",
                    f"{baseline}_seed_count",
                    f"{baseline}_seeds",
                }
            )
        missing = sorted(required_columns - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")

        table = {}
        for row in reader:
            metric = row["metric"]
            if metric not in REQUIRED_METRICS:
                continue
            common_seed_count = int(row["common_seed_count"])
            if common_seed_count != expected_seed_count:
                raise ValueError(f"Metric {metric} does not contain {expected_seed_count} seeds")
            common_seeds = _parse_seed_list(row["common_seeds"], f"Metric {metric}/common")
            if len(common_seeds) != common_seed_count:
                raise ValueError(
                    f"Metric {metric}/common seed list does not match common_seed_count"
                )
            values = {"unit": row["unit"], "direction": row["direction"]}
            for baseline in expected_baselines:
                if int(row[f"{baseline}_seed_count"]) != expected_seed_count:
                    raise ValueError(
                        f"Metric {metric}/{baseline} does not contain {expected_seed_count} seeds"
                    )
                baseline_seeds = _parse_seed_list(
                    row[f"{baseline}_seeds"], f"Metric {metric}/{baseline}"
                )
                if baseline_seeds != common_seeds:
                    raise ValueError(
                        f"Metric {metric}/{baseline} must contain the same seeds as common_seeds"
                    )
                scale = 100.0 if row["unit"] == "%" else 1.0
                mean = float(row[f"{baseline}_mean"]) * scale
                std = float(row[f"{baseline}_std"]) * scale
                if not math.isfinite(mean) or not math.isfinite(std):
                    raise ValueError(
                        f"Metric {metric}/{baseline} must contain finite mean/std values"
                    )
                if std < 0:
                    raise ValueError(
                        f"Metric {metric}/{baseline} std must be non-negative"
                    )
                values[baseline] = {"mean": mean, "std": std}
            if metric in table:
                raise ValueError(f"Duplicate metric row: {metric}")
            table[metric] = values

    missing_metrics = sorted(set(REQUIRED_METRICS) - set(table))
    if missing_metrics:
        raise ValueError(f"Missing required metrics: {', '.join(missing_metrics)}")
    return table


def beneficial_relative_change(old: float, new: float, direction: str) -> float:
    if old == 0:
        raise ValueError("Cannot calculate relative change from a zero baseline")
    raw = (new - old) / abs(old) * 100.0
    if direction == "higher_better":
        return raw
    if direction == "lower_better":
        return -raw
    raise ValueError(f"Unsupported direction: {direction}")


def _save_both(fig: plt.Figure, stem: Path, dpi: int) -> list[Path]:
    outputs = [stem.with_suffix(".png"), stem.with_suffix(".svg")]
    fig.tight_layout()
    fig.savefig(
        outputs[0], dpi=dpi, bbox_inches="tight", metadata=SAVEFIG_METADATA
    )
    svg_buffer = io.StringIO()
    fig.savefig(
        svg_buffer,
        format="svg",
        bbox_inches="tight",
        metadata=SAVEFIG_METADATA,
    )
    normalized_svg = "\n".join(
        line.rstrip() for line in svg_buffer.getvalue().splitlines()
    )
    outputs[1].write_text(f"{normalized_svg}\n", encoding="utf-8")
    plt.close(fig)
    return outputs


def _plot_four_baselines(ax: plt.Axes, metric: dict, title: str) -> None:
    positions = np.arange(len(BASELINES))
    means = [metric[baseline]["mean"] for baseline in BASELINES]
    errors = [metric[baseline]["std"] for baseline in BASELINES]
    ax.bar(
        positions,
        means,
        yerr=errors,
        capsize=4,
        color=[COLORS[baseline] for baseline in BASELINES],
        edgecolor="white",
    )
    ax.set_xticks(positions, [baseline.upper() for baseline in BASELINES])
    ax.set_ylabel(metric["unit"] or "Score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def _plot_core_kpis(metrics: dict, stem: Path, dpi: int) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    plots = (
        ("traffic.goodput_mbps", "Aggregate goodput"),
        ("traffic.packet_delay_p95_ms", "Packet delay (p95)"),
        ("traffic.expiry_drop_rate", "Expiry drop rate"),
        ("global_kpi.ue_goodput_jain", "UE goodput fairness (Jain)"),
    )
    for ax, (metric_name, title) in zip(axes.flat, plots):
        _plot_four_baselines(ax, metrics[metric_name], title)
    fig.suptitle("Uniform scenario: core KPIs")
    return _save_both(fig, stem, dpi)


def _plot_resource_tradeoffs(metrics: dict, stem: Path, dpi: int) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    _plot_four_baselines(axes[0], metrics["global_kpi.global_tb_bler"], "Global TB BLER")
    _plot_four_baselines(
        axes[1], metrics["global_kpi.prg_utilization_ratio"], "PRG utilization"
    )
    fig.suptitle("Uniform scenario: resource trade-offs")
    return _save_both(fig, stem, dpi)


def _plot_queue_aware_improvements(metrics: dict, stem: Path, dpi: int) -> list[Path]:
    metric_specs = (
        ("traffic.goodput_mbps", "Goodput"),
        ("traffic.packet_delay_p95_ms", "P95 delay"),
        ("traffic.expiry_drop_rate", "Expiry rate"),
        ("global_kpi.ue_goodput_p5_mbps", "UE p5 goodput"),
    )
    comparisons = (("rr", "rrq", "RR to RRQ"), ("pf", "pfq", "PF to PFQ"))
    positions = np.arange(len(metric_specs))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for index, (old, new, label) in enumerate(comparisons):
        changes = [
            beneficial_relative_change(
                metrics[metric_name][old]["mean"],
                metrics[metric_name][new]["mean"],
                metrics[metric_name]["direction"],
            )
            for metric_name, _ in metric_specs
        ]
        ax.bar(
            positions + (index - 0.5) * width,
            changes,
            width,
            label=label,
            color=COLORS[new],
        )
    ax.axhline(0.0, color="#374151", linewidth=0.8)
    ax.set_xticks(positions, [label for _, label in metric_specs])
    ax.set_ylabel("Beneficial relative change (%)")
    ax.set_title("Queue-aware improvement over matching baseline")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend()
    return _save_both(fig, stem, dpi)


def generate_figures(metrics: dict, output_dir: Path, dpi: int = 220) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    outputs += _plot_core_kpis(metrics, output_dir / "uniform_core_kpis", dpi)
    outputs += _plot_resource_tradeoffs(
        metrics, output_dir / "uniform_resource_tradeoffs", dpi
    )
    outputs += _plot_queue_aware_improvements(
        metrics, output_dir / "queue_aware_improvements", dpi
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Stage-B baseline verification report figures."
    )
    parser.add_argument(
        "--suite-dir", default="output/stageB_baseline_eval_suite_fullkpi_seed41_50"
    )
    parser.add_argument(
        "--output-dir", default="Doc/figures/stageB_baseline_verification"
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.suite_dir) / "suite_aggregate/uniform_four_baseline_metrics.csv"
    table = load_metric_table(csv_path, BASELINES, expected_seed_count=10)
    for output in generate_figures(table, Path(args.output_dir), dpi=args.dpi):
        print(output)


if __name__ == "__main__":
    main()

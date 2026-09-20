from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from training.stageb_hgraph.dpa_v1_historical_comparison import WindowSpec
from training.stageb_hgraph.pfq_dpa_aligned_comparison import (
    aggregate_paired_results,
    summarize_aligned_window,
    validate_seed_artifacts,
)


def _service(
    tti: int,
    ue_id: int,
    *,
    offered: int = 3000,
    backlog: int = 3000,
    scheduled: int = 0,
    successful: int = 0,
    expired: int = 0,
    prgs: int = 0,
    tb_tx: int = 0,
    tb_err: int = 0,
) -> dict[str, int]:
    return {
        "tti": tti,
        "ue_id": ue_id,
        "offered_bytes": offered,
        "pre_queue_bytes": backlog,
        "scheduled_dequeue_bytes": scheduled,
        "post_queue_bytes": max(0, backlog - scheduled - expired),
        "successful_bytes": successful,
        "failed_bytes": scheduled - successful,
        "expired_bytes": expired,
        "actual_prg_count": prgs,
        "tb_tx_count": tb_tx,
        "tb_err_count": tb_err,
    }


def test_shared_window_counts_expiry_in_every_headline_delay_and_rate():
    service_rows = [
        _service(
            1,
            0,
            scheduled=3000,
            successful=3000,
            prgs=1,
            tb_tx=1,
        ),
        _service(1, 1),
        _service(1, 2),
        _service(2, 0, backlog=0),
        _service(2, 1, expired=3000),
        _service(2, 2),
    ]
    outcomes = [
        {
            "terminal_tti": 1,
            "ue_id": 0,
            "outcome": "delivered",
            "packet_size_bytes": 3000,
            "system_time_ms": 1.0,
        },
        {
            "terminal_tti": 2,
            "ue_id": 1,
            "outcome": "expired",
            "packet_size_bytes": 3000,
            "system_time_ms": 200.0,
        },
    ]

    result = summarize_aligned_window(
        service_rows=service_rows,
        outcome_rows=outcomes,
        decision_rows=[
            {"tti": 1, "prg_reuse_ratio": 0.25},
            {"tti": 2, "prg_reuse_ratio": 0.0},
        ],
        window=WindowSpec("full_4000", 1, 2),
    )

    assert result["goodput_mbps"] == 24.0
    assert result["served_mbps"] == 24.0
    assert result["tb_bler"] == 0.0
    assert result["expiry_drop_rate"] == pytest.approx(1.0 / 6.0)
    assert result["outcome_delay_mean_ms"] == 100.5
    assert result["outcome_delay_p95_ms"] == pytest.approx(190.05)
    assert result["ue_expiry_aware_rate_mean_mbps"] == pytest.approx(8.0)
    assert result["ue_expiry_aware_rate_p10_mbps"] == 0.0
    assert result["ue_expiry_aware_rate_min_mbps"] == 0.0
    assert result["ue_expiry_aware_rate_no_outcome_ue_count"] == 1
    assert result["prg_reuse_ratio"] == 0.125
    assert not any(
        name.startswith("packet_delay_")
        or "packet_effective_service_rate" in name
        for name in result
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _minimal_aligned_dirs(tmp_path: Path) -> tuple[Path, Path]:
    dpa = tmp_path / "dpa"
    pfq = tmp_path / "pfq"
    dpa.mkdir()
    pfq.mkdir()
    _write_json(
        dpa / "eval_summary.json",
        {"sim_binary_sha256": "sim-hash", "completed_steps": 2},
    )
    _write_json(
        pfq / "reference_manifest.json",
        {
            "binary_sha256": "sim-hash",
            "warmup_steps": 0,
            "measure_steps": 2,
            "scenario": {"topology_seed": 41, "traffic_seed": 41},
        },
    )
    _write_jsonl(dpa / "eval_steps.jsonl", [{"tti": 1}, {"tti": 2}])
    _write_jsonl(pfq / "tti_trace.jsonl", [{"tti": 1}, {"tti": 2}])
    service = [
        _service(tti, ue_id)
        for tti in (1, 2)
        for ue_id in (0, 1)
    ]
    _write_csv(dpa / "service_balance_trace.csv", service)
    _write_csv(pfq / "service_balance_trace.csv", service)
    outcomes = [
        {
            "ue_id": 0,
            "packet_id": 1,
            "terminal_tti": 1,
            "outcome": "delivered",
            "packet_size_bytes": 3000,
            "system_time_ms": 1.0,
        }
    ]
    _write_csv(dpa / "packet_outcome_samples.csv", outcomes)
    _write_csv(pfq / "packet_outcome_samples.csv", outcomes)
    (dpa / "traffic_offered_packets.csv").write_text("same-stream\n", encoding="utf-8")
    (pfq / "traffic_offered_packets.csv").write_text("same-stream\n", encoding="utf-8")
    (dpa / "ue_topology.csv").write_text("same-topology\n", encoding="utf-8")
    (pfq / "ue_topology.csv").write_text("same-topology\n", encoding="utf-8")
    return dpa, pfq


def test_alignment_requires_same_binary_stream_topology_and_exact_coverage(tmp_path: Path):
    dpa, pfq = _minimal_aligned_dirs(tmp_path)

    report = validate_seed_artifacts(
        seed=41,
        dpa_dir=dpa,
        pfq_dir=pfq,
        expected_sim_sha256="sim-hash",
        expected_tti_count=2,
        expected_ue_count=2,
    )

    assert report["aligned"] is True
    assert all(report["checks"].values())

    (pfq / "traffic_offered_packets.csv").write_text("different-stream\n", encoding="utf-8")
    with pytest.raises(ValueError, match="offered_stream_hash"):
        validate_seed_artifacts(
            seed=41,
            dpa_dir=dpa,
            pfq_dir=pfq,
            expected_sim_sha256="sim-hash",
            expected_tti_count=2,
            expected_ue_count=2,
        )


def test_aggregate_is_paired_and_excludes_served_only_headline_metrics():
    seed_payloads = [
        {
            "seed": 41,
            "windows": {
                "full_4000": {
                    "dpa_v1": {
                        "goodput_mbps": 110.0,
                        "outcome_delay_mean_ms": 2.0,
                        "ue_expiry_aware_rate_mean_mbps": 8.0,
                    },
                    "pfq": {
                        "goodput_mbps": 100.0,
                        "outcome_delay_mean_ms": 3.0,
                        "ue_expiry_aware_rate_mean_mbps": 7.0,
                    },
                }
            },
        },
        {
            "seed": 42,
            "windows": {
                "full_4000": {
                    "dpa_v1": {
                        "goodput_mbps": 90.0,
                        "outcome_delay_mean_ms": 5.0,
                        "ue_expiry_aware_rate_mean_mbps": 6.0,
                    },
                    "pfq": {
                        "goodput_mbps": 100.0,
                        "outcome_delay_mean_ms": 4.0,
                        "ue_expiry_aware_rate_mean_mbps": 7.0,
                    },
                }
            },
        },
    ]

    report = aggregate_paired_results(seed_payloads)

    goodput = report["aggregate"]["full_4000"]["goodput_mbps"]
    assert goodput["dpa_v1"]["mean"] == 100.0
    assert goodput["pfq"]["mean"] == 100.0
    assert goodput["paired_delta_dpa_minus_pfq"]["mean"] == 0.0
    assert goodput["wins"] == {"dpa_v1": 1, "tie": 0, "pfq": 1}
    headline = report["headline_metrics"]
    assert "outcome_delay_mean_ms" in headline
    assert "ue_expiry_aware_rate_mean_mbps" in headline
    assert not any(
        name.startswith("packet_delay_")
        or "packet_effective_service_rate" in name
        for name in headline
    )

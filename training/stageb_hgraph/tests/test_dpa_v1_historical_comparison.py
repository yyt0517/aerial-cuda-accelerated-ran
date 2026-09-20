import csv
import json
import math
from pathlib import Path

import pytest

from training.stageb_hgraph import dpa_v1_historical_comparison as comparison
from training.stageb_hgraph.dpa_v1_historical_comparison import (
    WindowSpec,
    _aggregate_rows,
    classify_scorecard,
    materialize_historical_grid,
    parse_v9c_steady_blocks,
    resolve_recorded_path,
    summarize_dpa_window,
    validate_seed_summary,
)


def test_preflight_verify_only_checks_manifest_without_rewriting(tmp_path, monkeypatch):
    comparison_path = tmp_path / "v9c_compare.json"
    comparison_path.write_text(
        json.dumps(
            {
                "source_runs": [
                    {"seed": seed, "model_path": "/recorded/v9c_checkpoint.pt"}
                    for seed in range(41, 51)
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        comparison,
        "_find_frozen_file",
        lambda *_args: comparison_path,
    )
    monkeypatch.setattr(
        comparison,
        "resolve_recorded_path",
        lambda *_args, **_kwargs: tmp_path / "v9c_checkpoint.pt",
    )
    monkeypatch.setattr(
        comparison,
        "_hash_record",
        lambda path, expected: {
            "path": str(Path(path).resolve()),
            "sha256": expected,
            "expected_sha256": expected,
        },
    )
    monkeypatch.setattr(
        comparison,
        "_checkpoint_actor_state_sha256",
        lambda _path: comparison.FROZEN_HASHES["actor_state"],
    )

    out = tmp_path / "campaign_manifest.json"
    shared = [
        "--worktree-root",
        str(tmp_path / "worktree"),
        "--legacy-root",
        str(tmp_path / "legacy"),
        "--checkpoint",
        str(tmp_path / "actor.pt"),
        "--sim-bin",
        str(tmp_path / "simulator"),
        "--v9c-root",
        str(tmp_path / "v9c"),
        "--pfq-root",
        str(tmp_path / "pfq"),
        "--out",
        str(out),
    ]
    parser = comparison._build_parser()
    comparison.run_preflight(parser.parse_args(["preflight", *shared]))
    before_contents = out.read_text(encoding="utf-8")
    before_mtime_ns = out.stat().st_mtime_ns

    verify_args = parser.parse_args(["preflight", "--verify-only", *shared])
    comparison.run_preflight(verify_args)
    assert out.read_text(encoding="utf-8") == before_contents
    assert out.stat().st_mtime_ns == before_mtime_ns

    out.unlink()
    with pytest.raises(ValueError, match="missing or not a file"):
        comparison.run_preflight(verify_args)

    out.write_text("not JSON", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        comparison.run_preflight(verify_args)

    out.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match recomputed manifest"):
        comparison.run_preflight(verify_args)


def _service(tti, ue_id, backlog, scheduled, successful, expired=0, prgs=0):
    return {
        "tti": tti,
        "ue_id": ue_id,
        "pre_queue_bytes": backlog,
        "scheduled_dequeue_bytes": scheduled,
        "successful_bytes": successful,
        "failed_bytes": scheduled - successful,
        "expired_bytes": expired,
        "actual_prg_count": prgs,
    }


def test_steady_window_includes_expired_outcomes_and_detects_zero_service():
    rows = [
        _service(1, 0, 3000, 3000, 3000, prgs=1),
        _service(1, 1, 3000, 0, 0),
        _service(2, 0, 3000, 3000, 3000, prgs=1),
        _service(2, 1, 3000, 0, 0),
        _service(3, 0, 3000, 0, 0),
        _service(3, 1, 6000, 0, 0, expired=3000),
        _service(4, 0, 3000, 3000, 3000, prgs=1),
        _service(4, 1, 6000, 0, 0),
    ]
    outcomes = [
        {"terminal_tti": 2, "ue_id": 0, "outcome": "delivered", "packet_size_bytes": 3000, "system_time_ms": 1.0},
        {"terminal_tti": 3, "ue_id": 0, "outcome": "expired", "packet_size_bytes": 3000, "system_time_ms": 200.0},
        {"terminal_tti": 3, "ue_id": 1, "outcome": "expired", "packet_size_bytes": 3000, "system_time_ms": 200.0},
    ]
    summary = {
        "mean_goodput_mbps": 100.0,
        "mean_tb_err_rate": 0.20,
        "mean_expiry_drop_rate": 0.10,
        "mean_prg_utilization_ratio": 0.80,
        "mean_blank_ratio": 0.20,
        "stats_mean_goodput_mbps": 123.0,
        "stats_mean_tb_err_rate": 0.15,
        "stats_mean_expiry_drop_rate": 0.08,
        "stats_mean_prg_utilization_ratio": 0.75,
        "stats_mean_blank_ratio": 0.25,
    }

    result = summarize_dpa_window(
        eval_summary=summary,
        service_rows=rows,
        outcome_rows=outcomes,
        window=WindowSpec("steady_3000", 2, 4),
        slot_duration_s=0.0005,
    )

    assert result["tti_count"] == 3
    assert result["goodput_mbps"] == 123.0
    assert result["expired_bytes"] == 3000
    assert result["outcome_count"] == 3
    assert math.isclose(result["outcome_delay_mean_ms"], 401.0 / 3.0)
    assert result["outcome_delay_p50_ms"] == 200.0
    assert result["outcome_delay_p95_ms"] == 200.0
    assert result["outcome_delay_max_ms"] == 200.0
    assert result["zero_scheduled_dequeue_ue_ids"] == [1]
    assert result["max_backlogged_zero_dequeue_streak_tti"] == 3
    assert result["ue_goodput_min_mbps"] == 0.0
    assert result["ue_goodput_mean_mbps"] == 16.0
    assert result["ue_goodput_jain"] == 0.5
    assert result["ue_expiry_aware_rate_min_mbps"] == 0.0


def test_expiry_aware_rate_includes_all_active_ues():
    rows = [
        _service(1, 0, 3000, 3000, 3000, prgs=1),
        _service(1, 1, 3000, 0, 0),
        _service(1, 2, 3000, 0, 0),
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
            "terminal_tti": 1,
            "ue_id": 1,
            "outcome": "expired",
            "packet_size_bytes": 3000,
            "system_time_ms": 200.0,
        },
    ]
    summary = {
        "mean_goodput_mbps": 24.0,
        "mean_tb_err_rate": 0.0,
        "mean_expiry_drop_rate": 1.0 / 3.0,
        "mean_prg_utilization_ratio": 1.0 / 51.0,
        "mean_blank_ratio": 50.0 / 51.0,
    }

    result = summarize_dpa_window(
        eval_summary=summary,
        service_rows=rows,
        outcome_rows=outcomes,
        window=WindowSpec("full_4000", 1, 1),
        slot_duration_s=0.0005,
    )

    assert result["ue_expiry_aware_rate_mean_mbps"] == pytest.approx(8.0)
    assert result["ue_expiry_aware_rate_p10_mbps"] == 0.0
    assert result["ue_expiry_aware_rate_min_mbps"] == 0.0
    assert result["ue_expiry_aware_rate_ue_count"] == 3
    assert result["ue_expiry_aware_rate_no_outcome_ue_count"] == 1


def test_resolve_recorded_container_path_under_supplied_legacy_root(tmp_path):
    target = tmp_path / "output" / "run" / "kpi_summary.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}\n", encoding="utf-8")
    assert resolve_recorded_path(
        "/opt/nvidia/cuBB/output/run/kpi_summary.json",
        legacy_root=tmp_path,
    ) == target


def test_v9c_steady_parser_drops_the_first_two_500_tti_blocks(tmp_path):
    log = tmp_path / "checkpoint_eval.log"
    log.write_text(
        "\n".join(
            f"[hgraph-eval] tti={tti:04d} goodput={1000 + tti / 10:.4f} "
            f"tb_err=0.1000 blank=0.2000 fill=0.8000 uniq_ue=30.00"
            for tti in range(500, 4001, 500)
        ) + "\n",
        encoding="utf-8",
    )
    result = parse_v9c_steady_blocks(log, first_tti=1001, last_tti=4000)
    assert result["block_count"] == 6
    assert result["source_block_end_ttis"] == [1500, 2000, 2500, 3000, 3500, 4000]
    assert result["goodput_mbps"] == 1275.0
    assert result["tb_err_rate"] == 0.1
    assert result["prg_utilization_ratio"] == 0.8


def test_classification_prioritizes_starvation_over_mean_goodput_gain():
    report = {
        "structural_valid": True,
        "dpa_full_mean": {"goodput_mbps": 1600.0, "bler": 0.10, "expiry_drop_rate": 0.0},
        "pfq_full_mean": {"goodput_mbps": 1500.0, "bler": 0.11, "expiry_drop_rate": 0.001},
        "v9c_full_mean": {"goodput_mbps": 1490.0},
        "dpa_service_health": {"zero_service_seed_ues": [{"seed": 45, "ue_id": 4}], "ttl_streak_seed_ues": []},
    }
    assert classify_scorecard(report) == "cross_layout_service_failed_starvation"


def test_seed_summary_requires_the_frozen_structural_contract():
    summary = {
        "completed_steps": 4000,
        "stats_warmup_steps": 1000,
        "stats_count": 3000,
        "checkpoint_sha256": "checkpoint",
        "actor_state_dict_sha256": "actor-state",
        "sim_binary_sha256": "simulator",
        "decode_mode": "argmax",
        "independent_headroom_accounting": "ctbs-v1",
        "independent_non_starvation_guard": "off",
        "candidate_truncation": "forbidden",
        "candidate_width_policy": "runtime_topology",
        "topology_candidate_width": 16,
        "runtime_candidate_width": 16,
        "candidate_truncation_count": 0,
        "stats_sum_candidate_truncation_count": 0.0,
        "stats_sum_action_replay_mismatch_count": 0.0,
        "stats_sum_illegal_action_count": 0.0,
        "ctbs_mask_reactivation_count": 0,
        "ctbs_mcs_mismatch_count": 0,
        "ctbs_layer_mismatch_count": 0,
        "ctbs_tbs_mismatch_count": 0,
        "ctbs_mask_trigger_count_by_local_prg": [0] * 17,
        "ctbs_selected_count_by_local_prg": [0] * 17,
        "ctbs_valid_candidate_count_by_local_prg": [0] * 17,
    }
    observed = validate_seed_summary(
        summary,
        independent_runtime_candidate_width_request=0,
        expected_hashes={
            "checkpoint_file": "checkpoint",
            "actor_state": "actor-state",
            "simulator": "simulator",
        },
    )
    assert all(observed.values())

    summary["independent_non_starvation_guard"] = "ctbs-blank-v1"
    with pytest.raises(ValueError, match="guard_off"):
        validate_seed_summary(
            summary,
            independent_runtime_candidate_width_request=0,
            expected_hashes={
                "checkpoint_file": "checkpoint",
                "actor_state": "actor-state",
                "simulator": "simulator",
            },
        )


def test_seed_summary_rejects_padded_width_and_non_17_prg_evidence():
    summary = {
        "completed_steps": 4000,
        "stats_warmup_steps": 1000,
        "stats_count": 3000,
        "checkpoint_sha256": "checkpoint",
        "actor_state_dict_sha256": "actor-state",
        "sim_binary_sha256": "simulator",
        "decode_mode": "argmax",
        "independent_headroom_accounting": "ctbs-v1",
        "independent_non_starvation_guard": "off",
        "candidate_truncation": "forbidden",
        "candidate_width_policy": "runtime_topology",
        "topology_candidate_width": 16,
        "runtime_candidate_width": 17,
        "candidate_truncation_count": 0,
        "stats_sum_candidate_truncation_count": 0.0,
        "stats_sum_action_replay_mismatch_count": 0.0,
        "stats_sum_illegal_action_count": 0.0,
        "ctbs_mask_reactivation_count": 0,
        "ctbs_mcs_mismatch_count": 0,
        "ctbs_layer_mismatch_count": 0,
        "ctbs_tbs_mismatch_count": 0,
        "ctbs_mask_trigger_count_by_local_prg": [0] * 17,
        "ctbs_selected_count_by_local_prg": [0] * 17,
        "ctbs_valid_candidate_count_by_local_prg": [0] * 17,
    }
    hashes = {
        "checkpoint_file": "checkpoint",
        "actor_state": "actor-state",
        "simulator": "simulator",
    }
    with pytest.raises(ValueError, match="candidate_width_topology_resolved"):
        validate_seed_summary(
            summary,
            independent_runtime_candidate_width_request=0,
            expected_hashes=hashes,
        )

    summary["runtime_candidate_width"] = 16
    summary["ctbs_mask_trigger_count_by_local_prg"] = [0] * 16
    with pytest.raises(ValueError, match="prg_decision_count_17"):
        validate_seed_summary(
            summary,
            independent_runtime_candidate_width_request=0,
            expected_hashes=hashes,
        )

    summary["ctbs_mask_trigger_count_by_local_prg"] = [0] * 17
    with pytest.raises(ValueError, match="candidate_width_request_zero"):
        validate_seed_summary(
            summary,
            independent_runtime_candidate_width_request=17,
            expected_hashes=hashes,
        )


def test_historical_grid_serializes_unavailable_values_as_null_and_csv_empty(tmp_path):
    rows = [
        {
            "arm": "pfq_historical",
            "seed": 41,
            "window": "full_4000",
            "metric": "goodput_mbps",
            "value": 1500.0,
            "unit": "Mbps",
            "direction": "higher_better",
            "source_artifact": "/source/pfq.json",
            "comparison_scope": "historical_same_seed_cohort_full_4000",
        },
        {
            "arm": "pfq_historical",
            "seed": 41,
            "window": "full_4000",
            "metric": "bler",
            "value": None,
            "unit": "ratio",
            "direction": "lower_better",
            "source_artifact": "/source/pfq.json",
            "comparison_scope": "historical_same_seed_cohort_full_4000",
        }
    ]
    materialized = materialize_historical_grid(
        rows,
        arm="pfq_historical",
        seeds=[41],
        source_notes={
            "full_4000": "unavailable: not present in PFQ JSON",
            "steady_3000": "unavailable: no preserved PFQ steady trace",
        },
    )
    missing = next(
        row
        for row in materialized
        if row["window"] == "steady_3000" and row["metric"] == "goodput_mbps"
    )
    assert missing["value"] is None
    assert missing["comparison_scope"] == "unavailable_historical_metric"
    assert missing["source_artifact"] == "unavailable: no preserved PFQ steady trace"

    unavailable_full = next(
        row
        for row in materialized
        if row["window"] == "full_4000" and row["metric"] == "bler"
    )
    assert unavailable_full["value"] is None
    assert unavailable_full["comparison_scope"] == "unavailable_historical_metric"
    assert unavailable_full["source_artifact"] == "unavailable: not present in PFQ JSON"

    json_path = tmp_path / "rows.json"
    json_path.write_text(json.dumps(materialized), encoding="utf-8")
    assert next(
        row
        for row in json.loads(json_path.read_text(encoding="utf-8"))
        if row["window"] == "steady_3000" and row["metric"] == "goodput_mbps"
    )["value"] is None

    csv_path = tmp_path / "rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(materialized)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        serialized = list(csv.DictReader(handle))
    assert next(
        row
        for row in serialized
        if row["window"] == "steady_3000" and row["metric"] == "goodput_mbps"
    )["value"] == ""

    aggregate = next(
        row
        for row in _aggregate_rows(materialized)
        if row["window"] == "steady_3000" and row["metric"] == "goodput_mbps"
    )
    assert aggregate["count"] == 0
    assert aggregate["mean"] is None

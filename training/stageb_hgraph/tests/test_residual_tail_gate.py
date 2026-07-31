import csv
import json
import subprocess
import sys

import pytest

from training.stageb_hgraph.residual_tail_gate import evaluate_residual_tail_gate


def _summary():
    return {
        "iterations": 2,
        "completed_iterations": 2,
        "candidate_policy_mode": "direct",
        "pfq_residual_init": True,
        "ue_feat_dim": 16,
        "candidate_pfq_anchor_logit_coef": 1.0,
        "candidate_blank_logit_bias": -4.0,
        "candidate_success_logit_scale": 0.0,
        "candidate_blank_success_scale": 0.0,
        "queue_pressure_weight": 0.25,
        "queue_pressure_topk": 8,
    }


def _rows():
    common = {
        "rollout_steps": 64,
        "rollout_candidate_policy_direct_ratio": 1.0,
        "rollout_non_hard_override_count_mean": 0.0,
        "rollout_python_preflight_diagnostics_available_mean": 1.0,
        "rollout_native_reject_diagnostics_available_mean": 1.0,
        "rollout_action_replay_mismatch_count_mean": 0.0,
    }
    for prefix in ("rollout_python_preflight", "rollout_native_reject"):
        for suffix in (
            "wrong_cell_slot_count_mean",
            "empty_slot_count_mean",
            "prg_mask_count_mean",
            "oob_slot_count_mean",
            "duplicate_ue_count_mean",
        ):
            common[f"{prefix}_{suffix}"] = 0.0
    return [
        {
            **common,
            "iter": 1,
            "candidate_policy_grad_norm": 0.01,
            "rollout_entropy_mean": 0.4,
            "rollout_candidate_residual_logits_abs_mean": 0.0,
            "rollout_candidate_pfq_base_logits_abs_mean": 5.0,
            "rollout_queue_pressure_before_mean": 0.05,
            "rollout_queue_pressure_after_mean": 0.051,
            "rollout_queue_pressure_delta_mean": 0.00049,
            "rollout_queue_pressure_reward_mean": 0.0001225,
        },
        {
            **common,
            "iter": 2,
            "candidate_policy_grad_norm": 0.02,
            "rollout_entropy_mean": 0.7,
            "rollout_candidate_residual_logits_abs_mean": 0.01,
            "rollout_candidate_pfq_base_logits_abs_mean": 4.5,
            "rollout_queue_pressure_before_mean": 0.03,
            "rollout_queue_pressure_after_mean": 0.029,
            "rollout_queue_pressure_delta_mean": 0.00129,
            "rollout_queue_pressure_reward_mean": 0.0003225,
        },
    ]


def test_gate_accepts_zero_initial_residual_and_dense_pressure_credit():
    report = evaluate_residual_tail_gate(_summary(), _rows())

    assert report["passed"] is True
    assert all(report["checks"].values())
    assert report["evidence"]["initial_residual_abs_mean"] == 0.0
    assert report["evidence"]["initial_pfq_base_abs_mean"] == 5.0


@pytest.mark.parametrize(
    "mutation,failed_check",
    [
        (lambda rows: rows[0].update(
            rollout_candidate_residual_logits_abs_mean=1e-3
        ), "initial_residual_near_zero"),
        (lambda rows: rows[0].update(
            rollout_candidate_pfq_base_logits_abs_mean=0.0
        ), "initial_pfq_base_positive"),
        (lambda rows: rows[1].update(
            rollout_queue_pressure_reward_mean=-0.1
        ), "queue_reward_matches_pressure_delta"),
        (lambda rows: rows[1].update(
            rollout_native_reject_wrong_cell_slot_count_mean=1.0
        ), "native_reject_zero"),
    ],
)
def test_gate_fails_closed_on_invalid_residual_or_runtime_evidence(
    mutation, failed_check
):
    rows = _rows()
    mutation(rows)

    report = evaluate_residual_tail_gate(_summary(), rows)

    assert report["passed"] is False
    assert report["checks"][failed_check] is False


def test_cli_writes_auditable_gate_with_input_hashes(tmp_path):
    summary_path = tmp_path / "online_ppo_summary.json"
    history_path = tmp_path / "online_ppo_metrics.csv"
    output_path = tmp_path / "residual_tail_gate.json"
    summary_path.write_text(json.dumps(_summary()), encoding="utf-8")
    rows = _rows()
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.stageb_hgraph.residual_tail_gate",
            "--summary",
            str(summary_path),
            "--history",
            str(history_path),
            "--source-revision",
            "deadbeef",
            "--out",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["provenance"]["source_revision"] == "deadbeef"
    assert len(report["provenance"]["summary_sha256"]) == 64
    assert len(report["provenance"]["history_sha256"]) == 64
    assert output_path.read_text(encoding="utf-8").endswith("\n")

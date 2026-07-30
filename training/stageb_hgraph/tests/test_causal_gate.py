import csv
import json
import math
import subprocess
import sys

import pytest

from training.stageb_hgraph.causal_gate import evaluate_causal_gate


PASSING_EVIDENCE = {
    "candidate_policy_grad_norm": 0.2,
    "candidate_policy_entropy": 0.4,
    "non_hard_override_ratio": 0.0,
    "python_preflight_reject_count": 0.0,
    "native_reject_count": 0.0,
    "action_replay_mismatch_count": 0.0,
}


def test_causal_gate_requires_gradient_entropy_and_action_parity():
    report = evaluate_causal_gate(PASSING_EVIDENCE)
    assert report["passed"] is True
    assert report["checks"] == {
        "candidate_gradient_nonzero": True,
        "training_entropy_positive": True,
        "non_hard_override_below_5pct": True,
        "python_preflight_zero": True,
        "native_reject_zero": True,
        "action_replay_exact": True,
    }


def test_causal_gate_rejects_zero_candidate_gradient():
    report = evaluate_causal_gate(
        {**PASSING_EVIDENCE, "candidate_policy_grad_norm": 0.0}
    )
    assert report["passed"] is False
    assert report["checks"]["candidate_gradient_nonzero"] is False


def test_causal_gate_rejects_override_ratio_at_exact_boundary():
    report = evaluate_causal_gate(
        {**PASSING_EVIDENCE, "non_hard_override_ratio": 0.05}
    )
    assert report["passed"] is False
    assert report["checks"]["non_hard_override_below_5pct"] is False


@pytest.mark.parametrize("field", tuple(PASSING_EVIDENCE))
def test_causal_gate_rejects_missing_evidence(field):
    evidence = dict(PASSING_EVIDENCE)
    del evidence[field]
    report = evaluate_causal_gate(evidence)
    assert report["passed"] is False
    assert field in report["missing_evidence"]


@pytest.mark.parametrize("field", tuple(PASSING_EVIDENCE))
@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf, "not-a-number"])
def test_causal_gate_rejects_nonfinite_evidence(field, bad_value):
    report = evaluate_causal_gate({**PASSING_EVIDENCE, field: bad_value})
    assert report["passed"] is False
    assert field in report["invalid_evidence"]
    assert report["evidence"][field] is None


def _write_real_history(
    path,
    *,
    include_replay=True,
    override_mean=0.0,
    native_diagnostics_available=1.0,
    entropy_field="rollout_entropy_mean",
):
    rows = []
    for iteration, rollout_steps, grad, entropy in (
        (1, 2, 0.2, 0.3),
        (2, 6, 0.4, 0.5),
    ):
        row = {
            "iter": iteration,
            "rollout_steps": rollout_steps,
            "candidate_head_grad_norm": grad,
            entropy_field: entropy,
            "rollout_non_hard_override_count_mean": override_mean,
            "rollout_python_preflight_wrong_cell_slot_count_mean": 0.0,
            "rollout_python_preflight_empty_slot_count_mean": 0.0,
            "rollout_python_preflight_prg_mask_count_mean": 0.0,
            "rollout_python_preflight_oob_slot_count_mean": 0.0,
            "rollout_python_preflight_duplicate_ue_count_mean": 0.0,
            "rollout_native_reject_wrong_cell_slot_count_mean": 0.0,
            "rollout_native_reject_empty_slot_count_mean": 0.0,
            "rollout_native_reject_prg_mask_count_mean": 0.0,
            "rollout_native_reject_oob_slot_count_mean": 0.0,
            "rollout_native_reject_duplicate_ue_count_mean": 0.0,
        }
        if native_diagnostics_available is not None:
            row["rollout_native_reject_diagnostics_available_mean"] = (
                native_diagnostics_available
            )
        if include_replay:
            row["rollout_action_replay_mismatch_count_mean"] = 0.0
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_cli(summary_path, history_path, output_path):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "training.stageb_hgraph.causal_gate",
            "--summary",
            str(summary_path),
            "--history",
            str(history_path),
            "--out",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_aggregates_real_summary_and_history_fields_and_writes_pass(tmp_path):
    summary_path = tmp_path / "online_ppo_summary.json"
    history_path = tmp_path / "online_ppo_metrics.csv"
    output_path = tmp_path / "causal_gate.json"
    summary_path.write_text(
        json.dumps(
            {
                "iterations": 2,
                "completed_iterations": 2,
                "candidate_policy_mode": "direct",
                "history_tail": [{"candidate_head_grad_norm": 999.0}],
            }
        ),
        encoding="utf-8",
    )
    _write_real_history(history_path)

    completed = _run_cli(summary_path, history_path, output_path)

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["provenance_checks"] == {
        "candidate_policy_mode_direct": True,
        "two_iterations_complete": True,
    }
    assert report["run_config"]["candidate_policy_mode"] == "direct"
    assert report["run_config"]["completed_iterations"] == 2
    assert report["evidence"] == {
        "action_replay_mismatch_count": 0.0,
        "candidate_policy_entropy": 0.4,
        "candidate_policy_grad_norm": pytest.approx(0.3),
        "native_reject_count": 0.0,
        "non_hard_override_ratio": 0.0,
        "python_preflight_reject_count": 0.0,
    }
    assert output_path.read_text(encoding="utf-8").endswith("\n")


def test_cli_maps_ppo_update_entropy_when_rollout_entropy_is_absent(tmp_path):
    summary_path = tmp_path / "online_ppo_summary.json"
    history_path = tmp_path / "online_ppo_metrics.csv"
    output_path = tmp_path / "causal_gate.json"
    summary_path.write_text(
        json.dumps(
            {
                "iterations": 2,
                "completed_iterations": 2,
                "candidate_policy_mode": "direct",
                "history_tail": [],
            }
        ),
        encoding="utf-8",
    )
    _write_real_history(history_path, entropy_field="entropy_mean")

    completed = _run_cli(summary_path, history_path, output_path)

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["evidence"]["candidate_policy_entropy"] == pytest.approx(0.4)


def test_cli_fails_closed_when_real_history_omits_replay_evidence(tmp_path):
    summary_path = tmp_path / "online_ppo_summary.json"
    history_path = tmp_path / "online_ppo_metrics.csv"
    output_path = tmp_path / "causal_gate.json"
    summary_path.write_text(
        json.dumps(
            {
                "iterations": 2,
                "completed_iterations": 2,
                "candidate_policy_mode": "direct",
                "history_tail": [],
            }
        ),
        encoding="utf-8",
    )
    _write_real_history(history_path, include_replay=False)

    completed = _run_cli(summary_path, history_path, output_path)

    assert completed.returncode != 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert "action_replay_mismatch_count" in report["missing_evidence"]


@pytest.mark.parametrize("candidate_policy_mode", ["legacy", None])
def test_cli_rejects_legacy_or_missing_candidate_policy_mode(
    tmp_path, candidate_policy_mode
):
    summary_path = tmp_path / "online_ppo_summary.json"
    history_path = tmp_path / "online_ppo_metrics.csv"
    output_path = tmp_path / "causal_gate.json"
    summary = {
        "iterations": 2,
        "completed_iterations": 2,
        "history_tail": [],
    }
    if candidate_policy_mode is not None:
        summary["candidate_policy_mode"] = candidate_policy_mode
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    _write_real_history(history_path)

    completed = _run_cli(summary_path, history_path, output_path)

    assert completed.returncode != 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert all(report["checks"].values())
    assert report["provenance_checks"]["candidate_policy_mode_direct"] is False
    assert report["passed"] is False


@pytest.mark.parametrize(
    "iterations,completed_iterations",
    [(2, 1), (1, 1), (3, 2)],
)
def test_cli_rejects_incomplete_or_non_smoke_iteration_provenance(
    tmp_path, iterations, completed_iterations
):
    summary_path = tmp_path / "online_ppo_summary.json"
    history_path = tmp_path / "online_ppo_metrics.csv"
    output_path = tmp_path / "causal_gate.json"
    summary_path.write_text(
        json.dumps(
            {
                "iterations": iterations,
                "completed_iterations": completed_iterations,
                "candidate_policy_mode": "direct",
                "history_tail": [],
            }
        ),
        encoding="utf-8",
    )
    _write_real_history(history_path)

    completed = _run_cli(summary_path, history_path, output_path)

    assert completed.returncode != 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert all(report["checks"].values())
    assert report["provenance_checks"]["two_iterations_complete"] is False
    assert report["passed"] is False


@pytest.mark.parametrize("availability", [None, 0.0, 2.0])
def test_cli_fails_closed_when_native_diagnostics_are_not_fully_available(
    tmp_path, availability
):
    summary_path = tmp_path / "online_ppo_summary.json"
    history_path = tmp_path / "online_ppo_metrics.csv"
    output_path = tmp_path / "causal_gate.json"
    summary_path.write_text(
        json.dumps(
            {
                "iterations": 2,
                "completed_iterations": 2,
                "candidate_policy_mode": "direct",
                "history_tail": [],
            }
        ),
        encoding="utf-8",
    )
    _write_real_history(
        history_path,
        native_diagnostics_available=availability,
    )

    completed = _run_cli(summary_path, history_path, output_path)

    assert completed.returncode != 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["evidence"]["native_reject_count"] is None


def test_cli_exits_nonzero_and_writes_boundary_failure(tmp_path):
    summary_path = tmp_path / "online_ppo_summary.json"
    history_path = tmp_path / "online_ppo_metrics.csv"
    output_path = tmp_path / "causal_gate.json"
    summary_path.write_text(
        json.dumps(
            {
                "iterations": 2,
                "completed_iterations": 2,
                "candidate_policy_mode": "direct",
                "history_tail": [],
            }
        ),
        encoding="utf-8",
    )
    _write_real_history(history_path, override_mean=2.55)

    completed = _run_cli(summary_path, history_path, output_path)

    assert completed.returncode != 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["evidence"]["non_hard_override_ratio"] == pytest.approx(0.05)
    assert report["checks"]["non_hard_override_below_5pct"] is False

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "cuMAC/scripts/run_stageB_pfq_independent_reuse.sh"


def test_runner_recognizes_candidate_width_compatibility_phase() -> None:
    completed = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "candidate-width-compatibility",
            "--steps",
            "0",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--steps must be a positive integer" in completed.stderr
    assert "unsupported phase" not in completed.stderr


def test_runner_recognizes_separate_w0_v2_fp32_phase() -> None:
    completed = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "candidate-width-fp32-equivalence",
            "--steps",
            "0",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--steps must be a positive integer" in completed.stderr
    assert "unsupported phase" not in completed.stderr


def test_runner_recognizes_seed42_only_formal_pair_phase() -> None:
    completed = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "formal-seed42-pfq-b4",
            "--steps",
            "0",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--steps must be a positive integer" in completed.stderr
    assert "unsupported phase" not in completed.stderr


def test_runner_recognizes_seed42_per_prg_diagnostic_phase() -> None:
    completed = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "diagnose-seed42-per-prg",
            "--steps",
            "0",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--steps must be a positive integer" in completed.stderr
    assert "unsupported phase" not in completed.stderr


def test_runner_recognizes_seed42_service_accounting_audit_phase() -> None:
    completed = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "audit-seed42-service-accounting",
            "--steps",
            "0",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--steps must be a positive integer" in completed.stderr
    assert "unsupported phase" not in completed.stderr


def test_runner_recognizes_seed42_phy_tbs_alignment_audit_phase() -> None:
    completed = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "audit-seed42-phy-tbs-alignment",
            "--steps",
            "0",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--steps must be a positive integer" in completed.stderr
    assert "unsupported phase" not in completed.stderr


def test_runner_recognizes_seed42_phy_tbs_alignment_v2_cpu_olla_phase() -> None:
    completed = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "audit-seed42-phy-tbs-alignment-v2-cpu-olla",
            "--steps",
            "0",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--steps must be a positive integer" in completed.stderr
    assert "unsupported phase" not in completed.stderr


def test_runner_recognizes_seed42_cell2_ue2_service_balance_audit_phase() -> None:
    completed = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "audit-seed42-cell2-ue2-service-balance",
            "--steps",
            "0",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--steps must be a positive integer" in completed.stderr
    assert "unsupported phase" not in completed.stderr


def test_runner_recognizes_b4_ctbs_c1_short_treatment_phase() -> None:
    completed = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "b4-ctbs-c1-short-treatment",
            "--steps",
            "0",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--steps must be a positive integer" in completed.stderr
    assert "unsupported phase" not in completed.stderr


def test_runner_recognizes_b4_ctbs_c1_multiseed_long_phase() -> None:
    completed = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "b4-ctbs-c1-multiseed-long",
            "--steps",
            "0",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--steps must be a positive integer" in completed.stderr
    assert "unsupported phase" not in completed.stderr

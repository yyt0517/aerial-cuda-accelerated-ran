from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "cuMAC/scripts/run_stageB_pfq_independent_reuse.sh"


@pytest.mark.parametrize("phase", ("rescue-cohort-smoke", "rescue-cohort-seed41"))
def test_rescue_cohort_phase_routes_to_missing_simulator(tmp_path: Path, phase: str) -> None:
    """Catches either rescue-cohort phase being omitted from routing."""
    result = subprocess.run(
        [
            "bash",
            str(RUNNER),
            phase,
            "--build-dir",
            str(tmp_path / "missing-build"),
            "--out-dir",
            str(tmp_path / "missing-output"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "missing simulator binary" in result.stderr
    assert "unsupported phase" not in result.stderr


def _function_body(source: str, name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n(?P<body>.*?)^\}}$", source, re.MULTILINE | re.DOTALL)
    assert match is not None
    return match.group("body")


def test_rescue_cohort_phases_use_only_the_approved_evaluation_windows() -> None:
    """Catches a wrong TTI window, duplicate evaluation, or non-hierarchical decode."""
    source = RUNNER.read_text(encoding="utf-8")
    smoke_body = _function_body(source, "run_rescue_cohort_smoke")
    formal_body = _function_body(source, "run_rescue_cohort_seed41")

    assert smoke_body.count(
        'evaluate_checkpoint "${checkpoint}" 41 2 4 "${phase_dir}" 0 1 hierarchical_argmax'
    ) == 1
    assert formal_body.count(
        'evaluate_checkpoint "${checkpoint}" 41 1000 512 "${phase_dir}" 0 1 hierarchical_argmax'
    ) == 1

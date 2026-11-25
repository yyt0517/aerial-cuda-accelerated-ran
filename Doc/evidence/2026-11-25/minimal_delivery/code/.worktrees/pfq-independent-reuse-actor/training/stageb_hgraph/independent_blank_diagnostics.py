"""Detached, decision-level diagnostics for the independent policy's Blank class."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .independent_gate import CHECKPOINT_ITERATIONS


_TOLERANCE = 1.0e-6
_DIAGNOSTIC_LABELS = {"A4", "A8", "B"}
_DIAGNOSTIC_DECISIONS = 512 * 51
_BLANK_FIELDS = (
    "diagnostic_selected_blank_ratio",
    "diagnostic_greedy_blank_ratio",
    "diagnostic_forced_blank_ratio",
    "diagnostic_learned_blank_ratio",
    "diagnostic_fragmentation_blank_ratio",
    "diagnostic_conservative_blank_ratio",
    "diagnostic_fragmentation_share_among_learned_blank",
    "diagnostic_learned_blank_count",
    "diagnostic_mean_valid_ue_count",
    "diagnostic_valid_margin_count",
    "diagnostic_p_blank_mean",
    "diagnostic_p_blank_p10",
    "diagnostic_p_blank_p50",
    "diagnostic_p_blank_p90",
    "diagnostic_p_service_mean",
    "diagnostic_p_service_p10",
    "diagnostic_p_service_p50",
    "diagnostic_p_service_p90",
    "diagnostic_p_best_ue_mean",
    "diagnostic_p_best_ue_p10",
    "diagnostic_p_best_ue_p50",
    "diagnostic_p_best_ue_p90",
    "diagnostic_blank_margin_mean",
    "diagnostic_blank_margin_p10",
    "diagnostic_blank_margin_p50",
    "diagnostic_blank_margin_p90",
)


def _positive_iteration(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(summary: Mapping[str, object], key: str, label: str) -> float:
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} {key} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} {key} must be finite")
    return result


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-9)


def choose_b_diagnostic_iteration(selection: Mapping[str, object]) -> int:
    """Choose the formal B subject, or the best valid formal fallback candidate."""
    if "selected_iteration" not in selection:
        raise ValueError("B selection is missing selected_iteration")
    selected = selection["selected_iteration"]
    if selected is not None:
        return _positive_iteration(selected, "selected_iteration")

    candidates = selection.get("candidates")
    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("B selection has no fallback candidates")
    valid: list[tuple[float, int]] = []
    for iteration_text, candidate in candidates.items():
        try:
            iteration = _positive_iteration(int(iteration_text), "candidate iteration")
        except (TypeError, ValueError):
            continue
        if iteration not in CHECKPOINT_ITERATIONS:
            continue
        if not isinstance(candidate, Mapping) or candidate.get("present") is not True:
            continue
        metrics = candidate.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        try:
            goodput = _finite_number(metrics, "goodput_mbps", f"B{iteration}")
            replay = _finite_number(metrics, "action_replay_mismatch_count", f"B{iteration}")
            illegal = _finite_number(metrics, "illegal_action_count", f"B{iteration}")
        except ValueError:
            continue
        if (
            replay == 0.0
            and illegal == 0.0
            and metrics.get("independent_deletion_equal") is True
        ):
            valid.append((goodput, iteration))
    if not valid:
        raise ValueError("B selection has no valid fallback diagnostic candidate")
    return min(valid, key=lambda item: (-item[0], item[1]))[1]


def _validate_diagnostic_summary(
    label: str,
    summary: Mapping[str, object],
) -> dict[str, object]:
    if summary.get("decode_mode") != "argmax":
        raise ValueError(f"{label} diagnostic evaluation must use argmax")
    if summary.get("actor_family") != "pfq_independent_reuse_v1":
        raise ValueError(f"{label} diagnostic evaluation must use the independent actor")
    if summary.get("requires_pfq_features") is not False:
        raise ValueError(f"{label} actor must be PFQ-independent")
    if summary.get("independent_deletion_equal") is not True:
        raise ValueError(f"{label} PFQ-field deletion identity failed")
    if summary.get("independent_policy_diagnostics_enabled") != 1:
        raise ValueError(f"{label} independent diagnostics are not enabled")
    if summary.get("stats_count") != 512:
        raise ValueError(f"{label} diagnostic stats_count must equal 512")
    if (summary.get("measured_tti_first"), summary.get("measured_tti_last")) != (1001, 1512):
        raise ValueError(f"{label} measured TTI window must equal 1001--1512")
    decisions = _finite_number(summary, "diagnostic_decision_count", label)
    if decisions != float(_DIAGNOSTIC_DECISIONS):
        raise ValueError(f"{label} diagnostic decision count must equal {_DIAGNOSTIC_DECISIONS}")
    replay = _finite_number(summary, "stats_sum_action_replay_mismatch_count", label)
    if replay != 0.0:
        raise ValueError(f"{label} replay mismatch sum must equal zero")
    illegal = _finite_number(summary, "stats_sum_illegal_action_count", label)
    if illegal != 0.0:
        raise ValueError(f"{label} illegal action sum must equal zero")

    blank_values = {key: _finite_number(summary, key, label) for key in _BLANK_FIELDS}
    ratio_keys = (
        "diagnostic_selected_blank_ratio",
        "diagnostic_greedy_blank_ratio",
        "diagnostic_forced_blank_ratio",
        "diagnostic_learned_blank_ratio",
        "diagnostic_fragmentation_blank_ratio",
        "diagnostic_conservative_blank_ratio",
        "diagnostic_fragmentation_share_among_learned_blank",
    )
    if any(not 0.0 <= blank_values[key] <= 1.0 for key in ratio_keys):
        raise ValueError(f"{label} Blank ratios must be finite values in [0,1]")
    for prefix in ("diagnostic_p_blank", "diagnostic_p_service", "diagnostic_p_best_ue"):
        if any(not 0.0 <= blank_values[f"{prefix}_{suffix}"] <= 1.0 for suffix in ("mean", "p10", "p50", "p90")):
            raise ValueError(f"{label} probability summaries must be finite values in [0,1]")
    learned_count = blank_values["diagnostic_learned_blank_count"]
    valid_margin_count = blank_values["diagnostic_valid_margin_count"]
    if not 0.0 <= learned_count <= decisions or not 0.0 <= valid_margin_count <= decisions:
        raise ValueError(f"{label} diagnostic counts are outside the measured decision window")

    selected = blank_values["diagnostic_selected_blank_ratio"]
    greedy = blank_values["diagnostic_greedy_blank_ratio"]
    forced = blank_values["diagnostic_forced_blank_ratio"]
    learned = blank_values["diagnostic_learned_blank_ratio"]
    fragmentation = blank_values["diagnostic_fragmentation_blank_ratio"]
    conservative = blank_values["diagnostic_conservative_blank_ratio"]
    fragmentation_share = blank_values["diagnostic_fragmentation_share_among_learned_blank"]
    if not _close(selected, greedy):
        raise ValueError(f"{label} deterministic selected/greedy Blank identity failed")
    if not _close(greedy, forced + learned):
        raise ValueError(f"{label} forced/learned Blank identity failed")
    if not _close(learned, fragmentation + conservative):
        raise ValueError(f"{label} fragmentation/conservative Blank identity failed")
    if not _close(learned_count, learned * decisions):
        raise ValueError(f"{label} learned Blank count identity failed")
    expected_share = fragmentation / learned if learned_count > 0.0 else 0.0
    if not _close(fragmentation_share, expected_share):
        raise ValueError(f"{label} learned Blank fragmentation share identity failed")

    byte_keys = (
        "stats_sum_scheduled_bytes",
        "stats_sum_ue_goodput_bytes",
        "stats_sum_failed_bytes",
    )
    byte_values = {key: _finite_number(summary, key, label) for key in byte_keys}
    if any(value < 0.0 for value in byte_values.values()):
        raise ValueError(f"{label} measured byte values must be non-negative")
    if not _close(
        byte_values["stats_sum_scheduled_bytes"],
        byte_values["stats_sum_ue_goodput_bytes"] + byte_values["stats_sum_failed_bytes"],
    ):
        raise ValueError(f"{label} scheduled/goodput/failed byte identity failed")

    mbps_keys = (
        "stats_mean_scheduled_mbps",
        "stats_mean_ue_goodput_mbps",
        "stats_failed_mbps",
    )
    mbps_values = {key: _finite_number(summary, key, label) for key in mbps_keys}
    if any(value < 0.0 for value in mbps_values.values()):
        raise ValueError(f"{label} measured Mbps values must be non-negative")
    if not _close(
        mbps_values["stats_mean_scheduled_mbps"],
        mbps_values["stats_mean_ue_goodput_mbps"] + mbps_values["stats_failed_mbps"],
    ):
        raise ValueError(f"{label} scheduled/goodput/failed Mbps identity failed")
    for byte_key, mbps_key in zip(byte_keys, mbps_keys):
        if not _close(byte_values[byte_key] / 512.0 * 0.016, mbps_values[mbps_key]):
            raise ValueError(f"{label} measured byte/Mbps conversion identity failed")

    tb_tx = _finite_number(summary, "measured_tb_tx_count", label)
    tb_err = _finite_number(summary, "measured_tb_err_count", label)
    tb_bler = _finite_number(summary, "measured_tb_bler", label)
    if tb_tx <= 0.0 or tb_err < 0.0 or tb_err > tb_tx or not 0.0 <= tb_bler <= 1.0:
        raise ValueError(f"{label} measured TB counts are invalid")
    if not _close(tb_bler, tb_err / tb_tx):
        raise ValueError(f"{label} measured TB identity failed")

    checkpoint_hash = summary.get("checkpoint_sha256")
    sim_hash = summary.get("sim_binary_sha256")
    if not isinstance(checkpoint_hash, str) or not checkpoint_hash:
        raise ValueError(f"{label} checkpoint hash is missing")
    if not isinstance(sim_hash, str) or not sim_hash:
        raise ValueError(f"{label} simulator hash is missing")

    def distribution(prefix: str) -> dict[str, float]:
        return {
            suffix: blank_values[f"diagnostic_{prefix}_{suffix}"]
            for suffix in ("mean", "p10", "p50", "p90")
        }

    return {
        "checkpoint_sha256": checkpoint_hash,
        "sim_binary_sha256": sim_hash,
        "blank_composition": {
            "selected_ratio": selected,
            "greedy_ratio": greedy,
            "forced_ratio": forced,
            "learned_ratio": learned,
            "fragmentation_ratio": fragmentation,
            "conservative_ratio": conservative,
            "fragmentation_share_among_learned": fragmentation_share,
            "learned_count": learned_count,
        },
        "distributions": {
            "p_blank": distribution("p_blank"),
            "p_service": distribution("p_service"),
            "p_best_ue": distribution("p_best_ue"),
            "blank_margin": distribution("blank_margin"),
        },
        "transmission_bytes": {
            "scheduled": byte_values["stats_sum_scheduled_bytes"],
            "goodput": byte_values["stats_sum_ue_goodput_bytes"],
            "failed": byte_values["stats_sum_failed_bytes"],
        },
        "transmission_mbps": {
            "scheduled": mbps_values["stats_mean_scheduled_mbps"],
            "goodput": mbps_values["stats_mean_ue_goodput_mbps"],
            "failed": mbps_values["stats_failed_mbps"],
        },
        "tb": {"tx_count": tb_tx, "error_count": tb_err, "bler": tb_bler},
    }


def build_blank_diagnostic_report(
    *,
    summaries: Mapping[str, Mapping[str, object]],
    b_iteration: int,
    b_formal_selection: Mapping[str, object],
) -> dict[str, object]:
    """Validate fixed-window evidence and build the diagnostic conclusion."""
    if set(summaries) != _DIAGNOSTIC_LABELS:
        raise ValueError("diagnostic summary labels must be exactly A4, A8, and B")
    b_iteration = _positive_iteration(b_iteration, "B diagnostic iteration")
    subjects = {
        label: _validate_diagnostic_summary(label, summaries[label])
        for label in ("A4", "A8", "B")
    }
    simulator_hashes = {subject["sim_binary_sha256"] for subject in subjects.values()}
    if len(simulator_hashes) != 1:
        raise ValueError("diagnostic summaries must use one identical nonempty simulator hash")

    if "selected_iteration" not in b_formal_selection:
        raise ValueError("B formal selection is missing selected_iteration")
    formal_iteration_raw = b_formal_selection["selected_iteration"]
    formal_iteration = (
        None
        if formal_iteration_raw is None
        else _positive_iteration(formal_iteration_raw, "formal B selected_iteration")
    )
    subjects["A4"].update({"iteration": 4, "role": "positive_control"})
    subjects["A8"].update({"iteration": 8, "role": "primary_hypothesis_subject"})
    subjects["B"].update({"iteration": b_iteration, "role": "B_comparison"})

    a8 = subjects["A8"]["blank_composition"]
    learned_count = float(a8["learned_count"])
    fragmentation_share = float(a8["fragmentation_share_among_learned"])
    if learned_count == 0.0:
        hypothesis: dict[str, object] = {
            "name": "H_frag",
            "subject": "A8",
            "supported": None,
            "reason": "no learned Blank decisions",
            "learned_blank_count": learned_count,
            "fragmentation_share_among_learned_blank": fragmentation_share,
        }
    else:
        hypothesis = {
            "name": "H_frag",
            "subject": "A8",
            "supported": fragmentation_share > 0.5,
            "learned_blank_count": learned_count,
            "fragmentation_share_among_learned_blank": fragmentation_share,
        }

    forced = float(a8["forced_ratio"])
    fragmentation = float(a8["fragmentation_ratio"])
    conservative = float(a8["conservative_ratio"])
    if forced > 0.0 and forced >= max(fragmentation, conservative):
        next_experiment = (
            "Inspect candidate validity and sequential demand estimation before any policy change."
        )
    elif learned_count > 0.0 and fragmentation_share > 0.5:
        next_experiment = (
            "Test a minimal hierarchical Serve/Blank -> UE action head with matched PPO log-probability."
        )
    else:
        next_experiment = (
            "Retain the flat categorical and test learned conservatism evidence "
            "(reward/BC calibration) without forced fill."
        )

    return {
        "valid": True,
        "window": {
            "seed": 41,
            "warmup_ttis": 1000,
            "measured_ttis": 512,
            "measured_tti_first": 1001,
            "measured_tti_last": 1512,
            "decision_count_per_subject": _DIAGNOSTIC_DECISIONS,
            "sim_binary_sha256": next(iter(simulator_hashes)),
        },
        "subjects": subjects,
        "b_selection": {
            "iteration": b_iteration,
            "formal_selected_iteration": formal_iteration,
            "formal_gate_promoted": formal_iteration is not None and formal_iteration == b_iteration,
        },
        "hypothesis": hypothesis,
        "next_experiment": next_experiment,
    }


def _render_blank_diagnostic_markdown(report: Mapping[str, object]) -> str:
    subjects = report["subjects"]
    hypothesis = report["hypothesis"]
    b_selection = report["b_selection"]
    supported = hypothesis["supported"]
    conclusion = "inconclusive" if supported is None else ("supported" if supported else "not supported")

    def exact_number(value: object) -> str:
        number = float(value)
        return str(int(number)) if number.is_integer() else repr(number)

    lines = [
        "# Independent Blank Diagnostic Report",
        "",
        "## Blank composition",
        "",
        "| Subject | Iteration | Selected | Greedy | Forced | Learned | Fragmentation | Conservative | Frag / learned |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("A4", "A8", "B"):
        subject = subjects[label]
        blank = subject["blank_composition"]
        lines.append(
            f"| {label} | {subject['iteration']} | {blank['selected_ratio']:.6f} | "
            f"{blank['greedy_ratio']:.6f} | {blank['forced_ratio']:.6f} | "
            f"{blank['learned_ratio']:.6f} | {blank['fragmentation_ratio']:.6f} | "
            f"{blank['conservative_ratio']:.6f} | "
            f"{blank['fragmentation_share_among_learned']:.6f} |"
        )

    lines.extend([
        "",
        "## Probability and margin distributions",
        "",
        "| Subject | Distribution | Mean | P10 | P50 | P90 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for label in ("A4", "A8", "B"):
        distributions = subjects[label]["distributions"]
        for distribution_name in ("p_blank", "p_service", "p_best_ue", "blank_margin"):
            distribution = distributions[distribution_name]
            lines.append(
                f"| {label} | {distribution_name} | {distribution['mean']:.6f} | "
                f"{distribution['p10']:.6f} | {distribution['p50']:.6f} | "
                f"{distribution['p90']:.6f} |"
            )

    lines.extend([
        "",
        "## Transmission accounting",
        "",
        "| Subject | Scheduled bytes | Goodput bytes | Failed-transmission bytes | Scheduled Mbps | Goodput Mbps | Failed-transmission Mbps | TB errors / tx | BLER |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for label in ("A4", "A8", "B"):
        subject = subjects[label]
        byte_sums = subject["transmission_bytes"]
        mbps = subject["transmission_mbps"]
        tb = subject["tb"]
        lines.append(
            f"| {label} | {exact_number(byte_sums['scheduled'])} | "
            f"{exact_number(byte_sums['goodput'])} | {exact_number(byte_sums['failed'])} | "
            f"{mbps['scheduled']:.6f} | {mbps['goodput']:.6f} | {mbps['failed']:.6f} | "
            f"{exact_number(tb['error_count'])} / {exact_number(tb['tx_count'])} | "
            f"{tb['bler']:.6f} |"
        )
    lines.extend([
        "",
        "Failed-transmission bytes and Mbps represent scheduled transmission failures, not packet expiry.",
        "",
        "## Artifact identity",
        "",
        "| Subject | Checkpoint SHA-256 | Simulator SHA-256 |",
        "|---|---|---|",
    ])
    for label in ("A4", "A8", "B"):
        subject = subjects[label]
        lines.append(
            f"| {label} | `{subject['checkpoint_sha256']}` | "
            f"`{subject['sim_binary_sha256']}` |"
        )

    formal_iteration = b_selection["formal_selected_iteration"]
    formatted_formal_iteration = "none" if formal_iteration is None else str(formal_iteration)
    if b_selection["formal_gate_promoted"]:
        b_rationale = f"formal gate promoted iteration `{b_selection['iteration']}`."
    elif formal_iteration is None:
        b_rationale = (
            "the formal gate promoted no checkpoint; "
            f"iteration `{b_selection['iteration']}` is the highest-goodput valid "
            "checkpoint-cadence fallback."
        )
    else:
        b_rationale = (
            f"formal gate selected iteration `{formal_iteration}`; diagnostic iteration "
            f"`{b_selection['iteration']}` is a fallback."
        )
    lines.extend([
        "",
        "## Conclusion",
        "",
        f"- B diagnostic iteration: `{b_selection['iteration']}`",
        f"- B formal selected iteration: `{formatted_formal_iteration}`",
        f"- B selection rationale: {b_rationale}",
        f"- H_frag on A8: **{conclusion}** "
        f"(learned={hypothesis['learned_blank_count']:.0f}, "
        f"fragmentation share={hypothesis['fragmentation_share_among_learned_blank']:.6f})",
    ])
    if hypothesis.get("reason"):
        lines.append(f"- H_frag reason: {hypothesis['reason']}")
    lines.extend(["", f"Next experiment: {report['next_experiment']}", ""])
    return "\n".join(lines)


def _load_json_mapping(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select_parser = subparsers.add_parser("select-b")
    select_parser.add_argument("--selection", type=Path, required=True)
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--a4", type=Path, required=True)
    report_parser.add_argument("--a8", type=Path, required=True)
    report_parser.add_argument("--b", type=Path, required=True)
    report_parser.add_argument("--b-iteration", type=int, required=True)
    report_parser.add_argument("--b-selection", type=Path, required=True)
    report_parser.add_argument("--json-out", type=Path, required=True)
    report_parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "select-b":
        print(choose_b_diagnostic_iteration(_load_json_mapping(args.selection)))
        return 0

    report = build_blank_diagnostic_report(
        summaries={
            "A4": _load_json_mapping(args.a4),
            "A8": _load_json_mapping(args.a8),
            "B": _load_json_mapping(args.b),
        },
        b_iteration=args.b_iteration,
        b_formal_selection=_load_json_mapping(args.b_selection),
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_out.write_text(_render_blank_diagnostic_markdown(report), encoding="utf-8")
    return 0


def validate_measured_window_goodput_mbps(
    *,
    derived_goodput_mbps: float,
    existing_goodput_mbps: float,
) -> None:
    """Require finite, numerically equal goodput from the two measured sources."""
    if not np.isfinite(derived_goodput_mbps) or not np.isfinite(existing_goodput_mbps):
        raise ValueError("measured-window goodput values must be finite")
    float32_max = float(np.finfo(np.float32).max)
    if (
        abs(derived_goodput_mbps) > float32_max
        or abs(existing_goodput_mbps) > float32_max
    ):
        raise ValueError("measured-window goodput values must be within finite float32 range")
    magnitude = max(abs(derived_goodput_mbps), abs(existing_goodput_mbps))
    magnitude32 = np.float32(magnitude)
    direction = np.float32(-np.inf if magnitude32 == np.float32(float32_max) else np.inf)
    adjacent32 = np.nextafter(magnitude32, direction, dtype=np.float32)
    float32_ulp = abs(float(adjacent32) - float(magnitude32))
    tolerance = max(1.0e-6, float32_ulp)
    if abs(derived_goodput_mbps - existing_goodput_mbps) > tolerance:
        raise ValueError(
            "measured-window UE goodput does not match existing goodput: "
            f"ue={derived_goodput_mbps} existing={existing_goodput_mbps} "
            f"tolerance={tolerance}"
        )


def extract_failed_transmission_bytes(
    info: Mapping[str, object],
    *,
    expected_n_ue: int,
) -> dict[str, float]:
    """Return exact scheduled, goodput, and failed-attempt bytes for one TTI."""
    required = ("actual_ue_served_bytes", "actual_ue_goodput_bytes")
    missing = [key for key in required if key not in info]
    if missing:
        raise ValueError(f"missing protocol byte arrays: {', '.join(missing)}")

    arrays: dict[str, np.ndarray] = {}
    for key in required:
        try:
            values = np.asarray(info[key], dtype=np.float64).reshape((expected_n_ue,))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must reshape to ({expected_n_ue},)") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"{key} must contain only finite values")
        if np.any(values < 0.0):
            raise ValueError(f"{key} must contain only non-negative values")
        arrays[key] = values

    scheduled_bytes = np.sum(arrays["actual_ue_served_bytes"], dtype=np.float64)
    goodput_bytes = np.sum(arrays["actual_ue_goodput_bytes"], dtype=np.float64)
    if goodput_bytes > scheduled_bytes + _TOLERANCE:
        raise ValueError("goodput bytes must not exceed scheduled bytes")
    return {
        "scheduled_bytes": float(scheduled_bytes),
        "goodput_bytes": float(goodput_bytes),
        "failed_bytes": float(max(scheduled_bytes - goodput_bytes, 0.0)),
    }


@dataclass(frozen=True)
class IndependentPolicyDiagnostics:
    valid_ue_count: torch.Tensor
    p_blank: torch.Tensor
    p_service: torch.Tensor
    p_best_ue: torch.Tensor
    blank_margin: torch.Tensor
    margin_valid: torch.Tensor
    selected_blank: torch.Tensor
    greedy_blank: torch.Tensor
    changed_from_greedy: torch.Tensor
    forced_blank: torch.Tensor
    learned_blank: torch.Tensor
    fragmentation_blank: torch.Tensor
    fragmentation_rescued: torch.Tensor
    conservative_blank: torch.Tensor
    selected_forced_blank: torch.Tensor
    selected_fragmentation_blank: torch.Tensor
    selected_conservative_blank: torch.Tensor


def _detached_vector(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().reshape(-1).cpu()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_inputs(
    adjusted_logits: torch.Tensor,
    probabilities: torch.Tensor,
    valid_main: torch.Tensor,
    selected: torch.Tensor,
    blank_class_index: int,
) -> None:
    _require(adjusted_logits.dim() == 2, "adjusted_logits must have shape [N,C]")
    _require(probabilities.shape == adjusted_logits.shape, "probabilities shape must match adjusted_logits")
    _require(
        valid_main.shape == (adjusted_logits.shape[0], adjusted_logits.shape[1] - 1),
        "valid_main must have shape [N,C-1]",
    )
    _require(selected.shape == (adjusted_logits.shape[0],), "selected must have shape [N]")
    _require(blank_class_index == adjusted_logits.shape[1] - 1, "blank_class_index must be the final class")
    _require(torch.isfinite(adjusted_logits).all().item(), "adjusted_logits must be finite")
    _require(torch.isfinite(probabilities).all().item(), "probabilities must be finite")
    _require(((probabilities >= 0.0) & (probabilities <= 1.0)).all().item(), "probabilities must be in [0,1]")
    _require(
        ((selected >= 0) & (selected < adjusted_logits.shape[1])).all().item(),
        "selected contains an invalid class index",
    )


def compute_independent_policy_diagnostics(
    *,
    adjusted_logits: torch.Tensor,
    probabilities: torch.Tensor,
    valid_main: torch.Tensor,
    selected: torch.Tensor,
    blank_class_index: int,
) -> IndependentPolicyDiagnostics:
    """Classify the exact already-masked distribution without changing policy behavior."""
    _validate_inputs(adjusted_logits, probabilities, valid_main, selected, blank_class_index)

    valid_ue_count = valid_main.sum(dim=-1)
    margin_valid = valid_ue_count > 0
    p_blank = probabilities[:, blank_class_index]
    raw_p_service = probabilities[:, :blank_class_index].masked_fill(~valid_main, 0.0).sum(dim=-1)
    p_best_ue = probabilities[:, :blank_class_index].masked_fill(~valid_main, 0.0).max(dim=-1).values
    best_valid_logit = adjusted_logits[:, :blank_class_index].masked_fill(~valid_main, -torch.inf).max(dim=-1).values
    blank_margin = torch.where(
        margin_valid,
        adjusted_logits[:, blank_class_index] - best_valid_logit,
        torch.zeros_like(p_blank),
    )

    greedy_selected = adjusted_logits.argmax(dim=-1)
    changed_from_greedy = selected != greedy_selected
    greedy_blank = greedy_selected == blank_class_index
    selected_blank = selected == blank_class_index
    forced_blank = greedy_blank & ~margin_valid
    learned_blank = greedy_blank & margin_valid

    _require(torch.isfinite(raw_p_service).all().item(), "p_service must be finite")
    _require(torch.isfinite(p_best_ue).all().item(), "p_best_ue must be finite")
    _require(torch.isfinite(blank_margin).all().item(), "blank_margin must be finite")
    _require(((p_blank >= 0.0) & (p_blank <= 1.0)).all().item(), "p_blank must be in [0,1]")
    _require(
        ((raw_p_service >= -_TOLERANCE) & (raw_p_service <= 1.0 + _TOLERANCE)).all().item(),
        "p_service must be in [0,1] within tolerance",
    )
    _require(((p_best_ue >= 0.0) & (p_best_ue <= 1.0)).all().item(), "p_best_ue must be in [0,1]")
    _require(
        torch.allclose(p_blank + raw_p_service, torch.ones_like(p_blank), atol=_TOLERANCE, rtol=0.0),
        "p_blank + p_service must equal one",
    )
    _require(
        (p_best_ue <= raw_p_service + _TOLERANCE).all().item(),
        "p_best_ue must not exceed p_service",
    )

    p_service = raw_p_service.clamp(0.0, 1.0)
    fragmentation_blank = learned_blank & (p_service > p_blank)
    conservative_blank = learned_blank & (p_blank >= p_service)
    fragmentation_rescued = fragmentation_blank & ~selected_blank
    selected_forced_blank = forced_blank & selected_blank
    selected_fragmentation_blank = fragmentation_blank & selected_blank
    selected_conservative_blank = conservative_blank & selected_blank
    _require(torch.equal(learned_blank, fragmentation_blank | conservative_blank), "learned Blank must partition")
    _require(not (fragmentation_blank & conservative_blank).any().item(), "Blank classes must be disjoint")
    _require(
        (p_blank[forced_blank] >= 1.0 - _TOLERANCE).all().item(),
        "forced Blank decisions must have probability one",
    )

    return IndependentPolicyDiagnostics(
        valid_ue_count=_detached_vector(valid_ue_count),
        p_blank=_detached_vector(p_blank),
        p_service=_detached_vector(p_service),
        p_best_ue=_detached_vector(p_best_ue),
        blank_margin=_detached_vector(blank_margin),
        margin_valid=_detached_vector(margin_valid),
        selected_blank=_detached_vector(selected_blank),
        greedy_blank=_detached_vector(greedy_blank),
        changed_from_greedy=_detached_vector(changed_from_greedy),
        forced_blank=_detached_vector(forced_blank),
        learned_blank=_detached_vector(learned_blank),
        fragmentation_blank=_detached_vector(fragmentation_blank),
        fragmentation_rescued=_detached_vector(fragmentation_rescued),
        conservative_blank=_detached_vector(conservative_blank),
        selected_forced_blank=_detached_vector(selected_forced_blank),
        selected_fragmentation_blank=_detached_vector(selected_fragmentation_blank),
        selected_conservative_blank=_detached_vector(selected_conservative_blank),
    )


def concatenate_independent_policy_diagnostics(
    rows: Sequence[IndependentPolicyDiagnostics],
) -> IndependentPolicyDiagnostics:
    """Concatenate detached per-decision records collected across policy calls."""
    _require(bool(rows), "at least one diagnostics row is required")
    values: dict[str, torch.Tensor] = {}
    for field in fields(IndependentPolicyDiagnostics):
        tensors = [getattr(row, field.name) for row in rows]
        _require(all(tensor.dim() == 1 for tensor in tensors), f"{field.name} must be one-dimensional")
        values[field.name] = torch.cat(tensors)
    lengths = {tensor.numel() for tensor in values.values()}
    _require(len(lengths) == 1, "diagnostic fields must have equal lengths")
    return IndependentPolicyDiagnostics(**values)


def _distribution_summary(values: torch.Tensor, name: str, summary: dict[str, float]) -> None:
    summary[f"{name}_mean"] = float(values.mean().item())
    for percentile in (10, 50, 90):
        summary[f"{name}_p{percentile}"] = float(torch.quantile(values, percentile / 100.0).item())


def summarize_independent_policy_diagnostics(
    diagnostics: IndependentPolicyDiagnostics,
    *,
    classification_prefix: str,
) -> dict[str, float]:
    """Produce rollout/evaluation scalar metrics from collected decision records."""
    count = diagnostics.p_blank.numel()
    _require(count > 0, "cannot summarize empty diagnostics")
    for field in fields(IndependentPolicyDiagnostics):
        _require(getattr(diagnostics, field.name).shape == (count,), "diagnostic fields must have equal lengths")

    denominator = float(count)
    learned_count = int(diagnostics.learned_blank.sum().item())
    valid_margins = diagnostics.blank_margin[diagnostics.margin_valid]
    summary = {
        "decision_count": denominator,
        "selected_blank_ratio": float(diagnostics.selected_blank.sum().item()) / denominator,
        "changed_from_greedy_count": float(diagnostics.changed_from_greedy.sum().item()),
        "changed_from_greedy_ratio": float(diagnostics.changed_from_greedy.sum().item()) / denominator,
        f"{classification_prefix}greedy_blank_ratio": float(diagnostics.greedy_blank.sum().item()) / denominator,
        f"{classification_prefix}forced_blank_ratio": float(diagnostics.forced_blank.sum().item()) / denominator,
        f"{classification_prefix}learned_blank_ratio": float(learned_count) / denominator,
        f"{classification_prefix}fragmentation_blank_ratio": float(diagnostics.fragmentation_blank.sum().item()) / denominator,
        "fragmentation_blank_count": float(diagnostics.fragmentation_blank.sum().item()),
        "fragmentation_rescue_count": float(diagnostics.fragmentation_rescued.sum().item()),
        "fragmentation_rescue_ratio": float(diagnostics.fragmentation_rescued.sum().item()) / denominator,
        f"{classification_prefix}conservative_blank_ratio": float(diagnostics.conservative_blank.sum().item()) / denominator,
        f"{classification_prefix}fragmentation_share_among_learned_blank": (
            float(diagnostics.fragmentation_blank.sum().item()) / learned_count if learned_count else 0.0
        ),
        f"{classification_prefix}learned_blank_count": float(learned_count),
        "selected_forced_blank_count": float(diagnostics.selected_forced_blank.sum().item()),
        "selected_forced_blank_ratio": float(diagnostics.selected_forced_blank.sum().item()) / denominator,
        "selected_fragmentation_blank_count": float(
            diagnostics.selected_fragmentation_blank.sum().item()
        ),
        "selected_fragmentation_blank_ratio": float(
            diagnostics.selected_fragmentation_blank.sum().item()
        ) / denominator,
        "selected_conservative_blank_count": float(
            diagnostics.selected_conservative_blank.sum().item()
        ),
        "selected_conservative_blank_ratio": float(
            diagnostics.selected_conservative_blank.sum().item()
        ) / denominator,
        "mean_valid_ue_count": float(diagnostics.valid_ue_count.float().mean().item()),
        "valid_margin_count": float(valid_margins.numel()),
    }
    _distribution_summary(diagnostics.p_blank, "p_blank", summary)
    _distribution_summary(diagnostics.p_service, "p_service", summary)
    _distribution_summary(diagnostics.p_best_ue, "p_best_ue", summary)
    if valid_margins.numel() == 0:
        for suffix in ("mean", "p10", "p50", "p90"):
            summary[f"blank_margin_{suffix}"] = 0.0
    else:
        _distribution_summary(valid_margins, "blank_margin", summary)
    return summary


if __name__ == "__main__":
    raise SystemExit(_main())

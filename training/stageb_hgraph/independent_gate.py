"""Fail-closed research gates for the PFQ-independent boundary experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .online_protocol import VERSION


CHECKPOINT_ITERATIONS = (4, 8, 12, 16, 20)
PPO_TTIS_PER_ITERATION = 512
GOODPUT_RATIO_MIN = 0.99
BLER_ABSOLUTE_DELTA_MAX = 0.002
EXPIRY_ABSOLUTE_DELTA_MAX = 0.0005
SCORE_UE_WEIGHT = 0.25
SCORE_DELAY_WEIGHT = 0.25

_METRIC_FIELDS = (
    "goodput_mbps",
    "tb_bler",
    "expiry_rate",
    "outcome_delay_p95_ms",
    "ue_macro_ea_rate_mbps",
    "ue_p10_ea_rate_mbps",
    "action_replay_mismatch_count",
    "illegal_action_count",
    "measured_tti_first",
    "measured_tti_last",
)


def _finite(metrics: Mapping[str, object], field: str) -> float:
    if field not in metrics or metrics[field] is None:
        raise ValueError(f"missing required metric {field}")
    value = float(metrics[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite required metric {field}")
    return value


def _validated_metrics(metrics: Mapping[str, object]) -> dict[str, float | bool]:
    if not bool(metrics.get("ea_metrics_available", False)):
        raise ValueError("expiry-aware metrics are unavailable or pre-correction")
    values: dict[str, float | bool] = {
        field: _finite(metrics, field) for field in _METRIC_FIELDS
    }
    if "independent_deletion_equal" not in metrics:
        raise ValueError("missing required metric independent_deletion_equal")
    values["independent_deletion_equal"] = bool(metrics["independent_deletion_equal"])
    values["ea_metrics_available"] = True
    if values["goodput_mbps"] <= 0.0:
        raise ValueError("goodput_mbps comparator must be positive")
    if values["ue_macro_ea_rate_mbps"] <= 0.0:
        raise ValueError("ue_macro_ea_rate_mbps comparator must be positive")
    if values["measured_tti_last"] < values["measured_tti_first"]:
        raise ValueError("invalid measured TTI interval")
    return values


def compare_manifests(
    reference: Mapping[str, object], treatment: Mapping[str, object]
) -> dict[str, object]:
    """Require source, binary, protocol, and canonical scenario equality."""

    checks: dict[str, bool] = {}
    for field in ("binary_sha256", "source_revision", "protocol_version"):
        if field not in reference or field not in treatment:
            raise ValueError(f"manifest missing {field}")
        checks[field] = reference[field] == treatment[field]
        if not checks[field]:
            raise ValueError(
                f"manifest {field} mismatch: reference={reference[field]!r} "
                f"treatment={treatment[field]!r}"
            )
    if int(reference["protocol_version"]) != VERSION:
        raise ValueError(
            f"manifest protocol_version is pre-correction: "
            f"got={reference['protocol_version']} required={VERSION}"
        )
    if not isinstance(reference.get("scenario"), Mapping) or not isinstance(
        treatment.get("scenario"), Mapping
    ):
        raise ValueError("manifest scenario must be a mapping")
    checks["scenario"] = dict(reference["scenario"]) == dict(treatment["scenario"])
    if not checks["scenario"]:
        raise ValueError("manifest scenario mismatch")
    return {"passed": True, "checks": checks}


def _comparison(
    *,
    numerator: float,
    denominator: float,
    threshold: float,
    passed: bool,
    reason: str,
) -> dict[str, object]:
    absolute_delta = numerator - denominator
    relative_delta = absolute_delta / denominator if denominator != 0.0 else None
    return {
        "numerator": numerator,
        "denominator": denominator,
        "absolute_delta": absolute_delta,
        "relative_delta": relative_delta,
        "threshold": threshold,
        "passed": bool(passed),
        "reason": reason,
    }


def architecture_gate(
    pfq_metrics: Mapping[str, object], actor_metrics: Mapping[str, object]
) -> dict[str, object]:
    pfq = _validated_metrics(pfq_metrics)
    actor = _validated_metrics(actor_metrics)
    if (
        int(pfq["measured_tti_first"]) != int(actor["measured_tti_first"])
        or int(pfq["measured_tti_last"]) != int(actor["measured_tti_last"])
    ):
        raise ValueError("PFQ and actor measured TTI intervals do not match")

    goodput_ratio = float(actor["goodput_mbps"]) / float(pfq["goodput_mbps"])
    bler_delta = float(actor["tb_bler"]) - float(pfq["tb_bler"])
    expiry_delta = float(actor["expiry_rate"]) - float(pfq["expiry_rate"])
    checks = {
        "goodput": _comparison(
            numerator=float(actor["goodput_mbps"]),
            denominator=float(pfq["goodput_mbps"]),
            threshold=GOODPUT_RATIO_MIN,
            passed=goodput_ratio >= GOODPUT_RATIO_MIN,
            reason=f"actor/PFQ goodput={goodput_ratio:.9g} must be >= {GOODPUT_RATIO_MIN}",
        ),
        "tb_bler": _comparison(
            numerator=float(actor["tb_bler"]),
            denominator=float(pfq["tb_bler"]),
            threshold=BLER_ABSOLUTE_DELTA_MAX,
            passed=bler_delta <= BLER_ABSOLUTE_DELTA_MAX + 1.0e-12,
            reason=f"actor-PFQ BLER={bler_delta:.9g} must be <= {BLER_ABSOLUTE_DELTA_MAX}",
        ),
        "expiry_rate": _comparison(
            numerator=float(actor["expiry_rate"]),
            denominator=float(pfq["expiry_rate"]),
            threshold=EXPIRY_ABSOLUTE_DELTA_MAX,
            passed=expiry_delta <= EXPIRY_ABSOLUTE_DELTA_MAX + 1.0e-12,
            reason=f"actor-PFQ expiry={expiry_delta:.9g} must be <= {EXPIRY_ABSOLUTE_DELTA_MAX}",
        ),
        "action_replay": _comparison(
            numerator=float(actor["action_replay_mismatch_count"]),
            denominator=0.0,
            threshold=0.0,
            passed=float(actor["action_replay_mismatch_count"]) == 0.0,
            reason="action replay mismatch count must equal zero",
        ),
        "legality": _comparison(
            numerator=float(actor["illegal_action_count"]),
            denominator=0.0,
            threshold=0.0,
            passed=float(actor["illegal_action_count"]) == 0.0,
            reason="illegal action count must equal zero",
        ),
        "pfq_deletion": {
            "numerator": int(bool(actor["independent_deletion_equal"])),
            "denominator": 1,
            "absolute_delta": int(bool(actor["independent_deletion_equal"])) - 1,
            "relative_delta": 0.0 if bool(actor["independent_deletion_equal"]) else -1.0,
            "threshold": "exact_true",
            "passed": bool(actor["independent_deletion_equal"]),
            "reason": "PFQ-field deletion must preserve graph, logits, trace, and Type-0 action exactly",
        },
    }
    return {
        "passed": all(bool(check["passed"]) for check in checks.values()),
        "checks": checks,
        "measured_tti_first": int(actor["measured_tti_first"]),
        "measured_tti_last": int(actor["measured_tti_last"]),
    }


def score_checkpoint(
    pfq_metrics: Mapping[str, object], actor_metrics: Mapping[str, object]
) -> dict[str, object]:
    pfq = _validated_metrics(pfq_metrics)
    actor = _validated_metrics(actor_metrics)
    goodput_delta = (
        float(actor["goodput_mbps"]) - float(pfq["goodput_mbps"])
    ) / float(pfq["goodput_mbps"])
    ue_delta = (
        float(actor["ue_macro_ea_rate_mbps"])
        - float(pfq["ue_macro_ea_rate_mbps"])
    ) / float(pfq["ue_macro_ea_rate_mbps"])
    worse_delay = max(
        0.0,
        (
            float(actor["outcome_delay_p95_ms"])
            - float(pfq["outcome_delay_p95_ms"])
        )
        / max(float(pfq["outcome_delay_p95_ms"]), 1.0e-12),
    )
    score = goodput_delta + SCORE_UE_WEIGHT * ue_delta - SCORE_DELAY_WEIGHT * worse_delay
    return {
        "score": score,
        "components": {
            "relative_goodput_delta": goodput_delta,
            "relative_ue_ea_rate_delta": ue_delta,
            "worse_outcome_delay_relative": worse_delay,
        },
        "weights": {
            "ue_ea_rate": SCORE_UE_WEIGHT,
            "worse_outcome_delay": SCORE_DELAY_WEIGHT,
        },
    }


def select_treatment_checkpoints(
    pfq_metrics: Mapping[str, object],
    candidates: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    _validated_metrics(pfq_metrics)
    reports: dict[str, object] = {}
    selected_iteration: int | None = None
    selected_score = -math.inf
    first_gate_iteration: int | None = None
    for iteration in CHECKPOINT_ITERATIONS:
        metrics = candidates.get(iteration)
        if metrics is None:
            reports[str(iteration)] = {
                "present": False,
                "scored": False,
                "reason": "candidate evaluation is missing",
            }
            continue
        gate = architecture_gate(pfq_metrics, metrics)
        entry: dict[str, object] = {
            "present": True,
            "scored": False,
            "architecture_gate": gate,
            "metrics": dict(metrics),
        }
        if gate["passed"]:
            if first_gate_iteration is None:
                first_gate_iteration = iteration
            scored = score_checkpoint(pfq_metrics, metrics)
            entry.update(scored)
            entry["scored"] = True
            if float(scored["score"]) > selected_score:
                selected_score = float(scored["score"])
                selected_iteration = iteration
        else:
            entry["reason"] = "hard architecture/replay/legality gate failed"
        reports[str(iteration)] = entry
    selected_metrics = (
        dict(candidates[selected_iteration]) if selected_iteration is not None else None
    )
    return {
        "checkpoint_cadence_iterations": list(CHECKPOINT_ITERATIONS),
        "measured_ppo_ttis_per_iteration": PPO_TTIS_PER_ITERATION,
        "candidates": reports,
        "selected_iteration": selected_iteration,
        "selected_score": selected_score if selected_iteration is not None else None,
        "selected_metrics": selected_metrics,
        "first_gate_iteration": first_gate_iteration,
        "first_gate_measured_ppo_ttis": (
            first_gate_iteration * PPO_TTIS_PER_ITERATION
            if first_gate_iteration is not None
            else None
        ),
    }


def _auxiliary_preference(
    treatment: str,
    pfq_metrics: Mapping[str, object],
    selection: Mapping[str, object],
) -> dict[str, object]:
    metrics = selection.get("selected_metrics")
    if metrics is None:
        return {"passed": False, "reason": "no architecture-gate-passing checkpoint"}
    gate = architecture_gate(pfq_metrics, metrics)
    pfq = _validated_metrics(pfq_metrics)
    actor = _validated_metrics(metrics)
    checks = {
        "architecture_gate": bool(gate["passed"]),
        "outcome_delay_p95_no_worse": float(actor["outcome_delay_p95_ms"])
        <= float(pfq["outcome_delay_p95_ms"]),
        "ue_macro_ea_rate_no_worse": float(actor["ue_macro_ea_rate_mbps"])
        >= float(pfq["ue_macro_ea_rate_mbps"]),
        "ue_p10_ea_rate_no_worse": float(actor["ue_p10_ea_rate_mbps"])
        >= float(pfq["ue_p10_ea_rate_mbps"]),
    }
    return {
        "passed": treatment in ("C", "D") and all(checks.values()),
        "checks": checks,
        "reason": (
            "auxiliary treatment meets PFQ secondary KPIs"
            if treatment in ("C", "D") and all(checks.values())
            else "auxiliary treatment is absent or fails at least one PFQ secondary KPI"
        ),
    }


def compare_treatments(
    pfq_metrics: Mapping[str, object],
    selections: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    _validated_metrics(pfq_metrics)
    treatment_reports: dict[str, object] = {}
    eligible: list[tuple[float, int, str]] = []
    simplicity = {"B": 0, "A": 1, "C": 2, "D": 3}
    for treatment in ("A", "B", "C", "D"):
        if treatment not in selections:
            continue
        selection = selections[treatment]
        aux = _auxiliary_preference(treatment, pfq_metrics, selection)
        has_gate = selection.get("selected_iteration") is not None
        allowed = has_gate and (treatment in ("A", "B") or bool(aux["passed"]))
        treatment_reports[treatment] = {
            "selected_iteration": selection.get("selected_iteration"),
            "selected_score": selection.get("selected_score"),
            "first_gate_measured_ppo_ttis": selection.get(
                "first_gate_measured_ppo_ttis"
            ),
            "architecture_gate_passed": has_gate,
            "auxiliary_preference_eligible": bool(aux["passed"]),
            "auxiliary_preference": aux,
            "winner_eligible": allowed,
        }
        if allowed:
            eligible.append(
                (
                    float(selection["selected_score"]),
                    -simplicity[treatment],
                    treatment,
                )
            )

    a_ttis = selections.get("A", {}).get("first_gate_measured_ppo_ttis")
    b_ttis = selections.get("B", {}).get("first_gate_measured_ppo_ttis")
    h2_supported = b_ttis is not None and (a_ttis is None or int(b_ttis) < int(a_ttis))
    if b_ttis is not None and a_ttis is not None and int(b_ttis) == int(a_ttis):
        h2_reason = (
            "A and B first pass at the same checkpoint; no demonstrated benefit "
            "at the available 2048-TTI resolution"
        )
    elif h2_supported:
        h2_reason = "B reaches the H1 gate earlier than A, or A does not reach it within budget"
    else:
        h2_reason = "PFQ BC did not demonstrate a lower measured PPO interaction count to H1"

    selected_treatment = max(eligible)[2] if eligible else None
    return {
        "treatments": treatment_reports,
        "selected_treatment": selected_treatment,
        "launch_paired_seeds_41_43": selected_treatment is not None,
        "h1": {
            "supported": selected_treatment is not None,
            "reason": (
                "at least one independent treatment passes the seed-41 architecture gate"
                if selected_treatment is not None
                else "no independent treatment passes the seed-41 architecture gate"
            ),
        },
        "h2": {
            "supported": h2_supported,
            "a_first_gate_measured_ppo_ttis": a_ttis,
            "b_first_gate_measured_ppo_ttis": b_ttis,
            "reason": h2_reason,
        },
    }


def load_outcome_delay_p95(
    path: Path, *, measured_tti_first: int, measured_tti_last: int
) -> dict[str, object]:
    delays: list[float] = []
    terminal_ttis: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            terminal_tti = int(row["terminal_tti"])
            if int(measured_tti_first) <= terminal_tti <= int(measured_tti_last):
                delays.append(float(row["system_time_ms"]))
                terminal_ttis.append(terminal_tti)
    if not delays:
        raise ValueError("no packet terminal samples in the measured TTI interval")
    return {
        "sample_count": len(delays),
        "terminal_tti_first": min(terminal_ttis),
        "terminal_tti_last": max(terminal_ttis),
        "measured_tti_first": int(measured_tti_first),
        "measured_tti_last": int(measured_tti_last),
        "outcome_delay_p95_ms": float(np.percentile(delays, 95)),
    }


def normalize_gate_metrics(summary: Mapping[str, object]) -> dict[str, object]:
    """Convert a deterministic evaluator summary to the locked gate schema."""

    if int(summary.get("protocol_version", -1)) != VERSION:
        raise ValueError("evaluator summary is pre-correction or has the wrong protocol_version")
    ea_ready = (
        summary.get("packet_outcome_delay_p95_ms") is not None
        and float(summary.get("packet_outcome_measured_count", 0.0)) > 0.0
        and float(summary.get("ue_expiry_aware_rate_active_count", 0.0)) > 0.0
    )
    if not ea_ready:
        raise ValueError("expiry-aware evaluator metrics are missing or empty")
    measured_count = _finite(summary, "stats_count")
    if measured_count <= 0.0:
        raise ValueError("evaluator measured window is empty")
    mapping = {
        "goodput_mbps": "stats_mean_goodput_mbps",
        "tb_bler": "measured_tb_bler",
        "outcome_delay_p95_ms": "packet_outcome_delay_p95_ms",
        "ue_macro_ea_rate_mbps": "ue_macro_expiry_aware_rate_mbps",
        "ue_p10_ea_rate_mbps": "ue_p10_expiry_aware_rate_mbps",
        "action_replay_mismatch_count": "stats_sum_action_replay_mismatch_count",
        "illegal_action_count": "stats_sum_illegal_action_count",
        "measured_tti_first": "measured_tti_first",
        "measured_tti_last": "measured_tti_last",
    }
    normalized: dict[str, object] = {
        target: _finite(summary, source) for target, source in mapping.items()
    }
    expired_bytes = _finite(summary, "stats_sum_expired_bytes")
    normalized["expiry_rate"] = expired_bytes / (measured_count * 108_000.0)
    normalized["measured_tti_first"] = int(normalized["measured_tti_first"])
    normalized["measured_tti_last"] = int(normalized["measured_tti_last"])
    normalized["independent_deletion_equal"] = bool(
        summary.get("independent_deletion_equal", False)
    )
    normalized["ea_metrics_available"] = True
    _validated_metrics(normalized)
    return normalized


def _load_json(path: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_named_paths(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        parsed[name] = path
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select-checkpoints")
    select.add_argument("--pfq-metrics", required=True)
    select.add_argument("--pfq-manifest", required=True)
    select.add_argument("--treatment-manifest", required=True)
    select.add_argument("--candidate", action="append", default=[], help="ITER=eval_summary.json")
    select.add_argument("--out", required=True)

    compare = subparsers.add_parser("compare-treatments")
    compare.add_argument("--pfq-metrics", required=True)
    compare.add_argument("--selection", action="append", default=[], help="TREATMENT=selection.json")
    compare.add_argument("--out-dir", required=True)
    normalize = subparsers.add_parser("normalize-eval")
    normalize.add_argument("--summary", required=True)
    normalize.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "normalize-eval":
        _write_json(Path(args.out), normalize_gate_metrics(_load_json(args.summary)))
        return 0
    if args.command == "select-checkpoints":
        compare_manifests(_load_json(args.pfq_manifest), _load_json(args.treatment_manifest))
        candidate_paths = _parse_named_paths(args.candidate)
        candidates = {int(key): _load_json(value) for key, value in candidate_paths.items()}
        report = select_treatment_checkpoints(_load_json(args.pfq_metrics), candidates)
        _write_json(Path(args.out), report)
        return 0
    selections = {
        key: _load_json(value) for key, value in _parse_named_paths(args.selection).items()
    }
    pfq_metrics = _load_json(args.pfq_metrics)
    report = compare_treatments(pfq_metrics, selections)
    out_dir = Path(args.out_dir)
    _write_json(out_dir / "checkpoint_selection.json", selections)
    _write_json(out_dir / "treatment_comparison.json", report)
    selected = report["selected_treatment"]
    winner = selections[str(selected)] if selected is not None else None
    gate = (
        architecture_gate(pfq_metrics, winner["selected_metrics"])
        if winner is not None
        else {"passed": False, "reason": "no treatment selected"}
    )
    _write_json(out_dir / "seed41_gate.json", gate)
    return 0 if bool(gate["passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate a fixed Stage-B HGraph checkpoint through the online bridge."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys

    _script_dir = __file__.rsplit("/", 1)[0]
    _repo_root = _script_dir.rsplit("/", 2)[0]
    sys.path = [p for p in sys.path if p not in ("", _script_dir)]
    sys.path.insert(0, _repo_root)

import argparse
import csv
import hashlib
import json
import os
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
from torch.distributions import Categorical


def _configure_torch_threads_from_env() -> None:
    num_threads_raw = os.environ.get("CUMAC_TORCH_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
    interop_raw = os.environ.get("CUMAC_TORCH_NUM_INTEROP_THREADS")
    if num_threads_raw:
        try:
            torch.set_num_threads(max(1, int(num_threads_raw)))
        except (RuntimeError, ValueError):
            pass
    if interop_raw:
        try:
            torch.set_num_interop_threads(max(1, int(interop_raw)))
        except (RuntimeError, ValueError):
            pass


_configure_torch_threads_from_env()


def _count_candidate_mask_reactivations(
    valid_at_step: torch.Tensor,
    candidate_ue_ids_at_step: torch.Tensor,
    *,
    n_ue: int,
) -> int:
    """Count invalid-to-valid transitions after aligning candidate columns by UE."""
    valid = torch.as_tensor(valid_at_step, dtype=torch.bool).detach().cpu()
    ue_ids = torch.as_tensor(candidate_ue_ids_at_step, dtype=torch.long).detach().cpu()
    if valid.ndim != 3:
        raise ValueError("candidate mask trace must have shape [PRG, cell, class]")
    if ue_ids.shape != valid.shape:
        raise ValueError("candidate UE identities must match the mask trace shape")
    if n_ue <= 0:
        raise ValueError("candidate mask reactivation audit requires at least one UE")
    if valid.shape[0] < 2:
        return 0
    valid_by_ue = torch.zeros(
        (valid.shape[0], valid.shape[1], int(n_ue)), dtype=torch.bool
    )
    for prg in range(valid.shape[0]):
        for cell in range(valid.shape[1]):
            keep = valid[prg, cell] & (ue_ids[prg, cell] >= 0)
            row_ids = ue_ids[prg, cell, keep]
            if bool((row_ids >= n_ue).any().item()):
                raise ValueError("candidate UE identity is out of range")
            if row_ids.unique().numel() != row_ids.numel():
                raise ValueError("candidate UE identity is duplicated within a row")
            valid_by_ue[prg, cell, row_ids] = True
    return int(((~valid_by_ue[:-1]) & valid_by_ue[1:]).sum().item())

from training.stageb_hgraph.bridge_env import BridgeEnvClient, BridgeEnvConfig
from training.stageb_hgraph.ctbs_headroom import (
    CtbsHeadroomInputs,
    compare_ctbs_reference_labels,
)
from training.stageb_hgraph.ctbs_non_starvation_guard import (
    MAXIMUM_OVERRIDES_PER_CELL_PER_TTI,
    MAXIMUM_SCHEDULED_DEQUEUE_ADEQUACY,
    MINIMUM_LOG1P_DRAIN_PRESSURE,
    MINIMUM_TTL_NORMALIZED_SERVICE_AGE,
    CtbsNonStarvationGuardConfig,
    guard_diagnostics_to_rows,
    summarize_guard_diagnostics,
)
from training.stageb_hgraph.candidate_policy import (
    CandidatePolicyConfig,
    select_candidate_policy_action,
)
from training.stageb_hgraph.edge_generator import EdgeGeneratorConfig, SparseEdgeGenerator
from training.stageb_hgraph.feature_snapshot import FeatureSnapshotBuilder, SnapshotBuilderConfig
from training.gnnrl.masks import apply_prg_action_mask, apply_ue_action_mask, build_slot_selection_mask
from training.stageb_hgraph.graph_ops import (
    apply_candidate_blank_budget,
    build_prg_candidate_valid_mask,
    decode_candidate_actions_to_type0,
    decode_joint_actions_to_type0,
    materialize_candidate_choices_to_type0,
    pack_prg_candidates,
    select_and_decode_budgeted_candidate_actions,
)
from training.stageb_hgraph.independent_actor import (
    IndependentActorConfig,
    PFQIndependentReuseActor,
)
from training.stageb_hgraph.independent_blank_diagnostics import (
    IndependentPolicyDiagnostics,
    concatenate_independent_policy_diagnostics,
    extract_failed_transmission_bytes,
    summarize_independent_policy_diagnostics,
    validate_measured_window_goodput_mbps,
)
from training.stageb_hgraph.independent_contract import (
    assert_graph_tensor_equality,
    candidate_truncation_count,
    resolve_runtime_candidate_contract,
    state_dict_sha256,
    strip_pfq_fields,
    validate_independent_checkpoint,
    validate_independent_graph,
)
from training.stageb_hgraph.independent_policy import (
    W0_V2_CHOSEN_LOGPROB_ATOL,
    W0_V2_VALID_LOGITS_ATOL,
    compare_candidate_padding_fp32_equivalence,
    compare_candidate_padding_semantics,
    replay_independent_trace,
    resolve_full_candidate_width,
    select_independent_action,
)
from training.stageb_hgraph.rescue_cohort_diagnostics import (
    RescueCellTtiRow,
    RescueDecisionDiagnostics,
    RescueUeTtiRow,
    concatenate_rescue_decision_diagnostics,
    join_rescue_cohort_tti,
    summarize_local_prg_phy_interference,
    summarize_rescue_cohorts,
)
from training.stageb_hgraph.phy_tbs_alignment_audit import (
    PHY_TBS_AUDIT_V1_PROTOCOL_ID,
    PHY_TBS_AUDIT_V2_CPU_OLLA_PROTOCOL_ID,
    PhyTbsAuditUeTtiRow,
    PhyTbsConfig,
    join_phy_tbs_alignment_tti,
    summarize_phy_tbs_alignment_audit,
)
from training.stageb_hgraph.service_accounting_audit import (
    NEW_TRANSMISSION_QUEUE_DEQUEUE,
    ServiceAccountingUeTtiRow,
    join_service_accounting_tti,
    summarize_service_accounting_audit,
)
from training.stageb_hgraph.blank_expiry_attribution_audit import (
    FROZEN_B4_STATE_DICT_SHA256,
    V1_PROTOCOL_ID,
    V2_FULL_HORIZON_PROTOCOL_ID,
    BlankExpiryTtiTrace,
    build_atomic_exposures,
    build_blank_expiry_attribution_report,
    decision_diagnostics_to_rows,
    packet_lifecycle_rows,
    read_formal_expired_packets,
)
from training.stageb_hgraph.model import HGraphModelConfig, StageBHGraphPolicy, build_model_from_config
from training.stageb_hgraph.online_imitation import _pick_device, _set_seed
from training.stageb_hgraph.online_ppo import (
    _apply_candidate_direct_service_bias,
    _candidate_direct_bias_selected_mean,
    _graph_to_device,
    _repair_ue_action_for_coverage,
)
from training.stageb_hgraph.reward_utils import extract_reward_metrics
from training.stageb_hgraph.online_protocol import VERSION
from training.stageb_hgraph.pfq_reference_calibration import filter_outcome_samples


def _sha256(path: Path) -> str:
    if not path.is_file():
        return "unavailable"
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _independent_deletion_equal(
    actor: PFQIndependentReuseActor,
    observation: Dict[str, Any],
    edge_generator: SparseEdgeGenerator,
    snapshot_cfg: SnapshotBuilderConfig,
    device: torch.device,
    *,
    mode: str,
    runtime_candidate_width: int | None = None,
) -> bool:
    full = _graph_to_device(
        edge_generator.build_graph(FeatureSnapshotBuilder(snapshot_cfg).build(observation)),
        device,
    )
    deleted = _graph_to_device(
        edge_generator.build_graph(
            FeatureSnapshotBuilder(snapshot_cfg).build(strip_pfq_fields(observation))
        ),
        device,
    )
    assert_graph_tensor_equality(full, deleted)
    width = (
        int(runtime_candidate_width)
        if runtime_candidate_width is not None
        else resolve_full_candidate_width(full)
    )
    full_packed = pack_prg_candidates(full, max_candidates=width, fixed_width=True)
    deleted_packed = pack_prg_candidates(deleted, max_candidates=width, fixed_width=True)
    with torch.inference_mode():
        full_encoded = actor.encode_graph(full)
        deleted_encoded = actor.encode_graph(deleted)
        if not all(torch.equal(full_encoded[key], deleted_encoded[key]) for key in full_encoded):
            return False
        full_trace, full_action = select_independent_action(
            actor, full, full_packed, mode=mode
        )
        deleted_trace, deleted_action = select_independent_action(
            actor, deleted, deleted_packed, mode=mode
        )
    return bool(
        torch.equal(full_trace.choice, deleted_trace.choice)
        and torch.equal(full_trace.valid_at_step, deleted_trace.valid_at_step)
        and torch.equal(full_trace.logp, deleted_trace.logp)
        and torch.equal(full_action.action_ue_select, deleted_action.action_ue_select)
        and torch.equal(full_action.action_prg_alloc, deleted_action.action_prg_alloc)
    )


def _load_actor_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[StageBHGraphPolicy | PFQIndependentReuseActor, Dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu")
    actor_cfg = ckpt.get("actor_config")
    if isinstance(actor_cfg, dict) and actor_cfg.get("actor_family") == PFQIndependentReuseActor.actor_family:
        validate_independent_checkpoint(ckpt)
        independent_keys = IndependentActorConfig.__dataclass_fields__
        actor = PFQIndependentReuseActor(
            IndependentActorConfig(
                **{key: actor_cfg[key] for key in independent_keys if key in actor_cfg}
            )
        )
    elif isinstance(actor_cfg, dict):
        actor = build_model_from_config(actor_cfg)
    else:
        actor = StageBHGraphPolicy(HGraphModelConfig())

    state = ckpt.get("actor_state")
    if state is None:
        state = ckpt.get("model_state", ckpt)
    target_state = actor.state_dict()
    compatible = {k: v for k, v in state.items() if k in target_state and target_state[k].shape == v.shape}
    if isinstance(actor, PFQIndependentReuseActor) and set(compatible) != set(target_state):
        raise ValueError("independent checkpoint does not exactly match its actor configuration")
    actor.load_state_dict(compatible, strict=False)
    actor.to(device)
    actor.eval()
    return actor, ckpt


def _checkpoint_arg(ckpt: Dict[str, Any], name: str, fallback: Any) -> Any:
    args = ckpt.get("args")
    if isinstance(args, dict) and args.get(name) is not None:
        return args[name]
    return fallback


def _validate_direct_evaluation_config(args: argparse.Namespace) -> None:
    if str(getattr(args, "candidate_policy_mode", "legacy")).lower() != "direct":
        return
    required = (
        ("candidate_budgeted_select_mode", "sample"),
        ("candidate_budgeted_hard_cap", 1),
        ("candidate_blank_budget_max_per_cell", -1),
        ("slot_repair_max_per_cell", 0),
        ("safe_fill_max_per_cell", 0),
        ("candidate_budgeted_tail_bias_coef", 0.0),
        ("candidate_budgeted_rescue_bias_coef", 0.0),
        ("candidate_budgeted_completion_bonus", 0.0),
        ("candidate_budgeted_tail_quota_enable", 0),
    )
    incompatible = [
        f"{name}={getattr(args, name, None)!r} (required {expected!r})"
        for name, expected in required
        if getattr(args, name, None) != expected
    ]
    if incompatible:
        raise ValueError(
            "direct candidate evaluation has incompatible configuration: "
            + ", ".join(incompatible)
        )


_GUARD_TRACE_CSV_FIELDS = (
    "tti",
    "cell_index",
    "local_prg_index",
    "blank_class_index",
    "actor_choice",
    "executed_choice",
    "guard_override",
    "selected_class",
    "selected_ue",
    "candidate_was_valid",
    "log1p_drain_pressure",
    "scheduled_dequeue_adequacy",
    "ttl_normalized_service_age",
    "ctbs_capacity_before_bytes",
    "ctbs_capacity_after_bytes",
    "reason",
)
_GUARD_FORMAL_ACTION_TTIS = 5000
_GUARD_FORMAL_N_CELL = 3
_GUARD_FORMAL_N_PRG = 17
_GUARD_FORMAL_DECISIONS_PER_TTI = 51
_GUARD_FORMAL_FULL_DECISION_COUNT = 255000
_GUARD_FORMAL_MEASURED_DECISION_COUNT = 204000


def _validate_non_starvation_guard_startup(
    *,
    guard_mode: str,
    max_steps: int,
    episode_horizon: int,
) -> None:
    if guard_mode == "off":
        return
    if guard_mode != "ctbs-blank-v1":
        raise ValueError("unknown independent non-starvation guard mode")
    if int(max_steps) != _GUARD_FORMAL_ACTION_TTIS:
        raise ValueError("ctbs-blank-v1 requires --max-steps 5000")
    if int(episode_horizon) != _GUARD_FORMAL_ACTION_TTIS:
        raise ValueError("ctbs-blank-v1 requires --episode-horizon 5000")


def _validate_non_starvation_guard_topology(
    *,
    guard_mode: str,
    n_cell: int,
    n_prg: int,
) -> None:
    if guard_mode == "off":
        return
    if guard_mode != "ctbs-blank-v1":
        raise ValueError("unknown independent non-starvation guard mode")
    n_cell_value = int(n_cell)
    n_prg_value = int(n_prg)
    if (n_cell_value, n_prg_value) != (
        _GUARD_FORMAL_N_CELL,
        _GUARD_FORMAL_N_PRG,
    ) or n_cell_value * n_prg_value != _GUARD_FORMAL_DECISIONS_PER_TTI:
        raise ValueError(
            "ctbs-blank-v1 requires the frozen topology of 3 cells and 17 local "
            "PRGs (51 decisions per action TTI)"
        )


def _validate_non_starvation_guard_completion(
    *,
    guard_mode: str,
    completed_action_ttis: int,
    full_rows: Sequence[object],
    measured_rows: Sequence[object],
) -> None:
    if guard_mode == "off":
        return
    if guard_mode != "ctbs-blank-v1":
        raise ValueError("unknown independent non-starvation guard mode")
    if int(completed_action_ttis) != _GUARD_FORMAL_ACTION_TTIS:
        raise RuntimeError("ctbs-blank-v1 requires exactly 5000 completed action TTIs")
    if len(full_rows) != _GUARD_FORMAL_FULL_DECISION_COUNT:
        raise RuntimeError(
            "ctbs-blank-v1 full trace must contain exactly 255000 decisions: "
            f"got={len(full_rows)}"
        )
    if len(measured_rows) != _GUARD_FORMAL_MEASURED_DECISION_COUNT:
        raise RuntimeError(
            "ctbs-blank-v1 measured trace must contain exactly 204000 decisions: "
            f"got={len(measured_rows)}"
        )


def _guard_diagnostics_to_validated_rows(
    *,
    tti: int,
    trace,
    n_cell: int,
    n_prg: int,
) -> list[dict[str, object]]:
    diagnostics = getattr(trace, "guard_diagnostics", None)
    if diagnostics is None:
        raise RuntimeError("ctbs-blank-v1 step is missing guard diagnostics")
    expected_count = int(n_cell) * int(n_prg)
    for field_name in diagnostics.__dataclass_fields__:
        values = getattr(diagnostics, field_name, None)
        if not isinstance(values, torch.Tensor) or values.ndim != 1:
            raise RuntimeError(
                "ctbs-blank-v1 guard diagnostics field is not a vector: "
                f"{field_name}"
            )
        if int(values.numel()) != expected_count:
            raise RuntimeError(
                "ctbs-blank-v1 guard diagnostics field length mismatch: "
                f"field={field_name} got={values.numel()} expected={expected_count}"
            )
    rows = guard_diagnostics_to_rows(tti=int(tti), diagnostics=diagnostics)
    if len(rows) != expected_count:
        raise RuntimeError(
            "ctbs-blank-v1 guard decision count mismatch: "
            f"got={len(rows)} expected={expected_count}"
        )
    expected_coordinates = {
        (cell, local_prg)
        for local_prg in range(int(n_prg))
        for cell in range(int(n_cell))
    }
    coordinates = {
        (int(row["cell_index"]), int(row["local_prg_index"])) for row in rows
    }
    if coordinates != expected_coordinates:
        raise RuntimeError("ctbs-blank-v1 guard decision coordinates are incomplete")

    overrides_by_cell: dict[int, int] = {}
    for row in rows:
        if not bool(row["guard_override"]):
            if int(row["actor_choice"]) != int(row["executed_choice"]):
                raise ValueError(
                    "ctbs-blank-v1 changed a choice without an override diagnostic"
                )
            continue
        cell = int(row["cell_index"])
        overrides_by_cell[cell] = overrides_by_cell.get(cell, 0) + 1

    if max(overrides_by_cell.values(), default=0) > MAXIMUM_OVERRIDES_PER_CELL_PER_TTI:
        raise ValueError("ctbs-blank-v1 exceeded one override per cell/TTI")

    for row in rows:
        if not bool(row["guard_override"]):
            continue
        actor_choice = int(row["actor_choice"])
        executed_choice = int(row["executed_choice"])
        blank_class_index = int(row["blank_class_index"])
        if actor_choice != blank_class_index:
            raise ValueError("ctbs-blank-v1 violated the Blank-only override contract")
        if executed_choice == blank_class_index or actor_choice == executed_choice:
            raise ValueError("ctbs-blank-v1 override did not execute a nonblank choice")
        if not bool(row["candidate_was_valid"]):
            raise ValueError("ctbs-blank-v1 override selected an invalid candidate")
        if int(row["ctbs_capacity_after_bytes"]) <= int(
            row["ctbs_capacity_before_bytes"]
        ):
            raise ValueError("ctbs-blank-v1 override requires a positive CTBS capacity delta")
    return rows


def _build_non_starvation_guard_summary(
    *,
    guard_mode: str,
    full_rows: list[dict[str, object]],
    measured_rows: list[dict[str, object]],
) -> dict[str, object]:
    if guard_mode not in ("off", "ctbs-blank-v1"):
        raise ValueError("unknown independent non-starvation guard mode")
    if guard_mode == "off" and (full_rows or measured_rows):
        raise ValueError("guard-off evaluation must not accumulate guard rows")
    return {
        "independent_non_starvation_guard": guard_mode,
        "non_starvation_guard_config": {
            "minimum_log1p_drain_pressure": MINIMUM_LOG1P_DRAIN_PRESSURE,
            "maximum_scheduled_dequeue_adequacy": (
                MAXIMUM_SCHEDULED_DEQUEUE_ADEQUACY
            ),
            "minimum_ttl_normalized_service_age": (
                MINIMUM_TTL_NORMALIZED_SERVICE_AGE
            ),
            "maximum_overrides_per_cell_per_tti": (
                MAXIMUM_OVERRIDES_PER_CELL_PER_TTI
            ),
            "override_nonblank": False,
            "require_positive_ctbs_capacity_delta": True,
        },
        "guard_full_summary": summarize_guard_diagnostics(full_rows),
        "guard_measured_summary": summarize_guard_diagnostics(measured_rows),
    }


def _write_non_starvation_guard_trace(
    path: Path,
    rows: list[dict[str, object]],
    *,
    guard_mode: str,
) -> None:
    if guard_mode == "off":
        if rows:
            raise ValueError("guard-off evaluation must not write guard rows")
        return
    if guard_mode != "ctbs-blank-v1":
        raise ValueError("unknown independent non-starvation guard mode")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_GUARD_TRACE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _select_joint_action(
    actor: StageBHGraphPolicy | PFQIndependentReuseActor,
    graph,
    *,
    decode_mode: str,
    candidate_policy_mode: str = "legacy",
    slot_repair_max_per_cell: int = 0,
    slot_repair_min_backlog_bytes: float = 1.0,
    safe_fill_max_per_cell: int = 0,
    safe_fill_min_quality_db: float = -2.0,
    safe_fill_max_quality_gap_db: float = 8.0,
    safe_fill_max_prg_risk: float = 0.70,
    safe_fill_min_backlog_bytes: float = 1.0,
    safe_fill_max_ue_tb_err: float = 0.45,
    candidate_blank_budget_max_per_cell: int = -1,
    candidate_blank_budget_min_decision_logit: float = -1.0e9,
    candidate_decode_demand_slack_bytes: float = 3000.0,
    candidate_marginal_headroom_slack_bytes: float = -1.0,
    candidate_headroom_rank_mode: str = "stable",
    candidate_budgeted_select_mode: str = "off",
    candidate_budgeted_hard_cap: int = 1,
    candidate_budgeted_usage_penalty: float = 0.35,
    candidate_budgeted_overcap_penalty: float = 8.0,
    candidate_budgeted_first_service_bonus: float = 0.0,
    candidate_budgeted_tail_bias_coef: float = 0.0,
    candidate_budgeted_tail_bias_service_target: float = 0.75,
    candidate_budgeted_tail_bias_ttl_guard_ms: float = 20.0,
    candidate_budgeted_tail_bias_max: float = 1.5,
    candidate_budgeted_rescue_bias_coef: float = 0.0,
    candidate_budgeted_completion_bonus: float = 0.0,
    candidate_budgeted_tail_quota_enable: int = 0,
    candidate_seq_state_coef: float = 1.0,
    candidate_seq_fail_risk_feature_offset: int = -1,
    independent_policy_diagnostics: bool = False,
    collect_blank_expiry_attribution: bool = False,
    independent_runtime_candidate_width: int | None = None,
    ctbs_headroom_inputs: CtbsHeadroomInputs | None = None,
    independent_non_starvation_guard: str = "off",
) -> Dict[str, Any]:
    guard_mode = str(independent_non_starvation_guard)
    if guard_mode not in ("off", "ctbs-blank-v1"):
        raise ValueError(
            "independent non-starvation guard must be off or ctbs-blank-v1"
        )
    guard_config = CtbsNonStarvationGuardConfig(
        enabled=guard_mode == "ctbs-blank-v1"
    )
    if guard_config.enabled:
        if not isinstance(actor, PFQIndependentReuseActor):
            raise ValueError("ctbs-blank-v1 requires an independent actor")
        if decode_mode != "argmax":
            raise ValueError("ctbs-blank-v1 requires --decode-mode argmax")
        if not independent_policy_diagnostics:
            raise ValueError("ctbs-blank-v1 requires independent policy diagnostics")
        if ctbs_headroom_inputs is None:
            raise ValueError("ctbs-blank-v1 requires real CTBS headroom inputs")
    if independent_policy_diagnostics and not isinstance(actor, PFQIndependentReuseActor):
        raise ValueError("independent policy diagnostics require an independent actor")
    if (
        decode_mode == "hierarchical_argmax"
        and not isinstance(actor, PFQIndependentReuseActor)
    ):
        raise ValueError("--decode-mode hierarchical_argmax requires an independent actor")
    if isinstance(actor, PFQIndependentReuseActor):
        independent_decode_modes = {
            "argmax": "greedy",
            "hierarchical_argmax": "hierarchical_greedy",
        }
        if decode_mode not in independent_decode_modes:
            raise ValueError(
                "independent evaluation is deterministic and requires --decode-mode "
                "argmax or hierarchical_argmax"
            )
        runtime_width = (
            int(independent_runtime_candidate_width)
            if independent_runtime_candidate_width is not None
            else resolve_full_candidate_width(graph)
        )
        validate_independent_graph(
            graph,
            max_candidates=runtime_width,
        )
        packed = pack_prg_candidates(
            graph,
            max_candidates=runtime_width,
            fixed_width=True,
        )
        truncation_count = candidate_truncation_count(graph, packed)
        if truncation_count:
            raise ValueError(
                f"independent candidate truncation is forbidden: count={truncation_count}"
            )
        with torch.inference_mode():
            trace, decoded = select_independent_action(
                actor,
                graph,
                packed,
                mode=independent_decode_modes[decode_mode],
                collect_diagnostics=independent_policy_diagnostics,
                collect_blank_expiry_attribution=collect_blank_expiry_attribution,
                ctbs_headroom_inputs=ctbs_headroom_inputs,
                non_starvation_guard=guard_config,
            )
            _logp, _entropy, replayed = replay_independent_trace(
                actor,
                graph,
                packed,
                trace,
            )
        mismatch = int(
            (decoded.action_ue_select != replayed.action_ue_select).sum().item()
            + (decoded.action_prg_alloc != replayed.action_prg_alloc).sum().item()
        )
        chosen = decoded.chosen_ue_per_prg[decoded.chosen_ue_per_prg >= 0]
        return {
            "action_ue_select": decoded.action_ue_select.cpu().numpy(),
            "action_prg_alloc": decoded.action_prg_alloc.cpu().numpy(),
            "decoded_chosen_ue_per_prg": decoded.chosen_ue_per_prg.detach().cpu(),
            "prg_action_class": trace.choice.detach().cpu(),
            "independent_policy_trace": trace,
            "candidate_ue_ids_at_step": packed.ue_ids.reshape(
                graph.snapshot.static_meta.n_cell,
                graph.snapshot.static_meta.n_prg,
                packed.max_candidates,
            ).permute(1, 0, 2).detach().cpu(),
            "blank_ratio": float(
                (trace.choice == packed.blank_class_index).float().mean().item()
            ),
            "fill_ratio": float(decoded.prg_chosen_mask.float().mean().item()),
            "cap_drop_count": 0.0,
            "fallback_success_count": 0.0,
            "safe_fill_count": 0.0,
            "missed_safe_opportunity_count": 0.0,
            "blank_budget_repair_count": 0.0,
            "final_blank_count": float(decoded.final_blank_count),
            "post_decode_policy_blank_count": float(decoded.final_blank_count),
            "post_decode_no_valid_candidate_count": 0.0,
            "post_decode_demand_cap_count": 0.0,
            "post_decode_no_slot_count": 0.0,
            "post_decode_other_drop_count": 0.0,
            "slot_repair_count": 0.0,
            "coverage_repair_count": 0.0,
            "non_hard_override_count": 0.0,
            "candidate_policy_mode_direct": 0.0,
            "budgeted_changed_count": 0.0,
            "budgeted_no_headroom_count": 0.0,
            "budgeted_active_ue_count": 0.0,
            "budgeted_tail_bias_coef": 0.0,
            "budgeted_tail_bias_candidate_mean": 0.0,
            "budgeted_tail_bias_selected_mean": 0.0,
            "budgeted_tail_bias_active_count": 0.0,
            "uniq_ue": float(torch.unique(chosen).numel()) if chosen.numel() else 0.0,
            "action_replay_mismatch_count": float(mismatch),
            "runtime_candidate_width": int(runtime_width),
            "candidate_truncation_count": float(truncation_count),
            "independent_mode": 1.0,
        }
    _validate_direct_evaluation_config(
        argparse.Namespace(
            candidate_policy_mode=str(candidate_policy_mode).lower(),
            candidate_budgeted_select_mode=str(candidate_budgeted_select_mode).lower(),
            candidate_budgeted_hard_cap=int(candidate_budgeted_hard_cap),
            candidate_blank_budget_max_per_cell=int(candidate_blank_budget_max_per_cell),
            slot_repair_max_per_cell=int(slot_repair_max_per_cell),
            safe_fill_max_per_cell=int(safe_fill_max_per_cell),
            candidate_budgeted_tail_bias_coef=float(candidate_budgeted_tail_bias_coef),
            candidate_budgeted_rescue_bias_coef=float(candidate_budgeted_rescue_bias_coef),
            candidate_budgeted_completion_bonus=float(candidate_budgeted_completion_bonus),
            candidate_budgeted_tail_quota_enable=int(candidate_budgeted_tail_quota_enable),
        )
    )
    with torch.inference_mode():
        actor_out = actor.forward_graph(graph)

    if str(candidate_policy_mode).lower() == "direct":
        candidate_logits = actor_out.get("candidate_prg_logits")
        if candidate_logits is None:
            raise ValueError("direct candidate evaluation requires candidate_prg_logits")
        packed = pack_prg_candidates(
            graph,
            max_candidates=max(0, int(candidate_logits.shape[-1]) - 1),
            fixed_width=True,
        )
        trace = select_candidate_policy_action(
            graph,
            packed,
            packed.mask,
            candidate_logits,
            CandidatePolicyConfig(
                mode="greedy",
                demand_cap_slack_bytes=float(candidate_decode_demand_slack_bytes),
                hard_cap=bool(int(candidate_budgeted_hard_cap)),
                usage_penalty=float(candidate_budgeted_usage_penalty),
                overcap_penalty=float(candidate_budgeted_overcap_penalty),
                first_service_bonus=float(candidate_budgeted_first_service_bonus),
            ),
        )
        decoded = materialize_candidate_choices_to_type0(graph, packed, trace.choice)
        chosen = decoded.chosen_ue_per_prg[decoded.chosen_ue_per_prg >= 0]
        return {
            "action_ue_select": decoded.action_ue_select.cpu().numpy(),
            "action_prg_alloc": decoded.action_prg_alloc.cpu().numpy(),
            "prg_action_class": trace.choice.detach().cpu().to(torch.long),
            "candidate_policy_trace": trace,
            "blank_ratio": float(
                (trace.choice == packed.blank_class_index).float().mean().item()
            ),
            "fill_ratio": float(decoded.prg_chosen_mask.float().mean().item()),
            "cap_drop_count": float(decoded.cap_drop_count),
            "fallback_success_count": float(decoded.fallback_success_count),
            "safe_fill_count": 0.0,
            "missed_safe_opportunity_count": float(decoded.missed_safe_opportunity_count),
            "blank_budget_repair_count": 0.0,
            "final_blank_count": float(decoded.final_blank_count),
            "post_decode_policy_blank_count": float(
                (trace.choice == packed.blank_class_index).sum().item()
            ),
            "post_decode_no_valid_candidate_count": 0.0,
            "post_decode_demand_cap_count": 0.0,
            "post_decode_no_slot_count": float(decoded.slot_capacity_drop_count),
            "post_decode_other_drop_count": float(decoded.illegal_choice_count),
            "slot_repair_count": 0.0,
            "coverage_repair_count": 0.0,
            "non_hard_override_count": 0.0,
            "candidate_policy_mode_direct": 1.0,
            "budgeted_changed_count": 0.0,
            "budgeted_no_headroom_count": 0.0,
            "budgeted_active_ue_count": 0.0,
            "budgeted_tail_bias_coef": 0.0,
            "budgeted_tail_bias_candidate_mean": 0.0,
            "budgeted_tail_bias_selected_mean": 0.0,
            "budgeted_tail_bias_active_count": 0.0,
            "uniq_ue": float(torch.unique(chosen).numel()) if chosen.numel() > 0 else 0.0,
        }

    device = actor_out["ue_logits"].device
    meta = graph.snapshot.static_meta
    action_mask_ue = (
        meta.action_mask_ue
        if meta.action_mask_ue is not None
        else torch.ones((meta.n_ue,), dtype=torch.bool, device=device)
    ).to(device=device, dtype=torch.bool).unsqueeze(0)
    action_mask_cell_ue = meta.action_mask_cell_ue.to(device=device, dtype=torch.bool).unsqueeze(0)
    ue_logits, _ue_valid = apply_ue_action_mask(
        actor_out["ue_logits"].unsqueeze(0),
        action_mask_ue,
        n_cell=meta.n_cell,
        action_mask_cell_ue=action_mask_cell_ue,
    )
    ue_logits = ue_logits.squeeze(0)
    if decode_mode == "sample":
        ue_class_raw = Categorical(logits=ue_logits).sample()
    else:
        ue_class_raw = ue_logits.argmax(dim=-1)
    ue_class, coverage_repair_count = _repair_ue_action_for_coverage(
        ue_class_raw,
        graph,
        actor_out,
        max_repairs_per_cell=int(slot_repair_max_per_cell),
        min_backlog_bytes=float(slot_repair_min_backlog_bytes),
    )
    with torch.inference_mode():
        prg_actor_out = actor.forward_graph(graph, slot_ue_class_for_prg=ue_class)
    use_candidate_prg = "candidate_prg_logits" in prg_actor_out
    budgeted_metrics: Dict[str, float] = {}
    tail_scores: torch.Tensor | None = None
    budget_mode = str(candidate_budgeted_select_mode).lower()
    if use_candidate_prg:
        packed = pack_prg_candidates(
            graph,
            max_candidates=max(0, int(prg_actor_out["candidate_prg_logits"].shape[-1]) - 1),
            fixed_width=True,
        )
        candidate_valid, _selected_slot_mask, _selected_ue_mask = build_prg_candidate_valid_mask(
            graph,
            packed,
            ue_class,
        )
        prg_logits = prg_actor_out["candidate_prg_logits"].masked_fill(
            ~candidate_valid.to(device=device, dtype=torch.bool),
            -1.0e9,
        )
        blank_class = int(packed.blank_class_index)
        if budget_mode == "off" and float(candidate_budgeted_tail_bias_coef) > 0.0:
            prg_logits, budgeted_metrics, tail_scores, _rescue_scores = _apply_candidate_direct_service_bias(
                prg_logits,
                graph,
                packed,
                candidate_valid,
                tail_bias_coef=float(candidate_budgeted_tail_bias_coef),
                tail_bias_service_target=float(candidate_budgeted_tail_bias_service_target),
                tail_bias_ttl_guard_ms=float(candidate_budgeted_tail_bias_ttl_guard_ms),
                tail_bias_max=float(candidate_budgeted_tail_bias_max),
            )
    else:
        selected_slot_mask = build_slot_selection_mask(
            ue_class.unsqueeze(0),
            action_mask_ue,
            n_cell=meta.n_cell,
            action_mask_cell_ue=action_mask_cell_ue,
        )
        prg_logits, _prg_valid = apply_prg_action_mask(
            prg_actor_out["prg_logits"].unsqueeze(0),
            meta.action_mask_prg_cell.to(device=device, dtype=torch.bool).reshape(1, meta.n_cell, meta.n_prg),
            n_cell=meta.n_cell,
            action_mask_cell_ue=action_mask_cell_ue,
            action_mask_ue=action_mask_ue,
            selected_slot_mask=selected_slot_mask,
        )
        prg_logits = prg_logits.squeeze(0)
        blank_class = int(meta.n_sched_ue)
    if decode_mode == "sample":
        prg_class = Categorical(logits=prg_logits).sample()
    else:
        prg_class = prg_logits.argmax(dim=-1)

    blank_budget_repair_count = 0
    decoded = None
    if use_candidate_prg:
        if budget_mode != "off":
            decoded, prg_class, budgeted_metrics = select_and_decode_budgeted_candidate_actions(
                graph,
                packed,
                candidate_valid,
                ue_class.detach().cpu(),
                prg_logits,
                mode=("sample" if budget_mode == "sample" else "greedy"),
                demand_cap_slack_bytes=float(candidate_decode_demand_slack_bytes),
                marginal_headroom_slack_bytes=float(candidate_marginal_headroom_slack_bytes),
                hard_cap=bool(int(candidate_budgeted_hard_cap)),
                usage_penalty=float(candidate_budgeted_usage_penalty),
                overcap_penalty=float(candidate_budgeted_overcap_penalty),
                first_service_bonus=float(candidate_budgeted_first_service_bonus),
                tail_bias_coef=float(candidate_budgeted_tail_bias_coef),
                tail_bias_service_target=float(candidate_budgeted_tail_bias_service_target),
                tail_bias_ttl_guard_ms=float(candidate_budgeted_tail_bias_ttl_guard_ms),
                tail_bias_max=float(candidate_budgeted_tail_bias_max),
                seq_state_weights=prg_actor_out.get("candidate_seq_state_weights"),
                seq_state_coef=float(candidate_seq_state_coef),
                seq_fail_risk_feature_offset=int(candidate_seq_fail_risk_feature_offset),
                safe_fill_max_per_cell=int(safe_fill_max_per_cell),
                safe_fill_min_quality_db=float(safe_fill_min_quality_db),
                safe_fill_max_quality_gap_db=float(safe_fill_max_quality_gap_db),
                safe_fill_max_prg_risk=float(safe_fill_max_prg_risk),
                safe_fill_min_backlog_bytes=float(safe_fill_min_backlog_bytes),
                safe_fill_max_ue_tb_err=float(safe_fill_max_ue_tb_err),
                headroom_rank_mode=str(candidate_headroom_rank_mode),
                candidate_rank_scores=prg_logits[:, :blank_class].detach(),
            )
        else:
            if tail_scores is not None:
                budgeted_metrics["candidate_budgeted_tail_bias_selected_mean"] = _candidate_direct_bias_selected_mean(
                    tail_scores,
                    prg_class,
                    int(blank_class),
                )
            prg_class, blank_budget_repair_count = apply_candidate_blank_budget(
                graph,
                packed,
                prg_logits,
                prg_class,
                max_blank_per_cell=int(candidate_blank_budget_max_per_cell),
                min_decision_logit=float(candidate_blank_budget_min_decision_logit),
            )
        if decoded is None:
            decoded = decode_candidate_actions_to_type0(
                graph,
                packed,
                ue_class.detach().cpu(),
                prg_class.detach().cpu(),
                safe_fill_max_per_cell=int(safe_fill_max_per_cell),
                safe_fill_min_quality_db=float(safe_fill_min_quality_db),
                safe_fill_max_quality_gap_db=float(safe_fill_max_quality_gap_db),
                safe_fill_max_prg_risk=float(safe_fill_max_prg_risk),
                safe_fill_min_backlog_bytes=float(safe_fill_min_backlog_bytes),
                safe_fill_max_ue_tb_err=float(safe_fill_max_ue_tb_err),
                demand_cap_slack_bytes=float(candidate_decode_demand_slack_bytes),
                marginal_headroom_slack_bytes=float(candidate_marginal_headroom_slack_bytes),
                headroom_rank_mode=str(candidate_headroom_rank_mode),
                candidate_rank_scores=prg_logits[:, :blank_class].detach(),
            )
    else:
        decoded = decode_joint_actions_to_type0(
            graph,
            ue_class.detach().cpu(),
            prg_class.detach().cpu(),
        )
    return {
        "action_ue_select": decoded.action_ue_select.cpu().numpy(),
        "action_prg_alloc": decoded.action_prg_alloc.cpu().numpy(),
        "blank_ratio": float((prg_class == blank_class).float().mean().item()),
        "fill_ratio": float(decoded.prg_chosen_mask.float().mean().item()),
        "cap_drop_count": float(decoded.cap_drop_count),
        "fallback_success_count": float(decoded.fallback_success_count),
        "safe_fill_count": float(decoded.safe_fill_count),
        "missed_safe_opportunity_count": float(decoded.missed_safe_opportunity_count),
        "blank_budget_repair_count": float(blank_budget_repair_count),
        "final_blank_count": float(decoded.final_blank_count),
        "post_decode_policy_blank_count": float(decoded.post_decode_policy_blank_count),
        "post_decode_no_valid_candidate_count": float(decoded.post_decode_no_valid_candidate_count),
        "post_decode_demand_cap_count": float(decoded.post_decode_demand_cap_count),
        "post_decode_no_slot_count": float(decoded.post_decode_no_slot_count),
        "post_decode_other_drop_count": float(decoded.post_decode_other_drop_count),
        "slot_repair_count": float(coverage_repair_count),
        "coverage_repair_count": float(coverage_repair_count),
        "non_hard_override_count": float(decoded.non_hard_override_count),
        "candidate_policy_mode_direct": 0.0,
        "budgeted_changed_count": float(budgeted_metrics.get("candidate_budgeted_changed_count", 0.0)),
        "budgeted_no_headroom_count": float(budgeted_metrics.get("candidate_budgeted_no_headroom_count", 0.0)),
        "budgeted_active_ue_count": float(budgeted_metrics.get("candidate_budgeted_active_ue_count", 0.0)),
        "budgeted_tail_bias_coef": float(budgeted_metrics.get("candidate_budgeted_tail_bias_coef", 0.0)),
        "budgeted_tail_bias_candidate_mean": float(
            budgeted_metrics.get("candidate_budgeted_tail_bias_candidate_mean", 0.0)
        ),
        "budgeted_tail_bias_selected_mean": float(
            budgeted_metrics.get("candidate_budgeted_tail_bias_selected_mean", 0.0)
        ),
        "budgeted_tail_bias_active_count": float(
            budgeted_metrics.get("candidate_budgeted_tail_bias_active_count", 0.0)
        ),
        "uniq_ue": float(torch.unique(decoded.chosen_ue_per_prg[decoded.chosen_ue_per_prg >= 0]).numel())
        if bool((decoded.chosen_ue_per_prg >= 0).any().item())
        else 0.0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a fixed HGraph checkpoint against the online bridge")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--socket-path", required=True)
    p.add_argument("--connect-timeout-s", type=float, default=20.0)
    p.add_argument("--episode-horizon", type=int, default=4096)
    p.add_argument("--max-steps", type=int, default=0, help="0 means run until env signals done")
    p.add_argument("--ue-prg-topk", type=int, default=None, help="Default: checkpoint args, else 6")
    p.add_argument("--ue-prg-diversity-extra", type=int, default=None, help="Default: checkpoint args, else 0")
    p.add_argument("--tb-err-ema-alpha", type=float, default=None, help="Default: checkpoint args, else 0.10")
    p.add_argument(
        "--decode-mode",
        choices=["argmax", "hierarchical_argmax", "sample"],
        default="argmax",
    )
    p.add_argument(
        "--independent-policy-diagnostics",
        type=int,
        choices=(0, 1),
        default=0,
        help="collect detached PFQ-independent Blank diagnostics without changing actions",
    )
    p.add_argument(
        "--independent-headroom-accounting",
        choices=("raw-v1", "ctbs-v1"),
        default="raw-v1",
        help="B4 sequential headroom treatment; ctbs-v1 requires the causal pre-action handshake",
    )
    p.add_argument(
        "--independent-non-starvation-guard",
        choices=("off", "ctbs-blank-v1"),
        default="off",
        help="frozen blank-only CTBS non-starvation guard; evaluation only",
    )
    p.add_argument(
        "--independent-service-accounting-audit",
        type=int,
        choices=(0, 1),
        default=0,
        help="apply the frozen read-only scheduled/raw service audit to measured TTIs",
    )
    p.add_argument(
        "--independent-phy-tbs-alignment-audit",
        type=int,
        choices=(0, 1),
        default=0,
        help="apply the frozen read-only decision-time PHY/TBS reconstruction audit",
    )
    p.add_argument(
        "--independent-blank-expiry-attribution-audit",
        type=int,
        choices=(0, 1),
        default=0,
        help="collect the frozen read-only expired-packet exposure attribution audit",
    )
    p.add_argument(
        "--blank-expiry-trace-first-tti",
        type=int,
        default=602,
        help="first action TTI retained for full expired-packet lifecycle coverage",
    )
    p.add_argument(
        "--independent-blank-expiry-attribution-protocol",
        choices=(V1_PROTOCOL_ID, V2_FULL_HORIZON_PROTOCOL_ID),
        default=V1_PROTOCOL_ID,
        help="versioned window and reporting contract for Blank-expiry attribution",
    )
    p.add_argument(
        "--independent-phy-tbs-alignment-protocol",
        choices=(
            PHY_TBS_AUDIT_V1_PROTOCOL_ID,
            PHY_TBS_AUDIT_V2_CPU_OLLA_PROTOCOL_ID,
        ),
        default=PHY_TBS_AUDIT_V1_PROTOCOL_ID,
        help="versioned causal-input and gate contract for the read-only PHY/TBS audit",
    )
    p.add_argument(
        "--independent-runtime-candidate-width",
        type=int,
        default=0,
        help="0 resolves width from full association topology; a positive value is a fail-closed padding width",
    )
    p.add_argument(
        "--independent-equivalence-reference-width",
        type=int,
        default=0,
        help="positive width enables per-step semantic comparison against the runtime width",
    )
    p.add_argument(
        "--independent-equivalence-protocol",
        choices=("bitwise-w0-v1", "fp32-w0-v2"),
        default="bitwise-w0-v1",
        help="cross-width numerical protocol; W0-v2 freezes FP32 tolerances while keeping discrete gates exact",
    )
    p.add_argument("--slot-repair-max-per-cell", type=int, default=None, help="Default: checkpoint args, else 0")
    p.add_argument("--slot-repair-min-backlog-bytes", type=float, default=None, help="Default: checkpoint args, else 1.0")
    p.add_argument("--safe-fill-max-per-cell", type=int, default=None, help="Default: checkpoint args, else 0")
    p.add_argument("--safe-fill-min-quality-db", type=float, default=None, help="Default: checkpoint args, else -2.0")
    p.add_argument("--safe-fill-max-quality-gap-db", type=float, default=None, help="Default: checkpoint args, else 8.0")
    p.add_argument("--safe-fill-max-prg-risk", type=float, default=None, help="Default: checkpoint args, else 0.70")
    p.add_argument("--safe-fill-min-backlog-bytes", type=float, default=None, help="Default: checkpoint args, else 1.0")
    p.add_argument("--safe-fill-max-ue-tb-err", type=float, default=None, help="Default: checkpoint args, else 0.45")
    p.add_argument(
        "--candidate-blank-budget-max-per-cell",
        type=int,
        default=None,
        help="Default: checkpoint args, else -1. -1 disables deployment blank-budget repair.",
    )
    p.add_argument(
        "--candidate-blank-budget-min-decision-logit",
        type=float,
        default=None,
        help="Default: checkpoint args, else -1e9. Blank-minus-best-candidate threshold for budget slots.",
    )
    p.add_argument(
        "--candidate-decode-demand-slack-bytes",
        type=float,
        default=None,
        help="Default: checkpoint args, else 3000. Candidate decode demand-cap slack before leaving PRGs blank.",
    )
    p.add_argument(
        "--candidate-marginal-headroom-slack-bytes",
        type=float,
        default=None,
        help="Default: checkpoint args, else -1. Soft marginal-demand slack before excess PRGs are considered.",
    )
    p.add_argument(
        "--candidate-headroom-rank-mode",
        choices=["stable", "utility", "logit", "phy"],
        default=None,
        help="Default: env CUMAC_HGRAPH_CANDIDATE_HEADROOM_RANK_MODE, checkpoint args, else stable.",
    )
    p.add_argument(
        "--candidate-budgeted-select-mode",
        choices=["off", "greedy", "sample"],
        default=None,
        help="Default: checkpoint args, else off. Budget-aware PRG candidate selection using per-UE remaining headroom.",
    )
    p.add_argument("--candidate-budgeted-hard-cap", type=int, choices=[0, 1], default=None)
    p.add_argument("--candidate-budgeted-usage-penalty", type=float, default=None)
    p.add_argument("--candidate-budgeted-overcap-penalty", type=float, default=None)
    p.add_argument("--candidate-budgeted-first-service-bonus", type=float, default=None)
    p.add_argument("--candidate-budgeted-tail-bias-coef", type=float, default=None)
    p.add_argument("--candidate-budgeted-tail-bias-service-target", type=float, default=None)
    p.add_argument("--candidate-budgeted-tail-bias-ttl-guard-ms", type=float, default=None)
    p.add_argument("--candidate-budgeted-tail-bias-max", type=float, default=None)
    p.add_argument("--candidate-budgeted-rescue-bias-coef", type=float, default=None)
    p.add_argument("--candidate-budgeted-completion-bonus", type=float, default=None)
    p.add_argument("--candidate-budgeted-tail-quota-enable", type=int, choices=[0, 1], default=None)
    p.add_argument("--candidate-seq-state-coef", type=float, default=None)
    p.add_argument("--candidate-seq-fail-risk-feature-offset", type=int, default=None)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42, help="Bridge reset seed")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--print-every-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=1, help="1 writes eval_steps.jsonl; 0 writes summary only")
    p.add_argument(
        "--stats-warmup-steps",
        type=int,
        default=0,
        help="Ignore the first N evaluator steps in stats_mean_* summary fields; the environment still runs them.",
    )
    p.add_argument("--out-dir", default="")
    p.add_argument("--sim-bin", default="", help="optional simulator binary for SHA-256 provenance")
    p.add_argument(
        "--outcome-samples",
        default="",
        help="optional packet outcome CSV produced by the same evaluator run",
    )
    return p.parse_args()


def _merge_w0_v2_equivalence_reports(
    aggregate: dict[str, bool | float | int] | None,
    step: dict[str, bool | float | int],
) -> dict[str, bool | float | int]:
    if aggregate is None:
        return dict(step)
    merged = dict(aggregate)
    for key in (
        "candidate_choice_equal",
        "mask_equal",
        "type0_action_equal",
        "extra_padding_invalid",
        "bitwise_mean_logprob_equal",
        "valid_logits_within_tolerance",
        "chosen_logprob_within_tolerance",
        "passed",
    ):
        merged[key] = bool(aggregate[key]) and bool(step[key])
    for key in ("valid_logits_max_abs_diff", "chosen_logprob_max_abs_diff"):
        merged[key] = max(float(aggregate[key]), float(step[key]))
    for key in (
        "reference_replay_mismatch_count",
        "runtime_replay_mismatch_count",
        "nan_or_inf_count",
    ):
        merged[key] = int(aggregate[key]) + int(step[key])
    for key in ("valid_logits_atol", "chosen_logprob_atol"):
        if float(aggregate[key]) != float(step[key]):
            raise ValueError(f"W0-v2 tolerance changed during evaluation: {key}")
    return merged


def main() -> int:
    args = parse_args()
    _set_seed(int(args.sample_seed))
    device = _pick_device(args.device)

    ckpt_path = Path(args.checkpoint).resolve()
    actor, ckpt = _load_actor_checkpoint(ckpt_path, device)
    independent_mode = isinstance(actor, PFQIndependentReuseActor)
    independent_policy_diagnostics = bool(args.independent_policy_diagnostics)
    service_accounting_audit_enabled = bool(
        args.independent_service_accounting_audit
    )
    phy_tbs_alignment_audit_enabled = bool(
        args.independent_phy_tbs_alignment_audit
    )
    blank_expiry_audit_enabled = bool(
        args.independent_blank_expiry_attribution_audit
    )
    blank_expiry_trace_first_tti = int(args.blank_expiry_trace_first_tti)
    blank_expiry_attribution_protocol = str(
        args.independent_blank_expiry_attribution_protocol
    )
    phy_tbs_alignment_protocol = str(
        args.independent_phy_tbs_alignment_protocol
    )
    independent_headroom_accounting = str(args.independent_headroom_accounting)
    ctbs_enabled = independent_headroom_accounting == "ctbs-v1"
    guard_mode = str(args.independent_non_starvation_guard)
    guard_config = CtbsNonStarvationGuardConfig(
        enabled=guard_mode == "ctbs-blank-v1"
    )
    if guard_config.enabled:
        if not independent_mode:
            raise ValueError("ctbs-blank-v1 requires an independent actor")
        if str(args.decode_mode) != "argmax":
            raise ValueError("ctbs-blank-v1 requires --decode-mode argmax")
        if not ctbs_enabled:
            raise ValueError(
                "ctbs-blank-v1 requires --independent-headroom-accounting ctbs-v1"
            )
        if not independent_policy_diagnostics:
            raise ValueError(
                "ctbs-blank-v1 requires --independent-policy-diagnostics 1"
            )
    _validate_non_starvation_guard_startup(
        guard_mode=guard_mode,
        max_steps=int(args.max_steps),
        episode_horizon=int(args.episode_horizon),
    )
    if independent_policy_diagnostics and not independent_mode:
        raise ValueError("independent policy diagnostics require an independent actor")
    if service_accounting_audit_enabled and not independent_policy_diagnostics:
        raise ValueError(
            "service accounting audit requires independent policy diagnostics"
        )
    if phy_tbs_alignment_audit_enabled and not independent_policy_diagnostics:
        raise ValueError(
            "PHY/TBS alignment audit requires independent policy diagnostics"
        )
    if blank_expiry_audit_enabled and not independent_policy_diagnostics:
        raise ValueError(
            "Blank-expiry attribution audit requires independent policy diagnostics"
        )
    if blank_expiry_audit_enabled and not ctbs_enabled:
        raise ValueError("Blank-expiry attribution audit requires B4-CTBS-v1")
    if blank_expiry_audit_enabled and not args.outcome_samples:
        raise ValueError("Blank-expiry attribution audit requires --outcome-samples")
    if blank_expiry_audit_enabled and blank_expiry_trace_first_tti != 602:
        raise ValueError("frozen Blank-expiry audit requires trace_first_tti=602")
    if ctbs_enabled and not independent_mode:
        raise ValueError("CTBS headroom accounting requires an independent actor")
    if ctbs_enabled and not independent_policy_diagnostics:
        raise ValueError("B4-CTBS-v1 requires independent policy diagnostics")
    if independent_mode:
        independent_decode_mode = {
            "argmax": "greedy",
            "hierarchical_argmax": "hierarchical_greedy",
        }.get(str(args.decode_mode))
        if independent_decode_mode is None:
            raise ValueError(
                "independent evaluation is deterministic and requires --decode-mode "
                "argmax or hierarchical_argmax"
            )
    else:
        if str(args.decode_mode) == "hierarchical_argmax":
            raise ValueError("--decode-mode hierarchical_argmax requires an independent actor")
        independent_decode_mode = None
    candidate_policy_mode = (
        "independent"
        if independent_mode
        else str(_checkpoint_arg(ckpt, "candidate_policy_mode", "legacy")).lower()
    )
    if candidate_policy_mode not in ("legacy", "direct", "independent"):
        raise ValueError("checkpoint candidate_policy_mode must be legacy, direct, or independent")
    ue_prg_topk = int(args.ue_prg_topk if args.ue_prg_topk is not None else _checkpoint_arg(ckpt, "ue_prg_topk", 6))
    ue_prg_diversity_extra = int(
        args.ue_prg_diversity_extra
        if args.ue_prg_diversity_extra is not None
        else _checkpoint_arg(ckpt, "ue_prg_diversity_extra", 0)
    )
    candidate_mode = "all_demand_ues"
    if candidate_policy_mode in ("direct", "independent"):
        candidate_mode = str(
            _checkpoint_arg(ckpt, "candidate_mode", "all_demand_ues")
        ).lower()
        if candidate_mode not in ("all_demand_ues", "topk_diversity"):
            raise ValueError(
                "checkpoint candidate_mode must be all_demand_ues or topk_diversity"
            )
    tb_err_ema_alpha = float(
        args.tb_err_ema_alpha if args.tb_err_ema_alpha is not None else _checkpoint_arg(ckpt, "tb_err_ema_alpha", 0.10)
    )
    slot_repair_max_per_cell = int(
        args.slot_repair_max_per_cell
        if args.slot_repair_max_per_cell is not None
        else _checkpoint_arg(ckpt, "slot_repair_max_per_cell", 0)
    )
    slot_repair_min_backlog_bytes = float(
        args.slot_repair_min_backlog_bytes
        if args.slot_repair_min_backlog_bytes is not None
        else _checkpoint_arg(ckpt, "slot_repair_min_backlog_bytes", 1.0)
    )
    safe_fill_max_per_cell = int(
        args.safe_fill_max_per_cell
        if args.safe_fill_max_per_cell is not None
        else _checkpoint_arg(ckpt, "safe_fill_max_per_cell", 0)
    )
    safe_fill_min_quality_db = float(
        args.safe_fill_min_quality_db
        if args.safe_fill_min_quality_db is not None
        else _checkpoint_arg(ckpt, "safe_fill_min_quality_db", -2.0)
    )
    safe_fill_max_quality_gap_db = float(
        args.safe_fill_max_quality_gap_db
        if args.safe_fill_max_quality_gap_db is not None
        else _checkpoint_arg(ckpt, "safe_fill_max_quality_gap_db", 8.0)
    )
    safe_fill_max_prg_risk = float(
        args.safe_fill_max_prg_risk
        if args.safe_fill_max_prg_risk is not None
        else _checkpoint_arg(ckpt, "safe_fill_max_prg_risk", 0.70)
    )
    safe_fill_min_backlog_bytes = float(
        args.safe_fill_min_backlog_bytes
        if args.safe_fill_min_backlog_bytes is not None
        else _checkpoint_arg(ckpt, "safe_fill_min_backlog_bytes", 1.0)
    )
    safe_fill_max_ue_tb_err = float(
        args.safe_fill_max_ue_tb_err
        if args.safe_fill_max_ue_tb_err is not None
        else _checkpoint_arg(ckpt, "safe_fill_max_ue_tb_err", 0.45)
    )
    candidate_blank_budget_max_per_cell = int(
        args.candidate_blank_budget_max_per_cell
        if args.candidate_blank_budget_max_per_cell is not None
        else _checkpoint_arg(ckpt, "candidate_blank_budget_max_per_cell", -1)
    )
    candidate_blank_budget_min_decision_logit = float(
        args.candidate_blank_budget_min_decision_logit
        if args.candidate_blank_budget_min_decision_logit is not None
        else _checkpoint_arg(ckpt, "candidate_blank_budget_min_decision_logit", -1.0e9)
    )
    candidate_decode_demand_slack_bytes = float(
        args.candidate_decode_demand_slack_bytes
        if args.candidate_decode_demand_slack_bytes is not None
        else _checkpoint_arg(ckpt, "candidate_decode_demand_slack_bytes", 3000.0)
    )
    candidate_marginal_headroom_slack_bytes = float(
        args.candidate_marginal_headroom_slack_bytes
        if args.candidate_marginal_headroom_slack_bytes is not None
        else _checkpoint_arg(ckpt, "candidate_marginal_headroom_slack_bytes", -1.0)
    )
    candidate_headroom_rank_mode = str(
        args.candidate_headroom_rank_mode
        if args.candidate_headroom_rank_mode is not None
        else os.environ.get(
            "CUMAC_HGRAPH_CANDIDATE_HEADROOM_RANK_MODE",
            _checkpoint_arg(ckpt, "candidate_headroom_rank_mode", "stable"),
        )
    ).lower()
    candidate_budgeted_select_mode = str(
        args.candidate_budgeted_select_mode
        if args.candidate_budgeted_select_mode is not None
        else _checkpoint_arg(ckpt, "candidate_budgeted_select_mode", "off")
    ).lower()
    candidate_budgeted_hard_cap = int(
        args.candidate_budgeted_hard_cap
        if args.candidate_budgeted_hard_cap is not None
        else _checkpoint_arg(ckpt, "candidate_budgeted_hard_cap", 1)
    )
    candidate_budgeted_usage_penalty = float(
        args.candidate_budgeted_usage_penalty
        if args.candidate_budgeted_usage_penalty is not None
        else _checkpoint_arg(ckpt, "candidate_budgeted_usage_penalty", 0.35)
    )
    candidate_budgeted_overcap_penalty = float(
        args.candidate_budgeted_overcap_penalty
        if args.candidate_budgeted_overcap_penalty is not None
        else _checkpoint_arg(ckpt, "candidate_budgeted_overcap_penalty", 8.0)
    )
    candidate_budgeted_first_service_bonus = float(
        args.candidate_budgeted_first_service_bonus
        if args.candidate_budgeted_first_service_bonus is not None
        else _checkpoint_arg(ckpt, "candidate_budgeted_first_service_bonus", 0.0)
    )
    candidate_budgeted_tail_bias_coef = float(
        args.candidate_budgeted_tail_bias_coef
        if args.candidate_budgeted_tail_bias_coef is not None
        else _checkpoint_arg(ckpt, "candidate_budgeted_tail_bias_coef", 0.0)
    )
    candidate_budgeted_tail_bias_service_target = float(
        args.candidate_budgeted_tail_bias_service_target
        if args.candidate_budgeted_tail_bias_service_target is not None
        else _checkpoint_arg(ckpt, "candidate_budgeted_tail_bias_service_target", 0.75)
    )
    candidate_budgeted_tail_bias_ttl_guard_ms = float(
        args.candidate_budgeted_tail_bias_ttl_guard_ms
        if args.candidate_budgeted_tail_bias_ttl_guard_ms is not None
        else _checkpoint_arg(ckpt, "candidate_budgeted_tail_bias_ttl_guard_ms", 20.0)
    )
    candidate_budgeted_tail_bias_max = float(
        args.candidate_budgeted_tail_bias_max
        if args.candidate_budgeted_tail_bias_max is not None
        else _checkpoint_arg(ckpt, "candidate_budgeted_tail_bias_max", 1.5)
    )
    candidate_budgeted_rescue_bias_coef = float(
        args.candidate_budgeted_rescue_bias_coef
        if args.candidate_budgeted_rescue_bias_coef is not None
        else _checkpoint_arg(ckpt, "candidate_budgeted_rescue_bias_coef", 0.0)
    )
    candidate_budgeted_completion_bonus = float(
        args.candidate_budgeted_completion_bonus
        if args.candidate_budgeted_completion_bonus is not None
        else _checkpoint_arg(ckpt, "candidate_budgeted_completion_bonus", 0.0)
    )
    candidate_budgeted_tail_quota_enable = int(
        args.candidate_budgeted_tail_quota_enable
        if args.candidate_budgeted_tail_quota_enable is not None
        else _checkpoint_arg(ckpt, "candidate_budgeted_tail_quota_enable", 0)
    )
    candidate_seq_state_coef = float(
        args.candidate_seq_state_coef
        if args.candidate_seq_state_coef is not None
        else _checkpoint_arg(ckpt, "candidate_seq_state_coef", 1.0)
    )
    candidate_seq_fail_risk_feature_offset = int(
        args.candidate_seq_fail_risk_feature_offset
        if args.candidate_seq_fail_risk_feature_offset is not None
        else _checkpoint_arg(ckpt, "candidate_seq_fail_risk_feature_offset", -1)
    )
    if ue_prg_topk <= 0:
        raise ValueError("--ue-prg-topk must be > 0")
    if ue_prg_diversity_extra < 0:
        raise ValueError("--ue-prg-diversity-extra must be >= 0")
    if slot_repair_max_per_cell < 0:
        raise ValueError("--slot-repair-max-per-cell must be >= 0")
    if safe_fill_max_per_cell < 0:
        raise ValueError("--safe-fill-max-per-cell must be >= 0")
    if candidate_blank_budget_max_per_cell < -1:
        raise ValueError("--candidate-blank-budget-max-per-cell must be >= -1")
    if candidate_decode_demand_slack_bytes < 0.0:
        raise ValueError("--candidate-decode-demand-slack-bytes must be >= 0")
    if candidate_marginal_headroom_slack_bytes < -1.0:
        raise ValueError("--candidate-marginal-headroom-slack-bytes must be >= -1")
    if candidate_headroom_rank_mode not in ("stable", "utility", "logit", "phy"):
        raise ValueError("--candidate-headroom-rank-mode must be stable, utility, logit, or phy")
    if candidate_budgeted_select_mode not in ("off", "greedy", "sample"):
        raise ValueError("--candidate-budgeted-select-mode must be off, greedy, or sample")
    if candidate_budgeted_hard_cap not in (0, 1):
        raise ValueError("--candidate-budgeted-hard-cap must be 0 or 1")
    if candidate_budgeted_usage_penalty < 0.0:
        raise ValueError("--candidate-budgeted-usage-penalty must be >= 0")
    if candidate_budgeted_overcap_penalty < 0.0:
        raise ValueError("--candidate-budgeted-overcap-penalty must be >= 0")
    if candidate_budgeted_tail_bias_coef < 0.0:
        raise ValueError("--candidate-budgeted-tail-bias-coef must be >= 0")
    if candidate_budgeted_tail_bias_service_target <= 0.0:
        raise ValueError("--candidate-budgeted-tail-bias-service-target must be > 0")
    if candidate_budgeted_tail_bias_ttl_guard_ms <= 0.0:
        raise ValueError("--candidate-budgeted-tail-bias-ttl-guard-ms must be > 0")
    if candidate_budgeted_tail_bias_max < 0.0:
        raise ValueError("--candidate-budgeted-tail-bias-max must be >= 0")
    if candidate_budgeted_rescue_bias_coef < 0.0:
        raise ValueError("--candidate-budgeted-rescue-bias-coef must be >= 0")
    if candidate_budgeted_completion_bonus < 0.0:
        raise ValueError("--candidate-budgeted-completion-bonus must be >= 0")
    if candidate_budgeted_tail_quota_enable not in (0, 1):
        raise ValueError("--candidate-budgeted-tail-quota-enable must be 0 or 1")
    _validate_direct_evaluation_config(
        argparse.Namespace(
            candidate_policy_mode=("legacy" if independent_mode else candidate_policy_mode),
            candidate_budgeted_select_mode=candidate_budgeted_select_mode,
            candidate_budgeted_hard_cap=candidate_budgeted_hard_cap,
            candidate_blank_budget_max_per_cell=candidate_blank_budget_max_per_cell,
            slot_repair_max_per_cell=slot_repair_max_per_cell,
            safe_fill_max_per_cell=safe_fill_max_per_cell,
            candidate_budgeted_tail_bias_coef=candidate_budgeted_tail_bias_coef,
            candidate_budgeted_rescue_bias_coef=candidate_budgeted_rescue_bias_coef,
            candidate_budgeted_completion_bonus=candidate_budgeted_completion_bonus,
            candidate_budgeted_tail_quota_enable=candidate_budgeted_tail_quota_enable,
        )
    )
    if int(args.save_steps) not in (0, 1):
        raise ValueError("--save-steps must be 0 or 1")
    if int(args.stats_warmup_steps) < 0:
        raise ValueError("--stats-warmup-steps must be >= 0")

    out_dir = None
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    if guard_config.enabled and out_dir is None:
        raise ValueError("ctbs-blank-v1 requires --out-dir for the fixed trace CSV")
    if blank_expiry_audit_enabled and out_dir is None:
        raise ValueError("Blank-expiry attribution audit requires --out-dir")

    snapshot_cfg = SnapshotBuilderConfig(tb_err_ema_alpha=float(tb_err_ema_alpha))
    snapshot_builder = FeatureSnapshotBuilder(snapshot_cfg)
    up_edge_attr_dim = int(getattr(actor.cfg, "up_edge_attr_dim", 6))
    ckpt_anchor = _checkpoint_arg(ckpt, "include_pfq_score_anchor", None)
    if ckpt_anchor is not None:
        inferred_pfq_anchor = bool(int(ckpt_anchor))
    else:
        inferred_pfq_anchor = bool(
            int(
                getattr(
                    actor.cfg,
                    "has_pfq_score_anchor",
                    int(up_edge_attr_dim in (6, 9, 11, 14, 20, 30)),
                )
            )
        )
    inferred_risk_feature_dim = max(0, up_edge_attr_dim - 5 - int(inferred_pfq_anchor))
    inferred_candidate_risk_features = inferred_risk_feature_dim > 0
    include_candidate_risk_features = bool(
        int(_checkpoint_arg(ckpt, "include_candidate_risk_features", int(inferred_candidate_risk_features)))
    )
    candidate_risk_feature_dim = inferred_risk_feature_dim if include_candidate_risk_features else 0
    include_pfq_score_anchor = bool(
        int(_checkpoint_arg(ckpt, "include_pfq_score_anchor", int(inferred_pfq_anchor)))
    )
    edge_generator = SparseEdgeGenerator(
        EdgeGeneratorConfig(
            ue_prg_topk=int(ue_prg_topk),
            ue_prg_diversity_extra=int(ue_prg_diversity_extra),
            candidate_mode=str(candidate_mode),
            include_pfq_score_anchor=include_pfq_score_anchor,
            include_candidate_risk_features=include_candidate_risk_features,
            candidate_risk_feature_dim=candidate_risk_feature_dim,
        )
    )
    env = BridgeEnvClient(
        BridgeEnvConfig(
            socket_path=str(args.socket_path),
            connect_timeout_s=float(args.connect_timeout_s),
            enable_ctbs=ctbs_enabled,
        )
    )

    obs, _reward0, info = env.reset(seed=int(args.seed), episode_horizon=int(args.episode_horizon))
    runtime_harq_enabled = int(info.get("harq_enabled", -1))
    ctbs_config: PhyTbsConfig | None = None
    ctbs_config_sha256: str | None = None
    if ctbs_enabled:
        if runtime_harq_enabled != 0:
            raise ValueError("B4-CTBS-v1 requires runtime HARQ to be disabled")
        if not bool(info.get("ctbs_config_available", False)):
            raise ValueError("B4-CTBS-v1 reset did not provide a TBS configuration")
        ctbs_config = PhyTbsConfig.from_vector(info["ctbs_config"])
        ctbs_config_sha256 = str(info["ctbs_config_sha256"])
    if service_accounting_audit_enabled and runtime_harq_enabled != 0:
        raise ValueError(
            "service accounting audit requires runtime HARQ to be explicitly disabled"
        )
    if phy_tbs_alignment_audit_enabled and runtime_harq_enabled != 0:
        raise ValueError(
            "PHY/TBS alignment audit requires runtime HARQ to be explicitly disabled"
        )
    initial_graph = edge_generator.build_graph(snapshot_builder.build(obs))
    try:
        _validate_non_starvation_guard_topology(
            guard_mode=guard_mode,
            n_cell=int(initial_graph.snapshot.static_meta.n_cell),
            n_prg=int(initial_graph.snapshot.static_meta.n_prg),
        )
    except Exception:
        env.close()
        raise
    serving_cell_by_ue: tuple[int, ...] = ()
    if blank_expiry_audit_enabled:
        initial_association = initial_graph.snapshot.static_meta.action_mask_cell_ue.to(
            dtype=torch.bool
        ).detach().cpu()
        serving_cell_by_ue = tuple(
            int(torch.nonzero(initial_association[:, ue], as_tuple=False).flatten().item())
            for ue in range(initial_graph.snapshot.static_meta.n_ue)
        )
    checkpoint_actor_config = ckpt.get("actor_config", {})
    if not isinstance(checkpoint_actor_config, dict):
        checkpoint_actor_config = {}
    runtime_contract = None
    association_candidate_counts: list[int] | None = None
    actor_state_hash = ""
    equivalence_reference_width = int(args.independent_equivalence_reference_width)
    equivalence_protocol = str(args.independent_equivalence_protocol)
    if independent_mode:
        runtime_contract = resolve_runtime_candidate_contract(
            initial_graph,
            checkpoint_actor_config,
            requested_width=int(args.independent_runtime_candidate_width),
        )
        association_candidate_counts = [
            int(value)
            for value in initial_graph.snapshot.static_meta.action_mask_cell_ue.to(
                dtype=torch.int64
            ).sum(dim=1).tolist()
        ]
        checkpoint_state = ckpt.get("actor_state")
        if checkpoint_state is None:
            checkpoint_state = ckpt.get("model_state", ckpt)
        actor_state_hash = state_dict_sha256(checkpoint_state)
        if (
            blank_expiry_audit_enabled
            and actor_state_hash != FROZEN_B4_STATE_DICT_SHA256
        ):
            raise ValueError(
                "Blank-expiry audit frozen B4 state-dict hash mismatch: "
                f"got={actor_state_hash} expected={FROZEN_B4_STATE_DICT_SHA256}"
            )
        if equivalence_reference_width < 0:
            raise ValueError("independent equivalence reference width must be non-negative")
        if equivalence_reference_width:
            if equivalence_reference_width >= runtime_contract.runtime_candidate_width:
                raise ValueError(
                    "equivalence reference width must be smaller than runtime candidate width"
                )
            resolve_runtime_candidate_contract(
                initial_graph,
                checkpoint_actor_config,
                requested_width=equivalence_reference_width,
            )
        elif equivalence_protocol == "fp32-w0-v2":
            raise ValueError("fp32-w0-v2 requires an equivalence reference width")
    elif (
        int(args.independent_runtime_candidate_width)
        or equivalence_reference_width
        or equivalence_protocol != "bitwise-w0-v1"
    ):
        raise ValueError("independent candidate width flags require an independent actor")
    deletion_test_device = ""
    if independent_mode:
        # CUDA sparse reductions may differ by a few ULPs between identical
        # forward passes.  The independence contract is a bitwise test, so run
        # this one-time audit on the deterministic CPU path, then restore the
        # requested evaluation device.
        deletion_device = torch.device("cpu")
        actor.to(deletion_device)
        deletion_equal = _independent_deletion_equal(
            actor,
            obs,
            edge_generator,
            snapshot_cfg,
            deletion_device,
            mode=str(independent_decode_mode),
            runtime_candidate_width=runtime_contract.runtime_candidate_width,
        )
        actor.to(device)
        deletion_test_device = str(deletion_device)
    else:
        deletion_equal = None
    step_idx = 0
    measured_tti_first: int | None = None
    measured_tti_last: int | None = None
    n_active_ue = int(info.get("n_active_ue", 0))
    ue_measured_goodput_bytes = np.zeros((n_active_ue,), dtype=np.float64)
    ue_measured_outcome_time_ms = np.zeros((n_active_ue,), dtype=np.float64)
    measured_tb_tx_count = 0.0
    measured_tb_err_count = 0.0
    eval_diagnostics: list[IndependentPolicyDiagnostics] | None = (
        [] if independent_policy_diagnostics else None
    )
    rescue_decision_rows: list[RescueDecisionDiagnostics] | None = (
        [] if independent_policy_diagnostics else None
    )
    rescue_ue_rows: list[RescueUeTtiRow] | None = (
        [] if independent_policy_diagnostics else None
    )
    rescue_cell_rows: list[RescueCellTtiRow] | None = (
        [] if independent_policy_diagnostics else None
    )
    service_accounting_rows: list[ServiceAccountingUeTtiRow] | None = (
        [] if service_accounting_audit_enabled else None
    )
    phy_tbs_alignment_rows: list[PhyTbsAuditUeTtiRow] | None = (
        [] if phy_tbs_alignment_audit_enabled else None
    )
    blank_expiry_traces: list[BlankExpiryTtiTrace] | None = (
        [] if blank_expiry_audit_enabled else None
    )
    guard_full_rows: list[dict[str, object]] = []
    guard_measured_rows: list[dict[str, object]] = []
    stats_sum_scheduled_bytes: np.float64 | None = (
        np.float64(0.0) if independent_policy_diagnostics else None
    )
    stats_sum_ue_goodput_bytes: np.float64 | None = (
        np.float64(0.0) if independent_policy_diagnostics else None
    )
    stats_sum_failed_bytes: np.float64 | None = (
        np.float64(0.0) if independent_policy_diagnostics else None
    )
    last_delivered_time_ms = np.asarray(
        info.get("actual_ue_packet_system_time_ms", np.zeros((n_active_ue,))),
        dtype=np.float64,
    ).reshape(-1)
    last_expired_time_ms = np.asarray(
        info.get("actual_ue_packet_expired_system_time_ms", np.zeros((n_active_ue,))),
        dtype=np.float64,
    ).reshape(-1)
    rows = []
    tail_rows = deque(maxlen=max(1, int(args.print_every_steps) if int(args.print_every_steps) > 0 else 1))
    sums = {
        "goodput_mbps": 0.0,
        "tb_err_rate": 0.0,
        "expiry_drop_rate": 0.0,
        "expired_bytes": 0.0,
        "prg_utilization_ratio": 0.0,
        "prg_reuse_ratio": 0.0,
        "packet_completed_count": 0.0,
        "packet_delivered_bits": 0.0,
        "packet_system_time_ms": 0.0,
        "packet_delay_mean_ms": 0.0,
        "packet_effective_service_rate_mbps": 0.0,
        "packet_effective_service_rate_per_packet_mean_mbps": 0.0,
        "blank_ratio": 0.0,
        "fill_ratio": 0.0,
        "cap_drop_count": 0.0,
        "fallback_success_count": 0.0,
        "safe_fill_count": 0.0,
        "missed_safe_opportunity_count": 0.0,
        "blank_budget_repair_count": 0.0,
        "final_blank_count": 0.0,
        "post_decode_policy_blank_count": 0.0,
        "post_decode_no_valid_candidate_count": 0.0,
        "post_decode_demand_cap_count": 0.0,
        "post_decode_no_slot_count": 0.0,
        "post_decode_other_drop_count": 0.0,
        "uniq_ue": 0.0,
        "slot_repair_count": 0.0,
        "coverage_repair_count": 0.0,
        "non_hard_override_count": 0.0,
        "candidate_policy_mode_direct": 0.0,
        "budgeted_changed_count": 0.0,
        "budgeted_no_headroom_count": 0.0,
        "budgeted_active_ue_count": 0.0,
        "budgeted_tail_bias_candidate_mean": 0.0,
        "budgeted_tail_bias_selected_mean": 0.0,
        "budgeted_tail_bias_active_count": 0.0,
        "action_replay_mismatch_count": 0.0,
        "candidate_truncation_count": 0.0,
        "illegal_action_count": 0.0,
        "native_reject_count": 0.0,
        "native_reject_diagnostics_available": 0.0,
        "packet_expired_count": 0.0,
        "packet_expired_system_time_ms": 0.0,
        "packet_outcome_count": 0.0,
        "packet_outcome_system_time_ms": 0.0,
        "packet_expiry_aware_rate_mbps": 0.0,
        "packet_expiry_aware_rate_per_packet_mean_mbps": 0.0,
    }
    stats_sums = {key: 0.0 for key in sums}
    stats_count = 0
    equivalence_report: dict[str, bool | float | int] | None = None
    equivalence_steps = 0
    ctbs_snapshot_count = 0
    ctbs_snapshot_id_first: int | None = None
    ctbs_snapshot_id_last: int | None = None
    ctbs_mcs_mismatch_count = 0
    ctbs_layer_mismatch_count = 0
    ctbs_tbs_mismatch_count = 0
    ctbs_reference_label_deletion_equal = True
    ctbs_raw_selected_service_bytes_sum = 0.0
    ctbs_capacity_decrease_count = 0
    ctbs_remaining_rebound_count = 0
    ctbs_mask_reactivation_count = 0
    ctbs_n_prg = (
        int(initial_graph.snapshot.static_meta.n_prg) if ctbs_enabled else 0
    )
    ctbs_mask_trigger_count_by_local_prg = [0 for _ in range(ctbs_n_prg)]
    ctbs_selected_count_by_local_prg = [0 for _ in range(ctbs_n_prg)]
    ctbs_valid_candidate_count_by_local_prg = [0 for _ in range(ctbs_n_prg)]
    try:
        while True:
            graph = _graph_to_device(edge_generator.build_graph(snapshot_builder.build(obs)), device)
            action_meta = graph.snapshot.static_meta
            _validate_non_starvation_guard_topology(
                guard_mode=guard_mode,
                n_cell=int(action_meta.n_cell),
                n_prg=int(action_meta.n_prg),
            )
            ctbs_inputs: CtbsHeadroomInputs | None = None
            if ctbs_enabled:
                if ctbs_config is None or ctbs_config_sha256 is None:
                    raise RuntimeError("B4-CTBS-v1 reset configuration disappeared")
                pre_action = env.request_pre_action_snapshot(
                    expected_tti=int(info["tti"]),
                )
                ctbs_inputs = CtbsHeadroomInputs.from_arrays(
                    tti=pre_action.tti,
                    snapshot_id=pre_action.snapshot_id,
                    config_sha256=pre_action.config_sha256,
                    config=ctbs_config,
                    ue_ids=pre_action.ue_ids,
                    serving_cell=pre_action.serving_cell,
                    wb_sinr_first_layer_db=pre_action.wb_sinr_first_layer_db,
                    olla_delta=pre_action.olla_delta,
                    olla_delta_up=pre_action.olla_delta_up,
                    olla_delta_down=pre_action.olla_delta_down,
                    prior_tb_err=pre_action.prior_tb_err,
                    prg_layer_cap=pre_action.prg_layer_cap,
                )
            if independent_mode and equivalence_reference_width:
                if equivalence_protocol == "fp32-w0-v2":
                    step_equivalence = compare_candidate_padding_fp32_equivalence(
                        actor,
                        graph,
                        reference_width=equivalence_reference_width,
                        padded_width=runtime_contract.runtime_candidate_width,
                        mode=str(independent_decode_mode),
                        valid_logits_atol=W0_V2_VALID_LOGITS_ATOL,
                        chosen_logprob_atol=W0_V2_CHOSEN_LOGPROB_ATOL,
                    )
                    equivalence_report = _merge_w0_v2_equivalence_reports(
                        equivalence_report,
                        step_equivalence,
                    )
                    step_passed = bool(step_equivalence["passed"])
                else:
                    step_equivalence = compare_candidate_padding_semantics(
                        actor,
                        graph,
                        reference_width=equivalence_reference_width,
                        padded_width=runtime_contract.runtime_candidate_width,
                        mode=str(independent_decode_mode),
                    )
                    equivalence_report = (
                        dict(step_equivalence)
                        if equivalence_report is None
                        else {
                            key: bool(equivalence_report[key] and step_equivalence[key])
                            for key in equivalence_report
                        }
                    )
                    step_passed = all(bool(value) for value in step_equivalence.values())
                equivalence_steps += 1
                if not step_passed:
                    raise RuntimeError(
                        f"candidate padding failed {equivalence_protocol}: "
                        f"{step_equivalence}"
                    )
            selected = _select_joint_action(
                actor,
                graph,
                decode_mode=str(args.decode_mode),
                candidate_policy_mode=str(candidate_policy_mode),
                slot_repair_max_per_cell=int(slot_repair_max_per_cell),
                slot_repair_min_backlog_bytes=float(slot_repair_min_backlog_bytes),
                safe_fill_max_per_cell=int(safe_fill_max_per_cell),
                safe_fill_min_quality_db=float(safe_fill_min_quality_db),
                safe_fill_max_quality_gap_db=float(safe_fill_max_quality_gap_db),
                safe_fill_max_prg_risk=float(safe_fill_max_prg_risk),
                safe_fill_min_backlog_bytes=float(safe_fill_min_backlog_bytes),
                safe_fill_max_ue_tb_err=float(safe_fill_max_ue_tb_err),
                candidate_blank_budget_max_per_cell=int(candidate_blank_budget_max_per_cell),
                candidate_blank_budget_min_decision_logit=float(candidate_blank_budget_min_decision_logit),
                candidate_decode_demand_slack_bytes=float(candidate_decode_demand_slack_bytes),
                candidate_marginal_headroom_slack_bytes=float(candidate_marginal_headroom_slack_bytes),
                candidate_headroom_rank_mode=str(candidate_headroom_rank_mode),
                candidate_budgeted_select_mode=str(candidate_budgeted_select_mode),
                candidate_budgeted_hard_cap=int(candidate_budgeted_hard_cap),
                candidate_budgeted_usage_penalty=float(candidate_budgeted_usage_penalty),
                candidate_budgeted_overcap_penalty=float(candidate_budgeted_overcap_penalty),
                candidate_budgeted_first_service_bonus=float(candidate_budgeted_first_service_bonus),
                candidate_budgeted_tail_bias_coef=float(candidate_budgeted_tail_bias_coef),
                candidate_budgeted_tail_bias_service_target=float(candidate_budgeted_tail_bias_service_target),
                candidate_budgeted_tail_bias_ttl_guard_ms=float(candidate_budgeted_tail_bias_ttl_guard_ms),
                candidate_budgeted_tail_bias_max=float(candidate_budgeted_tail_bias_max),
                candidate_budgeted_rescue_bias_coef=float(candidate_budgeted_rescue_bias_coef),
                candidate_budgeted_completion_bonus=float(candidate_budgeted_completion_bonus),
                candidate_budgeted_tail_quota_enable=int(candidate_budgeted_tail_quota_enable),
                candidate_seq_state_coef=float(candidate_seq_state_coef),
                candidate_seq_fail_risk_feature_offset=int(candidate_seq_fail_risk_feature_offset),
                independent_policy_diagnostics=independent_policy_diagnostics,
                collect_blank_expiry_attribution=(
                    blank_expiry_audit_enabled
                    and (step_idx + 1) >= blank_expiry_trace_first_tti
                ),
                independent_runtime_candidate_width=(
                    runtime_contract.runtime_candidate_width
                    if runtime_contract is not None
                    else None
                ),
                ctbs_headroom_inputs=ctbs_inputs,
                independent_non_starvation_guard=guard_mode,
            )
            if guard_config.enabled:
                meta = graph.snapshot.static_meta
                action_tti = step_idx + 1
                step_guard_rows = _guard_diagnostics_to_validated_rows(
                    tti=action_tti,
                    trace=selected.get("independent_policy_trace"),
                    n_cell=int(meta.n_cell),
                    n_prg=int(meta.n_prg),
                )
                guard_full_rows.extend(step_guard_rows)
                if 1001 <= action_tti <= 5000:
                    guard_measured_rows.extend(step_guard_rows)
            next_obs, raw_reward, done, next_info = env.step(
                selected["action_ue_select"],
                selected["action_prg_alloc"],
            )
            if ctbs_inputs is not None:
                required_reference_labels = (
                    "actual_ue_mcs",
                    "actual_ue_layer",
                    "actual_ue_tbs_capacity_bytes",
                )
                missing_labels = [
                    name for name in required_reference_labels if name not in next_info
                ]
                if missing_labels:
                    raise RuntimeError(
                        "B4-CTBS-v1 post-action reference labels are missing: "
                        + ", ".join(missing_labels)
                    )
                parity = compare_ctbs_reference_labels(
                    ctbs_inputs,
                    chosen_ue_per_prg=selected["decoded_chosen_ue_per_prg"],
                    actual_mcs=next_info["actual_ue_mcs"],
                    actual_layer=next_info["actual_ue_layer"],
                    actual_tbs_capacity_bytes=next_info[
                        "actual_ue_tbs_capacity_bytes"
                    ],
                    n_cell=int(graph.snapshot.static_meta.n_cell),
                    n_prg=int(graph.snapshot.static_meta.n_prg),
                )
                ctbs_snapshot_count += 1
                ctbs_snapshot_id_first = (
                    ctbs_inputs.snapshot_id
                    if ctbs_snapshot_id_first is None
                    else ctbs_snapshot_id_first
                )
                ctbs_snapshot_id_last = ctbs_inputs.snapshot_id
                ctbs_mcs_mismatch_count += parity.mcs_mismatch_count
                ctbs_layer_mismatch_count += parity.layer_mismatch_count
                ctbs_tbs_mismatch_count += parity.tbs_mismatch_count
                ctbs_reference_label_deletion_equal = bool(
                    ctbs_reference_label_deletion_equal
                    and parity.reference_label_deletion_equal
                )
                if not parity.all_exact:
                    raise RuntimeError(
                        "B4-CTBS-v1 pre-action reconstruction mismatch: "
                        f"tti={ctbs_inputs.tti} snapshot_id={ctbs_inputs.snapshot_id} "
                        f"mcs={parity.mcs_mismatch_count} "
                        f"layer={parity.layer_mismatch_count} "
                        f"tbs={parity.tbs_mismatch_count}"
                    )
                if step_idx >= int(args.stats_warmup_steps):
                    trace = selected["independent_policy_trace"]
                    ctbs_diag = trace.ctbs_diagnostics
                    if ctbs_diag is None:
                        raise RuntimeError("B4-CTBS-v1 trace is missing CTBS diagnostics")
                    ctbs_mask_reactivation_count += _count_candidate_mask_reactivations(
                        trace.valid_at_step[:, :, :-1],
                        selected["candidate_ue_ids_at_step"],
                        n_ue=int(graph.snapshot.static_meta.n_ue),
                    )
                    selected_mask = ctbs_diag.selected_nonblank.to(dtype=torch.bool)
                    ctbs_raw_selected_service_bytes_sum += float(
                        ctbs_diag.raw_service_bytes[selected_mask].sum().item()
                    )
                    ctbs_capacity_decrease_count += int(
                        (
                            selected_mask
                            & (
                                ctbs_diag.cumulative_tbs_after_bytes
                                < ctbs_diag.cumulative_tbs_before_bytes
                            )
                        ).sum().item()
                    )
                    ctbs_remaining_rebound_count += int(
                        (
                            selected_mask
                            & (
                                ctbs_diag.remaining_after_bytes
                                > ctbs_diag.remaining_before_bytes
                            )
                        ).sum().item()
                    )
                    exhausted = (
                        selected_mask
                        & (ctbs_diag.remaining_before_bytes > 0.0)
                        & (ctbs_diag.remaining_after_bytes <= 0.0)
                    )
                    for local_prg in range(len(ctbs_mask_trigger_count_by_local_prg)):
                        local = ctbs_diag.local_prg_index == local_prg
                        ctbs_mask_trigger_count_by_local_prg[local_prg] += int(
                            (exhausted & local).sum().item()
                        )
                        ctbs_selected_count_by_local_prg[local_prg] += int(
                            (selected_mask & local).sum().item()
                        )
                        ctbs_valid_candidate_count_by_local_prg[local_prg] += int(
                            trace.valid_at_step[local_prg, :, :-1].sum().item()
                        )
            if blank_expiry_audit_enabled and (step_idx + 1) >= blank_expiry_trace_first_tti:
                trace_tti = int(next_info.get("tti", step_idx + 1))
                if trace_tti != step_idx + 1:
                    raise RuntimeError(
                        "Blank-expiry audit action/response TTI mismatch: "
                        f"step={step_idx + 1} response={trace_tti}"
                    )
                policy_trace = selected.get("independent_policy_trace")
                decision_diagnostics = getattr(
                    policy_trace, "blank_expiry_diagnostics", None
                )
                if decision_diagnostics is None or blank_expiry_traces is None:
                    raise RuntimeError(
                        "Blank-expiry audit decision diagnostics are missing"
                    )

                def audit_tuple(name: str, dtype, *, integer: bool = False):
                    values = np.asarray(next_info.get(name), dtype=dtype).reshape(-1)
                    if values.shape != (n_active_ue,):
                        raise RuntimeError(
                            f"Blank-expiry audit {name} shape mismatch: {values.shape}"
                        )
                    if not np.isfinite(values).all() or (values < 0).any():
                        raise RuntimeError(
                            f"Blank-expiry audit {name} values are invalid"
                        )
                    if integer and not np.equal(values, np.floor(values)).all():
                        raise RuntimeError(
                            f"Blank-expiry audit {name} must be integral"
                        )
                    return tuple(int(value) if integer else float(value) for value in values)

                blank_expiry_traces.append(
                    BlankExpiryTtiTrace(
                        tti=trace_tti,
                        decisions=decision_diagnostics,
                        actual_ue_served_bytes=audit_tuple(
                            "actual_ue_served_bytes", np.float64
                        ),
                        actual_ue_goodput_bytes=audit_tuple(
                            "actual_ue_goodput_bytes", np.float64
                        ),
                        actual_ue_tb_tx_count=audit_tuple(
                            "actual_ue_tb_tx_count", np.float64, integer=True
                        ),
                        actual_ue_tb_err_count=audit_tuple(
                            "actual_ue_tb_err_count", np.float64, integer=True
                        ),
                    )
                )
            reward_metrics = extract_reward_metrics(next_info)
            delivered_time_ms = np.asarray(
                next_info.get("actual_ue_packet_system_time_ms", last_delivered_time_ms),
                dtype=np.float64,
            ).reshape(-1)
            expired_time_ms = np.asarray(
                next_info.get(
                    "actual_ue_packet_expired_system_time_ms", last_expired_time_ms
                ),
                dtype=np.float64,
            ).reshape(-1)
            native_reject_available = bool(
                next_info.get("native_reject_diagnostics_available", False)
            )
            if independent_mode and not native_reject_available:
                raise RuntimeError(
                    "independent evaluation requires native reject diagnostics"
                )
            native_reject_count = 0.0
            if native_reject_available:
                for key in (
                    "native_reject_wrong_cell_slot_count",
                    "native_reject_empty_slot_count",
                    "native_reject_prg_mask_count",
                    "native_reject_oob_slot_count",
                    "native_reject_duplicate_ue_count",
                ):
                    native_reject_count += float(
                        np.asarray(next_info[key], dtype=np.float64).sum()
                    )
            row = {
                "step": step_idx,
                "tti": int(next_info.get("tti", step_idx)),
                "reward_raw_env": float(raw_reward),
                "goodput_mbps": float(reward_metrics.get("goodput_mbps", 0.0)),
                "tb_err_rate": float(reward_metrics.get("tb_err_rate", 0.0)),
                "expiry_drop_rate": float(reward_metrics.get("expiry_drop_rate", 0.0)),
                "expired_bytes": float(reward_metrics.get("expired_bytes", 0.0)),
                "prg_utilization_ratio": float(reward_metrics.get("prg_utilization_ratio", 0.0)),
                "prg_reuse_ratio": float(reward_metrics.get("prg_reuse_ratio", 0.0)),
                "packet_completed_count": float(reward_metrics.get("packet_completed_count", 0.0)),
                "packet_delivered_bits": float(reward_metrics.get("packet_delivered_bits", 0.0)),
                "packet_system_time_ms": float(reward_metrics.get("packet_system_time_ms", 0.0)),
                "packet_delay_mean_ms": float(reward_metrics.get("packet_delay_mean_ms", 0.0)),
                "packet_effective_service_rate_mbps": float(
                    reward_metrics.get("packet_effective_service_rate_mbps", 0.0)
                ),
                "packet_effective_service_rate_per_packet_mean_mbps": float(
                    reward_metrics.get("packet_effective_service_rate_per_packet_mean_mbps", 0.0)
                ),
                "packet_expired_count": float(
                    reward_metrics.get("packet_expired_count", 0.0)
                ),
                "packet_expired_system_time_ms": float(
                    reward_metrics.get("packet_expired_system_time_ms", 0.0)
                ),
                "packet_outcome_count": float(
                    reward_metrics.get("packet_outcome_count", 0.0)
                ),
                "packet_outcome_system_time_ms": float(
                    reward_metrics.get("packet_outcome_system_time_ms", 0.0)
                ),
                "packet_expiry_aware_rate_mbps": float(
                    reward_metrics.get("packet_expiry_aware_rate_mbps", 0.0)
                ),
                "packet_expiry_aware_rate_per_packet_mean_mbps": float(
                    reward_metrics.get(
                        "packet_expiry_aware_rate_per_packet_mean_mbps", 0.0
                    )
                ),
                "blank_ratio": float(selected["blank_ratio"]),
                "fill_ratio": float(selected["fill_ratio"]),
                "cap_drop_count": float(selected["cap_drop_count"]),
                "fallback_success_count": float(selected["fallback_success_count"]),
                "safe_fill_count": float(selected["safe_fill_count"]),
                "missed_safe_opportunity_count": float(selected["missed_safe_opportunity_count"]),
                "blank_budget_repair_count": float(selected["blank_budget_repair_count"]),
                "final_blank_count": float(selected["final_blank_count"]),
                "post_decode_policy_blank_count": float(selected["post_decode_policy_blank_count"]),
                "post_decode_no_valid_candidate_count": float(selected["post_decode_no_valid_candidate_count"]),
                "post_decode_demand_cap_count": float(selected["post_decode_demand_cap_count"]),
                "post_decode_no_slot_count": float(selected["post_decode_no_slot_count"]),
                "post_decode_other_drop_count": float(selected["post_decode_other_drop_count"]),
                "uniq_ue": float(selected["uniq_ue"]),
                "slot_repair_count": float(selected["slot_repair_count"]),
                "coverage_repair_count": float(selected["coverage_repair_count"]),
                "non_hard_override_count": float(selected["non_hard_override_count"]),
                "candidate_policy_mode_direct": float(selected["candidate_policy_mode_direct"]),
                "budgeted_changed_count": float(selected["budgeted_changed_count"]),
                "budgeted_no_headroom_count": float(selected["budgeted_no_headroom_count"]),
                "budgeted_active_ue_count": float(selected["budgeted_active_ue_count"]),
                "budgeted_tail_bias_candidate_mean": float(selected["budgeted_tail_bias_candidate_mean"]),
                "budgeted_tail_bias_selected_mean": float(selected["budgeted_tail_bias_selected_mean"]),
                "budgeted_tail_bias_active_count": float(selected["budgeted_tail_bias_active_count"]),
                "action_replay_mismatch_count": float(
                    selected.get("action_replay_mismatch_count", 0.0)
                ),
                "candidate_truncation_count": float(
                    selected.get("candidate_truncation_count", 0.0)
                ),
                "illegal_action_count": float(
                    selected["post_decode_no_slot_count"]
                    + selected["post_decode_other_drop_count"]
                    + native_reject_count
                ),
                "native_reject_count": native_reject_count,
                "native_reject_diagnostics_available": float(
                    native_reject_available
                ),
            }
            if int(args.save_steps) == 1:
                rows.append(row)
            tail_rows.append(row)
            for key in sums:
                sums[key] += float(row[key])
            if step_idx >= int(args.stats_warmup_steps):
                measured_tti = int(next_info.get("tti", step_idx + 1))
                measured_tti_first = (
                    measured_tti if measured_tti_first is None else measured_tti_first
                )
                measured_tti_last = measured_tti
                if eval_diagnostics is not None:
                    trace = selected.get("independent_policy_trace")
                    diagnostic = getattr(trace, "diagnostics", None)
                    if diagnostic is None:
                        raise ValueError("independent evaluation is missing policy diagnostics")
                    eval_diagnostics.append(diagnostic)
                    rescue_diagnostics = getattr(trace, "rescue_diagnostics", None)
                    chosen_ue_per_prg = selected.get("decoded_chosen_ue_per_prg")
                    if rescue_diagnostics is None or chosen_ue_per_prg is None:
                        raise ValueError("independent evaluation is missing rescue cohort diagnostics")
                    if rescue_decision_rows is None:
                        raise RuntimeError("rescue decision diagnostics accumulator is unavailable")
                    rescue_decision_rows.append(rescue_diagnostics)
                    meta = graph.snapshot.static_meta
                    step_ue_rows, step_cell_rows = join_rescue_cohort_tti(
                        rescue_diagnostics,
                        chosen_ue_per_prg=torch.as_tensor(chosen_ue_per_prg),
                        info=next_info,
                        n_ue=int(meta.n_ue),
                        n_cell=int(meta.n_cell),
                        n_prg=int(meta.n_prg),
                        tti=measured_tti,
                    )
                    if rescue_ue_rows is None or rescue_cell_rows is None:
                        raise RuntimeError("rescue cohort accumulators are unavailable")
                    rescue_ue_rows.extend(step_ue_rows)
                    rescue_cell_rows.extend(step_cell_rows)
                    if service_accounting_audit_enabled:
                        service_diagnostics = getattr(
                            trace, "service_accounting_diagnostics", None
                        )
                        if service_diagnostics is None or service_accounting_rows is None:
                            raise ValueError(
                                "independent evaluation is missing service accounting diagnostics"
                            )
                        service_accounting_rows.extend(
                            join_service_accounting_tti(
                                service_diagnostics,
                                info=next_info,
                                n_ue=int(meta.n_ue),
                                n_cell=int(meta.n_cell),
                                n_prg=int(meta.n_prg),
                                tti=measured_tti,
                            )
                        )
                    if phy_tbs_alignment_audit_enabled:
                        service_diagnostics = getattr(
                            trace, "service_accounting_diagnostics", None
                        )
                        if service_diagnostics is None or phy_tbs_alignment_rows is None:
                            raise ValueError(
                                "independent evaluation is missing PHY/TBS audit diagnostics"
                            )
                        phy_tbs_alignment_rows.extend(
                            join_phy_tbs_alignment_tti(
                                service_diagnostics,
                                info=next_info,
                                n_ue=int(meta.n_ue),
                                n_cell=int(meta.n_cell),
                                n_prg=int(meta.n_prg),
                                tti=measured_tti,
                                protocol_id=phy_tbs_alignment_protocol,
                            )
                        )
                    transmission = extract_failed_transmission_bytes(
                        next_info,
                        expected_n_ue=n_active_ue,
                    )
                    if (
                        stats_sum_scheduled_bytes is None
                        or stats_sum_ue_goodput_bytes is None
                        or stats_sum_failed_bytes is None
                    ):
                        raise RuntimeError("independent byte accumulators are unavailable")
                    stats_sum_scheduled_bytes += np.float64(transmission["scheduled_bytes"])
                    stats_sum_ue_goodput_bytes += np.float64(transmission["goodput_bytes"])
                    stats_sum_failed_bytes += np.float64(transmission["failed_bytes"])
                if n_active_ue > 0:
                    goodput_ue = np.asarray(
                        next_info.get(
                            "actual_ue_goodput_bytes", np.zeros((n_active_ue,))
                        ),
                        dtype=np.float64,
                    ).reshape(n_active_ue)
                    if (
                        delivered_time_ms.shape != (n_active_ue,)
                        or expired_time_ms.shape != (n_active_ue,)
                    ):
                        raise ValueError("expiry-aware evaluator UE diagnostic shape mismatch")
                    ue_measured_goodput_bytes += np.maximum(goodput_ue, 0.0)
                    ue_measured_outcome_time_ms += np.maximum(
                        delivered_time_ms - last_delivered_time_ms, 0.0
                    ) + np.maximum(expired_time_ms - last_expired_time_ms, 0.0)
                    tb_tx = np.asarray(
                        next_info.get("actual_ue_tb_tx_count", np.zeros((n_active_ue,))),
                        dtype=np.float64,
                    ).reshape(n_active_ue)
                    tb_err = np.asarray(
                        next_info.get("actual_ue_tb_err_count", np.zeros((n_active_ue,))),
                        dtype=np.float64,
                    ).reshape(n_active_ue)
                    measured_tb_tx_count += float(np.maximum(tb_tx, 0.0).sum())
                    measured_tb_err_count += float(np.maximum(tb_err, 0.0).sum())
                for key in stats_sums:
                    stats_sums[key] += float(row[key])
                stats_count += 1
            last_delivered_time_ms = delivered_time_ms
            last_expired_time_ms = expired_time_ms
            step_idx += 1

            if int(args.print_every_steps) > 0 and (
                (step_idx % int(args.print_every_steps)) == 0 or done or (args.max_steps > 0 and step_idx >= args.max_steps)
            ):
                tail = list(tail_rows)
                print(
                    f"[hgraph-eval] tti={step_idx:04d} "
                    f"goodput={float(np.mean([r['goodput_mbps'] for r in tail])):.4f} "
                    f"tb_err={float(np.mean([r['tb_err_rate'] for r in tail])):.4f} "
                    f"blank={float(np.mean([r['blank_ratio'] for r in tail])):.4f} "
                    f"fill={float(np.mean([r['fill_ratio'] for r in tail])):.4f} "
                    f"capDrop={float(np.mean([r['cap_drop_count'] for r in tail])):.2f} "
                    f"fbOK={float(np.mean([r['fallback_success_count'] for r in tail])):.2f} "
                    f"safeFill={float(np.mean([r['safe_fill_count'] for r in tail])):.2f} "
                    f"missedSafe={float(np.mean([r['missed_safe_opportunity_count'] for r in tail])):.2f} "
                    f"blankBudget={float(np.mean([r['blank_budget_repair_count'] for r in tail])):.2f} "
                    f"blankCnt={float(np.mean([r['final_blank_count'] for r in tail])):.2f} "
                    f"dropCap={float(np.mean([r['post_decode_demand_cap_count'] for r in tail])):.2f} "
                    f"dropSlot={float(np.mean([r['post_decode_no_slot_count'] for r in tail])):.2f} "
                    f"dropOth={float(np.mean([r['post_decode_other_drop_count'] for r in tail])):.2f} "
                    f"budgChg={float(np.mean([r['budgeted_changed_count'] for r in tail])):.2f} "
                    f"budgNoHr={float(np.mean([r['budgeted_no_headroom_count'] for r in tail])):.2f} "
                    f"tailBiasSel={float(np.mean([r['budgeted_tail_bias_selected_mean'] for r in tail])):.3f} "
                    f"uniq_ue={float(np.mean([r['uniq_ue'] for r in tail])):.2f}"
                )

            if done or (int(args.max_steps) > 0 and step_idx >= int(args.max_steps)):
                break
            obs, info = next_obs, next_info

    finally:
        env.close()

    _validate_non_starvation_guard_completion(
        guard_mode=guard_mode,
        completed_action_ttis=step_idx,
        full_rows=guard_full_rows,
        measured_rows=guard_measured_rows,
    )

    denom = max(step_idx, 1)
    def _stats_mean(key: str):
        return None if stats_count <= 0 else float(stats_sums[key] / float(stats_count))

    def _stats_sum(key: str):
        return None if stats_count <= 0 else float(stats_sums[key])

    diagnostic_summary: dict[str, float] | None = None
    transmission_summary: dict[str, Any] | None = None
    service_accounting_report: dict[str, object] | None = None
    phy_tbs_alignment_report: dict[str, object] | None = None
    if eval_diagnostics is not None:
        if stats_count <= 0 or not eval_diagnostics:
            raise ValueError("independent diagnostics require at least one measured TTI")
        eval_diag = concatenate_independent_policy_diagnostics(eval_diagnostics)
        expected_decisions = (
            stats_count
            * int(graph.snapshot.static_meta.n_cell)
            * int(graph.snapshot.static_meta.n_prg)
        )
        actual_decisions = int(eval_diag.valid_ue_count.numel())
        if actual_decisions != expected_decisions:
            raise ValueError(
                "independent evaluation diagnostic decision count mismatch: "
                f"got={actual_decisions} expected={expected_decisions}"
            )
        if (
            rescue_ue_rows is None
            or rescue_cell_rows is None
            or rescue_decision_rows is None
        ):
            raise RuntimeError("rescue cohort accumulators are unavailable")
        rescued_prgs = sum(row.rescued_prg_count for row in rescue_ue_rows)
        expected_rescued_prgs = int(eval_diag.fragmentation_rescued.sum().item())
        if rescued_prgs != expected_rescued_prgs:
            raise ValueError(
                "rescue cohort rescued PRG count mismatch: "
                f"got={rescued_prgs} expected={expected_rescued_prgs}"
            )
        planned_prgs = sum(row.planned_prg_count for row in rescue_ue_rows)
        expected_planned_prgs = int((~eval_diag.selected_blank).sum().item())
        if planned_prgs != expected_planned_prgs:
            raise ValueError(
                "rescue cohort planned PRG count mismatch: "
                f"got={planned_prgs} expected={expected_planned_prgs}"
            )
        if (
            stats_sum_scheduled_bytes is None
            or stats_sum_ue_goodput_bytes is None
            or stats_sum_failed_bytes is None
        ):
            raise RuntimeError("independent byte accumulators are unavailable")
        diagnostic_summary = summarize_independent_policy_diagnostics(
            eval_diag,
            classification_prefix="",
        )
        stats_mean_scheduled_mbps = float(
            stats_sum_scheduled_bytes / np.float64(stats_count) * np.float64(0.016)
        )
        stats_mean_ue_goodput_mbps = float(
            stats_sum_ue_goodput_bytes / np.float64(stats_count) * np.float64(0.016)
        )
        stats_failed_mbps = float(
            stats_sum_failed_bytes / np.float64(stats_count) * np.float64(0.016)
        )
        existing_goodput_mbps = _stats_mean("goodput_mbps")
        if existing_goodput_mbps is None:
            raise ValueError("measured-window goodput is unavailable")
        validate_measured_window_goodput_mbps(
            derived_goodput_mbps=stats_mean_ue_goodput_mbps,
            existing_goodput_mbps=existing_goodput_mbps,
        )
        transmission_summary = {
            "independent_policy_diagnostics_enabled": 1,
            "stats_sum_scheduled_bytes": float(stats_sum_scheduled_bytes),
            "stats_sum_ue_goodput_bytes": float(stats_sum_ue_goodput_bytes),
            "stats_sum_failed_bytes": float(stats_sum_failed_bytes),
            "stats_mean_scheduled_mbps": stats_mean_scheduled_mbps,
            "stats_mean_ue_goodput_mbps": stats_mean_ue_goodput_mbps,
            "stats_failed_mbps": stats_failed_mbps,
        }
        transmission_summary["rescue_cohort_diagnostics_enabled"] = 1
        transmission_summary["rescue_cohort_diagnostics"] = summarize_rescue_cohorts(
            rescue_ue_rows,
            rescue_cell_rows,
        )
        transmission_summary["local_prg_phy_interference_diagnostics"] = (
            summarize_local_prg_phy_interference(
                concatenate_rescue_decision_diagnostics(rescue_decision_rows),
                n_cell=int(graph.snapshot.static_meta.n_cell),
                n_prg=int(graph.snapshot.static_meta.n_prg),
            )
        )
        if service_accounting_audit_enabled:
            if service_accounting_rows is None:
                raise RuntimeError("service accounting audit rows are unavailable")
            service_accounting_report = summarize_service_accounting_audit(
                service_accounting_rows,
                scheduled_bytes_semantics=NEW_TRANSMISSION_QUEUE_DEQUEUE,
                per_prg_scheduled_bytes_available=False,
            )
            transmission_summary["service_accounting_audit"] = (
                service_accounting_report
            )
        if phy_tbs_alignment_audit_enabled:
            if phy_tbs_alignment_rows is None:
                raise RuntimeError("PHY/TBS alignment audit rows are unavailable")
            phy_tbs_alignment_report = summarize_phy_tbs_alignment_audit(
                phy_tbs_alignment_rows,
                protocol_id=phy_tbs_alignment_protocol,
            )
            transmission_summary["phy_tbs_alignment_audit"] = (
                phy_tbs_alignment_report
            )

    mean_packet_delay_weighted = (
        float(sums["packet_system_time_ms"] / max(sums["packet_completed_count"], 1.0))
        if sums["packet_completed_count"] > 0.0
        else 0.0
    )
    stats_packet_delay_weighted = (
        None
        if stats_count <= 0 or stats_sums["packet_completed_count"] <= 0.0
        else float(stats_sums["packet_system_time_ms"] / stats_sums["packet_completed_count"])
    )
    mean_packet_rate_weighted = (
        float((sums["packet_delivered_bits"] / 1000.0) / max(sums["packet_system_time_ms"], 1.0e-6))
        if sums["packet_system_time_ms"] > 0.0
        else 0.0
    )
    stats_packet_rate_weighted = (
        None
        if stats_count <= 0 or stats_sums["packet_system_time_ms"] <= 0.0
        else float((stats_sums["packet_delivered_bits"] / 1000.0) / stats_sums["packet_system_time_ms"])
    )
    stats_outcome_delay_weighted = (
        None
        if stats_count <= 0 or stats_sums["packet_outcome_count"] <= 0.0
        else float(
            stats_sums["packet_outcome_system_time_ms"]
            / stats_sums["packet_outcome_count"]
        )
    )
    ue_rates_mbps = np.zeros_like(ue_measured_goodput_bytes)
    ue_rate_active = ue_measured_outcome_time_ms > 0.0
    ue_rates_mbps[ue_rate_active] = (
        ue_measured_goodput_bytes[ue_rate_active]
        * 8.0
        / (ue_measured_outcome_time_ms[ue_rate_active] * 1000.0)
    )
    active_rates = ue_rates_mbps[ue_rate_active]
    outcome_delay_p95_ms: float | None = None
    outcome_sample_count = 0
    blank_expiry_attribution_report: dict[str, object] | None = None
    blank_expiry_decision_output_rows: list[dict[str, object]] | None = None
    blank_expiry_packet_output_rows: list[dict[str, object]] | None = None
    blank_expiry_atomic_output_rows: list[dict[str, object]] | None = None
    outcome_path = Path(args.outcome_samples).resolve() if args.outcome_samples else None
    if outcome_path is not None and not outcome_path.is_file():
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not outcome_path.is_file():
            time.sleep(0.05)
    if outcome_path is not None and outcome_path.is_file() and measured_tti_first is not None:
        outcome_rows = filter_outcome_samples(
            outcome_path,
            terminal_tti_exclusive=int(measured_tti_first) - 1,
            terminal_tti_inclusive=int(measured_tti_last),
        )
        delays = np.asarray(
            [float(row["system_time_ms"]) for row in outcome_rows], dtype=np.float64
        )
        outcome_sample_count = int(delays.size)
        outcome_delay_p95_ms = float(np.percentile(delays, 95)) if delays.size else None
    if blank_expiry_audit_enabled:
        if outcome_path is None or not outcome_path.is_file():
            raise RuntimeError("Blank-expiry attribution outcome CSV is unavailable")
        frozen_measured_window = {
            V1_PROTOCOL_ID: (1001, 1512),
            V2_FULL_HORIZON_PROTOCOL_ID: (1001, 5000),
        }[blank_expiry_attribution_protocol]
        if (measured_tti_first, measured_tti_last) != frozen_measured_window:
            raise RuntimeError(
                "frozen Blank-expiry attribution window mismatch: "
                f"observed={(measured_tti_first, measured_tti_last)} "
                f"expected={frozen_measured_window}"
            )
        if blank_expiry_traces is None:
            raise RuntimeError("Blank-expiry attribution traces are unavailable")
        expected_trace_count = measured_tti_last - blank_expiry_trace_first_tti + 1
        if len(blank_expiry_traces) != expected_trace_count:
            raise RuntimeError(
                "Blank-expiry attribution trace coverage mismatch: "
                f"got={len(blank_expiry_traces)} expected={expected_trace_count}"
            )
        formal_packets = read_formal_expired_packets(
            outcome_path,
            measured_tti_first=measured_tti_first,
            measured_tti_last=measured_tti_last,
            trace_first_tti=blank_expiry_trace_first_tti,
        )
        atomic_exposures = build_atomic_exposures(
            packets=formal_packets,
            traces=blank_expiry_traces,
            serving_cell_by_ue=serving_cell_by_ue,
            n_cell=int(initial_graph.snapshot.static_meta.n_cell),
            n_prg=int(initial_graph.snapshot.static_meta.n_prg),
        )
        blank_expiry_attribution_report = build_blank_expiry_attribution_report(
            packets=formal_packets,
            exposures=atomic_exposures,
            measured_tti_first=measured_tti_first,
            measured_tti_last=measured_tti_last,
            trace_first_tti=blank_expiry_trace_first_tti,
            n_prg=int(initial_graph.snapshot.static_meta.n_prg),
            protocol_id=blank_expiry_attribution_protocol,
        )
        blank_expiry_attribution_report["audit_status"] = "PASS"
        blank_expiry_attribution_report["interpretation"] = (
            "read-only exposure attribution; cell-PRG records are not causal effects"
        )
        blank_expiry_decision_output_rows = [
            row
            for trace in blank_expiry_traces
            for row in decision_diagnostics_to_rows(
                tti=trace.tti,
                diagnostics=trace.decisions,
            )
        ]
        blank_expiry_packet_output_rows = packet_lifecycle_rows(
            packets=formal_packets,
            traces=blank_expiry_traces,
            serving_cell_by_ue=serving_cell_by_ue,
            n_cell=int(initial_graph.snapshot.static_meta.n_cell),
            n_prg=int(initial_graph.snapshot.static_meta.n_prg),
        )
        blank_expiry_atomic_output_rows = [
            asdict(exposure) for exposure in atomic_exposures
        ]
    sim_bin_path = Path(args.sim_bin).resolve() if args.sim_bin else None

    summary = {
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": _sha256(ckpt_path),
        "actor_family": getattr(actor, "actor_family", "legacy_stageb_hgraph"),
        "requires_pfq_features": bool(
            getattr(actor, "requires_pfq_features", True)
        ),
        "protocol_version": VERSION,
        "runtime_harq_enabled": runtime_harq_enabled,
        "sim_binary_sha256": (
            _sha256(sim_bin_path)
            if sim_bin_path is not None and sim_bin_path.is_file()
            else os.environ.get("CUMAC_SIM_BINARY_SHA256", "")
        ),
        "independent_deletion_equal": deletion_equal,
        "independent_deletion_test_device": deletion_test_device,
        "independent_headroom_accounting": independent_headroom_accounting,
        "ctbs_pre_action_handshake_enabled": ctbs_enabled,
        "ctbs_config_sha256": ctbs_config_sha256,
        "ctbs_snapshot_count": ctbs_snapshot_count,
        "ctbs_snapshot_id_first": ctbs_snapshot_id_first,
        "ctbs_snapshot_id_last": ctbs_snapshot_id_last,
        "ctbs_mcs_mismatch_count": ctbs_mcs_mismatch_count,
        "ctbs_layer_mismatch_count": ctbs_layer_mismatch_count,
        "ctbs_tbs_mismatch_count": ctbs_tbs_mismatch_count,
        "ctbs_reference_label_deletion_equal": (
            ctbs_reference_label_deletion_equal if ctbs_enabled else None
        ),
        "ctbs_raw_selected_service_bytes_sum": ctbs_raw_selected_service_bytes_sum,
        "ctbs_capacity_decrease_count": ctbs_capacity_decrease_count,
        "ctbs_remaining_rebound_count": ctbs_remaining_rebound_count,
        "ctbs_mask_reactivation_count": ctbs_mask_reactivation_count,
        "ctbs_mask_trigger_count_by_local_prg": ctbs_mask_trigger_count_by_local_prg,
        "ctbs_selected_count_by_local_prg": ctbs_selected_count_by_local_prg,
        "ctbs_valid_candidate_count_by_local_prg": (
            ctbs_valid_candidate_count_by_local_prg
        ),
        "decode_mode": str(args.decode_mode),
        "candidate_policy_mode": str(candidate_policy_mode),
        "independent_service_accounting_audit_enabled": int(
            service_accounting_audit_enabled
        ),
        "independent_phy_tbs_alignment_audit_enabled": int(
            phy_tbs_alignment_audit_enabled
        ),
        "independent_phy_tbs_alignment_protocol": phy_tbs_alignment_protocol,
        "independent_blank_expiry_attribution_audit_enabled": int(
            blank_expiry_audit_enabled
        ),
        "blank_expiry_trace_first_tti": (
            blank_expiry_trace_first_tti if blank_expiry_audit_enabled else None
        ),
        "blank_expiry_trace_count": (
            len(blank_expiry_traces)
            if blank_expiry_traces is not None
            else None
        ),
        "blank_expiry_attribution_status": (
            blank_expiry_attribution_report.get("audit_status")
            if blank_expiry_attribution_report is not None
            else None
        ),
        "blank_expiry_attribution_classification": (
            blank_expiry_attribution_report.get("formal_conclusion")
            if blank_expiry_attribution_report is not None
            else None
        ),
        "ue_prg_topk": int(ue_prg_topk),
        "ue_prg_diversity_extra": int(ue_prg_diversity_extra),
        "candidate_mode": str(candidate_mode),
        "candidate_width_policy": (
            runtime_contract.candidate_width_policy if runtime_contract is not None else None
        ),
        "candidate_truncation": (
            runtime_contract.candidate_truncation if runtime_contract is not None else None
        ),
        "training_observed_max_candidates": (
            runtime_contract.training_observed_max_candidates
            if runtime_contract is not None
            else None
        ),
        "topology_candidate_width": (
            runtime_contract.topology_candidate_width if runtime_contract is not None else None
        ),
        "runtime_candidate_width": (
            runtime_contract.runtime_candidate_width if runtime_contract is not None else None
        ),
        "association_candidate_counts": association_candidate_counts,
        "candidate_full_coverage": True if independent_mode else None,
        "candidate_truncation_count": int(sums["candidate_truncation_count"]),
        "actor_state_dict_sha256": actor_state_hash if independent_mode else None,
        "candidate_width_equivalence_reference_width": (
            equivalence_reference_width if equivalence_reference_width else None
        ),
        "candidate_width_equivalence_protocol": (
            equivalence_protocol if equivalence_reference_width else None
        ),
        "candidate_width_valid_logits_atol": (
            W0_V2_VALID_LOGITS_ATOL
            if equivalence_reference_width and equivalence_protocol == "fp32-w0-v2"
            else None
        ),
        "candidate_width_chosen_logprob_atol": (
            W0_V2_CHOSEN_LOGPROB_ATOL
            if equivalence_reference_width and equivalence_protocol == "fp32-w0-v2"
            else None
        ),
        "candidate_width_equivalence_steps": int(equivalence_steps),
        "candidate_width_equivalence": equivalence_report,
        "include_pfq_score_anchor": int(include_pfq_score_anchor),
        "include_candidate_risk_features": int(include_candidate_risk_features),
        "candidate_risk_feature_dim": int(candidate_risk_feature_dim),
        "tb_err_ema_alpha": float(tb_err_ema_alpha),
        "slot_repair_max_per_cell": int(slot_repair_max_per_cell),
        "slot_repair_min_backlog_bytes": float(slot_repair_min_backlog_bytes),
        "safe_fill_max_per_cell": int(safe_fill_max_per_cell),
        "safe_fill_min_quality_db": float(safe_fill_min_quality_db),
        "safe_fill_max_quality_gap_db": float(safe_fill_max_quality_gap_db),
        "safe_fill_max_prg_risk": float(safe_fill_max_prg_risk),
        "safe_fill_min_backlog_bytes": float(safe_fill_min_backlog_bytes),
        "safe_fill_max_ue_tb_err": float(safe_fill_max_ue_tb_err),
        "candidate_blank_budget_max_per_cell": int(candidate_blank_budget_max_per_cell),
        "candidate_blank_budget_min_decision_logit": float(candidate_blank_budget_min_decision_logit),
        "candidate_decode_demand_slack_bytes": float(candidate_decode_demand_slack_bytes),
        "candidate_marginal_headroom_slack_bytes": float(candidate_marginal_headroom_slack_bytes),
        "candidate_headroom_rank_mode": str(candidate_headroom_rank_mode),
        "candidate_budgeted_select_mode": str(candidate_budgeted_select_mode),
        "candidate_budgeted_hard_cap": int(candidate_budgeted_hard_cap),
        "candidate_budgeted_usage_penalty": float(candidate_budgeted_usage_penalty),
        "candidate_budgeted_overcap_penalty": float(candidate_budgeted_overcap_penalty),
        "candidate_budgeted_first_service_bonus": float(candidate_budgeted_first_service_bonus),
        "candidate_budgeted_tail_bias_coef": float(candidate_budgeted_tail_bias_coef),
        "candidate_budgeted_tail_bias_service_target": float(candidate_budgeted_tail_bias_service_target),
        "candidate_budgeted_tail_bias_ttl_guard_ms": float(candidate_budgeted_tail_bias_ttl_guard_ms),
        "candidate_budgeted_tail_bias_max": float(candidate_budgeted_tail_bias_max),
        "candidate_budgeted_rescue_bias_coef": float(candidate_budgeted_rescue_bias_coef),
        "candidate_budgeted_completion_bonus": float(candidate_budgeted_completion_bonus),
        "candidate_budgeted_tail_quota_enable": int(candidate_budgeted_tail_quota_enable),
        "candidate_seq_state_coef": float(candidate_seq_state_coef),
        "candidate_seq_fail_risk_feature_offset": int(candidate_seq_fail_risk_feature_offset),
        "save_steps": int(args.save_steps),
        "completed_steps": int(step_idx),
        "stats_warmup_steps": int(args.stats_warmup_steps),
        "stats_count": int(stats_count),
        "measured_tti_first": measured_tti_first,
        "measured_tti_last": measured_tti_last,
        "packet_outcome_measured_count": outcome_sample_count,
        "packet_outcome_delay_p95_ms": outcome_delay_p95_ms,
        "ue_macro_expiry_aware_rate_mbps": (
            float(active_rates.mean()) if active_rates.size else 0.0
        ),
        "ue_p10_expiry_aware_rate_mbps": (
            float(np.percentile(active_rates, 10)) if active_rates.size else 0.0
        ),
        "ue_min_expiry_aware_rate_mbps": (
            float(active_rates.min()) if active_rates.size else 0.0
        ),
        "ue_expiry_aware_rate_active_count": int(active_rates.size),
        "mean_goodput_mbps": float(sums["goodput_mbps"] / denom),
        "mean_tb_err_rate": float(sums["tb_err_rate"] / denom),
        "mean_expiry_drop_rate": float(sums["expiry_drop_rate"] / denom),
        "mean_prg_utilization_ratio": float(sums["prg_utilization_ratio"] / denom),
        "mean_prg_reuse_ratio": float(sums["prg_reuse_ratio"] / denom),
        "mean_packet_completed_count": float(sums["packet_completed_count"] / denom),
        "mean_packet_delivered_bits": float(sums["packet_delivered_bits"] / denom),
        "mean_packet_system_time_ms": float(sums["packet_system_time_ms"] / denom),
        "mean_packet_delay_mean_ms": float(sums["packet_delay_mean_ms"] / denom),
        "mean_weighted_packet_delay_mean_ms": mean_packet_delay_weighted,
        "mean_packet_effective_service_rate_mbps": float(
            sums["packet_effective_service_rate_mbps"] / denom
        ),
        "mean_weighted_packet_effective_service_rate_mbps": mean_packet_rate_weighted,
        "mean_packet_effective_service_rate_per_packet_mean_mbps": float(
            sums["packet_effective_service_rate_per_packet_mean_mbps"] / denom
        ),
        "mean_blank_ratio": float(sums["blank_ratio"] / denom),
        "mean_fill_ratio": float(sums["fill_ratio"] / denom),
        "mean_cap_drop_count": float(sums["cap_drop_count"] / denom),
        "mean_fallback_success_count": float(sums["fallback_success_count"] / denom),
        "mean_safe_fill_count": float(sums["safe_fill_count"] / denom),
        "mean_missed_safe_opportunity_count": float(sums["missed_safe_opportunity_count"] / denom),
        "mean_blank_budget_repair_count": float(sums["blank_budget_repair_count"] / denom),
        "mean_final_blank_count": float(sums["final_blank_count"] / denom),
        "mean_post_decode_policy_blank_count": float(sums["post_decode_policy_blank_count"] / denom),
        "mean_post_decode_no_valid_candidate_count": float(sums["post_decode_no_valid_candidate_count"] / denom),
        "mean_post_decode_demand_cap_count": float(sums["post_decode_demand_cap_count"] / denom),
        "mean_post_decode_no_slot_count": float(sums["post_decode_no_slot_count"] / denom),
        "mean_post_decode_other_drop_count": float(sums["post_decode_other_drop_count"] / denom),
        "mean_uniq_ue": float(sums["uniq_ue"] / denom),
        "mean_slot_repair_count": float(sums["slot_repair_count"] / denom),
        "mean_coverage_repair_count": float(sums["coverage_repair_count"] / denom),
        "mean_non_hard_override_count": float(sums["non_hard_override_count"] / denom),
        "mean_candidate_policy_mode_direct": float(sums["candidate_policy_mode_direct"] / denom),
        "mean_budgeted_changed_count": float(sums["budgeted_changed_count"] / denom),
        "mean_budgeted_no_headroom_count": float(sums["budgeted_no_headroom_count"] / denom),
        "mean_budgeted_active_ue_count": float(sums["budgeted_active_ue_count"] / denom),
        "mean_budgeted_tail_bias_candidate_mean": float(sums["budgeted_tail_bias_candidate_mean"] / denom),
        "mean_budgeted_tail_bias_selected_mean": float(sums["budgeted_tail_bias_selected_mean"] / denom),
        "mean_budgeted_tail_bias_active_count": float(sums["budgeted_tail_bias_active_count"] / denom),
        "stats_mean_goodput_mbps": _stats_mean("goodput_mbps"),
        "stats_mean_tb_err_rate": _stats_mean("tb_err_rate"),
        "stats_mean_expiry_drop_rate": _stats_mean("expiry_drop_rate"),
        "stats_sum_expired_bytes": _stats_sum("expired_bytes"),
        "measured_tb_tx_count": measured_tb_tx_count,
        "measured_tb_err_count": measured_tb_err_count,
        "measured_tb_bler": (
            measured_tb_err_count / measured_tb_tx_count
            if measured_tb_tx_count > 0.0
            else None
        ),
        "stats_mean_prg_utilization_ratio": _stats_mean("prg_utilization_ratio"),
        "stats_mean_prg_reuse_ratio": _stats_mean("prg_reuse_ratio"),
        "stats_sum_packet_completed_count": _stats_sum("packet_completed_count"),
        "stats_sum_packet_delivered_bits": _stats_sum("packet_delivered_bits"),
        "stats_sum_packet_system_time_ms": _stats_sum("packet_system_time_ms"),
        "stats_mean_packet_completed_count": _stats_mean("packet_completed_count"),
        "stats_mean_packet_delivered_bits": _stats_mean("packet_delivered_bits"),
        "stats_mean_packet_system_time_ms": _stats_mean("packet_system_time_ms"),
        "stats_mean_packet_delay_mean_ms": _stats_mean("packet_delay_mean_ms"),
        "stats_weighted_packet_delay_mean_ms": stats_packet_delay_weighted,
        "stats_mean_packet_effective_service_rate_mbps": _stats_mean(
            "packet_effective_service_rate_mbps"
        ),
        "stats_weighted_packet_effective_service_rate_mbps": stats_packet_rate_weighted,
        "stats_mean_packet_effective_service_rate_per_packet_mean_mbps": _stats_mean(
            "packet_effective_service_rate_per_packet_mean_mbps"
        ),
        "stats_sum_packet_expired_count": _stats_sum("packet_expired_count"),
        "stats_sum_packet_expired_system_time_ms": _stats_sum(
            "packet_expired_system_time_ms"
        ),
        "stats_sum_packet_outcome_count": _stats_sum("packet_outcome_count"),
        "stats_sum_packet_outcome_system_time_ms": _stats_sum(
            "packet_outcome_system_time_ms"
        ),
        "stats_weighted_packet_outcome_delay_mean_ms": stats_outcome_delay_weighted,
        "stats_mean_packet_expiry_aware_rate_mbps": _stats_mean(
            "packet_expiry_aware_rate_mbps"
        ),
        "stats_mean_packet_expiry_aware_rate_per_packet_mean_mbps": _stats_mean(
            "packet_expiry_aware_rate_per_packet_mean_mbps"
        ),
        "stats_sum_action_replay_mismatch_count": _stats_sum(
            "action_replay_mismatch_count"
        ),
        "stats_sum_candidate_truncation_count": _stats_sum(
            "candidate_truncation_count"
        ),
        "stats_sum_illegal_action_count": _stats_sum("illegal_action_count"),
        "stats_sum_native_reject_count": _stats_sum("native_reject_count"),
        "stats_mean_native_reject_diagnostics_available": _stats_mean(
            "native_reject_diagnostics_available"
        ),
        "stats_mean_blank_ratio": _stats_mean("blank_ratio"),
        "stats_mean_fill_ratio": _stats_mean("fill_ratio"),
        "stats_mean_cap_drop_count": _stats_mean("cap_drop_count"),
        "stats_mean_fallback_success_count": _stats_mean("fallback_success_count"),
        "stats_mean_safe_fill_count": _stats_mean("safe_fill_count"),
        "stats_mean_missed_safe_opportunity_count": _stats_mean("missed_safe_opportunity_count"),
        "stats_mean_blank_budget_repair_count": _stats_mean("blank_budget_repair_count"),
        "stats_mean_final_blank_count": _stats_mean("final_blank_count"),
        "stats_mean_post_decode_policy_blank_count": _stats_mean("post_decode_policy_blank_count"),
        "stats_mean_post_decode_no_valid_candidate_count": _stats_mean("post_decode_no_valid_candidate_count"),
        "stats_mean_post_decode_demand_cap_count": _stats_mean("post_decode_demand_cap_count"),
        "stats_mean_post_decode_no_slot_count": _stats_mean("post_decode_no_slot_count"),
        "stats_mean_post_decode_other_drop_count": _stats_mean("post_decode_other_drop_count"),
        "stats_mean_uniq_ue": _stats_mean("uniq_ue"),
        "stats_mean_slot_repair_count": _stats_mean("slot_repair_count"),
        "stats_mean_coverage_repair_count": _stats_mean("coverage_repair_count"),
        "stats_mean_non_hard_override_count": _stats_mean("non_hard_override_count"),
        "stats_mean_candidate_policy_mode_direct": _stats_mean("candidate_policy_mode_direct"),
        "stats_mean_budgeted_changed_count": _stats_mean("budgeted_changed_count"),
        "stats_mean_budgeted_no_headroom_count": _stats_mean("budgeted_no_headroom_count"),
        "stats_mean_budgeted_active_ue_count": _stats_mean("budgeted_active_ue_count"),
        "stats_mean_budgeted_tail_bias_candidate_mean": _stats_mean("budgeted_tail_bias_candidate_mean"),
        "stats_mean_budgeted_tail_bias_selected_mean": _stats_mean("budgeted_tail_bias_selected_mean"),
        "stats_mean_budgeted_tail_bias_active_count": _stats_mean("budgeted_tail_bias_active_count"),
        "actor_config": ckpt.get("actor_config", {}),
    }
    summary.update(
        _build_non_starvation_guard_summary(
            guard_mode=guard_mode,
            full_rows=guard_full_rows,
            measured_rows=guard_measured_rows,
        )
    )
    if diagnostic_summary is not None and transmission_summary is not None:
        summary.update(
            {f"diagnostic_{key}": value for key, value in diagnostic_summary.items()}
        )
        summary.update(transmission_summary)

    if blank_expiry_attribution_report is not None:
        runtime_gates = {
            "candidate_truncation_count_zero": int(summary["candidate_truncation_count"]) == 0,
            "replay_mismatch_count_zero": float(
                summary["stats_sum_action_replay_mismatch_count"] or 0.0
            ) == 0.0,
            "illegal_action_count_zero": float(
                summary["stats_sum_illegal_action_count"] or 0.0
            ) == 0.0,
            "pfq_field_deletion_equality": summary["independent_deletion_equal"] is True,
            "full_candidate_coverage": summary["candidate_full_coverage"] is True,
            "state_dict_hash_exact": actor_state_hash == FROZEN_B4_STATE_DICT_SHA256,
            "ctbs_mcs_parity_exact": ctbs_mcs_mismatch_count == 0,
            "ctbs_layer_parity_exact": ctbs_layer_mismatch_count == 0,
            "ctbs_tbs_parity_exact": ctbs_tbs_mismatch_count == 0,
            "ctbs_reference_label_deletion_equality": (
                ctbs_reference_label_deletion_equal is True
            ),
            "lifecycle_coverage_complete": bool(
                blank_expiry_attribution_report["coverage"]["complete"]
            ),
            "unclassified_count_zero": int(
                blank_expiry_attribution_report["coverage"]["unclassified_count"]
            ) == 0,
        }
        blank_expiry_attribution_report["runtime_gates"] = runtime_gates
        blank_expiry_attribution_report["frozen_state_dict_sha256"] = (
            FROZEN_B4_STATE_DICT_SHA256
        )
        blank_expiry_attribution_report["observed_state_dict_sha256"] = actor_state_hash
        blank_expiry_attribution_report["audit_status"] = (
            "PASS" if all(runtime_gates.values()) else "FAIL"
        )
        summary["blank_expiry_attribution_status"] = (
            blank_expiry_attribution_report["audit_status"]
        )
        if blank_expiry_attribution_report["audit_status"] != "PASS":
            raise RuntimeError(
                "Blank-expiry attribution runtime Gate failed: "
                f"{runtime_gates}"
            )

    if out_dir is not None:
        (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _write_non_starvation_guard_trace(
            out_dir / "non_starvation_guard_trace.csv",
            guard_full_rows,
            guard_mode=guard_mode,
        )
        if service_accounting_report is not None:
            (out_dir / "service_accounting_audit.json").write_text(
                json.dumps(service_accounting_report, indent=2, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
        if phy_tbs_alignment_report is not None:
            (out_dir / "phy_tbs_alignment_audit.json").write_text(
                json.dumps(phy_tbs_alignment_report, indent=2, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            if phy_tbs_alignment_rows is None:
                raise RuntimeError("PHY/TBS audit rows disappeared before output")
            with (out_dir / "phy_tbs_alignment_rows.jsonl").open(
                "w", encoding="utf-8"
            ) as handle:
                for audit_row in phy_tbs_alignment_rows:
                    handle.write(
                        json.dumps(asdict(audit_row), ensure_ascii=False) + "\n"
                    )
        if blank_expiry_attribution_report is not None:
            (out_dir / "blank_expiry_attribution_audit.json").write_text(
                json.dumps(
                    blank_expiry_attribution_report,
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            for filename, output_rows in (
                ("blank_expiry_decision_rows.jsonl", blank_expiry_decision_output_rows),
                ("blank_expiry_packet_lifecycle_rows.jsonl", blank_expiry_packet_output_rows),
                ("blank_expiry_atomic_exposures.jsonl", blank_expiry_atomic_output_rows),
            ):
                if output_rows is None:
                    raise RuntimeError(f"Blank-expiry output rows disappeared: {filename}")
                with (out_dir / filename).open("w", encoding="utf-8") as handle:
                    for output_row in output_rows:
                        handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
        if int(args.save_steps) == 1:
            with open(out_dir / "eval_steps.jsonl", "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

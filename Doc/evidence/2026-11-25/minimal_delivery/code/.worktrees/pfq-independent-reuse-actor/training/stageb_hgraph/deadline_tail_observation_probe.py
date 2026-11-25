#!/usr/bin/env python3
"""Minimal offline probe for deadline-tail reward/observation alignment.

The probe replays one fixed checkpoint through the online bridge, retains a
single pre-action graph snapshot, and records the exact candidate set, masks,
logits, and PFQ label exposed at that TTI.  It is intentionally a diagnostic
spike: it does not train or modify the actor.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .bridge_env import BridgeEnvClient, BridgeEnvConfig
from .ctbs_headroom import CtbsHeadroomInputs
from .edge_generator import EdgeGeneratorConfig, SparseEdgeGenerator
from .eval_checkpoint_online import _load_actor_checkpoint
from .feature_snapshot import FeatureSnapshotBuilder, SnapshotBuilderConfig
from .graph_ops import pack_prg_candidates
from .independent_actor import PFQIndependentReuseActor
from .independent_objectives import ExpiryAwareUeWindowTracker
from .independent_policy import resolve_full_candidate_width, select_independent_action
from .phy_tbs_alignment_audit import PhyTbsConfig
from .reward_utils import extract_reward_metrics


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _upper_tail_cvar(values: np.ndarray, tail_fraction: float = 0.10) -> float:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    count = max(1, int(np.ceil(float(tail_fraction) * finite.size)))
    return float(np.sort(finite)[-count:].mean())


def _pressure(
    ue_features: torch.Tensor,
    scheduled_history: deque[np.ndarray],
    *,
    slot_duration_s: float = 0.0005,
    epsilon_bytes: float = 1.0,
) -> dict[str, Any]:
    """Compute the upper-tail CVaR of log(1+p_i) in byte/second units."""
    ue = ue_features.detach().cpu().numpy().astype(np.float64, copy=False)
    n_ue = int(ue.shape[0])
    backlog = np.maximum(ue[:, 0], 0.0)
    ttl_ms = ue[:, 9] if ue.shape[1] > 9 else np.full(n_ue, -1.0)
    active = (backlog > 0.0) & (ttl_ms >= 0.0)
    if scheduled_history:
        scheduled = np.stack(list(scheduled_history), axis=0)
        service_rate_bps = scheduled.sum(axis=0) / (len(scheduled) * float(slot_duration_s))
    else:
        service_rate_bps = np.zeros(n_ue, dtype=np.float64)
    ttl_s = np.maximum(ttl_ms, 0.0) * 1.0e-3
    pressure = backlog / (service_rate_bps * ttl_s + float(epsilon_bytes))
    log_pressure = np.log1p(np.maximum(pressure, 0.0))
    active_values = log_pressure[active]
    return {
        "active_count": int(active.sum()),
        "active_ue_ids": np.flatnonzero(active).astype(int).tolist(),
        "backlog_bytes": backlog.tolist(),
        "ttl_slack_ms": ttl_ms.tolist(),
        "scheduled_service_rate_bps": service_rate_bps.tolist(),
        "pressure_p": pressure.tolist(),
        "log_pressure": log_pressure.tolist(),
        "tail_ue_ids": np.flatnonzero(active)[
            np.argsort(active_values)[::-1][: max(1, int(np.ceil(0.10 * max(1, active_values.size))))]
        ].astype(int).tolist() if active_values.size else [],
        "phi_cvar10_upper": _upper_tail_cvar(active_values),
        "units": {
            "backlog": "bytes",
            "scheduled_service_rate": "bytes_per_second",
            "ttl_slack": "seconds internally; reported in milliseconds",
            "epsilon": float(epsilon_bytes),
        },
    }


def _target_service_diagnostics(
    *,
    target_ue_id: int,
    feature_names: tuple[str, ...] | list[str],
    ue_features: np.ndarray,
    scheduled_service_rate_bps: np.ndarray,
    active_ue_ids: list[int],
    actor_chosen_ue: np.ndarray,
    pfq_chosen_ue: np.ndarray,
    slot_duration_s: float = 0.0005,
) -> dict[str, Any]:
    """Summarize whether one UE is drainable with its recent scheduled service."""
    ue = np.asarray(ue_features, dtype=np.float64)
    service_bps = np.asarray(scheduled_service_rate_bps, dtype=np.float64).reshape(-1)
    ue_id = int(target_ue_id)
    if ue.ndim != 2 or not 0 <= ue_id < ue.shape[0] or service_bps.size != ue.shape[0]:
        raise ValueError("target UE diagnostics shape or UE id mismatch")
    names = tuple(str(name) for name in feature_names)
    if len(names) != ue.shape[1]:
        raise ValueError("feature name count does not match UE feature width")
    try:
        backlog_idx = names.index("buffer_bytes")
        ttl_idx = names.index("ttl_slack_ms")
    except ValueError as exc:
        raise ValueError("target diagnostics require buffer_bytes and ttl_slack_ms") from exc

    backlog = np.maximum(ue[:, backlog_idx], 0.0)
    ttl_ms = ue[:, ttl_idx]
    slot_ms = float(slot_duration_s) * 1.0e3
    if slot_ms <= 0.0:
        raise ValueError("slot_duration_s must be positive")
    ttl_steps = np.maximum(ttl_ms / slot_ms, 1.0)
    required = np.where(ttl_ms >= 0.0, backlog / ttl_steps, 0.0)
    recent = np.maximum(service_bps, 0.0) * float(slot_duration_s)
    deficit = np.maximum(required - recent, 0.0)
    active = np.asarray(
        [idx for idx in active_ue_ids if 0 <= int(idx) < ue.shape[0]],
        dtype=np.int64,
    )
    active_deficit = deficit[active] if active.size else np.asarray([], dtype=np.float64)
    target_deficit = float(deficit[ue_id])
    rank = 1 + int(np.sum(active_deficit > target_deficit)) if ue_id in set(active.tolist()) else None

    return {
        "ue_id": ue_id,
        "features": {name: float(ue[ue_id, idx]) for idx, name in enumerate(names)},
        "required_drain_bytes_per_tti": float(required[ue_id]),
        "recent_scheduled_bytes_per_tti": float(recent[ue_id]),
        "drain_deficit_bytes_per_tti": target_deficit,
        "drain_deficit_rank_desc": rank,
        "active_ue_count": int(active.size),
        "actor_prg_count": int(np.sum(np.asarray(actor_chosen_ue) == ue_id)),
        "pfq_prg_count": int(np.sum(np.asarray(pfq_chosen_ue) == ue_id)),
    }


def _parse_target_ue_ids(value: str, n_ue: int) -> list[int]:
    target_ids = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        ue_id = int(token)
        if not 0 <= ue_id < int(n_ue):
            raise ValueError(f"target UE id {ue_id} is outside [0, {int(n_ue)})")
        if ue_id not in target_ids:
            target_ids.append(ue_id)
    if not target_ids:
        raise ValueError("at least one target UE id is required")
    return target_ids


def _ctbs_inputs(env: BridgeEnvClient, info: Mapping[str, Any], config: PhyTbsConfig) -> CtbsHeadroomInputs:
    pre = env.request_pre_action_snapshot(expected_tti=int(info["tti"]))
    return CtbsHeadroomInputs.from_arrays(
        tti=pre.tti,
        snapshot_id=pre.snapshot_id,
        config_sha256=pre.config_sha256,
        config=config,
        ue_ids=pre.ue_ids,
        serving_cell=pre.serving_cell,
        wb_sinr_first_layer_db=pre.wb_sinr_first_layer_db,
        olla_delta=pre.olla_delta,
        olla_delta_up=pre.olla_delta_up,
        olla_delta_down=pre.olla_delta_down,
        prior_tb_err=pre.prior_tb_err,
        prg_layer_cap=pre.prg_layer_cap,
    )


def _pfq_candidate_labels(graph, packed) -> dict[str, Any]:
    meta = graph.snapshot.static_meta
    teacher_ue = meta.teacher_action_ue
    teacher_prg = meta.teacher_action_prg
    if teacher_ue is None or teacher_prg is None:
        raise RuntimeError("PFQ teacher label was not exported by the bridge")
    teacher_ue = teacher_ue.to(dtype=torch.long).detach().cpu()
    teacher_prg = teacher_prg.to(dtype=torch.long).detach().cpu()
    p, c, k = int(meta.n_prg), int(meta.n_cell), int(packed.max_candidates)
    candidate_ids = packed.ue_ids.reshape(c, p, k).permute(1, 0, 2).cpu()
    candidate_mask = packed.mask.reshape(c, p, k).permute(1, 0, 2).cpu()
    labels = torch.full((p, c), int(packed.blank_class_index), dtype=torch.long)
    label_ue = torch.full((p, c), -1, dtype=torch.long)
    for local_prg in range(p):
        for cell in range(c):
            slot = int(teacher_prg[local_prg * c + cell].item())
            if slot < 0:
                continue
            ue = int(teacher_ue[slot].item())
            matches = torch.nonzero(candidate_mask[local_prg, cell] & (candidate_ids[local_prg, cell] == ue), as_tuple=False).flatten()
            if matches.numel() != 1:
                raise RuntimeError(f"PFQ label UE {ue} is not uniquely represented in candidates at cell={cell} prg={local_prg}")
            labels[local_prg, cell] = int(matches.item())
            label_ue[local_prg, cell] = ue
    return {
        "teacher_action_ue_select": teacher_ue,
        "teacher_action_prg_alloc": teacher_prg,
        "candidate_class_by_local_prg_cell": labels,
        "candidate_ue_by_local_prg_cell": label_ue,
    }


def _run_logits(
    actor: PFQIndependentReuseActor,
    graph,
    packed,
    ctbs: CtbsHeadroomInputs | None = None,
    *,
    capture: bool = False,
) -> dict[str, Any]:
    with torch.inference_mode():
        trace, decoded = select_independent_action(
            actor,
            graph,
            packed,
            mode="greedy",
            collect_diagnostics=capture,
            collect_numerical_trace=capture,
            ctbs_headroom_inputs=ctbs,
        )
    if capture and trace.raw_logits_at_step is None:
        raise RuntimeError("independent actor did not return numerical logits")
    return {"trace": trace, "decoded": decoded}


def _candidate_contrast(actor, graph, packed, trace, *, ue_id: int) -> dict[str, Any]:
    logits = trace.raw_logits_at_step.detach().cpu()
    valid = trace.valid_at_step.detach().cpu()
    ids = packed.ue_ids.reshape(graph.snapshot.n_cell, graph.snapshot.n_prg, packed.max_candidates).permute(1, 0, 2).cpu()
    mask = packed.mask.reshape(graph.snapshot.n_cell, graph.snapshot.n_prg, packed.max_candidates).permute(1, 0, 2).cpu()
    rows = []
    for prg in range(int(graph.snapshot.n_prg)):
        for cell in range(int(graph.snapshot.n_cell)):
            hit = torch.nonzero(mask[prg, cell] & (ids[prg, cell] == int(ue_id)), as_tuple=False).flatten()
            if hit.numel() != 1:
                continue
            idx = int(hit.item())
            valid_nonblank = valid[prg, cell, : packed.max_candidates]
            values = logits[prg, cell, : packed.max_candidates][valid_nonblank]
            other = values[ids[prg, cell][valid_nonblank] != int(ue_id)]
            blank = float(logits[prg, cell, packed.blank_class_index].item())
            ue_logit = float(logits[prg, cell, idx].item())
            rows.append({
                "local_prg": prg,
                "cell": cell,
                "ue2_logit": ue_logit,
                "blank_logit": blank,
                "ue2_minus_blank": ue_logit - blank,
                "ue2_minus_best_other": ue_logit - float(other.max().item()) if other.numel() else None,
                "best_valid_nonblank": float(values.max().item()) if values.numel() else None,
            })
    margins = np.asarray([r["ue2_minus_best_other"] for r in rows if r["ue2_minus_best_other"] is not None], dtype=np.float64)
    all_valid = logits[valid]
    return {
        "ue_id": int(ue_id),
        "candidate_occurrences": len(rows),
        "rows": rows,
        "valid_logit_std": float(all_valid.std().item()) if all_valid.numel() > 1 else 0.0,
        "ue2_minus_best_other_mean": float(margins.mean()) if margins.size else None,
        "ue2_minus_best_other_max": float(margins.max()) if margins.size else None,
    }


def _perturb_graph_q(graph, ue_id: int, scale: float):
    features = graph.snapshot.ue_features.values.clone()
    features[ue_id, 0] = features[ue_id, 0] * float(scale)
    return replace(graph, snapshot=replace(graph.snapshot, ue_features=replace(graph.snapshot.ue_features, values=features)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sim-bin", required=True)
    p.add_argument("--socket-path", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episode-horizon", type=int, default=5000)
    p.add_argument("--target-tti", type=int, default=5000)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    p.add_argument("--runtime-candidate-width", type=int, default=14)
    p.add_argument("--service-window", type=int, default=100)
    p.add_argument("--target-ue-ids", default="2")
    p.add_argument("--rate-tail-fraction", type=float, default=0.10)
    p.add_argument("--rate-tail-weight", type=float, default=0.25)
    p.add_argument("--rate-tail-epsilon-mbps", type=float, default=1.0e-6)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--sim-cwd", default=".")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # This replay is dominated by many tiny CPU graph tensor operations; large
    # OpenMP pools add substantial per-TTI overhead without changing results.
    torch.set_num_threads(1)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    actor, checkpoint = _load_actor_checkpoint(Path(args.checkpoint).resolve(), device)
    if not isinstance(actor, PFQIndependentReuseActor):
        raise RuntimeError("deadline-tail probe requires PFQIndependentReuseActor")
    edge_generator = SparseEdgeGenerator(EdgeGeneratorConfig(
        ue_prg_topk=4,
        ue_prg_diversity_extra=0,
        candidate_mode="all_demand_ues",
        include_pfq_score_anchor=False,
        include_candidate_risk_features=False,
    ))
    snapshot_builder = FeatureSnapshotBuilder(SnapshotBuilderConfig(tb_err_ema_alpha=0.10))
    env = BridgeEnvClient(BridgeEnvConfig(socket_path=str(args.socket_path), enable_ctbs=True))
    obs, _reward, info = env.reset(seed=int(args.seed), episode_horizon=int(args.episode_horizon))
    if int(info.get("harq_enabled", -1)) != 0:
        raise RuntimeError("CTBS probe requires HARQ disabled")
    config = PhyTbsConfig.from_vector(info["ctbs_config"])
    history: deque[np.ndarray] = deque(maxlen=max(1, int(args.service_window)))
    rate_tracker: ExpiryAwareUeWindowTracker | None = None
    snapshot_record: dict[str, Any] | None = None
    transition_record: dict[str, Any] | None = None
    try:
        while True:
            graph = edge_generator.build_graph(snapshot_builder.build(obs))
            if rate_tracker is None:
                rate_tracker = ExpiryAwareUeWindowTracker(
                    int(graph.snapshot.static_meta.n_ue),
                    window_steps=max(1, int(args.service_window)),
                )
            ctbs = _ctbs_inputs(env, info, config)
            width = max(int(args.runtime_candidate_width), resolve_full_candidate_width(graph))
            packed = pack_prg_candidates(graph, max_candidates=width, fixed_width=True)
            is_target = int(info.get("tti", -1)) == int(args.target_tti)
            if is_target:
                selected = _run_logits(actor, graph, packed, ctbs, capture=True)
                trace = selected["trace"]
                decoded = selected["decoded"]
                phi_before = _pressure(graph.snapshot.ue_features.values, history)
                demand_before = (
                    graph.snapshot.ue_features.values[:, 0].detach().cpu().numpy().reshape(-1) > 0.0
                )
                rate_before = rate_tracker.rate_tail_utility(
                    demand_before,
                    tail_fraction=float(args.rate_tail_fraction),
                    epsilon_mbps=float(args.rate_tail_epsilon_mbps),
                )
                pfq = _pfq_candidate_labels(graph, packed)
                target_ue_ids = _parse_target_ue_ids(
                    args.target_ue_ids,
                    int(graph.snapshot.n_ue),
                )
                chosen_actor = decoded.chosen_ue_per_prg.detach().cpu().numpy()
                chosen_pfq = pfq["candidate_ue_by_local_prg_cell"].detach().cpu().numpy()
                ue_features_np = graph.snapshot.ue_features.values.detach().cpu().numpy()
                target_diagnostics = {
                    str(ue_id): _target_service_diagnostics(
                        target_ue_id=ue_id,
                        feature_names=graph.snapshot.ue_features.names,
                        ue_features=ue_features_np,
                        scheduled_service_rate_bps=np.asarray(
                            phi_before["scheduled_service_rate_bps"], dtype=np.float64
                        ),
                        active_ue_ids=phi_before["active_ue_ids"],
                        actor_chosen_ue=chosen_actor,
                        pfq_chosen_ue=chosen_pfq,
                    )
                    for ue_id in target_ue_ids
                }
                snapshot_record = {
                    "tti": int(info["tti"]),
                    "seed": int(args.seed),
                    "device": str(device),
                    "actor_family": actor.actor_family,
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "checkpoint_actor_config": checkpoint.get("actor_config", {}),
                    "observation": {
                        "cell_features": graph.snapshot.cell_features,
                        "ue_features": graph.snapshot.ue_features,
                        "prg_features": graph.snapshot.prg_features,
                        "action_mask_ue": graph.snapshot.static_meta.action_mask_ue,
                        "action_mask_cell_ue": graph.snapshot.static_meta.action_mask_cell_ue,
                        "action_mask_prg_cell": graph.snapshot.static_meta.action_mask_prg_cell,
                        "cell_adjacency": graph.snapshot.static_meta.cell_adjacency,
                        "cell_edge_attr": graph.snapshot.static_meta.cell_edge_attr,
                        "ue_serving_cell": graph.snapshot.static_meta.ue_serving_cell,
                        "post_eq_sinr": graph.snapshot.static_meta.post_eq_sinr,
                    },
                    "candidates": {
                        "max_candidates": int(packed.max_candidates),
                        "ue_ids": packed.ue_ids,
                        "mask": packed.mask,
                        "counts": packed.counts,
                        "edge_attr": packed.edge_attr,
                    },
                    "policy": {
                        "valid_at_step": trace.valid_at_step,
                        "raw_logits_at_step": trace.raw_logits_at_step,
                        "chosen_class": trace.choice,
                        "chosen_logprob_at_step": trace.chosen_logprob_at_step,
                        "chosen_ue_per_prg": decoded.chosen_ue_per_prg,
                        "action_ue_select": decoded.action_ue_select,
                        "action_prg_alloc": decoded.action_prg_alloc,
                    },
                    "pfq_label": pfq,
                    "pressure_before": phi_before,
                    "target_ue_ids": target_ue_ids,
                    "target_service_diagnostics": target_diagnostics,
                    "target_logit_contrasts": {
                        str(ue_id): _candidate_contrast(
                            actor, graph, packed, trace, ue_id=ue_id
                        )
                        for ue_id in target_ue_ids
                    },
                }
                all_perturbations = {}
                for ue_id in target_ue_ids:
                    perturbations = {}
                    for scale in (0.01, 1.0, 50.0):
                        perturbed = _perturb_graph_q(graph, ue_id, scale)
                        perturbed_result = _run_logits(actor, perturbed, packed, ctbs, capture=True)
                        raw = perturbed_result["trace"].raw_logits_at_step
                        perturbations[str(scale)] = {
                            "q_bytes": float(perturbed.snapshot.ue_features.values[ue_id, 0].item()),
                            "chosen_class": perturbed_result["trace"].choice,
                            "raw_logits_at_step": raw,
                        }
                    all_perturbations[str(ue_id)] = perturbations
                snapshot_record["target_backlog_perturbations"] = all_perturbations

            if not is_target:
                selected = _run_logits(actor, graph, packed, ctbs)
            next_obs, _raw_reward, done, next_info = env.step(
                selected["decoded"].action_ue_select.detach().cpu().numpy(),
                selected["decoded"].action_prg_alloc.detach().cpu().numpy(),
            )
            if "actual_ue_served_bytes" not in next_info:
                raise RuntimeError(
                    "deadline-tail probe requires actual_ue_served_bytes diagnostics; "
                    "refusing to substitute zeros for scheduled dequeue"
                )
            scheduled = np.asarray(next_info["actual_ue_served_bytes"], dtype=np.float64).reshape(-1)
            if scheduled.size != int(graph.snapshot.static_meta.n_ue):
                raise RuntimeError(
                    "actual_ue_served_bytes size mismatch: "
                    f"got={scheduled.size} expected={graph.snapshot.static_meta.n_ue}"
                )
            history.append(np.maximum(scheduled, 0.0))
            actual_expiry_aware = bool(
                next_info.get("actual_ue_packet_expiry_aware_diagnostics_available", False)
            )
            if not actual_expiry_aware:
                raise RuntimeError(
                    "deadline-tail probe requires expiry-aware UE packet diagnostics"
                )
            next_graph = edge_generator.build_graph(snapshot_builder.build(next_obs))
            next_reward_metrics = extract_reward_metrics(next_info)
            demand_after = (
                next_graph.snapshot.ue_features.values[:, 0].detach().cpu().numpy().reshape(-1) > 0.0
            )
            rate_tracker.update(
                demand_after,
                np.asarray(next_info["actual_ue_goodput_bytes"], dtype=np.float64),
                np.asarray(next_info["actual_ue_packet_system_time_ms"], dtype=np.float64),
                np.asarray(next_info["actual_ue_packet_expired_system_time_ms"], dtype=np.float64),
            )
            if is_target:
                rate_after = rate_tracker.rate_tail_utility(
                    demand_after,
                    tail_fraction=float(args.rate_tail_fraction),
                    epsilon_mbps=float(args.rate_tail_epsilon_mbps),
                )
                transition_record = {
                    "next_tti": int(next_info.get("tti", -1)),
                    "pressure_after": _pressure(next_graph.snapshot.ue_features.values, history),
                    "scheduled_bytes_this_tti": scheduled,
                    "actual_ue_goodput_bytes": np.asarray(
                        next_info["actual_ue_goodput_bytes"], dtype=np.float64
                    ),
                    "actual_ue_packet_system_time_ms": np.asarray(
                        next_info["actual_ue_packet_system_time_ms"], dtype=np.float64
                    ),
                    "actual_ue_packet_expired_system_time_ms": np.asarray(
                        next_info["actual_ue_packet_expired_system_time_ms"], dtype=np.float64
                    ),
                    "actual_ue_packet_expiry_aware_diagnostics_available": actual_expiry_aware,
                    "expired_bytes": float(next_reward_metrics.get("expired_bytes", 0.0)),
                    "expiry_drop_rate": float(next_reward_metrics.get("expiry_drop_rate", 0.0)),
                    "rate_tail_before": float(rate_before),
                    "rate_tail_after": float(rate_after),
                    "rate_tail_delta": float(float(args.gamma) * rate_after - rate_before),
                    "rate_tail_weighted": float(
                        args.rate_tail_weight * (float(args.gamma) * rate_after - rate_before)
                    ),
                }
                break
            obs, info = next_obs, next_info
            if done or int(info.get("tti", -1)) > int(args.target_tti):
                raise RuntimeError(f"environment ended before target TTI {args.target_tti}")
    finally:
        env.close()
    if snapshot_record is None or transition_record is None:
        raise RuntimeError("target snapshot was not captured")
    snapshot_record["transition"] = transition_record
    target_summary = {}
    for ue_id in snapshot_record["target_ue_ids"]:
        key = str(ue_id)
        item = dict(snapshot_record["target_service_diagnostics"][key])
        contrast = snapshot_record["target_logit_contrasts"][key]
        item.update({
            "in_phi_tail_before": bool(ue_id in snapshot_record["pressure_before"]["tail_ue_ids"]),
            "in_phi_tail_after": bool(ue_id in transition_record["pressure_after"]["tail_ue_ids"]),
            "logit_margin_vs_best_other_mean": contrast["ue2_minus_best_other_mean"],
            "actor_underallocates_vs_pfq": bool(
                int(item["actor_prg_count"]) < int(item["pfq_prg_count"])
            ),
        })
        target_summary[key] = item
    snapshot_record["summary"] = {
        "phi_before": snapshot_record["pressure_before"]["phi_cvar10_upper"],
        "phi_after": transition_record["pressure_after"]["phi_cvar10_upper"],
        "phi_delta_before_minus_after": snapshot_record["pressure_before"]["phi_cvar10_upper"] - transition_record["pressure_after"]["phi_cvar10_upper"],
        "targets": target_summary,
        "rate_tail_before": float(transition_record["rate_tail_before"]),
        "rate_tail_after": float(transition_record["rate_tail_after"]),
        "rate_tail_delta": float(transition_record["rate_tail_delta"]),
        "rate_tail_weighted": float(transition_record["rate_tail_weighted"]),
        "expired_bytes": float(transition_record["expired_bytes"]),
        "expiry_drop_rate": float(transition_record["expiry_drop_rate"]),
        "expiry_aware_diagnostics_available": bool(
            transition_record["actual_ue_packet_expiry_aware_diagnostics_available"]
        ),
        "directional_checks": {
            "all_targets_have_positive_drain_deficit": all(
                float(item["drain_deficit_bytes_per_tti"]) > 0.0
                for item in target_summary.values()
            ),
            "all_targets_identified_in_phi_tail": all(
                bool(item["in_phi_tail_before"]) for item in target_summary.values()
            ),
            "all_targets_underallocated_vs_pfq": all(
                bool(item["actor_underallocates_vs_pfq"])
                for item in target_summary.values()
            ),
            "phi_penalizes_actor_transition": bool(
                float(
                    snapshot_record["pressure_before"]["phi_cvar10_upper"]
                    - transition_record["pressure_after"]["phi_cvar10_upper"]
                ) <= 0.0
            ),
            "rate_tail_penalizes_or_does_not_credit_actor": bool(
                float(transition_record["rate_tail_delta"]) <= 0.0
            ),
            "expiry_diagnostics_available": bool(
                transition_record["actual_ue_packet_expiry_aware_diagnostics_available"]
            ),
        },
        "obs_ue_feature_dim": int(len(snapshot_record["observation"]["ue_features"].values[0])),
        "obs_ue_feature_names": list(snapshot_record["observation"]["ue_features"].names),
        "checkpoint_config_ue_feat_dim": int(getattr(actor.cfg, "ue_feat_dim", -1)),
        "scheduled_rate_source": "rolling prior actual_ue_served_bytes from bridge, interpreted as scheduled dequeue bytes",
        "target_tti": int(args.target_tti),
    }
    (out_dir / "snapshot.json").write_text(json.dumps(_jsonable(snapshot_record), indent=2) + "\n", encoding="utf-8")
    (out_dir / "probe_report.json").write_text(json.dumps(_jsonable({"summary": snapshot_record["summary"], "pressure_before": snapshot_record["pressure_before"], "pressure_after": transition_record["pressure_after"], "target_logit_contrasts": snapshot_record["target_logit_contrasts"]}), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable(snapshot_record["summary"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

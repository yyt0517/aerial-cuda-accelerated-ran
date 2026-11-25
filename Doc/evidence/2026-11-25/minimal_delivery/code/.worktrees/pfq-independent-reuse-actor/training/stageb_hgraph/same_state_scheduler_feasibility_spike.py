#!/usr/bin/env python3
"""Throwaway seed42 same-state PFQ/DPA feasibility spike.

The simulator follows the native PFQ teacher trajectory.  At every recorded
pre-action state the frozen DPA actor is evaluated read-only, then PFQ alone is
committed.  The spike intentionally distinguishes scheduler-independent
pre-action post-EQ quality and CTBS/TBS service proxies from unavailable
action-conditioned counterfactual PHY outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .bridge_env import BridgeEnvClient, BridgeEnvConfig
from .ctbs_headroom import CtbsHeadroomInputs, CtbsHeadroomState
from .deadline_tail_observation_probe import _ctbs_inputs, _run_logits
from .edge_generator import EdgeGeneratorConfig, SparseEdgeGenerator
from .eval_checkpoint_online import _load_actor_checkpoint
from .feature_snapshot import (
    FeatureSnapshotBuilder,
    SnapshotBuilderConfig,
    actor_visible_snapshot_tensors,
)
from .graph_ops import pack_prg_candidates
from .independent_actor import PFQIndependentReuseActor
from .independent_policy import resolve_full_candidate_width
from .pfq_reference_calibration import _extend_ld_library_path
from .phy_tbs_alignment_audit import PhyTbsConfig
from .same_state_four_action_attribution import (
    action_structure_metrics,
    build_four_actions,
    counterfactual_deadline_pressure,
    deadline_state_by_ue,
    infer_recent_scheduled_dequeue,
    select_representative_tti,
    shapley_two_patch_contributions,
)


def decode_type0_teacher_action(
    teacher_action_ue: Sequence[int],
    teacher_action_prg: Sequence[int],
    *,
    n_cell: int,
    n_prg: int,
) -> np.ndarray:
    """Decode native Type-0 labels into actor token order [cell, local PRG]."""
    ue_slots = np.asarray(teacher_action_ue, dtype=np.int64).reshape(-1)
    allocation = np.asarray(teacher_action_prg, dtype=np.int64).reshape(-1)
    if allocation.size != int(n_cell) * int(n_prg):
        raise ValueError("teacher Type-0 allocation dimensions disagree")
    chosen = np.full(int(n_cell) * int(n_prg), -1, dtype=np.int64)
    for local_prg in range(int(n_prg)):
        for cell in range(int(n_cell)):
            slot = int(allocation[local_prg * int(n_cell) + cell])
            if slot < 0:
                continue
            if slot >= ue_slots.size:
                raise ValueError("teacher Type-0 slot is out of range")
            ue = int(ue_slots[slot])
            if ue < 0:
                raise ValueError("teacher Type-0 slot resolves to blank UE")
            chosen[cell * int(n_prg) + local_prg] = ue
    return chosen


def _selected_sinr_db(
    chosen_ue_per_prg: np.ndarray,
    post_eq_sinr: np.ndarray,
    *,
    n_prg: int,
) -> list[float]:
    quality = np.asarray(post_eq_sinr, dtype=np.float64)
    if quality.ndim == 2:
        quality = quality[:, :, None]
    if quality.ndim != 3 or quality.shape[1] != int(n_prg):
        raise ValueError("post-EQ SINR must have shape [UE, PRG, layer]")
    values: list[float] = []
    for token, ue_value in enumerate(np.asarray(chosen_ue_per_prg).reshape(-1)):
        ue = int(ue_value)
        if ue < 0:
            continue
        if ue >= quality.shape[0]:
            raise ValueError("chosen UE is out of post-EQ SINR range")
        local_prg = token % int(n_prg)
        sinr_linear = float(np.maximum(quality[ue, local_prg], 1.0e-9).mean())
        values.append(float(10.0 * np.log10(sinr_linear)))
    return values


def summarize_same_state_actions(
    pfq_chosen_ue_per_prg: Sequence[int],
    ai_chosen_ue_per_prg: Sequence[int],
    *,
    post_eq_sinr: np.ndarray,
    n_cell: int,
    n_prg: int,
) -> dict[str, Any]:
    pfq = np.asarray(pfq_chosen_ue_per_prg, dtype=np.int64).reshape(-1)
    ai = np.asarray(ai_chosen_ue_per_prg, dtype=np.int64).reshape(-1)
    expected = int(n_cell) * int(n_prg)
    if pfq.size != expected or ai.size != expected:
        raise ValueError("same-state action dimensions disagree")
    changed = int(np.count_nonzero(pfq != ai))
    pfq_active = int(np.count_nonzero(pfq >= 0))
    ai_active = int(np.count_nonzero(ai >= 0))
    return {
        "changed_cell_prg_count": changed,
        "action_change_ratio": float(changed / expected),
        "pfq_scheduled_cell_prg_count": pfq_active,
        "ai_scheduled_cell_prg_count": ai_active,
        "scheduled_cell_prg_delta_ai_minus_pfq": int(ai_active - pfq_active),
        "pfq_selected_sinr_db": _selected_sinr_db(
            pfq, post_eq_sinr, n_prg=int(n_prg)
        ),
        "ai_selected_sinr_db": _selected_sinr_db(
            ai, post_eq_sinr, n_prg=int(n_prg)
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_sha256(graph) -> str:
    digest = hashlib.sha256()
    tensors = actor_visible_snapshot_tensors(graph.snapshot)
    meta = graph.snapshot.static_meta
    if meta.teacher_action_ue is not None:
        tensors["teacher_action_ue"] = meta.teacher_action_ue
    if meta.teacher_action_prg is not None:
        tensors["teacher_action_prg"] = meta.teacher_action_prg
    for name in sorted(tensors):
        array = tensors[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _ctbs_action_capacity(
    inputs: CtbsHeadroomInputs,
    chosen_ue_per_prg: Sequence[int],
    backlog_bytes: Sequence[float],
    *,
    n_cell: int,
    n_prg: int,
) -> dict[str, Any]:
    chosen = np.asarray(chosen_ue_per_prg, dtype=np.int64).reshape(-1)
    backlog = np.maximum(np.asarray(backlog_bytes, dtype=np.float64).reshape(-1), 0.0)
    n_ue = len(inputs.ue_ids)
    if chosen.size != int(n_cell) * int(n_prg) or backlog.size != n_ue:
        raise ValueError("CTBS action dimensions disagree")
    selected: list[list[int]] = [[] for _ in range(n_ue)]
    for token, ue_value in enumerate(chosen):
        ue = int(ue_value)
        if ue < 0:
            continue
        if ue >= n_ue:
            raise ValueError("CTBS chosen UE is out of range")
        cell = token // int(n_prg)
        local_prg = token % int(n_prg)
        if inputs.serving_cell[ue] != cell:
            raise ValueError("CTBS chosen UE belongs to a different cell")
        selected[ue].append(local_prg)

    capacity = np.zeros(n_ue, dtype=np.int64)
    reconstructed_mcs = np.full(n_ue, -1, dtype=np.int64)
    reconstructed_layer = np.zeros(n_ue, dtype=np.int64)
    for ue, local_prgs in enumerate(selected):
        if not local_prgs:
            continue
        state = CtbsHeadroomState(
            initial_demand_bytes=float(backlog[ue]),
            reconstructed_mcs=inputs.reconstructed_mcs_for_ue(ue),
            config=inputs.config,
        )
        for local_prg in local_prgs:
            state = state.with_selected_prg(
                local_prg,
                layer_cap=inputs.prg_layer_cap[ue][local_prg],
            )
        capacity[ue] = int(state.cumulative_tbs_bytes)
        reconstructed_mcs[ue] = int(state.reconstructed_mcs)
        reconstructed_layer[ue] = int(min(state.selected_layer_caps))
    demand_capped = np.minimum(capacity.astype(np.float64), backlog)
    return {
        "capacity_bytes_by_ue": capacity.tolist(),
        "demand_capped_service_bytes_by_ue": demand_capped.tolist(),
        "reconstructed_mcs_by_ue": reconstructed_mcs.tolist(),
        "reconstructed_layer_by_ue": reconstructed_layer.tolist(),
        "capacity_bytes_total": int(capacity.sum()),
        "demand_capped_service_bytes_total": float(demand_capped.sum()),
    }


def _actual_chosen_from_mask(
    mask: np.ndarray,
    serving_cell: Sequence[int],
    *,
    n_cell: int,
    n_prg: int,
) -> np.ndarray:
    values = np.asarray(mask, dtype=np.int64)
    if values.shape != (len(serving_cell), int(n_prg)):
        raise ValueError("actual UE-PRG mask dimensions disagree")
    chosen = np.full(int(n_cell) * int(n_prg), -1, dtype=np.int64)
    for ue, cell_value in enumerate(serving_cell):
        cell = int(cell_value)
        for local_prg in np.flatnonzero(values[ue]):
            token = cell * int(n_prg) + int(local_prg)
            if chosen[token] >= 0:
                raise ValueError("actual mask schedules multiple UEs on one Cell-PRG")
            chosen[token] = int(ue)
    return chosen


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    return float(np.percentile(array, percentile)) if array.size else None


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def vector(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows], dtype=np.float64)

    pfq_sinr = [value for row in rows for value in row["pfq_selected_sinr_db"]]
    ai_sinr = [value for row in rows for value in row["ai_selected_sinr_db"]]
    changed = vector("action_change_ratio")
    prg_delta = vector("scheduled_cell_prg_delta_ai_minus_pfq")
    return {
        "recorded_tti_count": len(rows),
        "recorded_tti_first": int(rows[0]["tti"]),
        "recorded_tti_last": int(rows[-1]["tti"]),
        "action_change_ratio_mean": float(changed.mean()),
        "action_change_ratio_median": float(np.median(changed)),
        "action_change_ratio_p90": float(np.percentile(changed, 90)),
        "scheduled_cell_prg_delta_ai_minus_pfq_mean": float(prg_delta.mean()),
        "scheduled_cell_prg_delta_ai_minus_pfq_median": float(np.median(prg_delta)),
        "ai_uses_no_more_cell_prgs_tti_count": int(np.count_nonzero(prg_delta <= 0)),
        "pfq_selected_pre_action_sinr_db_mean": float(np.mean(pfq_sinr)),
        "pfq_selected_pre_action_sinr_db_p10": _percentile(pfq_sinr, 10),
        "pfq_selected_pre_action_sinr_db_p5": _percentile(pfq_sinr, 5),
        "ai_selected_pre_action_sinr_db_mean": float(np.mean(ai_sinr)),
        "ai_selected_pre_action_sinr_db_p10": _percentile(ai_sinr, 10),
        "ai_selected_pre_action_sinr_db_p5": _percentile(ai_sinr, 5),
        "pfq_ctbs_capacity_bytes_per_tti_mean": float(
            vector("pfq_ctbs_capacity_bytes_total").mean()
        ),
        "ai_ctbs_capacity_bytes_per_tti_mean": float(
            vector("ai_ctbs_capacity_bytes_total").mean()
        ),
        "pfq_demand_capped_service_bytes_per_tti_mean": float(
            vector("pfq_demand_capped_service_bytes_total").mean()
        ),
        "ai_demand_capped_service_bytes_per_tti_mean": float(
            vector("ai_demand_capped_service_bytes_total").mean()
        ),
        "state_hash_mismatch_count": int(
            sum(not bool(row["shadow_inference_state_immutable"]) for row in rows)
        ),
        "pfq_action_acceptance_mismatch_count": int(
            sum(int(row["pfq_action_acceptance_mismatch_count"]) for row in rows)
        ),
        "pfq_ctbs_reference_mismatch_count": int(
            sum(int(row["pfq_ctbs_reference_mismatch_count"]) for row in rows)
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sim-bin", required=True)
    parser.add_argument("--sim-args", default="-d 1 -b 2 -f 0 -x 0 -g 100 -r 3000")
    parser.add_argument("--sim-cwd", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--record-steps", type=int, default=100)
    parser.add_argument("--runtime-candidate-width", type=int, default=14)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if int(args.warmup_steps) < 0 or not 50 <= int(args.record_steps) <= 100:
        raise ValueError("spike requires non-negative warmup and 50--100 recorded TTIs")
    torch.set_num_threads(1)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_bin = Path(args.sim_bin).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    socket_path = str(Path(tempfile.gettempdir()) / f"cumac_same_state_spike_{os.getpid()}.sock")
    sim_log = out_dir / "sim.log"
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    actor, _checkpoint = _load_actor_checkpoint(checkpoint_path, device)
    if not isinstance(actor, PFQIndependentReuseActor):
        raise RuntimeError("same-state spike requires PFQIndependentReuseActor")
    edge_generator = SparseEdgeGenerator(
        EdgeGeneratorConfig(
            ue_prg_topk=4,
            ue_prg_diversity_extra=0,
            candidate_mode="all_demand_ues",
            include_pfq_score_anchor=False,
            include_candidate_risk_features=False,
        )
    )
    snapshot_builder = FeatureSnapshotBuilder(SnapshotBuilderConfig(tb_err_ema_alpha=0.10))
    total_steps = int(args.warmup_steps) + int(args.record_steps)

    env_vars = os.environ.copy()
    env_vars.update(
        {
            "CUMAC_ONLINE_BRIDGE": "1",
            "CUMAC_ONLINE_SOCKET": socket_path,
            "CUMAC_ONLINE_PERSISTENT": "1",
            "CUMAC_ONLINE_EXPORT_TEACHER_ACTION": "1",
            "CUMAC_ONLINE_EXPORT_POST_EQ_SINR": "1",
            "CUMAC_ONLINE_EXPORT_PHY_TBS_AUDIT": "1",
            "CUMAC_TOPOLOGY_SEED": str(int(args.seed)),
            "CUMAC_TRAFFIC_SEED": str(int(args.seed)),
            "CUMAC_CELL_ASSOC_MODE": "strongest",
            "CUMAC_UE_PLACEMENT_MODE": "coop_boundary",
            "CUMAC_UE_VORONOI_CLIP": "1",
            "CUMAC_BS_TX_PATTERN": "omni",
            "CUMAC_PACKET_TTL_MS": "200",
            "CUMAC_PACKET_TTL_TTI": "0",
            "CUMAC_TRAFFIC_ARRIVAL_RATE": "1.0",
            "CUMAC_COMPACT_TTI_LOG": "1",
            "CUMAC_PROGRESS_TTI_INTERVAL": "0",
            "CUMAC_COMPARE_TTI_INTERVAL": "0",
            "CUMAC_EXEC_MODE": "both",
            "CUMAC_ONLINE_REWARD_MODE": "goodput_only",
        }
    )
    _extend_ld_library_path(env_vars, sim_bin)
    command = [str(sim_bin), *shlex.split(str(args.sim_args))]
    rows: list[dict[str, Any]] = []
    client: BridgeEnvClient | None = None
    with sim_log.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            command,
            cwd=str(Path(args.sim_cwd).resolve()) if args.sim_cwd else str(repo_root),
            env=env_vars,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        try:
            client = BridgeEnvClient(
                BridgeEnvConfig(socket_path=socket_path, connect_timeout_s=60.0, enable_ctbs=True)
            )
            obs, _reward, info = client.reset(seed=int(args.seed), episode_horizon=total_steps)
            if int(info.get("harq_enabled", -1)) != 0:
                raise RuntimeError("same-state spike requires HARQ disabled")
            config = PhyTbsConfig.from_vector(info["ctbs_config"])
            for step_idx in range(total_steps):
                graph = edge_generator.build_graph(snapshot_builder.build(obs))
                meta = graph.snapshot.static_meta
                if meta.teacher_action_ue is None or meta.teacher_action_prg is None:
                    raise RuntimeError("PFQ teacher action is unavailable")
                if meta.post_eq_sinr is None:
                    raise RuntimeError("pre-action post-EQ SINR is unavailable")
                ctbs = _ctbs_inputs(client, info, config)
                width = max(
                    int(args.runtime_candidate_width), resolve_full_candidate_width(graph)
                )
                packed = pack_prg_candidates(graph, max_candidates=width, fixed_width=True)
                state_hash_before = _snapshot_sha256(graph)
                selected = _run_logits(actor, graph, packed, ctbs)
                state_hash_after = _snapshot_sha256(graph)

                pfq_action_ue = meta.teacher_action_ue.detach().cpu().numpy().astype(np.int32)
                pfq_action_prg = meta.teacher_action_prg.detach().cpu().numpy().astype(np.int16)
                pfq_chosen = decode_type0_teacher_action(
                    pfq_action_ue,
                    pfq_action_prg,
                    n_cell=int(meta.n_cell),
                    n_prg=int(meta.n_prg),
                )
                ai_chosen = (
                    selected["decoded"].chosen_ue_per_prg.detach().cpu().numpy().astype(np.int64)
                )
                action_summary = summarize_same_state_actions(
                    pfq_chosen,
                    ai_chosen,
                    post_eq_sinr=meta.post_eq_sinr.detach().cpu().numpy(),
                    n_cell=int(meta.n_cell),
                    n_prg=int(meta.n_prg),
                )
                ue_features = graph.snapshot.ue_features.values.detach().cpu().numpy()
                ue_feature_names = list(graph.snapshot.ue_features.names)
                backlog = ue_features[:, ue_feature_names.index("buffer_bytes")]
                ttl_slack_ms = ue_features[:, ue_feature_names.index("ttl_slack_ms")]
                exported_log_pressure = ue_features[
                    :, ue_feature_names.index("log1p_drain_pressure")
                ]
                pre_action_deadline_state = deadline_state_by_ue(
                    ue_features,
                    ue_feature_names,
                )
                recent_scheduled_dequeue = infer_recent_scheduled_dequeue(
                    backlog,
                    ttl_slack_ms,
                    exported_log_pressure,
                    slot_duration_ms=0.5,
                )
                four_actions = build_four_actions(pfq_chosen, ai_chosen)
                four_ctbs: dict[str, dict[str, Any]] = {}
                four_action_metrics: dict[str, dict[str, Any]] = {}
                for action_name, chosen in four_actions.items():
                    action_ctbs = _ctbs_action_capacity(
                        ctbs,
                        chosen,
                        backlog,
                        n_cell=int(meta.n_cell),
                        n_prg=int(meta.n_prg),
                    )
                    pressure = counterfactual_deadline_pressure(
                        backlog,
                        ttl_slack_ms,
                        recent_scheduled_dequeue,
                        action_ctbs["demand_capped_service_bytes_by_ue"],
                        slot_duration_ms=0.5,
                        ewma_alpha=0.05,
                        tail_fraction=0.10,
                    )
                    structure = action_structure_metrics(
                        chosen,
                        n_cell=int(meta.n_cell),
                        n_prg=int(meta.n_prg),
                    )
                    four_ctbs[action_name] = action_ctbs
                    four_action_metrics[action_name] = {
                        **structure,
                        "capacity_bytes_total": int(action_ctbs["capacity_bytes_total"]),
                        "demand_capped_service_bytes_total": float(
                            action_ctbs["demand_capped_service_bytes_total"]
                        ),
                        "unused_capacity_bytes_total": float(
                            int(action_ctbs["capacity_bytes_total"])
                            - float(action_ctbs["demand_capped_service_bytes_total"])
                        ),
                        "phi_before": float(pressure["phi_before"]),
                        "phi_after": float(pressure["phi_after"]),
                        "phi_delta": float(pressure["phi_delta"]),
                        "capacity_bytes_by_ue": action_ctbs["capacity_bytes_by_ue"],
                        "demand_capped_service_bytes_by_ue": action_ctbs[
                            "demand_capped_service_bytes_by_ue"
                        ],
                        "log_pressure_after_by_ue": pressure[
                            "log_pressure_after"
                        ].tolist(),
                    }
                pfq_ctbs = four_ctbs["pfq"]
                ai_ctbs = four_ctbs["dpa"]
                attribution_metric_names = (
                    "capacity_bytes_total",
                    "demand_capped_service_bytes_total",
                    "unused_capacity_bytes_total",
                    "phi_delta",
                    "coactive_cell_pair_count",
                    "scheduled_cell_prg_count",
                )
                four_action_attribution = {
                    metric_name: shapley_two_patch_contributions(
                        {
                            action_name: float(values[metric_name])
                            for action_name, values in four_action_metrics.items()
                        }
                    )
                    for metric_name in attribution_metric_names
                }
                common_active = (pfq_chosen >= 0) & (ai_chosen >= 0)
                ue_patch_count = int(
                    np.count_nonzero(common_active & (pfq_chosen != ai_chosen))
                )
                topology_patch_count = int(
                    np.count_nonzero((pfq_chosen >= 0) != (ai_chosen >= 0))
                )
                active_pressure_ids = np.flatnonzero((backlog > 0.0) & (ttl_slack_ms >= 0.0))
                tail_count = max(1, int(np.ceil(0.10 * active_pressure_ids.size)))
                pressure_tail_ue_ids = (
                    active_pressure_ids[
                        np.argsort(exported_log_pressure[active_pressure_ids])[::-1][
                            :tail_count
                        ]
                    ].astype(int).tolist()
                    if active_pressure_ids.size
                    else []
                )
                next_obs, _raw_reward, done, next_info = client.step(
                    pfq_action_ue, pfq_action_prg
                )
                actual_chosen = _actual_chosen_from_mask(
                    np.asarray(next_info["actual_ue_prg_mask"]),
                    ctbs.serving_cell,
                    n_cell=int(meta.n_cell),
                    n_prg=int(meta.n_prg),
                )
                actual_tbs = np.asarray(
                    next_info["actual_ue_tbs_capacity_bytes"], dtype=np.int64
                ).reshape(-1)
                pfq_tbs = np.asarray(pfq_ctbs["capacity_bytes_by_ue"], dtype=np.int64)
                action_tti = step_idx + 1
                if action_tti > int(args.warmup_steps):
                    rows.append(
                        {
                            "tti": action_tti,
                            "pre_action_sim_tti": int(ctbs.tti),
                            "snapshot_id": int(ctbs.snapshot_id),
                            "pfq_chosen_ue_per_cell_prg": pfq_chosen.tolist(),
                            "ai_chosen_ue_per_cell_prg": ai_chosen.tolist(),
                            **action_summary,
                            "pfq_ctbs_capacity_bytes_total": int(
                                pfq_ctbs["capacity_bytes_total"]
                            ),
                            "ai_ctbs_capacity_bytes_total": int(
                                ai_ctbs["capacity_bytes_total"]
                            ),
                            "pfq_demand_capped_service_bytes_total": float(
                                pfq_ctbs["demand_capped_service_bytes_total"]
                            ),
                            "ai_demand_capped_service_bytes_total": float(
                                ai_ctbs["demand_capped_service_bytes_total"]
                            ),
                            "ue_reallocation_patch_count": ue_patch_count,
                            "topology_patch_count": topology_patch_count,
                            "pressure_tail_ue_ids": pressure_tail_ue_ids,
                            "pre_action_deadline_state": pre_action_deadline_state,
                            "four_action_metrics": four_action_metrics,
                            "four_action_attribution": four_action_attribution,
                            "dpa_minus_pfq_demand_capped_service_bytes": float(
                                four_action_metrics["dpa"][
                                    "demand_capped_service_bytes_total"
                                ]
                                - four_action_metrics["pfq"][
                                    "demand_capped_service_bytes_total"
                                ]
                            ),
                            "dpa_minus_pfq_phi_delta": float(
                                four_action_metrics["dpa"]["phi_delta"]
                                - four_action_metrics["pfq"]["phi_delta"]
                            ),
                            "shadow_inference_state_immutable": bool(
                                state_hash_before == state_hash_after
                            ),
                            "pfq_action_acceptance_mismatch_count": int(
                                np.count_nonzero(actual_chosen != pfq_chosen)
                            ),
                            "pfq_ctbs_reference_mismatch_count": int(
                                np.count_nonzero(actual_tbs != pfq_tbs)
                            ),
                        }
                    )
                obs, info = next_obs, next_info
                if done and action_tti < total_steps:
                    raise RuntimeError("environment ended before spike horizon")
        finally:
            if client is not None:
                client.close()
            try:
                return_code = proc.wait(timeout=60.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                return_code = proc.wait(timeout=30.0)
            if return_code != 0:
                raise RuntimeError(f"simulator exited with code {return_code}")

    if len(rows) != int(args.record_steps):
        raise RuntimeError(
            f"spike row count mismatch: got={len(rows)} expected={int(args.record_steps)}"
        )
    aggregate = _aggregate(rows)
    four_action_metric_names = (
        "capacity_bytes_total",
        "demand_capped_service_bytes_total",
        "unused_capacity_bytes_total",
        "phi_delta",
        "coactive_cell_pair_count",
        "scheduled_cell_prg_count",
    )
    aggregate["four_action_means"] = {
        action_name: {
            metric_name: float(
                np.mean(
                    [
                        float(row["four_action_metrics"][action_name][metric_name])
                        for row in rows
                    ]
                )
            )
            for metric_name in four_action_metric_names
        }
        for action_name in ("pfq", "ue_reallocation_only", "topology_only", "dpa")
    }
    aggregate["shapley_contribution_means"] = {
        metric_name: {
            contribution_name: float(
                np.mean(
                    [
                        float(
                            row["four_action_attribution"][metric_name][
                                contribution_name
                            ]
                        )
                        for row in rows
                    ]
                )
            )
            for contribution_name in (
                "ue_reallocation_contribution",
                "topology_contribution",
                "total_delta",
                "additivity_residual",
            )
        }
        for metric_name in four_action_metric_names
    }
    aggregate["dpa_minus_pfq_demand_capped_service_bytes_mean"] = float(
        np.mean([row["dpa_minus_pfq_demand_capped_service_bytes"] for row in rows])
    )
    aggregate["dpa_minus_pfq_phi_delta_mean"] = float(
        np.mean([row["dpa_minus_pfq_phi_delta"] for row in rows])
    )
    try:
        representative_tti = dict(select_representative_tti(rows))
        representative_selection_status = "AVAILABLE"
        representative_selection_reason = ""
    except ValueError as exc:
        representative_tti = None
        representative_selection_status = "UNAVAILABLE"
        representative_selection_reason = str(exc)
    report = {
        "status": "PASS" if (
            aggregate["state_hash_mismatch_count"] == 0
            and aggregate["pfq_action_acceptance_mismatch_count"] == 0
            and aggregate["pfq_ctbs_reference_mismatch_count"] == 0
        ) else "FAIL",
        "scope": "throwaway_seed42_same_state_four_action_attribution_spike",
        "state_source_policy": "PFQ teacher committed; DPA shadow inference only",
        "seed": int(args.seed),
        "warmup_steps": int(args.warmup_steps),
        "record_steps": int(args.record_steps),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "sim_binary": str(sim_bin),
        "sim_binary_sha256": _sha256_file(sim_bin),
        "capabilities": {
            "same_pre_action_state_dual_scheduler_actions": True,
            "read_only_dpa_shadow_inference": aggregate["state_hash_mismatch_count"] == 0,
            "scheduler_independent_pre_action_post_eq_selected_quality": True,
            "counterfactual_ctbs_tbs_capacity": aggregate[
                "pfq_ctbs_reference_mismatch_count"
            ] == 0,
            "four_action_service_and_pressure_attribution": True,
            "four_action_shapley_additive": all(
                abs(
                    aggregate["shapley_contribution_means"][metric_name][
                        "additivity_residual"
                    ]
                )
                <= 1.0e-9
                for metric_name in four_action_metric_names
            ),
            "counterfactual_interference_exposure_from_active_masks": True,
            "action_conditioned_counterfactual_post_eq_sinr": False,
            "simulator_snapshot_restore_branching": False,
        },
        "interpretation_limits": {
            "post_eq_sinr": (
                "pre-action full-reuse candidate quality; selecting different UE/PRG values is "
                "comparable, but it is not recomputed from each joint action"
            ),
            "dl_reuse_partner": (
                "current DL PHY changes interference when a neighboring Cell-PRG is blank/active; "
                "changing only the neighboring UE identity does not change target SINR"
            ),
            "delay": "not counterfactually attributable from this one-step shadow spike",
            "deadline_pressure": (
                "no-arrival one-step proxy using scheduled dequeue, exported TTL slack, and the "
                "same 0.05 scheduled-dequeue EWMA as the actor feature; not packet-level outcome delay"
            ),
        },
        "representative_tti_selection": {
            "status": representative_selection_status,
            "rule": (
                "abs(DPA-PFQ active Cell-PRGs)<=1, positive demand-capped service and "
                "pressure-reduction gains, then closest to the two-metric median"
            ),
            "reason": representative_selection_reason,
            "tti": int(representative_tti["tti"]) if representative_tti else None,
        },
        "aggregate": aggregate,
    }
    with (out_dir / "per_tti.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (out_dir / "feasibility_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if representative_tti is not None:
        (out_dir / "representative_tti.json").write_text(
            json.dumps(representative_tti, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Online imitation warm start for the Stage-B sparse entity graph policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.gnnrl.masks import (
    apply_prg_action_mask,
    apply_ue_action_mask,
    build_slot_selection_mask,
    classification_accuracy,
    predicted_legal_ratio,
    sanitize_targets,
)
from training.stageb_hgraph.bridge_env import BridgeEnvClient, BridgeEnvConfig
from training.stageb_hgraph.ctbs_headroom import CtbsHeadroomInputs, CtbsHeadroomState, compare_ctbs_reference_labels
from training.stageb_hgraph.edge_generator import EdgeGeneratorConfig, SparseEdgeGenerator
from training.stageb_hgraph.feature_snapshot import FeatureSnapshotBuilder, SnapshotBuilderConfig
from training.stageb_hgraph.graph_ops import (
    build_prg_candidate_valid_mask,
    joint_candidate_targets_from_type0_actions,
    pack_prg_candidates,
)
from training.stageb_hgraph.independent_actor import (
    IndependentActorConfig,
    PFQIndependentReuseActor,
)
from training.stageb_hgraph.independent_policy import (
    IndependentPolicyTrace,
    _run_policy,
    independent_teacher_targets_from_type0,
    independent_teacher_trace_from_type0,
    replay_independent_trace,
    resolve_full_candidate_width,
    select_independent_action,
)
from training.stageb_hgraph.model import HGraphModelConfig, StageBHGraphPolicy
from training.stageb_hgraph.phy_tbs_alignment_audit import PhyTbsConfig
from training.stageb_hgraph.teacher import HeuristicTeacherConfig, build_joint_teacher
from training.stageb_hgraph.types import SparseEntityGraph


_INDEPENDENT_BC_DATASET_SCHEMA = "independent-bc-ctbs-physical-bytes-v2"
_INDEPENDENT_BC_METADATA_KEYS = {
    "feature_stage",
    "headroom_accounting",
    "teacher_policy",
    "candidate_mode",
    "include_pfq_score_anchor",
    "actor_family",
    "collection_ablation",
    "topology_seed_mode",
    "topology_seed",
    "model_seed",
    "expected_episode_count",
    "expected_steps_per_episode",
    "expected_record_count",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _validate_independent_bc_metadata(
    metadata: object,
    initial_actor_sha256: object,
) -> Dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise ValueError("independent BC dataset metadata contract is missing")
    missing = sorted(_INDEPENDENT_BC_METADATA_KEYS - set(metadata))
    if missing:
        raise ValueError(f"independent BC dataset metadata contract missing: {missing}")
    result = dict(metadata)
    if result["feature_stage"] not in {"legacy", "dimensionless"}:
        raise ValueError("independent BC dataset feature_stage must be legacy or dimensionless")
    expected = {
        "headroom_accounting": "ctbs-v1",
        "teacher_policy": "pfq_passthrough",
        "candidate_mode": "all_demand_ues",
        "include_pfq_score_anchor": False,
        "actor_family": PFQIndependentReuseActor.actor_family,
        "collection_ablation": "full",
    }
    mode = result["topology_seed_mode"]
    if mode not in {"fixed", "list_cycle"}:
        raise ValueError("independent BC dataset topology_seed_mode must be fixed or list_cycle")
    if mode == "list_cycle":
        seeds = result.get("seed_list")
        if not isinstance(seeds, list) or not seeds or any(not isinstance(seed, int) or seed < 0 for seed in seeds):
            raise ValueError("independent BC list_cycle requires an explicit seed_list")
    for key, value in expected.items():
        if result[key] != value:
            raise ValueError(
                f"independent BC dataset metadata {key} mismatch: "
                f"got={result[key]!r} expected={value!r}"
            )
    for key in (
        "topology_seed",
        "model_seed",
        "expected_episode_count",
        "expected_steps_per_episode",
        "expected_record_count",
    ):
        if int(result[key]) <= 0:
            raise ValueError(f"independent BC dataset metadata {key} must be positive")
        result[key] = int(result[key])
    if result["expected_record_count"] != (
        result["expected_episode_count"] * result["expected_steps_per_episode"]
    ):
        raise ValueError("independent BC dataset expected_record_count is inconsistent")
    digest = str(initial_actor_sha256)
    if len(digest) != 64:
        raise ValueError("independent BC dataset initial actor hash is invalid")
    try:
        bytes.fromhex(digest)
    except ValueError as exc:
        raise ValueError("independent BC dataset initial actor hash is invalid") from exc
    return result


def _validate_independent_bc_step_sequence(
    rows: List[Mapping[str, object]],
    metadata: Mapping[str, object],
) -> None:
    expected_count = int(metadata["expected_record_count"])
    steps_per_episode = int(metadata["expected_steps_per_episode"])
    if len(rows) != expected_count:
        raise ValueError(
            "independent BC dataset expected_record_count mismatch: "
            f"got={len(rows)} expected={expected_count}"
        )
    first_tti: Dict[int, int] = {}
    for index, row in enumerate(rows):
        expected_episode = index // steps_per_episode + 1
        expected_step = index % steps_per_episode
        episode = int(row.get("episode", -1))
        step = int(row.get("step", -1))
        global_step = int(row.get("global_step", -1))
        tti = int(row.get("tti", -1))
        if (episode, step, global_step) != (expected_episode, expected_step, index):
            raise ValueError(
                "independent BC dataset episode/step/global-step continuity mismatch "
                f"at record {index}"
            )
        if metadata["topology_seed_mode"] == "list_cycle":
            seeds = metadata["seed_list"]
            expected_seed = seeds[(episode - 1) % len(seeds)]
            if row.get("topology_seed") != expected_seed or row.get("episode_seed") != expected_seed:
                raise ValueError(f"independent BC scene seed mismatch at record {index}")
        first_tti.setdefault(episode, tti)
        if tti != first_tti[episode] + step:
            raise ValueError(f"independent BC dataset TTI continuity mismatch at record {index}")
        if bool(row.get("done", False)) and step < steps_per_episode - 1:
            raise ValueError(f"independent BC dataset early done at record {index}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_tensor_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _cpu_detached(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{field.name: _cpu_detached(getattr(value, field.name)) for field in fields(value)}
        )
    if isinstance(value, dict):
        return {key: _cpu_detached(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_detached(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_detached(item) for item in value)
    return value


def _align_ctbs_teacher(graph, action_ue, action_prg, inputs):
    """BC-only deterministic prefix pruning; never relax the actor's legal mask."""
    meta=graph.snapshot.static_meta
    ue=torch.as_tensor(action_ue).reshape(meta.n_sched_ue)
    raw=torch.as_tensor(action_prg).reshape(meta.n_prg*meta.n_cell)
    aligned=raw.clone()
    states={}; dropped=[]
    for prg in range(meta.n_prg):
        for cell in range(meta.n_cell):
            index=prg*meta.n_cell+cell; slot=int(raw[index])
            if slot==-1:continue
            if not 0<=slot<meta.n_sched_ue:raise ValueError('teacher slot out of range')
            if not bool(meta.action_mask_prg_cell[cell,prg]):
                raise ValueError('teacher PRG unavailable')
            target=int(ue[slot])
            if not 0<=target<meta.n_ue:raise ValueError('teacher UE out of range')
            if inputs.serving_cell[target]!=cell or not bool(meta.action_mask_cell_ue[cell,target]):
                raise ValueError('teacher serving cell mismatch')
            if target not in states:
                states[target]=CtbsHeadroomState(float(graph.snapshot.ue_features.values[target,0]),
                    inputs.reconstructed_mcs_for_ue(target),inputs.config)
            state=states[target]
            # Only trim surplus after a genuinely allocated prefix. Invalid initial
            # demand/schedulability still fails the canonical policy trace below.
            if state.selected_local_prgs and state.remaining_demand_bytes<=0:
                aligned[index]=-1; dropped.append(index)
            else:
                states[target]=state.with_selected_prg(prg,layer_cap=inputs.prg_layer_cap[target][prg])
    return aligned,{'version':'ctbs_prefix_prune_v1','raw_action_prg':raw.cpu().tolist(),
                    'dropped_native_indices':dropped,'dropped_count':len(dropped)}


def _canonical_independent_teacher_trace(
    actor: PFQIndependentReuseActor,
    graph: SparseEntityGraph,
    packed,
    teacher_action_ue: torch.Tensor,
    teacher_action_prg: torch.Tensor,
    ctbs_headroom_inputs: CtbsHeadroomInputs,
) -> IndependentPolicyTrace:
    """Build a policy-RNG-independent, replayable CTBS-v1 teacher trace."""
    was_training = actor.training
    actor.train(False)
    try:
        with torch.no_grad():
            trace = _run_policy(
                actor,
                graph,
                packed,
                mode="greedy",
                teacher_actions=(teacher_action_ue, teacher_action_prg),
                ctbs_headroom_inputs=ctbs_headroom_inputs,
            )
    finally:
        actor.train(was_training)
    zero = torch.zeros((), dtype=torch.float32)
    return _cpu_detached(
        replace(
            trace,
            logp=zero,
            entropy=zero,
            diagnostics=None,
            rescue_diagnostics=None,
            service_accounting_diagnostics=None,
            raw_logits_at_step=None,
            chosen_logprob_at_step=None,
            ctbs_diagnostics=None,
            blank_expiry_diagnostics=None,
        )
    )


def _independent_bc_train_step(
    actor: PFQIndependentReuseActor,
    optimizer: torch.optim.Optimizer,
    graph: SparseEntityGraph,
    teacher_trace: IndependentPolicyTrace,
    grad_clip: float,
    *,
    update: bool = True,
) -> Dict[str, float]:
    actor.train(True)
    logp, _entropy, _decoded = replay_independent_trace(
        actor,
        graph,
        pack_prg_candidates(
            graph,
            max_candidates=int(actor.cfg.max_candidates),
            fixed_width=True,
        ),
        teacher_trace,
    )
    loss = -logp
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite BC loss")
    grad_norm = 0.0
    if update:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=float(grad_clip), error_if_nonfinite=True)
        )
        optimizer.step()
    return {"loss": float(loss.detach().item()), "grad_norm": float(grad_norm)}


class IndependentBcDatasetWriter:
    def __init__(
        self,
        directory: Path | str,
        *,
        initial_actor_sha256: str,
        metadata: Mapping[str, object],
    ) -> None:
        self.directory = Path(directory).resolve()
        if self.directory.exists() and any(self.directory.iterdir()):
            raise ValueError(f"dataset output directory must be empty: {self.directory}")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict[str, object]] = []
        self.initial_actor_sha256 = str(initial_actor_sha256)
        self.metadata = _validate_independent_bc_metadata(
            metadata, self.initial_actor_sha256
        )
        self.step_metadata: List[Mapping[str, object]] = []

    def write(
        self,
        *,
        graph: SparseEntityGraph,
        teacher_trace: IndependentPolicyTrace,
        teacher_action_ue: torch.Tensor,
        teacher_action_prg: torch.Tensor,
        next_obs: Mapping[str, object],
        next_info: Mapping[str, object],
        step_metadata: Mapping[str, object],
    ) -> None:
        index = len(self.records)
        name = f"step_{index:08d}.pt"
        record = {
            "schema": _INDEPENDENT_BC_DATASET_SCHEMA,
            "index": index,
            "graph": _cpu_detached(graph),
            "teacher_trace": _cpu_detached(teacher_trace),
            "teacher_action_ue": _cpu_detached(teacher_action_ue.to(dtype=torch.long)),
            "teacher_action_prg": _cpu_detached(teacher_action_prg.to(dtype=torch.long)),
            "next_obs": _cpu_detached(dict(next_obs)),
            "next_info": _cpu_detached(dict(next_info)),
            "metadata": dict(step_metadata),
        }
        path = self.directory / name
        torch.save(record, path)
        self.records.append({"file": name, "sha256": _sha256_file(path)})
        self.step_metadata.append(dict(step_metadata))

    def finalize(self, *, max_candidates: int) -> Dict[str, object]:
        if not self.records:
            raise ValueError("cannot finalize an empty independent BC dataset")
        _validate_independent_bc_step_sequence(self.step_metadata, self.metadata)
        dataset_digest = hashlib.sha256()
        dataset_digest.update(_canonical_json_bytes(self.metadata))
        for record in self.records:
            dataset_digest.update(bytes.fromhex(str(record["sha256"])))
        manifest: Dict[str, object] = {
            "schema": _INDEPENDENT_BC_DATASET_SCHEMA,
            "count": len(self.records),
            "records": self.records,
            "dataset_sha256": dataset_digest.hexdigest(),
            "initial_actor_sha256": self.initial_actor_sha256,
            "max_candidates": int(max_candidates),
            "metadata": self.metadata,
        }
        manifest["manifest_sha256"] = hashlib.sha256(
            _canonical_json_bytes(manifest)
        ).hexdigest()
        (self.directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return manifest


class IndependentBcDatasetReader:
    def __init__(
        self,
        directory: Path | str,
        *,
        expected_metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.directory = Path(directory).resolve()
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"independent BC dataset manifest is missing: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("independent BC dataset manifest is malformed") from exc
        if not isinstance(manifest, dict) or manifest.get("schema") != _INDEPENDENT_BC_DATASET_SCHEMA:
            raise ValueError("independent BC dataset manifest schema mismatch")
        records = manifest.get("records")
        if not isinstance(records, list) or int(manifest.get("count", -1)) != len(records) or not records:
            raise ValueError("independent BC dataset manifest count is invalid")
        dataset_digest = hashlib.sha256()
        for index, entry in enumerate(records):
            if not isinstance(entry, dict) or entry.get("file") != f"step_{index:08d}.pt":
                raise ValueError("independent BC dataset record sequence is malformed")
            path = self.directory / str(entry["file"])
            if not path.is_file() or _sha256_file(path) != str(entry.get("sha256", "")):
                raise ValueError(f"independent BC dataset record hash mismatch: {path}")
            dataset_digest.update(bytes.fromhex(str(entry["sha256"])))
        metadata = _validate_independent_bc_metadata(
            manifest.get("metadata"), manifest.get("initial_actor_sha256")
        )
        expected_count = int(metadata["expected_record_count"])
        if len(records) != expected_count:
            raise ValueError(
                "independent BC dataset expected_record_count mismatch: "
                f"got={len(records)} expected={expected_count}"
            )
        if expected_metadata is not None:
            for key, expected_value in expected_metadata.items():
                if metadata.get(key) != expected_value:
                    raise ValueError(
                        f"independent BC dataset expected metadata {key} mismatch: "
                        f"got={metadata.get(key)!r} expected={expected_value!r}"
                    )
        dataset_digest = hashlib.sha256()
        dataset_digest.update(_canonical_json_bytes(metadata))
        for entry in records:
            dataset_digest.update(bytes.fromhex(str(entry["sha256"])))
        if dataset_digest.hexdigest() != str(manifest.get("dataset_sha256", "")):
            raise ValueError("independent BC dataset aggregate hash mismatch")
        manifest_without_hash = dict(manifest)
        manifest_sha256 = str(manifest_without_hash.pop("manifest_sha256", ""))
        if hashlib.sha256(_canonical_json_bytes(manifest_without_hash)).hexdigest() != manifest_sha256:
            raise ValueError("independent BC dataset manifest hash mismatch")
        if int(manifest.get("max_candidates", 0)) <= 0:
            raise ValueError("independent BC dataset max_candidates is invalid")
        self.manifest = manifest
        self.records = records

    def __iter__(self) -> Iterator[Dict[str, object]]:
        step_rows: List[Mapping[str, object]] = []
        for index, entry in enumerate(self.records):
            record = torch.load(
                self.directory / str(entry["file"]),
                map_location="cpu",
                weights_only=False,
            )
            if (
                not isinstance(record, dict)
                or record.get("schema") != _INDEPENDENT_BC_DATASET_SCHEMA
                or int(record.get("index", -1)) != index
                or not isinstance(record.get("graph"), SparseEntityGraph)
                or not isinstance(record.get("teacher_trace"), IndependentPolicyTrace)
                or not isinstance(record.get("next_obs"), Mapping)
                or not isinstance(record.get("next_info"), Mapping)
            ):
                raise ValueError(f"independent BC dataset record {index} is malformed")
            graph = record["graph"]
            trace = record["teacher_trace"]
            metadata = record.get("metadata")
            if not isinstance(metadata, Mapping):
                raise ValueError(f"independent BC dataset record {index} metadata is malformed")
            if trace.ctbs_headroom_inputs is None or int(trace.ctbs_headroom_inputs.tti) != int(
                metadata.get("tti", -1)
            ):
                raise ValueError(
                    f"independent BC dataset record {index} CTBS tuple TTI mismatch"
                )
            contract = self.manifest["metadata"]
            expected_seed = int(contract["topology_seed"])
            if contract["topology_seed_mode"] == "list_cycle":
                expected_seed = contract["seed_list"][(int(metadata.get("episode", 0)) - 1) % len(contract["seed_list"])]
            if int(metadata.get("topology_seed", -1)) != expected_seed:
                raise ValueError(f"independent BC dataset record {index} topology seed mismatch")
            if int(metadata.get("episode_seed", -1)) != expected_seed:
                raise ValueError(f"independent BC dataset record {index} episode seed mismatch")
            expected_tail = (
                ("critical_ttl_backlog_bytes", "near_ttl_backlog_bytes", "age_weighted_backlog_bytes", "service_gap_tti")
                if self.manifest["metadata"]["feature_stage"] == "legacy"
                else ("log1p_drain_pressure", "scheduled_dequeue_adequacy", "ttl_normalized_service_age", "reserved_zero")
            )
            if tuple(graph.snapshot.ue_features.names[12:16]) != expected_tail:
                raise ValueError(f"independent BC dataset record {index} feature stage mismatch")
            edge_attr = graph.edges["E_UP"].attr
            if edge_attr is None or edge_attr.dim() != 2 or int(edge_attr.shape[1]) != 5:
                raise ValueError(f"independent BC dataset record {index} E_UP schema mismatch")
            stored_ue = torch.as_tensor(record.get("teacher_action_ue"), dtype=torch.long)
            stored_prg = torch.as_tensor(record.get("teacher_action_prg"), dtype=torch.long)
            packed = pack_prg_candidates(
                graph,
                max_candidates=int(self.manifest["max_candidates"]),
                fixed_width=True,
            )
            try:
                teacher_choice = independent_teacher_targets_from_type0(
                    graph,
                    packed,
                    stored_ue,
                    stored_prg,
                    trace.valid_at_step,
                )
            except ValueError as exc:
                raise ValueError(
                    f"independent BC dataset record {index} teacher actions disagree with frozen trace"
                ) from exc
            if not torch.equal(teacher_choice.cpu(), trace.choice.cpu()):
                raise ValueError(
                    f"independent BC dataset record {index} teacher actions disagree with frozen trace"
                )
            step_rows.append(metadata)
            yield record
        _validate_independent_bc_step_sequence(step_rows, self.manifest["metadata"])


def _parse_seed_list(seed_list_arg: str) -> List[int]:
    if not seed_list_arg:
        return []
    out: List[int] = []
    for token in seed_list_arg.replace(",", " ").split():
        out.append(int(token))
    return out


def _parse_sim_env_entries(entries: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for kv in entries:
        if "=" not in kv:
            continue
        key, value = kv.split("=", 1)
        out[key] = value
    return out


def _resolve_topology_seed_mode(mode: str, explicit_seed_list: List[int]) -> str:
    if mode != "auto":
        return mode
    return "list_cycle" if explicit_seed_list else "fixed"


def _pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_prg_candidate_count(topk: int, diversity_extra: int) -> int:
    return max(1, int(topk)) + max(0, int(diversity_extra))


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class EpisodeSummary:
    episode: int
    episode_seed: int
    topology_seed: int
    steps: int
    mean_loss: float
    mean_acc: float
    mean_teacher_blank_ratio: float
    mean_reward: float
    mean_prev_assigned_ratio: float


class OnlineEpisodeRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.ctbs_enabled = (
            str(getattr(args, "independent_headroom_accounting", "raw-v1"))
            == "ctbs-v1"
        )
        self.proc: Optional[subprocess.Popen] = None
        self.env: Optional[BridgeEnvClient] = None
        self.socket_path: Optional[str] = None
        self.live_topology_seed: Optional[int] = None
        self.cached_obs = None
        self.cached_info = None
        self.cached_reward: float = 0.0
        self.proc_log_handle = None
        self.explicit_seed_list = _parse_seed_list(args.seed_list)
        self.resolved_topology_seed_mode = _resolve_topology_seed_mode(args.topology_seed_mode, self.explicit_seed_list)
        self.base_topology_seed = int(args.topology_seed)
        self.sim_env_overrides = _parse_sim_env_entries(getattr(args, "sim_env", []))
        self.sim_log_path = Path(args.out_dir).resolve() / "sim_runtime" / "online_bridge_imitation.log"

    def _bridge_env_config(self) -> BridgeEnvConfig:
        return BridgeEnvConfig(
            socket_path=str(self.socket_path),
            connect_timeout_s=float(self.args.connect_timeout_s),
            enable_ctbs=self.ctbs_enabled,
        )

    def _episode_seed_for(self, episode_idx: int) -> int:
        if self.resolved_topology_seed_mode == "list_cycle":
            return int(self.explicit_seed_list[(episode_idx - 1) % len(self.explicit_seed_list)])
        if self.resolved_topology_seed_mode == "sequential":
            return int(self.base_topology_seed + max(0, episode_idx - 1))
        return int(self.base_topology_seed)

    def _launch_sim(self, socket_path: str, topology_seed: int, episode_idx: int, episode_seed: int) -> subprocess.Popen:
        sim_env = os.environ.copy()
        sim_env["CUMAC_ONLINE_BRIDGE"] = "1"
        sim_env["CUMAC_ONLINE_SOCKET"] = socket_path
        sim_env["CUMAC_ONLINE_PERSISTENT"] = "1" if int(self.args.online_persistent) == 1 else "0"
        sim_env.setdefault("CUMAC_COMPACT_TTI_LOG", "1")
        sim_env.setdefault("CUMAC_COMPARE_TTI_INTERVAL", "0")
        if self.args.action_head_mode == "independent":
            sim_env.setdefault("CUMAC_ONLINE_EXPORT_TEACHER_ACTION", "1")
            sim_env.setdefault("CUMAC_ONLINE_EXPORT_POST_EQ_SINR", "1")
            sim_env.setdefault("CUMAC_CELL_ASSOC_MODE", "strongest")
            sim_env.setdefault("CUMAC_UE_PLACEMENT_MODE", "coop_boundary")
            sim_env.setdefault("CUMAC_UE_VORONOI_CLIP", "1")
            sim_env.setdefault("CUMAC_BS_TX_PATTERN", "omni")
            sim_env.setdefault("CUMAC_TRAFFIC_SEED", str(int(topology_seed)))
            sim_env.setdefault("CUMAC_TRAFFIC_ARRIVAL_RATE", "1.0")
            sim_env.setdefault("CUMAC_PACKET_TTL_TTI", "0")
            sim_env.setdefault("CUMAC_PACKET_TTL_MS", "200")
            # The independent policy has a zero-slack hard service-headroom
            # mask. Teacher collection must share that action space; the PFQ
            # comparator calibration remains unchanged.
            sim_env.setdefault("CUMAC_PFQ_DEMAND_CAP_SLACK_BYTES", "0")
            if self.ctbs_enabled:
                sim_env.setdefault("CUMAC_ONLINE_EXPORT_PHY_TBS_AUDIT", "1")
        for key, value in self.sim_env_overrides.items():
            sim_env[key] = value
        sim_env["CUMAC_TOPOLOGY_SEED"] = str(int(topology_seed))
        self._maybe_extend_ld_library_path(sim_env)

        cmd = [self.args.sim_bin] + shlex.split(self.args.sim_args)
        if self.args.sim_log_mode == "file":
            self.sim_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.proc_log_handle = open(self.sim_log_path, "a", encoding="utf-8", buffering=1)
            launch_ts = time.strftime("%Y-%m-%d %H:%M:%S")
            self.proc_log_handle.write(
                f"\n===== [{launch_ts}] episode={episode_idx} episode_seed={episode_seed} "
                f"topology_seed={topology_seed} socket={socket_path} =====\n"
            )
            return subprocess.Popen(
                cmd,
                env=sim_env,
                cwd=self.args.sim_cwd,
                stdout=self.proc_log_handle,
                stderr=subprocess.STDOUT,
            )
        return subprocess.Popen(cmd, env=sim_env, cwd=self.args.sim_cwd)

    def _maybe_extend_ld_library_path(self, sim_env: Dict[str, str]) -> None:
        sim_bin = Path(self.args.sim_bin).resolve()
        build_cumac_dir = sim_bin.parents[2] if len(sim_bin.parents) >= 3 else None
        if build_cumac_dir is None:
            return
        lib_dirs: List[str] = []
        for candidate in (
            build_cumac_dir / "src",
            build_cumac_dir / "lib" / "cumac_msg",
        ):
            if candidate.is_dir():
                lib_dirs.append(str(candidate))
        if not lib_dirs:
            return
        current = sim_env.get("LD_LIBRARY_PATH", "")
        sim_env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([current] if current else []))

    def close_episode(self, graceful: bool = True) -> None:
        if self.env is not None:
            try:
                if graceful:
                    self.env.close()
            except Exception:
                pass
            finally:
                self.env = None
        if self.proc is not None:
            try:
                self.proc.wait(timeout=self.args.sim_wait_timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
            self.proc = None
        if self.proc_log_handle is not None:
            try:
                self.proc_log_handle.flush()
                self.proc_log_handle.close()
            except Exception:
                pass
            finally:
                self.proc_log_handle = None
        if self.socket_path:
            try:
                os.unlink(self.socket_path)
            except FileNotFoundError:
                pass
            self.socket_path = None
        self.live_topology_seed = None
        self.cached_obs = None
        self.cached_info = None
        self.cached_reward = 0.0

    def _can_reuse_episode(self, topology_seed: int) -> bool:
        if int(self.args.online_persistent) != 1:
            return False
        if self.proc is None or self.env is None or self.socket_path is None:
            return False
        if self.proc.poll() is not None:
            return False
        return self.live_topology_seed == int(topology_seed)

    def can_continue_stream(self, topology_seed: int) -> bool:
        return self._can_reuse_episode(topology_seed) and self.cached_obs is not None and self.cached_info is not None

    def begin_rollout(self, episode_idx: int):
        episode_seed = self._episode_seed_for(episode_idx)
        topology_seed = int(episode_seed)
        if self.can_continue_stream(topology_seed):
            print(
                f"[episode {episode_idx:04d}] continue "
                f"episode_seed={episode_seed} topology_seed={topology_seed} "
                f"persistent={int(self.args.online_persistent)} socket={self.socket_path}"
            )
            return self.cached_obs, float(self.cached_reward), self.cached_info, episode_seed, topology_seed, True

        obs, reward, info, episode_seed, topology_seed = self.reset_episode(episode_idx)
        self.cached_obs = obs
        self.cached_info = info
        self.cached_reward = float(reward)
        return obs, float(reward), info, episode_seed, topology_seed, False

    def record_transition(self, obs, reward: float, info) -> None:
        self.cached_obs = obs
        self.cached_info = info
        self.cached_reward = float(reward)

    def reset_episode(self, episode_idx: int):
        episode_seed = self._episode_seed_for(episode_idx)
        topology_seed = int(episode_seed)
        if self._can_reuse_episode(topology_seed):
            assert self.env is not None
            print(
                f"[episode {episode_idx:04d}] reuse "
                f"episode_seed={episode_seed} topology_seed={topology_seed} "
                f"persistent={int(self.args.online_persistent)} socket={self.socket_path}"
            )
            try:
                obs, reward, info = self.env.reset(seed=episode_seed, episode_horizon=self.args.episode_horizon)
                self.cached_obs = obs
                self.cached_info = info
                self.cached_reward = float(reward)
                return obs, float(reward), info, episode_seed, topology_seed
            except Exception as exc:
                print(
                    f"[episode {episode_idx:04d}] reuse_failed "
                    f"topology_seed={topology_seed} err={exc}; relaunching simulator"
                )
                self.close_episode(graceful=False)

        self.close_episode(graceful=True)
        self.socket_path = f"{self.args.socket_path}.{os.getpid()}.{episode_idx}"
        self.proc = self._launch_sim(
            socket_path=self.socket_path,
            topology_seed=topology_seed,
            episode_idx=episode_idx,
            episode_seed=episode_seed,
        )
        print(
            f"[episode {episode_idx:04d}] launch "
            f"episode_seed={episode_seed} topology_seed={topology_seed} "
            f"persistent={int(self.args.online_persistent)} socket={self.socket_path}"
        )
        self.env = BridgeEnvClient(self._bridge_env_config())
        self.live_topology_seed = topology_seed
        obs, reward, info = self.env.reset(seed=episode_seed, episode_horizon=self.args.episode_horizon)
        self.cached_obs = obs
        self.cached_info = info
        self.cached_reward = float(reward)
        return obs, float(reward), info, episode_seed, topology_seed


def _ctbs_inputs_for_imitation_step(
    runner: OnlineEpisodeRunner,
    info: Mapping[str, object],
) -> CtbsHeadroomInputs | None:
    if not runner.ctbs_enabled:
        return None
    if runner.env is None:
        raise RuntimeError("CTBS imitation requires a live bridge environment")
    if bool(info.get("ctbs_config_available", False)):
        if int(info.get("harq_enabled", -1)) != 0:
            raise ValueError("independent CTBS-v1 BC requires runtime HARQ to be disabled")
        runner.ctbs_config = PhyTbsConfig.from_vector(info["ctbs_config"])
        runner.ctbs_config_sha256 = str(info["ctbs_config_sha256"])
    config = getattr(runner, "ctbs_config", None)
    config_sha256 = getattr(runner, "ctbs_config_sha256", None)
    if config is None or config_sha256 is None:
        raise RuntimeError("independent CTBS-v1 BC reset did not provide a TBS configuration")
    if "tti" not in info:
        raise RuntimeError("independent CTBS-v1 BC state is missing its TTI identity")
    pre_action = runner.env.request_pre_action_snapshot(expected_tti=int(info["tti"]))
    return CtbsHeadroomInputs.from_arrays(
        tti=pre_action.tti,
        snapshot_id=pre_action.snapshot_id,
        config_sha256=pre_action.config_sha256,
        config=config,
        ue_ids=pre_action.ue_ids,
        serving_cell=pre_action.serving_cell,
        wb_sinr_first_layer_db=pre_action.wb_sinr_first_layer_db,
        olla_delta=pre_action.olla_delta,
        olla_delta_up=pre_action.olla_delta_up,
        olla_delta_down=pre_action.olla_delta_down,
        prior_tb_err=pre_action.prior_tb_err,
        prg_layer_cap=pre_action.prg_layer_cap,
    )


def _run_offline_independent_bc(
    args: argparse.Namespace,
    device: torch.device,
    metrics_path: Path,
    summary_path: Path,
    ckpt_last_path: Path,
    ckpt_best_path: Path,
) -> int:
    sim_env = _parse_sim_env_entries(list(getattr(args, "sim_env", [])))
    feature_stage = sim_env.get(
        "CUMAC_DPA_FEATURE_MODE", os.environ.get("CUMAC_DPA_FEATURE_MODE", "")
    )
    expected_steps = int(args.max_steps) if int(args.max_steps) > 0 else int(args.episode_horizon)
    reader = IndependentBcDatasetReader(
        args.independent_bc_dataset_in,
        expected_metadata={
            "feature_stage": feature_stage,
            "topology_seed_mode": _resolve_topology_seed_mode(
                str(args.topology_seed_mode), _parse_seed_list(str(args.seed_list))
            ),
            "topology_seed": int(args.topology_seed),
            "expected_episode_count": int(args.episodes),
            "expected_steps_per_episode": expected_steps,
            "expected_record_count": int(args.episodes) * expected_steps,
        },
    )
    max_candidates = int(reader.manifest["max_candidates"])
    model = PFQIndependentReuseActor(
        IndependentActorConfig(
            hidden_dim=int(args.hidden_dim),
            message_layers=int(args.message_layers),
            dropout=float(args.dropout),
            max_candidates=max_candidates,
            ablation=str(args.independent_ablation),
        )
    ).to(device)
    initial_actor_sha256 = _model_tensor_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    rows: List[Dict[str, object]] = []
    episode_losses: Dict[int, List[float]] = {}
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for global_step, record in enumerate(reader):
            trace = record["teacher_trace"]
            if trace.ctbs_headroom_inputs is None:
                raise ValueError(f"independent BC dataset record {global_step} lacks CTBS-v1 inputs")
            step_result = _independent_bc_train_step(
                model,
                optimizer,
                record["graph"],
                trace,
                float(args.grad_clip),
            )
            metadata = record.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise ValueError(f"independent BC dataset record {global_step} metadata is malformed")
            episode = int(metadata.get("episode", 1))
            row: Dict[str, object] = {
                "episode": episode,
                "step": int(metadata.get("step", global_step)),
                "global_step": global_step,
                "tti": int(metadata.get("tti", -1)),
                "loss": float(step_result["loss"]),
                "grad_norm": float(step_result["grad_norm"]),
                "reward_raw": float(metadata.get("reward_raw", 0.0)),
                "action_head_mode": "independent",
                "independent_ablation": str(args.independent_ablation),
                "dataset_replay": 1,
            }
            metrics_file.write(json.dumps(row, ensure_ascii=True) + "\n")
            rows.append(row)
            episode_losses.setdefault(episode, []).append(float(step_result["loss"]))

    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": model.model_config_dict(),
        "actor_state": model.state_dict(),
        "actor_config": model.model_config_dict(),
        "graph_config": {
            "candidate_mode": "all_demand_ues",
            "include_pfq_score_anchor": False,
            "include_candidate_risk_features": False,
        },
        "train_args": vars(args),
        "episode": max(episode_losses, default=0),
        "global_step": len(rows),
        "mean_loss": float(np.mean([row["loss"] for row in rows])) if rows else 0.0,
        "dataset": reader.manifest,
        "initial_actor_sha256": initial_actor_sha256,
    }
    torch.save(checkpoint, ckpt_last_path)
    torch.save(checkpoint, ckpt_best_path)
    summary = {
        "episodes": len(episode_losses),
        "resolved_max_steps": len(rows),
        "device": str(device),
        "teacher_mode": "pfq_passthrough",
        "action_head_mode": "independent",
        "independent_ablation": str(args.independent_ablation),
        "actor_family": model.actor_family,
        "requires_pfq_features": False,
        "resolved_max_candidates": max_candidates,
        "metrics_path": str(metrics_path),
        "last_checkpoint": str(ckpt_last_path),
        "best_checkpoint": str(ckpt_best_path),
        "best_mean_loss": min(
            (float(np.mean(values)) for values in episode_losses.values()),
            default=0.0,
        ),
        "mean_loss_all": float(np.mean([row["loss"] for row in rows])) if rows else 0.0,
        "initial_actor_sha256": initial_actor_sha256,
        "dataset": reader.manifest,
        "dataset_mode": "offline_replay",
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    print(f"[online-imitation] replayed dataset: {args.independent_bc_dataset_in}")
    print(f"[online-imitation] wrote metrics: {metrics_path}")
    print(f"[online-imitation] wrote summary: {summary_path}")
    print(f"[online-imitation] last checkpoint: {ckpt_last_path}")
    print(f"[online-imitation] best checkpoint: {ckpt_best_path}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Stage-B sparse-graph online imitation warm start")
    p.add_argument("--sim-bin", default="", help="Path to multiCellSchedulerUeSelection binary; omitted for dataset replay")
    p.add_argument("--sim-args", default="-d 1 -b 2 -x 0 -f 0 -g 100 -r 3000", help="CLI args for the simulation binary")
    p.add_argument("--sim-cwd", default="", help="Optional cwd for launching the simulation")
    p.add_argument("--sim-env", action="append", default=[], help="Extra simulator env vars, format KEY=VALUE")
    p.add_argument("--sim-log-mode", choices=["file", "inherit"], default="file")
    p.add_argument("--socket-path", default="/tmp/cumac_stageb_hgraph_imitation.sock")
    p.add_argument("--connect-timeout-s", type=float, default=20.0)
    p.add_argument("--sim-wait-timeout", type=float, default=10.0)
    p.add_argument("--online-persistent", type=int, choices=[0, 1], default=1)
    p.add_argument("--episode-horizon", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=0, help="0 means use episode-horizon as the runner-side cap")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--topology-seed", type=int, default=42)
    p.add_argument("--topology-seed-mode", choices=["auto", "fixed", "sequential", "list_cycle"], default="auto")
    p.add_argument("--seed-list", default="", help="Comma-separated topology seeds for list_cycle mode")
    p.add_argument("--out-dir", default="training/stageb_hgraph/runs/online_imitation")
    p.add_argument("--ue-prg-topk", type=int, default=4)
    p.add_argument("--ue-prg-diversity-extra", type=int, default=0)
    p.add_argument("--tb-err-ema-alpha", type=float, default=0.10)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--message-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument(
        "--action-head-mode",
        choices=["slot", "candidate", "independent"],
        default="slot",
        help="independent trains the PFQ-free fixed-order reuse actor; other values retain legacy paths",
    )
    p.add_argument(
        "--candidate-mode",
        choices=["all_demand_ues", "topk_diversity"],
        default="all_demand_ues",
    )
    p.add_argument("--include-pfq-score-anchor", type=int, choices=[0, 1], default=1)
    p.add_argument("--candidate-blank-logit-bias", type=float, default=0.0)
    p.add_argument("--candidate-success-logit-scale", type=float, default=1.0)
    p.add_argument("--candidate-blank-success-scale", type=float, default=1.0)
    p.add_argument("--candidate-pfq-anchor-logit-coef", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--teacher-mode", choices=["heuristic_graph", "pfq_passthrough"], default="pfq_passthrough")
    p.add_argument("--save-every-episodes", type=int, default=1)
    p.add_argument(
        "--independent-headroom-accounting",
        choices=["raw-v1", "ctbs-v1"],
        default="raw-v1",
    )
    p.add_argument(
        "--independent-ablation",
        choices=["full", "no_reuse_attention", "no_dimensionless"],
        default="full",
    )
    p.add_argument("--independent-bc-max-candidates", type=int, default=0, help="0 resolves first scene; explicit width must cover every collection scene")
    p.add_argument("--independent-bc-dataset-out", default="")
    p.add_argument("--independent-bc-dataset-in", default="")
    p.add_argument(
        "--independent-bc-collect-only",
        type=int,
        choices=[0, 1],
        default=0,
        help="capture PFQ/CTBS records without optimizer updates; requires dataset-out",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.sim_cwd == "":
        args.sim_cwd = None
    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1")
    if args.episode_horizon < 1:
        raise ValueError("--episode-horizon must be >= 1")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be >= 0")
    if args.ue_prg_diversity_extra < 0:
        raise ValueError("--ue-prg-diversity-extra must be >= 0")
    if args.candidate_pfq_anchor_logit_coef < 0.0:
        raise ValueError("--candidate-pfq-anchor-logit-coef must be >= 0")
    if args.independent_bc_dataset_in and args.independent_bc_dataset_out:
        raise ValueError("--independent-bc-dataset-in and --independent-bc-dataset-out are mutually exclusive")
    dataset_mode = bool(args.independent_bc_dataset_in or args.independent_bc_dataset_out)
    if int(args.independent_bc_collect_only) == 1 and not args.independent_bc_dataset_out:
        raise ValueError("--independent-bc-collect-only 1 requires --independent-bc-dataset-out")
    if not args.independent_bc_dataset_in and not args.sim_bin:
        raise ValueError("--sim-bin is required unless --independent-bc-dataset-in is used")
    if args.action_head_mode == "independent":
        if args.teacher_mode != "pfq_passthrough":
            raise ValueError("independent BC requires --teacher-mode pfq_passthrough")
        if args.candidate_mode != "all_demand_ues":
            raise ValueError("independent BC requires --candidate-mode all_demand_ues")
        if int(args.include_pfq_score_anchor) != 0:
            raise ValueError("independent BC requires --include-pfq-score-anchor 0")
    if dataset_mode:
        if args.action_head_mode != "independent":
            raise ValueError("independent BC dataset mode requires --action-head-mode independent")
        if args.independent_headroom_accounting != "ctbs-v1":
            raise ValueError("independent BC dataset mode requires --independent-headroom-accounting ctbs-v1")
        if args.independent_bc_dataset_in and args.independent_ablation == "no_dimensionless":
            raise ValueError("offline legacy S0 dataset replay does not support no_dimensionless")
    explicit_seed_list = _parse_seed_list(args.seed_list)
    if args.topology_seed_mode == "list_cycle" and not explicit_seed_list:
        raise ValueError("--topology-seed-mode list_cycle requires --seed-list")
    if explicit_seed_list and args.topology_seed_mode not in ("auto", "list_cycle"):
        raise ValueError("--seed-list may only be used with --topology-seed-mode auto or list_cycle")

    _set_seed(int(args.seed))
    device = _pick_device(args.device)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "online_imitation_metrics.jsonl"
    summary_path = out_dir / "online_imitation_summary.json"
    independent_mode = args.action_head_mode == "independent"
    ckpt_last_path = out_dir / (
        "hgraph_independent_bc_last.pt" if independent_mode else "hgraph_imitation_last.pt"
    )
    ckpt_best_path = out_dir / (
        "hgraph_independent_bc_best.pt" if independent_mode else "hgraph_imitation_best.pt"
    )
    if args.independent_bc_dataset_in:
        return _run_offline_independent_bc(
            args,
            device,
            metrics_path,
            summary_path,
            ckpt_last_path,
            ckpt_best_path,
        )
    max_prg_candidates = (
        0
        if independent_mode
        else _resolve_prg_candidate_count(args.ue_prg_topk, args.ue_prg_diversity_extra)
    )

    requested_feature_mode = _parse_sim_env_entries(list(args.sim_env)).get("CUMAC_DPA_FEATURE_MODE")
    previous_feature_mode = os.environ.get("CUMAC_DPA_FEATURE_MODE")
    if requested_feature_mode is not None:
        os.environ["CUMAC_DPA_FEATURE_MODE"] = requested_feature_mode
    try:
        snapshot_builder = FeatureSnapshotBuilder(
            SnapshotBuilderConfig(tb_err_ema_alpha=float(args.tb_err_ema_alpha))
        )
    finally:
        if requested_feature_mode is not None:
            if previous_feature_mode is None:
                os.environ.pop("CUMAC_DPA_FEATURE_MODE", None)
            else:
                os.environ["CUMAC_DPA_FEATURE_MODE"] = previous_feature_mode
    edge_generator = SparseEdgeGenerator(
        EdgeGeneratorConfig(
            ue_prg_topk=int(args.ue_prg_topk),
            ue_prg_diversity_extra=int(args.ue_prg_diversity_extra),
            candidate_mode=str(args.candidate_mode),
            include_pfq_score_anchor=(
                False if independent_mode else bool(int(args.include_pfq_score_anchor))
            ),
            include_candidate_risk_features=False,
        )
    )
    teacher = build_joint_teacher(
        args.teacher_mode,
        heuristic_cfg=HeuristicTeacherConfig(max_candidates=max_prg_candidates),
        max_candidates=max_prg_candidates,
    )

    if independent_mode:
        model = None
    else:
        model = StageBHGraphPolicy(
            HGraphModelConfig(
                hidden_dim=int(args.hidden_dim),
                message_layers=int(args.message_layers),
                dropout=float(args.dropout),
                max_prg_candidates=max_prg_candidates,
                action_head_mode=str(args.action_head_mode),
                candidate_blank_logit_bias=float(args.candidate_blank_logit_bias),
                candidate_success_logit_scale=float(args.candidate_success_logit_scale),
                candidate_blank_success_scale=float(args.candidate_blank_success_scale),
                candidate_pfq_anchor_logit_coef=float(args.candidate_pfq_anchor_logit_coef),
            )
        ).to(device)
    optimizer = (
        None
        if model is None
        else torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
    )

    runner = OnlineEpisodeRunner(args)
    max_steps = int(args.max_steps) if int(args.max_steps) > 0 else int(args.episode_horizon)
    if max_steps < 1:
        raise ValueError("resolved max_steps must be >= 1")
    ue_loss_weight = 1.0

    all_rows: List[Dict] = []
    episode_summaries: List[EpisodeSummary] = []
    best_mean_loss = float("inf")
    global_step = 0
    dataset_writer: IndependentBcDatasetWriter | None = None
    dataset_manifest: Dict[str, object] | None = None
    initial_actor_sha256 = ""

    try:
        with open(metrics_path, "w", encoding="utf-8") as f_jsonl:
            for episode_idx in range(1, args.episodes + 1):
                obs, _reward0, info, episode_seed, topology_seed, resumed_stream = runner.begin_rollout(episode_idx)
                if not resumed_stream:
                    snapshot_builder.reset()
                step = 0
                episode_rows: List[Dict] = []
                while True:
                    snapshot = snapshot_builder.build(obs)
                    graph = edge_generator.build_graph(snapshot)
                    meta = graph.snapshot.static_meta
                    if independent_mode and model is None:
                        max_prg_candidates = max(resolve_full_candidate_width(graph), int(getattr(args, "independent_bc_max_candidates", 0)))
                        model = PFQIndependentReuseActor(
                            IndependentActorConfig(
                                hidden_dim=int(args.hidden_dim),
                                message_layers=int(args.message_layers),
                                dropout=float(args.dropout),
                                max_candidates=max_prg_candidates,
                                ablation=str(args.independent_ablation),
                            )
                        ).to(device)
                        initial_actor_sha256 = _model_tensor_sha256(model)
                        optimizer = torch.optim.AdamW(
                            model.parameters(),
                            lr=float(args.lr),
                            weight_decay=float(args.weight_decay),
                        )
                        print(
                            "[online-imitation] independent full candidate width "
                            f"resolved from static association mask: {max_prg_candidates}"
                        )
                        if args.independent_bc_dataset_out:
                            dataset_writer = IndependentBcDatasetWriter(
                                args.independent_bc_dataset_out,
                                initial_actor_sha256=initial_actor_sha256,
                                metadata={
                                    "feature_stage": runner.sim_env_overrides.get(
                                        "CUMAC_DPA_FEATURE_MODE",
                                        os.environ.get("CUMAC_DPA_FEATURE_MODE", ""),
                                    ),
                                    "headroom_accounting": "ctbs-v1",
                                    "teacher_policy": "pfq_passthrough",
                                    "teacher_alignment": "ctbs_prefix_prune_v1",
                                    "candidate_mode": "all_demand_ues",
                                    "include_pfq_score_anchor": False,
                                    "actor_family": model.actor_family,
                                    "collection_ablation": str(args.independent_ablation),
                                    "topology_seed_mode": runner.resolved_topology_seed_mode,
                                    "seed_list": _parse_seed_list(str(args.seed_list)),
                                    "topology_seed": int(args.topology_seed),
                                    "model_seed": int(args.seed),
                                    "expected_episode_count": int(args.episodes),
                                    "expected_steps_per_episode": int(max_steps),
                                    "expected_record_count": int(args.episodes) * int(max_steps),
                                },
                            )
                    if model is None or optimizer is None:
                        raise RuntimeError("model initialization failed")

                    teacher_out = teacher.label_graph(graph)
                    teacher_alignment = None
                    teacher_target_ue = teacher_out["target_ue_class"].to(device=device, dtype=torch.long)
                    teacher_target_prg = teacher_out["target_prg_class"].to(device=device, dtype=torch.long)

                    model.train(True)
                    out: Dict[str, torch.Tensor] = {}
                    if independent_mode:
                        if resolve_full_candidate_width(graph) > max_prg_candidates:
                            raise ValueError("BC candidate width would truncate this scene; set --independent-bc-max-candidates to paired-scene maximum")
                        packed = pack_prg_candidates(
                            graph,
                            max_candidates=max_prg_candidates,
                            fixed_width=True,
                        )
                        teacher_action_ue = teacher_out["action_ue_select"].to(
                            device=device, dtype=torch.long
                        )
                        teacher_action_prg = teacher_out["action_prg_alloc"].to(
                            device=device, dtype=torch.long
                        )
                        if dataset_writer is not None:
                            ctbs_inputs = _ctbs_inputs_for_imitation_step(runner, info)
                            if ctbs_inputs is None:
                                raise RuntimeError("dataset collection requires CTBS-v1 inputs")
                            teacher_action_prg, teacher_alignment = _align_ctbs_teacher(
                                graph,teacher_action_ue,teacher_action_prg,ctbs_inputs)
                            teacher_out["action_prg_alloc"]=teacher_action_prg
                            teacher_out["teacher_blank_ratio"]=(teacher_action_prg<0).float().mean()
                            teacher_trace = _canonical_independent_teacher_trace(
                                model,
                                graph,
                                packed,
                                teacher_action_ue,
                                teacher_action_prg,
                                ctbs_inputs,
                            )
                            replayed_logp, _replayed_entropy, _replayed_action = replay_independent_trace(
                                model,
                                graph,
                                packed,
                                teacher_trace,
                            )
                            loss = -replayed_logp
                        else:
                            teacher_trace = independent_teacher_trace_from_type0(
                                model,
                                graph,
                                packed,
                                teacher_action_ue,
                                teacher_action_prg,
                            )
                            loss = -teacher_trace.logp
                        prg_loss = loss
                        ue_loss = loss * 0.0
                        with torch.no_grad():
                            greedy_trace, _ = select_independent_action(
                                model,
                                graph,
                                packed,
                                mode="greedy",
                                ctbs_headroom_inputs=(
                                    teacher_trace.ctbs_headroom_inputs
                                    if dataset_writer is not None
                                    else None
                                ),
                            )
                            prg_acc = float(
                            (greedy_trace.choice == teacher_trace.choice.to(greedy_trace.choice.device)).float().mean().item()
                            )
                        ue_acc = 1.0
                        joint_acc = prg_acc
                        ue_legal_ratio = 1.0
                        prg_legal_ratio = 1.0
                        ue_bad_target_ratio = 0.0
                        prg_bad_target_ratio = 0.0
                    else:
                        out = model(graph)
                        ue_logits_raw = out["ue_logits"].unsqueeze(0)
                        prg_logits_raw = out["prg_logits"].unsqueeze(0)

                        action_mask_ue = (
                            meta.action_mask_ue
                            if meta.action_mask_ue is not None
                            else torch.ones((meta.n_ue,), dtype=torch.bool)
                        ).to(device=device, dtype=torch.bool).unsqueeze(0)
                        action_mask_cell_ue = meta.action_mask_cell_ue.to(device=device, dtype=torch.bool).unsqueeze(0)
                        action_mask_prg_cell = meta.action_mask_prg_cell.to(device=device, dtype=torch.bool).unsqueeze(0)

                        ue_logits, ue_valid = apply_ue_action_mask(
                            ue_logits_raw,
                            action_mask_ue,
                            n_cell=meta.n_cell,
                            action_mask_cell_ue=action_mask_cell_ue,
                        )
                        ue_target, ue_bad = sanitize_targets(
                            teacher_target_ue.unsqueeze(0),
                            ue_valid,
                        )
                        if "candidate_prg_logits" in out:
                            packed = pack_prg_candidates(
                                graph,
                                max_candidates=max(0, int(out["candidate_prg_logits"].shape[-1]) - 1),
                                fixed_width=True,
                            )
                            teacher_target_ue, teacher_target_prg = joint_candidate_targets_from_type0_actions(
                                graph,
                                packed,
                                action_ue_select=teacher_out["action_ue_select"].to(device=device, dtype=torch.long),
                                action_prg_alloc=teacher_out["action_prg_alloc"].to(device=device, dtype=torch.long),
                            )
                            teacher_target_ue = teacher_target_ue.to(device=device, dtype=torch.long)
                            teacher_target_prg = teacher_target_prg.to(device=device, dtype=torch.long)
                            ue_target, ue_bad = sanitize_targets(
                                teacher_target_ue.unsqueeze(0),
                                ue_valid,
                            )
                            candidate_valid, _selected_slot_mask, _selected_ue_mask = build_prg_candidate_valid_mask(
                                graph,
                                packed,
                                teacher_target_ue,
                            )
                            prg_valid = candidate_valid.to(device=device, dtype=torch.bool).unsqueeze(0)
                            prg_logits = out["candidate_prg_logits"].unsqueeze(0).masked_fill(~prg_valid, -1.0e9)
                        else:
                            selected_slot_mask = build_slot_selection_mask(
                                teacher_target_ue.unsqueeze(0),
                                action_mask_ue,
                                n_cell=meta.n_cell,
                                action_mask_cell_ue=action_mask_cell_ue,
                            )
                            prg_logits, prg_valid = apply_prg_action_mask(
                                prg_logits_raw,
                                meta.action_mask_prg_cell.to(device=device, dtype=torch.bool).reshape(1, meta.n_cell, meta.n_prg),
                                n_cell=meta.n_cell,
                                action_mask_cell_ue=action_mask_cell_ue,
                                action_mask_ue=action_mask_ue,
                                selected_slot_mask=selected_slot_mask,
                            )
                        prg_target, prg_bad = sanitize_targets(
                            teacher_target_prg.unsqueeze(0),
                            prg_valid,
                        )

                        ue_loss = F.cross_entropy(
                            ue_logits.reshape(-1, meta.n_ue + 1),
                            ue_target.reshape(-1),
                            ignore_index=-100,
                        )
                        prg_loss = F.cross_entropy(
                            prg_logits.reshape(-1, prg_logits.shape[-1]),
                            prg_target.reshape(-1),
                            ignore_index=-100,
                        )
                        loss = prg_loss + ue_loss_weight * ue_loss

                    if int(args.independent_bc_collect_only) == 0:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
                        optimizer.step()

                    if not independent_mode:
                        with torch.no_grad():
                            ue_acc = classification_accuracy(ue_logits, ue_target, ignore_index=-100)
                            prg_acc = classification_accuracy(prg_logits, prg_target, ignore_index=-100)
                            joint_acc = 0.5 * (ue_acc + prg_acc)
                            ue_legal_ratio = predicted_legal_ratio(ue_logits, ue_valid)
                            prg_legal_ratio = predicted_legal_ratio(prg_logits, prg_valid)
                            ue_bad_target_ratio = float(ue_bad.float().mean().item())
                            prg_bad_target_ratio = float(prg_bad.float().mean().item())

                    assert runner.env is not None
                    next_obs, reward_raw, done, next_info = runner.env.step(
                        teacher_out["action_ue_select"].cpu().numpy(),
                        teacher_out["action_prg_alloc"].cpu().numpy(),
                    )
                    runner.record_transition(next_obs, float(reward_raw), next_info)
                    if dataset_writer is not None:
                        slots=teacher_action_prg.detach().cpu().reshape(meta.n_prg,meta.n_cell)
                        selected=teacher_action_ue.detach().cpu()
                        chosen=torch.full_like(slots,-1)
                        valid=slots>=0; chosen[valid]=selected[slots[valid]]
                        parity=compare_ctbs_reference_labels(ctbs_inputs,
                            chosen_ue_per_prg=chosen.T.reshape(-1).tolist(),
                            actual_mcs=next_info['actual_ue_mcs'],actual_layer=next_info['actual_ue_layer'],
                            actual_tbs_capacity_bytes=next_info['actual_ue_tbs_capacity_bytes'],
                            n_cell=meta.n_cell,n_prg=meta.n_prg)
                        if not parity.all_exact:raise ValueError('aligned BC teacher CTBS post-action parity failed')
                        dataset_writer.write(
                            graph=graph,
                            teacher_trace=teacher_trace,
                            teacher_action_ue=teacher_action_ue,
                            teacher_action_prg=teacher_action_prg,
                            next_obs=next_obs,
                            next_info=next_info,
                            step_metadata={
                                "episode": episode_idx,
                                "episode_seed": episode_seed,
                                "topology_seed": topology_seed,
                                "step": step,
                                "global_step": global_step,
                                "tti": int(info["tti"]),
                                "done": bool(done),
                                "reward_raw": float(reward_raw),
                                "teacher_alignment": teacher_alignment,
                                "ctbs_post_action_exact": True,
                            },
                        )

                    prg_feat = snapshot.prg_features.values
                    row = {
                        "episode": episode_idx,
                        "episode_seed": episode_seed,
                        "topology_seed": topology_seed,
                        "step": step,
                        "global_step": global_step,
                        "tti": int(info["tti"]),
                        "loss": float(loss.item()),
                        "ue_loss": float(ue_loss.item()),
                        "prg_loss": float(prg_loss.item()),
                        "ue_acc": ue_acc,
                        "prg_acc": prg_acc,
                        "joint_acc": joint_acc,
                        "ue_legal_ratio": ue_legal_ratio,
                        "prg_legal_ratio": prg_legal_ratio,
                        "ue_bad_target_ratio": ue_bad_target_ratio,
                        "prg_bad_target_ratio": prg_bad_target_ratio,
                        "teacher_blank_ratio": float(teacher_out["teacher_blank_ratio"].item()),
                        "reward_raw": float(reward_raw),
                        "action_head_mode": str(args.action_head_mode),
                        "collect_only": int(args.independent_bc_collect_only),
                        "candidate_pfq_anchor_logit_mean": (
                            float(out["candidate_pfq_anchor_logits"].detach().mean().item())
                            if "candidate_pfq_anchor_logits" in out
                            and int(out["candidate_pfq_anchor_logits"].numel()) > 0
                            else 0.0
                        ),
                        "candidate_mode": (
                            "all_demand_ues"
                            if independent_mode
                            else graph.snapshot.static_meta.candidate_mode
                        ),
                        "post_eq_layer_dim": (
                            int(graph.snapshot.static_meta.post_eq_sinr.shape[-1])
                            if graph.snapshot.static_meta.post_eq_sinr is not None
                            and graph.snapshot.static_meta.post_eq_sinr.dim() == 3
                            else (1 if graph.snapshot.static_meta.post_eq_sinr is not None else 0)
                        ),
                        "n_edges_up": int(graph.edges["E_UP"].num_edges),
                        "n_edges_pp": int(graph.edges["E_PP"].num_edges),
                        "prg_prev_assigned_ratio": float(prg_feat[:, 2].mean().item()),
                        "prg_top1_sinr_db_mean": float(prg_feat[:, 0].mean().item()),
                        "prg_same_prg_conflict_ratio_mean": float(prg_feat[:, 6].mean().item()),
                        "prg_ici_proxy_mean": float(prg_feat[:, 7].mean().item()),
                    }
                    if prg_feat.shape[1] > 8:
                        row["prg_post_eq_p10_sinr_db_mean"] = float(prg_feat[:, 8].mean().item())
                    if prg_feat.shape[1] > 9:
                        row["prg_post_eq_mean_sinr_db_mean"] = float(prg_feat[:, 9].mean().item())
                    if prg_feat.shape[1] > 10:
                        row["prg_backlog_weighted_expected_goodput_mbps_mean"] = float(prg_feat[:, 10].mean().item())
                    if prg_feat.shape[1] > 11:
                        row["prg_low_sinr_hol_ttl_weight_mean"] = float(prg_feat[:, 11].mean().item())
                    f_jsonl.write(json.dumps(row, ensure_ascii=True) + "\n")
                    all_rows.append(row)
                    episode_rows.append(row)

                    if args.print_every > 0 and ((step % args.print_every) == 0 or step == max_steps - 1):
                        print(
                            f"[episode {episode_idx:04d}] step={step:04d} tti={row['tti']} "
                            f"loss={row['loss']:.4f} ue_acc={row['ue_acc']:.3f} prg_acc={row['prg_acc']:.3f} "
                            f"blank={row['teacher_blank_ratio']:.3f} reward={row['reward_raw']:.4f} "
                            f"mode={row['candidate_mode']} L={row['post_eq_layer_dim']} "
                            f"pfqA={row['candidate_pfq_anchor_logit_mean']:.3f} "
                            f"top1_mu={row['prg_top1_sinr_db_mean']:.2f} "
                            f"prev_on={row['prg_prev_assigned_ratio']:.3f}"
                        )

                    global_step += 1
                    step += 1
                    obs, info = next_obs, next_info
                    if done or step >= max_steps:
                        break

                mean_loss = float(np.mean([row["loss"] for row in episode_rows])) if episode_rows else 0.0
                mean_acc = float(np.mean([row["joint_acc"] for row in episode_rows])) if episode_rows else 0.0
                mean_blank = (
                    float(np.mean([row["teacher_blank_ratio"] for row in episode_rows])) if episode_rows else 0.0
                )
                mean_reward = float(np.mean([row["reward_raw"] for row in episode_rows])) if episode_rows else 0.0
                mean_prev = (
                    float(np.mean([row["prg_prev_assigned_ratio"] for row in episode_rows])) if episode_rows else 0.0
                )
                episode_summaries.append(
                    EpisodeSummary(
                        episode=episode_idx,
                        episode_seed=episode_seed,
                        topology_seed=topology_seed,
                        steps=len(episode_rows),
                        mean_loss=mean_loss,
                        mean_acc=mean_acc,
                        mean_teacher_blank_ratio=mean_blank,
                        mean_reward=mean_reward,
                        mean_prev_assigned_ratio=mean_prev,
                    )
                )

                checkpoint = {
                    "model_state": model.state_dict(),
                    "model_config": model.model_config_dict(),
                    "actor_state": model.state_dict(),
                    "actor_config": model.model_config_dict(),
                    "graph_config": (
                        {
                            "candidate_mode": "all_demand_ues",
                            "include_pfq_score_anchor": False,
                            "include_candidate_risk_features": False,
                        }
                        if independent_mode
                        else {
                            "candidate_mode": str(args.candidate_mode),
                            "include_pfq_score_anchor": bool(int(args.include_pfq_score_anchor)),
                            "include_candidate_risk_features": False,
                        }
                    ),
                    "train_args": vars(args),
                    "episode": episode_idx,
                    "global_step": global_step,
                    "mean_loss": mean_loss,
                    "mean_acc": mean_acc,
                    "initial_actor_sha256": initial_actor_sha256,
                    "collect_only": int(args.independent_bc_collect_only),
                    "teacher_collection": (
                        {
                            "policy": "pfq",
                            "demand_cap_slack_bytes": 0.0,
                            "reason": "match independent zero-slack hard headroom action space",
                        }
                        if independent_mode
                        else {}
                    ),
                }
                if int(args.save_every_episodes) > 0 and (episode_idx % int(args.save_every_episodes) == 0):
                    torch.save(checkpoint, ckpt_last_path)
                if mean_loss < best_mean_loss:
                    best_mean_loss = mean_loss
                    torch.save(checkpoint, ckpt_best_path)

    finally:
        runner.close_episode(graceful=False)

    if dataset_writer is not None:
        dataset_manifest = dataset_writer.finalize(max_candidates=max_prg_candidates)
        checkpoint["dataset"] = dataset_manifest
        checkpoint["initial_actor_sha256"] = initial_actor_sha256
        torch.save(checkpoint, ckpt_last_path)

    summary = {
        "episodes": int(args.episodes),
        "resolved_max_steps": int(max_steps),
        "resolved_topology_seed_mode": runner.resolved_topology_seed_mode,
        "device": str(device),
        "teacher_mode": args.teacher_mode,
        "action_head_mode": str(args.action_head_mode),
        "independent_ablation": str(args.independent_ablation),
        "actor_family": getattr(model, "actor_family", "legacy_stageb_hgraph"),
        "requires_pfq_features": bool(getattr(model, "requires_pfq_features", True)),
        "resolved_max_candidates": int(max_prg_candidates),
        "candidate_pfq_anchor_logit_coef": float(args.candidate_pfq_anchor_logit_coef),
        "sim_bin": args.sim_bin,
        "sim_args": args.sim_args,
        "sim_cwd": args.sim_cwd,
        "metrics_path": str(metrics_path),
        "last_checkpoint": str(ckpt_last_path),
        "best_checkpoint": str(ckpt_best_path),
        "best_mean_loss": float(best_mean_loss if np.isfinite(best_mean_loss) else 0.0),
        "initial_actor_sha256": initial_actor_sha256,
        "dataset": dataset_manifest or {},
        "dataset_mode": "online_collect" if dataset_manifest is not None else "disabled",
        "collect_only": int(args.independent_bc_collect_only),
        "episode_summaries": [asdict(ep) for ep in episode_summaries],
        "mean_loss_all": float(np.mean([row["loss"] for row in all_rows])) if all_rows else 0.0,
        "mean_ue_acc_all": float(np.mean([row["ue_acc"] for row in all_rows])) if all_rows else 0.0,
        "mean_prg_acc_all": float(np.mean([row["prg_acc"] for row in all_rows])) if all_rows else 0.0,
        "mean_acc_all": float(np.mean([row["joint_acc"] for row in all_rows])) if all_rows else 0.0,
        "mean_reward_all": float(np.mean([row["reward_raw"] for row in all_rows])) if all_rows else 0.0,
    }
    with open(summary_path, "w", encoding="utf-8") as f_summary:
        json.dump(summary, f_summary, ensure_ascii=True, indent=2)

    print(f"[online-imitation] wrote metrics: {metrics_path}")
    print(f"[online-imitation] wrote summary: {summary_path}")
    print(f"[online-imitation] last checkpoint: {ckpt_last_path}")
    print(f"[online-imitation] best checkpoint: {ckpt_best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

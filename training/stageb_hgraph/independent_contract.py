from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass
from typing import Mapping

import torch

from .feature_snapshot import actor_visible_snapshot_tensors
from .graph_ops import PackedPrgCandidates, demand_active_ue_mask_from_features
from .types import SparseEntityGraph

_INDEPENDENT_EDGE_ATTR_DIM = 5
W0_V2_VALID_LOGITS_ATOL = 1.0e-5
W0_V2_CHOSEN_LOGPROB_ATOL = 1.0e-6


@dataclass(frozen=True)
class RuntimeCandidateContract:
    runtime_candidate_width: int
    topology_candidate_width: int
    training_observed_max_candidates: int
    candidate_width_policy: str = "runtime_topology"
    candidate_truncation: str = "forbidden"


def _normalized_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(key).lower())


def strip_pfq_fields(observation: Mapping[str, object]) -> dict[str, object]:
    """Delete every PFQ/teacher-named observation field, including future ones."""
    stripped: dict[str, object] = {}
    for key, value in observation.items():
        normalized = _normalized_key(key)
        if "pfq" in normalized or "teacher" in normalized:
            continue
        stripped[key] = strip_pfq_fields(value) if isinstance(value, Mapping) else value
    return stripped


def _maximum_static_association(meta) -> int:
    if meta.n_cell <= 0:
        return 0
    return int(meta.action_mask_cell_ue.to(dtype=torch.int64).sum(dim=1).max().item())


def validate_independent_graph(
    graph: SparseEntityGraph,
    *,
    max_candidates: int,
) -> None:
    meta = graph.snapshot.static_meta
    expected_width = _maximum_static_association(meta)
    if int(max_candidates) < expected_width:
        raise ValueError(
            "max_candidates would truncate the maximum static association count: "
            f"got={int(max_candidates)} expected={expected_width}"
        )
    if "E_UP" not in graph.edges:
        raise ValueError("independent graph requires E_UP")
    edge = graph.edges["E_UP"]
    if edge.attr is None or edge.attr.dim() != 2:
        raise ValueError("independent E_UP requires explicit edge attributes")
    if int(edge.attr.shape[1]) != _INDEPENDENT_EDGE_ATTR_DIM:
        raise ValueError(
            "independent E_UP must contain five PFQ-free base columns; "
            f"got={int(edge.attr.shape[1])}, possible PFQ anchor/risk columns present"
        )

    demand = demand_active_ue_mask_from_features(
        graph.snapshot.ue_features.values,
        action_mask_ue=meta.action_mask_ue,
    )
    for cell_idx in range(meta.n_cell):
        expected = torch.nonzero(
            meta.action_mask_cell_ue[cell_idx].to(dtype=torch.bool) & demand,
            as_tuple=False,
        ).flatten()
        expected_ids = sorted(int(value) for value in expected.tolist())
        for local_prg in range(meta.n_prg):
            token_idx = cell_idx * meta.n_prg + local_prg
            selected = edge.index[0, edge.index[1] == token_idx]
            actual_ids = [int(value) for value in selected.tolist()]
            if not bool(meta.action_mask_prg_cell[cell_idx, local_prg].item()):
                if actual_ids:
                    raise ValueError(f"masked PRG token {token_idx} has candidates")
                continue
            if len(actual_ids) != len(set(actual_ids)):
                raise ValueError(f"duplicate UE candidate at PRG token {token_idx}")
            if sorted(actual_ids) != expected_ids:
                raise ValueError(
                    f"incomplete independent candidates at PRG token {token_idx}: "
                    f"actual={sorted(actual_ids)} expected={expected_ids}"
                )


def resolve_runtime_candidate_contract(
    graph: SparseEntityGraph,
    actor_config: Mapping[str, object],
    *,
    requested_width: int = 0,
) -> RuntimeCandidateContract:
    topology_width = _maximum_static_association(graph.snapshot.static_meta)
    requested = int(requested_width)
    if requested < 0:
        raise ValueError("requested runtime candidate width must be non-negative")
    runtime_width = topology_width if requested == 0 else requested
    if runtime_width < topology_width:
        raise ValueError(
            "runtime candidate width would truncate topology candidates: "
            f"got={runtime_width} topology={topology_width}"
        )
    training_width = int(
        actor_config.get(
            "training_observed_max_candidates",
            actor_config.get("max_candidates", 0),
        )
    )
    return RuntimeCandidateContract(
        runtime_candidate_width=runtime_width,
        topology_candidate_width=topology_width,
        training_observed_max_candidates=training_width,
    )


def candidate_truncation_count(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
) -> int:
    edge = graph.edges.get("E_UP")
    if edge is None:
        raise ValueError("independent graph requires E_UP")
    expected = torch.zeros(
        (graph.snapshot.n_prg_tokens,),
        dtype=torch.int64,
        device=edge.index.device,
    )
    if edge.index.numel() > 0:
        destination = edge.index[1].to(dtype=torch.int64)
        expected.scatter_add_(0, destination, torch.ones_like(destination))
    materialized = packed.mask.to(device=expected.device, dtype=torch.int64).sum(dim=1)
    return int(torch.clamp_min(expected - materialized, 0).sum().item())


def _state_tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = torch.as_tensor(tensor).detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.int16).numpy().tobytes(order="C")
    return value.numpy().tobytes(order="C")


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash sorted tensor names, dtypes, shapes, and raw bytes only."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = torch.as_tensor(state_dict[name]).detach().cpu().contiguous()
        encoded_name = name.encode("utf-8")
        encoded_dtype = str(tensor.dtype).encode("ascii")
        digest.update(struct.pack("<I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack("<I", len(encoded_dtype)))
        digest.update(encoded_dtype)
        digest.update(struct.pack("<I", tensor.dim()))
        for size in tensor.shape:
            digest.update(struct.pack("<q", int(size)))
        digest.update(_state_tensor_bytes(tensor))
    return digest.hexdigest()


def state_dicts_bitwise_equal(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> bool:
    if set(left) != set(right):
        return False
    for name in sorted(left):
        left_tensor = torch.as_tensor(left[name]).detach().cpu().contiguous()
        right_tensor = torch.as_tensor(right[name]).detach().cpu().contiguous()
        if left_tensor.dtype != right_tensor.dtype or left_tensor.shape != right_tensor.shape:
            return False
        if _state_tensor_bytes(left_tensor) != _state_tensor_bytes(right_tensor):
            return False
    return True


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def validate_independent_checkpoint(checkpoint: Mapping[str, object]) -> None:
    actor_cfg = _mapping(checkpoint.get("actor_config"), "actor_config")
    graph_cfg = _mapping(checkpoint.get("graph_config"), "graph_config")
    if actor_cfg.get("actor_family") != "pfq_independent_reuse_v1":
        raise ValueError("checkpoint actor_family is not pfq_independent_reuse_v1")
    if bool(actor_cfg.get("requires_pfq_features", True)):
        raise ValueError("independent actor requires_pfq_features must be false")
    if graph_cfg.get("candidate_mode") != "all_demand_ues":
        raise ValueError("independent candidate_mode must be all_demand_ues")
    if bool(graph_cfg.get("include_pfq_score_anchor", True)):
        raise ValueError("independent graph cannot include PFQ score anchors")
    if bool(graph_cfg.get("include_candidate_risk_features", True)):
        raise ValueError("independent graph cannot include candidate risk extensions")
    for key in checkpoint:
        normalized = _normalized_key(key)
        if "pfq" in normalized and "residual" in normalized:
            raise ValueError(f"independent checkpoint contains PFQ residual configuration: {key}")


def _assert_tensor_dict_equal(
    label: str,
    a: Mapping[str, torch.Tensor],
    b: Mapping[str, torch.Tensor],
) -> None:
    if set(a) != set(b):
        raise AssertionError(f"{label} tensor keys differ: {sorted(a)} != {sorted(b)}")
    for key in sorted(a):
        if not torch.equal(a[key], b[key]):
            raise AssertionError(f"{label}.{key} differs")


def assert_graph_tensor_equality(a: SparseEntityGraph, b: SparseEntityGraph) -> None:
    a_snapshot = a.snapshot
    b_snapshot = b.snapshot
    structural_a = (
        a_snapshot.n_cell,
        a_snapshot.n_ue,
        a_snapshot.n_prg,
        a_snapshot.static_meta.n_sched_ue,
        a_snapshot.static_meta.topology,
        a_snapshot.cell_features.names,
        a_snapshot.ue_features.names,
        a_snapshot.prg_features.names,
    )
    structural_b = (
        b_snapshot.n_cell,
        b_snapshot.n_ue,
        b_snapshot.n_prg,
        b_snapshot.static_meta.n_sched_ue,
        b_snapshot.static_meta.topology,
        b_snapshot.cell_features.names,
        b_snapshot.ue_features.names,
        b_snapshot.prg_features.names,
    )
    if structural_a != structural_b:
        raise AssertionError(f"graph structure differs: {structural_a} != {structural_b}")
    _assert_tensor_dict_equal(
        "snapshot",
        actor_visible_snapshot_tensors(a_snapshot),
        actor_visible_snapshot_tensors(b_snapshot),
    )
    if set(a.edges) != set(b.edges):
        raise AssertionError(f"edge sets differ: {sorted(a.edges)} != {sorted(b.edges)}")
    for name in sorted(a.edges):
        edge_a = a.edges[name]
        edge_b = b.edges[name]
        if (edge_a.name, edge_a.src_type, edge_a.dst_type) != (
            edge_b.name,
            edge_b.src_type,
            edge_b.dst_type,
        ):
            raise AssertionError(f"edge metadata differs for {name}")
        if not torch.equal(edge_a.index, edge_b.index):
            raise AssertionError(f"edge index differs for {name}")
        if (edge_a.attr is None) != (edge_b.attr is None):
            raise AssertionError(f"edge attr presence differs for {name}")
        if edge_a.attr is not None and not torch.equal(edge_a.attr, edge_b.attr):
            raise AssertionError(f"edge attr differs for {name}")


__all__ = [
    "W0_V2_CHOSEN_LOGPROB_ATOL",
    "W0_V2_VALID_LOGITS_ATOL",
    "assert_graph_tensor_equality",
    "state_dict_sha256",
    "state_dicts_bitwise_equal",
    "strip_pfq_fields",
    "validate_independent_checkpoint",
    "validate_independent_graph",
]

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from training.stageb_hgraph import independent_contract as independent_contract_module
from training.stageb_hgraph.edge_generator import EdgeGeneratorConfig, SparseEdgeGenerator
from training.stageb_hgraph.feature_snapshot import FeatureSnapshotBuilder
from training.stageb_hgraph.graph_ops import pack_prg_candidates
from training.stageb_hgraph.independent_contract import (
    assert_graph_tensor_equality,
    strip_pfq_fields,
    validate_independent_checkpoint,
    validate_independent_graph,
)
from training.stageb_hgraph.tests.graph_fixtures import make_candidate_observation


def _independent_graph(obs: dict[str, object]):
    snapshot = FeatureSnapshotBuilder().build(obs)
    return SparseEdgeGenerator(
        EdgeGeneratorConfig(
            candidate_mode="all_demand_ues",
            include_pfq_score_anchor=False,
            include_candidate_risk_features=False,
        )
    ).build_graph(snapshot)


def test_full_candidates_cover_each_active_associated_ue_once_without_anchor() -> None:
    obs = make_candidate_observation(n_prg=3, ue_feat_dim=16)
    obs["action_mask_cell_ue"] = np.asarray(
        [[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 1]],
        dtype=np.uint8,
    )
    obs["obs_ue_features"][1, 0] = 0.0
    obs["obs_ue_features"][1, 6] = 0.0
    generator = SparseEdgeGenerator(
        EdgeGeneratorConfig(
            candidate_mode="all_demand_ues",
            include_pfq_score_anchor=False,
            include_candidate_risk_features=False,
        )
    )
    graph = generator.build_graph(FeatureSnapshotBuilder().build(obs))

    validate_independent_graph(graph, max_candidates=3)
    assert not any("pfq" in name for name in generator.ue_prg_edge_feature_names)
    edge = graph.edges["E_UP"]
    assert edge.attr is not None
    assert edge.attr.shape[1] == 5
    for cell in range(3):
        expected = {
            ue
            for ue in range(6)
            if bool(obs["action_mask_cell_ue"][cell, ue])
            and bool(obs["action_mask_ue"][ue])
            and (
                float(obs["obs_ue_features"][ue, 0]) >= 1.0
                or float(obs["obs_ue_features"][ue, 6]) > 0.5
            )
        }
        for local_prg in range(3):
            token = cell * 3 + local_prg
            actual = edge.index[0, edge.index[1] == token].tolist()
            assert len(actual) == len(set(actual))
            assert set(actual) == expected


def test_no_demand_leaves_blank_as_only_valid_class() -> None:
    obs = make_candidate_observation(
        n_prg=2,
        backlog_bytes=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    obs["obs_ue_features"][:, 6] = 0.0
    graph = _independent_graph(obs)

    validate_independent_graph(graph, max_candidates=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    assert packed.max_candidates == 2
    assert not bool(packed.mask.any().item())
    assert torch.equal(packed.counts, torch.zeros_like(packed.counts))
    assert packed.blank_class_index == 2


def test_candidate_width_cannot_be_smaller_than_static_association() -> None:
    graph = _independent_graph(make_candidate_observation(n_prg=2))

    validate_independent_graph(graph, max_candidates=2)
    validate_independent_graph(graph, max_candidates=3)
    with pytest.raises(ValueError, match="max_candidates"):
        validate_independent_graph(graph, max_candidates=1)


def test_runtime_width_uses_full_topology_and_legacy_width_is_training_metadata() -> None:
    resolver = getattr(independent_contract_module, "resolve_runtime_candidate_contract", None)
    assert callable(resolver), "runtime topology candidate contract is missing"
    graph = _independent_graph(make_candidate_observation(n_prg=2))

    automatic = resolver(graph, {"max_candidates": 13})
    padded = resolver(graph, {"max_candidates": 13}, requested_width=3)

    assert automatic.runtime_candidate_width == 2
    assert automatic.topology_candidate_width == 2
    assert automatic.training_observed_max_candidates == 13
    assert automatic.candidate_width_policy == "runtime_topology"
    assert automatic.candidate_truncation == "forbidden"
    assert padded.runtime_candidate_width == 3
    with pytest.raises(ValueError, match="topology|truncat"):
        resolver(graph, {"max_candidates": 13}, requested_width=1)


def test_candidate_truncation_counter_detects_dropped_edges() -> None:
    counter = getattr(independent_contract_module, "candidate_truncation_count", None)
    assert callable(counter), "candidate truncation audit is missing"
    graph = _independent_graph(make_candidate_observation(n_prg=2))
    full = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    truncated = pack_prg_candidates(graph, max_candidates=1, fixed_width=True)

    assert counter(graph, full) == 0
    assert counter(graph, truncated) == 6


def test_state_dict_identity_uses_sorted_tensor_content_not_serialization() -> None:
    hasher = getattr(independent_contract_module, "state_dict_sha256", None)
    equal = getattr(independent_contract_module, "state_dicts_bitwise_equal", None)
    assert callable(hasher)
    assert callable(equal), "bitwise state-dict value comparison is missing"
    left = {
        "z.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "a.bias": torch.tensor([3, 4], dtype=torch.int16),
    }
    reordered = {
        "a.bias": left["a.bias"].clone(),
        "z.weight": left["z.weight"].clone(),
    }
    different_dtype = {
        "a.bias": left["a.bias"].to(torch.int32),
        "z.weight": left["z.weight"].clone(),
    }
    different_shape = {
        "a.bias": left["a.bias"].clone(),
        "z.weight": left["z.weight"].reshape(2, 1),
    }
    different_value = {
        "a.bias": left["a.bias"].clone(),
        "z.weight": torch.tensor([[1.0, 2.5]], dtype=torch.float32),
    }

    assert hasher(left) == hasher(reordered)
    assert equal(left, reordered)
    for changed in (different_dtype, different_shape, different_value):
        assert hasher(left) != hasher(changed)
        assert not equal(left, changed)


def test_deleting_all_pfq_and_teacher_fields_preserves_actor_graph_tensors() -> None:
    obs = make_candidate_observation(n_prg=2, ue_feat_dim=16)
    obs["teacher_action_ue_select"] = np.arange(6, dtype=np.int32)
    obs["teacher_action_prg_alloc"] = np.zeros((6,), dtype=np.int16)
    obs["PFQ-future.score"] = np.arange(6, dtype=np.float32)
    obs["future_teacher_margin"] = np.arange(6, dtype=np.float32) + 100.0

    full_graph = _independent_graph(obs)
    stripped = strip_pfq_fields(obs)
    stripped_graph = _independent_graph(stripped)

    assert "teacher_action_ue_select" not in stripped
    assert "teacher_action_prg_alloc" not in stripped
    assert "PFQ-future.score" not in stripped
    assert "future_teacher_margin" not in stripped
    assert_graph_tensor_equality(full_graph, stripped_graph)


def test_contract_rejects_topk_anchor_residual_or_pfq_requirement() -> None:
    valid = {
        "actor_config": {
            "actor_family": "pfq_independent_reuse_v1",
            "requires_pfq_features": False,
        },
        "graph_config": {
            "candidate_mode": "all_demand_ues",
            "include_pfq_score_anchor": False,
            "include_candidate_risk_features": False,
        },
    }
    validate_independent_checkpoint(valid)

    invalid = []
    for key, value in (
        ("candidate_mode", "topk_diversity"),
        ("include_pfq_score_anchor", True),
    ):
        checkpoint = deepcopy(valid)
        checkpoint["graph_config"][key] = value
        invalid.append(checkpoint)
    checkpoint = deepcopy(valid)
    checkpoint["actor_config"]["requires_pfq_features"] = True
    invalid.append(checkpoint)
    checkpoint = deepcopy(valid)
    checkpoint["pfq_residual_config"] = {"enabled": True}
    invalid.append(checkpoint)

    for checkpoint in invalid:
        with pytest.raises(ValueError):
            validate_independent_checkpoint(checkpoint)


def test_graph_validator_rejects_non_independent_generator_configuration() -> None:
    obs = make_candidate_observation(n_prg=2)
    snapshot = FeatureSnapshotBuilder().build(obs)
    graph = SparseEdgeGenerator(
        EdgeGeneratorConfig(
            candidate_mode="all_demand_ues",
            include_pfq_score_anchor=True,
        )
    ).build_graph(snapshot)
    with pytest.raises(ValueError, match="PFQ|anchor"):
        validate_independent_graph(graph, max_candidates=2)

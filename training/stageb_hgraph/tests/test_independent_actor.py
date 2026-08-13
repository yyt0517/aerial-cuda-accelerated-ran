from __future__ import annotations

import io

import numpy as np
import torch

from training.stageb_hgraph.edge_generator import EdgeGeneratorConfig, SparseEdgeGenerator
from training.stageb_hgraph.feature_snapshot import FeatureSnapshotBuilder
from training.stageb_hgraph.graph_ops import pack_prg_candidates
from training.stageb_hgraph.independent_actor import (
    IndependentActorConfig,
    PFQIndependentReuseActor,
)


def _canonical_graph():
    n_cell, n_ue, n_prg = 3, 36, 17
    ue = np.zeros((n_ue, 16), dtype=np.float32)
    ue[:, 0] = 3000.0
    ue[:, 1] = 1.0
    ue[:, 2] = np.linspace(2.0, 20.0, n_ue, dtype=np.float32)
    ue[:, 3] = 10.0
    ue[:, 4] = 1.0
    ue[:, 6] = 1.0
    association = np.zeros((n_cell, n_ue), dtype=np.uint8)
    for cell in range(n_cell):
        association[cell, cell * 12 : (cell + 1) * 12] = 1
    obs = {
        "obs_cell_features": np.zeros((n_cell, 5), dtype=np.float32),
        "obs_ue_features": ue,
        "obs_prg_features": np.zeros((n_cell, n_prg, 12), dtype=np.float32),
        "action_mask_ue": np.ones((n_ue,), dtype=np.uint8),
        "action_mask_cell_ue": association,
        "action_mask_prg_cell": np.ones((n_cell, n_prg), dtype=np.uint8),
        "post_eq_sinr": np.full((n_ue, n_prg, 1), 10.0, dtype=np.float32),
        "n_sched_ue": np.int32(n_ue),
    }
    snapshot = FeatureSnapshotBuilder().build(obs)
    return SparseEdgeGenerator(
        EdgeGeneratorConfig(
            candidate_mode="all_demand_ues",
            include_pfq_score_anchor=False,
            include_candidate_risk_features=False,
        )
    ).build_graph(snapshot)


def _actor(hidden_dim: int = 32) -> PFQIndependentReuseActor:
    torch.manual_seed(9)
    return PFQIndependentReuseActor(
        IndependentActorConfig(hidden_dim=hidden_dim, message_layers=1, max_candidates=12)
    )


def test_encode_shapes_attention_heads_and_finite_gradients() -> None:
    graph = _canonical_graph()
    actor = _actor()

    encoded = actor.encode_graph(graph)

    assert encoded["cell_embeddings"].shape == (3, 32)
    assert encoded["ue_embeddings"].shape == (36, 32)
    assert encoded["prg_embeddings"].shape == (51, 32)
    assert encoded["reuse_embeddings"].shape == (17, 3, 32)
    assert actor.reuse_attention.num_heads == 4
    packed = pack_prg_candidates(graph, max_candidates=12, fixed_width=True)
    logits = actor.score_reuse_group(
        encoded,
        graph,
        packed,
        local_prg=0,
        sequential_features=torch.zeros((3, 12, 3)),
    )
    loss = sum(value.square().mean() for value in encoded.values()) + logits.square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in actor.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)


def test_reuse_attention_is_same_local_prg_only_and_cross_cell() -> None:
    actor = _actor(hidden_dim=16).eval()
    base = torch.randn((3 * 4, 16), generator=torch.Generator().manual_seed(3))
    original = actor.apply_reuse_attention(base, n_cell=3, n_prg=4)

    same_prg_other_cell = base.clone()
    same_prg_other_cell[1 * 4 + 2] += 5.0
    cross_changed = actor.apply_reuse_attention(same_prg_other_cell, n_cell=3, n_prg=4)
    assert not torch.allclose(original[2, 0], cross_changed[2, 0])

    different_prg = base.clone()
    different_prg[1 * 4 + 3] += 5.0
    local_unchanged = actor.apply_reuse_attention(different_prg, n_cell=3, n_prg=4)
    assert torch.equal(original[2], local_unchanged[2])


def test_score_group_returns_unmasked_ue_and_blank_logits() -> None:
    graph = _canonical_graph()
    actor = _actor()
    encoded = actor.encode_graph(graph)
    packed = pack_prg_candidates(graph, max_candidates=12, fixed_width=True)
    sequential = torch.zeros((3, 12, 3), dtype=torch.float32)
    sequential[:, :, 0] = 1.0

    logits = actor.score_reuse_group(
        encoded,
        graph,
        packed,
        local_prg=5,
        sequential_features=sequential,
    )

    assert logits.shape == (3, 13)
    assert bool(torch.isfinite(logits).all().item())


def test_actor_has_no_legacy_correction_heads_and_reports_checkpoint_metadata() -> None:
    actor = _actor()
    forbidden = ("pfq", "proposal", "residual_action", "fallback")
    internal_names = [name.lower() for name, _ in actor.named_modules() if name]
    internal_names += [name.lower() for name, _ in actor.named_parameters()]
    assert not any(token in name for name in internal_names for token in forbidden)

    config = actor.model_config_dict()
    assert config["actor_family"] == "pfq_independent_reuse_v1"
    assert config["requires_pfq_features"] is False
    assert config["attention_heads"] == 4
    assert config["max_candidates"] == 12

    parameter_count = sum(parameter.numel() for parameter in actor.parameters())
    payload = io.BytesIO()
    torch.save({"actor_config": config, "actor_state": actor.state_dict()}, payload)
    assert parameter_count > 0
    assert len(payload.getvalue()) > 0

from __future__ import annotations

import numpy as np
import torch

from training.stageb_hgraph.edge_generator import EdgeGeneratorConfig, SparseEdgeGenerator
from training.stageb_hgraph.feature_snapshot import FeatureSnapshotBuilder
from training.stageb_hgraph.independent_actor import IndependentActorConfig, PFQIndependentReuseActor
from training.stageb_hgraph.independent_contract import assert_graph_tensor_equality, strip_pfq_fields
from training.stageb_hgraph.independent_policy import select_independent_action
from training.stageb_hgraph.graph_ops import pack_prg_candidates
from training.stageb_hgraph.tests.graph_fixtures import make_candidate_observation


def _graph(obs):
    return SparseEdgeGenerator(
        EdgeGeneratorConfig(
            candidate_mode="all_demand_ues",
            include_pfq_score_anchor=False,
            include_candidate_risk_features=False,
        )
    ).build_graph(FeatureSnapshotBuilder().build(obs))


def test_pfq_field_deletion_is_bitwise_equal_through_logits_trace_and_type0() -> None:
    torch.manual_seed(23)
    obs = make_candidate_observation(n_prg=3, ue_feat_dim=16)
    obs["teacher_action_ue_select"] = np.arange(6, dtype=np.int32)
    obs["teacher_action_prg_alloc"] = np.zeros((9,), dtype=np.int16)
    obs["pfq_score"] = np.arange(6, dtype=np.float32)
    full = _graph(obs)
    deleted = _graph(strip_pfq_fields(obs))
    assert_graph_tensor_equality(full, deleted)

    actor = PFQIndependentReuseActor(
        IndependentActorConfig(
            hidden_dim=16,
            message_layers=1,
            max_candidates=2,
            up_edge_attr_dim=5,
        )
    ).eval()
    packed_full = pack_prg_candidates(full, max_candidates=2, fixed_width=True)
    packed_deleted = pack_prg_candidates(deleted, max_candidates=2, fixed_width=True)
    encoded_full = actor.encode_graph(full)
    encoded_deleted = actor.encode_graph(deleted)
    for key in encoded_full:
        assert torch.equal(encoded_full[key], encoded_deleted[key])

    trace_full, action_full = select_independent_action(
        actor, full, packed_full, mode="greedy"
    )
    trace_deleted, action_deleted = select_independent_action(
        actor, deleted, packed_deleted, mode="greedy"
    )
    assert torch.equal(trace_full.choice, trace_deleted.choice)
    assert torch.equal(trace_full.valid_at_step, trace_deleted.valid_at_step)
    assert torch.equal(trace_full.logp, trace_deleted.logp)
    assert torch.equal(action_full.action_ue_select, action_deleted.action_ue_select)
    assert torch.equal(action_full.action_prg_alloc, action_deleted.action_prg_alloc)

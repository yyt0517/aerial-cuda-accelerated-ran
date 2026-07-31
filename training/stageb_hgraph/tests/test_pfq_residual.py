import importlib

import torch

from training.stageb_hgraph.model import HGraphModelConfig, StageBHGraphPolicy, build_model_from_config
from training.stageb_hgraph.tests.graph_fixtures import make_candidate_graph


def test_zero_initialized_residual_main_logits_equal_pfq_base():
    pfq_residual = importlib.import_module("training.stageb_hgraph.pfq_residual")
    graph = make_candidate_graph(n_prg=2)
    actor = StageBHGraphPolicy(
        HGraphModelConfig(
            ue_feat_dim=12,
            action_head_mode="candidate",
            max_prg_candidates=2,
            candidate_success_logit_scale=0.0,
            candidate_blank_success_scale=0.0,
            candidate_pfq_anchor_logit_coef=1.0,
            candidate_blank_logit_bias=-4.0,
        )
    )

    pfq_residual.initialize_pfq_residual_head(actor)
    out = actor(graph)
    valid = out["candidate_mask"]

    assert torch.allclose(
        out["candidate_residual_logits"][valid],
        torch.zeros_like(out["candidate_residual_logits"][valid]),
        atol=0.0,
    )
    assert torch.allclose(
        out["candidate_prg_logits"][:, :-1][valid],
        out["candidate_pfq_base_logits"][valid],
        atol=1e-6,
    )
    assert torch.allclose(
        out["candidate_prg_logits"][:, -1],
        torch.full_like(out["candidate_prg_logits"][:, -1], -4.0),
        atol=0.0,
    )


def test_residual_initialization_is_applied_exactly_once():
    pfq_residual = importlib.import_module("training.stageb_hgraph.pfq_residual")

    assert pfq_residual.should_initialize_pfq_residual(None, enabled=True) is True
    assert pfq_residual.should_initialize_pfq_residual({}, enabled=True) is True
    assert (
        pfq_residual.should_initialize_pfq_residual(
            {"initialization": {"mode": "legacy"}},
            enabled=True,
        )
        is True
    )
    assert (
        pfq_residual.should_initialize_pfq_residual(
            {"initialization": {"mode": "pfq_residual"}},
            enabled=True,
        )
        is False
    )
    assert pfq_residual.should_initialize_pfq_residual(None, enabled=False) is False


def test_legacy_candidate_checkpoint_loads_with_output_aliases():
    graph = make_candidate_graph(n_prg=2)
    original = StageBHGraphPolicy(
        HGraphModelConfig(
            action_head_mode="candidate",
            max_prg_candidates=2,
            candidate_pfq_anchor_logit_coef=0.0,
        )
    )
    legacy_config = original.model_config_dict()
    legacy_config.pop("candidate_pfq_anchor_logit_coef")
    restored = build_model_from_config(legacy_config)

    restored.load_state_dict(original.state_dict(), strict=True)
    out = restored(graph)

    assert torch.equal(
        out["candidate_residual_logits"],
        out["candidate_service_value_logits"],
    )
    assert torch.equal(
        out["candidate_pfq_base_logits"],
        out["candidate_pfq_anchor_logits"],
    )

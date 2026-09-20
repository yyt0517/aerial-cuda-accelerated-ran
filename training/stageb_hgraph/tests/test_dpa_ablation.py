from copy import deepcopy

import pytest
import torch

from training.stageb_hgraph.independent_actor import IndependentActorConfig, PFQIndependentReuseActor
from training.stageb_hgraph.tests.test_independent_actor import _canonical_graph


def actor(variant):
    torch.manual_seed(701)
    return PFQIndependentReuseActor(IndependentActorConfig(hidden_dim=32, ablation=variant))


def test_removed_attention_cannot_transfer_other_cell_information():
    model = actor("no_reuse_attention")
    x = torch.randn(51, 32)
    before = model.apply_reuse_attention(x, 3, 17)
    changed = x.clone()
    changed[17:34] = torch.randn(17, 32) * 10
    after = model.apply_reuse_attention(changed, 3, 17)
    assert torch.equal(before[:, 0], after[:, 0])
    assert not torch.equal(before[:, 1], after[:, 1])
    before.square().sum().backward()
    assert all(p.grad is None and not p.requires_grad for p in model.reuse_attention.parameters())


def test_removed_deadline_features_cannot_change_any_encoded_output():
    graph = _canonical_graph()
    changed = deepcopy(graph)
    changed.snapshot.ue_features.values[:, 12:15] = torch.rand(36, 3) * 7
    model = actor("no_dimensionless")
    a, b = model.encode_graph(graph), model.encode_graph(changed)
    assert all(torch.equal(a[k], b[k]) for k in a)
    full = actor("full")
    assert not torch.equal(full.encode_graph(graph)["ue_embeddings"], full.encode_graph(changed)["ue_embeddings"])
    assert torch.count_nonzero(changed.snapshot.ue_features.values[:, 12:15]) == 108


def test_paired_random_initialization_and_checkpoint_roundtrip(tmp_path):
    from training.stageb_hgraph.eval_checkpoint_online import _load_actor_checkpoint

    full = actor("full")
    for variant in ("no_reuse_attention", "no_dimensionless"):
        model = actor(variant)
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in full.state_dict().items())
        path = tmp_path / f"{variant}.pt"
        torch.save({"actor_config": model.model_config_dict(), "actor_state": model.state_dict(),
                    "graph_config": {"candidate_mode": "all_demand_ues", "include_pfq_score_anchor": False,
                                     "include_candidate_risk_features": False}}, path)
        restored, _ = _load_actor_checkpoint(path, torch.device("cpu"))
        assert restored.cfg.ablation == variant
        a, b = model.encode_graph(_canonical_graph()), restored.encode_graph(_canonical_graph())
        assert all(torch.equal(a[k], b[k]) for k in a)


def test_unknown_ablation_is_rejected():
    with pytest.raises(ValueError, match="ablation"):
        actor("typo")


def test_ppo_optimizer_keeps_removed_branch_frozen():
    from training.stageb_hgraph.online_ppo import _configure_actor_trainable_params
    model = actor("no_reuse_attention")
    _configure_actor_trainable_params(model)
    assert all(not p.requires_grad for p in model.reuse_attention.parameters())


def test_campaign_commands_have_equal_budget_and_explicit_random_init(monkeypatch, tmp_path):
    import sys
    from training.stageb_hgraph.dpa_ablation_experiment import train_command
    from training.stageb_hgraph.online_ppo import parse_args
    for variant in ("full", "no_reuse_attention", "no_dimensionless"):
        command = train_command(tmp_path / "sim", tmp_path / variant, variant, 701, smoke=False)
        monkeypatch.setattr(sys, "argv", ["online_ppo", *command[3:]])
        args = parse_args()
        assert args.independent_ablation == variant
        assert args.independent_random_init == 1 and not args.init_checkpoint
        assert args.iterations == 48 and args.rollout_steps == 512
        assert args.rollout_warmup_steps == 1000 and args.rollout_warmup_once == 0
        assert args.update_epochs == 4 and args.minibatch_size == 64
        assert args.target_kl == 0 and args.early_stop_patience == 0


def test_statistics_preserve_training_seed_as_replication_unit():
    from training.stageb_hgraph.dpa_ablation_experiment import paired_deltas
    import numpy as np
    result = paired_deltas(np.asarray([[1.] * 5, [2.] * 5, [3.] * 5]))
    assert result["training_seed_means"] == [1., 2., 3.]
    assert result["mean_delta"] == 2.
    assert result["n_training_replicates"] == 3
    assert result["conditional_ci95"][0] < 0


def test_evaluation_audit_rejects_a_single_illegal_action():
    from training.stageb_hgraph.dpa_ablation_experiment import audit_eval
    summary = {"independent_deletion_equal": True, "candidate_full_coverage": True,
               "ctbs_reference_label_deletion_equal": True,
               "ctbs_mcs_mismatch_count": 0, "ctbs_layer_mismatch_count": 0,
               "ctbs_tbs_mismatch_count": 0}
    row = {"illegal_action_count": 1, "candidate_truncation_count": 0,
           "action_replay_mismatch_count": 0, "native_reject_count": 0}
    with pytest.raises(ValueError, match="illegal_action_count"):
        audit_eval(summary, [row])
    row["illegal_action_count"] = 0
    audit_eval(summary, [row])


def test_reference_name_selects_the_actual_native_scheduler():
    import shlex
    from training.stageb_hgraph.dpa_ablation_experiment import REFERENCE_SIM_ARGS
    for policy, mode in (("pfq", "2"), ("rrq", "3")):
        args = shlex.split(REFERENCE_SIM_ARGS[policy])
        assert args[args.index("-b") + 1] == mode


def test_reference_trace_starts_at_tti_one(monkeypatch, tmp_path):
    from training.stageb_hgraph import dpa_ablation_experiment as experiment
    captured = []
    monkeypatch.setattr(experiment, "run_command", lambda command, *args: captured.append(command))
    experiment.evaluate(tmp_path / "sim", tmp_path / "pfq", None, 201, {}, True, "pfq")
    command = captured[0]
    assert command[command.index("--service-balance-trace-first-tti") + 1] == "1"

import argparse
from collections import defaultdict
import shutil
import subprocess
import sys
from dataclasses import MISSING, fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from training.stageb_hgraph import eval_checkpoint_online, online_ppo
from training.stageb_hgraph.candidate_policy import (
    CandidatePolicyConfig,
    select_candidate_policy_action,
)
from training.stageb_hgraph.graph_ops import (
    materialize_candidate_choices_to_type0,
    pack_prg_candidates,
)
from training.stageb_hgraph.model import HGraphModelConfig, StageBHGraphPolicy
from training.stageb_hgraph.online_ppo import (
    GraphValueHead,
    RolloutStep,
    _direct_candidate_logp_entropy,
    _direct_candidate_logp_entropy_batch,
    _load_checkpoint_compatible,
    _ppo_update,
    _sample_joint_action,
    _validate_candidate_policy_mode,
    parse_args,
)
from training.stageb_hgraph.tests.graph_fixtures import make_candidate_graph


def test_direct_evaluation_bypasses_legacy_decode_and_materializes_greedy_choices(
    monkeypatch,
):
    graph = make_candidate_graph(n_prg=2)
    candidate_logits = torch.tensor(
        [
            [9.0, 1.0, -9.0],
            [8.0, 7.0, -9.0],
            [9.0, 1.0, -9.0],
            [8.0, 7.0, -9.0],
            [9.0, 1.0, -9.0],
            [8.0, 7.0, -9.0],
        ]
    )

    class DirectEvalActor:
        def forward_graph(self, _graph, **kwargs):
            assert not kwargs, "direct evaluation invoked the UE-slot-conditioned forward pass"
            return {"candidate_prg_logits": candidate_logits}

    def fail_legacy(*_args, **_kwargs):
        pytest.fail("direct evaluation invoked legacy UE selection/coverage/repair")

    for name in (
        "apply_ue_action_mask",
        "_repair_ue_action_for_coverage",
        "build_slot_selection_mask",
        "apply_prg_action_mask",
        "build_prg_candidate_valid_mask",
        "select_and_decode_budgeted_candidate_actions",
        "apply_candidate_blank_budget",
        "decode_candidate_actions_to_type0",
        "decode_joint_actions_to_type0",
    ):
        monkeypatch.setattr(eval_checkpoint_online, name, fail_legacy)

    selected = eval_checkpoint_online._select_joint_action(
        DirectEvalActor(),
        graph,
        decode_mode="argmax",
        candidate_policy_mode="direct",
        candidate_budgeted_select_mode="sample",
        candidate_budgeted_hard_cap=1,
        candidate_budgeted_usage_penalty=0.4,
        candidate_budgeted_overcap_penalty=8.0,
        candidate_budgeted_first_service_bonus=0.01,
        candidate_decode_demand_slack_bytes=0.0,
    )

    assert selected["prg_action_class"].tolist() == [0, 1, 0, 1, 0, 1]
    assert selected["action_ue_select"].tolist() == [0, 1, 2, 3, 4, 5]
    assert selected["action_prg_alloc"].tolist() == [0, 2, 4, 1, 3, 5]
    assert selected["slot_repair_count"] == 0.0
    assert selected["coverage_repair_count"] == 0.0
    assert selected["safe_fill_count"] == 0.0
    assert selected["blank_budget_repair_count"] == 0.0
    assert selected["non_hard_override_count"] == 0.0
    assert selected["candidate_policy_mode_direct"] == 1.0


def _direct_eval_args(**overrides):
    values = {
        "candidate_policy_mode": "direct",
        "candidate_budgeted_select_mode": "sample",
        "candidate_budgeted_hard_cap": 1,
        "candidate_blank_budget_max_per_cell": -1,
        "slot_repair_max_per_cell": 0,
        "safe_fill_max_per_cell": 0,
        "candidate_budgeted_tail_bias_coef": 0.0,
        "candidate_budgeted_rescue_bias_coef": 0.0,
        "candidate_budgeted_completion_bonus": 0.0,
        "candidate_budgeted_tail_quota_enable": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_budgeted_select_mode", "greedy"),
        ("candidate_budgeted_hard_cap", 0),
        ("candidate_blank_budget_max_per_cell", 0),
        ("slot_repair_max_per_cell", 1),
        ("safe_fill_max_per_cell", 1),
        ("candidate_budgeted_tail_bias_coef", 0.1),
        ("candidate_budgeted_rescue_bias_coef", 0.1),
        ("candidate_budgeted_completion_bonus", 0.1),
        ("candidate_budgeted_tail_quota_enable", 1),
    ],
)
def test_direct_evaluation_rejects_incompatible_configuration(field, value):
    with pytest.raises(ValueError, match="direct"):
        eval_checkpoint_online._validate_direct_evaluation_config(
            _direct_eval_args(**{field: value})
        )


def _run_evaluator_main_boundary(monkeypatch, checkpoint_args):
    graph = object()
    captured = {}

    class FakeEnv:
        def reset(self, **_kwargs):
            return {}, 0.0, {}

        def step(self, *_args):
            return {}, 0.0, True, {"tti": 0}

        def close(self):
            pass

    def select(_actor, _graph, **kwargs):
        captured.update(kwargs)
        return defaultdict(
            float,
            action_ue_select=[0],
            action_prg_alloc=[0],
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_checkpoint_online.py",
            "--checkpoint",
            "checkpoint.pt",
            "--socket-path",
            "unused.sock",
            "--device",
            "cpu",
            "--max-steps",
            "1",
            "--save-steps",
            "0",
        ],
    )
    monkeypatch.setattr(eval_checkpoint_online, "_set_seed", lambda _seed: None)
    monkeypatch.setattr(
        eval_checkpoint_online,
        "_pick_device",
        lambda _device: torch.device("cpu"),
    )
    monkeypatch.setattr(
        eval_checkpoint_online,
        "_load_actor_checkpoint",
        lambda *_args: (SimpleNamespace(cfg=SimpleNamespace(up_edge_attr_dim=6)), {"args": checkpoint_args}),
    )
    monkeypatch.setattr(
        eval_checkpoint_online,
        "FeatureSnapshotBuilder",
        lambda _cfg: SimpleNamespace(build=lambda _obs: object()),
    )
    monkeypatch.setattr(
        eval_checkpoint_online,
        "SparseEdgeGenerator",
        lambda cfg: (
            captured.__setitem__("edge_generator_config", cfg)
            or SimpleNamespace(build_graph=lambda _snapshot: graph)
        ),
    )
    monkeypatch.setattr(eval_checkpoint_online, "BridgeEnvClient", lambda _cfg: FakeEnv())
    monkeypatch.setattr(eval_checkpoint_online, "_graph_to_device", lambda value, _device: value)
    monkeypatch.setattr(eval_checkpoint_online, "_select_joint_action", select)
    monkeypatch.setattr(eval_checkpoint_online, "extract_reward_metrics", lambda _info: {})
    return captured


def _direct_evaluation_checkpoint_args(**overrides):
    values = {
        "candidate_policy_mode": "direct",
        "candidate_budgeted_select_mode": "sample",
        "candidate_budgeted_hard_cap": 1,
        "candidate_decode_demand_slack_bytes": 1234.5,
        "candidate_budgeted_usage_penalty": 0.41,
        "candidate_budgeted_overcap_penalty": 8.2,
        "candidate_budgeted_first_service_bonus": 0.03,
        "candidate_blank_budget_max_per_cell": -1,
        "slot_repair_max_per_cell": 0,
        "safe_fill_max_per_cell": 0,
        "candidate_budgeted_tail_bias_coef": 0.0,
        "candidate_budgeted_rescue_bias_coef": 0.0,
        "candidate_budgeted_completion_bonus": 0.0,
        "candidate_budgeted_tail_quota_enable": 0,
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    ("checkpoint_candidate_mode", "expected_candidate_mode"),
    [
        ("topk_diversity", "topk_diversity"),
        (None, "all_demand_ues"),
    ],
)
def test_evaluator_uses_direct_checkpoint_policy_config_at_selection_boundary(
    monkeypatch,
    checkpoint_candidate_mode,
    expected_candidate_mode,
):
    checkpoint_args = _direct_evaluation_checkpoint_args()
    if checkpoint_candidate_mode is not None:
        checkpoint_args["candidate_mode"] = checkpoint_candidate_mode
    captured = _run_evaluator_main_boundary(monkeypatch, checkpoint_args)

    assert eval_checkpoint_online.main() == 0
    assert captured["edge_generator_config"].candidate_mode == expected_candidate_mode
    assert captured["candidate_policy_mode"] == "direct"
    assert captured["candidate_budgeted_select_mode"] == "sample"
    assert captured["candidate_decode_demand_slack_bytes"] == 1234.5
    assert captured["candidate_budgeted_hard_cap"] == 1
    assert captured["candidate_budgeted_usage_penalty"] == 0.41
    assert captured["candidate_budgeted_overcap_penalty"] == 8.2
    assert captured["candidate_budgeted_first_service_bonus"] == 0.03
    assert captured["candidate_blank_budget_max_per_cell"] == -1
    assert captured["slot_repair_max_per_cell"] == 0
    assert captured["safe_fill_max_per_cell"] == 0
    assert captured["candidate_budgeted_tail_bias_coef"] == 0.0
    assert captured["candidate_budgeted_rescue_bias_coef"] == 0.0
    assert captured["candidate_budgeted_completion_bonus"] == 0.0
    assert captured["candidate_budgeted_tail_quota_enable"] == 0


def test_evaluator_rejects_unsupported_checkpoint_candidate_mode(monkeypatch):
    _run_evaluator_main_boundary(
        monkeypatch,
        _direct_evaluation_checkpoint_args(candidate_mode="unsupported"),
    )

    with pytest.raises(ValueError, match="candidate_mode.*all_demand_ues.*topk_diversity"):
        eval_checkpoint_online.main()


@pytest.mark.parametrize("checkpoint_candidate_mode", ["topk_diversity", "unsupported"])
def test_legacy_evaluator_ignores_checkpoint_candidate_mode(
    monkeypatch,
    checkpoint_candidate_mode,
):
    captured = _run_evaluator_main_boundary(
        monkeypatch,
        _direct_evaluation_checkpoint_args(
            candidate_policy_mode="legacy",
            candidate_mode=checkpoint_candidate_mode,
        ),
    )

    assert eval_checkpoint_online.main() == 0
    assert captured["edge_generator_config"].candidate_mode == "all_demand_ues"
    assert captured["candidate_policy_mode"] == "legacy"


def test_causal_direct_preset_executes_with_frozen_compatible_arguments(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    sandbox_root = tmp_path / "repo"
    script_dir = sandbox_root / "cuMAC" / "scripts"
    training_dir = sandbox_root / "training" / "stageb_hgraph"
    venv_bin = sandbox_root / ".venv" / "bin"
    script_dir.mkdir(parents=True)
    training_dir.mkdir(parents=True)
    venv_bin.mkdir(parents=True)

    source_script = repo_root / "cuMAC" / "scripts" / "run_stageB_hgraph_online_ppo.sh"
    script = script_dir / source_script.name
    shutil.copy2(source_script, script)
    (training_dir / "online_ppo.py").write_text("", encoding="utf-8")
    run_script = script_dir / "run_stageB_main_experiment.sh"
    run_script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    run_script.chmod(0o755)
    python_stub = venv_bin / "python"
    python_stub.write_text(
        "#!/usr/bin/env bash\nprintf '[fake-python]'\nprintf ' %s' \"$@\"\nprintf '\\n'\n",
        encoding="utf-8",
    )
    python_stub.chmod(0o755)

    build_dir = tmp_path / "build"
    simulator = (
        build_dir
        / "cuMAC"
        / "examples"
        / "multiCellSchedulerUeSelection"
        / "multiCellSchedulerUeSelection"
    )
    simulator.parent.mkdir(parents=True)
    simulator.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    simulator.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            str(script),
            "--ppo-preset",
            "goodput_delay_rate_v20_causal_direct_sample_fast",
            "--build-dir",
            str(build_dir),
            "--out-dir",
            str(tmp_path / "out"),
            "--iterations",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "candidate_mode=all_demand_ues" in completed.stdout
    assert "max_prg_candidates=12 action_head_mode=candidate" in completed.stdout
    assert "candidate_policy_mode=direct" in completed.stdout
    assert (
        "candidate_budgeted_select_mode=sample hard_cap=1 usage_penalty=0.4 "
        "overcap_penalty=8.0 first_service_bonus=0.01"
    ) in completed.stdout
    assert "safe_fill_max_per_cell=0" in completed.stdout
    assert "candidate_blank_budget_max_per_cell=-1" in completed.stdout
    assert "candidate_tail_bias_coef=0.0" in completed.stdout
    assert "candidate_rescue_bias_coef=0.0" in completed.stdout
    assert "candidate_completion_bonus=0.0" in completed.stdout
    assert "candidate_tail_quota_enable=0" in completed.stdout
    assert "slot_repair_max_per_cell=0" in completed.stdout
    assert "--candidate-policy-mode direct" in completed.stdout


def test_direct_mode_excludes_ue_selector_logp_and_updates_candidate_head():
    torch.manual_seed(11)
    graph = make_candidate_graph(n_prg=2)
    actor = StageBHGraphPolicy(
        HGraphModelConfig(action_head_mode="candidate", max_prg_candidates=2)
    )
    actor_out = actor(graph)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    trace = select_candidate_policy_action(
        graph,
        packed,
        packed.mask,
        actor_out["candidate_prg_logits"],
        CandidatePolicyConfig(mode="sample"),
        generator=torch.Generator().manual_seed(11),
    )

    new_logp, entropy = _direct_candidate_logp_entropy(
        actor_out,
        graph,
        trace,
        CandidatePolicyConfig(mode="sample"),
    )
    loss = -new_logp
    loss.backward()

    grad = actor.action_head.candidate_pair_score[-1].weight.grad
    assert torch.isfinite(loss)
    assert grad is not None and grad.abs().sum().item() > 0.0
    assert entropy.item() > 0.0


def test_direct_rollout_returns_exact_trace_materialization_and_metrics(monkeypatch):
    torch.manual_seed(17)
    graph = make_candidate_graph(n_prg=2)
    actor = StageBHGraphPolicy(
        HGraphModelConfig(action_head_mode="candidate", max_prg_candidates=2)
    )
    actor_out = actor(graph)
    monkeypatch.setattr(
        actor,
        "forward_graph",
        lambda *args, **kwargs: pytest.fail("direct mode sampled the legacy UE selector"),
    )

    sampled = _sample_joint_action(
        actor,
        actor_out,
        graph,
        candidate_policy_mode="direct",
        candidate_budgeted_select_mode="sample",
        candidate_decode_demand_slack_bytes=0.0,
    )

    trace = sampled["candidate_policy_trace"]
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    expected = materialize_candidate_choices_to_type0(graph, packed, trace.choice)
    assert torch.equal(sampled["prg_action_class"], trace.choice.cpu())
    assert torch.equal(torch.from_numpy(sampled["action_ue_select"]), expected.action_ue_select.cpu())
    assert torch.equal(torch.from_numpy(sampled["action_prg_alloc"]), expected.action_prg_alloc.cpu())
    assert torch.equal(sampled["logp"], trace.logp.cpu())
    assert sampled["coverage_repair_count"] == 0.0
    assert sampled["safe_fill_count"] == 0.0
    assert sampled["blank_budget_repair_count"] == 0.0
    assert sampled["non_hard_override_count"] == 0.0
    assert torch.equal(sampled["candidate_packed"].ue_ids, packed.ue_ids)
    assert "action_replay_mismatch_count" not in sampled


def test_rollout_replay_metric_compares_exact_arrays_submitted_to_env(monkeypatch):
    torch.manual_seed(19)
    graph = make_candidate_graph(n_prg=2)
    actor = StageBHGraphPolicy(
        HGraphModelConfig(action_head_mode="candidate", max_prg_candidates=2)
    )
    actor_out = actor(graph)
    candidate_logits = torch.full_like(actor_out["candidate_prg_logits"], -40.0)
    candidate_logits[:, 0] = 40.0
    actor_out["candidate_prg_logits"] = candidate_logits
    monkeypatch.setattr(
        online_ppo,
        "_evaluate_graph",
        lambda *_args, **_kwargs: (actor_out, torch.tensor(0.0)),
    )

    submitted = {}
    sampled_for_submission = {}

    class FakeEnv:
        def step(self, action_ue_select, action_prg_alloc):
            submitted["action_ue_select"] = np.asarray(
                action_ue_select, dtype=np.int32
            ).copy()
            submitted["action_prg_alloc"] = np.asarray(
                action_prg_alloc, dtype=np.int32
            ).copy()
            return {}, 0.0, True, {}

    class FakeRunner:
        def __init__(self):
            self.args = SimpleNamespace(
                action_head_mode="candidate",
                candidate_policy_mode="direct",
                candidate_budgeted_select_mode="sample",
                candidate_budgeted_hard_cap=1,
            )
            self.env = FakeEnv()

        def begin_rollout(self, _episode_idx):
            return {}, 0.0, {}, 19, 19, False

        def record_transition(self, *_args):
            pass

    original_sample = online_ppo._sample_joint_action

    def sample_then_tamper(*args, **kwargs):
        sampled = original_sample(*args, **kwargs)
        assert sampled["candidate_policy_trace"].choice[0].item() == 0
        tampered_alloc = np.asarray(
            sampled["action_prg_alloc"], dtype=np.int32
        ).copy()
        first_native_prg = (
            int(graph.snapshot.static_meta.prg_local_index[0])
            * int(graph.snapshot.static_meta.n_cell)
            + int(graph.snapshot.static_meta.prg_owner_cell[0])
        )
        tampered_alloc[first_native_prg] = -1
        sampled["action_prg_alloc"] = tampered_alloc
        sampled_for_submission["action_ue_select"] = np.asarray(
            sampled["action_ue_select"], dtype=np.int32
        ).copy()
        sampled_for_submission["action_prg_alloc"] = tampered_alloc.copy()
        return sampled

    monkeypatch.setattr(online_ppo, "_sample_joint_action", sample_then_tamper)
    monkeypatch.setattr(
        online_ppo,
        "extract_reward_metrics",
        lambda _info: {
            "goodput_mbps": 0.0,
            "tb_err_rate": 0.0,
            "expiry_drop_rate": 0.0,
            "prg_utilization_ratio": 0.0,
            "prg_reuse_ratio": 0.0,
        },
    )
    snapshot_builder = SimpleNamespace(
        reset=lambda: None,
        build=lambda _obs: object(),
    )
    edge_generator = SimpleNamespace(build_graph=lambda _snapshot: graph)

    rollout, summary, _per_ue, _expiry = online_ppo._collect_rollout(
        runner=FakeRunner(),
        snapshot_builder=snapshot_builder,
        edge_generator=edge_generator,
        actor=actor,
        critic=GraphValueHead(hidden_dim=128),
        device=torch.device("cpu"),
        rollout_steps=1,
        rollout_warmup_steps=0,
        reward_mode="raw",
        reward_cfg={},
        episode_idx=0,
        rollout_log_every_steps=0,
    )

    assert len(rollout) == 1
    assert np.array_equal(
        submitted["action_prg_alloc"],
        sampled_for_submission["action_prg_alloc"],
    )
    assert np.array_equal(
        submitted["action_ue_select"],
        sampled_for_submission["action_ue_select"],
    )
    assert not hasattr(rollout[0], "candidate_packed")
    assert rollout[0].action_replay_mismatch_count == 1.0
    assert summary["rollout_action_replay_mismatch_count_mean"] == 1.0


def test_action_replay_uses_native_prg_order_and_checks_blank_choices():
    graph = make_candidate_graph(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    choice = torch.tensor(
        [
            packed.blank_class_index,
            0,
            1,
            packed.blank_class_index,
            0,
            packed.blank_class_index,
        ],
        dtype=torch.long,
    )

    # Candidate token order is cell-major, while the native Type-0 array is
    # local-PRG-major: native indices [0, 3, 1, 4, 2, 5].
    action_ue_select = np.array([0, -1, 3, -1, 4, -1], dtype=np.int32)
    action_prg_alloc = np.array([-1, 2, 4, 0, -1, -1], dtype=np.int32)
    exact = online_ppo._candidate_action_replay_mismatch_count(
        graph,
        packed,
        choice,
        action_ue_select,
        action_prg_alloc,
    )

    filled_blank = action_prg_alloc.copy()
    filled_blank[0] = 0
    blank_mismatch = online_ppo._candidate_action_replay_mismatch_count(
        graph,
        packed,
        choice,
        action_ue_select,
        filled_blank,
    )

    assert exact == 0.0
    assert blank_mismatch == 1.0


def test_action_replay_diagnostic_detects_tampered_type0_action():
    graph = make_candidate_graph(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    choice = torch.zeros((packed.num_prg_tokens,), dtype=torch.long)
    decoded = materialize_candidate_choices_to_type0(graph, packed, choice)

    exact = online_ppo._candidate_action_replay_mismatch_count(
        graph,
        packed,
        choice,
        decoded.action_ue_select,
        decoded.action_prg_alloc,
    )
    tampered_alloc = decoded.action_prg_alloc.clone()
    first_native_prg = (
        int(graph.snapshot.static_meta.prg_local_index[0])
        * int(graph.snapshot.static_meta.n_cell)
        + int(graph.snapshot.static_meta.prg_owner_cell[0])
    )
    tampered_alloc[first_native_prg] = -1
    tampered = online_ppo._candidate_action_replay_mismatch_count(
        graph,
        packed,
        choice,
        decoded.action_ue_select,
        tampered_alloc,
    )

    assert exact == 0.0
    assert tampered == 1.0


def test_replay_summary_requires_all_direct_steps_with_real_diagnostics():
    direct = [
        SimpleNamespace(
            candidate_policy_mode="direct",
            action_replay_mismatch_count=0.0,
        ),
        SimpleNamespace(
            candidate_policy_mode="direct",
            action_replay_mismatch_count=2.0,
        ),
    ]
    assert online_ppo._action_replay_summary(direct) == {
        "rollout_action_replay_mismatch_count_mean": 1.0,
    }
    assert online_ppo._action_replay_summary(
        [*direct, SimpleNamespace(candidate_policy_mode="legacy")]
    ) == {}


def _direct_args(**overrides):
    values = {
        "candidate_policy_mode": "direct",
        "action_head_mode": "candidate",
        "candidate_budgeted_select_mode": "sample",
        "candidate_budgeted_hard_cap": 1,
        "candidate_blank_budget_max_per_cell": -1,
        "slot_repair_max_per_cell": 0,
        "safe_fill_max_per_cell": 0,
        "candidate_budgeted_tail_bias_coef": 0.0,
        "candidate_budgeted_rescue_bias_coef": 0.0,
        "candidate_budgeted_completion_bonus": 0.0,
        "candidate_budgeted_tail_quota_enable": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action_head_mode", "slot"),
        ("candidate_budgeted_select_mode", "greedy"),
        ("candidate_blank_budget_max_per_cell", 0),
        ("slot_repair_max_per_cell", 1),
        ("safe_fill_max_per_cell", 1),
        ("candidate_budgeted_tail_bias_coef", 0.1),
        ("candidate_budgeted_rescue_bias_coef", 0.1),
        ("candidate_budgeted_completion_bonus", 0.1),
        ("candidate_budgeted_tail_quota_enable", 1),
    ],
)
def test_direct_mode_rejects_incompatible_configuration(field, value):
    with pytest.raises(ValueError, match="direct"):
        _validate_candidate_policy_mode(_direct_args(**{field: value}))


def test_direct_mode_requires_hard_cap():
    _validate_candidate_policy_mode(_direct_args(candidate_budgeted_hard_cap=1))
    with pytest.raises(ValueError, match="candidate_budgeted_hard_cap"):
        _validate_candidate_policy_mode(_direct_args(candidate_budgeted_hard_cap=0))


def test_ppo_direct_batch_replays_each_stored_trace():
    torch.manual_seed(23)
    graphs = [make_candidate_graph(n_prg=2), make_candidate_graph(n_prg=2)]
    actor = StageBHGraphPolicy(
        HGraphModelConfig(action_head_mode="candidate", max_prg_candidates=2)
    )
    outputs = [actor(graph) for graph in graphs]
    cfg = CandidatePolicyConfig(mode="sample")
    traces = []
    for graph, actor_out in zip(graphs, outputs):
        packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
        traces.append(
            select_candidate_policy_action(
                graph,
                packed,
                packed.mask,
                actor_out["candidate_prg_logits"],
                cfg,
            )
        )
    batched_out = {
        "candidate_prg_logits": torch.stack(
            [out["candidate_prg_logits"] for out in outputs],
            dim=0,
        ),
        "ue_logits": torch.full((2, 1), float("nan")),
    }

    logp, entropy = _direct_candidate_logp_entropy_batch(
        batched_out,
        graphs,
        traces,
        cfg,
    )

    assert logp.shape == (2,)
    assert entropy.shape == (2,)
    assert torch.isfinite(logp).all()
    assert torch.isfinite(entropy).all()
    assert torch.allclose(logp, torch.stack([trace.logp for trace in traces]))
    assert torch.allclose(entropy, torch.stack([trace.entropy for trace in traces]))


def test_legacy_checkpoint_load_and_default_startup_mode_remain_legacy(
    monkeypatch,
    tmp_path,
):
    source = StageBHGraphPolicy(
        HGraphModelConfig(action_head_mode="candidate", max_prg_candidates=2)
    )
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "actor_state": source.state_dict(),
            "args": {"action_head_mode": "candidate"},
        },
        checkpoint,
    )
    restored = StageBHGraphPolicy(
        HGraphModelConfig(action_head_mode="candidate", max_prg_candidates=2)
    )

    loaded, total = _load_checkpoint_compatible(restored, str(checkpoint))

    assert loaded == total
    for name, value in source.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)
    monkeypatch.setattr(sys, "argv", ["online_ppo.py", "--sim-bin", "unused"])
    assert parse_args().candidate_policy_mode == "legacy"


def test_direct_ppo_eval_replay_matches_rollout_with_dropout_and_batch_two():
    torch.manual_seed(29)
    graphs = [make_candidate_graph(n_prg=2), make_candidate_graph(n_prg=2)]
    actor = StageBHGraphPolicy(
        HGraphModelConfig(
            action_head_mode="candidate",
            max_prg_candidates=2,
            dropout=0.35,
        )
    )
    actor.eval()
    rollout = []
    for graph in graphs:
        sampled = _sample_joint_action(
            actor,
            actor(graph),
            graph,
            candidate_policy_mode="direct",
            candidate_budgeted_select_mode="sample",
            candidate_decode_demand_slack_bytes=0.0,
        )
        required = {
            field.name: 0.0
            for field in fields(RolloutStep)
            if field.default is MISSING and field.default_factory is MISSING
        }
        required.update(
            graph=graph,
            ue_action_class=sampled["ue_action_class"],
            prg_action_class=sampled["prg_action_class"],
            old_logp=sampled["logp"],
            candidate_policy_trace=sampled["candidate_policy_trace"],
            candidate_policy_mode="direct",
        )
        rollout.append(RolloutStep(**required))
    critic = GraphValueHead(hidden_dim=128)

    metrics = _ppo_update(
        actor=actor,
        critic=critic,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=0.0),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=0.0),
        rollout=rollout,
        advantages=torch.ones(2),
        returns=torch.zeros(2),
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        update_epochs=1,
        minibatch_size=2,
        iter_idx=1,
        update_log_every_minibatches=0,
        target_kl=0.0,
        candidate_policy_mode="direct",
        candidate_budgeted_select_mode="sample",
        candidate_decode_demand_slack_bytes=0.0,
    )

    assert metrics["num_updates"] == 1.0
    assert metrics["ue_selector_policy_logp_included"] == 0.0
    assert torch.isfinite(torch.tensor(metrics["policy_loss"]))
    assert torch.isfinite(torch.tensor(metrics["entropy_mean"]))
    assert metrics["candidate_head_grad_norm"] > 0.0
    assert abs(metrics["approx_kl_mean"]) < 1.0e-6
    assert actor.training is False

from __future__ import annotations

import csv
import math
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from training.stageb_hgraph.edge_generator import EdgeGeneratorConfig, SparseEdgeGenerator
from training.stageb_hgraph.bridge_env import PreActionPhySnapshot
from training.stageb_hgraph.ctbs_headroom import CtbsHeadroomInputs, CtbsHeadroomState
from training.stageb_hgraph.eval_checkpoint_online import (
    _count_candidate_mask_reactivations,
)
from training.stageb_hgraph.feature_snapshot import FeatureSnapshotBuilder
from training.stageb_hgraph.independent_actor import (
    IndependentActorConfig,
    PFQIndependentReuseActor,
)
from training.stageb_hgraph.independent_blank_diagnostics import (
    summarize_independent_policy_diagnostics,
)
from training.stageb_hgraph.independent_policy import (
    IndependentPolicyTrace,
    replay_independent_trace,
)
from training.stageb_hgraph.graph_ops import type0_action_preflight_reject_counts
from training.stageb_hgraph.online_ppo import (
    GraphValueHead,
    _evaluate_graph,
    _evaluate_graph_batch,
    _ctbs_inputs_for_rollout_step,
    _record_submitted_action_replay_mismatch,
    _sample_joint_action,
    _validate_ctbs_rollout_parity,
)
from training.stageb_hgraph.online_imitation import OnlineEpisodeRunner
from training.stageb_hgraph.phy_tbs_alignment_audit import PhyTbsConfig
from training.stageb_hgraph.tests.graph_fixtures import make_candidate_observation


def _graph():
    obs = make_candidate_observation(n_prg=3, ue_feat_dim=16)
    return SparseEdgeGenerator(
        EdgeGeneratorConfig(
            candidate_mode="all_demand_ues",
            include_pfq_score_anchor=False,
            include_candidate_risk_features=False,
        )
    ).build_graph(FeatureSnapshotBuilder().build(obs))


def _actor():
    return PFQIndependentReuseActor(
        IndependentActorConfig(
            hidden_dim=16,
            message_layers=1,
            max_candidates=2,
            up_edge_attr_dim=5,
        )
    )


def _ctbs_inputs(n_prg: int = 3) -> CtbsHeadroomInputs:
    return CtbsHeadroomInputs.from_arrays(
        tti=1,
        snapshot_id=41,
        config_sha256="12" * 32,
        config=PhyTbsConfig(
            mcs_lut_type=0,
            mcs_sinr_cap_db=25.99,
            layer_sinval_threshold=0.1,
            prbs_per_prg=16,
            pdsch_data_symbols=11,
            effective_layer_cap=4,
        ),
        ue_ids=np.arange(6, dtype=np.int32),
        serving_cell=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32),
        wb_sinr_first_layer_db=np.full((6,), -4.0, dtype=np.float32),
        olla_delta=np.zeros((6,), dtype=np.float32),
        olla_delta_up=np.ones((6,), dtype=np.float32),
        olla_delta_down=np.full((6,), 0.1, dtype=np.float32),
        prior_tb_err=np.full((6,), -1, dtype=np.int32),
        prg_layer_cap=np.full((6, n_prg), 4, dtype=np.uint8),
    )


class _BlankIndependentActor(PFQIndependentReuseActor):
    def score_reuse_group(
        self,
        encoded,
        graph,
        packed,
        local_prg: int,
        sequential_features: torch.Tensor,
    ) -> torch.Tensor:
        del encoded, graph, local_prg, sequential_features
        anchor = next(self.parameters())
        logits = torch.full(
            (3, packed.max_candidates + 1),
            -4.0,
            dtype=anchor.dtype,
            device=anchor.device,
        )
        logits[:, packed.blank_class_index] = 4.0
        return logits


def _guard_graph(n_prg: int = 3):
    obs = make_candidate_observation(
        n_prg=n_prg,
        ue_feat_dim=16,
        backlog_bytes=(12000.0,) * 6,
    )
    obs["obs_ue_features"][:, 12] = math.log(3.0)
    obs["obs_ue_features"][:, 13] = 0.5
    obs["obs_ue_features"][:, 14] = 0.5
    return SparseEdgeGenerator(
        EdgeGeneratorConfig(
            candidate_mode="all_demand_ues",
            include_pfq_score_anchor=False,
            include_candidate_risk_features=False,
        )
    ).build_graph(FeatureSnapshotBuilder().build(obs))


def _blank_actor() -> PFQIndependentReuseActor:
    return _BlankIndependentActor(
        IndependentActorConfig(
            hidden_dim=16,
            message_layers=1,
            max_candidates=2,
            up_edge_attr_dim=5,
        )
    )


def test_candidate_mask_reactivation_counter_detects_false_to_true_only() -> None:
    # Candidate columns are quality-ranked independently per PRG. The same UE
    # can move columns without becoming valid again.
    candidate_ue_ids = torch.tensor(
        [
            [[0, 1, 2]],
            [[1, 0, 2]],
            [[2, 1, 0]],
        ]
    )
    mask_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    monotone_by_ue = torch.tensor(
        [
            [[True, True, False]],   # UE 0,1 valid
            [[True, True, False]],   # reordered: UE 1,0 valid
            [[False, True, False]],  # reordered: only UE 1 remains valid
        ],
        device=mask_device,
    )
    reactivated_by_ue = monotone_by_ue.clone()
    reactivated_by_ue[1, 0, 1] = False  # UE 0 is first masked.
    reactivated_by_ue[2, 0, 2] = True  # UE 0 becomes valid again.

    assert _count_candidate_mask_reactivations(
        monotone_by_ue,
        candidate_ue_ids,
        n_ue=3,
    ) == 0
    assert _count_candidate_mask_reactivations(
        reactivated_by_ue,
        candidate_ue_ids,
        n_ue=3,
    ) == 1


def test_independent_graph_and_batch_evaluation_reuse_existing_critic_pooling() -> None:
    actor = _actor()
    critic = GraphValueHead(hidden_dim=16)
    graph = _graph()

    one, one_value = _evaluate_graph(actor, critic, graph)
    batch, batch_value = _evaluate_graph_batch(actor, critic, [graph, graph])

    assert one["reuse_embeddings"].shape == (3, 3, 16)
    assert one_value.shape == ()
    assert batch["reuse_embeddings"].shape == (2, 3, 3, 16)
    assert batch_value.shape == (2,)


def test_rollout_action_stores_trace_exact_arrays_and_replays_without_mismatch() -> None:
    torch.manual_seed(11)
    actor = _actor()
    critic = GraphValueHead(hidden_dim=16)
    graph = _graph()
    actor_out, _ = _evaluate_graph(actor, critic, graph)

    torch.manual_seed(23)
    sampled_off = _sample_joint_action(
        actor,
        actor_out,
        graph,
        independent_policy_diagnostics=False,
    )
    rng_off = torch.random.get_rng_state()
    torch.manual_seed(23)
    sampled = _sample_joint_action(
        actor,
        actor_out,
        graph,
        independent_policy_diagnostics=True,
    )
    rng_on = torch.random.get_rng_state()
    _record_submitted_action_replay_mismatch(graph, sampled)

    trace_off = sampled_off["independent_policy_trace"]
    trace = sampled["independent_policy_trace"]
    assert isinstance(trace, IndependentPolicyTrace)
    assert trace_off.diagnostics is None
    assert trace.diagnostics is not None
    assert trace.diagnostics.valid_ue_count.numel() == 9
    assert np.array_equal(sampled_off["action_ue_select"], sampled["action_ue_select"])
    assert np.array_equal(sampled_off["action_prg_alloc"], sampled["action_prg_alloc"])
    assert torch.equal(trace_off.choice, trace.choice)
    assert torch.equal(trace_off.logp, trace.logp)
    assert torch.equal(trace_off.entropy, trace.entropy)
    assert torch.equal(rng_off, rng_on)
    assert sampled["candidate_policy_mode"] == "independent"
    assert sampled["action_replay_mismatch_count"] == 0.0
    assert np.asarray(sampled["action_ue_select"]).shape == (6,)
    assert np.asarray(sampled["action_prg_alloc"]).shape == (9,)

    summary = summarize_independent_policy_diagnostics(
        trace.diagnostics,
        classification_prefix="shadow_",
    )
    assert "shadow_greedy_blank_ratio" in summary
    assert "shadow_forced_blank_ratio" in summary
    assert "shadow_learned_blank_ratio" in summary
    assert "shadow_fragmentation_blank_ratio" in summary
    assert "shadow_conservative_blank_ratio" in summary

    _, _, replayed = replay_independent_trace(
        actor,
        graph,
        sampled["candidate_packed"],
        trace,
    )
    assert np.array_equal(
        replayed.action_ue_select.cpu().numpy(),
        sampled["action_ue_select"],
    )
    assert np.array_equal(
        replayed.action_prg_alloc.cpu().numpy(),
        sampled["action_prg_alloc"],
    )


def test_online_ppo_sampling_freezes_ctbs_tuple_into_replay_trace() -> None:
    torch.manual_seed(29)
    actor = _actor()
    critic = GraphValueHead(hidden_dim=16)
    graph = _graph()
    actor_out, _ = _evaluate_graph(actor, critic, graph)
    inputs = _ctbs_inputs()

    sampled = _sample_joint_action(
        actor,
        actor_out,
        graph,
        ctbs_headroom_inputs=inputs,
    )
    trace = sampled["independent_policy_trace"]
    replay_logp, replay_entropy, replayed = replay_independent_trace(
        actor,
        graph,
        sampled["candidate_packed"],
        trace,
    )

    assert trace.ctbs_headroom_inputs is inputs
    assert torch.equal(replay_logp, trace.logp)
    assert torch.equal(replay_entropy, trace.entropy)
    assert np.array_equal(
        replayed.action_prg_alloc.cpu().numpy(),
        sampled["action_prg_alloc"],
    )


def test_deterministic_evaluator_collects_independent_policy_diagnostics() -> None:
    from training.stageb_hgraph.eval_checkpoint_online import _select_joint_action

    selected = _select_joint_action(
        _actor(),
        _graph(),
        decode_mode="argmax",
        candidate_policy_mode="independent",
        independent_policy_diagnostics=True,
    )

    trace = selected["independent_policy_trace"]
    assert trace.diagnostics is not None
    assert trace.diagnostics.valid_ue_count.numel() == 9


def test_non_starvation_guard_retains_raw_and_executed_actions_with_override() -> None:
    from training.stageb_hgraph.eval_checkpoint_online import _select_joint_action

    selected = _select_joint_action(
        _blank_actor(),
        _guard_graph(),
        decode_mode="argmax",
        candidate_policy_mode="independent",
        independent_policy_diagnostics=True,
        ctbs_headroom_inputs=_ctbs_inputs(),
        independent_non_starvation_guard="ctbs-blank-v1",
    )

    trace = selected["independent_policy_trace"]
    assert torch.equal(trace.actor_choice, torch.full_like(trace.actor_choice, 2))
    assert int(trace.guard_override.sum().item()) == 3
    assert torch.equal(trace.guard_override, trace.actor_choice != trace.choice)
    assert trace.guard_diagnostics.guard_override.any()


@pytest.mark.parametrize(
    ("actor", "decode_mode", "ctbs_inputs", "diagnostics", "error"),
    [
        (
            _blank_actor(),
            "hierarchical_argmax",
            _ctbs_inputs(),
            True,
            "--decode-mode argmax",
        ),
        (_blank_actor(), "argmax", None, True, "CTBS"),
        (object(), "argmax", _ctbs_inputs(), True, "independent actor"),
        (_blank_actor(), "argmax", _ctbs_inputs(), False, "policy diagnostics"),
    ],
)
def test_non_starvation_guard_fails_closed_for_evaluator_contract(
    actor,
    decode_mode: str,
    ctbs_inputs: CtbsHeadroomInputs | None,
    diagnostics: bool,
    error: str,
) -> None:
    from training.stageb_hgraph.eval_checkpoint_online import _select_joint_action

    with pytest.raises(ValueError, match=error):
        _select_joint_action(
            actor,
            _guard_graph(),
            decode_mode=decode_mode,
            candidate_policy_mode="independent",
            independent_policy_diagnostics=diagnostics,
            ctbs_headroom_inputs=ctbs_inputs,
            independent_non_starvation_guard="ctbs-blank-v1",
        )


def test_non_starvation_guard_output_contract_retains_complete_two_tti_trace(
    tmp_path,
) -> None:
    from training.stageb_hgraph.eval_checkpoint_online import (
        _build_non_starvation_guard_summary,
        _guard_diagnostics_to_validated_rows,
        _select_joint_action,
        _write_non_starvation_guard_trace,
    )

    selected = _select_joint_action(
        _blank_actor(),
        _guard_graph(),
        decode_mode="argmax",
        candidate_policy_mode="independent",
        independent_policy_diagnostics=True,
        ctbs_headroom_inputs=_ctbs_inputs(),
        independent_non_starvation_guard="ctbs-blank-v1",
    )
    trace = selected["independent_policy_trace"]
    first = _guard_diagnostics_to_validated_rows(
        tti=1,
        trace=trace,
        n_cell=3,
        n_prg=3,
    )
    measured = _guard_diagnostics_to_validated_rows(
        tti=1001,
        trace=trace,
        n_cell=3,
        n_prg=3,
    )
    full = first + measured
    summary = _build_non_starvation_guard_summary(
        guard_mode="ctbs-blank-v1",
        full_rows=full,
        measured_rows=measured,
    )
    output = tmp_path / "non_starvation_guard_trace.csv"
    _write_non_starvation_guard_trace(
        output,
        full,
        guard_mode="ctbs-blank-v1",
    )

    assert len(full) == 18
    assert summary["guard_full_summary"]["total_override_count"] == 6
    assert summary["guard_measured_summary"]["total_override_count"] == 3
    assert summary["guard_full_summary"]["cell_budget_used_count"] == 12
    assert summary["guard_measured_summary"]["cell_budget_used_count"] == 6
    assert summary["guard_full_summary"]["override_count_by_cell"] == {
        0: 2,
        1: 2,
        2: 2,
    }
    assert summary["guard_full_summary"]["override_count_by_ue"] == {
        0: 2,
        2: 2,
        4: 2,
    }
    assert (
        summary["guard_full_summary"]["maximum_overrides_per_cell_per_tti"]
        == 1
    )
    assert summary["non_starvation_guard_config"] == {
        "minimum_log1p_drain_pressure": math.log(2.0),
        "maximum_scheduled_dequeue_adequacy": 1.0,
        "minimum_ttl_normalized_service_age": 0.25,
        "maximum_overrides_per_cell_per_tti": 1,
        "override_nonblank": False,
        "require_positive_ctbs_capacity_delta": True,
    }

    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        written = list(reader)
    assert reader.fieldnames == [
        "tti",
        "cell_index",
        "local_prg_index",
        "blank_class_index",
        "actor_choice",
        "executed_choice",
        "guard_override",
        "selected_class",
        "selected_ue",
        "candidate_was_valid",
        "log1p_drain_pressure",
        "scheduled_dequeue_adequacy",
        "ttl_normalized_service_age",
        "ctbs_capacity_before_bytes",
        "ctbs_capacity_after_bytes",
        "reason",
    ]
    assert len(written) == 18
    assert [int(row["actor_choice"]) for row in written] == [2] * 18
    assert [int(row["executed_choice"]) for row in written[:3]] == [0, 0, 0]
    assert [int(row["executed_choice"]) for row in written[3:9]] == [2] * 6
    assert [row["ctbs_capacity_before_bytes"] for row in written[:3]] == ["0"] * 3
    assert [row["ctbs_capacity_after_bytes"] for row in written[:3]] == ["253"] * 3


def test_non_starvation_guard_off_summary_is_zero_and_cli_has_no_thresholds(
    monkeypatch,
    tmp_path,
) -> None:
    from training.stageb_hgraph import eval_checkpoint_online

    summary = eval_checkpoint_online._build_non_starvation_guard_summary(
        guard_mode="off",
        full_rows=[],
        measured_rows=[],
    )
    assert summary["independent_non_starvation_guard"] == "off"
    assert summary["guard_full_summary"]["total_override_count"] == 0
    assert summary["guard_measured_summary"]["total_override_count"] == 0
    assert summary["guard_full_summary"]["override_count_by_cell"] == {}
    assert summary["guard_full_summary"]["override_count_by_ue"] == {}
    for guard_summary in (
        summary["guard_full_summary"],
        summary["guard_measured_summary"],
    ):
        assert guard_summary["maximum_overrides_per_cell_per_tti"] == 0
        assert all(
            value == 0
            for key, value in guard_summary.items()
            if key.endswith("_count")
        )
    off_trace = tmp_path / "non_starvation_guard_trace.csv"
    eval_checkpoint_online._write_non_starvation_guard_trace(
        off_trace,
        [],
        guard_mode="off",
    )
    assert not off_trace.exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_checkpoint_online.py",
            "--checkpoint",
            "actor.pt",
            "--socket-path",
            "bridge.sock",
            "--independent-non-starvation-guard",
            "ctbs-blank-v1",
        ],
    )
    args = eval_checkpoint_online.parse_args()
    assert args.independent_non_starvation_guard == "ctbs-blank-v1"
    forbidden = (
        "minimum_log1p_drain_pressure",
        "maximum_scheduled_dequeue_adequacy",
        "minimum_ttl_normalized_service_age",
        "maximum_overrides_per_cell_per_tti",
    )
    assert not any(hasattr(args, name) for name in forbidden)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("actor_choice", torch.tensor([0] + [2] * 8), "Blank-only"),
        (
            "guard_override",
            torch.tensor([True, True, True, True] + [False] * 5),
            "one override",
        ),
        (
            "ctbs_capacity_after_bytes",
            torch.tensor([0, 2024, 2024] + [0] * 6),
            "positive CTBS",
        ),
    ],
)
def test_non_starvation_guard_trace_validation_fails_closed(
    field: str,
    value: torch.Tensor,
    error: str,
) -> None:
    from training.stageb_hgraph.eval_checkpoint_online import (
        _guard_diagnostics_to_validated_rows,
        _select_joint_action,
    )

    selected = _select_joint_action(
        _blank_actor(),
        _guard_graph(),
        decode_mode="argmax",
        candidate_policy_mode="independent",
        independent_policy_diagnostics=True,
        ctbs_headroom_inputs=_ctbs_inputs(),
        independent_non_starvation_guard="ctbs-blank-v1",
    )
    trace = selected["independent_policy_trace"]
    broken = replace(
        trace,
        guard_diagnostics=replace(trace.guard_diagnostics, **{field: value}),
    )

    with pytest.raises((ValueError, RuntimeError), match=error):
        _guard_diagnostics_to_validated_rows(
            tti=1,
            trace=broken,
            n_cell=3,
            n_prg=3,
        )


def test_non_starvation_guard_trace_validation_rejects_missing_diagnostics() -> None:
    from training.stageb_hgraph.eval_checkpoint_online import (
        _guard_diagnostics_to_validated_rows,
    )

    with pytest.raises(RuntimeError, match="missing guard diagnostics"):
        _guard_diagnostics_to_validated_rows(
            tti=1,
            trace=SimpleNamespace(guard_diagnostics=None),
            n_cell=3,
            n_prg=17,
        )


def test_non_starvation_guard_trace_validation_rejects_incomplete_diagnostics() -> None:
    from training.stageb_hgraph.eval_checkpoint_online import (
        _guard_diagnostics_to_validated_rows,
        _select_joint_action,
    )

    selected = _select_joint_action(
        _blank_actor(),
        _guard_graph(),
        decode_mode="argmax",
        candidate_policy_mode="independent",
        independent_policy_diagnostics=True,
        ctbs_headroom_inputs=_ctbs_inputs(),
        independent_non_starvation_guard="ctbs-blank-v1",
    )
    trace = selected["independent_policy_trace"]
    incomplete = replace(
        trace.guard_diagnostics,
        reason_code=trace.guard_diagnostics.reason_code[:-1],
    )

    with pytest.raises(RuntimeError, match="guard diagnostics.*length"):
        _guard_diagnostics_to_validated_rows(
            tti=1,
            trace=replace(trace, guard_diagnostics=incomplete),
            n_cell=3,
            n_prg=3,
        )


def _guard_main_argv(tmp_path, *extra: str, include_out_dir: bool = True) -> list[str]:
    argv = [
        "eval_checkpoint_online.py",
        "--checkpoint",
        str(tmp_path / "actor.pt"),
        "--socket-path",
        "simulator-must-not-start.sock",
        "--independent-policy-diagnostics",
        "1",
        "--independent-headroom-accounting",
        "ctbs-v1",
        "--independent-non-starvation-guard",
        "ctbs-blank-v1",
        "--device",
        "cpu",
    ]
    if include_out_dir:
        argv.extend(("--out-dir", str(tmp_path / "guard-output")))
    argv.extend(extra)
    return argv


@pytest.mark.parametrize(
    ("extra", "error"),
    [
        ((), "--max-steps 5000"),
        (("--max-steps", "4999", "--episode-horizon", "5000"), "--max-steps 5000"),
        (("--max-steps", "5000"), "--episode-horizon 5000"),
    ],
)
def test_non_starvation_guard_rejects_nonformal_horizon_before_simulator_start(
    monkeypatch,
    tmp_path,
    extra: tuple[str, ...],
    error: str,
) -> None:
    from training.stageb_hgraph import eval_checkpoint_online

    class SimulatorMustNotStart:
        def __init__(self, *_args, **_kwargs) -> None:
            pytest.fail("simulator startup was reached before horizon validation")

    monkeypatch.setattr(eval_checkpoint_online, "BridgeEnvClient", SimulatorMustNotStart)
    monkeypatch.setattr(
        eval_checkpoint_online,
        "_load_actor_checkpoint",
        lambda *_args: (_blank_actor(), {}),
    )
    monkeypatch.setattr(sys, "argv", _guard_main_argv(tmp_path, *extra))

    with pytest.raises(ValueError, match=error):
        eval_checkpoint_online.main()


def test_non_starvation_guard_rejects_missing_output_before_simulator_start(
    monkeypatch,
    tmp_path,
) -> None:
    from training.stageb_hgraph import eval_checkpoint_online

    class SimulatorMustNotStart:
        def __init__(self, *_args, **_kwargs) -> None:
            pytest.fail("simulator startup was reached before output validation")

    monkeypatch.setattr(eval_checkpoint_online, "BridgeEnvClient", SimulatorMustNotStart)
    monkeypatch.setattr(
        eval_checkpoint_online,
        "_load_actor_checkpoint",
        lambda *_args: (_blank_actor(), {}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _guard_main_argv(
            tmp_path,
            "--max-steps",
            "5000",
            "--episode-horizon",
            "5000",
            include_out_dir=False,
        ),
    )

    with pytest.raises(ValueError, match="--out-dir"):
        eval_checkpoint_online.main()


def test_non_starvation_guard_requires_frozen_three_by_seventeen_topology() -> None:
    from training.stageb_hgraph.eval_checkpoint_online import (
        _validate_non_starvation_guard_topology,
    )

    for n_cell, n_prg in ((2, 17), (3, 16), (4, 17)):
        with pytest.raises(ValueError, match="3 cells.*17.*PRGs"):
            _validate_non_starvation_guard_topology(
                guard_mode="ctbs-blank-v1",
                n_cell=n_cell,
                n_prg=n_prg,
            )
    _validate_non_starvation_guard_topology(
        guard_mode="ctbs-blank-v1",
        n_cell=3,
        n_prg=17,
    )


def test_non_starvation_guard_main_rejects_wrong_topology_at_first_graph(
    monkeypatch,
    tmp_path,
) -> None:
    from training.stageb_hgraph import eval_checkpoint_online

    source = _ctbs_inputs()
    wrong_graph = _guard_graph(n_prg=3)

    class FakeEnv:
        closed = False

        def reset(self, **_kwargs):
            return {}, 0.0, {
                "harq_enabled": 0,
                "ctbs_config_available": True,
                "ctbs_config": source.config.as_vector(),
                "ctbs_config_sha256": source.config_sha256,
            }

        def close(self):
            self.closed = True

    env = FakeEnv()

    monkeypatch.setattr(
        sys,
        "argv",
        _guard_main_argv(
            tmp_path,
            "--max-steps",
            "5000",
            "--episode-horizon",
            "5000",
        ),
    )
    monkeypatch.setattr(
        eval_checkpoint_online,
        "_load_actor_checkpoint",
        lambda *_args: (_blank_actor(), {}),
    )
    monkeypatch.setattr(
        eval_checkpoint_online,
        "FeatureSnapshotBuilder",
        lambda _cfg: SimpleNamespace(build=lambda _obs: object()),
    )
    monkeypatch.setattr(
        eval_checkpoint_online,
        "SparseEdgeGenerator",
        lambda _cfg: SimpleNamespace(build_graph=lambda _snapshot: wrong_graph),
    )
    monkeypatch.setattr(eval_checkpoint_online, "BridgeEnvClient", lambda _cfg: env)

    with pytest.raises(ValueError, match="3 cells.*17.*PRGs"):
        eval_checkpoint_online.main()
    assert env.closed


@pytest.mark.parametrize("drift_n_prg", [16, 18])
def test_non_starvation_guard_main_rejects_every_action_topology_drift_and_closes(
    monkeypatch,
    tmp_path,
    drift_n_prg: int,
) -> None:
    from training.stageb_hgraph import eval_checkpoint_online

    source = _ctbs_inputs(n_prg=17)
    graphs = iter((_guard_graph(n_prg=17), _guard_graph(n_prg=drift_n_prg)))

    class FakeEnv:
        closed = False

        def reset(self, **_kwargs):
            return {}, 0.0, {
                "tti": 1,
                "harq_enabled": 0,
                "ctbs_config_available": True,
                "ctbs_config": source.config.as_vector(),
                "ctbs_config_sha256": source.config_sha256,
                "n_active_ue": 6,
            }

        def request_pre_action_snapshot(self, **_kwargs):
            pytest.fail("invalid action topology reached CTBS snapshot collection")

        def close(self):
            self.closed = True

    env = FakeEnv()
    monkeypatch.setattr(
        sys,
        "argv",
        _guard_main_argv(
            tmp_path,
            "--max-steps",
            "5000",
            "--episode-horizon",
            "5000",
        ),
    )
    monkeypatch.setattr(
        eval_checkpoint_online,
        "_load_actor_checkpoint",
        lambda *_args: (_blank_actor(), {}),
    )
    monkeypatch.setattr(
        eval_checkpoint_online,
        "FeatureSnapshotBuilder",
        lambda _cfg: SimpleNamespace(build=lambda _obs: object()),
    )
    monkeypatch.setattr(
        eval_checkpoint_online,
        "SparseEdgeGenerator",
        lambda _cfg: SimpleNamespace(build_graph=lambda _snapshot: next(graphs)),
    )
    monkeypatch.setattr(eval_checkpoint_online, "BridgeEnvClient", lambda _cfg: env)
    monkeypatch.setattr(
        eval_checkpoint_online,
        "resolve_runtime_candidate_contract",
        lambda *_args, **_kwargs: SimpleNamespace(runtime_candidate_width=2),
    )
    monkeypatch.setattr(
        eval_checkpoint_online,
        "_independent_deletion_equal",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        eval_checkpoint_online,
        "_graph_to_device",
        lambda value, _device: value,
    )

    with pytest.raises(ValueError, match="3 cells.*17.*PRGs"):
        eval_checkpoint_online.main()
    assert env.closed


@pytest.mark.parametrize(
    ("completed", "full_count", "measured_count", "error"),
    [
        (4999, 255000, 204000, "exactly 5000 completed"),
        (5000, 254999, 204000, "255000"),
        (5000, 255000, 203999, "204000"),
    ],
)
def test_non_starvation_guard_completion_requires_fixed_formal_counts(
    completed: int,
    full_count: int,
    measured_count: int,
    error: str,
) -> None:
    from training.stageb_hgraph.eval_checkpoint_online import (
        _validate_non_starvation_guard_completion,
    )

    with pytest.raises(RuntimeError, match=error):
        _validate_non_starvation_guard_completion(
            guard_mode="ctbs-blank-v1",
            completed_action_ttis=completed,
            full_rows=range(full_count),
            measured_rows=range(measured_count),
        )
    _validate_non_starvation_guard_completion(
        guard_mode="ctbs-blank-v1",
        completed_action_ttis=5000,
        full_rows=range(255000),
        measured_rows=range(204000),
    )


def test_non_starvation_guard_main_collects_51_rows_then_rejects_early_done(
    monkeypatch,
    tmp_path,
) -> None:
    from training.stageb_hgraph import eval_checkpoint_online

    source = _ctbs_inputs(n_prg=17)
    graph = _guard_graph(n_prg=17)
    captured: dict[str, int] = {}

    class FakeEnv:
        def reset(self, **_kwargs):
            return {}, 0.0, {
                "tti": 1,
                "harq_enabled": 0,
                "ctbs_config_available": True,
                "ctbs_config": source.config.as_vector(),
                "ctbs_config_sha256": source.config_sha256,
                "n_active_ue": 6,
            }

        def request_pre_action_snapshot(self, *, expected_tti: int):
            assert expected_tti == 1
            return PreActionPhySnapshot(
                tti=source.tti,
                snapshot_id=source.snapshot_id,
                config_sha256=source.config_sha256,
                olla_source_code=1,
                capture_phase_code=1,
                ue_ids=np.asarray(source.ue_ids, dtype=np.int32),
                serving_cell=np.asarray(source.serving_cell, dtype=np.int32),
                wb_sinr_first_layer_db=np.asarray(
                    source.wb_sinr_first_layer_db, dtype=np.float32
                ),
                olla_delta=np.asarray(source.olla_delta, dtype=np.float32),
                olla_delta_up=np.asarray(source.olla_delta_up, dtype=np.float32),
                olla_delta_down=np.asarray(source.olla_delta_down, dtype=np.float32),
                prior_tb_err=np.asarray(source.prior_tb_err, dtype=np.int32),
                prg_layer_cap=np.asarray(source.prg_layer_cap, dtype=np.uint8),
            )

        def step(self, *_args):
            reject_zeros = np.zeros((3,), dtype=np.int32)
            return {}, 0.0, True, {
                "tti": 1,
                "actual_ue_mcs": np.zeros((6,), dtype=np.int32),
                "actual_ue_layer": np.zeros((6,), dtype=np.int32),
                "actual_ue_tbs_capacity_bytes": np.zeros((6,), dtype=np.int32),
                "native_reject_diagnostics_available": True,
                "native_reject_wrong_cell_slot_count": reject_zeros,
                "native_reject_empty_slot_count": reject_zeros,
                "native_reject_prg_mask_count": reject_zeros,
                "native_reject_oob_slot_count": reject_zeros,
                "native_reject_duplicate_ue_count": reject_zeros,
            }

        def close(self):
            pass

    original_completion = (
        eval_checkpoint_online._validate_non_starvation_guard_completion
    )

    def capture_completion(**kwargs):
        captured["completed"] = int(kwargs["completed_action_ttis"])
        captured["full_count"] = len(kwargs["full_rows"])
        captured["measured_count"] = len(kwargs["measured_rows"])
        return original_completion(**kwargs)

    monkeypatch.setattr(
        sys,
        "argv",
        _guard_main_argv(
            tmp_path,
            "--max-steps",
            "5000",
            "--episode-horizon",
            "5000",
            "--stats-warmup-steps",
            "5000",
            "--save-steps",
            "0",
        ),
    )
    monkeypatch.setattr(
        eval_checkpoint_online,
        "_load_actor_checkpoint",
        lambda *_args: (_blank_actor(), {}),
    )
    monkeypatch.setattr(
        eval_checkpoint_online,
        "FeatureSnapshotBuilder",
        lambda _cfg: SimpleNamespace(build=lambda _obs: object()),
    )
    monkeypatch.setattr(
        eval_checkpoint_online,
        "SparseEdgeGenerator",
        lambda _cfg: SimpleNamespace(build_graph=lambda _snapshot: graph),
    )
    monkeypatch.setattr(eval_checkpoint_online, "BridgeEnvClient", lambda _cfg: FakeEnv())
    monkeypatch.setattr(eval_checkpoint_online, "_graph_to_device", lambda value, _device: value)
    monkeypatch.setattr(eval_checkpoint_online, "_independent_deletion_equal", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        eval_checkpoint_online,
        "resolve_runtime_candidate_contract",
        lambda *_args, **_kwargs: SimpleNamespace(runtime_candidate_width=2),
    )
    monkeypatch.setattr(
        eval_checkpoint_online,
        "compare_ctbs_reference_labels",
        lambda *_args, **_kwargs: SimpleNamespace(
            mcs_mismatch_count=0,
            layer_mismatch_count=0,
            tbs_mismatch_count=0,
            reference_label_deletion_equal=True,
            all_exact=True,
        ),
    )
    monkeypatch.setattr(eval_checkpoint_online, "extract_reward_metrics", lambda _info: {})
    monkeypatch.setattr(
        eval_checkpoint_online,
        "_validate_non_starvation_guard_completion",
        capture_completion,
    )

    with pytest.raises(RuntimeError, match="exactly 5000 completed"):
        eval_checkpoint_online.main()
    assert captured == {"completed": 1, "full_count": 51, "measured_count": 0}


def test_non_starvation_guard_main_requires_ctbs_v1_before_simulator_start(
    monkeypatch,
    tmp_path,
) -> None:
    from training.stageb_hgraph import eval_checkpoint_online

    class SimulatorMustNotStart:
        def __init__(self, *_args, **_kwargs) -> None:
            pytest.fail("simulator startup was reached before guard validation")

    monkeypatch.setattr(eval_checkpoint_online, "BridgeEnvClient", SimulatorMustNotStart)
    monkeypatch.setattr(
        eval_checkpoint_online,
        "_load_actor_checkpoint",
        lambda *_args: (_blank_actor(), {}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_checkpoint_online.py",
            "--checkpoint",
            str(tmp_path / "actor.pt"),
            "--socket-path",
            "simulator-must-not-start.sock",
            "--independent-policy-diagnostics",
            "1",
            "--independent-headroom-accounting",
            "raw-v1",
            "--independent-non-starvation-guard",
            "ctbs-blank-v1",
            "--device",
            "cpu",
        ],
    )

    with pytest.raises(ValueError, match="headroom-accounting ctbs-v1"):
        eval_checkpoint_online.main()


def test_deterministic_evaluator_uses_frozen_runtime_width_without_truncation() -> None:
    from training.stageb_hgraph.eval_checkpoint_online import _select_joint_action

    selected = _select_joint_action(
        _actor(),
        _graph(),
        decode_mode="argmax",
        candidate_policy_mode="independent",
        independent_runtime_candidate_width=3,
    )

    assert selected["runtime_candidate_width"] == 3
    assert selected["candidate_truncation_count"] == 0.0


def test_deterministic_evaluator_exposes_materialized_chosen_ue_vector_for_diagnostics() -> None:
    """Catches cohort collection reconstructing a different action than the submitted one."""
    from training.stageb_hgraph.eval_checkpoint_online import _select_joint_action

    selected = _select_joint_action(
        _actor(),
        _graph(),
        decode_mode="hierarchical_argmax",
        candidate_policy_mode="independent",
        independent_policy_diagnostics=True,
    )

    chosen = selected["decoded_chosen_ue_per_prg"]
    assert isinstance(chosen, torch.Tensor)
    assert chosen.device.type == "cpu"
    assert chosen.shape == (9,)
    assert int((chosen >= 0).sum().item()) == int(
        (~selected["independent_policy_trace"].diagnostics.selected_blank).sum().item()
    )


@pytest.mark.parametrize(
    ("decode_mode", "expected_internal_mode"),
    [("argmax", "greedy"), ("hierarchical_argmax", "hierarchical_greedy")],
)
def test_independent_deterministic_evaluator_routes_requested_decode_mode(
    monkeypatch,
    decode_mode: str,
    expected_internal_mode: str,
) -> None:
    """Catches evaluation silently using flat argmax for hierarchical requests."""
    from training.stageb_hgraph import eval_checkpoint_online

    original = eval_checkpoint_online.select_independent_action
    observed_modes: list[str] = []

    def select_with_mode_spy(*args, **kwargs):
        observed_modes.append(kwargs["mode"])
        return original(*args, **kwargs)

    monkeypatch.setattr(eval_checkpoint_online, "select_independent_action", select_with_mode_spy)
    graph = _graph()
    selected = eval_checkpoint_online._select_joint_action(
        _actor(),
        graph,
        decode_mode=decode_mode,
        candidate_policy_mode="independent",
    )

    assert observed_modes == [expected_internal_mode]
    assert selected["action_replay_mismatch_count"] == 0.0
    assert all(
        int(reject.sum().item()) == 0
        for reject in type0_action_preflight_reject_counts(
            graph,
            torch.as_tensor(selected["action_ue_select"]),
            torch.as_tensor(selected["action_prg_alloc"]),
        ).values()
    )


def test_evaluator_cli_accepts_hierarchical_argmax(monkeypatch) -> None:
    from training.stageb_hgraph import eval_checkpoint_online

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_checkpoint_online.py",
            "--checkpoint",
            "actor.pt",
            "--socket-path",
            "bridge.sock",
            "--decode-mode",
            "hierarchical_argmax",
        ],
    )

    assert eval_checkpoint_online.parse_args().decode_mode == "hierarchical_argmax"


def test_evaluator_cli_accepts_runtime_width_equivalence_flags(monkeypatch) -> None:
    from training.stageb_hgraph import eval_checkpoint_online

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_checkpoint_online.py",
            "--checkpoint",
            "actor.pt",
            "--socket-path",
            "bridge.sock",
            "--independent-runtime-candidate-width",
            "14",
            "--independent-equivalence-reference-width",
            "13",
            "--independent-equivalence-protocol",
            "fp32-w0-v2",
        ],
    )

    args = eval_checkpoint_online.parse_args()
    assert args.independent_runtime_candidate_width == 14
    assert args.independent_equivalence_reference_width == 13
    assert args.independent_equivalence_protocol == "fp32-w0-v2"


def test_evaluator_cli_enables_only_the_ctbs_headroom_treatment(monkeypatch) -> None:
    from training.stageb_hgraph import eval_checkpoint_online

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_checkpoint_online.py",
            "--checkpoint",
            "actor.pt",
            "--socket-path",
            "bridge.sock",
            "--independent-headroom-accounting",
            "ctbs-v1",
        ],
    )

    args = eval_checkpoint_online.parse_args()
    assert args.independent_headroom_accounting == "ctbs-v1"


def test_evaluator_cli_accepts_full_horizon_blank_expiry_protocol(monkeypatch) -> None:
    from training.stageb_hgraph import eval_checkpoint_online
    from training.stageb_hgraph.blank_expiry_attribution_audit import (
        V2_FULL_HORIZON_PROTOCOL_ID,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_checkpoint_online.py",
            "--checkpoint",
            "actor.pt",
            "--socket-path",
            "bridge.sock",
            "--independent-blank-expiry-attribution-protocol",
            V2_FULL_HORIZON_PROTOCOL_ID,
        ],
    )

    assert (
        eval_checkpoint_online.parse_args().independent_blank_expiry_attribution_protocol
        == V2_FULL_HORIZON_PROTOCOL_ID
    )


def test_nonindependent_hierarchical_argmax_rejects_before_simulator_start(
    monkeypatch,
    tmp_path,
) -> None:
    from training.stageb_hgraph import eval_checkpoint_online

    class SimulatorMustNotStart:
        def __init__(self, *_args, **_kwargs) -> None:
            pytest.fail("simulator startup was reached before decode validation")

    monkeypatch.setattr(eval_checkpoint_online, "BridgeEnvClient", SimulatorMustNotStart)
    monkeypatch.setattr(eval_checkpoint_online, "_load_actor_checkpoint", lambda *_args: (object(), {}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_checkpoint_online.py",
            "--checkpoint",
            str(tmp_path / "candidate.pt"),
            "--socket-path",
            "simulator-must-not-start.sock",
            "--decode-mode",
            "hierarchical_argmax",
            "--device",
            "cpu",
        ],
    )

    with pytest.raises(ValueError, match="hierarchical_argmax.*independent"):
        eval_checkpoint_online.main()


def test_nonindependent_decode_path_rejects_hierarchical_argmax() -> None:
    from training.stageb_hgraph.eval_checkpoint_online import _select_joint_action

    with pytest.raises(ValueError, match="hierarchical_argmax.*independent"):
        _select_joint_action(object(), object(), decode_mode="hierarchical_argmax")


def test_independent_sample_evaluation_remains_rejected() -> None:
    from training.stageb_hgraph.eval_checkpoint_online import _select_joint_action

    with pytest.raises(ValueError, match="deterministic"):
        _select_joint_action(
            _actor(),
            _graph(),
            decode_mode="sample",
            candidate_policy_mode="independent",
        )


def test_pfq_deletion_equality_uses_hierarchical_treatment_mode(monkeypatch) -> None:
    """Catches the deletion audit defaulting to greedy under a hierarchical request."""
    from training.stageb_hgraph import eval_checkpoint_online

    original = eval_checkpoint_online.select_independent_action
    observed_modes: list[str] = []

    def select_with_mode_spy(*args, **kwargs):
        observed_modes.append(kwargs["mode"])
        return original(*args, **kwargs)

    monkeypatch.setattr(eval_checkpoint_online, "select_independent_action", select_with_mode_spy)
    observation = make_candidate_observation(n_prg=3, ue_feat_dim=16)
    edge_generator = SparseEdgeGenerator(
        EdgeGeneratorConfig(
            candidate_mode="all_demand_ues",
            include_pfq_score_anchor=False,
            include_candidate_risk_features=False,
        )
    )

    assert eval_checkpoint_online._independent_deletion_equal(
        _actor().eval(),
        observation,
        edge_generator,
        eval_checkpoint_online.SnapshotBuilderConfig(),
        torch.device("cpu"),
        mode="hierarchical_greedy",
        runtime_candidate_width=3,
    )
    assert observed_modes == ["hierarchical_greedy", "hierarchical_greedy"]


def test_deterministic_evaluator_rejects_diagnostics_for_non_independent_actor() -> None:
    from training.stageb_hgraph.eval_checkpoint_online import _select_joint_action

    with pytest.raises(ValueError, match="independent actor"):
        _select_joint_action(
            object(),
            object(),
            decode_mode="argmax",
            independent_policy_diagnostics=True,
        )


def test_online_clis_accept_independent_policy_diagnostics_flag(monkeypatch) -> None:
    from training.stageb_hgraph import eval_checkpoint_online, online_ppo

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "online_ppo.py",
            "--sim-bin",
            "simulator",
            "--independent-policy-diagnostics",
            "1",
        ],
    )
    assert online_ppo.parse_args().independent_policy_diagnostics == 1

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_checkpoint_online.py",
            "--checkpoint",
            "actor.pt",
            "--socket-path",
            "bridge.sock",
            "--independent-policy-diagnostics",
            "1",
        ],
    )
    assert eval_checkpoint_online.parse_args().independent_policy_diagnostics == 1


def test_online_ppo_cli_accepts_ctbs_headroom_treatment(monkeypatch) -> None:
    from training.stageb_hgraph import online_ppo

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "online_ppo.py",
            "--sim-bin",
            "simulator",
            "--independent-headroom-accounting",
            "ctbs-v1",
        ],
    )

    assert online_ppo.parse_args().independent_headroom_accounting == "ctbs-v1"


def test_online_ppo_cli_accepts_staged_safety_prefix(monkeypatch) -> None:
    from training.stageb_hgraph import online_ppo

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "online_ppo.py",
            "--sim-bin",
            "simulator",
            "--safety-prefix-iters",
            "8",
        ],
    )

    assert online_ppo.parse_args().safety_prefix_iters == 8


def test_online_episode_runner_enables_ctbs_on_the_bridge_only_when_requested(
    tmp_path,
) -> None:
    base = dict(
        seed_list="",
        topology_seed_mode="fixed",
        topology_seed=41,
        sim_env=[],
        out_dir=str(tmp_path),
        connect_timeout_s=20.0,
    )
    ctbs = OnlineEpisodeRunner(
        SimpleNamespace(**base, independent_headroom_accounting="ctbs-v1")
    )
    raw = OnlineEpisodeRunner(
        SimpleNamespace(**base, independent_headroom_accounting="raw-v1")
    )
    ctbs.socket_path = "/tmp/ctbs.sock"
    raw.socket_path = "/tmp/raw.sock"

    assert ctbs._bridge_env_config().enable_ctbs is True
    assert raw._bridge_env_config().enable_ctbs is False


def test_online_ppo_ctbs_handshake_and_post_action_exact_parity() -> None:
    graph = _graph()
    actor = _actor()
    actor_out, _ = _evaluate_graph(actor, GraphValueHead(hidden_dim=16), graph)
    source = _ctbs_inputs()

    class FakeEnv:
        def request_pre_action_snapshot(self, *, expected_tti: int):
            assert expected_tti == source.tti
            return PreActionPhySnapshot(
                tti=source.tti,
                snapshot_id=source.snapshot_id,
                config_sha256=source.config_sha256,
                olla_source_code=1,
                capture_phase_code=1,
                ue_ids=np.asarray(source.ue_ids, dtype=np.int32),
                serving_cell=np.asarray(source.serving_cell, dtype=np.int32),
                wb_sinr_first_layer_db=np.asarray(
                    source.wb_sinr_first_layer_db, dtype=np.float32
                ),
                olla_delta=np.asarray(source.olla_delta, dtype=np.float32),
                olla_delta_up=np.asarray(source.olla_delta_up, dtype=np.float32),
                olla_delta_down=np.asarray(source.olla_delta_down, dtype=np.float32),
                prior_tb_err=np.asarray(source.prior_tb_err, dtype=np.int32),
                prg_layer_cap=np.asarray(source.prg_layer_cap, dtype=np.uint8),
            )

    runner = SimpleNamespace(
        args=SimpleNamespace(independent_headroom_accounting="ctbs-v1"),
        env=FakeEnv(),
    )
    inputs = _ctbs_inputs_for_rollout_step(
        runner,
        {
            "tti": source.tti,
            "harq_enabled": 0,
            "ctbs_config_available": True,
            "ctbs_config": source.config.as_vector(),
            "ctbs_config_sha256": source.config_sha256,
        },
    )
    sampled = _sample_joint_action(
        actor,
        actor_out,
        graph,
        ctbs_headroom_inputs=inputs,
    )
    chosen = np.asarray(sampled["chosen_ue_per_prg"], dtype=np.int32).reshape(3, 3)
    actual_mcs = np.full((6,), -1, dtype=np.int32)
    actual_layer = np.zeros((6,), dtype=np.int32)
    actual_tbs = np.zeros((6,), dtype=np.int32)
    for ue in range(6):
        cell = source.serving_cell[ue]
        local_prgs = np.flatnonzero(chosen[cell] == ue).tolist()
        if not local_prgs:
            continue
        actual_mcs[ue] = source.reconstructed_mcs_for_ue(ue)
        state = CtbsHeadroomState(
            initial_demand_bytes=0.0,
            reconstructed_mcs=int(actual_mcs[ue]),
            config=source.config,
        )
        for local_prg in local_prgs:
            state = state.with_selected_prg(
                local_prg,
                layer_cap=source.prg_layer_cap[ue][local_prg],
            )
        actual_layer[ue] = min(state.selected_layer_caps)
        actual_tbs[ue] = state.cumulative_tbs_bytes

    parity = _validate_ctbs_rollout_parity(
        inputs,
        sampled,
        {
            "actual_ue_mcs": actual_mcs,
            "actual_ue_layer": actual_layer,
            "actual_ue_tbs_capacity_bytes": actual_tbs,
        },
        graph,
    )

    assert inputs is not None
    assert inputs.snapshot_id == source.snapshot_id
    assert parity is not None and parity.all_exact


def test_online_ppo_rejects_diagnostics_for_non_independent_actor_before_simulator_start(
    monkeypatch,
    tmp_path,
) -> None:
    from training.stageb_hgraph import online_ppo

    class SimulatorMustNotStart:
        def __init__(self, _args) -> None:
            pytest.fail("simulator startup was reached before diagnostics validation")

    monkeypatch.setattr(online_ppo, "OnlineEpisodeRunner", SimulatorMustNotStart)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "online_ppo.py",
            "--sim-bin",
            "simulator-must-not-start",
            "--action-head-mode",
            "candidate",
            "--independent-policy-diagnostics",
            "1",
            "--device",
            "cpu",
            "--out-dir",
            str(tmp_path / "run"),
        ],
    )

    with pytest.raises(
        ValueError,
        match="--independent-policy-diagnostics requires --action-head-mode independent",
    ):
        online_ppo.main()


def test_independent_sampling_resolves_runtime_topology_width_from_graph() -> None:
    graph = _graph()  # Two associated UEs per cell.
    actor = PFQIndependentReuseActor(
        IndependentActorConfig(
            hidden_dim=16,
            message_layers=1,
            max_candidates=1,  # Training-observed width, not an inference cap.
            up_edge_attr_dim=5,
        )
    )
    critic = GraphValueHead(hidden_dim=16)
    actor_out, _ = _evaluate_graph(actor, critic, graph)

    sampled = _sample_joint_action(actor, actor_out, graph)

    assert sampled["candidate_packed"].max_candidates == 2
    logp, _entropy, replayed = replay_independent_trace(
        actor,
        graph,
        sampled["candidate_packed"],
        sampled["independent_policy_trace"],
    )
    assert torch.isfinite(logp)
    assert torch.equal(
        replayed.action_ue_select.cpu(),
        torch.as_tensor(sampled["action_ue_select"], dtype=torch.int32),
    )
    assert torch.equal(
        replayed.action_prg_alloc.cpu(),
        torch.as_tensor(sampled["action_prg_alloc"], dtype=torch.int16),
    )


def test_independent_replay_policy_gradient_reaches_attention_and_ue_scorer() -> None:
    torch.manual_seed(17)
    actor = _actor()
    critic = GraphValueHead(hidden_dim=16)
    graph = _graph()
    actor_out, _ = _evaluate_graph(actor, critic, graph)
    sampled = _sample_joint_action(actor, actor_out, graph)
    trace = sampled["independent_policy_trace"]

    logp, _entropy, decoded = replay_independent_trace(
        actor,
        graph,
        sampled["candidate_packed"],
        trace,
    )
    (-logp).backward()

    assert torch.equal(
        decoded.action_ue_select.cpu(),
        torch.as_tensor(sampled["action_ue_select"], dtype=torch.int32),
    )
    assert torch.equal(
        decoded.action_prg_alloc.cpu(),
        torch.as_tensor(sampled["action_prg_alloc"], dtype=torch.int16),
    )
    assert actor.reuse_attention.in_proj_weight.grad is not None
    assert actor.reuse_attention.in_proj_weight.grad.abs().sum() > 0.0
    scorer_grads = [p.grad for p in actor.ue_scorer.parameters()]
    assert any(grad is not None and grad.abs().sum() > 0.0 for grad in scorer_grads)
    forbidden = ("pfq", "slot", "safe_fill", "repair", "teacher")
    assert not any(any(word in name.lower() for word in forbidden) for name, _ in actor.named_parameters())


def test_independent_checkpoint_loader_requires_explicit_family_and_no_pfq(tmp_path) -> None:
    from training.stageb_hgraph.eval_checkpoint_online import _load_actor_checkpoint

    actor = _actor()
    valid = {
        "actor_state": actor.state_dict(),
        "actor_config": actor.model_config_dict(),
        "graph_config": {
            "candidate_mode": "all_demand_ues",
            "include_pfq_score_anchor": False,
            "include_candidate_risk_features": False,
        },
    }
    valid_path = tmp_path / "valid.pt"
    torch.save(valid, valid_path)
    loaded, _ = _load_actor_checkpoint(valid_path, torch.device("cpu"))
    assert isinstance(loaded, PFQIndependentReuseActor)

    invalid = dict(valid)
    invalid["actor_config"] = dict(valid["actor_config"], requires_pfq_features=True)
    invalid_path = tmp_path / "invalid.pt"
    torch.save(invalid, invalid_path)
    with pytest.raises(ValueError, match="PFQ|pfq"):
        _load_actor_checkpoint(invalid_path, torch.device("cpu"))

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import torch

from training.stageb_hgraph.ctbs_headroom import CtbsHeadroomInputs
from training.stageb_hgraph.bridge_env import PreActionPhySnapshot
from training.stageb_hgraph.graph_ops import pack_prg_candidates
from training.stageb_hgraph.independent_actor import (
    IndependentActorConfig,
    PFQIndependentReuseActor,
)
from training.stageb_hgraph.online_imitation import (
    IndependentBcDatasetReader,
    IndependentBcDatasetWriter,
    _canonical_independent_teacher_trace,
    _independent_bc_train_step,
    _model_tensor_sha256,
)
from training.stageb_hgraph.phy_tbs_alignment_audit import PhyTbsConfig
from training.stageb_hgraph.edge_generator import EdgeGeneratorConfig, SparseEdgeGenerator
from training.stageb_hgraph.tests.graph_fixtures import (
    make_candidate_graph,
    make_candidate_observation,
)


def _metadata(*, count: int = 1, feature_stage: str = "dimensionless") -> dict[str, object]:
    return {
        "feature_stage": feature_stage,
        "headroom_accounting": "ctbs-v1",
        "teacher_policy": "pfq_passthrough",
        "candidate_mode": "all_demand_ues",
        "include_pfq_score_anchor": False,
        "actor_family": "pfq_independent_reuse_v1",
        "collection_ablation": "full",
        "topology_seed_mode": "fixed",
        "topology_seed": 41,
        "model_seed": 801,
        "expected_episode_count": 1,
        "expected_steps_per_episode": count,
        "expected_record_count": count,
    }


def _ctbs_inputs() -> CtbsHeadroomInputs:
    return CtbsHeadroomInputs.from_arrays(
        tti=1,
        snapshot_id=41,
        config_sha256="12" * 32,
        config=PhyTbsConfig(0, 25.99, 0.1, 16, 11, 4),
        ue_ids=np.arange(6),
        serving_cell=np.asarray([0, 0, 1, 1, 2, 2]),
        wb_sinr_first_layer_db=np.full((6,), -4.0),
        olla_delta=np.zeros((6,)),
        olla_delta_up=np.ones((6,)),
        olla_delta_down=np.full((6,), 0.1),
        prior_tb_err=np.full((6,), -1),
        prg_layer_cap=np.full((6, 2), 4),
    )


def _actor() -> PFQIndependentReuseActor:
    return PFQIndependentReuseActor(
        IndependentActorConfig(
            hidden_dim=16,
            message_layers=1,
            max_candidates=2,
            up_edge_attr_dim=5,
        )
    )


def _graph():
    from training.stageb_hgraph.feature_snapshot import FeatureSnapshotBuilder
    graph=SparseEdgeGenerator(EdgeGeneratorConfig(
        candidate_mode="all_demand_ues",ue_prg_topk=2,
        include_pfq_score_anchor=False,include_candidate_risk_features=False,
    )).build_graph(FeatureSnapshotBuilder().build(make_candidate_observation(n_prg=2,ue_feat_dim=16)))
    return graph


def test_ctbs_teacher_prunes_only_exhausted_prefix_and_remains_strict():
    from training.stageb_hgraph.online_imitation import _align_ctbs_teacher
    graph=_graph()
    graph.snapshot.ue_features.values[:,0]=1
    ue=torch.arange(6)
    raw=torch.tensor([0,2,4,0,3,4])
    aligned,audit=_align_ctbs_teacher(graph,ue,raw,_ctbs_inputs())
    assert aligned.tolist()==[0,2,4,-1,3,-1]
    assert raw.tolist()==[0,2,4,0,3,4]
    assert audit['dropped_native_indices']==[3,5]
    trace=_canonical_independent_teacher_trace(_actor(),graph,
        pack_prg_candidates(graph,max_candidates=2,fixed_width=True),ue,aligned,_ctbs_inputs())
    assert trace.ctbs_headroom_inputs is not None
    wrong=torch.tensor([2,2,4,0,3,4])
    with pytest.raises(ValueError,match='serving cell'):
        _align_ctbs_teacher(graph,ue,wrong,_ctbs_inputs())
    graph.snapshot.static_meta.action_mask_prg_cell[0,1]=False
    with pytest.raises(ValueError,match='PRG unavailable'):
        _align_ctbs_teacher(graph,ue,raw,_ctbs_inputs())


def test_streamed_ctbs_dataset_replay_matches_online_update_exactly(tmp_path) -> None:
    """Catches offline replay changing the legal trace, loss, or optimizer update."""
    torch.manual_seed(7)
    graph = _graph()
    action_ue = torch.tensor([0, 1, 2, 3, 4, 5])
    action_prg = torch.tensor([0, 2, 4, -1, 3, 5])
    online_actor = _actor()
    offline_actor = _actor()
    offline_actor.load_state_dict(online_actor.state_dict())
    online_opt = torch.optim.AdamW(online_actor.parameters(), lr=3.0e-4, weight_decay=1.0e-5)
    offline_opt = torch.optim.AdamW(offline_actor.parameters(), lr=3.0e-4, weight_decay=1.0e-5)
    trace = _canonical_independent_teacher_trace(
        online_actor,
        graph,
        pack_prg_candidates(graph, max_candidates=2, fixed_width=True),
        action_ue,
        action_prg,
        _ctbs_inputs(),
    )

    writer = IndependentBcDatasetWriter(
        tmp_path / "dataset",
        initial_actor_sha256=_model_tensor_sha256(online_actor),
        metadata=_metadata(),
    )
    writer.write(
        graph=graph,
        teacher_trace=trace,
        teacher_action_ue=action_ue,
        teacher_action_prg=action_prg,
        next_obs={"obs_ue_features": np.ones((6, 16), dtype=np.float32)},
        next_info={"actual_ue_goodput_bytes": np.arange(6, dtype=np.float64)},
        step_metadata={"episode": 1, "episode_seed":41,"topology_seed":41,"step": 0, "global_step": 0, "tti": 1, "done": False, "reward_raw": 0.5},
    )
    writer.finalize(max_candidates=2)

    online = _independent_bc_train_step(online_actor, online_opt, graph, trace, 5.0)
    record = next(iter(IndependentBcDatasetReader(tmp_path / "dataset")))
    assert np.array_equal(record["next_obs"]["obs_ue_features"], np.ones((6, 16)))
    assert np.array_equal(record["next_info"]["actual_ue_goodput_bytes"], np.arange(6))
    offline = _independent_bc_train_step(
        offline_actor,
        offline_opt,
        record["graph"],
        record["teacher_trace"],
        5.0,
    )

    assert offline["loss"] == pytest.approx(online["loss"], abs=0.0)
    assert _model_tensor_sha256(offline_actor) == _model_tensor_sha256(online_actor)

    # Corrected-unit training must reject a legacy-capacity archive even if
    # its files and hashes are otherwise intact; it cannot be relabeled/reused.
    manifest_path = tmp_path / "dataset" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema"] = "independent-bc-ctbs-v1"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema mismatch"):
        IndependentBcDatasetReader(tmp_path / "dataset")


def test_dataset_reader_fails_closed_on_partial_or_modified_archive(tmp_path) -> None:
    """Catches training from a partial archive or a record whose hash no longer matches."""
    partial = tmp_path / "partial"
    partial.mkdir()
    torch.save({"index": 0}, partial / "step_00000000.pt")
    with pytest.raises(ValueError, match="manifest"):
        IndependentBcDatasetReader(partial)

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "independent-bc-ctbs-physical-bytes-v2",
                "count": 1,
                "records": [{"file": "step_00000000.pt", "sha256": "00" * 32}],
                "dataset_sha256": "00" * 32,
                "max_candidates": 2,
            }
        ),
        encoding="utf-8",
    )
    torch.save({"index": 0}, malformed / "step_00000000.pt")
    with pytest.raises(ValueError, match="hash"):
        IndependentBcDatasetReader(malformed)


def test_collect_only_loss_does_not_update_initialized_actor() -> None:
    """Catches collection-only mode training before the first-transition audit."""
    torch.manual_seed(9)
    graph = _graph()
    actor = _actor()
    optimizer = torch.optim.AdamW(actor.parameters(), lr=3.0e-4)
    trace = _canonical_independent_teacher_trace(
        actor,
        graph,
        pack_prg_candidates(graph, max_candidates=2, fixed_width=True),
        torch.tensor([0, 1, 2, 3, 4, 5]),
        torch.tensor([0, 2, 4, -1, 3, 5]),
        _ctbs_inputs(),
    )
    before = _model_tensor_sha256(actor)

    result = _independent_bc_train_step(
        actor, optimizer, graph, trace, 5.0, update=False
    )

    assert result["loss"] > 0.0
    assert _model_tensor_sha256(actor) == before


def test_dataset_reader_rejects_teacher_actions_that_disagree_with_frozen_trace(
    tmp_path,
) -> None:
    """Catches silently projecting a malformed teacher label during offline replay."""
    graph = _graph()
    actor = _actor()
    action_ue = torch.tensor([0, 1, 2, 3, 4, 5])
    action_prg = torch.tensor([0, 2, 4, -1, 3, 5])
    trace = _canonical_independent_teacher_trace(
        actor,
        graph,
        pack_prg_candidates(graph, max_candidates=2, fixed_width=True),
        action_ue,
        action_prg,
        _ctbs_inputs(),
    )
    writer = IndependentBcDatasetWriter(
        tmp_path / "dataset",
        initial_actor_sha256=_model_tensor_sha256(actor),
        metadata=_metadata(),
    )
    wrong_action_ue = action_ue.clone()
    wrong_action_ue[0] = 2
    writer.write(
        graph=graph,
        teacher_trace=trace,
        teacher_action_ue=wrong_action_ue,
        teacher_action_prg=action_prg,
        next_obs={},
        next_info={},
        step_metadata={"episode": 1,"episode_seed":41,"topology_seed":41, "step": 0, "global_step": 0, "tti": 1, "done": False},
    )
    writer.finalize(max_candidates=2)

    with pytest.raises(ValueError, match="teacher actions disagree"):
        next(iter(IndependentBcDatasetReader(tmp_path / "dataset")))


def test_dataset_contract_rejects_truncation_wrong_stage_and_tti_mismatch(tmp_path) -> None:
    """Catches self-consistent but incomplete or semantically wrong archives."""
    graph = _graph()
    actor = _actor()
    action_ue = torch.tensor([0, 1, 2, 3, 4, 5])
    action_prg = torch.tensor([0, 2, 4, -1, 3, 5])
    trace = _canonical_independent_teacher_trace(
        actor,
        graph,
        pack_prg_candidates(graph, max_candidates=2, fixed_width=True),
        action_ue,
        action_prg,
        _ctbs_inputs(),
    )
    writer = IndependentBcDatasetWriter(
        tmp_path / "truncated",
        initial_actor_sha256=_model_tensor_sha256(actor),
        metadata=_metadata(count=2),
    )
    writer.write(
        graph=graph,
        teacher_trace=trace,
        teacher_action_ue=action_ue,
        teacher_action_prg=action_prg,
        next_obs={},
        next_info={},
        step_metadata={"episode": 1,"episode_seed":41,"topology_seed":41, "step": 0, "global_step": 0, "tti": 1, "done": True},
    )
    with pytest.raises(ValueError, match="early done|expected_record_count"):
        writer.finalize(max_candidates=2)

    complete = IndependentBcDatasetWriter(
        tmp_path / "complete",
        initial_actor_sha256=_model_tensor_sha256(actor),
        metadata=_metadata(feature_stage="dimensionless"),
    )
    complete.write(
        graph=graph,
        teacher_trace=trace,
        teacher_action_ue=action_ue,
        teacher_action_prg=action_prg,
        next_obs={},
        next_info={},
        step_metadata={"episode": 1,"episode_seed":41,"topology_seed":41, "step": 0, "global_step": 0, "tti": 2, "done": False},
    )
    complete.finalize(max_candidates=2)
    with pytest.raises(ValueError, match="feature_stage"):
        IndependentBcDatasetReader(
            tmp_path / "complete",
            expected_metadata={"feature_stage": "legacy"},
        )
    with pytest.raises(ValueError, match="CTBS tuple TTI"):
        list(IndependentBcDatasetReader(tmp_path / "complete"))


def test_online_main_collect_only_writes_complete_archive_without_weight_update(
    monkeypatch, tmp_path
) -> None:
    """Catches the CLI collection path updating weights or emitting an incomplete archive."""
    from training.stageb_hgraph import online_imitation

    obs = make_candidate_observation(n_prg=2, ue_feat_dim=16)
    obs["teacher_action_ue_select"] = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int32)
    obs["obs_ue_features"][:, 0] = 1.0
    obs["teacher_action_prg_alloc"] = np.asarray([0, 2, 4, 0, 3, 4], dtype=np.int16)
    config = PhyTbsConfig(0, 25.99, 0.1, 16, 11, 4)
    info = {
        "tti": 1,
        "harq_enabled": 0,
        "ctbs_config_available": True,
        "ctbs_config": config.as_vector(),
        "ctbs_config_sha256": "12" * 32,
    }

    class FakeEnv:
        def request_pre_action_snapshot(self, expected_tti):
            assert expected_tti == 1
            return PreActionPhySnapshot(
                tti=1,
                snapshot_id=41,
                config_sha256="12" * 32,
                olla_source_code=1,
                capture_phase_code=1,
                ue_ids=np.arange(6),
                serving_cell=np.asarray([0, 0, 1, 1, 2, 2]),
                wb_sinr_first_layer_db=np.full(6, -4.0),
                olla_delta=np.zeros(6),
                olla_delta_up=np.ones(6),
                olla_delta_down=np.full(6, 0.1),
                prior_tb_err=np.full(6, -1),
                prg_layer_cap=np.full((6, 2), 4),
            )

        def step(self, _action_ue, _action_prg):
            assert _action_prg.tolist() == [0, 2, 4, -1, 3, -1]
            return obs, 0.25, True, {
                **info,
                "actual_ue_mcs": [0, 0, 0, 0, 0, 0],
                "actual_ue_layer": [4, 0, 4, 4, 4, 0],
                "actual_ue_tbs_capacity_bytes": [253, 0, 253, 253, 253, 0],
            }

    class FakeRunner:
        def __init__(self, args):
            self.args = args
            self.ctbs_enabled = True
            self.env = FakeEnv()
            self.sim_env_overrides = {"CUMAC_DPA_FEATURE_MODE": "legacy"}
            self.resolved_topology_seed_mode = "fixed"

        def begin_rollout(self, _episode):
            return obs, 0.0, info, 41, 41, False

        def record_transition(self, *_args):
            return None

        def close_episode(self, **_kwargs):
            return None

    monkeypatch.setattr(online_imitation, "OnlineEpisodeRunner", FakeRunner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "online_imitation.py",
            "--sim-bin",
            "unused",
            "--out-dir",
            str(tmp_path / "out"),
            "--episodes",
            "1",
            "--episode-horizon",
            "1",
            "--max-steps",
            "1",
            "--action-head-mode",
            "independent",
            "--include-pfq-score-anchor",
            "0",
            "--independent-headroom-accounting",
            "ctbs-v1",
            "--independent-bc-dataset-out",
            str(tmp_path / "dataset"),
            "--independent-bc-collect-only",
            "1",
            "--sim-env",
            "CUMAC_DPA_FEATURE_MODE=legacy",
            "--hidden-dim",
            "16",
            "--message-layers",
            "1",
            "--device",
            "cpu",
            "--seed",
            "801",
            "--topology-seed",
            "41",
        ],
    )

    assert online_imitation.main() == 0
    checkpoint = torch.load(
        tmp_path / "out" / "hgraph_independent_bc_last.pt",
        map_location="cpu",
        weights_only=False,
    )
    restored = PFQIndependentReuseActor(
        IndependentActorConfig(**{
            key: value
            for key, value in checkpoint["actor_config"].items()
            if key in IndependentActorConfig.__dataclass_fields__
        })
    )
    restored.load_state_dict(checkpoint["actor_state"])
    assert _model_tensor_sha256(restored) == checkpoint["initial_actor_sha256"]
    reader = IndependentBcDatasetReader(
        tmp_path / "dataset",
        expected_metadata={
            "feature_stage": "legacy",
            "topology_seed": 41,
            "model_seed": 801,
            "expected_record_count": 1,
            "teacher_alignment": "ctbs_prefix_prune_v1",
        },
    )
    records = list(reader)
    assert len(records) == 1
    assert records[0]["teacher_action_prg"].tolist() == [0, 2, 4, -1, 3, -1]
    assert records[0]["metadata"]["teacher_alignment"]["raw_action_prg"] == [0, 2, 4, 0, 3, 4]
    assert records[0]["metadata"]["teacher_alignment"]["dropped_count"] == 2
    assert records[0]["metadata"]["ctbs_post_action_exact"] is True

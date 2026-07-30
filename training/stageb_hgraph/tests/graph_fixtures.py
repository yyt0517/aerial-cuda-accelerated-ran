from __future__ import annotations

import numpy as np

from training.stageb_hgraph.edge_generator import EdgeGeneratorConfig, SparseEdgeGenerator
from training.stageb_hgraph.feature_snapshot import FeatureSnapshotBuilder, SnapshotBuilderConfig
from training.stageb_hgraph.types import SparseEntityGraph


def make_candidate_observation(
    n_prg: int = 2,
    ue_feat_dim: int = 12,
    backlog_bytes: tuple[float, float, float, float, float, float] = (
        3000.0,
        3000.0,
        3000.0,
        3000.0,
        3000.0,
        3000.0,
    ),
) -> dict[str, np.ndarray]:
    ue = np.zeros((6, ue_feat_dim), dtype=np.float32)
    ue[:, 0] = np.asarray(backlog_bytes, dtype=np.float32)
    ue[:, 1] = 1.0
    ue[:, 2] = 10.0
    ue[:, 3] = 10.0
    ue[:, 4] = 1.0
    ue[:, 6] = 1.0
    return {
        "obs_cell_features": np.zeros((3, 5), dtype=np.float32),
        "obs_ue_features": ue,
        "obs_prg_features": np.zeros((3, n_prg, 12), dtype=np.float32),
        "action_mask_ue": np.ones((6,), dtype=np.uint8),
        "action_mask_cell_ue": np.array(
            [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]],
            dtype=np.uint8,
        ),
        "action_mask_prg_cell": np.ones((3, n_prg), dtype=np.uint8),
        "post_eq_sinr": np.full((6, n_prg, 1), 10.0, dtype=np.float32),
        "n_sched_ue": np.int32(6),
    }


def make_candidate_graph(
    n_prg: int = 2,
    backlog_bytes: tuple[float, float, float, float, float, float] = (
        3000.0,
        3000.0,
        3000.0,
        3000.0,
        3000.0,
        3000.0,
    ),
) -> SparseEntityGraph:
    obs = make_candidate_observation(n_prg=n_prg, backlog_bytes=backlog_bytes)
    snapshot = FeatureSnapshotBuilder(SnapshotBuilderConfig()).build(obs)
    return SparseEdgeGenerator(
        EdgeGeneratorConfig(
            candidate_mode="all_demand_ues",
            ue_prg_topk=2,
        )
    ).build_graph(snapshot)

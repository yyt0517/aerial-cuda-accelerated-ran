from training.stageb_hgraph.feature_snapshot import FeatureSnapshotBuilder
from training.stageb_hgraph.tests.graph_fixtures import make_candidate_observation


def test_snapshot_exposes_sixteen_ue_features_in_stable_order():
    obs = make_candidate_observation(n_prg=2, ue_feat_dim=16)
    obs["obs_ue_features"][0, 12:16] = [3000.0, 6000.0, 1500.0, 7.0]

    snapshot = FeatureSnapshotBuilder().build(obs)

    assert snapshot.ue_features.dim == 16
    assert snapshot.ue_features.names[12:] == (
        "critical_ttl_backlog_bytes",
        "near_ttl_backlog_bytes",
        "age_weighted_backlog_bytes",
        "service_gap_tti",
    )
    assert snapshot.ue_features.values[0, 12:].tolist() == [
        3000.0,
        6000.0,
        1500.0,
        7.0,
    ]

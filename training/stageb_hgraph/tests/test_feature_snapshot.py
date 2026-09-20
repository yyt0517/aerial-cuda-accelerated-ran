from training.stageb_hgraph.feature_snapshot import FeatureSnapshotBuilder
from training.stageb_hgraph.tests.graph_fixtures import make_candidate_observation


def test_legacy_stage_labels_do_not_reinterpret_or_zero_values(monkeypatch):
    monkeypatch.setenv("CUMAC_DPA_FEATURE_MODE", "legacy")
    obs = make_candidate_observation(n_prg=2, ue_feat_dim=16)
    obs["obs_ue_features"][0, 12:16] = [3000, 3000, 1500, 9]
    snapshot = FeatureSnapshotBuilder().build(obs)
    assert snapshot.ue_features.names[12:] == (
        "critical_ttl_backlog_bytes", "near_ttl_backlog_bytes",
        "age_weighted_backlog_bytes", "service_gap_tti")
    assert snapshot.ue_features.values[0, 12:].tolist() == [3000, 3000, 1500, 9]


def test_snapshot_exposes_sixteen_ue_features_in_stable_order():
    obs = make_candidate_observation(n_prg=2, ue_feat_dim=16)
    obs["obs_ue_features"][0, 12:16] = [0.6931472, 1.0, 0.25, 0.0]

    snapshot = FeatureSnapshotBuilder().build(obs)

    assert snapshot.ue_features.dim == 16
    assert snapshot.ue_features.names[12:] == (
        "log1p_drain_pressure",
        "scheduled_dequeue_adequacy",
        "ttl_normalized_service_age",
        "reserved_zero",
    )
    assert snapshot.ue_features.values[0, 12:].tolist() == [
        0.6931471824645996,
        1.0,
        0.25,
        0.0,
    ]

from training.stageb_hgraph import dpa_reproduction_mechanisms as mechanisms
from types import SimpleNamespace

import numpy as np
import pytest


def test_snapshot_schedule_keeps_uniform_and_pressure_samples_separate():
    assert mechanisms.uniform_snapshot_tti(1001)
    assert mechanisms.uniform_snapshot_tti(3971)
    assert not mechanisms.uniform_snapshot_tti(1000)
    assert not mechanisms.uniform_snapshot_tti(1002)
    assert sum(mechanisms.uniform_snapshot_tti(i) for i in range(1,4001))==100


def test_paired_aggregation_does_not_treat_scenarios_as_training_replicates():
    result=mechanisms.aggregate_differences([[1,3],[2,4],[-2,0]])
    assert result['training_seed_means']==[2,3,-1]
    assert result['mean']==4/3
    assert result['n_training_replicates']==3


def test_common_prefix_comparison_uses_effective_legal_logits():
    from copy import deepcopy
    from dataclasses import replace
    import numpy as np
    from training.stageb_hgraph.tests.test_independent_bc_dataset import _actor, _ctbs_inputs, _graph
    full=_actor().eval()
    removed=deepcopy(full)
    removed.cfg=replace(removed.cfg,ablation='no_reuse_attention')
    record={'graph':_graph(),'ctbs_inputs':_ctbs_inputs(),
            'teacher_action_ue':np.arange(6),'teacher_action_prg':np.array([0,2,4,-1,3,5])}
    result=mechanisms.compare_snapshot(record,{'801/full':full,'801/no_reuse_attention':removed})
    assert result['prefix']=='PFQ'
    assert not result['causal_service_claim']
    assert result['comparisons']['801/full']['effective_logit_max_abs_difference']==0
    for row in result['comparisons'].values():
        assert 0<=row['conditional_argmax_changed_fraction']<=1
        assert np.asarray(row['conditional_ue_matrix']).shape==(3,2)


def test_endpoint_flags_efficiency_delay_tradeoff_and_reward_low_expiry_fallback():
    baseline={'goodput_mbps':100,'expiry_drop_rate':0,'outcome_delay_mean_ms':2,
              'outcome_delay_p95_ms':4,'ue_expiry_aware_rate_p10_mbps':1,
              'end_backlog_bytes':100,'offered_bytes':10000,'successful_bytes_per_active_prg':100}
    full={**baseline,'successful_bytes_per_active_prg':104,'outcome_delay_p95_ms':6}
    result=mechanisms.endpoint_assessment([[full]*5]*3,[[baseline]*5]*3,'no_reuse_attention')
    assert result['engineering_effect_met']
    assert not result['protection_passed_all_pairs']
    assert result['classification']=='performance_tradeoff'
    result=mechanisms.endpoint_assessment([[baseline]*5]*3,[[baseline]*5]*3,'no_reward')
    assert result['metric']=='ue_expiry_aware_rate_p10_mbps'
    assert result['classification']=='no_stable_engineering_gain'


def test_reference_capture_uses_formal_actor_graph_schema(tmp_path):
    capture=mechanisms.ReferenceCapture(tmp_path/'snapshots',object())
    assert capture.edges.cfg.candidate_mode=='all_demand_ues'
    assert capture.edges.cfg.include_pfq_score_anchor is False
    assert capture.edges.cfg.include_candidate_risk_features is False
    assert capture.edges.ue_prg_edge_feature_names==(
        'quality_db','quality_gap_db','urgency','service_debt','prg_risk')


def test_reference_capture_fails_closed_on_ctbs_post_action_mismatch(tmp_path,monkeypatch):
    capture=mechanisms.ReferenceCapture(tmp_path/'snapshots',object())
    action_ue=np.arange(6); action_prg=np.array([0,2,4,-1,3,5])
    capture._pending=(1,SimpleNamespace(),SimpleNamespace(snapshot=SimpleNamespace(
        static_meta=SimpleNamespace(n_cell=3,n_prg=2))),action_ue.copy())
    mismatch=SimpleNamespace(all_exact=False,reference_label_deletion_equal=True,
                             mcs_mismatch_count=1,layer_mismatch_count=0,tbs_mismatch_count=0)
    seen={}
    def compare(*args,**kwargs):seen.update(kwargs); return mismatch
    monkeypatch.setattr(mechanisms,'compare_ctbs_reference_labels',compare)
    with pytest.raises(RuntimeError,match='causal PHY/TBS parity failed'):
        capture.after_step(action_ue,action_prg,{
            'actual_ue_mcs':[1]*6,'actual_ue_layer':[1]*6,
            'actual_ue_tbs_capacity_bytes':[100]*6})
    assert seen['chosen_ue_per_prg']==[0,-1,2,3,4,5]

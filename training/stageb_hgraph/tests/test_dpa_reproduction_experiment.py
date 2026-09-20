import pytest
import json
from pathlib import Path
from training.stageb_hgraph import dpa_reproduction_experiment as experiment


def metrics(**overrides):
    result=dict(goodput_mbps=1500., expiry_drop_rate=0., outcome_delay_mean_ms=1.,
                outcome_delay_p95_ms=2., ue_expiry_aware_rate_p10_mbps=10.,
                end_backlog_bytes=1000., offered_bytes=1000000.,
                max_backlogged_zero_dequeue_streak_tti=4,
                max_backlogged_schedulable_zero_dequeue_streak_tti=4)
    result.update(overrides); return result


def test_gate_rejects_goodput_improvement_with_weak_ue_collapse():
    old = metrics()
    new = dict(old, goodput_mbps=1510., ue_expiry_aware_rate_p10_mbps=4.)
    assert not experiment.acceptance(new, old, old)['passed']
    assert experiment.acceptance(old, old, old)['passed']


def test_selection_uses_feasible_candidates_before_shorter_delay():
    old=metrics()
    bad=dict(old, outcome_delay_p95_ms=1., ue_expiry_aware_rate_p10_mbps=1.)
    rows=[{'iteration':4, 'metrics':bad}, {'iteration':8, 'metrics':old}]
    assert experiment.choose_checkpoint(rows, old, old)['iteration']==8


def test_calibration_chooses_first_in_range_not_most_favorable():
    rows=[{'ttl_tti':200,'expiry_rates':[0.,0.,0.]},
          {'ttl_tti':100,'expiry_rates':[0.,.002,.003]},
          {'ttl_tti':50,'expiry_rates':[.01,.02,.03]}]
    assert experiment.choose_pressure(rows)['ttl_tti']==100


def test_full_failure_prevents_expansion():
    assert experiment.may_expand([{'passed':True},{'passed':False}]) is False
    assert experiment.may_expand([{'passed':True},{'passed':True}]) is True
    assert experiment.may_expand([]) is False


def test_schedulable_starvation_carries_warmup_streak_and_resets_unschedulable():
    rows=[]
    for tti in range(998,1004):
        rows.append({'tti':tti,'ue_id':0,'pre_queue_bytes':1,'serving_cell':0,
                     'demand_valid_prg_count':1,'scheduled_dequeue_bytes':0})
    rows.append({'tti':1004,'ue_id':0,'pre_queue_bytes':1,'serving_cell':-1,
                 'demand_valid_prg_count':1,'scheduled_dequeue_bytes':0})
    rows.append({'tti':1005,'ue_id':0,'pre_queue_bytes':1,'serving_cell':0,
                 'demand_valid_prg_count':1,'scheduled_dequeue_bytes':0})
    assert experiment.max_schedulable_zero_dequeue_streak(rows,1001,1005)==6


def test_common_dataset_collection_command_is_collect_only_and_stage_bound(tmp_path):
    command=experiment.bc_collection_command(Path('/sim'),tmp_path,tmp_path/'data',
                                             steps=4000,feature_mode='legacy')
    options=dict(zip(command[3::2],command[4::2]))
    assert options['--independent-bc-collect-only']=='1'
    assert options['--independent-bc-dataset-out']==str(tmp_path/'data')
    assert options['--independent-headroom-accounting']=='ctbs-v1'
    assert options['--seed']=='801'
    assert options['--topology-seed']=='41'
    assert 'CUMAC_DPA_FEATURE_MODE=legacy' in command


def test_after_p0_wait_does_not_overwrite_and_rejects_failed_status(tmp_path):
    path=tmp_path/'status.json'; original={'status':'p0_references_complete','sentinel':7}
    path.write_text(json.dumps(original))
    assert experiment.wait_for_p0(tmp_path,poll_seconds=0)==original
    assert json.loads(path.read_text())==original
    path.write_text(json.dumps({'status':'failed','error':'x'}))
    with pytest.raises(RuntimeError,match='P0 failed'):
        experiment.wait_for_p0(tmp_path,poll_seconds=0)


def test_eval_identity_binds_complete_execution_contract():
    identity=experiment.eval_identity(seed=211,ttl_tti=400,feature_mode='dimensionless',
        policy='pfq',checkpoint_sha256=None,binary_sha256='b',source_fingerprint='s',
        protocol_sha256='p',mechanism_snapshots=True)
    assert identity['window']=={'total_tti':4000,'warmup_tti':1000,'measured_tti':3000}
    assert identity['decode']=='argmax' and identity['ctbs_guard']=='off'
    assert identity['traffic_profile']=='homogeneous'
    assert identity['mechanism_snapshots'] is True


def test_acceptance_uses_schedulable_not_legacy_starvation():
    old=metrics(); candidate=metrics(max_backlogged_zero_dequeue_streak_tti=999,
                                    max_backlogged_schedulable_zero_dequeue_streak_tti=3)
    assert experiment.acceptance(candidate,old,old)['passed']


def test_real_reward_validation_reports_exact_removed_term_errors():
    from training.stageb_hgraph.tests.test_independent_reward import _graph_with_debt, _info
    from training.stageb_hgraph.tests.graph_fixtures import make_candidate_observation
    before=_graph_with_debt(0.0)
    record={'graph':before,'next_obs':make_candidate_observation(n_prg=2,ue_feat_dim=16),
            'next_info':_info(goodput_bytes=105_000.,scheduled_bytes=108_000.,expired_bytes=3_000.)}
    result=experiment.validate_reward_records([record])
    assert result['passed']
    assert result['max_removed_terms_error']<=1e-6
    assert result['max_no_reward_formula_error']<=1e-6
    assert result['max_shared_raw_component_error']<=1e-6
    assert result['all_finite']


def test_feature_audit_uses_response_clock_for_dequeue_events():
    from types import SimpleNamespace
    import numpy as np
    import torch
    from training.stageb_hgraph.feature_snapshot import FeatureSnapshotBuilder
    from training.stageb_hgraph.tests.graph_fixtures import make_candidate_observation
    snapshot=FeatureSnapshotBuilder().build(make_candidate_observation(n_prg=2,ue_feat_dim=16))
    dimensionless_names=snapshot.ue_features.names
    legacy_names=(*dimensionless_names[:12], 'critical_ttl_backlog_bytes',
                  'near_ttl_backlog_bytes','age_weighted_backlog_bytes','service_gap_tti')
    legacy=[]; dimensionless=[]; recent=0.
    # First traffic update precedes RESET; first dequeue is recorded at clock 1.
    expected_ages=[1,0,*range(1,15)]
    for tti,age in enumerate(expected_ages):
        values=torch.zeros((1,16)); values[0,0]=100.; values[0,9]=200.
        def record(features,names):
            return {'graph':SimpleNamespace(snapshot=SimpleNamespace(ue_features=SimpleNamespace(values=features,names=names))),
                    'metadata':{'tti':tti},'next_info':{'tti':tti+1,'actual_ue_served_bytes':[100 if tti==0 else 0]}}
        legacy.append(record(values.clone(),legacy_names))
        values[0,12]=np.log1p(100/(recent*400+1))
        values[0,13]=min(recent*400/100,1)
        values[0,14]=age/400.
        dimensionless.append(record(values,dimensionless_names))
        recent=.95*recent+(.05*100 if tti==0 else 0)
    assert experiment.validate_feature_records(legacy,dimensionless)['passed']
    dimensionless[1]['graph'].snapshot.ue_features.values[0,14]=1/400.
    with pytest.raises(ValueError,match='service age'):
        experiment.validate_feature_records(legacy,dimensionless)


def test_cross_condition_audit_checks_all_75_policy_scene_pairs(tmp_path,monkeypatch):
    names=['pfq','rrq','old_dpa',*[f'{seed}/{arm}' for seed in (801,802,803)
                                  for arm in experiment.ARMS]]
    monkeypatch.setattr(experiment,'offered_identity',lambda folder:('same-stream',[400]))
    monkeypatch.setattr(experiment,'sha',lambda path:'same-topology')
    result=experiment.validate_cross_condition_inputs(tmp_path,names)
    assert result['passed'] and result['pair_count']==75


def test_e_reward_contract_is_explicit_and_keeps_fixed_reward_denominator():
    contract=experiment.EXPECTED_E_REWARD_CONTRACT
    assert contract['independent_offered_bytes_per_tti']==108000.
    assert contract['independent_expired_weight']==2.
    assert contract['deadline_tail_eta']==.25
    assert contract['deadline_tail_window_steps']==100
    assert contract['deadline_tail_rate_epsilon_mbps']==1e-6


def test_actual_ppo_actor_checkpoint_uses_args_schema():
    import torch
    checkpoint=torch.load(experiment.OLD,map_location='cpu',weights_only=False)
    assert isinstance(checkpoint['args'],dict)
    assert checkpoint['args']['reward_mode']=='deadline_tail'
    assert checkpoint['reward_cfg']['gamma']==.99

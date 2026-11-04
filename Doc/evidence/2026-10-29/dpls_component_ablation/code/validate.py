"""Verify paired exogenous inputs and native service before analyzing effects."""
import json
import numpy as np
import pandas as pd
import torch
from common import ARMS, OUT, REPO, REFERENCE, SEEDS, baseline, load_file, path_for, sha, write


def validate(arm, seed):
    path=path_for(arm,seed)
    reference=path_for('pfq',seed)
    execution=json.loads((path/'execution.json').read_text())
    check=json.loads((path/'validation.json').read_text())
    assert check['passed']
    for key in ['native_actions_exact','native_mcs_layer_tbs_dequeue_exact','fresh_observation_snapshot_queue_exact','fresh_ctbs_tti_binding_exact','current_association_mask_exact']:
        assert check[key]==28400,(arm,seed,key)
    assert check['common_pfq_warmup']==4000 and check['frozen_sources_unchanged']
    assert execution['horizon']==28400 and execution['measured_ttis']==20000
    assert execution['settling_ttis']==4000 and execution['followup_ttis']==400
    physical=json.loads((path/'physical_validation.json').read_text())
    assert physical['passed'] and len(physical['samples'])==6
    assert all(x['direct_native_sinr_max_abs_db']<.005 and x['direct_native_bler_max_abs']<.0005 for x in physical['samples'])
    saved=np.load(path/'prefix.npz'); old=np.load(reference/'prefix.npz')
    for key in ['channel_fingerprints','precoder_fingerprints','fresh_input_tti']:
        np.testing.assert_array_equal(saved[key],old[key],err_msg=f'{arm}/{seed}/{key}')
    for key in ['input_hashes','ctbs_hashes']:
        np.testing.assert_array_equal(saved[key][:4001],old[key][:4001])
    for key in ['actions','ue','prg','goodput']:
        np.testing.assert_array_equal(saved[key][:4000],old[key][:4000])
    for name in ['traffic_offered_packets.csv','ue_topology.csv']:
        assert sha(path/name)==sha(reference/name)
    actions=saved['actions'].reshape(28400,3,17)
    slots,prg=saved['ue'],saved['prg']
    decoded=np.where(prg>=0,np.take_along_axis(slots,np.maximum(prg,0),axis=1),-1)
    np.testing.assert_array_equal(actions,decoded.reshape(28400,17,3).transpose(0,2,1))
    fields=list(baseline.FIELDS)
    raw=np.fromfile(path/'metrics.bin',dtype='<f8').reshape(28400,36,len(fields))
    raw_old=np.fromfile(reference/'metrics.bin',dtype='<f8').reshape(raw.shape)
    np.testing.assert_array_equal(raw[:4000],raw_old[:4000])
    for ue in range(36):
        np.testing.assert_array_equal((actions==ue).sum(axis=(1,2)),raw[:,ue,fields.index('actual_prg_count')])
    np.testing.assert_array_equal(saved['goodput'],raw[:,:,fields.index('successful_bytes')].sum(1))
    states=torch.load(path/'states.pt',map_location='cpu',weights_only=False)
    old_states=torch.load(reference/'states.pt',map_location='cpu',weights_only=False)
    independent=load_file('ablation_independent_inputs',REPO/'deliverables/boundary08_corrected_physics_20261020/verify_inputs.py')
    samples=[]
    for state,other in zip(states,old_states):
        assert state['tti']==other['tti']
        for key in ['h','w']:
            np.testing.assert_array_equal(state['snapshot'][key],other['snapshot'][key])
        samples.append(independent.check_state(state))
    assert len(samples)==6
    if not ARMS[arm]['reused']:
        cv=json.loads((path/'component_validation.json').read_text())
        assert cv['passed'] and cv['arm']==arm
        decisions=pd.read_csv(path/'decisions.csv')
        audit=pd.read_csv(path/'alignment_audit.csv')
        assert len(decisions)==len(audit)==24400
        np.testing.assert_array_equal(decisions.draft_dequeue,audit.draft_dequeue)
        np.testing.assert_array_equal(decisions.keep_dequeue,(90*audit.draft_dequeue.astype(np.int64)+99)//100)
        assert decisions.chosen_dequeue.ge(decisions.keep_dequeue).all()
        assert decisions.converged.all() and not decisions.cap_reached.any()
        if ARMS[arm]['search']=='none':
            assert decisions.candidate_evaluations.eq(0).all()
            np.testing.assert_array_equal(actions[4000:],np.load(path/'drafts.npy'))
        else:
            history=pd.read_csv(path/'rounds.csv')
            np.testing.assert_array_equal(history.groupby('tti').candidates.sum(),decisions.candidate_evaluations)
        weights=np.load(path/'weights.npy')
        if not ARMS[arm]['deficit']:
            np.testing.assert_array_equal(weights,np.ones((24400,36)))
    result=dict(passed=True,arm=arm,seed=seed,full_exogenous_ttis=28400,
        common_pfq_ttis=4000,common_first_decision_input=True,native_decode_counts_and_goodput_exact=True,
        independent_input_samples=samples,complete_horizon=True,
        files={name:sha(path/name) for name in ['metrics.bin','prefix.npz','launch.json','execution.json','validation.json','physical_validation.json']})
    write(OUT/'checks'/f'{arm}_seed{seed}.json',result)
    print('PAIRED_RUN_VALIDATED',arm,seed,flush=True)
    return result


if __name__=='__main__':
    (OUT/'checks').mkdir(exist_ok=True)
    for arm in ARMS:
        for seed in SEEDS:validate(arm,seed)

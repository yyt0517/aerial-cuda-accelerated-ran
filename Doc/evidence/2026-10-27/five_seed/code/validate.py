"""Pairing, causal-floor and continuous-input checks for the promoted policy."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .run import OUT,REFERENCE,REPO,baseline


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_analysis():
    return load_file('rho090_reference_analysis',REPO/'deliverables/boundary08_steady_state_20261021/analyze.py')


def check_inputs(seed,partial=False):
    check=load_file('rho090_independent_inputs',REPO/'deliverables/boundary08_corrected_physics_20261020/verify_inputs.py')
    path=OUT/f'seed{seed}_current_20000'
    if not (path/'states.pt').exists():
        return dict(passed=False,pending=True,seed=seed)
    try:
        states=torch.load(path/'states.pt',map_location='cpu',weights_only=False)
    except (RuntimeError,EOFError,OSError):
        if partial:return dict(passed=False,pending=True,seed=seed)
        raise
    expected=[4000,4001,8000,8001,18000,28000]
    assert [x['tti'] for x in states]==expected[:len(states)]
    if not partial:assert len(states)==6
    independent=[check.check_state(x) for x in states]
    pfq=torch.load(REFERENCE/f'seed{seed}_pfq0_20000/states.pt',map_location='cpu',weights_only=False)
    by_tti={x['tti']:x for x in states}
    ref={x['tti']:x for x in pfq}
    for tti,state in by_tti.items():
        for field in ['h','w']:
            np.testing.assert_array_equal(state['snapshot'][field],ref[tti]['snapshot'][field])
    recurrence=None
    if 4001 in by_tti:
        legacy=baseline.r.historical.OUT.parent/'boundary08_corrected_physics_20261020_v1'/f'seed{seed}_pfq0_2000/states.pt'
        # Use the original periodic-clock PFQ snapshot only as an audit innovation sample.
        reset=next(x for x in torch.load(legacy,map_location='cpu',weights_only=False) if x['tti']==4001)
        before=np.asarray(by_tti[4000]['snapshot']['h'],dtype=np.complex128)
        after=np.asarray(by_tti[4001]['snapshot']['h'],dtype=np.complex128)
        innovation=np.asarray(reset['snapshot']['h'],dtype=np.complex128)
        predicted=float(np.float32(.9938))*before+float(np.float32(.0786))*innovation
        scale=float(np.abs(predicted).max())
        np.testing.assert_allclose(after,predicted,rtol=1e-6,atol=scale*1e-7)
        assert not np.allclose(after,innovation,rtol=1e-3,atol=scale*1e-5)
        recurrence=dict(passed=True,max_error_over_global_scale=float(np.abs(after-predicted).max()/scale))
    result=dict(passed=len(states)==6,available_checks_passed=True,seed=seed,observed_ttis=list(by_tti),
                samples=independent,continuous_first_rollover=recurrence,baseline_H_W_samples_exact=True)
    baseline.r.write(path/('partial_input_validation.json' if partial else 'input_validation.json'),result)
    print('INPUT_CHECKS',seed,list(by_tti),flush=True)
    return result


def validate(seed,smoke=False):
    directory=OUT/(f'smoke_seed{seed}_current_136' if smoke else f'seed{seed}_current_20000')
    assert json.loads((directory/'validation.json').read_text())['passed']
    causal=json.loads((directory/'causal_floor_validation.json').read_text())
    assert causal['passed'] and causal['floor_ratio_num']==90
    assert causal['coordinator_forward_calls']==causal['external_floor_reads']==0
    decisions=pd.read_csv(directory/'decisions.csv');audit=pd.read_csv(directory/'alignment_audit.csv')
    assert len(decisions)==(8 if smoke else 24400)
    np.testing.assert_array_equal(decisions.draft_dequeue,audit.draft_dequeue_independent)
    np.testing.assert_array_equal(decisions.keep_dequeue,(90*decisions.draft_dequeue.astype(np.int64)+99)//100)
    np.testing.assert_array_equal(decisions.keep_dequeue,audit.causal_floor_bytes)
    assert (decisions.chosen_dequeue>=decisions.keep_dequeue).all()
    assert (decisions.keep_dequeue<=decisions.draft_dequeue).all()
    assert decisions.converged.all() and not decisions.cap_reached.any() and decisions.rounds_evaluated.lt(256).all()
    history=pd.read_csv(directory/'rounds.csv')
    np.testing.assert_array_equal(history.groupby('tti').candidates.sum(),decisions.candidate_evaluations)
    result=dict(passed=True,seed=seed,causal_floor_decisions=len(decisions),full_search_converged=True)
    if smoke:
        ref=baseline.r.historical.OUT.parent/'boundary08_corrected_physics_20261020_v1'/f'seed{seed}_pfq0_2000'
        saved=np.load(directory/'prefix.npz');old=np.load(ref/'prefix.npz')
        for key in ['channel_fingerprints','precoder_fingerprints','fresh_input_tti']:
            np.testing.assert_array_equal(saved[key],old[key][:136])
        for key in ['input_hashes','ctbs_hashes']:
            np.testing.assert_array_equal(saved[key][:129],old[key][:129])
        for key in ['actions','ue','prg','goodput']:
            np.testing.assert_array_equal(saved[key][:128],old[key][:128])
        if seed==41:
            selected=OUT.parent/'boundary08_causal_floor_ratio_seed41_20261027_v1/rho90/smoke_seed41_current_136'
            assert (directory/'metrics.bin').read_bytes()==(selected/'metrics.bin').read_bytes()
            result['entire_selected_rho090_smoke_native_metrics_bit_exact']=True
    else:
        helper=load_analysis()
        result['pairing']=helper.verify_saved_pairing(dict(pfq0=REFERENCE/f'seed{seed}_pfq0_20000',current=directory))
        result['independent_inputs']=check_inputs(seed)
        protocol=json.loads((OUT/'protocol.json').read_text())
        for name,digest in protocol['reference_files_sha256'][str(seed)].items():
            assert baseline.r.sha(REFERENCE/f'seed{seed}_pfq0_20000'/name)==digest
    baseline.r.write(directory/'paired_validation.json',result)
    print('LATEST_RHO090_NATIVE_VALID',seed,'smoke' if smoke else '28400TTI',flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=range(41,46),required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--partial-inputs',action='store_true')
    args=parser.parse_args()
    check_inputs(args.seed,True) if args.partial_inputs else validate(args.seed,args.smoke)

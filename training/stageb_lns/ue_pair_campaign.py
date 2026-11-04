"""Frozen per-UE within-state pairwise terminal risk and confidence fallback."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
from unittest.mock import patch
import numpy as np
from .directed_reorder import DirectedConfig,library as directed_library
from .terminal_cached import solve_cached
from .ue_pair_solver import solve_ue_pair
from .ue_pair_risk import UEPairModel,RISK_CAP,CONFIDENCE
from .ue_pair_train import prepare,train
from .terminal_risk import RiskModel
from .terminal_service_campaign import assert_parity
from .weak_service_campaign import RecordedMemo
from .joint_reorder import library as delta_library
from .resource_condition_campaign import teachers,compressed_condition,REGISTRY
from .resource_conditions import satisfies,library as physical_library
from .shortlist_trajectory import service_metrics
from .shortlist_campaign import selection_key
from .bounded_probes import csv_write,seed_mean
from .physics import ROOT,ReferencePhy,EXPECTED,DEQUEUE
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .ranked_dpls import solve_ranked
from .ranked_dpls_campaign import read_rows,identity
from .experiment import write_json,sha

CONFIG=DirectedConfig(joint_candidates=False)
NAMES=['full','cached_maxj','aggregate_guard','ue_heuristic_guard','ue_pair','ue_pair_guard']
PREVIOUS=ROOT/'deliverables/terminal_service_20261103_v1'


def run(row,arm,teacher,model=None):
    RecordedMemo.latest=None
    start=time.perf_counter();p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs']);group=None
    if arm=='full':a,v,r=solve_ranked(p,draft)
    else:
        condition=compressed_condition(p,*teacher,'joint212')
        if arm=='cached_maxj':a,v,r,group=solve_cached(p,draft,condition,CONFIG,mode='maxj',offline_diagnostic=True)
        else:a,v,r,group=solve_ue_pair(p,draft,condition,CONFIG,mode='maxj' if arm=='new_maxj' else arm,model=model,offline_diagnostic=True)
    r['score']=float(v[:,EXPECTED]@p.weights);r['e2e_ms']=(time.perf_counter()-start)*1000;r.pop('history',None)
    r.update(state_id=identity(row),seed=row['seed'],source=row['family'],tti=row['tti'],arm=arm)
    memo=RecordedMemo.latest;observed=visited=None
    if memo is not None:
        observed=np.stack(list(memo.rows.values()));visited=np.frombuffer(b''.join(memo.rows),np.int64).reshape(-1,p.C,p.P).copy()
        r['query_sha256']=hashlib.sha256(b''.join(memo.events)).hexdigest()
        r['archive_sha256']=hashlib.sha256(b''.join(memo.rows)+observed.tobytes()).hexdigest()
    return a,v,r,group,observed,visited


def engineering(rows,out,models):
    ta=np.load(PREVIOUS/'training_teacher_actions.npz');tv=np.load(PREVIOUS/'training_teacher_values.npz')
    work=[];checks=[]
    for row in (rows[0],rows[-1]):
        sid=identity(row);ref=(ta[sid],tv[sid]);p=problem_from_raw_row(row);results={}
        for arm in ['cached_maxj','new_maxj',*NAMES[2:]]:
            result=run(row,arm,ref,models);results[arm]=result;a,v,r,*_=result;work.append(r)
            ind=ReferencePhy(p).evaluate(a)[0];np.testing.assert_allclose(v,ind,rtol=2e-9,atol=2e-6)
            checks.append(dict(state_id=sid,arm=arm,max_error=float(abs(v-ind).max())))
        for arm in ['new_maxj',*NAMES[2:]]:assert_parity(results['cached_maxj'],results[arm],final=arm=='new_maxj')
    csv_write(out/'engineering_measurements.csv',work);write_json(out/'engineering_checks.json',dict(independent_phy=checks,
        copied_search_maxj_equivalent=True,all_terminal_queries_equivalent=True,teacher_reused_training_only=True))


def benchmark(rows,refs,out,phase,model):
    raw=[];saved={k:{} for k in ('actions','values','service_groups','observed_values','observed_actions')}
    for n,row in enumerate(rows):
        state_results={}
        for repeat in range(3):
            shift=(n+repeat)%len(NAMES)
            for arm in NAMES[shift:]+NAMES[:shift]:
                result=run(row,arm,refs[identity(row)],model);a,v,r,g,vv,aa=result;r['repeat']=repeat;raw.append(r);key=f'{identity(row)}|{arm}'
                for name,value in zip(saved,(a,v,g,vv,aa)):
                    if value is None:continue
                    if key in saved[name]:np.testing.assert_array_equal(saved[name][key],value)
                    else:saved[name][key]=value
                if repeat==0:state_results[arm]=result
        for arm in NAMES[2:]:assert_parity(state_results['cached_maxj'],state_results[arm],final=False)
        print('UE PAIR BENCH',phase,identity(row),flush=True)
    state=[];checks=[];parity=[];previous=np.load(PREVIOUS/f'{phase}_actions.npz');pv=np.load(PREVIOUS/f'{phase}_values.npz')
    for row in rows:
        p=problem_from_raw_row(row);sid=identity(row);ref=saved['values'][f'{sid}|full'];jref=float(ref[:,EXPECTED]@p.weights)
        full_ms=float(np.median([r['e2e_ms'] for r in raw if r['state_id']==sid and r['arm']=='full']))
        np.testing.assert_array_equal(saved['actions'][f'{sid}|full'],refs[sid][0])
        np.testing.assert_array_equal(saved['actions'][f'{sid}|cached_maxj'],previous[f'{sid}|cached_maxj'])
        np.testing.assert_array_equal(saved['values'][f'{sid}|cached_maxj'],pv[f'{sid}|cached_maxj'])
        parity.append(dict(state_id=sid,previous_baseline_equal=True,cached_maxj_previous_final_equal=True,all_terminal_arms_query_archive_equal=True))
        for arm in NAMES:
            key=f'{sid}|{arm}';v=saved['values'][key];a=saved['actions'][key];ind=ReferencePhy(p).evaluate(a)[0]
            np.testing.assert_allclose(v,ind,rtol=2e-9,atol=2e-6)
            rr=[r for r in raw if r['state_id']==sid and r['arm']==arm]
            r={k:v for k,v in rr[0].items() if k!='repeat'};r['e2e_ms']=float(np.median([v['e2e_ms'] for v in rr]))
            r.update(ratio=r['score']/jref,reference_ms=full_ms,execution_time_ratio=r['e2e_ms']/full_ms,**service_metrics(v,ref))
            assert v[:,DEQUEUE].sum()>=r['floor']
            if arm!='full':
                assert r['candidate_evaluations']==212 and r['feature_roots']<=56 and r['condition_feasible'] and not r['fallback_to_raw']
                assert satisfies(p,a,compressed_condition(p,*refs[sid],'joint212'))[0] and r['mmse_misses']==2448
                assert r['selected_j_loss_fraction']<=RISK_CAP+1.e-12
                assert len({z['query_sha256'] for z in rr})==len({z['archive_sha256'] for z in rr})==1
            state.append(r);checks.append(dict(state_id=sid,arm=arm,max_error=float(abs(v-ind).max())))
    csv_write(out/f'{phase}_measurements.csv',raw);csv_write(out/f'{phase}_per_state.csv',state)
    csv_write(out/f'{phase}_summary.csv',summarize(state));csv_write(out/f'{phase}_equivalence.csv',parity)
    for name,data in saved.items():np.savez_compressed(out/f'{phase}_{name}.npz',**data)
    write_json(out/f'{phase}_checks.json',dict(independent_phy=checks,repeats_and_equivalence=True,budget_floor_conditions=True))
    return state


def freeze(out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(date='2026-11-03',version=1,methods=NAMES,search_config=asdict(CONFIG),
        objective='canonical native BLER causal deficit-weightedJ,own rawPFQ90%dequeue floor',
        conditions='paid offline teacher counts/exact min-layer/activity only;no teacher UE positions/J/service/effective membership at runtime',
        unchanged_search='same existing-pool directed2 slots,quota-exp frontier,root-local pool/proxy cache;212 complete-action scores/56 roots/16 seeds/8batch',
        terminal='only already paid condition/floor-feasible archive;2%ownbestJwindow unchanged;full99.5%J gate unchanged',
        features='20 causal context features perUE;7 monotone ownservice-loss bases plus learned reference loss;no UE ID embedding;shared20-24tanh-10 model',
        loss='within-state risk=10*relu(.995-J/fullJ)+relu(.98-P5ratio)+relu(.98-fixedweakratio);pair gap>.005;ties<=.0001;up to1024each/state,prioritize maxJ pairs',
        auxiliary='0.2*(referencepositive BCE+referenceE/D MSE),teacher labels only in training',
        training='reuse previous paid own-visited41-43 archives;deduplicate by immutable inputs/archive/reference hash;3leave-seed-out fits forpositive calibration scale;3all-training initialization fits',
        fixed_training='240epochs each Adam lr.01 decay.001;pairlogit scale20;no epoch/model/threshold search;1440gradientsteps',
        calibration='monotone zero-intercept pair probability from held-seed predictions,including ties;heuristic scale from same training-only pairs',
        confidence='propose highest mean pairscore within2%;accept only if every ensemble member calibrated P(win vs maxJ)>=.9 and positive delta,else return exact maxJ;this is a relative model confidence,not a hard feasibility guarantee',
        heuristic='sameperUE losses with uniform coefficients,ownpositive frequency andq90reference;single score with its training-fitted probabilityscale andsame.9gate',
        controls='aggregate_guard uses frozen previous aggregate model but falls back to maxJ when no>=.9 candidate;ue_pair disables guard solely for attribution and cannot be validation-selected',
        training_seeds=[41,42,43],validation_seeds=[44,45],evaluation_seeds=[46,47,48],
        train_validation_ttis=[4001,8001,18000],evaluation_ttis=[3001,4001,5000],historical_development=True,repeats=3,
        gates='eachJ>=.995full,nonewlyunallocatedUE,fullpositiveP5/fixedweak20>=.98full,seed-equalcompleteexecution<=.8full',
        timing='input/PFQ,freshPHY/MMSE,conditioncompression,unchangedsearch,terminalfeatures/3modelinference/calibration/fallback,cleanup;common memoevent copies included',
        exclusions='teachers,training,initiallibrary/TBS/modelIO,archiveIO,digests and independent audits',
        selection='validation only;cachedmaxJ,aggregate_guard,ue_heuristic_guard,ue_pair_guard eligible;freeze beforedevelopment;no development tuning',
        final_seeds_not_read=[311,312,313,314,315],native_trajectories=0,ctbs=False,ppo=False,
        registry_sha256=sha(REGISTRY),physics_source_sha256=sha(ROOT/'training/stageb_lns/physics_kernel.cpp'),
        previous_aggregate_model_sha256=sha(PREVIOUS/'risk_model.npz'),platform=platform.platform(),
        threads=dict(OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1))
    write_json(out/'protocol.json',protocol);manifest={}
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.cpp','.so','.rst'):
            target=out/'source'/p.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target);manifest[p.name]=sha(p)
    write_json(out/'source_manifest.json',manifest);return protocol


def main(output):
    out=Path(output).resolve();protocol=freeze(out);start=time.perf_counter()
    rows,files=read_rows((41,42,43));rows=[r for r in rows if r['tti'] in protocol['train_validation_ttis']]
    oldfiles=json.loads((PREVIOUS/'training_input_manifest.json').read_text())['files'];assert files==oldfiles
    write_json(out/'training_input_manifest.json',dict(files=files,states=[identity(r) for r in rows]))
    problem_from_raw_row(rows[0]);physical_library();delta_library();directed_library()
    data=prepare(rows,PREVIOUS,out);train(data,out)
    shutil.copyfile(PREVIOUS/'risk_model.npz',out/'aggregate_model.npz')
    models=dict(pair=UEPairModel.load(out/'ue_pair_model.npz'),aggregate=RiskModel.load(out/'aggregate_model.npz'))
    modelhash={n:sha(out/n) for n in ('ue_pair_model.npz','aggregate_model.npz')}
    write_json(out/'models_frozen.json',dict(hashes=modelhash,validation_not_read=True,development_not_read=True))
    with patch('training.stageb_lns.terminal_cached.ExactMemo',RecordedMemo),patch('training.stageb_lns.ue_pair_solver.ExactMemo',RecordedMemo):
        engineering(rows,out,models)
        for phase,seeds,evaluation in [('validation',(44,45),False),('evaluation',(46,47,48),True)]:
            rows,files=read_rows(seeds,evaluation=evaluation);ttis=protocol['evaluation_ttis'] if evaluation else protocol['train_validation_ttis']
            rows=[r for r in rows if r['tti'] in ttis];write_json(out/f'{phase}_input_manifest.json',dict(files=files,states=[identity(r) for r in rows]))
            refs=teachers(rows,out,phase);state=benchmark(rows,refs,out,phase,models)
            if not evaluation:
                selection=choose(state);write_json(out/'selection.json',selection);print('UE PAIR SELECTION FROZEN',selection,flush=True)
    assert sha(REGISTRY)==protocol['registry_sha256'] and sha(ROOT/'training/stageb_lns/physics_kernel.cpp')==protocol['physics_source_sha256']
    assert all(sha(out/n)==h for n,h in modelhash.items())
    assert all(sha(Path(__file__).with_name(n))==h for n,h in json.loads((out/'source_manifest.json').read_text()).items())
    write_json(out/'completion.json',dict(completed=True,formal_decisions=378,formal_independent_finals=126,new_offline_teachers=21,
        engineering_decoders=12,engineering_independent_finals=12,reused_training_states=18,unique_training_states=len(data),
        new_training_teacher_scores=0,new_training_decoder_scores=0,training_updates=1440,source_model_registry_unchanged=True,
        final_seeds_read=False,native_trajectories=0,elapsed_s=time.perf_counter()-start))
    print('UE PAIR COMPLETE',out,flush=True)


def summarize(state):
    full=[r for r in state if r['arm']=='full'];out=[]
    for arm in NAMES:
        rr=[r for r in state if r['arm']==arm]
        q=all(r['ratio']>=.995 for r in rr)
        c=all(r['newly_unserved_ues']==0 and r['positive_p5_ratio']>=.98 and r['positive_weak_ratio']>=.98 for r in rr)
        cost=seed_mean(rr,'e2e_ms')/seed_mean(full,'e2e_ms')
        out.append(dict(arm=arm,mean_j=seed_mean(rr,'score'),mean_ms=seed_mean(rr,'e2e_ms'),execution_time_ratio=cost,
            mean_relative_j_pct=100*(seed_mean(rr,'ratio')-1),worst_relative_j_pct=100*(min(r['ratio'] for r in rr)-1),
            mean_newly_unserved=seed_mean(rr,'newly_unserved_ues'),mean_positive_p5=seed_mean(rr,'positive_p5_ratio'),
            worst_positive_p5=min(r['positive_p5_ratio'] for r in rr),mean_positive_weak=seed_mean(rr,'positive_weak_ratio'),
            worst_positive_weak=min(r['positive_weak_ratio'] for r in rr),mean_actual_phy=seed_mean(rr,'candidate_evaluations'),
            mean_feature_roots=seed_mean(rr,'feature_roots') if arm!='full' else None,
            quality_failures=sum(r['ratio']<.995 for r in rr),coverage_failures=sum(not(r['newly_unserved_ues']==0 and r['positive_p5_ratio']>=.98 and r['positive_weak_ratio']>=.98) for r in rr),
            quality_pass=q,coverage_pass=c,execution_cost_pass=cost<=.8,conditional_target_pass=q and c and cost<=.8,
            online_ai_success=False))
    return out


def choose(state):
    choices=[]
    for arm in ['cached_maxj','aggregate_guard','ue_heuristic_guard','ue_pair_guard']:
        rr=[r for r in state if r['arm']==arm];key=selection_key(rr)
        q=all(r['ratio']>=.995 for r in rr)
        c=all(r['newly_unserved_ues']==0 and r['positive_p5_ratio']>=.98 and r['positive_weak_ratio']>=.98 for r in rr)
        cost=seed_mean(rr,'e2e_ms')/seed_mean(rr,'reference_ms')
        choices.append(((int(q and c),*key[1:]),arm,cost,seed_mean(rr,'e2e_ms')))
    best=max(choices,key=lambda v:(v[0],-v[3]));feasible=[v for v in choices if v[2]<=.8]
    within=max(feasible,key=lambda v:(v[0],-v[3])) if feasible else min(choices,key=lambda v:v[3])
    return dict(quality_candidate=best[1],execution_cost_candidate=within[1],execution_cost_feasible=bool(feasible),
        validation_only=True,development_not_read=True,historical_exposure=True,teacher_preparation_excluded=True,
        rule='same prior rule: all-state quality/service then mean quality shortfall, service shortfall, mean ratio; execution cost filter, lower time on ties',online_ai_claim=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)

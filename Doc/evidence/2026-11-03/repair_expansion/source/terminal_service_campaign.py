"""Frozen six-arm cached search and candidate-risk calibration experiment."""
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
from .directed_reorder import DirectedConfig,solve_directed,library as directed_library
from .terminal_cached import solve_cached
from .terminal_risk import candidate_features,RiskModel,RISK_CAP,PROBABILITY_THRESHOLD
from .terminal_risk_train import train
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
NAMES=['full','existing2','cached_maxj','continuous','attainable','calibrated']
PREVIOUS=ROOT/'deliverables/directed_reorder_20261102_v1'
PARITY=('candidate_evaluations','ue_evaluations','mmse_requests','mmse_misses','feature_roots','iterations',
    'proxy_candidates','seed_gamma_array_reads','service_group_size','frontier_peak','best_observed_j','query_sha256','archive_sha256')


def run(row,arm,teacher,model=None):
    RecordedMemo.latest=None
    start=time.perf_counter();p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs']);group=None
    if arm=='full':a,v,r=solve_ranked(p,draft)
    else:
        condition=compressed_condition(p,*teacher,'joint212')
        if arm=='existing2':a,v,r,group=solve_directed(p,draft,condition,CONFIG,offline_diagnostic=True)
        else:a,v,r,group=solve_cached(p,draft,condition,CONFIG,mode='maxj' if arm=='cached_maxj' else arm,model=model,offline_diagnostic=True)
    r['score']=float(v[:,EXPECTED]@p.weights);r['e2e_ms']=(time.perf_counter()-start)*1000;r.pop('history',None)
    r.update(state_id=identity(row),seed=row['seed'],source=row['family'],tti=row['tti'],arm=arm)
    memo=RecordedMemo.latest;observed=visited=None
    if memo is not None:
        observed=np.stack(list(memo.rows.values()));visited=np.frombuffer(b''.join(memo.rows),np.int64).reshape(-1,p.C,p.P).copy()
        r['query_sha256']=hashlib.sha256(b''.join(memo.events)).hexdigest()
        r['archive_sha256']=hashlib.sha256(b''.join(memo.rows)+observed.tobytes()).hexdigest()
    return a,v,r,group,observed,visited


def assert_parity(old,new,final=True):
    for i in ((0,1,3,4,5) if final else (3,4,5)):np.testing.assert_array_equal(old[i],new[i])
    for field in PARITY:assert old[2][field]==new[2][field],(field,old[2][field],new[2][field])


def collect_training(rows,refs,out):
    datasets=[];work=[];checks=[];observed={};visited={};parity=[]
    for n,row in enumerate(rows):
        p=problem_from_raw_row(row);sid=identity(row);ref=refs[sid][1];c=compressed_condition(p,*refs[sid],'joint212')
        result=run(row,'cached_maxj',refs[sid]);a,v,r,g,vv,aa=result;work.append(r);observed[sid]=vv;visited[sid]=aa
        ind=ReferencePhy(p).evaluate(a)[0];np.testing.assert_allclose(v,ind,rtol=2e-9,atol=2e-6)
        checks.append(dict(state_id=sid,arm='cached_maxj',max_error=float(abs(v-ind).max())))
        if n in (0,len(rows)-1):
            old=run(row,'existing2',refs[sid]);assert_parity(old,result);parity.append(dict(state_id=sid,full_query_action_value_parity=True))
            work.append(old[2]);ind=ReferencePhy(p).evaluate(old[0])[0];np.testing.assert_allclose(old[1],ind,rtol=2e-9,atol=2e-6)
            checks.append(dict(state_id=sid,arm='existing2',max_error=float(abs(old[1]-ind).max())))
        keep=satisfies(p,aa,c)&(vv[:,:,DEQUEUE].sum(1)>=r['floor']);candidate=vv[keep]
        x,j,h,names=candidate_features(p,c,candidate);labels=[service_metrics(v,ref) for v in candidate]
        datasets.append(dict(x=x,j=j,j_ratio=j/float(ref[:,EXPECTED]@p.weights),
            coverage=np.array([v['newly_unserved_ues']==0 and v['positive_p5_ratio']>=.98 and v['positive_weak_ratio']>=.98 for v in labels]),
            p5=np.array([v['positive_p5_ratio'] for v in labels]),weak=np.array([v['positive_weak_ratio'] for v in labels]),
            seed=np.full(len(x),row['seed']),state=np.full(len(x),sid),memo_index=np.flatnonzero(keep)))
        print('TERMINAL TRAIN VISIT',sid,flush=True)
    data={k:np.concatenate([v[k] for v in datasets]) for k in datasets[0]}
    np.savez_compressed(out/'training_candidates.npz',**data)
    np.savez_compressed(out/'training_observed_values.npz',**observed);np.savez_compressed(out/'training_observed_actions.npz',**visited)
    csv_write(out/'training_decoder_work.csv',work);write_json(out/'feature_names.json',names)
    write_json(out/'training_checks.json',dict(independent_phy=checks,engineering_parity=parity,paid_own_search_only=True,
        extra_candidate_label_scores=0,reference_labels_post_search=True,raw_initial_included_only_if_condition_satisfies=True))
    return train(data,out)


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
        for arm in NAMES[2:]:assert_parity(state_results['existing2'],state_results[arm],final=arm=='cached_maxj')
        print('TERMINAL BENCH',phase,identity(row),flush=True)
    state=[];checks=[];parity=[];previous=np.load(PREVIOUS/f'{phase}_actions.npz');pv=np.load(PREVIOUS/f'{phase}_values.npz')
    for row in rows:
        p=problem_from_raw_row(row);sid=identity(row);ref=saved['values'][f'{sid}|full'];jref=float(ref[:,EXPECTED]@p.weights)
        full_ms=float(np.median([r['e2e_ms'] for r in raw if r['state_id']==sid and r['arm']=='full']))
        np.testing.assert_array_equal(saved['actions'][f'{sid}|full'],refs[sid][0])
        np.testing.assert_array_equal(saved['actions'][f'{sid}|existing2'],previous[f'{sid}|existing2'])
        np.testing.assert_array_equal(saved['values'][f'{sid}|existing2'],pv[f'{sid}|existing2'])
        parity.append(dict(state_id=sid,previous_baseline_equal=True,cached_maxj_final_equal=True,all_terminal_arms_query_archive_equal=True))
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
        objective='canonical native BLER causal deficit-weighted J; own raw unpruned PFQ90% dequeue floor',
        conditions='fresh paid offline teacher counts/exact min-layer/activity only; no teacher UE positions,J,service or effective mask at runtime',
        cache='decision-local deterministic wide pool keyed full root/wide/seed; lazy proxy keyed full root/candidate; no skipping periodic global RNG seed calls; ordered queries and physical values must match existing2',
        terminal='same212 paid search for all conditional arms; maximumJ vs quota-exp vs attainable-envelope heuristic vs tiny calibrated neural coverage risk',
        risk_cap=RISK_CAP,risk_cap_scope='fraction of own maximumJ only; chosen from previously exposed diagnostic headroom, not a new blind hypothesis; global99.5%-fullJ gate unchanged',
        continuous='lexicographic quota-exp thenmean reliability thenJ within2% ownbest',
        attainable='own feasible-paid archive perUEmax E>1 group;min(candidateP5/envelopeP5,fixedweak20sum/envelopesum);choose maxJ withscore>=.98 else bestscore thenJ within2%',
        calibrated='same causal feature summaries;42input->32tanh->16tanh->1 service-coverage logit;positive monotone Platt from leave-seed-out41-43;maxJ probability>=.9 else bestprob thenJ within2%',
        training='18 own-visited states seeds41-43; all condition-and-floor-feasible paid candidates; offline full coverage labels;240 epochs fixed Adam lr.01 decay.001;3 leave-seed-out fits plus1all;no epoch/model tuning',
        splits='group by seed, never random candidate split; all train normalization fitted only to fold training',
        training_seeds=[41,42,43],validation_seeds=[44,45],evaluation_seeds=[46,47,48],
        train_validation_ttis=[4001,8001,18000],evaluation_ttis=[3001,4001,5000],historical_development=True,
        final_seeds_not_read=[311,312,313,314,315],repeats=3,
        gates='each J>=.995full,no newlyunallocatedUE,fullpositive P5/fixedweak20service>=.98full,seed-equal mean complete cost<=.8full',
        selection='validation choose quality/service rule and separate cost-feasible route; cachedmaxJ eligible;freeze beforedevelopment;no tuning on44-48',
        timing='numeric input/PFQ,freshPHY/MMSE,condition compression,all search/caches/terminalfeatures/network/contextcleanup; same memo event copies',
        exclusions='teacherprep,compilation/immutablefirstTBS/library/model file IO,training,data/archiveIO,digests/independent audits',
        limits='teacher-conditioned historical states;not online AI deployment timing or blind generalization;no additional recall scores/native trajectories/default promotion',
        threads=dict(OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1),
        registry_sha256=sha(REGISTRY),physics_source_sha256=sha(ROOT/'training/stageb_lns/physics_kernel.cpp'),platform=platform.platform())
    write_json(out/'protocol.json',protocol);manifest={}
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.cpp','.so','.rst'):
            target=out/'source'/p.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target);manifest[p.name]=sha(p)
    write_json(out/'source_manifest.json',manifest);return protocol


def main(output):
    out=Path(output).resolve();protocol=freeze(out);start=time.perf_counter()
    rows,files=read_rows((41,42,43));rows=[r for r in rows if r['tti'] in protocol['train_validation_ttis']]
    write_json(out/'training_input_manifest.json',dict(files=files,states=[identity(r) for r in rows]))
    problem_from_raw_row(rows[0]);physical_library();delta_library();directed_library()
    with patch('training.stageb_lns.directed_reorder.ExactMemo',RecordedMemo),patch('training.stageb_lns.terminal_cached.ExactMemo',RecordedMemo):
        refs=teachers(rows,out,'training');model=collect_training(rows,refs,out)
        model=RiskModel.load(out/'risk_model.npz');model_sha=sha(out/'risk_model.npz')
        write_json(out/'model_frozen.json',dict(sha256=model_sha,validation_not_read=True,development_not_read=True,training_only=True))
        for phase,seeds,evaluation in [('validation',(44,45),False),('evaluation',(46,47,48),True)]:
            rows,files=read_rows(seeds,evaluation=evaluation);ttis=protocol['evaluation_ttis'] if evaluation else protocol['train_validation_ttis']
            rows=[r for r in rows if r['tti'] in ttis];write_json(out/f'{phase}_input_manifest.json',dict(files=files,states=[identity(r) for r in rows]))
            refs=teachers(rows,out,phase);state=benchmark(rows,refs,out,phase,model)
            if not evaluation:
                selection=choose(state);write_json(out/'selection.json',selection);print('TERMINAL SELECTION FROZEN',selection,flush=True)
    assert sha(REGISTRY)==protocol['registry_sha256'] and sha(ROOT/'training/stageb_lns/physics_kernel.cpp')==protocol['physics_source_sha256']
    assert sha(out/'risk_model.npz')==model_sha
    assert all(sha(Path(__file__).with_name(name))==digest for name,digest in json.loads((out/'source_manifest.json').read_text()).items())
    write_json(out/'completion.json',dict(completed=True,formal_decisions=378,formal_independent_finals=126,offline_teachers=39,
        training_decoders=18,extra_engineering_decoders=2,training_decoder_independent_finals=20,training_updates=960,
        additional_candidate_label_scores=0,registry_and_physics_unchanged=True,executed_sources_unchanged=True,
        final_seeds_read=False,native_trajectories=0,elapsed_s=time.perf_counter()-start))
    print('TERMINAL SERVICE COMPLETE',out,flush=True)


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
    for arm in ['cached_maxj','continuous','attainable','calibrated']:
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
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);main(parser.parse_args().output)

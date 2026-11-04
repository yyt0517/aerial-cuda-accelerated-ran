"""Frozen-model PHY-budget curves; no training or changes to the solver."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np

from .shortlist_trajectory import Config,solve,service_metrics
from .shortlist_campaign import selection_key
from .bounded_probes import Predictor,probe_library,csv_write,seed_mean
from .physics import ROOT,ReferencePhy,EXPECTED,DEQUEUE
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .ranked_dpls import solve_ranked
from .ranked_dpls_campaign import read_rows,identity
from .experiment import write_json,sha


PREVIOUS=ROOT/'deliverables/shortlist_trajectory_20261102_v1'
REGISTRY=ROOT/'training/stageb_hgraph/current_algorithm.json'
BUDGETS=(106,160,212)
FAMILIES=('heuristic','initial','trained')


def arm_name(family,k,budget):return f'{family}_k{k}_b{budget}'


def parse_arm(arm):
    if arm=='full':return 'full',0,0
    family,k,budget=arm.split('_')
    return family,int(k[1:]),int(budget[1:])


def arms():
    return ['full']+[arm_name(f,k,b) for b in BUDGETS for f in FAMILIES for k in (2,4)]


def run(row,arm,models):
    begin=time.perf_counter();p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs'])
    family,k,budget=parse_arm(arm)
    if family=='full':a,v,r=solve_ranked(p,draft)
    else:a,v,r=solve(p,draft,Config(k,True,56,budget),models.get(family))
    r['score']=float(v[:,EXPECTED]@p.weights);r['e2e_ms']=(time.perf_counter()-begin)*1000
    r.pop('trace',None);r.pop('history',None)
    r.update(state_id=identity(row),seed=row['seed'],family=row['family'],tti=row['tti'],arm=arm,
        model=family,top_k=k,budget=budget)
    return a,v,r


def summarize(records):
    full=[r for r in records if r['arm']=='full'];result=[]
    for arm in dict.fromkeys(r['arm'] for r in records):
        rr=[r for r in records if r['arm']==arm];family,k,budget=parse_arm(arm)
        quality=all(r['ratio']>=.995 for r in rr)
        cover=all(r['newly_unserved_ues']==0 and r['positive_p5_ratio']>=.98 and r['positive_weak_ratio']>=.98 for r in rr)
        cost=seed_mean(rr,'e2e_ms')/seed_mean(full,'e2e_ms')
        result.append(dict(arm=arm,model=family,top_k=k,budget=budget,
            mean_j=seed_mean(rr,'score'),mean_ms=seed_mean(rr,'e2e_ms'),time_ratio=cost,
            state_p95_ms=float(np.quantile([r['e2e_ms'] for r in rr],.95)),
            mean_relative_j_pct=100*(seed_mean(rr,'ratio')-1),worst_relative_j_pct=100*(min(r['ratio'] for r in rr)-1),
            mean_actual_phy=seed_mean(rr,'candidate_evaluations'),
            mean_feature_roots=seed_mean(rr,'feature_roots') if family!='full' else None,
            mean_accepted_edits=seed_mean(rr,'accepted_edits') if family!='full' else None,
            physical_budget_stops=sum(r.get('stop_reason')=='physical_budget' for r in rr),
            feature_budget_stops=sum(r.get('stop_reason')=='feature_budget' for r in rr),
            domain_exhausted_stops=sum(r.get('stop_reason')=='finite_domain_exhausted' for r in rr),
            mean_newly_unserved=seed_mean(rr,'newly_unserved_ues'),
            mean_positive_p5=seed_mean(rr,'positive_p5_ratio'),worst_positive_p5=min(r['positive_p5_ratio'] for r in rr),
            mean_positive_weak=seed_mean(rr,'positive_weak_ratio'),
            mean_ue_drops=seed_mean(rr,'ue_service_drops_gt2pct'),
            quality_pass=quality,coverage_pass=cover,cost_pass=cost<=.8,joint_pass=quality and cover and cost<=.8))
    return result


def benchmark(rows,models,out,phase):
    names=arms();records=[];actions={};values={}
    for n,row in enumerate(rows):
        for repeat in range(3):
            shift=(n+repeat)%len(names)
            for arm in names[shift:]+names[:shift]:
                a,v,r=run(row,arm,models);r['repeat']=repeat;records.append(r)
                key=f'{identity(row)}|{arm}'
                if key in actions:np.testing.assert_array_equal(actions[key],a)
                else:actions[key]=a;values[key]=v
        print('BUDGET_BENCH',phase,identity(row),flush=True)
    state=[];audits=[]
    for row in rows:
        p=problem_from_raw_row(row);sid=identity(row);reference=values[f'{sid}|full']
        full=[r for r in records if r['state_id']==sid and r['arm']=='full']
        full_ms=float(np.median([r['e2e_ms'] for r in full]))
        for arm in names:
            key=f'{sid}|{arm}';a=actions[key];v=values[key]
            independent=ReferencePhy(p).evaluate(a)[0]
            np.testing.assert_allclose(v,independent,rtol=2.e-9,atol=2.e-6)
            rr=[r for r in records if r['state_id']==sid and r['arm']==arm]
            assert independent[:,DEQUEUE].sum()>=rr[0]['floor']
            r={k:v for k,v in rr[0].items() if k!='repeat'}
            r['e2e_ms']=float(np.median([v['e2e_ms'] for v in rr]))
            r.update(ratio=r['score']/float(reference[:,EXPECTED]@p.weights),
                reference_ms=full_ms,time_ratio=r['e2e_ms']/full_ms,**service_metrics(v,reference))
            if arm!='full':
                assert r['candidate_evaluations']<=r['budget'] and r['feature_roots']<=56
                assert r['repeated_exact_candidates']==0 and r['full_search_calls']==0
            state.append(r);audits.append(dict(state_id=sid,arm=arm,max_error=float(abs(v-independent).max())))
    csv_write(out/f'{phase}_measurements.csv',records);csv_write(out/f'{phase}_per_state.csv',state)
    np.savez_compressed(out/f'{phase}_actions.npz',**actions);np.savez_compressed(out/f'{phase}_values.npz',**values)
    summary=summarize(state);csv_write(out/f'{phase}_summary.csv',summary)
    write_json(out/f'{phase}_checks.json',dict(repeats_action_identical=True,common_floor=True,
        caps_valid=True,root_repeated_phy_zero=True,independent_phy=audits))
    return state,summary


def choose(records):
    choices=[]
    for arm in dict.fromkeys(r['arm'] for r in records if r['arm']!='full'):
        rr=[r for r in records if r['arm']==arm]
        cost=seed_mean(rr,'e2e_ms')/seed_mean(rr,'reference_ms')
        prior_key=selection_key(rr)
        quality=all(r['ratio']>=.995 for r in rr)
        cover=all(r['newly_unserved_ues']==0 and r['positive_p5_ratio']>=.98 and r['positive_weak_ratio']>=.98 for r in rr)
        key=(int(quality and cover),*prior_key[1:])
        choices.append((key,arm,cost,seed_mean(rr,'e2e_ms')))
    # Separate quality ceiling from a usable cost-feasible research point.
    quality=max(choices,key=lambda v:(v[0],-v[3],-parse_arm(v[1])[2]))
    feasible=[v for v in choices if v[2]<=.8]
    within=max(feasible,key=lambda v:(v[0],-v[3],-parse_arm(v[1])[2])) if feasible else min(choices,key=lambda v:v[3])
    return dict(quality_best=dict(arm=quality[1],time_ratio=quality[2]),
        within_cost_best=dict(arm=within[1],time_ratio=within[2],cost_feasible=bool(feasible)),
        selection='validation only; shared prior key, complete-mean cost filter, lower cost/budget on exact ties',
        development_not_read_in_this_run=True,historical_exposure_retained=True)


def budget_deltas(records):
    index={(r['state_id'],r['arm']):r for r in records};result=[]
    for sid in dict.fromkeys(r['state_id'] for r in records):
        for family in FAMILIES:
            for k in (2,4):
                for old,new in ((106,160),(160,212)):
                    a=index[sid,arm_name(family,k,old)];b=index[sid,arm_name(family,k,new)]
                    result.append(dict(state_id=sid,seed=a['seed'],model=family,top_k=k,from_budget=old,to_budget=new,
                        j_delta=b['score']-a['score'],relative_j_delta_pp=100*(b['ratio']-a['ratio']),
                        ms_delta=b['e2e_ms']-a['e2e_ms'],actual_phy_delta=b['candidate_evaluations']-a['candidate_evaluations'],
                        feature_roots_delta=b['feature_roots']-a['feature_roots'],new_stop_reason=b['stop_reason'],
                        newly_unserved_delta=b['newly_unserved_ues']-a['newly_unserved_ues'],
                        positive_p5_delta=b['positive_p5_ratio']-a['positive_p5_ratio']))
    return result


def freeze(out):
    out.mkdir(parents=True,exist_ok=False)
    previous_protocol=json.loads((PREVIOUS/'protocol.json').read_text())
    previous_selection=json.loads((PREVIOUS/'selection.json').read_text())
    model_paths=dict(initial=previous_protocol['initial']['checkpoint'],trained=previous_selection['checkpoint'])
    protocol=dict(date='2026-11-02',version=1,training=False,ppo=False,ctbs=False,headroom_masks=False,
        models={k:dict(path=v,sha256=sha(v)) for k,v in model_paths.items()},
        control='same feature/rank cache, failed-index deduplication and canonical score layout for heuristic and both AI checkpoints',
        budgets=list(BUDGETS),top_k=[2,4],max_feature_roots=56,methods=arms(),
        numerical_and_acceptance='unchanged shortlist solver; >1B edit acceptance, full DPLS original1/16B thresholds',
        floor='ceil(90% own unpruned raw PFQ dequeue), causal deficit-weighted J',
        partial_batch='last shortlist truncated to remaining scores; different budgets may change later paths; do not assume monotone J',
        validation_seeds=[44,45],validation_sources=['raw_global_deficit','pfq'],validation_ttis=[4001,8001,18000],
        evaluation_seeds=[46,47,48],evaluation_sources=['pfq'],evaluation_ttis=[3001,4001,5000],
        historical_development=True,final_seeds_not_read=[311,312,313,314,315],
        selection='freeze quality-best and best within80%-full cost on validation, before reading development results in this run',
        quality_gate='all states J>=99.5% full',
        coverage_gate='no newly unallocated COUNT>0 UE vs full; P5 and bottom20% service among full expected>1B UEs each>=98%',
        cost_gate='seed-equal medians complete mean time<=80% full',
        repeats=3,timing='numeric input/PFQ decode through fresh PHY/cache/features/network/validation/scoring/cleanup',
        excludes='archive/model/library IO and first immutable TBS table setup',
        registry_sha256=sha(REGISTRY),solver_sha256=sha(Path(__file__).with_name('shortlist_trajectory.py')),
        prior_protocol_sha256=sha(PREVIOUS/'protocol.json'),prior_selection_sha256=sha(PREVIOUS/'selection.json'),
        platform=platform.platform())
    write_json(out/'protocol.json',protocol)
    sources={}
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.cpp','.so','.rst'):
            target=out/'source'/p.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target);sources[p.name]=sha(p)
    write_json(out/'source_manifest.json',sources)
    return protocol


def main(output):
    import torch
    torch.set_num_threads(1);out=Path(output).resolve();protocol=freeze(out);started=time.perf_counter()
    models={k:Predictor(v['path']) for k,v in protocol['models'].items()}
    validation,files=read_rows((44,45));validation=[r for r in validation if r['tti'] in protocol['validation_ttis']]
    write_json(out/'validation_input_manifest.json',dict(files=files,states=[identity(r) for r in validation]))
    problem_from_raw_row(validation[0]);probe_library()
    state,summary=benchmark(validation,models,out,'validation')
    selected=choose(state);write_json(out/'selection.json',selected)
    csv_write(out/'validation_budget_deltas.csv',budget_deltas(state))
    print('BUDGET SELECTION FROZEN',selected,flush=True)
    evaluation,files=read_rows((46,47,48),evaluation=True)
    evaluation=[r for r in evaluation if r['tti'] in protocol['evaluation_ttis']]
    write_json(out/'evaluation_input_manifest.json',dict(files=files,states=[identity(r) for r in evaluation]))
    state,summary=benchmark(evaluation,models,out,'evaluation')
    csv_write(out/'evaluation_budget_deltas.csv',budget_deltas(state))
    assert all(sha(v['path'])==v['sha256'] for v in protocol['models'].values())
    assert sha(Path(__file__).with_name('shortlist_trajectory.py'))==protocol['solver_sha256']
    assert sha(REGISTRY)==protocol['registry_sha256']
    write_json(out/'completion.json',dict(completed=True,validation_states=len(validation),evaluation_states=len(evaluation),
        validation_decisions=len(validation)*len(arms())*3,evaluation_decisions=len(evaluation)*len(arms())*3,
        frozen_models_and_solver_unchanged=True,registry_unchanged=True,native_trajectories=0,
        final_seeds_read=False,training_updates=0,elapsed_s=time.perf_counter()-started))
    print('BUDGET CURVE COMPLETE',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)

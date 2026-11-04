"""Frozen small whole-TB/continuation experiment; no training or default change."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import shutil
import time
import numpy as np

from .tb_macro import MacroConfig,solve_macro
from .shortlist_trajectory import Config,solve,service_metrics
from .shortlist_campaign import selection_key
from .bounded_probes import Predictor,probe_library,csv_write,seed_mean
from .physics import ROOT,ReferencePhy,EXPECTED,DEQUEUE
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .ranked_dpls import solve_ranked
from .ranked_dpls_campaign import read_rows,identity
from .experiment import write_json,sha

REGISTRY=ROOT/'training/stageb_hgraph/current_algorithm.json'
MODEL=ROOT/'deliverables/shortlist_trajectory_20261102_v1/own_r1_e10.npz'
CONFIGS={
    'single_b212':MacroConfig('single',False,212),
    'joint_b212':MacroConfig('joint',False,212),
    'single_continue_b212':MacroConfig('single',True,212),
    'joint_continue_b212':MacroConfig('joint',True,212),
    'single_b424':MacroConfig('single',False,424),
    'joint_b424':MacroConfig('joint',False,424),
    'joint_continue_b424':MacroConfig('joint',True,424),
}
NAMES=['full','shortlist_trained212',*CONFIGS]


def run(row,arm,model):
    start=time.perf_counter();p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs']);snapshot=None
    if arm=='full':a,v,r=solve_ranked(p,draft)
    elif arm=='shortlist_trained212':a,v,r=solve(p,draft,Config(4,True,56,212),model)
    else:a,v,r,snapshot=solve_macro(p,draft,CONFIGS[arm])
    r['score']=float(v[:,EXPECTED]@p.weights);r['e2e_ms']=(time.perf_counter()-start)*1000
    events=r.pop('continuation_records',[]);r.pop('trace',None);r.pop('history',None)
    r.update(state_id=identity(row),seed=row['seed'],source=row['family'],tti=row['tti'],arm=arm,
        budget=(CONFIGS[arm].max_candidates if arm in CONFIGS else (212 if arm!='full' else 0)))
    return a,v,r,snapshot,events


def summarize(records):
    full=[r for r in records if r['arm']=='full'];out=[]
    for arm in NAMES:
        rr=[r for r in records if r['arm']==arm]
        q=all(r['ratio']>=.995 for r in rr)
        c=all(r['newly_unserved_ues']==0 and r['positive_p5_ratio']>=.98 and r['positive_weak_ratio']>=.98 for r in rr)
        cost=seed_mean(rr,'e2e_ms')/seed_mean(full,'e2e_ms')
        out.append(dict(arm=arm,mean_j=seed_mean(rr,'score'),mean_ms=seed_mean(rr,'e2e_ms'),time_ratio=cost,
            mean_relative_j_pct=100*(seed_mean(rr,'ratio')-1),worst_relative_j_pct=100*(min(r['ratio'] for r in rr)-1),
            mean_actual_phy=seed_mean(rr,'candidate_evaluations'),
            mean_feature_roots=seed_mean(rr,'feature_roots') if arm!='full' else None,
            mean_newly_unserved=seed_mean(rr,'newly_unserved_ues'),mean_positive_p5=seed_mean(rr,'positive_p5_ratio'),
            worst_positive_p5=min(r['positive_p5_ratio'] for r in rr),mean_positive_weak=seed_mean(rr,'positive_weak_ratio'),
            mean_macro_accepts=seed_mean(rr,'macro_accepts') if arm in CONFIGS else None,
            mean_continuation_phy=seed_mean(rr,'continuation_phy') if arm in CONFIGS else None,
            continuation_rescues=sum(r.get('continuation_rescues',0) for r in rr),
            quality_pass=q,coverage_pass=c,cost_pass=cost<=.8,joint_pass=q and c and cost<=.8))
    return out


def choose(records):
    candidates=[]
    for arm in NAMES[1:]:
        rr=[r for r in records if r['arm']==arm];key=selection_key(rr)
        q=all(r['ratio']>=.995 for r in rr)
        c=all(r['newly_unserved_ues']==0 and r['positive_p5_ratio']>=.98 and r['positive_weak_ratio']>=.98 for r in rr)
        cost=seed_mean(rr,'e2e_ms')/seed_mean(rr,'reference_ms')
        candidates.append(((int(q and c),*key[1:]),arm,cost,seed_mean(rr,'e2e_ms')))
    best=max(candidates,key=lambda v:(v[0],-v[3]))
    feasible=[v for v in candidates if v[2]<=.8]
    usable=max(feasible,key=lambda v:(v[0],-v[3])) if feasible else min(candidates,key=lambda v:v[3])
    return dict(quality_best=best[1],within_cost_best=usable[1],cost_feasible=bool(feasible),
        rule='validation only; gates, J shortfall, service shortfall, mean ratio; lower complete time on ties',
        development_not_read_in_this_run=True,historical_exposure=True)


def benchmark(rows,model,out,phase):
    records=[];actions={};values={};events=[];snapshots={}
    for n,row in enumerate(rows):
        for repeat in range(3):
            shift=(n+repeat)%len(NAMES)
            for arm in NAMES[shift:]+NAMES[:shift]:
                a,v,r,s,e=run(row,arm,model);r['repeat']=repeat;records.append(r)
                key=f'{identity(row)}|{arm}'
                if key in actions:
                    np.testing.assert_array_equal(actions[key],a);np.testing.assert_array_equal(values[key],v)
                else:
                    actions[key]=a;values[key]=v
                    if s is not None:snapshots[key]=s
                    events.extend(dict(state_id=identity(row),arm=arm,**x) for x in e)
        print('TB_BENCH',phase,identity(row),flush=True)
    state=[];audits=[]
    for row in rows:
        p=problem_from_raw_row(row);sid=identity(row);ref=values[f'{sid}|full']
        common=snapshots[f'{sid}|single_b212']
        for arm in CONFIGS:np.testing.assert_array_equal(snapshots[f'{sid}|{arm}'],common)
        full_ms=float(np.median([r['e2e_ms'] for r in records if r['state_id']==sid and r['arm']=='full']))
        for arm in NAMES:
            key=f'{sid}|{arm}';v=values[key];independent=ReferencePhy(p).evaluate(actions[key])[0]
            np.testing.assert_allclose(v,independent,rtol=2.e-9,atol=2.e-6)
            rr=[r for r in records if r['state_id']==sid and r['arm']==arm]
            assert independent[:,DEQUEUE].sum()>=rr[0]['floor']
            r={k:v for k,v in rr[0].items() if k!='repeat'}
            r['e2e_ms']=float(np.median([x['e2e_ms'] for x in rr]))
            r.update(ratio=r['score']/float(ref[:,EXPECTED]@p.weights),reference_ms=full_ms,
                time_ratio=r['e2e_ms']/full_ms,**service_metrics(v,ref))
            if arm!='full':assert r['candidate_evaluations']<=r['budget'] and r['feature_roots']<=56
            if arm in CONFIGS:
                assert r['covered_prgs']==p.P and r['second_pass_prgs']==p.P
                assert r['continuation_calls']<=4 and r['full_search_calls']==0
            state.append(r);audits.append(dict(state_id=sid,arm=arm,max_error=float(abs(v-independent).max())))
    csv_write(out/f'{phase}_measurements.csv',records);csv_write(out/f'{phase}_per_state.csv',state)
    csv_write(out/f'{phase}_summary.csv',summarize(state));write_json(out/f'{phase}_continuations.json',events)
    np.savez_compressed(out/f'{phase}_actions.npz',**actions);np.savez_compressed(out/f'{phase}_values.npz',**values)
    write_json(out/f'{phase}_checks.json',dict(repeat_actions_values_identical=True,common_first_sweep=True,
        all_prgs_visited_twice_before_continuation=True,caps_and_floor_valid=True,independent_phy=audits))
    return state


def freeze(out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(date='2026-11-02',training=False,ctbs=False,headroom_masks=False,ppo=False,
        methods=NAMES,configs={k:asdict(v) for k,v in CONFIGS.items()},model=str(MODEL),model_sha256=sha(MODEL),
        objective='canonical native BLER, causal deficit-weighted J, own unpruned PFQ90% dequeue floor',
        direct='first sweep 4 singles/PRG; later 4 singles or 2 singles+2 coordinated macros/PRG; logical full-TB floor filter',
        proxy='fixed current-root reliability plus activity interference proxy; exact all-PRG TB count/min-layer/dequeue; no candidate BLER/MMSE',
        macro='joint UE/activity on same PRG; cross-PRG move/swap/compensate, at most3 changed slots',
        continuation='only after second full sweep; <=4 PRGs, width2, horizon<=2, fixed causal domains; all nodes paid exact PHY',
        speculative='below-floor or low-J intermediate actions may expand, only exact feasible >1B final actions execute',
        missing='truncated or absent continuation values are unknown, not zero; no optimal-value assertion',
        cache='complete action exact memo local to immutable decision; logical descriptors keyed by complete incumbent; no cross-decision reuse',
        controls='single and joint share pools, proxy, memo, descriptors, sweeps and timings; frozen trained top4/212 and full references',
        validation_seeds=[44,45],validation_ttis=[4001,8001,18000],validation_sources=['raw_global_deficit','pfq'],
        evaluation_seeds=[46,47,48],evaluation_ttis=[3001,4001,5000],evaluation_sources=['pfq'],historical_development=True,
        final_seeds_not_read=[311,312,313,314,315],repeats=3,
        quality_gate='every state J>=99.5% full',coverage_gate='no newly unallocated UE vs full; effective-UE P5/bottom20% expected service each>=98% full',
        cost_gate='seed-equal complete median mean<=80% full',
        timing='input/PFQ decode through fresh PHY/cache/pools/proxies/exact nodes/ranking/final scoring and cleanup; no free continuation teacher',
        exclusions='archive/model/library IO and first immutable TBS setup; independent audit after timing',
        registry_sha256=sha(REGISTRY),old_solver_sha256=sha(Path(__file__).with_name('shortlist_trajectory.py')),platform=platform.platform())
    write_json(out/'protocol.json',protocol);sources={}
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.cpp','.so','.rst'):
            target=out/'source'/p.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target);sources[p.name]=sha(p)
    write_json(out/'source_manifest.json',sources)
    return protocol


def main(output):
    import torch
    torch.set_num_threads(1);out=Path(output).resolve();protocol=freeze(out);start=time.perf_counter();model=Predictor(MODEL)
    rows,files=read_rows((44,45));rows=[r for r in rows if r['tti'] in protocol['validation_ttis']]
    write_json(out/'validation_input_manifest.json',dict(files=files,states=[identity(r) for r in rows]))
    problem_from_raw_row(rows[0]);probe_library()
    state=benchmark(rows,model,out,'validation');selected=choose(state);write_json(out/'selection.json',selected)
    print('TB SELECTION FROZEN',selected,flush=True);nval=len(rows)
    rows,files=read_rows((46,47,48),evaluation=True);rows=[r for r in rows if r['tti'] in protocol['evaluation_ttis']]
    write_json(out/'evaluation_input_manifest.json',dict(files=files,states=[identity(r) for r in rows]))
    benchmark(rows,model,out,'evaluation')
    assert sha(MODEL)==protocol['model_sha256'] and sha(REGISTRY)==protocol['registry_sha256']
    assert sha(Path(__file__).with_name('shortlist_trajectory.py'))==protocol['old_solver_sha256']
    write_json(out/'completion.json',dict(completed=True,validation_states=nval,evaluation_states=len(rows),
        decisions=(nval+len(rows))*len(NAMES)*3,independent_final_audits=(nval+len(rows))*len(NAMES),
        training_updates=0,native_trajectories=0,final_seeds_read=False,frozen_model_and_solver_unchanged=True,
        registry_unchanged=True,elapsed_s=time.perf_counter()-start))
    print('TB MACRO COMPLETE',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);main(parser.parse_args().output)

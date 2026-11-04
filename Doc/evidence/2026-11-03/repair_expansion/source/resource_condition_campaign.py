"""Three resource-level conditional ablations, no new model or online oracle."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import shutil
import time
import numpy as np

from .resource_conditions import Conditions,ConditionalConfig,solve_conditional,satisfies,library
from .selective_tb import SelectiveConfig,solve_selective,feature_library
from .shortlist_trajectory import service_metrics
from .shortlist_campaign import selection_key
from .bounded_probes import csv_write,seed_mean,probe_library
from .physics import ROOT,COUNT,LAYER,EXPECTED,DEQUEUE,ReferencePhy
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .ranked_dpls import solve_ranked
from .ranked_dpls_campaign import read_rows,identity
from .experiment import sha,write_json

REGISTRY=ROOT/'training/stageb_hgraph/current_algorithm.json'
NAMES=['full','ordinary212','bank_none212','quota_layer212','activity212','joint212']
CONDITIONAL=('quota_layer212','activity212','joint212')
CONFIG=ConditionalConfig()


def compressed_condition(p,action,values,arm):
    counts=values[:,COUNT].astype(np.int64).copy() if arm in ('quota_layer212','joint212') else None
    layers=np.maximum(values[:,LAYER],0).astype(np.int64).copy() if counts is not None else None
    active=(action>=0).copy() if arm in ('activity212','joint212') else None
    return Conditions(counts,layers,active).validate(p)


def teachers(rows,out,phase):
    result={};work=[];checks=[];actions={};values={};start_all=time.perf_counter()
    for row in rows:
        start=time.perf_counter();p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs']);a,v,r=solve_ranked(p,draft)
        r['teacher_ms']=(time.perf_counter()-start)*1000;r.pop('history',None)
        sid=identity(row);r.update(state_id=sid,seed=row['seed']);work.append(r);result[sid]=(a,v)
        independent=ReferencePhy(p).evaluate(a)[0];np.testing.assert_allclose(v,independent,rtol=2.e-9,atol=2.e-6)
        checks.append(dict(state_id=sid,max_error=float(abs(v-independent).max())));actions[sid]=a;values[sid]=v
        print('RESOURCE_TEACHER',phase,sid,flush=True)
    csv_write(out/f'{phase}_teacher_work.csv',work);np.savez_compressed(out/f'{phase}_teacher_actions.npz',**actions)
    np.savez_compressed(out/f'{phase}_teacher_values.npz',**values)
    write_json(out/f'{phase}_teacher_checks.json',dict(independent_phy=checks,states=len(rows),
        elapsed_s=time.perf_counter()-start_all,physical_scores=sum(r['candidate_evaluations'] for r in work),
        teacher_preparation_outside_solver_timing=True,complete_teacher_position_not_passed_to_solver=True))
    return result


def run(row,arm,teacher,*,solver=solve_conditional):
    start=time.perf_counter();p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs'])
    if arm=='full':a,v,r=solve_ranked(p,draft)
    elif arm=='ordinary212':a,v,r,_=solve_selective(p,draft,SelectiveConfig('selective',False))
    else:
        condition=compressed_condition(p,*teacher,arm)
        a,v,r=solver(p,draft,condition,CONFIG,offline_diagnostic=arm in CONDITIONAL)
    r['score']=float(v[:,EXPECTED]@p.weights);r['e2e_ms']=(time.perf_counter()-start)*1000
    r.pop('history',None);r.update(state_id=identity(row),seed=row['seed'],source=row['family'],tti=row['tti'],arm=arm)
    return a,v,r


def summarize(records):
    full=[r for r in records if r['arm']=='full'];out=[]
    for arm in NAMES:
        rr=[r for r in records if r['arm']==arm]
        q=all(r['ratio']>=.995 for r in rr);c=all(r['newly_unserved_ues']==0 and r['positive_p5_ratio']>=.98 and r['positive_weak_ratio']>=.98 for r in rr)
        time_ratio=seed_mean(rr,'e2e_ms')/seed_mean(full,'e2e_ms')
        valid=all(r.get('condition_feasible',True) for r in rr)
        out.append(dict(arm=arm,mean_j=seed_mean(rr,'score'),mean_ms=seed_mean(rr,'e2e_ms'),execution_time_ratio=time_ratio,
            mean_relative_j_pct=100*(seed_mean(rr,'ratio')-1),worst_relative_j_pct=100*(min(r['ratio'] for r in rr)-1),
            mean_actual_phy=seed_mean(rr,'candidate_evaluations'),mean_mmse_requests=seed_mean(rr,'mmse_requests'),
            mean_mmse_misses=seed_mean(rr,'mmse_misses'),mean_feature_roots=seed_mean(rr,'feature_roots') if arm!='full' else None,
            mean_proxy_candidates=seed_mean(rr,'proxy_candidates') if arm not in ('full','ordinary212') else None,
            mean_newly_unserved=seed_mean(rr,'newly_unserved_ues'),mean_positive_p5=seed_mean(rr,'positive_p5_ratio'),
            worst_positive_p5=min(r['positive_p5_ratio'] for r in rr),mean_positive_weak=seed_mean(rr,'positive_weak_ratio'),
            condition_valid=valid,quality_pass=valid and q,coverage_pass=valid and c,
            execution_cost_pass=time_ratio<=.8,conditional_target_pass=valid and q and c and time_ratio<=.8,
            online_ai_success=False,teacher_cost_excluded=arm in CONDITIONAL))
    return out


def benchmark(rows,teacher,out,phase,*,solver=solve_conditional):
    raw=[];actions={};values={}
    for n,row in enumerate(rows):
        for repeat in range(3):
            shift=(n+repeat)%len(NAMES)
            for arm in NAMES[shift:]+NAMES[:shift]:
                a,v,r=run(row,arm,teacher[identity(row)],solver=solver);r['repeat']=repeat;raw.append(r);key=f'{identity(row)}|{arm}'
                if key in actions:np.testing.assert_array_equal(actions[key],a);np.testing.assert_array_equal(values[key],v)
                else:actions[key]=a;values[key]=v
        print('RESOURCE_BENCH',phase,identity(row),flush=True)
    state=[];checks=[]
    for row in rows:
        p=problem_from_raw_row(row);sid=identity(row);ref=values[f'{sid}|full'];jref=float(ref[:,EXPECTED]@p.weights)
        np.testing.assert_array_equal(actions[f'{sid}|full'],teacher[sid][0])
        full_ms=float(np.median([r['e2e_ms'] for r in raw if r['state_id']==sid and r['arm']=='full']))
        for arm in NAMES:
            key=f'{sid}|{arm}';v=values[key];independent=ReferencePhy(p).evaluate(actions[key])[0]
            np.testing.assert_allclose(v,independent,rtol=2.e-9,atol=2.e-6)
            rr=[r for r in raw if r['state_id']==sid and r['arm']==arm];r={k:v for k,v in rr[0].items() if k!='repeat'}
            r['e2e_ms']=float(np.median([v['e2e_ms'] for v in rr]));r.update(ratio=r['score']/jref,reference_ms=full_ms,
                execution_time_ratio=r['e2e_ms']/full_ms,**service_metrics(v,ref))
            assert v[:,DEQUEUE].sum()>=r['floor']
            if arm!='full':assert r['candidate_evaluations']<=212 and r['feature_roots']<=56
            if arm not in ('full','ordinary212'):
                condition=compressed_condition(p,*teacher[sid],arm)
                assert satisfies(p,actions[key],condition)[0]==r['condition_satisfied']
                if r['condition_feasible']:assert r['condition_satisfied'] and r['best_conditional_j'] is not None
                else:assert r['best_conditional_j'] is None and r['fallback_to_raw']
                assert r['mmse_misses']==r['prepared_cache_misses']==p.U*p.P*2**(p.C-1)
            state.append(r);checks.append(dict(state_id=sid,arm=arm,max_error=float(abs(v-independent).max())))
    csv_write(out/f'{phase}_measurements.csv',raw);csv_write(out/f'{phase}_per_state.csv',state)
    csv_write(out/f'{phase}_summary.csv',summarize(state));np.savez_compressed(out/f'{phase}_actions.npz',**actions)
    np.savez_compressed(out/f'{phase}_values.npz',**values)
    write_json(out/f'{phase}_checks.json',dict(repeats_actions_values_identical=True,raw_floor_and_caps_valid=True,
        exact_conditions_audited=True,teacher_preparation_separate=True,no_teacher_UE_positions_in_solver=True,independent_phy=checks))
    return state


def choose(state):
    choices=[]
    for arm in CONDITIONAL:
        rr=[r for r in state if r['arm']==arm];key=selection_key([dict(r,time_ratio=r['execution_time_ratio']) for r in rr])
        q=all(r['ratio']>=.995 for r in rr)
        c=all(r['newly_unserved_ues']==0 and r['positive_p5_ratio']>=.98 and r['positive_weak_ratio']>=.98 for r in rr)
        valid=all(r['condition_feasible'] for r in rr);cost=seed_mean(rr,'e2e_ms')/seed_mean(rr,'reference_ms')
        choices.append(((int(valid and q and c),*key[1:]),arm,cost,seed_mean(rr,'e2e_ms')))
    best=max(choices,key=lambda item:(item[0],-item[3]));feasible=[v for v in choices if v[2]<=.8]
    within=max(feasible,key=lambda item:(item[0],-item[3])) if feasible else min(choices,key=lambda item:item[3])
    return dict(quality_candidate=best[1],execution_cost_candidate=within[1],execution_cost_feasible=bool(feasible),
        validation_only=True,development_not_read=True,historical_exposure=True,
        selection_rule='conditional validity/quality/service, J shortfall, service shortfall, ratio; execution-only cost filter',
        teacher_preparation_excluded=True,online_ai_claim=False)


def engineering(rows,out,*,solver=solve_conditional):
    reference=teachers(rows,out,'engineering');records=[];checks=[]
    for row in rows:
        p=problem_from_raw_row(row)
        for arm in ('bank_none212',*CONDITIONAL):
            a,v,r=run(row,arm,reference[identity(row)],solver=solver);independent=ReferencePhy(p).evaluate(a)[0]
            np.testing.assert_allclose(v,independent,rtol=2.e-9,atol=2.e-6)
            assert r['candidate_evaluations']<=212 and r['condition_feasible'] and r['condition_satisfied']
            records.append(r);checks.append(dict(state_id=identity(row),arm=arm,max_error=float(abs(v-independent).max())))
    csv_write(out/'engineering_per_state.csv',records)
    write_json(out/'engineering_checks.json',dict(passed=True,independent_phy=checks,training_states_only=True,
        hyperparameter_selection=False,decoder_frozen_before_pilot=True))


def freeze(out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(date='2026-11-02',version=1,training=False,ppo=False,ctbs=False,headroom_masks=False,methods=NAMES,
        config=asdict(CONFIG),conditions={'quota_layer212':['perUE count','exact perUE minimum layer'],
            'activity212':['per-cell/per-PRG activity bits'],'joint212':['both']},
        excluded_teacher_information=['UE-to-PRG positions','teacher expected service','teacher J inside solver','teacher edit path'],
        decoder='linear assignment with one cap==L witness token/UE and remaining cap>=L; blanks allowed only where conditions permit',
        initial_candidates='16 fixed causal gamma/TBS/weight/log-utility or greedy variants; deterministic random seed',
        search='same 4-beam archive, pair swaps/3-cycles, legal3-UE alternatives when quotas free, periodic full redecoding; top8 exact per root',
        proxy='root-anchored smooth reliability from causal precomputed gamma; never candidate native BLER or exact complete J',
        physics_bank='all U*P*2^(C-1) MMSE solves inside solver timing, requests/misses and proxy complete schedules/UE rows charged separately',
        physical_budget='212 exact complete-schedule evaluations including initial raw PFQ; not equal MMSE/proxy work or equal wall time',
        floor='canonical native BLER, causal deficit-weighted J; own unpruned PFQ90% dequeue floor',
        feasibility='strict supplied conditions; no layer relaxation; no feasible exact candidate=>unknown conditional score and explicit raw-PFQ fallback',
        scope='conditional finite-solver headroom only, not global optimum or complete upper bound; no causal learner trained this round',
        engineering='first/last state of training seeds41-43, fixed before validation; 2teacher+8decoder runs, no hyperparameter selection',
        validation_seeds=[44,45],validation_sources=['raw_global_deficit','pfq'],validation_ttis=[4001,8001,18000],
        development_seeds=[46,47,48],development_sources=['pfq'],development_ttis=[3001,4001,5000],historical_development=True,
        final_seeds_not_read=[311,312,313,314,315],repeats=3,
        teacher='fresh separately charged full-DPLS preparation/decision; same teacher compressed into each condition; no positions passed to solve_conditional',
        selection='freeze conditional quality candidate and execution-cost candidate on validation before reading development; cost excludes teacher and is not online AI latency',
        quality_gate='every state J>=99.5% full',coverage_gate='no newly unallocated UE; full-positive UE P5/bottom20%>=98% full',
        execution_cost_gate='seed-equal mean complete per-state median solver time<=80% full, teacher excluded and explicitly nondeployable',
        timing='raw input/PFQ decode, fresh PHY/cache, condition validation, MMSE bank, matching, all proxies/pools, exact scores, final scoring and cleanup',
        exclusions='teacher preparation, compilation/archive/library IO and first immutable TBS setup; condition compression/validation inside, audits separate',
        registry_sha256=sha(REGISTRY),physics_source_sha256=sha(Path(__file__).with_name('physics_kernel.cpp')),platform=platform.platform())
    write_json(out/'protocol.json',protocol);manifest={}
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.cpp','.so','.rst'):
            target=out/'source'/p.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target);manifest[p.name]=sha(p)
    write_json(out/'source_manifest.json',manifest);return protocol


def main(output):
    out=Path(output).resolve();protocol=freeze(out);start=time.perf_counter()
    training,files=read_rows((41,42,43));training=[r for r in training if r['tti'] in protocol['validation_ttis']]
    engineering_rows=[training[0],training[-1]]
    write_json(out/'engineering_input_manifest.json',dict(files=files,states=[identity(r) for r in engineering_rows]))
    problem_from_raw_row(engineering_rows[0]);library();probe_library();feature_library();engineering(engineering_rows,out)
    validation,files=read_rows((44,45));validation=[r for r in validation if r['tti'] in protocol['validation_ttis']]
    write_json(out/'validation_input_manifest.json',dict(files=files,states=[identity(r) for r in validation]))
    reference=teachers(validation,out,'validation');state=benchmark(validation,reference,out,'validation')
    selection=choose(state);write_json(out/'selection.json',selection);print('RESOURCE SELECTION FROZEN',selection,flush=True)
    evaluation,files=read_rows((46,47,48),evaluation=True);evaluation=[r for r in evaluation if r['tti'] in protocol['development_ttis']]
    write_json(out/'evaluation_input_manifest.json',dict(files=files,states=[identity(r) for r in evaluation]))
    reference=teachers(evaluation,out,'evaluation');benchmark(evaluation,reference,out,'evaluation')
    assert sha(REGISTRY)==protocol['registry_sha256'] and sha(Path(__file__).with_name('physics_kernel.cpp'))==protocol['physics_source_sha256']
    assert all(sha(Path(__file__).with_name(name))==digest for name,digest in json.loads((out/'source_manifest.json').read_text()).items())
    write_json(out/'completion.json',dict(completed=True,formal_decisions=(len(validation)+len(evaluation))*len(NAMES)*3,
        formal_independent_finals=(len(validation)+len(evaluation))*len(NAMES),offline_teacher_decisions=23,
        engineering_decoder_decisions=8,validation_states=len(validation),development_states=len(evaluation),
        training_updates=0,registry_and_physics_unchanged=True,executed_sources_unchanged=True,
        no_teacher_UE_positions_in_solver=True,native_trajectories=0,final_seeds_read=False,elapsed_s=time.perf_counter()-start))
    print('RESOURCE CONDITIONS COMPLETE',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);main(parser.parse_args().output)

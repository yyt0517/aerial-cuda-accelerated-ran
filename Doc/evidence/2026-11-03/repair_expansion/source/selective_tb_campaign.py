"""Small incremental-feature, guarded selective-ranking distillation experiment."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import shutil
import time
import numpy as np

from .selective_tb import Incremental,ServiceGuard,SelectiveConfig,Ranker,feature_library,solve_selective
from .selective_tb_training import collect,fit
from .tb_macro import Descriptors,candidate_pools,MacroConfig,solve_macro
from .shortlist_trajectory import Config,solve,service_metrics
from .shortlist_campaign import selection_key
from .bounded_probes import Predictor,ProbePhy,probe_library,csv_write,seed_mean
from .physics import ROOT,ReferencePhy,EXPECTED,DEQUEUE,COUNT,LAYER
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft,raw_pfq_floor
from .ranked_dpls import solve_ranked
from .ranked_dpls_campaign import read_rows,identity
from .experiment import write_json,sha

REGISTRY=ROOT/'training/stageb_hgraph/current_algorithm.json'
OLD_MODEL=ROOT/'deliverables/shortlist_trajectory_20261102_v1/own_r1_e10.npz'
CONFIGS={
    'increment_fixed_unprotected':SelectiveConfig('fixed',False),
    'increment_selective_unprotected':SelectiveConfig('selective',False),
    'increment_single_guard':SelectiveConfig('single',True),
    'increment_fixed_guard':SelectiveConfig('fixed',True),
    'increment_selective_guard':SelectiveConfig('selective',True),
    'trained_guard':SelectiveConfig('selective',True),
    'trained_gain_only_guard':SelectiveConfig('selective',True,True),
}
NAMES=['full','legacy_ai212','old_joint212',*CONFIGS]


def run(row,arm,old_model,model):
    start=time.perf_counter();p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs']);snapshot=None
    if arm=='full':a,v,r=solve_ranked(p,draft)
    elif arm=='legacy_ai212':a,v,r=solve(p,draft,Config(4,True,56,212),old_model)
    elif arm=='old_joint212':a,v,r,snapshot=solve_macro(p,draft,MacroConfig('joint',False,212))
    else:a,v,r,snapshot=solve_selective(p,draft,CONFIGS[arm],model if arm.startswith('trained') else None)
    r['score']=float(v[:,EXPECTED]@p.weights);r['e2e_ms']=(time.perf_counter()-start)*1000
    for key in ('trace','history','continuation_records'):r.pop(key,None)
    r.update(state_id=identity(row),seed=row['seed'],source=row['family'],tti=row['tti'],arm=arm)
    return a,v,r,snapshot


def summarize(records):
    full=[r for r in records if r['arm']=='full'];result=[]
    for arm in NAMES:
        rr=[r for r in records if r['arm']==arm]
        q=all(r['ratio']>=.995 for r in rr)
        c=all(r['newly_unserved_ues']==0 and r['positive_p5_ratio']>=.98 and r['positive_weak_ratio']>=.98 for r in rr)
        cost=seed_mean(rr,'e2e_ms')/seed_mean(full,'e2e_ms')
        result.append(dict(arm=arm,mean_j=seed_mean(rr,'score'),mean_ms=seed_mean(rr,'e2e_ms'),time_ratio=cost,
            mean_relative_j_pct=100*(seed_mean(rr,'ratio')-1),worst_relative_j_pct=100*(min(r['ratio'] for r in rr)-1),
            mean_actual_phy=seed_mean(rr,'candidate_evaluations'),mean_feature_roots=seed_mean(rr,'feature_roots') if arm!='full' else None,
            mean_newly_unserved=seed_mean(rr,'newly_unserved_ues'),mean_positive_p5=seed_mean(rr,'positive_p5_ratio'),
            worst_positive_p5=min(r['positive_p5_ratio'] for r in rr),mean_positive_weak=seed_mean(rr,'positive_weak_ratio'),
            mean_joint_selected=seed_mean(rr,'selected_joint') if arm in CONFIGS else None,
            mean_joint_accepted=seed_mean(rr,'joint_accepted') if arm in CONFIGS else None,
            mean_guard_rejects=seed_mean(rr,'guard_rejects') if arm in CONFIGS else None,
            quality_pass=q,coverage_pass=c,cost_pass=cost<=.8,joint_pass=q and c and cost<=.8))
    return result


def benchmark(rows,old_model,model,out,phase):
    records=[];actions={};values={};snapshots={}
    for n,row in enumerate(rows):
        for repeat in range(3):
            shift=(n+repeat)%len(NAMES)
            for arm in NAMES[shift:]+NAMES[:shift]:
                a,v,r,s=run(row,arm,old_model,model);r['repeat']=repeat;records.append(r)
                key=f'{identity(row)}|{arm}'
                if key in actions:np.testing.assert_array_equal(actions[key],a);np.testing.assert_array_equal(values[key],v)
                else:
                    actions[key]=a;values[key]=v
                    if s is not None:snapshots[key]=s
        print('SELECTIVE_BENCH',phase,identity(row),flush=True)
    state=[];audits=[]
    for row in rows:
        p=problem_from_raw_row(row);sid=identity(row);ref=values[f'{sid}|full']
        anchor=ReferencePhy(p).evaluate(raw_pfq_draft(p,row['obs']))[0]
        common=snapshots[f'{sid}|increment_single_guard']
        for arm,config in CONFIGS.items():
            if config.protect:np.testing.assert_array_equal(snapshots[f'{sid}|{arm}'],common)
        full_ms=float(np.median([r['e2e_ms'] for r in records if r['state_id']==sid and r['arm']=='full']))
        for arm in NAMES:
            key=f'{sid}|{arm}';v=values[key];independent=ReferencePhy(p).evaluate(actions[key])[0]
            np.testing.assert_allclose(v,independent,rtol=2.e-9,atol=2.e-6)
            rr=[r for r in records if r['state_id']==sid and r['arm']==arm]
            assert independent[:,DEQUEUE].sum()>=rr[0]['floor']
            r={k:v for k,v in rr[0].items() if k!='repeat'};r['e2e_ms']=float(np.median([v['e2e_ms'] for v in rr]))
            r.update(ratio=r['score']/float(ref[:,EXPECTED]@p.weights),reference_ms=full_ms,time_ratio=r['e2e_ms']/full_ms,
                anchor_guard_pass=bool(ServiceGuard.against(v,anchor)[0]),**service_metrics(v,ref))
            if arm!='full':assert r['candidate_evaluations']<=212 and r['feature_roots']<=56
            if arm in CONFIGS:
                assert r['covered_prgs']==p.P
                if CONFIGS[arm].protect:assert r['anchor_guard_pass']
            state.append(r);audits.append(dict(state_id=sid,arm=arm,max_error=float(abs(v-independent).max())))
    csv_write(out/f'{phase}_measurements.csv',records);csv_write(out/f'{phase}_per_state.csv',state)
    csv_write(out/f'{phase}_summary.csv',summarize(state))
    np.savez_compressed(out/f'{phase}_actions.npz',**actions);np.savez_compressed(out/f'{phase}_values.npz',**values)
    write_json(out/f'{phase}_checks.json',dict(repeat_actions_values_identical=True,common_guarded_first_sweep=True,
        raw_floor_and_caps_valid=True,guarded_anchor_invariant=True,independent_phy=audits))
    return state


def select_arms(records):
    choices=[]
    for arm in NAMES[1:]:
        rr=[r for r in records if r['arm']==arm];prior=selection_key(rr)
        q=all(r['ratio']>=.995 for r in rr)
        c=all(r['newly_unserved_ues']==0 and r['positive_p5_ratio']>=.98 and r['positive_weak_ratio']>=.98 for r in rr)
        cost=seed_mean(rr,'e2e_ms')/seed_mean(rr,'reference_ms')
        choices.append(((int(q and c),*prior[1:]),arm,cost,seed_mean(rr,'e2e_ms')))
    best=max(choices,key=lambda v:(v[0],-v[3]));feasible=[v for v in choices if v[2]<=.8]
    usable=max(feasible,key=lambda v:(v[0],-v[3])) if feasible else min(choices,key=lambda v:v[3])
    return dict(quality_best=best[1],within_cost_best=usable[1],cost_feasible=bool(feasible),
        rule='validation only: quality/coverage, J shortfall, service shortfall, mean ratio; complete-cost filter for within-cost point',
        development_not_read_in_this_run=True,historical_exposure_retained=True)


def engineering(rows,out):
    measurements=[];checks=[];work=[]
    for row in (rows[0],rows[-1]):
        p=problem_from_raw_row(row);phy=ProbePhy(p)
        try:
            base=phy.evaluate(raw_pfq_draft(p,row['obs']));floor=raw_pfq_floor(base.values[0])
            d=Incremental(p,phy);wide=Descriptors(p,phy,56);d.get(base);wide.get(base)
            for prg in range(min(4,p.P)):
                for family,actions in zip(('single','joint'),candidate_pools(p,d,base,prg)):
                    ms={'incremental':[],'wide':[]}
                    for repeat in range(3):
                        for method in (('incremental','wide') if repeat%2==0 else ('wide','incremental')):
                            before=phy.stats.copy();start=time.perf_counter()
                            if method=='incremental':f=d.features(base,actions,floor)
                            else:proxy,total=wide.proxies(base,actions)
                            ms[method].append((time.perf_counter()-start)*1000);assert phy.stats==before
                    np.testing.assert_allclose(f['prior'],proxy,rtol=1.e-11,atol=1.e-7)
                    exact=phy.evaluate(actions,base);independent=ReferencePhy(p).evaluate(actions)
                    np.testing.assert_allclose(exact.values,independent,rtol=2.e-9,atol=2.e-6)
                    np.testing.assert_array_equal(f['counts'],exact.values[:,:,COUNT])
                    np.testing.assert_array_equal(f['layers'],np.maximum(exact.values[:,:,LAYER],0))
                    np.testing.assert_allclose(f['dequeue'],exact.values[:,:,DEQUEUE],rtol=0,atol=1.e-8)
                    measurements.append(dict(state_id=identity(row),prg=prg,family=family,candidates=len(actions),
                        incremental_ms=float(np.median(ms['incremental'])),wide_ms=float(np.median(ms['wide']))))
                    checks.append(dict(state_id=identity(row),prg=prg,family=family,
                        proxy_max_error=float(abs(f['prior']-proxy).max(initial=0)),
                        independent_max_error=float(abs(exact.values-independent).max(initial=0))))
            work.append(dict(state_id=identity(row),**phy.stats))
        finally:phy.close()
    csv_write(out/'engineering_features.csv',measurements)
    write_json(out/'engineering_checks.json',dict(passed=True,checks=checks,work=work,
        no_candidate_bler_or_mmse_in_features=True,whole_tb_logical_exact=True,
        interpretation='cached feature microbenchmark only; formal decisions include new context, pools and cleanup'))


def freeze(out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(date='2026-11-02',version=1,ctbs=False,headroom_masks=False,ppo=False,
        methods=NAMES,configs={k:asdict(v) for k,v in CONFIGS.items()},max_candidates=212,max_feature_roots=56,max_sweeps=5,
        objective='canonical native BLER, causal deficit-weighted J; own unpruned PFQ90% dequeue floor',
        incremental='C++ logical layer histogram updates on edited UEs, sparse activity-denominator updates; equivalent frozen old proxy; no candidate BLER/MMSE',
        features='60 causal dimensions:12 global including nominal weak-service risk and6 semantic UE slots x8; normalization training only',
        model='60->32->16->2 ReLU; residual gain plus feasibility logit; score=prior+3000*residual-3000*(1-sigmoid(logit))',
        protection='exact no newly unallocated UE, effective P5 and bottom20%>=98% vs both immutable initial PFQ and current root; candidate count prefilter exact',
        quota='first full sweep4 single edits, no model; then reserve2 singles and allocate remaining2 by joint/single score; fixed2+2 and single-only controls',
        cache='complete-action exact values local to immutable decision; rejection mask root-local, including unsafe failures; unsafe cache values may be reconsidered at new root',
        training_seeds=[41,42,43],validation_seeds=[44,45],evaluation_seeds=[46,47,48],
        training_validation_sources=['raw_global_deficit','pfq'],training_validation_ttis=[4001,8001,18000],
        evaluation_sources=['pfq'],evaluation_ttis=[3001,4001,5000],historical_development=True,
        final_seeds_not_read=[311,312,313,314,315],
        own_states='one initial zero-residual/heuristic selective guarded student; <=4 actual post-first-sweep visits/decision; root/PRG/failed mask preserved',
        teacher='separate post-decision fresh PHY, full local single/joint pool, exact service labels; all scores and replay work charged offline',
        training=dict(rounds=1,epochs=20,checkpoints=[0,5,10,20],seed=20261102,optimizer='fresh AdamW lr0.001',
            loss='safe listwise near-optimal gain with retained-root option +0.25 feasibility BCE +0.1 gain residual Huber'),
        checkpoint_selection='validation trajectory quality/coverage only, smaller epoch on ties; cost tested in formal repeated benchmark',
        policy_selection='quality-best and complete-cost<=80% best frozen on formal validation before development is read',
        learning_gate='changed selected weights, positive mean J delta every development seed and>=6/9 wins vs same selective guarded heuristic, plus joint quality/service/cost gate',
        quality_gate='each state J>=99.5% full',coverage_gate='no newly unallocated UE vs full; effective full-positive UE P5 and bottom20%>=98% full',
        cost_gate='seed-equal mean of complete per-state medians<=80% full',repeats=3,
        timing='numerical input/PFQ decode through fresh PHY/cache/pools/features/network/exact validation/guard/final scoring/cleanup',
        exclusions='archive/model/library IO, compilation and first immutable TBS setup; teachers/training/independent audits separate',
        old_model=str(OLD_MODEL),old_model_sha256=sha(OLD_MODEL),registry_sha256=sha(REGISTRY),platform=platform.platform())
    write_json(out/'protocol.json',protocol);sources={}
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.cpp','.so','.rst'):
            target=out/'source'/p.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target);sources[p.name]=sha(p)
    write_json(out/'source_manifest.json',sources);return protocol


def main(output):
    import torch
    torch.set_num_threads(1);out=Path(output).resolve();protocol=freeze(out);start=time.perf_counter()
    training,files=read_rows((41,42,43));training=[r for r in training if r['tti'] in protocol['training_validation_ttis']]
    write_json(out/'training_input_manifest.json',dict(files=files,states=[identity(r) for r in training]))
    problem_from_raw_row(training[0]);probe_library();feature_library();engineering(training,out)
    groups=collect(training,out);old_model=Predictor(OLD_MODEL)
    validation,files=read_rows((44,45));validation=[r for r in validation if r['tti'] in protocol['training_validation_ttis']]
    write_json(out/'validation_input_manifest.json',dict(files=files,states=[identity(r) for r in validation]))
    references={identity(row):run(row,'full',old_model,None)[1] for row in validation};checkpoint_records=[]
    def validate(model,epoch):
        rr=[]
        for row in validation:
            a,v,r,_=run(row,'trained_guard',old_model,model);reference=references[identity(row)]
            r.update(epoch=epoch,ratio=r['score']/float(reference[:,EXPECTED]@problem_from_raw_row(row).weights),
                **service_metrics(v,reference));rr.append(r);checkpoint_records.append(r)
        q=all(r['ratio']>=.995 for r in rr)
        c=all(r['newly_unserved_ues']==0 and r['positive_p5_ratio']>=.98 and r['positive_weak_ratio']>=.98 for r in rr)
        qshort=float(np.mean([max(.995-r['ratio'],0) for r in rr]))
        cshort=float(np.mean([max(.98-r['positive_p5_ratio'],0)+max(.98-r['positive_weak_ratio'],0)+r['newly_unserved_ues']/max(r['reference_positive_ues'],1) for r in rr]))
        metric=dict(mean_relative_j_pct=100*(seed_mean(rr,'ratio')-1),quality=q,coverage=c,mean_positive_p5=seed_mean(rr,'positive_p5_ratio'))
        return (int(q and c),-qshort,-cshort,seed_mean(rr,'ratio')),metric
    model,epoch=fit(groups,out,validate);csv_write(out/'checkpoint_validation.csv',checkpoint_records)
    write_json(out/'model_selection.json',dict(checkpoint=str(model.path),sha256=sha(model.path),epoch=epoch,
        normalization_training_only=True,one_training_round=True,development_not_read=True))
    state=benchmark(validation,old_model,model,out,'validation');selected=select_arms(state)
    write_json(out/'selection.json',selected);print('SELECTIVE SELECTION FROZEN',selected,flush=True)
    evaluation,files=read_rows((46,47,48),evaluation=True);evaluation=[r for r in evaluation if r['tti'] in protocol['evaluation_ttis']]
    write_json(out/'evaluation_input_manifest.json',dict(files=files,states=[identity(r) for r in evaluation]))
    benchmark(evaluation,old_model,model,out,'evaluation')
    assert sha(OLD_MODEL)==protocol['old_model_sha256'] and sha(REGISTRY)==protocol['registry_sha256']
    manifest=json.loads((out/'source_manifest.json').read_text())
    assert all(sha(Path(__file__).with_name(name))==digest for name,digest in manifest.items())
    write_json(out/'completion.json',dict(completed=True,formal_decisions=(len(validation)+len(evaluation))*len(NAMES)*3,
        formal_independent_final_audits=(len(validation)+len(evaluation))*len(NAMES),
        checkpoint_validation_decisions=len(checkpoint_records),validation_reference_decisions=len(references),
        training_states=len(training),validation_states=len(validation),development_states=len(evaluation),
        training_rounds=1,selected_epoch=epoch,registry_and_old_model_unchanged=True,executed_sources_unchanged=True,
        native_trajectories=0,final_seeds_read=False,elapsed_s=time.perf_counter()-start))
    print('SELECTIVE COMPLETE',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);main(parser.parse_args().output)

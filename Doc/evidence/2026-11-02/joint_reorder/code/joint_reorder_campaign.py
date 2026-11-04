"""Controlled wide-reorder and J/service-frontier ablations, offline conditions."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import shutil
import time
import numpy as np

from .joint_reorder import ReorderConfig,solve_joint,JointFeatures,library as delta_library
from .resource_condition_campaign import teachers,compressed_condition,REGISTRY
from .resource_conditions import ConditionalPhy,library as physical_library,seed_action
from .resource_conditions_fast import solve_conditional as old_solve,pool_library
from .shortlist_trajectory import service_metrics
from .shortlist_campaign import selection_key
from .bounded_probes import csv_write,seed_mean
from .physics import ROOT,ReferencePhy,EXPECTED,DEQUEUE,COUNT
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .ranked_dpls import solve_ranked
from .ranked_dpls_campaign import read_rows,identity
from .search import ProposalGraph
from .experiment import write_json,sha

CONFIGS={
 'narrow_delta_j':ReorderConfig(False,False,0),
 'wide_j':ReorderConfig(True,False,0),
 'narrow_pareto005':ReorderConfig(False,True,.005),
 'wide_pareto_j':ReorderConfig(True,True,0),
 'wide_pareto005':ReorderConfig(True,True,.005),
 'wide_pareto02':ReorderConfig(True,True,.02),
}
NAMES=['full','activity_reference','joint_reference',*CONFIGS]


def run(row,arm,teacher):
    start=time.perf_counter();p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs']);group=None
    if arm=='full':a,v,r=solve_ranked(p,draft)
    elif arm in ('activity_reference','joint_reference'):
        condition=compressed_condition(p,*teacher,'activity212' if arm=='activity_reference' else 'joint212')
        a,v,r=old_solve(p,draft,condition,offline_diagnostic=True)
    else:
        condition=compressed_condition(p,*teacher,'joint212')
        a,v,r,group=solve_joint(p,draft,condition,CONFIGS[arm],offline_diagnostic=True)
    r['score']=float(v[:,EXPECTED]@p.weights);r['e2e_ms']=(time.perf_counter()-start)*1000;r.pop('history',None)
    r.update(state_id=identity(row),seed=row['seed'],source=row['family'],tti=row['tti'],arm=arm)
    return a,v,r,group


def summarize(state):
    full=[r for r in state if r['arm']=='full'];out=[]
    for arm in NAMES:
        rr=[r for r in state if r['arm']==arm];q=all(r['ratio']>=.995 for r in rr)
        c=all(r['newly_unserved_ues']==0 and r['positive_p5_ratio']>=.98 and r['positive_weak_ratio']>=.98 for r in rr)
        valid=all(r.get('condition_feasible',True) for r in rr);cost=seed_mean(rr,'e2e_ms')/seed_mean(full,'e2e_ms')
        out.append(dict(arm=arm,mean_j=seed_mean(rr,'score'),mean_ms=seed_mean(rr,'e2e_ms'),execution_time_ratio=cost,
            mean_relative_j_pct=100*(seed_mean(rr,'ratio')-1),worst_relative_j_pct=100*(min(r['ratio'] for r in rr)-1),
            mean_newly_unserved=seed_mean(rr,'newly_unserved_ues'),mean_positive_p5=seed_mean(rr,'positive_p5_ratio'),
            worst_positive_p5=min(r['positive_p5_ratio'] for r in rr),mean_positive_weak=seed_mean(rr,'positive_weak_ratio'),
            mean_actual_phy=seed_mean(rr,'candidate_evaluations'),mean_feature_roots=seed_mean(rr,'feature_roots') if arm!='full' else None,
            mean_selected_j_loss=seed_mean(rr,'selected_j_loss_fraction') if arm in CONFIGS else None,
            quality_pass=valid and q,coverage_pass=valid and c,execution_cost_pass=cost<=.8,
            conditional_target_pass=valid and q and c and cost<=.8,online_ai_success=False))
    return out


def benchmark(rows,teacher,out,phase):
    raw=[];actions={};values={};groups={}
    for n,row in enumerate(rows):
        for repeat in range(3):
            shift=(n+repeat)%len(NAMES)
            for arm in NAMES[shift:]+NAMES[:shift]:
                a,v,r,g=run(row,arm,teacher[identity(row)]);r['repeat']=repeat;raw.append(r);key=f'{identity(row)}|{arm}'
                if key in actions:
                    np.testing.assert_array_equal(actions[key],a);np.testing.assert_array_equal(values[key],v)
                    if g is not None:np.testing.assert_array_equal(groups[key],g)
                else:
                    actions[key]=a;values[key]=v
                    if g is not None:groups[key]=g
        print('JOINT_REORDER',phase,identity(row),flush=True)
    state=[];audits=[];equivalence=[]
    for row in rows:
        p=problem_from_raw_row(row);sid=identity(row);ref=values[f'{sid}|full'];ref_j=float(ref[:,EXPECTED]@p.weights)
        full_ms=float(np.median([r['e2e_ms'] for r in raw if r['state_id']==sid and r['arm']=='full']))
        np.testing.assert_array_equal(actions[f'{sid}|full'],teacher[sid][0])
        equivalence.append(dict(state_id=sid,actions_equal=bool(np.array_equal(actions[f'{sid}|joint_reference'],actions[f'{sid}|narrow_delta_j'])),
            values_equal=bool(np.array_equal(values[f'{sid}|joint_reference'],values[f'{sid}|narrow_delta_j']))))
        for arm in NAMES:
            key=f'{sid}|{arm}';v=values[key];independent=ReferencePhy(p).evaluate(actions[key])[0]
            np.testing.assert_allclose(v,independent,rtol=2.e-9,atol=2.e-6)
            rr=[r for r in raw if r['state_id']==sid and r['arm']==arm]
            r={k:v for k,v in rr[0].items() if k!='repeat'};r['e2e_ms']=float(np.median([v['e2e_ms'] for v in rr]))
            r.update(ratio=r['score']/ref_j,reference_ms=full_ms,execution_time_ratio=r['e2e_ms']/full_ms,**service_metrics(v,ref))
            if arm!='full':assert r['candidate_evaluations']<=212 and r['feature_roots']<=56 and r['condition_feasible']
            if arm in CONFIGS:
                assert r['score']+1.e-8>=(1-CONFIGS[arm].final_j_loss)*r['best_observed_j']
                target=ref[:,EXPECTED]>1.;group=groups[key]
                r.update(service_group_reference_recall=float((group&target).sum()/max(target.sum(),1)),
                    service_group_extra=int((group&~target).sum()))
            state.append(r);audits.append(dict(state_id=sid,arm=arm,max_error=float(abs(v-independent).max())))
        # These three variants differ ONLY in the final archive selector.
        selector=['wide_pareto_j','wide_pareto005','wide_pareto02']
        for field in ('best_observed_j','candidate_evaluations','proxy_candidates','feature_roots','iterations'):
            vv=[next(r[field] for r in state if r['state_id']==sid and r['arm']==arm) for arm in selector]
            assert vv.count(vv[0])==len(vv)
        for arm in selector[1:]:np.testing.assert_array_equal(groups[f'{sid}|{arm}'],groups[f'{sid}|{selector[0]}'])
    csv_write(out/f'{phase}_measurements.csv',raw);csv_write(out/f'{phase}_per_state.csv',state)
    csv_write(out/f'{phase}_summary.csv',summarize(state));csv_write(out/f'{phase}_delta_equivalence.csv',equivalence)
    np.savez_compressed(out/f'{phase}_actions.npz',**actions);np.savez_compressed(out/f'{phase}_values.npz',**values)
    np.savez_compressed(out/f'{phase}_service_groups.npz',**groups)
    write_json(out/f'{phase}_checks.json',dict(repeats_actions_values_groups_identical=True,full_phy=audits,
        phase_budgets_and_fixed_observed_j_bound_valid=True,selector_only_searches_identical=True,
        no_teacher_service_group_or_positions=True,delta_reference_equivalent=sum(r['actions_equal'] and r['values_equal'] for r in equivalence)))
    return state


def choose(state):
    choices=[]
    for arm in CONFIGS:
        rr=[r for r in state if r['arm']==arm];key=selection_key(rr)
        q=all(r['ratio']>=.995 for r in rr)
        c=all(r['newly_unserved_ues']==0 and r['positive_p5_ratio']>=.98 and r['positive_weak_ratio']>=.98 for r in rr)
        cost=seed_mean(rr,'e2e_ms')/seed_mean(rr,'reference_ms')
        choices.append(((int(q and c),*key[1:]),arm,cost,seed_mean(rr,'e2e_ms')))
    best=max(choices,key=lambda v:(v[0],-v[3]));feasible=[v for v in choices if v[2]<=.8]
    within=max(feasible,key=lambda v:(v[0],-v[3])) if feasible else min(choices,key=lambda v:v[3])
    return dict(quality_candidate=best[1],execution_cost_candidate=within[1],execution_cost_feasible=bool(feasible),
        validation_only=True,development_not_read=True,historical_exposure=True,teacher_preparation_excluded=True,
        rule='final quality/service, J shortfall, service shortfall, mean ratio; execution cost filter and lower time on ties',online_ai_claim=False)


def engineering(rows,out):
    refs=teachers(rows,out,'engineering');records=[];checks=[]
    for row in rows:
        p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs']);condition=compressed_condition(p,*refs[identity(row)],'joint212')
        phy=ConditionalPhy(p)
        try:
            bank=phy.bank();a=seed_action(p,phy.evaluate(draft),bank,ProposalGraph(phy),condition,np.random.default_rng(20261102),0)
            base=phy.evaluate(a);features=JointFeatures(p,bank,condition);pool=features.pool(base,True,20261102)
            sparse,_,valid=features.proxy(base,pool);dense,_,_,_=phy.proxy(base,pool)
            np.testing.assert_allclose(sparse[valid],dense[valid],rtol=1.e-10,atol=1.e-7)
            checks.append(dict(state_id=identity(row),wide_candidates=len(pool),max_error=float(abs(sparse[valid]-dense[valid]).max(initial=0))))
        finally:phy.close()
        for arm in ('joint_reference','narrow_delta_j','wide_j','wide_pareto005'):
            a,v,r,_=run(row,arm,refs[identity(row)]);records.append(r)
            np.testing.assert_allclose(v,ReferencePhy(p).evaluate(a)[0],rtol=2.e-9,atol=2.e-6)
    csv_write(out/'engineering_per_state.csv',records);write_json(out/'engineering_checks.json',dict(passed=True,checks=checks,training_states_only=True,tuning=False))


def freeze(out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(date='2026-11-02',version=1,methods=NAMES,configs={k:asdict(v) for k,v in CONFIGS.items()},
        ctbs=False,headroom_masks=False,ppo=False,training=False,
        objective='canonical native BLER, causal deficit-weighted J; own unpruned PFQ90% dequeue floor',
        conditions='fresh offline teacher counts/exact min-layer/activity only; no UE-to-PRG positions, teacher J/service/group in solver',
        wider='all active PRG3-cycles plus4-slot cyclic windows in8 deterministic orders, allPRG coverage; paired narrow pool remains first6',
        sparse_proxy='same root-anchored smooth reliability but update affected UE gamma sums only; no candidate native BLER or exact J; no new MMSE solves after bank',
        work='count native gamma requests/misses PLUS direct gamma array reads, seed matching reads, generation attempts and approximate candidates; equal212 is not equal work/time',
        service_group='freeze union of >1B UE service observed in paid16-seed archive, before main search; no teacher-positive mask',
        frontier='exact-observed J/P5/bottom20% nondomination; 4 roots incl maxJ/service extremes within95% best available-root J; at most2/8 proposals for projected service',
        final_selector='max J or maximize observed service within0.5%/2% of ONE fixed archive bestJ; does not relax final99.5%-full gate',
        engineering='2 training states41-43,8solver runs and sparse/dense proxy checks, no tuning',
        validation_seeds=[44,45],validation_sources=['raw_global_deficit','pfq'],validation_ttis=[4001,8001,18000],
        evaluation_seeds=[46,47,48],evaluation_sources=['pfq'],evaluation_ttis=[3001,4001,5000],historical_development=True,
        final_seeds_not_read=[311,312,313,314,315],repeats=3,
        selection='freeze breadth/frontier/final selector on validation before development results; no larger model trained this round',
        gates='each J>=99.5% full, no newly unallocated UE, full-positive P5/bottom20%>=98%, execution mean<=80% full',
        timing='numeric input/PFQ through new PHY/MMSE bank, condition compression, reorder pools, proxies/frontiers, exact scoring, final selection/cleanup',
        exclusions='fresh teacher preparation, compilation/archive/library IO and first immutable TBS setup; all teacher/proxy work separately recorded',
        registry_sha256=sha(REGISTRY),physics_source_sha256=sha(Path(__file__).with_name('physics_kernel.cpp')),platform=platform.platform())
    write_json(out/'protocol.json',protocol);manifest={}
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.cpp','.so','.rst'):
            target=out/'source'/p.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target);manifest[p.name]=sha(p)
    write_json(out/'source_manifest.json',manifest);return protocol


def main(output):
    out=Path(output).resolve();protocol=freeze(out);start=time.perf_counter()
    training,files=read_rows((41,42,43));training=[r for r in training if r['tti'] in protocol['validation_ttis']]
    erows=[training[0],training[-1]];write_json(out/'engineering_input_manifest.json',dict(files=files,states=[identity(r) for r in erows]))
    problem_from_raw_row(erows[0]);physical_library();pool_library();delta_library();engineering(erows,out)
    validation,files=read_rows((44,45));validation=[r for r in validation if r['tti'] in protocol['validation_ttis']]
    write_json(out/'validation_input_manifest.json',dict(files=files,states=[identity(r) for r in validation]))
    reference=teachers(validation,out,'validation');state=benchmark(validation,reference,out,'validation')
    selection=choose(state);write_json(out/'selection.json',selection);print('JOINT SELECTION FROZEN',selection,flush=True)
    evaluation,files=read_rows((46,47,48),evaluation=True);evaluation=[r for r in evaluation if r['tti'] in protocol['evaluation_ttis']]
    write_json(out/'evaluation_input_manifest.json',dict(files=files,states=[identity(r) for r in evaluation]))
    reference=teachers(evaluation,out,'evaluation');benchmark(evaluation,reference,out,'evaluation')
    assert sha(REGISTRY)==protocol['registry_sha256'] and sha(Path(__file__).with_name('physics_kernel.cpp'))==protocol['physics_source_sha256']
    assert all(sha(Path(__file__).with_name(name))==digest for name,digest in json.loads((out/'source_manifest.json').read_text()).items())
    write_json(out/'completion.json',dict(completed=True,formal_decisions=(len(validation)+len(evaluation))*len(NAMES)*3,
        formal_independent_finals=(len(validation)+len(evaluation))*len(NAMES),offline_teacher_decisions=23,engineering_decoder_decisions=8,
        validation_states=len(validation),development_states=len(evaluation),training_updates=0,
        teacher_positions_and_service_not_in_solver=True,registry_and_physics_unchanged=True,executed_sources_unchanged=True,
        native_trajectories=0,final_seeds_read=False,elapsed_s=time.perf_counter()-start))
    print('JOINT REORDER COMPLETE',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);main(parser.parse_args().output)

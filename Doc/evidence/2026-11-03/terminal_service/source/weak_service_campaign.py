"""Frozen seven-arm continuous-service and equivalent-cost experiment."""
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

from .weak_service import ServiceConfig,solve_service
from .joint_reorder import ReorderConfig,solve_joint,library as delta_library
from .resource_condition_campaign import teachers,compressed_condition,REGISTRY
from .resource_conditions import satisfies,library as physical_library
from .resource_conditions_fast import solve_conditional as activity_solve,pool_library
from .shortlist_trajectory import service_metrics
from .shortlist_campaign import selection_key
from .bounded_probes import csv_write,seed_mean
from .physics import ROOT,ReferencePhy,EXPECTED,DEQUEUE
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .ranked_dpls import solve_ranked
from .ranked_dpls_campaign import read_rows,identity
from .experiment import write_json,sha
from .tb_macro import ExactMemo

CONFIGS={
    'frontier_fast':ServiceConfig(),
    'quota_p5':ServiceConfig(metric='quota_p5'),
    'quota_log':ServiceConfig(metric='quota_log'),
    'quota_exp':ServiceConfig(metric='quota_exp'),
}
NAMES=['full','activity_reference','frontier_reference',*CONFIGS]
PARITY_FIELDS=('candidate_evaluations','ue_evaluations','mmse_requests','mmse_misses','feature_roots',
    'iterations','proxy_candidates','proposal_attempts','proxy_gamma_array_reads','seed_gamma_array_reads',
    'service_group_size','frontier_peak','best_observed_j','query_sha256','archive_sha256')


class RecordedMemo(ExactMemo):
    """Identical observation hook for old/new frontiers; digest outside timing."""
    latest=None
    def __init__(self,*args):
        super().__init__(*args);self.events=[self.root.actions.tobytes()];RecordedMemo.latest=self
    def evaluate(self,actions,base):
        result=super().evaluate(actions,base)
        self.events.extend([base.actions.tobytes(),result[0].actions.tobytes() if result[0] is not None else b''])
        return result


def run(row,arm,teacher):
    RecordedMemo.latest=None
    start=time.perf_counter();p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs']);group=None
    if arm=='full':a,v,r=solve_ranked(p,draft)
    elif arm=='activity_reference':
        condition=compressed_condition(p,*teacher,'activity212')
        a,v,r=activity_solve(p,draft,condition,offline_diagnostic=True)
    else:
        condition=compressed_condition(p,*teacher,'joint212')
        if arm=='frontier_reference':a,v,r,group=solve_joint(p,draft,condition,ReorderConfig(True,True,0),offline_diagnostic=True)
        else:a,v,r,group=solve_service(p,draft,condition,CONFIGS[arm],offline_diagnostic=True)
    r['score']=float(v[:,EXPECTED]@p.weights);r['e2e_ms']=(time.perf_counter()-start)*1000;r.pop('history',None)
    r.update(state_id=identity(row),seed=row['seed'],source=row['family'],tti=row['tti'],arm=arm)
    memo=RecordedMemo.latest;observed=None
    if memo is not None:
        observed=np.stack(list(memo.rows.values()))
        r['query_sha256']=hashlib.sha256(b''.join(memo.events)).hexdigest()
        r['archive_sha256']=hashlib.sha256(b''.join(memo.rows)+observed.tobytes()).hexdigest()
    return a,v,r,group,observed


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


def benchmark(rows,teachers_by_id,out,phase):
    raw=[];actions={};values={};groups={};observed={}
    for n,row in enumerate(rows):
        for repeat in range(3):
            shift=(n+repeat)%len(NAMES)
            for arm in NAMES[shift:]+NAMES[:shift]:
                a,v,r,g,visited=run(row,arm,teachers_by_id[identity(row)]);r['repeat']=repeat;raw.append(r)
                key=f'{identity(row)}|{arm}'
                if key in actions:
                    np.testing.assert_array_equal(actions[key],a);np.testing.assert_array_equal(values[key],v)
                    if g is not None:
                        np.testing.assert_array_equal(groups[key],g);np.testing.assert_array_equal(observed[key],visited)
                else:
                    actions[key]=a;values[key]=v
                    if g is not None:groups[key]=g;observed[key]=visited
        print('WEAK_SERVICE',phase,identity(row),flush=True)
    state=[];audits=[];parity=[]
    for row in rows:
        p=problem_from_raw_row(row);sid=identity(row);ref=values[f'{sid}|full'];ref_j=float(ref[:,EXPECTED]@p.weights)
        full_ms=float(np.median([r['e2e_ms'] for r in raw if r['state_id']==sid and r['arm']=='full']))
        np.testing.assert_array_equal(actions[f'{sid}|full'],teachers_by_id[sid][0])
        for arm in NAMES:
            key=f'{sid}|{arm}';v=values[key];independent=ReferencePhy(p).evaluate(actions[key])[0]
            np.testing.assert_allclose(v,independent,rtol=2e-9,atol=2e-6)
            rr=[r for r in raw if r['state_id']==sid and r['arm']==arm]
            r={k:v for k,v in rr[0].items() if k!='repeat'};r['e2e_ms']=float(np.median([v['e2e_ms'] for v in rr]))
            r.update(ratio=r['score']/ref_j,reference_ms=full_ms,execution_time_ratio=r['e2e_ms']/full_ms,**service_metrics(v,ref))
            assert v[:,DEQUEUE].sum()>=r['floor']
            if arm!='full':
                assert r['candidate_evaluations']<=212 and r['feature_roots']<=56 and r['condition_feasible']
                cond=compressed_condition(p,*teachers_by_id[sid],'activity212' if arm=='activity_reference' else 'joint212')
                assert satisfies(p,actions[key],cond)[0] and r['mmse_misses']==2448
            if key in groups:
                target=ref[:,EXPECTED]>1;g=groups[key]
                r.update(service_group_reference_recall=float((g&target).sum()/max(target.sum(),1)),service_group_extra=int((g&~target).sum()))
                assert len({z['query_sha256'] for z in rr})==len({z['archive_sha256'] for z in rr})==1
                assert r['score']==r['best_observed_j']
            state.append(r);audits.append(dict(state_id=sid,arm=arm,max_error=float(abs(v-independent).max())))
        old=next(r for r in state if r['state_id']==sid and r['arm']=='frontier_reference')
        new=next(r for r in state if r['state_id']==sid and r['arm']=='frontier_fast')
        for field in PARITY_FIELDS:assert old[field]==new[field],(sid,field,old[field],new[field])
        for data in (actions,values,groups,observed):np.testing.assert_array_equal(data[f'{sid}|frontier_reference'],data[f'{sid}|frontier_fast'])
        parity.append(dict(state_id=sid,actions_values_groups_all_observed_equal=True,query_roots_order_and_work_equal=True,
            query_sha256=new['query_sha256'],archive_sha256=new['archive_sha256']))
    csv_write(out/f'{phase}_measurements.csv',raw);csv_write(out/f'{phase}_per_state.csv',state)
    csv_write(out/f'{phase}_summary.csv',summarize(state));csv_write(out/f'{phase}_equivalence.csv',parity)
    for name,data in [('actions',actions),('values',values),('service_groups',groups),('observed_values',observed)]:np.savez_compressed(out/f'{phase}_{name}.npz',**data)
    write_json(out/f'{phase}_checks.json',dict(repeats_actions_values_groups_queries_identical=True,full_phy=audits,
        budget_floor_conditions_valid=True,fast_equivalent_states=len(parity),no_teacher_service_group_or_positions=True))
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
        rule='same prior rule: all-state quality/service then mean quality shortfall, service shortfall, mean ratio; execution cost filter, lower time on ties',online_ai_claim=False)


def engineering(rows,out):
    refs=teachers(rows,out,'engineering');records=[];checks=[]
    for row in rows:
        result={}
        for arm in ['frontier_reference',*CONFIGS]:
            a,v,r,g,visited=run(row,arm,refs[identity(row)]);records.append(r);result[arm]=(a,v,r,g,visited)
            independent=ReferencePhy(problem_from_raw_row(row)).evaluate(a)[0]
            np.testing.assert_allclose(v,independent,rtol=2e-9,atol=2e-6)
            checks.append(dict(state_id=identity(row),arm=arm,max_error=float(abs(v-independent).max())))
        old=result['frontier_reference'];new=result['frontier_fast']
        for i in (0,1,3,4):np.testing.assert_array_equal(old[i],new[i])
        for field in PARITY_FIELDS:assert old[2][field]==new[2][field],field
    csv_write(out/'engineering_per_state.csv',records)
    write_json(out/'engineering_checks.json',dict(passed=True,training_states_only=True,tuning=False,independent_phy=checks,
        fast_reference_actions_all_queries_all_values_work_equal=True))


def freeze(out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(date='2026-11-02',version=1,methods=NAMES,configs={k:asdict(v) for k,v in CONFIGS.items()},
        ctbs=False,headroom_masks=False,ppo=False,training=False,objective='canonical native BLER, causal deficit-weighted J; own unpruned PFQ90% dequeue floor',
        conditions='fresh paid offline teacher counts/exact min-layer/activity only; no teacher UE positions/J/service/group in solver',
        moves='unchanged wide: all active PRG3-cycles plus4-slot cyclic windows in8 deterministic orders',
        service_metrics=dict(bootstrap_p5='paid seed union >1B; original raw-byte P5/bottom20%',quota_p5='all logical-positive-dequeue quota UEs, same raw-byte P5/bottom20%',
            quota_log='r=clip(expected/dequeue,0,1); mean(log1p(r/0.01))/log1p(100), mean(r)',
            quota_exp='same r; -0.1*log(mean(exp(-r/0.1))), mean(r)'),
        constants='fixed before validation, no smoothing/retention/budget tuning',
        final_selector='maximum exact observed J for every new arm; no final J-risk relaxation',
        frontier='same4 roots /95% current-expandable-best retention / at most2 of8 service proposals; new metric replaces both exact and proxy service coordinates',
        optimization='batch only newly observed service; cache dequeue; incremental pairwise dominance/equality; one frontier query; stable query-order ties and revival after dominator exhaustion',
        instrumentation='same lightweight ExactMemo event copies in old/new frontiers, charged inside timing; ordered root/query/all-value digests and diagnostic arrays outside timing',
        engineering='2 training states41-43 x5 decoders; full query/action/value/work parity, no tuning',
        validation_seeds=[44,45],validation_sources=['raw_global_deficit','pfq'],validation_ttis=[4001,8001,18000],
        evaluation_seeds=[46,47,48],evaluation_sources=['pfq'],evaluation_ttis=[3001,4001,5000],historical_development=True,
        final_seeds_not_read=[311,312,313,314,315],repeats=3,
        gates='each J>=99.5%full; no newly unallocatedUE; full-positive P5/bottom20%>=98%; complete execution mean<=80%full',
        timing='numeric input/PFQ through freshPHY/MMSE bank, condition compression, all pools/proxies/frontiers/exact scoring/final selection/cleanup',
        exclusions='teacher preparation, compilation/library/archiveIO, first immutableTBS setup; digests/independent audits/posthoc diagnosis outside timing',
        work='all exact scores,2448 unique MMSE, native requests PLUS direct proxy/seed gamma reads, generation attempts, frontier pairs/queries/metric rows',
        threads=dict(OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1),
        registry_sha256=sha(REGISTRY),physics_source_sha256=sha(ROOT/'training/stageb_lns/physics_kernel.cpp'),platform=platform.platform())
    write_json(out/'protocol.json',protocol);manifest={}
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.cpp','.so','.rst'):
            target=out/'source'/p.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target);manifest[p.name]=sha(p)
    write_json(out/'source_manifest.json',manifest);return protocol


def main(output):
    out=Path(output).resolve();protocol=freeze(out);start=time.perf_counter()
    training,files=read_rows((41,42,43));training=[r for r in training if r['tti'] in protocol['validation_ttis']]
    erows=[training[0],training[-1]];write_json(out/'engineering_input_manifest.json',dict(files=files,states=[identity(r) for r in erows]))
    problem_from_raw_row(erows[0]);physical_library();pool_library();delta_library()
    with patch('training.stageb_lns.joint_reorder.ExactMemo',RecordedMemo),patch('training.stageb_lns.weak_service.ExactMemo',RecordedMemo):
        engineering(erows,out)
        validation,files=read_rows((44,45));validation=[r for r in validation if r['tti'] in protocol['validation_ttis']]
        write_json(out/'validation_input_manifest.json',dict(files=files,states=[identity(r) for r in validation]))
        reference=teachers(validation,out,'validation');state=benchmark(validation,reference,out,'validation')
        selection=choose(state);write_json(out/'selection.json',selection);print('WEAK SERVICE SELECTION FROZEN',selection,flush=True)
        evaluation,files=read_rows((46,47,48),evaluation=True);evaluation=[r for r in evaluation if r['tti'] in protocol['evaluation_ttis']]
        write_json(out/'evaluation_input_manifest.json',dict(files=files,states=[identity(r) for r in evaluation]))
        reference=teachers(evaluation,out,'evaluation');benchmark(evaluation,reference,out,'evaluation')
    assert sha(REGISTRY)==protocol['registry_sha256'] and sha(ROOT/'training/stageb_lns/physics_kernel.cpp')==protocol['physics_source_sha256']
    assert all(sha(Path(__file__).with_name(name))==digest for name,digest in json.loads((out/'source_manifest.json').read_text()).items())
    write_json(out/'completion.json',dict(completed=True,formal_decisions=441,formal_independent_finals=147,
        offline_teacher_decisions=23,engineering_decoder_decisions=10,validation_states=12,development_states=9,
        training_updates=0,registry_and_physics_unchanged=True,executed_sources_unchanged=True,native_trajectories=0,
        final_seeds_read=False,elapsed_s=time.perf_counter()-start))
    print('WEAK SERVICE COMPLETE',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);main(parser.parse_args().output)

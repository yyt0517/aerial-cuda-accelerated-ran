"""Six-arm fixed-budget directed joint reorder plus separately paid recall."""
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
from .weak_service import ServiceConfig,solve_service
from .weak_service_campaign import RecordedMemo,PARITY_FIELDS
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

CONFIGS={
    'existing2':DirectedConfig(joint_candidates=False),
    'joint2':DirectedConfig(),
    'joint4':DirectedConfig(directed_slots=4),
    'sparse2':DirectedConfig(alternate_sparse=True),
}
NAMES=['full','baseline',*CONFIGS]
PREVIOUS=ROOT/'deliverables/weak_service_20261102_v1'


def run(row,arm,teacher):
    RecordedMemo.latest=None
    start=time.perf_counter();p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs']);group=None
    if arm=='full':a,v,r=solve_ranked(p,draft)
    else:
        condition=compressed_condition(p,*teacher,'joint212')
        if arm=='baseline':a,v,r,group=solve_service(p,draft,condition,ServiceConfig(metric='quota_exp'),offline_diagnostic=True)
        else:
            cfg=DirectedConfig(directed_slots=0,joint_candidates=False) if arm=='disabled' else CONFIGS[arm]
            a,v,r,group=solve_directed(p,draft,condition,cfg,offline_diagnostic=True)
    r['score']=float(v[:,EXPECTED]@p.weights);r['e2e_ms']=(time.perf_counter()-start)*1000;r.pop('history',None)
    r.update(state_id=identity(row),seed=row['seed'],source=row['family'],tti=row['tti'],arm=arm)
    memo=RecordedMemo.latest;observed=visited=None
    if memo is not None:
        observed=np.stack(list(memo.rows.values()))
        visited=np.frombuffer(b''.join(memo.rows),np.int64).reshape(-1,p.C,p.P).copy()
        r['query_sha256']=hashlib.sha256(b''.join(memo.events)).hexdigest()
        r['archive_sha256']=hashlib.sha256(b''.join(memo.rows)+observed.tobytes()).hexdigest()
    return a,v,r,group,observed,visited


def benchmark(rows,refs,out,phase):
    raw=[];actions={};values={};groups={};observed={};visited={}
    for n,row in enumerate(rows):
        for repeat in range(3):
            shift=(n+repeat)%len(NAMES)
            for arm in NAMES[shift:]+NAMES[:shift]:
                a,v,r,g,vv,aa=run(row,arm,refs[identity(row)]);r['repeat']=repeat;raw.append(r);key=f'{identity(row)}|{arm}'
                if key in actions:
                    np.testing.assert_array_equal(actions[key],a);np.testing.assert_array_equal(values[key],v)
                    if g is not None:
                        for old,new in ((groups[key],g),(observed[key],vv),(visited[key],aa)):np.testing.assert_array_equal(old,new)
                else:
                    actions[key]=a;values[key]=v
                    if g is not None:groups[key]=g;observed[key]=vv;visited[key]=aa
        print('DIRECTED_REORDER',phase,identity(row),flush=True)
    state=[];checks=[];parity=[]
    previous_actions=np.load(PREVIOUS/f'{phase}_actions.npz');previous_values=np.load(PREVIOUS/f'{phase}_values.npz')
    for row in rows:
        p=problem_from_raw_row(row);sid=identity(row);ref=values[f'{sid}|full'];jref=float(ref[:,EXPECTED]@p.weights)
        full_ms=float(np.median([r['e2e_ms'] for r in raw if r['state_id']==sid and r['arm']=='full']))
        np.testing.assert_array_equal(actions[f'{sid}|full'],refs[sid][0])
        np.testing.assert_array_equal(actions[f'{sid}|baseline'],previous_actions[f'{sid}|quota_exp'])
        np.testing.assert_array_equal(values[f'{sid}|baseline'],previous_values[f'{sid}|quota_exp'])
        parity.append(dict(state_id=sid,baseline_previous_actions_values_equal=True))
        for arm in NAMES:
            key=f'{sid}|{arm}';v=values[key];independent=ReferencePhy(p).evaluate(actions[key])[0]
            np.testing.assert_allclose(v,independent,rtol=2e-9,atol=2e-6)
            rr=[r for r in raw if r['state_id']==sid and r['arm']==arm]
            r={k:v for k,v in rr[0].items() if k!='repeat'};r['e2e_ms']=float(np.median([v['e2e_ms'] for v in rr]))
            r.update(ratio=r['score']/jref,reference_ms=full_ms,execution_time_ratio=r['e2e_ms']/full_ms,**service_metrics(v,ref))
            assert v[:,DEQUEUE].sum()>=r['floor']
            if arm!='full':
                assert r['candidate_evaluations']<=212 and r['feature_roots']<=56 and r['condition_feasible'] and not r['fallback_to_raw']
                condition=compressed_condition(p,*refs[sid],'joint212');assert satisfies(p,actions[key],condition)[0]
                assert r['mmse_misses']==2448 and r['score']==r['best_observed_j']
                assert len({z['query_sha256'] for z in rr})==len({z['archive_sha256'] for z in rr})==1
                np.testing.assert_array_equal(groups[key],groups[f'{sid}|baseline'])
            state.append(r);checks.append(dict(state_id=sid,arm=arm,max_error=float(abs(v-independent).max())))
    csv_write(out/f'{phase}_measurements.csv',raw);csv_write(out/f'{phase}_per_state.csv',state)
    csv_write(out/f'{phase}_summary.csv',summarize(state));csv_write(out/f'{phase}_baseline_equivalence.csv',parity)
    for name,data in [('actions',actions),('values',values),('service_groups',groups),('observed_values',observed),('observed_actions',visited)]:np.savez_compressed(out/f'{phase}_{name}.npz',**data)
    write_json(out/f'{phase}_checks.json',dict(repeats_actions_values_groups_queries_identical=True,full_phy=checks,
        budget_floor_conditions_valid=True,baseline_previous_equivalent_states=len(parity),no_teacher_service_group_or_positions=True))
    return state


def engineering(rows,out):
    refs=teachers(rows,out,'engineering');records=[];checks=[]
    for row in rows:
        result={}
        for arm in ['baseline','disabled',*CONFIGS]:
            a,v,r,g,vv,aa=run(row,arm,refs[identity(row)]);records.append(r);result[arm]=(a,v,r,g,vv,aa)
            independent=ReferencePhy(problem_from_raw_row(row)).evaluate(a)[0]
            np.testing.assert_allclose(v,independent,rtol=2e-9,atol=2e-6)
            checks.append(dict(state_id=identity(row),arm=arm,max_error=float(abs(v-independent).max())))
        old=result['baseline'];new=result['disabled']
        for i in (0,1,3,4,5):np.testing.assert_array_equal(old[i],new[i])
        for field in PARITY_FIELDS:assert old[2][field]==new[2][field],field
    csv_write(out/'engineering_per_state.csv',records)
    write_json(out/'engineering_checks.json',dict(passed=True,training_states_only=True,tuning=False,independent_phy=checks,disabled_full_query_work_parity=True))


def freeze(out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(date='2026-11-02',version=1,methods=NAMES,configs={k:asdict(v) for k,v in CONFIGS.items()},
        ctbs=False,headroom_masks=False,ppo=False,training=False,
        objective='canonical native BLER, causal deficit-weighted J; own unpruned rawPFQ90% dequeue floor',
        conditions='fresh paid offline teacher counts/exact min-layer/activity only; no teacher positions/J/service/group in solver',
        baseline='previous quota_exp fast frontier; same complete query/value observation hook, final maximum exact observed J',
        targets='interleave3 lowest E/D and3 highest weight*(D-E), deduplicate at most6; causal positive-dequeue quota UEs only; exclude saturated',
        directed_pool='up to4 disjoint target-acquisition swaps/UE; donor penalties0/.25/1; each intermediate keeps exact counts/layers/activity; combine first/last anchor plans across disjoint cells;<=256 proposals/root',
        target_selector='rotate causal targets by iteration, reserve2 or4 exact candidates for positive projected target reliability gains, proxyJ>=95%root; retain2 global service slots and fill with J',
        budget='212 exact complete actions includinginitialPFQ;56 roots;16 seeds;8 perbatch; no additional online teacher',
        sparse='alternate odd iterations use directed pool only if nonempty, even use wide+directed; fallback to wide if empty; this changes search, not an equivalence optimization',
        fallback='keep own exact archive/root and original wide/service proposals; no guarantee of matching separately run baseline after reallocating budget',
        diagnostics='after validation selection AND development:up to2 failing states/phase by10*Jshortfall+P5shortfall+weakshortfall;baseline+quality-selected method<=8 roots;score all unseen wide+directed candidates<=8192/root;replay root and MMSE separately charged;independent selected audits',
        diagnostic_limits='finite one-step pool excludes main visited actions and seed rematching;labels only,no policy change;iteration0 synthetic shortlist,not claimed as actual continuation',
        validation_seeds=[44,45],validation_sources=['raw_global_deficit','pfq'],validation_ttis=[4001,8001,18000],
        evaluation_seeds=[46,47,48],evaluation_sources=['pfq'],evaluation_ttis=[3001,4001,5000],historical_development=True,
        engineering='2 training states41-43 x6 decoders including disabled-mechanism equivalence;no tuning',
        final_seeds_not_read=[311,312,313,314,315],repeats=3,
        gates='each J>=99.5%full; no newly unallocatedUE;full-positive P5/bottom20%>=98%;complete execution mean<=80%full',
        selection='prior quality/service shortfall rule on validation, baseline eligible;separate cost-feasible route;no development reselection',
        timing='numeric input/PFQ,freshPHY/MMSE bank,condition compression,all pools/proxies/frontiers/exact scoring/final selection/context cleanup;common memo event copying included',
        exclusions='teacher preparation,compilation/library/archiveIO,first immutableTBS setup,digests/independent audits/separate paid recall diagnosis',
        work='all exact scores,2448 unique MMSE,native requests,direct proxy/seed/directed gamma reads,directed attempts/compositions and selected slots',
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
    problem_from_raw_row(erows[0]);physical_library();delta_library();directed_library()
    with patch('training.stageb_lns.weak_service.ExactMemo',RecordedMemo),patch('training.stageb_lns.directed_reorder.ExactMemo',RecordedMemo):
        engineering(erows,out)
        validation,files=read_rows((44,45));validation=[r for r in validation if r['tti'] in protocol['validation_ttis']]
        write_json(out/'validation_input_manifest.json',dict(files=files,states=[identity(r) for r in validation]))
        refs=teachers(validation,out,'validation');state=benchmark(validation,refs,out,'validation')
        selection=choose(state);write_json(out/'selection.json',selection);print('DIRECTED SELECTION FROZEN',selection,flush=True)
        evaluation,files=read_rows((46,47,48),evaluation=True);evaluation=[r for r in evaluation if r['tti'] in protocol['evaluation_ttis']]
        write_json(out/'evaluation_input_manifest.json',dict(files=files,states=[identity(r) for r in evaluation]))
        refs=teachers(evaluation,out,'evaluation');benchmark(evaluation,refs,out,'evaluation')
    formal_elapsed=time.perf_counter()-start
    from .directed_reorder_diagnosis import audit
    audit(out,CONFIGS)
    assert sha(REGISTRY)==protocol['registry_sha256'] and sha(ROOT/'training/stageb_lns/physics_kernel.cpp')==protocol['physics_source_sha256']
    assert all(sha(Path(__file__).with_name(name))==digest for name,digest in json.loads((out/'source_manifest.json').read_text()).items())
    diag=json.loads((out/'recall_diagnosis/summary.json').read_text())
    write_json(out/'completion.json',dict(completed=True,formal_decisions=378,formal_independent_finals=126,
        offline_teacher_decisions=23,engineering_decoder_decisions=12,validation_states=12,development_states=9,
        diagnostic_roots=diag['roots'],diagnostic_exact_scores=diag['exact_scores'],diagnostic_independent_audits=diag['independent_audits'],
        training_updates=0,registry_and_physics_unchanged=True,executed_sources_unchanged=True,native_trajectories=0,
        final_seeds_read=False,formal_elapsed_s=formal_elapsed,elapsed_s=time.perf_counter()-start))
    print('DIRECTED REORDER COMPLETE',out,flush=True)


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
    for arm in ['baseline',*CONFIGS]:
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

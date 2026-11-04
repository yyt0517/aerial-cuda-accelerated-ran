"""Frozen top2/top4 mechanism comparison, then one own-state training round."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np

from .shortlist_trajectory import Config,solve,collect_own,service_metrics,state_key
from .own_trajectory import fine_tune,generate as legacy_generate,DEEP
from .bounded_probes import Predictor,csv_write,seed_mean,probe_library
from .physics import ROOT,ReferencePhy,EXPECTED,DEQUEUE
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .ranked_dpls import solve_ranked
from .ranked_dpls_campaign import read_rows,identity
from .experiment import write_json,sha


PREVIOUS=ROOT/'deliverables/three_bounded_ai_probes_20261102_v2'
REGISTRY=ROOT/'training/stageb_hgraph/current_algorithm.json'


def run(row,method,initial,trained=None):
    start=time.perf_counter();p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs'])
    if method=='full':a,v,r=solve_ranked(p,draft)
    elif method=='legacy_top1':a,v,r=legacy_generate(p,draft,DEEP,initial)
    else:
        prefix,k=method.rsplit('_',1);k=int(k)
        config=Config(k,deduplicate=prefix!='repeat')
        model=None if prefix=='heuristic' else (trained if prefix=='trained' else initial)
        a,v,r=solve(p,draft,config,model)
    r['score']=float(v[:,EXPECTED]@p.weights);r['e2e_ms']=(time.perf_counter()-start)*1000
    r.pop('trace',None);r.pop('history',None)
    r.update(state_id=identity(row),seed=row['seed'],arm=method,family=row['family'],tti=row['tti'])
    return a,v,r


def selection_key(records):
    quality=all(r['ratio']>=.995 for r in records)
    coverage=all(r['newly_unserved_ues']==0 and r['positive_p5_ratio']>=.98 and r['positive_weak_ratio']>=.98 for r in records)
    q_short=float(np.mean([max(.995-r['ratio'],0) for r in records]))
    c_short=float(np.mean([max(.98-r['positive_p5_ratio'],0)+max(.98-r['positive_weak_ratio'],0)+
        r['newly_unserved_ues']/max(r['reference_positive_ues'],1) for r in records]))
    cost=seed_mean(records,'e2e_ms')/seed_mean(records,'reference_ms')
    return (int(quality and coverage and cost<=.8),-q_short,-c_short,seed_mean(records,'ratio'))


def metric(records):
    return dict(quality=all(r['ratio']>=.995 for r in records),
        coverage=all(r['newly_unserved_ues']==0 and r['positive_p5_ratio']>=.98 and r['positive_weak_ratio']>=.98 for r in records),
        mean_ratio=seed_mean(records,'ratio'),mean_time_ratio=seed_mean(records,'e2e_ms')/seed_mean(records,'reference_ms'),
        mean_positive_p5=seed_mean(records,'positive_p5_ratio'),mean_newly_unserved=seed_mean(records,'newly_unserved_ues'))


def summarize(records):
    full=[r for r in records if r['arm']=='full'];result=[]
    for arm in dict.fromkeys(r['arm'] for r in records):
        rr=[r for r in records if r['arm']==arm];m=metric(rr)
        cost=seed_mean(rr,'e2e_ms')/seed_mean(full,'e2e_ms')
        result.append(dict(arm=arm,mean_j=seed_mean(rr,'score'),mean_ms=seed_mean(rr,'e2e_ms'),time_ratio=cost,
            state_p95_ms=float(np.quantile([r['e2e_ms'] for r in rr],.95)),
            mean_relative_j_pct=100*(seed_mean(rr,'ratio')-1),worst_relative_j_pct=100*(min(r['ratio'] for r in rr)-1),
            mean_candidates=seed_mean(rr,'candidate_evaluations'),
            mean_feature_roots=seed_mean(rr,'feature_roots') if arm!='full' else None,
            mean_accepted_edits=seed_mean(rr,'accepted_edits') if arm not in ('full','legacy_top1') else None,
            mean_repeated_exact=seed_mean(rr,'repeated_exact_candidates') if arm not in ('full','legacy_top1') else None,
            mean_newly_unserved=seed_mean(rr,'newly_unserved_ues'),
            mean_positive_p5=seed_mean(rr,'positive_p5_ratio'),worst_positive_p5=min(r['positive_p5_ratio'] for r in rr),
            mean_positive_weak=seed_mean(rr,'positive_weak_ratio'),mean_ue_drops=seed_mean(rr,'ue_service_drops_gt2pct'),
            quality_pass=m['quality'],coverage_pass=m['coverage'],cost_pass=cost<=.8,
            joint_pass=m['quality'] and m['coverage'] and cost<=.8))
    return result


def benchmark(rows,methods,initial,trained,out,phase):
    records=[];actions={};values={};audits=[]
    for n,row in enumerate(rows):
        for repeat in range(3):
            shift=(n+repeat)%len(methods)
            for method in methods[shift:]+methods[:shift]:
                a,v,r=run(row,method,initial,trained);r['repeat']=repeat;records.append(r)
                key=f'{identity(row)}|{method}'
                if key in actions:np.testing.assert_array_equal(actions[key],a)
                else:actions[key]=a;values[key]=v
        print('BENCH_SHORTLIST',phase,identity(row),flush=True)
    state=[]
    for row in rows:
        p=problem_from_raw_row(row);sid=identity(row);ref=values[f'{sid}|full']
        full=[r for r in records if r['state_id']==sid and r['arm']=='full']
        full_ms=float(np.median([r['e2e_ms'] for r in full]))
        for method in methods:
            key=f'{sid}|{method}';a=actions[key];v=values[key]
            independent=ReferencePhy(p).evaluate(a)[0]
            np.testing.assert_allclose(v,independent,rtol=2.e-9,atol=2.e-6)
            rr=[r for r in records if r['state_id']==sid and r['arm']==method]
            assert independent[:,DEQUEUE].sum()>=rr[0]['floor']
            r={k:v for k,v in rr[0].items() if k!='repeat'};r['e2e_ms']=float(np.median([r['e2e_ms'] for r in rr]))
            r.update(ratio=r['score']/float(ref[:,EXPECTED]@p.weights),time_ratio=r['e2e_ms']/full_ms,reference_ms=full_ms,
                **service_metrics(v,ref));state.append(r)
            audits.append(dict(state_id=sid,arm=method,max_error=float(abs(v-independent).max())))
    csv_write(out/f'{phase}_measurements.csv',records);csv_write(out/f'{phase}_per_state.csv',state)
    np.savez_compressed(out/f'{phase}_actions.npz',**actions);np.savez_compressed(out/f'{phase}_values.npz',**values)
    summary=summarize(state);csv_write(out/f'{phase}_summary.csv',summary)
    write_json(out/f'{phase}_checks.json',dict(repeats_action_identical=True,common_floor=True,independent_phy=audits))
    return state,values,summary


def freeze(out):
    out.mkdir(parents=True,exist_ok=False);previous=json.loads((PREVIOUS/'selection.json').read_text())
    protocol=dict(date='2026-11-02',version=1,ctbs=False,headroom_masks=False,ppo=False,
        objective='canonical native-BLER deficit-weighted J; own unpruned PFQ90% dequeue floor',
        initial=previous['scalar'],initial_context='original scalar ranker; previous shallow-only finetuning retained as history',
        top_k_grid=[2,4],max_feature_roots=56,max_candidates=106,
        root_cache='candidate/features/ranking cached at same root for every arm; all refreshed after accepting edit',
        rejection='failed exact candidate indices skipped only at the same decision/root; caches never cross decisions/CSI',
        threshold='every accepted edit >1B, matching scalar generator; full DPLS retains native1/16B thresholds',
        numerical_scores='full original-neighborhood reduction layout/row indices, entirely inside timer',
        training_seeds=[41,42,43],validation_seeds=[44,45],evaluation_seeds=[46,47,48],
        train_validation_sources=['raw_global_deficit','pfq'],train_validation_ttis=[4001,8001,18000],
        evaluation_sources=['pfq'],evaluation_ttis=[3001,4001,5000],historical_development=True,
        final_seeds_not_read=[311,312,313,314,315],
        phase_a_methods=['full','legacy_top1','heuristic_2','initial_2','heuristic_4','initial_4','repeat_2','repeat_4'],
        selection='validation only; joint first, J shortfall, positive-service coverage shortfall, mean J ratio; smaller K on ties',
        training_entry='after engineering checks, one bounded diagnostic training round on selected mechanism, even if final quality gate is not passed',
        training_rounds=1,training_epochs=20,checkpoints=[0,5,10,20],max_own_states_per_decision=8,
        visited_state_key='stateID/root SHA/failed-mask SHA; correlated rejection states not independent physical scenes',
        teacher='independent post-decision full-neighborhood gain; availability excludes student failed-before indices',
        anchors=str(PREVIOUS/'training_groups.pt'),anchor_sha256=sha(PREVIOUS/'training_groups.pt'),
        training='frozen scalar normalization, warm weights, fresh AdamW lr0.0005; same fine_tune as own_trajectory',
        quality_gate='all states J>=99.5% full',
        coverage_gate='all states no newly unserved UE; P5 and bottom20% service among full expected>1B UEs each >=98% reference',
        cost_gate='seed-equal median complete time<=80% full',
        learning_gate='changed weights, positive J gain every seed and at least6/9 state wins vs initial same K; joint gate required',
        evaluation_methods=['full','legacy_top1','heuristic_2','initial_2','heuristic_4','initial_4','repeat_2','repeat_4','trained_2','trained_4'],
        repeats=3,timing='raw numerical input/PFQ decode through fresh PHY/caches, candidates/features/network, validation, scoring and cleanup',
        exclusions='archive/model/library IO and first immutable TBS setup; offline teacher/training separately charged',
        registry_sha256=sha(REGISTRY),platform=platform.platform())
    write_json(out/'protocol.json',protocol)
    sources={}
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.cpp','.so','.rst'):
            target=out/'source'/p.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target);sources[p.name]=sha(p)
    write_json(out/'source_manifest.json',sources)
    return protocol


def main(output):
    import torch
    torch.set_num_threads(1);out=Path(output).resolve();protocol=freeze(out);start=time.perf_counter()
    initial=Predictor(protocol['initial']['checkpoint']);assert sha(initial.path)==protocol['initial']['sha256']
    training,train_files=read_rows((41,42,43));validation,val_files=read_rows((44,45))
    training=[r for r in training if r['tti'] in protocol['train_validation_ttis']]
    validation=[r for r in validation if r['tti'] in protocol['train_validation_ttis']]
    write_json(out/'input_manifest.json',dict(training=train_files,validation=val_files,
        training_states=[identity(r) for r in training],validation_states=[identity(r) for r in validation]))
    problem_from_raw_row(training[0]);probe_library()
    val_rows,val_values,summary=benchmark(validation,protocol['phase_a_methods'],initial,None,out,'phase_a')
    options=[]
    for k in (2,4):
        rows=[r for r in val_rows if r['arm']==f'initial_{k}']
        options.append((selection_key(rows),k,metric(rows)))
    selected=max(options,key=lambda x:(x[0],-x[1]));k=selected[1];config=Config(k)
    write_json(out/'mechanism_selection.json',dict(top_k=k,metric=selected[2],
        alternatives=[dict(top_k=v[1],metric=v[2]) for v in options],development_not_read_in_this_run=True))
    print('MECHANISM',k,selected[2],flush=True)
    own,visits=collect_own(training,initial,config,out)
    anchors=torch.load(protocol['anchors'],weights_only=False)
    dataset={state_key(g['state_id'],g['root'],g.get('failed_before',np.empty(0,np.int64))):g for g in anchors}
    before=set(dataset)
    for g in own:dataset.setdefault(state_key(g['state_id'],g['root'],g['failed_before']),g)
    assert all(g['seed'] in (41,42,43) for g in dataset.values())
    def validate(model):
        records=[]
        refs={r['state_id']:r for r in val_rows if r['arm']=='full'}
        for row in validation:
            a,v,r=run(row,f'initial_{k}',model);ref=refs[identity(row)]
            r.update(ratio=r['score']/ref['score'],time_ratio=r['e2e_ms']/ref['e2e_ms'],reference_ms=ref['e2e_ms'],
                **service_metrics(v,val_values[f'{identity(row)}|full']));records.append(r)
        return selection_key(records),metric(records)
    trained,record=fine_tune(list(dataset.values()),initial,out,1,validate)
    changed=any(not np.array_equal(getattr(initial,v),getattr(trained,v)) for v in ('w1','b1','w2','b2','w3','b3'))
    selection=dict(top_k=k,checkpoint=str(trained.path),sha256=sha(trained.path),initial_sha256=sha(initial.path),
        weights_changed=changed,training_groups=len(dataset),collected_groups=len(own),new_states=len(set(dataset)-before),
        training_selection=record,development_not_read_in_this_run=True,historical_exposure_retained=True)
    write_json(out/'selection.json',selection);csv_write(out/'own_visits.csv',visits)
    torch.save(list(dataset.values()),out/'combined_training.pt')
    print('SHORTLIST MODEL FROZEN',selection,flush=True)
    evaluation,files=read_rows((46,47,48),evaluation=True)
    evaluation=[r for r in evaluation if r['tti'] in protocol['evaluation_ttis']]
    write_json(out/'evaluation_input_manifest.json',dict(files=files,states=[identity(r) for r in evaluation]))
    final,values,summary=benchmark(evaluation,protocol['evaluation_methods'],initial,trained,out,'evaluation')
    paired=[];index={(r['state_id'],r['arm']):r for r in final}
    for row in evaluation:
        for k in (2,4):
            a=index[identity(row),f'trained_{k}'];b=index[identity(row),f'initial_{k}']
            paired.append(dict(state_id=identity(row),seed=row['seed'],top_k=k,j_delta=a['score']-b['score'],
                j_relative_pct=100*(a['score']/b['score']-1),ms_delta=a['e2e_ms']-b['e2e_ms'],
                positive_p5_delta=a['positive_p5_ratio']-b['positive_p5_ratio']))
    csv_write(out/'paired_learning.csv',paired)
    assert sha(REGISTRY)==protocol['registry_sha256']
    write_json(out/'completion.json',dict(completed=True,validation_states=len(validation),evaluation_states=len(evaluation),
        phase_a_decisions=len(validation)*len(protocol['phase_a_methods'])*3,
        evaluation_decisions=len(evaluation)*len(protocol['evaluation_methods'])*3,
        elapsed_s=time.perf_counter()-start,native_trajectories=0,final_seeds_read=False,registry_unchanged=True))
    print('SHORTLIST COMPLETE',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);main(parser.parse_args().output)

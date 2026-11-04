"""Compare shallow/deep generation, then train only on student-visited roots."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np

from .own_trajectory import (SHALLOW,DEEP,generate,coverage,collect_own,fine_tune,group_key)
from .bounded_probes import Predictor,csv_write,seed_mean,probe_library
from .physics import ROOT,ReferencePhy,EXPECTED,DEQUEUE,COUNT
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .ranked_dpls import solve_ranked
from .ranked_dpls_campaign import read_rows,identity
from .experiment import write_json,sha


PREVIOUS=ROOT/'deliverables/three_bounded_ai_probes_20261102_v2'
REGISTRY=ROOT/'training/stageb_hgraph/current_algorithm.json'


def run(row,method,initial,trained=None):
    start=time.perf_counter();p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs'])
    if method=='full':a,v,r=solve_ranked(p,draft)
    else:
        prefix,shape=method.split('_');geometry=SHALLOW if shape=='shallow' else DEEP
        model=None if prefix=='heuristic' else (trained if prefix=='trained' else initial)
        a,v,r=generate(p,draft,geometry,model)
    r['score']=float(v[:,EXPECTED]@p.weights)
    r['e2e_ms']=(time.perf_counter()-start)*1000;r.pop('history',None)
    r.update(state_id=identity(row),seed=row['seed'],family=row['family'],tti=row['tti'],arm=method)
    return a,v,r


def benchmark(rows,methods,initial,trained,out,phase,repeats=3):
    records=[];actions={};values={};audits=[]
    for state_index,row in enumerate(rows):
        for repeat in range(repeats):
            shift=(state_index+repeat)%len(methods);order=methods[shift:]+methods[:shift]
            for method in order:
                a,v,r=run(row,method,initial,trained);r['repeat']=repeat;records.append(r)
                key=(identity(row),method)
                if key in actions:np.testing.assert_array_equal(actions[key],a)
                else:actions[key]=a;values[key]=v
        print('BENCH',phase,identity(row),flush=True)
    # All independent audits happen after the serial timing phase.
    medians=[]
    for row in rows:
        p=problem_from_raw_row(row);ref=values[identity(row),'full']
        ref_j=float(ref[:,EXPECTED]@p.weights)
        for method in methods:
            key=(identity(row),method);a=actions[key];v=values[key]
            independent=ReferencePhy(p).evaluate(a)[0]
            np.testing.assert_allclose(v,independent,rtol=2.e-9,atol=2.e-6)
            rr=[r for r in records if r['state_id']==identity(row) and r['arm']==method]
            assert independent[:,DEQUEUE].sum()>=rr[0]['floor']
            audits.append(dict(state_id=identity(row),arm=method,max_error=float(np.max(abs(v-independent)))))
            record={k:v for k,v in rr[0].items() if k!='repeat'}
            record.update(e2e_ms=float(np.median([r['e2e_ms'] for r in rr])),ratio=rr[0]['score']/ref_j,
                **coverage(v,ref))
            medians.append(record)
    csv_write(out/f'{phase}_measurements.csv',records);csv_write(out/f'{phase}_per_state.csv',medians)
    np.savez_compressed(out/f'{phase}_actions.npz',**{f'{s}|{m}':a for (s,m),a in actions.items()})
    np.savez_compressed(out/f'{phase}_values.npz',**{f'{s}|{m}':v for (s,m),v in values.items()})
    write_json(out/f'{phase}_checks.json',dict(repeats_action_identical=True,independent_phy=audits,
        common_floor=True,ctbs=False,full_search_in_generator=False))
    summary=summarize(medians);csv_write(out/f'{phase}_summary.csv',summary)
    return medians,values,summary


def summarize(records):
    full=[r for r in records if r['arm']=='full'];summary=[]
    for arm in dict.fromkeys(r['arm'] for r in records):
        rr=[r for r in records if r['arm']==arm]
        quality=all(r['ratio']>=.995 for r in rr)
        cover=all(r['newly_unserved_ues']==0 and r['expected_p5_ratio']>=.98 for r in rr)
        cost=seed_mean(rr,'e2e_ms')/seed_mean(full,'e2e_ms')
        summary.append(dict(arm=arm,mean_j=seed_mean(rr,'score'),mean_ms=seed_mean(rr,'e2e_ms'),time_ratio=cost,
            state_p95_ms=float(np.quantile([r['e2e_ms'] for r in rr],.95)),
            mean_relative_j_pct=100*(seed_mean(rr,'ratio')-1),worst_relative_j_pct=100*(min(r['ratio'] for r in rr)-1),
            mean_candidates=seed_mean(rr,'candidate_evaluations'),mean_feature_roots=seed_mean(rr,'feature_roots') if arm!='full' else None,
            mean_unique_roots=seed_mean(rr,'unique_feature_roots') if arm!='full' else None,
            mean_best_path_edits=seed_mean(rr,'best_path_edits') if arm!='full' else None,
            mean_stagnant_roots=seed_mean(rr,'stagnant_roots') if arm!='full' else None,
            mean_served_ues=seed_mean(rr,'served_ues'),mean_newly_unserved_ues=seed_mean(rr,'newly_unserved_ues'),
            mean_expected_p5_ratio=seed_mean(rr,'expected_p5_ratio'),worst_expected_p5_ratio=min(r['expected_p5_ratio'] for r in rr),
            mean_weak_service_ratio=seed_mean(rr,'weak_service_ratio'),mean_ue_drops_gt2pct=seed_mean(rr,'ue_service_drops_gt2pct'),
            quality_pass=quality,coverage_pass=cover,cost_pass=cost<=.8,joint_pass=quality and cover and cost<=.8))
    return summary


def selection_key(records,reference):
    rows=[]
    for r in records:
        ref=reference[r['state_id']]
        rows.append(dict(r,ratio=r['score']/ref['score']))
    q=all(r['ratio']>=.995 for r in rows)
    c=all(r['newly_unserved_ues']==0 and r['expected_p5_ratio']>=.98 for r in rows)
    time_ratio=seed_mean(rows,'e2e_ms')/seed_mean(list(reference.values()),'e2e_ms')
    shortfall=float(np.mean([max(.995-r['ratio'],0) for r in rows]))
    coverage_shortfall=float(np.mean([max(.98-r['expected_p5_ratio'],0)+r['newly_unserved_ues']/36 for r in rows]))
    # Fixed quality/coverage rules, then ratio. Timing breaks only exact ties.
    key=(int(q and c and time_ratio<=.8),-shortfall,-coverage_shortfall,seed_mean(rows,'ratio'))
    return key,dict(quality=q,coverage=c,cost=time_ratio<=.8,time_ratio=time_ratio,
        mean_ratio=seed_mean(rows,'ratio'),quality_shortfall=shortfall,coverage_shortfall=coverage_shortfall)


def teacher_depth_reference(rows,values,out,phase):
    result=[];work=[]
    for row in rows:
        p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs']);accepted=0;prefix=draft.copy()
        def observe(info):
            nonlocal prefix,accepted
            if info['chosen_index']>=0:
                accepted+=1
                if accepted<=32:prefix=info['actions'][info['chosen_index']].copy()
        _,_,r=solve_ranked(p,draft,observer=observe)
        v=ReferencePhy(p).evaluate(prefix)[0];ref=values[identity(row),'full']
        result.append(dict(state_id=identity(row),seed=row['seed'],score=float(v[:,EXPECTED]@p.weights),
            ratio=float(v[:,EXPECTED]@p.weights/(ref[:,EXPECTED]@p.weights)),
            accepted_edits=min(32,accepted),**coverage(v,ref)))
        work.append(dict(state_id=identity(row),candidate_evaluations=r['candidate_evaluations'],
            ue_evaluations=r['ue_evaluations'],mmse_misses=r['mmse_misses']))
    csv_write(out/f'{phase}_teacher32.csv',result)
    write_json(out/f'{phase}_teacher32_work.json',dict(teacher_work=work,
        finite_greedy_prefix_only=True,global_upper_bound=False,online_capability=False))
    return result


def freeze(out):
    out.mkdir(parents=True,exist_ok=False)
    previous=json.loads((PREVIOUS/'selection.json').read_text())
    protocol=dict(date='2026-11-02',version=1,ctbs=False,headroom_masks=False,ppo=False,
        objective='canonical native-BLER deficit-weighted J; own raw PFQ 90% floor',
        geometries=dict(shallow=SHALLOW.__dict__,deep=DEEP.__dict__),
        resource_comparison='same max56 feature roots/max106 physical scores; actual usage and time reported, not equal wall time',
        training_seeds=[41,42,43],validation_seeds=[44,45],evaluation_seeds=[46,47,48],
        train_validation_sources=['raw_global_deficit','pfq'],train_validation_ttis=[4001,8001,18000],
        evaluation_sources=['pfq'],evaluation_ttis=[3001,4001,5000],
        historical_development=True,final_seeds_not_read=[311,312,313,314,315],
        prior_scalar=previous['scalar'],prior_selection_sha256=sha(PREVIOUS/'selection.json'),
        phase_a_methods=['full','heuristic_shallow','initial_shallow','heuristic_deep','initial_deep'],
        geometry_selection='validation only: joint first, mean J shortfall, coverage shortfall, mean J ratio; prefer deep on exact ties',
        own_rounds=2,max_unique_roots_per_state_per_round=8,
        own_root_sampling='deduplicate actual student roots, at most8 evenly spaced first visits after decision completed',
        anchor_groups=str(PREVIOUS/'training_groups.pt'),anchor_sha256=sha(PREVIOUS/'training_groups.pt'),
        teacher='separate post-decision full one-PRG neighborhood; immediate exact gain, no suffix/MPC oracle',
        training='warm scalar weights and frozen normalization; fresh AdamW lr0.0005 each round; 20epochs; CE temp0.15 +0.1 Huber',
        sampling='up to32 exact leaders +32 current-model leaders +32 heuristic leaders +32 uniform per eligible list',
        checkpoints=[0,5,10,20],model_selection='full validation generator output under the selected geometry; same quality/coverage key; earliest tie',
        evaluation_methods=['full','heuristic_shallow','initial_shallow','heuristic_deep','initial_deep','trained_shallow','trained_deep'],
        repeats=3,quality_gate='each state J>=99.5% full',
        coverage_gate='each state no newly unserved UE vs full and expected-service P5>=98% full; other UE diagnostics reported',
        cost_gate='seed-equal median complete time<=80% full',
        learning_gate='changed nonzero weights, positive J gain in every seed vs initial same geometry, at least6/9 state wins, quality+coverage+cost',
        timing='raw numeric state/PFQ decode through fresh PHY/feature context, generation, inference, verification and cleanup',
        exclusions='archive/model/library IO and first immutable TBS table; teacher labels/training separately charged',
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
    initial=Predictor(protocol['prior_scalar']['checkpoint'])
    assert sha(initial.path)==protocol['prior_scalar']['sha256']
    training,train_files=read_rows((41,42,43));validation,val_files=read_rows((44,45))
    training=[r for r in training if r['tti'] in protocol['train_validation_ttis']]
    validation=[r for r in validation if r['tti'] in protocol['train_validation_ttis']]
    write_json(out/'input_manifest.json',dict(training=train_files,validation=val_files,
        training_states=[identity(r) for r in training],validation_states=[identity(r) for r in validation]))
    problem_from_raw_row(training[0]);probe_library()
    phase_a,val_values,val_summary=benchmark(validation,protocol['phase_a_methods'],initial,None,out,'phase_a')
    refs={r['state_id']:r for r in phase_a if r['arm']=='full'}
    candidates=[]
    for name in ('shallow','deep'):
        rr=[r for r in phase_a if r['arm']=='initial_'+name]
        key,metric=selection_key(rr,refs);candidates.append((key,name,metric))
    chosen=max(candidates,key=lambda v:(v[0],v[1]=='deep'))
    geometry=DEEP if chosen[1]=='deep' else SHALLOW
    write_json(out/'geometry_selection.json',dict(geometry=chosen[1],metric=chosen[2],
        alternatives=[dict(geometry=v[1],metric=v[2]) for v in candidates],development_results_not_read=True))
    print('GEOMETRY',chosen[1],chosen[2],flush=True)
    teacher_depth_reference(validation,val_values,out,'validation')
    anchors=torch.load(protocol['anchor_groups'],weights_only=False)
    dataset={group_key(g):g for g in anchors};seen=set(dataset);history=[];visits=[];current=initial
    own_keys=set()
    def validate(model):
        records=[]
        for row in validation:
            a,v,r=run(row,'initial_'+chosen[1],model)
            r.update(coverage(v,val_values[identity(row),'full']));records.append(r)
        return selection_key(records,refs)
    for round_index in (1,2):
        groups,own_visits=collect_own(training,current,geometry,out,round_index)
        unique_keys={group_key(g) for g in groups};novel=len(unique_keys-seen)
        for g in groups:dataset.setdefault(group_key(g),g)
        seen.update(unique_keys);own_keys.update(unique_keys);visits.extend(own_visits)
        current,selected=fine_tune(list(dataset.values()),current,out,round_index,validate)
        history.append(dict(round=round_index,collected_groups=len(groups),new_roots_vs_existing=novel,
            unique_own_roots=len(own_keys),combined_groups=len(dataset),selection=selected))
    selection=dict(geometry=chosen[1],checkpoint=str(current.path),sha256=sha(current.path),
        initial_sha256=sha(initial.path),weights_changed=any(not np.array_equal(getattr(current,k),getattr(initial,k))
            for k in ('w1','b1','w2','b2','w3','b3')),rounds=history,
        evaluation_not_read_in_this_run=True,historical_exposure_retained=True)
    write_json(out/'selection.json',selection);csv_write(out/'own_visits.csv',visits)
    torch.save(list(dataset.values()),out/'combined_training.pt')
    print('MODEL FROZEN',selection,flush=True)
    evaluation,eval_files=read_rows((46,47,48),evaluation=True)
    evaluation=[r for r in evaluation if r['tti'] in protocol['evaluation_ttis']]
    write_json(out/'evaluation_input_manifest.json',dict(files=eval_files,states=[identity(r) for r in evaluation]))
    final,eval_values,summary=benchmark(evaluation,protocol['evaluation_methods'],initial,current,out,'evaluation')
    teacher_depth_reference(evaluation,eval_values,out,'evaluation')
    paired=[]
    index={(r['state_id'],r['arm']):r for r in final}
    for row in evaluation:
        for shape in ('shallow','deep'):
            a=index[identity(row),'trained_'+shape];b=index[identity(row),'initial_'+shape]
            paired.append(dict(state_id=identity(row),seed=row['seed'],geometry=shape,
                j_delta=a['score']-b['score'],j_relative_pct=100*(a['score']/b['score']-1),
                ms_delta=a['e2e_ms']-b['e2e_ms'],p5_ratio_delta=a['expected_p5_ratio']-b['expected_p5_ratio']))
    csv_write(out/'paired_learning.csv',paired)
    assert sha(REGISTRY)==protocol['registry_sha256']
    write_json(out/'completion.json',dict(completed=True,validation_states=len(validation),evaluation_states=len(evaluation),
        phase_a_decisions=len(validation)*len(protocol['phase_a_methods'])*3,
        evaluation_decisions=len(evaluation)*len(protocol['evaluation_methods'])*3,
        native_trajectories=0,final_seeds_read=False,registry_unchanged=True,elapsed_s=time.perf_counter()-start))
    print('COMPLETE',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    main(parser.parse_args().output)

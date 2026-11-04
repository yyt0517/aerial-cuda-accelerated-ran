"""Causal service-set proposal, finite-budget terminal repair and conditional probe."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import shutil
import time
import numpy as np

from .service_repair import RepairConfig,solve_service
from .service_sets import set_metrics
from .service_set_training import label_states,fit
from .selective_tb import SelectiveConfig,solve_selective,feature_library
from .selective_tb_campaign import select_arms as unused_previous_selector
from .shortlist_trajectory import Config,solve,service_metrics
from .shortlist_campaign import selection_key
from .bounded_probes import Predictor,probe_library,csv_write,seed_mean
from .physics import ROOT,ReferencePhy,EXPECTED,COUNT,DEQUEUE
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .ranked_dpls import solve_ranked
from .ranked_dpls_campaign import read_rows,identity
from .experiment import write_json,sha

REGISTRY=ROOT/'training/stageb_hgraph/current_algorithm.json'
OLD_MODEL=ROOT/'deliverables/shortlist_trajectory_20261102_v1/own_r1_e10.npz'
CONFIGS={
    'unprotected212':RepairConfig('none',False,0),
    'main180':RepairConfig('none',False,0,180),
    'heuristic_lock_repair32':RepairConfig('heuristic',True,32),
    'learned_lock_repair32':RepairConfig('learned',True,32),
    'heuristic_repair32':RepairConfig('heuristic',False,32),
    'matched_repair32':RepairConfig('matched',False,32),
    'learned_repair32':RepairConfig('learned',False,32),
    'learned_repair64':RepairConfig('learned',False,64),
}
NAMES=['full','legacy_ai212','unprotected212','main180','all_pfq_guard212',
       'heuristic_lock_repair32','learned_lock_repair32','heuristic_repair32','matched_repair32','learned_repair32','learned_repair64']


def run(row,arm,old_model,model,thresholds,heuristic_counts):
    start=time.perf_counter();p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs']);masks=None;main=None
    if arm=='full':a,v,r=solve_ranked(p,draft)
    elif arm=='legacy_ai212':a,v,r=solve(p,draft,Config(4,True,56,212),old_model)
    elif arm=='all_pfq_guard212':a,v,r,_=solve_selective(p,draft,SelectiveConfig())
    else:a,v,r,masks,main=solve_service(p,draft,CONFIGS[arm],model=model,thresholds=thresholds,heuristic_counts=heuristic_counts)
    r['score']=float(v[:,EXPECTED]@p.weights);r['e2e_ms']=(time.perf_counter()-start)*1000
    events=r.pop('repair_events',[])
    for k in ('trace','history'):r.pop(k,None)
    r.update(state_id=identity(row),seed=row['seed'],source=row['family'],tti=row['tti'],arm=arm)
    return a,v,r,masks,main,events


def summarize(records):
    full=[r for r in records if r['arm']=='full'];out=[]
    for arm in dict.fromkeys(r['arm'] for r in records):
        rr=[r for r in records if r['arm']==arm]
        q=all(r['ratio']>=.995 for r in rr)
        c=all(r['newly_unserved_ues']==0 and r['positive_p5_ratio']>=.98 and r['positive_weak_ratio']>=.98 for r in rr)
        cost=seed_mean(rr,'e2e_ms')/seed_mean(full,'e2e_ms') if full else None
        has_sets=rr[0].get('allocation_recall') is not None
        set_gate=(all(r['allocation_recall']>=.98 and r['effective_recall']>=.98 for r in rr)
            and seed_mean(rr,'size_ratio')<=.75) if has_sets else None
        out.append(dict(arm=arm,mean_j=seed_mean(rr,'score'),mean_ms=seed_mean(rr,'e2e_ms'),time_ratio=cost,
            mean_relative_j_pct=100*(seed_mean(rr,'ratio')-1),worst_relative_j_pct=100*(min(r['ratio'] for r in rr)-1),
            mean_actual_phy=seed_mean(rr,'candidate_evaluations'),mean_feature_roots=seed_mean(rr,'feature_roots') if arm!='full' else None,
            mean_repair_phy=seed_mean(rr,'repair_phy') if arm in CONFIGS or arm.startswith('teacher_') else None,
            mean_newly_unserved=seed_mean(rr,'newly_unserved_ues'),mean_positive_p5=seed_mean(rr,'positive_p5_ratio'),
            worst_positive_p5=min(r['positive_p5_ratio'] for r in rr),mean_positive_weak=seed_mean(rr,'positive_weak_ratio'),
            allocation_recall=seed_mean(rr,'allocation_recall') if has_sets else None,
            min_allocation_recall=min(r['allocation_recall'] for r in rr) if has_sets else None,
            effective_recall=seed_mean(rr,'effective_recall') if has_sets else None,
            mean_set_size=seed_mean(rr,'allocation_size') if has_sets else None,mean_set_size_ratio=seed_mean(rr,'size_ratio') if has_sets else None,
            set_gate=set_gate,quality_pass=q,coverage_pass=c,cost_pass=cost<=.8 if cost is not None else None,
            joint_pass=q and c and cost<=.8 if cost is not None else None))
    return out


def benchmark(rows,old_model,model,thresholds,heuristic_counts,out,phase):
    raw=[];actions={};values={};sets={};mains={};events=[]
    for n,row in enumerate(rows):
        for repeat in range(3):
            shift=(n+repeat)%len(NAMES)
            for arm in NAMES[shift:]+NAMES[:shift]:
                a,v,r,m,main,ee=run(row,arm,old_model,model,thresholds,heuristic_counts);r['repeat']=repeat;raw.append(r)
                key=f'{identity(row)}|{arm}'
                if key in actions:
                    np.testing.assert_array_equal(actions[key],a);np.testing.assert_array_equal(values[key],v)
                    if m is not None:np.testing.assert_array_equal(sets[key],m);np.testing.assert_array_equal(mains[key],main[0])
                else:
                    actions[key]=a;values[key]=v
                    if m is not None:sets[key]=m;mains[key]=main[0]
                    events.extend(dict(state_id=identity(row),arm=arm,**event) for event in ee)
        print('SERVICE_BENCH',phase,identity(row),flush=True)
    state=[];audits=[]
    for row in rows:
        p=problem_from_raw_row(row);sid=identity(row);reference=values[f'{sid}|full']
        target=np.stack([reference[:,COUNT]>0,reference[:,EXPECTED]>1],1);eligible=p.legal.any((0,1))
        full_ms=float(np.median([r['e2e_ms'] for r in raw if r['state_id']==sid and r['arm']=='full']))
        for arm in ('heuristic_repair32','matched_repair32','learned_repair32'):
            np.testing.assert_array_equal(mains[f'{sid}|{arm}'],actions[f'{sid}|main180'])
        learned=sets[f'{sid}|learned_repair32'];matched=sets[f'{sid}|matched_repair32']
        for cell in range(p.C):np.testing.assert_array_equal(learned[p.serving==cell].sum(0),matched[p.serving==cell].sum(0))
        for arm in NAMES:
            key=f'{sid}|{arm}';v=values[key];independent=ReferencePhy(p).evaluate(actions[key])[0]
            np.testing.assert_allclose(v,independent,rtol=2.e-9,atol=2.e-6)
            rr=[r for r in raw if r['state_id']==sid and r['arm']==arm]
            r={k:v for k,v in rr[0].items() if k!='repeat'};r['e2e_ms']=float(np.median([v['e2e_ms'] for v in rr]))
            r.update(ratio=r['score']/float(reference[:,EXPECTED]@p.weights),reference_ms=full_ms,time_ratio=r['e2e_ms']/full_ms,
                **service_metrics(v,reference))
            assert independent[:,DEQUEUE].sum()>=r['floor']
            if arm!='full':assert r['candidate_evaluations']<=212 and r['feature_roots']<=56
            if arm in CONFIGS:
                assert not r['diagnostic'] and r['covered_prgs']==p.P
                assert r['repair_phy']<=CONFIGS[arm].repair_budget and r['score']+1.e-8>=r['quality_fence']
                if CONFIGS[arm].source!='none':r.update(set_metrics(sets[key],target,eligible))
            state.append(r);audits.append(dict(state_id=sid,arm=arm,max_error=float(abs(v-independent).max())))
    csv_write(out/f'{phase}_measurements.csv',raw);csv_write(out/f'{phase}_per_state.csv',state)
    csv_write(out/f'{phase}_summary.csv',summarize(state));write_json(out/f'{phase}_repair_events.json',events)
    np.savez_compressed(out/f'{phase}_actions.npz',**actions);np.savez_compressed(out/f'{phase}_values.npz',**values)
    np.savez_compressed(out/f'{phase}_sets.npz',**sets);np.savez_compressed(out/f'{phase}_main_actions.npz',**mains)
    write_json(out/f'{phase}_checks.json',dict(repeats_actions_values_sets_identical=True,unlocked_main_prefix_identical=True,
        matched_cardinalities=True,quality_fence_and_floor_valid=True,caps_valid=True,teacher_masks_never_online=True,independent_phy=audits))
    return state,actions,values


def choose(records):
    choices=[]
    for arm in NAMES[1:]:
        rr=[r for r in records if r['arm']==arm];prior=selection_key(rr)
        q=all(r['ratio']>=.995 for r in rr)
        c=all(r['newly_unserved_ues']==0 and r['positive_p5_ratio']>=.98 and r['positive_weak_ratio']>=.98 for r in rr)
        cost=seed_mean(rr,'e2e_ms')/seed_mean(rr,'reference_ms')
        choices.append(((int(q and c),*prior[1:]),arm,cost,seed_mean(rr,'e2e_ms')))
    quality=max(choices,key=lambda v:(v[0],-v[3]));feasible=[v for v in choices if v[2]<=.8]
    usable=max(feasible,key=lambda v:(v[0],-v[3])) if feasible else min(choices,key=lambda v:v[3])
    return dict(quality_best=quality[1],within_cost_best=usable[1],cost_feasible=bool(feasible),
        validation_only=True,development_not_read=True,conditional_masks_excluded=True,
        historical_exposure=True,rule='quality/service, J shortfall, service shortfall, ratio; complete-cost filter for within-cost point')


def conditional_probe(rows,groups,validation_state,out):
    by_id={g['state_id']:g for g in groups};refs={r['state_id']:r for r in validation_state if r['arm']=='full'}
    records=[];actions={};values={};checks=[];start_all=time.perf_counter()
    for row in rows:
        p=problem_from_raw_row(row);sid=identity(row);g=by_id[sid];masks=g['target']
        for lock in (False,True):
            arm='teacher_lock_repair32' if lock else 'teacher_repair32';start=time.perf_counter()
            a,v,r,m,main=solve_service(p,raw_pfq_draft(p,row['obs']),RepairConfig('diagnostic',lock,32),diagnostic_masks=masks)
            r['e2e_ms']=(time.perf_counter()-start)*1000;r.pop('repair_events')
            independent=ReferencePhy(p).evaluate(a)[0];np.testing.assert_allclose(v,independent,rtol=2.e-9,atol=2.e-6)
            r.update(state_id=sid,seed=row['seed'],arm=arm,ratio=r['score']/g['teacher_j'],
                reference_ms=refs[sid]['e2e_ms'],**service_metrics(v,g['teacher_values']),
                **set_metrics(m,g['target'],p.legal.any((0,1))))
            records.append(r);key=f'{sid}|{arm}';actions[key]=a;values[key]=v
            checks.append(dict(state_id=sid,arm=arm,max_error=float(abs(v-independent).max())))
    csv_write(out/'conditional_per_state.csv',records);csv_write(out/'conditional_summary.csv',summarize(records))
    np.savez_compressed(out/'conditional_actions.npz',**actions);np.savez_compressed(out/'conditional_values.npz',**values)
    write_json(out/'conditional_checks.json',dict(passed=True,validation_only=True,policy_selection_changed=False,
        diagnostic_decisions=len(records),independent_phy=checks,elapsed_s=time.perf_counter()-start_all,
        physical_scores=sum(r['candidate_evaluations'] for r in records),
        interpretation='teacher masks are an offline conditional mechanism probe, not a complete upper bound or deployable algorithm; times exclude separately charged teacher solve'))
    return records


def freeze(out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(date='2026-11-02',version=1,methods=NAMES,configs={k:asdict(v) for k,v in CONFIGS.items()},
        ctbs=False,headroom_masks=False,ppo=False,objective='canonical native BLER, causal deficit-weighted J, raw PFQ90% dequeue floor',
        training_seeds=[41,42,43],validation_seeds=[44,45],evaluation_seeds=[46,47,48],
        training_validation_sources=['raw_global_deficit','pfq'],training_validation_ttis=[4001,8001,18000],
        evaluation_sources=['pfq'],evaluation_ttis=[3001,4001,5000],historical_development=True,final_seeds_not_read=[311,312,313,314,315],
        teacher='one fresh full-DPLS terminal/decision on training and validation only, labels COUNT>0 and EXPECTED>1B; explicit offline physics ledger and independent audit',
        features='40 per-UE causal initial-PFQ/geometry/MCS/queue/weight/cell-rank/context features; one initial PHY is part of online212 cap',
        model='40->24->12->2 ReLU, sigmoid sets; training-only normalization; BCE +0.1 cardinality loss',
        training='one60epoch round, AdamW lr0.003 wd0.0001, seed20261102; checkpoints0/10/30/60',
        calibration='validation9 threshold pairs0.3/0.5/0.7; per-state allocation/effective recall>=98% and mean size<=75% eligible; otherwise predefined recall/size/precision loss',
        heuristic='calibrate allocated K=4/6/8 and effective K=2/4 per cell on same set objective; utility based on causal PFQ reliability/TBS/quality/weights',
        matched='same learned cardinalities per cell, heuristic UE identities; model/cardinality cost remains included',
        main='same incremental selective pool/proxy; >1B improving PFQ-feasible edits; optional fixed proposed-member count lock only, no all-PFQ P5 anchor',
        repair='total212 scores; reserve32 or64 new exact scores; max4 fair PRG passes,1 candidate/block; root-local phase rejection and decision-local exact cache',
        repair_goal='lexicographic proposed allocation coverage, proposed effective-positive count, P5/weak20% in1B bins, then J',
        repair_fence='every accepted terminal action J>=99.5% fixed main-stage terminal J and raw PFQ dequeue floor; not a99.5%-full guarantee',
        conditional='validation only, explicit diagnostic mode, teacher masks never enter formal online arms; single run per diagnostic, teacher preparation separately charged; not a complete upper bound',
        max_feature_roots=56,main_sweeps=5,repeats=3,
        timing='input/PFQ decode through fresh PHY, proposal features/network, caches/pools, exact verification and terminal repair/scoring/cleanup',
        exclusions='archive/model/library IO and first immutable TBS setup; all labels, training, conditional probes and independent audits separate',
        quality_gate='every state J>=99.5% full',coverage_gate='no newly unallocated UE vs full; full-positive UE P5/bottom20%>=98% reference',
        cost_gate='seed-equal complete per-state median mean<=80% full',
        learning_gate='changed model, set recall/compression gate,>=6/9 J wins and positive every-seed J delta vs size-matched control, plus final joint gate',
        old_model=str(OLD_MODEL),old_model_sha256=sha(OLD_MODEL),registry_sha256=sha(REGISTRY),platform=platform.platform())
    write_json(out/'protocol.json',protocol);manifest={}
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.cpp','.so','.rst'):
            target=out/'source'/p.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target);manifest[p.name]=sha(p)
    write_json(out/'source_manifest.json',manifest);return protocol


def main(output):
    import torch
    torch.set_num_threads(1);out=Path(output).resolve();protocol=freeze(out);started=time.perf_counter()
    training,files=read_rows((41,42,43));training=[r for r in training if r['tti'] in protocol['training_validation_ttis']]
    write_json(out/'training_input_manifest.json',dict(files=files,states=[identity(r) for r in training]))
    problem_from_raw_row(training[0]);probe_library();feature_library();train_groups=label_states(training,out,'training')
    validation,files=read_rows((44,45));validation=[r for r in validation if r['tti'] in protocol['training_validation_ttis']]
    write_json(out/'validation_input_manifest.json',dict(files=files,states=[identity(r) for r in validation]))
    val_groups=label_states(validation,out,'validation');model,thresholds,counts=fit(train_groups,val_groups,validation,out)
    old_model=Predictor(OLD_MODEL);state,actions,values=benchmark(validation,old_model,model,thresholds,counts,out,'validation')
    for g in val_groups:np.testing.assert_array_equal(actions[f'{g["state_id"]}|full'],g['teacher_action'])
    selected=choose(state);write_json(out/'selection.json',selected);print('SERVICE SELECTION FROZEN',selected,flush=True)
    conditional_probe(validation,val_groups,state,out)
    evaluation,files=read_rows((46,47,48),evaluation=True);evaluation=[r for r in evaluation if r['tti'] in protocol['evaluation_ttis']]
    write_json(out/'evaluation_input_manifest.json',dict(files=files,states=[identity(r) for r in evaluation]))
    benchmark(evaluation,old_model,model,thresholds,counts,out,'evaluation')
    assert sha(OLD_MODEL)==protocol['old_model_sha256'] and sha(REGISTRY)==protocol['registry_sha256']
    assert all(sha(Path(__file__).with_name(name))==digest for name,digest in json.loads((out/'source_manifest.json').read_text()).items())
    write_json(out/'completion.json',dict(completed=True,formal_decisions=(len(validation)+len(evaluation))*len(NAMES)*3,
        formal_independent_audits=(len(validation)+len(evaluation))*len(NAMES),conditional_decisions=len(validation)*2,
        training_states=len(training),validation_states=len(validation),development_states=len(evaluation),
        training_rounds=1,selected_epoch=json.loads((out/'model_selection.json').read_text())['epoch'],
        source_model_and_registry_unchanged=True,teacher_masks_never_formal_online=True,native_trajectories=0,final_seeds_read=False,
        elapsed_s=time.perf_counter()-started))
    print('SERVICE REPAIR COMPLETE',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);main(parser.parse_args().output)

"""First three-arm AI acceleration experiment, with CTBS removed throughout.

Training/selection use no-CTBS and PFQ-only histories. Historical development
evaluation uses PFQ-only histories and is unread until model/K selection ends.
No final seeds, native simulation, or active algorithm registry are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import numpy as np

from .physics import ROOT, NativePhy, ReferencePhy, EXPECTED, DEQUEUE
from .archive import enable_legacy_types, ARCHIVE
from .no_ctbs_inputs import detach_raw_phy_row, problem_from_raw_row, raw_pfq_draft
from .ranked_dpls import RankedDplsConfig, solve_ranked
from .edit_ranker import (EditFeatureContext, candidate_features, EditRanker,
                         make_model, export_model, FEATURE_DIM, BYTE_SCALE)
from .experiment import write_json, sha


REGISTRY = ROOT/'training/stageb_hgraph/current_algorithm.json'


def identity(row):
    return f"{row['family']}:seed{row['seed']}:tti{row['tti']}"


def read_rows(seeds, *, evaluation=False):
    import torch
    enable_legacy_types()
    rows,files=[],{}
    for seed in seeds:
        if evaluation:
            paths=[('pfq',ROOT/f'deliverables/tb_joint_repair_20261027_native_v2/seed{seed}_pfq0/states.pt')]
            ttis=(2001,2501,3001,3501,4001,4501,5000)
        else:
            paths=[('raw_global_deficit',ROOT/f'deliverables/dpls_component_ablation_20261029_v1/native/raw_global_deficit/seed{seed}_current_20000/states.pt'),
                   ('pfq',ROOT/f'deliverables/boundary08_rho090_five_seed_20261027/raw_runs/seed{seed}_pfq0/states.pt')]
            ttis=(4000,4001,8000,8001,18000,28000)
        for family,path in paths:
            files[str(path)]=sha(path)
            saved=torch.load(path,map_location='cpu',weights_only=False)
            for row in saved:
                if row['tti'] not in ttis:continue
                raw=detach_raw_phy_row(dict(row,seed=seed,family=family,source=str(path)))
                # Discard post-action information before any training/decision.
                raw={k:raw[k] for k in ('tti','obs','snapshot','ema','raw_phy','seed','family','source')}
                if 'ctbs' in raw:raise AssertionError('Legacy CTBS object reached the new decision path')
                rows.append(raw)
    return rows,files


def run_decision(row, mode, top_k=64, ranker=None, observer=None):
    started=time.perf_counter()
    problem=problem_from_raw_row(row)
    draft=raw_pfq_draft(problem,row['obs'])
    action,values,record=solve_ranked(problem,draft,RankedDplsConfig(mode,top_k),
                                     ranker=ranker,observer=observer)
    record['e2e_ms']=(time.perf_counter()-started)*1000
    record.update(state_id=identity(row),seed=row['seed'],family=row['family'],tti=row['tti'],arm=mode)
    return action,values,record


def collect(rows,out,round_index,ranker=None,top_k=64):
    """Exact labels on full-teacher or the selected student's actual roots.

    Extra labels use a second PHY context and cannot change student candidates,
    counters, or caches. All such offline teacher work is separately recorded.
    """
    import torch
    groups,decisions=[],[]
    extra_cost={k:0 for k in ('candidate_evaluations','ue_evaluations','mmse_requests','mmse_misses')}
    for row in rows:
        p=problem_from_raw_row(row)
        teacher=NativePhy(p)
        teacher_context=None
        checks=[]
        try:
            def observe(info):
                nonlocal teacher_context
                base=info['base'];actions=info['actions']
                signature=hashlib.sha256(base.actions.tobytes()+actions.tobytes()).hexdigest()
                if info['features'] is None:
                    if teacher_context is None or teacher_context.received is not info['phy'].received:
                        teacher_context=EditFeatureContext(info['problem'],info['phy'])
                    x,heuristic=candidate_features(info['problem'],teacher_context,base,actions,floor=info['floor'])
                    logical=teacher_context.last_dequeue_totals
                else:
                    x,heuristic=info['features'],info['heuristic']
                    logical=info['context'].last_dequeue_totals
                if info['full_scan']:
                    score,eligible=info['scores'],info['eligible']
                else:
                    before=teacher.stats
                    independent_base=teacher.evaluate(base.actions[0])
                    exact=teacher.evaluate(actions,independent_base)
                    score=exact.values[:,:,EXPECTED]@p.weights
                    eligible=exact.values[:,:,DEQUEUE].sum(1)>=info['floor']
                    np.testing.assert_array_equal(exact.values[:,:,DEQUEUE].sum(1),logical)
                    for key,value in teacher.stats.items():extra_cost[key]+=value-before[key]
                if not np.array_equal(eligible,logical>=info['floor']):
                    raise AssertionError('Logical floor screen differs from exact physical dequeue')
                if signature!=hashlib.sha256(base.actions.tobytes()+actions.tobytes()).hexdigest():
                    raise AssertionError('Teacher observer mutated student candidates')
                root_j=float(base.values[0,:,EXPECTED]@p.weights)
                groups.append(dict(x=x.copy(),heuristic=heuristic.copy(),
                    gain=np.asarray(score-root_j,np.float32),eligible=eligible.copy(),
                    state_id=identity(row),seed=row['seed'],iteration=info['iteration'],
                    floor=info['floor'],root=base.actions[0].copy(),
                    root_sha256=hashlib.sha256(base.actions.tobytes()).hexdigest(),round=round_index))
            mode='full' if ranker is None else 'ai'
            action,values,record=run_decision(row,mode,top_k,ranker,observe)
            # Collection decisions are not used for wall-clock comparisons.
            np.testing.assert_allclose(values,ReferencePhy(p).evaluate(action)[0],rtol=1.e-8,atol=1.e-6)
            decisions.append({k:v for k,v in record.items() if k!='history'})
            print('LABELS',round_index,identity(row),record['rounds'],len(groups),flush=True)
        finally:
            if teacher_context is not None:teacher_context.close()
            teacher.close()
    torch.save(groups,out/f'labels_round{round_index}.pt')
    teacher_cost={k:sum(d[k] for d in decisions) if ranker is None else extra_cost[k] for k in extra_cost}
    write_json(out/f'collection_round{round_index}.json',dict(round=round_index,groups=len(groups),
        candidate_labels=sum(len(g['gain']) for g in groups),states=len(rows),
        teacher_cost=teacher_cost,student_decisions=decisions,
        source='full DPLS roots' if ranker is None else 'selected student visited roots; separate complete teacher scoring',
        ctbs=False,post_action_inputs=False))
    return groups


def mean_seed(records,key):
    d={}
    for r in records:d.setdefault(r['seed'],[]).append(r[key])
    return float(np.mean([np.mean(v) for v in d.values()]))


def validation_metric(records,reference):
    ref={r['state_id']:r for r in reference}
    ratios=np.array([r['score']/ref[r['state_id']]['score'] for r in records])
    quality=bool((ratios>=.995).all() and all(r['converged'] for r in records))
    deficit=float(np.maximum(.995-ratios,0).mean())
    ms=mean_seed(records,'e2e_ms')
    # Quality first, then minimize mean relative shortfall, then actual time.
    key=(int(quality),-deficit,-ms)
    return key,dict(quality_pass=quality,quality_shortfall=deficit,mean_ms=ms,
        worst_relative_j=float(ratios.min()-1),mean_j=mean_seed(records,'score'))


def training_arrays(groups,seed):
    rng=np.random.default_rng(seed)
    groups=[g for g in groups if g['eligible'].any()]
    all_x=np.concatenate([g['x'][g['eligible']] for g in groups])
    mean=all_x.mean(0);scale=np.maximum(all_x.std(0),.1)
    width=192
    x=np.zeros((len(groups),width,FEATURE_DIM),np.float32)
    prior=np.zeros((len(groups),width),np.float32)
    target=np.zeros_like(prior);mask=np.zeros_like(prior,dtype=bool)
    for i,g in enumerate(groups):
        legal=np.flatnonzero(g['eligible'])
        by_gain=legal[np.argsort(-g['gain'][legal],kind='stable')[:64]]
        by_prior=legal[np.argsort(-g['heuristic'][legal],kind='stable')[:64]]
        random=rng.choice(legal,min(64,len(legal)),replace=False)
        selected=np.array(list(dict.fromkeys(np.r_[by_gain,by_prior,random].tolist())),np.int64)
        n=len(selected)
        x[i,:n]=g['x'][selected];prior[i,:n]=g['heuristic'][selected]/BYTE_SCALE
        target[i,:n]=g['gain'][selected]/BYTE_SCALE;mask[i,:n]=True
    return np.clip((x-mean)/scale,-10,10),prior,target,mask,mean,scale


def train(groups,validation,reference,out,round_index,epochs):
    import torch
    torch.manual_seed(8411+round_index)
    x,prior,target,mask,mean,scale=training_arrays(groups,8411+round_index)
    tx,tp,ty,tm=map(torch.from_numpy,(x,prior,target,mask))
    model=make_model()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(8411+round_index)
    best=None;history=[]
    for epoch in range(epochs+1):
        losses=[]
        if epoch:
            order=rng.permutation(len(x))
            for begin in range(0,len(x),32):
                ix=order[begin:begin+32]
                pred=tp[ix]+model(tx[ix])[...,0]
                m=tm[ix]
                q=torch.softmax((ty[ix]/.3).masked_fill(~m,-1.e9),-1)
                logits=(pred/.3).masked_fill(~m,-1.e9)
                loss=-(q*torch.log_softmax(logits,-1)).sum(-1).mean()
                loss+=.1*torch.nn.functional.smooth_l1_loss(pred[m],ty[ix][m])
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step();losses.append(float(loss.detach()))
        item=dict(epoch=epoch,loss=float(np.mean(losses)) if losses else None)
        if epoch in (0,5,10,epochs):
            checkpoint=out/f'edit_ranker_r{round_index}_e{epoch:02d}.npz'
            export_model(checkpoint,model,mean,scale)
            deployed=EditRanker(checkpoint)
            with torch.no_grad():
                predicted=(tp[:8]+model(tx[:8])[...,0]).numpy()*BYTE_SCALE
            # Train input is normalized; compare NumPy on original features.
            original_x=tx[:8].numpy()*scale+mean
            np.testing.assert_allclose(deployed.predict(original_x,tp[:8].numpy()*BYTE_SCALE),
                                       predicted,rtol=3.e-5,atol=.02)
            variants=[]
            for k in (64,128):
                records=[run_decision(r,'ai',k,deployed)[2] for r in validation]
                key,metric=validation_metric(records,reference)
                metric.update(top_k=k,checkpoint=str(checkpoint),epoch=epoch,round=round_index)
                variants.append(metric)
                candidate=(key,checkpoint,k,epoch,round_index)
                if best is None or candidate[0]>best[0]:best=candidate
                print('MODEL',round_index,epoch,k,metric['quality_pass'],metric['mean_ms'],metric['worst_relative_j'],flush=True)
            item['validation']=variants
        history.append(item)
    write_json(out/f'training_round{round_index}.json',dict(groups=len(x),features=FEATURE_DIM,
        parameters=sum(p.numel() for p in model.parameters()),epochs=epochs,
        candidate_sampling='64 exact leaders + 64 heuristic leaders + 64 uniform; full labels retained',
        selection='quality first (all J>=99.5% and converged), then mean shortfall, then seed-equal wall time',
        selected=str(best[1]),top_k=best[2],selected_epoch=best[3],history=history))
    return best


def evaluate(rows,out,checkpoint,k,repeats):
    import pandas as pd
    model=EditRanker(checkpoint)
    arms=('full','heuristic','ai')
    # First-use immutable tables and libraries are warmed equally before timing.
    for arm in arms:run_decision(rows[0],arm,k,model if arm=='ai' else None)
    rng=np.random.default_rng(8431)
    records=[];outputs={};audits=[]
    for row in rows:
        for repeat in range(repeats):
            for arm in rng.permutation(arms):
                arm=str(arm)
                action,values,record=run_decision(row,arm,k,model if arm=='ai' else None)
                key=(identity(row),arm)
                if key in outputs:
                    np.testing.assert_array_equal(action,outputs[key][0])
                    np.testing.assert_array_equal(values,outputs[key][1])
                    if record['candidate_evaluations']!=outputs[key][2]:
                        raise AssertionError('Repeated work changed without a clock cutoff')
                else:outputs[key]=(action.copy(),values.copy(),record['candidate_evaluations'])
                records.append(dict({key:value for key,value in record.items() if key!='history'},
                                    repetition=repeat))
        print('TIMED',identity(row),flush=True)
    for row in rows:
        p=problem_from_raw_row(row);reference=ReferencePhy(p)
        for arm in arms:
            action,values,_=outputs[(identity(row),arm)]
            expected=reference.evaluate(action)[0]
            np.testing.assert_allclose(values,expected,rtol=1.e-8,atol=1.e-6)
            audits.append(dict(state_id=identity(row),arm=arm,max_error=float(np.abs(values-expected).max())))
    f=pd.DataFrame(records);f.to_csv(out/'measurements.csv',index=False)
    per=f.groupby(['state_id','seed','family','tti','arm','top_k']).median(numeric_only=True).reset_index()
    per.to_csv(out/'per_state.csv',index=False)
    ref=per[per.arm=='full'].set_index('state_id')
    summaries=[]
    for arm,g in per.groupby('arm'):
        g=g.set_index('state_id');base=ref.loc[g.index]
        relative=g.score/base.score-1
        delta=g.score-base.score
        records_=g.reset_index().to_dict('records')
        quality=bool((relative>=-.005).all() and g.converged.all())
        ms=mean_seed(records_,'e2e_ms')
        base_ms=mean_seed(base.reset_index().to_dict('records'),'e2e_ms')
        summaries.append(dict(arm=arm,top_k=k,score=mean_seed(records_,'score'),mean_ms=ms,
            p95_ms=float(g.e2e_ms.quantile(.95)),candidates=mean_seed(records_,'candidate_evaluations'),
            mean_relative_j_pct=float(relative.groupby(g.seed).mean().mean()*100),
            worst_relative_j_pct=float(relative.min()*100),wins=int((delta>1).sum()),losses=int((delta<-1).sum()),
            quality_pass=quality,cost_pass=bool(ms<=.8*base_ms),relative_cost_pct=(ms/base_ms-1)*100))
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    by_arm={r['arm']:r for r in summaries}
    ai,heur=by_arm['ai'],by_arm['heuristic']
    learned_nonzero=bool(np.any(model.w3) or np.any(model.b3))
    contribution=bool(learned_nonzero and ai['quality_pass'] and ai['cost_pass'] and
        ((heur['quality_pass'] and ai['mean_ms']<heur['mean_ms']) or not heur['quality_pass']))
    write_json(out/'evaluation_checks.json',dict(passed=True,scientific_gate_passed=bool(ai['quality_pass'] and ai['cost_pass']),
        learned_contribution_gate_passed=contribution,learned_residual_nonzero=learned_nonzero,
        repeated_actions_values_and_work_exact=True,independent_audits=audits,
        ctbs=False,raw_pfq_floor=True,all_converged=bool(per.converged.all()),
        caveat='Historical PFQ-only development states; no native packet or scene-generalization claim'))
    np.savez_compressed(out/'actions.npz',**{f'{i}_{arm}':outputs[(identity(row),arm)][0]
                      for i,row in enumerate(rows) for arm in arms})


def main(args):
    import torch
    torch.set_num_threads(1)
    if args.epochs<10 or args.repeats<3:
        raise ValueError('Use at least ten epochs and three paired timing repeats')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    registry_sha=sha(REGISTRY)
    protocol=dict(date='2026-11-02',version=1,ctbs=False,demand_headroom_masks=False,
        objective='J=sum deficit_weight*expected_success_bytes; original native-BLER canonical kernel',
        floor='ceil(.90*own raw PFQ physical dequeue), fixed throughout each decision',
        training_seeds=[41,42,43],validation_seeds=[44,45],training_sources=['raw_global_deficit','pfq'],
        training_ttis=[4000,4001,8000,8001,18000,28000],
        evaluation_seeds=[46,47,48],evaluation_sources=['pfq'],
        evaluation_ttis=[2001,2501,3001,3501,4001,4501,5000],historical_development=True,
        final_seeds_not_read=[311,312,313,314,315],top_k_grid=[64,128],self_rounds=2,
        epochs_per_round=args.epochs,checkpoints=[0,5,10,args.epochs],repeats=args.repeats,
        selection='All-state >=99.5% J quality first, mean shortfall second, seed-equal complete time third',
        independent_ai_gate='nonzero learned residual; AI passes quality and >=20% speedup vs full; faster than quality-passing heuristic or heuristic fails quality',
        quality_gate='every state J >=99.5% of full, all converge with final full-neighborhood stop',
        cost_gate='seed-equal complete mean time <=80% of full; P95 also reported',
        batch_semantics='one immutable root and at most one accepted edit per iteration; stable original index ties',
        fallback='top-K no >threshold gain -> score only remaining candidates; final complete scan required',
        threshold='>1 byte first two iterations, >16 thereafter; no earlier stop than iteration3',
        timing='single-thread serial raw snapshot->MCS/TBS/current PFQ->fresh PHY/context->features/network/sort/search/fallback->cleanup',
        excluded='archive unpacking, model/library loading, first immutable table setup; teacher labels separately charged',
        registry_sha256=registry_sha,legacy_carrier='archive-only physical metadata unpacked to raw_phy; no CTBS methods or demand pruning',
        current_registry_changed=False,native_campaign='not started unless separately justified by completed offline gates')
    write_json(out/'protocol.json',protocol)
    paths=[p for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.cpp','.so','.rst')]
    paths += [ROOT/'cuMAC/src/cumac.h',REGISTRY,ARCHIVE/'training/stageb_hgraph/phy_tbs_alignment_audit.py']
    manifest={}
    for path in paths:
        rel=path.relative_to(ROOT);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest);manifest[str(rel)]=sha(path)
    write_json(out/'source_manifest.json',manifest)
    write_json(out/'runtime_environment.json',dict(python=platform.python_version(),numpy=np.__version__,
        torch=torch.__version__,platform=platform.platform(),gpu_used=False,threads=1))
    train_rows,train_files=read_rows((41,42,43));validation,val_files=read_rows((44,45))
    write_json(out/'input_manifest.json',dict(training=train_files,validation=val_files))
    # Numeric input preparation warms ONLY immutable native TBS construction.
    for row in train_rows+validation:problem_from_raw_row(row)
    run_decision(validation[0],'full')
    reference=[run_decision(r,'full')[2] for r in validation]
    write_json(out/'validation_reference.json',reference)
    groups=collect(train_rows,out,0)
    first=train(groups,validation,reference,out,0,args.epochs)
    groups+=collect(train_rows,out,1,EditRanker(first[1]),first[2])
    second=train(groups,validation,reference,out,1,args.epochs)
    # Re-evaluate round champions together before committing the final choice.
    finalists=[]
    for candidate in (first,second):
        model=EditRanker(candidate[1]);k=candidate[2]
        records=[run_decision(r,'ai',k,model)[2] for r in validation]
        key,metric=validation_metric(records,reference)
        finalists.append((key,candidate,metric))
    _,best,metric=max(finalists,key=lambda x:x[0])
    checkpoint,k,epoch,round_index=best[1:]
    with np.load(checkpoint,allow_pickle=False) as z:zero=not (np.any(z['w3']) or np.any(z['b3']))
    write_json(out/'selection.json',dict(selected=str(checkpoint),sha256=sha(checkpoint),top_k=k,
        selected_epoch=epoch,selected_round=round_index,zero_residual=zero,
        validation=metric,evaluation_unread_at_selection=True))
    rows,eval_files=read_rows((46,47,48),evaluation=True)
    write_json(out/'evaluation_input_manifest.json',eval_files)
    evaluate(rows,out,checkpoint,k,args.repeats)
    if sha(REGISTRY)!=registry_sha:raise AssertionError('Active registry changed')
    for name,digest in manifest.items():
        if sha(ROOT/name)!=digest:raise AssertionError(f'Frozen source changed: {name}')
    write_json(out/'completion.json',dict(completed=True,ctbs=False,protected_registry_unchanged=True,
        final_seeds_not_read=True,native_trajectories=0))
    print('COMPLETED',str(out),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--repeats',type=int,default=3)
    main(parser.parse_args())

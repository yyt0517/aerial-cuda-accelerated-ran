"""Bounded scalar residual trajectories and passive own-state supervision.

No CTBS, future observations, teacher labels, full search or PPO are available
inside generate(). Exact full-neighborhood labels are queried only after the
student has finished, using a separate physical context.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

import numpy as np

from .bounded_probes import ProbePhy, Predictor, ranking_metric, seed_mean
from .edit_ranker import EditFeatureContext, candidate_features
from .physics import EXPECTED, DEQUEUE, COUNT
from .search import greedy_candidates
from .no_ctbs_inputs import raw_pfq_floor, problem_from_raw_row, raw_pfq_draft
from .ranked_dpls_campaign import identity
from .experiment import write_json, sha


@dataclass(frozen=True)
class Geometry:
    width: int
    depth: int
    max_roots: int = 56
    max_candidates: int = 106

    def __post_init__(self):
        if min(self.width,self.depth,self.max_roots,self.max_candidates)<1:
            raise ValueError('Positive geometry and resource limits required')


SHALLOW=Geometry(7,8)
DEEP=Geometry(1,32)


def generate(p,draft,geometry=DEEP,ranker=None,*,observer=None):
    """Same expansion rule as bounded_probes.generate, with explicit budgets.

    A width-one path verifies one top-ranked edit per step. The shallow beam
    verifies up to seven edits while it has one parent, otherwise two/parent.
    A failed edit retains its parent. The depth limit is attempted steps, not
    a promise of that many successful edits; repeated/stalled roots are logged.
    """
    phy=ProbePhy(p);context=EditFeatureContext(p,phy)
    record=None
    try:
        initial=phy.evaluate(draft);floor=raw_pfq_floor(initial.values[0])
        frontier=[(initial,0)];roots=0;attempted=0;accepted=0;generated=0
        feature_ms=network_ms=generation_ms=physical_ms=0.
        stagnant=0;unique_roots=set();visited_prgs=set();stop='depth_limit'
        for step in range(geometry.depth):
            children=[];budget_hit=False
            for path,(base,path_edits) in enumerate(frontier):
                if roots>=geometry.max_roots or phy.stats['candidate_evaluations']>=geometry.max_candidates:
                    budget_hit=True;stop='resource_limit';break
                roots+=1;fingerprint=hashlib.sha256(base.actions.tobytes()).hexdigest()
                unique_roots.add(fingerprint)
                stamp=time.perf_counter();actions=greedy_candidates(p,base.actions[0]);generated+=len(actions)
                generation_ms+=(time.perf_counter()-stamp)*1000
                stamp=time.perf_counter();x,prior=candidate_features(p,context,base,actions,floor=floor)
                feasible=context.last_dequeue_totals>=floor
                feature_ms+=(time.perf_counter()-stamp)*1000
                stamp=time.perf_counter();prediction=prior if ranker is None else ranker.predict(x,prior)
                order=np.lexsort((np.arange(len(actions)),-prediction))
                quota=geometry.width if len(frontier)==1 else 2
                quota=min(quota,geometry.max_candidates-phy.stats['candidate_evaluations'])
                chosen=order[feasible[order]][:quota]
                network_ms+=(time.perf_counter()-stamp)*1000
                gained=0
                if len(chosen):
                    stamp=time.perf_counter();exact=phy.evaluate(actions[chosen],base)
                    physical_ms+=(time.perf_counter()-stamp)*1000;attempted+=len(chosen)
                    for i,index in enumerate(chosen):
                        changed=np.flatnonzero((actions[index]!=base.actions[0]).any(0))
                        visited_prgs.update(map(int,changed))
                        if exact.values[i,:,EXPECTED]@p.weights>base.values[0,:,EXPECTED]@p.weights+1:
                            children.append((phy.take(exact,i),path_edits+1));accepted+=1;gained+=1
                stagnant+=gained==0
                children.append((base,path_edits))
                if observer is not None:
                    # Copies only. The observer cannot touch the executing PHY,
                    # features, chosen indices or incumbent through its payload.
                    observer(dict(root=base.actions[0].copy(),step=step+1,path=path,ordinal=roots,
                        fingerprint=fingerprint,selected=chosen.copy(),floor=floor,
                        accepted=gained,path_edits=path_edits))
            if budget_hit:
                # Keep every unprocessed parent instead of silently deleting it.
                children.extend(frontier)
            unique={v.actions.tobytes():(v,n) for v,n in children}
            frontier=sorted(unique.values(),key=lambda item:-float(item[0].values[0,:,EXPECTED]@p.weights))[:geometry.width]
            if budget_hit:break
        pool={draft.tobytes():(initial,0)}
        for v,n in frontier:pool[v.actions.tobytes()]=(v,n)
        pool=list(pool.values())[:geometry.width+1]
        best,edits=max(pool,key=lambda item:float(item[0].values[0,:,EXPECTED]@p.weights))
        record=dict(score=float(best.values[0,:,EXPECTED]@p.weights),initial_score=float(initial.values[0,:,EXPECTED]@p.weights),
            floor=floor,width=geometry.width,depth=geometry.depth,feature_roots=roots,
            unique_feature_roots=len(unique_roots),stagnant_roots=stagnant,
            attempted_edits=attempted,accepted_branch_edits=accepted,best_path_edits=edits,
            pool_size=len(pool),proposal_prgs=len(visited_prgs),generated_candidates=generated,
            feature_ms=feature_ms,network_ms=network_ms,generation_ms=generation_ms,physical_ms=physical_ms,
            stop_reason=stop,full_search_calls=0,**phy.stats)
        assert record['candidate_evaluations']<=geometry.max_candidates
        assert roots<=geometry.max_roots
        return best.actions[0].copy(),best.values[0].copy(),record
    finally:context.close();phy.close()


def coverage(values,reference):
    """Unweighted per-UE expected bytes, not packet goodput or UE throughput."""
    actual=values[:,EXPECTED];target=reference[:,EXPECTED]
    active=target>1.;weak=np.argsort(target,kind='stable')[:max(1,int(np.ceil(.2*len(target))))]
    target_p5=float(np.quantile(target,.05));p5=float(np.quantile(actual,.05))
    return dict(served_ues=int((values[:,COUNT]>0).sum()),reference_served_ues=int((reference[:,COUNT]>0).sum()),
        newly_unserved_ues=int(((reference[:,COUNT]>0)&(values[:,COUNT]==0)).sum()),
        expected_p5=p5,reference_expected_p5=target_p5,
        expected_p5_ratio=p5/target_p5 if target_p5>1.e-9 else (1. if p5>=target_p5 else 0.),
        weak_service_ratio=float(actual[weak].sum()/target[weak].sum()) if target[weak].sum()>1.e-9 else 1.,
        ue_service_drops_gt2pct=int((actual[active]<.98*target[active]).sum()))


def collect_own(rows,model,geometry,out,round_index,max_roots=8):
    import torch
    groups=[];student_records=[];teacher_records=[];visits=[]
    for row in rows:
        p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs']);observed=[]
        action,values,student=generate(p,draft,geometry,model,observer=observed.append)
        student_records.append(dict(state_id=identity(row),seed=row['seed'],**student))
        # Deduplicate before sampling; a stalled root is not a new sample.
        unique={}
        for v in observed:unique.setdefault(v['fingerprint'],v)
        unique=list(unique.values())
        indices=np.unique(np.linspace(0,len(unique)-1,min(max_roots,len(unique)),dtype=int)) if unique else []
        phy=ProbePhy(p);context=EditFeatureContext(p,phy)
        try:
            for index in indices:
                visit=unique[int(index)];root=visit['root'];base=phy.evaluate(root)
                actions=greedy_candidates(p,root)
                x,prior=candidate_features(p,context,base,actions,floor=student['floor'])
                exact=phy.evaluate(actions,base)
                gain=exact.values[:,:,EXPECTED]@p.weights-base.values[0,:,EXPECTED]@p.weights
                eligible=exact.values[:,:,DEQUEUE].sum(1)>=student['floor']
                np.testing.assert_array_equal(eligible,context.last_dequeue_totals>=student['floor'])
                groups.append(dict(state_id=identity(row),seed=row['seed'],iteration=visit['ordinal'],
                    x48=x,prior=prior,gain=gain,eligible=eligible,root=root.copy(),floor=student['floor'],
                    root_sha256=visit['fingerprint'],round=round_index,student_step=visit['step']))
                visits.append(dict(state_id=identity(row),round=round_index,root_sha256=visit['fingerprint'],
                    student_step=visit['step'],ordinal=visit['ordinal'],actual_visited=True,
                    candidate_labels=len(gain),teacher_best_gain=float(max(0,gain[eligible].max())) if eligible.any() else 0.))
            teacher_records.append(dict(state_id=identity(row),**phy.stats))
        finally:context.close();phy.close()
        # Re-execution proves offline teacher calls cannot influence the student.
        again=generate(p,draft,geometry,model)
        np.testing.assert_array_equal(action,again[0])
        assert student['candidate_evaluations']==again[2]['candidate_evaluations']
        print('OWN',round_index,identity(row),'visited',len(observed),'unique',len(unique),'labels',len(indices),flush=True)
    torch.save(groups,out/f'own_round{round_index}.pt')
    write_json(out/f'own_round{round_index}_work.json',dict(student_decisions=student_records,
        teacher=teacher_records,groups=len(groups),candidate_labels=sum(len(g['gain']) for g in groups),
        post_decision_teacher=True,student_replay_identical=True,visits=visits))
    return groups,visits


def group_key(g):
    return g['state_id'],hashlib.sha256(g['root'].tobytes()).hexdigest()


def fine_tune(groups,starting,out,round_index,validate,epochs=20):
    """Continue frozen scalar weights/normalization; fresh declared AdamW state."""
    import torch
    torch.manual_seed(20261103+round_index)
    model=torch.nn.Sequential(torch.nn.Linear(48,32),torch.nn.ReLU(),torch.nn.Linear(32,16),
        torch.nn.ReLU(),torch.nn.Linear(16,1))
    with torch.no_grad():
        for layer,w,b in [(model[0],starting.w1,starting.b1),(model[2],starting.w2,starting.b2),(model[4],starting.w3,starting.b3)]:
            layer.weight.copy_(torch.from_numpy(w));layer.bias.copy_(torch.from_numpy(b))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005)
    rng=np.random.default_rng(20261103+round_index);arrays=[]
    for g in groups:
        legal=np.flatnonzero(g['eligible'])
        if not len(legal):continue
        prior=g['prior'];prediction=starting.predict(g['x48'],prior)
        sample=np.unique(np.r_[legal[np.argsort(-g['gain'][legal],kind='stable')[:32]],
            legal[np.argsort(-prediction[legal],kind='stable')[:32]],
            legal[np.argsort(-prior[legal],kind='stable')[:32]],rng.choice(legal,min(32,len(legal)),replace=False)])
        x=np.clip((g['x48'][sample]-starting.mean)/starting.scale,-10,10)
        arrays.append((torch.from_numpy(x),torch.tensor(prior[sample]/3000,dtype=torch.float32),
            torch.tensor(g['gain'][sample]/3000,dtype=torch.float32)))
    history=[];best=None
    for epoch in range(epochs+1):
        losses=[]
        if epoch:
            for index in rng.permutation(len(arrays)):
                x,prior,y=arrays[index];pred=prior+model(x).ravel()
                target=torch.softmax(y/.15,0)
                loss=-(target*torch.log_softmax(pred/.15,0)).sum()+.1*torch.nn.functional.smooth_l1_loss(pred,y)
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step();losses.append(float(loss.detach()))
        if epoch in (0,5,10,epochs):
            path=out/f'own_r{round_index}_e{epoch:02d}.npz'
            np.savez(path,mean=starting.mean,scale=starting.scale,**{k:v.detach().numpy() for k,v in (
                ('w1',model[0].weight),('b1',model[0].bias),('w2',model[2].weight),
                ('b2',model[2].bias),('w3',model[4].weight),('b3',model[4].bias))})
            predictor=Predictor(path)
            # Verify the deployed numerical path before validation selection.
            if arrays:
                x,prior,_=arrays[0]
                with torch.no_grad():expected=prior.numpy()*3000+model(x).numpy().ravel()*3000
                original_x=x.numpy()*starting.scale+starting.mean
                np.testing.assert_allclose(predictor.predict(original_x,prior.numpy()*3000),expected,rtol=5.e-5,atol=.05)
            key,metric=validate(predictor)
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)) if losses else None,**metric))
            if best is None or key>best[0]:best=(key,predictor,epoch,metric)
            print('FIT',round_index,epoch,metric,flush=True)
    write_json(out/f'own_round{round_index}_training.json',dict(groups=len(groups),epochs=epochs,
        normalization='frozen original scalar training mean/scale',optimizer='fresh AdamW lr0.0005',
        labels='immediate exact whole-TB gain; no multi-TTI or suffix-value supervision',
        selected_epoch=best[2],selected=str(best[1].path),history=history))
    return best[1],dict(epoch=best[2],checkpoint=str(best[1].path),sha256=sha(best[1].path),metric=best[3])

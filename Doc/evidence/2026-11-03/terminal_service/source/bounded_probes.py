"""Three small, CTBS-free diagnostics; no native rollout or PPO.

Models see causal inputs only. Exact labels, certificates, and their costs are
separate. Frozen development states are never called blind tests.
"""
from __future__ import annotations

import argparse
import ctypes
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import weakref

import numpy as np

from .physics import (ROOT, NativePhy, ReferencePhy, COUNT, LAYER, DEQUEUE,
                      EXPECTED, BLER, curves)
from .no_ctbs_inputs import problem_from_raw_row, raw_pfq_draft, raw_pfq_floor
from .ranked_dpls_campaign import read_rows, identity
from .ranked_dpls import solve_ranked
from .search import greedy_candidates, ProposalGraph
from .edit_ranker import EditFeatureContext, candidate_features
from .experiment import write_json, sha


def build_kernel():
    directory=Path(__file__).parent
    subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-ffp-contract=off',
        '-shared','-fPIC','-I/usr/include/eigen3',str(directory/'probe_kernel.cpp'),
        '-o',str(directory/'_probe.so')],check=True)


@lru_cache(maxsize=1)
def probe_library():
    lib=ctypes.CDLL(str(Path(__file__).with_name('_probe.so')))
    i=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    d=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    f=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    lib.lns_create.argtypes=[ctypes.c_int]*3+[ctypes.c_double]+[i]*4+[d]*3+[i,d,d]
    lib.lns_create.restype=ctypes.c_void_p
    lib.lns_destroy.argtypes=[ctypes.c_void_p];lib.lns_destroy.restype=None
    lib.lns_evaluate.argtypes=[ctypes.c_void_p,ctypes.c_int,i,ctypes.c_void_p,ctypes.c_void_p,d]
    lib.lns_evaluate.restype=ctypes.c_int
    lib.lns_stats.argtypes=[ctypes.c_void_p,i];lib.lns_stats.restype=None
    lib.probe_prepare.argtypes=[ctypes.c_void_p];lib.probe_prepare.restype=ctypes.c_int
    lib.probe_features.argtypes=[ctypes.c_void_p,ctypes.c_int,i,i,d,d,ctypes.c_int,f,d,d,d]
    lib.probe_features.restype=ctypes.c_int
    return lib


class ProbePhy(NativePhy):
    def __init__(self,p):
        self.problem=p;self.lib=probe_library()
        self.covariance=np.ascontiguousarray(p.h@p.h.conj().swapaxes(-1,-2))
        self.signal=np.ascontiguousarray(p.h[np.arange(p.U),:,p.serving]@p.w)
        self.received=np.trace(self.covariance,axis1=-2,axis2=-1).real
        offsets=np.r_[0,np.cumsum([len(x) for x,_ in curves()])].astype(np.int64)
        xs=np.concatenate([x for x,_ in curves()]);ys=np.concatenate([y for _,y in curves()])
        self.context=self.lib.lns_create(p.U,p.C,p.P,p.noise,p.serving,p.mcs,p.caps,p.tbs,
            p.queue,self.covariance.view(np.float64),self.signal.view(np.float64),offsets,xs,ys)
        if not self.context:raise RuntimeError('Probe PHY construction failed')
        self._release=weakref.finalize(self,self.lib.lns_destroy,self.context)

    def prepare_bank(self):
        if not self._release.alive:raise RuntimeError('Closed probe PHY')
        if self.lib.probe_prepare(self.context):raise ValueError('Bank supports at most three cells')

    def features(self,base,actions,spatial=True):
        if not self._release.alive:raise RuntimeError('Closed probe PHY')
        if base.owner is not self or len(base.actions)!=1:raise ValueError('Foreign base')
        actions=self.problem.actions(actions);n=len(actions)
        x=np.empty((n,72),np.float32)
        upper=np.empty(n);total=np.empty(n);proxy=np.empty(n)
        error=self.lib.probe_features(self.context,n,actions,base.actions[0],base.values[0],self.problem.weights,
            int(spatial),x,upper,total,proxy)
        if error:raise ValueError(f'Unsupported probe candidate ({error})')
        if not all(np.isfinite(a).all() for a in (x,upper,total,proxy)):raise AssertionError('Nonfinite probe features')
        return x,upper,total,proxy


class Predictor:
    def __init__(self,path):
        self.path=Path(path)
        with np.load(path) as z:
            for k in ('mean','scale','w1','b1','w2','b2','w3','b3'):setattr(self,k,z[k])
    def predict(self,x,prior=None):
        z=np.clip((x-self.mean)/self.scale,-10,10)
        z=np.maximum(z@self.w1.T+self.b1,0)
        z=np.maximum(z@self.w2.T+self.b2,0)
        out=(z@self.w3.T+self.b3).ravel().astype(float)*3000
        return out if prior is None else out+prior


def seed_mean(rows,key):
    return float(np.mean([np.mean([r[key] for r in rows if r['seed']==s]) for s in sorted({r['seed'] for r in rows})]))

def ranking_metric(g,pred,k=8):
    eligible=np.flatnonzero(g['eligible'])
    if not len(eligible):return dict(regret=0.,near=1.,retained=1.)
    truth=g['gain'];best=float(max(0,truth[eligible].max()))
    chosen=eligible[np.argsort(-pred[eligible],kind='stable')[:k]]
    achieved=float(max(0,truth[chosen].max()))
    return dict(regret=best-achieved,near=float(achieved>=best-max(16.,.005*best)),
                retained=achieved/best if best>16 else 1.)


def train_model(groups,validation,out,name,feature_key,epochs=30):
    import torch
    torch.manual_seed(20261102)
    all_x=np.concatenate([g[feature_key] for g in groups])
    mean=all_x.mean(0);scale=np.maximum(all_x.std(0),.1)
    model=torch.nn.Sequential(torch.nn.Linear(len(mean),32),torch.nn.ReLU(),
        torch.nn.Linear(32,16),torch.nn.ReLU(),torch.nn.Linear(16,1))
    torch.nn.init.zeros_(model[-1].weight);torch.nn.init.zeros_(model[-1].bias)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    rng=np.random.default_rng(20261102)
    history=[];best=None
    # Fixed sampled lists: leaders, prior leaders and random candidates.
    arrays=[]
    for g in groups:
        legal=np.flatnonzero(g['eligible'])
        indices=np.unique(np.r_[legal[np.argsort(-g['gain'][legal])[:32]],
            legal[np.argsort(-g['prior'][legal])[:32]],rng.choice(legal,min(64,len(legal)),replace=False)])
        arrays.append((torch.from_numpy(np.clip((g[feature_key][indices]-mean)/scale,-10,10)),
            torch.tensor(g['prior'][indices]/3000,dtype=torch.float32),
            torch.tensor(g['gain'][indices]/3000,dtype=torch.float32)))
    for epoch in range(epochs+1):
        losses=[]
        if epoch:
            for index in rng.permutation(len(arrays)):
                x,prior,y=arrays[index];pred=prior+model(x).ravel()
                q=torch.softmax(y/.3,0)
                loss=-(q*torch.log_softmax(pred/.3,0)).sum()+.1*torch.nn.functional.smooth_l1_loss(pred,y)
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step();losses.append(float(loss.detach()))
        if epoch in (0,10,20,epochs):
            path=out/f'{name}_e{epoch:02d}.npz'
            np.savez(path,mean=mean,scale=scale,**{k:v.detach().numpy() for k,v in (
                ('w1',model[0].weight),('b1',model[0].bias),('w2',model[2].weight),
                ('b2',model[2].bias),('w3',model[4].weight),('b3',model[4].bias))})
            deployed=Predictor(path)
            rows=[dict(seed=g['seed'],**ranking_metric(g,deployed.predict(g[feature_key],g['prior']))) for g in validation]
            metric=seed_mean(rows,'regret')
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)) if losses else None,
                validation_regret=metric,validation_near=seed_mean(rows,'near')))
            if best is None or metric<best[0]:best=(metric,path,epoch)
    write_json(out/f'{name}_training.json',dict(groups=len(groups),validation_groups=len(validation),
        feature_dim=len(mean),epochs=epochs,selected_epoch=best[2],selected=str(best[1]),history=history))
    return Predictor(best[1]),dict(checkpoint=str(best[1]),sha256=sha(best[1]),epoch=best[2])


def collect_groups(rows,out,role):
    import torch
    groups=[];cost=[]
    for row in rows:
        p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs'])
        roots={}
        def observe(info):
            # Iterations fixed before outcome inspection; duplicates stay marked.
            if info['iteration'] in (1,5,13):roots[info['iteration']]=info['base'].actions[0].copy()
        solve_ranked(p,draft,observer=observe)
        phy=ProbePhy(p)
        try:
            stamp=time.perf_counter();before=phy.stats;phy.prepare_bank()
            bank_ms=(time.perf_counter()-stamp)*1000
            bank_misses=phy.stats['mmse_misses']-before['mmse_misses']
            floor=raw_pfq_floor(phy.evaluate(draft).values[0])
            context=EditFeatureContext(p,phy)
            try:
                for iteration,action in roots.items():
                    base=phy.evaluate(action);actions=greedy_candidates(p,action)
                    stamp=time.perf_counter();x48,prior=candidate_features(p,context,base,actions,floor=floor)
                    old_ms=(time.perf_counter()-stamp)*1000
                    stamp=time.perf_counter();spatial,upper,total,_=phy.features(base,actions)
                    spatial_ms=(time.perf_counter()-stamp)*1000
                    exact=phy.evaluate(actions,base)
                    gain=exact.values[:,:,EXPECTED]@p.weights-base.values[0,:,EXPECTED]@p.weights
                    eligible=exact.values[:,:,DEQUEUE].sum(1)>=floor
                    np.testing.assert_array_equal(total,exact.values[:,:,DEQUEUE].sum(1))
                    assert np.all(upper+1.e-7>=exact.values[:,:,EXPECTED]@p.weights)
                    groups.append(dict(state_id=identity(row),seed=row['seed'],iteration=iteration,
                        x48=x48,x120=np.concatenate((x48,spatial),1),prior=prior,gain=gain,
                        eligible=eligible,root=action,floor=floor))
                    cost.append(dict(state_id=identity(row),seed=row['seed'],iteration=iteration,
                        candidates=len(actions),old_feature_ms=old_ms,spatial_feature_ms=spatial_ms,
                        bank_ms=bank_ms,bank_mmse_misses=bank_misses))
            finally:context.close()
        finally:phy.close()
        print('COLLECT',role,identity(row),len(groups),flush=True)
    torch.save(groups,out/f'{role}_groups.pt')
    return groups,cost


def certified_search(p,draft,order='bound',ranker=None):
    """Exact best-improvement path, pruning solely by conservative bounds."""
    phy=ProbePhy(p);context=None
    try:
        base=phy.evaluate(draft);floor=raw_pfq_floor(base.values[0]);history=[]
        feature_ms=network_ms=0.;pruned=0
        if order!='bound':
            if order=='learned':phy.prepare_bank()
            context=EditFeatureContext(p,phy)
        for iteration in range(1,257):
            actions=greedy_candidates(p,base.actions[0]);stamp=time.perf_counter()
            spatial,upper,total,prior=phy.features(base,actions,order=='learned')
            valid=total>=floor
            if order!='bound':
                old,prior=candidate_features(p,context,base,actions,floor=floor)
            feature_ms+=(time.perf_counter()-stamp)*1000
            stamp=time.perf_counter()
            features=np.concatenate((old,spatial),1) if order=='learned' else (old if order=='scalar' else None)
            pred=upper if order=='bound' else ranker.predict(features,prior)
            indices=np.flatnonzero(valid);indices=indices[np.argsort(-pred[indices],kind='stable')]
            network_ms+=(time.perf_counter()-stamp)*1000
            score=float(base.values[0,:,EXPECTED]@p.weights);best_score=score;best_index=-1;best=None
            scored=0
            # Match full DPLS's [original-neighborhood, UE, field] reduction
            # shape and row positions. NumPy may accumulate a short strided
            # matrix-vector product differently; mathematical ties otherwise
            # change with the evaluation batch. This buffer contains no PHY
            # work for skipped candidates, and is charged inside the timer.
            score_buffer=np.zeros((len(actions),p.U,8),np.float64)
            for offset in range(0,len(indices),32):
                batch=indices[offset:offset+32]
                live=batch[upper[batch]>=best_score] # equality evaluated to preserve native ties
                pruned+=len(batch)-len(live)
                if not len(live):continue
                evaluated=phy.evaluate(actions[live],base);scored+=len(live)
                score_buffer[live]=evaluated.values
                js=(score_buffer[:,:,EXPECTED]@p.weights)[live]
                maximum=float(js.max());tied=live[js==maximum];chosen=int(tied.min())
                if maximum>best_score or (maximum==best_score and (best_index<0 or chosen<best_index)):
                    best_score=maximum;best_index=chosen
                    best=phy.take(evaluated,int(np.flatnonzero(live==chosen)[0]))
            take=best_index>=0 and best_score-score>(1 if iteration<=2 else 16)
            history.append(dict(iteration=iteration,best_index=best_index if take else -1,scored=scored))
            if take:base=best
            if iteration>=3 and not take:break
        else:raise RuntimeError('Certified search did not converge')
        record=dict(score=float(base.values[0,:,EXPECTED]@p.weights),floor=floor,rounds=len(history),
            feature_ms=feature_ms,network_ms=network_ms,certified_pruned=pruned,history=history,**phy.stats)
        return base.actions[0].copy(),base.values[0].copy(),record
    finally:
        if context is not None:context.close()
        phy.close()


def local_domains(p,graph,base,prg):
    domains=[]
    for c in range(p.C):
        choices=graph.options(base.actions[0],c,prg,2,base.values[0])
        domains.append(tuple(dict.fromkeys(choices)))
    return domains


def local_actions(p,root,prg,domains):
    from itertools import product
    choices=np.array(list(product(*domains)),np.int64)
    result=np.broadcast_to(root,(len(choices),p.C,p.P)).copy()
    result[:,:,prg]=choices
    return result


def block_features(p,graph,base):
    """17 small block descriptions, never a Cartesian candidate feature pass."""
    result=[]
    for prg in range(p.P):
        features=[]
        for c,domain in enumerate(local_domains(p,graph,base,prg)):
            # Three active representatives: incumbent and at most two alternatives.
            users=list(dict.fromkeys([int(base.actions[0,c,prg]),*domain]))
            users=[u for u in users if u>=0][:3]
            for index in range(3):
                if index>=len(users):features.extend([0.]*11);continue
                u=users[index];v=base.values[0,u]
                features.extend([1,np.log1p(p.queue[u]/3000),p.weights[u]/4,p.mcs[u]/27,
                    v[COUNT]/p.P,max(v[LAYER],0)/4,v[DEQUEUE]/3000,max(v[BLER],0),
                    v[EXPECTED]/3000,np.log1p(graph.quality[u,prg]),p.caps[u,prg]/4])
        result.append(features)
    return np.asarray(result,np.float32)


def collect_block_groups(rows,groups,out,role):
    import torch
    by_state={}
    for g in groups:by_state.setdefault(g['state_id'],[]).append(g)
    labels=[];teacher_candidates=0
    for row in rows:
        p=problem_from_raw_row(row);phy=ProbePhy(p)
        try:
            graph=ProposalGraph(phy)
            for g in by_state[identity(row)]:
                base=phy.evaluate(g['root']);x=block_features(p,graph,base)
                gains=[];prior=[]
                for prg in range(p.P):
                    actions=local_actions(p,g['root'],prg,local_domains(p,graph,base,prg))
                    exact=phy.evaluate(actions,base);teacher_candidates+=len(actions)
                    j=exact.values[:,:,EXPECTED]@p.weights
                    ok=exact.values[:,:,DEQUEUE].sum(1)>=g['floor']
                    gain=max(0,float(j[ok].max()-base.values[0,:,EXPECTED]@p.weights))
                    gains.append(gain)
                    # Static interference/TB risk priority, no counterfactual PHY.
                    users=g['root'][:,prg];users=users[users>=0]
                    prior.append(float((p.weights[users]*base.values[0,users,DEQUEUE]*
                        np.maximum(base.values[0,users,BLER],0)).sum()))
                labels.append(dict(state_id=g['state_id'],seed=g['seed'],iteration=g['iteration'],
                    xblock=x,prior=np.asarray(prior),gain=np.asarray(gains),eligible=np.ones(p.P,bool)))
        finally:phy.close()
    torch.save(labels,out/f'{role}_blocks.pt')
    return labels,teacher_candidates


def block_search(p,draft,mode,model=None,budget=4096):
    phy=ProbePhy(p)
    try:
        base=phy.evaluate(draft);floor=raw_pfq_floor(base.values[0]);graph=ProposalGraph(phy)
        score=float(base.values[0,:,EXPECTED]@p.weights);rng=np.random.default_rng(20261102)
        covered=set();blocks=0;accepted=0;feature_ms=network_ms=0
        for sweep in range(4):
            stamp=time.perf_counter();x=block_features(p,graph,base)
            priority=[]
            for prg in range(p.P):
                u=base.actions[0,:,prg];u=u[u>=0]
                priority.append(float((p.weights[u]*base.values[0,u,DEQUEUE]*np.maximum(base.values[0,u,BLER],0)).sum()))
            feature_ms+=(time.perf_counter()-stamp)*1000
            stamp=time.perf_counter()
            order=(rng.permutation(p.P) if mode=='random' else
                np.argsort(-(np.asarray(priority) if mode=='heuristic' else model.predict(x,np.asarray(priority))),kind='stable'))
            network_ms+=(time.perf_counter()-stamp)*1000
            updates=0
            for prg in order:
                remaining=budget-phy.stats['candidate_evaluations']
                actions=local_actions(p,base.actions[0],int(prg),local_domains(p,graph,base,int(prg)))
                # Never spend a partial block's budget pretending to solve it exactly.
                if len(actions)>remaining:break
                exact=phy.evaluate(actions,base);covered.add(int(prg));blocks+=1
                js=exact.values[:,:,EXPECTED]@p.weights;ok=exact.values[:,:,DEQUEUE].sum(1)>=floor
                best=int(np.argmax(np.where(ok,js,-np.inf)))
                if js[best]-score>(1 if sweep==0 else 16):
                    base=phy.take(exact,best);score=float(js[best]);accepted+=1;updates+=1
            if updates==0 or phy.stats['candidate_evaluations']>=budget-64:break
        return base.actions[0].copy(),base.values[0].copy(),dict(score=score,floor=floor,
            covered_prgs=len(covered),scored_blocks=blocks,accepted=accepted,feature_ms=feature_ms,
            network_ms=network_ms,**phy.stats)
    finally:phy.close()


def generate(p,draft,mode,ranker=None,width=7,depth=8):
    """PFQ plus seven residual paths; at most eight accepted edits each.

    All proposal features and PHY validation are paid online. No full-search
    or teacher call. Candidate collection is an independently usable generator.
    """
    phy=ProbePhy(p);context=EditFeatureContext(p,phy)
    try:
        initial=phy.evaluate(draft);floor=raw_pfq_floor(initial.values[0]);frontier=[initial]
        if mode=='spatial':phy.prepare_bank()
        accepted=0;feature_ms=network_ms=0
        for step in range(depth):
            children=[]
            for base in frontier:
                actions=greedy_candidates(p,base.actions[0]);stamp=time.perf_counter()
                x,prior=candidate_features(p,context,base,actions,floor=floor)
                if mode=='spatial':
                    spatial,_,total,_=phy.features(base,actions)
                    x=np.concatenate((x,spatial),1)
                else:total=context.last_dequeue_totals
                feature_ms+=(time.perf_counter()-stamp)*1000
                stamp=time.perf_counter()
                prediction=prior if mode=='heuristic' else ranker.predict(x,prior)
                order=np.lexsort((np.arange(len(actions)),-prediction))
                count=width if len(frontier)==1 else 2
                chosen=order[total[order]>=floor][:count]
                network_ms+=(time.perf_counter()-stamp)*1000
                if len(chosen):
                    exact=phy.evaluate(actions[chosen],base)
                    for i in range(len(exact.actions)):
                        # Each residual path keeps its verified parent if the edit hurts.
                        if exact.values[i,:,EXPECTED]@p.weights>base.values[0,:,EXPECTED]@p.weights+1:
                            children.append(phy.take(exact,i));accepted+=1
                children.append(base)
            unique={v.actions.tobytes():v for v in children}
            candidates=sorted(unique.values(),key=lambda v:-float(v.values[0,:,EXPECTED]@p.weights))
            # Retain distinct complete actions by exact J; no service-diversity guarantee.
            frontier=candidates[:width]
        pool={draft.tobytes():initial}
        for value in frontier:pool[value.actions.tobytes()]=value
        values=list(pool.values())[:width+1]
        best=max(values,key=lambda v:float(v.values[0,:,EXPECTED]@p.weights))
        root_j=float(initial.values[0,:,EXPECTED]@p.weights)
        return best.actions[0].copy(),best.values[0].copy(),dict(score=float(best.values[0,:,EXPECTED]@p.weights),
            floor=floor,pool_size=len(values),accepted_edits=accepted,feature_ms=feature_ms,
            network_ms=network_ms,initial_score=root_j,full_search_calls=0,**phy.stats)
    finally:context.close();phy.close()


def timed(row,method,models):
    begin=time.perf_counter();p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs'])
    if method=='full':action,values,record=solve_ranked(p,draft)
    elif method.startswith('cert_'):
        action,values,record=certified_search(p,draft,method[5:],models['spatial'])
    elif method.startswith('block_'):
        action,values,record=block_search(p,draft,method[6:],models['block'])
    elif method.startswith('gen_'):
        mode=method[4:];action,values,record=generate(p,draft,mode,models.get(mode))
    else:raise ValueError(method)
    record['e2e_ms']=(time.perf_counter()-begin)*1000
    record.update(state_id=identity(row),seed=row['seed'],family=row['family'],tti=row['tti'],arm=method)
    return action,values,record


def freeze_protocol(out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(date='2026-11-02',version=2,ctbs=False,headroom_masks=False,ppo=False,
        objective='canonical native-BLER J with causal deficit weights; not equal-weight PPO',
        floor='ceil(90% own unpruned raw PFQ physical dequeue)',training_seeds=[41,42,43],
        validation_seeds=[44,45],training_sources=['raw_global_deficit','pfq'],
        training_ttis=[4001,8001,18000],root_iterations=[1,5,13],
        evaluation_seeds=[46,47,48],evaluation_sources=['pfq'],evaluation_ttis=[3001,4001,5000],
        historical_development=True,final_seeds_not_read=[311,312,313,314,315],
        epochs=30,checkpoints=[0,10,20,30],model='32/16 ReLU residual MLP, same optimizer and lists',
        selection='seed-equal validation regret@8, earliest epoch on ties, frozen before evaluation reads',
        input_probe='48 scalar features vs 48+72 causal affected-UE spatial/TB features; gamma bank fully charged',
        near_optimal='within max(16 bytes,0.5% best positive gain)',
        block_probe='random/physical-risk/learned ordering; max4096 actual candidate scores, max4 sweeps; 2 alternatives/cell',
        bound_probe='BLER in [0,1] upper certificate; batches32; stable original-index ties and thresholds; action parity required',
        numerical_ties='Sparse PHY scores reduced in original full-neighborhood shape and row positions; preparation v1 retained',
        generator_probe='PFQ plus <=7 verified residual paths, depth8, <=2 proposals/path after first expansion; no teacher/PPO',
        methods=['full','cert_bound','cert_learned','block_random','block_heuristic','block_learned',
                 'gen_heuristic','gen_scalar','gen_spatial'],repeats=3,
        timing='serial raw numeric adapter/PFQ decode through fresh PHY/graph/bank/features/network/candidates/verifier/cleanup',
        excludes='archive IO, model/library loading, first immutable TBS table construction; offline teacher work separately counted',
        quality_gate='all evaluation states J>=99.5% full; floor and independent PHY valid',
        cost_gate='seed-equal median complete time <=80% full; empirical state P95 reported',
        learning_gate='nonzero trained checkpoint and improvement over matching heuristic under shared implementation',
        input_gate='spatial vs scalar lower seed-equal regret@8 in every development seed; near-optimal hit not lower',
        native_gate='no native rollout in this pilot; saved-state gates alone cannot promote scheduler',
        registry_sha256=sha(ROOT/'training/stageb_hgraph/current_algorithm.json'),platform=platform.platform())
    write_json(out/'protocol.json',protocol)
    sources={}
    for path in Path(__file__).parent.iterdir():
        if path.suffix in ('.py','.cpp','.so','.rst'):
            target=out/'source'/path.name;target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path,target);sources[path.name]=sha(path)
    write_json(out/'source_manifest.json',sources)
    return protocol


def csv_write(path,rows):
    import csv
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x') as stream:
        w=csv.DictWriter(stream,keys);w.writeheader();w.writerows(rows)


def main(output):
    import torch
    torch.set_num_threads(1)
    out=Path(output).resolve();protocol=freeze_protocol(out);started=time.perf_counter()
    training,train_files=read_rows((41,42,43));validation,val_files=read_rows((44,45))
    training=[r for r in training if r['tti'] in protocol['training_ttis']]
    validation=[r for r in validation if r['tti'] in protocol['training_ttis']]
    write_json(out/'input_manifest.json',dict(training=train_files,validation=val_files,
        training_state_ids=[identity(r) for r in training],validation_state_ids=[identity(r) for r in validation]))
    # Immutable table and dynamic libraries only; never prewarm a decision's gamma bank.
    p=problem_from_raw_row(training[0]);probe_library()
    print('PHASE training inputs',flush=True)
    train_groups,train_cost=collect_groups(training,out,'training')
    val_groups,val_cost=collect_groups(validation,out,'validation')
    scalar,scalar_selection=train_model(train_groups,val_groups,out,'scalar','x48')
    spatial,spatial_selection=train_model(train_groups,val_groups,out,'spatial','x120')
    print('PHASE block teacher',flush=True)
    train_blocks,tc=collect_block_groups(training,train_groups,out,'training')
    val_blocks,vc=collect_block_groups(validation,val_groups,out,'validation')
    block,block_selection=train_model(train_blocks,val_blocks,out,'block','xblock')
    selection=dict(scalar=scalar_selection,spatial=spatial_selection,block=block_selection,
        evaluation_unread_at_selection=True,protocol_sha256=sha(out/'protocol.json'))
    write_json(out/'selection.json',selection)
    csv_write(out/'feature_cost.csv',train_cost+val_cost)
    print('SELECTION FROZEN',selection,flush=True)
    evaluation,eval_files=read_rows((46,47,48),evaluation=True)
    evaluation=[r for r in evaluation if r['tti'] in protocol['evaluation_ttis']]
    write_json(out/'evaluation_input_manifest.json',dict(files=eval_files,state_ids=[identity(r) for r in evaluation]))
    eval_groups,eval_cost=collect_groups(evaluation,out,'evaluation')
    csv_write(out/'evaluation_feature_cost.csv',eval_cost)
    rank_rows=[]
    for g in eval_groups:
        for name,model,key in [('heuristic',None,None),('scalar',scalar,'x48'),('spatial',spatial,'x120')]:
            pred=g['prior'] if model is None else model.predict(g[key],g['prior'])
            rank_rows.append(dict(state_id=g['state_id'],seed=g['seed'],iteration=g['iteration'],arm=name,
                **ranking_metric(g,pred)))
    csv_write(out/'ranking.csv',rank_rows)
    models=dict(scalar=scalar,spatial=spatial,block=block)
    records=[];actions={};audits=[];references={}
    for state_index,row in enumerate(evaluation):
        for repeat in range(protocol['repeats']):
            methods=protocol['methods'];shift=(state_index+repeat)%len(methods)
            for method in methods[shift:]+methods[:shift]:
                action,values,record=timed(row,method,models)
                record.pop('history',None);record['repeat']=repeat;records.append(record)
                key=f'{identity(row)}|{method}'
                if key in actions:np.testing.assert_array_equal(actions[key],action)
                else:actions[key]=action
                if repeat==0:
                    independent=ReferencePhy(problem_from_raw_row(row)).evaluate(action)[0]
                    np.testing.assert_allclose(values,independent,rtol=2.e-9,atol=2.e-6)
                    assert values[:,DEQUEUE].sum()>=record['floor']
                    audits.append(dict(state_id=identity(row),arm=method,max_error=float(np.abs(independent-values).max())))
                if method=='full':references[identity(row)]=dict(score=record['score'],values=values)
                print('EVAL',identity(row),repeat,method,round(record['score'],2),round(record['e2e_ms'],2),flush=True)
    csv_write(out/'measurements.csv',records);np.savez_compressed(out/'actions.npz',**actions)
    medians=[]
    for row in evaluation:
        for method in protocol['methods']:
            rr=[r for r in records if r['state_id']==identity(row) and r['arm']==method]
            reference=references[identity(row)]
            action=actions[f'{identity(row)}|{method}']
            p=problem_from_raw_row(row);v=ReferencePhy(p).evaluate(action)[0]
            good=reference['values'][:,EXPECTED]>0
            weak_drop=int((v[good,EXPECTED]<.98*reference['values'][good,EXPECTED]).sum())
            medians.append(dict(state_id=identity(row),seed=row['seed'],arm=method,
                score=rr[0]['score'],ratio=rr[0]['score']/reference['score'],e2e_ms=float(np.median([r['e2e_ms'] for r in rr])),
                candidate_evaluations=rr[0]['candidate_evaluations'],ue_evaluations=rr[0]['ue_evaluations'],
                mmse_misses=rr[0]['mmse_misses'],weak_ue_expected_drops_gt2pct=weak_drop,
                exact_full_action=bool(np.array_equal(action,actions[f'{identity(row)}|full'])),
                covered_prgs=rr[0].get('covered_prgs'),pool_size=rr[0].get('pool_size')))
    csv_write(out/'per_state.csv',medians)
    full=[r for r in medians if r['arm']=='full'];summary=[]
    for method in protocol['methods']:
        rr=[r for r in medians if r['arm']==method]
        quality=bool(all(r['ratio']>=.995 for r in rr));cost=seed_mean(rr,'e2e_ms')/seed_mean(full,'e2e_ms')
        summary.append(dict(arm=method,mean_j=seed_mean(rr,'score'),mean_ms=seed_mean(rr,'e2e_ms'),
            state_p95_ms=float(np.quantile([r['e2e_ms'] for r in rr],.95)),time_ratio=cost,
            mean_relative_j_pct=(seed_mean(rr,'ratio')-1)*100,worst_relative_j_pct=(min(r['ratio'] for r in rr)-1)*100,
            mean_candidates=seed_mean(rr,'candidate_evaluations'),quality_pass=quality,cost_pass=cost<=.8,
            joint_pass=quality and cost<=.8,exact_actions=sum(r['exact_full_action'] for r in rr)))
    csv_write(out/'summary.csv',summary)
    rank_summary=[]
    for s in (46,47,48):
        for name in ('heuristic','scalar','spatial'):
            rr=[r for r in rank_rows if r['seed']==s and r['arm']==name]
            rank_summary.append(dict(seed=s,arm=name,regret=float(np.mean([r['regret'] for r in rr])),
                near=float(np.mean([r['near'] for r in rr])),retained=float(np.mean([r['retained'] for r in rr]))))
    csv_write(out/'ranking_per_seed.csv',rank_summary)
    bound=[r for r in medians if r['arm'].startswith('cert_')]
    assert all(r['exact_full_action'] for r in bound)
    assert sha(ROOT/'training/stageb_hgraph/current_algorithm.json')==protocol['registry_sha256']
    write_json(out/'checks.json',dict(engineering_pass=True,certified_action_parity=True,
        final_phy_audits=audits,repeats_action_identical=True,ctbs=False,ppo=False,native_trajectories=0,
        final_seeds_read=False,registry_unchanged=True,selection_sha256=sha(out/'selection.json'),
        offline_block_teacher_candidates=tc+vc,finite_state_gates_only=True))
    write_json(out/'completion.json',dict(completed=True,states=len(evaluation),decisions=len(records),
        elapsed_s=time.perf_counter()-started,source_manifest_sha256=sha(out/'source_manifest.json')))
    print('COMPLETE',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output');parser.add_argument('--build',action='store_true')
    args=parser.parse_args()
    if args.build:build_kernel()
    if args.output:main(args.output)

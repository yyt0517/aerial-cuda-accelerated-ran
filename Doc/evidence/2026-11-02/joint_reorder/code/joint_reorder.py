"""Wider joint-condition permutations and exact-observed service frontiers."""
from dataclasses import dataclass
import ctypes
from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess
import time
import numpy as np

from .resource_conditions import Conditions,ConditionalPhy,satisfies,edges,seed_action,DecodeInfeasible
from .resource_conditions_fast import ordered_unique
from .physics import COUNT,DEQUEUE,EXPECTED,curves
from .search import ProposalGraph
from .tb_macro import ExactMemo
from .no_ctbs_inputs import raw_pfq_floor


def build_kernel():
    d=Path(__file__).parent
    subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-ffp-contract=off','-shared','-fPIC',
        str(d/'joint_reorder_kernel.cpp'),'-o',str(d/'_joint_reorder.so')],check=True)


@lru_cache(maxsize=1)
def library():
    lib=ctypes.CDLL(str(Path(__file__).with_name('_joint_reorder.so')))
    i=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS');d=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    b=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
    lib.joint_reorders.argtypes=[ctypes.c_int]*3+[i]*4+[b,b,ctypes.c_int,i]+[ctypes.c_int]*2+[i,i]
    lib.joint_reorders.restype=ctypes.c_int
    lib.joint_delta.argtypes=[ctypes.c_int]*4+[i]*5+[b]+[d]*6+[b,i]
    lib.joint_delta.restype=ctypes.c_int
    return lib


class JointFeatures:
    def __init__(self,p,bank,condition):
        if condition.counts is None or condition.activity is None:raise ValueError('Joint features require both conditions')
        self.p=p;self.condition=condition;self.lib=library();self.legal=np.ascontiguousarray(p.legal,np.uint8)
        self.activity=np.ascontiguousarray(condition.activity,np.uint8)
        self.gamma=np.ascontiguousarray(edges(p,bank,condition.activity,condition.layers))
        self.threshold=np.array([xs[np.argmin(abs(ys-.5))] for xs,ys in (curves()[int(m)] for m in p.mcs)])
        self.gamma_reads=0;self.attempts=0

    def pool(self,base,wide,seed):
        p=self.p;condition=self.condition;K=8;orders=np.full((p.C,K,p.P),-1,np.int64)
        digest=int.from_bytes(hashlib.sha256(base.actions[0].tobytes()).digest()[:8],'little')
        rng=np.random.default_rng((digest+seed)%(1<<64))
        for c in range(p.C):
            prgs=np.flatnonzero(condition.activity[c]);n=len(prgs)
            q=self.gamma[base.actions[0,c,prgs],prgs]
            weak=prgs[np.argsort(q,kind='stable')]
            for k,values in enumerate([prgs,prgs[::-1],weak,weak[::-1],*[rng.permutation(prgs) for _ in range(4)]]):orders[c,k,:n]=values
        capacity=p.C*(p.P*(p.P-1)//2+2*p.P*(p.P-1)*(p.P-2)//6+K*p.P*3)
        output=np.empty((max(capacity,1),p.C,p.P),np.int64);stats=np.zeros(2,np.int64)
        n=self.lib.joint_reorders(p.U,p.C,p.P,base.actions[0],condition.counts,condition.layers,p.caps,self.legal,
            self.activity,int(wide),orders,K,len(output),output,stats)
        if n<0:raise RuntimeError('Joint permutation enumeration failed')
        self.attempts+=int(stats[0]);return output[:n].copy()

    def proxy(self,base,actions):
        p=self.p;a=p.actions(actions);score=np.empty(len(a));predicted=np.empty((len(a),p.U));valid=np.empty(len(a),np.uint8);stats=np.zeros(1,np.int64)
        code=self.lib.joint_delta(len(a),p.U,p.C,p.P,a,base.actions[0],self.condition.counts,self.condition.layers,
            p.caps,self.legal,self.gamma,self.threshold,p.weights,base.values[0],score,predicted,valid,stats)
        if code:raise RuntimeError('Joint delta proxy failed')
        self.gamma_reads+=int(stats[0]);return score,predicted,valid.astype(bool)


def service_vector(values,group):
    e=np.asarray(values)[...,group,EXPECTED]
    if not group.any():return np.zeros(e.shape[:-1]+(2,))
    weak=max(1,int(np.ceil(.2*group.sum())))
    return np.stack([np.quantile(e,.05,axis=-1),np.sort(e,axis=-1)[...,:weak].sum(-1)],axis=-1)


def nondominated(keys,objectives):
    kept=[]
    for key in keys:
        value=objectives[key]
        if any(np.all(objectives[k]>=value) and np.any(objectives[k]>value) for k in kept):continue
        kept=[k for k in kept if not (np.all(value>=objectives[k]) and np.any(value>objectives[k]))]
        # Equal endpoints retain their first complete-action representative.
        if not any(np.array_equal(objectives[k],value) for k in kept):kept.append(key)
    return kept


def frontier_roots(ordered,objectives,width,exploration_retention=.95):
    if not ordered:return []
    best_j=max(objectives[k][0] for k in ordered);near=[k for k in ordered if objectives[k][0]>=exploration_retention*best_j]
    front=nondominated(near,objectives);selected=[ordered[0]]
    for axis in (1,2):
        if front:
            key=max(front,key=lambda k:(objectives[k][axis],objectives[k][0]))
            if key not in selected:selected.append(key)
    for key in ordered:
        if len(selected)>=width:break
        if key in front and key not in selected:selected.append(key)
    for key in ordered:
        if len(selected)>=width:break
        if key not in selected:selected.append(key)
    return selected[:width]


def final_key(keys,objectives,loss):
    if not keys:raise ValueError('No feasible selection')
    best=max(keys,key=lambda k:objectives[k][0])
    if loss==0:return best
    allowed=[k for k in keys if objectives[k][0]+1.e-8>=(1-loss)*objectives[best][0]]
    return max(allowed,key=lambda k:(objectives[k][1],objectives[k][2],objectives[k][0]))


@dataclass(frozen=True)
class ReorderConfig:
    wide:bool=True
    pareto:bool=False
    final_j_loss:float=0.
    max_candidates:int=212
    max_feature_roots:int=56
    rounds:int=56
    seed_candidates:int=16
    batch_size:int=8
    beam_width:int=4
    seed:int=20261102

    def __post_init__(self):
        if not 0<=self.final_j_loss<=.05 or min(self.max_candidates,self.max_feature_roots,self.rounds,self.seed_candidates,self.batch_size,self.beam_width)<1:
            raise ValueError('Invalid joint reorder configuration')


def solve_joint(p,draft,condition,config=ReorderConfig(),*,offline_diagnostic=False):
    if not offline_diagnostic:raise ValueError('Compressed teacher conditions require offline diagnostic mode')
    condition=condition.validate(p)
    if condition.counts is None or condition.activity is None:raise ValueError('Joint conditions required')
    phy=ConditionalPhy(p)
    try:
        memo=ExactMemo(phy,draft,config.max_candidates);original=memo.root;floor=raw_pfq_floor(original.values[0]);graph=ProposalGraph(phy)
        stamp=time.perf_counter();bank=phy.bank();bank_ms=(time.perf_counter()-stamp)*1000
        features=JointFeatures(p,bank,condition);rng=np.random.default_rng(config.seed);archive={};expanded={};roots=set()
        matching_ms=proxy_ms=generation_ms=service_ms=0.;proxy_candidates=decode_failures=seed_calls=0;stop='round_limit'
        def store(exact):
            if exact is None:return
            for j in np.flatnonzero(satisfies(p,exact.actions,condition)):
                archive[exact.actions[j].tobytes()]=phy.take(exact,int(j))
        store(original);seeds=[];seen=set(memo.rows);stamp=time.perf_counter()
        for k in range(config.seed_candidates):
            try:a=seed_action(p,original,bank,graph,condition,rng,k);seed_calls+=1
            except DecodeInfeasible:decode_failures+=1;continue
            if a.tobytes() not in seen:seeds.append(a);seen.add(a.tobytes())
        matching_ms+=(time.perf_counter()-stamp)*1000
        if seeds:exact,_=memo.evaluate(np.stack(seeds),original);store(exact)
        # Fixed after paid bootstrap: union of its observed >1B effective UEs.
        # No teacher service mask or service amount is consulted.
        group=np.zeros(p.U,bool)
        for value in archive.values():group |= value.values[0,:,EXPECTED]>1.
        objectives={}
        def refresh():
            nonlocal service_ms
            stamp=time.perf_counter()
            for k,value in archive.items():
                if k not in objectives:
                    v=value.values[0];objectives[k]=np.r_[float(v[:,EXPECTED]@p.weights),service_vector(v,group)]
            service_ms+=(time.perf_counter()-stamp)*1000
        refresh();iterations=0;frontier_peak=0
        for iteration in range(config.rounds):
            if memo.remaining<=0:stop='physical_budget';break
            candidates=sorted(archive,key=lambda k:(int(archive[k].values[0,:,DEQUEUE].sum()>=floor),objectives[k][0],
                float(archive[k].values[0,:,DEQUEUE].sum())),reverse=True)
            if not candidates:stop='no_conditional_seed';break
            available=[k for k in candidates if expanded.get(k,0)>=0]
            stamp=time.perf_counter()
            usable=frontier_roots(available,objectives,config.beam_width) if config.pareto else available[:config.beam_width]
            if config.pareto:frontier_peak=max(frontier_peak,len(nondominated(available,objectives)))
            service_ms+=(time.perf_counter()-stamp)*1000
            if not usable:stop='finite_pool_stalled';break
            key=usable[0] if iteration%2==0 else min(usable,key=lambda k:expanded.get(k,0));base=archive[key]
            if key not in roots and len(roots)>=config.max_feature_roots:stop='feature_budget';break
            roots.add(key);expanded[key]=expanded.get(key,0)+1;iterations+=1
            stamp=time.perf_counter();actions=features.pool(base,config.wide,config.seed);generation_ms+=(time.perf_counter()-stamp)*1000
            regenerated=[];stamp=time.perf_counter()
            if iteration%4==0:
                for j in range(2):
                    try:regenerated.append(seed_action(p,base,bank,graph,condition,rng,3+(iteration+j)%9));seed_calls+=1
                    except DecodeInfeasible:decode_failures+=1
            matching_ms+=(time.perf_counter()-stamp)*1000
            if regenerated:actions=np.concatenate([actions,np.stack(regenerated)])
            if len(actions):actions=ordered_unique(actions);actions=actions[memo.unseen(actions)]
            if not len(actions):expanded[key]=-1;continue
            stamp=time.perf_counter();scores,predicted,valid=features.proxy(base,actions);proxy_candidates+=len(actions)
            ids=np.flatnonzero(valid)
            order=ids[np.argsort(-np.round(scores[ids],7),kind='stable')]
            # In frontier mode reserve2 of8 proposals for projected weak-service
            # improvements, inside95% root J; all are still exactly verified.
            selected=list(order[:min(config.batch_size,memo.remaining)])
            if config.pareto and len(selected)>=4 and group.any():
                service=np.stack([np.quantile(predicted[:,group],.05,axis=1),
                    np.sort(predicted[:,group],axis=1)[:,:max(1,int(np.ceil(.2*group.sum())))].sum(1)],1)
                near=ids[scores[ids]>=.95*objectives[key][0]]
                extra=[]
                for axis in (0,1):
                    if len(near):
                        candidate=int(near[np.argmax(service[near,axis])])
                        if candidate not in extra:extra.append(candidate)
                selected=list(order[:max(2,len(selected)-len(extra))])
                for j in [*extra,*order]:
                    if len(selected)>=min(config.batch_size,memo.remaining):break
                    if int(j) not in selected:selected.append(int(j))
            proxy_ms+=(time.perf_counter()-stamp)*1000
            if not selected:expanded[key]=-1;continue
            exact,_=memo.evaluate(actions[np.asarray(selected,np.int64)],base);store(exact);refresh()
        feasible=[k for k,v in archive.items() if v.values[0,:,DEQUEUE].sum()>=floor]
        if feasible:
            best_key=final_key(feasible,objectives,config.final_j_loss);best=archive[best_key]
            jbest=max(objectives[k][0] for k in feasible)
        else:best=original;jbest=float(original.values[0,:,EXPECTED]@p.weights)
        v=best.values[0];score=float(v[:,EXPECTED]@p.weights)
        assert phy.stats['candidate_evaluations']<=config.max_candidates and len(roots)<=config.max_feature_roots
        assert v[:,DEQUEUE].sum()>=floor and score+1.e-8>=(1-config.final_j_loss)*jbest
        if feasible:assert satisfies(p,best.actions,condition)[0]
        r=dict(score=score,floor=floor,best_observed_j=jbest,selected_j_loss_fraction=(jbest-score)/max(jbest,1),
            condition_feasible=bool(feasible),condition_satisfied=bool(satisfies(p,best.actions,condition)[0]),
            fallback_to_raw=not bool(feasible),feature_roots=len(roots),iterations=iterations,proxy_candidates=proxy_candidates,
            proposal_attempts=features.attempts,proxy_gamma_array_reads=features.gamma_reads,seed_gamma_array_reads=seed_calls*p.U*p.P,
            service_group_size=int(group.sum()),frontier_peak=frontier_peak,bank_ms=bank_ms,matching_ms=matching_ms,
            generation_ms=generation_ms,proxy_ms=proxy_ms,service_ms=service_ms,decode_failures=decode_failures,
            stop_reason=stop,offline_diagnostic=True,full_search_calls=0,**phy.stats)
        return best.actions[0].copy(),v.copy(),r,group.copy()
    finally:phy.close()


if __name__=='__main__':build_kernel()

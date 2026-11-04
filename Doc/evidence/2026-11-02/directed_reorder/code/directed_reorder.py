"""Budgeted weak-UE directed proposals on the frozen fast continuous frontier."""
from dataclasses import dataclass
import ctypes
from functools import lru_cache
from pathlib import Path
import subprocess
import time
import numpy as np
from .weak_service import ServiceConfig,ServiceMetric,FrontierCache
from .joint_reorder import JointFeatures,final_key
from .resource_conditions import ConditionalPhy,satisfies,seed_action,DecodeInfeasible
from .resource_conditions_fast import ordered_unique
from .physics import DEQUEUE,EXPECTED
from .search import ProposalGraph
from .tb_macro import ExactMemo
from .no_ctbs_inputs import raw_pfq_floor


@dataclass(frozen=True)
class DirectedConfig(ServiceConfig):
    metric:str='quota_exp'
    directed_slots:int=2
    joint_candidates:bool=True
    alternate_sparse:bool=False
    target_count:int=6
    chain_depth:int=4

    def __post_init__(self):
        super().__post_init__()
        if self.directed_slots not in (0,2,4) or not 1<=self.target_count<=6 or not 1<=self.chain_depth<=4:
            raise ValueError('Unsupported bounded directed configuration')
        if self.alternate_sparse and not self.joint_candidates:raise ValueError('Sparse rounds need directed candidates')


def build_kernel():
    d=Path(__file__).parent
    subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-ffp-contract=off','-shared','-fPIC',
        str(d/'directed_reorder_kernel.cpp'),'-o',str(d/'_directed_reorder.so')],check=True)


@lru_cache(maxsize=1)
def library():
    lib=ctypes.CDLL(str(Path(__file__).with_name('_directed_reorder.so')))
    i=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    d=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    b=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
    lib.directed_reorders.argtypes=[ctypes.c_int]*4+[i]*6+[b]+[d]*3+[ctypes.c_int]*2+[i,i]
    lib.directed_reorders.restype=ctypes.c_int
    return lib


def target_ues(p,values,group,limit=6):
    deq=values[:,DEQUEUE];ratio=np.divide(values[:,EXPECTED],deq,out=np.ones(p.U),where=deq>0)
    ids=np.flatnonzero(group&(deq>0)&(ratio<1-1.e-9))
    weak=ids[np.lexsort((ids,-p.weights[ids]*deq[ids],ratio[ids]))]
    deficit=ids[np.argsort(-p.weights[ids]*(deq[ids]-values[ids,EXPECTED]),kind='stable')]
    out=[]
    for rank in range(min(3,len(ids))):
        for order in (weak,deficit):
            u=int(order[rank])
            if u not in out:out.append(u)
    return np.asarray(out[:limit],np.int64)


class DirectedFeatures:
    def __init__(self,features):
        self.features=features;self.lib=library();self.attempts=self.gamma_reads=self.swap_steps=0
        self.combinations=self.candidates=self.multicell=0

    def pool(self,base,targets,depth):
        f=self.features;p=f.p;c=f.condition;output=np.empty((256,p.C,p.P),np.int64);stats=np.zeros(6,np.int64)
        n=self.lib.directed_reorders(p.U,p.C,p.P,len(targets),base.actions[0],p.serving,c.counts,c.layers,p.caps,
            np.ascontiguousarray(targets,np.int64),f.legal,f.gamma,base.values[0],p.weights,depth,len(output),output,stats)
        if n<0:raise RuntimeError(f'Directed enumeration failed:{n}')
        self.attempts+=int(stats[0]);self.gamma_reads+=int(stats[1]);self.swap_steps+=int(stats[2])
        self.combinations+=int(stats[3]);self.candidates+=int(stats[4]);self.multicell+=int(stats[5])
        return output[:n].copy()


def choose_batch(scores,predicted,valid,base,metric,targets,remaining,config,iteration):
    """Preserve J and continuous-service proposals, then reserve directed slots.

    Only root-causal projected gains guide the reservation. Every selected
    action still consumes the same exact complete-schedule budget.
    """
    ids=np.flatnonzero(valid);order=ids[np.argsort(-np.round(scores[ids],7),kind='stable')]
    size=min(config.batch_size,remaining,len(order));selected=list(order[:size]);directed=[]
    if not config.pareto or size<4 or not metric.group.any():return selected,directed
    root_j=float(base.values[0,:,EXPECTED]@metric.weights)
    service=metric.expected(predicted);near=ids[scores[ids]>=.95*root_j];extras=[]
    for axis in (0,1):
        if len(near):
            candidate=int(near[np.argmax(service[near,axis])])
            if candidate not in extras:extras.append(candidate)
    # With zero reservations this is exactly the preceding implementation.
    reserve=min(config.directed_slots,max(0,size-2-len(extras)))
    selected=list(order[:max(2,size-len(extras)-reserve)])
    for j in extras:
        if j not in selected:selected.append(j)
    if reserve and len(targets) and len(near):
        rotated=np.roll(targets,-iteration%len(targets));deq=base.values[0,:,DEQUEUE]
        for u in rotated:
            gain=(predicted[near,u]-base.values[0,u,EXPECTED])/max(deq[u],1.e-12)
            ranking=np.lexsort((near,-scores[near],-np.round(gain,12)))
            for pos in ranking:
                j=int(near[pos])
                if gain[pos]>1.e-12 and j not in selected:
                    selected.append(j);directed.append(j);break
            if len(directed)>=reserve:break
    for j in order:
        if len(selected)>=size:break
        if int(j) not in selected:selected.append(int(j))
    return selected,directed


def solve_directed(p,draft,condition,config=DirectedConfig(),*,offline_diagnostic=False):
    if not offline_diagnostic:raise ValueError('Compressed teacher conditions require offline diagnostic mode')
    condition=condition.validate(p)
    if condition.counts is None or condition.activity is None:raise ValueError('Joint conditions required')
    phy=ConditionalPhy(p)
    try:
        memo=ExactMemo(phy,draft,config.max_candidates);original=memo.root
        floor=raw_pfq_floor(original.values[0]);graph=ProposalGraph(phy)
        stamp=time.perf_counter();bank=phy.bank();bank_ms=(time.perf_counter()-stamp)*1000
        features=JointFeatures(p,bank,condition);directed_features=DirectedFeatures(features);rng=np.random.default_rng(config.seed)
        archive={};expanded={};roots=set();pending=[];dequeues={};objectives={}
        cache=FrontierCache(config.max_candidates)
        matching_ms=proxy_ms=generation_ms=service_ms=0.
        proxy_candidates=decode_failures=seed_calls=0;stop='round_limit'
        directed_queries=directed_pool_queries=multicell_queries=wide_rounds=sparse_rounds=0;directed_generation_ms=0.

        def store(exact):
            if exact is None:return
            for j in np.flatnonzero(satisfies(p,exact.actions,condition)):
                key=exact.actions[j].tobytes()
                if key not in archive:
                    archive[key]=phy.take(exact,int(j));pending.append(key)
                    dequeues[key]=float(exact.values[j,:,DEQUEUE].sum())

        store(original);seeds=[];seen=set(memo.rows);stamp=time.perf_counter()
        for k in range(config.seed_candidates):
            try:a=seed_action(p,original,bank,graph,condition,rng,k);seed_calls+=1
            except DecodeInfeasible:decode_failures+=1;continue
            if a.tobytes() not in seen:seeds.append(a);seen.add(a.tobytes())
        matching_ms+=(time.perf_counter()-stamp)*1000
        if seeds:exact,_=memo.evaluate(np.stack(seeds),original);store(exact)
        bootstrap=np.zeros(p.U,bool)
        for value in archive.values():bootstrap|=value.values[0,:,EXPECTED]>1.
        metric=ServiceMetric(p,condition,bootstrap,config);metric.weights=p.weights;group=metric.group

        def refresh():
            nonlocal service_ms
            if not pending:return
            stamp=time.perf_counter()
            expected=np.stack([archive[k].values[0,:,EXPECTED] for k in pending])
            services=metric.expected(expected)
            for k,s in zip(pending,services):
                # Retain the original 1-D dot operation: batched BLAS changes
                # last-bit ties that can alter exact frontier decisions.
                j=float(archive[k].values[0,:,EXPECTED]@p.weights)
                objectives[k]=np.r_[j,s]
            cache.add(pending,objectives);pending.clear()
            service_ms+=(time.perf_counter()-stamp)*1000

        refresh();iterations=0;frontier_peak=0
        for iteration in range(config.rounds):
            if memo.remaining<=0:stop='physical_budget';break
            candidates=sorted(archive,key=lambda k:(int(dequeues[k]>=floor),objectives[k][0],dequeues[k]),reverse=True)
            if not candidates:stop='no_conditional_seed';break
            available=[k for k in candidates if expanded.get(k,0)>=0]
            stamp=time.perf_counter()
            if config.pareto:
                usable,peak=cache.roots(available,objectives,config.beam_width);frontier_peak=max(frontier_peak,peak)
            else:usable=available[:config.beam_width]
            service_ms+=(time.perf_counter()-stamp)*1000
            if not usable:stop='finite_pool_stalled';break
            key=usable[0] if iteration%2==0 else min(usable,key=lambda k:expanded.get(k,0));base=archive[key]
            if key not in roots and len(roots)>=config.max_feature_roots:stop='feature_budget';break
            roots.add(key);expanded[key]=expanded.get(key,0)+1;iterations+=1
            stamp=time.perf_counter();targets=target_ues(p,base.values[0],group,config.target_count)
            macro=directed_features.pool(base,targets,config.chain_depth) if config.joint_candidates else np.empty((0,p.C,p.P),np.int64)
            macro_keys={a.tobytes() for a in macro};directed_generation_ms+=(time.perf_counter()-stamp)*1000
            stamp=time.perf_counter()
            if config.alternate_sparse and iteration%2 and len(macro):actions=macro;sparse_rounds+=1
            else:
                actions=features.pool(base,config.wide,config.seed);wide_rounds+=1
                if len(macro):actions=np.concatenate([actions,macro])
            generation_ms+=(time.perf_counter()-stamp)*1000
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
            selected,dedicated=choose_batch(scores,predicted,valid,base,metric,targets,memo.remaining,config,iteration)
            directed_queries+=len(dedicated)
            directed_pool_queries+=sum(actions[j].tobytes() in macro_keys for j in selected)
            multicell_queries+=sum(int(np.any(actions[j]!=base.actions[0],axis=1).sum()>1) for j in selected)
            proxy_ms+=(time.perf_counter()-stamp)*1000
            if not selected:expanded[key]=-1;continue
            exact,_=memo.evaluate(actions[np.asarray(selected,np.int64)],base);store(exact);refresh()
        feasible=[k for k in archive if dequeues[k]>=floor]
        if feasible:
            best=archive[final_key(feasible,objectives,config.final_j_loss)]
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
            service_group_size=int(group.sum()),bootstrap_group_size=int(bootstrap.sum()),frontier_peak=frontier_peak,
            bank_ms=bank_ms,matching_ms=matching_ms,generation_ms=generation_ms,proxy_ms=proxy_ms,service_ms=service_ms,
            directed_queries=directed_queries,directed_pool_queries=directed_pool_queries,multicell_queries=multicell_queries,
            directed_generation_ms=directed_generation_ms,directed_candidates=directed_features.candidates,
            directed_attempts=directed_features.attempts,directed_gamma_reads=directed_features.gamma_reads,
            directed_swap_steps=directed_features.swap_steps,directed_combinations=directed_features.combinations,
            directed_multicell_candidates=directed_features.multicell,wide_rounds=wide_rounds,sparse_rounds=sparse_rounds,
            service_metric=config.metric,service_rows=metric.rows,dominance_pairs=cache.comparisons,frontier_queries=cache.queries,
            decode_failures=decode_failures,stop_reason=stop,offline_diagnostic=True,full_search_calls=0,**phy.stats)
        return best.actions[0].copy(),v.copy(),r,group.copy()
    finally:phy.close()


if __name__=='__main__':build_kernel()

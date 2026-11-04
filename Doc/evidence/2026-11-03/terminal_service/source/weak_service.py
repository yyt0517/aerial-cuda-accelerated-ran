"""Continuous quota-service objectives and equivalent batched frontier engine.

Only compressed offline counts/layers/activity enter this decoder. All metrics
use its own exact/proxy service; neither teacher service nor teacher UE positions
are available. The older joint_reorder implementation remains frozen.
"""
from dataclasses import dataclass
import time
import numpy as np

from .joint_reorder import JointFeatures,ReorderConfig,final_key,service_vector
from .resource_conditions import ConditionalPhy,satisfies,seed_action,DecodeInfeasible
from .resource_conditions_fast import ordered_unique
from .physics import DEQUEUE,EXPECTED
from .search import ProposalGraph
from .tb_macro import ExactMemo
from .no_ctbs_inputs import raw_pfq_floor


@dataclass(frozen=True)
class ServiceConfig(ReorderConfig):
    pareto:bool=True
    metric:str='bootstrap_p5'
    log_epsilon:float=.01
    tail_temperature:float=.1

    def __post_init__(self):
        super().__post_init__()
        if self.metric not in ('bootstrap_p5','quota_p5','quota_log','quota_exp'):
            raise ValueError('Unknown service objective')
        if self.log_epsilon<=0 or self.tail_temperature<=0:
            raise ValueError('Positive smoothing scales required')


class ServiceMetric:
    def __init__(self,p,condition,bootstrap,config):
        self.config=config
        self.dequeue=np.minimum(p.queue,p.tbs[np.arange(p.U),condition.counts,condition.layers])
        quota=(condition.counts>0)&(self.dequeue>0)
        self.group=np.asarray(bootstrap,bool).copy() if config.metric=='bootstrap_p5' else quota
        self.rows=0

    def expected(self,e):
        e=np.asarray(e,dtype=np.float64)
        self.rows+=int(np.prod(e.shape[:-1])) if e.ndim>1 else 1
        x=e[...,self.group]
        if not self.group.any():return np.zeros(x.shape[:-1]+(2,))
        if self.config.metric in ('bootstrap_p5','quota_p5'):
            weak=max(1,int(np.ceil(.2*self.group.sum())))
            return np.stack([np.quantile(x,.05,axis=-1),np.sort(x,axis=-1)[...,:weak].sum(-1)],axis=-1)
        ratio=np.clip(x/self.dequeue[self.group],0.,1.)
        if self.config.metric=='quota_log':
            eps=self.config.log_epsilon
            first=np.log1p(ratio/eps).mean(-1)/np.log1p(1/eps)
        else:
            tau=self.config.tail_temperature
            first=-tau*np.log(np.exp(-ratio/tau).mean(-1))
        return np.stack([first,ratio.mean(-1)],axis=-1)


class FrontierCache:
    """Cache pairwise dominance once, while honoring each query's stable ties.

    A dominated candidate may become nondominated after its dominator is marked
    exhausted. Keep the full relation matrix and query only expandable IDs;
    caching only the current frontier would incorrectly discard such candidates.
    """
    def __init__(self,capacity):
        self.capacity=capacity;self.ids={};self.n=0
        self.values=np.empty((capacity,3),np.float64)
        self.dominates=np.zeros((capacity,capacity),bool)
        self.equal=np.zeros((capacity,capacity),bool)
        self.comparisons=0;self.queries=0

    def add(self,keys,objectives):
        keys=[k for k in keys if k not in self.ids]
        if not keys:return
        old=self.n;new=old+len(keys)
        if new>self.capacity:raise ValueError('Frontier capacity exceeded')
        for i,k in enumerate(keys,old):self.ids[k]=i;self.values[i]=objectives[k]
        a=self.values[old:new,None,:];b=self.values[None,:new,:]
        ge=np.all(a>=b,axis=-1);le=np.all(a<=b,axis=-1);eq=np.all(a==b,axis=-1)
        self.dominates[old:new,:new]=ge&~eq
        self.dominates[:new,old:new]=(le&~eq).T
        self.equal[old:new,:new]=eq;self.equal[:new,old:new]=eq.T
        self.comparisons+=len(keys)*new;self.n=new

    def front(self,ordered):
        self.queries+=1
        if not ordered:return []
        ids=np.array([self.ids[k] for k in ordered],np.int64)
        dominated=self.dominates[np.ix_(ids,ids)].any(axis=0)
        duplicate=np.triu(self.equal[np.ix_(ids,ids)],k=1).any(axis=0)
        return [k for k,keep in zip(ordered,~(dominated|duplicate)) if keep]

    def roots(self,ordered,objectives,width,retention=.95):
        front=self.front(ordered)
        if not ordered:return [],0
        best=max(objectives[k][0] for k in ordered)
        # A dominator of a near-J point has at least that J, so it is near too.
        near=[k for k in front if objectives[k][0]>=retention*best]
        near_set=set(near);selected=[ordered[0]]
        for axis in (1,2):
            if near:
                key=max(near,key=lambda k:(objectives[k][axis],objectives[k][0]))
                if key not in selected:selected.append(key)
        for key in ordered:
            if len(selected)>=width:break
            if key in near_set and key not in selected:selected.append(key)
        for key in ordered:
            if len(selected)>=width:break
            if key not in selected:selected.append(key)
        return selected[:width],len(front)


def solve_service(p,draft,condition,config=ServiceConfig(),*,offline_diagnostic=False):
    if not offline_diagnostic:raise ValueError('Compressed teacher conditions require offline diagnostic mode')
    condition=condition.validate(p)
    if condition.counts is None or condition.activity is None:raise ValueError('Joint conditions required')
    phy=ConditionalPhy(p)
    try:
        memo=ExactMemo(phy,draft,config.max_candidates);original=memo.root
        floor=raw_pfq_floor(original.values[0]);graph=ProposalGraph(phy)
        stamp=time.perf_counter();bank=phy.bank();bank_ms=(time.perf_counter()-stamp)*1000
        features=JointFeatures(p,bank,condition);rng=np.random.default_rng(config.seed)
        archive={};expanded={};roots=set();pending=[];dequeues={};objectives={}
        cache=FrontierCache(config.max_candidates)
        matching_ms=proxy_ms=generation_ms=service_ms=0.
        proxy_candidates=decode_failures=seed_calls=0;stop='round_limit'

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
        metric=ServiceMetric(p,condition,bootstrap,config);group=metric.group

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
            stamp=time.perf_counter();actions=features.pool(base,config.wide,config.seed)
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
            ids=np.flatnonzero(valid);order=ids[np.argsort(-np.round(scores[ids],7),kind='stable')]
            selected=list(order[:min(config.batch_size,memo.remaining)])
            if config.pareto and len(selected)>=4 and group.any():
                service=metric.expected(predicted);near=ids[scores[ids]>=.95*objectives[key][0]];extra=[]
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
            service_metric=config.metric,service_rows=metric.rows,dominance_pairs=cache.comparisons,frontier_queries=cache.queries,
            decode_failures=decode_failures,stop_reason=stop,offline_diagnostic=True,full_search_calls=0,**phy.stats)
        return best.actions[0].copy(),v.copy(),r,group.copy()
    finally:phy.close()

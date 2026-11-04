"""Bounded physical headroom under resource/layer and/or activity conditions.

Only counts[UE], minimum layers[UE] and activity[cell,PRG] may be conditions.
The solver never receives teacher UE-to-PRG positions or teacher service/J.
"""
from dataclasses import dataclass
import ctypes
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import subprocess
import time
import weakref
import numpy as np
from scipy.optimize import linear_sum_assignment

from .physics import NativePhy,Evaluation,COUNT,LAYER,DEQUEUE,EXPECTED,curves
from .search import ProposalGraph
from .tb_macro import ExactMemo
from .no_ctbs_inputs import raw_pfq_floor


class DecodeInfeasible(ValueError):pass


@dataclass(frozen=True)
class Conditions:
    counts: np.ndarray|None=None
    layers: np.ndarray|None=None
    activity: np.ndarray|None=None

    def validate(self,p):
        n=l=a=None
        if (self.counts is None)!=(self.layers is None):raise ValueError('Counts and layers must be supplied together')
        if self.counts is not None:
            n=np.asarray(self.counts);l=np.asarray(self.layers)
            if n.shape!=(p.U,) or l.shape!=(p.U,) or not np.isfinite(n).all() or not np.isfinite(l).all():raise ValueError('Invalid quota shape')
            if not np.equal(n,np.floor(n)).all() or not np.equal(l,np.floor(l)).all():raise ValueError('Integer resource conditions required')
            n=n.astype(np.int64).copy();l=l.astype(np.int64).copy()
            if ((n<0)|(n>p.P)|(l<0)|(l>4)).any() or ((n==0)!=(l==0)).any():raise ValueError('Invalid quota/layer values')
            if any(n[p.serving==c].sum()>p.P for c in range(p.C)):raise ValueError('Cell resource capacity exceeded')
        if self.activity is not None:
            raw=np.asarray(self.activity)
            if raw.shape!=(p.C,p.P) or not np.isin(raw,[0,1]).all():raise ValueError('Invalid activity condition')
            a=raw.astype(bool).copy()
            if n is not None and any(n[p.serving==c].sum()!=a[c].sum() for c in range(p.C)):
                raise ValueError('Quotas and activity totals disagree')
        for value in (n,l,a):
            if value is not None:value.flags.writeable=False
        return Conditions(n,l,a)


def logical(p,actions):
    a=p.actions(actions);assigned=a[:,p.serving,:]==np.arange(p.U)[None,:,None]
    n=assigned.sum(2);l=np.where(assigned,p.caps[None],5).min(2);l=np.where(n>0,l,0)
    return n,l


def satisfies(p,actions,condition):
    a=p.actions(actions);ok=np.ones(len(a),bool)
    if condition.counts is not None:
        n,l=logical(p,a);ok &= (n==condition.counts).all(1)&(l==condition.layers).all(1)
    if condition.activity is not None:ok &= ((a>=0)==condition.activity).all((1,2))
    return ok


def build_kernel():
    d=Path(__file__).parent
    subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-ffp-contract=off','-shared','-fPIC',
        '-I/usr/include/eigen3',str(d/'resource_condition_kernel.cpp'),'-o',str(d/'_resource_condition.so')],check=True)


@lru_cache(maxsize=1)
def library():
    lib=ctypes.CDLL(str(Path(__file__).with_name('_resource_condition.so')))
    i=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS');d=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    lib.lns_create.argtypes=[ctypes.c_int]*3+[ctypes.c_double]+[i]*4+[d]*3+[i,d,d];lib.lns_create.restype=ctypes.c_void_p
    lib.lns_destroy.argtypes=[ctypes.c_void_p];lib.lns_destroy.restype=None
    lib.lns_evaluate.argtypes=[ctypes.c_void_p,ctypes.c_int,i,ctypes.c_void_p,ctypes.c_void_p,d];lib.lns_evaluate.restype=ctypes.c_int
    lib.lns_stats.argtypes=[ctypes.c_void_p,i];lib.lns_stats.restype=None
    lib.rc_bank.argtypes=[ctypes.c_void_p,d];lib.rc_bank.restype=ctypes.c_int
    lib.rc_proxy.argtypes=[ctypes.c_void_p,ctypes.c_int,i,d,d,d,d,i,i];lib.rc_proxy.restype=ctypes.c_int
    return lib


class ConditionalPhy(NativePhy):
    def __init__(self,p):
        self.problem=p;self.lib=library();self.covariance=np.ascontiguousarray(p.h@p.h.conj().swapaxes(-1,-2))
        self.signal=np.ascontiguousarray(p.h[np.arange(p.U),:,p.serving]@p.w)
        self.received=np.trace(self.covariance,axis1=-2,axis2=-1).real
        offsets=np.r_[0,np.cumsum([len(x) for x,_ in curves()])].astype(np.int64)
        xs=np.concatenate([x for x,_ in curves()]);ys=np.concatenate([y for _,y in curves()])
        self.context=self.lib.lns_create(p.U,p.C,p.P,p.noise,p.serving,p.mcs,p.caps,p.tbs,p.queue,
            self.covariance.view(np.float64),self.signal.view(np.float64),offsets,xs,ys)
        if not self.context:raise RuntimeError('Conditional PHY context failed')
        self._release=weakref.finalize(self,self.lib.lns_destroy,self.context)

    def bank(self):
        p=self.problem
        if p.C>3:raise ValueError('Conditional bank limited to3cells')
        out=np.empty((p.U,p.P,1<<p.C,4))
        if self.lib.rc_bank(self.context,out):raise RuntimeError('Gamma bank failed')
        return out

    def proxy(self,base,actions):
        p=self.problem;a=p.actions(actions);n=len(a);score=np.empty(n);total=np.empty(n)
        counts=np.empty((n,p.U),np.int64);layers=np.empty_like(counts)
        if self.lib.rc_proxy(self.context,n,a,base.values[0],p.weights,score,total,counts,layers):raise RuntimeError('Conditional proxy failed')
        return score,total,counts,layers


def edges(p,bank,activity,layers):
    mask=np.sum(np.asarray(activity,np.int64)*(1<<np.arange(p.C))[:,None],axis=0)
    return bank[np.arange(p.U)[:,None],np.arange(p.P)[None],mask[None],np.maximum(layers,1)[:,None]-1]


def matching(p,condition,utility):
    """Exact feasibility via duplicated UE slots; one cap==L witness per UE."""
    if condition.counts is None:raise ValueError('Matching requires quotas')
    action=np.full((p.C,p.P),-1,np.int64)
    for c in range(p.C):
        prgs=np.flatnonzero(condition.activity[c]) if condition.activity is not None else np.arange(p.P)
        tokens=[];witness=[]
        for u in np.flatnonzero((p.serving==c)&(condition.counts>0)):
            tokens.extend([int(u)]*int(condition.counts[u]));witness.extend([True]+[False]*(int(condition.counts[u])-1))
        blank=len(prgs)-len(tokens)
        if blank<0:raise DecodeInfeasible('Insufficient active slots')
        tokens.extend([-1]*blank);witness.extend([False]*blank)
        if not len(prgs):continue
        cost=np.zeros((len(prgs),len(tokens)))
        for j,(u,first) in enumerate(zip(tokens,witness)):
            if u<0:continue
            valid=p.legal[c,prgs,u]&(p.caps[u,prgs]==condition.layers[u] if first else p.caps[u,prgs]>=condition.layers[u])
            cost[:,j]=np.where(valid,-utility[u,prgs],np.inf)
        try:r,s=linear_sum_assignment(cost)
        except ValueError as exc:raise DecodeInfeasible('No feasible quota/layer matching') from exc
        if not np.isfinite(cost[r,s]).all():raise DecodeInfeasible('No finite assignment')
        action[c,prgs[r]]=np.asarray(tokens,np.int64)[s]
    if not satisfies(p,action,condition)[0]:raise AssertionError('Matching violated exact condition')
    return action


def seed_action(p,base,bank,graph,condition,rng,variant):
    root=base.actions[0];v=base.values[0];activity=condition.activity if condition.activity is not None else root>=0
    layers=condition.layers if condition.layers is not None else np.where(v[:,COUNT]>0,np.maximum(v[:,LAYER],1),p.caps.min(1)).astype(int)
    gamma=edges(p,bank,activity,layers)
    reliability=np.where(v[:,COUNT]>0,1-np.maximum(v[:,5],0),.5)
    if condition.counts is not None:
        count=condition.counts;deq=np.minimum(p.queue,p.tbs[np.arange(p.U),count,condition.layers])
        if variant%3==0:utility=p.weights[:,None]*deq[:,None]/np.maximum(count[:,None],1)*np.log1p(gamma)
        elif variant%3==1:utility=p.weights[:,None]*deq[:,None]/np.maximum(count[:,None],1)*gamma/np.maximum(gamma.mean(1)[:,None],1.e-12)
        else:utility=p.weights[:,None]*deq[:,None]/np.maximum(count[:,None],1)*(.1+1-reliability[:,None])*np.log1p(gamma)
        if variant>=3:utility+=rng.gumbel(size=utility.shape)*max(float(np.std(utility)),1)*(.02,.08,.2)[(variant//3-1)%3]
        return matching(p,condition,utility)
    active=activity.copy()
    if condition.activity is None and variant%4:
        for c in range(p.C):
            current=np.flatnonzero(active[c]);merit=[]
            for prg in current:
                u=root[c,prg];merit.append(p.weights[u]*v[u,EXPECTED]/max(v[u,COUNT],1) if u>=0 else 0.)
            mute=current[np.argsort(merit,kind='stable')[:min(variant%4*2,len(current))]];active[c,mute]=False
    action=np.full_like(root,-1);counts=np.zeros(p.U,int);minimum=np.full(p.U,4,int)
    prgs=np.arange(p.P)
    if variant%3==1:prgs=prgs[::-1]
    elif variant%3==2:prgs=rng.permutation(prgs)
    for prg in prgs:
        for c in range(p.C):
            if not active[c,prg]:continue
            us=np.flatnonzero(p.legal[c,prg]);n=counts[us];l=np.minimum(minimum[us],p.caps[us,prg])
            before=np.minimum(p.queue[us],p.tbs[us,n,minimum[us]])
            after=np.minimum(p.queue[us],p.tbs[us,n+1,l])
            score=p.weights[us]*(after-before)*(.1+reliability[us])*np.log1p(gamma[us,prg])
            score+=.02*p.weights[us]*after/np.maximum(n+1,1)*np.log1p(gamma[us,prg])
            if variant>=3:score+=rng.gumbel(size=len(us))*max(float(np.std(score)),1)*.1
            if not len(us):raise DecodeInfeasible('Active PRG without legal UE')
            u=int(us[np.argmax(score)]);action[c,prg]=u;counts[u]+=1;minimum[u]=min(minimum[u],p.caps[u,prg])
    return action


def local_pool(p,graph,base,condition):
    root=base.actions[0];result=[];seen={root.tobytes()}
    def add(a):
        key=a.tobytes()
        if key in seen:return
        c,j=np.nonzero(a!=root);u=a[c,j]
        if np.any((u>=0)&~p.legal[c,j,np.maximum(u,0)]):return
        seen.add(key);result.append(a)
    if condition.counts is None:
        for c in range(p.C):
            for j in range(p.P):
                if condition.activity is not None and not condition.activity[c,j]:continue
                choices=graph.options(root,c,j,3,base.values[0])
                for u in choices:
                    if u<0 and condition.activity is not None:continue
                    a=root.copy();a[c,j]=u;add(a)
    for c in range(p.C):
        prgs=np.flatnonzero(condition.activity[c]) if condition.activity is not None else np.arange(p.P)
        for j,k in combinations(prgs,2):
            if root[c,j]==root[c,k]:continue
            a=root.copy();a[c,j],a[c,k]=a[c,k],a[c,j];add(a)
        # A 3-cycle can preserve required layer witnesses when no pair swap can.
        for j,k,l in combinations(prgs[:min(len(prgs),6)],3):
            for shift in (1,2):
                a=root.copy();a[c,[j,k,l]]=np.roll(root[c,[j,k,l]],shift);add(a)
    return np.asarray(result,np.int64).reshape(-1,p.C,p.P)


@dataclass(frozen=True)
class ConditionalConfig:
    max_candidates:int=212
    max_feature_roots:int=56
    rounds:int=56
    seed_candidates:int=16
    batch_size:int=8
    beam_width:int=4
    seed:int=20261102

    def __post_init__(self):
        if min(self.max_candidates,self.max_feature_roots,self.rounds,self.seed_candidates,self.batch_size,self.beam_width)<1:raise ValueError('Positive conditional limits required')


def solve_conditional(p,draft,condition=Conditions(),config=ConditionalConfig(),*,offline_diagnostic=False,
                      pool_builder=None,deduplicator=None):
    if not offline_diagnostic and any(x is not None for x in (condition.counts,condition.layers,condition.activity)):
        raise ValueError('Teacher conditions require explicit offline diagnostic mode')
    condition=condition.validate(p);start=time.perf_counter();phy=ConditionalPhy(p)
    try:
        memo=ExactMemo(phy,draft,config.max_candidates);original=memo.root;floor=raw_pfq_floor(original.values[0]);graph=ProposalGraph(phy)
        initial_stats=phy.stats.copy();stamp=time.perf_counter();bank=phy.bank();bank_ms=(time.perf_counter()-stamp)*1000
        bank_stats=phy.stats.copy();rng=np.random.default_rng(config.seed);archive={};expanded={};roots=set()
        matching_ms=proxy_ms=generation_ms=0.;proxy_candidates=decode_failures=0;stop='round_limit'
        def store_exact(exact):
            if exact is None:return
            ok=satisfies(p,exact.actions,condition)
            for i in np.flatnonzero(ok):
                key=exact.actions[i].tobytes();archive[key]=phy.take(exact,int(i))
        store_exact(original)
        seeds=[];seen=set(memo.rows)
        stamp=time.perf_counter()
        for k in range(config.seed_candidates):
            try:a=seed_action(p,original,bank,graph,condition,rng,k)
            except DecodeInfeasible:decode_failures+=1;continue
            if a.tobytes() not in seen:seeds.append(a);seen.add(a.tobytes())
        matching_ms+=(time.perf_counter()-stamp)*1000
        if seeds:
            exact,_=memo.evaluate(np.stack(seeds),original);store_exact(exact)
        iterations=0
        for iteration in range(config.rounds):
            if memo.remaining<=0:stop='physical_budget';break
            def key_score(key):
                v=archive[key].values[0];return (int(v[:,DEQUEUE].sum()>=floor),float(v[:,EXPECTED]@p.weights),float(v[:,DEQUEUE].sum()))
            candidates=sorted(archive,key=key_score,reverse=True)
            if not candidates:stop='no_conditional_seed';break
            usable=[key for key in candidates if expanded.get(key,0)>=0][:config.beam_width]
            if not usable:stop='finite_pool_stalled';break
            key=usable[0] if iteration%2==0 else min(usable,key=lambda k:expanded.get(k,0))
            base=archive[key]
            if key not in roots and len(roots)>=config.max_feature_roots:stop='feature_budget';break
            roots.add(key);expanded[key]=expanded.get(key,0)+1;iterations+=1
            stamp=time.perf_counter();actions=(local_pool if pool_builder is None else pool_builder)(p,graph,base,condition);generation_ms+=(time.perf_counter()-stamp)*1000
            stamp=time.perf_counter();regenerated=[]
            if iteration%4==0:
                for j in range(2):
                    try:regenerated.append(seed_action(p,base,bank,graph,condition,rng,3+(iteration+j)%9))
                    except DecodeInfeasible:decode_failures+=1
            matching_ms+=(time.perf_counter()-stamp)*1000
            if regenerated:actions=np.concatenate([actions,np.stack(regenerated)])
            if len(actions):
                if deduplicator is None:
                    _,indices=np.unique(actions.reshape(len(actions),-1),axis=0,return_index=True);actions=actions[np.sort(indices)]
                else:actions=deduplicator(actions)
                actions=actions[memo.unseen(actions)]
            if not len(actions):expanded[key]=-1;continue
            stamp=time.perf_counter();scores,total,n,l=phy.proxy(base,actions);proxy_candidates+=len(actions)
            ok=total>=floor
            if condition.counts is not None:ok &= (n==condition.counts).all(1)&(l==condition.layers).all(1)
            if condition.activity is not None:ok &= ((actions>=0)==condition.activity).all((1,2))
            indices=np.flatnonzero(ok)
            # If no floor-feasible proposal is seen, explore projected complete
            # schedules without executing an infeasible one as the answer.
            if not len(indices):indices=np.flatnonzero(satisfies(p,actions,condition))
            order=indices[np.argsort(-np.round(scores[indices],7),kind='stable')]
            proxy_ms+=(time.perf_counter()-stamp)*1000
            if not len(order):expanded[key]=-1;continue
            exact,_=memo.evaluate(actions[order[:min(config.batch_size,memo.remaining)]],base);store_exact(exact)
        feasible=[value for value in archive.values() if value.values[0,:,DEQUEUE].sum()>=floor]
        if feasible:best=max(feasible,key=lambda value:float(value.values[0,:,EXPECTED]@p.weights))
        else:best=original
        v=best.values[0];score=float(v[:,EXPECTED]@p.weights);condition_ok=bool(satisfies(p,best.actions,condition)[0])
        record=dict(score=score,floor=floor,condition_feasible=bool(feasible),condition_satisfied=condition_ok,
            best_conditional_j=score if feasible else None,fallback_to_raw=not bool(feasible),
            bank_ms=bank_ms,initial_mmse_misses=initial_stats['mmse_misses'],
            bank_mmse_requests=bank_stats['mmse_requests']-initial_stats['mmse_requests'],
            bank_mmse_misses=bank_stats['mmse_misses']-initial_stats['mmse_misses'],
            prepared_cache_misses=bank_stats['mmse_misses'],
            matching_ms=matching_ms,proxy_ms=proxy_ms,generation_ms=generation_ms,proxy_candidates=proxy_candidates,
            proxy_ue_rows=proxy_candidates*p.U,feature_roots=len(roots),iterations=iterations,decode_failures=decode_failures,
            stop_reason=stop,offline_diagnostic=offline_diagnostic,full_search_calls=0,solve_ms=(time.perf_counter()-start)*1000,**phy.stats)
        assert record['candidate_evaluations']<=config.max_candidates and len(roots)<=config.max_feature_roots
        assert v[:,DEQUEUE].sum()>=floor
        if feasible:assert condition_ok
        return best.actions[0].copy(),v.copy(),record
    finally:phy.close()


if __name__=='__main__':build_kernel()

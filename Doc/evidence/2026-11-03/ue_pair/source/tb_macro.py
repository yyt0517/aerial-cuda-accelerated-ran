"""Whole-TB coordinated candidates and paid finite continuation.

No new model, CTBS, headroom mask or free teacher. Each decision has frozen
CSI/W/MCS/queues/weights. Exact complete-action memoization may therefore span
incumbents inside this decision; candidate rankings and logical descriptors
remain keyed by their own complete root. Nothing survives a new decision.
"""
from dataclasses import dataclass
from itertools import product
import time

import numpy as np

from .physics import Evaluation,EXPECTED,DEQUEUE,COUNT,LAYER,BLER
from .bounded_probes import ProbePhy
from .search import ProposalGraph,greedy_candidates
from .no_ctbs_inputs import raw_pfq_floor


@dataclass(frozen=True)
class MacroConfig:
    family: str = 'joint'
    continuation: bool = False
    max_candidates: int = 212
    max_feature_roots: int = 56
    max_sweeps: int = 5
    continuation_blocks: int = 4

    def __post_init__(self):
        if self.family not in ('single','joint'):raise ValueError('Unknown candidate family')
        if min(self.max_candidates,self.max_feature_roots,self.max_sweeps)>0 and self.continuation_blocks>=0:return
        raise ValueError('Invalid macro resources')


class ExactMemo:
    """Complete-action cache for exactly one immutable physical decision."""
    def __init__(self,phy,draft,limit):
        self.phy=phy;self.limit=limit;self.rows={};self.requests=0;self.cache_hits=0
        self.root=phy.evaluate(draft)
        self.rows[self.root.actions[0].tobytes()]=self.root.values[0]

    @property
    def remaining(self):return self.limit-self.phy.stats['candidate_evaluations']

    def unseen(self,actions):return np.array([a.tobytes() not in self.rows for a in actions],bool)

    def evaluate(self,actions,base):
        actions=self.phy.problem.actions(actions);self.requests+=len(actions)
        missing={a.tobytes():a for a in actions if a.tobytes() not in self.rows}
        count=min(len(missing),max(self.remaining,0))
        if count:
            exact=self.phy.evaluate(np.stack(list(missing.values())[:count]),base)
            for a,v in zip(exact.actions,exact.values):self.rows[a.tobytes()]=v.copy()
        keys=[a.tobytes() for a in actions];keep=np.array([key in self.rows for key in keys])
        self.cache_hits+=sum(key not in missing for key in keys)
        if not keep.any():return None,False
        selected=actions[keep].copy();values=np.stack([self.rows[key] for key,k in zip(keys,keep) if k])
        selected.flags.writeable=values.flags.writeable=False
        return Evaluation(selected,values,self.phy),bool(keep.all())


class Descriptors:
    """Cheap full-TB logical summaries; no candidate BLER or MMSE."""
    def __init__(self,p,phy,limit):
        self.p=p;self.phy=phy;self.limit=limit;self.rows={};self.feature_ms=0.;self.pool_ms=0.
        self.graph=ProposalGraph(phy);self.proxy_candidates=0

    def get(self,base):
        key=base.actions[0].tobytes()
        if key not in self.rows:
            if len(self.rows)>=self.limit:return None
            start=time.perf_counter();p=self.p;root=base.actions[0];v=base.values[0]
            received=self.phy.received.copy();received[np.arange(p.U),:,p.serving]=0.
            denominator=4*p.noise+np.einsum('cp,upc->up',root>=0,received)
            reliability=np.where(v[:,COUNT]>0,1-np.maximum(v[:,BLER],0),.5)
            self.rows[key]=dict(root=root,values=v,received=received,denominator=denominator,
                reliability=reliability,score=float(v[:,EXPECTED]@p.weights))
            self.feature_ms+=(time.perf_counter()-start)*1000
        return self.rows[key]

    def proxies(self,base,actions):
        descriptor=self.get(base)
        if descriptor is None:return None
        start=time.perf_counter();p=self.p;actions=p.actions(actions);n=len(actions)
        if not n:return np.empty(0),np.empty(0)
        assigned=actions[:,p.serving,:]==np.arange(p.U)[None,:,None]
        counts=assigned.sum(2)
        layers=np.where(assigned,p.caps[None],5).min(2)
        layers=np.where(counts>0,layers,0)
        dequeue=np.minimum(p.queue[None],p.tbs[np.arange(p.U)[None],counts,layers])
        fixed=(dequeue*descriptor['reliability'][None]-descriptor['values'][None,:,EXPECTED])*p.weights[None]
        activity=(actions>=0).astype(np.int8)-(descriptor['root'][None]>=0)
        after=descriptor['denominator'][None]+np.einsum('ncp,upc->nup',activity,descriptor['received'])
        logratio=np.log(descriptor['denominator'][None]/np.maximum(after,1.e-30))
        ratio=(logratio*assigned).sum(2)/np.maximum(counts,1)
        risk=np.maximum(dequeue,descriptor['values'][None,:,DEQUEUE])*(1-descriptor['reliability'])[None]
        proxy=fixed.sum(1)+(.5*risk*np.tanh(ratio)*p.weights[None]).sum(1)
        self.proxy_candidates+=n;self.feature_ms+=(time.perf_counter()-start)*1000
        return proxy,dequeue.sum(1)


def alternatives(p,graph,base,c,prg,n=2):
    return [u for u in graph.options(base.actions[0],c,prg,n,base.values[0])
            if u>=0 and u!=base.actions[0,c,prg]][:n]


def candidate_pools(p,descriptors,base,prg):
    """Singles plus new 2--3-token same-PRG and cross-PRG macro actions."""
    start=time.perf_counter();root=base.actions[0];graph=descriptors.graph
    # Original neighborhood on this PRG, including directed mute/reassignment.
    singles=[]
    choices=[np.flatnonzero(p.legal[c,prg] & (np.arange(p.U)!=root[c,prg])) for c in range(p.C)]
    for receiver in range(p.C):
        for u in choices[receiver]:
            a=root.copy();a[receiver,prg]=u;singles.append(a)
    for source in range(p.C):
        if root[source,prg]<0:continue
        a=root.copy();a[source,prg]=-1;singles.append(a)
        for receiver in range(p.C):
            if receiver==source:continue
            for u in choices[receiver]:
                a=root.copy();a[source,prg]=-1;a[receiver,prg]=u;singles.append(a)
    singles=np.asarray(singles,np.int64).reshape(-1,p.C,p.P)
    seen={root.tobytes(),*(a.tobytes() for a in singles)};macros=[]
    def add(a):
        key=a.tobytes()
        if key in seen:return
        c,j=np.nonzero(a!=root);users=a[c,j]
        if np.any((users>=0)&~p.legal[c,j,np.maximum(users,0)]):return
        seen.add(key);macros.append(a)
    alt=[alternatives(p,graph,base,c,prg) for c in range(p.C)]
    # Joint UE identity changes, not just original mute+one-recipient edits.
    for c in range(p.C):
        for d in range(c+1,p.C):
            if alt[c] and alt[d]:
                a=root.copy();a[c,prg]=alt[c][0];a[d,prg]=alt[d][0];add(a)
    if all(alt):
        a=root.copy()
        for c in range(p.C):a[c,prg]=alt[c][0]
        add(a)
    # Enumerate local activity intentions; keep current or best alternate IDs.
    for bits in product((False,True),repeat=p.C):
        for replace in (False,True):
            a=root.copy();valid=True
            for c,active in enumerate(bits):
                options=np.flatnonzero(p.legal[c,prg])
                if active and not len(options):valid=False;break
                a[c,prg]=(-1 if not active else (alt[c][0] if replace and alt[c] else
                    (root[c,prg] if root[c,prg]>=0 else int(options[0]))))
            if valid:add(a)
    # Drop a poor/min-layer allocation, compensate on another PRG, optionally
    # change a protected victim's identity. Every other allocation is retained.
    for c in range(p.C):
        u=int(root[c,prg])
        if u<0:continue
        qs=np.flatnonzero((np.arange(p.P)!=prg)&(root[c]!=u)&p.legal[c,:,u])
        merits=p.caps[u,qs]/4+.05*np.log1p(graph.quality[u,qs])
        qs=qs[np.argsort(-merits,kind='stable')[:2]]
        for q in qs:
            v=int(root[c,q])
            a=root.copy();a[c,prg]=-1;a[c,q]=u;add(a)
            if v>=0 and p.legal[c,prg,v]:
                a=root.copy();a[c,prg]=v;a[c,q]=u;add(a)
            for other in range(p.C):
                if other!=c and alt[other]:
                    a=root.copy();a[c,prg]=-1;a[c,q]=u;a[other,prg]=alt[other][0];add(a)
    descriptors.pool_ms+=(time.perf_counter()-start)*1000
    return singles,np.asarray(macros,np.int64).reshape(-1,p.C,p.P)


def shortlist(p,descriptors,memo,base,prg,floor,family,quota):
    pools=candidate_pools(p,descriptors,base,prg);ranked=[]
    for actions in pools:
        proxy=descriptors.proxies(base,actions)
        if proxy is None:return None
        scores,total=proxy;indices=np.flatnonzero((total>=floor)&memo.unseen(actions))
        ranked.append(actions[indices[np.argsort(-scores[indices],kind='stable')]])
    if family=='single':return ranked[0][:quota]
    # Preserve two single candidates whenever a four-row macro quota is used.
    split=min(2,quota);selected=list(ranked[0][:split])+list(ranked[1][:max(0,quota-split)])
    used={a.tobytes() for a in selected}
    for a in [*ranked[0],*ranked[1]]:
        if len(selected)>=quota:break
        if a.tobytes() not in used:selected.append(a);used.add(a.tobytes())
    return np.asarray(selected,np.int64).reshape(-1,p.C,p.P)


def prg_priority(p,descriptors,base):
    root=base.actions[0];v=base.values[0];scores=np.zeros(p.P)
    for prg in range(p.P):
        for c in range(p.C):
            u=root[c,prg]
            if u<0:continue
            scores[prg]+=p.weights[u]*v[u,DEQUEUE]*max(0,v[u,BLER])
            if v[u,COUNT]>1 and p.caps[u,prg]==v[u,LAYER]:scores[prg]+=.05*p.weights[u]*v[u,DEQUEUE]
    return np.argsort(-scores,kind='stable')


def continuation_plan(p,descriptors,base,prg,family):
    """Fixed small domains from the verified root, never future labels."""
    root=base.actions[0];v=base.values[0];graph=descriptors.graph
    sources=[c for c in range(p.C) if root[c,prg]>=0]
    if not sources:return None
    source=max(sources,key=lambda c:float(p.weights[root[c,prg]]*v[root[c,prg],DEQUEUE]*
        (max(v[root[c,prg],BLER],0)+.05*(p.caps[root[c,prg],prg]==v[root[c,prg],LAYER]))))
    user=int(root[source,prg]);qs=np.flatnonzero((np.arange(p.P)!=prg)&(root[source]!=user)&p.legal[source,:,user])
    if not len(qs):return None
    merit=p.caps[user,qs]/4+.05*np.log1p(graph.quality[user,qs]);q=int(qs[np.argmax(merit)])
    prefix=root.copy();suffix=[]
    if family=='joint':
        prefix[source,prg]=-1
        suffix.append(((source,q),tuple(dict.fromkeys([int(root[source,q]),user]))))
    else:
        alt=alternatives(p,graph,base,source,prg,1)
        prefix[source,prg]=alt[0] if alt else -1
        suffix.append(((source,q),tuple(dict.fromkeys([int(root[source,q]),user]))))
    others=[c for c in range(p.C) if c!=source]
    if others:
        other=max(others,key=lambda c:float(descriptors.phy.received[int(max(root[c,prg],0)),prg,source])
            if root[c,prg]>=0 else 0.)
        alt=alternatives(p,graph,base,other,prg,1)
        suffix.append(((other,prg),tuple(dict.fromkeys([int(root[other,prg]),*(alt[:1])]))))
    return prefix,tuple(suffix[:2])


def finite_continue(p,memo,base,prefix,suffix,floor):
    """Width2/horizon<=2, all nodes complete actions; incomplete values stay None."""
    start_count=memo.phy.stats['candidate_evaluations'];requested=0;complete=True
    best=None;best_j=-np.inf;seed_j=None;seed_feasible=None
    def score(actions):
        nonlocal requested,complete,best,best_j
        requested+=len(actions);result,finished=memo.evaluate(actions,base);complete &= finished
        if result is None:return None,None
        js=np.array([float(v[:,EXPECTED]@p.weights) for v in result.values])
        feasible=result.values[:,:,DEQUEUE].sum(1)>=floor
        if feasible.any():
            index=int(np.argmax(np.where(feasible,js,-np.inf)))
            if js[index]>best_j:best_j=float(js[index]);best=memo.phy.take(result,index)
        return result,js
    first,js=score(prefix[None])
    if first is None:return None,dict(complete=False,seed_j=None,best_observed_j=None,submitted=requested,
        physical_scores=memo.phy.stats['candidate_evaluations']-start_count)
    seed_j=float(js[0]);seed_feasible=bool(first.values[0,:,DEQUEUE].sum()>=floor);frontier=first.actions
    for token,options in suffix:
        proposals=np.repeat(frontier,len(options),axis=0)
        proposals[:,token[0],token[1]]=np.tile(options,len(frontier))
        result,js=score(proposals)
        if result is None:break
        order=np.argsort(-js,kind='stable')[:2];frontier=result.actions[order]
        if not complete:break
    root_j=float(base.values[0,:,EXPECTED]@p.weights)
    return best,dict(complete=bool(complete),seed_j=seed_j,seed_feasible=seed_feasible,
        best_observed_j=None if best is None else best_j,submitted=requested,
        physical_scores=memo.phy.stats['candidate_evaluations']-start_count,
        observed_future_gain=(best_j-seed_j) if best is not None and seed_feasible else None,
        rescued_unacceptable_seed=bool(best is not None and best_j>root_j+1 and (not seed_feasible or seed_j<=root_j+1)),
        improvement_vs_root=None if best is None else best_j-root_j)


def solve_macro(p,draft,config=MacroConfig()):
    phy=ProbePhy(p)
    try:
        memo=ExactMemo(phy,draft,config.max_candidates);root=memo.root;floor=raw_pfq_floor(root.values[0])
        initial=float(root.values[0,:,EXPECTED]@p.weights)
        descriptors=Descriptors(p,phy,config.max_feature_roots)
        covered=set();second=set();calls=0;accepted=0;macro_accepts=0;continuations=[];stop='sweep_limit'
        order=prg_priority(p,descriptors,root);coverage_snapshot=None;continuation_ran=False
        def accept(result):
            nonlocal root,accepted,macro_accepts
            if result is None:return
            js=np.array([float(v[:,EXPECTED]@p.weights) for v in result.values]);ok=result.values[:,:,DEQUEUE].sum(1)>=floor
            if not ok.any():return
            best=int(np.argmax(np.where(ok,js,-np.inf)))
            if js[best]>float(root.values[0,:,EXPECTED]@p.weights)+1:
                macro_accepts+=int(np.count_nonzero(result.actions[best]!=root.actions[0])>1)
                root=phy.take(result,best);accepted+=1
        def continue_stage():
            nonlocal continuation_ran
            continuation_ran=True
            for prg in prg_priority(p,descriptors,root)[:config.continuation_blocks]:
                if memo.remaining<=0:break
                if descriptors.get(root) is None:break
                plan=continuation_plan(p,descriptors,root,int(prg),config.family)
                if plan is None:
                    continuations.append(dict(prg=int(prg),complete=False,seed_j=None,best_observed_j=None,
                        reason='no_compensation_domain',physical_scores=0));continue
                best,record=finite_continue(p,memo,root,*plan,floor)
                continuations.append(dict(prg=int(prg),**record))
                if best is not None:accept(best)
        for sweep in range(config.max_sweeps):
            progress=accepted
            if sweep>0:order=prg_priority(p,descriptors,root)
            for prg in order:
                if memo.remaining<=0:stop='physical_budget';break
                if descriptors.get(root) is None:stop='feature_budget';break
                family='single' if sweep==0 else config.family
                proposals=shortlist(p,descriptors,memo,root,int(prg),floor,family,min(4,memo.remaining))
                if proposals is None:stop='feature_budget';break
                if len(proposals):
                    result,_=memo.evaluate(proposals,root);calls+=1;accept(result)
                covered.add(int(prg))
                if sweep==1:second.add(int(prg))
            if sweep==0:coverage_snapshot=root.actions[0].copy()
            # No speculative work before every PRG's first two passes.
            if sweep==1 and config.continuation and len(second)==p.P:continue_stage()
            if stop in ('physical_budget','feature_budget'):break
            if sweep>=1 and accepted==progress and (not config.continuation or continuation_ran):
                stop='proxy_domain_stalled';break
        score=float(root.values[0,:,EXPECTED]@p.weights)
        result=dict(score=score,initial_score=initial,floor=floor,family=config.family,continuation=config.continuation,
            covered_prgs=len(covered),second_pass_prgs=len(second),feature_roots=len(descriptors.rows),
            accepted_edits=accepted,macro_accepts=macro_accepts,direct_batches=calls,
            proxy_candidates=descriptors.proxy_candidates,feature_ms=descriptors.feature_ms,candidate_generation_ms=descriptors.pool_ms,
            memo_requests=memo.requests,memo_hits=memo.cache_hits,continuation_calls=len(continuations),
            continuation_complete=sum(r['complete'] for r in continuations),
            continuation_rescues=sum(r.get('rescued_unacceptable_seed',False) for r in continuations),
            continuation_phy=sum(r['physical_scores'] for r in continuations),
            continuation_records=continuations,stop_reason=stop,full_search_calls=0,**phy.stats)
        assert phy.stats['candidate_evaluations']<=config.max_candidates and len(descriptors.rows)<=config.max_feature_roots
        assert root.values[0,:,DEQUEUE].sum()>=floor and score+1.e-8>=initial
        return root.actions[0].copy(),root.values[0].copy(),result,coverage_snapshot
    finally:phy.close()

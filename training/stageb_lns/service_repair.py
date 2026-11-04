"""Causal service masks plus paid terminal repair in one fixed PHY budget."""
from dataclasses import dataclass
import time
import numpy as np
from .physics import COUNT,DEQUEUE,EXPECTED
from .bounded_probes import ProbePhy
from .no_ctbs_inputs import raw_pfq_floor
from .tb_macro import ExactMemo,candidate_pools,prg_priority
from .selective_tb import Incremental,choose_indices
from .service_sets import propose,validate_masks


@dataclass(frozen=True)
class RepairConfig:
    source: str='learned'
    lock: bool=False
    repair_budget: int=32
    max_candidates: int=212
    max_feature_roots: int=56
    max_sweeps: int=5
    repair_passes: int=4
    quality_retention: float=.995

    def __post_init__(self):
        if self.source not in ('none','heuristic','matched','learned','diagnostic'):raise ValueError('Unknown set source')
        if not 0<=self.repair_budget<self.max_candidates or min(self.max_feature_roots,self.max_sweeps,self.repair_passes)<1:
            raise ValueError('Invalid repair budget')
        if not 0<self.quality_retention<=1:raise ValueError('Invalid terminal quality fence')


def goal(values,masks):
    a,e=masks.T;expected=values[e,EXPECTED]
    missing_a=int((values[a,COUNT]==0).sum());missing_e=int((expected<=1).sum())
    if len(expected):
        p5=float(np.quantile(expected,.05));weak=float(np.sort(expected)[:max(1,int(np.ceil(.2*len(expected))))].sum())
    else:p5=weak=0.
    return (-missing_a,-missing_e,int(np.floor(p5)),int(np.floor(weak)))


def projected_order(p,base,actions,f,masks,available):
    reliability=np.where(base.values[0,:,COUNT]>0,1-np.maximum(base.values[0,:,5],0),.5)
    expected=f['dequeue']*reliability[None];a,e=masks.T
    missing_a=(f['counts'][:,a]==0).sum(1);missing_e=(expected[:,e]<=1).sum(1)
    if e.any():
        p5=np.quantile(expected[:,e],.05,axis=1)
        weak=np.sort(expected[:,e],axis=1)[:,:max(1,int(np.ceil(.2*e.sum())))].sum(1)
    else:p5=weak=np.zeros(len(actions))
    ids=np.flatnonzero(available)
    return ids[np.lexsort((ids,-np.round(f['prior'][ids],7),-np.floor(weak[ids]),-np.floor(p5[ids]),missing_e[ids],missing_a[ids]))]


def solve_service(p,draft,config=RepairConfig(),*,model=None,thresholds=(.5,.5),heuristic_counts=(6,4),diagnostic_masks=None):
    if (diagnostic_masks is not None)!=(config.source=='diagnostic'):
        raise ValueError('Teacher masks are permitted only in explicit offline diagnostic mode')
    if config.source in ('learned','matched') and model is None:raise ValueError('Missing causal set model')
    phy=ProbePhy(p)
    try:
        memo=ExactMemo(phy,draft,config.max_candidates);root=memo.root;floor=raw_pfq_floor(root.values[0])
        if config.source=='diagnostic':
            masks=validate_masks(p,*np.asarray(diagnostic_masks,bool).T,p.legal.any((0,1)));pr=dict(proposal_ms=0.)
        else:masks,pr=propose(p,phy,root,config.source,model,thresholds,heuristic_counts)
        d=Incremental(p,phy,config.max_feature_roots);failed=set();covered=set();main_accepted=0
        main_cap=config.max_candidates-config.repair_budget;main_stop='sweep_limit';start=time.perf_counter()
        for sweep in range(config.max_sweeps):
            previous=main_accepted
            for prg in prg_priority(p,d,root):
                if phy.stats['candidate_evaluations']>=main_cap:main_stop='main_budget';break
                if d.get(root) is None:main_stop='feature_budget';break
                singles,macros=candidate_pools(p,d,root,int(prg));actions=np.concatenate([singles,macros])
                f=d.features(root,actions,floor);joint=np.arange(len(actions))>=len(singles)
                available=(f['total']>=floor)&np.array([a.tobytes() not in failed for a in actions],bool)
                if config.lock:
                    available &= ~((root.values[0,None,:,COUNT]>0)&masks[None,:,0]&(f['counts']==0)).any(1)
                quota=min(4,main_cap-phy.stats['candidate_evaluations'],memo.remaining)
                selected=choose_indices(f['prior'],joint,available,'single' if sweep==0 else 'selective',quota)
                if len(selected):
                    exact,_=memo.evaluate(actions[selected],root);failed.update(a.tobytes() for a in actions[selected])
                    js=np.array([float(v[:,EXPECTED]@p.weights) for v in exact.values]);ok=exact.values[:,:,DEQUEUE].sum(1)>=floor
                    if ok.any():
                        best=int(np.argmax(np.where(ok,js,-np.inf)))
                        if js[best]>float(root.values[0,:,EXPECTED]@p.weights)+1:
                            root=phy.take(exact,best);main_accepted+=1;failed=set()
                covered.add(int(prg))
            if main_stop in ('main_budget','feature_budget'):break
            if sweep>=1 and main_accepted==previous:main_stop='proxy_domain_stalled';break
        main_ms=(time.perf_counter()-start)*1000;main_action=root.actions[0].copy();main_values=root.values[0].copy()
        main_j=float(main_values[:,EXPECTED]@p.weights);fence=config.quality_retention*main_j
        repair_start=phy.stats['candidate_evaluations'];repair_stop='disabled' if not config.repair_budget else 'pass_limit'
        repair_submitted=repair_accepted=repair_j_losses=0;repair_visited=set();events=[];failed=set();start=time.perf_counter()
        # Acceptance objective changed: reset failed mask, retain exact value cache.
        if config.repair_budget:
            for iteration in range(config.repair_passes):
                for prg in prg_priority(p,d,root):
                    if phy.stats['candidate_evaluations']-repair_start>=config.repair_budget or memo.remaining<=0:
                        repair_stop='repair_budget';break
                    if d.get(root) is None:repair_stop='feature_budget';break
                    actions=np.concatenate(candidate_pools(p,d,root,int(prg)));f=d.features(root,actions,floor)
                    available=(f['total']>=floor)&(f['dequeue']@p.weights+1.e-8>=fence)
                    available &= np.array([a.tobytes() not in failed for a in actions],bool)
                    order=projected_order(p,root,actions,f,masks,available);repair_visited.add(int(prg))
                    if not len(order):continue
                    action=actions[order[:1]];before_j=float(root.values[0,:,EXPECTED]@p.weights);before_goal=goal(root.values[0],masks)
                    before_count=phy.stats['candidate_evaluations'];exact,_=memo.evaluate(action,root)
                    failed.add(action[0].tobytes());repair_submitted+=1;v=exact.values[0];j=float(v[:,EXPECTED]@p.weights);after_goal=goal(v,masks)
                    feasible=bool(v[:,DEQUEUE].sum()>=floor and j+1.e-8>=fence)
                    accepted=feasible and (after_goal>before_goal or (after_goal==before_goal and j>before_j+1))
                    events.append(dict(prg=int(prg),iteration=iteration,physical_scores=phy.stats['candidate_evaluations']-before_count,
                        before_goal=before_goal,after_goal=after_goal,before_j=before_j,candidate_j=j,feasible=feasible,accepted=bool(accepted)))
                    if accepted:
                        root=phy.take(exact,0);repair_accepted+=1;repair_j_losses+=int(j<before_j);failed=set()
                if repair_stop in ('repair_budget','feature_budget'):break
        repair_ms=(time.perf_counter()-start)*1000;v=root.values[0];score=float(v[:,EXPECTED]@p.weights)
        actual_repair=phy.stats['candidate_evaluations']-repair_start
        assert phy.stats['candidate_evaluations']<=config.max_candidates and actual_repair<=config.repair_budget
        assert len(d.rows)<=config.max_feature_roots and v[:,DEQUEUE].sum()>=floor and score+1.e-8>=fence
        r=dict(score=score,floor=floor,main_j=main_j,quality_fence=fence,main_phy=repair_start,repair_phy=actual_repair,
            main_accepted=main_accepted,main_ms=main_ms,main_stop=main_stop,repair_ms=repair_ms,repair_stop=repair_stop,
            repair_submitted=repair_submitted,repair_accepted=repair_accepted,repair_j_losses=repair_j_losses,
            covered_prgs=len(covered),repair_visited_prgs=len(repair_visited),feature_roots=len(d.rows),
            feature_ms=d.feature_ms,candidate_generation_ms=d.pool_ms,proxy_candidates=d.proxy_candidates,
            memo_hits=memo.cache_hits,repair_events=events,diagnostic=config.source=='diagnostic',full_search_calls=0,**pr,**phy.stats)
        return root.actions[0].copy(),v.copy(),r,masks.copy(),(main_action,main_values)
    finally:phy.close()

"""Shared-floor cap4 moves and stale-exact dirty shortlists over exact caching.

Full initial scan is mandatory. Invalid cache entries stay invalid across
rounds until rescored. A failed dirty shortlist triggers its unscored complement
before claiming convergence; no current full-teacher labels enter decisions.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from .physics import NativePhy,Evaluation,EXPECTED,DEQUEUE
from .ranked_dpls import repair_warm_start
from .dpls_operations import compact_neighborhood,expand_edits,edit_metadata,canonical_choice,canonical_order


@dataclass(frozen=True)
class BudgetedCacheConfig:
    mode: str = 'cached'
    dirty_m: int = 16
    batch_cap: int = 4
    epsilon: float = 1.e-6
    max_rounds: int = 256
    diagnostics: bool = True

    def __post_init__(self):
        if self.mode not in ('cached','batch','dirty'):
            raise ValueError('Unknown cached search mode')
        if min(self.dirty_m,self.batch_cap,self.max_rounds)<1 or self.epsilon<0 or not np.isfinite(self.epsilon):
            raise ValueError('Invalid budget/configuration')


def solve_budgeted(problem,raw_pfq,config=BudgetedCacheConfig(),*,warm_start=None,observer=None):
    begin=time.perf_counter()
    phy=NativePhy(problem,profile=False)
    record=None
    try:
        root=phy.evaluate(raw_pfq)
        score=float(root.values[0,:,EXPECTED]@problem.weights)
        raw_score=score
        floor=(90*int(root.values[0,:,DEQUEUE].sum())+99)//100
        if warm_start is not None:
            warm=phy.evaluate(repair_warm_start(problem,warm_start))
            warm_score=float(warm.values[0,:,EXPECTED]@problem.weights)
            if warm.values[0,:,DEQUEUE].sum()>=floor and warm_score>score:
                root,score=warm,warm_score
        search_initial=score
        old_keys=old_values=old_valid=old_labels=None
        last_footprint=np.uint64(0)
        generated=scored=reused_total=fallback_total=fallback_rounds=joint_evals=accepted_moves=0
        first_scan_candidates=0
        first_scan_ms=0.
        setup_ms=(time.perf_counter()-begin)*1000 if config.diagnostics else None
        history=[]
        converged=False
        for iteration in range(1,config.max_rounds+1):
            round_begin=time.perf_counter() if config.diagnostics else 0.
            prg,assignment=compact_neighborhood(root.actions[0],problem.legal)
            keys,affected,footprints=edit_metadata(root.actions[0],prg,assignment,problem.U)
            n=len(keys)
            generated+=n
            valid=np.zeros(n,bool)
            labels=np.full(n,np.nan)
            values=np.broadcast_to(root.values[0],(n,*root.values.shape[1:])).copy()
            if old_keys is not None:
                _,previous,current=np.intersect1d(old_keys,keys,return_indices=True)
                labels[current]=old_labels[previous]
                clean=old_valid[previous] & ((footprints[current]&last_footprint)==0)
                new,old=current[clean],previous[clean]
                valid[new]=True
                support=((affected[new,None] >> np.arange(problem.U,dtype=np.uint64))&1)!=0
                rows,users=np.nonzero(support)
                values[new[rows],users]=old_values[old[rows],users]
            reused=int(valid.sum())
            reused_total+=reused
            if config.mode!='dirty' or old_keys is None:
                first=np.flatnonzero(~valid)
            else:
                novel=np.flatnonzero(~valid & ~np.isfinite(labels))
                dirty=np.flatnonzero(~valid & np.isfinite(labels))
                order=canonical_order(labels[dirty],keys[dirty],config.epsilon)
                first=np.r_[novel,dirty[order[:config.dirty_m]]]
            fresh_mask=np.zeros(n,bool)
            prior_labels=labels.copy() if observer is not None else None
            def refresh(indices):
                nonlocal scored
                if not len(indices):
                    return
                if valid[indices].any():
                    raise AssertionError('Repeated current-root PHY evaluation')
                exact=phy.evaluate(expand_edits(root.actions[0],prg[indices],assignment[indices]),root)
                values[indices]=exact.values
                valid[indices]=True
                fresh_mask[indices]=True
                scored+=len(indices)
            refresh(first)
            scores=values[:,:,EXPECTED]@problem.weights
            gain=scores-score
            eligible=valid & (values[:,:,DEQUEUE].sum(1)>=floor)
            threshold=1. if iteration<=2 else 16.
            chosen=canonical_choice(gain,eligible,keys,threshold,config.epsilon)
            fallback=0
            if chosen<0 and not valid.all():
                rest=np.flatnonzero(~valid)
                fallback=len(rest)
                fallback_total+=fallback
                fallback_rounds+=1
                refresh(rest)
                scores=values[:,:,EXPECTED]@problem.weights
                gain=scores-score
                eligible=values[:,:,DEQUEUE].sum(1)>=floor
                chosen=canonical_choice(gain,eligible,keys,threshold,config.epsilon)
            # Only exact/current labels may be propagated as newly exact labels.
            labels[valid]=gain[valid]
            selected=[]
            if chosen>=0:
                selected=[chosen]
                if config.mode=='batch':
                    used=int(footprints[chosen])
                    root_d=float(root.values[0,:,DEQUEUE].sum())
                    change=float(values[chosen,:,DEQUEUE].sum())-root_d
                    pool=np.flatnonzero(eligible & (gain>threshold))
                    order=pool[canonical_order(gain[pool],keys[pool],config.epsilon)]
                    for index in order:
                        if len(selected)>=config.batch_cap:
                            break
                        if int(footprints[index])&used:
                            continue
                        delta_d=float(values[index,:,DEQUEUE].sum())-root_d
                        if root_d+change+delta_d<floor:
                            continue
                        selected.append(int(index))
                        used|=int(footprints[index])
                        change+=delta_d
            if observer is not None:
                observer(dict(iteration=iteration,root=root,keys=keys,prg=prg,assignment=assignment,
                    chosen=chosen,selected=np.array(selected,np.int64),valid=valid.copy(),
                    fresh=fresh_mask.copy(),values=values,labels=labels.copy(),
                    prior_labels=prior_labels,affected=affected,footprints=footprints,
                    gain=np.where(valid,gain,np.nan),eligible=eligible,floor=floor,
                    threshold=threshold,fallback=fallback,root_score=score))
            old_keys,old_values,old_valid,old_labels=keys,values,valid,labels
            last_footprint=np.bitwise_or.reduce(footprints[selected],initial=np.uint64(0))
            root_score=score
            if selected:
                action=root.actions[0].copy()
                for index in selected:
                    action[:,prg[index]]=assignment[index]
                if len(selected)>1:
                    # Joint verification is part of online cost, not an oracle.
                    candidate=phy.evaluate(action,root)
                    candidate_score=float(candidate.values[0,:,EXPECTED]@problem.weights)
                    if candidate.values[0,:,DEQUEUE].sum()<floor or candidate_score-score<=threshold:
                        raise AssertionError('Independent batch violated exact score/floor acceptance')
                    joint_evals+=1
                    root=candidate
                    score=candidate_score
                else:
                    action=action[None].copy()
                    selected_values=values[chosen:chosen+1].copy()
                    action.flags.writeable=selected_values.flags.writeable=False
                    root=Evaluation(action,selected_values,phy)
                    score=float(scores[chosen])
                accepted_moves+=len(selected)
            if iteration==1:
                first_scan_candidates=int(fresh_mask.sum())
                if config.diagnostics:
                    first_scan_ms=(time.perf_counter()-round_begin)*1000
            if config.diagnostics:
                history.append(dict(iteration=iteration,candidates=n,scored=int(fresh_mask.sum()),
                    reused=reused,unscored_dirty=int((~valid).sum()),fallback_candidates=fallback,
                    selected_keys=[int(keys[j]) for j in selected],moves=len(selected),
                    root_score=root_score,score=score,gain=score-root_score,
                    round_ms=(time.perf_counter()-round_begin)*1000))
            if iteration>=3 and chosen<0:
                if not valid.all():
                    raise AssertionError('Convergence claimed with unscored dirty candidates')
                converged=True
                break
        record=dict(version='budgeted_cache_v1',mode=config.mode,dirty_m=config.dirty_m,batch_cap=config.batch_cap,
            initial_score=raw_score,search_initial_score=search_initial,score=score,floor=floor,
            chosen_dequeue=float(root.values[0,:,DEQUEUE].sum()),rounds=iteration,converged=converged,
            stop_reason='complete_neighborhood' if converged else 'round_limit',
            generated_candidates=generated,scored_neighborhood_candidates=scored,reused_candidates=reused_total,
            fallback_candidates=fallback_total,fallback_rounds=fallback_rounds,joint_evaluations=joint_evals,
            accepted_moves=accepted_moves,first_scan_candidates=first_scan_candidates,
            history=history,diagnostics_enabled=config.diagnostics,**phy.stats)
        if config.diagnostics:
            record.update(setup_ms=setup_ms,first_scan_ms=first_scan_ms)
        return root.actions[0].copy(),root.values[0].copy(),record
    finally:
        phy.close()
        if record is not None:
            record['core_ms']=(time.perf_counter()-begin)*1000

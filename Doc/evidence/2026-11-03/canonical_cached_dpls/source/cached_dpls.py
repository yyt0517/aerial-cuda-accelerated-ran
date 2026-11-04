"""Versioned canonical-tie full DPLS and exact cross-round dependency cache.

The old ranked_dpls baseline stays frozen. Both new modes share the same
max-anchored epsilon tie contract. Cached entries hold exact affected-UE PHY
values, not old global scores or eligibility. No cache crosses a TTI.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from .physics import NativePhy, Evaluation, EXPECTED, DEQUEUE
from .ranked_dpls import repair_warm_start
from .dpls_operations import compact_neighborhood, expand_edits, edit_metadata, canonical_choice


@dataclass(frozen=True)
class CanonicalDplsConfig:
    mode: str = 'full'
    epsilon: float = 1.e-6
    max_rounds: int = 256

    def __post_init__(self):
        if self.mode not in ('full','cached'):
            raise ValueError('Use full or cached canonical DPLS')
        if not np.isfinite(self.epsilon) or self.epsilon < 0 or self.max_rounds < 1:
            raise ValueError('Invalid epsilon or round cap')


def solve_canonical(problem, raw_pfq, config=CanonicalDplsConfig(), *, warm_start=None, observer=None):
    begin = time.perf_counter()
    phy = NativePhy(problem, profile=True)
    record = None
    try:
        root = phy.evaluate(raw_pfq)
        raw_score = float(root.values[0,:,EXPECTED]@problem.weights)
        score = raw_score
        floor = (90*int(root.values[0,:,DEQUEUE].sum())+99)//100
        initialization = 'raw_pfq'
        if warm_start is not None:
            warm = phy.evaluate(repair_warm_start(problem,warm_start))
            warm_score = float(warm.values[0,:,EXPECTED]@problem.weights)
            if warm.values[0,:,DEQUEUE].sum() >= floor and warm_score > score:
                root,score,initialization = warm,warm_score,'previous_executed'
        search_initial_score = score
        previous_keys = previous_values = None
        previous_move = np.uint64(0)
        history = []
        converged = False
        timings = dict(generation_ms=0.,mapping_ms=0.,reassembly_ms=0.,
                       evaluate_ms=0.,score_ms=0.,observer_ms=0.)
        for iteration in range(1,config.max_rounds+1):
            stamp = time.perf_counter()
            prg,assignment = compact_neighborhood(root.actions[0],problem.legal)
            keys,affected,footprints = edit_metadata(root.actions[0],prg,assignment,problem.U)
            n = len(keys)
            timings['generation_ms'] += (time.perf_counter()-stamp)*1000
            stamp = time.perf_counter()
            refresh = np.ones(n,bool)
            reused_new = reused_old = np.empty(0,np.int64)
            if config.mode == 'cached' and previous_keys is not None:
                _,old,new = np.intersect1d(previous_keys,keys,return_indices=True)
                valid = (footprints[new] & previous_move) == 0
                reused_new,reused_old = new[valid],old[valid]
                refresh[reused_new] = False
            timings['mapping_ms'] += (time.perf_counter()-stamp)*1000
            stamp = time.perf_counter()
            values = np.broadcast_to(root.values[0],(n,*root.values.shape[1:])).copy()
            if len(reused_new):
                support = ((affected[reused_new,None] >> np.arange(problem.U,dtype=np.uint64)) & 1) != 0
                rows,users = np.nonzero(support)
                values[reused_new[rows],users] = previous_values[reused_old[rows],users]
            timings['reassembly_ms'] += (time.perf_counter()-stamp)*1000
            stamp = time.perf_counter()
            fresh = np.flatnonzero(refresh)
            if len(fresh):
                result = phy.evaluate(expand_edits(root.actions[0],prg[fresh],assignment[fresh]),root)
                values[fresh] = result.values
            timings['evaluate_ms'] += (time.perf_counter()-stamp)*1000
            stamp = time.perf_counter()
            # Identical complete [candidate,UE,field] layout/reduction in both
            # modes. Unaffected rows come from the CURRENT incumbent, so old
            # global-score cancellation and stale floor flags are not reused.
            scores = values[:,:,EXPECTED]@problem.weights
            gain = scores-score
            eligible = values[:,:,DEQUEUE].sum(1) >= floor
            threshold = 1. if iteration <= 2 else 16.
            chosen = canonical_choice(gain,eligible,keys,threshold,config.epsilon)
            timings['score_ms'] += (time.perf_counter()-stamp)*1000
            entry = dict(iteration=iteration,candidates=n,scored=len(fresh),reused=len(reused_new),
                chosen_index=chosen,chosen_key=int(keys[chosen]) if chosen >= 0 else None,
                gain=float(gain[chosen]) if chosen >= 0 else 0.,root_score=score,
                threshold=threshold,eligible=int(eligible.sum()),**phy.stats)
            if observer is not None:
                stamp = time.perf_counter()
                observer(dict(iteration=iteration,root=root,keys=keys,prg=prg,assignment=assignment,
                              values=values,scores=scores,eligible=eligible,gain=gain,chosen=chosen,
                              refresh=refresh,floor=floor))
                timings['observer_ms'] += (time.perf_counter()-stamp)*1000
            history.append(entry)
            previous_keys,previous_values = keys,values
            previous_move = footprints[chosen] if chosen >= 0 else np.uint64(0)
            if chosen >= 0:
                action = expand_edits(root.actions[0],prg[chosen:chosen+1],assignment[chosen:chosen+1])
                selected = values[chosen:chosen+1].copy()
                action.flags.writeable = selected.flags.writeable = False
                root = Evaluation(action,selected,phy)
                score = float(scores[chosen])
            if iteration >= 3 and chosen < 0:
                converged = True
                break
        action,values = root.actions[0].copy(),root.values[0].copy()
        record = dict(version='canonical_dependency_v1',mode=config.mode,epsilon=config.epsilon,
            initialization=initialization,initial_score=raw_score,search_initial_score=search_initial_score,
            score=score,floor=floor,chosen_dequeue=float(values[:,DEQUEUE].sum()),
            converged=converged,stop_reason='complete_neighborhood' if converged else 'round_limit',
            rounds=len(history),history=history,
            scored_neighborhood_candidates=sum(h['scored'] for h in history),
            generated_candidates=sum(h['candidates'] for h in history),
            reused_candidates=sum(h['reused'] for h in history),**phy.stats,**phy.timings,**timings)
        return action,values,record
    finally:
        phy.close()
        if record is not None:
            record['core_ms'] = (time.perf_counter()-begin)*1000

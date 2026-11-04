"""Independent bounded masked-joint experiment; does not change active policies."""
from dataclasses import dataclass, asdict
import time
import numpy as np

from .physics import NativePhy, EXPECTED, DEQUEUE, BLER, LAYER, COUNT
from .search import ProposalGraph, SearchConfig
from .masked_joint import JointContext


@dataclass(frozen=True)
class MaskedConfig:
    rounds: int = 2
    blocks_per_round: int = 4
    global_reconstruction: bool = True
    seed: int = 1307


def raw_mask_blocks(graph, base):
    """The exact graph.blocks ordering before size-dependent truncation/dedup.

    Computing this once avoids repeating its identical priority/neighbor work
    for sizes 4, 8, 16 and 8. Arithmetic order matches ProposalGraph.blocks.
    """
    p=graph.problem;action=base.actions[0];prediction=base.values[0]
    priority=np.zeros((p.C,p.P));neighbors={}
    for c in range(p.C):
        for prg in range(p.P):
            targets=[]
            for other in range(p.C):
                u=action[other,prg]
                if other==c or u<0:continue
                fraction=graph.received[u,prg,c]/max(4*p.noise+graph.received[u,prg].sum()-graph.received[u,prg,other],1.e-30)
                benefit=p.weights[u]*prediction[u,DEQUEUE]*max(0.,prediction[u,BLER])*fraction
                priority[c,prg]+=benefit;targets.append((benefit,other))
            targets.sort(key=lambda x:(-x[0],x[1]))
            neighbors[c,prg]=[other for _,other in targets[:2]]
            u=action[c,prg]
            if u>=0:
                priority[c,prg]+=p.weights[u]*prediction[u,DEQUEUE]*max(0.,prediction[u,BLER])/(1+graph.quality[u,prg])
                if p.caps[u,prg]==prediction[u,LAYER] and prediction[u,COUNT]>1:
                    priority[c,prg]+=.05*p.weights[u]*prediction[u,DEQUEUE]
    raw=[]
    for token in np.argsort(-priority.ravel(),kind='stable'):
        c,prg=divmod(int(token),p.P)
        block=[(c,prg)]+[(other,prg) for other in neighbors[c,prg]]
        for cell,local in tuple(block):
            u=action[cell,local]
            if u<0:continue
            related=np.flatnonzero(action[cell]==u)
            related=sorted(related,key=lambda x:(graph.quality[u,x],int(x)))
            block.extend((cell,int(x)) for x in related if x!=local)
        raw.append(tuple(dict.fromkeys(block)))
    return raw


def masks(graph, base, number=4, visited=None, sizes=(4, 8, 16, 8)):
    """Directed-interference/TB masks of multiple sizes, refreshed per incumbent."""
    p = graph.problem
    visited = set() if visited is None else visited
    result = []
    raw=raw_mask_blocks(graph,base)
    by_size={}
    for size in sizes:
        if size not in by_size:
            seen=set();candidates=[]
            for full in raw:
                block=full[:min(size,p.C*p.P)];key=tuple(sorted(block))
                if key not in seen:seen.add(key);candidates.append(block)
            by_size[size]=candidates
        candidates=by_size[size]
        for block in candidates:
            flat = tuple(c*p.P+prg for c, prg in block)
            key = tuple(sorted(flat))
            if key in visited:
                continue
            visited.add(key); result.append(np.array(flat, dtype=np.int64))
            break
        if len(result) >= number:
            break
    return result


def unique_actions(actions, incumbent):
    seen = {incumbent.tobytes()}
    kept = []
    for action in actions:
        key = action.tobytes()
        if key not in seen:
            kept.append(action); seen.add(key)
    if not kept:
        return np.empty((0, *incumbent.shape), dtype=np.int64)
    return np.ascontiguousarray(kept, dtype=np.int64)


def solve_masked(problem, draft, repairer, config=MaskedConfig(), initial_action=None, trace=None):
    start = time.perf_counter()
    model_before = repairer.stats.copy()
    phy = NativePhy(problem)
    record = None
    try:
        incumbent = phy.evaluate(draft)
        draft_score = float(incumbent.values[0, :, EXPECTED]@problem.weights)
        draft_dequeue = int(incumbent.values[0, :, DEQUEUE].sum())
        floor = (90*draft_dequeue+99)//100
        if initial_action is not None and not np.array_equal(initial_action, draft):
            candidate = phy.evaluate(initial_action)
            if (candidate.values[0, :, DEQUEUE].sum() < floor or
                candidate.values[0, :, EXPECTED]@problem.weights+1.e-7 < draft_score):
                raise ValueError('Warm start violates original draft floor/objective')
            incumbent = candidate
        initial_score = score = float(incumbent.values[0, :, EXPECTED]@problem.weights)
        initialized = time.perf_counter()
        graph = ProposalGraph(phy)
        context = JointContext(phy, graph)
        setup_done = time.perf_counter()
        rng = np.random.default_rng(config.seed)
        accepted = 0
        verification_ms = 0.
        mask_ms = 0.
        stage_records = []
        visited = set()

        def try_blocks(blocks, stage):
            nonlocal incumbent, score, accepted, verification_ms, visited
            previous = incumbent
            proposals = repairer.propose_many(context, incumbent, floor, blocks, rng)
            proposals = unique_actions(proposals, incumbent.actions[0])
            old_score = score
            if len(proposals):
                t = time.perf_counter()
                values = phy.evaluate(proposals, incumbent)
                scores = values.values[:, :, EXPECTED]@problem.weights
                eligible = values.values[:, :, DEQUEUE].sum(1) >= floor
                best = int(np.argmax(np.where(eligible, scores, -np.inf)))
                threshold = 1. if stage <= 2 else 16.
                if eligible[best] and scores[best] > score+threshold:
                    incumbent = phy.take(values, best)
                    score = float(scores[best]); accepted += 1
                    visited = set()
                verification_ms += (time.perf_counter()-t)*1000
                if trace is not None:
                    trace(dict(stage=stage, base=previous.actions[0].copy(), blocks=[b.copy() for b in blocks],
                               actions=values.actions.copy(), values=values.values.copy(),
                               eligible=eligible.copy(), best_score=score))
            stage_records.append(dict(stage=stage, blocks=len(blocks), candidates=len(proposals),
                                      gain=score-old_score, score=score))

        if config.global_reconstruction:
            try_blocks([np.arange(problem.C*problem.P)], 1)
        for round_index in range(config.rounds):
            t = time.perf_counter()
            blocks = masks(graph, incumbent, config.blocks_per_round, visited)
            mask_ms += (time.perf_counter()-t)*1000
            if not blocks:
                break
            try_blocks(blocks, round_index+2)
        action = incumbent.actions[0].copy()
        values = incumbent.values[0].copy()
        model_stats = {k: repairer.stats[k]-model_before[k] for k in model_before}
        record = dict(config=asdict(config), score=score, initial_score=initial_score,
                      draft_score=draft_score, improvement=score-initial_score,
                      draft_dequeue=draft_dequeue, floor=floor,
                      chosen_dequeue=float(values[:, DEQUEUE].sum()), accepted_updates=accepted,
                      setup_ms=(setup_done-start)*1000, phy_initialization_ms=(initialized-start)*1000,
                      feature_setup_ms=(setup_done-initialized)*1000, mask_ms=mask_ms,
                      verification_ms=verification_ms, stages=stage_records,
                      model=model_stats, **phy.stats)
        return action, values, record
    finally:
        phy.close()
        if record is not None:
            record['core_ms'] = (time.perf_counter()-start)*1000

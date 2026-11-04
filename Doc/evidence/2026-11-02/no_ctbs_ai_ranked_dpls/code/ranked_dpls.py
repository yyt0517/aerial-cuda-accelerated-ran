"""CTBS-free DPLS with learned/heuristic original-neighborhood screening.

Each iteration has ONE fixed root and accepts at most one edit. Full scans
use stable original-neighborhood argmax. A failed shortlist evaluates only
its unscored complement; termination always follows a complete neighborhood
check with the original >1-byte / >16-byte thresholds. This certifies a local
stop, not equality to the full-search trajectory or its packet outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

from .physics import NativePhy, EXPECTED, DEQUEUE
from .search import greedy_candidates


@dataclass(frozen=True)
class RankedDplsConfig:
    mode: str = 'full'
    top_k: int = 64
    max_rounds: int = 256

    def __post_init__(self):
        if self.mode not in ('full', 'heuristic', 'ai'):
            raise ValueError('Unknown neighborhood-ranking mode')
        if self.top_k <= 0 or self.max_rounds < 3:
            raise ValueError('Invalid shortlist or iteration limit')


def solve_ranked(problem, raw_pfq, config=RankedDplsConfig(), *, ranker=None, observer=None):
    """Fresh PHY and feature context, all decision-local work timed.

    observer is for offline teacher collection only and runs after the current
    root's scored stream is fixed. It cannot return an action or alter ranking.
    No wall-clock cutoff changes the frozen quality-first search semantics.
    """
    if config.mode == 'ai' and ranker is None:
        raise ValueError('AI ranking requires a trained checkpoint')
    begin = time.perf_counter()
    phy = NativePhy(problem)
    context = None
    record = None
    try:
        root = phy.evaluate(raw_pfq)
        score = float(root.values[0, :, EXPECTED]@problem.weights)
        initial_score = score
        draft_dequeue = int(root.values[0, :, DEQUEUE].sum())
        floor = (90*draft_dequeue+99)//100
        if config.mode != 'full':
            from .edit_ranker import EditFeatureContext
            context = EditFeatureContext(problem, phy)
        generated = shortlisted = fallback_candidates = 0
        fallback_rounds = accepted = final_scan_candidates = 0
        feature_ms = inference_ms = generation_ms = phy_ms = 0.
        history = []
        converged = False
        for iteration in range(1, config.max_rounds+1):
            stamp = time.perf_counter()
            actions = greedy_candidates(problem, root.actions[0])
            generation_ms += (time.perf_counter()-stamp)*1000
            n = len(actions)
            generated += n
            threshold = 1. if iteration <= 2 else 16.
            scores = np.full(n, -np.inf)
            eligible = np.zeros(n, bool)
            evaluated = np.zeros(n, bool)
            best_index, best_score, best_values = -1, score, None

            def evaluate(indices):
                nonlocal phy_ms, best_index, best_score, best_values
                if not len(indices):
                    return
                if evaluated[indices].any():
                    raise AssertionError('Fallback attempted to repeat a scored candidate')
                start = time.perf_counter()
                result = phy.evaluate(actions[indices], root)
                phy_ms += (time.perf_counter()-start)*1000
                values = result.values
                local_j = values[:, :, EXPECTED]@problem.weights
                local_eligible = values[:, :, DEQUEUE].sum(1) >= floor
                scores[indices] = local_j
                eligible[indices] = local_eligible
                evaluated[indices] = True
                if local_eligible.any():
                    maximum = float(local_j[local_eligible].max())
                    # PHY order can differ from original candidate order.
                    tied = indices[local_eligible & (local_j == maximum)]
                    chosen = int(tied.min())
                    if maximum > best_score or (maximum == best_score and
                            (best_index < 0 or chosen < best_index)):
                        best_index, best_score = chosen, maximum
                        best_values = values[int(np.flatnonzero(indices == chosen)[0])].copy()

            x = heuristic = None
            if config.mode == 'full' or config.top_k >= n:
                evaluate(np.arange(n))
                complete_scan = True
            else:
                from .edit_ranker import candidate_features
                stamp = time.perf_counter()
                x, heuristic = candidate_features(problem, context, root, actions, floor=floor)
                feature_ms += (time.perf_counter()-stamp)*1000
                stamp = time.perf_counter()
                ranking = heuristic.copy() if config.mode == 'heuristic' else ranker.predict(x, heuristic)
                if not np.isfinite(ranking).all():
                    raise ValueError('Nonfinite learned/heuristic ranking')
                feasible = context.last_dequeue_totals >= floor
                order = np.lexsort((np.arange(n), -ranking))
                first = order[feasible[order]][:config.top_k]
                inference_ms += (time.perf_counter()-stamp)*1000
                shortlisted += len(first)
                evaluate(first)
                complete_scan = bool(evaluated.all())
                if best_score-score <= threshold:
                    remaining = np.flatnonzero(~evaluated)
                    fallback_rounds += 1
                    fallback_candidates += len(remaining)
                    evaluate(remaining)
                    complete_scan = True
            gain = best_score-score
            take = best_index >= 0 and gain > threshold
            if observer is not None:
                observer(dict(problem=problem, phy=phy, context=context, base=root,
                    actions=actions, scores=scores.copy(), eligible=eligible.copy(),
                    evaluated=evaluated.copy(), full_scan=complete_scan, iteration=iteration,
                    floor=floor, threshold=threshold, features=x, heuristic=heuristic,
                    chosen_index=best_index if take else -1))
            history.append(dict(iteration=iteration,candidates=n,scored=int(evaluated.sum()),
                full_scan=complete_scan,accepted=bool(take),gain=float(gain if take else 0.),
                root_score=score,best_scored_score=best_score))
            if take:
                from .physics import Evaluation
                action = actions[best_index:best_index+1].copy()
                value = best_values[None].copy()
                action.flags.writeable = value.flags.writeable = False
                root = Evaluation(action, value, phy)
                score = best_score
                accepted += 1
            if iteration >= 3 and not take:
                if not complete_scan:
                    raise AssertionError('Search stopped without full-neighborhood verification')
                converged = True
                final_scan_candidates = n
                break
        action, values = root.actions[0].copy(), root.values[0].copy()
        record = dict(mode=config.mode,top_k=config.top_k,score=score,initial_score=initial_score,
            improvement=score-initial_score,draft_dequeue=draft_dequeue,floor=floor,
            chosen_dequeue=float(values[:, DEQUEUE].sum()),rounds=len(history),accepted_updates=accepted,
            generated_candidates=generated,shortlisted_candidates=shortlisted,
            fallback_candidates=fallback_candidates,fallback_rounds=fallback_rounds,
            fallback_fraction=fallback_rounds/len(history),final_scan_candidates=final_scan_candidates,
            final_full_scan=bool(converged),converged=converged,
            stop_reason='full_neighborhood_stop' if converged else 'round_limit',
            feature_ms=feature_ms,inference_and_sort_ms=inference_ms,
            generation_ms=generation_ms,physical_ms=phy_ms,history=history,**phy.stats)
        return action, values, record
    finally:
        if context is not None:
            context.close()
        phy.close()
        if record is not None:
            record['core_ms'] = (time.perf_counter()-begin)*1000

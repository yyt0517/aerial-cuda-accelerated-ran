"""Frozen-actor proposals, budgeted continuation values, and own-prefix labels.

The value target is the best *observed feasible* J under a specified finite
physical continuation, not an optimal suffix, process advantage, or multi-TTI
return. Missing continuations are explicitly masked, never filled with zero.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .cached_repair import CachedJointRepair
from .neural_repair import destroy
from .physics import EXPECTED, DEQUEUE
from .structured_repair import VerifiedMemo, validate_block


VALUE_VERSION = 1
EXTRA_FEATURES = (
    'legal_suffix_slots', 'lower_layer_suffix_slots', 'reachable_count',
    'reachable_min_layer', 'reachable_dequeue', 'queue_gap_before',
    'queue_gap_after', 'reachable_gain_after', 'same_layer_suffix_slots',
    'editable_same_prg_neighbors', 'root_failure_exposure', 'weighted_reachable_gain',
    'suffix_min_layer', 'suffix_max_layer', 'logical_floor_slack', 'continuation_horizon',
)


def structural_features(graph, base, block, depth, choices, count, layer, floor, horizon=2):
    """Sixteen causal completion/feasibility features; no candidate PHY query."""
    p = graph.problem
    c, prg = block[depth]
    u = np.maximum(choices, 0)
    valid = choices >= 0
    current_count, current_layer = count[:, u], layer[:, u]
    new_count = np.minimum(current_count+1, p.P)
    new_layer = np.minimum(current_layer, p.caps[u, prg])
    suffix = [t for t in block[depth+1:] if t[0] == c]
    allowed = np.array([[p.legal[cc, pp, user] for cc, pp in suffix] for user in u], bool).reshape(len(u), len(suffix))
    caps = np.array([[p.caps[user, pp] for _, pp in suffix] for user in u], int).reshape(len(u), len(suffix))
    available = allowed.sum(1)
    minimum = np.where(allowed, caps, 5).min(1) if suffix else np.full(len(u), 5)
    maximum = np.where(allowed, caps, 0).max(1) if suffix else np.zeros(len(u))
    reachable_count = np.minimum(new_count+available, p.P)
    reachable_layer = np.minimum(new_layer, minimum)
    before = np.minimum(p.queue[u], p.tbs[u[None], current_count, current_layer])
    after = np.minimum(p.queue[u], p.tbs[u[None], new_count, new_layer])
    reachable = np.minimum(p.queue[u], p.tbs[u[None], reachable_count, reachable_layer])
    lower = ((caps[None] < new_layer[..., None]) & allowed[None]).sum(-1)
    equal = ((caps[None] == new_layer[..., None]) & allowed[None]).sum(-1)
    logical = np.minimum(p.queue, p.tbs[np.arange(p.U)[None], count, layer]).sum(1)
    failure = np.maximum(base.values[0, u, 5], 0)
    size = len(block)
    neighbors = sum(cc != c and pp == prg for cc, pp in block[depth+1:])
    values = np.broadcast_arrays(
        available[None]/size, lower/size, reachable_count/p.P, reachable_layer/4,
        reachable/3000., (p.queue[u]-before)/3000., (p.queue[u]-after)/3000.,
        (reachable-after)/3000., equal/size, np.full((len(count), len(u)), neighbors/p.C),
        failure[None]*reachable/3000., p.weights[u][None]*(reachable-after)/3000.,
        np.minimum(minimum, 4)[None]/4., maximum[None]/4.,
        (logical[:, None]-floor)/3000.,
        np.full((len(count), len(u)), min(horizon, len(block)-depth-1)/2.),
    )
    x = np.stack(values, -1).astype(np.float32)
    x[..., :14] *= valid[None, :, None]
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite structural continuation features')
    return x


def complete_prefix(base_action, partial, block, next_depth):
    action = base_action.copy()
    for token in block[:next_depth]:
        action[token] = partial[token]
    return action


@dataclass
class ContinuationResult:
    value: float | None
    action: np.ndarray | None
    complete: bool
    submitted: int


def bounded_continuation(graph, base, partial, block, next_depth, floor, evaluate, *,
                         horizon=2, width=4, max_candidates=128):
    """Finite all-legal physical beam; fixed prefix, incumbent-filled suffix.

    Each current completion and continuation is a legal complete schedule and
    goes through the caller's evaluator. Below-floor prefixes may still be
    expanded. Their unobserved feasible continuation is not assigned value 0.
    """
    p = graph.problem
    block = validate_block(p, block)
    if not 0 <= next_depth <= len(block) or horizon < 0 or min(width, max_candidates) <= 0:
        raise ValueError('Invalid continuation depth or budget')
    submitted = 0
    best, best_action = -np.inf, None

    def score(actions):
        nonlocal submitted, best, best_action
        left = max_candidates-submitted
        if left <= 0:
            return None, False
        requested = min(len(actions), left)
        result = evaluate(actions[:requested], base)
        if result is None:
            return None, False
        submitted += len(result.actions)
        j = result.values[:, :, EXPECTED] @ p.weights
        eligible = result.values[:, :, DEQUEUE].sum(1) >= floor
        if eligible.any():
            index = int(np.argmax(np.where(eligible, j, -np.inf)))
            if j[index] > best:
                best, best_action = float(j[index]), result.actions[index].copy()
        return (result, j), len(result.actions) == len(actions)

    first = complete_prefix(base.actions[0], partial, block, next_depth)[None]
    result, complete = score(first)
    if result is None:
        return ContinuationResult(None, None, False, submitted)
    frontier = result[0].actions
    for token in block[next_depth:next_depth+horizon]:
        choices = np.r_[-1, np.flatnonzero(p.legal[token])]
        actions = np.repeat(frontier, len(choices), axis=0)
        actions[:, token[0], token[1]] = np.tile(choices, len(frontier))
        result, full = score(actions)
        complete &= full
        if result is None:
            break
        evaluation, j = result
        order = np.argsort(-j, kind='stable')[:width]
        frontier = evaluation.actions[order]
        if not full:
            break
    return ContinuationResult(None if best_action is None else best,
                              best_action, bool(complete), submitted)


class ContinuationValue:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            if int(z['version']) != VALUE_VERSION:
                raise ValueError('Unsupported continuation-value checkpoint')
            for key in ('mean', 'scale', 'w1', 'b1', 'w2', 'b2', 'w3', 'b3'):
                setattr(self, key, z[key].copy())

    def predict(self, x):
        x = np.clip((x-self.mean)/self.scale, -10, 10)
        x = np.maximum(x@self.w1.T+self.b1, 0)
        x = np.maximum(x@self.w2.T+self.b2, 0)
        return (x@self.w3.T+self.b3)[..., 0]


class ContinuationRepair(CachedJointRepair):
    """Frozen 79 actor plus optional learned or explicit continuation ranking.

    likelihood exactly follows the cached decoder's proposal ordering. The
    recorder samples a real retained parent before pruning, and cannot return
    teacher actions. current/rollout call the same budgeted solve evaluator.
    """
    def __init__(self, checkpoint, floor, *, mode='likelihood', value=None,
                 width=16, shortlist=32, horizon=2, rollout_width=2,
                 recorder=None, record_blocks=4):
        super().__init__(checkpoint, width)
        if mode not in ('likelihood', 'learned', 'current', 'rollout'):
            raise ValueError('Unknown continuation mode')
        if mode == 'learned' and value is None:
            raise ValueError('Learned ranking needs a frozen value checkpoint')
        if min(shortlist, record_blocks, rollout_width) <= 0 or horizon < 0:
            raise ValueError('Invalid continuation controls')
        if mode in ('current', 'rollout') and shortlist <= width:
            raise ValueError('Value-ranking shortlist must exceed the retained beam width')
        self.floor, self.mode, self.value = floor, mode, value
        self.shortlist, self.horizon, self.rollout_width = shortlist, horizon, rollout_width
        self.recorder, self.record_blocks = recorder, record_blocks
        self.rollout_requests = self.incomplete_rollouts = 0

    def repair(self, graph, base, block, evaluate):
        p = graph.problem
        block = validate_block(p, block)
        cache = self.cache_for(graph, base)
        root = destroy(base.actions[0], block)
        assigned = root[p.serving, :] == np.arange(p.U)[:, None]
        count = assigned.sum(-1)[None]
        layer = np.where(assigned, p.caps, 5).min(-1)[None]
        layer = np.where(count > 0, layer, 4)
        active = (root >= 0)[None]
        edits = np.full((1, len(block)), -1, np.int64)
        logp = np.zeros(1)
        memo = VerifiedMemo(base, evaluate)
        self.calls += 1
        self.prefix_initializations += 1
        for depth, (cell, prg) in enumerate(block):
            choices = cache.node(cell, prg)[0]
            x79 = cache.features(block, depth, count, layer, active)
            scores = self.logits(x79)
            self.option_scores += scores.size
            self.logit_calls += 1
            scores -= scores.max(-1, keepdims=True)
            scores -= np.log(np.exp(scores).sum(-1, keepdims=True))
            joint = logp[:, None]+scores
            need_extra = self.mode == 'learned' or (self.recorder is not None and self.calls <= self.record_blocks)
            x = None
            if need_extra:
                extra = structural_features(graph, base, block, depth, choices, count, layer, self.floor, self.horizon)
                x = np.concatenate((x79, extra), -1)
            if self.recorder is not None and self.calls <= self.record_blocks:
                partial = root.copy()
                for step, token in enumerate(block[:depth]):
                    partial[token] = edits[0, step]
                self.recorder(dict(base_action=base.actions[0].copy(), partial=partial,
                    block=block, depth=depth, choices=choices.copy(), x=x[0].copy(),
                    floor=int(self.floor), root_score=float(base.values[0, :, EXPECTED]@p.weights),
                    block_visit=self.calls, parent_logp=float(logp[0])))
            if self.mode in ('likelihood', 'learned'):
                rank = joint if self.mode == 'likelihood' else self.value.predict(x)
                order = np.lexsort((np.arange(rank.size), -joint.ravel(), -rank.ravel()))[:self.width]
                parent, option = np.divmod(order, len(choices))
            else:
                candidates = np.argsort(-joint.ravel(), kind='stable')[:self.shortlist]
                parent, option = np.divmod(candidates, len(choices))
                prefixes = edits[parent].copy()
                prefixes[:, depth] = choices[option]
                complete = np.repeat(base.actions, len(parent), axis=0)
                for step, token in enumerate(block[:depth+1]):
                    complete[:, token[0], token[1]] = prefixes[:, step]
                current = memo(complete)
                if current is None:
                    return
                values = current.values[:, :, EXPECTED]@p.weights
                if self.mode == 'rollout':
                    for i, action in enumerate(complete):
                        result = bounded_continuation(graph, base, action, block, depth+1,
                            self.floor, lambda a, _: memo(a), horizon=self.horizon,
                            width=self.rollout_width, max_candidates=128)
                        self.rollout_requests += result.submitted
                        self.incomplete_rollouts += int(not result.complete)
                        values[i] = -np.inf if result.value is None else result.value
                        if not result.complete:
                            return
                keep = np.lexsort((np.arange(len(values)), -joint.ravel()[candidates], -values))[:self.width]
                parent, option = parent[keep], option[keep]
            selected = choices[option]
            edits = edits[parent].copy()
            edits[:, depth] = selected
            count, layer, active = count[parent], layer[parent], active[parent]
            active[:, cell, prg] = selected >= 0
            rows = np.flatnonzero(selected >= 0)
            users = selected[rows]
            count[rows, users] += 1
            layer[rows, users] = np.minimum(layer[rows, users], p.caps[users, prg])
            logp = joint[parent, option]
        proposals = np.repeat(base.actions, len(edits), axis=0)
        for step, token in enumerate(block):
            proposals[:, token[0], token[1]] = edits[:, step]
        self.proposals += len(proposals)
        # Preserve the original stream, including duplicate physical work, in
        # likelihood mode so its regression is an exact baseline comparison.
        evaluate(proposals, base)


def label_own_prefix(problem, observation, *, horizon=2, width=4, max_candidates=128):
    """Teacher query AFTER the student decision, in an independent PHY context."""
    from .physics import NativePhy
    from .search import ProposalGraph
    phy = NativePhy(problem)
    try:
        base = phy.evaluate(observation['base_action'])
        graph = ProposalGraph(phy)
        score = float(base.values[0, :, EXPECTED]@problem.weights)
        if abs(score-observation['root_score']) > 1.e-6:
            raise AssertionError('Teacher root does not match the visited student root')
        block, depth = observation['block'], observation['depth']
        target, known, complete, submitted = [], [], [], []
        for choice in observation['choices']:
            partial = observation['partial'].copy()
            partial[block[depth]] = choice
            result = bounded_continuation(graph, base, partial, block, depth+1,
                observation['floor'], phy.evaluate, horizon=horizon, width=width,
                max_candidates=max_candidates)
            target.append(np.nan if result.value is None else (result.value-score)/3000.)
            known.append(result.value is not None)
            complete.append(result.complete)
            submitted.append(result.submitted)
        return dict(x=observation['x'], target=np.array(target, np.float32),
                    known=np.array(known), complete=np.array(complete),
                    submitted=np.array(submitted), **phy.stats)
    finally:
        phy.close()

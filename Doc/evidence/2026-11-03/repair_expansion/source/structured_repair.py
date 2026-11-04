"""Exact activity-pattern / cell-assignment decomposition for small blocks.

This is valid for the existing activity-only PHY, fixed CSI/W/MCS, additive J,
and the original global dequeue floor. Whole TBs, including PRGs outside the
block, are evaluated. All PHY work goes through the caller's budget callback.
No change is made to the frozen neural decoder or active algorithm registry.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
import numpy as np

from .physics import Evaluation, EXPECTED, DEQUEUE


class FactorizationLimit(ValueError):
    """The exact local problem exceeds its declared combinatorial limit."""


def validate_block(problem, block):
    block = tuple((int(c), int(p)) for c, p in block)
    if not block or len(set(block)) != len(block):
        raise ValueError('A block must contain distinct positions')
    if any(not (0 <= c < problem.C and 0 <= p < problem.P) for c, p in block):
        raise ValueError('Block position outside the problem')
    return block


def pareto_indices(j, d):
    """Exact two-objective nondominance; stable first representative of ties.

    No floating tolerance is used to delete a potentially better J endpoint.
    Larger J and larger dequeue are better for this fixed-block objective only.
    """
    j, d = np.asarray(j, float), np.asarray(d, float)
    if j.ndim != 1 or j.shape != d.shape or not np.isfinite(j).all() or not np.isfinite(d).all():
        raise ValueError('Invalid Pareto objectives')
    order = np.lexsort((np.arange(len(j)), -j, -d))
    kept, best_j = [], -np.inf
    for index in order:
        if j[index] > best_j:
            kept.append(int(index))
            best_j = j[index]
    return np.asarray(kept, np.int64)


class VerifiedMemo:
    """Reuse only exact evaluations from this decision and verified base.

    Callback truncation is propagated, never interpreted as an absent gain.
    The memo is local to one repair call and cannot survive a graph/root change.
    """
    def __init__(self, base, evaluate):
        self.base, self.evaluate = base, evaluate
        self.rows = {base.actions[0].tobytes(): base.values[0]}
        self.new_candidates = 0

    def __call__(self, actions):
        actions = self.base.owner.problem.actions(actions)
        missing = dict.fromkeys(a.tobytes() for a in actions if a.tobytes() not in self.rows)
        if missing:
            shape = actions.shape[1:]
            new = np.stack([np.frombuffer(k, np.int64).reshape(shape) for k in missing])
            result = self.evaluate(new, self.base)
            if result is None:
                return None
            self.new_candidates += len(result.actions)
            for a, v in zip(result.actions, result.values):
                self.rows[a.tobytes()] = v
            if len(result.actions) != len(new):
                return None
        values = np.stack([self.rows[a.tobytes()] for a in actions])
        return Evaluation(actions, values, self.base.owner)


@dataclass
class FactorizedResult:
    action: np.ndarray | None
    predicted_j: float | None
    predicted_dequeue: float | None
    complete: bool
    patterns: int
    local_assignments: int
    merged_pairs: int
    verified_candidates: int


def factorized_block(graph, base, block, floor, evaluate, *,
                     max_cell_assignments=4096, max_merge_pairs=65536,
                     max_block_size=4):
    """Enumerate activities, retain cell (J,D) frontiers, and merge globally.

    Complete=True certifies exhaustive coverage of this *fixed* block, not of
    the full scheduling problem. All combinatorial caps fail explicitly.
    A budget-exhausted callback yields complete=False and no purported optimum.
    """
    p = graph.problem
    block = validate_block(p, block)
    if base.owner.problem is not p or graph.received is not base.owner.received:
        raise ValueError('Graph and verified base belong to different PHY contexts')
    if len(block) > max_block_size or min(max_cell_assignments, max_merge_pairs) <= 0:
        raise FactorizationLimit('Exact block size or work limit exceeded')
    domains = {token: np.flatnonzero(p.legal[token]) for token in block}
    for c in range(p.C):
        sizes = [len(domains[t]) for t in block if t[0] == c and len(domains[t])]
        if prod(sizes) > max_cell_assignments:
            raise FactorizationLimit('Exact same-cell assignment limit exceeded')
    memo = VerifiedMemo(base, evaluate)
    patterns = assignments = merged = 0
    best_action, best_j, best_d = None, -np.inf, None
    best_values = None

    def finish(complete):
        return FactorizedResult(best_action if complete else None,
            float(best_j) if complete and best_action is not None else None,
            float(best_d) if complete and best_action is not None else None,
            complete, patterns, assignments, merged, memo.new_candidates)

    bits = [(False, True) if len(domains[t]) else (False,) for t in block]
    for pattern in product(*bits):
        patterns += 1
        skeleton = base.actions[0].copy()
        for token, active in zip(block, pattern):
            skeleton[token] = domains[token][0] if active else -1
        skeleton_eval = memo(skeleton)
        if skeleton_eval is None:
            return finish(False)
        # Every cell, including cells without editable positions, contributes
        # its whole-TB response to the fixed cross-cell activity pattern.
        frontier = [(0., 0., skeleton.copy())]
        for c in range(p.C):
            tokens = [t for t, active in zip(block, pattern) if t[0] == c and active]
            combinations = list(product(*(domains[t] for t in tokens)))
            actions = np.repeat(skeleton[None], len(combinations), axis=0)
            for i, values in enumerate(combinations):
                for token, value in zip(tokens, values):
                    actions[i][token] = value
            assignments += len(actions)
            result = memo(actions)
            if result is None:
                return finish(False)
            users = np.flatnonzero(p.serving == c)
            local_j = result.values[:, users, EXPECTED] @ p.weights[users]
            local_d = result.values[:, users, DEQUEUE].sum(1)
            keep = pareto_indices(local_j, local_d)
            if len(frontier)*len(keep) > max_merge_pairs:
                raise FactorizationLimit('Exact Pareto merge limit exceeded')
            combined = []
            for previous_j, previous_d, previous_action in frontier:
                for index in keep:
                    action = previous_action.copy()
                    action[c] = result.actions[index, c]
                    combined.append((previous_j+local_j[index], previous_d+local_d[index], action))
            merged += len(combined)
            keep = pareto_indices([x[0] for x in combined], [x[1] for x in combined])
            frontier = [combined[i] for i in keep]
        eligible = [x for x in frontier if x[1] >= floor]
        if eligible:
            j, d, action = max(eligible, key=lambda x: x[0])
            if j > best_j:
                best_j, best_d, best_action = float(j), float(d), action.copy()
    if best_action is not None:
        verified = memo(best_action)
        if verified is None:
            return finish(False)
        best_values = verified.values[0]
        actual_j = float(best_values[:, EXPECTED] @ p.weights)
        actual_d = float(best_values[:, DEQUEUE].sum())
        if not np.isclose(actual_j, best_j, rtol=1.e-10, atol=1.e-6) or actual_d != best_d:
            raise AssertionError('Activity decomposition disagrees with the exact full PHY')
    return finish(True)


class StructuredRepair:
    """Search adapter; exact actions remain subject to solve's original guards."""
    deterministic = True

    def __init__(self, floor, *, max_cell_assignments=4096, max_blocks=8):
        if max_blocks <= 0:
            raise ValueError('max_blocks must be positive')
        self.floor = floor
        self.max_cell_assignments = max_cell_assignments
        self.max_blocks = max_blocks
        self.calls = self.completed = self.limits = 0
        self.patterns = self.local_assignments = self.merged_pairs = 0

    def repair(self, graph, base, block, evaluate):
        if self.calls >= self.max_blocks:
            return
        self.calls += 1
        try:
            result = factorized_block(graph, base, block, self.floor, evaluate,
                max_cell_assignments=self.max_cell_assignments)
        except FactorizationLimit:
            self.limits += 1
            return
        self.completed += int(result.complete)
        self.patterns += result.patterns
        self.local_assignments += result.local_assignments
        self.merged_pairs += result.merged_pairs

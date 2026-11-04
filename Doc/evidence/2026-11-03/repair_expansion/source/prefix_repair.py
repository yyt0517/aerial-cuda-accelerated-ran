"""Bounded neural proposal search with verified prefix values and coverage.

The actor ranks a small proposal set, rather than the verifier enumerating all
legal continuations. Prefixes are completed with the incumbent suffix before
verification. Search owns verification, acceptance and candidate budgets.
"""
import numpy as np

from .cached_repair import CachedDeadlineRepair
from .deadline import packet_objective
from .neural_repair import destroy


def coverage_order(order, groups, width, reserve):
    """Keep the best core and reserve slots for unrepresented first choices.

    Groups provide branch coverage, not a claim of physical equivalence.
    Stable input ordering supplies all score and prior tie breaks.
    """
    width = min(width, len(order))
    reserve = min(reserve, max(0, width-1))
    selected = list(map(int, order[:width-reserve]))
    represented = set(map(int, groups[selected]))
    for index in order:
        index = int(index)
        group = int(groups[index])
        if len(selected) >= width:
            break
        if group not in represented:
            selected.append(index)
            represented.add(group)
    present = set(selected)
    for index in order:
        index = int(index)
        if len(selected) >= width:
            break
        if index not in present:
            selected.append(index)
            present.add(index)
    return np.asarray(selected, dtype=np.int64)


def shortlist(joint, choices, incumbent_value, width, budget, per_parent):
    """Bound global likelihood leaders plus parent-local and suffix anchors."""
    global_order = np.argsort(-joint.ravel(), kind='stable')
    local_order = np.argsort(-joint, axis=1, kind='stable')
    parent_order = np.argsort(-joint.max(1), kind='stable')
    selected = dict.fromkeys(map(int, global_order[:width]))
    for rank in range(min(per_parent, len(choices))):
        for parent in parent_order:
            selected.setdefault(int(parent*len(choices)+local_order[parent, rank]), None)
    anchor = np.flatnonzero(choices == incumbent_value)
    if len(anchor):
        for parent in parent_order:
            selected.setdefault(int(parent*len(choices)+anchor[0]), None)
    # Fallback expansion avoids silently using less than the declared cap.
    for index in global_order:
        if len(selected) >= min(budget, joint.size):
            break
        selected.setdefault(int(index), None)
    return np.asarray(list(selected)[:budget], dtype=np.int64)


class PrefixRepair(CachedDeadlineRepair):
    """Exact prefix-value diagnostic with a frozen neural proposal policy.

    mode='coverage' verifies only the selected likelihood/diversity beam.
    mode='verified' scores a capped shortlist before retaining its best beam.
    mode='verified_cover' additionally reserves structurally different branches.
    All complete candidates go through solve.evaluate, with no hidden PHY calls.
    """
    MODES = ('coverage', 'verified', 'verified_cover')

    def __init__(self, checkpoint, width=16, *, mode='verified_cover',
                 prefix_budget=48, reserve=4, per_parent=2):
        super().__init__(checkpoint, width)
        if mode not in self.MODES:
            raise ValueError('Unknown prefix repair mode')
        if prefix_budget < width or per_parent <= 0 or not 0 <= reserve < width:
            raise ValueError('Invalid prefix search budget or reserve')
        self.mode = mode
        self.prefix_budget = prefix_budget
        self.reserve = reserve
        self.per_parent = per_parent
        self.prefix_candidates = self.prefix_batches = self.diversity_changes = 0

    def propose(self, graph, base, block):
        raise RuntimeError('PrefixRepair requires solve verification callbacks')

    def repair(self, graph, base, block, evaluate):
        p = graph.problem
        block = tuple(block)
        if not block or len(set(block)) != len(block):
            raise ValueError('Repair block must contain distinct tokens')
        cache = self.cache_for(graph, base)
        root = destroy(base.actions[0], block)
        assigned = root[p.serving, :] == np.arange(p.U)[:, None]
        count = assigned.sum(-1)[None]
        layer = np.where(assigned, p.caps, 5).min(-1)[None]
        layer = np.where(count > 0, layer, 4)
        active = (root >= 0)[None]
        edits = np.full((1, len(block)), -1, dtype=np.int64)
        logp = np.zeros(1)
        objective = packet_objective(p)
        self.calls += 1
        self.prefix_initializations += 1
        for depth, (cell, prg) in enumerate(block):
            choices = cache.node(cell, prg)[0]
            scores = self.logits(cache.features(block, depth, count, layer, active))
            self.option_scores += scores.size
            self.logit_calls += 1
            scores -= scores.max(-1, keepdims=True)
            scores -= np.log(np.exp(scores).sum(-1, keepdims=True))
            joint = logp[:, None]+scores
            indices = shortlist(joint, choices, int(base.actions[0, cell, prg]),
                                self.width, self.prefix_budget, self.per_parent)
            parent, value = np.divmod(indices, len(choices))
            groups = choices[value] if depth == 0 else edits[parent, 0]
            candidate_logp = joint.ravel()[indices]
            if self.mode == 'coverage':
                order = np.argsort(-candidate_logp, kind='stable')
                keep = coverage_order(order, groups, self.width, self.reserve)
                indices, parent, value = indices[keep], parent[keep], value[keep]
                groups, candidate_logp = groups[keep], candidate_logp[keep]
            prefix = edits[parent].copy()
            prefix[:, depth] = choices[value]
            # Complete unassigned positions with the incumbent, not blank.
            complete = np.broadcast_to(base.actions[0], (len(prefix), p.C, p.P)).copy()
            for step, (old_cell, old_prg) in enumerate(block[:depth+1]):
                complete[:, old_cell, old_prg] = prefix[:, step]
            self.proposals += len(complete)
            self.prefix_candidates += len(complete)
            result = evaluate(complete, base)
            self.prefix_batches += 1
            if result is None:
                break
            size = len(result.actions)
            # A candidate budget may truncate the submitted stream.
            parent, value = parent[:size], value[:size]
            prefix, groups, candidate_logp = prefix[:size], groups[:size], candidate_logp[:size]
            if self.mode == 'coverage':
                keep = np.arange(size)
            else:
                utility = objective(result.values)
                # Primary score: verified value. Neural log probability only
                # breaks value ties, followed by original stable input order.
                order = np.lexsort((np.arange(size), -candidate_logp, -utility))
                reserve = self.reserve if self.mode == 'verified_cover' else 0
                keep = coverage_order(order, groups, self.width, reserve)
                self.diversity_changes += int(not np.array_equal(keep, order[:len(keep)]))
            selected = choices[value[keep]]
            previous = parent[keep]
            edits = prefix[keep]
            count, layer, active = count[previous], layer[previous], active[previous]
            active[:, cell, prg] = selected >= 0
            rows = np.flatnonzero(selected >= 0)
            users = selected[rows]
            count[rows, users] += 1
            layer[rows, users] = np.minimum(layer[rows, users], p.caps[users, prg])
            logp = candidate_logp[keep]

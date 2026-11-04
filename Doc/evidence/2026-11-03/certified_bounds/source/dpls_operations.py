"""Canonical one-PRG edits, dependency footprints and deterministic choices."""
from __future__ import annotations
import numpy as np


def edit_metadata(root, prg, assignment, users):
    if users+root.shape[1] > 63:
        raise ValueError('Research dependency masks support at most 63 UE+PRG resources')
    old = root[:, prg].T
    edited = assignment != old
    targets = np.where(edited, assignment, -2)
    radix = users+3
    if root.shape[1]*radix**root.shape[0] >= 2**63:
        raise ValueError('Canonical operation key exceeds int64')
    keys = prg.astype(np.int64).copy()
    for c in range(root.shape[0]):
        keys = keys*radix+targets[:, c]+2
    if len(np.unique(keys)) != len(keys):
        raise ValueError('Duplicate operations')
    activity_changed = np.any((old >= 0) != (assignment >= 0), axis=1)
    affected = np.zeros(len(prg), np.uint64)
    for c in range(root.shape[0]):
        for ids in (old[:, c], assignment[:, c]):
            include = (ids >= 0) & (edited[:, c] | activity_changed)
            affected |= np.where(include, np.left_shift(np.uint64(1), np.maximum(ids, 0).astype(np.uint64)), 0)
    footprint = affected | np.left_shift(np.uint64(1), (users+prg).astype(np.uint64))
    return keys, affected, footprint


def compact_neighborhood(root, legal):
    """Same stream/order as greedy_candidates, without duplicating other PRGs."""
    cells, prgs, _ = legal.shape
    indices, assignments = [], []
    for p in range(prgs):
        current = root[:, p]
        choices = [np.flatnonzero(legal[c, p] & (np.arange(legal.shape[2]) != current[c])) for c in range(cells)]
        def add(source, receiver, targets):
            n = 1 if receiver < 0 else len(targets)
            if not n:
                return
            values = np.broadcast_to(current, (n, cells)).copy()
            if source >= 0:
                values[:, source] = -1
            if receiver >= 0:
                values[:, receiver] = targets
            assignments.append(values)
            indices.append(np.full(n, p, np.int64))
        for receiver in range(cells):
            add(-1, receiver, choices[receiver])
        for source in range(cells):
            if current[source] < 0:
                continue
            add(source, -1, None)
            for receiver in range(cells):
                if receiver != source:
                    add(source, receiver, choices[receiver])
    return (np.concatenate(indices), np.concatenate(assignments)) if indices else (
        np.empty(0, np.int64), np.empty((0, cells), np.int64))


def expand_edits(root, prg, assignment):
    actions = np.broadcast_to(root, (len(prg), *root.shape)).copy()
    actions[np.arange(len(prg))[:, None], np.arange(root.shape[0]), prg[:, None]] = assignment
    return np.ascontiguousarray(actions)


def infer_legal(root, prg, assignment, users):
    """Recover the constant legal domain from a COMPLETE original neighborhood."""
    legal = np.zeros((*root.shape, users), bool)
    for c in range(root.shape[0]):
        for p in range(root.shape[1]):
            ids = np.r_[root[c, p], assignment[prg == p, c]]
            legal[c, p, ids[ids >= 0]] = True
    return legal


def canonical_choice(gain, eligible, keys, threshold, epsilon=1.e-6):
    """Anchor ties at the true maximum; never accept a sub-threshold edit.

    Pairwise epsilon comparators are nontransitive. This definition is invariant
    to row/evaluation order and picks the smallest overwrite-operation key.
    """
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError('Invalid tie epsilon')
    pool = np.flatnonzero(eligible & (gain > threshold))
    if not len(pool):
        return -1
    maximum = gain[pool].max()
    tied = pool[gain[pool] >= maximum-epsilon]
    return int(tied[np.argmin(keys[tied])])


def canonical_order(scores, keys, epsilon=1.e-6, *, limit=None):
    """Descending score bands, each anchored at its own maximum, then key."""
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError('Invalid tie epsilon')
    if limit is not None and limit < 0:
        raise ValueError('Invalid ranking limit')
    if not np.isfinite(scores).all():
        raise ValueError('Rank only operators with a finite historical label')
    order = np.argsort(-scores, kind='stable')
    negative = -scores[order]
    result = []
    i = 0
    while i < len(order) and (limit is None or len(result) < limit):
        end = max(i+1, int(np.searchsorted(negative, -(scores[order[i]]-epsilon), side='right')))
        group = order[i:end]
        group=group[np.argsort(keys[group], kind='stable')]
        result.extend(group if limit is None else group[:max(0,limit-len(result))])
        i = end
    return np.asarray(result, np.int64)

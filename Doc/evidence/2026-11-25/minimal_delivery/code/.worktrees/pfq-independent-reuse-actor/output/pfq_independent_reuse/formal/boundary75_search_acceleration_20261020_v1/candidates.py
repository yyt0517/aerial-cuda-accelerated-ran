"""Same ordered neighborhood, with one action allocation and no unused metadata.

Preserve the original duplicate old/draft rows and every candidate position:
the frozen interpolation and argmax tie breaking depend on that ordering.
"""
import numpy as np


def _legal_choices(obs, queue):
    legal = np.asarray(obs['action_mask_cell_ue'], bool) & np.asarray(obs['action_mask_ue'], bool)[None]
    legal &= (np.asarray(queue) > 0)[None]
    return [np.flatnonzero(legal[c]) for c in range(3)], np.asarray(obs['action_mask_prg_cell'], bool)


def _construct(draft, old, choices, prg_legal, joint):
    draft = np.asarray(draft).reshape(3, 17)
    old = np.asarray(old).reshape(3, 17)
    changed = old != draft
    assert changed.sum() <= 1 and np.all(old[changed] == -1)
    # One block per receiver/source group, rather than one Python object and
    # action copy per candidate. Blocks retain the original nested-loop order.
    blocks = []
    size = 2
    for p in range(17):
        eligible = [choices[c][choices[c] != draft[c, p]].astype(draft.dtype, copy=False) for c in range(3)]
        for receiver in range(3):
            if prg_legal[receiver, p] and len(eligible[receiver]):
                ues = eligible[receiver]
                blocks.append((-1, receiver, p, ues))
                size += len(ues)
        for source in range(3):
            if draft[source, p] < 0:
                continue
            blocks.append((source, -1, p, None))
            size += 1
            if not joint:
                continue
            for receiver in range(3):
                if receiver == source or not prg_legal[receiver, p]:
                    continue
                ues = eligible[receiver]
                if len(ues):
                    blocks.append((source, receiver, p, ues))
                    size += len(ues)
    actions = np.empty((size, 3, 17), dtype=np.result_type(draft.dtype, old.dtype))
    actions[:] = draft
    actions[0] = old
    # Original edits occur in draft.dtype before np.stack promotes mixed roots.
    blank = np.asarray(-1).astype(draft.dtype).item()
    offset = 2
    for source, receiver, p, ues in blocks:
        end = offset + (1 if receiver < 0 else len(ues))
        if source >= 0:
            actions[offset:end, source, p] = blank
        if receiver >= 0:
            actions[offset:end, receiver, p] = ues
        offset = end
    assert offset == size
    return actions, None


def candidates(draft, old, obs, queue, joint):
    choices, prg_legal = _legal_choices(obs, queue)
    return _construct(draft, old, choices, prg_legal, joint)


fast_candidates = candidates


def prepare_candidates(obs, queue):
    """Prepare once for one fixed pre-action state; reuse across its rounds.

    The returned callable has the original five-argument signature. Its obs and
    queue arguments must refer to the state supplied here; no cross-TTI reuse.
    """
    choices, prg_legal = _legal_choices(obs, queue)

    def prepared(draft, old, obs, queue, joint):
        return _construct(draft, old, choices, prg_legal, joint)

    return prepared

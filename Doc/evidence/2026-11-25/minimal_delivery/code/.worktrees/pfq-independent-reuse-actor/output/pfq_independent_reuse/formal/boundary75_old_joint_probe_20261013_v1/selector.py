"""Disposable same-budget local reference, rooted in the frozen BC draft."""
import numpy as np


def candidates(draft, old, obs, queue, joint):
    draft = np.asarray(draft).reshape(3, 17)
    old = np.asarray(old).reshape(3, 17)
    changed = old != draft
    assert changed.sum() <= 1 and np.all(old[changed] == -1)
    legal = np.asarray(obs['action_mask_cell_ue'], bool) & np.asarray(obs['action_mask_ue'], bool)[None]
    legal &= (np.asarray(queue) > 0)[None]
    prg_legal = np.asarray(obs['action_mask_prg_cell'], bool)
    choices = [np.flatnonzero(legal[c]) for c in range(3)]
    actions = [old.copy(), draft.copy()]
    meta = [('old', -1, -1, -1, -1), ('draft', -1, -1, -1, -1)]
    for p in range(17):
        for receiver in range(3):
            if not prg_legal[receiver, p]:
                continue
            for u in choices[receiver]:
                if u == draft[receiver, p]:
                    continue
                a = draft.copy(); a[receiver, p] = u
                actions.append(a); meta.append(('ue', -1, receiver, p, int(u)))
        for source in range(3):
            if draft[source, p] < 0:
                continue
            mute = draft.copy(); mute[source, p] = -1
            actions.append(mute); meta.append(('blank', source, -1, p, -1))
            if not joint:
                continue
            for receiver in range(3):
                if receiver == source or not prg_legal[receiver, p]:
                    continue
                for u in choices[receiver]:
                    if u == draft[receiver, p]:
                        continue
                    a = mute.copy(); a[receiver, p] = u
                    actions.append(a); meta.append(('pair', source, receiver, p, int(u)))
    return np.stack(actions), meta


def select(draft, old, phy, obs, joint=True, details=False):
    actions, meta = candidates(draft, old, obs, phy.snapshot['queue'], joint)
    scores, dequeues = [], []
    for start in range(0, len(actions), 256):
        pred = phy.evaluate(actions[start:start + 256])
        scores.extend(pred['expected'].sum(1)); dequeues.extend(pred['dequeue'].sum(1))
    scores, dequeues = np.asarray(scores), np.asarray(dequeues)
    assert np.isfinite(scores).all() and np.isfinite(dequeues).all()
    eligible = (dequeues >= dequeues[0]) & (scores > scores[0] + 1.)
    def best(mask):
        allowed = eligible & mask
        return int(np.argmax(np.where(allowed, scores, -np.inf))) if allowed.any() else 0
    is_pair = np.asarray([m[0] == 'pair' for m in meta])
    index = best(np.ones(len(actions), bool))
    single_index = best(~is_pair)
    chosen = actions[index]
    kind, source, receiver, prg, ue = meta[index]
    info = dict(kind=kind, source=source, receiver=receiver, prg=prg, ue=ue,
        candidates=len(actions), eligible_pairs=int((eligible & is_pair).sum()),
        gain_vs_old_bytes=float(scores[index] - scores[0]),
        gain_vs_single_bytes=float(scores[index] - scores[single_index]),
        dequeue_delta_vs_old_bytes=float(dequeues[index] - dequeues[0]),
        old_expected_bytes=float(scores[0]), selected_expected_bytes=float(scores[index]),
        extra_blanks=int(((draft >= 0) & (chosen < 0)).sum()),
        changed_from_draft=int((chosen != draft).sum()),
        blank_credit_bytes=0., donor_credit_bytes=0., neighbor_credit_bytes=0., ue_credit_bytes=0.)
    if source >= 0:
        fill = chosen.copy(); fill[source, prg] = draft[source, prg]
        blank_only = chosen.copy()
        if receiver >= 0:
            blank_only[receiver, prg] = draft[receiver, prg]
        pp = phy.evaluate(np.stack([chosen, fill, blank_only]))
        delta = pp['expected'][0] - pp['expected'][1]
        donor = float(delta[phy.serving == source].sum())
        info.update(blank_credit_bytes=float(delta.sum()), donor_credit_bytes=donor,
            neighbor_credit_bytes=float(delta.sum()) - donor,
            ue_credit_bytes=float((pp['expected'][0] - pp['expected'][2]).sum()))
    assert info['extra_blanks'] <= 1 and info['changed_from_draft'] <= 2
    assert info['gain_vs_old_bytes'] >= 0 and info['dequeue_delta_vs_old_bytes'] >= 0
    if details:
        rows = [dict(index=i, kind=m[0], source=m[1], receiver=m[2], prg=m[3], ue=m[4],
            gain_vs_old_bytes=float(scores[i]-scores[0]), dequeue_delta_vs_old_bytes=float(dequeues[i]-dequeues[0]),
            eligible=bool(eligible[i]), selected=i == index) for i, m in enumerate(meta)]
        return chosen, info, rows, actions
    return chosen, info

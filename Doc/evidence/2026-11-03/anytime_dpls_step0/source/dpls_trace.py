"""Passive, compact full-neighborhood trajectories for stopping/proposal labels.

One NPZ per captured decision; candidate edits store the PRG and its complete
cell assignment instead of duplicating every unchanged PRG. No extra PHY calls.
Observer copying is charged inside solve_ranked; compression/I/O is separate.
"""
from pathlib import Path
import numpy as np
from .physics import EXPECTED


def solution_overlap(previous, current):
    previous, current = np.asarray(previous), np.asarray(current)
    if previous.shape != current.shape or previous.ndim != 2:
        raise ValueError('Expected equal cell-by-PRG actions')
    union = (previous >= 0) | (current >= 0)
    common = (previous == current) & (previous >= 0)
    return dict(slot_agreement=float(np.mean(previous == current)),
                active_assignment_jaccard=float(common.sum()/max(1, union.sum())) if union.any() else 1.,
                activity_agreement=float(np.mean((previous >= 0) == (current >= 0))),
                changed_slots=int(np.sum(previous != current)))


class DplsTrace:
    def __init__(self):
        self.rounds = []

    def __call__(self, info):
        root, actions = info['base'].actions[0], info['actions']
        changed = np.any(actions != root, axis=1)
        if np.any(changed.sum(1) != 1):
            raise ValueError('Compact trace requires original one-PRG edits')
        prg = np.argmax(changed, axis=1) if len(actions) else np.empty(0, dtype=int)
        assignment = actions[np.arange(len(actions))[:, None], np.arange(root.shape[0]), prg[:, None]]
        self.rounds.append(dict(root=root.copy(), prg=prg.astype(np.int32),
            assignment=assignment.astype(np.int32), scores=info['scores'].copy(),
            eligible=info['eligible'].copy(), evaluated=info['evaluated'].copy(),
            root_score=float(info['base'].values[0, :, EXPECTED]@info['problem'].weights),
            chosen_index=info['chosen_index'], threshold=info['threshold'],
            floor=info['floor'], iteration=info['iteration']))

    def write(self, path, *, action, values):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        lengths = [len(r['scores']) for r in self.rounds]
        shape = np.asarray(action).shape
        data = dict(schema_version=np.array(1), action=action, values=values,
                    offsets=np.r_[0, np.cumsum(lengths)].astype(np.int64))
        for key in ('prg', 'assignment', 'scores', 'eligible', 'evaluated'):
            data[key] = (np.concatenate([r[key] for r in self.rounds]) if self.rounds else
                         np.empty((0, shape[0]) if key == 'assignment' else (0,)))
        data['roots'] = (np.stack([r['root'] for r in self.rounds]) if self.rounds else
                         np.empty((0, *shape), dtype=np.int64))
        for key in ('root_score', 'chosen_index', 'threshold', 'floor', 'iteration'):
            data[key] = np.array([r[key] for r in self.rounds])
        # NaN denotes an unscored candidate; it is never a gain label.
        data['delta'] = np.where(data['evaluated'], data['scores']-
                                np.repeat(data['root_score'], lengths), np.nan)
        with path.open('xb') as handle:
            np.savez_compressed(handle, **data)

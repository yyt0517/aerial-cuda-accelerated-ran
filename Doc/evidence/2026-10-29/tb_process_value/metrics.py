"""Common frozen-pool budget metrics; never count oracle parent labels as inputs."""
import numpy as np
import pandas as pd


def high_yield(delta, eligible):
    good = delta[eligible]
    if not len(good):
        return np.zeros(len(delta), dtype=bool)
    threshold = np.quantile(good, .9)
    return eligible & (delta >= threshold)


def curves(arrays, groups, score, budgets, method):
    rows = []
    for g in groups.itertuples():
        index = np.flatnonzero((arrays['group'] == g.group) & arrays['unique'])
        delta, eligible = arrays['delta'][index], arrays['eligible'][index]
        target = high_yield(delta, eligible)
        order = np.argsort(-score[index], kind='stable')
        best = float(np.max(np.r_[0., delta[eligible]]))
        for fraction in budgets:
            k = max(1, int(np.ceil(len(index)*fraction-1.e-9)))
            kept = order[:k]
            chosen_gain = float(np.max(np.r_[0., delta[kept][eligible[kept]]]))
            rows.append(dict(group=g.group, seed=g.seed, tti=g.tti, origin=g.origin,
                split=g.split, method=method, budget=fraction, candidates=len(index),
                exact_evaluations=k, actual_fraction=k/len(index), high_yield_count=int(target.sum()),
                recall=float(target[kept].sum()/target.sum()) if target.any() else np.nan,
                any_high_yield=float(target[kept].any()) if target.any() else np.nan,
                oracle_gain=best, retained_gain=chosen_gain, regret=best-chosen_gain,
                retained_gain_fraction=chosen_gain/best if best > 0 else np.nan))
    return pd.DataFrame(rows)

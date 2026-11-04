"""Cheap process-value inputs and a NumPy MLP ensemble for offline screening.

Exact counterfactual PHY values and future packet outcomes are never inputs.
The target is the risk-score increment relative to a verified frozen root,
not a guarantee about future packet delivery or an optimal suffix value.
"""
import numpy as np

from .physics import COUNT, LAYER, DEQUEUE, BLER, SINR
from .deadline import urgency


def risk_from_expected(problem, expected):
    urgent = urgency(problem)
    target = np.minimum(problem.queue, 2400.)
    return (expected @ (problem.weights + .5*urgent)
            - .5*(np.maximum(target-expected, 0)*urgent).sum(-1)
            + .5*(np.minimum(expected, target)*problem.deadline[:, 5]).sum(-1))


def cheap_completion_features(graph, base, actions):
    """Whole-TB effects, retaining six largest causal resource/channel changes.

    Channel inputs are covariance traces and observed per-PRG SINR hints;
    there is no counterfactual MMSE solve or native BLER evaluation here.
    The scalar control freezes verified-root BLER (unserved UE: .5).
    """
    p = graph.problem
    actions = p.actions(actions)
    assigned = actions[:, p.serving, :] == np.arange(p.U)[None, :, None]
    count = assigned.sum(-1)
    layers = np.where(assigned, p.caps, 5).min(-1)
    layers = np.where(count > 0, layers, 0)
    dequeue = np.minimum(p.queue, p.tbs[np.arange(p.U)[None], count, layers])
    root = base.values[0]
    root_count = root[:, COUNT]
    root_layer = np.maximum(root[:, LAYER], 0)
    reliability = np.where(root_count > 0, 1-np.maximum(root[:, BLER], 0), .5)
    expected = dequeue*reliability
    root_expected = root[:, DEQUEUE]*reliability
    proxy = risk_from_expected(p, expected)-risk_from_expected(p, root_expected)

    def rough_db(a, mask, n, l):
        active = a >= 0
        received = graph.received.copy()
        received[np.arange(p.U), :, p.serving] = 0
        denominator = 4*p.noise + np.einsum('bcp,upc->bup', active, received)
        root_active = base.actions[0] >= 0
        original = 4*p.noise + np.einsum('cp,upc->up', root_active, received)
        if p.sinr_hint is None:
            signal = graph.received[np.arange(p.U), :, p.serving]
            gamma = np.broadcast_to(signal[None]/denominator, mask.shape)
        else:
            include = np.arange(4)[None, None, None] < l[..., None, None]
            hints = (p.sinr_hint[None]*include).sum(-1)/np.maximum(l[..., None], 1)
            gamma = hints * original[None]/denominator
        mean = (gamma*mask).sum(-1)/np.maximum(n, 1)
        return np.where(n > 0, 10*np.log10(np.maximum(mean, 1.e-12)), 0)

    db = rough_db(actions, assigned, count, layers)
    root_assigned = base.actions[:, p.serving, :] == np.arange(p.U)[None, :, None]
    root_db = rough_db(base.actions, root_assigned, root_count[None], root_layer[None])[0]
    dd = dequeue-root[:, DEQUEUE]
    dc = count-root_count
    dl = layers-root_layer
    ds = db-root_db
    changed = np.any(assigned != root_assigned, axis=-1)
    # A trace-ratio activity change also affects UEs whose resources are fixed.
    affected = changed | (np.abs(ds) > 1.e-7)
    urgent = urgency(p)
    ue = np.stack(np.broadcast_arrays(
        dc/p.P, dl/4, dd/2400., ds/30, count/p.P, layers/4,
        dequeue/np.maximum(p.queue, 2400.), db/30,
        reliability[None], p.weights[None]/4,
        np.log1p(p.queue[None]/2400.), p.mcs[None]/27,
        urgent[None], p.deadline[None, :, 5]), axis=-1)
    priority = np.abs(dd)*p.weights + np.minimum(p.queue, 2400)*np.abs(ds)/30
    priority = np.where(affected, priority+1.e-6, -1)
    order = np.argsort(-priority, axis=1, kind='stable')[:, :6]
    picked = np.take_along_axis(ue, order[..., None], axis=1)
    picked *= np.take_along_axis(affected, order, axis=1)[..., None]
    if picked.shape[1] < 6:
        picked = np.pad(picked, ((0, 0), (0, 6-picked.shape[1]), (0, 0)))
    delta_expected = dd*reliability
    summary = np.stack((proxy/2400., dd.sum(-1)/2400.,
        np.maximum(dd, 0).sum(-1)/2400., np.minimum(dd, 0).sum(-1)/2400.,
        delta_expected@p.weights/2400., delta_expected@urgent/2400.,
        affected.sum(-1)/p.U, changed.sum(-1)/p.U,
        (actions >= 0).sum((1, 2))/(p.C*p.P),
        ((actions >= 0) != (base.actions[0] >= 0)).sum((1, 2))/(p.C*p.P),
        (ds*affected).sum(-1)/(30*np.maximum(affected.sum(-1), 1)),
        np.abs(ds).max(-1)/30), axis=-1)
    result = np.concatenate((summary, picked.reshape(len(actions), -1)), axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite cheap value features')
    return result, proxy


class ProcessValue:
    """Two heads: root-relative gain and auxiliary parent-relative increment."""
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as saved:
            if int(saved['version']) != 1:
                raise ValueError('Unsupported process value version')
            self.features = int(saved['features'])
            for key in ('mean', 'scale', 'w1', 'b1', 'w2', 'b2', 'w3', 'b3'):
                setattr(self, key, saved[key].copy())

    def predict(self, x):
        x = np.clip((x[:, :self.features]-self.mean)/self.scale, -10, 10)
        outputs = []
        for i in range(len(self.w1)):
            h = np.maximum(x@self.w1[i].T+self.b1[i], 0)
            h = np.maximum(h@self.w2[i].T+self.b2[i], 0)
            outputs.append(h@self.w3[i].T+self.b3[i])
        z = np.asarray(outputs)*2400.
        return z.mean(0), z.std(0)


def conformal_upper_quantile(values, alpha):
    """Finite-sample order statistic; insufficient calibration => +infinity."""
    values = np.sort(np.asarray(values, dtype=float))
    rank = int(np.ceil((len(values)+1)*(1-alpha)))
    return np.inf if rank > len(values) else float(values[rank-1])

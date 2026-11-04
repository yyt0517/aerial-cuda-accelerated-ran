"""Autoregressive, candidate-conditioned joint block repair.

Every next assignment depends on earlier assignments in the same destroyed
block. The network sees causal PHY/queue summaries and logical partial-TB
statistics, not exact candidate scores. Only completed schedules are verified.
"""
from pathlib import Path
import numpy as np

from .physics import COUNT, LAYER, DEQUEUE, BLER, EXPECTED, SINR
from .learning import block_features, BYTE_SCALE

REPAIR_FEATURE_VERSION = 1


def options(problem, cell, prg):
    return np.r_[-1, np.flatnonzero(problem.legal[cell, prg])].astype(np.int64)


def destroy(action, block):
    result = action.copy()
    for cell, prg in block:
        result[cell, prg] = -1
    return result


def features(graph, base, block, partials, depth, choices, context=None):
    """[beam, legal options, features], vectorized over prefixes and options."""
    p = graph.problem
    partials = np.asarray(partials)
    cell, prg = block[depth]
    prediction = base.values[0]
    if context is None:
        context = block_features(graph, base, [block])[0]
    assigned = partials[:, p.serving, :] == np.arange(p.U)[None, :, None]
    count = assigned.sum(-1)
    layer = np.where(assigned, p.caps[None], 5).min(-1)
    layer = np.where(count > 0, layer, 4)
    u = np.maximum(choices, 0)
    valid = choices >= 0
    b, m = len(partials), len(choices)
    old_capacity = p.tbs[u[None], count[:, u], layer[:, u]]
    new_count = np.minimum(count[:, u]+1, p.P)
    new_layer = np.minimum(layer[:, u], p.caps[u, prg][None])
    new_capacity = p.tbs[u[None], new_count, new_layer]
    queue = p.queue[u]
    before = np.minimum(queue[None], old_capacity)
    after = np.minimum(queue[None], new_capacity)
    hint = p.sinr_hint[u, prg] if p.sinr_hint is not None else np.repeat(graph.quality[u, prg, None], 4, axis=1)
    hint_mean = np.maximum(hint.mean(-1), 1.e-30)
    active = partials[:, :, prg] >= 0
    interference = np.einsum("bc,mc->bm", active, graph.received[u, prg]*(np.arange(p.C)[None] != p.serving[u, None]))
    signal = graph.received[u, prg, p.serving[u]]
    rough_sinr = signal[None]/np.maximum(4*p.noise+interference, 1.e-30)
    # Direct broadcast assignment avoids creating twenty-seven broadcast views
    # and stacking/concatenating them for every autoregressive prefix.
    result = np.empty((b, m, len(context)+27), dtype=np.float32)
    result[..., :len(context)] = context
    raw = result[..., len(context):len(context)+21]
    values = [
        np.log1p(queue/BYTE_SCALE), p.weights[u]/4, p.mcs[u]/27,
        p.caps[u, prg]/4, prediction[u, COUNT]/p.P,
        np.maximum(prediction[u, LAYER], 0)/4, np.maximum(prediction[u, BLER], 0),
        prediction[u, EXPECTED]/np.maximum(queue, BYTE_SCALE),
        np.clip(prediction[u, SINR]/30., -1., 2.),
        count[:, u]/p.P, new_count/p.P, layer[:, u]/4, new_layer/4,
        before/np.maximum(queue[None], BYTE_SCALE), after/np.maximum(queue[None], BYTE_SCALE),
        (after-before)/BYTE_SCALE,
        np.log1p(hint_mean), np.log1p(hint.min(-1)), np.log1p(hint.max(-1)),
        np.log1p(rough_sinr), np.log1p(graph.quality[u, prg]),
    ]
    for index, value in enumerate(values):
        raw[..., index] = value
    raw *= valid[None, :, None]
    for index, value in enumerate([
        ~valid, choices == base.actions[0, cell, prg],
        depth/max(1, len(block)-1), (len(block)-depth)/6,
        active.sum(-1)[:, None]/p.C,
        (partials >= 0).sum((1, 2))[:, None]/(p.C*p.P),
    ]):
        result[..., len(context)+21+index] = value
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite autoregressive repair features")
    return result


class JointRepair:
    # A no-change complete scan will produce identical candidates on repetition.
    deterministic = True
    version = REPAIR_FEATURE_VERSION

    def __init__(self, checkpoint, width=16):
        if width <= 0:
            raise ValueError("Repair width must be positive")
        self.width = width
        with np.load(checkpoint, allow_pickle=False) as saved:
            if int(saved["feature_version"]) != self.version:
                raise ValueError("Unsupported joint-repair feature version")
            for key in ("mean", "scale", "w1", "b1", "w2", "b2", "w3", "b3"):
                setattr(self, key, saved[key].copy())
        self.calls = self.proposals = 0
        self.option_scores = self.logit_calls = 0

    def logits(self, x):
        x = np.clip((x-self.mean)/self.scale, -10., 10.)
        x = np.maximum(x@self.w1.T+self.b1, 0)
        x = np.maximum(x@self.w2.T+self.b2, 0)
        return (x@self.w3.T+self.b3)[..., 0]

    def features(self, graph, base, block, beam, depth, choices, context):
        return features(graph, base, block, beam, depth, choices, context)

    def propose(self, graph, base, block):
        p = graph.problem
        context = block_features(graph, base, [block])[0]
        beam = destroy(base.actions[0], block)[None]
        logp = np.zeros(1)
        for depth, (cell, prg) in enumerate(block):
            choices = options(p, cell, prg)
            scores = self.logits(self.features(graph, base, block, beam, depth, choices, context))
            self.option_scores += scores.size
            self.logit_calls += 1
            scores -= scores.max(-1, keepdims=True)
            scores -= np.log(np.exp(scores).sum(-1, keepdims=True))
            joint = logp[:, None]+scores
            order = np.argsort(-joint.ravel(), kind="stable")[:self.width]
            parent, value = np.divmod(order, len(choices))
            beam = beam[parent].copy()
            beam[:, cell, prg] = choices[value]
            logp = joint.ravel()[order]
        self.calls += 1
        self.proposals += len(beam)
        return p.actions(beam)


def export_model(path, model, mean, scale):
    np.savez(path, feature_version=REPAIR_FEATURE_VERSION, mean=mean, scale=scale,
        w1=model[0].weight.detach().cpu().numpy(), b1=model[0].bias.detach().cpu().numpy(),
        w2=model[2].weight.detach().cpu().numpy(), b2=model[2].bias.detach().cpu().numpy(),
        w3=model[4].weight.detach().cpu().numpy(), b3=model[4].bias.detach().cpu().numpy())

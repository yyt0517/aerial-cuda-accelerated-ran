"""Equivalent autoregressive decoding with decision-local feature reuse.

Weights, float32 feature values, logits and stable beam ordering follow the
reference decoder. Caches belong to one graph (one solve), with incumbent data
invalidated by Evaluation identity. Prefix resources are updated only when a
previously destroyed token is assigned. No cache survives a changed graph.
"""
import numpy as np

from .learning import BYTE_SCALE, token_features, pool_block_features
from .neural_repair import JointRepair, options, destroy
from .physics import COUNT, LAYER, BLER, EXPECTED, SINR


class DecodeCache:
    def __init__(self, graph, owner):
        self.graph = graph
        self.owner = owner
        self.base = None
        self.nodes = {}
        self.contexts = {}

    def set_base(self, base):
        if self.base is base:
            return
        p = self.graph.problem
        self.base = base
        self.tokens = token_features(self.graph, base)
        self.owner.token_builds += 1
        self.contexts.clear()
        prediction = base.values[0]
        self.prediction = np.stack((
            prediction[:, COUNT]/p.P,
            np.maximum(prediction[:, LAYER], 0)/4,
            np.maximum(prediction[:, BLER], 0),
            prediction[:, EXPECTED]/np.maximum(p.queue, BYTE_SCALE),
            np.clip(prediction[:, SINR]/30., -1., 2.),
        ), axis=-1)

    def context(self, block):
        if block not in self.contexts:
            self.contexts[block] = pool_block_features(self.tokens, [block], self.graph.problem.P)[0]
            self.owner.context_builds += 1
        return self.contexts[block]

    def node(self, cell, prg):
        key = cell, prg
        if key not in self.nodes:
            p, graph = self.graph.problem, self.graph
            choices = options(p, cell, prg)
            u = np.maximum(choices, 0)
            hint = (p.sinr_hint[u, prg] if p.sinr_hint is not None
                    else np.repeat(graph.quality[u, prg, None], 4, axis=1))
            # Keep the reference arithmetic and float32 conversion order.
            static = np.zeros((len(choices), 21), dtype=np.float32)
            for index, value in (
                (0, np.log1p(p.queue[u]/BYTE_SCALE)), (1, p.weights[u]/4),
                (2, p.mcs[u]/27), (3, p.caps[u, prg]/4),
                (16, np.log1p(np.maximum(hint.mean(-1), 1.e-30))),
                (17, np.log1p(hint.min(-1))), (18, np.log1p(hint.max(-1))),
                (20, np.log1p(graph.quality[u, prg])),
            ):
                static[:, index] = value
            extra = None
            if self.owner.version == 2:
                extra = p.deadline[u].copy()
                extra[choices < 0] = 0
            received = graph.received[u, prg]*(np.arange(p.C)[None] != p.serving[u, None])
            signal = graph.received[u, prg, p.serving[u]]
            self.nodes[key] = choices, u, static, extra, received, signal
            self.owner.option_builds += 1
        return self.nodes[key]

    def features(self, block, depth, count, layer, active):
        p = self.graph.problem
        cell, prg = block[depth]
        choices, u, static, extra, received, signal = self.node(cell, prg)
        context = self.context(block)
        b, m, n = len(count), len(choices), len(context)
        result = np.empty((b, m, n+27+(8 if extra is not None else 0)), dtype=np.float32)
        result[..., :n] = context
        raw = result[..., n:n+21]
        raw[:] = static
        raw[..., 4:9] = self.prediction[u]
        current_count = count[:, u]
        current_layer = layer[:, u]
        new_count = np.minimum(current_count+1, p.P)
        new_layer = np.minimum(current_layer, p.caps[u, prg][None])
        before = np.minimum(p.queue[u][None], p.tbs[u[None], current_count, current_layer])
        after = np.minimum(p.queue[u][None], p.tbs[u[None], new_count, new_layer])
        denominator = np.maximum(p.queue[u][None], BYTE_SCALE)
        interference = np.einsum('bc,mc->bm', active[:, :, prg], received)
        rough_sinr = signal[None]/np.maximum(4*p.noise+interference, 1.e-30)
        for index, value in (
            (9, current_count/p.P), (10, new_count/p.P),
            (11, current_layer/4), (12, new_layer/4),
            (13, before/denominator), (14, after/denominator),
            (15, (after-before)/BYTE_SCALE), (19, np.log1p(rough_sinr)),
        ):
            raw[..., index] = value
        valid = choices >= 0
        raw *= valid[None, :, None]
        for index, value in enumerate((
            ~valid, choices == self.base.actions[0, cell, prg],
            depth/max(1, len(block)-1), (len(block)-depth)/6,
            active[:, :, prg].sum(-1)[:, None]/p.C,
            count.sum(-1)[:, None]/(p.C*p.P),
        )):
            result[..., n+21+index] = value
        if extra is not None:
            result[..., n+27:] = extra
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite autoregressive repair features')
        return result


class CachedJointRepair(JointRepair):
    def __init__(self, checkpoint, width=16):
        super().__init__(checkpoint, width)
        self._cache = None
        self.token_builds = self.context_builds = self.option_builds = 0
        self.prefix_initializations = 0

    def cache_for(self, graph, base):
        if self._cache is None or self._cache.graph is not graph:
            self._cache = DecodeCache(graph, self)
        self._cache.set_base(base)
        return self._cache

    def propose(self, graph, base, block):
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
        self.prefix_initializations += 1
        for depth, (cell, prg) in enumerate(block):
            choices = cache.node(cell, prg)[0]
            scores = self.logits(cache.features(block, depth, count, layer, active))
            self.option_scores += scores.size
            self.logit_calls += 1
            scores -= scores.max(-1, keepdims=True)
            scores -= np.log(np.exp(scores).sum(-1, keepdims=True))
            joint = logp[:, None]+scores
            # Stable full ordering is intentional: ties must match the reference.
            order = np.argsort(-joint.ravel(), kind='stable')[:self.width]
            parent, value = np.divmod(order, len(choices))
            selected = choices[value]
            edits = edits[parent]
            edits[:, depth] = selected
            count, layer, active = count[parent], layer[parent], active[parent]
            active[:, cell, prg] = selected >= 0
            rows = np.flatnonzero(selected >= 0)
            users = selected[rows]
            count[rows, users] += 1
            layer[rows, users] = np.minimum(layer[rows, users], p.caps[users, prg])
            logp = joint.ravel()[order]
        beam = np.broadcast_to(root, (len(edits), p.C, p.P)).copy()
        for depth, (cell, prg) in enumerate(block):
            beam[:, cell, prg] = edits[:, depth]
        self.calls += 1
        self.proposals += len(beam)
        return p.actions(beam)


class CachedDeadlineRepair(CachedJointRepair):
    version = 2

    def features(self, graph, base, block, beam, depth, choices, context):
        from .deadline import deadline_features
        return deadline_features(graph, base, block, beam, depth, choices, context)

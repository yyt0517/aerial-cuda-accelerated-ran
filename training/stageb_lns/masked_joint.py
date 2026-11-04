"""Experimental MACSIM-inspired joint heads and MaskCO-style conditional repair.

No active scheduler imports this module. Static channel features are computed
once per physical context; dynamic features are bound to a verified incumbent.
The decoder changes only the requested mask and never uses candidate PHY labels.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
import time
import numpy as np

from .physics import COUNT, LAYER, CAPACITY, DEQUEUE, SINR, BLER, EXPECTED
from .learning import token_features, BYTE_SCALE

VERSION = 1
DIM = 32
HEADS = 4
PHI_DIM = 10
PHI_NAMES = ('count_before', 'count_after', 'layer_before', 'layer_after',
             'dequeue_before', 'dequeue_after', 'marginal_dequeue', 'rough_sinr',
             'blank', 'copy_incumbent')


def hermitian_real(matrix):
    """Lossless real coordinates for a complex Hermitian 4x4 matrix."""
    i, j = np.triu_indices(4, 1)
    return np.concatenate((np.diagonal(matrix, axis1=-2, axis2=-1).real,
                           matrix[..., i, j].real, matrix[..., i, j].imag), -1)


class JointContext:
    def __init__(self, phy, graph):
        p = self.problem = phy.problem
        self.graph = graph
        self.phy = phy
        self.N = p.C*p.P
        self.slots, self.users = np.nonzero(p.legal.reshape(self.N, p.U))
        self.prgs = self.slots % p.P
        u, prg = self.users, self.prgs
        den = np.maximum(4*p.noise + graph.received.sum(-1), 1.e-30)
        cov = phy.covariance[u, prg]/den[u, prg, None, None, None]
        signal = phy.signal[u, prg]/np.sqrt(den[u, prg, None, None])
        spatial = np.concatenate((hermitian_real(cov).reshape(len(u), p.C*16),
                                  signal.real.reshape(len(u), 16),
                                  signal.imag.reshape(len(u), 16),
                                  (p.noise/den[u, prg])[:, None]), -1)
        hints = (p.sinr_hint[u, prg] if p.sinr_hint is not None else
                 np.repeat(graph.quality[u, prg, None], 4, axis=-1))
        cheap = np.column_stack((p.caps[u, prg]/4, np.log1p(hints.mean(-1))/8,
                                 np.log1p(hints.min(-1))/8, np.log1p(hints.max(-1))/8,
                                 np.log1p(graph.quality[u, prg])/8,
                                 graph.received[u, prg]/den[u, prg, None]))
        self.edges = np.ascontiguousarray(np.concatenate((cheap, spatial), -1), dtype=np.float32)
        self.edge_dim = self.edges.shape[-1]
        self.legal = np.column_stack((np.ones(self.N, bool), p.legal.reshape(self.N, p.U)))
        self.received = np.ascontiguousarray(graph.received, dtype=np.float64)
        self.static_embedding = None
        self.static_owner = None
        self.base = None
        if not np.isfinite(self.edges).all():
            raise ValueError('Nonfinite masked-joint channel features')

    def dynamic(self, base, floor):
        p = self.problem
        if base.owner is not self.phy or len(base.actions) != 1:
            raise ValueError('Masked-joint context needs its own single verified incumbent')
        v = base.values[0]
        scale = np.maximum(p.queue, BYTE_SCALE)
        served = v[:, COUNT] > 0
        ue = np.column_stack((np.log1p(p.queue/BYTE_SCALE), p.weights/4, p.mcs/27,
                              v[:, COUNT]/p.P, np.maximum(v[:, LAYER], 0)/4,
                              v[:, CAPACITY]/scale, v[:, DEQUEUE]/scale,
                              np.maximum(v[:, BLER], 0), v[:, EXPECTED]/scale,
                              np.where(served, np.clip(v[:, SINR]/30, -1, 2), 0),
                              served, np.maximum(p.queue-v[:, CAPACITY], 0)/scale)).astype(np.float32)
        slots = token_features(self.graph, base).astype(np.float32)
        assigned = (base.actions[0].ravel()[self.slots] == self.users).astype(np.float32)
        globals_ = np.array([v[:, EXPECTED]@p.weights/BYTE_SCALE/p.U,
                             floor/BYTE_SCALE/p.U,
                             (v[:, DEQUEUE].sum()-floor)/BYTE_SCALE/p.U,
                             (base.actions[0] >= 0).mean()], dtype=np.float32)
        return dict(ue=ue, slots=slots, assigned=assigned, globals=globals_,
                    action=base.actions[0].ravel().copy())


def make_torch_model(edge_dim=89, dim=DIM, heads=HEADS):
    """Training model; deployment exports the same weights to NumPy."""
    import torch
    from torch import nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.edge = nn.Linear(edge_dim, dim)
            self.ue = nn.Linear(12, dim)
            self.slot = nn.Linear(17, dim)
            self.uu = nn.ModuleList([nn.Linear(2*dim, dim) for _ in range(2)])
            self.ss = nn.ModuleList([nn.Linear(3*dim, dim) for _ in range(2)])
            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.context = nn.Linear(dim+4, dim)
            self.edge_bias = nn.Linear(dim, 1)
            self.modes = nn.Parameter(torch.randn(heads, dim)*.08)
            self.blank = nn.Parameter(torch.randn(dim)*.08)
            self.beta = nn.Parameter(torch.zeros(heads, PHI_DIM))
            self.copy_bias = nn.Parameter(torch.tensor(.5))
            self.blank_bias = nn.Parameter(torch.tensor(-.5))

        def encode(self, data):
            e = torch.relu(self.edge(data['edges']))
            u, s = torch.relu(self.ue(data['ue'])), torch.relu(self.slot(data['slots']))
            ei, eu = data['edge_slots'], data['edge_users']
            weight = 1+3*data['assigned']
            ud = torch.zeros(len(u), device=u.device).index_add(0, eu, weight).clamp_min(1)
            sd = torch.zeros(len(s), device=s.device).index_add(0, ei, weight).clamp_min(1)
            for ul, sl in zip(self.uu, self.ss):
                msg = torch.relu(e+u[eu]+s[ei])*weight[:, None]
                um = torch.zeros_like(u).index_add(0, eu, msg)/ud[:, None]
                sm = torch.zeros_like(s).index_add(0, ei, msg)/sd[:, None]
                view = s.reshape(data['C'], data['P'], -1)
                peers = ((view.sum(0)[None]-view)/max(1, data['C']-1)).reshape_as(s)
                u = torch.relu(ul(torch.cat((u, um), -1)))
                s = torch.relu(sl(torch.cat((s, sm, peers), -1)))
            bias = torch.zeros((len(s), len(u)+1), device=s.device)
            bias[ei, eu+1] = self.edge_bias(e)[:, 0]
            keys = self.key(torch.cat((self.blank[None], u), 0))
            return s, keys, bias

        def logits(self, data, encoded, block):
            s, keys, bias = encoded
            ctx = self.context(torch.cat((s[block].mean(0), data['globals']), -1))
            query = torch.tanh(self.query(s[block])[None]+ctx[None, None]+self.modes[:, None])
            out = query@keys.T/np.sqrt(dim)+bias[block][None]
            copy = (torch.arange(len(keys), device=s.device)[None] == data['action'][block, None]+1)
            out = out+self.copy_bias*copy[None]
            blank = torch.arange(len(keys), device=s.device) == 0
            out = out+self.blank_bias*blank
            return out.masked_fill(~data['legal'][block][None], -1.e9)

    return Model()


def torch_data(context, dynamic):
    import torch
    result = {k: torch.from_numpy(v) for k, v in dynamic.items()}
    result.update(edges=torch.from_numpy(context.edges),
                  edge_slots=torch.from_numpy(context.slots),
                  edge_users=torch.from_numpy(context.users),
                  legal=torch.from_numpy(context.legal), C=context.problem.C, P=context.problem.P)
    return result


class MaskedJointRepair:
    def __init__(self, checkpoint=None, state=None, dim=DIM, heads=HEADS, compiled=True):
        if checkpoint is not None:
            with np.load(checkpoint, allow_pickle=False) as data:
                if int(data['version']) != VERSION:
                    raise ValueError('Unsupported masked-joint checkpoint')
                dim, heads = int(data['dim']), int(data['heads'])
                state = {k[6:]: data[k].copy() for k in data.files if k.startswith('state.')}
        if state is None:
            raise ValueError('Checkpoint or weights required')
        self.state = {k: np.asarray(v, dtype=np.float32) for k, v in state.items()}
        self.dim, self.heads, self.compiled = dim, heads, compiled
        self.stats = dict(static_encodes=0, dynamic_encodes=0, joint_batches=0,
                          generated=0, decode_steps=0, encoder_ms=0., head_ms=0., decoder_ms=0.)
        self.library = decoder_library() if compiled else None

    def linear(self, x, name):
        return x@self.state[name+'.weight'].T+self.state[name+'.bias']

    def encode(self, context, dynamic):
        start = time.perf_counter()
        if context.static_owner is not self:
            context.static_embedding = np.maximum(self.linear(context.edges, 'edge'), 0)
            context.static_owner = self
            self.stats['static_encodes'] += 1
        e = context.static_embedding
        u = np.maximum(self.linear(dynamic['ue'], 'ue'), 0)
        s = np.maximum(self.linear(dynamic['slots'], 'slot'), 0)
        ei, eu = context.slots, context.users
        weight = 1+3*dynamic['assigned']
        ud = np.maximum(np.bincount(eu, weights=weight, minlength=len(u)), 1).astype(np.float32)
        sd = np.maximum(np.bincount(ei, weights=weight, minlength=len(s)), 1).astype(np.float32)
        for r in range(2):
            msg = np.maximum(e+u[eu]+s[ei], 0)*weight[:, None]
            um, sm = np.zeros_like(u), np.zeros_like(s)
            np.add.at(um, eu, msg); np.add.at(sm, ei, msg)
            um /= ud[:, None]; sm /= sd[:, None]
            view = s.reshape(context.problem.C, context.problem.P, -1)
            peers = ((view.sum(0)[None]-view)/max(1, context.problem.C-1)).reshape(s.shape)
            u = np.maximum(self.linear(np.concatenate((u, um), -1), f'uu.{r}'), 0)
            s = np.maximum(self.linear(np.concatenate((s, sm, peers), -1), f'ss.{r}'), 0)
        bias = np.zeros((len(s), len(u)+1), np.float32)
        bias[ei, eu+1] = self.linear(e, 'edge_bias')[:, 0]
        keys = self.linear(np.concatenate((self.state['blank'][None], u)), 'key')
        self.stats['dynamic_encodes'] += 1
        self.stats['encoder_ms'] += (time.perf_counter()-start)*1000
        return s, keys, bias

    def logits(self, context, dynamic, encoded, block):
        s, keys, bias = encoded
        block = np.asarray(block, dtype=np.int64)
        ctx = self.linear(np.r_[s[block].mean(0), dynamic['globals']], 'context')
        query = np.tanh(self.linear(s[block], 'query')[None]+ctx[None, None]+self.state['modes'][:, None])
        out = query@keys.T/np.float32(np.sqrt(self.dim))+bias[block][None]
        out += self.state['copy_bias']*(np.arange(len(keys))[None] == dynamic['action'][block, None]+1)[None]
        out[..., 0] += self.state['blank_bias']
        return np.where(context.legal[block][None], out, -1.e9).astype(np.float32)

    def propose_many(self, context, base, floor, blocks, rng, stochastic_all=False, return_trace=False):
        dynamic = context.dynamic(base, floor)
        encoded = self.encode(context, dynamic)
        start = time.perf_counter()
        logits = [self.logits(context, dynamic, encoded, b) for b in blocks]
        self.stats['head_ms'] += (time.perf_counter()-start)*1000
        self.stats['joint_batches'] += 1
        candidates, traces = [], []
        for block, values in zip(blocks, logits):
            start = time.perf_counter()
            modes = np.repeat(np.arange(self.heads), 2).astype(np.int64)
            noise = rng.gumbel(size=(len(modes), len(block), context.problem.U+1)).astype(np.float64)
            if not stochastic_all:
                noise[::2] = 0
            actions = decode(context, base.actions[0], block, values, self.state['beta'], modes, noise,
                             self.library)
            self.stats['decoder_ms'] += (time.perf_counter()-start)*1000
            self.stats['decode_steps'] += len(modes)*len(block)
            self.stats['generated'] += len(actions)
            candidates.extend(actions)
            if return_trace:
                traces.append(dict(block=np.asarray(block), actions=actions, modes=modes, noise=noise,
                                   dynamic=dynamic))
        result = context.problem.actions(np.asarray(candidates))
        return (result, traces) if return_trace else result


def prefix_features(context, action, block, targets=None):
    """Teacher-forced features, indexed by canonical step and global UE class.

    Only already assigned target positions are used. Missing positions really
    are blank for resource accounting; the original action is a separate copy
    reference. Targets at future positions are never read into features.
    """
    p = context.problem
    block = np.asarray(block, dtype=np.int64)
    partial = action.ravel().copy(); partial[block] = -1
    count = np.bincount(partial[partial >= 0], minlength=p.U)
    layer = np.full(p.U, 4, dtype=np.int64)
    for slot in np.flatnonzero(partial >= 0):
        u = partial[slot]; layer[u] = min(layer[u], p.caps[u, slot % p.P])
    result = []
    for depth, slot in enumerate(block):
        c, prg = divmod(int(slot), p.P)
        before = np.minimum(p.queue, p.tbs[np.arange(p.U), count, layer])
        n = np.minimum(count+1, p.P); l = np.minimum(layer, p.caps[:, prg])
        after = np.minimum(p.queue, p.tbs[np.arange(p.U), n, l])
        active = partial.reshape(p.C, p.P)[:, prg] >= 0
        incoming = context.received[:, prg].copy()
        incoming[np.arange(p.U), p.serving] = 0
        signal = context.received[np.arange(p.U), prg, p.serving]
        rough = signal/np.maximum(4*p.noise+incoming@active, 1.e-30)
        phi = np.zeros((p.U+1, PHI_DIM), np.float32)
        phi[1:, :8] = np.column_stack((count/p.P, n/p.P, layer/4, l/4,
                                      before/np.maximum(p.queue, BYTE_SCALE),
                                      after/np.maximum(p.queue, BYTE_SCALE),
                                      (after-before)/BYTE_SCALE, np.log1p(rough)/8))
        phi[0, 8] = 1
        phi[:, 9] = np.arange(p.U+1) == action.ravel()[slot]+1
        result.append(phi)
        if targets is not None:
            u = int(targets[depth])
            partial[slot] = u
            if u >= 0:
                count[u] += 1; layer[u] = min(layer[u], p.caps[u, prg])
    return np.stack(result)


_LIBRARY = None


def decoder_library():
    global _LIBRARY
    if _LIBRARY is None:
        path = Path(__file__).with_name('_masked_decode.so')
        if not path.exists():
            raise RuntimeError('Build first: python -m training.stageb_lns.build_masked')
        lib = ctypes.CDLL(str(path))
        i64 = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
        f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
        u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
        lib.masked_decode.argtypes = [ctypes.c_int]*5+[ctypes.c_double]+[i64]*5+[f64]*2+[u8,f32,f32,i64,f64,i64]
        lib.masked_decode.restype = ctypes.c_int
        _LIBRARY = lib
    return _LIBRARY


def decode(context, action, block, logits, beta, modes, noise, library=None):
    p = context.problem
    block = np.ascontiguousarray(block, dtype=np.int64)
    if len(block) == 0 or len(np.unique(block)) != len(block) or np.any((block < 0) | (block >= p.C*p.P)):
        raise ValueError('Repair mask must contain distinct valid positions')
    out = np.broadcast_to(action, (len(modes), p.C, p.P)).copy()
    if library is not None:
        code = library.masked_decode(p.U, p.C, p.P, len(block), len(modes), p.noise,
                                     np.ascontiguousarray(action, dtype=np.int64), block, p.serving,
                                     p.caps, p.tbs, p.queue, context.received,
                                     context.legal.astype(np.uint8), np.ascontiguousarray(logits),
                                     np.ascontiguousarray(beta), np.ascontiguousarray(modes),
                                     np.ascontiguousarray(noise), out)
        if code:
            raise RuntimeError('Compiled masked decoder failed')
    else:
        for k, mode in enumerate(modes):
            target = np.full(len(block), -1, np.int64)
            for depth, slot in enumerate(block):
                # Account for every destroyed future position, not just prefix.
                root = action.ravel().copy(); root[block] = -1
                for j in range(depth): root[block[j]] = target[j]
                # Reuse the reference feature routine with the original copy flag.
                phi = prefix_features(context, root.reshape(action.shape), [slot])[0]
                phi[:, 9] = np.arange(p.U+1) == action.ravel()[slot]+1
                scores = logits[mode, depth]+phi@beta[mode]+noise[k, depth]
                scores = np.where(context.legal[slot], scores, -np.inf)
                target[depth] = int(np.argmax(scores))-1
            out[k].ravel()[block] = target
    return p.actions(out)


def export_model(path, model, metadata=None):
    state = {f'state.{k}': v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(path, version=VERSION, dim=model.modes.shape[1], heads=len(model.modes), **state)

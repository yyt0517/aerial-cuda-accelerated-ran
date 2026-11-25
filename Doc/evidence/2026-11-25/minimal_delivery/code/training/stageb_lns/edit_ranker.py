"""Cheap ranking inputs for the original DPLS one-PRG edit neighborhood.

Only incumbent PHY values and causal geometry are inputs. Candidate logical
TB resource changes use [candidate, UE] arrays, never [candidate, UE, PRG]
scans. No candidate MMSE, BLER lookup, or PHY evaluation occurs here. The
heuristic and learned controls share exactly the same feature construction.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
import weakref
import numpy as np

from .physics import COUNT, LAYER, DEQUEUE, BLER, EXPECTED


FEATURE_VERSION = 1
BYTE_SCALE = 3000.
FEATURE_NAMES = (
    'heuristic_gain', 'fixed_reliability_gain', 'activity_risk_gain',
    'dequeue_delta', 'positive_dequeue_delta', 'negative_dequeue_delta',
    'resource_changed_ues', 'edited_slots', 'muted_cells', 'activated_cells',
    'identity_swaps', 'activity_changes', 'activity_affected_victims',
    'root_j', 'root_dequeue_slack', 'root_activity', 'new_dequeue_slack',
    'positive_count_delta', 'negative_count_delta', 'positive_layer_delta',
    'negative_layer_delta', 'unique_bottleneck_removals', 'newly_served_ues',
    'newly_unserved_ues', 'newly_drained_ues', 'positive_victim_risk',
    'negative_victim_risk', 'mean_tb_log_ratio', 'max_tb_log_ratio', 'min_tb_log_ratio',
    'recipient_root_count', 'donor_root_count', 'recipient_root_layer', 'donor_root_layer',
    'recipient_mcs', 'donor_mcs', 'recipient_log_queue', 'donor_log_queue',
    'recipient_log_quality', 'donor_log_quality', 'recipient_prg_layer_cap',
    'donor_prg_layer_cap', 'recipient_root_bler', 'donor_root_bler',
    'recipient_weight', 'donor_weight', 'largest_positive_fixed_gain',
    'largest_negative_fixed_gain',
)
FEATURE_DIM = len(FEATURE_NAMES)


@lru_cache(maxsize=1)
def feature_library():
    path=Path(__file__).with_name('_edit_features.so')
    if not path.exists():
        raise RuntimeError('Build candidate features first: python -m training.stageb_lns.build_edit_ranker')
    lib=ctypes.CDLL(str(path))
    i=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    f=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    x=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    m=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
    lib.edit_features_create.argtypes=[ctypes.c_int]*3+[ctypes.c_double]+[i]*4+[m]+[f]*4
    lib.edit_features_create.restype=ctypes.c_void_p
    lib.edit_features_destroy.argtypes=[ctypes.c_void_p]
    lib.edit_features_destroy.restype=None
    lib.edit_features_base.argtypes=[ctypes.c_void_p,i,f]
    lib.edit_features_base.restype=ctypes.c_int
    lib.edit_features_evaluate.argtypes=[ctypes.c_void_p,ctypes.c_int,i,ctypes.c_double,x,f,f]
    lib.edit_features_evaluate.restype=ctypes.c_int
    return lib


class EditFeatureContext:
    """Decision-local causal geometry; incumbent-dependent state is invalidated.

    Construct once per decision from its NativePhy or ProposalGraph and reuse
    every search round. No cache may survive changed CSI/problem geometry.
    """
    def __init__(self, problem, phy_or_graph, *, use_native=True):
        if phy_or_graph.problem is not problem:
            raise ValueError('Ranking geometry belongs to a different problem')
        self.problem = problem
        self.received = phy_or_graph.received
        p = problem
        self.other_received = self.received.copy()
        self.other_received[np.arange(p.U), :, p.serving] = 0.
        signal = self.received[np.arange(p.U), :, p.serving]
        self.quality = signal/np.maximum(4*p.noise+self.other_received.sum(-1), 1.e-30)
        self.base = None
        self.use_native = bool(use_native)
        self.native_context = None
        if self.use_native:
            self.lib=feature_library()
            self.native_context=self.lib.edit_features_create(p.U,p.C,p.P,p.noise,
                p.serving,p.mcs,p.caps,p.tbs,np.ascontiguousarray(p.legal,np.uint8),
                p.queue,p.weights,self.other_received,self.quality)
            if not self.native_context:
                raise RuntimeError('Could not create edit-feature context')
            self._release=weakref.finalize(self,self.lib.edit_features_destroy,self.native_context)

    def set_base(self, base):
        if self.use_native and not self._release.alive:
            raise RuntimeError('Edit-feature context is closed')
        if base.owner.problem is not self.problem or base.owner.received is not self.received:
            raise ValueError('Ranking base belongs to a different PHY context')
        if len(base.actions) != 1:
            raise ValueError('Ranking needs a single verified incumbent')
        if self.base is base:
            return
        self.base = base
        p = self.problem
        root = base.actions[0]
        values = base.values[0]
        assigned = root[p.serving] == np.arange(p.U)[:, None]
        self.count = assigned.sum(1)
        self.layer = np.where(self.count > 0, np.maximum(values[:, LAYER], 0), 4).astype(np.int64)
        caps = np.where(assigned, p.caps, 5)
        smallest = caps.min(1)
        self.smallest = smallest
        self.smallest_count = (caps == smallest[:, None]).sum(1)
        self.second_layer = np.where(caps > smallest[:, None], caps, 5).min(1)
        self.root_active = root >= 0
        self.denominator = 4*p.noise + np.einsum('cp,upc->up', self.root_active, self.other_received)
        self.reliability = np.where(self.count > 0, 1-np.maximum(values[:, BLER], 0), .5)
        self.failure = np.where(self.count > 0, np.maximum(values[:, BLER], 0), .5)
        if self.use_native and self.lib.edit_features_base(self.native_context,base.actions[0],values):
            raise RuntimeError('Could not set edit-feature incumbent')

    def close(self):
        if self.use_native:
            self._release()


def candidate_features(problem, phy_or_context, base, actions, *, floor=None):
    """Compiled implementation of the independently checked reference below.

    Public shape/dtype checks are followed by in-kernel legality and original
    neighborhood checks. Context construction, feature calls and destruction
    belong inside decision timing; shared-library compilation/loading do not.
    """
    p=problem
    context=phy_or_context if isinstance(phy_or_context,EditFeatureContext) else EditFeatureContext(p,phy_or_context)
    if not context.use_native:
        return candidate_features_reference(p,context,base,actions,floor=floor)
    if context.problem is not p:
        raise ValueError('Ranking context belongs to a different problem')
    context.set_base(base)
    actions=np.asarray(actions)
    if actions.ndim==2:actions=actions[None]
    if actions.ndim!=3 or actions.shape[1:]!=(p.C,p.P) or not np.issubdtype(actions.dtype,np.integer):
        raise ValueError('Invalid candidate shape or dtype')
    actions=np.ascontiguousarray(actions,np.int64)
    floor=0. if floor is None else float(floor)
    if not np.isfinite(floor) or floor<0:
        raise ValueError('Invalid fixed raw-PFQ floor')
    n=len(actions)
    x=np.empty((n,FEATURE_DIM),np.float32)
    heuristic=np.empty(n,np.float64)
    totals=np.empty(n,np.float64)
    if n:
        error=context.lib.edit_features_evaluate(context.native_context,n,actions,floor,x,heuristic,totals)
        if error:
            raise ValueError(f'Invalid edit-ranking candidate stream (kernel error {error})')
    context.last_dequeue_totals=totals
    return x,heuristic


def candidate_features_reference(problem, phy_or_context, base, actions, *, floor=None):
    """Return ``(float32[N,48], float64[N])`` without candidate PHY calls.

    Candidates must belong to greedy_candidates' neighborhood: at most two
    cells edited on one PRG. Every affected UE's *whole TB* count/min-layer is
    updated, including all its allocations outside that PRG. A trace-ratio
    activity proxy also accounts for resource-unchanged neighboring victims.

    The optional fixed raw-PFQ floor conditions logical dequeue-slack inputs.
    context.last_dequeue_totals exposes exact logical total dequeues for the
    just-processed candidate stream, without an additional PHY calculation.
    A supplied EditFeatureContext avoids rebuilding static geometry. Supplying
    NativePhy/ProposalGraph directly is supported for standalone checks only.
    """
    p = problem
    context = phy_or_context if isinstance(phy_or_context, EditFeatureContext) else EditFeatureContext(p, phy_or_context,use_native=False)
    if context.problem is not p:
        raise ValueError('Ranking context belongs to a different problem')
    context.set_base(base)
    actions = p.actions(actions)
    n = len(actions)
    if not n:
        context.last_dequeue_totals = np.empty(0, np.float64)
        return np.empty((0, FEATURE_DIM), np.float32), np.empty(0, np.float64)
    root, verified = base.actions[0], base.values[0]
    changed = actions != root
    changed_prgs = changed.any(1)
    if (changed_prgs.sum(1) > 1).any() or (changed.sum((1, 2)) > 2).any():
        raise ValueError('Edit ranker supports at most two edits on one PRG')
    prg = changed_prgs.argmax(1)
    rows = np.arange(n)
    old = root[:, prg].T
    new = actions[rows[:, None], np.arange(p.C)[None], prg[:, None]]
    edited = old != new
    removed = edited & (old >= 0)
    added = edited & (new >= 0)
    dc = np.zeros((n, p.U), np.int16)
    candidate_rows = np.broadcast_to(rows[:, None], old.shape)
    np.add.at(dc, (candidate_rows[removed], old[removed]), -1)
    np.add.at(dc, (candidate_rows[added], new[added]), 1)
    donors, recipients = dc < 0, dc > 0
    count = context.count[None] + dc
    caps = p.caps[:, prg].T
    unique_min = (caps == context.smallest[None]) & (context.smallest_count[None] == 1)
    layer = np.broadcast_to(context.layer, count.shape).copy()
    layer = np.where(donors & unique_min, context.second_layer[None], layer)
    layer = np.where(recipients, np.minimum(layer, caps), layer)
    layer = np.where(count > 0, layer, 0)
    if (count < 0).any() or (layer > 4).any():
        raise AssertionError('Invalid logical TB update')
    dequeue = np.minimum(p.queue, p.tbs[np.arange(p.U)[None], count, layer])
    total_dequeue = dequeue.sum(1)
    context.last_dequeue_totals = total_dequeue
    floor = 0. if floor is None else float(floor)
    if not np.isfinite(floor) or floor < 0:
        raise ValueError('Invalid fixed raw-PFQ floor')
    dd = dequeue - verified[None, :, DEQUEUE]
    fixed = dd * (context.reliability*p.weights)[None]
    fixed_gain = fixed.sum(1)

    active = new >= 0
    delta_active = active.astype(np.int8)-context.root_active[:, prg].T
    # Gather only [candidate, UE, cell] trace data for the edited PRG.
    received = context.other_received[:, prg].transpose(1, 0, 2)
    before_denominator = context.denominator[:, prg].T
    after_denominator = before_denominator+np.einsum('nc,nuc->nu', delta_active, received)
    log_ratio = np.log(before_denominator/np.maximum(after_denominator, 1.e-30))
    selected = actions[rows[:, None], p.serving[None], prg[:, None]] == np.arange(p.U)[None]
    tb_log_ratio = np.where(selected, log_ratio/np.maximum(count, 1), 0.)
    victim = selected & (np.abs(log_ratio) > 1.e-12)
    # A bounded reliability-risk proxy, not a predicted exact candidate BLER.
    risk = np.minimum(p.queue, np.maximum(dequeue, verified[None, :, DEQUEUE]))
    risk *= (context.failure*p.weights)[None]
    activity_gain_ue = .5*risk*np.tanh(tb_log_ratio)
    activity_gain = activity_gain_ue.sum(1)
    heuristic = fixed_gain+activity_gain
    resource_changed = dc != 0
    old_layer = np.where(context.count > 0, context.layer, 0)
    dl = layer-old_layer[None]

    def average(values, mask):
        return (np.asarray(values)*mask).sum(1)/np.maximum(mask.sum(1), 1)

    def positive_sum(value):
        return np.maximum(value, 0).sum(1)

    def negative_sum(value):
        return np.minimum(value, 0).sum(1)

    root_j = float(verified[:, EXPECTED]@p.weights)
    root_d = float(verified[:, DEQUEUE].sum())
    log_quality = np.log1p(context.quality[:, prg].T)
    values = (
        heuristic/BYTE_SCALE, fixed_gain/BYTE_SCALE, activity_gain/BYTE_SCALE,
        dd.sum(1)/BYTE_SCALE, positive_sum(dd)/BYTE_SCALE, negative_sum(dd)/BYTE_SCALE,
        resource_changed.sum(1)/p.U, edited.sum(1)/(p.C*p.P),
        (delta_active < 0).sum(1)/p.C, (delta_active > 0).sum(1)/p.C,
        (edited & (old >= 0) & (new >= 0)).sum(1)/p.C,
        (delta_active != 0).sum(1)/p.C, victim.sum(1)/p.U,
        np.full(n, root_j/BYTE_SCALE), np.full(n, (root_d-floor)/BYTE_SCALE),
        np.full(n, context.root_active.mean()), (total_dequeue-floor)/BYTE_SCALE,
        positive_sum(dc)/p.P, negative_sum(dc)/p.P,
        positive_sum(dl)/4, negative_sum(dl)/4,
        (donors & unique_min).sum(1)/p.U,
        ((context.count[None] == 0) & (count > 0)).sum(1)/p.U,
        ((context.count[None] > 0) & (count == 0)).sum(1)/p.U,
        ((verified[None, :, DEQUEUE] < p.queue) & (dequeue >= p.queue)).sum(1)/p.U,
        positive_sum(activity_gain_ue)/BYTE_SCALE, negative_sum(activity_gain_ue)/BYTE_SCALE,
        average(tb_log_ratio, victim)/3, tb_log_ratio.max(1)/3, tb_log_ratio.min(1)/3,
        average(context.count[None], recipients)/p.P, average(context.count[None], donors)/p.P,
        average(old_layer[None], recipients)/4, average(old_layer[None], donors)/4,
        average(p.mcs[None], recipients)/27, average(p.mcs[None], donors)/27,
        average(np.log1p(p.queue/BYTE_SCALE)[None], recipients),
        average(np.log1p(p.queue/BYTE_SCALE)[None], donors),
        average(log_quality, recipients), average(log_quality, donors),
        average(caps, recipients)/4, average(caps, donors)/4,
        average(context.failure[None], recipients), average(context.failure[None], donors),
        average(p.weights[None], recipients)/4, average(p.weights[None], donors)/4,
        np.maximum(fixed, 0).max(1)/BYTE_SCALE, np.minimum(fixed, 0).min(1)/BYTE_SCALE,
    )
    x = np.stack(values, 1).astype(np.float32)
    if x.shape != (n, FEATURE_DIM) or not np.isfinite(x).all() or not np.isfinite(heuristic).all():
        raise ValueError('Invalid edit-ranking features')
    return x, heuristic


class EditRanker:
    """Small NumPy residual MLP; zero final layer equals physical heuristic."""
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            if int(z['feature_version']) != FEATURE_VERSION:
                raise ValueError('Unsupported edit-ranking feature version')
            if tuple(z['feature_names'].tolist()) != FEATURE_NAMES:
                raise ValueError('Edit-ranking feature order mismatch')
            for key in ('mean', 'scale', 'w1', 'b1', 'w2', 'b2', 'w3', 'b3'):
                setattr(self, key, z[key].copy())
            self.residual_scale = float(z['residual_scale'])
        if self.mean.shape != (FEATURE_DIM,) or self.scale.shape != (FEATURE_DIM,) or (self.scale <= 0).any():
            raise ValueError('Invalid edit-ranking normalization')

    def residual(self, x):
        x = np.clip((np.asarray(x, np.float32)-self.mean)/self.scale, -10, 10)
        x = np.maximum(x@self.w1.T+self.b1, 0)
        x = np.maximum(x@self.w2.T+self.b2, 0)
        return (x@self.w3.T+self.b3)[..., 0]*self.residual_scale

    def predict(self, x, heuristic_score):
        heuristic_score = np.asarray(heuristic_score, np.float64)
        residual = self.residual(x)
        if heuristic_score.shape != residual.shape:
            raise ValueError('Heuristic and feature rows disagree')
        return heuristic_score+residual.astype(np.float64)

    @property
    def parameters(self):
        return sum(getattr(self, name).size for name in ('w1','b1','w2','b2','w3','b3'))


def make_model(features=FEATURE_DIM, *, zero_output=True):
    import torch
    if features != FEATURE_DIM:
        raise ValueError('Unexpected edit-ranking feature dimension')
    model = torch.nn.Sequential(torch.nn.Linear(features, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    if zero_output:
        torch.nn.init.zeros_(model[4].weight)
        torch.nn.init.zeros_(model[4].bias)
    return model


def export_model(path, model, mean, scale, *, residual_scale=BYTE_SCALE):
    mean, scale = np.asarray(mean, np.float32), np.asarray(scale, np.float32)
    if mean.shape != (FEATURE_DIM,) or scale.shape != (FEATURE_DIM,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Invalid edit-ranking normalization')
    path = Path(path)
    np.savez(path, feature_version=FEATURE_VERSION, feature_names=np.asarray(FEATURE_NAMES),
        mean=mean, scale=scale, residual_scale=float(residual_scale),
        **{name: value.detach().cpu().numpy().astype(np.float32) for name, value in (
            ('w1',model[0].weight),('b1',model[0].bias),('w2',model[2].weight),
            ('b2',model[2].bias),('w3',model[4].weight),('b3',model[4].bias))})

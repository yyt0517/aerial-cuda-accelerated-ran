"""Validated causal inputs and two independent implementations of TB scoring."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import time
import weakref

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("count", "layer", "capacity", "dequeue", "sinr_db", "bler", "expected", "interference")
COUNT, LAYER, CAPACITY, DEQUEUE, SINR, BLER, EXPECTED, INTERFERENCE = range(8)


@lru_cache(maxsize=1)
def curves():
    text = (ROOT / "cuMAC/src/cumac.h").read_text()
    result = []
    for m in range(28):
        row = []
        for name in ("snrMcs", "blerMcs"):
            match = re.search(rf"{name}{m}\[\]\s*=\s*\{{([^}}]+)\}}", text)
            if match is None:
                raise ValueError(f"Missing native curve {name}{m}")
            row.append(np.array([float(x) for x in match[1].split(",")], dtype=np.float32).astype(np.float64))
        if len(row[0]) != len(row[1]):
            raise ValueError("Native curve lengths disagree")
        result.append(tuple(row))
    return tuple(result)


def native_bler(db, mcs):
    """Scalar native scan per element; no sorted-table assumption or batch state."""
    db, mcs = np.broadcast_arrays(np.maximum(np.asarray(db), -4.75), mcs)
    output = np.empty(db.shape, dtype=np.float64)
    for index in np.ndindex(db.shape):
        value = db[index]
        xs, ys = curves()[int(mcs[index])]
        if value < xs[-1]:
            output[index] = 1.
        elif value > xs[0]:
            output[index] = 0.
        else:
            output[index] = 0.
            for j in range(len(xs)-1, 0, -1):
                if value <= xs[j-1]:
                    output[index] = ys[j] + (value-xs[j])/(xs[j-1]-xs[j])*(ys[j-1]-ys[j])
                    break
    return output


@dataclass
class Problem:
    h: np.ndarray
    w: np.ndarray
    noise: float
    queue: np.ndarray
    serving: np.ndarray
    mcs: np.ndarray
    caps: np.ndarray
    tbs: np.ndarray  # [UE, PRG count, layer], current MCS already selected
    weights: np.ndarray
    legal: np.ndarray  # [cell, PRG, UE], blank is always legal
    sinr_hint: np.ndarray | None = None  # Existing causal per-PRG observation, never counterfactual labels.
    deadline: np.ndarray | None = None  # Eight causal deadline/service features per UE.

    def __post_init__(self):
        self.h = np.ascontiguousarray(self.h, dtype=np.complex128)
        self.w = np.ascontiguousarray(self.w, dtype=np.complex128)
        if self.h.ndim != 5 or self.h.shape[-2:] != (4, 4):
            raise ValueError("This research kernel supports 4T4R only")
        self.U, self.P, self.C = self.h.shape[:3]
        if not (self.U > 0 and self.P > 0 and 0 < self.C <= 63):
            raise ValueError("Invalid dimensions or unsupported activity-mask width")
        for name, dtype, shape in (
            ("w", np.complex128, (self.U, self.P, 4, 4)),
            ("queue", np.float64, (self.U,)), ("weights", np.float64, (self.U,)),
            ("serving", np.int64, (self.U,)), ("mcs", np.int64, (self.U,)),
            ("caps", np.int64, (self.U, self.P)),
            ("tbs", np.int64, (self.U, self.P+1, 5)),
            ("legal", bool, (self.C, self.P, self.U)),
        ):
            value = np.ascontiguousarray(getattr(self, name), dtype=dtype)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name}")
            setattr(self, name, value)
        if not np.isfinite(self.h).all() or not np.isfinite(self.noise) or self.noise <= 0:
            raise ValueError("Invalid CSI or noise")
        if (self.queue < 0).any() or (self.weights <= 0).any():
            raise ValueError("Invalid demand or weights")
        if ((self.serving < 0) | (self.serving >= self.C)).any() or ((self.mcs < 0) | (self.mcs > 27)).any():
            raise ValueError("Invalid serving cell or MCS")
        if ((self.caps < 1) | (self.caps > 4)).any() or (self.tbs < 0).any():
            raise ValueError("Invalid layers or TBS")
        association = np.arange(self.C)[:, None, None] == self.serving[None, None, :]
        if (self.legal & (~association | (self.queue[None, None, :] <= 0))).any():
            raise ValueError("Legal mask must enforce association and positive demand")
        if self.tbs[:, 0].any() or self.tbs[:, :, 0].any():
            raise ValueError("Zero-resource TBS must be zero")
        if self.sinr_hint is not None:
            self.sinr_hint = np.ascontiguousarray(self.sinr_hint, dtype=np.float64)
            if self.sinr_hint.shape != (self.U, self.P, 4) or not np.isfinite(self.sinr_hint).all() or (self.sinr_hint < 0).any():
                raise ValueError("Invalid causal SINR hint")
        if self.deadline is not None:
            self.deadline = np.ascontiguousarray(self.deadline, dtype=np.float64)
            if self.deadline.shape != (self.U, 8) or not np.isfinite(self.deadline).all():
                raise ValueError("Invalid causal deadline features")

    def actions(self, values):
        values = np.asarray(values)
        if values.ndim == 2:
            values = values[None]
        if values.ndim != 3 or values.shape[1:] != (self.C, self.P):
            raise ValueError("Invalid action dimensions")
        if not np.issubdtype(values.dtype, np.integer) or ((values < -1) | (values >= self.U)).any():
            raise ValueError("Invalid UE index")
        b, c, p = np.nonzero(values >= 0)
        if not self.legal[c, p, values[b, c, p]].all():
            raise ValueError("Illegal assignment")
        return np.ascontiguousarray(values, dtype=np.int64)


class ReferencePhy:
    """Independent NumPy full-UE reference; deliberately not an online fast path."""
    def __init__(self, problem):
        self.problem = problem
        self.covariance = problem.h @ problem.h.conj().swapaxes(-1, -2)
        self.signal = problem.h[np.arange(problem.U), :, problem.serving] @ problem.w

    def evaluate(self, actions):
        p = self.problem
        actions = p.actions(actions)
        result = np.zeros((len(actions), p.U, len(FIELDS)))
        for b, action in enumerate(actions):
            for u in range(p.U):
                selected = np.flatnonzero(action[p.serving[u]] == u)
                if not len(selected):
                    result[b, u, [LAYER, SINR, BLER]] = [-1, -999, -1]
                    continue
                layers = int(p.caps[u, selected].min())
                count = len(selected)
                r = np.tile(p.noise*np.eye(4, dtype=complex), (count, 1, 1))
                interference = 0.
                for c in range(p.C):
                    if c == p.serving[u]:
                        continue
                    active = action[c, selected] >= 0
                    r[active] += self.covariance[u, selected[active], c]
                    interference += np.trace(self.covariance[u, selected[active], c], axis1=-2, axis2=-1).real.sum()
                s = self.signal[u, selected]
                e = np.linalg.inv(np.eye(4) + s.conj().swapaxes(-1, -2) @ np.linalg.solve(r, s))
                gamma = np.maximum(1./np.diagonal(e, axis1=-2, axis2=-1).real-1., 0.)
                db = 10*np.log10(max(gamma[:, :layers].mean(), 1.e-30))
                capacity = p.tbs[u, count, layers]
                dequeue = min(p.queue[u], capacity)
                bler = float(native_bler(db, p.mcs[u]))
                result[b, u] = [count, layers, capacity, dequeue, db, bler, dequeue*(1-bler), interference/count]
        return result


@lru_cache(maxsize=1)
def library():
    path = Path(__file__).with_name("_physics.so")
    if not path.exists():
        raise RuntimeError("Build the kernel first: python -m training.stageb_lns.build")
    lib = ctypes.CDLL(str(path))
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    lib.lns_create.argtypes = [ctypes.c_int]*3 + [ctypes.c_double] + [i64]*4 + [f64]*3 + [i64, f64, f64]
    lib.lns_create.restype = ctypes.c_void_p
    lib.lns_destroy.argtypes = [ctypes.c_void_p]
    lib.lns_destroy.restype = None
    lib.lns_evaluate.argtypes = [ctypes.c_void_p, ctypes.c_int, i64, ctypes.c_void_p, ctypes.c_void_p, f64]
    lib.lns_evaluate.restype = ctypes.c_int
    lib.lns_stats.argtypes = [ctypes.c_void_p, i64]
    lib.lns_stats.restype = None
    return lib


@dataclass(frozen=True)
class Evaluation:
    actions: np.ndarray
    values: np.ndarray
    owner: object


class NativePhy:
    def __init__(self, problem, *, profile=False):
        self.profile = profile
        self.timings = dict(validation_ms=0., native_kernel_ms=0., output_copy_ms=0.)
        self.problem = problem
        self.lib = library()
        self.covariance = np.ascontiguousarray(problem.h @ problem.h.conj().swapaxes(-1, -2))
        self.signal = np.ascontiguousarray(problem.h[np.arange(problem.U), :, problem.serving] @ problem.w)
        self.received = np.trace(self.covariance, axis1=-2, axis2=-1).real
        offsets = np.r_[0, np.cumsum([len(x) for x, _ in curves()])].astype(np.int64)
        xs = np.concatenate([x for x, _ in curves()])
        ys = np.concatenate([y for _, y in curves()])
        self.context = self.lib.lns_create(problem.U, problem.C, problem.P, problem.noise,
            problem.serving, problem.mcs, problem.caps, problem.tbs, problem.queue,
            self.covariance.view(np.float64), self.signal.view(np.float64), offsets, xs, ys)
        if not self.context:
            raise RuntimeError("Could not allocate physical context")
        self._release = weakref.finalize(self, self.lib.lns_destroy, self.context)

    def evaluate(self, actions, base=None):
        profile = getattr(self, 'profile', False)
        stamp = time.perf_counter() if profile else 0.
        if not self._release.alive:
            raise RuntimeError("Physical context is closed")
        actions = self.problem.actions(actions)
        if base is not None and (base.owner is not self or len(base.actions) != 1):
            raise ValueError("Incremental base must be a single result from this PHY context")
        if base is not None and (base.actions.shape != (1, self.problem.C, self.problem.P)
                or base.values.shape != (1, self.problem.U, len(FIELDS))
                or base.actions.dtype != np.int64 or base.values.dtype != np.float64
                or not base.actions.flags.c_contiguous or not base.values.flags.c_contiguous):
            raise ValueError("Invalid incremental base layout")
        values = np.empty((len(actions), self.problem.U, len(FIELDS)), dtype=np.float64)
        if profile:
            now = time.perf_counter()
            self.timings['validation_ms'] += (now-stamp)*1000
            stamp = now
        result = self.lib.lns_evaluate(self.context, len(actions), actions,
            None if base is None else base.actions.ctypes.data,
            None if base is None else base.values.ctypes.data, values)
        if profile:
            now = time.perf_counter()
            self.timings['native_kernel_ms'] += (now-stamp)*1000
            stamp = now
        if result or not np.isfinite(values).all():
            raise RuntimeError("Physical evaluation failed")
        # Own immutable arrays: callers cannot invalidate an incremental base.
        actions = actions.copy()
        actions.flags.writeable = values.flags.writeable = False
        if profile:
            self.timings['output_copy_ms'] += (time.perf_counter()-stamp)*1000
        return Evaluation(actions, values, self)

    def take(self, evaluation, index):
        if evaluation.owner is not self:
            raise ValueError("Foreign evaluation")
        if not 0 <= index < len(evaluation.actions):
            raise IndexError("Evaluation index out of range")
        return Evaluation(evaluation.actions[index:index+1], evaluation.values[index:index+1], self)

    @property
    def stats(self):
        if not self._release.alive:
            raise RuntimeError("Physical context is closed")
        values = np.empty(5, dtype=np.int64)
        self.lib.lns_stats(self.context, values)
        return dict(zip(("candidate_evaluations", "ue_evaluations", "mmse_requests", "mmse_misses"), map(int, values[:4])))

    def close(self):
        self._release()

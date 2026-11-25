"""Neighbor-only, causal spatial reports and a cached scheduler receiver.

R = 4 C / trace(C), C_hat = 4 beta_RSRP R.  R has trace four;
its 4x4 Hermitian representation has 15 independent real parameters.
There is no temporal covariance smoothing: the update-period experiment
isolates sampling/holding and the fixed one-TTI transport delay.
"""
from collections import deque
from dataclasses import replace
import numpy as np
from training.stageb_rsrp.model import approximate_problem

PERIODS = (1, 4, 20, 100, 200)
GRANULARITIES = ('prg', 'wideband')
SPATIAL_STAGES = tuple(f'{g}_t{k}' for g in GRANULARITIES for k in PERIODS)
STAGES = ('exact', 'reported') + SPATIAL_STAGES
UPPER = np.triu_indices(4, 1)


def setting(stage):
    if stage not in SPATIAL_STAGES:
        raise ValueError(stage)
    granularity, period = stage.rsplit('_t', 1)
    return granularity, int(period)


def neighbors(serving, cells):
    return np.array([[c for c in range(cells) if c != s] for s in serving], dtype=np.int64)


def normalized_shapes(h, serving):
    """Measurement emulator only: access the neighbors' actual full-port H."""
    h = np.asarray(h, dtype=np.complex128)
    u, p, c = h.shape[:3]
    nb = neighbors(serving, c)
    hn = h[np.arange(u)[:, None, None], np.arange(p)[None, :, None], nb[:, None, :]]
    covariance = hn @ hn.conj().swapaxes(-1, -2)
    result = {}
    for name, matrix in [('prg', covariance), ('wideband', covariance.mean(axis=1, keepdims=True))]:
        trace = np.trace(matrix, axis1=-2, axis2=-1).real
        safe = np.where(trace > 0, trace, 1.)
        normalized = matrix * (4 / safe)[..., None, None]
        normalized[trace <= 0] = np.eye(4)
        result[name] = normalized
    return result


def pack_shape(shape):
    shape = np.asarray(shape)
    off = shape[..., UPPER[0], UPPER[1]]
    return np.concatenate((np.diagonal(shape, axis1=-2, axis2=-1).real[..., :3], off.real, off.imag), axis=-1)


def unpack_shape(packet):
    packet = np.asarray(packet, dtype=np.float64)
    if packet.shape[-1] != 15 or not np.isfinite(packet).all():
        raise ValueError('Expected 15 finite real parameters per trace-four covariance')
    shape = np.zeros(packet.shape[:-1] + (4, 4), dtype=np.complex128)
    diagonal = np.concatenate((packet[..., :3], 4-packet[..., :3].sum(-1, keepdims=True)), axis=-1)
    shape[..., np.arange(4), np.arange(4)] = diagonal
    off = packet[..., 3:9] + 1j*packet[..., 9:15]
    shape[..., UPPER[0], UPPER[1]] = off
    shape[..., UPPER[1], UPPER[0]] = off.conj()
    return shape


def decode_factor(packet):
    """Scheduler work, performed ONLY when a new spatial report arrives."""
    shape = unpack_shape(packet)
    eigenvalues, vectors = np.linalg.eigh(shape)
    if eigenvalues.min(initial=0) < -1e-10:
        raise ValueError('Spatial report is not positive semidefinite')
    return vectors * np.sqrt(np.maximum(eigenvalues, 0))[..., None, :]


class SpatialReports:
    def __init__(self, serving, stages, delay_tti=1):
        if delay_tti != 1:
            raise ValueError('This frozen comparison uses one-TTI transport delay')
        self.serving = np.asarray(serving, dtype=np.int64).copy()
        self.stages = tuple(stages)
        for stage in self.stages:
            setting(stage)
        self.delay = delay_tti
        self.last_tti = -1
        self.pending = deque()
        self.available = {}

    def observe(self, h, tti):
        if tti != self.last_tti + 1:
            raise ValueError('Covariance samples must be consecutive and causal')
        due = [s for s in self.stages if tti % setting(s)[1] == 0]
        if due:
            shapes = normalized_shapes(h, self.serving)
            packets = {g: pack_shape(shapes[g]) for g in {setting(s)[0] for s in due}}
            for stage in due:
                packet = packets[setting(stage)[0]].copy()
                packet.flags.writeable = False
                self.pending.append((tti+self.delay, stage, tti, packet))
        while self.pending and self.pending[0][0] <= tti:
            _, stage, sampled, packet = self.pending.popleft()
            self.available[stage] = (sampled, packet)
        self.last_tti = tti

    def read(self, stage):
        if stage not in self.available:
            raise RuntimeError('No causal spatial report yet')
        return self.available[stage]

    def snapshot(self):
        return {s: dict(sample_tti=t, packet=p.copy()) for s, (t, p) in self.available.items()}


class SpatialReceiver:
    def __init__(self):
        self.sample_tti = None
        self.factor = None

    def accept(self, sample_tti, packet):
        if self.sample_tti == sample_tti:
            return False
        if self.sample_tti is not None and sample_tti < self.sample_tti:
            raise ValueError('Out-of-order spatial report')
        self.factor = decode_factor(packet)
        self.sample_tti = sample_tti
        return True


def spatial_problem(p, rsrp, factor):
    """No access to instantaneous neighbor H: serving CSI + reports only."""
    factor = np.asarray(factor)
    if factor.shape not in ((p.U, p.P, p.C-1, 4, 4), (p.U, 1, p.C-1, 4, 4)):
        raise ValueError('Spatial factor has the wrong dimensions')
    rsrp = np.asarray(rsrp, dtype=np.float64)
    if rsrp.shape != (p.U, p.C) or not np.isfinite(rsrp).all() or (rsrp < 0).any():
        raise ValueError('Invalid calibrated RSRP power')
    nb = neighbors(p.serving, p.C)
    h = np.zeros_like(p.h)
    ui = np.arange(p.U)
    for n in range(p.C-1):
        scale = np.sqrt(4*rsrp[ui, nb[:, n]])
        h[ui, :, nb[:, n]] = factor[:, :, n] * scale[:, None, None, None]
    h[ui, :, p.serving] = p.h[ui, :, p.serving]
    return replace(p, h=h)


def from_row(p, stage, row, receiver=None):
    if stage == 'exact':
        return p
    if stage == 'reported':
        return approximate_problem(p, row['rsrp_report'])
    report = row['spatial_reports'][stage]
    if receiver is None:
        receiver = SpatialReceiver()
    receiver.accept(int(report['sample_tti']), report['packet'])
    return spatial_problem(p, row['rsrp_report'], receiver.factor)


def diagnostic_times(seed, smoke=False):
    if smoke:
        return (50, 59, 67, 81, 97, 109)
    # Stratified phases, independently dithered to avoid aliasing the 4/20/100-TTI periods.
    rng = np.random.default_rng(20261110 + seed)
    phase = np.floor((np.arange(6)+rng.uniform(.1, .9, 6))*200/6).astype(int)
    return tuple(int(cycle*200 + ph + 1) for cycle, ph in zip((48, 63, 78, 93, 108, 123), phase))


def feedback_rows(users=36, cells=3, prgs=17):
    links = users*(cells-1)
    rows = []
    for stage in STAGES:
        cov_rate = 0.
        if stage == 'exact':
            rsrp_rate = 0.
            cov_rate = links*prgs*16/.0005
            granularity, period_ms = 'instantaneous_full_covariance', .5
        elif stage == 'reported':
            rsrp_rate = links/.01
            granularity, period_ms = 'scalar_only', None
        else:
            granularity, period = setting(stage)
            period_ms = period*.5
            rsrp_rate = links/.01
            cov_rate = links*(prgs if granularity == 'prg' else 1)*15/(period*.0005)
        total = rsrp_rate+cov_rate
        rows.append(dict(stage=stage, granularity=granularity, period_ms=period_ms,
            neighbor_links=links, rsrp_real_parameters_per_s=rsrp_rate,
            covariance_real_parameters_per_s=cov_rate, total_real_parameters_per_s=total,
            payload_mbps_at_16bits=total*16/1e6, payload_mbps_at_32bits=total*32/1e6,
            float64_serialized_mbps=total*64/1e6,
            exact_raw_complex_H_real_parameters_per_s=links*prgs*32/.0005))
    return rows

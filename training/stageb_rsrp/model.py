"""Measurement emulation is separate from the scheduler's scalar-only interface.

The simulator does not export standardized RSRP. This idealized proxy measures
one transmit reference port, averages linear power over PRGs and receive
branches, filters causally, and reports one scalar per UE/cell. Internal H
already includes data transmit power and scaling, so equal reference/data EPRE
is assumed; the proxy is not a dBm or standards-compliant RSRP report.
"""
from collections import deque
from dataclasses import dataclass, replace
import numpy as np


@dataclass(frozen=True)
class RsrpConfig:
    period_tti: int = 20
    delay_tti: int = 1
    tau_tti: float = 200.0

    def __post_init__(self):
        if self.period_tti < 1 or self.delay_tti < 1 or self.tau_tti <= 0:
            raise ValueError("Invalid RSRP reporting configuration")


class RsrpMeasurements:
    def __init__(self, config=RsrpConfig()):
        self.config = config
        self.last_tti = -1
        self.filtered = None
        self.pending = deque()
        self.report = None
        self.report_tti = None
        self.alpha = -np.expm1(-1.0 / config.tau_tti)

    def observe(self, h, tti):
        """Emulator only: consume current physical H, publish delayed scalars."""
        if tti != self.last_tti + 1:
            raise ValueError("Measurements require consecutive causal snapshots")
        h = np.asarray(h)
        if h.ndim != 5 or h.shape[-2:] != (4, 4) or not np.isfinite(h).all():
            raise ValueError("Expected finite [UE, PRG, cell, RX, TX] channel")
        # Orthogonal pilot on TX port 0; average PRGs and receive branches.
        power = np.square(np.abs(h[..., 0])).mean(axis=(1, 3))
        if self.filtered is None:
            self.filtered = power.copy()
        else:
            if self.filtered.shape != power.shape:
                raise ValueError("Measurement dimensions changed")
            self.filtered += self.alpha * (power - self.filtered)
        if tti % self.config.period_tti == 0:
            self.pending.append((tti + self.config.delay_tti, tti, self.filtered.copy()))
        while self.pending and self.pending[0][0] <= tti:
            _, self.report_tti, self.report = self.pending.popleft()
        self.last_tti = tti

    def read(self):
        if self.report is None:
            raise RuntimeError("No causal RSRP report available yet")
        return self.report.copy(), int(self.report_tti)


def scheduler_channels(serving_h, serving, rsrp, prgs):
    """Build scheduler H from serving CSI and scalar reports ONLY.

    Four equal-power transmit ports imply C_hat = 4 * RSRP * I_4.
    sqrt(4 * RSRP) * I is just a factor of this covariance, not an
    estimate of the neighbor's physical complex channel.
    """
    serving_h = np.asarray(serving_h)
    serving = np.asarray(serving)
    rsrp = np.asarray(rsrp, dtype=np.float64)
    users = len(serving)
    if rsrp.ndim != 2 or rsrp.shape[0] != users:
        raise ValueError("Invalid scalar report dimensions")
    if serving_h.shape != (users, prgs, 4, 4):
        raise ValueError("Invalid serving CSI dimensions")
    if not np.isfinite(rsrp).all() or (rsrp < 0).any():
        raise ValueError("Invalid RSRP values")
    h = np.zeros((users, prgs, rsrp.shape[1], 4, 4), np.complex128)
    diagonal = np.arange(4)
    h[..., diagonal, diagonal] = np.sqrt(4.0 * rsrp)[:, None, :, None]
    h[np.arange(users), :, serving] = serving_h
    return h


def approximate_problem(p, rsrp):
    own = p.h[np.arange(p.U), :, p.serving].copy()
    h = scheduler_channels(own, p.serving, rsrp, p.P)
    # No mutation of native snapshot, p.h, serving CSI, or native link adaptation.
    return replace(p, h=h)

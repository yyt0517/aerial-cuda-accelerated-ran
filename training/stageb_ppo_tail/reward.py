"""Charge all unresolved full packets, using only the causal post-step FIFO.

A partial packet costs the same as an untouched packet of the same age.
CRC-damaged packets remain charged until dequeue/expiry, so failed packets
cannot disappear from the waiting objective. There are no UE/class weights.
"""
import numpy as np


def pending_cost(fifo, tti, cfg):
    ages = np.asarray([tti-p[0]+1 for queue in fifo.queues for p in queue], dtype=float)
    assert np.all((ages >= 1) & (ages <= fifo.ttl))
    exposure = np.clip((ages-cfg['age_threshold_tti'])/cfg['age_ramp_tti'], 0., 1.)
    full_bytes = len(ages)*fifo.size
    age_bytes = float(exposure.sum())*fifo.size
    return dict(pending_packets=len(ages), pending_full_bytes=full_bytes,
        pending_age_sum_tti=float(ages.sum()),
        pending_over_threshold=int((ages > cfg['age_threshold_tti']).sum()),
        age_exposure_bytes=age_bytes, whole_wait_cost_bytes=cfg['holding']*full_bytes,
        age_cost_bytes=cfg['holding']*cfg['age_multiplier']*age_bytes)


def reward(record, fifo, phi_before, phi_after, cfg):
    costs = pending_cost(fifo, record['tti'], cfg)
    complete = record['complete_payload_bytes']
    expired = record['expired_payload_bytes']
    base = (complete-expired-costs['whole_wait_cost_bytes']-costs['age_cost_bytes'])/86400.
    shaping = cfg['gamma']*phi_after-phi_before
    return dict(tti=record['tti'], **costs, complete_payload_bytes=complete,
        expired_payload_bytes=expired, base_reward=base, shaping_reward=shaping,
        reward=base+shaping, potential_before=phi_before, potential_after=phi_after)


def reconstruct_pending(pk, horizon, cfg):
    """Independent packet-lifecycle audit, after native FIFO reconstruction.

    MAC completions leave during terminal_tti. Expiry is reported at the
    last eligible TTI and removed before the following TTI. Censored packets
    remain present after the final executed step.
    """
    arrivals=pk.arrival_tti.to_numpy(int)
    ends=pk.terminal_tti.to_numpy(int)+pk.outcome.eq('expired').to_numpy(int)
    ends=np.where(pk.censored.to_numpy(bool),horizon+1,ends)
    sizes=pk.packet_size_bytes.to_numpy(float)
    diff=np.zeros(horizon+2); age_diff=np.zeros(horizon+2)
    np.add.at(diff,arrivals,sizes);np.add.at(diff,np.minimum(ends,horizon+1),-sizes)
    full=np.cumsum(diff)[1:horizon+1]
    # An exposure at age a applies at t=arrival+a-1, provided t<end.
    threshold=cfg['age_threshold_tti'];ramp=cfg['age_ramp_tti']
    for k in range(1,ramp+1):
        starts=arrivals+threshold+k-1
        valid=(starts<ends)&(starts<=horizon)
        np.add.at(age_diff,starts[valid],sizes[valid]/ramp)
        np.add.at(age_diff,np.minimum(ends[valid],horizon+1),-sizes[valid]/ramp)
    return full,np.cumsum(age_diff)[1:horizon+1]

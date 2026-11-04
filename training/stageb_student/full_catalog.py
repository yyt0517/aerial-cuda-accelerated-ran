"""Full-search control with the same deployable observations as the student."""
import time
import numpy as np
from training.stageb_ppo_v2.catalog import Catalog
from training.stageb_lns.archive import problem_from_row
from training.stageb_lns.physics import NativePhy, EXPECTED, DEQUEUE, COUNT, BLER, SINR
from .features import packet_features


class FullCatalog:
    def __init__(self): self.catalog = Catalog(.98)

    def build(self, env):
        started = time.perf_counter(); row = self.catalog.build(env)
        p = problem_from_row(dict(ctbs=env.inputs, obs=env.obs, snapshot=env.snapshot, ema=np.full(36, 2400.)))
        p.weights[:] = 1.; p.deadline = None
        packet, _ = packet_features(env.fifo, env.tti+1, p.queue)
        phy = NativePhy(p)
        try: raw = phy.evaluate(env.raw_pfq()).values[0]
        finally: phy.close()
        row['state'] = np.clip(np.concatenate((packet[:, :8], np.log1p(p.queue[:, None]/3000), p.mcs[:, None]/27,
            raw[:, EXPECTED, None]/3000, raw[:, COUNT, None]/17,
            raw[:, DEQUEUE, None]/np.maximum(p.queue[:, None], 3000), np.maximum(raw[:, BLER, None], 0),
            raw[:, SINR, None]/30, packet[:, 8:]), -1), -10, 10).astype(np.float32)
        # Re-evaluate the small complete catalog to avoid reconstructing clipped
        # feature values. All additional work belongs in the full control's cost.
        phy = NativePhy(p)
        try: v = phy.evaluate(row['actions'][row['mask']]).values
        finally: phy.close()
        n = len(v); row['candidate'][:n, :, 1] = np.clip((v[:, :, EXPECTED]-raw[:, EXPECTED])/3000, -10, 10)
        row['candidate'][:n, :, 3] = np.clip((v[:, :, DEQUEUE]-raw[:, DEQUEUE])/3000, -10, 10)
        row['candidate'][:n, :, 5] = np.clip((v[:, :, COUNT]-raw[:, COUNT])/17, -10, 10)
        row['summary'][:n, 1] = np.clip((v[:, :, EXPECTED].sum(-1)-raw[:, EXPECTED].sum())/3000, -10, 10)
        row['summary'][:n, 6] = (row['actions'][:n] == env.raw_pfq()).all((1, 2))
        row['audit'].update(full_search_calls=1, k=16, wall_s=time.perf_counter()-started)
        return row

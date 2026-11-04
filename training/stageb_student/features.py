"""Causal FIFO features with no search-result inputs."""
import numpy as np
from training.stageb_ppo_v2.state import packet_state

PACKET_DIM = 14
STATE_DIM = 21


def packet_features(fifo, tti, queue):
    base, potential = packet_state(fifo, tti, queue)
    age = np.zeros((len(queue), 6), np.float32)
    for u, items in enumerate(fifo.queues):
        active = [p for p in items if tti-p[0]+1 <= fifo.ttl]
        incoming = int(queue[u])-sum(p[1] for p in active)
        assert incoming >= 0 and incoming % fifo.size == 0
        ages = np.array([tti-p[0]+1 for p in active] + [1]*(incoming//fifo.size))
        if len(ages):
            age[u] = [*(np.mean(ages > threshold) for threshold in (20, 40, 100, 200)),
                      ages.mean()/fifo.ttl, np.quantile(ages, .95)/fifo.ttl]
    return np.concatenate((base, age), -1), potential

"""Causal packet/PHY observations; capacity is information, never a mask."""
import numpy as np


def packet_state(fifo,tti,pre_queue):
    features=np.zeros((36,8),np.float32)
    potential=0.
    for u,queue in enumerate(fifo.queues):
        active=[p for p in queue if tti-p[0]+1<=fifo.ttl]
        incoming=int(pre_queue[u])-sum(p[1] for p in active)
        assert incoming>=0 and incoming%fifo.size==0
        active += [[tti,fifo.size,False] for _ in range(incoming//fifo.size)]
        clean=sum(p[1] for p in active if not p[2])
        damaged=sum(p[1] for p in active if p[2])
        progress=sum(fifo.size-p[1] for p in active if not p[2])
        near=sum(p[1] for p in active if fifo.ttl-(tti-p[0]+1)<=40)
        hol=active[0] if active else [tti,0,False]
        age=(tti-hol[0]+1)/fifo.ttl if active else 0.
        features[u]=[np.log1p(clean/3000),np.log1p(damaged/3000),hol[1]/3000,
            float(hol[2]),age,near/max(float(pre_queue[u]),3000),progress/3000,np.log1p(len(active))]
        potential+=progress/86400.
    return features,potential


def reward(record,phi_before,phi_after,gamma,holding=1/400):
    base=(record['complete_payload_bytes']-record['expired_payload_bytes']-holding*record['post_queue_bytes'])/86400.
    shaping=gamma*phi_after-phi_before
    return dict(tti=record['tti'],base_reward=base,shaping_reward=shaping,reward=base+shaping,
        potential_before=phi_before,potential_after=phi_after)

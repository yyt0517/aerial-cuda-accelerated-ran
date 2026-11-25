"""Causal per-UE fixed-payload FIFO; preserves packet CRC and expiration accounting."""
from collections import deque
import numpy as np
class HeterogeneousPacketReward:
    def __init__(self,sizes,ttl=400):
        self.sizes=np.asarray(sizes,dtype=np.int64)
        assert self.sizes.shape==(36,) and (self.sizes>0).all()
        self.ttl=ttl;self.queues=[deque() for _ in self.sizes];self.expired=0
    def before(self,tti,pre_queue):
        self.expired=0
        for u,q in enumerate(self.queues):
            size=int(self.sizes[u])
            while q and tti-q[0][0]+1>self.ttl:q.popleft();self.expired+=size
            arrived=int(pre_queue[u])-sum(p[1] for p in q)
            assert arrived>=0 and arrived%size==0,(tti,u,arrived,size)
            q.extend([tti,size,False] for _ in range(arrived//size))
    def after(self,served,good):
        complete=failed_complete=0
        for u,q in enumerate(self.queues):
            left=int(served[u]);assert good[u]==0 or good[u]==left
            failed=good[u]<left
            while left:
                take=min(left,q[0][1]);q[0][1]-=take;q[0][2]|=failed;left-=take
                if q[0][1]==0:
                    _,_,damaged=q.popleft()
                    if damaged:failed_complete+=int(self.sizes[u])
                    else:complete+=int(self.sizes[u])
        remaining=sum(p[1] for q in self.queues for p in q)
        return dict(reward=(complete-self.expired-.01*remaining)/86400.,complete_payload_bytes=complete,failed_complete_bytes=failed_complete,expired_payload_bytes=self.expired,post_queue_bytes=remaining,goodput_bytes=float(np.sum(good)))

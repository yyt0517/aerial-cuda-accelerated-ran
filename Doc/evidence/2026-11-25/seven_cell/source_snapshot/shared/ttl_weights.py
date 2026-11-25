"""One frozen causal HOL urgency family; no learned parameters or future labels."""
import numpy as np

COEFFICIENTS=(0.,.25,.5,1.,2.)

def hol_urgency(fifo,tti,pre_queue):
    """Read without calling before(): it would reset pending expiry rewards.

    Discard would-expire heads in a read-only view; a positive queue with no
    surviving previous packet consists of newly arrived packets with full TTL.
    """
    out=np.zeros(len(pre_queue));slack=np.full(len(pre_queue),fifo.ttl,dtype=np.int64)
    for u,q in enumerate(fifo.queues):
        if pre_queue[u]<=0:continue
        for born,remaining,damaged in q:
            if tti-born+1<=fifo.ttl:
                slack[u]=fifo.ttl-tti+born;break
        out[u]=np.clip(1.-slack[u]/fifo.ttl,0.,1.)
    return out,slack

def objective_weights(base,urgency,coefficient):
    if coefficient<0 or not np.isfinite(coefficient):raise ValueError('Invalid urgency coefficient')
    if coefficient==0:return base.copy() # exact fixed-weight control, no second normalization
    weights=np.asarray(base)*(1.+coefficient*np.asarray(urgency))
    weights/=weights.mean()
    if not np.isfinite(weights).all() or np.any(weights<=0):raise ValueError('Invalid weights')
    return np.ascontiguousarray(weights,np.float64)

"""Causal deadline/service inputs and explicitly labelled risk proxies.

HOL-based urgency is not an exact count of bytes expiring soon. Actual TTL
losses are obtained from completed native packet trajectories during training.
"""
import numpy as np
from .physics import EXPECTED
from .neural_repair import JointRepair, features


def causal_features(obs, ema):
    x=np.asarray(obs['obs_ue_features'],dtype=float)
    if x.ndim != 2 or x.shape[1] < 15 or not np.isfinite(x).all():
        raise ValueError('Deadline training requires dimensionless native UE features')
    active=x[:,0]>0
    result=np.stack((np.clip(x[:,8]/200.,0,1),np.where(x[:,9]>=0,np.clip(x[:,9]/200.,0,1),1.),
        np.clip(x[:,12]/8.,0,1),np.clip(x[:,13],0,1),np.clip(x[:,14],0,1),
        np.clip(1-np.asarray(ema)/2400.,0,1),np.clip(x[:,11],0,1),np.clip(np.asarray(ema)/4800.,0,1)),axis=-1)
    result[~active]=0
    return result


def urgency(problem):
    if problem.deadline is None:
        raise ValueError('Missing deadline state')
    return np.clip(1-problem.deadline[:,1]/.3,0,1)*(problem.queue>0)


def components(problem, values):
    """Physical expected bytes plus deadline and weak-service risk surrogates."""
    expected=np.asarray(values)[...,EXPECTED]
    if expected.ndim==1:expected=expected[None]
    urgent=urgency(problem);weak=problem.deadline[:,5]
    target=np.minimum(problem.queue,2400.)
    physical=expected@problem.weights
    ttl_shortfall=(np.maximum(target[None]-expected,0)*urgent).sum(-1)
    urgent_success=expected@urgent
    weak_success=(np.minimum(expected,target[None])*weak).sum(-1)
    shaped=physical+.5*urgent_success-.5*ttl_shortfall+.5*weak_success
    return physical,shaped,ttl_shortfall,weak_success


def packet_objective(problem):
    return lambda values:components(problem,values)[1]


def deadline_features(graph,base,block,partials,depth,choices,context=None):
    x=features(graph,base,block,partials,depth,choices,context)
    extra=graph.problem.deadline[np.maximum(choices,0)].copy()
    extra[choices<0]=0
    return np.concatenate((x,np.broadcast_to(extra,x.shape[:2]+(8,))),axis=-1).astype(np.float32)


class DeadlineRepair(JointRepair):
    version=2

    def features(self,graph,base,block,beam,depth,choices,context):
        return deadline_features(graph,base,block,beam,depth,choices,context)

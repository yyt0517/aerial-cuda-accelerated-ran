"""Causal per-UE service-set proposals, with training-only terminal labels."""
from pathlib import Path
import time
import numpy as np

from .physics import COUNT,LAYER,DEQUEUE,EXPECTED,BLER,SINR
from .search import ProposalGraph


FEATURE_NAMES=(
    'log_queue','log_weight','weight','mcs','pfq_count','pfq_layer','pfq_dequeue','pfq_expected',
    'pfq_reliability','pfq_sinr','mean_cap','min_cap','max_cap','cap1_fraction',
    'quality_mean','quality_q25','quality_q75','quality_max','signal_mean','interference_mean',
    'one_prg_tbs','four_prg_tbs','all_prg_tbs','legal_fraction','pfq_utilization',
    'utility','cell_utility_rank','cell_quality_rank','cell_expected_rank','cell_weight_rank',
    'cell_eligible','cell_queue_mean','cell_weight_mean','cell_quality_mean','cell_expected_mean',
    'cell_reliability_mean','root_activity','root_j','root_dequeue','eligible')


def causal_features(p,phy,base):
    """Initial verified PFQ only; no terminal actions/labels or future fields."""
    if base.owner is not phy:raise ValueError('Foreign initial PHY')
    v=base.values[0];g=ProposalGraph(phy);quality=np.log1p(g.quality)
    signal=phy.received[np.arange(p.U),:,p.serving]
    other=np.maximum(phy.received.sum(2)-signal,0)
    eligible=p.legal.any((0,1));reli=np.where(v[:,COUNT]>0,1-np.maximum(v[:,BLER],0),.5)
    nominal=np.minimum(p.queue,p.tbs[np.arange(p.U),min(4,p.P),np.maximum(p.caps.min(1),1)])
    utility=p.weights*nominal*np.maximum(reli,.05)*(1+quality.mean(1))
    columns=[np.log1p(p.queue)/10,np.log1p(p.weights)/3,p.weights/4,p.mcs/27,
        v[:,COUNT]/p.P,np.maximum(v[:,LAYER],0)/4,v[:,DEQUEUE]/3000,v[:,EXPECTED]/3000,
        reli,np.clip(v[:,SINR],-40,40)/20,p.caps.mean(1)/4,p.caps.min(1)/4,p.caps.max(1)/4,
        (p.caps==1).mean(1),quality.mean(1)/4,np.quantile(quality,.25,axis=1)/4,
        np.quantile(quality,.75,axis=1)/4,quality.max(1)/4,
        np.log1p(signal/(4*p.noise)).mean(1)/10,np.log1p(other/(4*p.noise)).mean(1)/10,
        p.tbs[:,1,1]/3000,p.tbs[np.arange(p.U),min(4,p.P),np.maximum(p.caps.min(1),1)]/10000,
        p.tbs[:,p.P,4]/30000,p.legal.sum((0,1))/p.P,
        v[:,DEQUEUE]/np.maximum(p.queue,1),np.log1p(utility)/10]
    rank=np.zeros((p.U,4));context=np.zeros((p.U,6))
    for cell in range(p.C):
        us=np.flatnonzero(eligible&(p.serving==cell))
        if not len(us):continue
        for j,score in enumerate((utility,quality.mean(1),v[:,EXPECTED],p.weights)):
            order=us[np.argsort(-score[us],kind='stable')]
            rank[order,j]=np.arange(len(us))/max(len(us)-1,1)
        context[us]=[len(us)/p.P,np.log1p(p.queue[us]).mean()/10,p.weights[us].mean()/4,
            quality[us].mean()/4,v[us,EXPECTED].mean()/3000,reli[us].mean()]
    columns.extend(rank.T);columns.extend(context.T)
    columns.extend([np.full(p.U,(base.actions[0]>=0).mean()),
        np.full(p.U,float(v[:,EXPECTED]@p.weights)/100000),np.full(p.U,v[:,DEQUEUE].sum()/30000),eligible.astype(float)])
    x=np.column_stack(columns).astype(np.float32)
    assert x.shape==(p.U,len(FEATURE_NAMES)) and np.isfinite(x).all()
    return x,utility,eligible


def validate_masks(p,allocated,effective,eligible):
    a=np.asarray(allocated,bool);e=np.asarray(effective,bool)
    if a.shape!=(p.U,) or e.shape!=(p.U,) or (e&~a).any() or (a&~eligible).any():
        raise ValueError('Invalid causal service sets')
    return np.stack([a,e],1)


def decode_probabilities(p,prob,utility,eligible,thresholds):
    ta,te=thresholds;e=(prob[:,1]>=te)&eligible;a=((prob[:,0]>=ta)&eligible)|e
    for cell in range(p.C):
        us=np.flatnonzero(eligible&(p.serving==cell))
        if len(us) and not e[us].any():
            selected=int(us[np.argmax(prob[us,1]+1.e-10*utility[us]/max(float(utility.max()),1))]);e[selected]=a[selected]=True
    return validate_masks(p,a,e,eligible)


def heuristic_masks(p,utility,eligible,counts):
    a=np.zeros(p.U,bool);e=a.copy()
    for cell in range(p.C):
        us=np.flatnonzero(eligible&(p.serving==cell));order=us[np.argsort(-utility[us],kind='stable')]
        ka,ke=(counts[cell] if np.asarray(counts).ndim==2 else counts)
        a[order[:int(ka)]]=True;e[order[:min(int(ka),int(ke))]]=True
    return validate_masks(p,a,e,eligible)


class SetModel:
    def __init__(self,path):
        self.path=Path(path)
        with np.load(path) as z:
            for name in ('mean','scale','w1','b1','w2','b2','w3','b3'):setattr(self,name,z[name])

    def logits(self,x):
        z=np.clip((x-self.mean)/self.scale,-10,10)
        z=np.maximum(z@self.w1.T+self.b1,0);z=np.maximum(z@self.w2.T+self.b2,0)
        return z@self.w3.T+self.b3

    def predict(self,x):return 1/(1+np.exp(-np.clip(self.logits(x),-30,30)))


def propose(p,phy,base,source,model,thresholds,heuristic_counts):
    if source=='none':return np.zeros((p.U,2),bool),dict(proposal_ms=0.)
    start=time.perf_counter();x,utility,eligible=causal_features(p,phy,base)
    if source=='heuristic':masks=heuristic_masks(p,utility,eligible,heuristic_counts)
    elif source in ('learned','matched'):
        prob=model.predict(x);masks=decode_probabilities(p,prob,utility,eligible,thresholds)
        if source=='matched':
            counts=np.array([masks[p.serving==c].sum(0) for c in range(p.C)])
            masks=heuristic_masks(p,utility,eligible,counts)
    else:raise ValueError('Online proposal source must be causal')
    return masks,dict(proposal_ms=(time.perf_counter()-start)*1000)


def set_metrics(masks,target,eligible):
    out={}
    for j,name in enumerate(('allocation','effective')):
        prediction=masks[:,j];truth=target[:,j];tp=int((prediction&truth).sum())
        out[f'{name}_recall']=tp/max(int(truth.sum()),1) if truth.any() else 1.
        out[f'{name}_precision']=tp/max(int(prediction.sum()),1) if prediction.any() else float(not truth.any())
        out[f'{name}_missing']=int((truth&~prediction).sum());out[f'{name}_size']=int(prediction.sum())
    out['size_ratio']=float(masks[:,0].sum()/max(eligible.sum(),1))
    return out


def calibration_key(records):
    gate=all(r['allocation_recall']>=.98 and r['effective_recall']>=.98 for r in records)
    size=float(np.mean([r['size_ratio'] for r in records]));compressed=size<=.75
    loss=float(np.mean([2*(1-r['allocation_recall'])+(1-r['effective_recall'])+
        2*max(r['size_ratio']-.75,0)+.25*(1-r['allocation_precision']) for r in records]))
    return (int(gate and compressed),-loss,-size)

"""Causal candidate features; tiny service-risk model; bounded terminal choice.

The runtime API never receives reference service, quality or membership. An
attainable envelope is an own-paid-archive statistic, not a simultaneous action.
"""
import numpy as np
from .physics import EXPECTED,DEQUEUE

RISK_CAP=.02
PROBABILITY_THRESHOLD=.9


def candidate_features(p,condition,values):
    e=values[:,:,EXPECTED];d=np.minimum(p.queue,p.tbs[np.arange(p.U),condition.counts,condition.layers])
    quota=(condition.counts>0)&(d>0);den=np.maximum(d,1.e-12)
    r=np.clip(e/den,0,1);j=np.array([row@p.weights for row in e]);best=int(np.argmax(j))
    envelope=e.max(0);group=quota&(envelope>1.);features=[];names=[]
    def add(name,x):
        names.append(name);features.append(np.broadcast_to(x,(len(e),)))
    add('j_fraction',j/max(j[best],1))
    for label,g in [('quota',quota),('attainable',group)]:
        rr=r[:,g] if g.any() else np.ones((len(e),1))
        for q in (0,.05,.2,.5):add(f'{label}_reliability_q{q}',np.quantile(rr,q,axis=1))
        add(label+'_reliability_mean',rr.mean(1));add(label+'_exp',-.1*np.log(np.exp(-rr/.1).mean(1)))
        add(label+'_zero_fraction',(rr<1.e-8).mean(1));add(label+'_ue_fraction',g.mean())
    references=[('max',envelope),('q90',np.quantile(e,.9,axis=0)),('mean',e.mean(0)),('bestj',e[best])]
    ratios=[]
    for label,ref in references:
        ids=np.flatnonzero(group)
        if not len(ids):p5=weak=np.ones(len(e));rel=np.ones((len(e),1))
        else:
            weakids=ids[np.argsort(ref[ids],kind='stable')[:max(1,int(np.ceil(.2*len(ids))))]]
            p5=np.quantile(e[:,ids],.05,axis=1)/max(float(np.quantile(ref[ids],.05)),1.)
            weak=e[:,weakids].sum(1)/max(float(ref[weakids].sum()),1.)
            rel=e[:,ids]/np.maximum(ref[ids],1.)
        ratios.append((p5,weak))
        add(label+'_p5_ratio',np.clip(p5,0,3));add(label+'_fixedweak_ratio',np.clip(weak,0,3))
        for q in (0,.05,.2):add(f'{label}_relative_q{q}',np.clip(np.quantile(rel,q,axis=1),0,3))
    add('log_total_quota',np.log1p(d.sum())/10);add('quota_fraction',quota.mean())
    add('max_attainability_mean',np.mean(envelope[quota]/den[quota]) if quota.any() else 1.)
    add('max_attainability_p5',np.quantile(envelope[quota]/den[quota],.05) if quota.any() else 1.)
    add('weight_cv',np.std(p.weights)/max(float(np.mean(p.weights)),1.e-12))
    x=np.stack(features,axis=1)
    assert np.isfinite(x).all()
    return x,j,np.minimum(*ratios[0]),names


def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-40,40)))


class RiskModel:
    def __init__(self,arrays):self.arrays=arrays

    def logits(self,x):
        a=self.arrays;z=np.clip((x-a['mean'])/a['scale'],-8,8)
        for i in range(2):z=np.tanh(z@a[f'w{i}']+a[f'b{i}'])
        return (z@a['w2']+a['b2']).ravel()

    def predict(self,x):
        a=self.arrays;return sigmoid(a['calibration'][0]*self.logits(x)+a['calibration'][1])

    @classmethod
    def load(cls,path):
        with np.load(path) as data:return cls({k:data[k].copy() for k in data.files})


def select_index(j,score,threshold,cap=RISK_CAP):
    allowed=np.flatnonzero(j+1.e-8>=(1-cap)*j.max())
    safe=allowed[score[allowed]>=threshold]
    if len(safe):return int(safe[np.argmax(j[safe])]),True
    # No declared-safe candidate: best risk, then quality, within frozen cap.
    return int(max(allowed,key=lambda i:(score[i],j[i]))),False


def select_terminal(p,condition,archive,keys,objectives,mode,model):
    if mode=='maxj':return max(keys,key=lambda k:objectives[k][0]),{}
    values=np.stack([archive[k].values[0] for k in keys]);j=np.array([objectives[k][0] for k in keys])
    allowed=np.flatnonzero(j+1.e-8>=(1-RISK_CAP)*j.max())
    if mode=='continuous':
        i=int(max(allowed,key=lambda i:(objectives[keys[i]][1],objectives[keys[i]][2],j[i])))
        return keys[i],dict(terminal_candidates=len(keys),terminal_allowed=len(allowed),terminal_safe_found=False)
    x,j2,heuristic,names=candidate_features(p,condition,values)
    np.testing.assert_array_equal(j,j2)
    if mode=='attainable':scores=heuristic;threshold=.98
    elif mode=='calibrated':
        if model is None:raise ValueError('Frozen risk model required')
        scores=model.predict(x);threshold=PROBABILITY_THRESHOLD
    else:raise ValueError('Unknown terminal selector')
    i,safe=select_index(j,scores,threshold)
    return keys[i],dict(terminal_candidates=len(keys),terminal_allowed=len(allowed),terminal_safe_found=safe,
        terminal_risk_score=float(scores[i]),terminal_maxj_risk_score=float(scores[np.argmax(j)]))

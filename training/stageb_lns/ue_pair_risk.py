"""Shared per-UE monotone service-loss ranking and conservative terminal gate.

Runtime inputs are causal problem/conditions and own exact paid archive only.
No UE index embedding or teacher membership is available at inference.
"""
import time
import numpy as np
from .physics import EXPECTED
from .terminal_risk import RISK_CAP,sigmoid,candidate_features

CONFIDENCE=.9
CONTEXT_NAMES=('count','layer','mcs','log_quota','queue_fraction','relative_weight','weighted_quota_share',
    'max_r','q90_r','mean_r','median_r','std_r','positive_frequency','reliable_frequency','bestj_r',
    'cell_activity_fraction','legal_fraction','quota_share','min_r','quota_group_fraction')


def ue_features(p,condition,values):
    e=values[:,:,EXPECTED];d=np.minimum(p.queue,p.tbs[np.arange(p.U),condition.counts,condition.layers])
    group=(condition.counts>0)&(d>0);r=np.clip(e/np.maximum(d,1.e-12),0,1)
    j=np.array([v@p.weights for v in e]);best=int(np.argmax(j));maximum=r.max(0);q90=np.quantile(r,.9,axis=0)
    mean=r.mean(0);positive=(e>1).mean(0);wj=p.weights*d
    context=np.stack([condition.counts/p.P,condition.layers/4,p.mcs/27,np.log1p(d)/10,
        d/np.maximum(p.queue,1),p.weights/max(float(p.weights.mean()),1.e-12),wj/max(float(wj.sum()),1),
        maximum,q90,mean,np.median(r,axis=0),r.std(0),positive,(r>=.98).mean(0),r[best],
        condition.activity.mean(1)[p.serving],p.legal[p.serving,:,np.arange(p.U)].mean(1),
        d/max(float(d.sum()),1),r.min(0),np.full(p.U,group.mean())],axis=1)
    losses=np.stack([1-r,(1-r)**2,np.maximum(maximum-r,0),np.maximum(q90-r,0),
        np.maximum(mean-r,0),np.maximum(r[best]-r,0),sigmoid((.01-r)/.02)],axis=-1)
    assert np.isfinite(context).all() and np.isfinite(losses).all()
    return dict(context=context,losses=losses,r=r,group=group,j=j,best=best,heuristic_target=q90,heuristic_effective=positive)


def service_scores(f,parameters):
    # All loss bases decrease with candidate service for a fixed archive.
    weights=2*sigmoid(parameters[:,:8]);effective=sigmoid(parameters[:,8]);target=sigmoid(parameters[:,9])
    loss=np.concatenate([f['losses'],np.maximum(target[None]-f['r'],0)[...,None]],axis=-1)
    contribution=(loss*weights[None]).mean(2)*(.1+.9*effective)[None]*f['group'][None]
    return -contribution.sum(1)/max(int(f['group'].sum()),1)


def heuristic_scores(f):
    loss=np.concatenate([f['losses'],np.maximum(f['heuristic_target'][None]-f['r'],0)[...,None]],axis=-1)
    return -(loss.mean(2)*(.1+.9*f['heuristic_effective'])[None]*f['group'][None]).sum(1)/max(int(f['group'].sum()),1)


class UEPairModel:
    def __init__(self,arrays):self.arrays=arrays

    def scores(self,f):
        a=self.arrays;out=[]
        for i in range(3):
            z=np.clip((f['context']-a[f'mean{i}'])/a[f'scale{i}'],-8,8)
            z=np.tanh(z@a[f'w{i}_0']+a[f'b{i}_0'])
            out.append(service_scores(f,z@a[f'w{i}_1']+a[f'b{i}_1']))
        return np.stack(out)

    @classmethod
    def load(cls,path):
        with np.load(path) as z:return cls({k:z[k].copy() for k in z.files})


def rank_select(j,scores,alpha,guard=True):
    scores=np.atleast_2d(scores);base=int(np.argmax(j));allowed=np.flatnonzero(j+1.e-8>=(1-RISK_CAP)*j[base])
    average=scores.mean(0);proposed=int(max(allowed,key=lambda i:(average[i],j[i])))
    delta=scores[:,proposed]-scores[:,base];confidence=float(sigmoid(alpha*delta).min())
    if proposed==base:chosen=base;reason='maxj_already_best'
    elif guard and (confidence<CONFIDENCE or np.any(delta<=0)):chosen=base;reason='low_confidence'
    else:chosen=proposed;reason='accepted' if guard else 'unguarded'
    return chosen,dict(proposed_index=proposed,selected_index=chosen,reference_maxj_index=base,
        pair_confidence=confidence,pair_min_gain=float(delta.min()),pair_mean_gain=float(delta.mean()),
        fallback_to_maxj=bool(chosen==base and proposed!=base),terminal_reason=reason,
        proposed_j_loss=float(1-j[proposed]/max(j[base],1)),terminal_allowed=len(allowed))


def select_terminal(p,condition,archive,keys,objectives,mode,model):
    if mode=='maxj':return max(keys,key=lambda k:objectives[k][0]),{}
    start=time.perf_counter();values=np.stack([archive[k].values[0] for k in keys])
    if mode=='aggregate_guard':
        x,j,_,_=candidate_features(p,condition,values);feature_ms=(time.perf_counter()-start)*1000
        stamp=time.perf_counter();prob=model['aggregate'].predict(x);network_ms=(time.perf_counter()-stamp)*1000
        stamp=time.perf_counter();base=int(np.argmax(j));allowed=np.flatnonzero(j+1.e-8>=(1-RISK_CAP)*j[base]);safe=allowed[prob[allowed]>=CONFIDENCE]
        chosen=int(safe[np.argmax(j[safe])]) if len(safe) else base
        stats=dict(selected_index=chosen,reference_maxj_index=base,pair_confidence=float(prob[chosen]),
            fallback_to_maxj=not bool(len(safe)),terminal_reason='accepted' if len(safe) else 'low_absolute_confidence',
            terminal_allowed=len(allowed))
    else:
        f=ue_features(p,condition,values);feature_ms=(time.perf_counter()-start)*1000
        stamp=time.perf_counter();learned=model['pair']
        if mode=='ue_heuristic_guard':scores=heuristic_scores(f)[None];alpha=float(learned.arrays['heuristic_alpha'])
        elif mode in ('ue_pair','ue_pair_guard'):scores=learned.scores(f);alpha=float(learned.arrays['alpha'])
        else:raise ValueError('Unknown per-UE selector')
        network_ms=(time.perf_counter()-stamp)*1000;stamp=time.perf_counter()
        chosen,stats=rank_select(f['j'],scores,alpha,guard=mode!='ue_pair')
    stats.update(terminal_feature_ms=feature_ms,terminal_network_ms=network_ms,
        terminal_selection_ms=(time.perf_counter()-stamp)*1000,terminal_candidates=len(keys))
    return keys[chosen],stats

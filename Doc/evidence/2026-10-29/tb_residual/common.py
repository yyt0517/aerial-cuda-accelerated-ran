"""Fast common-depth metrics and sparse residual score reconstruction."""
from pathlib import Path
import numpy as np
import pandas as pd
from training.stageb_lns.residual import terms
from training.stageb_lns.prefix_repair import shortlist

HERE=Path(__file__).resolve().parent

def load():
    with np.load(HERE/'data/candidates.npz') as z:a={k:z[k] for k in z.files}
    with np.load(HERE/'data/pairs.npz') as z:p={k:z[k] for k in z.files}
    return a,p,pd.read_csv(HERE/'data/groups.csv')

def targets(a,groups):
    target=np.zeros(len(a['delta']),bool)
    for g in groups.itertuples():
        ids=np.arange(g.start,g.end);good=a['eligible'][ids]&a['unique'][ids]
        if good.any():target[ids]=good&(a['delta'][ids]>=np.quantile(a['delta'][ids][good],.9))
    return target

def metrics(a,groups,scores,budgets,method,target=None):
    if target is None:target=targets(a,groups)
    rows=[]
    for g in groups.itertuples():
        ids=np.arange(g.start,g.end);ids=ids[a['unique'][ids]]
        high=target[ids];n=len(ids)
        order=np.argsort(-scores[ids],kind='stable')
        oracle=float(np.max(np.r_[0.,a['delta'][ids][a['eligible'][ids]]]))
        for budget in budgets:
            k=max(1,int(np.ceil(budget*n-1.e-9)));chosen=ids[order[:k]]
            gain=float(np.max(np.r_[0.,a['delta'][chosen][a['eligible'][chosen]]]))
            rows.append(dict(group=g.group,block=g.block,split=g.split,seed=g.seed,tti=g.tti,origin=g.origin,depth=g.depth,
                method=method,budget=budget,n=n,k=k,actual_fraction=k/n,high_yield_count=int(high.sum()),
                recall=float(target[chosen].sum()/high.sum()) if high.any() else np.nan,
                any_high_yield=float(target[chosen].any()) if high.any() else np.nan,
                gain=gain,oracle_gain=oracle,regret=oracle-gain,
                retained_gain=gain/oracle if oracle>0 else np.nan))
    return pd.DataFrame(rows)

def residual_scores(a,p,predictions,mode):
    raw=p['raw'];approx=raw[:,1+mode];normalizer=np.maximum(raw[:,0],2400.)
    out=[]
    for pred in predictions:
        e=np.clip(approx+pred*normalizer,0,raw[:,0])
        score=a['draft_scores'][:,mode].copy()
        np.add.at(score,p['candidate'],terms(e,raw)-terms(approx,raw))
        out.append(score)
    out=np.asarray(out)
    return out.mean(0),out.std(0)

def actor_joint(a,groups,logits):
    result=np.zeros(len(a['delta']))
    for g in groups.itertuples():
        ids=slice(g.start,g.end);x=logits[ids].reshape(g.parents,g.options).copy()
        x-=x.max(-1,keepdims=True);x-=np.log(np.exp(x).sum(-1,keepdims=True))
        parent=a['parent'][ids];previous=np.zeros(len(parent))
        valid=parent>=0;previous[valid]=result[parent[valid]]
        result[ids]=x.ravel()+previous
    return result

def actor_coverage(a,groups,joint,method,target=None):
    if target is None:target=targets(a,groups)
    result=[]
    for g in groups.itertuples():
        ids=np.arange(g.start,g.end);choices=np.asarray(__import__('json').loads(g.choices))
        anchor=np.flatnonzero(a['actor_x'][g.start:g.start+g.options,74]>.5)
        assert len(anchor)==1
        selected=shortlist(joint[ids].reshape(g.parents,g.options),choices,int(choices[anchor[0]]),16,48,2)
        chosen=ids[selected];high=target[ids]
        outside=high&~a['old_short'][ids]
        mask=np.zeros(len(ids),bool);mask[selected]=True
        result.append(dict(group=g.group,split=g.split,seed=g.seed,tti=g.tti,origin=g.origin,depth=g.depth,
            method=method,n=len(ids),k=len(chosen),high_yield_count=int(high.sum()),outside_high_yield=int(outside.sum()),
            recall=float(target[chosen].sum()/high.sum()) if high.any() else np.nan,
            outside_recall=float((outside&mask).sum()/outside.sum()) if outside.any() else np.nan,
            any_high_yield=float(target[chosen].any()) if high.any() else np.nan))
    return pd.DataFrame(result)

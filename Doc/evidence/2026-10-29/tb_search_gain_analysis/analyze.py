"""Reproduce the loss/continuation analysis into a new output directory."""
from pathlib import Path
import argparse,sys,json
import numpy as np
import pandas as pd

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];source=root/'deliverables/tb_residual_20261029_v1'
    sys.path.insert(0,str(source));import common
    from training.stageb_lns.prefix_repair import shortlist
    a,_,g=common.load()
    with np.load(source/'evaluation/scores.npz') as z:score=z['residual'];new_joint=z['new_joint']
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False);rows=[]
    for row in g[g.split=='evaluation'].itertuples():
        ids=np.arange(row.start,row.end);choices=np.asarray(json.loads(row.choices))
        anchor=int(choices[np.flatnonzero(a['actor_x'][row.start:row.start+row.options,74]>.5)[0]])
        best=float(np.max(np.r_[0.,a['delta'][ids][a['eligible'][ids]]]))
        for name,joint in [('original',a['old_logp']),('new',new_joint)]:
            keep=shortlist(joint[ids].reshape(row.parents,row.options),choices,anchor,16,48,2);short=ids[keep]
            short_best=float(np.max(np.r_[0.,a['delta'][short][a['eligible'][short]]]))
            k=int(np.ceil(.1*len(ids)-1.e-9));selected=short[np.argsort(-score[short],kind='stable')[:k]]
            chosen=float(np.max(np.r_[0.,a['delta'][selected][a['eligible'][selected]]]))
            rows.append(dict(seed=row.seed,tti=row.tti,origin=row.origin,group=row.group,depth=row.depth,actor=name,
                oracle_gain=best,coverage_loss=best-short_best,ranking_loss=short_best-chosen,total_loss=best-chosen,
                any_positive=best>1,shortlist_misses_best=short_best<best-1,selection_misses_shortlist_best=chosen<short_best-1))
    d=pd.DataFrame(rows);d.to_csv(out/'loss_decomposition.csv',index=False)
    np.testing.assert_allclose(d.total_loss,d.coverage_loss+d.ranking_loss,atol=1.e-9,rtol=0)
    parent=a['parent'];own=np.where(a['eligible'],a['delta'],0.);backed=own.copy();expanded=np.zeros(len(parent),bool)
    valid=parent>=0;expanded[parent[valid]]=True
    for depth in sorted(np.unique(a['depth']),reverse=True):
        ids=np.flatnonzero((a['depth']==depth)&valid);np.maximum.at(backed,parent[ids],backed[ids])
    flips=[];guard_rows=[]
    for row in g[g.split=='evaluation'].itertuples():
        ids=np.arange(row.start,row.end);ids=ids[expanded[ids]]
        if len(ids)<2:continue
        current=ids[np.argmax(a['delta'][ids])];future=ids[np.argmax(backed[ids])]
        flips.append(dict(seed=row.seed,group=row.group,depth=row.depth,n=len(ids),best_known_future=float(backed[future]),
            current_leader_future=float(backed[current]),opportunity_gap=float(backed[future]-backed[current]),
            ranking_flip=bool(backed[future]>backed[current]+1)))
        allowed=ids[a['eligible'][ids]];strict=False;strict_gap=np.nan
        if len(allowed)>=2:
            now=allowed[np.argmax(a['delta'][allowed])];later=allowed[np.argmax(backed[allowed])]
            strict_gap=backed[later]-backed[now];strict=bool(strict_gap>1)
        guard_rows.append(dict(group=row.group,seed=row.seed,leader_currently_eligible=bool(a['eligible'][current]),
            flip=bool(backed[future]>backed[current]+1),strict_available=len(allowed)>=2,strict_flip=strict,strict_gap=strict_gap))
    f=pd.DataFrame(flips);f.to_csv(out/'observed_continuation_rank_flips.csv',index=False)
    pd.DataFrame(guard_rows).to_csv(out/'continuation_guard_strata.csv',index=False)
    print(d.groupby(['seed','actor'])[['coverage_loss','ranking_loss','total_loss']].mean().to_string())
    print('observed future-rank flips',int(f.ranking_flip.sum()),'/',len(f))

if __name__=='__main__':main()

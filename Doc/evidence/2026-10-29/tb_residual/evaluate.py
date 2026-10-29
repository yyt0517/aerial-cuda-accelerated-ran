"""Frozen residual and proposal comparison on common full legal prefix pools."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from common import load,targets,metrics,residual_scores,actor_joint,actor_coverage
from training.stageb_lns.residual import ResidualValue
from training.stageb_lns.cached_repair import CachedDeadlineRepair
from training.stageb_lns.experiment import write_json,sha

def main():
    out=HERE/'evaluation';out.mkdir(exist_ok=False);plan=json.loads((HERE/'plan.json').read_text())
    rs=json.loads((HERE/'residual_training/selected.json').read_text());ps=json.loads((HERE/'proposer_training/selected.json').read_text())
    assert sha(HERE/'residual_frozen.npz')==rs['sha256']
    assert sha(HERE/'actor_frozen.npz')==ps['sha256']
    a,p,g=load();high=targets(a,g);model=ResidualValue(HERE/'residual_frozen.npz')
    pred=np.concatenate([model.predict_rows(p['x'][start:start+65536]) for start in range(0,len(p['x']),65536)],axis=1)
    score,std=residual_scores(a,p,pred,model.mode)
    actor=CachedDeadlineRepair(HERE/'actor_frozen.npz',16)
    logits=np.concatenate([actor.logits(a['actor_x'][i:i+65536]) for i in range(0,len(a['delta']),65536)])
    new_joint=actor_joint(a,g,logits)
    np.savez_compressed(out/'scores.npz',residual=score,std=std,new_joint=new_joint)
    retained=g[g.split!='train']
    covers=pd.concat([actor_coverage(a,retained,a['old_logp'],'original_actor',high),
        actor_coverage(a,retained,new_joint,'outside_positive_actor',high)],ignore_index=True)
    covers.to_csv(out/'proposal_per_group.csv',index=False)
    covers.groupby(['split','seed','method']).mean(numeric_only=True).to_csv(out/'proposal_per_seed.csv')
    # Rebuild each short-list mask from model outputs and frozen legal options.
    from training.stageb_lns.prefix_repair import shortlist
    restricted={name:np.full(len(score),-1.e30) for name in ('original_actor_residual','new_actor_residual')}
    for row in g.itertuples():
        choices=np.asarray(json.loads(row.choices));ids=np.arange(row.start,row.end)
        anchor=int(choices[np.flatnonzero(a['actor_x'][row.start:row.start+row.options,74]>.5)[0]])
        for name,joint in (('original_actor_residual',a['old_logp']),('new_actor_residual',new_joint)):
            keep=shortlist(joint[ids].reshape(row.parents,row.options),choices,anchor,16,48,2)
            restricted[name][ids[keep]]=score[ids[keep]]
    scores=dict(trace_draft=a['draft_scores'][:,0],spatial_draft=a['draft_scores'][:,1],
        residual=score,original_actor=a['old_logp'],new_actor=new_joint,
        exact_oracle=np.where(a['eligible'],a['delta'],-1.e30))
    frames=[metrics(a,retained,s,plan['budgets'],name,high) for name,s in scores.items()]
    # At10% full-pool budget k<=21<=48; no artificial filling outside the shortlist.
    frames += [metrics(a,retained,s,[.1],name,high) for name,s in restricted.items()]
    d=pd.concat(frames,ignore_index=True);d.to_csv(out/'curves_per_group.csv',index=False)
    means=d.groupby(['split','seed','method','budget']).mean(numeric_only=True).reset_index()
    means.to_csv(out/'curves_per_seed.csv',index=False)
    means.groupby(['split','method','budget']).mean(numeric_only=True).to_csv(out/'curves_macro.csv')
    calibration=g[g.split=='calibration'];crit=[];weights=[]
    for row in calibration.itertuples():
        ids=np.arange(row.start,row.end);ids=ids[high[ids]]
        if len(ids):crit.extend(-score[ids]/(std[ids]+200));weights.extend(np.repeat(1/len(ids),len(ids)))
    crit=np.asarray(crit);weights=np.asarray(weights);order=np.argsort(crit,kind='stable')
    cdf=np.cumsum(weights[order]);cdf/=cdf[-1]
    lam=float(crit[order[min(np.searchsorted(cdf,.98),len(order)-1)]]+1.e-10)
    rows=[]
    for row in retained.itertuples():
        ids=np.arange(row.start,row.end);keep=(score[ids]+lam*(std[ids]+200))>0
        target=high[ids]
        rows.append(dict(split=row.split,seed=row.seed,group=row.group,kept_fraction=float(keep.mean()),
            recall=float((keep&target).sum()/target.sum()) if target.any() else np.nan))
    threshold=pd.DataFrame(rows);threshold.to_csv(out/'empirical_threshold_groups.csv',index=False)
    threshold.groupby(['split','seed']).mean(numeric_only=True).to_csv(out/'empirical_threshold_seeds.csv')
    write_json(out/'calibration.json',dict(lambda_empirical=lam,alpha=.02,seed=45,
        independent_seed_units=1,formal_nontrivial_guarantee=False,formal_fallback='verify_all',
        historical_development_exposure=True))
    mae=[]
    for seed in plan['evaluation_seeds']:
        ids=p['seed']==seed;raw=p['raw'][ids];base=raw[:,1+model.mode]
        estimated=np.clip(base[None]+pred[:,ids]*np.maximum(raw[:,0],2400)[None],0,raw[:,0][None]).mean(0)
        mae.append(dict(seed=seed,pairs=int(ids.sum()),draft_expected_mae=float(np.abs(base-p['exact_expected'][ids]).mean()),
            residual_expected_mae=float(np.abs(estimated-p['exact_expected'][ids]).mean())))
    pd.DataFrame(mae).to_csv(out/'per_ue_mae.csv',index=False)
    gate=means[(means.split=='evaluation')&(means.method=='residual')&(means.budget==.1)]
    baseline=means[(means.split=='evaluation')&(means.method==('trace_draft' if model.mode==0 else 'spatial_draft'))&(means.budget==.1)]
    passed=bool(len(gate)==2 and (gate.recall>=.95).all() and np.all(gate.recall.to_numpy()>baseline.recall.to_numpy()))
    write_json(out/'checks.json',dict(passed=True,scientific_gate_passed=passed,mode=model.mode,
        residual_sha256=rs['sha256'],actor_sha256=ps['sha256'],
        evaluation_seed_results=gate[['seed','recall','any_high_yield','actual_fraction','regret']].to_dict('records'),
        no_evaluation_retraining=True,new_final_seed_campaign_started=False,
        interpretation='Full-legal teacher-forced prefix pools; a larger population than the previous48-shortlist pilot. Not an end-to-end or native-delay result'))
    print('FROZEN EVALUATION\n',means[(means.split=='evaluation')&(means.budget==.1)][['seed','method','actual_fraction','recall','any_high_yield','regret']].to_string(index=False),flush=True)
    print('PROPOSAL\n',covers[covers.split=='evaluation'].groupby(['seed','method'])[['recall','outside_recall']].mean().to_string(),flush=True)
    print('GATE',passed,flush=True)

if __name__=='__main__':main()

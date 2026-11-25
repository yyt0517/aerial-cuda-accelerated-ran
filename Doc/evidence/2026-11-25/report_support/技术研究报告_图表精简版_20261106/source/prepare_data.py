"""Recompute figure inputs from the archived 120 mature packet cohorts; no simulation."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1]
REPO=BASE.parents[1]
SOURCE=REPO/'deliverables/stage2_closure_20261105_r2'
OUT=BASE/'data';OUT.mkdir(exist_ok=True)
raw=pd.read_csv(SOURCE/'analysis/per_seed.csv')
arms=['rr','rrq','pf','pfq','deficit','mlwdf_pf']
per=raw[raw.seed.between(41,50)&raw.arm.isin(arms)&(((raw.scene=='boundary')&(raw.rate==.8))|((raw.scene=='uniform')&(raw.rate==1)))].sort_values(['scene','arm','seed']).copy()
assert len(per)==120 and not per.duplicated(['scene','arm','seed']).any()
assert all(set(g.seed)==set(range(41,51)) for _,g in per.groupby(['scene','arm']))
per.to_csv(OUT/'per_seed_metrics.csv',index=False)
bins=[];ues=[];manifest={};checks=[]
for j,r in enumerate(per.itertuples(),1):
    folder=SOURCE/'analysis'/r.key
    path=folder/'cohort.csv.gz'
    manifest[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ['per_ue.csv','summary.json']:
        f=folder/name;manifest[str(f.resolve())]=hashlib.sha256(f.read_bytes()).hexdigest()
    f=pd.read_csv(path,usecols=['ue_id','arrival_tti','packet_size_bytes','outcome','system_time_ms','censored','successfully_delivered'])
    assert len(f)==r.cohort_packets and not f.censored.any()
    assert f.arrival_tti.between(8001,28000).all()
    assert set(f.outcome)<= {'mac_completed','expired'}
    assert f.system_time_ms.between(.5,200).all()
    exp=f.outcome.eq('expired');d=f.system_time_ms;good=f.successfully_delivered
    assert (d[exp]==200).all()
    metrics=[f.loc[good,'packet_size_bytes'].sum()*8/10e6,100*exp.mean(),d[good].mean()]
    np.testing.assert_allclose(metrics,[r.payload_mbps,r.expiry_pct,r.delay_mean_ms],rtol=1e-10,atol=1e-10)
    masks=[(~exp)&(d<=5),(~exp)&(d>5)&(d<=20),(~exp)&(d>20)&(d<=100),(~exp)&(d>100)&(d<=200),exp]
    counts=[int(m.sum()) for m in masks];assert sum(counts)==len(f)
    for label,count in zip(['le5','5_20','20_100','100_200','expired'],counts):
        bins.append(dict(scene=r.scene,arm=r.arm,seed=r.seed,bin=label,count=count,total=len(f),pct=100*count/len(f)))
    uf=pd.read_csv(folder/'per_ue.csv').sort_values('ue')
    assert list(uf.ue)==list(range(36))
    rates=f.assign(expired=exp).groupby('ue_id').expired.mean()*100
    np.testing.assert_allclose(rates,uf.expiry_pct,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(rates.max(),r.worst_ue_expiry_pct,rtol=1e-10,atol=1e-10)
    for row in uf.itertuples():ues.append(dict(scene=r.scene,arm=r.arm,seed=r.seed,ue=row.ue,expiry_pct=row.expiry_pct))
    checks.append(dict(key=r.key,packets=len(f),expired=int(exp.sum()),core_metrics_reproduced=True,per_ue_reproduced=True))
    if j%10==0: print(f'Validated {j}/120 packet cohorts',flush=True)
b=pd.DataFrame(bins);b.to_csv(OUT/'delay_bins_per_seed.csv',index=False)
b.groupby(['scene','arm','bin'],sort=False).pct.mean().reset_index().to_csv(OUT/'delay_bins_mean.csv',index=False)
u=pd.DataFrame(ues);u.to_csv(OUT/'per_seed_per_ue.csv',index=False)
rank=[]
for (scene,arm,seed),g in u.groupby(['scene','arm','seed']):
    for i,v in enumerate(np.sort(g.expiry_pct),1):rank.append(dict(scene=scene,arm=arm,seed=seed,rank=i,expiry_pct=v))
ranks=pd.DataFrame(rank);ranks.to_csv(OUT/'ue_rank_per_seed.csv',index=False)
rankmean=ranks.groupby(['scene','arm','rank']).expiry_pct.mean().reset_index();rankmean.to_csv(OUT/'ue_rank_mean.csv',index=False)
u.groupby(['scene','arm','ue']).expiry_pct.mean().reset_index().to_csv(OUT/'fixed_ue_mean.csv',index=False)
assert u[u.arm=='mlwdf_pf'].expiry_pct.eq(0).all()
for scene in ['boundary','uniform']:
    for arm in arms:
        got=rankmean[(rankmean.scene==scene)&(rankmean.arm==arm)&(rankmean['rank']==36)].expiry_pct.iloc[0]
        expected=per[(per.scene==scene)&(per.arm==arm)].worst_ue_expiry_pct.mean()
        np.testing.assert_allclose(got,expected)
casefile=SOURCE/'audit/boundary_r0.8_mlwdf_pf_s43_tti18001.json'
if not casefile.exists():
    casefile=next(p for p in sorted((SOURCE/'audit').glob('boundary_r0.8_mlwdf_pf_s43_tti*.json')) if json.loads(p.read_text()).get('tti')==18001)
case=json.loads(casefile.read_text())
(OUT/'joint_case.json').write_text(json.dumps(case,ensure_ascii=False,indent=2)+'\n')
manifest[str(casefile.resolve())]=hashlib.sha256(casefile.read_bytes()).hexdigest()
(OUT/'input_sha256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(OUT/'validation.json').write_text(json.dumps(dict(passed=True,simulation_rerun=False,cohorts=len(checks),packets=sum(x['packets'] for x in checks),seeds=list(range(41,51)),aggregation='Equal seed means, not pooled packet weighting',delay_bins='MAC-completed (including CRC-damaged) split by delay; expired packets exclusively in final bin; all mature arrivals in denominator',ue_ranking='Sort 36 UE rates within each seed and algorithm, then average each rank across seeds. Rank is not a fixed UE identity.',da_pls_all_720_ue_seed_rates_zero=True,checks=checks),ensure_ascii=False,indent=2)+'\n')
print('Done:',OUT,flush=True)

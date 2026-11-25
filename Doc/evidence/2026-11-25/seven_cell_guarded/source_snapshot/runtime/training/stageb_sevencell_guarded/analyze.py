"""Frozen packet metrics, exact input pairing and guard replay for every decision."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from training.stageb_ppo.native import sha,write
from training.stageb_sevencell import analyze as frozen
from training.stageb_sevencell.campaign import LABELS as OLD_LABELS
from .policy import ARM,LABEL,TIE_ORDER,LAMBDA,BETA

BASELINES=('deficit','mlwdf_pf','da_three_branch')

def one(directory):
    frozen.LABELS={**OLD_LABELS,ARM:LABEL}
    return frozen.one(directory)

def main(root,baseline,multistart,workers=3):
    root=Path(root);baseline=Path(baseline);multistart=Path(multistart)
    source=root/'campaign';output=root/'analysis';output.mkdir(exist_ok=False)
    protocol=json.loads((source/'protocol.json').read_text());complete=json.loads((source/'completion.json').read_text())
    assert complete['completed'] and not complete['failed']
    assert protocol['lambda_value']==LAMBDA and protocol['beta']==BETA
    paths=[source/f'{ARM}_s{s}' for s in protocol['seeds']]
    with ProcessPoolExecutor(workers) as pool:rows=list(pool.map(one,map(str,paths)))
    historical=baseline/('smoke_final' if protocol['smoke'] else 'campaign')
    three=multistart/('smoke/campaign' if protocol['smoke'] else 'campaign')
    paired=[];branch_rows=[];row_paths={(ARM,s):p for s,p in zip(protocol['seeds'],paths)}
    for seed,path in zip(protocol['seeds'],paths):
        warm=protocol['windows'][0]
        refs={'pfq':historical/f'pfq_s{seed}','deficit':historical/f'deficit_s{seed}',
              'mlwdf_pf':historical/f'mlwdf_pf_s{seed}','da_three_branch':three/f'da_three_branch_s{seed}'}
        shape=(-1,84,len(frozen.FIELDS));a=np.memmap(path/'metrics.bin',dtype='<f8',mode='r').reshape(shape)
        for arm,ref in refs.items():
            for name in ('traffic_offered_packets.csv','ue_topology.csv','serving.npy','channel_hashes.npy'):
                assert sha(path/name)==sha(ref/name),(seed,arm,name)
            with np.load(path/'actions.npz') as x,np.load(ref/'actions.npz') as y:
                np.testing.assert_array_equal(x['actions'][:warm],y['actions'][:warm])
                np.testing.assert_array_equal(x['input_hashes'][:warm+1],y['input_hashes'][:warm+1])
            b=np.memmap(ref/'metrics.bin',dtype='<f8',mode='r').reshape(shape)
            np.testing.assert_array_equal(a[:warm],b[:warm])
            paired.append(dict(seed=seed,reference=arm,topology_traffic_all_channels_equal=True,
                warmup_actions_observations_and_native_metrics_equal=True,first_policy_input_equal=True))
            if arm in BASELINES:
                rows.append(json.loads((ref/'summary.json').read_text()));row_paths[(arm,seed)]=ref
        assert json.loads((path/'policy_validation.json').read_text())['passed']
        launch=json.loads((path/'launch.json').read_text());search=pd.read_csv(path/'search.csv.gz',float_precision='round_trip')
        assert len(search)==sum(protocol['windows'][1:]) and search.converged.all() and search.service_guard_passed.all()
        scores=search[[n+'_score_q' for n in TIE_ORDER]].to_numpy()
        services=search[[n+'_expected_bytes' for n in TIE_ORDER]].to_numpy()
        floor=BETA*search.dpls_expected_bytes.to_numpy();eligible=services>=floor[:,None]
        np.testing.assert_array_equal(search.service_floor,floor)
        np.testing.assert_array_equal(search[[n+'_eligible' for n in TIE_ORDER]],eligible)
        choice=np.where(eligible,scores,-1).argmax(1);ix=np.arange(len(search))
        np.testing.assert_array_equal(search.selected_branch,np.array(TIE_ORDER)[choice])
        np.testing.assert_array_equal(search.score_q,scores[ix,choice])
        np.testing.assert_array_equal(search.selected_expected_bytes,services[ix,choice])
        assert (search.score_q>=search.dpls_score_q).all() and (search.selected_expected_bytes>=floor).all()
        assert (search.new_from_dpls_score_q>=search.dpls_score_q).all()
        np.testing.assert_array_equal(search.unconstrained_winner,np.array(TIE_ORDER)[scores.argmax(1)])
        np.testing.assert_array_equal(search.guard_changed_winner,choice!=scores.argmax(1))
        z=search[search.tti.between(launch['measurement_first_tti'],launch['measurement_last_tti'])]
        assert len(z)==launch['measured']
        ratio=np.divide(z.selected_expected_bytes,z.dpls_expected_bytes,out=np.ones(len(z)),where=z.dpls_expected_bytes>0)
        rec=dict(seed=seed,decision_ttis=len(search),measured_ttis=len(z),
            guard_changed_winner_pct=float(100*z.guard_changed_winner.mean()),
            new_score_strict_gain_vs_dpls_pct=float(100*(z.score_q>z.dpls_score_q).mean()),
            selected_vs_dpls_expected_service_pct=float(100*(z.selected_expected_bytes.mean()/z.dpls_expected_bytes.mean()-1)),
            min_expected_service_ratio=float(ratio.min()),mean_rounds_total=float(z['rounds'].mean()),wall_ms_mean=float(z.wall_ms.mean()))
        for n in TIE_ORDER:
            rec[n+'_selected_pct']=float(100*z.selected_branch.eq(n).mean())
            rec[n+'_eligible_pct']=float(100*z[n+'_eligible'].mean())
        branch_rows.append(rec)
    for row in rows:
        p=row_paths[(row['arm'],row['seed'])]
        pk=pd.read_csv(p/'cohort.csv.gz',usecols=['system_time_ms','successfully_delivered'])
        delay=pk.loc[pk.successfully_delivered,'system_time_ms']
        row.update(delay_p50_ms=float(delay.quantile(.5)),success_le10ms_pct=float(100*(delay<=10).mean()))
    df=pd.DataFrame(rows);df['reused_baseline']=df.arm!=ARM;df.to_csv(output/'per_seed.csv',index=False)
    metrics=['payload_mbps','goodput_mbps','delay_mean_ms','delay_p50_ms','delay_p95_ms','success_le10ms_pct','expiry_pct',
        'outcome_mean_ms','outcome_p95_ms','ue_payload_p5_mbps','jain_payload','worst_ue_expiry_pct',
        'inr_pooled_db','sinr_mean_db','occupancy_pct','other_nondelivery_pct','queue_mean_mb']
    aggregates=[];pairs=[]
    for arm,z in df.groupby('arm'):
        for m in metrics:aggregates.append(dict(arm=arm,label=z.label.iloc[0],metric=m,n=len(z),mean=float(z[m].mean()),std=float(z[m].std()) if len(z)>1 else None))
    a=df[df.arm.eq(ARM)].set_index('seed')
    for arm in BASELINES:
        b=df[df.arm.eq(arm)].set_index('seed')
        for m in metrics:
            d=a[m]-b[m];n=len(d);half=float(student_t.ppf(.975,n-1)*d.std()/np.sqrt(n)) if n>1 else np.nan
            pairs.append(dict(arm=ARM,baseline=arm,metric=m,n=n,delta=float(d.mean()),ci_low=float(d.mean()-half),ci_high=float(d.mean()+half),
                relative_mean_change_pct=float(100*(a[m].mean()/b[m].mean()-1)) if b[m].mean()!=0 else np.nan))
    pd.DataFrame(aggregates).to_csv(output/'aggregate.csv',index=False);pd.DataFrame(pairs).to_csv(output/'paired.csv',index=False)
    pd.DataFrame(branch_rows).to_csv(output/'branch_selection.csv',index=False)
    write(output/'paired_validation.json',dict(passed=True,checks=paired,every_decision_guard_and_max_eligible_score=True))
    write(output/'completion.json',dict(completed=True,new_runs=len(paths),reused_runs=len(rows)-len(paths),smoke=protocol['smoke'],seeds=protocol['seeds']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--baseline',required=True);p.add_argument('--multistart',required=True);p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();main(a.root,a.baseline,a.multistart,a.workers)

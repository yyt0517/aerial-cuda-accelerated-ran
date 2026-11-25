"""Reuse the frozen complete-packet metrics, then pair with archived baselines."""
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
from .policy import ARM,LABEL

def one(directory):
    frozen.LABELS={**OLD_LABELS,ARM:LABEL}
    return frozen.one(directory)

def main(root,baseline,workers=3):
    root=Path(root);baseline=Path(baseline);source=root/'campaign';output=root/'analysis';output.mkdir(exist_ok=False)
    protocol=json.loads((source/'protocol.json').read_text());complete=json.loads((source/'completion.json').read_text())
    assert complete['completed'] and not complete['failed']
    paths=[source/f'{ARM}_s{s}' for s in protocol['seeds']]
    with ProcessPoolExecutor(workers) as pool:rows=list(pool.map(one,map(str,paths)))
    historical=baseline/('smoke_final' if protocol['smoke'] else 'campaign');paired=[];branch_rows=[]
    for seed,path in zip(protocol['seeds'],paths):
        ref=historical/f'pfq_s{seed}';warm=protocol['windows'][0]
        for name in ('traffic_offered_packets.csv','ue_topology.csv','serving.npy','channel_hashes.npy'):
            assert sha(path/name)==sha(ref/name),(seed,name)
        with np.load(path/'actions.npz') as a,np.load(ref/'actions.npz') as b:
            np.testing.assert_array_equal(a['actions'][:warm],b['actions'][:warm])
            np.testing.assert_array_equal(a['input_hashes'][:warm+1],b['input_hashes'][:warm+1])
        shape=(-1,84,len(frozen.FIELDS))
        a=np.memmap(path/'metrics.bin',dtype='<f8',mode='r').reshape(shape)
        b=np.memmap(ref/'metrics.bin',dtype='<f8',mode='r').reshape(shape)
        np.testing.assert_array_equal(a[:warm],b[:warm])
        assert json.loads((path/'policy_validation.json').read_text())['passed']
        paired.append(dict(seed=seed,topology_traffic_all_channels_equal=True,warmup_actions_observations_and_all_native_metrics_equal=True,
            first_policy_input_equal=True))
        launch=json.loads((path/'launch.json').read_text());search=pd.read_csv(path/'search.csv.gz')
        assert len(search)==sum(protocol['windows'][1:]) and search.converged.all()
        branch_q=search[['original_da_score_q','from_dpls_da_score_q','dpls_score_q']].to_numpy()
        np.testing.assert_array_equal(search.score_q,branch_q.max(axis=1))
        expected=np.array(['original_da','from_dpls_da','dpls'])[branch_q.argmax(axis=1)]
        np.testing.assert_array_equal(search.selected_branch,expected)
        z=search[search.tti.between(launch['measurement_first_tti'],launch['measurement_last_tti'])]
        assert len(z)==launch['measured']
        rec=dict(seed=seed,decision_ttis=len(search),measured_ttis=len(z),
            da_score_strict_improvement_pct=float(100*np.mean(z.gain_over_original_da_q>0)),
            mean_da_score_gain_bytes=float(z.gain_over_original_da_q.mean()/1e6),
            mean_da_score_gain_pct=float(100*(z.score_q/z.original_da_score_q-1).mean()),
            mean_rounds_total=float(z.rounds.mean()),wall_ms_mean=float(z.wall_ms.mean()))
        for name in ('dpls','original_da','from_dpls_da'):rec[name+'_selected_pct']=float(100*np.mean(z.selected_branch==name))
        branch_rows.append(rec)
        for arm in OLD_LABELS:
            old=json.loads((historical/f'{arm}_s{seed}'/'summary.json').read_text());rows.append(old)
    df=pd.DataFrame(rows);df['reused_baseline']=df.arm!=ARM;df.to_csv(output/'per_seed.csv',index=False)
    metrics=['payload_mbps','goodput_mbps','delay_mean_ms','delay_p95_ms','expiry_pct','outcome_mean_ms','outcome_p95_ms',
        'ue_payload_p5_mbps','jain_payload','worst_ue_expiry_pct','inr_pooled_db','sinr_mean_db','occupancy_pct','other_nondelivery_pct','queue_mean_mb']
    aggregates=[];pairs=[]
    for arm,z in df.groupby('arm'):
        for metric in metrics:aggregates.append(dict(arm=arm,label=z.label.iloc[0],metric=metric,n=len(z),mean=float(z[metric].mean()),std=float(z[metric].std()) if len(z)>1 else None))
    a=df[df.arm==ARM].set_index('seed')
    for arm in OLD_LABELS:
        b=df[df.arm==arm].set_index('seed')
        for metric in metrics:
            d=a[metric]-b[metric];n=len(d);half=float(student_t.ppf(.975,n-1)*d.std()/np.sqrt(n)) if n>1 else np.nan
            pairs.append(dict(arm=ARM,baseline=arm,metric=metric,n=n,delta=float(d.mean()),ci_low=float(d.mean()-half),ci_high=float(d.mean()+half),
                relative_mean_change_pct=float(100*(a[metric].mean()/b[metric].mean()-1)) if b[metric].mean()!=0 else np.nan))
    pd.DataFrame(aggregates).to_csv(output/'aggregate.csv',index=False);pd.DataFrame(pairs).to_csv(output/'paired.csv',index=False)
    pd.DataFrame(branch_rows).to_csv(output/'branch_selection.csv',index=False)
    write(output/'paired_validation.json',dict(passed=True,checks=paired))
    write(output/'completion.json',dict(completed=True,new_runs=len(paths),reused_runs=len(rows)-len(paths),smoke=protocol['smoke'],seeds=protocol['seeds']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--baseline',required=True);p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();main(a.root,a.baseline,a.workers)

"""Complete-packet, PHY and interference results with strict paired-input checks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from training.stageb_ppo.native import ROOT,sha,write
from training.stage2_closure.revise_metrics import FIELDS
from .packets import packets_from_fifo
from .campaign import ARMS,LABELS

def local(p):return Path(str(p).replace('/opt/nvidia/cuBB',str(ROOT)))

def one(directory):
    out=Path(directory);launch=json.loads((out/'launch.json').read_text())
    assert json.loads((out/'validation.json').read_text())['passed']
    U,C,P,T=launch['users'],launch['cells'],launch['prgs'],launch['horizon']
    first,last=launch['measurement_first_tti'],launch['measurement_last_tti'];seconds=launch['measured']*.0005
    data=np.memmap(out/'metrics.bin',dtype='<f8',mode='r').reshape(T,U,len(FIELDS))
    service=pd.DataFrame(data.reshape(-1,len(FIELDS)),columns=FIELDS)
    pk=packets_from_fifo(service,pd.read_csv(out/'traffic_offered_packets.csv'),pd.read_csv(out/'packet_outcome_samples.csv'),T)
    rewards=pd.read_csv(out/'rewards.csv')
    np.testing.assert_array_equal(pk[pk.successfully_delivered].groupby('terminal_tti').packet_size_bytes.sum().reindex(range(1,T+1),fill_value=0),rewards.complete_payload_bytes)
    np.testing.assert_array_equal(pk[pk.outcome.eq('expired')].groupby('terminal_tti').packet_size_bytes.sum().reindex(range(T),fill_value=0),rewards.expired_payload_bytes)
    cohort=pk[pk.arrival_tti.between(first,last)].copy();assert len(cohort)>0 and not cohort.censored.any()
    cohort.to_csv(out/'cohort.csv.gz',index=False);good=cohort[cohort.successfully_delivered]
    s=service[service.tti.between(first,last)];tx=s.actual_prg_count>0
    payload=good.groupby('ue_id').packet_size_bytes.sum().reindex(range(U),fill_value=0)*8/(seconds*1e6)
    gp=s.groupby('ue_id').successful_bytes.sum().reindex(range(U),fill_value=0)*8/(seconds*1e6)
    exp=cohort.assign(expired=cohort.outcome.eq('expired')).groupby('ue_id').expired.mean().reindex(range(U))*100
    serving=np.load(out/'serving.npy');per=pd.DataFrame(dict(ue=np.arange(U),cell=serving,payload_mbps=payload.values,goodput_mbps=gp.values,expiry_pct=exp.values))
    for name,frame in [('success',good),('outcome',cohort)]:
        per[name+'_delay_mean_ms']=frame.groupby('ue_id').system_time_ms.mean().reindex(range(U)).values
        per[name+'_delay_p95_ms']=frame.groupby('ue_id').system_time_ms.quantile(.95).reindex(range(U)).values
    per.to_csv(out/'per_ue.csv',index=False)
    cells=[]
    for c in range(C):
        ids=np.flatnonzero(serving==c);z=s[s.ue_id.isin(ids)];active=z.actual_prg_count>0
        cells.append(dict(cell=c,users=len(ids),payload_mbps=float(payload.loc[ids].sum()),goodput_mbps=float(gp.loc[ids].sum()),
            occupancy_pct=float(100*z.actual_prg_count.sum()/(launch['measured']*P)),sinr_mean_db=float(z.loc[active,'phy_sinr_db_cpu'].mean())))
    pd.DataFrame(cells).to_csv(out/'per_cell.csv',index=False)
    row=dict(arm=launch['arm'],label=LABELS[launch['arm']],seed=launch['seed'],cells=C,users=U,
        payload_mbps=float(payload.sum()),goodput_mbps=float(gp.sum()),payload_per_cell_mbps=float(payload.sum()/C),
        offered_mbps=float(s.offered_bytes.sum()*8/(seconds*1e6)),success_pct=float(100*cohort.successfully_delivered.mean()),
        expiry_pct=float(100*cohort.outcome.eq('expired').mean()),
        other_nondelivery_pct=float(100*((~cohort.successfully_delivered)&(~cohort.outcome.eq('expired'))).mean()),
        delay_mean_ms=float(good.system_time_ms.mean()),delay_p95_ms=float(good.system_time_ms.quantile(.95)),
        outcome_mean_ms=float(cohort.system_time_ms.mean()),outcome_p95_ms=float(cohort.system_time_ms.quantile(.95)),
        ue_payload_p5_mbps=float(payload.quantile(.05)),jain_payload=float(payload.sum()**2/(U*(payload**2).sum())),
        worst_ue_expiry_pct=float(exp.max()),sinr_mean_db=float(s.loc[tx,'phy_sinr_db_cpu'].mean()),
        mcs_mean=float(s.loc[tx,'mcs'].mean()),tb_error_pct=float(100*s.tb_err_count.sum()/s.tb_tx_count.sum()),
        occupancy_pct=float(100*s.actual_prg_count.sum()/(launch['measured']*C*P)),
        queue_mean_mb=float(s.groupby('tti').post_queue_bytes.sum().mean()/1e6),
        queue_start_mb=float(s[s.tti==first].previous_post_queue_bytes.sum()/1e6),queue_end_mb=float(s[s.tti==last].post_queue_bytes.sum()/1e6),
        cohort_packets=len(cohort),censored=0)
    ir=pd.read_csv(out/'interference.csv.gz');row['inr_pooled']=float(ir.inr_sum.sum()/ir.links.sum());row['inr_pooled_db']=float(10*np.log10(row['inr_pooled']))
    with np.load(out/'actions.npz') as a:
        occupancy=(a['actions'][first-1:last]>=0).sum(axis=1)
        for n in range(C+1):row[f'reuse_{n}_pct']=float(100*np.mean(occupancy==n))
    if launch['arm']!='pfq':
        r=pd.read_csv(out/'search.csv.gz');r=r[r.tti.between(first,last)];assert len(r)==launch['measured'] and r.converged.all()
        for m in ['wall_ms','cpu_ms','rounds','candidate_evaluations']:
            row[m+'_mean']=float(r[m].mean());row[m+'_p95']=float(r[m].quantile(.95));row[m+'_max']=float(r[m].max())
    # Selected action's PHY is independently rebuilt and checked against native.
    import torch
    from training.stageb_lns.no_ctbs_inputs import problem_from_raw_row
    from training.stageb_lns.physics import ReferencePhy
    checks=[]
    for e in json.loads((out/'saved_states.json').read_text()):
        f=local(e['path']);assert sha(f)==e['sha256'];state=torch.load(f,map_location='cpu',weights_only=False)
        p=problem_from_raw_row(state);v=ReferencePhy(p).evaluate(state['selected_action'])[0];actual=data[int(state['tti'])-1];active=v[:,0]>0
        if 'selected_values' in state:np.testing.assert_allclose(v,state['selected_values'],rtol=1e-8,atol=1e-6)
        for i,column in [(0,'actual_prg_count'),(1,'layer'),(3,'scheduled_dequeue_bytes')]:np.testing.assert_array_equal(v[active,i],actual[active,FIELDS.index(column)])
        err=float(np.abs(v[active,4]-actual[active,FIELDS.index('phy_sinr_db_cpu')]).max(initial=0));assert err<.005
        bler=float(np.abs(v[active,5]-actual[active,FIELDS.index('predicted_bler_cpu')]).max(initial=0));assert bler<.0005
        checks.append(dict(tti=state['tti'],sinr_max_error_db=err,bler_max_error=bler))
    write(out/'analysis_validation.json',dict(passed=True,cohort_mature=True,fifo_crc_native_terminals_exact=True,selected_physics=checks))
    write(out/'summary.json',row);print('ANALYZED',out.name,flush=True);return row

def summarize(source,output):
    protocol=json.loads((source/'protocol.json').read_text());rows=[];paired_checks=[]
    for seed in protocol['seeds']:
        base=source/f'pfq_s{seed}'
        with np.load(base/'actions.npz') as b:
            warm=protocol['windows'][0]
            for arm in protocol['arms']:
                p=source/f'{arm}_s{seed}';rows.append(json.loads((p/'summary.json').read_text()))
                for name in ('traffic_offered_packets.csv','ue_topology.csv','serving.npy','channel_hashes.npy'):
                    assert sha(p/name)==sha(base/name),(seed,arm,name)
                with np.load(p/'actions.npz') as a:
                    np.testing.assert_array_equal(a['actions'][:warm],b['actions'][:warm]);np.testing.assert_array_equal(a['input_hashes'][:warm],b['input_hashes'][:warm])
                paired_checks.append(dict(seed=seed,arm=arm,topology_traffic_all_channels_and_common_pfq_prefix=True))
    output.mkdir(parents=True,exist_ok=True);df=pd.DataFrame(rows);df.to_csv(output/'per_seed.csv',index=False)
    metrics=[c for c in df.select_dtypes('number') if c not in ('seed','cells','users','censored')]
    aggregate=[];pairs=[]
    for arm,z in df.groupby('arm'):
        for m in metrics:
            a=z[m].dropna();aggregate.append(dict(arm=arm,label=LABELS[arm],metric=m,n=len(a),mean=float(a.mean()) if len(a) else None,std=float(a.std()) if len(a)>1 else None))
    for arm,baseline in [('deficit','pfq'),('mlwdf_pf','pfq'),('mlwdf_pf','deficit')]:
        a=df[df.arm==arm].set_index('seed');b=df[df.arm==baseline].set_index('seed')
        if not len(a) or not len(b):continue
        for m in metrics:
            d=(a[m]-b[m]).dropna();n=len(d)
            if not n:continue
            mean=float(d.mean());half=float(student_t.ppf(.975,n-1)*d.std()/np.sqrt(n)) if n>1 else None
            pairs.append(dict(arm=arm,baseline=baseline,metric=m,n=n,delta=mean,ci_low=None if half is None else mean-half,ci_high=None if half is None else mean+half))
    pd.DataFrame(aggregate).to_csv(output/'aggregate.csv',index=False);pd.DataFrame(pairs).to_csv(output/'paired.csv',index=False)
    write(output/'paired_validation.json',dict(passed=True,checks=paired_checks))
    write(output/'completion.json',dict(completed=True,runs=len(df),smoke=protocol['smoke'],seeds=protocol['seeds']))
    return df

def main(a):
    source=Path(a.source).resolve();out=Path(a.output).resolve();paths=[p.parent for p in source.glob('*/done.json')]
    protocol=json.loads((source/'protocol.json').read_text());assert len(paths)==len(protocol['seeds'])*len(protocol['arms'])
    with ProcessPoolExecutor(a.workers) as pool:list(pool.map(one,map(str,paths)))
    summarize(source,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=3);main(p.parse_args())

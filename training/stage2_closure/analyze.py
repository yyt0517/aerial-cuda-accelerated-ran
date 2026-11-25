"""Mature packet cohorts, equal-seed statistics and matched-prefix validation."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from training.stageb_ppo.native import ROOT, legacy, sha, write, EXPECTED_SIMULATOR
from .campaign import FIRST, LAST, MEASURED, SEEDS

def local(path):
    return Path(str(path).replace('/opt/nvidia/cuBB',str(ROOT)))

def analyze_one(done,output):
    spec=json.loads(Path(done).read_text());d=local(spec['directory']);out=Path(output)/spec['key'];out.mkdir(parents=True,exist_ok=True)
    if spec['reused']:
        for name,digest in spec['input_sha256'].items():
            assert sha(d/name)==digest, f'Frozen reused input changed: {d/name}'
    launch=json.loads((d/'launch.json').read_text())
    assert launch['horizon']==28400 and launch['simulator_sha256']==EXPECTED_SIMULATOR
    assert json.loads((d/'validation.json').read_text())['passed']
    assert launch['env']['CUMAC_SHADOW_FADING_STD_DB']=='3.0'
    assert float(launch['env']['CUMAC_TRAFFIC_ARRIVAL_RATE'])==spec['rate']
    assert launch['env']['CUMAC_UE_PLACEMENT_MODE']==('uniform' if spec['scene']=='uniform' else 'coop_boundary')
    libs=[v for k,v in launch['source_sha256'].items() if k.endswith('/libcumac.so')]
    assert libs==['7b414f13851e50da73c91db3fc1342017113c8180a04148606f24a5adcc08eea']
    shadows=re.findall(r'Shadow fading configuration: std_db=([0-9.eE+-]+) ln_std=([0-9.eE+-]+)',(d/'sim.log').read_text())
    assert shadows and all(abs(float(db)-3.)<1e-6 and abs(float(ln)-3.*np.log(10.)/10.)<1e-6 for db,ln in shadows)
    write(out/'source_validation.json',dict(passed=True,horizon=28400,simulator=EXPECTED_SIMULATOR,libcumac=libs[0],
        native_shadow_parameters=[dict(std_db=float(a),ln_std=float(b)) for a,b in shadows],native_log_sha256=sha(d/'sim.log')))
    if (out/'summary.json').exists():return json.loads((out/'summary.json').read_text())
    b=legacy()
    from output.pfq_independent_reuse.formal.boundary08_fast_multiseed_20261010_v1.summarize import packets_from_fifo
    fields=b.FIELDS
    data=np.memmap(d/'metrics.bin',dtype='<f8',mode='r').reshape(-1,36,len(fields))
    assert data.shape[0]==28400
    service=pd.DataFrame(data.reshape(-1,len(fields)),columns=fields)
    if not spec['reused'] and spec['arm'] not in ('pfq','rr','rrq','pf'):
        import torch
        from training.stageb_lns.archive import enable_legacy_types
        from training.stageb_lns.no_ctbs_inputs import problem_from_raw_row
        from training.stageb_lns.physics import ReferencePhy
        enable_legacy_types();samples=json.loads((d/'saved_states.json').read_text());checks=[]
        assert len(samples)==3
        for entry in samples:
            path=local(entry['path']);assert sha(path)==entry['sha256']
            row=torch.load(path,map_location='cpu',weights_only=False);v=row['selected_values'];actual=data[int(row['tti'])-1]
            independent=ReferencePhy(problem_from_raw_row(row)).evaluate(row['selected_action'])[0]
            np.testing.assert_allclose(v,independent,rtol=1e-8,atol=1e-6)
            active=v[:,0]>0
            for i,column in [(0,'actual_prg_count'),(1,'layer'),(3,'scheduled_dequeue_bytes')]:
                np.testing.assert_array_equal(v[active,i],actual[active,fields.index(column)])
            sinr=float(np.max(np.abs(v[active,4]-actual[active,fields.index('phy_sinr_db_cpu')]),initial=0))
            bler=float(np.max(np.abs(v[active,5]-actual[active,fields.index('predicted_bler_cpu')]),initial=0))
            assert sinr<.005 and bler<.0005, (spec['key'],row['tti'],sinr,bler)
            checks.append(dict(tti=int(row['tti']),tx_ues=int(active.sum()),sinr_max_abs_db=sinr,bler_max_abs=bler,passed=True))
        write(out/'physical_validation.json',dict(passed=True,samples=checks,sinr_tolerance_db=.005,bler_tolerance=.0005,
            scope='selected full-TB values vs independent NumPy MMSE and actual native transmission; separate float tolerance from exact actions/bytes'))
    offered=pd.read_csv(d/'traffic_offered_packets.csv');outcomes=pd.read_csv(d/'packet_outcome_samples.csv')
    pk=packets_from_fifo(service,offered,outcomes,28400)
    cohort=pk[pk.arrival_tti.between(FIRST,LAST)].copy()
    assert len(cohort)>0 and not cohort.censored.any()
    good=cohort[cohort.successfully_delivered];s=service[service.tti.between(FIRST,LAST)]
    per_payload=good.groupby('ue_id').packet_size_bytes.sum().reindex(range(36),fill_value=0)*8/10e6
    per_goodput=s.groupby('ue_id').successful_bytes.sum().reindex(range(36),fill_value=0)*8/10e6
    expired=cohort.assign(expired=cohort.outcome.eq('expired')).groupby('ue_id').expired.mean().reindex(range(36),fill_value=0)*100
    tx=s.actual_prg_count>0
    per=pd.DataFrame(dict(ue=np.arange(36),payload_mbps=per_payload.values,goodput_mbps=per_goodput.values,expiry_pct=expired.values))
    per['success_delay_mean_ms']=good.groupby('ue_id').system_time_ms.mean().reindex(range(36)).values
    per.to_csv(out/'per_ue.csv',index=False)
    cohort.to_csv(out/'cohort.csv.gz',index=False)
    rows=dict(scene=spec['scene'],rate=spec['rate'],arm=spec['arm'],seed=spec['seed'],key=spec['key'],
        payload_mbps=float(per_payload.sum()),goodput_mbps=float(per_goodput.sum()),
        expiry_pct=float(100*cohort.outcome.eq('expired').mean()),
        success_pct=float(100*cohort.successfully_delivered.mean()),
        other_nondelivery_pct=float(100*((~cohort.successfully_delivered)&(~cohort.outcome.eq('expired'))).mean()),
        delay_mean_ms=float(good.system_time_ms.mean()),delay_p95_ms=float(good.system_time_ms.quantile(.95)),
        delay_p99_ms=float(good.system_time_ms.quantile(.99)),terminal_mean_ms=float(cohort.system_time_ms.mean()),
        ue_payload_p5_mbps=float(per_payload.quantile(.05)),ue_goodput_p5_mbps=float(per_goodput.quantile(.05)),
        jain_payload=float(per_payload.sum()**2/(36*(per_payload**2).sum())),
        jain_goodput=float(per_goodput.sum()**2/(36*(per_goodput**2).sum())),worst_ue_expiry_pct=float(expired.max()),
        sinr_tx_mean_db=float(s.loc[tx,'phy_sinr_db_cpu'].mean()),mcs_tx_mean=float(s.loc[tx,'mcs'].mean()),
        prg_occupancy_pct=float(100*s.actual_prg_count.sum()/(MEASURED*51)),
        bytes_per_used_prg=float(s.successful_bytes.sum()/s.actual_prg_count.sum()),
        tb_error_pct=float(100*s.tb_err_count.sum()/s.tb_tx_count.sum()),cohort_packets=len(cohort),
        censored=0,measurement_seconds=10.,native_directory=str(d),reused=spec['reused'])
    # Preserve per-seed distributions; plotted aggregation must give seeds equal weight.
    for column,bins in [('phy_sinr_db_cpu',np.linspace(-20,60,161)),('mcs',np.arange(-.5,28.5,1))]:
        counts,_=np.histogram(s.loc[tx,column],bins)
        assert counts.sum()==int(tx.sum()), f'Histogram range excludes transmitted samples: {spec["key"]}, {column}'
        pd.DataFrame(dict(left=bins[:-1],right=bins[1:],count=counts,mass=counts/counts.sum())).to_csv(out/f'{column}_hist.csv',index=False)
    action_file=d/('prefix.npz' if spec['reused'] else 'actions.npz')
    with np.load(action_file) as z:
        actions=z['actions'].reshape(-1,3,17)
        assert len(actions)==28400
        occupancy=(actions[8000:28000]>=0).sum(axis=1)
        for n in range(4):rows[f'reuse_{n}_pct']=float(100*np.mean(occupancy==n))
        np.savez_compressed(out/'prefix.npz',actions=actions[:4000],input_hashes=z['input_hashes'][:4000])
    traffic=offered[offered.arrival_tti<=28400].sort_values(['ue_id','packet_id']).to_numpy(dtype=np.int64)
    import hashlib
    rows['traffic_sha256']=hashlib.sha256(np.ascontiguousarray(traffic).tobytes()).hexdigest()
    topology=pd.read_csv(d/'ue_topology.csv')
    rows['topology_sha256']=hashlib.sha256(topology.to_csv(index=False).encode()).hexdigest()
    if not spec['reused'] and (d/'search.csv.gz').exists():
        try:records=pd.read_csv(d/'search.csv.gz')
        except pd.errors.EmptyDataError:
            assert spec['arm']=='pfq', 'Only the native PFQ arm omits search records'
            records=pd.DataFrame()
        if len(records):
            r=records[records.tti.between(FIRST,LAST)]
            assert len(r)==MEASURED
            for metric in ['wall_ms','cpu_ms','candidate_evaluations','ue_evaluations','mmse_misses']:
                if metric in r:
                    rows[metric+'_mean']=float(r[metric].mean());rows[metric+'_p95']=float(r[metric].quantile(.95));rows[metric+'_max']=float(r[metric].max())
    if not spec['reused']:
        rewards=pd.read_csv(d/'rewards.csv')
        success_tti=pk[pk.successfully_delivered].groupby('terminal_tti').packet_size_bytes.sum().reindex(range(1,28401),fill_value=0)
        np.testing.assert_array_equal(success_tti,rewards.complete_payload_bytes)
    write(out/'input_manifest.json',{str(d/n):sha(d/n) for n in ['launch.json','validation.json','sim.log','metrics.bin','traffic_offered_packets.csv','packet_outcome_samples.csv',action_file.name]})
    write(out/'validation.json',dict(passed=True,fifo_conservation=True,native_terminal_pairing=True,no_censoring=True,whole_packet_crc=True))
    write(out/'summary.json',rows);return rows

METRICS=['payload_mbps','goodput_mbps','expiry_pct','other_nondelivery_pct','delay_mean_ms','delay_p95_ms','delay_p99_ms',
 'ue_payload_p5_mbps','ue_goodput_p5_mbps','jain_payload','jain_goodput','worst_ue_expiry_pct','sinr_tx_mean_db','mcs_tx_mean',
 'prg_occupancy_pct','bytes_per_used_prg','tb_error_pct','reuse_0_pct','reuse_1_pct','reuse_2_pct','reuse_3_pct']

def summarize(root,output,partial=False):
    output=Path(output);rows=[json.loads(p.read_text()) for p in output.glob('*/summary.json')]
    df=pd.DataFrame(rows).sort_values(['scene','rate','arm','seed']);df.to_csv(output/'per_seed.csv',index=False)
    aggregate=[];paired=[];prefix=[];cost=[];weight_pairs=[]
    for (scene,rate,arm),group in df.groupby(['scene','rate','arm']):
        if not partial:assert sorted(group.seed)==list(SEEDS)
        for metric in ['wall_ms','cpu_ms','candidate_evaluations','ue_evaluations','mmse_misses']:
            if metric+'_mean' in group:
                z=group[group[metric+'_mean'].notna()]
                if len(z):cost.append(dict(scene=scene,rate=rate,arm=arm,metric=metric,n=len(z),
                    seed_mean=float(z[metric+'_mean'].mean()),seed_p95_mean=float(z[metric+'_p95'].mean()),
                    observed_max=float(z[metric+'_max'].max())))
        for subset,seeds in [('41-50',SEEDS),('44-50',range(44,51))]:
            z=group[group.seed.isin(seeds)]
            for metric in METRICS:
                a=z[metric].dropna().to_numpy();n=len(a)
                if not n:continue
                m=float(np.mean(a));sd=float(np.std(a,ddof=1)) if n>1 else 0.
                half=float(student_t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else None
                aggregate.append(dict(scene=scene,rate=rate,arm=arm,subset=subset,metric=metric,n=n,mean=m,std=sd,ci_low=None if half is None else m-half,ci_high=None if half is None else m+half))
        base=df[(df.scene==scene)&(df.rate==rate)&(df.arm=='pfq')]
        for _,r in group.iterrows():
            b=base[base.seed==r.seed]
            if not len(b):continue
            b=b.iloc[0]
            assert r.traffic_sha256==b.traffic_sha256 and r.topology_sha256==b.topology_sha256
            with np.load(output/r.key/'prefix.npz') as x,np.load(output/b.key/'prefix.npz') as y:
                for k in ['actions','input_hashes']:np.testing.assert_array_equal(x[k],y[k])
            prefix.append(dict(scene=scene,rate=rate,arm=arm,seed=int(r.seed),common_pfq_ttis=4000,passed=True))
        if arm=='pfq':continue
        join=group.merge(base,on='seed',suffixes=('','_pfq'))
        for subset,seeds in [('41-50',SEEDS),('44-50',range(44,51))]:
            z=join[join.seed.isin(seeds)]
            for metric in METRICS:
                x=(z[metric]-z[metric+'_pfq']).to_numpy();n=len(x)
                if n<2:continue
                mean=float(x.mean());half=float(student_t.ppf(.975,n-1)*np.std(x,ddof=1)/np.sqrt(n))
                denominator=float(z[metric+'_pfq'].mean())
                paired.append(dict(scene=scene,rate=rate,arm=arm,subset=subset,metric=metric,n=n,delta=mean,ci_low=mean-half,ci_high=mean+half,
                    relative_change_pct=100*mean/denominator if denominator else None))
    pd.DataFrame(aggregate).to_csv(output/'aggregate.csv',index=False);pd.DataFrame(paired).to_csv(output/'paired.csv',index=False)
    for (scene,rate),group in df.groupby(['scene','rate']):
        for baseline in ['deficit','equal','ttl1','ttl2','exp_pf']:
            joint=group[group.arm=='mlwdf_pf'].merge(group[group.arm==baseline],on='seed',suffixes=('','_baseline'))
            for subset,seeds in [('41-50',SEEDS),('44-50',range(44,51))]:
                z=joint[joint.seed.isin(seeds)]
                for metric in METRICS:
                    x=(z[metric]-z[metric+'_baseline']).to_numpy();n=len(x)
                    if n<2:continue
                    mean=float(x.mean());half=float(student_t.ppf(.975,n-1)*np.std(x,ddof=1)/np.sqrt(n))
                    base=float(z[metric+'_baseline'].mean())
                    weight_pairs.append(dict(scene=scene,rate=rate,arm='mlwdf_pf',baseline=baseline,subset=subset,metric=metric,n=n,
                        delta=mean,ci_low=mean-half,ci_high=mean+half,relative_change_pct=100*mean/base if base else None))
    weight_frame=pd.DataFrame(weight_pairs,columns=['scene','rate','arm','baseline','subset','metric','n','delta','ci_low','ci_high','relative_change_pct'])
    weight_frame.to_csv(output/'ablation_paired.csv',index=False)
    weight_frame[weight_frame.baseline=='deficit'].to_csv(output/'weight_paired.csv',index=False)
    pd.DataFrame(prefix).to_csv(output/'prefix_validation.csv',index=False)
    pd.DataFrame(cost).to_csv(output/'continuous_cost.csv',index=False)
    physical=[sample for p in output.glob('*/physical_validation.json') for sample in json.loads(p.read_text())['samples']]
    write(output/'completion.json',dict(completed=not partial,points=len(df),prefix_checks=len(prefix),statistical_unit='seed',pool_packets=False,
        physical_states=len(physical),physical_sinr_max_abs_db=max((p['sinr_max_abs_db'] for p in physical),default=0),
        physical_bler_max_abs=max((p['bler_max_abs'] for p in physical),default=0)))

def main(args):
    root=Path(args.source);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    files=sorted(root.glob('*/done.json'))
    if not args.partial:assert len(files)==220
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        fs={pool.submit(analyze_one,str(f),str(out)):f for f in files}
        for f in as_completed(fs):
            row=f.result();print('ANALYZED',row['key'],flush=True)
    summarize(root,out,args.partial)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--partial',action='store_true');main(p.parse_args())

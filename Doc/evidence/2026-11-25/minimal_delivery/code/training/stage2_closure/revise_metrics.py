"""Extend the verified 220 runs for the chapter 6/7 revision; never rerun PHY."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from . import analyze

FIELDS = ['tti','ue_id','offered_bytes','accepted_bytes','rejected_bytes','expired_bytes',
          'previous_post_queue_bytes','pre_queue_bytes','scheduled_dequeue_bytes','post_queue_bytes',
          'successful_bytes','failed_bytes','actual_prg_count','mcs','layer','phy_sinr_db_cpu',
          'predicted_bler_cpu','tb_tx_count','tb_err_count']
ADDED = ['success_pct','outcome_mean_ms','outcome_p95_ms','offered_mbps','served_mbps',
         'ue_payload_mean_mbps','ue_goodput_mean_mbps','queue_mean_mb','queue_start_mb','queue_end_mb',
         'layer_tx_mean','cluster_goodput_spectral_efficiency','worst_cell_goodput_spectral_efficiency']

def prepare(source, target):
    target.mkdir(parents=True, exist_ok=True)
    for name in ['audit','e0','source_documents']:
        if not (target/name).exists():shutil.copytree(source/name,target/name)
    if not (target/'campaign').exists():(target/'campaign').symlink_to((source/'campaign').resolve(),target_is_directory=True)
    for name in ['tests.log','early_native_prefix_check.json','baseline_feasibility.json','execution_priority.json']:
        shutil.copy2(source/name,target/name)
    for path in source.glob('environment_*.json'):shutil.copy2(path,target/path.name)
    out=target/'analysis';out.mkdir(exist_ok=True)
    for original in (source/'analysis').iterdir():
        if not original.is_dir():continue
        dest=out/original.name;dest.mkdir(exist_ok=True)
        for file in original.iterdir():
            if not file.is_file():continue
            if file.name=='cohort.csv.gz':
                if not (dest/file.name).exists():(dest/file.name).symlink_to(file.resolve())
            elif not (dest/file.name).exists():shutil.copy2(file,dest/file.name)

def one(path):
    path=Path(path);summary=json.loads((path/'summary.json').read_text())
    if (path/'revision_validation.json').exists() and json.loads((path/'revision_validation.json').read_text()).get('revision',0)==2:return summary['key']
    cols=['ue_id','arrival_tti','packet_size_bytes','terminal_tti','outcome','system_time_ms','censored','successfully_delivered']
    frame=pd.read_csv(path/'cohort.csv.gz',usecols=cols)
    assert len(frame)==summary['cohort_packets'] and not frame.censored.any()
    assert frame.arrival_tti.between(8001,28000).all()
    assert set(frame.outcome.unique())<={'mac_completed','expired'}
    assert frame.system_time_ms.between(.5,200).all()
    good=frame[frame.successfully_delivered]
    np.testing.assert_allclose([good.system_time_ms.mean(),good.system_time_ms.quantile(.95),frame.system_time_ms.mean()],
        [summary['delay_mean_ms'],summary['delay_p95_ms'],summary.get('terminal_mean_ms',summary.get('outcome_mean_ms'))],rtol=1e-12,atol=1e-10)
    summary['outcome_mean_ms']=float(frame.system_time_ms.mean())
    summary['outcome_p95_ms']=float(frame.system_time_ms.quantile(.95))
    summary.pop('delay_p99_ms',None);summary.pop('terminal_mean_ms',None)
    per=pd.read_csv(path/'per_ue.csv').set_index('ue')
    for name,group in [('success',good),('outcome',frame)]:
        per[name+'_delay_mean_ms']=group.groupby('ue_id').system_time_ms.mean()
        per[name+'_delay_p95_ms']=group.groupby('ue_id').system_time_ms.quantile(.95)
    per.to_csv(path/'per_ue.csv')
    bins=np.arange(.25,200.76,.5)
    success_hist,_=np.histogram(good.system_time_ms,bins)
    outcome_hist,_=np.histogram(frame.system_time_ms,bins)
    assert success_hist.sum()==len(good) and outcome_hist.sum()==len(frame)
    pd.DataFrame(dict(delay_ms=(bins[:-1]+bins[1:])/2,success_mass=success_hist/len(good),
                      outcome_mass=outcome_hist/len(frame))).to_csv(path/'delay_hist.csv',index=False)
    native=analyze.local(summary['native_directory'])
    data=np.memmap(native/'metrics.bin',dtype='<f8',mode='r').reshape(28400,36,len(FIELDS))
    measured=data[8000:28000]
    get=lambda name:measured[:,:,FIELDS.index(name)]
    tx=get('actual_prg_count')>0
    summary.update(offered_mbps=float(get('offered_bytes').sum()*8/10e6),
        served_mbps=float(get('scheduled_dequeue_bytes').sum()*8/10e6),
        ue_payload_mean_mbps=summary['payload_mbps']/36,ue_goodput_mean_mbps=summary['goodput_mbps']/36,
        queue_mean_mb=float(get('post_queue_bytes').sum(axis=1).mean()/1e6),
        queue_start_mb=float(data[8000,:,FIELDS.index('previous_post_queue_bytes')].sum()/1e6),
        queue_end_mb=float(data[27999,:,FIELDS.index('post_queue_bytes')].sum()/1e6),
        layer_tx_mean=float(get('layer')[tx].mean()),
        cluster_goodput_spectral_efficiency=summary['goodput_mbps']/(3*97.92))
    np.testing.assert_allclose(get('successful_bytes').sum()*8/10e6,summary['goodput_mbps'],rtol=1e-12)
    np.testing.assert_allclose(get('scheduled_dequeue_bytes'),get('successful_bytes')+get('failed_bytes'),rtol=0,atol=0)
    # The topology CSV's serving_cell_id is the nominal generation cell. Recover
    # the actual strongest-cell assignment from the verified executed prefix.
    with np.load(path/'prefix.npz') as z:actions=z['actions'].reshape(-1,3,17)
    serving=np.full(36,-1,dtype=int)
    for cell in range(3):
        ids=np.unique(actions[:,cell,:]);ids=ids[ids>=0].astype(int)
        assert np.all(serving[ids]==-1), 'A UE appeared under multiple serving cells'
        serving[ids]=cell
    assert np.all(serving>=0), 'The common prefix must identify all 36 serving cells'
    cells=[]
    for cell in range(3):
        ids=np.flatnonzero(serving==cell)
        active=tx[:,ids];gp=float(get('successful_bytes')[:,ids].sum()*8/10e6)
        cells.append(dict(cell=cell,ue_count=len(ids),goodput_mbps=gp,spectral_efficiency=gp/97.92,
            mcs_tx_mean=float(get('mcs')[:,ids][active].mean()),layer_tx_mean=float(get('layer')[:,ids][active].mean()),
            sinr_tx_mean_db=float(get('phy_sinr_db_cpu')[:,ids][active].mean()),
            occupancy_pct=float(get('actual_prg_count')[:,ids].sum()/(20000*17)*100)))
    np.testing.assert_allclose(sum(c['goodput_mbps'] for c in cells),summary['goodput_mbps'],rtol=1e-12)
    summary['worst_cell_goodput_spectral_efficiency']=min(c['spectral_efficiency'] for c in cells)
    pd.DataFrame(cells).to_csv(path/'per_cell.csv',index=False)
    (path/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    check=dict(passed=True,revision=2,scope='Same mature arrival cohort for success and outcome; no PHY rerun',
        packets=len(frame),success_packets=len(good),expired_packets=int(frame.outcome.eq('expired').sum()),
        other_packets=int((~frame.successfully_delivered&~frame.outcome.eq('expired')).sum()),
        outcome_definition='All MAC-completed packets (including CRC-damaged packets) and naturally expired packets',
        original_success_and_outcome_mean_reproduced=True,existing_goodput_reproduced=True,
        serving_assignment_source='Verified common-PFQ executed action prefix; not nominal topology labels',
        actual_serving_cells=serving.tolist(),
        cohort_sha256=hashlib.sha256((path/'cohort.csv.gz').read_bytes()).hexdigest())
    (path/'revision_validation.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
    return summary['key']

def main(args):
    source=Path(args.source).resolve();root=Path(args.output).resolve();prepare(source,root)
    paths=sorted(p.parent for p in (root/'analysis').glob('*/summary.json'));assert len(paths)==220
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,f in enumerate(as_completed([pool.submit(one,str(p)) for p in paths]),1):
            print('EXTENDED',i,f.result(),flush=True)
    analyze.METRICS=[m for m in analyze.METRICS if m!='delay_p99_ms']+ADDED
    analyze.summarize(root/'campaign',root/'analysis')
    old=pd.read_csv(source/'analysis/per_seed.csv').set_index('key');new=pd.read_csv(root/'analysis/per_seed.csv').set_index('key')
    common=[m for m in analyze.METRICS if m in old.columns]
    np.testing.assert_allclose(new.loc[old.index,common],old[common],rtol=1e-12,atol=1e-10)
    cellrows=[]
    for p in paths:
        s=json.loads((p/'summary.json').read_text());f=pd.read_csv(p/'per_cell.csv')
        for k in ['scene','rate','arm','seed']:f[k]=s[k]
        cellrows.append(f)
    cell=pd.concat(cellrows,ignore_index=True);cell.to_csv(root/'analysis/cell_per_seed.csv',index=False)
    cell.groupby(['scene','rate','arm','cell'],as_index=False).agg({k:'mean' for k in ['ue_count','goodput_mbps','spectral_efficiency','mcs_tx_mean','layer_tx_mean','sinr_tx_mean_db','occupancy_pct']}).to_csv(root/'analysis/cell_aggregate.csv',index=False)
    pairs=[]
    df=new.reset_index()
    for (scene,rate),g in df.groupby(['scene','rate']):
        for baseline in ['rr','rrq','pf']:
            z=g[g.arm=='pfq'].merge(g[g.arm==baseline],on='seed',suffixes=('','_base'))
            if not len(z):continue
            assert len(z)==10
            for metric in analyze.METRICS:
                x=z[metric]-z[metric+'_base'];d=float(x.mean());h=float(student_t.ppf(.975,9)*x.std(ddof=1)/np.sqrt(10))
                pairs.append(dict(scene=scene,rate=rate,arm='pfq',baseline=baseline,metric=metric,n=10,delta=d,ci_low=d-h,ci_high=d+h))
    pd.DataFrame(pairs).to_csv(root/'analysis/baseline_paired.csv',index=False)
    (root/'指标补充核验.json').write_text(json.dumps(dict(passed=True,points=220,existing_metrics_unchanged=common,
        new_metrics=ADDED,report_delay_quantiles=[.95],previous_release=str(source),simulation_rerun=False),ensure_ascii=False,indent=2)+'\n')
    print('REVISION_METRICS_READY',root,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4);main(p.parse_args())

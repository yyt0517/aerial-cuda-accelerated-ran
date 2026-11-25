"""Whole-packet metrics, physical execution audits and paired input checks."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from training.stageb_ppo.native import ROOT, write, sha, packets, EXPECTED_SIMULATOR
from training.stage2_closure.revise_metrics import FIELDS
from training.stageb_lns.archive import enable_legacy_types
from training.stageb_lns.no_ctbs_inputs import problem_from_raw_row
from training.stageb_lns.physics import ReferencePhy
from .model import approximate_problem

METRICS=('payload_mbps','goodput_mbps','delay_mean_ms','delay_p95_ms','delay_p99_ms',
         'outcome_p95_ms','expiry_pct','success_pct','other_nondelivery_pct','queue_mean_mb')


def local(path):
    return Path(str(path).replace('/opt/nvidia/cuBB',str(ROOT)))


def physical_audit(directory, spec, data):
    import torch
    enable_legacy_types(); results=[]
    for sample in json.loads((directory/'saved_states.json').read_text()):
        path=local(sample['path']); assert sha(path)==sample['sha256']
        row=torch.load(path,map_location='cpu',weights_only=False)
        p=problem_from_raw_row(row)
        p.weights=np.asarray(row['objective_weights'])
        q=approximate_problem(p,row['rsrp_report']) if spec['arm'].startswith('rsrp_') else p
        original=p.h.copy();action=row['selected_action']
        predicted=ReferencePhy(q).evaluate(action)[0]
        actual_model=ReferencePhy(p).evaluate(action)[0]
        np.testing.assert_allclose(row['selected_values'],predicted,rtol=1e-8,atol=1e-6)
        np.testing.assert_array_equal(p.h,original)
        actual=data[int(row['tti'])-1];active=actual_model[:,0]>0
        for field,column in [(0,'actual_prg_count'),(1,'layer'),(3,'scheduled_dequeue_bytes')]:
            np.testing.assert_array_equal(actual_model[active,field],actual[active,FIELDS.index(column)])
        sinr_error=np.abs(actual_model[active,4]-actual[active,FIELDS.index('phy_sinr_db_cpu')])
        bler_error=np.abs(actual_model[active,5]-actual[active,FIELDS.index('predicted_bler_cpu')])
        assert sinr_error.max(initial=0)<.005 and bler_error.max(initial=0)<.0005
        error=predicted[active,4]-actual_model[active,4]
        results.append(dict(tti=int(row['tti']),native_sinr_max_error_db=float(sinr_error.max(initial=0)),
            native_bler_max_error=float(bler_error.max(initial=0)),
            approximation_sinr_errors_db=error.tolist(),
            approximation_expected_bytes_error=float((predicted[:,6]-actual_model[:,6]).sum()),
            passed=True))
    assert len(results)==3
    return results


def one(done, output):
    spec=json.loads(Path(done).read_text());d=local(spec['directory']);out=Path(output)/spec['key']
    out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():return json.loads((out/'summary.json').read_text())
    if spec['reused']:
        for name,digest in spec['hashes'].items():assert sha(d/name)==digest,name
    launch=json.loads((d/'launch.json').read_text())
    assert launch['simulator_sha256']==EXPECTED_SIMULATOR
    assert json.loads((d/'validation.json').read_text())['passed']
    assert float(launch['env']['CUMAC_TRAFFIC_ARRIVAL_RATE'])==spec['rate']
    first,last=int(launch['measurement_first_tti']),int(launch['measurement_last_tti'])
    horizon=int(launch['horizon']);warmup=int(launch['common_pfq_warmup'])
    seconds=(last-first+1)*.0005
    data=np.memmap(d/'metrics.bin',dtype='<f8',mode='r').reshape(horizon,36,len(FIELDS))
    checks=physical_audit(d,spec,data)
    write(out/'physical_audit.json',dict(passed=True,samples=checks))
    if spec['reused']:
        old=ROOT/'deliverables/stage2_closure_20261104/analysis'/spec['key']
        assert json.loads((old/'validation.json').read_text())['passed']
        cohort=pd.read_csv(old/'cohort.csv.gz')
        # Existing cohorts are checked against the unchanged native artifact hashes.
        cohort_source=dict(path=str(old/'cohort.csv.gz'),sha256=sha(old/'cohort.csv.gz'))
    else:
        _,pk=packets(d)
        cohort=pk[pk.arrival_tti.between(first,last)].copy()
        cohort.to_csv(out/'cohort.csv.gz',index=False)
        cohort_source=dict(path=str(out/'cohort.csv.gz'),sha256=sha(out/'cohort.csv.gz'))
    assert len(cohort)>0 and not cohort.censored.any()
    assert cohort.arrival_tti.between(first,last).all()
    good=cohort[cohort.successfully_delivered]
    measured=data[first-1:last]
    get=lambda name: measured[:,:,FIELDS.index(name)]
    np.testing.assert_array_equal(get('scheduled_dequeue_bytes'),get('successful_bytes')+get('failed_bytes'))
    successful=float(good.packet_size_bytes.sum()*8/seconds/1e6)
    summary=dict(**{k:spec[k] for k in ('scene','rate','arm','seed','key','reused')},
        payload_mbps=successful,goodput_mbps=float(get('successful_bytes').sum()*8/seconds/1e6),
        delay_mean_ms=float(good.system_time_ms.mean()),delay_p95_ms=float(good.system_time_ms.quantile(.95)),
        delay_p99_ms=float(good.system_time_ms.quantile(.99)),outcome_p95_ms=float(cohort.system_time_ms.quantile(.95)),
        expiry_pct=float(100*cohort.outcome.eq('expired').mean()),success_pct=float(100*cohort.successfully_delivered.mean()),
        other_nondelivery_pct=float(100*((~cohort.successfully_delivered)&(~cohort.outcome.eq('expired'))).mean()),
        queue_mean_mb=float(get('post_queue_bytes').sum(axis=1).mean()/1e6),
        cohort_packets=len(cohort),measurement_seconds=seconds,native_directory=str(d),
        topology_sha256=sha(d/'ue_topology.csv'),traffic_sha256=sha(d/'traffic_offered_packets.csv'),
        cohort_source=cohort_source)
    costs=pd.read_csv(d/'search.csv.gz');costs=costs[costs.tti.between(first,last)]
    assert len(costs)==last-first+1
    for col in ['wall_ms','cpu_ms','measurement_emulator_ms','candidate_evaluations','ue_evaluations','mmse_misses','rounds']:
        if col in costs:
            summary[col+'_mean']=float(costs[col].mean())
            summary[col+'_p95']=float(costs[col].quantile(.95))
    if spec['reused']:
        old_summary=json.loads((old/'summary.json').read_text())
        for col in ['payload_mbps','goodput_mbps','delay_p95_ms','expiry_pct']:
            np.testing.assert_allclose(summary[col],old_summary[col],rtol=1e-12,atol=1e-9)
    errors=np.concatenate([x['approximation_sinr_errors_db'] for x in checks])
    summary['sample_sinr_bias_db']=float(errors.mean())
    summary['sample_sinr_mae_db']=float(np.abs(errors).mean())
    with np.load(d/'actions.npz') as z:
        np.savez_compressed(out/'prefix.npz',actions=z['actions'][:warmup],input_hashes=z['input_hashes'][:warmup])
    write(out/'validation.json',dict(passed=True,native_original_channel_verified=True,
          no_censored_packets=True,whole_packet_metrics=True,warmup=warmup))
    write(out/'summary.json',summary)
    return summary


def paired_inputs(df, output):
    import torch
    enable_legacy_types();checks=[]
    for (scene,seed),group in df.groupby(['scene','seed']):
        base=group[group.arm=='deficit'].iloc[0]
        for _,row in group.iterrows():
            assert row.topology_sha256==base.topology_sha256
            assert row.traffic_sha256==base.traffic_sha256
            with np.load(output/row.key/'prefix.npz') as a,np.load(output/base.key/'prefix.npz') as b:
                for name in ('actions','input_hashes'):np.testing.assert_array_equal(a[name],b[name])
            d=Path(row.native_directory);bd=Path(base.native_directory)
            for sample in sorted((d/'states').glob('*.pt')):
                current=torch.load(sample,map_location='cpu',weights_only=False)
                reference=torch.load(bd/'states'/sample.name,map_location='cpu',weights_only=False)
                for name in ('h','w'):
                    np.testing.assert_array_equal(current['snapshot'][name],reference['snapshot'][name])
            checks.append(dict(scene=scene,seed=int(seed),arm=row.arm,passed=True,
                               topology=True,traffic=True,warmup=True,saved_physical_channels=True))
        approximate=group[group.arm.str.startswith('rsrp_')]
        if len(approximate)==2:
            a,b=[Path(x) for x in approximate.native_directory]
            np.testing.assert_array_equal(np.load(a/'channel_hashes.npy'),np.load(b/'channel_hashes.npy'))
            np.testing.assert_array_equal(np.load(a/'report_hashes.npy'),np.load(b/'report_hashes.npy'))
    write(output/'pairing.json',dict(passed=True,checks=checks))


def summarize(output):
    output=Path(output)
    df=pd.DataFrame([json.loads(p.read_text()) for p in output.glob('*/summary.json')])
    df=df.sort_values(['scene','seed','arm']);df.to_csv(output/'per_seed.csv',index=False)
    paired_inputs(df,output)
    aggregate=[];pairs=[]
    for (scene,arm),g in df.groupby(['scene','arm']):
        for metric in METRICS:
            aggregate.append(dict(scene=scene,arm=arm,metric=metric,n=len(g),mean=float(g[metric].mean()),
                                  std=float(g[metric].std(ddof=1)) if len(g)>1 else 0.))
        if not arm.startswith('rsrp_'):continue
        for control in ('deficit','mlwdf_pf'):
            b=df[(df.scene==scene)&(df.arm==control)]
            joined=g.merge(b,on='seed',suffixes=('','_base'),validate='one_to_one')
            for metric in METRICS:
                delta=joined[metric]-joined[metric+'_base'];n=len(delta)
                half=float(student_t.ppf(.975,n-1)*delta.std(ddof=1)/np.sqrt(n)) if n>1 else None
                mean=float(delta.mean())
                pairs.append(dict(scene=scene,arm=arm,control=control,metric=metric,n=n,mean_delta=mean,
                    ci_low=mean-half if half is not None else None,ci_high=mean+half if half is not None else None))
    pd.DataFrame(aggregate).to_csv(output/'aggregate.csv',index=False)
    pd.DataFrame(pairs).to_csv(output/'paired.csv',index=False)
    write(output/'completion.json',dict(completed=True,runs=len(df),passed=True,
          unit='equal-weight seed; paired nominal 95% t intervals; exploratory study',
          seeds=sorted(int(x) for x in df.seed.unique())))


def main(args):
    source=Path(args.source);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    completion=json.loads((source/'completion.json').read_text());assert completion['completed']
    done=sorted(source.glob('*/done.json'));assert len(done)==completion['total']
    with ProcessPoolExecutor(args.workers) as pool:
        futures={pool.submit(one,p,output):p for p in done}
        for future in as_completed(futures):print('ANALYZED',future.result()['key'],flush=True)
    summarize(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=3);main(parser.parse_args())

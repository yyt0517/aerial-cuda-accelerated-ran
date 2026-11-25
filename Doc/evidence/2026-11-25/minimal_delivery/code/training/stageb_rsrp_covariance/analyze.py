"""Packet-cohort analysis derived from the validated prior ablation analyzer."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from .model import from_row, STAGES, SPATIAL_STAGES, setting, feedback_rows

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
        q=from_row(p,spec['stage'],row)
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
        old=local(spec['prior_analysis'])
        assert json.loads((old/'validation.json').read_text())['passed']
        old_summary=json.loads((old/'summary.json').read_text())
        old_cohort=local(old_summary['cohort_source']['path'])
        assert sha(old_cohort)==old_summary['cohort_source']['sha256']
        cohort=pd.read_csv(old_cohort)
        # Existing cohorts are checked against the unchanged native artifact hashes.
        cohort_source=dict(path=str(old_cohort),sha256=sha(old_cohort))
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
    summary=dict(**{k:spec[k] for k in ('scene','rate','stage','objective','seed','key','reused')},
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
    count_columns=('rounds','accepted_moves','generated_edits','regenerated_prgs','scored_edits',
        'sum_neighborhood_sizes','first_scan_evaluations','initialization_evaluations','bound_updates',
        'heap_pushes','certificate_heap_pruned','known_exact_reuses','floor_pruned','threshold_pruned',
        'candidate_evaluations','ue_evaluations','mmse_requests','mmse_misses')
    np.testing.assert_array_equal(costs.sum_neighborhood_sizes,
        costs[['scored_edits','known_exact_reuses','floor_pruned','threshold_pruned','certificate_heap_pruned']].sum(axis=1))
    np.testing.assert_array_equal(costs.candidate_evaluations,costs.scored_edits+costs.initialization_evaluations)
    np.testing.assert_array_equal(costs.heap_pushes,costs.scored_edits+costs.certificate_heap_pruned)
    for col in (*count_columns,'wall_ms','cpu_ms','measurement_emulator_ms'):
        if col in costs:
            summary[col+'_mean']=float(costs[col].mean())
            summary[col+'_p95']=float(costs[col].quantile(.95))
    total=costs.sum_neighborhood_sizes.sum()
    for col in ('scored_edits','known_exact_reuses','floor_pruned','threshold_pruned','certificate_heap_pruned'):
        summary[col+'_fraction']=float(costs[col].sum()/total) if total else 0.
    summary['neighborhood_per_round']=float(total/costs['rounds'].sum())
    summary['mmse_hit_fraction']=float(1-costs.mmse_misses.sum()/costs.mmse_requests.sum())
    if spec['reused']:
        old_summary=json.loads((old/'summary.json').read_text())
        for col in ['payload_mbps','goodput_mbps','delay_p95_ms','expiry_pct']:
            np.testing.assert_allclose(summary[col],old_summary[col],rtol=1e-12,atol=1e-9)
    errors=np.concatenate([x['approximation_sinr_errors_db'] for x in checks])
    summary['sample_sinr_bias_db']=float(errors.mean())
    summary['sample_sinr_mae_db']=float(np.abs(errors).mean())
    summary.update(extended_metrics(d, out, spec, cohort, measured, first, last, costs))
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
        base=group[(group.stage=='exact')&(group.objective=='deficit')].iloc[0]
        hashed=group[[ (Path(d)/'channel_hashes.npy').exists() for d in group.native_directory]]
        reference_hashes=Path(hashed.iloc[0].native_directory)
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
            full=(d/'channel_hashes.npy').exists()
            if full:
                for name in ('channel_hashes.npy','report_hashes.npy'):
                    np.testing.assert_array_equal(np.load(d/name),np.load(reference_hashes/name))
            checks.append(dict(scene=scene,seed=int(seed),stage=row.stage,objective=row.objective,passed=True,
                topology=True,traffic=True,warmup=True,saved_physical_channels=True,full_channel_stream=full))
    write(output/'pairing.json',dict(passed=True,checks=checks))


def extended_metrics(directory, output, spec, cohort, measured, first, last, costs):
    get = lambda name: measured[:, :, FIELDS.index(name)]
    tx, err = get('tb_tx_count'), get('tb_err_count')
    active = tx > 0
    prediction = None
    if (directory/'selected_predictions.npz').exists():
        with np.load(directory/'selected_predictions.npz') as z:
            indices = np.searchsorted(z['tti'], np.arange(first, last+1))
            np.testing.assert_array_equal(z['tti'][indices], np.arange(first, last+1))
            prediction = z['sinr_db'][indices]-get('phy_sinr_db_cpu')
    per_ue = []
    for u in range(36):
        packets = cohort[cohort.ue_id == u]
        good = packets[packets.successfully_delivered]
        n = tx[:, u].sum(); e = err[:, u].sum(); selected = active[:, u]
        record = dict(key=spec['key'], scene=spec['scene'], seed=spec['seed'], objective=spec['objective'],
            stage=spec['stage'], ue_id=u, tb_tx_count=int(n), tb_err_count=int(e),
            actual_bler=e/n if n else np.nan, native_mcs_mean=get('mcs')[selected, u].mean() if n else np.nan,
            expired_packets=int(packets.outcome.eq('expired').sum()), expired_pct=100*packets.outcome.eq('expired').mean(),
            whole_packet_mbps=good.packet_size_bytes.sum()*8/((last-first+1)*.0005)/1e6,
            delay_p95_ms=good.system_time_ms.quantile(.95), delay_p99_ms=good.system_time_ms.quantile(.99))
        if prediction is not None:
            errors = prediction[selected, u]
            record.update(sinr_samples=len(errors), sinr_bias_db=errors.mean() if len(errors) else np.nan,
                sinr_var_db2=errors.var(ddof=1) if len(errors)>1 else np.nan,
                sinr_mae_db=np.abs(errors).mean() if len(errors) else np.nan)
        per_ue.append(record)
    pd.DataFrame(per_ue).to_csv(output/'per_ue.csv', index=False)
    summary = dict(actual_tb_bler=float(err.sum()/tx.sum()),
                   native_mcs_mean=float(get('mcs')[active].mean()), full_window_prediction=prediction is not None)
    if prediction is not None:
        errors = prediction[active]
        summary.update(full_sinr_bias_db=float(errors.mean()), full_sinr_var_db2=float(errors.var(ddof=1)),
                       full_sinr_mae_db=float(np.abs(errors).mean()), full_sinr_samples=int(len(errors)))
    if spec['stage'] in SPATIAL_STAGES:
        granularity, period = setting(spec['stage'])
        assert costs.covariance_age_tti.between(1, period).all()
        np.testing.assert_array_equal(costs.covariance_refresh, ((costs.tti-2)%period==0).astype(int))
        summary.update(granularity=granularity, period_ms=period*.5,
            covariance_updates=int(costs.covariance_refresh.sum()),
            covariance_age_mean_ms=float(costs.covariance_age_tti.mean()*.5),
            covariance_decode_cpu_ms_mean=float(costs.covariance_decode_cpu_ms.mean()))
    return summary


def spatial_pairing(df, output):
    import torch
    enable_legacy_types()
    checks=[]
    for (scene, seed), group in df.groupby(['scene','seed']):
        witness=group[(group.stage=='prg_t1') & (group.objective=='deficit')]
        if len(witness)!=1:continue
        witness=Path(witness.iloc[0].native_directory)
        for row in group[group.stage.isin(SPATIAL_STAGES)].itertuples():
            for path in (Path(row.native_directory)/'states').glob('*.pt'):
                a=torch.load(path,map_location='cpu',weights_only=False)
                b=torch.load(witness/'states'/path.name,map_location='cpu',weights_only=False)
                x,y=a['spatial_reports'][row.stage],b['spatial_reports'][row.stage]
                assert x['sample_tti']==y['sample_tti']
                np.testing.assert_array_equal(x['packet'],y['packet'])
            checks.append(dict(key=row.key,shared_exogenous_spatial_reports_exact=True))
        for stage,g in group[group.stage.isin(SPATIAL_STAGES)].groupby('stage'):
            arrays=[np.load(Path(d)/'covariance_hashes.npy') for d in g.native_directory]
            for a in arrays[1:]:np.testing.assert_array_equal(a,arrays[0])
    write(output/'spatial_pairing.json',dict(passed=True,checks=checks,
        all_spatial_streams_identical_between_objectives=True))


def summarize(output):
    output=Path(output)
    df=pd.DataFrame([json.loads(p.read_text()) for p in output.glob('*/summary.json')])
    df=df.sort_values(['scene','objective','stage','seed']);df.to_csv(output/'per_seed.csv',index=False)
    paired_inputs(df,output);spatial_pairing(df,output)
    metrics=list(METRICS)+['actual_tb_bler','native_mcs_mean','scored_edits_mean','rounds_mean',
        'sum_neighborhood_sizes_mean','scored_edits_fraction','mmse_misses_mean']
    aggregate=df.groupby(['scene','objective','stage'],as_index=False)[metrics].mean()
    aggregate.to_csv(output/'means.csv',index=False)
    pairs=[];recoveries=[]
    for (scene,objective),g in df.groupby(['scene','objective']):
        exact=g[g.stage=='exact'].set_index('seed');scalar=g[g.stage=='reported'].set_index('seed')
        assert len(exact)==len(scalar)>0
        for stage,part in g.groupby('stage'):
            if stage=='exact':continue
            for control in ('exact','reported'):
                base=g[g.stage==control]
                joined=part.merge(base,on='seed',suffixes=('','_base'),validate='one_to_one')
                for metric in metrics:
                    delta=joined[metric]-joined[metric+'_base'];n=len(delta);mean=float(delta.mean())
                    half=float(student_t.ppf(.975,n-1)*delta.std(ddof=1)/np.sqrt(n)) if n>1 else None
                    pairs.append(dict(scene=scene,objective=objective,stage=stage,control=control,metric=metric,
                        n=n,mean_delta=mean,ci_low=mean-half if half is not None else None,
                        ci_high=mean+half if half is not None else None))
            if stage not in SPATIAL_STAGES:continue
            hybrid=part.set_index('seed')
            for metric in ('payload_mbps','delay_p95_ms','delay_p99_ms','expiry_pct'):
                sign=1 if metric=='payload_mbps' else -1
                original_gap=sign*(exact[metric]-scalar[metric]);recovered=sign*(hybrid[metric]-scalar[metric])
                denom=float(original_gap.mean())
                recovery=100*float(recovered.mean())/denom if abs(denom)>1e-12 else None
                recoveries.append(dict(scene=scene,objective=objective,stage=stage,metric=metric,
                    original_mean=float(exact[metric].mean()),scalar_mean=float(scalar[metric].mean()),
                    hybrid_mean=float(hybrid[metric].mean()),gap_recovered_pct=recovery,
                    original_gap=denom,improvement_over_scalar=float(recovered.mean())))
    pd.DataFrame(pairs).to_csv(output/'paired.csv',index=False)
    pd.DataFrame(recoveries).to_csv(output/'recovery.csv',index=False)
    pd.DataFrame(feedback_rows()).to_csv(output/'feedback.csv',index=False)
    per_ue=pd.concat([pd.read_csv(p) for p in output.glob('*/per_ue.csv')],ignore_index=True)
    assert len(per_ue)==len(df)*36
    per_ue.to_csv(output/'per_ue.csv',index=False)
    write(output/'completion.json',dict(completed=True,passed=True,runs=len(df),
        seeds=sorted(int(s) for s in df.seed.unique()),
        recovery='ratio of equal-seed mean improvements to original-minus-scalar gap; not clipped; zero denominator undefined',
        intervals='nominal paired 95% t; exploratory n=3; no multiplicity adjustment',
        feedback='neighbor-link payload only; 15 real shape parameters; 16/32-bit rates hypothetical, PHY unquantized'))


def main(args):
    source=Path(args.source);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    completion=json.loads((source/'completion.json').read_text());assert completion['completed']
    done=sorted(source.glob('*/done.json'));assert len(done)==completion['total']
    with ProcessPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(one,p,output) for p in done]):
            print('ANALYZED',future.result()['key'],flush=True)
    summarize(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=4);main(parser.parse_args())


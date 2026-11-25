"""Full-window per-UE CRC/OLLA/expiry plus sampled signed SINR errors.

OLLA is reconstructed using native float32 updates on each scheduled TTI and
the last transmitted CRC (carried across unscheduled TTIs). Every archived
pre-decision OLLA vector and prior-CRC vector must match exactly.
"""
import argparse
import gzip
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import numpy as np
import pandas as pd
from training.stageb_ppo.native import write,sha
from training.stage2_closure.revise_metrics import FIELDS
from training.stageb_lns.archive import enable_legacy_types
from training.stageb_lns.no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from training.stageb_lns.physics import NativePhy
from .analyze import local
from .model import from_row


def one(summary,output):
    import torch
    torch.set_num_threads(1);enable_legacy_types()
    out=Path(output)/summary['key'];out.mkdir(parents=True,exist_ok=True)
    if (out/'validation.json').exists():return summary['key']
    d=Path(summary['native_directory']);launch=json.loads((d/'launch.json').read_text())
    first,last=launch['measurement_first_tti'],launch['measurement_last_tti'];horizon=launch['horizon']
    data=np.memmap(d/'metrics.bin',dtype='<f8',mode='r').reshape(horizon,36,len(FIELDS))
    measured=data[first-1:last];get=lambda name:measured[:,:,FIELDS.index(name)]
    samples={};errors=[]
    identity={k:summary[k] for k in ('scene','seed','objective','stage','key')}
    for sample in json.loads((d/'saved_states.json').read_text()):
        path=local(sample['path']);assert sha(path)==sample['sha256']
        row=torch.load(path,map_location='cpu',weights_only=False);samples[int(row['tti'])]=row
        p=problem_from_raw_row(row);p.weights=np.asarray(row['objective_weights']);q=from_row(p,summary['stage'],row)
        actions=np.stack([row['selected_action'],raw_pfq_draft(p,row['obs'])])
        truth=NativePhy(p);approx=NativePhy(q)
        try:actual=truth.evaluate(actions).values;predicted=approx.evaluate(actions).values
        finally:truth.close();approx.close()
        np.testing.assert_array_equal(predicted[0],row['selected_values'])
        np.testing.assert_array_equal(q.mcs,p.mcs)
        for j,action_name in enumerate(('selected','raw_pfq')):
            for u in np.flatnonzero(actual[j,:,0]>0):
                errors.append(dict(**identity,tti=int(row['tti']),ue_id=int(u),action=action_name,
                    sinr_estimate_db=float(predicted[j,u,4]),sinr_original_db=float(actual[j,u,4]),
                    sinr_error_db=float(predicted[j,u,4]-actual[j,u,4]),
                    bler_estimate=float(predicted[j,u,5]),bler_original=float(actual[j,u,5]),
                    bler_error=float(predicted[j,u,5]-actual[j,u,5]),native_mcs=int(p.mcs[u]),
                    pre_olla_delta_db=float(row['raw_phy']['olla_delta'][u])))
    # Exact reconstruction, including updates after the policy selected an action.
    raw=next(iter(samples.values()))['raw_phy']
    up=np.asarray(raw['olla_delta_up'],np.float32);down=np.asarray(raw['olla_delta_down'],np.float32)
    delta=np.zeros(36,np.float32);prior=np.full(36,-1,np.int64)
    pre=np.empty((last-first+1,36),np.float32);post=np.empty_like(pre)
    checks=[]
    for t,a in enumerate(data,1):
        if t in samples:
            v=samples[t]['raw_phy']
            for name,array in [('olla_delta',delta),('prior_tb_err',prior),('olla_delta_up',up),('olla_delta_down',down)]:
                np.testing.assert_array_equal(array,v[name])
            checks.append(dict(tti=t,all_36_offsets_and_prior_crc_exact=True))
        if first<=t<=last:pre[t-first]=delta
        active=a[:,FIELDS.index('actual_prg_count')]>0
        mask=active&(prior==1);delta[mask]+=up[mask]
        mask=active&(prior==0);delta[mask]-=down[mask]
        if first<=t<=last:post[t-first]=delta
        tx=a[:,FIELDS.index('tb_tx_count')]>0
        assert np.all(a[tx,FIELDS.index('tb_tx_count')]==1)
        prior[tx]=a[tx,FIELDS.index('tb_err_count')].astype(np.int64)
    assert len(checks)==3
    tx=get('tb_tx_count');err=get('tb_err_count');active=tx>0
    path=local(summary['cohort_source']['path']);assert sha(path)==summary['cohort_source']['sha256']
    cohort=pd.read_csv(path);ue=[]
    for u in range(36):
        packets=cohort[cohort.ue_id==u];good=packets[packets.successfully_delivered]
        n=float(tx[:,u].sum());e=float(err[:,u].sum());ok=active[:,u]
        row=dict(**identity,ue_id=u,tb_tx_count=int(n),tb_err_count=int(e),
            actual_tb_bler=e/n if n else np.nan,native_predicted_bler_mean=float(get('predicted_bler_cpu')[ok,u].mean()) if n else np.nan,
            scheduled_tti_pct=float(100*ok.mean()),native_mcs_mean=float(get('mcs')[ok,u].mean()) if n else np.nan,
            native_mcs_p95=float(np.quantile(get('mcs')[ok,u],.95)) if n else np.nan,
            successful_tb_payload_mean_bytes=float(get('successful_bytes')[:,u].sum()/(n-e)) if n>e else np.nan,
            failed_bytes_fraction=float(get('failed_bytes')[:,u].sum()/get('scheduled_dequeue_bytes')[:,u].sum()) if n else np.nan,
            queue_mean_bytes=float(get('post_queue_bytes')[:,u].mean()),
            cohort_packets=len(packets),expired_packets=int(packets.outcome.eq('expired').sum()),
            expired_pct=float(100*packets.outcome.eq('expired').mean()),
            crc_damaged_mac_completed_packets=int(((packets.outcome=='mac_completed')&(~packets.successfully_delivered)).sum()),
            whole_packet_mbps=float(good.packet_size_bytes.sum()*8/summary['measurement_seconds']/1e6),
            successful_packet_p95_ms=float(good.system_time_ms.quantile(.95)),
            successful_packet_p99_ms=float(good.system_time_ms.quantile(.99)))
        for name,values in [('pre_olla_all_tti',pre[:,u]),('used_olla_scheduled',post[ok,u])]:
            row[name+'_mean_db']=float(values.mean()) if len(values) else np.nan
            row[name+'_var_db2']=float(values.astype(np.float64).var()) if len(values) else np.nan
            for quant in (.05,.5,.95):row[name+f'_p{int(quant*100):02d}_db']=float(np.quantile(values,quant)) if len(values) else np.nan
        ue.append(row)
    pd.DataFrame(ue).to_csv(out/'per_ue.csv',index=False)
    errors=pd.DataFrame(errors);errors.to_csv(out/'sinr_samples.csv',index=False)
    # Save actual complete OLLA distributions without creating millions of text rows.
    np.savez_compressed(out/'olla_window.npz',pre_delta_db=pre,used_delta_db=post,scheduled=active,first_tti=first)
    costs=pd.read_csv(d/'search.csv.gz');costs=costs[costs.tti.between(first,last)]
    cols=['tti','rounds','generated_edits','sum_neighborhood_sizes','scored_edits','candidate_evaluations',
          'first_scan_evaluations','known_exact_reuses','floor_pruned','threshold_pruned','certificate_heap_pruned']
    cost=costs[cols].copy()
    cost['floor_eligible_occurrences']=cost.sum_neighborhood_sizes-cost.floor_pruned
    cost['fresh_scored_per_candidate_occurrence']=cost.scored_edits/cost.sum_neighborhood_sizes
    cost['fresh_scored_per_floor_eligible']=cost.scored_edits/cost.floor_eligible_occurrences
    cost.to_csv(out/'per_tti_candidates.csv.gz',index=False)
    # The full-window BLER is TB-weighted, rather than a mean of UE ratios.
    value=dict(**identity,actual_tb_bler=float(err.sum()/tx.sum()),
        native_predicted_bler_mean=float(get('predicted_bler_cpu')[active].mean()),
        native_mcs_mean=float(get('mcs')[active].mean()),
        successful_tb_payload_mean_bytes=float(get('successful_bytes').sum()/(tx-err).sum()),
        used_olla_mean_db=float(post[active].mean()),used_olla_p05_db=float(np.quantile(post[active],.05)),
        used_olla_p50_db=float(np.quantile(post[active],.5)),used_olla_p95_db=float(np.quantile(post[active],.95)),
        expired_packets=int(cohort.outcome.eq('expired').sum()),
        crc_damaged_mac_completed_packets=int(((cohort.outcome=='mac_completed')&(~cohort.successfully_delivered)).sum()))
    for label,part in errors.groupby('action'):
        value[label+'_sinr_error_n']=len(part);value[label+'_sinr_error_mean_db']=float(part.sinr_error_db.mean())
        value[label+'_sinr_error_var_db2']=float(part.sinr_error_db.var(ddof=0))
        value[label+'_optimistic_fraction']=float((part.sinr_error_db>0).mean())
        value[label+'_bler_error_mean']=float(part.bler_error.mean())
    write(out/'summary.json',value)
    write(out/'validation.json',dict(passed=True,olla_checks=checks,olla_samples=int(pre.size),
        full_window_bler_expiry_mcs_olla=True,sinr_scope='three inherited saved states only; counts and signed errors retained',
        mcs_unchanged_by_model_on_same_state=True,
        candidate_scope='all measured TTIs; occurrences across rounds are distinct from unique generated edit entries'))
    return summary['key']


def main(args):
    root=Path(args.root);output=root/'ue_diagnostics';output.mkdir(exist_ok=True)
    summaries=[json.loads(p.read_text()) for p in (root/'analysis').glob('*/summary.json')]
    assert len(summaries)==72
    with ProcessPoolExecutor(args.workers) as pool:
        for f in as_completed([pool.submit(one,s,output) for s in summaries]):print('UE_DIAGNOSTICS',f.result(),flush=True)
    per_ue=pd.concat([pd.read_csv(p) for p in output.glob('*/per_ue.csv')],ignore_index=True)
    assert len(per_ue)==72*36 and not per_ue.duplicated(['key','ue_id']).any()
    assert set(per_ue.key)=={s['key'] for s in summaries}
    per_ue.to_csv(output/'per_ue.csv',index=False)
    samples=pd.concat([pd.read_csv(p) for p in output.glob('*/sinr_samples.csv')],ignore_index=True)
    samples.to_csv(output/'sinr_samples.csv',index=False)
    sample_stats=samples.groupby(['scene','seed','objective','stage','ue_id','action']).agg(
        n=('sinr_error_db','size'),sinr_error_mean_db=('sinr_error_db','mean'),
        sinr_error_var_db2=('sinr_error_db',lambda x:x.var(ddof=1)),bler_error_mean=('bler_error','mean')).reset_index()
    index=per_ue[['scene','seed','objective','stage','ue_id']].merge(pd.DataFrame({'action':['selected','raw_pfq']}),how='cross')
    sample_stats=index.merge(sample_stats,how='left',validate='one_to_one')
    assert len(sample_stats)==72*36*2
    sample_stats['n']=sample_stats.n.fillna(0).astype(int)
    sample_stats.to_csv(output/'sinr_per_ue.csv',index=False)
    summary=pd.DataFrame([json.loads(p.read_text()) for p in output.glob('*/summary.json')]);summary.to_csv(output/'per_seed.csv',index=False)
    exclude=['scene','seed','objective','stage','key'];cols=[c for c in summary if c not in exclude]
    summary.groupby(['scene','objective','stage'],as_index=False)[cols].mean().to_csv(output/'summary.csv',index=False)
    saturation=[]
    for s in summaries:
        d=Path(s['native_directory']);launch=json.loads((d/'launch.json').read_text())
        first,last=launch['measurement_first_tti'],launch['measurement_last_tti']
        data=np.memmap(d/'metrics.bin',dtype='<f8',mode='r').reshape(-1,36,len(FIELDS))[first-1:last]
        np.testing.assert_array_equal(data[:,:,FIELDS.index('pre_queue_bytes')]-data[:,:,FIELDS.index('scheduled_dequeue_bytes')],
                                      data[:,:,FIELDS.index('post_queue_bytes')])
        np.testing.assert_array_equal(data[:,:,FIELDS.index('scheduled_dequeue_bytes')],
            data[:,:,FIELDS.index('successful_bytes')]+data[:,:,FIELDS.index('failed_bytes')])
        with np.load(output/s['key']/'olla_window.npz') as z:used=z['used_delta_db'];active=z['scheduled']
        for u in range(36):
            values=used[active[:,u],u];mcs=data[active[:,u],u,FIELDS.index('mcs')]
            saturation.append(dict(**{k:s[k] for k in ('scene','seed','objective','stage','key')},ue_id=u,
                scheduled_samples=len(values),mcs27_fraction=float((mcs==27).mean()) if len(mcs) else np.nan,
                used_olla_below_minus20_fraction=float((values < -20).mean()) if len(values) else np.nan,
                used_olla_min_db=float(values.min()) if len(values) else np.nan,
                used_olla_max_db=float(values.max()) if len(values) else np.nan))
    pd.DataFrame(saturation).to_csv(output/'olla_saturation_per_ue.csv',index=False)
    first_file=True
    with gzip.open(output/'per_tti_candidates.csv.gz','wt',compresslevel=3) as handle:
        for s in summaries:
            cost=pd.read_csv(output/s['key']/'per_tti_candidates.csv.gz')
            for name in ('scene','seed','objective','stage','key'):cost[name]=s[name]
            cost.to_csv(handle,index=False,header=first_file);first_file=False
    write(output/'validation.json',dict(passed=True,runs=len(summaries),ue_rows=len(per_ue),
        full_window_olla_samples=72*20000*36,sinr_sample_rows=len(samples),
        per_ue_sinr_variance='sample variance ddof=1; undefined when fewer than two samples; all-TTI OLLA and pooled descriptive variances use ddof=0',
        queue_identity='post = pre - (successful bytes + failed bytes), verified for all measured UE/TTI entries',
        interpretation='full-window native BLER/OLLA/MCS/expiry; signed SINR variance is saved-state descriptive, not full-window variance'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--workers',type=int,default=4)
    main(parser.parse_args())

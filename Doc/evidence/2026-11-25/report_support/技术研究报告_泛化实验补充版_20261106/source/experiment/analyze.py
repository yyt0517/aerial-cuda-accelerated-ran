"""Mature cohorts, CRC/FIFO conservation, subgroup results and paired-seed validation."""
import argparse,json,hashlib,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from training.stageb_ppo.native import legacy,write,sha
from .protocol import ROOT,OUT,SEEDS,SCENARIOS,ARMS
FIELDS=['tti','ue_id','offered_bytes','accepted_bytes','rejected_bytes','expired_bytes','previous_post_queue_bytes','pre_queue_bytes','scheduled_dequeue_bytes','post_queue_bytes','successful_bytes','failed_bytes','actual_prg_count','mcs','layer','phy_sinr_db_cpu','predicted_bler_cpu','tb_tx_count','tb_err_count']
def local(path):return Path(str(path).replace('/opt/nvidia/cuBB',str(ROOT)))
def digest_frame(df):return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
def ratios(c,seconds):
    good=c[c.successfully_delivered];expired=c.outcome.eq('expired');totalbytes=c.packet_size_bytes.sum()
    return dict(packets=len(c),offered_payload_bytes=int(totalbytes),payload_mbps=float(good.packet_size_bytes.sum()*8/seconds/1e6),success_pct=float(100*c.successfully_delivered.mean()),expiry_pct=float(100*expired.mean()),expiry_byte_pct=float(100*c.loc[expired,'packet_size_bytes'].sum()/totalbytes),delay_mean_ms=float(good.system_time_ms.mean()) if len(good) else None,delay_p95_ms=float(good.system_time_ms.quantile(.95)) if len(good) else None,outcome_mean_ms=float(c.system_time_ms.mean()),outcome_p95_ms=float(c.system_time_ms.quantile(.95)))

def one(done,output,audit=True):
    started=time.monotonic();job=json.loads(Path(done).read_text());folder=local(job['directory']);dest=Path(output)/job['key'];dest.mkdir(parents=True,exist_ok=True)
    if (dest/'summary.json').exists():return json.loads((dest/'summary.json').read_text())
    launch=json.loads((folder/'launch.json').read_text());validation=json.loads((folder/'validation.json').read_text());assert validation['passed']
    assert launch['simulator_sha256']==json.loads((OUT/'native/build_manifest.json').read_text())['simulator_sha256']
    planpath=local(launch['flow_file']);plan=pd.read_csv(planpath).sort_values('ue');assert list(plan.ue)==list(range(36))
    horizon=launch['horizon'];first,last=launch['measurement_first_tti'],launch['measurement_last_tti'];seconds=(last-first+1)*.0005
    data=np.memmap(folder/'metrics.bin',dtype='<f8',mode='r').reshape(horizon,36,len(FIELDS));service=pd.DataFrame(data.reshape(-1,len(FIELDS)),columns=FIELDS)
    offered=pd.read_csv(folder/'traffic_offered_packets.csv');outcomes=pd.read_csv(folder/'packet_outcome_samples.csv')
    offered=offered[offered.arrival_tti<=horizon].copy()
    sizes=plan.set_index('ue').packet_size_bytes
    assert offered.packet_size_bytes.eq(offered.ue_id.map(sizes)).all()
    assert offered.ttl_tti.eq(400).all()
    legacy()
    from output.pfq_independent_reuse.formal.boundary08_fast_multiseed_20261010_v1.summarize import packets_from_fifo
    pk=packets_from_fifo(service,offered,outcomes,horizon)
    reward=pd.read_csv(folder/'rewards.csv')
    completed=pk[pk.successfully_delivered].groupby('terminal_tti').packet_size_bytes.sum().reindex(range(1,horizon+1),fill_value=0)
    np.testing.assert_array_equal(completed,reward.complete_payload_bytes)
    expired=pk[pk.outcome.eq('expired')].groupby('terminal_tti').packet_size_bytes.sum().reindex(range(horizon),fill_value=0)
    np.testing.assert_array_equal(expired,reward.expired_payload_bytes)
    cohort=pk[pk.arrival_tti.between(first,last)].copy();assert len(cohort)>0 and not cohort.censored.any()
    cohort=cohort.merge(plan[['ue','serving_cell','placement_group','traffic_group']],left_on='ue_id',right_on='ue',validate='many_to_one').drop(columns='ue')
    cohort.to_csv(dest/'cohort.csv.gz',index=False)
    measured=data[first-1:last];get=lambda col:measured[:,:,FIELDS.index(col)]
    tx=get('actual_prg_count')>0
    per=[]
    for u in range(36):
        c=cohort[cohort.ue_id==u];d=ratios(c,seconds);d.update(ue=u,**plan.iloc[u][['serving_cell','placement_group','traffic_group','packet_size_bytes','arrival_rate_packets_per_tti']].to_dict());per.append(d)
    ue=pd.DataFrame(per);ue.to_csv(dest/'per_ue.csv',index=False)
    groups=[]
    for dimension in ['placement_group','traffic_group','serving_cell']:
        for group,c in cohort.groupby(dimension):
            d=ratios(c,seconds);d.update(dimension=dimension,group=str(group),ues=int(c.ue_id.nunique()),worst_ue_expiry_pct=float(ue[ue.ue.isin(c.ue_id.unique())].expiry_pct.max()));groups.append(d)
    heavy=int(plan.heavy_cell.iloc[0]);cells=[]
    for c in range(3):
        ids=plan[plan.serving_cell==c].ue.to_numpy();mask=tx[:,ids]
        row=ratios(cohort[cohort.serving_cell==c],seconds)
        row.update(cell=c,users=len(ids),heavy=c==heavy,offered_mbps=float(get('offered_bytes')[:,ids].sum()*8/seconds/1e6),served_mbps=float(get('scheduled_dequeue_bytes')[:,ids].sum()*8/seconds/1e6),goodput_mbps=float(get('successful_bytes')[:,ids].sum()*8/seconds/1e6),queue_mean_mb=float(get('post_queue_bytes')[:,ids].sum(1).mean()/1e6),prg_occupancy_pct=float(get('actual_prg_count')[:,ids].sum()/(len(measured)*17)*100))
        cells.append(row)
    if heavy>=0:
        for group,c in [('heavy_cell',cohort[cohort.serving_cell==heavy]),('light_cells',cohort[cohort.serving_cell!=heavy])]:
            d=ratios(c,seconds);d.update(dimension='cell_load',group=group,ues=int(c.ue_id.nunique()),worst_ue_expiry_pct=float(ue[ue.ue.isin(c.ue_id.unique())].expiry_pct.max()));groups.append(d)
        for group,c in cohort[cohort.serving_cell==heavy].groupby('traffic_group'):
            d=ratios(c,seconds);d.update(dimension='heavy_cell_traffic',group=str(group),ues=int(c.ue_id.nunique()),worst_ue_expiry_pct=float(ue[ue.ue.isin(c.ue_id.unique())].expiry_pct.max()));groups.append(d)
    pd.DataFrame(groups).to_csv(dest/'groups.csv',index=False);pd.DataFrame(cells).to_csv(dest/'per_cell.csv',index=False)
    goodrates=ue.payload_mbps.to_numpy();r=ratios(cohort,seconds)
    r.update(**{k:job[k] for k in ['scenario','arm','seed','key']},offered_mbps=float(get('offered_bytes').sum()*8/seconds/1e6),served_mbps=float(get('scheduled_dequeue_bytes').sum()*8/seconds/1e6),goodput_mbps=float(get('successful_bytes').sum()*8/seconds/1e6),worst_ue_expiry_pct=float(ue.expiry_pct.max()),ue_payload_p5_mbps=float(np.quantile(goodrates,.05)),jain_payload=float(goodrates.sum()**2/(36*np.sum(goodrates**2))),queue_mean_mb=float(get('post_queue_bytes').sum(1).mean()/1e6),queue_start_mb=float(data[first-1,:,FIELDS.index('previous_post_queue_bytes')].sum()/1e6),queue_end_mb=float(data[last-1,:,FIELDS.index('post_queue_bytes')].sum()/1e6),prg_occupancy_pct=float(get('actual_prg_count').sum()/(len(measured)*51)*100),sinr_tx_mean_db=float(get('phy_sinr_db_cpu')[tx].mean()),mcs_tx_mean=float(get('mcs')[tx].mean()),tb_error_pct=float(get('tb_err_count').sum()/get('tb_tx_count').sum()*100),heavy_cell=heavy,measurement_seconds=seconds,cohort_packets=len(cohort),censored=0)
    with np.load(folder/'actions.npz') as a:
        actions=a['actions'].reshape(-1,3,17);assert len(actions)==horizon
        for c in range(3):
            us=np.unique(actions[:,c]);us=us[us>=0].astype(int);assert np.all(plan.set_index('ue').loc[us,'serving_cell']==c)
        np.savez_compressed(dest/'prefix.npz',actions=actions[:launch['common_pfq_warmup']],input_hashes=a['input_hashes'][:launch['common_pfq_warmup']])
    r['traffic_sha256']=digest_frame(offered.sort_values(['ue_id','packet_id']).reset_index(drop=True))
    r['topology_sha256']=sha(folder/'ue_topology.csv');r['plan_sha256']=sha(planpath)
    state_checks=[]
    if audit and job['arm']!='pfq':
        import torch
        torch.set_num_threads(1)
        from training.stageb_lns.archive import enable_legacy_types
        from training.stageb_lns.no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
        from training.stageb_lns.physics import ReferencePhy
        from training.stageb_lns.persistent_dpls import solve_integer_reference
        from training.stageb_lns.bounded_dpls import solve_bounded
        from training.stageb_lns.bounded_campaign import audit_round
        enable_legacy_types()
        for entry in json.loads((folder/'saved_states.json').read_text()):
            file=local(entry['path']);assert sha(file)==entry['sha256'];row=torch.load(file,map_location='cpu',weights_only=False)
            p=problem_from_raw_row(row)
            if job['arm']=='deficit':
                ww=plan.packet_size_bytes.to_numpy()*plan.arrival_rate_packets_per_tti.to_numpy()/np.maximum(row['ema'],1.e-9);ww=np.clip(ww,1.,4.);ww/=ww.mean()
            else:
                from training.stageb_lns.crossload_native import weights_for
                ww=weights_for('mlwdf_pf',p.weights,row['ema'],row['hol_urgency'],p.queue,0.)
            np.testing.assert_array_equal(ww,row['objective_weights'])
            p.weights=ww;a=row['selected_action'];v=row['selected_values'];native=data[row['tti']-1];active=v[:,0]>0
            expected=ReferencePhy(p).evaluate(a)[0];np.testing.assert_allclose(v,expected,rtol=1e-8,atol=1e-6)
            for vi,col in [(0,'actual_prg_count'),(1,'layer'),(3,'scheduled_dequeue_bytes')]:np.testing.assert_array_equal(v[active,vi],native[active,FIELDS.index(col)])
            sinr=float(np.max(np.abs(v[active,4]-native[active,FIELDS.index('phy_sinr_db_cpu')]),initial=0));bler=float(np.max(np.abs(v[active,5]-native[active,FIELDS.index('predicted_bler_cpu')]),initial=0));assert sinr<.005 and bler<.0005
            trace=[];ref=solve_integer_reference(p,raw_pfq_draft(p,row['obs']),observer=lambda z:trace.append({k:w.copy() if isinstance(w,np.ndarray) else w for k,w in z.items()}))
            np.testing.assert_array_equal(ref[0],a);np.testing.assert_array_equal(ref[1],v)
            rounds=[]
            bounded=solve_bounded(p,raw_pfq_draft(p,row['obs']),observer=lambda z:rounds.append(audit_round(z,trace[z['iteration']-1])))
            np.testing.assert_array_equal(bounded[0],a);assert bounded[2]['converged']
            state_checks.append(dict(tti=row['tti'],sinr_max_abs_db=sinr,bler_max_abs=bler,full_enumeration_exact=True,rounds=len(rounds),bounded_certificates_valid=True))
    write(dest/'validation.json',dict(passed=True,packet_fifo_crc_conservation=True,no_censoring=True,per_ue_packet_sizes_verified=True,native_reward_matches_reconstruction=True,actual_serving_association_verified=True,saved_states=state_checks))
    write(dest/'input_manifest.json',{str(p):sha(p) for p in [folder/'launch.json',folder/'metrics.bin',folder/'traffic_offered_packets.csv',folder/'packet_outcome_samples.csv',folder/'actions.npz',planpath]})
    r['analysis_seconds']=time.monotonic()-started;write(dest/'summary.json',r);return r

def aggregate(output,smoke=False,partial=False):
    output=Path(output);rows=[json.loads(p.read_text()) for p in output.glob('*/summary.json')];df=pd.DataFrame(rows)
    if df.empty:return
    df=df.sort_values(['scenario','arm','seed']);df.to_csv(output/'per_seed.csv',index=False)
    expected=12 if smoke else 60
    if not partial:assert len(df)==expected
    checks=[];groups=[];cells=[];ues=[]
    for (scenario,seed),q in df.groupby(['scenario','seed']):
        if len(q)!=3:
            assert partial;continue
        assert set(q.arm)==set(ARMS)
        for col in ['traffic_sha256','topology_sha256','plan_sha256']:assert q[col].nunique()==1,(scenario,seed,col)
        base=q[q.arm=='pfq'].iloc[0]
        with np.load(output/base.key/'prefix.npz') as a:
            for r in q.itertuples():
                with np.load(output/r.key/'prefix.npz') as b:
                    np.testing.assert_array_equal(a['actions'],b['actions']);np.testing.assert_array_equal(a['input_hashes'],b['input_hashes'])
        checks.append(dict(scenario=scenario,seed=int(seed),identical_topology_arrivals_and_pfq_warmup=True))
    for r in df.itertuples():
        for name,collection in [('groups',groups),('per_cell',cells),('per_ue',ues)]:
            f=pd.read_csv(output/r.key/f'{name}.csv');f['scenario']=r.scenario;f['arm']=r.arm;f['seed']=r.seed;collection.append(f)
    metric_cols=[c for c in df.select_dtypes(include='number').columns if c not in ['seed','heavy_cell']]
    ag=df.groupby(['scenario','arm'])[metric_cols].agg(['mean','min','max','std'])
    ag.columns=['__'.join(c) for c in ag.columns];ag.reset_index().to_csv(output/'aggregate.csv',index=False)
    g=pd.concat(groups,ignore_index=True);g.to_csv(output/'groups_per_seed.csv',index=False)
    g.groupby(['scenario','arm','dimension','group']).mean(numeric_only=True).reset_index().to_csv(output/'groups_mean.csv',index=False)
    pd.concat(cells,ignore_index=True).to_csv(output/'per_cell_all.csv',index=False);pd.concat(ues,ignore_index=True).to_csv(output/'per_ue_all.csv',index=False)
    paired=[]
    for scenario,frame in df.groupby('scenario'):
        for baseline in ['pfq','deficit']:
            a=frame[frame.arm=='mlwdf_pf'].set_index('seed');b=frame[frame.arm==baseline].set_index('seed');ss=sorted(set(a.index)&set(b.index))
            for metric in ['payload_mbps','expiry_pct','expiry_byte_pct','delay_mean_ms','worst_ue_expiry_pct']:
                delta=a.loc[ss,metric]-b.loc[ss,metric]
                if len(delta):paired.append(dict(scenario=scenario,baseline=baseline,metric=metric,n=len(delta),mean=float(delta.mean()),min=float(delta.min()),max=float(delta.max()),positive=int((delta>0).sum()),negative=int((delta<0).sum()),zero=int((delta==0).sum())))
    pd.DataFrame(paired).to_csv(output/'paired_differences.csv',index=False)
    statechecks=[z for p in output.glob('*/validation.json') for z in json.loads(p.read_text())['saved_states']]
    write(output/'validation.json',dict(passed=True,complete=not partial,runs=len(df),expected=expected,paired_checks=checks,packets=int(df.cohort_packets.sum()),saved_states_verified=len(statechecks),exact_search_rounds=sum(z['rounds'] for z in statechecks),scope='Mature cohort and full-payload CRC, per-seed averaging, paired topology/arrivals/warmup; saved-state exact enumeration and native PHY'))

def main(args):
    smoke=args.smoke;source=OUT/('smoke' if smoke else 'campaign');output=OUT/('smoke_analysis' if smoke else 'analysis');output.mkdir(exist_ok=True)
    paths=sorted(source.glob('*/done.json'))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(one,p,output,True):p for p in paths}
        for f in as_completed(pending):r=f.result();print('ANALYZED',r['key'],r['payload_mbps'],r['expiry_pct'],flush=True)
    aggregate(output,smoke,args.partial)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--partial',action='store_true');p.add_argument('--workers',type=int,default=3);main(p.parse_args())

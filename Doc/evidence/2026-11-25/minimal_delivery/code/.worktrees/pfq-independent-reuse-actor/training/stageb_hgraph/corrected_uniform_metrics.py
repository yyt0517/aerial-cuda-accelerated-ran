"""Scientific accounting for the fixed 3cell/36UE corrected uniform campaign."""
from collections import defaultdict, deque
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .capacity_units_decision_probe import check_capacity_row, canonical_actions
from .phy_tbs_alignment_audit import MCS_CODE_RATES, MCS_QAM_ORDERS

BANDWIDTH_HZ = 97_920_000


def describe(values, prefix):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    result = {prefix + '_count': int(len(x)), prefix + '_mean': float(x.mean()) if len(x) else None}
    for name, q in [('min',0),('p5',5),('p10',10),('p50',50),('p90',90),('p95',95),('p99',99),('max',100)]:
        result[prefix+'_'+name] = float(np.percentile(x,q)) if len(x) else None
    return result


def reconstruct_packets(service, offered, outcomes, total):
    """Replay existing FIFO, checked against terminal records and every queue balance.

    Expiry terminal TTI is one earlier than the TTI that removes its bytes.
    Only executed TTIs are replayed; pending is censored at the requested horizon.
    """
    queues = defaultdict(deque)
    arrivals = defaultdict(list)
    for r in offered.to_dict('records'):
        if int(r['arrival_tti']) <= total:
            arrivals[(int(r['arrival_tti']),int(r['ue_id']))].append(r)
    terminal = {(int(r['ue_id']),int(r['packet_id'])):r for r in outcomes.to_dict('records')}
    records = []
    for r in service.sort_values(['tti','ue_id']).to_dict('records'):
        t,u = int(r['tti']),int(r['ue_id'])
        assert r['rejected_bytes'] == 0, 'rejected packets require explicit admission records'
        assert r['offered_bytes'] == r['accepted_bytes']
        assert r['pre_queue_bytes'] == r['previous_post_queue_bytes'] + r['accepted_bytes'] - r['expired_bytes']
        assert r['post_queue_bytes'] == r['pre_queue_bytes'] - r['scheduled_dequeue_bytes']
        assert r['successful_bytes'] + r['failed_bytes'] == r['scheduled_dequeue_bytes']
        # A single UE TB outcome unambiguously assigns the success of each FIFO byte.
        assert not (r['successful_bytes'] > 0 and r['failed_bytes'] > 0), 'mixed TB byte success cannot be attributed to packets'
        q = queues[u]
        assert sum(p['remaining_bytes'] for p in q) == r['previous_post_queue_bytes']
        expired = 0
        while q and t - q[0]['arrival_tti'] + 1 > q[0]['ttl_tti']:
            p = q.popleft()
            p.update(outcome='expired', terminal_tti=t-1, event_tti=t,
                     system_time_ms=p['ttl_tti']*.5)
            expired += p['remaining_bytes']
            observed = terminal[(u,p['packet_id'])]
            assert observed['outcome'] == 'expired' and observed['terminal_tti'] == t-1
            assert observed['remaining_bytes_at_terminal'] == p['remaining_bytes']
        assert expired == r['expired_bytes']
        assert sum(p['packet_size_bytes'] for p in arrivals[(t,u)]) == r['offered_bytes']
        for a in arrivals[(t,u)]:
            p = dict(a, remaining_bytes=int(a['packet_size_bytes']), successful_bytes=0,
                     failed_dequeue_bytes=0, outcome='pending', terminal_tti=None,
                     event_tti=None, system_time_ms=None)
            q.append(p); records.append(p)
        left = int(r['scheduled_dequeue_bytes'])
        while left:
            p = q[0]
            take = min(left,p['remaining_bytes'])
            p['remaining_bytes'] -= take
            p['successful_bytes' if r['successful_bytes'] else 'failed_dequeue_bytes'] += take
            left -= take
            if p['remaining_bytes'] == 0:
                q.popleft()
                p.update(outcome='mac_completed', terminal_tti=t, event_tti=t,
                         system_time_ms=(t-p['arrival_tti']+1)*.5)
                observed = terminal[(u,p['packet_id'])]
                assert observed['outcome'] == 'delivered' and observed['terminal_tti'] == t
                assert observed['system_time_ms'] == p['system_time_ms'], (observed, p)
        assert sum(p['remaining_bytes'] for p in q) == r['post_queue_bytes']
    for p in records:
        p['scheduled_dequeue_bytes'] = p['successful_bytes'] + p['failed_dequeue_bytes']
        p['successfully_delivered'] = p['outcome']=='mac_completed' and p['failed_dequeue_bytes']==0
        p['age_at_horizon_ms'] = (total-p['arrival_tti']+1)*.5 if p['outcome']=='pending' else None
        p['censored'] = p['outcome']=='pending'
    return pd.DataFrame(records)


def summarize_group(s, packets, first, last, ncell):
    ntti=last-first+1
    ue = sorted(s.ue_id.unique())
    sums=s.sum(numeric_only=True)
    rate=lambda key:float(sums[key])*8/(ntti*.0005)/1e6
    ratio=lambda a,b:float(a/b) if b else None
    result={key:int(sums[key]) for key in ('offered_bytes','accepted_bytes','rejected_bytes','expired_bytes',
            'successful_bytes','failed_bytes','scheduled_dequeue_bytes','tbs_capacity_bits','tb_tx_count','tb_err_count')}
    result.update(goodput_mbps=rate('successful_bytes'), served_mbps=rate('scheduled_dequeue_bytes'),
                  offered_mbps=rate('offered_bytes'), failed_mbps=rate('failed_bytes'),
                  scheduled_phy_capacity_mbps=float(sums.tbs_capacity_bits)/(ntti*.0005)/1e6,
                  tb_bler=ratio(sums.tb_err_count,sums.tb_tx_count),
                  expiry_bytes_over_offered=ratio(sums.expired_bytes,sums.offered_bytes),
                  goodput_over_offered=ratio(sums.successful_bytes,sums.offered_bytes),
                  prg_utilization=ratio(sums.actual_prg_count,ntti*ncell*17) if ncell else None,
                  successful_bytes_per_active_prg=ratio(sums.successful_bytes,sums.actual_prg_count))
    for key in ('goodput','served'):
        result[key+'_spectral_efficiency_bps_hz']=result[key+'_mbps']*1e6/BANDWIDTH_HZ
        result[key+'_spectral_efficiency_per_cell_bandwidth_bps_hz']=result[key+'_mbps']*1e6/(max(1,ncell)*BANDWIDTH_HZ)
    end=s[s.tti==last]
    result['end_backlog_bytes']=int(end.post_queue_bytes.sum())
    result.update(describe(s.groupby('tti').post_queue_bytes.sum(),'total_backlog_bytes'))
    result.update(describe(s.hol_delay_ms[s.hol_delay_ms>=0],'hol_delay_ms'))
    result.update(describe(s.ttl_slack_ms[s.ttl_slack_ms>=0],'ttl_slack_ms'))
    tx=s[s.actual_prg_count>0]
    for field in ['mcs','layer','qam_order','code_rate','phy_sinr_db_cpu','predicted_bler_cpu','phy_sinr_db_gpu','predicted_bler_gpu']:
        values=tx[field]
        values=values[values > -900] if 'sinr' in field else values[values>=0]
        result.update(describe(values, field+'_tx_only'))
    result['mcs_all_tti_zero_mean']=float(s.mcs.where(s.actual_prg_count>0,0).mean())
    result['layer_all_tti_zero_mean']=float(s.layer.where(s.actual_prg_count>0,0).mean())
    for field in s.columns:
        if field.startswith('pre_') and 'post_eq' not in field:
            result.update(describe(s[field],field))
    terminal=packets[(packets.terminal_tti>=first)&(packets.terminal_tti<=last)&packets.ue_id.isin(ue)]
    completed=terminal[terminal.outcome=='mac_completed']
    successful=terminal[terminal.successfully_delivered]
    for name, rows in [('mac_dequeue_completion_delay_ms',completed),('successful_delivery_delay_ms',successful),
                       ('mac_completion_or_expiry_outcome_delay_ms',terminal),('expiry_delay_ms',terminal[terminal.outcome=='expired'])]:
        result.update(describe(rows.system_time_ms,name))
    pending=packets[(packets.outcome=='pending')&packets.ue_id.isin(ue)]
    result.update(pending_packet_count=len(pending), mac_completed_packet_count=len(completed),
                  successfully_delivered_packet_count=len(successful), expired_packet_count=int((terminal.outcome=='expired').sum()),
                  prior_window_arrival_terminal_count=int((terminal.arrival_tti<first).sum()))
    result.update(describe(pending.age_at_horizon_ms,'pending_age_ms'))
    result['mac_packet_service_rate_mbps']=ratio(completed.packet_size_bytes.sum()*8,completed.system_time_ms.sum()*1000)
    result['mac_packet_expiry_aware_rate_mbps']=ratio(completed.packet_size_bytes.sum()*8,terminal.system_time_ms.sum()*1000)
    result['successful_byte_outcome_time_rate_mbps']=ratio(sums.successful_bytes*8,terminal.system_time_ms.sum()*1000)
    rates=s.groupby('ue_id').successful_bytes.sum()*8/(ntti*.0005)/1e6
    result.update(describe(rates,'ue_goodput_mbps'))
    result['ue_goodput_jain']=ratio(rates.sum()**2,len(rates)*(rates**2).sum())
    result['zero_goodput_ue_count']=int((rates==0).sum())
    result['zero_dequeue_ue_count']=int((s.groupby('ue_id').scheduled_dequeue_bytes.sum()==0).sum())
    result.update(describe((s.scheduled_dequeue_bytes>0).groupby(s.tti).sum(),'served_ue_per_tti'))
    streaks=[]
    for _, rows in s.groupby('ue_id'):
        streak=longest=0
        for r in rows.sort_values('tti').itertuples():
            streak=streak+1 if r.pre_queue_bytes>0 and r.scheduled_dequeue_bytes==0 else 0
            longest=max(longest,streak)
        streaks.append(longest)
    result['max_backlogged_zero_dequeue_streak_tti']=max(streaks,default=0)
    result['queue_delay_est_ms']=ratio(result['end_backlog_bytes']*8,result['served_mbps']*1000)
    return result


def export_arm(out, seed, policy, warmup, measured):
    out=Path(out); total=warmup+measured
    s=pd.read_csv(out/'service_balance_trace.csv')
    if len(s)!=total*36 or s.duplicated(['tti','ue_id']).any() or set(s.ue_id)!=set(range(36)) or set(s.tti)!=set(range(1,total+1)):
        raise ValueError('incomplete service grid')
    for r in s.to_dict('records'): check_capacity_row(r)
    topology=pd.read_csv(out/'ue_topology.csv')
    assert set(topology.placement)=={'uniform'} and set(topology.topology_seed)=={seed}
    assert 'ue_placement=uniform' in (out/'sim.log').read_text()
    manifest=json.loads((out/'reference_manifest.json').read_text())
    assert manifest['scenario']['ue_placement']=='uniform'
    observations=pd.read_csv(out/'ue_tti_observations.csv')
    assert len(observations)==len(s) and not observations.duplicated(['tti','ue_id']).any()
    s=s.merge(observations,on=['tti','ue_id'],validate='one_to_one')
    assert len(s)==36*total
    s['seed']=seed;s['policy']=policy
    s['goodput_mbps']=s.successful_bytes*.016;s['served_mbps']=s.scheduled_dequeue_bytes*.016
    s['qam_order']=s.mcs.map(dict(enumerate(MCS_QAM_ORDERS))).where(s.actual_prg_count>0)
    s['code_rate']=s.mcs.map(dict(enumerate(MCS_CODE_RATES))).where(s.actual_prg_count>0)/1024
    s.to_csv(out/'ue_tti_metrics.csv',index=False)
    offered=pd.read_csv(out/'traffic_offered_packets.csv'); outcomes=pd.read_csv(out/'packet_outcome_samples.csv')
    packets=reconstruct_packets(s,offered,outcomes,total)
    packets['seed']=seed;packets['policy']=policy
    packets['serving_cell']=packets.ue_id.map(s.groupby('ue_id').serving_cell.first())
    packets.to_csv(out/'packet_lifecycle.csv',index=False)
    actions=canonical_actions(s.to_dict('records'),total_tti=total).reshape(total,3,17)
    indices=np.indices(actions.shape)
    a=pd.DataFrame(dict(tti=indices[0].ravel()+1,cell_id=indices[1].ravel(),prg_id=indices[2].ravel(),ue_id=actions.ravel()))
    a['seed']=seed;a['policy']=policy;a['active']=a.ue_id>=0
    a['same_prg_active_cells']=a.groupby(['tti','prg_id']).active.transform('sum')
    a.to_csv(out/'cell_prg_actions.csv',index=False)
    tti=[]
    keys=['offered_bytes','accepted_bytes','rejected_bytes','expired_bytes','successful_bytes','failed_bytes','scheduled_dequeue_bytes','post_queue_bytes','tbs_capacity_bits','tb_tx_count','tb_err_count','actual_prg_count']
    for cell in [-1,0,1,2]:
        subset=s if cell==-1 else s[s.serving_cell==cell]
        d=subset.groupby('tti')[keys].sum().reset_index()
        d['seed']=seed;d['policy']=policy;d['cell_id']=cell
        d['goodput_mbps']=d.successful_bytes*.016;d['served_mbps']=d.scheduled_dequeue_bytes*.016
        d['tb_bler']=d.tb_err_count/d.tb_tx_count.replace(0,np.nan)
        d['prg_utilization']=d.actual_prg_count/(51 if cell==-1 else 17)
        d['blank_ratio']=1-d.prg_utilization
        d['goodput_spectral_efficiency_bps_hz']=d.goodput_mbps*1e6/BANDWIDTH_HZ
        d['served_spectral_efficiency_bps_hz']=d.served_mbps*1e6/BANDWIDTH_HZ
        d['served_ue_count']=(subset.scheduled_dequeue_bytes>0).groupby(subset.tti).sum().to_numpy()
        if cell==-1:
            counts=(actions>=0).sum(axis=1)
            d['reuse_ratio']=np.maximum(counts-1,0).sum(axis=1)/np.maximum(1,2*(counts>0).sum(axis=1))
        tti.append(d)
    pd.concat(tti).to_csv(out/'tti_metrics.csv',index=False)
    globals=[]; cells=[]; ues=[]; hist=[]; aligned=[]
    for window,first in [('full',1),('measured',warmup+1)]:
        selected=s[s.tti>=first]
        identity=dict(seed=seed,policy=policy,window=window,first_tti=first,last_tti=total)
        globals.append({**identity,**summarize_group(selected,packets,first,total,3)})
        from .pfq_dpa_aligned_comparison import summarize_aligned_window
        from .dpa_v1_historical_comparison import WindowSpec
        packet_rows=packets[packets.outcome!='pending'].rename(columns={'remaining_bytes':'remaining_bytes_at_terminal'}).copy()
        packet_rows['outcome']=packet_rows.outcome.replace({'mac_completed':'delivered'})
        old=summarize_aligned_window(service_rows=selected.to_dict('records'),outcome_rows=packet_rows.to_dict('records'),
              decision_rows=tti[0][tti[0].tti>=first].rename(columns={'reuse_ratio':'prg_reuse_ratio'}).to_dict('records'),
              window=WindowSpec(window,first,total))
        aligned.append({**identity,**{k:v for k,v in old.items() if not isinstance(v,(dict,list))}})
        globals[-1]['prg_reuse_ratio']=old['prg_reuse_ratio']
        globals[-1]['blank_ratio']=old['blank_ratio']
        for key,value in old.items():
            if key not in globals[-1] and not isinstance(value,(dict,list)):
                globals[-1][key]=value
        globals[-1]['success_rate']=1-globals[-1]['tb_bler'] if globals[-1]['tb_bler'] is not None else None
        globals[-1]['fill_ratio']=globals[-1]['prg_utilization']
        globals[-1]['offered_bytes_per_tti']=globals[-1]['offered_bytes']/(total-first+1)
        globals[-1]['residual_buffer_ratio']=globals[-1]['end_backlog_bytes']/globals[-1]['offered_bytes']
        globals[-1]['packet_effective_service_rate_mbps']=globals[-1]['mac_packet_service_rate_mbps']
        for cell in range(3):
            cells.append({**identity,'cell_id':cell,**summarize_group(selected[selected.serving_cell==cell],packets,first,total,1)})
        for u in range(36):
            ues.append({**identity,'ue_id':u,'cell_id':int(s[s.ue_id==u].serving_cell.iloc[0]),**summarize_group(selected[selected.ue_id==u],packets,first,total,0)})
        cell_rates=np.array([r['goodput_mbps'] for r in cells[-3:]])
        globals[-1]['cell_goodput_jain']=float(cell_rates.sum()**2/(3*(cell_rates**2).sum())) if np.any(cell_rates) else None
        valid_cell_bler=[r['tb_bler'] for r in cells[-3:] if r['tb_bler'] is not None]
        globals[-1]['cell_tb_bler_mean']=float(np.mean(valid_cell_bler)) if valid_cell_bler else None
        globals[-1]['goodput_spectral_efficiency_mean_bps_hz']=float(cell_rates.mean()*1e6/BANDWIDTH_HZ)
        globals[-1]['goodput_spectral_efficiency_min_bps_hz']=float(cell_rates.min()*1e6/BANDWIDTH_HZ)
        rates=[r['mac_packet_service_rate_mbps'] for r in ues[-36:] if r['mac_packet_service_rate_mbps'] is not None]
        globals[-1]['ue_macro_packet_effective_service_rate_mbps']=float(np.mean(rates)) if rates else None
        for field in ['mcs','layer','qam_order','code_rate']:
            for value,count in selected.loc[selected.actual_prg_count>0,field].value_counts().items():
                hist.append({**identity,'metric':field,'value':value,'count':int(count),'denominator':'tx UE-TTI'})
        hist.append({**identity,'metric':'no_tx','value':-1,'count':int((selected.actual_prg_count==0).sum()),'denominator':'all UE-TTI'})
    for name,rows in [('global_metrics',globals),('per_cell_metrics',cells),('per_ue_metrics',ues),('mcs_histograms',hist),('historical_aligned_metrics',aligned)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    pd.DataFrame([dict(seed=seed,policy=policy,window=r['window'],metric=k,value=v) for r in globals
                  for k,v in r.items() if k.endswith(('_mean','_min','_max','_p5','_p10','_p50','_p90','_p95','_p99','_count'))]).to_csv(out/'distributions.csv',index=False)
    return globals


def export_legacy_catalog(out):
    """Keep the historical dictionary and full-window outputs, with explicit aliases."""
    import importlib.util
    import subprocess
    import sys
    import os
    root=Path(__file__).resolve().parents[2]
    script=root/'cuMAC/scripts/summarize_stageA_kpi.py'
    legacy=Path(out)/'historical_full_window'
    legacy.mkdir(exist_ok=True)
    for source,dest in [('output.txt','output.txt'),('output_short.txt','output_short.txt'),('sim.log','run.log')]:
        if not (legacy/dest).exists(): os.link(Path(out)/source,legacy/dest)
    with (legacy/'summary_run.log').open('w') as log:
        subprocess.run([sys.executable,str(script),'--output-dir',str(legacy),'--slot-duration-ms','0.5',
                        '--traffic-percent','100','--packet-size-bytes','3000','--prbs-per-group','16','--prg-count','17'],
                        stdout=log,stderr=subprocess.STDOUT,check=True)
    spec=importlib.util.spec_from_file_location('old_kpi',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    catalog=[]
    for name,definition in module.METRIC_DEFINITIONS.items():
        semantics='Historical scheduler or PHY diagnostic; CPU and GPU paths must remain separate.'
        if any(x in name for x in ('packet_delay','packet_effective','delivered')):
            semantics='MAC dequeue completion, NOT successful packet delivery. Canonical successful_delivery_delay_ms is separately reconstructed from FIFO and TB outcomes.'
        if 'outcome' in name or 'expiry_aware' in name:
            semantics='MAC-completed plus expired terminal outcomes; pending excluded and separately censored.'
        catalog.append(dict(metric=name,legacy_definition=definition,source='historical_full_window/kpi_summary.json; sim.log; output_short.txt',
                            window='full',semantics=semantics,availability='see value and sample count in historical full-window output',
                            canonical_source='per_ue_metrics.csv; per_cell_metrics.csv; global_metrics.csv'))
    from .pfq_dpa_aligned_comparison import HEADLINE_METRICS
    from .dpa_v1_historical_comparison import METRIC_META
    from .granularity_experiment import _COHORT_METRICS, _CSV_METRICS, _CAMPAIGN_METRIC_DIRECTIONS
    oldkeys=set(HEADLINE_METRICS)|set(METRIC_META)|set(_COHORT_METRICS)|set(_CSV_METRICS)|set(_CAMPAIGN_METRIC_DIRECTIONS)
    globals=pd.read_csv(Path(out)/'global_metrics.csv')
    aliases={'completion_delay_mean_ms':'mac_dequeue_completion_delay_ms_mean',
             'completion_delay_p95_ms':'mac_dequeue_completion_delay_ms_p95',
             'outcome_delay_mean_ms':'mac_completion_or_expiry_outcome_delay_ms_mean',
             'outcome_delay_p95_ms':'mac_completion_or_expiry_outcome_delay_ms_p95',
             'expiry_drop_rate':'expiry_bytes_over_offered','expiry_rate':'expiry_bytes_over_offered',
             'prg_utilization_ratio':'prg_utilization','ue_goodput_p10_mbps':'ue_goodput_mbps_p10',
             'total_backlog_mean_bytes':'total_backlog_bytes_mean','total_backlog_p95_bytes':'total_backlog_bytes_p95',
             'bler':'tb_bler','expired_packets':'expired_packet_count','goodput_efficiency':'goodput_over_offered',
             'mcs_tx_weighted_mean':'mcs_tx_only_mean','layer_tx_weighted_mean':'layer_tx_only_mean',
             'prg_utilization_mean':'prg_utilization','successful_bytes_per_active_prg_mean':'successful_bytes_per_active_prg'}
    for stat in ['mean','p50','p90','p95','max']:
        aliases['packet_delay_'+stat+'_ms']='mac_dequeue_completion_delay_ms_'+stat
    for name in sorted(oldkeys):
        canonical=aliases.get(name,name)
        available=canonical in globals.columns
        catalog.append(dict(metric=name,legacy_definition='',source='archived aligned/granularity/service report key union',window='full/measured',
                            semantics=canonical if available else 'Historical campaign/configuration/comparison key; retained for audit, not assumed to be a per-arm metric.',
                            availability='canonical CSV' if available else 'requires individual mapping review',canonical_source=canonical if available else ''))
    for name in globals.columns:
        catalog.append(dict(metric=name,legacy_definition='',source='packet_lifecycle.csv or ue_tti_metrics.csv',window='full/measured',
                            semantics='Quantiles pooled over raw samples; counts are sample counts; ratios use summed bytes/TBs, not mean UE ratios.',
                            availability='canonical CSV',canonical_source=name))
    for row in catalog:
        name=row['metric']
        row['unit']=next((unit for token,unit in [('spectral_efficiency','bit/s/Hz'),('mbps','Mbit/s'),('delay','ms'),('_ms','ms'),('bytes','byte'),('bits','bit'),('sinr_db','dB'),('sinr_lin','linear'),('count','count'),('tti','TTI')] if token in name),'dimensionless or identifier; see definition')
        row['denominator']='sample count for mean/quantile; corresponding raw sample population'
        if 'mbps' in name: row['denominator']='window TTIs * 0.0005 seconds; packet service rates instead use summed packet system time in seconds'
        if 'spectral_efficiency' in name: row['denominator']='97,920,000 Hz; per_cell_bandwidth divides additionally by cell count'
        if 'bler' in name: row['denominator']='transmitted TB count (CPU/GPU separate); predicted BLER is an arithmetic UE-TTI sample statistic'
        if 'expiry' in name and ('rate' in name or 'over_offered' in name): row['denominator']='actual offered bytes; packet expiry-aware service rates use completed+expired system time'
        if 'jain' in name: row['denominator']='N * sum(x^2), numerator sum(x)^2, including zero-service UEs/cells'
        row['zero_sample']='NA with count=0 in canonical outputs; legacy outputs retain historical aliases and may contain zero placeholders'
    missing=[r['metric'] for r in catalog if r['availability']=='requires individual mapping review']
    if missing: raise ValueError('unmapped historical metrics: '+str(missing))
    pd.DataFrame(catalog).to_csv(Path(out)/'metric_catalog.csv',index=False)

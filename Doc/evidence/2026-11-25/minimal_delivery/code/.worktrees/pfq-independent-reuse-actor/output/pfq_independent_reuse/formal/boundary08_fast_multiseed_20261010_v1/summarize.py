"""Final business metrics only; FIFO byte attribution without per-TTI oracle audits."""
import argparse
from collections import deque
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from output.pfq_independent_reuse.formal.boundary08_extended_review_20261009_v1.analyze import per_ue_metrics
from .build import FIELDS

def packets_from_fifo(service, offered, outcomes, total):
    # Keep the exact previous FIFO and CRC definition, operating on arrays instead
    # of materializing 878400 Python service-row dictionaries and summing queues.
    offered = offered[offered.arrival_tti <= total].sort_values(['ue_id','arrival_tti','packet_id']).reset_index(drop=True)
    # Persistent bridge prepares TTI total+1 before receiving CloseReq. Expiry
    # stamped total is removed in that unexecuted step, as in the old replay.
    outcomes = outcomes[(outcomes.terminal_tti <= total) &
        ~((outcomes.outcome == 'expired') & (outcomes.terminal_tti == total))]
    arrivals = offered.arrival_tti.to_numpy(dtype=np.int64)
    size = offered.packet_size_bytes.to_numpy(dtype=np.int64)
    ttl = offered.ttl_tti.to_numpy(dtype=np.int64)
    remaining = size.copy(); successful = np.zeros(len(size),dtype=np.int64)
    terminal = np.full(len(size),-1,dtype=np.int64); expired = np.zeros(len(size),dtype=bool)
    assert (service.rejected_bytes == 0).all() and np.array_equal(service.offered_bytes,service.accepted_bytes)
    assert np.array_equal(service.pre_queue_bytes,service.previous_post_queue_bytes+service.accepted_bytes-service.expired_bytes)
    assert np.array_equal(service.post_queue_bytes,service.pre_queue_bytes-service.scheduled_dequeue_bytes)
    assert np.array_equal(service.successful_bytes+service.failed_bytes,service.scheduled_dequeue_bytes)
    assert not ((service.successful_bytes>0)&(service.failed_bytes>0)).any()
    for u in range(36):
        indices = np.flatnonzero(offered.ue_id.to_numpy()==u)
        cursor=0; q=deque()
        s = service[service.ue_id==u]
        assert np.array_equal(s.tti,np.arange(1,total+1))
        assert np.array_equal(s.previous_post_queue_bytes.to_numpy()[1:],s.post_queue_bytes.to_numpy()[:-1])
        for t,off,exp,left,succ in s[['tti','offered_bytes','expired_bytes','scheduled_dequeue_bytes','successful_bytes']].to_numpy(dtype=np.int64):
            removed=0
            while q and t-arrivals[q[0]]+1 > ttl[q[0]]:
                i=q.popleft(); removed+=remaining[i]; terminal[i]=t-1; expired[i]=True
            assert removed==exp,(u,t,'expiry')
            added=0
            while cursor<len(indices) and arrivals[indices[cursor]]==t:
                i=indices[cursor]; q.append(i); added+=size[i]; cursor+=1
            assert added==off,(u,t,'offered')
            while left:
                i=q[0]; take=min(left,remaining[i]); remaining[i]-=take; left-=take
                if succ: successful[i]+=take
                if remaining[i]==0: q.popleft(); terminal[i]=t
        assert sum(remaining[i] for i in q)==int(s.post_queue_bytes.iloc[-1])
    result=offered.copy()
    result['remaining_bytes']=remaining; result['successful_bytes']=successful
    result['failed_dequeue_bytes']=size-remaining-successful
    result['terminal_tti']=terminal
    result['outcome']=np.where(terminal<0,'pending',np.where(expired,'expired','mac_completed'))
    result['system_time_ms']=np.where(expired,ttl*.5,np.where(terminal<0,np.nan,(terminal-arrivals+1)*.5))
    result['censored']=terminal<0
    result['successfully_delivered']=(terminal>=0)&~expired&(result.failed_dequeue_bytes==0)
    observed=result[result.terminal_tti>=0].merge(outcomes,on=['ue_id','packet_id'],suffixes=('','_native'),validate='one_to_one')
    assert len(observed)==int((terminal>=0).sum())==len(outcomes)
    for col in ['arrival_tti','packet_size_bytes','terminal_tti','system_time_ms']:
        assert np.array_equal(observed[col],observed[col+'_native']),col
    assert np.array_equal(observed.remaining_bytes,observed.remaining_bytes_at_terminal)
    assert np.array_equal(observed.outcome.replace({'mac_completed':'delivered'}),observed.outcome_native)
    return result

def summarize(out):
    out=Path(out); started=time.monotonic()
    execution=json.loads((out/'execution.json').read_text())
    total=execution['horizon']; first=execution['warmup_ttis']+1; last=first+execution['measured_ttis']-1
    data=np.fromfile(out/'metrics.bin',dtype='<f8').reshape(-1,len(FIELDS))
    service=pd.DataFrame(data,columns=FIELDS)
    assert len(service)==total*36
    offered=pd.read_csv(out/'traffic_offered_packets.csv')
    outcomes=pd.read_csv(out/'packet_outcome_samples.csv')
    packets=packets_from_fifo(service,offered,outcomes,total)
    cohort=packets[packets.arrival_tti.between(first,last)]
    assert not cohort.censored.any(),'400-TTI followup must mature all measurement arrivals'
    s=service[service.tti.between(first,last)]; tx=s.actual_prg_count>0
    rates=s.groupby('ue_id').successful_bytes.sum()*8/(execution['measured_ttis']*.0005)/1e6
    per=per_ue_metrics(cohort); per['fixed_window_goodput_mbps']=rates
    per['expired_bytes']=s.groupby('ue_id').expired_bytes.sum()
    per.to_csv(out/'per_ue.csv')
    success=cohort[cohort.successfully_delivered]
    row=dict(seed=execution['seed'],arm=execution['arm'],goodput_mbps=float(rates.sum()),
        ue_mean_goodput_mbps=float(rates.mean()), ue_goodput_p5_mbps=float(rates.quantile(.05)),
        ue_goodput_p10_mbps=float(rates.quantile(.1)),ue_goodput_jain=float(rates.sum()**2/(36*(rates**2).sum())),
        successful_bytes=int(s.successful_bytes.sum()),failed_bytes=int(s.failed_bytes.sum()),
        dequeue_bytes=int(s.scheduled_dequeue_bytes.sum()),expired_bytes=int(s.expired_bytes.sum()),
        accepted_bytes=int(s.accepted_bytes.sum()),start_backlog_bytes=int(s[s.tti==first].previous_post_queue_bytes.sum()),
        end_backlog_bytes=int(s[s.tti==last].post_queue_bytes.sum()),allocated_prgs=int(s.actual_prg_count.sum()),
        prg_utilization=float(s.actual_prg_count.sum()/(execution['measured_ttis']*51)),
        successful_bytes_per_prg=float(s.successful_bytes.sum()/s.actual_prg_count.sum()),
        mcs_tx_mean=float(s.loc[tx,'mcs'].mean()),sinr_tx_mean_db=float(s.loc[tx,'phy_sinr_db_cpu'].mean()),
        tb_bler=float(s.tb_err_count.sum()/s.tb_tx_count.sum()),
        interference_tx_mean_scaled=execution['interference_tx_mean_scaled'],
        interference_prg_weighted_scaled=execution['interference_prg_weighted_scaled'],
        cohort_packets=len(cohort),complete_success=len(success),expired_packets=int((cohort.outcome=='expired').sum()),
        complete_success_pct=float(100*cohort.successfully_delivered.mean()),
        expiry_pct=float(100*(cohort.outcome=='expired').mean()),
        outcome_mean_ms=float(cohort.system_time_ms.mean()),outcome_p50_ms=float(cohort.system_time_ms.quantile(.5)),
        outcome_p95_ms=float(cohort.system_time_ms.quantile(.95)),outcome_p99_ms=float(cohort.system_time_ms.quantile(.99)),
        success_delay_mean_ms=float(success.system_time_ms.mean()),success_delay_p95_ms=float(success.system_time_ms.quantile(.95)),
        ue_macro_mac_packet_rate_mbps=float(per.mac_packet_rate_mbps.mean()),
        ue_macro_success_packet_rate_mbps=float(per.success_packet_rate_mbps.mean()),
        ue_macro_success_byte_outcome_rate_mbps=float(per.success_byte_outcome_rate_mbps.mean()),
        success_within_1ms_pct=float(100*(cohort.successfully_delivered&(cohort.system_time_ms<=1)).mean()),
        success_within_5ms_pct=float(100*(cohort.successfully_delivered&(cohort.system_time_ms<=5)).mean()),
        elapsed_native_s=execution['elapsed_native_s'],elapsed_summary_s=time.monotonic()-started)
    assert row['successful_bytes']==row['dequeue_bytes']-row['failed_bytes']
    assert row['start_backlog_bytes']+row['accepted_bytes']-row['expired_bytes']-row['dequeue_bytes']==row['end_backlog_bytes']
    for name,stats in execution['timing'].items():
        for key,v in stats.items(): row[name+'_'+key]=v
    pd.DataFrame([row]).to_csv(out/'metrics.csv',index=False)
    (out/'metrics.json.tmp').write_text(json.dumps(row,indent=2)+'\n')
    (out/'metrics.json.tmp').replace(out/'metrics.json')
    print('METRICS',out.name,row['goodput_mbps'],row['elapsed_summary_s'],flush=True)
    return row

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('out'); summarize(parser.parse_args().out)

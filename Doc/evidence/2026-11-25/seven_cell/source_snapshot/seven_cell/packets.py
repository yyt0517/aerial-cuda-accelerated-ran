"""Dimension-aware frozen FIFO/CRC reconstruction; same terminal matching."""
from collections import deque
import numpy as np
import pandas as pd

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
    for u in sorted(service.ue_id.unique().astype(int)):
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

"""New cross-load weight-grid runner; copied frozen native lifecycle, same solver."""
import json
from pathlib import Path
import time
import numpy as np
from .bounded_native import identifier
from .bounded_dpls import solve_bounded
from .no_ctbs_inputs import detach_raw_phy_row,problem_from_raw_row,raw_pfq_draft
from .ttl_weights import hol_urgency,objective_weights
from .analyze_dpls_moves import write_json,write_csv,sha

FEATURES=('queue_mean_norm','queue_P95_norm','active_fraction','urgency_mean','urgency_P95','ema_mean_norm')
COEFFICIENTS=(0.,1.,2.,4.,8.)
RATES=(.8,1.12,1.28)

def run_key(seed,rate,k,family='ttl'):
    return identifier(seed,rate,k) if family=='ttl' else identifier(seed,rate,0).replace('_k0_',f'_{family}_')

def state_features(queue,ema,urgency):
    return np.array([np.mean(queue)/2400.,np.quantile(queue,.95)/2400.,np.mean(np.asarray(queue)>0),
        np.mean(urgency),np.quantile(urgency,.95),np.mean(ema)/2400.])

def weights_for(family,base,ema,urgency,queue,coefficient):
    if family=='ttl':return objective_weights(base,urgency,coefficient)
    # Delay in TTI units. New arrivals have one-slot scheduling age; empty
    # queues get the same finite floor but never have a positive service term.
    age=np.maximum(1.,400.*np.asarray(urgency));a=-np.log(.01)/400.
    pf=1./np.maximum(np.asarray(ema,np.float64),1.e-9)
    if family=='mlwdf_pf':w=pf*(a*age)
    elif family=='exp_pf':
        active=np.asarray(queue)>0;x=float(np.mean((a*age)[active])) if active.any() else 0.
        w=pf*np.exp((a*age-x)/(1.+np.sqrt(x)))
    else:raise ValueError('Unknown weight family')
    w/=w.mean()
    if not np.isfinite(w).all() or (w<=0).any():raise ValueError('Invalid literature weights')
    return np.ascontiguousarray(w)

def one(output,seed,rate,coefficient,warmup,settling,measured,followup,save_every,family="ttl"):
    import pandas as pd
    import torch
    from .rate_native import RateNativeEnv
    from training.stageb_ppo.native import packets,write,REFERENCE
    from training.stageb_ppo.evaluate import metrics
    out=Path(output)/run_key(seed,rate,coefficient,family)
    env=RateNativeEnv(out,seed,warmup+settling+measured+followup,rate)
    launch=json.loads((out/'launch.json').read_text());launch.update(policy='capacity_bound_q1e6_v1',
        weight_family=family,urgency_coefficient=coefficient,urgency='1-HOL_slack/400, read-only causal FIFO',initializer='cold raw PFQ')
    write(out/'launch.json',launch)
    ema=np.zeros(36);records=[];saved=[];first=warmup+settling+1;last=first+measured-1
    if save_every:(out/'states').mkdir()
    try:
        for _ in range(env.horizon):
            if env.tti<warmup:action=env.raw_pfq()
            else:
                started=time.perf_counter();cpu_started=time.process_time()
                row=detach_raw_phy_row(dict(tti=env.tti+1,obs=env.obs,snapshot=env.snapshot,ema=ema.copy(),ctbs=env.inputs))
                p=problem_from_raw_row(row)
                urgency,slack=hol_urgency(env.fifo,env.tti+1,p.queue)
                p.weights=weights_for(family,p.weights,ema,urgency,p.queue,coefficient)
                features=state_features(p.queue,ema,urgency)
                draft=raw_pfq_draft(p,row['obs']);action,_,r=solve_bounded(p,draft)
                elapsed=1000*(time.perf_counter()-started);cpu_ms=1000*(time.process_time()-cpu_started)
                if not r['converged']:raise AssertionError('Certified native decision hit round cap')
                records.append(dict(tti=env.tti+1,**dict(zip(FEATURES,features)),wall_ms=elapsed,cpu_ms=cpu_ms,rounds=r['rounds'],
                    candidates=r['candidate_evaluations'],scored=r['scored_edits'],
                    full_candidate_count=r['sum_neighborhood_sizes']+r['initialization_evaluations'],
                    first_scan=r['first_scan_evaluations'],bound_updates=r['bound_updates'],
                    ue_evaluations=r['ue_evaluations'],mmse_requests=r['mmse_requests'],mmse_misses=r['mmse_misses']))
                if save_every and first<=env.tti+1<=last and (env.tti+1-first)%save_every==0:
                    row['objective_weights']=p.weights.copy();row['hol_urgency']=urgency.copy();row['hol_slack']=slack.copy();file=out/'states'/f'tti{env.tti+1:06d}.pt'
                    torch.save(row,file);saved.append(dict(path=str(file),sha256=sha(file)))
            env.step(action);ema=.995*ema+.005*np.asarray(env.info['actual_ue_goodput_bytes'])
            if env.tti%2000==0:print('BOUND_NATIVE',seed,rate,coefficient,env.tti,flush=True)
        env.close()
    except BaseException:
        if not env.log.closed:env.close(validate=False)
        raise
    write_csv(out/'search.csv',records);write_json(out/'saved_states.json',saved)
    if rate==.8 and warmup>=4000:
        ref=REFERENCE/f'seed{seed}_pfq0_20000'
        with np.load(ref/'prefix.npz') as b,np.load(out/'actions.npz') as a:
            np.testing.assert_array_equal(b['input_hashes'][:warmup],a['input_hashes'][:warmup])
            np.testing.assert_array_equal(b['actions'][:warmup].reshape(-1,3,17),a['actions'][:warmup])
        offered=pd.read_csv(out/'traffic_offered_packets.csv');expected=pd.read_csv(ref/'traffic_offered_packets.csv')
        pd.testing.assert_frame_equal(offered[offered.arrival_tti<=env.horizon].reset_index(drop=True),
            expected[expected.arrival_tti<=env.horizon].reset_index(drop=True))
    if followup<400:
        result=dict(seed=seed,rate=rate,coefficient=coefficient,smoke=True,native_passed=True)
    else:
        service,pk=packets(out);cohort=pk[pk.arrival_tti.between(first,last)]
        cohort.to_csv(out/'cohort.csv.gz',index=False);result=metrics(service,pk,first,last,seed,'bounded')
        per=[]
        for u in range(36):
            c=cohort[cohort.ue_id==u];successful=c[c.successfully_delivered]
            flags=service[(service.ue_id==u)&service.tti.between(first,last)].successful_bytes.to_numpy()>0
            longest=run=0
            for f in flags:run=0 if f else run+1;longest=max(longest,run)
            per.append(dict(ue=u,payload_mbps=float(successful.packet_size_bytes.sum()*8/(measured*.0005)/1e6),
                expiry_pct=float(100*c.outcome.eq('expired').mean()),packets=len(c),longest_no_successful_byte_ttis=longest))
        write_csv(out/'per_ue.csv',per)
        result.update(rate=rate,load_multiplier=rate/.8,coefficient=coefficient,
            worst_ue_expiry_pct=max(r['expiry_pct'] for r in per),zero_payload_UEs=sum(r['payload_mbps']==0 for r in per),
            TTL_long_gap_UEs=sum(r['longest_no_successful_byte_ttis']>=400 for r in per),native_passed=True)
    measured_records=[r for r in records if first<=r['tti']<=last]
    if measured_records:
        ms=np.array([r['wall_ms'] for r in measured_records]);c=np.array([r['candidates'] for r in measured_records])
        full=np.array([r['full_candidate_count'] for r in measured_records]);ratio=c/full
        result.update(wall_mean_ms=float(ms.mean()),wall_P95_ms=float(np.quantile(ms,.95)),wall_max_ms=float(ms.max()),
            candidates_mean=float(c.mean()),candidate_fraction_mean=float(ratio.mean()),
            candidate_fraction_P95=float(np.quantile(ratio,.95)),candidate_fraction_max=float(ratio.max()),
            candidate_work_ratio=float(c.sum()/full.sum()))
    result['weight_family']=family
    write(out/'summary.json',result)
    return dict(directory=str(out),**result)

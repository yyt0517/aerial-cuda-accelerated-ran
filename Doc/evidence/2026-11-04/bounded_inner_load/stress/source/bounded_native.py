"""Frozen native load-cost checks, then five-coefficient causal TTL frontier."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
import numpy as np
from .physics import ROOT
from .bounded_dpls import solve_bounded
from .no_ctbs_inputs import detach_raw_phy_row,problem_from_raw_row,raw_pfq_draft
from .ttl_weights import hol_urgency,objective_weights,COEFFICIENTS
from .analyze_dpls_moves import write_json,write_csv,sha

def identifier(seed,rate,k):
    return f"r{format(rate,'.12g').replace('.','p')}_k{format(k,'.12g').replace('.','p')}_seed{seed}"

def pin_worker(queue):os.sched_setaffinity(0,{queue.get()})

def one(output,seed,rate,coefficient,warmup,settling,measured,followup,save_every):
    import pandas as pd
    import torch
    from .rate_native import RateNativeEnv
    from training.stageb_ppo.native import packets,write,REFERENCE
    from training.stageb_ppo.evaluate import metrics
    out=Path(output)/identifier(seed,rate,coefficient)
    env=RateNativeEnv(out,seed,warmup+settling+measured+followup,rate)
    launch=json.loads((out/'launch.json').read_text());launch.update(policy='capacity_bound_q1e6_v1',
        urgency_coefficient=coefficient,urgency='1-HOL_slack/400, read-only causal FIFO',initializer='cold raw PFQ')
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
                if coefficient:
                    urgency,slack=hol_urgency(env.fifo,env.tti+1,p.queue)
                    p.weights=objective_weights(p.weights,urgency,coefficient)
                draft=raw_pfq_draft(p,row['obs']);action,_,r=solve_bounded(p,draft)
                elapsed=1000*(time.perf_counter()-started);cpu_ms=1000*(time.process_time()-cpu_started)
                if not r['converged']:raise AssertionError('Certified native decision hit round cap')
                records.append(dict(tti=env.tti+1,wall_ms=elapsed,cpu_ms=cpu_ms,rounds=r['rounds'],
                    candidates=r['candidate_evaluations'],scored=r['scored_edits'],
                    full_candidate_count=r['sum_neighborhood_sizes']+r['initialization_evaluations'],
                    first_scan=r['first_scan_evaluations'],bound_updates=r['bound_updates'],
                    ue_evaluations=r['ue_evaluations'],mmse_requests=r['mmse_requests'],mmse_misses=r['mmse_misses']))
                if save_every and first<=env.tti+1<=last and (env.tti+1-first)%save_every==0:
                    row['objective_weights']=p.weights.copy();file=out/'states'/f'tti{env.tti+1:06d}.pt'
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
    write(out/'summary.json',result)
    return dict(directory=str(out),**result)

def main(args):
    gate=Path(args.gate).resolve();g=json.loads((gate/'completion.json').read_text())
    if not (g['equivalence_passed'] and g['wall_passed']):raise ValueError('Exact/wall gate must pass first')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    if args.mode=='smoke':
        specs=[(41,.8,0.),(41,1.28,0.),(41,.8,1.)];warmup,settling,measured,followup=8,0,8,0;save=0
    elif args.mode=='stress':
        specs=[(s,r,0.) for r in (.8,1.12,1.28) for s in (41,42)]
        warmup=settling=measured=4000;followup=400;save=100
        if args.workers!=1:raise ValueError('Stress perTTI latency requires serial execution')
    else:
        stress=Path(args.stress).resolve();s=json.loads((stress/'completion.json').read_text())
        if not s['B_entry_passed']:raise ValueError('Stress equivalence/cost entry not passed')
        specs=[(seed,.8,k) for k in COEFFICIENTS for seed in (41,42,43) if k!=0 or seed==43]
        warmup=settling=measured=4000;followup=400;save=1000
    files=[Path(__file__),*[Path(__file__).with_name(n) for n in ('rate_native.py','ttl_weights.py','bounded_dpls.py',
        'bounded_kernel.cpp','_bounded.so','persistent_kernel.cpp','physics_kernel.cpp','no_ctbs_inputs.py')]]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files};(out/'source').mkdir()
    for f in files:shutil.copy2(f,out/'source'/f.name)
    write_json(out/'source_manifest.json',hashes)
    protocol=dict(mode=args.mode,seeds=[41,42] if args.mode=='stress' else [41,42,43],
        jobs=[dict(seed=s,rate=r,coefficient=k) for s,r,k in specs],
        warmup=warmup,settling=settling,measured=measured,followup=followup,
        initializer='cold raw PFQ; fixed causal deficit weights at coefficient0',
        weights='base deficit weights*(1+k*(1-HOL_slack/400)), normalized mean1; k0 preserves exact baseline bits',
        feature='causal FIFO remaining head, read without mutating expiry accounting',
        baseline_rate=.8,load_multipliers=[1,1.4,1.6],coefficients=list(COEFFICIENTS),
        selection='single grid frozen before outcomes; no online training or coefficient reselection',
        gate=str(gate),workers=args.workers,cpu_pins=list(range(args.cpu,args.cpu+args.workers)),
        timing='serial perTTI cost including causal adaptation/PFQ/context/search/close' if args.mode=='stress' else
            'parallel quality trajectories; elapsed policy times are not fair speed measurements',
        scientific_scope='training41--43 development; no final confirmation seeds')
    reuse=[]
    if args.mode=='ttl':
        stress_source=Path(args.stress_source).resolve()
        for seed in (41,42):
            d=stress_source/identifier(seed,.8,0.)
            reuse.append(dict(directory=str(d),**json.loads((d/'summary.json').read_text())))
        protocol['reused_zero_coefficient']=reuse
    write_json(out/'protocol.json',protocol)
    ctx=mp.get_context('spawn');queue=ctx.Queue()
    for cpu in range(args.cpu,args.cpu+args.workers):queue.put(cpu)
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=ctx,initializer=pin_worker,initargs=(queue,)) as pool:
        futures=[pool.submit(one,str(out),seed,rate,k,warmup,settling,measured,followup,save) for seed,rate,k in specs]
        for f in as_completed(futures):results.append(f.result())
    for f,digest in hashes.items():assert sha(ROOT/f)==digest
    write_json(out/'completion.json',dict(completed=True,new_native_runs=len(results),
        new_native_steps=len(results)*(warmup+settling+measured+followup),runs=results,reused_runs=reuse,
        training_steps=0,mode=args.mode))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('smoke','stress','ttl'),required=True)
    p.add_argument('--gate',default='deliverables/bounded_gate_wall_20261104_v1');p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--cpu',type=int,default=255)
    p.add_argument('--stress');p.add_argument('--stress-source');main(p.parse_args())

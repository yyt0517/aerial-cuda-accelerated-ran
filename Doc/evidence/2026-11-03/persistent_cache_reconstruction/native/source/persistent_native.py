"""Conditional two-seed two-second native development, no current teacher."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
from .physics import ROOT
from .no_ctbs_inputs import detach_raw_phy_row,problem_from_raw_row,raw_pfq_draft
from .persistent_dpls import solve_persistent,PersistentConfig
from .analyze_dpls_moves import write_json,write_csv,sha


def one(output,seed,mode,warmup,settling,measured,followup):
    from training.stageb_ppo.native import NativeEnv,packets,write,REFERENCE
    from training.stageb_ppo.evaluate import metrics
    import pandas as pd
    out=Path(output)/f'{mode}_seed{seed}'
    env=NativeEnv(out,seed,warmup+settling+measured+followup)
    launch=json.loads((out/'launch.json').read_text())
    launch.update(business_weights=True,policy='persistent_q1e6_v1',mode=mode,initialization='cold raw PFQ',
                  current_teacher_calls=0)
    write(out/'launch.json',launch)
    ema=np.zeros(36);records=[]
    try:
        for _ in range(env.horizon):
            if env.tti<warmup:action=env.raw_pfq()
            else:
                row=detach_raw_phy_row(dict(tti=env.tti+1,obs=env.obs,snapshot=env.snapshot,ema=ema.copy(),ctbs=env.inputs))
                p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs'])
                action,_,r=solve_persistent(p,draft,PersistentConfig(mode))
                if not r['converged']:raise AssertionError('Native decision hit round cap')
                records.append(dict(tti=env.tti+1,rounds=r['rounds'],candidates=r['candidate_evaluations'],
                    scored=r['scored_edits'],generated=r['generated_edits'],guard_rejections=r['guard_rejections'],
                    accepted_batches=r['accepted_batches'],score_q=r['score_q']))
            env.step(action)
            ema=.995*ema+.005*np.asarray(env.info['actual_ue_goodput_bytes'])
            if env.tti%2000==0:print('NATIVE',seed,mode,env.tti,flush=True)
        env.close()
    except BaseException:
        if not env.log.closed:env.close(validate=False)
        raise
    service,pk=packets(out);first=warmup+settling+1;last=first+measured-1
    ref=REFERENCE/f'seed{seed}_pfq0_20000'
    with np.load(ref/'prefix.npz') as before,np.load(out/'actions.npz') as after:
        np.testing.assert_array_equal(before['input_hashes'][:warmup],after['input_hashes'][:warmup])
        np.testing.assert_array_equal(before['actions'][:warmup].reshape(-1,3,17),after['actions'][:warmup])
    offered=pd.read_csv(out/'traffic_offered_packets.csv');expected=pd.read_csv(ref/'traffic_offered_packets.csv')
    pd.testing.assert_frame_equal(offered[offered.arrival_tti<=env.horizon].reset_index(drop=True),
        expected[expected.arrival_tti<=env.horizon].reset_index(drop=True))
    write_csv(out/'search.csv',records)
    if followup<400:
        result=dict(seed=seed,mode=mode,smoke=True,native_passed=True,cohort_evaluated=False)
    else:
        cohort=pk[pk.arrival_tti.between(first,last)];cohort.to_csv(out/'cohort.csv.gz',index=False)
        result=metrics(service,pk,first,last,seed,mode)
        per=[]
        seconds=measured*.0005
        for ue in range(36):
            c=cohort[cohort.ue_id==ue];success=c[c.successfully_delivered]
            s=service[(service.ue_id==ue)&service.tti.between(first,last)].sort_values('tti')
            flags=s.successful_bytes.to_numpy()>0
            longest=run=0
            for served in flags:
                run=0 if served else run+1
                longest=max(longest,run)
            per.append(dict(ue=ue,payload_mbps=float(success.packet_size_bytes.sum()*8/seconds/1e6),
                expiry_pct=float(100*c.outcome.eq('expired').mean()),packets=len(c),
                longest_no_successful_byte_ttis=longest))
        write_csv(out/'per_ue.csv',per)
        result.update(worst_ue_expiry_pct=max(r['expiry_pct'] for r in per),
            zero_complete_payload_ues=sum(r['payload_mbps']==0 for r in per),
            UEs_with_TTL_long_no_service=sum(r['longest_no_successful_byte_ttis']>=400 for r in per),
            native_passed=True)
    measured_records=[r for r in records if first<=r['tti']<=last]
    if measured_records:
        result.update(mean_candidate_evaluations=float(np.mean([r['candidates'] for r in measured_records])),
            p95_candidate_evaluations=float(np.quantile([r['candidates'] for r in measured_records],.95)),
            mean_generated_edits=float(np.mean([r['generated'] for r in measured_records])),
            guard_rejections=sum(r['guard_rejections'] for r in measured_records),
            accepted_batches=sum(r['accepted_batches'] for r in measured_records))
    write_json(out/'summary.json',result)
    return result


def main(args):
    gate=Path(args.gate).resolve();out=Path(args.output).resolve()
    completed=json.loads((gate/'completion.json').read_text())
    if not completed['integer_equivalence_passed'] or not completed['native_entry_authorized_by_gate']:
        raise ValueError('Persistent integer/guard envelope entry gates not passed')
    out.mkdir(parents=True,exist_ok=False)
    if args.smoke:seeds,modes,warmup,settling,measured,followup=[41],['full','guarded4'],8,0,8,0
    else:seeds,modes,warmup,settling,measured,followup=[44,45],['full','guarded4'],4000,4000,4000,400
    files=[Path(__file__),*[Path(__file__).with_name(n) for n in ('persistent_dpls.py','persistent_kernel.cpp','_persistent.so',
        'physics_kernel.cpp','physics.py','no_ctbs_inputs.py')]]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files};(out/'source').mkdir()
    import shutil
    for p in files:shutil.copy2(p,out/'source'/p.name)
    write_json(out/'source_manifest.json',hashes)
    write_json(out/'protocol.json',dict(seeds=seeds,modes=modes,warmup=warmup,settling=settling,measured=measured,
        followup=followup,measured_seconds=measured*.0005,gate=str(gate),workers=args.workers,
        scene='boundary/rate0.8 noCTBS causal deficit',initializer='cold raw PFQ, same integer objective both arms',
        weak_guard='causal deficit-top20%, compare batch vs best single; no current full teacher',
        sampling='exposed44/45 development, mature arrival cohort; not final new-seed confirmation',
        timing='concurrent native trajectories; native elapsed/policy times not fair speed measurement',
        selection='one guarded configuration frozen before native outcomes; no retuning',
        numerical='same 1e-6B integer contributions, strict1/16B threshold and canonical one-unit tie'))
    write_json(out/'quality_protocol.json',dict(payload_paired_mean_min_ratio=.997,payload_each_seed_min_ratio=.994,
        expiry_max_increase_pp=.10,mean_success_delay_max_increase_ms=.5,p95_success_delay_max_increase_ms=1.,
        ue_payload_P5_min_ratio=.97,worst_ue_expiry='<=max(reference worst+2 percentage points,5%)',
        newly_zero_complete_payload_ues=0,newly_TTL_long_service_gap_ues=0,
        TTL_gap='400 consecutive measurement TTIs without successful bytes =200ms; report actual perUE maxima',
        mature_cohort=True,packet_metrics='same offered arrival window8001--12000 with400TTI followup',
        costs='report paired candidate counts; fair wall time remains the separate serial saved-input benchmark',
        inference='two exposed development seeds; no final confidence/generalization claim'))
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        jobs=[pool.submit(one,str(out),s,m,warmup,settling,measured,followup) for s in seeds for m in modes]
        for job in as_completed(jobs):
            results.append(job.result())
    for name,digest in hashes.items():
        if sha(ROOT/name)!=digest:raise AssertionError('Native source changed: '+name)
    write_json(out/'completion.json',dict(completed=True,runs=results,native_runs=len(results),
        native_steps=len(results)*(warmup+settling+measured+followup),smoke=args.smoke))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gate',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--smoke',action='store_true');main(p.parse_args())

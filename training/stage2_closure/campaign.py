"""Frozen ten-seed closure experiment, native simulator and exact solver unchanged."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

import numpy as np
from training.stageb_ppo.native import ROOT, FORMAL, EXPECTED_SIMULATOR, write, sha, packets
from training.stageb_lns.no_ctbs_inputs import detach_raw_phy_row, problem_from_raw_row, raw_pfq_draft
from training.stageb_lns.bounded_dpls import solve_bounded
from training.stageb_lns.ttl_weights import hol_urgency
from training.stageb_lns.crossload_native import weights_for
from .native import ReportNativeEnv

WARMUP, SETTLING, MEASURED, FOLLOWUP = 4000, 4000, 20000, 400
FIRST, LAST = 8001, 28000
SEEDS = tuple(range(41, 51))

def key(job):
    return f"{job['scene']}_r{job['rate']:g}_{job['arm']}_s{job['seed']}"

def jobs():
    # E1, E3, E4 and E5. E2/E6 are optional extensions, not implicit completed work.
    result=[]
    for scene,rate,arms in [('boundary',.8,['pfq','rr','rrq','pf','equal','deficit','ttl1','ttl2','exp_pf','mlwdf_pf']),
                            ('uniform',1.,['pfq','rr','rrq','pf','deficit','mlwdf_pf']),
                            ('boundary',1.12,['pfq','deficit','mlwdf_pf']),
                            ('boundary',1.28,['pfq','deficit','mlwdf_pf'])]:
        for arm in arms:
            for seed in SEEDS:result.append(dict(scene=scene,rate=rate,arm=arm,seed=seed))
    priority={'mlwdf_pf':0,'deficit':1,'pfq':2,'equal':3,'ttl1':4,'ttl2':5,'exp_pf':6,'rr':7,'rrq':8,'pf':9}
    return sorted(result,key=lambda j:(j['rate']>1.,priority[j['arm']],j['scene'],j['seed']))

def legacy_pfq(job):
    if job['arm']!='pfq' or job['seed']>45:return None
    name={'boundary':'boundary08','uniform':'uniform10'}[job['scene']]
    if job['rate']!=({'boundary':.8,'uniform':1.}[job['scene']]):return None
    p=FORMAL/f'{name}_steady_state_20261021_v1'/f"seed{job['seed']}_pfq0_20000"
    launch=json.loads((p/'launch.json').read_text())
    assert launch['horizon']==28400 and launch['simulator_sha256']==EXPECTED_SIMULATOR
    assert launch['env']['CUMAC_SHADOW_FADING_STD_DB']=='3.0'
    assert json.loads((p/'validation.json').read_text())['passed']
    return p

def baseline_action(env,arm,p,ema,cursors):
    """Explicit reproducible textbook baselines; no native-baseline identity claim.

    RR: rotate positive-queue users; do not cap allocation by queue size.
    RRQ: same rotation, stop allocating to a UE once fixed-MCS TBS covers Q.
    PF: maximize instantaneous sum-layer log2(1+SINR) / successful-byte EMA
    independently per PRG; no queue-sized cap. Native PFQ is exported unchanged.
    """
    action=np.full((3,17),-1,np.int64)
    for cell in range(3):
        users=np.flatnonzero((p.serving==cell)&(p.queue>0))
        if not len(users):continue
        if arm=='pf':
            gamma=np.asarray(env.obs['post_eq_sinr'])[users]
            layers=np.arange(4)[None,None,:]<p.caps[users,:,None]
            rate=(np.log2(1+np.maximum(gamma,0))*layers).sum(-1)
            score=rate/np.maximum(ema[users,None],1e-9)
            for prg in range(17):
                legal=p.legal[cell,prg,users]
                if legal.any():action[cell,prg]=users[np.argmax(np.where(legal,score[:,prg],-np.inf))]
            continue
        order=np.roll(users,-(int(cursors[cell])%len(users)))
        counts=np.zeros(p.U,int);minlayers=np.full(p.U,4,int);cursor=0
        for prg in range(17):
            for probe in range(len(order)):
                i=(cursor+probe)%len(order);u=order[i]
                if not p.legal[cell,prg,u]:continue
                if arm=='rrq' and counts[u]>0 and p.tbs[u,counts[u],minlayers[u]]>=p.queue[u]:continue
                action[cell,prg]=u;counts[u]+=1;minlayers[u]=min(minlayers[u],p.caps[u,prg]);cursor=(i+1)%len(order);break
        cursors[cell]+=1
    return action

def one(root,job,smoke=False):
    import torch
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled=False
    # Quality trajectories may run concurrently; their wall times are descriptive.
    affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{affinity[os.getpid()%len(affinity)]})
    out=Path(root)/key(job)
    if (out/'done.json').exists():return json.loads((out/'done.json').read_text())
    if not smoke:
        old=legacy_pfq(job)
        if old:
            out.mkdir(exist_ok=False)
            result=dict(**job,reused=True,directory=str(old),key=key(job),
                        input_sha256={n:sha(old/n) for n in ['launch.json','validation.json','metrics.bin','prefix.npz','traffic_offered_packets.csv','packet_outcome_samples.csv']})
            write(out/'done.json',result);return result
    warmup,settling,measured,followup=(16,0,16,0) if smoke else (WARMUP,SETTLING,MEASURED,FOLLOWUP)
    first,last=warmup+settling+1,warmup+settling+measured
    env=ReportNativeEnv(out,job['seed'],warmup+settling+measured+followup,job['rate'],
                        'uniform' if job['scene']=='uniform' else 'coop_boundary')
    launch=json.loads((out/'launch.json').read_text());launch.update(job,common_pfq_warmup=warmup,
        settling=settling,measured=measured,followup=followup,measurement_first_tti=first,measurement_last_tti=last,
        report_protocol='stage2_20261104_rawPFQ_exact_v1',ctbs_action_clipping=False)
    write(out/'launch.json',launch)
    ema=np.zeros(36);records=[];states=[];cursors=np.zeros(3,int)
    source_digest=sha(Path(__file__));(out/'states').mkdir()
    try:
        for _ in range(env.horizon):
            t=env.tti+1
            if t<=warmup or job['arm']=='pfq':action=env.raw_pfq()
            else:
                start=time.perf_counter();cpu=time.process_time()
                row=detach_raw_phy_row(dict(tti=t,obs=env.obs,snapshot=env.snapshot,ema=ema.copy(),ctbs=env.inputs))
                p=problem_from_raw_row(row)
                if job['arm'] in ('rr','rrq','pf'):
                    action=baseline_action(env,job['arm'],p,ema,cursors);r={}
                else:
                    urgency,slack=hol_urgency(env.fifo,t,p.queue)
                    arm=job['arm']
                    if arm=='equal':p.weights=np.ones(36)
                    elif arm.startswith('ttl'):p.weights=weights_for('ttl',p.weights,ema,urgency,p.queue,float(arm[3:]))
                    elif arm in ('mlwdf_pf','exp_pf'):p.weights=weights_for(arm,p.weights,ema,urgency,p.queue,0.)
                    elif arm!='deficit':raise ValueError(arm)
                    draft=raw_pfq_draft(p,row['obs']);action,values,r=solve_bounded(p,draft)
                    assert r['converged']
                    if t in ({first,first+measured//2,last} if not smoke else {first}):
                        row.update(objective_weights=p.weights.copy(),hol_urgency=urgency,hol_slack=slack,
                                   selected_action=action.copy(),selected_values=values.copy())
                        file=out/'states'/f'tti{t:06d}.pt';torch.save(row,file);states.append(dict(path=str(file),sha256=sha(file)))
                records.append(dict(tti=t,wall_ms=(time.perf_counter()-start)*1000,
                    cpu_ms=(time.process_time()-cpu)*1000,**{k:v for k,v in r.items() if k not in ('core_ms','version')}))
            env.step(action);ema=.995*ema+.005*np.asarray(env.info['actual_ue_goodput_bytes'])
        env.close()
    except BaseException:
        if not env.log.closed:env.close(validate=False)
        raise
    assert source_digest==sha(Path(__file__))
    import pandas as pd
    pd.DataFrame(records).to_csv(out/'search.csv.gz',index=False)
    write(out/'saved_states.json',states)
    result=dict(**job,reused=False,directory=str(out),key=key(job),native_passed=True,smoke=smoke)
    # Cohorts are reconstructed by the separate analysis process, not hidden in run time.
    write(out/'done.json',result)
    return result

def main(args):
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    alljobs=jobs()
    if args.smoke:alljobs=[dict(scene=s,rate=r,arm=a,seed=41) for s,r,a in
        [('boundary',.8,'mlwdf_pf'),('uniform',1.,'mlwdf_pf'),('boundary',.8,'rr'),('boundary',.8,'rrq'),('boundary',.8,'pf'),('boundary',1.28,'deficit')]]
    if args.arms:alljobs=[j for j in alljobs if j['arm'] in args.arms.split(',')]
    # Freeze before outcomes; resumes reject changes.
    source_files=[Path(__file__).with_name(n) for n in ('campaign.py','native.py','__init__.py')]+[
        ROOT/'training/stageb_lns'/n for n in ('_bounded.so','_persistent.so','bounded_dpls.py','persistent_dpls.py','physics.py','no_ctbs_inputs.py','crossload_native.py','ttl_weights.py')]
    protocol=dict(version='stage2_closure_v1',seeds=list(SEEDS),jobs=alljobs,smoke=args.smoke,
        windows=dict(warmup=WARMUP,settling=SETTLING,measured=MEASURED,followup=FOLLOWUP),
        sources={str(p.relative_to(ROOT)):sha(p) for p in source_files},simulator=EXPECTED_SIMULATOR,
        statistical_unit='seed; equal seed means; paired t CI; separate 44-50 robustness subset, not pristine unseen seeds',
        timing='parallel trajectories descriptive only; serial saved-state cost measured separately',
        variants='all search arms raw PFQ initialization and 90% raw-PFQ dequeue floor; no CTBS scheduler',
        execution=dict(mps_pipe=os.environ.get('CUDA_MPS_PIPE_DIRECTORY'),cuda_connections=os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS')))
    if (root/'protocol.json').exists():assert json.loads((root/'protocol.json').read_text())==protocol
    else:write(root/'protocol.json',protocol)
    completed=[];failed=[]
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        pending={pool.submit(one,str(root),j,args.smoke):j for j in alljobs}
        for future in as_completed(pending):
            job=pending[future]
            try:
                completed.append(future.result());print('DONE',key(job),len(completed),'/',len(alljobs),flush=True)
            except BaseException:
                detail=dict(job=job,error=traceback.format_exc());failed.append(detail);print('FAILED',detail,flush=True)
            write(root/'progress_campaign.json',dict(completed=completed,failed=failed,total=len(alljobs)))
    write(root/'completion.json',dict(completed=not failed,runs=completed,failed=failed,total=len(alljobs)))
    if failed:raise RuntimeError(f'{len(failed)} experiments failed')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--arms');main(parser.parse_args())

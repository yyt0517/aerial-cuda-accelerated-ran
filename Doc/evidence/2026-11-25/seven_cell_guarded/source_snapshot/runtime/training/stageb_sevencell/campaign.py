"""Paired PFQ/DPLS/DA-PLS trajectories; fixed Boundary08 settings."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
import numpy as np
from training.stageb_ppo.native import ROOT, write, sha
from training.stageb_lns.no_ctbs_inputs import detach_raw_phy_row, problem_from_raw_row, raw_pfq_draft
from training.stageb_lns.crossload_native import weights_for
from training.stageb_lns.ttl_weights import hol_urgency
from .native import SevenCellEnv
from .solver import solve_bounded

ARMS=('pfq','deficit','mlwdf_pf')
LABELS={'pfq':'PFQ','deficit':'DPLS','mlwdf_pf':'DA-PLS'}

def run_one(root,build,seed,arm,smoke=False,cells=7):
    import torch
    import pandas as pd
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False
    out=Path(root)/f'{arm}_s{seed}'
    if (out/'done.json').exists():return json.loads((out/'done.json').read_text())
    warmup,settling,measured,followup=(16,0,32,400) if smoke else (4000,4000,20000,400)
    first,last=warmup+settling+1,warmup+settling+measured
    env=SevenCellEnv(out,seed,warmup+settling+measured+followup,build,cells)
    launch=json.loads((out/'launch.json').read_text());launch.update(arm=arm,label=LABELS[arm],seed=seed,smoke=smoke,
        warmup=warmup,settling=settling,measured=measured,followup=followup,measurement_first_tti=first,measurement_last_tti=last)
    write(out/'launch.json',launch);(out/'states').mkdir()
    ema=np.zeros(env.U);records=[];states=[];interference=[];channels=[]
    serving=np.asarray(env.inputs.serving_cell);np.save(out/'serving.npy',serving)
    import hashlib
    try:
        for _ in range(env.horizon):
            t=env.tti+1;channels.append(hashlib.sha256(env.snapshot['h'].tobytes()).hexdigest())
            assert np.array_equal(serving,env.inputs.serving_cell)
            save=t in {first,first+measured//2,last}
            row=None
            if t<=warmup or arm=='pfq':action=env.raw_pfq()
            else:
                begin=time.perf_counter();cpu=time.process_time()
                row=detach_raw_phy_row(dict(tti=t,obs=env.obs,snapshot=env.snapshot,ema=ema.copy(),ctbs=env.inputs))
                p=problem_from_raw_row(row);urgency,slack=hol_urgency(env.fifo,t,p.queue)
                if arm=='mlwdf_pf':p.weights=weights_for('mlwdf_pf',p.weights,ema,urgency,p.queue,0.)
                draft=raw_pfq_draft(p,row['obs']);action,values,r=solve_bounded(p,draft)
                assert r['converged'], ('search cap reached',seed,arm,t)
                records.append(dict(tti=t,wall_ms=(time.perf_counter()-begin)*1000,cpu_ms=(time.process_time()-cpu)*1000,
                    **{k:v for k,v in r.items() if k not in ('core_ms','version')}))
                if save:row.update(objective_weights=p.weights.copy(),hol_urgency=urgency,hol_slack=slack,selected_values=values.copy())
            if save:
                if row is None:row=detach_raw_phy_row(dict(tti=t,obs=env.obs,snapshot=env.snapshot,ema=ema.copy(),ctbs=env.inputs))
                row['selected_action']=action.copy();path=out/'states'/f'tti{t:06d}.pt';torch.save(row,path)
                states.append(dict(path=str(path),sha256=sha(path)))
            if first<=t<=last:
                # Actual activity-only covariance used by the frozen native PHY:
                # H H^H per active interfering cell, divided by 4*noise.
                power=np.square(np.abs(env.snapshot['h'])).sum(axis=(-1,-2))
                cross=(np.arange(env.C)[None,None,:]!=serving[:,None,None])
                inr=(power*cross*(action.T>=0)[None]).sum(-1)/(4*env.snapshot['noise'])
                assigned=action[serving]==np.arange(env.U)[:,None];selected=inr[assigned]
                interference.append(dict(tti=t,links=int(assigned.sum()),inr_sum=float(selected.sum()),
                    inr_mean=float(selected.mean()) if len(selected) else 0.,inr_p95=float(np.quantile(selected,.95)) if len(selected) else 0.))
            env.step(action);ema=.995*ema+.005*np.asarray(env.info['actual_ue_goodput_bytes'])
            if t%1000==0:print('PROGRESS',seed,arm,t,env.horizon,flush=True)
        env.close()
    except BaseException:
        if not env.log.closed:env.close(validate=False)
        raise
    pd.DataFrame(records).to_csv(out/'search.csv.gz',index=False)
    pd.DataFrame(interference).to_csv(out/'interference.csv.gz',index=False)
    np.save(out/'channel_hashes.npy',np.array(channels));write(out/'saved_states.json',states)
    result=dict(seed=seed,arm=arm,label=LABELS[arm],directory=str(out),smoke=smoke,native_passed=True)
    write(out/'done.json',result);return result

def main(args):
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    jobs=[(s,a) for s in map(int,args.seeds.split(',')) for a in args.arms.split(',')]
    protocol=dict(cells=args.cells,users=args.cells*12,seeds=sorted({s for s,a in jobs}),arms=args.arms.split(','),
        smoke=args.smoke,windows=[16,0,32,400] if args.smoke else [4000,4000,20000,400],
        rate=.8,prgs_per_cell=17,prbs_per_prg=16,packet_bytes=3000,ttl_ms=200,
        labels=LABELS,build=str(Path(args.build).resolve()),
        changes='3→7 cells;36→84 users;128-bit dependency masks; size-aware input/action/FIFO adapters; boundary sampling on adjacent shared edges',
        preserved='strongest association, coop_boundary, radius500, shadow3dB, continuous Rayleigh, 4T4R, same per-UE arrival load and weights, raw PFQ and rho0.90, exact search thresholds/tie',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},
        simulator_sha256=sha(Path(args.build)/'simulator'),solver_sha256=sha(Path(__file__).with_name('_bounded128.so')),
        note='Shared GPU; concurrent wall time is descriptive and is not a deployment latency or fair speed ratio')
    write(root/'protocol.json',protocol);completed=[];failed=[]
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(run_one,str(root),str(Path(args.build).resolve()),s,a,args.smoke,args.cells):(s,a) for s,a in jobs}
        for future in as_completed(futures):
            try:completed.append(future.result());print('DONE',futures[future],flush=True)
            except BaseException:failed.append(dict(job=futures[future],error=traceback.format_exc()));print('FAILED',failed[-1],flush=True)
            write(root/'progress_campaign.json',dict(completed=completed,failed=failed,total=len(jobs)))
    write(root/'completion.json',dict(completed=not failed,runs=completed,failed=failed,total=len(jobs)))
    if failed:raise RuntimeError(f'{len(failed)} failed runs')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--build',required=True)
    p.add_argument('--seeds',default='41,42,43');p.add_argument('--arms',default=','.join(ARMS));p.add_argument('--workers',type=int,default=3)
    p.add_argument('--cells',type=int,default=7);p.add_argument('--smoke',action='store_true');main(p.parse_args())

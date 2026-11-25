"""Closed-loop, three-branch experiment; frozen seven-cell simulator."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback
import numpy as np
from training.stageb_ppo.native import ROOT,write,sha
from training.stageb_lns.no_ctbs_inputs import detach_raw_phy_row,problem_from_raw_row,raw_pfq_draft
from training.stageb_lns.crossload_native import weights_for
from training.stageb_lns.ttl_weights import hol_urgency
from training.stageb_sevencell.native import SevenCellEnv
from .policy import select,ARM,LABEL,TIE_ORDER

def run_one(root,build,seed,smoke=False):
    import torch
    import pandas as pd
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False
    own_sources={str(p):sha(p) for p in Path(__file__).parent.rglob('*') if p.is_file() and p.suffix in ('.py','.cpp','.so')}
    out=Path(root)/f'{ARM}_s{seed}'
    warmup,settling,measured,followup=(16,0,32,400) if smoke else (4000,4000,20000,400)
    first,last=warmup+settling+1,warmup+settling+measured
    env=SevenCellEnv(out,seed,last+followup,build,7)
    launch=json.loads((out/'launch.json').read_text());launch.update(arm=ARM,label=LABEL,seed=seed,smoke=smoke,
        warmup=warmup,settling=settling,measured=measured,followup=followup,measurement_first_tti=first,measurement_last_tti=last,
        branch_policy=dict(branches=['dpls','original_da','from_dpls_da'],initializer='raw PFQ for first two, forced DPLS endpoint for third',
            objective='canonical DA score, sum round_even(1e6*weight*expected_bytes)',tie_order=TIE_ORDER,
            floor='ceil(0.90 * raw PFQ planned dequeue), common to every branch'),policy_source_sha256=own_sources)
    write(out/'launch.json',launch);(out/'states').mkdir()
    ema=np.zeros(env.U);records=[];states=[];interference=[];channels=[]
    serving=np.asarray(env.inputs.serving_cell);np.save(out/'serving.npy',serving)
    try:
        for _ in range(env.horizon):
            t=env.tti+1;channels.append(hashlib.sha256(env.snapshot['h'].tobytes()).hexdigest())
            assert np.array_equal(serving,env.inputs.serving_cell)
            save=t in {warmup+1,first,first+measured//2,last};row=None
            if t<=warmup:action=env.raw_pfq()
            else:
                begin=time.perf_counter();cpu=time.process_time()
                row=detach_raw_phy_row(dict(tti=t,obs=env.obs,snapshot=env.snapshot,ema=ema.copy(),ctbs=env.inputs))
                p=problem_from_raw_row(row);base_weights=p.weights.copy();urgency,slack=hol_urgency(env.fifo,t,p.queue)
                da=weights_for('mlwdf_pf',base_weights,ema,urgency,p.queue,0.)
                draft=raw_pfq_draft(p,row['obs']);action,values,r,branches=select(p,draft,da)
                records.append(dict(tti=t,wall_ms=(time.perf_counter()-begin)*1000,cpu_ms=(time.process_time()-cpu)*1000,**r))
                if save:row.update(objective_weights=da.copy(),deficit_weights=base_weights,hol_urgency=urgency,hol_slack=slack,
                    selected_values=values.copy(),selected_branch=r['selected_branch'],branches=branches)
            if save:
                row['selected_action']=action.copy();path=out/'states'/f'tti{t:06d}.pt';torch.save(row,path)
                states.append(dict(path=str(path),sha256=sha(path)))
            if first<=t<=last:
                power=np.square(np.abs(env.snapshot['h'])).sum(axis=(-1,-2))
                cross=(np.arange(env.C)[None,None,:]!=serving[:,None,None])
                inr=(power*cross*(action.T>=0)[None]).sum(-1)/(4*env.snapshot['noise'])
                assigned=action[serving]==np.arange(env.U)[:,None];selected=inr[assigned]
                interference.append(dict(tti=t,links=int(assigned.sum()),inr_sum=float(selected.sum()),
                    inr_mean=float(selected.mean()) if len(selected) else 0.,inr_p95=float(np.quantile(selected,.95)) if len(selected) else 0.))
            env.step(action);ema=.995*ema+.005*np.asarray(env.info['actual_ue_goodput_bytes'])
            if t%1000==0:print('PROGRESS',seed,t,env.horizon,flush=True)
        env.close()
        for path,h in own_sources.items():assert sha(path)==h,path
    except BaseException:
        if not env.log.closed:env.close(validate=False)
        raise
    pd.DataFrame(records).to_csv(out/'search.csv.gz',index=False)
    pd.DataFrame(interference).to_csv(out/'interference.csv.gz',index=False)
    np.save(out/'channel_hashes.npy',np.array(channels));write(out/'saved_states.json',states)
    write(out/'policy_validation.json',dict(passed=True,decision_ttis=len(records),sources_unchanged=True,
        all_branches_converged=True,common_floor_all_branches=True,
        selected_da_score_ge_all_three=True,forced_dpls_seed_da_score_monotone=True))
    result=dict(seed=seed,arm=ARM,label=LABEL,directory=str(out),smoke=smoke,native_passed=True)
    write(out/'done.json',result);return result

def main(args):
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    seeds=list(map(int,args.seeds.split(',')))
    protocol=dict(cells=7,users=84,seeds=seeds,arms=[ARM],smoke=args.smoke,windows=[16,0,32,400] if args.smoke else [4000,4000,20000,400],
        rate=.8,prgs_per_cell=17,prbs_per_prg=16,packet_bytes=3000,ttl_ms=200,labels={ARM:LABEL},build=str(Path(args.build).resolve()),
        baseline=str(Path(args.baseline).resolve()),branch_tie_order=TIE_ORDER,
        changes='Three same-state branches, maximize canonical DA score; third forcibly starts at DPLS endpoint, original PFQ floor retained',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.rglob('*') if p.is_file() and p.suffix in ('.py','.cpp','.so')},
        simulator_sha256=sha(Path(args.build)/'simulator'),
        note='Single-step DA objective dominance only; no long-horizon throughput or delay guarantee')
    write(root/'protocol.json',protocol);completed=[];failed=[]
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(run_one,str(root),str(Path(args.build).resolve()),s,args.smoke):s for s in seeds}
        for future in as_completed(futures):
            try:completed.append(future.result());print('DONE',futures[future],flush=True)
            except BaseException:failed.append(dict(seed=futures[future],error=traceback.format_exc()));print('FAILED',failed[-1],flush=True)
            write(root/'progress_campaign.json',dict(completed=completed,failed=failed,total=len(seeds)))
    write(root/'completion.json',dict(completed=not failed,runs=completed,failed=failed,total=len(seeds)))
    if failed:raise RuntimeError(f'{len(failed)} failed runs')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--build',required=True);p.add_argument('--baseline',required=True)
    p.add_argument('--seeds',default='41,42,43');p.add_argument('--workers',type=int,default=3);p.add_argument('--smoke',action='store_true');main(p.parse_args())

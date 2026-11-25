"""Four new ablation stages; immutable exact and RSRP endpoints reused."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback
import numpy as np
from training.stageb_ppo.native import ROOT,write,sha,EXPECTED_SIMULATOR
from training.stage2_closure.native import ReportNativeEnv
from training.stageb_lns.no_ctbs_inputs import detach_raw_phy_row,problem_from_raw_row,raw_pfq_draft
from training.stageb_lns.bounded_dpls import solve_bounded
from training.stageb_lns.ttl_weights import hol_urgency
from training.stageb_lns.crossload_native import weights_for
from training.stageb_rsrp.model import RsrpConfig,RsrpMeasurements
from training.stageb_rsrp.campaign import source_files as prior_sources
from .model import STAGES,FACTORS,transform,check_counters

CONFIG=RsrpConfig()
PRIOR=ROOT/'deliverables/rsrp_interference_20261109'


def key(job):return f"{job['scene']}_r{job['rate']:g}_{job['stage']}_{job['objective']}_s{job['seed']}"


def sources():
    return prior_sources()+[Path(__file__).with_name(n) for n in ('__init__.py','campaign.py','model.py')]


def reuse(out,job):
    arm=('rsrp_' if job['stage']=='reported' else '')+job['objective']
    old_key=f"{job['scene']}_r{job['rate']:g}_{arm}_s{job['seed']}"
    prior=json.loads((PRIOR/'campaign'/old_key/'done.json').read_text())
    original=Path(prior['directory'].replace('/opt/nvidia/cuBB',str(ROOT)))
    assert json.loads((original/'validation.json').read_text())['passed']
    launch=json.loads((original/'launch.json').read_text())
    assert launch['simulator_sha256']==EXPECTED_SIMULATOR and launch['horizon']==28400
    for name,digest in json.loads((PRIOR/'campaign/protocol.json').read_text())['sources'].items():
        assert sha(ROOT/name)==digest,name
    out.mkdir()
    files=['launch.json','validation.json','metrics.bin','actions.npz','traffic_offered_packets.csv',
           'packet_outcome_samples.csv','ue_topology.csv','saved_states.json','search.csv.gz']
    result=dict(**job,key=key(job),reused=True,directory=str(original),prior_key=old_key,
                prior_analysis=str(PRIOR/'analysis'/old_key),hashes={n:sha(original/n) for n in files})
    write(out/'done.json',result);return result


def run_one(root,job,smoke=False):
    import torch
    import pandas as pd
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False
    out=Path(root)/key(job)
    if out.exists():raise FileExistsError(out)
    if job['stage'] in ('exact','reported') and not smoke:return reuse(out,job)
    warm,settle,measure,follow=(32,16,64,400) if smoke else (4000,4000,20000,400)
    first,last=warm+settle+1,warm+settle+measure
    env=ReportNativeEnv(out,job['seed'],last+follow,job['rate'],
        'uniform' if job['scene']=='uniform' else 'coop_boundary')
    launch=json.loads((out/'launch.json').read_text());launch.update(job,common_pfq_warmup=warm,
        settling=settle,measured=measure,followup=follow,measurement_first_tti=first,measurement_last_tti=last,
        rsrp_config=asdict(CONFIG),execution='unchanged original native H; same PFQ/MCS/layers/weights/search')
    write(out/'launch.json',launch);(out/'states').mkdir()
    meter=RsrpMeasurements(CONFIG);ema=np.zeros(36)
    records=[];states=[];channels=[];reports=[];frozen={str(p):sha(p) for p in sources()}
    try:
        for _ in range(env.horizon):
            t=env.tti+1;digest=hashlib.sha256(env.snapshot['h'].tobytes()).hexdigest();channels.append(digest)
            stamp=time.perf_counter();meter.observe(env.snapshot['h'],env.tti)
            measurement_ms=(time.perf_counter()-stamp)*1000
            if t<=warm:action=env.raw_pfq()
            else:
                begin=time.perf_counter();cpu=time.process_time()
                row=detach_raw_phy_row(dict(tti=t,obs=env.obs,snapshot=env.snapshot,ema=ema.copy(),ctbs=env.inputs))
                p=problem_from_raw_row(row);urgency,slack=hol_urgency(env.fifo,t,p.queue)
                if job['objective']=='mlwdf_pf':p.weights=weights_for('mlwdf_pf',p.weights,ema,urgency,p.queue,0.)
                report,report_tti=meter.read();reports.append(hashlib.sha256(report.tobytes()).hexdigest())
                draft=raw_pfq_draft(p,row['obs'])
                q=transform(p,job['stage'],filtered=meter.filtered,reported=report)
                action,values,info=solve_bounded(q,draft);assert info['converged'];check_counters(info)
                records.append(dict(tti=t,wall_ms=(time.perf_counter()-begin)*1000,cpu_ms=(time.process_time()-cpu)*1000,
                    measurement_emulator_ms=measurement_ms,
                    **{k:v for k,v in info.items() if k not in ('core_ms','version')}))
                if t in {first,first+measure//2,last}:
                    row.update(objective_weights=p.weights.copy(),hol_urgency=urgency,hol_slack=slack,
                        selected_action=action.copy(),selected_values=values.copy(),stage=job['stage'],
                        rsrp_report=report.copy(),rsrp_filtered=meter.filtered.copy(),rsrp_report_tti=report_tti,
                        rsrp_config=asdict(CONFIG))
                    path=out/'states'/f'tti{t:06d}.pt';torch.save(row,path)
                    states.append(dict(path=str(path),sha256=sha(path)))
            assert hashlib.sha256(env.snapshot['h'].tobytes()).hexdigest()==digest
            env.step(action);ema=.995*ema+.005*np.asarray(env.info['actual_ue_goodput_bytes'])
            if t%4000==0:print('PROGRESS',key(job),t,env.horizon,flush=True)
        env.close()
    except BaseException:
        if not env.log.closed:env.close(validate=False)
        raise
    for path,digest in frozen.items():assert sha(path)==digest,path
    pd.DataFrame(records).to_csv(out/'search.csv.gz',index=False)
    np.save(out/'channel_hashes.npy',np.asarray(channels));np.save(out/'report_hashes.npy',np.asarray(reports))
    write(out/'saved_states.json',states)
    write(out/'ablation_validation.json',dict(passed=True,original_H_unchanged=env.horizon,
        candidate_accounting_checks=len(records),source_sha256=frozen))
    result=dict(**job,key=key(job),reused=False,directory=str(out),native_passed=True,smoke=smoke)
    write(out/'done.json',result);return result


def main(args):
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    seeds=list(map(int,args.seeds.split(',')));stages=args.stages.split(',') if args.stages else STAGES
    assert set(stages)<=set(STAGES)
    jobs=[dict(scene=scene,rate=rate,stage=stage,objective=objective,seed=seed)
        for stage in stages for scene,rate in [('boundary',.8),('uniform',1.)]
        for objective in ('deficit','mlwdf_pf') for seed in (seeds[:1] if args.smoke else seeds)]
    write(root/'protocol.json',dict(version='ordered_rsrp_ablation_v1',stages=list(STAGES),factors=list(FACTORS),
        jobs=jobs,seeds=seeds,smoke=args.smoke,windows=[32,16,64,400] if args.smoke else [4000,4000,20000,400],
        rsrp=asdict(CONFIG),simulator=EXPECTED_SIMULATOR,workers=args.workers,
        interpretation='adjacent causal contrasts conditional on order; not additive order-invariant main effects',
        port_bridge='all-port instantaneous power to single TX reference port isolated before temporal filtering',
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources()}))
    completed=[];failed=[]
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        pending={pool.submit(run_one,str(root),job,args.smoke):job for job in jobs}
        for future in as_completed(pending):
            try:completed.append(future.result());print('DONE',key(pending[future]),len(completed),'/',len(jobs),flush=True)
            except BaseException:
                failed.append(dict(job=pending[future],error=traceback.format_exc()));print('FAILED',failed[-1],flush=True)
            write(root/'progress_campaign.json',dict(completed=completed,failed=failed,total=len(jobs)))
    write(root/'completion.json',dict(completed=not failed,runs=completed,failed=failed,total=len(jobs)))
    if failed:raise RuntimeError(f'{len(failed)} failed runs')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=32);parser.add_argument('--seeds',default='41,42,43')
    parser.add_argument('--stages');parser.add_argument('--smoke',action='store_true');main(parser.parse_args())

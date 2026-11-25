"""Paired RSRP study: exact native execution, frozen original controls."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
import numpy as np
from training.stageb_ppo.native import ROOT, write, sha, EXPECTED_SIMULATOR
from training.stage2_closure.native import ReportNativeEnv
from training.stage2_closure.campaign import key
from training.stageb_lns.no_ctbs_inputs import detach_raw_phy_row, problem_from_raw_row, raw_pfq_draft
from training.stageb_lns.bounded_dpls import solve_bounded
from training.stageb_lns.ttl_weights import hol_urgency
from training.stageb_lns.crossload_native import weights_for
from .model import RsrpConfig, RsrpMeasurements, approximate_problem

CONFIG = RsrpConfig()
ARMS = ('deficit', 'mlwdf_pf', 'rsrp_deficit', 'rsrp_mlwdf_pf')
SOURCE = ROOT/'deliverables/stage2_closure_20261104/campaign'


def source_files():
    return [Path(__file__).with_name(n) for n in ('__init__.py','model.py','campaign.py')] + [ROOT/'training'/p for p in (
        'stage2_closure/native.py', 'stageb_ppo/native.py', 'stageb_ppo/reward.py',
        'stageb_lns/physics.py', 'stageb_lns/no_ctbs_inputs.py', 'stageb_lns/crossload_native.py',
        'stageb_lns/ttl_weights.py', 'stageb_lns/bounded_dpls.py', 'stageb_lns/_bounded.so',
        'stageb_lns/persistent_dpls.py', 'stageb_lns/physics_kernel.cpp',
        'stageb_lns/persistent_kernel.cpp', 'stageb_lns/bounded_kernel.cpp')]


def run_one(root, job, smoke=False):
    import torch
    import pandas as pd
    torch.set_num_threads(1); torch.backends.mkldnn.enabled=False
    out = Path(root)/key(job)
    if out.exists(): raise FileExistsError(out)
    if not job['arm'].startswith('rsrp_') and not smoke:
        original = SOURCE/key(job)
        assert json.loads((original/'done.json').read_text())['native_passed']
        assert json.loads((original/'validation.json').read_text())['passed']
        launch = json.loads((original/'launch.json').read_text())
        assert launch['simulator_sha256'] == EXPECTED_SIMULATOR and launch['horizon'] == 28400
        # Refuse historical reuse if any decision implementation changed.
        historical = launch['source_sha256']
        protocol_sources=json.loads((SOURCE/'protocol.json').read_text())['sources']
        for relative,digest in protocol_sources.items():
            assert sha(ROOT/relative)==digest,relative
        for name in ('bounded_dpls.py','bounded_kernel.cpp','_bounded.so','persistent_kernel.cpp',
                     'physics_kernel.cpp','no_ctbs_inputs.py','ttl_weights.py'):
            path = ROOT/'training/stageb_lns'/name
            old = next(v for k,v in historical.items() if k.endswith('/'+name))
            assert sha(path) == old, name
        out.mkdir()
        files = ['launch.json','validation.json','metrics.bin','actions.npz','traffic_offered_packets.csv',
                 'packet_outcome_samples.csv','ue_topology.csv','saved_states.json','search.csv.gz']
        result = dict(**job,key=key(job),reused=True,directory=str(original),
                      hashes={name:sha(original/name) for name in files})
        write(out/'done.json', result)
        return result
    warmup, settling, measured, followup = (32,16,64,400) if smoke else (4000,4000,20000,400)
    first, last = warmup+settling+1, warmup+settling+measured
    env = ReportNativeEnv(out,job['seed'],last+followup,job['rate'],
                          'uniform' if job['scene']=='uniform' else 'coop_boundary')
    launch=json.loads((out/'launch.json').read_text())
    launch.update(job,common_pfq_warmup=warmup,settling=settling,measured=measured,followup=followup,
                  measurement_first_tti=first,measurement_last_tti=last,
                  rsrp_config=asdict(CONFIG),measurement_kind='ideal_port0_power_proxy',
                  execution='original native channel, MCS, layers, PFQ initialization and PHY')
    write(out/'launch.json',launch)
    (out/'states').mkdir()
    meter=RsrpMeasurements(CONFIG)
    ema=np.zeros(36);records=[];states=[];channel_hashes=[];report_hashes=[]
    sources={str(p):sha(p) for p in source_files()}
    try:
        for _ in range(env.horizon):
            t=env.tti+1
            original_h=env.snapshot['h']
            digest=hashlib.sha256(original_h.tobytes()).hexdigest()
            channel_hashes.append(digest)
            stamp=time.perf_counter()
            meter.observe(original_h,env.tti)
            measurement_ms=(time.perf_counter()-stamp)*1000
            if t<=warmup:
                action=env.raw_pfq()
            else:
                begin=time.perf_counter();cpu=time.process_time()
                row=detach_raw_phy_row(dict(tti=t,obs=env.obs,snapshot=env.snapshot,ema=ema.copy(),ctbs=env.inputs))
                p=problem_from_raw_row(row)
                urgency,slack=hol_urgency(env.fifo,t,p.queue)
                objective=job['arm'].removeprefix('rsrp_')
                if objective=='mlwdf_pf':p.weights=weights_for('mlwdf_pf',p.weights,ema,urgency,p.queue,0.)
                report,report_tti=meter.read()
                report_hashes.append(hashlib.sha256(report.tobytes()).hexdigest())
                draft=raw_pfq_draft(p,row['obs'])
                q=approximate_problem(p,report) if job['arm'].startswith('rsrp_') else p
                action,values,info=solve_bounded(q,draft)
                assert info['converged']
                records.append(dict(tti=t,wall_ms=(time.perf_counter()-begin)*1000,
                    cpu_ms=(time.process_time()-cpu)*1000,measurement_emulator_ms=measurement_ms,
                    report_age_tti=env.tti-report_tti,
                    **{k:v for k,v in info.items() if k not in ('core_ms','version')}))
                if t in {first,first+measured//2,last}:
                    row.update(objective_weights=p.weights.copy(),hol_urgency=urgency,hol_slack=slack,
                        selected_action=action.copy(),selected_values=values.copy(),
                        rsrp_report=report.copy(),rsrp_report_tti=report_tti,
                        rsrp_filtered=meter.filtered.copy(),rsrp_config=asdict(CONFIG))
                    path=out/'states'/f'tti{t:06d}.pt';torch.save(row,path)
                    states.append(dict(path=str(path),sha256=sha(path)))
            # The original channel is never replaced in env or passed back from q.
            assert hashlib.sha256(env.snapshot['h'].tobytes()).hexdigest()==digest
            env.step(action)
            ema=.995*ema+.005*np.asarray(env.info['actual_ue_goodput_bytes'])
            if t%2000==0:print('PROGRESS',key(job),t,env.horizon,flush=True)
        env.close()
    except BaseException:
        if not env.log.closed:env.close(validate=False)
        raise
    for p,digest in sources.items():assert sha(p)==digest,p
    pd.DataFrame(records).to_csv(out/'search.csv.gz',index=False)
    np.save(out/'channel_hashes.npy',np.asarray(channel_hashes))
    np.save(out/'report_hashes.npy',np.asarray(report_hashes))
    write(out/'saved_states.json',states)
    write(out/'approximation_validation.json',dict(passed=True,original_channel_unchanged=env.horizon,
        report_minimum_delay_tti=CONFIG.delay_tti,source_sha256=sources))
    result=dict(**job,key=key(job),reused=False,directory=str(out),native_passed=True,smoke=smoke)
    write(out/'done.json',result)
    return result


def main(args):
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    seeds=[int(x) for x in args.seeds.split(',')]
    jobs=[dict(scene=scene,rate=rate,arm=arm,seed=seed)
          for scene,rate in [('boundary',.8),('uniform',1.)]
          for seed in seeds for arm in ARMS]
    if args.smoke:jobs=[j for j in jobs if j['seed']==seeds[0]]
    protocol=dict(version='rsrp_isotropic_v1',jobs=jobs,seeds=seeds,smoke=args.smoke,
        windows=[32,16,64,400] if args.smoke else [4000,4000,20000,400],
        config=asdict(CONFIG),tti_ms=.5,simulator=EXPECTED_SIMULATOR,
        measurement='ideal noise-free TX port 0; linear PRG/RX average; causal EMA; periodic delayed report',
        covariance='4 * filtered scalar power * I4, constant across PRGs',
        retained='instantaneous serving CSI, native PFQ, MCS/OLLA and layer caps; queue/HOL/service weights',
        scope='neighbor interference evaluation only; not a full limited-feedback deployment',
        sources={str(p.relative_to(ROOT)):sha(p) for p in source_files()})
    write(root/'protocol.json',protocol)
    completed=[];failed=[]
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        pending={pool.submit(run_one,str(root),j,args.smoke):j for j in jobs}
        for future in as_completed(pending):
            job=pending[future]
            try:
                completed.append(future.result());print('DONE',key(job),len(completed),'/',len(jobs),flush=True)
            except BaseException:
                failed.append(dict(job=job,error=traceback.format_exc()));print('FAILED',failed[-1],flush=True)
            write(root/'progress_campaign.json',dict(completed=completed,failed=failed,total=len(jobs)))
    write(root/'completion.json',dict(completed=not failed,runs=completed,failed=failed,total=len(jobs)))
    if failed:raise RuntimeError(f'{len(failed)} failed experiments')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--seeds',default='41,42,43');parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--smoke',action='store_true');main(parser.parse_args())

"""Predeclared paired scenario campaign, using unchanged exact search and PHY."""
import argparse,json,os,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd
from training.stageb_ppo.native import write,sha,REFERENCE
from training.stageb_lns.no_ctbs_inputs import detach_raw_phy_row,problem_from_raw_row,raw_pfq_draft
from training.stageb_lns.bounded_dpls import solve_bounded
from training.stageb_lns.ttl_weights import hol_urgency
from training.stageb_lns.crossload_native import weights_for
from .native import GeneralizationEnv
from .protocol import OUT,ROOT,SEEDS,SCENARIOS,ARMS,WARMUP,SETTLING,MEASURED,FOLLOWUP,FIRST,LAST,flows

def configure_cpu():
    import torch
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False
    aff=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{aff[os.getpid()%len(aff)]})

def setup_seed(seed):
    configure_cpu();folder=OUT/'setup'/f'seed{seed}'
    if (folder/'plan.json').exists():return json.loads((folder/'plan.json').read_text())
    env=GeneralizationEnv(folder,seed,8)
    serving=np.asarray(env.inputs.serving_cell,dtype=int).copy()
    for _ in range(env.horizon):env.step(env.raw_pfq())
    env.close()
    assert set(serving)=={0,1,2}
    configurations={}
    for scenario in SCENARIOS:
        sizes,rates,heavy=flows(scenario,seed,serving)
        frame=pd.DataFrame(dict(ue=np.arange(36),packet_size_bytes=sizes,arrival_rate_packets_per_tti=rates,serving_cell=serving,placement_group=np.where(np.arange(36)%12<6,'uniform','boundary'),traffic_group=np.where(sizes<3000,'small_high',np.where(sizes>3000,'large_low','homogeneous')),heavy_cell=heavy if SCENARIOS[scenario]['imbalanced'] else -1))
        path=OUT/'plans'/f'{scenario}_s{seed}.csv';path.parent.mkdir(exist_ok=True);frame.to_csv(path,index=False,float_format='%.15g')
        configurations[scenario]=dict(path=str(path),sha256=sha(path),expected_cluster_mbps=float(np.sum(sizes*rates)*.016),expected_cell_mbps=[float(np.sum((sizes*rates)[serving==c])*.016) for c in range(3)],heavy_cell=heavy if SCENARIOS[scenario]['imbalanced'] else None)
    result=dict(seed=seed,serving=serving.tolist(),cell_ue_counts=np.bincount(serving,minlength=3).tolist(),configurations=configurations)
    write(folder/'plan.json',result);return result

def parity():
    configure_cpu();out=OUT/'parity_disabled_seed41'
    if (out/'parity.json').exists():return
    if not (out/'validation.json').exists():
        env=GeneralizationEnv(out,41,512,mixed=False)
        for _ in range(env.horizon):env.step(env.raw_pfq())
        env.close()
    with np.load(out/'actions.npz') as a,np.load(REFERENCE/'seed41_pfq0_20000/prefix.npz') as b:
        np.testing.assert_array_equal(a['actions'],b['actions'][:512].reshape(-1,3,17))
        np.testing.assert_array_equal(a['input_hashes'],b['input_hashes'][:512])
    got=np.fromfile(out/'metrics.bin',dtype='<f8')
    original=np.memmap(REFERENCE/'seed41_pfq0_20000/metrics.bin',dtype='<f8',mode='r')
    a=got.reshape(-1,36,19).copy();b=np.asarray(original[:got.size]).reshape(-1,36,19).copy()
    # The non-transmitting layer sentinel is signed -1 versus uint8 255.
    # It is excluded from all physical averages; all active samples and all
    # queue/service/traffic fields must still be bit-exact.
    differing=a[:,:,14]!=b[:,:,14]
    assert np.all(a[:,:,12][differing]==0) and np.all(b[:,:,12][differing]==0)
    assert np.all(a[:,:,14][differing]==-1) and np.all(b[:,:,14][differing]==255)
    a[:,:,14][differing]=255
    np.testing.assert_array_equal(a,b)
    write(out/'parity.json',dict(passed=True,ttis=512,exact_actions_inputs_active_phy_queues_service_traffic=True,inactive_layer_sentinel_encoding_differences=int(differing.sum()),scenario_extension_disabled=True))

def key(j):return f"{j['scenario']}_{j['arm']}_s{j['seed']}"
def one(job,smoke=False):
    configure_cpu();out=OUT/('smoke' if smoke else 'campaign')/key(job)
    if (out/'done.json').exists():return json.loads((out/'done.json').read_text())
    warm,settled,measure,follow=(16,0,64,400) if smoke else (WARMUP,SETTLING,MEASURED,FOLLOWUP)
    first,last=warm+settled+1,warm+settled+measure
    flow_file=OUT/'plans'/f"{job['scenario']}_s{job['seed']}.csv"
    plan=pd.read_csv(flow_file);expected=plan.packet_size_bytes.to_numpy()*plan.arrival_rate_packets_per_tti.to_numpy()
    env=GeneralizationEnv(out,job['seed'],warm+settled+measure+follow,flow_file=flow_file)
    np.testing.assert_array_equal(env.inputs.serving_cell,plan.serving_cell)
    launch=json.loads((out/'launch.json').read_text());launch.update(job,common_pfq_warmup=warm,settling=settled,measured=measure,followup=follow,measurement_first_tti=first,measurement_last_tti=last,smoke=smoke,deficit_weights='clip(per-UE configured mean bytes per TTI / successful-byte EMA, 1, 4), mean normalized',protocol='stage2_generalization_20261106_v1');write(out/'launch.json',launch)
    ema=np.zeros(36);records=[];states=[];(out/'states').mkdir()
    try:
        for _ in range(env.horizon):
            t=env.tti+1
            if t<=warm or job['arm']=='pfq':action=env.raw_pfq()
            else:
                start=time.perf_counter()
                row=detach_raw_phy_row(dict(tti=t,obs=env.obs,snapshot=env.snapshot,ema=ema.copy(),ctbs=env.inputs))
                p=problem_from_raw_row(row)
                urgency,slack=hol_urgency(env.fifo,t,p.queue)
                if job['arm']=='deficit':
                    p.weights=np.clip(expected/np.maximum(ema,1.e-9),1.,4.);p.weights/=p.weights.mean()
                elif job['arm']=='mlwdf_pf':p.weights=weights_for('mlwdf_pf',p.weights,ema,urgency,p.queue,0.)
                else:raise ValueError(job['arm'])
                action,values,r=solve_bounded(p,raw_pfq_draft(p,row['obs']));assert r['converged']
                records.append(dict(tti=t,wall_ms=1000*(time.perf_counter()-start),**{k:v for k,v in r.items() if k not in ('core_ms','version')}))
                if t in {first,first+measure//2,last}:
                    import torch
                    row.update(objective_weights=p.weights.copy(),hol_urgency=urgency,hol_slack=slack,selected_action=action.copy(),selected_values=values.copy())
                    f=out/'states'/f'tti{t:06d}.pt';torch.save(row,f);states.append(dict(path=str(f),sha256=sha(f)))
            env.step(action);ema=.995*ema+.005*np.asarray(env.info['actual_ue_goodput_bytes'])
        env.close()
    except BaseException:
        if not env.log.closed:env.close(validate=False)
        raise
    pd.DataFrame(records).to_csv(out/'search.csv.gz',index=False);write(out/'saved_states.json',states)
    result=dict(**job,key=key(job),directory=str(out),horizon=env.horizon,native_passed=True,smoke=smoke);write(out/'done.json',result);return result

def run_pool(jobs,workers,smoke):
    completed=[];failed=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        pending={pool.submit(one,j,smoke):j for j in jobs}
        for f in as_completed(pending):
            try:completed.append(f.result());print('DONE',key(pending[f]),len(completed),'/',len(jobs),flush=True)
            except BaseException:
                detail=dict(job=pending[f],error=traceback.format_exc());failed.append(detail);print('FAILED',detail,flush=True)
            write(OUT/('smoke_progress.json' if smoke else 'progress_campaign.json'),dict(completed=completed,failed=failed,total=len(jobs)))
    write(OUT/('smoke_completion.json' if smoke else 'completion.json'),dict(completed=not failed,runs=completed,failed=failed,total=len(jobs)))
    if failed:raise RuntimeError(f'{len(failed)} failures')

def main(args):
    OUT.mkdir(exist_ok=True)
    if args.phase=='setup':
        parity()
        for seed in SEEDS:print('SETUP',setup_seed(seed),flush=True)
        return
    smoke=args.phase=='smoke';seeds=[SEEDS[0]] if smoke else SEEDS
    jobs=[dict(scenario=s,arm=a,seed=seed) for seed in seeds for s in SCENARIOS for a in ARMS]
    (OUT/('smoke' if smoke else 'campaign')).mkdir(exist_ok=True)
    if not smoke:
        assert json.loads((OUT/'smoke_completion.json').read_text())['completed']
        assert json.loads((OUT/'smoke_analysis/validation.json').read_text())['passed']
        protocol=dict(version='stage2_generalization_v1',scenarios=SCENARIOS,seeds=list(SEEDS),arms=ARMS,windows=dict(warmup=WARMUP,settling=SETTLING,measured=MEASURED,followup=FOLLOWUP),all_users=36,mixed_placement='18 boundary + 18 original uniform sampler; 6+6 per nominal cell; strongest serving association retained',mixed_traffic='18 x (1500 B,1.6/TTI) + 18 x (12000 B,0.2/TTI); equal expected bytes/UE in G2',imbalanced='Actual serving-cell arrival byte ratio 2:1:1, heavy=(seed-51)%3; expected cluster 1382.40 Mbps',fixed_ttl_ms=200,deficit_extension='Use per-UE configured mean arrival bytes in original clipped deficit formula; no tuning',primary_metrics=['payload_mbps','expiry_pct','expiry_byte_pct','delay_mean_ms','worst_ue_expiry_pct'],statistical_unit='within seed, then equal means across 5 paired seeds; fresh seeds 51-55',sources={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')},build=json.loads((OUT/'native/build_manifest.json').read_text()),plans={str(p):sha(p) for p in sorted((OUT/'plans').glob('*.csv'))})
        path=OUT/'protocol.json'
        if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(protocol))
        else:write(path,protocol)
    run_pool(jobs,args.workers,smoke)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['setup','smoke','formal'],required=True);p.add_argument('--workers',type=int,default=24);main(p.parse_args())

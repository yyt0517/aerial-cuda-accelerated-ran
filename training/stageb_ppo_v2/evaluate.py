"""Paired development trials: final PPO versus identical physical catalog."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from training.stageb_ppo.native import ROOT,REFERENCE,ABLATION,NativeEnv,legacy,packets,sha,write
from training.stageb_ppo.evaluate import metrics
from .catalog import Catalog
from .model import ServiceActor


def evaluate_one(root,arm,seed,cfg):
    root=Path(root).resolve();torch.set_num_threads(1);torch.backends.mkldnn.enabled=False
    checkpoint=root/('physical_greedy.pt' if arm=='physical_greedy' else 'ppo_final.pt')
    actor=ServiceActor();actor.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['actor']);actor.eval()
    warmup,settling,measured,followup=[cfg[f'validation_{k}'] for k in ['warmup','settling','measured','followup']]
    env=NativeEnv(root/f'eval_{arm}_seed{seed}',seed,warmup+settling+measured+followup)
    catalog=Catalog(cfg['retention']);audits=[];times=[]
    try:
        for _ in range(env.horizon):
            if env.tti<warmup:action=env.raw_pfq()
            else:
                start=time.monotonic();row=catalog.build(env)
                choice,lp,_=actor.sample(row,greedy=True)
                greedy=int(np.where(row['mask'],row['prior'],-1.e9).argmax())
                action=row['actions'][choice];times.append(time.monotonic()-start)
                audits.append(dict(row['audit'],selected=choice,physical_greedy=greedy,
                    differs_from_physical_greedy=choice!=greedy,selected_logprob=lp,
                    selected_expected_ratio=float(row['summary'][choice,0]/row['summary'][0,0]) if row['summary'][0,0]>0 else 1.))
            env.step(action)
            if env.tti%2000==0:print('EVAL',arm,seed,env.tti,flush=True)
        env.close()
    except BaseException:
        if not env.log.closed:env.close(validate=False)
        raise
    service,pk=packets(env.out);first,last=warmup+settling+1,warmup+settling+measured
    pk[pk.arrival_tti.between(first,last)].to_csv(env.out/'cohort.csv.gz',index=False)
    pd.DataFrame(audits).to_csv(env.out/'decision_audit.csv',index=False)
    ref=REFERENCE/f'seed{seed}_pfq0_20000';before=np.load(ref/'prefix.npz');after=np.load(env.out/'actions.npz')
    np.testing.assert_array_equal(before['input_hashes'][:warmup],after['input_hashes'][:warmup])
    np.testing.assert_array_equal(before['actions'][:warmup].reshape(-1,3,17),after['actions'][:warmup])
    offered=pd.read_csv(env.out/'traffic_offered_packets.csv');expected=pd.read_csv(ref/'traffic_offered_packets.csv')
    pd.testing.assert_frame_equal(offered[offered.arrival_tti<=env.horizon].reset_index(drop=True),
        expected[expected.arrival_tti<=env.horizon].reset_index(drop=True))
    result=metrics(service,pk,first,last,seed,arm)
    result.update(checkpoint_sha256=sha(checkpoint),native_validation=True,common_pfq_prefix_exact=warmup,
        mean_shared_host_policy_ms=1000*float(np.mean(times)),
        differs_from_physical_greedy_pct=100*float(np.mean([a['differs_from_physical_greedy'] for a in audits])),
        mean_neural_catalog=float(np.mean([a['neural_catalog'] for a in audits])))
    write(env.out/'evaluation.json',result);return result


def summarize(root,cfg):
    rows=[]
    for arm in ['physical_greedy','ppo']:
        for seed in cfg['validation_seeds']:
            path=root/f'eval_{arm}_seed{seed}/evaluation.json'
            if not path.exists():return False
            rows.append(json.loads(path.read_text()))
    b=legacy();first=cfg['validation_warmup']+cfg['validation_settling']+1;last=first+cfg['validation_measured']-1
    old=ROOT/'deliverables/ppo_no_ctbs_equal_20261029_v1/formal'
    for seed in cfg['validation_seeds']:
        references=[('pfq',REFERENCE/f'seed{seed}_pfq0_20000',ABLATION/f'packet_cohorts/pfq_seed{seed}.csv.gz'),
            ('raw_global_equal',ABLATION/f'native/raw_global_equal/seed{seed}_current_20000',ABLATION/f'packet_cohorts/raw_global_equal_seed{seed}.csv.gz'),
            ('ppo_v1',old/f'eval_ppo_seed{seed}',old/f'eval_ppo_seed{seed}/cohort.csv.gz')]
        for arm,folder,cohort_path in references:
            service=pd.DataFrame(np.fromfile(folder/'metrics.bin',dtype='<f8').reshape(-1,len(b.FIELDS)),columns=b.FIELDS)
            cohort=pd.read_csv(cohort_path)
            rows.append(metrics(service,cohort,first,last,seed,arm))
    table=pd.DataFrame(rows).sort_values(['seed','arm']);table.to_csv(root/'evaluation.csv',index=False)
    gates=[]
    for seed in cfg['validation_seeds']:
        subset=table[table.seed==seed].set_index('arm');a=subset.loc['ppo'];d=subset.loc['raw_global_equal'];g=subset.loc['physical_greedy']
        checks=dict(payload=a.complete_payload_mbps>=.995*d.complete_payload_mbps,
            expiry=a.expiry_pct<=d.expiry_pct+.1,mean_delay=a.success_delay_mean_ms<=d.success_delay_mean_ms,
            p95_delay=a.success_delay_p95_ms<=d.success_delay_p95_ms,ue_p5=a.ue_complete_payload_p5_mbps>=d.ue_complete_payload_p5_mbps)
        gates.append(dict(seed=seed,checks={k:bool(v) for k,v in checks.items()},passed=bool(all(checks.values())),
            ppo_minus_catalog_greedy_mbps=float(a.complete_payload_mbps-g.complete_payload_mbps),
            ppo_minus_catalog_greedy_expiry_pp=float(a.expiry_pct-g.expiry_pct)))
    write(root/'evaluation_gate.json',dict(completed=True,quality_passed=all(x['passed'] for x in gates),
        seeds=gates,promoted=False,selection=cfg['checkpoint_selection'],limits=cfg['limits']))
    return True


def main(root,arms):
    root=Path(root).resolve();cfg=json.loads((root/'protocol.json').read_text())
    source=json.loads((root/'source_manifest.json').read_text())
    for path,digest in source.items():assert sha(ROOT/path)==digest,path
    tasks=[(arm,seed) for arm in arms for seed in cfg['validation_seeds']
        if not (root/f'eval_{arm}_seed{seed}/evaluation.json').exists()]
    if tasks:
        with ProcessPoolExecutor(max_workers=len(tasks),mp_context=mp.get_context('spawn')) as pool:
            jobs=[pool.submit(evaluate_one,str(root),arm,seed,cfg) for arm,seed in tasks]
            for future in as_completed(jobs):print(json.dumps(future.result()),flush=True)
    for path,digest in source.items():assert sha(ROOT/path)==digest,path
    print('EVALUATION_SUMMARIZED',summarize(root,cfg),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--arms',nargs='+',choices=['physical_greedy','ppo'],default=['physical_greedy','ppo'])
    a=p.parse_args();main(a.root,a.arms)

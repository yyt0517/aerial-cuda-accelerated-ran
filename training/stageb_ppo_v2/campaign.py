"""Frozen bounded experiment with independent persistent training streams."""
import argparse
import json
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path
import random
import shutil
import time
import traceback
import numpy as np
import torch
from training.stageb_ppo.native import ROOT,sha,write
from .catalog import REPAIR
from .model import ServiceActor,Value
from .learn import learn
from .worker import worker


def config(smoke=False):
    cfg=dict(training_seeds=[41,42,43],warmup=4000,settling=512,rollout=1024,max_iterations=12,
        gamma=2**(-1/400),gae_lambda=.999,reward_scale=.01,holding=1/400,
        retention=.98,actor_lr=3.e-4,critic_lr=1.e-3,ppo_epochs=8,critic_epochs=8,
        target_kl=.02,entropy_coef=.001,ctbs=False,business_weights=False,
        validation_seeds=[44,45],validation_warmup=4000,validation_settling=4000,
        validation_measured=4000,validation_followup=400,
        checkpoint_selection='fixed final iteration; no evaluation-based selection',
        stopping='fixed 12 iterations; reaching the budget is not evidence of convergence',
        physical_prior='unweighted expected successful bytes; residual initialized to zero',
        limits=['full equal DPLS remains in every catalog; no speed claim',
                'development seeds 44–45 already exposed; no independent generalization claim',
                '2-second measurement; no automatic promotion'])
    if smoke:cfg.update(training_seeds=[41],warmup=128,settling=16,rollout=32,max_iterations=2,ppo_epochs=2,critic_epochs=2)
    return cfg


def receive_all(connections,kind,root,processes):
    pending=list(connections);responses=[];started=time.monotonic()
    while pending:
        for connection in wait(pending,timeout=30):
            message=connection.recv()
            if message['type']=='error':raise RuntimeError(message['error'])
            assert message['type']==kind,message.get('type')
            responses.append(message);pending.remove(connection)
        for process,connection in zip(processes,connections):
            if connection in pending and not process.is_alive():
                raise RuntimeError(f'Worker {process.pid} exited {process.exitcode} while waiting for {kind}')
        write(root/'heartbeat.json',dict(waiting_for=kind,pending=len(pending),elapsed_wait_s=time.monotonic()-started))
    return responses


def main(root,smoke=False):
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False
    torch.manual_seed(20261030);np.random.seed(20261030);random.seed(20261030)
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    cfg=config(smoke);write(root/'protocol.json',cfg)
    sources={str(p.relative_to(ROOT)):sha(p) for package in ['stageb_ppo_v2','stageb_ppo','stageb_lns']
        for p in (ROOT/'training'/package).rglob('*.py')}
    sources[str(REPAIR.relative_to(ROOT))]=sha(REPAIR)
    write(root/'source_manifest.json',sources)
    for package in ['stageb_ppo_v2','stageb_ppo','stageb_lns']:
        shutil.copytree(ROOT/'training'/package,root/'source_snapshot'/package,ignore=shutil.ignore_patterns('__pycache__'))
    actor,critic=ServiceActor(),Value()
    actor_opt=torch.optim.Adam(actor.parameters(),lr=cfg['actor_lr'])
    critic_opt=torch.optim.Adam(critic.parameters(),lr=cfg['critic_lr'])
    torch.save(dict(actor=actor.state_dict(),protocol=cfg),root/'physical_greedy.pt')
    context=mp.get_context('spawn');processes=[];connections=[];history=[];start=time.monotonic()
    try:
        for seed in cfg['training_seeds']:
            parent,child=context.Pipe()
            process=context.Process(target=worker,args=(child,str(root),seed,cfg))
            process.start();child.close();processes.append(process);connections.append(parent)
        write(root/'status.json',dict(stage='warmup',worker_pids=[p.pid for p in processes]))
        receive_all(connections,'ready',root,processes)
        for iteration in range(1,cfg['max_iterations']+1):
            write(root/'status.json',dict(stage='training',iteration=iteration,max_iterations=cfg['max_iterations']))
            for connection in connections:connection.send(dict(type='rollout',iteration=iteration,actor=actor.state_dict()))
            segments=[m['data'] for m in receive_all(connections,'rollout',root,processes)]
            segments.sort(key=lambda s:s['seed'])
            report,credit=learn(actor,critic,actor_opt,critic_opt,segments,cfg,iteration)
            report['elapsed_s']=time.monotonic()-start
            history.append(report);write(root/'learning_history.json',history)
            torch.save(credit,root/f'credit_{iteration:03d}.pt')
            checkpoint=dict(actor=actor.state_dict(),critic=critic.state_dict(),actor_optimizer=actor_opt.state_dict(),
                critic_optimizer=critic_opt.state_dict(),iteration=iteration,protocol=cfg,source_manifest=sources,
                torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
            torch.save(checkpoint,root/f'checkpoint_{iteration:03d}.pt')
            print(json.dumps({k:v for k,v in report.items() if k!='steps'}),flush=True)
        for connection in connections:connection.send(dict(type='finish'))
        receive_all(connections,'finished',root,processes)
        for process in processes:
            process.join(timeout=30);assert process.exitcode==0,process.exitcode
        for path,digest in sources.items():assert sha(ROOT/path)==digest,path
        shutil.copyfile(root/f'checkpoint_{cfg["max_iterations"]:03d}.pt',root/'ppo_final.pt')
        result=dict(stage='training_complete',iterations=len(history),samples=sum(r['samples'] for r in history),
            elapsed_s=time.monotonic()-start,checkpoint_sha256=sha(root/'ppo_final.pt'),sources_unchanged=True,
            convergence_proven=False,promoted=False)
        write(root/'status.json',result);print(json.dumps(result),flush=True)
    except BaseException:
        write(root/'status.json',dict(stage='failed',traceback=traceback.format_exc()))
        for process in processes:
            if process.is_alive():process.terminate()
        for process in processes:process.join(timeout=15)
        raise
    finally:
        for connection in connections:connection.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();main(args.out,args.smoke)

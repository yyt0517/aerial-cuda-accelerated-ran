"""Continue the same native histories with minibatch PPO optimizer updates."""
import argparse
import json
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path
import random
import shutil
import os
import time
import traceback
import numpy as np
import torch
from training.stageb_ppo.native import ROOT,sha,write
from .catalog import REPAIR
from .model import ServiceActor,Value
from .minibatch_learn import learn
from .refine_worker import worker
from .campaign import receive_all


def main(root,source):
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False
    torch.manual_seed(20261030);np.random.seed(20261030);random.seed(20261030)
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    source=Path(source).resolve();resume=6
    saved=torch.load(source/f'checkpoint_{resume:03d}.pt',weights_only=False)
    cfg=dict(saved['protocol'],resume_root=str(source),resume_iteration=resume,actor_minibatch=128,
        optimizer_refinement='128-state minibatches, 24 steps per epoch; entire epoch rollback if KL exceeds 0.02',
        resume_checkpoint_sha256=sha(source/f'checkpoint_{resume:03d}.pt'),
        refinement_reason='six full-batch rounds yielded per-round KL around 1e-7; actor optimized only 8 steps per round',
        prefix_replay='exact native actions and rewards, final catalog features and sampler state checked',
        new_training_samples=6*3*1024,reused_prefix_training_samples=6*3*1024,
        reused_control=str(source),checkpoint_selection='fixed iteration 12; no PPO evaluation used to select refinement')
    write(root/'protocol.json',cfg)
    for name in ['physical_greedy.pt','source_manifest.json','runtime_supplement_manifest.json']:
        shutil.copyfile(source/name,root/('base_source_manifest.json' if name=='source_manifest.json' else name))
    for iteration in range(1,resume+1):
        for path in [source/f'checkpoint_{iteration:03d}.pt',source/f'credit_{iteration:03d}.pt',
                *[source/f'rollout_{iteration:03d}_seed{s}.pt' for s in cfg['training_seeds']]]:
            os.link(path,root/path.name)
    for seed in cfg['validation_seeds']:
        shutil.copytree(source/f'eval_physical_greedy_seed{seed}',root/f'eval_physical_greedy_seed{seed}',copy_function=os.link)
    sources={str(p.relative_to(ROOT)):sha(p) for package in ['stageb_ppo_v2','stageb_ppo','stageb_lns']
        for p in (ROOT/'training'/package).rglob('*.py')}
    sources[str(REPAIR.relative_to(ROOT))]=sha(REPAIR)
    write(root/'source_manifest.json',sources)
    for package in ['stageb_ppo_v2','stageb_ppo','stageb_lns']:
        shutil.copytree(ROOT/'training'/package,root/'source_snapshot'/package,ignore=shutil.ignore_patterns('__pycache__'))
    actor,critic=ServiceActor(),Value()
    actor_opt=torch.optim.Adam(actor.parameters(),lr=cfg['actor_lr'])
    critic_opt=torch.optim.Adam(critic.parameters(),lr=cfg['critic_lr'])
    actor.load_state_dict(saved['actor']);critic.load_state_dict(saved['critic'])
    actor_opt.load_state_dict(saved['actor_optimizer']);critic_opt.load_state_dict(saved['critic_optimizer'])
    torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
    history=json.loads((source/'learning_history.json').read_text())[:resume]
    elapsed_prefix=history[-1]['elapsed_s']
    context=mp.get_context('spawn');processes=[];connections=[];start=time.monotonic()
    try:
        for seed in cfg['training_seeds']:
            parent,child=context.Pipe()
            process=context.Process(target=worker,args=(child,str(root),seed,cfg))
            process.start();child.close();processes.append(process);connections.append(parent)
        write(root/'status.json',dict(stage='exact_prefix_replay',resume_iteration=resume,worker_pids=[p.pid for p in processes]))
        receive_all(connections,'ready',root,processes)
        for iteration in range(resume+1,cfg['max_iterations']+1):
            write(root/'status.json',dict(stage='training',iteration=iteration,max_iterations=cfg['max_iterations']))
            for connection in connections:connection.send(dict(type='rollout',iteration=iteration,actor=actor.state_dict()))
            segments=[m['data'] for m in receive_all(connections,'rollout',root,processes)]
            segments.sort(key=lambda s:s['seed'])
            report,credit=learn(actor,critic,actor_opt,critic_opt,segments,cfg,iteration)
            report['elapsed_s']=elapsed_prefix+time.monotonic()-start
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
            new_training_samples=cfg['new_training_samples'],elapsed_s=time.monotonic()-start,checkpoint_sha256=sha(root/'ppo_final.pt'),sources_unchanged=True,
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
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args();main(args.out,args.source)

"""Bounded retraining campaign; never overwrite a run or promote by train reward."""
import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .learn import learn
from .model import JointActor, Value, observation, action_tokens, native_action
from .native import ROOT, ABLATION, NativeEnv, teacher, write, sha, packets


def checkpoint(path, actor, critic, actor_opt, critic_opt, generator, cfg, iteration):
    torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(),
        actor_optimizer=actor_opt.state_dict(), critic_optimizer=critic_opt.state_dict(),
        sampler_state=generator.get_state(), torch_rng_state=torch.get_rng_state(),
        config=cfg, iteration=iteration, architecture='joint_prg_cell_gru64_no_ctbs_v1'), path)


def run(out, smoke=False):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.manual_seed(20261029)
    np.random.seed(20261029)
    cfg = dict(train_seeds=[41] if smoke else [41,42,43], validation_seeds=[] if smoke else [44,45],
        warmup=128 if smoke else 4000, teacher_steps=16 if smoke else 128,
        iterations=2 if smoke else 12, rollout=16 if smoke else 512,
        bc_epochs=2 if smoke else 60, ppo_epochs=2, critic_epochs=4,
        gamma=2**(-1/400), gae_lambda=.995, clip=.1, target_kl=.02,
        actor_lr=3.e-5, critic_lr=3.e-4, entropy_coef=.002,
        ctbs=False, business_weights=False, deficit_weights=False,
        action='autoregressive 51 PRG/cell tokens; all associated nonempty UEs plus blank; static legality only',
        initialization='new model; equal no-CTBS DPLS behavior cloning on training trajectories only',
        reward='(fully successful completed packet payload - full expired packet payload - 0.01*postqueue bytes)/86400',
        expiry_timing='charged at causal pre-action removal, one TTI after last usable slot',
        ppo_floor='none; rho0.90 applies only to teacher and physical-search comparator',
        validation_warmup=4000, validation_settling=4000, validation_measured=4000, validation_followup=400,
        selection='final fixed-budget PPO checkpoint; also evaluate BC endpoint; no training-peak promotion',
        comparator='raw_global_equal and PFQ, matched 8001..12000 arrivals with full followup',
        limits=['Development seeds, not blind tests.','2-second pilot measurement, not historical 10-second validation.',
                'New compact joint actor; not a continuation of historical HGraph weights.',
                'No automatic replacement of current_algorithm.json.'], smoke=smoke)
    source = {str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}
    write(out/'protocol.json', dict(config=cfg, source_sha256=source,
        ablation_protocol_sha256=sha(ABLATION/'protocol.json'), date='2026-10-29'))
    generator = torch.Generator().manual_seed(20261029)
    actor, critic = JointActor(), Value()
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg['actor_lr'])
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg['critic_lr'])
    envs, examples, history = [], [], []
    started = time.monotonic()
    try:
        for seed in cfg['train_seeds']:
            horizon = cfg['warmup'] + cfg['teacher_steps'] + cfg['rollout'] * cfg['iterations']//len(cfg['train_seeds'])
            env = NativeEnv(out/f'train_seed{seed}', seed, horizon)
            envs.append(env)
            write(out/'status.json', dict(stage='warmup', seed=seed, tti=0))
            for _ in range(cfg['warmup']):
                env.step(env.raw_pfq())
                if env.tti % 1000 == 0:
                    print('WARMUP', seed, env.tti, flush=True)
            study = teacher()
            for k in range(cfg['teacher_steps']):
                x, mask = observation(env.obs, env.inputs)
                action = env.expert(study)
                examples.append((x,mask,action_tokens(action)))
                env.step(action)
                if (k+1)%32 == 0:
                    print('TEACHER', seed, k+1, flush=True)
            write(out/f'teacher_seed{seed}.json', dict(samples=cfg['teacher_steps'], ctbs_pruned_prgs=0,
                weights='all ones', causal_floor=.9))
        x, mask, labels = [torch.stack([e[i] for e in examples]) for i in range(3)]
        torch.save(dict(x=x,mask=mask,labels=labels,seeds=cfg['train_seeds']), out/'bc_dataset.pt')
        bc_opt = torch.optim.Adam(actor.parameters(),lr=3.e-4)
        for epoch in range(cfg['bc_epochs']):
            losses=[]
            for index in torch.randperm(len(x)).split(16):
                bc_opt.zero_grad()
                logp,_,_=actor.replay(x[index],mask[index],labels[index])
                loss=-logp.mean()/51
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
                bc_opt.step()
                losses.append(float(loss.detach()))
            if (epoch+1)%10 == 0 or epoch == cfg['bc_epochs']-1:
                print('BC',epoch+1,float(np.mean(losses)),flush=True)
                write(out/'status.json',dict(stage='bc',epoch=epoch+1,loss=float(np.mean(losses))))
        checkpoint(out/'bc.pt',actor,critic,actor_opt,critic_opt,generator,cfg,0)
        del examples,x,mask,labels,bc_opt
        for iteration in range(1,cfg['iterations']+1):
            env=envs[(iteration-1)%len(envs)]
            rows=[]
            write(out/'status.json',dict(stage='ppo_rollout',iteration=iteration,seed=env.seed,tti=env.tti))
            for _ in range(cfg['rollout']):
                x,mask=observation(env.obs,env.inputs)
                actions,lp,dist=actor.sample(x,mask,generator=generator)
                record=env.step(native_action(actions))
                rows.append((x,mask,actions,lp,dist,torch.tensor(record['reward'],dtype=torch.float32)))
            # A horizon is a collection truncation, never a terminal physical state.
            # Fetch the already prepared next state even on the final step.
            if env.tti == env.horizon:
                env.refresh()
            next_x,next_mask=observation(env.obs,env.inputs)
            data=(*[torch.stack([r[i] for r in rows]) for i in range(6)],next_x,next_mask)
            torch.save(dict(data=data,iteration=iteration,seed=env.seed,tti=env.tti),out/f'rollout_{iteration:03d}.pt')
            result=learn(actor,critic,actor_opt,critic_opt,data,cfg)
            result.update(iteration=iteration,seed=env.seed,elapsed_s=time.monotonic()-started)
            history.append(result)
            write(out/'training_metrics.json',history)
            checkpoint(out/f'ppo_iter_{iteration:03d}.pt',actor,critic,actor_opt,critic_opt,generator,cfg,iteration)
            print('PPO',json.dumps({k:v for k,v in result.items() if k!='steps'}),flush=True)
            del data,rows
        checkpoint(out/'ppo_final.pt',actor,critic,actor_opt,critic_opt,generator,cfg,cfg['iterations'])
        for env in envs:
            env.close()
            packets(env.out)
        envs=[]
        assert sum(r['actor_updates'] for r in history)>0
        write(out/'training_validation.json',dict(passed=True,iterations=cfg['iterations'],
            actor_updates=sum(r['actor_updates'] for r in history),
            steps=cfg['rollout']*cfg['iterations'],native_actions_and_fifo_rewards_verified=True,
            final_sha256=sha(out/'ppo_final.pt'), elapsed_s=time.monotonic()-started))
        write(out/'status.json',dict(stage='training_complete',checkpoint=str(out/'ppo_final.pt')))
        if not smoke:
            from .evaluate import evaluate_campaign
            evaluate_campaign(out,cfg)
        print('CAMPAIGN_COMPLETE',out,flush=True)
    except BaseException:
        write(out/'status.json',dict(stage='failed',error=traceback.format_exc()))
        for env in envs:
            if not env.log.closed:
                env.close(validate=False)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    run(args.out,args.smoke)

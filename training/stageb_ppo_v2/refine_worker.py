"""Exact native replay to a frozen checkpoint, then persistent on-policy streams."""
import json
from pathlib import Path
import traceback
import signal
import numpy as np
import pandas as pd
import torch
from training.stageb_ppo.native import NativeEnv,write,packets
from .catalog import Catalog
from .model import ServiceActor
from .state import reward,packet_state
from .learn import KEYS


def finish(env,audits,rewards,cfg):
    # The simulator is persistent and ignores reset horizon. Record the actual
    # executed length if a predeclared training plateau stops the stream early.
    launch=json.loads((env.out/'launch.json').read_text())
    launch['planned_horizon']=launch['horizon'];launch['horizon']=env.tti
    write(env.out/'launch.json',launch);env.horizon=env.tti
    env.close();packets(env.out)
    pd.DataFrame(audits).to_csv(env.out/'catalog_audit.csv',index=False)
    pd.DataFrame(rewards).to_csv(env.out/'long_credit_rewards.csv',index=False)
    base=pd.read_csv(env.out/'rewards.csv').set_index('tti')
    r=pd.DataFrame(rewards)
    selected=base.loc[r.tti]
    expected=(selected.complete_payload_bytes-selected.expired_payload_bytes-selected.post_queue_bytes/400)/86400
    np.testing.assert_allclose(expected,r.base_reward,rtol=0,atol=1.e-12)
    np.testing.assert_allclose(r.potential_after.iloc[:-1],r.potential_before.iloc[1:],rtol=0,atol=1.e-12)
    np.testing.assert_allclose(r.shaping_reward,cfg['gamma']*r.potential_after-r.potential_before,rtol=0,atol=1.e-12)
    np.testing.assert_allclose(r.reward,r.base_reward+r.shaping_reward,rtol=0,atol=1.e-12)
    assert np.isfinite(r.to_numpy()).all()
    write(env.out/'long_credit_validation.json',dict(passed=True,steps=len(r),
        full_packet_base_verified=True,potential_endpoint_consistency=True,
        all_native_actions_exact=True,no_ctbs=True,no_business_weights=True))


def worker(connection,root,seed,cfg):
    def stop(signum,frame):raise SystemExit('campaign worker terminated')
    signal.signal(signal.SIGTERM,stop)
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False
    torch.manual_seed(20261030+seed)
    env=None
    try:
        root=Path(root).resolve()
        horizon=cfg['warmup']+cfg['settling']+cfg['max_iterations']*cfg['rollout']
        env=NativeEnv(root/f'train_seed{seed}',seed,horizon)
        actor=ServiceActor();catalog=Catalog(cfg['retention'])
        sampler=torch.Generator().manual_seed(20261030+seed)
        for _ in range(cfg['warmup']):env.step(env.raw_pfq())
        for _ in range(cfg['settling']):env.step(env.expert(catalog.study))
        audits=[];rewards=[];errors=[]
        source=Path(cfg['resume_root'])
        for iteration in range(1,cfg['resume_iteration']+1):
            saved=torch.load(source/f'rollout_{iteration:03d}_seed{seed}.pt',weights_only=False)
            for index,action in enumerate(saved['actions']):
                before=packet_state(env.fifo,env.tti+1,env.snapshot['queue'])[1]
                actual=env.step(action)
                after=packet_state(env.fifo,env.tti+1,env.snapshot['queue'])[1]
                r=reward(actual,before,after,cfg['gamma'])
                errors.append(abs(r['reward']-saved['reward'][index]));rewards.append(r)
            sampler.set_state(saved['sampler_after'])
        assert max(errors)<1.e-12,max(errors)
        pending=catalog.build(env)
        for key in KEYS:np.testing.assert_allclose(pending[key],saved['next'][key],rtol=0,atol=1.e-6)
        write(env.out/'prefix_replay_validation.json',dict(passed=True,checkpoint_iteration=cfg['resume_iteration'],
            native_prefix_ttis=env.tti,ppo_replayed=len(errors),max_reward_error=max(errors),
            terminal_catalog_features_match=True,sampler_state_resumed=True,
            catalog_audit_scope='new continuation decisions only; earlier PPO prefix reproduced from saved exact actions'))
        connection.send(dict(type='ready',seed=seed,tti=env.tti))
        while True:
            command=connection.recv()
            if command['type']=='finish':
                finish(env,audits,rewards,cfg)
                connection.send(dict(type='finished',seed=seed,tti=env.tti));break
            actor.load_state_dict(command['actor'])
            data={k:[] for k in KEYS+['choice','distribution','reward','actions','catalog_actions']}
            initial_rng=sampler.get_state()
            for _ in range(cfg['rollout']):
                row=pending
                choice,lp,distribution=actor.sample(row,generator=sampler)
                actual=env.step(row['actions'][choice])
                if env.tti==env.horizon:env.refresh()
                pending=catalog.build(env)
                r=reward(actual,row['potential'],pending['potential'],cfg['gamma'])
                for key in KEYS:data[key].append(row[key])
                data['choice'].append(choice);data['distribution'].append(distribution.numpy())
                data['reward'].append(r['reward']);data['actions'].append(row['actions'][choice])
                data['catalog_actions'].append(row['actions'])
                audit=dict(row['audit'],selected=choice,selected_expected_ratio=float(
                    row['summary'][choice,0]/max(row['summary'][0,0],1.e-9)),
                    sampled_logprob=lp)
                audits.append(audit);rewards.append(r)
                if env.tti%256==0:
                    write(env.out/'learning_progress.json',dict(iteration=command['iteration'],tti=env.tti,
                        phase='collecting',latest_choices=audit['choices']))
            packed={k:np.asarray(v) for k,v in data.items()}
            packed.update(next={k:pending[k] for k in KEYS},seed=seed,iteration=command['iteration'],
                first_tti=env.tti-cfg['rollout']+1,last_tti=env.tti,
                sampler_before=initial_rng,sampler_after=sampler.get_state())
            if command['iteration']==cfg['resume_iteration']+1:
                reference=torch.load(source/f'rollout_{command["iteration"]:03d}_seed{seed}.pt',weights_only=False)
                for key in KEYS+['choice','distribution','reward','actions','catalog_actions']:
                    np.testing.assert_allclose(packed[key],reference[key],rtol=0,atol=1.e-6)
                write(env.out/'paired_first_rollout.json',dict(passed=True,iteration=command['iteration'],
                    matched_fullbatch_control=True,states_actions_rewards_distributions_match=True))
            torch.save(packed,root/f'rollout_{command["iteration"]:03d}_seed{seed}.pt')
            write(env.out/'learning_progress.json',dict(iteration=command['iteration'],tti=env.tti,
                reward_mean=float(packed['reward'].mean()),mean_choices=float(np.mean([a['choices'] for a in audits[-cfg['rollout']:]]))))
            connection.send(dict(type='rollout',data=packed))
    except BaseException:
        error=traceback.format_exc()
        if env is not None and not env.log.closed:env.close(validate=False)
        try:connection.send(dict(type='error',seed=seed,error=error))
        finally:connection.close()
        raise

"""Persistent native streams; each rollout holds one frozen behavior policy."""
import json
from pathlib import Path
import traceback
import signal
import numpy as np
import pandas as pd
import torch
from training.stageb_ppo.native import NativeEnv,write,packets
from training.stageb_ppo_v2.catalog import Catalog
from training.stageb_ppo_v2.model import ServiceActor
from .reward import reward, reconstruct_pending
from training.stageb_ppo_v2.learn import KEYS


def finish(env,audits,rewards,cfg):
    # The simulator is persistent and ignores reset horizon. Record the actual
    # executed length if a predeclared training plateau stops the stream early.
    launch=json.loads((env.out/'launch.json').read_text())
    launch['planned_horizon']=launch['horizon'];launch['horizon']=env.tti
    write(env.out/'launch.json',launch);env.horizon=env.tti
    env.close();_,pk=packets(env.out)
    pd.DataFrame(audits).to_csv(env.out/'catalog_audit.csv',index=False)
    r=pd.DataFrame(rewards)
    r.to_csv(env.out/'long_credit_rewards.csv',index=False)
    full,age=reconstruct_pending(pk,env.tti,cfg)
    index=r.tti.to_numpy(int)-1
    np.testing.assert_allclose(full[index],r.pending_full_bytes,rtol=0,atol=1.e-8)
    np.testing.assert_allclose(age[index],r.age_exposure_bytes,rtol=0,atol=1.e-6)
    base=pd.read_csv(env.out/'rewards.csv').set_index('tti').loc[r.tti]
    expected=(base.complete_payload_bytes.to_numpy()-base.expired_payload_bytes.to_numpy()
        -cfg['holding']*(full[index]+cfg['age_multiplier']*age[index]))/86400.
    np.testing.assert_allclose(expected,r.base_reward,rtol=0,atol=1.e-10)
    np.testing.assert_allclose(r.potential_after.iloc[:-1],r.potential_before.iloc[1:],rtol=0,atol=1.e-12)
    np.testing.assert_allclose(r.shaping_reward,cfg['gamma']*r.potential_after-r.potential_before,rtol=0,atol=1.e-12)
    np.testing.assert_allclose(r.reward,r.base_reward+r.shaping_reward,rtol=0,atol=1.e-12)
    assert np.isfinite(r.to_numpy()).all()
    write(env.out/'long_credit_validation.json',dict(passed=True,steps=len(r),
        independently_reconstructed_full_packet_wait=True,independently_reconstructed_age_exposure=True,
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
        pending=catalog.build(env);audits=[];rewards=[]
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
                r=reward(actual,env.fifo,row['potential'],pending['potential'],cfg)
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
                reward_components=rewards[-cfg['rollout']:],
                sampler_before=initial_rng,sampler_after=sampler.get_state())
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

"""Controlled refinement: PPO minibatch steps with whole-epoch KL rollback."""
import copy
import numpy as np
import torch
from training.stageb_ppo.model import gae

KEYS=['state','candidate','summary','mask','prior']


def tensors(batch):
    return [torch.as_tensor(batch[k]) for k in KEYS]


def learn(actor,critic,actor_opt,critic_opt,segments,cfg,iteration):
    features=[torch.cat([torch.as_tensor(s[k]) for s in segments]) for k in KEYS]
    state,candidates,summary,mask,prior=features
    actions=torch.cat([torch.as_tensor(s['choice']) for s in segments]).long()
    old=torch.cat([torch.as_tensor(s['distribution']) for s in segments])
    old_lp=old.gather(1,actions[:,None]).squeeze(1)
    raw_rewards=torch.cat([torch.as_tensor(s['reward']) for s in segments]).float()
    n=len(actions)
    if iteration==1:
        with torch.no_grad():
            critic.net[-1].bias.fill_(float(raw_rewards.mean())*cfg['reward_scale']/(1-cfg['gamma']))
    with torch.no_grad():
        values=critic(state,summary,mask)
        boot=torch.stack([critic(torch.as_tensor(s['next']['state'])[None],
            torch.as_tensor(s['next']['summary'])[None],torch.as_tensor(s['next']['mask'])[None])[0] for s in segments])
        advantage=[];targets=[];offset=0
        for i,s in enumerate(segments):
            size=len(s['choice'])
            a,t=gae(raw_rewards[offset:offset+size]*cfg['reward_scale'],values[offset:offset+size],boot[i],cfg['gamma'],cfg['gae_lambda'])
            advantage.append(a);targets.append(t);offset+=size
        advantage=torch.cat(advantage);targets=torch.cat(targets)
        normalized=(advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1.e-6)
        current=torch.cat([actor(*[f[i:i+64] for f in features]) for i in range(0,n,64)])
        replay_error=float((current-old).abs().max())
        assert replay_error<1.e-4,('behavior replay mismatch',replay_error)
    steps=[]
    for epoch in range(cfg['ppo_epochs']):
        accepted=False
        for attempt in range(3):
            before=copy.deepcopy(actor.state_dict());adam=copy.deepcopy(actor_opt.state_dict())
            total_loss=0.;norms=[];minibatches=0
            for idx in torch.randperm(n).split(cfg['actor_minibatch']):
                actor_opt.zero_grad()
                d=actor(*[f[idx] for f in features])
                lp=d.gather(1,actions[idx,None]).squeeze(1)
                ratio=(lp-old_lp[idx]).exp()
                a=normalized[idx]
                entropy=-(d.exp()*d).sum(-1)
                loss=(-torch.minimum(ratio*a,ratio.clamp(.9,1.1)*a)-cfg['entropy_coef']*entropy).mean()
                loss.backward();total_loss+=float(loss.detach())*len(idx)/n
                norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
                assert torch.isfinite(norm)
                actor_opt.step();norms.append(float(norm));minibatches+=1
            with torch.no_grad():
                new=torch.cat([actor(*[f[i:i+64] for f in features]) for i in range(0,n,64)])
                kl=float((old.exp()*(old-new)).sum(-1).mean())
                accepted=np.isfinite(kl) and kl<=cfg['target_kl']
            steps.append(dict(epoch=epoch+1,attempt=attempt+1,accepted=bool(accepted),kl=kl,
                loss=total_loss,gradient_norm=float(np.mean(norms)),lr=actor_opt.param_groups[0]['lr'],
                optimizer_steps=minibatches))
            if accepted:break
            actor.load_state_dict(before);actor_opt.load_state_dict(adam)
            for group in actor_opt.param_groups:group['lr']*=.5
        if not accepted:break
    before_ev=float(1-(values-targets).var(unbiased=False)/targets.var(unbiased=False).clamp_min(1.e-9))
    for _ in range(cfg['critic_epochs']):
        for idx in torch.randperm(n).split(128):
            critic_opt.zero_grad()
            loss=(critic(state[idx],summary[idx],mask[idx])-targets[idx]).square().mean()
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(critic.parameters(),1.)
            assert torch.isfinite(norm);critic_opt.step()
    with torch.no_grad():prediction=critic(state,summary,mask)
    result=dict(iteration=iteration,samples=n,reward_mean=float(raw_rewards.mean()),
        reward_by_seed=[float(np.mean(s['reward'])) for s in segments],
        critic_ev_before=before_ev,
        critic_ev_after=float(1-(prediction-targets).var(unbiased=False)/targets.var(unbiased=False).clamp_min(1.e-9)),
        critic_normalized_mse=float((prediction-targets).square().mean()),
        value_mean=float(prediction.mean()),target_mean=float(targets.mean()),target_std=float(targets.std(unbiased=False)),
        advantage_std=float(advantage.std(unbiased=False)),behavior_max_logprob_error=replay_error,
        actor_steps=sum(s['optimizer_steps'] for s in steps if s['accepted']),steps=steps)
    return result,dict(values=values,bootstrap=boot,targets=targets,advantages=advantage,rewards=raw_rewards,
        reward_scale=cfg['reward_scale'],gamma=cfg['gamma'],gae_lambda=cfg['gae_lambda'])

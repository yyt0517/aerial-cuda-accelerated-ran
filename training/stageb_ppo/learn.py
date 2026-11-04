"""Clipped on-policy PPO with full-action ratios and rollback of Adam state."""
import copy
import torch
from .model import gae


def clone_state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def learn(actor, critic, actor_opt, critic_opt, data, cfg):
    x, mask, actions, old_logp, old_dist, rewards, next_x, next_mask = data
    n = len(rewards)
    with torch.no_grad():
        values = torch.cat([critic(x[i:i+16], mask[i:i+16]) for i in range(0, n, 16)])
        bootstrap = critic(next_x[None], next_mask[None])[0]
        advantages, targets = gae(rewards, values, bootstrap, cfg['gamma'], cfg['gae_lambda'])
        adv = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1.e-6)
        initial_logp = torch.cat([actor.replay(x[i:i+16], mask[i:i+16], actions[i:i+16])[0] for i in range(0,n,16)])
        error = float((initial_logp - old_logp).abs().max())
        assert error < 2.e-4, ('behavior probability mismatch', error)
    actor_before = clone_state(actor)
    records = []
    for epoch in range(cfg['ppo_epochs']):
        accepted = False
        for attempt in range(3):
            before, adam_before = clone_state(actor), copy.deepcopy(actor_opt.state_dict())
            actor_opt.zero_grad()
            loss_total = 0.
            for i in range(0,n,16):
                lp, ent, _ = actor.replay(x[i:i+16], mask[i:i+16], actions[i:i+16])
                ratio = (lp - old_logp[i:i+16]).exp()
                a = adv[i:i+16]
                objective = torch.minimum(ratio*a, ratio.clamp(1-cfg['clip'], 1+cfg['clip'])*a)
                loss = (-objective - cfg['entropy_coef']*ent/51).sum()/n
                loss.backward()
                loss_total += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), .5)
            assert torch.isfinite(norm)
            actor_opt.step()
            with torch.no_grad():
                divergences = []
                for i in range(0,n,16):
                    new = actor.distributions(x[i:i+16], mask[i:i+16], actions[i:i+16])
                    divergences.append((old_dist[i:i+16].exp()*(old_dist[i:i+16]-new)).sum((-1,-2)))
                kl = float(torch.cat(divergences).mean())
            accepted = bool(torch.isfinite(torch.tensor(kl)) and kl <= cfg['target_kl'])
            records.append(dict(epoch=epoch+1, attempt=attempt+1, accepted=accepted,
                conditional_kl_sum_mean=kl, loss=loss_total, gradient_norm=float(norm), lr=actor_opt.param_groups[0]['lr']))
            if accepted:
                break
            actor.load_state_dict(before)
            actor_opt.load_state_dict(adam_before)
            for group in actor_opt.param_groups:
                group['lr'] *= .5
        if not accepted:
            break
    for _ in range(cfg['critic_epochs']):
        for index in torch.randperm(n).split(32):
            critic_opt.zero_grad()
            loss = (critic(x[index],mask[index])-targets[index]).square().mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.)
            assert torch.isfinite(norm)
            critic_opt.step()
    with torch.no_grad():
        pred = torch.cat([critic(x[i:i+16],mask[i:i+16]) for i in range(0,n,16)])
    return dict(samples=n, behavior_logp_max_error=error, actor_updates=sum(r['accepted'] for r in records),
        steps=records, reward_mean=float(rewards.mean()), advantage_std=float(advantages.std(unbiased=False)),
        critic_mse=float((pred-targets).square().mean()),
        parameter_change_l2=float(torch.sqrt(sum((actor.state_dict()[k]-v).square().sum() for k,v in actor_before.items()))))

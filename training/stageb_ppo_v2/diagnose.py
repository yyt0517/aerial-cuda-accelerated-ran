"""Read-only diagnosis of the frozen 2026-10-29 experiment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from training.stageb_ppo.model import Value, gae
from training.stageb_ppo.native import ROOT, ABLATION, legacy, write

OLD=ROOT/'deliverables/ppo_no_ctbs_equal_20261029_v1/formal'
OUT=ROOT/'deliverables/ppo_diagnosis_continue_20261030_v1'


def main():
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled=False
    OUT.mkdir(parents=True,exist_ok=True)
    value_rows=[]
    for i in [1,6,12]:
        before=torch.load(OLD/('bc.pt' if i==1 else f'ppo_iter_{i-1:03d}.pt'),weights_only=False)
        after=torch.load(OLD/f'ppo_iter_{i:03d}.pt',weights_only=False)
        x,mask,actions,lp,dist,reward,nx,nm=torch.load(OLD/f'rollout_{i:03d}.pt',weights_only=False)['data']
        critic=Value();critic.load_state_dict(before['critic'])
        with torch.no_grad():
            v=critic(x,mask);last=critic(nx[None],nm[None])[0]
            adv,target=gae(reward,v,last,before['config']['gamma'],before['config']['gae_lambda'])
            critic.load_state_dict(after['critic']);pred=critic(x,mask)
        prob=dist.exp()
        value_rows.append(dict(iteration=i,value_before_mean=float(v.mean()),value_after_mean=float(pred.mean()),
            target_mean=float(target.mean()),target_std=float(target.std(unbiased=False)),
            mse=float((pred-target).square().mean()),explained_variance=float(1-(pred-target).var(unbiased=False)/target.var(unbiased=False)),
            mean_predictor_mse=float(target.var(unbiased=False)),
            entropy_per_token=float(-(prob*dist).sum(-1).mean()),
            mean_max_choice_probability=float(prob.max(-1).values.mean()),
            sampled_vs_prefix_argmax_fraction=float((actions!=dist.argmax(-1)).float().mean())))
        if i==12:
            trials=[]
            for scale in [1.,.01]:
                torch.manual_seed(13)
                fitted=Value();fitted.load_state_dict(before['critic'])
                with torch.no_grad():
                    fitted.net[-1].weight.mul_(scale);fitted.net[-1].bias.mul_(scale)
                opt=torch.optim.Adam(fitted.parameters(),lr=3.e-4)
                for step in range(400):
                    indices=torch.randperm(384)[:32]
                    loss=(fitted(x[indices],mask[indices])-scale*target[indices]).square().mean()
                    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(fitted.parameters(),1.);opt.step()
                with torch.no_grad():
                    prediction=fitted(x,mask)/scale
                for label,index in [('fit',slice(0,384)),('time_holdout',slice(384,None))]:
                    error=prediction[index]-target[index]
                    trials.append(dict(scale=scale,split=label,mse=float(error.square().mean()),
                        explained_variance=float(1-error.var(unbiased=False)/target[index].var(unbiased=False))))
            write(OUT/'critic_scale_probe.json',dict(steps_per_arm=400,rows=trials,
                scope='Fixed saved GAE targets; temporal split within one rollout, diagnostic only, not an independent return test.'))
    pd.DataFrame(value_rows).to_csv(OUT/'value_and_exploration.csv',index=False)
    b=legacy();fields=b.FIELDS
    statistics=[];ue_rows=[]
    for seed in [41,42,43]:
        rewards=pd.read_csv(OLD/f'train_seed{seed}/rewards.csv')
        data=np.fromfile(OLD/f'train_seed{seed}/metrics.bin',dtype='<f8').reshape(-1,36,len(fields))
        for stage,first,last in [('teacher',4001,4128),('ppo',4129,6176)]:
            s=rewards[rewards.tti.between(first,last)]
            statistics.append(dict(seed=seed,stage=stage,queue_mean_bytes=float(s.post_queue_bytes.mean()),
                complete_credit=float(s.complete_payload_bytes.mean()/86400),
                expiry_cost=float(s.expired_payload_bytes.mean()/86400),
                holding_cost=float(.01*s.post_queue_bytes.mean()/86400),reward=float(s.reward.mean())))
    for seed in [44,45]:
        for arm in ['bc','ppo','raw_global_equal']:
            path=OLD/f'eval_{arm}_seed{seed}' if arm!='raw_global_equal' else ABLATION/f'native/raw_global_equal/seed{seed}_current_20000'
            data=np.fromfile(path/'metrics.bin',dtype='<f8').reshape(-1,36,len(fields))[8000:12000]
            for ue in range(36):
                d=data[:,ue]
                ue_rows.append(dict(seed=seed,arm=arm,ue=ue,
                    goodput_mbps=float(d[:,fields.index('successful_bytes')].sum()*8/2e6),
                    prg_count=float(d[:,fields.index('actual_prg_count')].sum()),
                    queue_mean=float(d[:,fields.index('pre_queue_bytes')].mean()),
                    expired_bytes=float(d[:,fields.index('expired_bytes')].sum())))
    pd.DataFrame(statistics).to_csv(OUT/'reward_decomposition.csv',index=False)
    pd.DataFrame(ue_rows).to_csv(OUT/'matched_ue_diagnosis.csv',index=False)
    write(OUT/'diagnosis.json',dict(training_steps=6144,actor_optimizer_steps=24,
        bc_unique_states=384,teacher_ttis=[4001,4128],teacher_time_span_ms=64,
        gamma=2**(-1/400),gae_lambda=.995,gae_trace_effective_ttis=1/(1-2**(-1/400)*.995),
        old_holding_cost_per_full_ttl_in_packet_units=.01*400,
        state_omissions=['Explicit per-UE PRG counts and cumulative TBS in the sampled prefix',
            'Per-cell interference changes under blanking',
            'FIFO partial-packet CRC damage and clean completion progress'],
        observations=value_rows,reward_decomposition=statistics,
        limits=['MSE alone does not establish critic failure; scale probe and explained variance are reported.',
            'Diagnostic associations are not a factorial causal attribution.',
            'No policy, trajectory, or model was modified in this diagnosis.']))
    print(json.dumps(dict(values=value_rows,rewards=statistics,scale_probe=trials),indent=2),flush=True)


if __name__=='__main__':main()

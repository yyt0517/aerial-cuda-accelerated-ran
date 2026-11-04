"""Check checkpoint continuity and saved behavior against every native action.

Run in the existing CUDA container with the full local experiment directory.
This verifier does not rerun simulation or update any model.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from training.stageb_ppo.model import JointActor


def main(root, output):
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled=False
    root=Path(root)
    previous=torch.load(root/'bc.pt',map_location='cpu',weights_only=False)
    cfg=previous['config']
    history=json.loads((root/'training_metrics.json').read_text())
    model=JointActor()
    native={seed:np.load(root/f'train_seed{seed}/actions.npz')['actions'] for seed in cfg['train_seeds']}
    actor_steps=0
    checks=[]
    for iteration in range(1,cfg['iterations']+1):
        saved=torch.load(root/f'rollout_{iteration:03d}.pt',map_location='cpu',weights_only=False)
        x,mask,actions,old_lp,old_dist,*_=saved['data']
        current=torch.load(root/f'ppo_iter_{iteration:03d}.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(previous['actor'])
        generator=torch.Generator();generator.set_state(previous['sampler_state'])
        sampled,lp,dist=model.sample(x[0],mask[0],generator=generator)
        torch.testing.assert_close(sampled,actions[0],atol=0,rtol=0)
        torch.testing.assert_close(lp,old_lp[0],atol=0,rtol=0)
        torch.testing.assert_close(dist,old_dist[0],atol=0,rtol=0)
        # Reconstruct the remainder's RNG from the saved behavior distribution.
        probabilities=old_dist[1:].exp()
        for i in range(len(probabilities)):
            for t in range(51):
                draw=torch.multinomial(probabilities[i,t][None],1,generator=generator)[0,0]
                assert draw==actions[i+1,t]
        torch.testing.assert_close(generator.get_state(),current['sampler_state'],atol=0,rtol=0)
        actor_steps+=history[iteration-1]['actor_updates']
        assert {int(s['step']) for s in current['actor_optimizer']['state'].values()}=={actor_steps}
        critic_steps=iteration*cfg['critic_epochs']*(cfg['rollout']//32)
        assert {int(s['step']) for s in current['critic_optimizer']['state'].values()}=={critic_steps}
        for value in current['actor'].values():assert torch.isfinite(value).all()
        decoded=actions.numpy().reshape(-1,17,3).transpose(0,2,1).copy()
        decoded[decoded==36]=-1
        start=cfg['warmup']+cfg['teacher_steps']+((iteration-1)//len(cfg['train_seeds']))*cfg['rollout']
        np.testing.assert_array_equal(decoded,native[saved['seed']][start:start+len(actions)])
        checks.append(dict(iteration=iteration,seed=saved['seed'],native_actions=len(actions),
            actor_optimizer_steps=actor_steps,critic_optimizer_steps=critic_steps,
            behavior_distribution_exact_first_state=True,sampler_chain_exact=True))
        previous=current
    result=dict(passed=True,iterations=checks,total_native_ppo_actions=sum(r['native_actions'] for r in checks),
        actor_optimizer_steps=actor_steps,final_critic_optimizer_steps=critic_steps)
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='iterations'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    main(args.root,args.output)

"""Exercise minibatch optimization on a frozen, genuinely on-policy rollout."""
import json
import torch
from training.stageb_ppo.native import ROOT,write
from .model import ServiceActor,Value
from .minibatch_learn import learn


def main():
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False
    root=ROOT/'deliverables/ppo_diagnosis_continue_20261030_v1/formal'
    before=torch.load(root/'checkpoint_006.pt',weights_only=False)
    actor,critic=ServiceActor(),Value();actor.load_state_dict(before['actor']);critic.load_state_dict(before['critic'])
    actor_opt=torch.optim.Adam(actor.parameters());critic_opt=torch.optim.Adam(critic.parameters())
    actor_opt.load_state_dict(before['actor_optimizer']);critic_opt.load_state_dict(before['critic_optimizer'])
    initial_steps={int(s['step']) for s in actor_opt.state.values()}
    cfg=dict(before['protocol'],actor_minibatch=128,ppo_epochs=1,critic_epochs=1)
    segments=[torch.load(root/f'rollout_007_seed{s}.pt',weights_only=False) for s in [41,42,43]]
    torch.manual_seed(113)
    result,_=learn(actor,critic,actor_opt,critic_opt,segments,cfg,7)
    final_steps={int(s['step']) for s in actor_opt.state.values()}
    assert len(initial_steps)==len(final_steps)==1
    assert result['actor_steps']==24 and final_steps.pop()-initial_steps.pop()==24
    assert result['behavior_max_logprob_error']<1.e-4
    write(root.parent/'minibatch_optimizer_smoke.json',dict(passed=True,optimizer_steps=24,report=result))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

import unittest
import numpy as np
import torch

from training.stageb_ppo.model import JointActor, observation, action_tokens, native_action, gae, FEATURES
from training.stageb_ppo.reward import PacketReward


class Contracts(unittest.TestCase):
    def test_no_deficit_features_or_ctbs_headroom_mask(self):
        from types import SimpleNamespace
        ue=np.zeros((36,16),np.float32)
        ue[0,0]=1  # One byte remains; all 17 PRGs remain policy choices.
        obs=dict(obs_ue_features=ue,post_eq_sinr=np.ones((36,17,4)),
            teacher_action_ue_select=np.arange(36),teacher_action_prg_alloc=np.full(51,-1),
            action_mask_cell_ue=np.arange(3)[:,None]==(np.arange(36)//12)[None],
            action_mask_ue=ue[:,0]>0,action_mask_prg_cell=np.ones((3,17),bool))
        inputs=SimpleNamespace(reconstructed_mcs_for_ue=lambda u:27,prg_layer_cap=np.full((36,17),4))
        x,mask=observation(obs,inputs)
        self.assertTrue(mask[::3,0].all())
        self.assertFalse(mask[1::3,:36].any())
        ue[:,11]=100  # The legacy deficit field cannot affect this policy.
        other,other_mask=observation(obs,inputs)
        torch.testing.assert_close(x,other,atol=0,rtol=0)
        torch.testing.assert_close(mask,other_mask)

    def test_sampling_probability_replay_and_identity(self):
        torch.set_num_threads(1)
        torch.backends.mkldnn.enabled=False
        torch.manual_seed(7)
        actor=JointActor()
        x=torch.randn(51,37,FEATURES)
        mask=torch.zeros(51,37,dtype=torch.bool)
        mask[:,36]=True
        for t in range(51): mask[t,(t%3)*12:(t%3+1)*12]=True
        actions,old,dist=actor.sample(x,mask,generator=torch.Generator().manual_seed(11))
        logp,_,replayed=actor.replay(x[None],mask[None],actions[None])
        torch.testing.assert_close(logp[0],old,atol=2.e-4,rtol=0)
        torch.testing.assert_close(replayed[0],dist,atol=2.e-6,rtol=0)
        torch.testing.assert_close(action_tokens(native_action(actions)),actions)
        self.assertTrue(mask.gather(-1,actions[:,None]).all())
        (-logp.mean()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters()))

    def test_empty_queues_allow_blank_only(self):
        actor=JointActor()
        x=torch.zeros(51,37,FEATURES)
        mask=torch.zeros(51,37,dtype=torch.bool);mask[:,36]=True
        choices,lp,_=actor.sample(x,mask)
        self.assertTrue((choices==36).all());self.assertEqual(float(lp),0.)

    def test_fifo_partial_crc_and_full_expiry(self):
        r=PacketReward(users=1,ttl=3)
        r.before(1,[6000]);a=r.after([1000],[0])
        self.assertEqual(a['complete_payload_bytes'],0)
        r.before(2,[5000]);a=r.after([2000],[2000])
        self.assertEqual(a['complete_payload_bytes'],0)
        self.assertEqual(a['failed_complete_bytes'],3000)
        r.before(3,[3000]);r.after([1000],[1000])
        r.before(4,[3000]);a=r.after([3000],[3000])
        self.assertEqual(a['expired_payload_bytes'],3000)
        self.assertEqual(a['complete_payload_bytes'],3000)
        self.assertEqual(a['post_queue_bytes'],0)

    def test_truncation_bootstraps_next_state(self):
        a,target=gae(torch.tensor([1.,2.]),torch.tensor([.5,.75]),torch.tensor(3.),.9,1.)
        torch.testing.assert_close(target,torch.tensor([1+.9*(2+.9*3),2+.9*3]))
        torch.testing.assert_close(a,target-torch.tensor([.5,.75]))


if __name__=='__main__': unittest.main()

import copy
import unittest
import numpy as np
import torch
from training.stageb_ppo.reward import PacketReward
from training.stageb_ppo_v2.state import packet_state,reward
from training.stageb_ppo_v2.catalog import completion_bytes
from training.stageb_ppo_v2.model import ServiceActor


class Contracts(unittest.TestCase):
    def test_damage_and_partial_progress_are_causal(self):
        fifo=PacketReward();fifo.queues[0].extend([[1,700,False],[2,3000,True]])
        before=copy.deepcopy(fifo.queues)
        queues=np.zeros(36);queues[0]=6700
        features,phi=packet_state(fifo,10,queues)
        self.assertAlmostEqual(phi,2300/86400)
        self.assertAlmostEqual(features[0,3],0)
        self.assertGreater(features[0,1],0)
        self.assertEqual(fifo.queues,before)
        served=np.zeros((3,36));served[:,0]=[699,700,6700]
        complete=completion_bytes(fifo,10,queues,served)
        np.testing.assert_array_equal(complete[:,0],[0,3000,6000])

    def test_potential_shaping_telescopes(self):
        gamma=2**(-1/400);phi=[.03,.09,.05,0.]
        rec=dict(tti=1,complete_payload_bytes=3000,expired_payload_bytes=0,post_queue_bytes=6000)
        rewards=[reward(rec,a,b,gamma) for a,b in zip(phi,phi[1:])]
        shaped=sum(gamma**i*r['shaping_reward'] for i,r in enumerate(rewards))
        self.assertAlmostEqual(shaped,-phi[0]+gamma**3*phi[-1])
        self.assertAlmostEqual(rewards[0]['base_reward'],(3000-6000/400)/86400)

    def test_physical_prior_and_exact_behavior_replay(self):
        torch.manual_seed(3);actor=ServiceActor()
        batch=dict(state=np.zeros((36,15),np.float32),candidate=np.zeros((16,36,9),np.float32),
            summary=np.zeros((16,7),np.float32),mask=np.arange(16)<3,prior=np.zeros(16,np.float32))
        batch['prior'][:3]=[-1.,0.,-.2]
        choice,lp,dist=actor.sample(batch,greedy=True)
        self.assertEqual(choice,1);self.assertEqual(float(dist[3:].exp().sum()),0)
        action,lp,dist=actor.sample(batch,generator=torch.Generator().manual_seed(7))
        replay=actor(*[torch.as_tensor(batch[k])[None] for k in ['state','candidate','summary','mask','prior']])[0]
        self.assertEqual(float(replay[action].detach()),lp)
        self.assertTrue(torch.equal(replay,dist))


if __name__=='__main__':unittest.main()

import unittest
from collections import deque
from types import SimpleNamespace
import numpy as np
import pandas as pd
from training.stageb_ppo_tail_balance.reward import pending_cost, reward, reconstruct_pending


CFG=dict(holding=1/400,age_threshold_tti=20,age_ramp_tti=20,age_multiplier=2.,expiry_penalty=2.,gamma=2**(-1/400))


class WholePacketRewardTests(unittest.TestCase):
    def test_partial_and_crc_damage_do_not_discount_waiting(self):
        fifo=SimpleNamespace(size=3000,ttl=400,queues=[deque([[1,3000,False],[1,1,False],[1,2,True]])])
        cost=pending_cost(fifo,40,CFG)
        self.assertEqual(cost['pending_full_bytes'],9000)
        self.assertEqual(cost['whole_wait_cost_bytes'],22.5)
        self.assertEqual(cost['age_cost_bytes'],45.)

    def test_age_ramp_and_complete_expired_credit(self):
        fifo=SimpleNamespace(size=3000,ttl=400,queues=[deque([[1,1,False]])])
        self.assertEqual(pending_cost(fifo,20,CFG)['age_cost_bytes'],0.)
        self.assertEqual(pending_cost(fifo,30,CFG)['age_cost_bytes'],7.5)
        self.assertEqual(pending_cost(fifo,400,CFG)['age_cost_bytes'],15.)
        result=reward(dict(tti=30,complete_payload_bytes=6000,expired_payload_bytes=3000),fifo,0.,0.,CFG)
        self.assertAlmostEqual(result['reward'],(6000-2*3000-7.5-7.5)/86400)

    def test_independent_lifecycle_boundaries(self):
        pk=pd.DataFrame(dict(arrival_tti=[1,1,1],terminal_tti=[2,40,-1],
            packet_size_bytes=[3000]*3,outcome=['mac_completed','expired','pending'],censored=[False,False,True]))
        full,age=reconstruct_pending(pk,42,CFG)
        self.assertEqual(full[0],9000);self.assertEqual(full[1],6000)
        self.assertEqual(full[39],6000);self.assertEqual(full[40],3000)
        self.assertEqual(full[41],3000)
        self.assertEqual(age[19],0);self.assertEqual(age[29],3000)
        self.assertEqual(age[39],6000);self.assertEqual(age[41],3000)


if __name__=='__main__':unittest.main()

import unittest
from collections import deque
import numpy as np
from .test_physics import fixture
from .physics import NativePhy
from .bounded_dpls import BoundedPhy,solve_bounded
from .persistent_dpls import solve_integer_reference
from .bounded_campaign import audit_round
from .ttl_weights import hol_urgency,objective_weights
from training.stageb_ppo.reward import PacketReward

class BoundedTests(unittest.TestCase):
    def test_same_frozen_phy_binary_results(self):
        p,a=fixture();x,y=NativePhy(p),BoundedPhy(p)
        try:np.testing.assert_array_equal(x.evaluate(a).values,y.evaluate(a).values)
        finally:x.close();y.close()

    def test_every_bound_and_sparse_cache_on_full_independent_paths(self):
        for seed,cells,prgs in ((17,3,5),(39,3,7),(113,3,5),(19,7,3)):
            p,a=fixture(cells=cells,prgs=prgs,seed=seed)
            p.weights[:]=np.random.default_rng(seed).uniform(.2,3,p.U)
            trace=[]
            def save(i):trace.append({k:v.copy() if isinstance(v,np.ndarray) else v for k,v in i.items()})
            ref=solve_integer_reference(p,a,observer=save)
            def audit(i):audit_round(i,trace[i['iteration']-1])
            actual=solve_bounded(p,a,observer=audit)
            np.testing.assert_array_equal(actual[0],ref[0]);np.testing.assert_array_equal(actual[1],ref[1])
            self.assertEqual(actual[2]['rounds'],len(trace));self.assertTrue(actual[2]['converged'])
            self.assertEqual(actual[2]['candidate_evaluations'],actual[2]['scored_edits']+actual[2]['initialization_evaluations'])

    def test_causal_urgency_expiry_boundary_and_no_fifo_mutation(self):
        f=PacketReward(users=3);f.queues=[deque([[1,3000,False],[2,1000,True]]),deque(),deque([[1,1000,False]])]
        before=[list(q) for q in f.queues];u,s=hol_urgency(f,401,[4000,3000,0])
        np.testing.assert_array_equal(s,[1,400,400]);np.testing.assert_allclose(u,[.9975,0,0])
        self.assertEqual(before,[list(q) for q in f.queues]);self.assertEqual(f.expired,0)

    def test_zero_coefficient_is_exact_control_and_positive_weights_are_normalized(self):
        b=np.array([.777777777,1.222222223]);u=np.array([0.,1.])
        np.testing.assert_array_equal(objective_weights(b,u,0),b)
        w=objective_weights(b,u,2);self.assertAlmostEqual(w.mean(),1.);self.assertTrue(np.all(w>0))
        self.assertGreater(w[1]/w[0],b[1]/b[0])

if __name__=='__main__':unittest.main()

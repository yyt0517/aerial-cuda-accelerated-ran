import unittest
import numpy as np
from .closed_data import discounted_window
from .deadline import causal_features,components,packet_objective
from .test_physics import fixture
from .physics import NativePhy,EXPECTED
from .search import solve,SearchConfig
from .quality import select


class DeadlineTests(unittest.TestCase):
    def test_delayed_return_matches_explicit_finite_horizon(self):
        x=np.zeros((9,2,3));x[3,0,1]=4;x[6,1,0]=2
        got=discounted_window(x,4,.9)
        for t in range(9):
            expected=sum(.9**k*x[t+k] for k in range(min(4,9-t)))/sum(.9**k for k in range(4))
            np.testing.assert_allclose(got[t],expected,atol=1.e-14)
        self.assertEqual(got[4,0,1],0)

    def test_deadline_inputs_ignore_post_action_outcomes(self):
        x=np.zeros((2,16));x[:,0]=3000;x[:,8]=[190,1];x[:,9]=[10,199]
        obs=dict(obs_ue_features=x,native_info={'future':123})
        a=causal_features(obs,np.array([500,2400]));obs['native_info']['future']=-999
        np.testing.assert_array_equal(a,causal_features(obs,np.array([500,2400])))
        self.assertGreater(a[0,5],a[1,5]);self.assertLess(a[0,1],a[1,1])

    def test_urgent_weak_service_changes_risk_reward(self):
        p,root=fixture();p.deadline=np.zeros((p.U,8));p.deadline[:,1]=1
        p.deadline[0,1]=0;p.deadline[0,5]=1
        values=np.zeros((2,p.U,8));values[0,0,EXPECTED]=100;values[1,1,EXPECTED]=100
        raw,risk,*_=components(p,values)
        self.assertEqual(raw[0],raw[1]);self.assertGreater(risk[0],risk[1])

    def test_shaped_objective_cannot_spend_physical_guarantee(self):
        p,root=fixture()
        full,_,baseline=solve(p,root,SearchConfig(mode='greedy',budget_ms=None,max_candidates=10000))
        # Adversarial shaping asks to reduce exactly the protected objective.
        action,_,result=solve(p,root,SearchConfig(budget_ms=None,max_candidates=2000),
            initial_action=full,objective=lambda values:-(values[:,:,EXPECTED]@p.weights))
        np.testing.assert_array_equal(action,full)
        self.assertGreaterEqual(result['score'],baseline['score'])
        self.assertGreaterEqual(result['chosen_dequeue'],result['floor'])

    def test_zero_risk_reproduces_original_search(self):
        p,root=fixture();p.deadline=np.zeros((p.U,8));p.deadline[:,1]=1
        config=SearchConfig(budget_ms=None,max_candidates=2000,max_rounds=3)
        old,_,a=solve(p,root,config)
        new,_,b=solve(p,root,config,objective=packet_objective(p))
        np.testing.assert_array_equal(old,new)
        self.assertEqual(a['score'],b['score'])
        self.assertEqual(a['candidate_evaluations'],b['candidate_evaluations'])

    def test_direct_and_guarded_physical_risk_controls_remain_verified(self):
        p,root=fixture();p.deadline=np.zeros((p.U,8));p.deadline[:,1]=1
        p.deadline[0,1]=0;p.deadline[0,5]=1
        for arm in ('risk_physical_direct','risk_physical_lns'):
            with self.subTest(arm=arm):
                action,values,record=select(p,root,arm)
                phy=NativePhy(p)
                try:
                    independent=phy.evaluate(action).values[0]
                    draft=phy.evaluate(root).values[0]
                finally:
                    phy.close()
                np.testing.assert_array_equal(values,independent)
                self.assertGreaterEqual(record['score'],float(draft[:,EXPECTED]@p.weights)-1.e-7)
                self.assertGreaterEqual(record['chosen_dequeue'],record['floor'])
                if arm=='risk_physical_direct':
                    self.assertAlmostEqual(record['objective_score'],float(packet_objective(p)(values)[0]))
                    self.assertEqual(record['config']['repair_width'],16)
                    self.assertEqual(record['config']['ue_topk'],p.U)
                else:
                    self.assertGreaterEqual(record['score'],record['baseline_score']-1.e-7)


if __name__=='__main__':unittest.main()

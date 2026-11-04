"""Sparse proxy equivalence, wide move coverage and causal frontier selection."""
import unittest
import numpy as np
from .test_physics import fixture
from .physics import ReferencePhy,DEQUEUE
from .resource_conditions import Conditions,ConditionalPhy,ConditionalConfig,logical,satisfies
from .resource_conditions_fast import fast_pool,solve_conditional
from .joint_reorder import JointFeatures,ReorderConfig,solve_joint,nondominated,final_key,frontier_roots,service_vector
from .search import ProposalGraph


class JointReorderTests(unittest.TestCase):
    def context(self,prgs=5,seed=17,uniform=False):
        p,a=fixture(prgs=prgs,seed=seed);a[:,1::2]+=1
        if uniform:p.caps[:]=2
        n,l=logical(p,a);condition=Conditions(n[0],l[0],a>=0).validate(p)
        phy=ConditionalPhy(p);self.addCleanup(phy.close);base=phy.evaluate(a);bank=phy.bank()
        return p,a,condition,phy,base,JointFeatures(p,bank,condition)

    def test_narrow_pool_matches_old_valid_candidate_order(self):
        for seed in (17,29,43):
            p,a,c,phy,base,f=self.context(seed=seed)
            old=fast_pool(p,ProposalGraph(phy),base,c);old=old[satisfies(p,old,c)]
            np.testing.assert_array_equal(old,f.pool(base,False,20261102))

    def test_sparse_proxy_matches_root_anchored_full_proxy(self):
        p,a,c,phy,base,f=self.context(prgs=8,uniform=True)
        actions=f.pool(base,True,20261102);dense,_,_,_=phy.proxy(base,actions)
        before=phy.stats.copy();sparse,predicted,valid=f.proxy(base,actions)
        self.assertEqual(phy.stats,before);self.assertTrue(valid.all());self.assertGreater(f.gamma_reads,0)
        np.testing.assert_allclose(sparse,dense,rtol=1.e-10,atol=1.e-7)
        self.assertTrue(np.isfinite(predicted).all())

    def test_wide_pool_adds_four_slot_moves_and_all_prg_coverage(self):
        p,a,c,phy,base,f=self.context(prgs=8,uniform=True)
        narrow=f.pool(base,False,3);wide=f.pool(base,True,3)
        self.assertTrue({x.tobytes() for x in narrow}.issubset({x.tobytes() for x in wide}))
        new=[x for x in wide if np.count_nonzero(x!=a)==4]
        self.assertGreater(len(new),0);self.assertTrue(any(np.any(np.nonzero(x!=a)[1]>=6) for x in new))
        self.assertTrue(satisfies(p,wide,c).all())

    def test_invalid_activity_or_quota_delta_is_rejected(self):
        p,a,c,phy,base,f=self.context()
        b=a.copy();b[0,0]=-1
        _,_,valid=f.proxy(base,b)
        self.assertFalse(valid[0])

    def test_narrow_max_j_decision_matches_previous_decoder(self):
        p,a,c,_,_,_=self.context()
        old=solve_conditional(p,a,c,ConditionalConfig(max_candidates=73),offline_diagnostic=True)
        new=solve_joint(p,a,c,ReorderConfig(False,False,0,max_candidates=73),offline_diagnostic=True)
        np.testing.assert_array_equal(old[0],new[0]);np.testing.assert_array_equal(old[1],new[1])

    def test_frontier_keeps_service_tradeoff_and_first_tie(self):
        objectives={'j':np.array([100,1,2]),'s':np.array([99,10,12]),'d':np.array([98,1,2]),'tie':np.array([99,10,12])}
        self.assertEqual(nondominated(list(objectives),objectives),['j','s'])
        roots=frontier_roots(list(objectives),objectives,2)
        self.assertEqual(roots,['j','s'])

    def test_final_j_loss_is_relative_to_one_fixed_best(self):
        obj={'best':np.array([100.,1,2]),'half':np.array([99.6,10,12]),'two':np.array([98.1,20,22]),'too_low':np.array([95,30,40])}
        self.assertEqual(final_key(list(obj),obj,0),'best')
        self.assertEqual(final_key(list(obj),obj,.005),'half')
        self.assertEqual(final_key(list(obj),obj,.02),'two')

    def test_service_metric_fixed_mask_and_empty_mask(self):
        v=np.zeros((3,8));v[:,6]=[0,20,30]
        np.testing.assert_array_equal(service_vector(v,np.zeros(3,bool)),[0,0])
        s=service_vector(v,np.array([False,True,True]));self.assertGreater(s[0],0);self.assertEqual(s[1],20)

    def test_joint_search_caps_conditions_and_independent_physics(self):
        p,a,c,_,_,_=self.context(prgs=8,uniform=True)
        for config in (ReorderConfig(True,False,0,65),ReorderConfig(True,True,.005,65),ReorderConfig(True,True,.02,65)):
            action,v,r,group=solve_joint(p,a,c,config,offline_diagnostic=True)
            self.assertTrue(satisfies(p,action,c)[0]);self.assertLessEqual(r['candidate_evaluations'],65)
            self.assertLessEqual(r['feature_roots'],56);self.assertGreaterEqual(v[:,DEQUEUE].sum(),r['floor'])
            self.assertGreaterEqual(r['score']+1.e-8,(1-config.final_j_loss)*r['best_observed_j'])
            np.testing.assert_allclose(v,ReferencePhy(p).evaluate(action)[0],rtol=1.e-10,atol=1.e-8)
            self.assertEqual(group.shape,(p.U,))

    def test_teacher_conditions_rejected_without_offline_flag(self):
        p,a,c,_,_,_=self.context()
        with self.assertRaises(ValueError):solve_joint(p,a,c)


if __name__=='__main__':unittest.main()

"""Directed multi-swap legality, causal slots, budget and disabled parity."""
from unittest import TestCase,main
from unittest.mock import patch
import numpy as np
from .test_physics import fixture
from .test_weak_service import TraceMemo
from .weak_service import ServiceConfig,ServiceMetric,solve_service
from .directed_reorder import DirectedConfig,DirectedFeatures,target_ues,choose_batch,solve_directed
from .joint_reorder import JointFeatures
from .resource_conditions import Conditions,ConditionalPhy,logical,satisfies
from .physics import EXPECTED,DEQUEUE,ReferencePhy


class DirectedTests(TestCase):
    def context(self,seed=17,uniform=True):
        p,a=fixture(prgs=8,seed=seed);a[:,1::2]+=1
        if uniform:p.caps[:]=2
        n,l=logical(p,a);c=Conditions(n[0],l[0],a>=0).validate(p)
        phy=ConditionalPhy(p);self.addCleanup(phy.close);base=phy.evaluate(a);f=JointFeatures(p,phy.bank(),c)
        return p,a,c,phy,base,f,DirectedFeatures(f)

    def test_pool_preserves_exact_conditions_across_random_layers(self):
        for seed in (17,23,41):
            p,a,c,phy,base,f,d=self.context(seed,False);targets=target_ues(p,base.values[0],c.counts>0)
            pool=d.pool(base,targets,4)
            self.assertTrue(satisfies(p,pool,c).all());self.assertLessEqual(len(pool),256)

    def test_greedy_chains_reach_more_than_four_changed_slots(self):
        p,a,c,phy,base,f,d=self.context()
        f.gamma[:]=1
        for u in (0,2,4):f.gamma[u,::2]=.1;f.gamma[u,1::2]=10
        pool=d.pool(base,np.array([0,2,4]),4)
        self.assertTrue((np.count_nonzero(pool!=a,axis=(1,2))>4).any())
        self.assertTrue(satisfies(p,pool,c).all())

    def test_compositions_change_multiple_cells_without_activity_changes(self):
        p,a,c,phy,base,f,d=self.context();f.gamma[:]=1
        for u in (0,2,4):f.gamma[u,::2]=.1;f.gamma[u,1::2]=10
        pool=d.pool(base,np.array([0,2,4]),3)
        self.assertTrue((np.any(pool!=a,axis=2).sum(1)==3).any())
        np.testing.assert_array_equal(pool>=0,np.broadcast_to(a>=0,pool.shape))

    def test_pool_deterministic_unique_and_root_excluded(self):
        p,a,c,phy,base,f,d=self.context();targets=np.array([0,2,4])
        one=d.pool(base,targets,4);two=d.pool(base,targets,4)
        np.testing.assert_array_equal(one,two);self.assertEqual(len(one),len({x.tobytes() for x in one}))
        self.assertNotIn(a.tobytes(),{x.tobytes() for x in one})

    def test_pool_uses_no_candidate_phy_and_accounts_gamma_reads(self):
        p,a,c,phy,base,f,d=self.context();before=phy.stats.copy();pool=d.pool(base,np.array([0,2,4]),4)
        self.assertEqual(before,phy.stats);self.assertGreater(d.gamma_reads,0)
        if len(pool):self.assertTrue(f.proxy(base,pool)[2].all())

    def test_empty_targets_emit_no_candidates(self):
        p,a,c,phy,base,f,d=self.context();pool=d.pool(base,np.array([],dtype=np.int64),4)
        self.assertEqual(pool.shape,(0,p.C,p.P))

    def test_target_selection_excludes_empty_and_saturated_ues(self):
        p,a,c,phy,base,f,d=self.context();v=base.values[0].copy();v[:,DEQUEUE]=100;v[:,EXPECTED]=0
        v[0,DEQUEUE]=0;v[1,EXPECTED]=100;group=np.ones(p.U,bool);group[2]=False
        targets=target_ues(p,v,group)
        self.assertNotIn(0,targets);self.assertNotIn(1,targets);self.assertNotIn(2,targets)
        self.assertEqual(len(targets),len(set(targets)))

    def test_directed_slots_respect_retention_and_partial_budget(self):
        p,a,c,phy,base,f,d=self.context();metric=ServiceMetric(p,c,np.ones(p.U,bool),ServiceConfig(metric='quota_exp'));metric.weights=p.weights
        root_j=float(base.values[0,:,EXPECTED]@p.weights);pred=np.tile(base.values[0,:,EXPECTED],(12,1))
        pred[0]=metric.dequeue;scores=np.full(12,root_j);scores[0]=.5*root_j
        for size in (1,3,4,5,8):
            chosen,reserved=choose_batch(scores,pred,np.ones(12,bool),base,metric,np.arange(p.U),size,DirectedConfig(directed_slots=4),0)
            self.assertLessEqual(len(chosen),size);self.assertEqual(len(chosen),len(set(chosen)));self.assertNotIn(0,reserved)

    def test_disabled_proposals_match_previous_complete_queries_and_values(self):
        p,a,c,phy,base,f,d=self.context()
        with patch('training.stageb_lns.weak_service.ExactMemo',TraceMemo):
            old=solve_service(p,a,c,ServiceConfig(metric='quota_exp',max_candidates=73),offline_diagnostic=True);events=TraceMemo.latest.events.copy()
        with patch('training.stageb_lns.directed_reorder.ExactMemo',TraceMemo):
            new=solve_directed(p,a,c,DirectedConfig(directed_slots=0,joint_candidates=False,max_candidates=73),offline_diagnostic=True)
            self.assertEqual(events,TraceMemo.latest.events)
        for i in (0,1,3):np.testing.assert_array_equal(old[i],new[i])
        for k in ('candidate_evaluations','ue_evaluations','mmse_requests','mmse_misses','proxy_candidates','feature_roots'):
            self.assertEqual(old[2][k],new[2][k])

    def test_all_variants_keep_floor_budget_conditions_and_physics(self):
        p,a,c,phy,base,f,d=self.context()
        configs=[DirectedConfig(joint_candidates=False,max_candidates=65),DirectedConfig(max_candidates=65),
            DirectedConfig(directed_slots=4,max_candidates=65),DirectedConfig(alternate_sparse=True,max_candidates=65)]
        for cfg in configs:
            action,v,r,g=solve_directed(p,a,c,cfg,offline_diagnostic=True)
            self.assertLessEqual(r['candidate_evaluations'],65);self.assertLessEqual(r['feature_roots'],56)
            self.assertGreaterEqual(v[:,DEQUEUE].sum(),r['floor']);self.assertTrue(satisfies(p,action,c)[0])
            np.testing.assert_allclose(v,ReferencePhy(p).evaluate(action)[0],rtol=1e-10,atol=1e-8)
            self.assertEqual(r['score'],r['best_observed_j'])

    def test_sparse_rounds_retain_wide_rounds(self):
        p,a,c,phy,base,f,d=self.context()
        _,_,r,_=solve_directed(p,a,c,DirectedConfig(alternate_sparse=True,max_candidates=65),offline_diagnostic=True)
        self.assertGreater(r['wide_rounds'],0);self.assertGreater(r['sparse_rounds'],0)

    def test_invalid_configuration_and_offline_fence(self):
        with self.assertRaises(ValueError):DirectedConfig(directed_slots=3)
        with self.assertRaises(ValueError):DirectedConfig(chain_depth=5)
        with self.assertRaises(ValueError):DirectedConfig(alternate_sparse=True,joint_candidates=False)
        p,a,c,phy,base,f,d=self.context()
        with self.assertRaises(ValueError):solve_directed(p,a,c)


if __name__=='__main__':main()

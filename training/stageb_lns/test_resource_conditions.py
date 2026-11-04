"""Exact quota/layer feasibility, no position leakage and paid physics work."""
import itertools
import unittest
import numpy as np
from .resource_conditions import (Conditions,ConditionalPhy,ConditionalConfig,DecodeInfeasible,
    logical,satisfies,matching,solve_conditional)
from .test_physics import fixture
from .physics import ReferencePhy,COUNT,LAYER,SINR,DEQUEUE


class ResourceConditionTests(unittest.TestCase):
    def test_matching_equals_small_exhaustive_quota_and_layer_assignment(self):
        p,a=fixture(cells=1,prgs=4);p.caps[:]=2;p.caps[0,0]=1
        condition=Conditions(np.array([2,2]),np.array([1,2])).validate(p)
        utility=np.random.default_rng(3).normal(size=(p.U,p.P));actual=matching(p,condition,utility)
        scores=[]
        for row in itertools.product(range(2),repeat=4):
            row=np.array([row],np.int64)
            if satisfies(p,row,condition)[0]:scores.append(sum(utility[u,j] for j,u in enumerate(row[0])))
        self.assertAlmostEqual(sum(utility[u,j] for j,u in enumerate(actual[0])),max(scores))
        self.assertEqual(actual[0,0],0)

    def test_exact_min_layer_requires_a_witness_not_just_lower_bound(self):
        p,a=fixture(cells=1,prgs=3);p.caps[:]=4
        condition=Conditions(np.array([1,0]),np.array([2,0])).validate(p)
        with self.assertRaises(DecodeInfeasible):matching(p,condition,np.ones((p.U,p.P)))

    def test_joint_condition_enforces_activity_positions_and_conservation(self):
        p,a=fixture(cells=1,prgs=4);p.caps[:]=2
        condition=Conditions(np.array([1,1]),np.array([2,2]),np.array([[0,1,0,1]])).validate(p)
        actual=matching(p,condition,np.zeros((p.U,p.P)))
        self.assertTrue(satisfies(p,actual,condition)[0]);np.testing.assert_array_equal(actual>=0,condition.activity)
        with self.assertRaises(ValueError):Conditions(condition.counts,condition.layers,np.ones((1,4))).validate(p)

    def test_bank_matches_independent_single_prg_physics(self):
        p,a=fixture(prgs=3);phy=ConditionalPhy(p);self.addCleanup(phy.close)
        bank=phy.bank();self.assertEqual(phy.stats['candidate_evaluations'],0)
        self.assertEqual(phy.stats['mmse_misses'],p.U*p.P*2**(p.C-1))
        reference=ReferencePhy(p)
        for u in range(p.U):
            for mask in range(1<<p.C):
                action=np.full_like(a,-1);action[p.serving[u],0]=u
                for c in range(p.C):
                    if c!=p.serving[u] and mask&(1<<c):action[c,0]=2*c
                expected=reference.evaluate(action)[0,u,SINR]
                actual=10*np.log10(max(bank[u,0,mask,p.caps[u,0]-1],1.e-30))
                self.assertAlmostEqual(actual,expected,places=8)

    def test_proxy_has_no_exact_schedule_or_native_bler_calls(self):
        p,a=fixture();phy=ConditionalPhy(p);self.addCleanup(phy.close);base=phy.evaluate(a);phy.bank()
        b=a.copy();b[0,0]=-1;before=phy.stats.copy();_,_,n,l=phy.proxy(base,np.stack([a,b]))
        after=phy.stats;self.assertEqual(after['candidate_evaluations'],before['candidate_evaluations'])
        self.assertEqual(after['ue_evaluations'],before['ue_evaluations']);self.assertEqual(after['mmse_misses'],before['mmse_misses'])
        self.assertGreater(after['mmse_requests'],before['mmse_requests'])
        exact=ReferencePhy(p).evaluate(np.stack([a,b]))
        np.testing.assert_array_equal(n,exact[:,:,COUNT]);np.testing.assert_array_equal(l,np.maximum(exact[:,:,LAYER],0))

    def test_compressed_conditions_cannot_carry_teacher_identity_positions(self):
        p,a=fixture(cells=1,prgs=4);p.caps[:]=2
        one=np.array([[0,1,0,1]],np.int64);two=np.array([[1,0,1,0]],np.int64)
        n,l=logical(p,one);m,k=logical(p,two)
        np.testing.assert_array_equal(n,m);np.testing.assert_array_equal(l,k)
        c1=Conditions(n[0],l[0],one>=0);c2=Conditions(m[0],k[0],two>=0)
        config=ConditionalConfig(max_candidates=30,seed_candidates=6)
        r1=solve_conditional(p,a,c1,config,offline_diagnostic=True);r2=solve_conditional(p,a,c2,config,offline_diagnostic=True)
        np.testing.assert_array_equal(r1[0],r2[0]);np.testing.assert_array_equal(r1[1],r2[1])

    def test_teacher_conditions_require_offline_mode(self):
        p,a=fixture();n,l=logical(p,a)
        with self.assertRaises(ValueError):solve_conditional(p,a,Conditions(n[0],l[0]))

    def test_capped_search_has_independent_final_physics_and_exact_conditions(self):
        p,a=fixture();a[:,1::2]+=1;n,l=logical(p,a)
        for condition in (Conditions(),Conditions(n[0],l[0]),Conditions(activity=a>=0),Conditions(n[0],l[0],a>=0)):
            result=solve_conditional(p,a,condition,ConditionalConfig(max_candidates=65),offline_diagnostic=True)
            self.assertLessEqual(result[2]['candidate_evaluations'],65);self.assertLessEqual(result[2]['feature_roots'],56)
            self.assertGreaterEqual(result[1][:,DEQUEUE].sum(),result[2]['floor'])
            np.testing.assert_allclose(result[1],ReferencePhy(p).evaluate(result[0])[0],rtol=1.e-10,atol=1.e-8)
            self.assertTrue(result[2]['condition_feasible']);self.assertTrue(satisfies(p,result[0],condition.validate(p))[0])

    def test_missing_conditional_feasibility_is_unknown_not_zero(self):
        p,a=fixture();condition=Conditions(activity=np.zeros_like(a,bool))
        r=solve_conditional(p,a,condition,ConditionalConfig(max_candidates=1),offline_diagnostic=True)
        self.assertFalse(r[2]['condition_feasible']);self.assertIsNone(r[2]['best_conditional_j'])
        self.assertTrue(r[2]['fallback_to_raw']);self.assertFalse(r[2]['condition_satisfied'])


if __name__=='__main__':unittest.main()

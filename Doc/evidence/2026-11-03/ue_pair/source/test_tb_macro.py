"""Physical closure, paid continuation, cache scope and controlled coverage."""
import unittest
import numpy as np

from .tb_macro import (MacroConfig,ExactMemo,Descriptors,candidate_pools,
                       finite_continue,solve_macro)
from .bounded_probes import ProbePhy
from .test_physics import fixture
from .physics import ReferencePhy,DEQUEUE,EXPECTED
from .search import greedy_candidates


class MacroTests(unittest.TestCase):
    def context(self,p,a,budget=212):
        phy=ProbePhy(p);self.addCleanup(phy.close)
        memo=ExactMemo(phy,a,budget)
        return phy,memo,Descriptors(p,phy,56)

    def test_full_tb_proxy_dequeue_includes_min_layer_outside_block(self):
        p,a=fixture();p.caps[:]=4;p.caps[0,0]=1
        phy,memo,d=self.context(p,a)
        b=a.copy();b[0,0]=-1
        before=phy.stats.copy()
        _,total=d.proxies(memo.root,np.stack([a,b]))
        reference=ReferencePhy(p).evaluate(np.stack([a,b]))
        np.testing.assert_allclose(total,reference[:,:,DEQUEUE].sum(1))
        self.assertGreater(reference[1,0,DEQUEUE],reference[0,0,DEQUEUE])
        self.assertEqual(before,phy.stats)

    def test_macro_candidates_are_legal_and_extend_original_neighborhood(self):
        p,a=fixture();a[:,1::2]+=1;phy,memo,d=self.context(p,a)
        singles,macros=candidate_pools(p,d,memo.root,0)
        old={v.tobytes() for v in greedy_candidates(p,a)}
        self.assertTrue(all(v.tobytes() in old for v in singles))
        self.assertGreater(len(macros),0)
        self.assertTrue(all(v.tobytes() not in old for v in macros))
        self.assertTrue(any(len(np.unique(np.nonzero(v!=a)[1]))>1 for v in macros))
        p.actions(macros)
        self.assertTrue(all(2<=np.count_nonzero(v!=a)<=3 for v in macros))

    def test_complete_action_memo_is_exact_across_roots_and_local_to_decision(self):
        p,a=fixture();phy,memo,d=self.context(p,a)
        b=a.copy();b[0,0]=1;c=a.copy();c[1,2]=-1
        scored,_=memo.evaluate(np.stack([b,c]),memo.root)
        count=phy.stats['candidate_evaluations']
        reused,complete=memo.evaluate(np.stack([a,c]),phy.take(scored,0))
        self.assertTrue(complete);self.assertEqual(count,phy.stats['candidate_evaluations'])
        np.testing.assert_allclose(reused.values,ReferencePhy(p).evaluate(np.stack([a,c])),rtol=1.e-10,atol=1.e-8)
        _,fresh,_=self.context(p,a)
        self.assertTrue(fresh.unseen(c[None])[0])

    def bridge(self,budget=20):
        p,a=fixture(cells=1,prgs=2);p.h*=100;p.queue[:]=1000;p.caps[:]=1
        a[0]=[0,1];p.caps[0,1]=4
        phy,memo,d=self.context(p,a,budget)
        prefix=a.copy();prefix[0,0]=-1
        floor=.9*memo.root.values[0,:,DEQUEUE].sum()
        return p,a,phy,memo,prefix,floor

    def test_paid_continuation_can_bridge_an_infeasible_intermediate(self):
        p,a,phy,memo,prefix,floor=self.bridge()
        best,r=finite_continue(p,memo,memo.root,prefix,(((0,1),(1,0)),),floor)
        self.assertFalse(r['seed_feasible']);self.assertTrue(r['rescued_unacceptable_seed'])
        self.assertTrue(r['complete']);self.assertGreater(r['physical_scores'],0)
        self.assertIsNone(r['observed_future_gain'])
        self.assertGreater(best.values[0,:,DEQUEUE].sum(),floor)
        np.testing.assert_allclose(best.values,ReferencePhy(p).evaluate(best.actions),rtol=1.e-10,atol=1.e-8)

    def test_missing_continuation_value_is_unknown_not_zero(self):
        p,a,phy,memo,prefix,floor=self.bridge(budget=1)
        best,r=finite_continue(p,memo,memo.root,prefix,(((0,1),(1,0)),),floor)
        self.assertIsNone(best);self.assertFalse(r['complete'])
        self.assertIsNone(r['seed_j']);self.assertIsNone(r['best_observed_j'])
        self.assertEqual(r['physical_scores'],0)

    def test_common_prefix_and_complete_prg_coverage_before_continuation(self):
        p,a=fixture(prgs=5);snapshots=[]
        for family,cont in [('single',False),('joint',False),('single',True),('joint',True)]:
            action,v,r,snapshot=solve_macro(p,a,MacroConfig(family,cont,212))
            snapshots.append(snapshot)
            self.assertEqual(r['covered_prgs'],p.P);self.assertEqual(r['second_pass_prgs'],p.P)
            self.assertLessEqual(r['continuation_calls'],4)
            self.assertLessEqual(r['candidate_evaluations'],212);self.assertLessEqual(r['feature_roots'],56)
            self.assertGreaterEqual(v[:,DEQUEUE].sum(),r['floor'])
            np.testing.assert_allclose(v,ReferencePhy(p).evaluate(action)[0],rtol=1.e-10,atol=1.e-8)
        for s in snapshots[1:]:np.testing.assert_array_equal(s,snapshots[0])

    def test_small_budget_and_new_decision_are_reproducible(self):
        p,a=fixture();config=MacroConfig('joint',True,13)
        first=solve_macro(p,a,config);second=solve_macro(p,a,config)
        self.assertLessEqual(first[2]['candidate_evaluations'],13)
        np.testing.assert_array_equal(first[0],second[0])
        self.assertEqual(first[2]['continuation_calls'],0)


if __name__=='__main__':unittest.main()

"""Stable full-scan parity, fixed-root edits, fallback and no-CTBS budgets."""
from dataclasses import replace
import unittest
import numpy as np

from .physics import NativePhy, ReferencePhy, DEQUEUE, EXPECTED
from .search import greedy_candidates
from .test_physics import fixture
from .ranked_dpls import RankedDplsConfig, solve_ranked


def independent_full(p, draft):
    phy=NativePhy(p)
    try:
        current=phy.evaluate(draft)
        floor=(90*int(current.values[0,:,DEQUEUE].sum())+99)//100
        score=float(current.values[0,:,EXPECTED]@p.weights)
        count=1
        for number in range(1,257):
            candidates=greedy_candidates(p,current.actions[0])
            result=phy.evaluate(candidates,current)
            count+=len(candidates)
            j=result.values[:,:,EXPECTED]@p.weights
            eligible=result.values[:,:,DEQUEUE].sum(1)>=floor
            best=int(np.argmax(np.where(eligible,j,-np.inf)))
            accepted=eligible[best] and j[best]-score>(1. if number<=2 else 16.)
            if accepted:current=phy.take(result,best);score=float(j[best])
            if number>=3 and not accepted:break
        else:raise RuntimeError('Independent full search failed')
        return current.actions[0],current.values[0],count,number
    finally:phy.close()


class RankedDplsTests(unittest.TestCase):
    def test_full_matches_independent_fixed_root_reference(self):
        for seed in (17,19,113):
            p,draft=fixture(seed=seed)
            expected=independent_full(p,draft)
            action,values,record=solve_ranked(p,draft)
            np.testing.assert_array_equal(action,expected[0])
            np.testing.assert_array_equal(values,expected[1])
            self.assertEqual(record['candidate_evaluations'],expected[2])
            self.assertEqual(record['rounds'],expected[3])
            self.assertTrue(record['final_full_scan'])

    def test_ranked_full_k_has_identical_choices_and_no_duplicate_fallback(self):
        p,draft=fixture()
        expected=solve_ranked(p,draft)
        actual=solve_ranked(p,draft,RankedDplsConfig('heuristic',100000))
        np.testing.assert_array_equal(actual[0],expected[0])
        self.assertEqual(actual[2]['candidate_evaluations'],expected[2]['candidate_evaluations'])

    def test_tiny_shortlist_ends_with_exact_full_stop_and_independent_phy(self):
        p,draft=fixture(seed=39)
        action,values,record=solve_ranked(p,draft,RankedDplsConfig('heuristic',1))
        self.assertTrue(record['converged'])
        self.assertGreaterEqual(values[:,DEQUEUE].sum(),record['floor'])
        self.assertGreaterEqual(record['score'],record['initial_score'])
        self.assertEqual(record['candidate_evaluations'],1+sum(x['scored'] for x in record['history']))
        self.assertTrue(record['history'][-1]['full_scan'])
        np.testing.assert_allclose(values,ReferencePhy(p).evaluate(action)[0],rtol=1.e-10,atol=1.e-8)
        phy=NativePhy(p)
        try:
            base=phy.evaluate(action);next_=phy.evaluate(greedy_candidates(p,action),base)
            j=next_.values[:,:,EXPECTED]@p.weights
            eligible=next_.values[:,:,DEQUEUE].sum(1)>=record['floor']
            self.assertLessEqual(float(np.max(np.where(eligible,j,-np.inf)))-record['score'],16.)
        finally:phy.close()

    def test_first_two_rejections_still_reach_round_three(self):
        p,draft=fixture(cells=1,prgs=1)
        p.queue[:]=1;p.tbs[:,1,1:]=1000
        _,_,record=solve_ranked(p,draft)
        self.assertEqual(record['rounds'],3)
        self.assertTrue(record['converged'])
        self.assertEqual(record['accepted_updates'],0)

    def test_passive_teacher_observer_does_not_change_decisions(self):
        p,draft=fixture()
        observed=[]
        expected=solve_ranked(p,draft,RankedDplsConfig('heuristic',4))
        actual=solve_ranked(p,draft,RankedDplsConfig('heuristic',4),observer=lambda g:observed.append(g['base'].actions[0].copy()))
        np.testing.assert_array_equal(expected[0],actual[0])
        self.assertEqual(expected[2]['candidate_evaluations'],actual[2]['candidate_evaluations'])
        self.assertEqual(len(observed),actual[2]['rounds'])


if __name__=='__main__':unittest.main()

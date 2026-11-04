"""Frozen-grid budgets and selection without development outcomes."""
import unittest

import numpy as np

from .budget_curve import arm_name,parse_arm,arms,choose
from .shortlist_trajectory import Config,solve
from .test_physics import fixture
from .physics import ReferencePhy,DEQUEUE


class Prior:
    def predict(self,x,prior):return prior


class BudgetCurveTests(unittest.TestCase):
    def test_grid_has_exactly18_points_and_one_full_control(self):
        names=arms();self.assertEqual(len(names),19);self.assertEqual(len(set(names)),19)
        for family in ('heuristic','initial','trained'):
            for k in (2,4):
                for budget in (106,160,212):self.assertEqual(parse_arm(arm_name(family,k,budget)),(family,k,budget))

    def test_uneven_score_caps_and_fixed_feature_cap(self):
        p,draft=fixture(seed=113)
        for k in (2,4):
            for budget in (106,160,212):
                a,v,r=solve(p,draft,Config(k,True,56,budget),Prior())
                self.assertLessEqual(r['candidate_evaluations'],budget)
                self.assertLessEqual(r['feature_roots'],56)
                self.assertEqual(r['repeated_exact_candidates'],0)
                self.assertGreaterEqual(v[:,DEQUEUE].sum(),r['floor'])
                np.testing.assert_allclose(v,ReferencePhy(p).evaluate(a)[0],rtol=1.e-10,atol=1.e-8)

    def test_common_full_batches_do_not_depend_on_final_budget(self):
        p,draft=fixture(seed=39)
        for k in (2,4):
            small=solve(p,draft,Config(k,True,56,10),Prior())[2]
            large=solve(p,draft,Config(k,True,56,16),Prior())[2]
            for before,after in zip(small['trace'],large['trace']):
                if len(before['selected'])<k:break
                self.assertEqual(before['root_sha256'],after['root_sha256'])
                self.assertEqual(before['selected'],after['selected'])
                self.assertEqual(before['accepted'],after['accepted'])

    def test_cost_filtered_choice_does_not_promote_a_slow_quality_ceiling(self):
        rows=[]
        for seed in (44,45):
            for arm,ratio,time in [('initial_k2_b106',.97,50.),('trained_k4_b212',1.,90.)]:
                rows.append(dict(seed=seed,arm=arm,ratio=ratio,e2e_ms=time,reference_ms=100.,
                    newly_unserved_ues=0,positive_p5_ratio=1.,positive_weak_ratio=1.,reference_positive_ues=10))
        selected=choose(rows)
        self.assertEqual(selected['quality_best']['arm'],'trained_k4_b212')
        self.assertEqual(selected['within_cost_best']['arm'],'initial_k2_b106')
        self.assertTrue(selected['within_cost_best']['cost_feasible'])

    def test_quality_ceiling_is_not_silently_cost_filtered(self):
        rows=[]
        for seed in (44,45):
            for arm,ratio,time in [('initial_k2_b106',1.,50.),('trained_k4_b212',1.02,90.)]:
                rows.append(dict(seed=seed,arm=arm,ratio=ratio,e2e_ms=time,reference_ms=100.,
                    newly_unserved_ues=0,positive_p5_ratio=1.,positive_weak_ratio=1.,reference_positive_ues=10))
        result=choose(rows)
        self.assertEqual(result['quality_best']['arm'],'trained_k4_b212')
        self.assertEqual(result['within_cost_best']['arm'],'initial_k2_b106')


if __name__=='__main__':unittest.main()

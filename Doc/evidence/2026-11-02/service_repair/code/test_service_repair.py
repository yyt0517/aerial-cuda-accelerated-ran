"""Causal membership, separate diagnostic access and paid terminal behavior."""
import unittest
import numpy as np
from .test_physics import fixture
from .bounded_probes import ProbePhy
from .physics import ReferencePhy,COUNT,DEQUEUE,EXPECTED
from .service_sets import causal_features,decode_probabilities,heuristic_masks,set_metrics,calibration_key
from .service_repair import RepairConfig,solve_service,goal
from .selective_tb import SelectiveConfig,solve_selective


class ConstantModel:
    def predict(self,x):return np.tile([.6,.4],(len(x),1))


class ServiceRepairTests(unittest.TestCase):
    def test_features_only_use_initial_verified_context(self):
        p,a=fixture();phy=ProbePhy(p);self.addCleanup(phy.close);base=phy.evaluate(a)
        before=phy.stats.copy();x,u,eligible=causal_features(p,phy,base)
        self.assertEqual(phy.stats,before);self.assertEqual(x.shape,(p.U,40));self.assertTrue(np.isfinite(x).all())
        independent=ReferencePhy(p).evaluate(a)
        np.testing.assert_allclose(base.values,independent,rtol=1.e-10,atol=1.e-8)

    def test_masks_are_nested_eligible_with_per_cell_fallback(self):
        p,a=fixture();eligible=p.legal.any((0,1));prob=np.zeros((p.U,2));utility=np.arange(p.U,dtype=float)
        masks=decode_probabilities(p,prob,utility,eligible,(.7,.7))
        self.assertTrue((~masks[:,1]|masks[:,0]).all())
        for c in range(p.C):self.assertEqual(masks[p.serving==c,1].sum(),1)

    def test_size_matched_control_preserves_both_cardinalities(self):
        p,a=fixture();eligible=p.legal.any((0,1));utility=np.arange(p.U,dtype=float)
        counts=np.array([[2,1],[1,1],[2,2]])
        masks=heuristic_masks(p,utility,eligible,counts)
        for c in range(p.C):np.testing.assert_array_equal(masks[p.serving==c].sum(0),counts[c])

    def test_calibration_penalizes_protecting_every_ue(self):
        all_rows=[dict(allocation_recall=1.,effective_recall=1.,size_ratio=1.,allocation_precision=.5)]
        selective=[dict(allocation_recall=.95,effective_recall=.95,size_ratio=.65,allocation_precision=.8)]
        self.assertGreater(calibration_key(selective),calibration_key(all_rows))

    def test_no_repair_matches_frozen_incremental_unprotected_solver(self):
        p,a=fixture();reference=solve_selective(p,a,SelectiveConfig('selective',False,max_candidates=101))
        result=solve_service(p,a,RepairConfig('none',False,0,101))
        np.testing.assert_array_equal(result[0],reference[0]);np.testing.assert_array_equal(result[1],reference[1])

    def test_teacher_masks_rejected_by_online_path(self):
        p,a=fixture();masks=np.ones((p.U,2),bool)
        with self.assertRaises(ValueError):solve_service(p,a,RepairConfig('learned'),model=ConstantModel(),diagnostic_masks=masks)
        with self.assertRaises(ValueError):solve_service(p,a,RepairConfig('diagnostic'))

    def test_shared_unlocked_main_prefix_and_complete_final_physics(self):
        p,a=fixture(prgs=5);results=[]
        for source in ('heuristic','learned','matched'):
            r=solve_service(p,a,RepairConfig(source,False,16,100),model=ConstantModel())
            results.append(r);np.testing.assert_allclose(r[1],ReferencePhy(p).evaluate(r[0])[0],rtol=1.e-10,atol=1.e-8)
            self.assertLessEqual(r[2]['candidate_evaluations'],100);self.assertLessEqual(r[2]['repair_phy'],16)
            self.assertLessEqual(r[2]['feature_roots'],56);self.assertGreaterEqual(r[2]['score']+1.e-8,r[2]['quality_fence'])
            self.assertGreaterEqual(r[1][:,DEQUEUE].sum(),r[2]['floor'])
        for r in results[1:]:np.testing.assert_array_equal(r[4][0],results[0][4][0])

    def test_quality_fence_does_not_compound_with_multiple_repairs(self):
        p,a=fixture(prgs=5);r=solve_service(p,a,RepairConfig('heuristic',False,32,130))
        self.assertAlmostEqual(r[2]['quality_fence'],.995*r[2]['main_j'])
        for event in r[2]['repair_events']:
            if event['accepted']:self.assertGreaterEqual(event['candidate_j']+1.e-8,r[2]['quality_fence'])
        self.assertEqual(r[2]['full_search_calls'],0)

    def test_repair_can_change_acceptance_goal_and_is_reproducible(self):
        p,a=fixture(prgs=5);config=RepairConfig('heuristic',False,32,130)
        one=solve_service(p,a,config);two=solve_service(p,a,config)
        np.testing.assert_array_equal(one[0],two[0]);np.testing.assert_array_equal(one[1],two[1])
        self.assertEqual(one[2]['repair_submitted'],two[2]['repair_submitted'])
        before=goal(one[4][1],one[3]);after=goal(one[1],one[3]);self.assertGreaterEqual(after,before)


if __name__=='__main__':unittest.main()

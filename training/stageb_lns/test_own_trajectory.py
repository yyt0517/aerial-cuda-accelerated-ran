"""Geometry parity, paid budgets, passive data and instant service diagnostics."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .test_physics import fixture
from .physics import ReferencePhy,EXPECTED,COUNT,DEQUEUE
from .bounded_probes import Predictor,generate as previous_generate
from .own_trajectory import generate,Geometry,SHALLOW,DEEP,coverage,collect_own,fine_tune
from .own_trajectory_diagnostics import positive_coverage


class Prior:
    def predict(self,x,prior):return prior


class TrajectoryTests(unittest.TestCase):
    def test_shallow_and_deep_reproduce_previous_generator(self):
        for seed in (17,39):
            p,draft=fixture(seed=seed)
            for geometry in (SHALLOW,DEEP):
                old=previous_generate(p,draft,'scalar',Prior(),width=geometry.width,depth=geometry.depth)
                actual=generate(p,draft,geometry,Prior())
                np.testing.assert_array_equal(actual[0],old[0])
                np.testing.assert_array_equal(actual[1],old[1])
                self.assertEqual(actual[2]['candidate_evaluations'],old[2]['candidate_evaluations'])
                self.assertEqual(actual[2]['score'],old[2]['score'])

    def test_observer_mutation_cannot_change_action_or_work(self):
        p,draft=fixture()
        expected=generate(p,draft,SHALLOW,Prior())
        def malicious(payload):
            payload['root'][:]=-1;payload['selected'][:]=0;payload['floor']=0
        actual=generate(p,draft,SHALLOW,Prior(),observer=malicious)
        np.testing.assert_array_equal(actual[0],expected[0])
        self.assertEqual(actual[2]['candidate_evaluations'],expected[2]['candidate_evaluations'])

    def test_resource_truncation_preserves_a_verified_parent_and_floor(self):
        p,draft=fixture()
        for geometry in (Geometry(7,8,max_roots=3,max_candidates=7),Geometry(1,32,max_roots=2,max_candidates=3)):
            a,v,r=generate(p,draft,geometry,Prior())
            self.assertLessEqual(r['feature_roots'],geometry.max_roots)
            self.assertLessEqual(r['candidate_evaluations'],geometry.max_candidates)
            self.assertGreaterEqual(v[:,DEQUEUE].sum(),r['floor'])
            self.assertGreaterEqual(r['score'],r['initial_score'])
            self.assertEqual(r['stop_reason'],'resource_limit')
            np.testing.assert_allclose(v,ReferencePhy(p).evaluate(a)[0],rtol=1.e-10,atol=1.e-8)

    def test_attempted_depth_does_not_claim_successful_edits(self):
        p,draft=fixture(cells=1,prgs=1);p.queue[:]=1
        a,v,r=generate(p,draft,DEEP,Prior())
        self.assertEqual(r['best_path_edits'],0)
        self.assertEqual(r['unique_feature_roots'],1)
        self.assertEqual(r['feature_roots'],32)
        self.assertEqual(r['stagnant_roots'],32)
        self.assertEqual(r['candidate_evaluations'],33)
        np.testing.assert_array_equal(a,draft)

    def test_coverage_exposes_newly_unserved_and_tail_service(self):
        reference=np.zeros((5,8));reference[:,COUNT]=1;reference[:,EXPECTED]=[10,20,30,40,50]
        actual=reference.copy();actual[0,COUNT]=0;actual[0,EXPECTED]=0
        r=coverage(actual,reference)
        self.assertEqual(r['newly_unserved_ues'],1)
        self.assertEqual(r['served_ues'],4)
        self.assertLess(r['expected_p5_ratio'],1)
        self.assertEqual(r['weak_service_ratio'],0)
        self.assertEqual(r['ue_service_drops_gt2pct'],1)

    def test_teacher_labels_only_actual_deduplicated_student_roots(self):
        p,draft=fixture(cells=1,prgs=1);p.queue[:]=1
        row=dict(seed=41,family='fixture',tti=4001,obs={})
        with tempfile.TemporaryDirectory() as temp:
            with patch('training.stageb_lns.own_trajectory.problem_from_raw_row',return_value=p),\
                 patch('training.stageb_lns.own_trajectory.raw_pfq_draft',return_value=draft):
                groups,visits=collect_own([row],Prior(),DEEP,Path(temp),1)
            self.assertEqual(len(groups),1)
            self.assertEqual(len(visits),1)
            self.assertTrue(visits[0]['actual_visited'])
            np.testing.assert_array_equal(groups[0]['root'],draft)
            self.assertEqual(groups[0]['round'],1)

    def test_invalid_geometry_is_rejected(self):
        for args in [(0,8),(1,0),(1,1,0,10),(1,1,3,0)]:
            with self.assertRaises(ValueError):Geometry(*args)

    def test_positive_service_diagnostic_exposes_zero_inclusive_p5_degeneracy(self):
        reference=np.zeros((5,8));reference[:,EXPECTED]=[0,0,10,20,30]
        actual=reference.copy();actual[:,EXPECTED]=[1,5,5,20,30]
        r=positive_coverage(actual,reference)
        self.assertTrue(r['all_ue_p5_degenerate'])
        self.assertEqual(r['reference_positive_ues'],3)
        self.assertLess(r['positive_p5_ratio'],1.)
        self.assertEqual(r['positive_weak_ratio'],.5)

    def test_finetuning_keeps_original_weights_on_validation_ties(self):
        rng=np.random.default_rng(31)
        data=dict(mean=np.zeros(48,np.float32),scale=np.ones(48,np.float32))
        for k,shape in [('w1',(32,48)),('b1',(32,)),('w2',(16,32)),('b2',(16,)),('w3',(1,16)),('b3',(1,))]:
            data[k]=rng.normal(0,.01,shape).astype(np.float32)
        group=dict(x48=rng.normal(size=(20,48)).astype(np.float32),prior=np.zeros(20),
            gain=np.linspace(-100,100,20),eligible=np.ones(20,bool))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);np.savez(path/'initial.npz',**data);initial=Predictor(path/'initial.npz')
            selected,record=fine_tune([group],initial,path,1,lambda model:((0,),dict(mean_ratio=.5)),epochs=1)
            self.assertEqual(record['epoch'],0)
            for k in data:np.testing.assert_array_equal(getattr(selected,k),getattr(initial,k))


if __name__=='__main__':unittest.main()

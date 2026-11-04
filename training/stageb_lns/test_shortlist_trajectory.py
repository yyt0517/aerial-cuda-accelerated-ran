"""Real verification budgets, root-local failed masks and own-state labels."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .test_physics import fixture
from .physics import ReferencePhy,EXPECTED,DEQUEUE
from .bounded_probes import ProbePhy
from .shortlist_trajectory import Config,solve,collect_own,state_key
from .shortlist_campaign import selection_key,metric


class Prior:
    def predict(self,x,prior):return prior


class FixedOrder:
    def predict(self,x,prior):return -np.arange(len(x),dtype=float)


class ShortlistTests(unittest.TestCase):
    def test_dedup_escapes_a_rejected_top_candidate(self):
        p,draft=fixture(cells=1,prgs=2)
        p.caps[:]=1;p.weights[:]=[1.,2.];p.queue[:]=1000
        p.h[0]*=100;p.h[1,0]*=.00001;p.h[1,1]*=100
        escaped=solve(p,draft,Config(1,True,max_candidates=3),FixedOrder())
        repeated=solve(p,draft,Config(1,False,max_candidates=3),FixedOrder())
        self.assertEqual(escaped[2]['accepted_edits'],1)
        self.assertEqual(repeated[2]['accepted_edits'],0)
        self.assertGreater(escaped[2]['score'],repeated[2]['score'])
        self.assertEqual(escaped[2]['repeated_exact_candidates'],0)
        self.assertEqual(repeated[2]['repeated_exact_candidates'],1)

    def test_top2_and_top4_never_rescore_a_root_candidate_or_hide_full_search(self):
        p,draft=fixture(seed=39)
        original=ProbePhy.evaluate
        for k in (2,4):
            batches=[]
            def checked(phy,actions,base=None):
                n=1 if np.asarray(actions).ndim==2 else len(actions)
                batches.append(n)
                self.assertLessEqual(n,k)
                return original(phy,actions,base)
            with patch.object(ProbePhy,'evaluate',checked):
                a,v,r=solve(p,draft,Config(k),Prior())
            self.assertEqual(r['candidate_evaluations'],sum(batches))
            self.assertEqual(r['candidate_evaluations'],1+sum(len(x['selected']) for x in r['trace']))
            self.assertEqual(r['repeated_exact_candidates'],0)
            self.assertEqual(r['full_search_calls'],0)
            seen={}
            for visit in r['trace']:
                used=seen.setdefault(visit['root_sha256'],set())
                self.assertFalse(used.intersection(visit['selected']));used.update(visit['selected'])
            np.testing.assert_allclose(v,ReferencePhy(p).evaluate(a)[0],rtol=1.e-10,atol=1.e-8)

    def test_new_roots_and_new_decisions_begin_without_old_failed_indices(self):
        p,draft=fixture(cells=1,prgs=2)
        p.caps[:]=1;p.weights[:]=[1.,2.];p.queue[:]=1000
        p.h[0]*=100;p.h[1,0]*=.00001;p.h[1,1]*=100
        for _ in range(2):
            a,v,r=solve(p,draft,Config(1,True,max_candidates=6),FixedOrder())
            seen=set()
            for visit in r['trace']:
                if visit['root_sha256'] not in seen:
                    self.assertEqual(visit['failed_before'],[]);seen.add(visit['root_sha256'])
            self.assertGreaterEqual(len(seen),2)
        p.h*=1.1
        self.assertEqual(solve(p,draft,Config(2),Prior())[2]['trace'][0]['failed_before'],[])

    def test_observer_is_passive(self):
        p,draft=fixture()
        expected=solve(p,draft,Config(2),Prior())
        def mutate(v):v['root'][:]=-1;v['failed_before'][:]=0;v['selected'][:]=0
        actual=solve(p,draft,Config(2),Prior(),observer=mutate)
        np.testing.assert_array_equal(actual[0],expected[0])
        self.assertEqual(actual[2]['candidate_evaluations'],expected[2]['candidate_evaluations'])

    def test_partial_last_shortlist_and_small_budgets_are_respected(self):
        p,draft=fixture(seed=113)
        for config in (Config(4,max_candidates=4),Config(2,max_feature_roots=1),Config(4,max_candidates=1)):
            a,v,r=solve(p,draft,config,Prior())
            self.assertLessEqual(r['candidate_evaluations'],config.max_candidates)
            self.assertLessEqual(r['feature_roots'],config.max_feature_roots)
            self.assertGreaterEqual(v[:,DEQUEUE].sum(),r['floor'])
            self.assertGreaterEqual(r['score'],r['initial_score'])

    def test_own_labels_preserve_actual_failed_mask_and_availability(self):
        p,draft=fixture(cells=1,prgs=3)
        p.queue[:]=1;p.h*=100;p.caps[:]=1
        row=dict(seed=41,family='fixture',tti=4001,obs={})
        with tempfile.TemporaryDirectory() as temp:
            with patch('training.stageb_lns.shortlist_trajectory.problem_from_raw_row',return_value=p),\
                 patch('training.stageb_lns.shortlist_trajectory.raw_pfq_draft',return_value=draft):
                groups,visits=collect_own([row],Prior(),Config(1),Path(temp))
            self.assertTrue(any(len(g['failed_before']) for g in groups))
            self.assertEqual(len({state_key(g['state_id'],g['root'],g['failed_before']) for g in groups}),len(groups))
            for g in groups:self.assertFalse(g['eligible'][g['failed_before']].any())
            self.assertTrue(all(v['actual_visited'] for v in visits))

    def test_invalid_configuration_is_rejected(self):
        for k in (0,3,8):
            with self.assertRaises(ValueError):Config(k)
        with self.assertRaises(ValueError):Config(2,max_candidates=0)

    def test_selection_cost_is_ratio_of_seed_equal_complete_means(self):
        records=[]
        for seed,actual,reference in [(44,2.,1.),(45,98.,199.)]:
            records.append(dict(seed=seed,ratio=1.,e2e_ms=actual,reference_ms=reference,
                time_ratio=actual/reference,newly_unserved_ues=0,positive_p5_ratio=1.,
                positive_weak_ratio=1.,reference_positive_ues=10))
        self.assertEqual(selection_key(records)[0],1)
        self.assertEqual(metric(records)['mean_time_ratio'],.5)


if __name__=='__main__':unittest.main()

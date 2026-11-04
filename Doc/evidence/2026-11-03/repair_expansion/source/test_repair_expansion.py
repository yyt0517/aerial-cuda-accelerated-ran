"""Repair-pair coverage, cohort weighting and variable archive sizes."""
from unittest import TestCase,main
import numpy as np
from .repair_pair_train import focused_pairs,cohort_weights,auxiliary_weights,padded,fit
from .ue_pair_train import predict


def example(good=(False,True,False,True),n=4):
    r=np.array([[.2,.9],[.9,.8],[.1,.8],[.7,.7]])[:n]
    d=dict(j=np.array([100.,99.,98.5,97.])[:n],good=np.array(good[:n]),group=np.ones(2,bool),
        effective=np.array([1.,1.]),target=np.array([.9,.9]),r=r,losses=np.repeat((1-r)[...,None],7,axis=-1),context=np.zeros((2,20)))
    d['pairs']=focused_pairs(d);return d


class ExpansionTests(TestCase):
    def test_all_qualified_repairs_compare_to_failed_incumbent(self):
        d=example();p=focused_pairs(d)
        self.assertEqual(set(map(tuple,p)),{(1.,0.,1.),(1.,2.,1.)})
        self.assertNotIn(3,p[:,:2])

    def test_good_incumbent_has_bad_comparisons_and_good_ties(self):
        p=focused_pairs(example((True,True,False,False)))
        self.assertEqual(set(map(tuple,p)),{(0.,2.,1.),(0.,1.,.5)})

    def test_unrepairable_state_has_no_false_positive_pair(self):
        self.assertEqual(focused_pairs(example((False,False,False,False))).shape,(0,3))

    def test_cohort_weights_balance_states_and_exclude_unrepairable(self):
        ds=[example(),example((True,True,False,False)),example((True,False,False,False)),example((False,False,False,False))]
        np.testing.assert_array_equal(cohort_weights(ds),[.5,.25,.25,0.])

    def test_underprotected_reference_ue_is_not_diluted(self):
        ds=[example()];w=auxiliary_weights(ds)
        np.testing.assert_allclose(w,[[.5,.5]])
        self.assertAlmostEqual(w.sum(),1.)

    def test_variable_length_archives_fit_and_export_finite_scores(self):
        ds=[example(),example(n=3)];p=padded(ds,'r')
        self.assertEqual(p.shape,(2,4,2));np.testing.assert_array_equal(p[1,3],0)
        a,history=fit(ds,7,True)
        self.assertEqual(len(predict(a,ds[0])),4);self.assertTrue(np.isfinite(predict(a,ds[1])).all())
        self.assertEqual(history[-1]['epoch'],240)


if __name__=='__main__':main()

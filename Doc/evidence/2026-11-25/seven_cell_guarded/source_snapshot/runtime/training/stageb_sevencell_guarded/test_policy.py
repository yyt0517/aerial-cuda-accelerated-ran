"""Guard boundaries, priority semantics and golden DPLS endpoint preservation."""
import unittest
from pathlib import Path
import numpy as np
import torch
from training.stageb_lns.no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from training.stageb_sevencell_multistart.audit import audit_branch
from .policy import select,make_weights,choose,BETA

ROOT=Path(__file__).resolve().parents[2]/'deliverables/seven_cell_boundary08_20261105_v1'

class PolicyTests(unittest.TestCase):
    def test_guard_rejects_high_score_and_keeps_boundary_and_ties(self):
        b={'dpls':dict(expected_bytes=100.,score_q=100),
           'new_cold':dict(expected_bytes=97.999,score_q=300),
           'new_from_dpls':dict(expected_bytes=98.,score_q=200)}
        winner,floor,eligible=choose(b)
        self.assertEqual(winner,'new_from_dpls');self.assertEqual(floor,98.);self.assertFalse(eligible['new_cold'])
        b['new_from_dpls']['expected_bytes']=97.999
        self.assertEqual(choose(b)[0],'dpls')
        b['new_cold']['expected_bytes']=98.;b['new_cold']['score_q']=100
        self.assertEqual(choose(b)[0],'new_cold')

    def test_young_queues_keep_base_and_weights_are_bounded(self):
        b=np.array([1.,4.]);b/=b.mean();w=make_weights(b,np.array([0.,1.]))
        self.assertAlmostEqual(w[1]/w[0],4.4)
        np.testing.assert_allclose(make_weights(b,np.zeros(2)),b,atol=1e-15,rtol=1e-15)

    def test_real_dpls_endpoint_and_new_search_full_oracle(self):
        for seed in (41,42,43):
            row=torch.load(ROOT/f'campaign/deficit_s{seed}/states/tti008001.pt',map_location='cpu',weights_only=False)
            p=problem_from_raw_row(row);base=p.weights.copy();w=make_weights(base,row['hol_urgency']);draft=raw_pfq_draft(p,row['obs'])
            a,v,r,b=select(p,draft,w)
            np.testing.assert_array_equal(b['dpls']['action'],row['selected_action'])
            np.testing.assert_array_equal(b['dpls']['values'],row['selected_values'])
            np.testing.assert_array_equal(p.weights,base)
            self.assertGreaterEqual(r['selected_expected_bytes'],BETA*b['dpls']['expected_bytes'])
            self.assertGreaterEqual(r['score_q'],b['dpls']['score_q'])
            p.weights=w
            for name,initial in [('new_cold',None),('new_from_dpls',b['dpls']['action'])]:
                x,y,_=audit_branch(p,draft,initial)
                np.testing.assert_array_equal(b[name]['action'],x);np.testing.assert_array_equal(b[name]['values'],y)

if __name__=='__main__':unittest.main()

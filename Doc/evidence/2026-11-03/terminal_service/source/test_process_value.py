"""Value screening must not read counterfactual PHY or reuse missing resources."""
import unittest
from unittest.mock import patch
import numpy as np

from .test_physics import fixture
from .physics import NativePhy, EXPECTED
from .search import ProposalGraph
from .deadline import packet_objective
from .process_value import cheap_completion_features, risk_from_expected, conformal_upper_quantile


class ProcessValueTests(unittest.TestCase):
    def test_cheap_inputs_need_no_new_phy_and_track_other_cell_victims(self):
        p, action = fixture()
        p.deadline = np.zeros((p.U, 8));p.deadline[:, 1] = 1
        p.sinr_hint = np.ones((p.U, p.P, 4))*10
        phy = NativePhy(p);self.addCleanup(phy.close)
        base = phy.evaluate(action);graph = ProposalGraph(phy)
        changed = action.copy();changed[0, 0] = -1
        before = phy.stats
        with patch.object(phy, 'evaluate', side_effect=AssertionError('hidden exact PHY')):
            x, score = cheap_completion_features(graph, base, np.stack((action, changed)))
        self.assertEqual(phy.stats, before)
        self.assertEqual(x.shape, (2, 96))
        self.assertEqual(score[0], 0)
        self.assertEqual(x[0, 6], 0)
        self.assertGreater(x[1, 6], x[1, 7])  # victims beyond resource-changed UE
        rev, rev_score = cheap_completion_features(graph, base, np.stack((changed, action)))
        np.testing.assert_array_equal(rev, x[::-1])
        np.testing.assert_array_equal(rev_score, score[::-1])

    def test_expected_byte_risk_formula_matches_verified_objective(self):
        p, action = fixture()
        p.deadline = np.random.default_rng(123).uniform(0, 1, (p.U, 8))
        phy = NativePhy(p);self.addCleanup(phy.close)
        blank = np.full_like(action, -1)
        values = phy.evaluate(np.stack((action, blank))).values
        np.testing.assert_allclose(risk_from_expected(p, values[:, :, EXPECTED]),
                                   packet_objective(p)(values), rtol=0, atol=1.e-10)

    def test_two_percent_seed_cluster_bound_cannot_use_one_seed_as_many_candidates(self):
        self.assertTrue(np.isinf(conformal_upper_quantile([0.2], .02)))
        self.assertTrue(np.isinf(conformal_upper_quantile(np.arange(48), .02)))
        self.assertEqual(conformal_upper_quantile(np.arange(49), .02), 48)


if __name__ == '__main__':
    unittest.main()

"""Exact full/cached equality, not just objective tolerance or final actions."""
import unittest
import numpy as np
from .cached_dpls import CanonicalDplsConfig, solve_canonical
from .test_physics import fixture
from .physics import ReferencePhy


class CachedDplsTests(unittest.TestCase):
    def test_disabled_diagnostics_has_identical_actions_phy_and_work(self):
        p,draft=fixture(seed=39)
        for mode in ('full','cached'):
            before=solve_canonical(p,draft,CanonicalDplsConfig(mode))
            after=solve_canonical(p,draft,CanonicalDplsConfig(mode,diagnostics=False))
            np.testing.assert_array_equal(before[0],after[0])
            np.testing.assert_array_equal(before[1],after[1])
            self.assertEqual(before[2]['candidate_evaluations'],after[2]['candidate_evaluations'])
            self.assertEqual(after[2]['history'],[])
            self.assertNotIn('score_ms',after[2])

    def test_every_candidate_phy_every_round_matches_fresh_full(self):
        for seed in (17,19,39,113):
            p,draft = fixture(seed=seed)
            observed = []
            def capture(r):
                observed.append({k:r[k].copy() for k in ('values','scores','eligible','keys')})
                observed[-1].update(chosen=r['chosen'],root=r['root'].actions.copy())
            full = solve_canonical(p,draft,observer=capture)
            checked = []
            def audit(r):
                expected = observed[r['iteration']-1]
                for k in ('values','scores','eligible','keys'):
                    np.testing.assert_array_equal(r[k],expected[k])
                np.testing.assert_array_equal(r['root'].actions,expected['root'])
                self.assertEqual(r['chosen'],expected['chosen'])
                checked.append(r['iteration'])
            cached = solve_canonical(p,draft,CanonicalDplsConfig('cached'),observer=audit)
            self.assertEqual(len(checked),len(observed))
            np.testing.assert_array_equal(full[0],cached[0])
            np.testing.assert_array_equal(full[1],cached[1])
            self.assertEqual(full[2]['score'],cached[2]['score'])
            self.assertTrue(cached[2]['converged'])
            self.assertLess(cached[2]['candidate_evaluations'],full[2]['candidate_evaluations'])
            np.testing.assert_allclose(cached[1],ReferencePhy(p).evaluate(cached[0])[0],atol=1.e-8)

    def test_warm_bootstrap_is_current_phy_and_common_budget_floor(self):
        p,draft = fixture()
        previous = solve_canonical(p,draft)[0]
        full = solve_canonical(p,draft,warm_start=previous)
        cached = solve_canonical(p,draft,CanonicalDplsConfig('cached'),warm_start=previous)
        np.testing.assert_array_equal(full[0],cached[0])
        np.testing.assert_array_equal(full[1],cached[1])
        self.assertEqual(full[2]['floor'],cached[2]['floor'])
        self.assertEqual(cached[2]['candidate_evaluations'],2+cached[2]['scored_neighborhood_candidates'])

    def test_empty_neighborhood_retains_verified_zero_floor(self):
        p,draft = fixture(cells=1,prgs=1)
        p.queue[:] = 0
        p.legal[:] = False
        draft[:] = -1
        for mode in ('full','cached'):
            action,_,record = solve_canonical(p,draft,CanonicalDplsConfig(mode))
            np.testing.assert_array_equal(action,draft)
            self.assertTrue(record['converged'])
            self.assertEqual(record['rounds'],3)
            self.assertEqual(record['candidate_evaluations'],1)


if __name__ == '__main__':
    unittest.main()

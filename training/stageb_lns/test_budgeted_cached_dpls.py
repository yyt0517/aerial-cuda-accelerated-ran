"""Current-valid caches, certified stops, physical joint moves and shared floors."""
import unittest
import numpy as np
from .test_physics import fixture
from .physics import ReferencePhy,EXPECTED,DEQUEUE
from .cached_dpls import solve_canonical,CanonicalDplsConfig
from .budgeted_cached_dpls import solve_budgeted,BudgetedCacheConfig


class BudgetedCacheTests(unittest.TestCase):
    def test_full_first_scan_and_all_online_evaluations_are_charged(self):
        p,draft=fixture(seed=39)
        for cfg in (BudgetedCacheConfig('batch'),BudgetedCacheConfig('dirty',dirty_m=1)):
            _,_,r=solve_budgeted(p,draft,cfg)
            self.assertEqual(r['history'][0]['scored'],r['history'][0]['candidates'])
            self.assertEqual(r['first_scan_candidates'],r['history'][0]['candidates'])
            self.assertEqual(r['candidate_evaluations'],1+sum(h['scored'] for h in r['history'])+r['joint_evaluations'])
            self.assertEqual(r['joint_evaluations'],sum(h['moves']>1 for h in r['history']))

    def test_exact_control_preserves_canonical_trajectory_and_work(self):
        for seed in (17,39,113):
            p,draft=fixture(seed=seed)
            expected=solve_canonical(p,draft,CanonicalDplsConfig('cached'))
            actual=solve_budgeted(p,draft)
            np.testing.assert_array_equal(actual[0],expected[0])
            np.testing.assert_array_equal(actual[1],expected[1])
            self.assertEqual(actual[2]['candidate_evaluations'],expected[2]['candidate_evaluations'])
            self.assertEqual([h['selected_keys'][0] if h['selected_keys'] else None for h in actual[2]['history']],
                             [h['chosen_key'] for h in expected[2]['history']])

    def test_batch_floor_and_terminal_phy(self):
        for seed in (17,19,39):
            p,draft=fixture(seed=seed)
            checked=[]
            def inspect(info):
                selected=info['selected']
                footprints=info['footprints'][selected]
                used=0
                for mask in footprints:
                    self.assertEqual(used&int(mask),0)
                    used|=int(mask)
                if len(selected):
                    root_d=info['root'].values[0,:,DEQUEUE].sum()
                    combined=root_d+sum(info['values'][i,:,DEQUEUE].sum()-root_d for i in selected)
                    self.assertGreaterEqual(combined,info['floor'])
                checked.append(len(selected))
            action,values,r=solve_budgeted(p,draft,BudgetedCacheConfig('batch'),observer=inspect)
            self.assertLessEqual(max(checked),4)
            self.assertTrue(r['converged'])
            np.testing.assert_allclose(values,ReferencePhy(p).evaluate(action)[0],atol=1.e-8)
            self.assertGreaterEqual(values[:,DEQUEUE].sum(),r['floor'])

    def test_dirty_invalid_entries_persist_and_stop_has_complete_verification(self):
        for seed in (17,39):
            p,draft=fixture(seed=seed)
            roots=[]
            action,values,r=solve_budgeted(p,draft,BudgetedCacheConfig('dirty',dirty_m=1),observer=lambda i:roots.append(
                (i['root'].actions[0].copy(),i['valid'].copy(),i['gain'].copy(),i['chosen'])))
            self.assertTrue(r['converged'])
            self.assertTrue(roots[-1][1].all())
            self.assertGreater(r['fallback_candidates'],0)
            for root,valid,gain,_ in roots:
                self.assertTrue(np.isnan(gain[~valid]).all())
            np.testing.assert_allclose(values,ReferencePhy(p).evaluate(action)[0],atol=1.e-8)
            self.assertGreaterEqual(values[:,DEQUEUE].sum(),r['floor'])
            # Returned incumbent is actually locally stopped, not shortlist stopped.
            from .search import greedy_candidates
            all_values=ReferencePhy(p).evaluate(greedy_candidates(p,action))
            gain=all_values[:,:,EXPECTED]@p.weights-values[:,EXPECTED]@p.weights
            eligible=all_values[:,:,DEQUEUE].sum(1)>=r['floor']
            self.assertLessEqual(float(np.max(gain[eligible],initial=0.)),16.+1.e-8)

    def test_diagnostics_off_does_not_change_budget_behavior(self):
        p,draft=fixture()
        for mode in ('batch','dirty'):
            a=solve_budgeted(p,draft,BudgetedCacheConfig(mode,dirty_m=2))
            b=solve_budgeted(p,draft,BudgetedCacheConfig(mode,dirty_m=2,diagnostics=False))
            np.testing.assert_array_equal(a[0],b[0])
            np.testing.assert_array_equal(a[1],b[1])
            self.assertEqual(a[2]['candidate_evaluations'],b[2]['candidate_evaluations'])
            self.assertEqual(b[2]['history'],[])


if __name__=='__main__':unittest.main()

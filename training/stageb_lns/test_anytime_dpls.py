"""Budget accounting, causal initialization and lossless passive trajectories."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np

from .physics import NativePhy, ReferencePhy, DEQUEUE, EXPECTED
from .test_physics import fixture
from .ranked_dpls import RankedDplsConfig, solve_ranked, repair_warm_start
from .dpls_trace import DplsTrace, solution_overlap


class AnytimeDplsTests(unittest.TestCase):
    def test_hard_candidate_budgets_include_both_initializations(self):
        p, draft = fixture(seed=39)
        previous = solve_ranked(p, draft)[0]
        for warm in (None, previous):
            for cap in (1, 2, 7, 106, 160, 212):
                action, values, record = solve_ranked(p, draft,
                    RankedDplsConfig(max_candidates=cap, profile=True), warm_start=warm)
                initial_evaluations = 1+int(warm is not None and cap > 1)
                self.assertEqual(record['candidate_evaluations'], initial_evaluations+
                                 sum(r['scored'] for r in record['history']))
                self.assertLessEqual(record['candidate_evaluations'], cap)
                self.assertGreaterEqual(values[:, DEQUEUE].sum(), record['floor'])
                self.assertGreaterEqual(record['score'], record['initial_score'])
                np.testing.assert_allclose(values, ReferencePhy(p).evaluate(action)[0], atol=1.e-8)
                if cap <= 7:
                    self.assertFalse(record['converged'])
                for key in ('native_kernel_ms', 'validation_ms', 'output_copy_ms'):
                    self.assertGreaterEqual(record[key], 0.)

    def test_one_and_two_round_controls_are_prefixes_of_full(self):
        p, draft = fixture()
        for rounds in (1, 2, 3, 5):
            roots = []
            solve_ranked(p, draft, observer=lambda r: roots.append(r['base'].actions[0].copy()))
            action, _, record = solve_ranked(p, draft, RankedDplsConfig(max_rounds=rounds))
            self.assertLessEqual(record['rounds'], rounds)
            if rounds < len(roots):
                np.testing.assert_array_equal(action, roots[rounds])

    def test_current_legality_and_current_phy_replace_stale_state(self):
        p, draft = fixture()
        previous = solve_ranked(p, draft)[0]
        p = replace(p, queue=p.queue.copy(), legal=p.legal.copy(), h=p.h*.3)
        p.queue[0] = 0
        p.legal[:, :, 0] = False
        p.legal[1, 0] = False
        repaired = repair_warm_start(p, previous)
        self.assertFalse(np.any(repaired == 0))
        self.assertEqual(repaired[1, 0], -1)
        p.actions(repaired)
        action, values, record = solve_ranked(p, draft if not np.any(draft == 0) else
            repair_warm_start(p, draft), RankedDplsConfig(max_candidates=2), warm_start=previous)
        np.testing.assert_allclose(values, ReferencePhy(p).evaluate(action)[0], atol=1.e-8)
        phy = NativePhy(p)
        try:
            self.assertEqual(record['warm_score'], float(phy.evaluate(repaired).values[0, :, EXPECTED]@p.weights))
        finally:
            phy.close()

    def test_passive_trace_reconstructs_every_candidate_and_delta(self):
        p, draft = fixture()
        expected = solve_ranked(p, draft)
        trace = DplsTrace()
        actual = solve_ranked(p, draft, observer=trace)
        np.testing.assert_array_equal(expected[0], actual[0])
        self.assertEqual(expected[2]['candidate_evaluations'], actual[2]['candidate_evaluations'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.npz'
            trace.write(path, action=actual[0], values=actual[1])
            with np.load(path) as data:
                phy = NativePhy(p)
                try:
                    for i, root in enumerate(data['roots']):
                        start, end = data['offsets'][i:i+2]
                        actions = np.broadcast_to(root, (end-start, *root.shape)).copy()
                        for j, prg in enumerate(data['prg'][start:end]):
                            actions[j, :, prg] = data['assignment'][start+j]
                        score = phy.evaluate(actions).values[:, :, EXPECTED]@p.weights
                        np.testing.assert_allclose(score-data['root_score'][i], data['delta'][start:end], atol=1.e-8)
                        take = data['chosen_index'][i]
                        after = data['roots'][i+1] if i+1 < len(data['roots']) else data['action']
                        np.testing.assert_array_equal(after, actions[take] if take >= 0 else root)
                finally:
                    phy.close()

    def test_overlap_does_not_hide_changes_behind_blank_agreement(self):
        a = np.array([[0, -1, -1, -1]])
        b = np.array([[1, -1, -1, -1]])
        result = solution_overlap(a, b)
        self.assertEqual(result['slot_agreement'], .75)
        self.assertEqual(result['active_assignment_jaccard'], 0.)
        self.assertEqual(result['activity_agreement'], 1.)


if __name__ == '__main__':
    unittest.main()

"""Operator identity, dependency closure, common-floor packing and scalar proxy."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .analyze_dpls_moves import operators, greedy_pack, pack_bounds, ScalarSir, analyze_trace
from .test_physics import fixture
from .search import greedy_candidates


class MoveAnalysisTests(unittest.TestCase):
    def test_support_distinguishes_swap_and_neighbor_activity_change(self):
        root = np.array([[0, 0], [2, 2]])
        _, affected, _ = operators(root, np.array([0, 0]), np.array([[1, 2], [-1, 2]]), 4)
        self.assertEqual(int(affected[0]), (1 << 0) | (1 << 1))
        self.assertEqual(int(affected[1]), (1 << 0) | (1 << 2))

    def test_operator_keys_survive_unrelated_root_and_row_order_changes(self):
        root = np.array([[0, 0], [2, 2]])
        before = operators(root, np.array([0, 1]), np.array([[1, 2], [0, 3]]), 4)[0]
        after = root.copy()
        after[0, 0] = 1
        later = operators(after, np.array([1]), np.array([[0, 3]]), 4)[0]
        self.assertEqual(before[1], later[0])
        self.assertNotEqual(before[0], later[0])

    def test_structurally_independent_moves_can_violate_combined_floor(self):
        footprints = np.array([1, 2], np.uint64)
        gain, pool = np.array([10., 9.]), np.arange(2)
        self.assertEqual(pack_bounds(footprints, gain, pool, 2)['lower'], 2)
        actual = pack_bounds(footprints, gain, pool, 2, delta_d=np.array([-7., -7.]), slack=10.)
        self.assertTrue(actual['optimal'])
        self.assertEqual(actual['lower'], 1)
        self.assertEqual(greedy_pack(footprints, gain, pool, 0, np.array([-7., -7.]), 10.), [0])

    def test_delta_matching_removes_global_score_shift_without_phy(self):
        root = np.array([[0, 0], [2, 2]])
        after = root.copy()
        after[0, 0] = 1
        final = after.copy()
        final[1, 1] = 3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.npz'
            np.savez(path, values=np.zeros((4, 8)), roots=np.array([root, after]),
                     action=final, offsets=np.array([0, 2, 3]),
                     prg=np.array([0, 1, 1]), assignment=np.array([[1, 2], [0, 3], [0, 3]]),
                     scores=np.array([110., 105., 115.]), delta=np.array([10., 5., 5.]),
                     eligible=np.ones(3, bool), evaluated=np.ones(3, bool),
                     chosen_index=np.array([0, 0]), threshold=np.ones(2), floor=np.zeros(2))
            records, transitions = analyze_trace(path)
        self.assertEqual(transitions[0]['delta_changed'], 0)
        self.assertEqual(transitions[0]['common'], 1)
        self.assertEqual(records[1]['lazy_refresh_candidates'], 0)
        self.assertTrue(all(r['cached_chosen_matches'] for r in records))

    def test_scalar_proxy_and_logical_tb_match_independent_full_actions(self):
        p, root = fixture(cells=2, prgs=3)
        actions = greedy_candidates(p, root)
        changed = np.any(actions != root, axis=1)
        prg = np.argmax(changed, axis=1)
        assignment = actions[np.arange(len(actions))[:, None], np.arange(p.C), prg[:, None]]
        actual = ScalarSir(p).score(root, prg, assignment)
        received = np.sum(np.abs(p.h)**2, axis=(-1, -2))
        def reference(action):
            utility, dequeue = 0., 0.
            for u in range(p.U):
                selected = np.flatnonzero(action[p.serving[u]] == u)
                if not len(selected):
                    continue
                capacity = p.tbs[u, len(selected), p.caps[u, selected].min()]
                d = min(p.queue[u], capacity)
                sir = []
                for q in selected:
                    interference = sum(received[u, q, c] for c in range(p.C)
                                       if c != p.serving[u] and action[c, q] >= 0)
                    sir.append(received[u, q, p.serving[u]]/(4*p.noise+interference))
                utility += p.weights[u]*d*np.log1p(np.mean(sir))
                dequeue += d
            return utility, dequeue
        initial = reference(root)
        expected = np.array([reference(action) for action in actions])-initial
        np.testing.assert_allclose(actual['tb'], expected[:, 0], atol=1.e-8)
        np.testing.assert_array_equal(actual['delta_d'], expected[:, 1])

    def test_roundoff_in_reused_gain_can_break_original_index_tie(self):
        root = np.array([[0, 0], [3, 3]])
        after = root.copy()
        after[0, 0] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.npz'
            np.savez(path, values=np.zeros((6, 8)), roots=np.array([root, after]),
                     offsets=np.array([0, 2, 4]), prg=np.array([0, 1, 0, 1]),
                     assignment=np.array([[1, 3], [0, 4], [2, 3], [0, 4]]),
                     delta=np.array([10., 5.+2.e-11, 5., 5.]),
                     eligible=np.ones(4, bool), evaluated=np.ones(4, bool),
                     chosen_index=np.array([0, 0]), threshold=np.ones(2), floor=np.zeros(2))
            records, _ = analyze_trace(path)
        self.assertFalse(records[1]['cached_chosen_matches'])
        self.assertEqual(records[1]['cached_chosen_index'], 1)
        self.assertTrue(records[1]['guarded_chosen_matches'])


if __name__ == '__main__':
    unittest.main()

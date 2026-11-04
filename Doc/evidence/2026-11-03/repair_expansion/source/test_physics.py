"""Behavioral tests for TB closure, weak interference, invariance and budgets."""
import unittest
from unittest.mock import patch
import numpy as np

from .physics import Problem, NativePhy, ReferencePhy, EXPECTED, LAYER, DEQUEUE, native_bler
from .search import SearchConfig, solve


def fixture(cells=3, prgs=5, seed=17):
    rng = np.random.default_rng(seed)
    users = cells*2
    h = (rng.normal(size=(users, prgs, cells, 4, 4)) + 1j*rng.normal(size=(users, prgs, cells, 4, 4)))/3
    serving = np.repeat(np.arange(cells), 2)
    h[np.arange(users), :, serving] *= 2
    w = np.tile(np.eye(4, dtype=complex), (users, prgs, 1, 1))
    queue = rng.integers(500, 3000, users).astype(float)
    tbs = np.tile(np.arange(prgs+1)[None, :, None]*np.arange(5)[None, None, :]*130, (users, 1, 1))
    legal = np.broadcast_to(np.arange(cells)[:, None, None] == serving[None, None, :], (cells, prgs, users)).copy()
    problem = Problem(h, w, 1., queue, serving, np.full(users, 3),
                      rng.integers(1, 5, (users, prgs)), tbs, np.ones(users), legal)
    action = np.array([[2*c]*prgs for c in range(cells)])
    return problem, action


class PhysicalTests(unittest.TestCase):
    def test_full_and_incremental_match_independent_reference(self):
        problem, root = fixture()
        phy = NativePhy(problem)
        self.addCleanup(phy.close)
        base = phy.evaluate(root)
        rng = np.random.default_rng(319)
        actions = [root.copy(), np.full_like(root, -1)]
        for _ in range(80):
            action = root.copy()
            for _ in range(int(rng.integers(1, 7))):
                c, p = int(rng.integers(problem.C)), int(rng.integers(problem.P))
                action[c, p] = rng.choice([-1, 2*c, 2*c+1])
            actions.append(action)
        actions = np.array(actions)
        full = phy.evaluate(actions).values
        incremental = phy.evaluate(actions, base).values
        expected = ReferencePhy(problem).evaluate(actions)
        np.testing.assert_allclose(full, expected, rtol=1.e-10, atol=1.e-8)
        np.testing.assert_array_equal(incremental, full)
        perm = rng.permutation(len(actions))
        np.testing.assert_array_equal(phy.evaluate(actions[perm], base).values, full[perm])
        chunks = [phy.evaluate(actions[i:i+7], base).values for i in range(0, len(actions), 7)]
        np.testing.assert_array_equal(np.concatenate(chunks), full)

    def test_tb_layer_change_recomputes_resources_outside_block(self):
        problem, root = fixture()
        problem.caps[:] = 4
        problem.caps[0, 0] = 1
        phy = NativePhy(problem)
        self.addCleanup(phy.close)
        before = phy.evaluate(root)
        after = root.copy(); after[0, 0] = -1
        result = phy.evaluate(after, before)
        self.assertEqual(before.values[0, 0, LAYER], 1)
        self.assertEqual(result.values[0, 0, LAYER], 4)
        np.testing.assert_allclose(result.values, ReferencePhy(problem).evaluate(after), rtol=1.e-10, atol=1.e-8)

    def test_activity_change_recomputes_all_physical_neighbors(self):
        problem, root = fixture(cells=7)
        phy = NativePhy(problem)
        self.addCleanup(phy.close)
        base = phy.evaluate(root)
        before = phy.stats
        action = root.copy(); action[0, 0] = -1
        actual = phy.evaluate(action, base).values
        self.assertEqual(phy.stats["ue_evaluations"]-before["ue_evaluations"], 7)
        np.testing.assert_allclose(actual, ReferencePhy(problem).evaluate(action), rtol=1.e-10, atol=1.e-8)
        self.assertLess(phy.stats["mmse_misses"], problem.U*problem.P*2**problem.C)

    def test_ue_swap_does_not_invalidate_neighbor_mmse(self):
        problem, root = fixture()
        phy = NativePhy(problem)
        self.addCleanup(phy.close)
        base = phy.evaluate(root)
        before = phy.stats
        action = root.copy(); action[0, 0] = 1
        phy.evaluate(action, base)
        self.assertEqual(phy.stats["ue_evaluations"]-before["ue_evaluations"], 2)
        self.assertEqual(phy.stats["mmse_misses"]-before["mmse_misses"], 1)

    def test_invalid_inputs_and_context_are_rejected(self):
        problem, root = fixture()
        phy, other = NativePhy(problem), NativePhy(problem)
        self.addCleanup(phy.close); self.addCleanup(other.close)
        for value in (-2, problem.U, 2):
            action = root.copy(); action[0, 0] = value
            with self.assertRaises(ValueError):
                phy.evaluate(action)
        with self.assertRaises(ValueError):
            phy.evaluate(root, other.evaluate(root))
        with self.assertRaises(ValueError):
            phy.evaluate(root.astype(float))

    def test_nonmonotonic_native_curve_is_batch_invariant(self):
        x = np.array([3.01, 3.24, 3.26, 3.5, -10, 20])
        value = native_bler(x, 3)
        np.testing.assert_array_equal(value, [float(native_bler(v, 3)) for v in x])
        np.testing.assert_array_equal(value[::-1], native_bler(x[::-1], 3))
        self.assertTrue(np.all((value >= 0) & (value <= 1)))

    def test_anytime_budget_floor_and_reproducibility(self):
        problem, root = fixture()
        for mode in ("greedy", "graph_lns", "random_lns"):
            config = SearchConfig(mode=mode, budget_ms=None, max_candidates=150, batch_size=8)
            a, values, result = solve(problem, root, config)
            repeated, _, _ = solve(problem, root, config)
            np.testing.assert_array_equal(a, repeated)
            self.assertGreaterEqual(result["score"], result["initial_score"])
            self.assertGreaterEqual(values[:, DEQUEUE].sum(), result["floor"])
            self.assertLessEqual(result["candidate_evaluations"], 150)
            problem.actions(a)
        a, _, result = solve(problem, root, SearchConfig(budget_ms=0))
        np.testing.assert_array_equal(a, root)
        self.assertEqual(result["candidate_evaluations"], 1)
        self.assertEqual(result["stop_reason"], "time_budget")

    def test_joint_block_crosses_a_single_edit_reliability_valley(self):
        # Two separate mutings each lose source service without crossing the
        # victim's BLER threshold. Muting both unlocks the victim's complete TB.
        problem, root = fixture(cells=2, prgs=2)
        problem.h[:] = 0
        problem.h[0, :, 0] = 10*np.eye(4)
        problem.h[2, :, 0] = 10*np.eye(4)
        problem.h[2, :, 1] = np.sqrt(10)*np.eye(4)
        problem.caps[:] = 1
        problem.mcs[:] = 0; problem.mcs[2] = 10
        problem.queue[0] = 400; problem.queue[2] = 4000
        problem.tbs[:] = 0
        for n in range(1, 3):
            problem.tbs[0, n, 1:] = 200*n
            problem.tbs[2, n, 1:] = 2000*n
        problem.legal[:] = False
        problem.legal[0, :, 0] = True; problem.legal[1, :, 2] = True
        _, _, greedy = solve(problem, root, SearchConfig(mode="greedy", budget_ms=None, max_candidates=500))
        action, _, block = solve(problem, root, SearchConfig(mode="graph_lns", budget_ms=None, max_candidates=500))
        self.assertAlmostEqual(greedy["improvement"], 0.)
        self.assertGreater(block["improvement"], 3000.)
        np.testing.assert_array_equal(action[0], [-1, -1])
        self.assertGreaterEqual(block["chosen_dequeue"], block["floor"])

    def test_policy_can_stop_and_return_verified_incumbent(self):
        class Stop:
            adaptive = True
            def __call__(self, graph, base, blocks, remaining_ms):
                return []
        problem, root = fixture()
        action, _, result = solve(problem, root, SearchConfig(budget_ms=None), block_orderer=Stop())
        np.testing.assert_array_equal(action, root)
        self.assertEqual(result["stop_reason"], "policy_stop")
        self.assertEqual(result["candidate_evaluations"], 1)

    def test_late_verification_cannot_replace_incumbent(self):
        from . import search
        problem, root = fixture()
        clock = [0.]
        original = NativePhy.evaluate
        def delayed(phy, actions, base=None):
            result = original(phy, actions, base)
            if base is not None:
                clock[0] = .1  # First candidate batch finishes after 1 ms.
            return result
        with patch.object(search.time, "perf_counter", side_effect=lambda: clock[0]), patch.object(NativePhy, "evaluate", delayed):
            action, _, result = solve(problem, root, SearchConfig(mode="greedy", budget_ms=1.))
        np.testing.assert_array_equal(action, root)
        self.assertEqual(result["improvement"], 0.)
        self.assertGreater(result["candidate_evaluations"], 1)
        self.assertGreater(result["overrun_ms"], 0.)

    def test_full_greedy_result_is_independent_of_batch_size(self):
        problem, root = fixture()
        reference = None
        for size in (1, 7, 64, 256):
            action, _, result = solve(problem, root, SearchConfig(mode="greedy", budget_ms=None, max_candidates=100000, batch_size=size))
            self.assertTrue(result["converged"])
            if reference is None:
                reference = action
            np.testing.assert_array_equal(action, reference)


if __name__ == "__main__":
    unittest.main()

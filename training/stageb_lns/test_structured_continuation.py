"""Independent exhaustive checks, teacher-prefix integrity, and hard budgets."""
import itertools
from pathlib import Path
import tempfile
import unittest
import numpy as np

from .test_physics import fixture
from .physics import NativePhy, ReferencePhy, EXPECTED, DEQUEUE
from .search import ProposalGraph, SearchConfig, solve
from .cached_repair import CachedJointRepair
from .structured_repair import factorized_block, FactorizationLimit, StructuredRepair, pareto_indices
from .continuation_repair import ContinuationRepair, bounded_continuation, label_own_prefix


class StructuredContinuationTests(unittest.TestCase):
    def checkpoint(self, ties=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)/'actor.npz'
        rng = np.random.default_rng(73)
        np.savez(path, feature_version=1, mean=np.zeros(79), scale=np.ones(79),
            w1=rng.normal(0, .1, (64, 79)), b1=np.zeros(64),
            w2=rng.normal(0, .1, (32, 64)), b2=np.zeros(32),
            w3=np.zeros((1, 32)) if ties else rng.normal(0, .1, (1, 32)), b3=np.zeros(1))
        return path

    def context(self, **kwargs):
        p, root = fixture(**kwargs)
        phy = NativePhy(p)
        self.addCleanup(phy.close)
        return p, root, phy, phy.evaluate(root), ProposalGraph(phy)

    def exhaustive(self, p, root, block, floor):
        domains = [np.r_[-1, np.flatnonzero(p.legal[t])] for t in block]
        actions = np.repeat(root[None], np.prod([len(x) for x in domains]), axis=0)
        for a, choice in zip(actions, itertools.product(*domains)):
            for token, value in zip(block, choice):
                a[token] = value
        values = ReferencePhy(p).evaluate(actions)
        j = values[:, :, EXPECTED]@p.weights
        d = values[:, :, DEQUEUE].sum(1)
        return float(np.max(np.where(d >= floor, j, -np.inf)))

    def test_exact_factorization_matches_independent_exhaustive_blocks(self):
        for seed in (17, 39, 72):
            p, root, phy, base, graph = self.context(seed=seed, prgs=3)
            p.weights[:] = np.linspace(.5, 3, p.U)
            for block in (((0, 0), (1, 0), (2, 0)),
                          ((0, 0), (0, 1), (1, 0), (1, 2))):
                floor = int(np.ceil(.9*base.values[0, :, DEQUEUE].sum()))
                result = factorized_block(graph, base, block, floor, phy.evaluate)
                self.assertTrue(result.complete)
                self.assertAlmostEqual(result.predicted_j, self.exhaustive(p, root, block, floor), places=6)
                independent = ReferencePhy(p).evaluate(result.action)[0]
                self.assertAlmostEqual(result.predicted_j, independent[:, EXPECTED]@p.weights, places=6)
                self.assertEqual(result.predicted_dequeue, independent[:, DEQUEUE].sum())

    def test_empty_legal_and_no_feasible_schedule_are_explicit(self):
        p, root = fixture(prgs=2)
        p.legal[0, 0] = False
        root[0, 0] = -1
        phy = NativePhy(p); self.addCleanup(phy.close)
        base, graph = phy.evaluate(root), ProposalGraph(phy)
        result = factorized_block(graph, base, ((0, 0), (1, 0)), 1.e20, phy.evaluate)
        self.assertTrue(result.complete)
        self.assertIsNone(result.action)
        self.assertIsNone(result.predicted_j)

    def test_decomposition_limits_and_callback_budget_do_not_claim_optimum(self):
        p, root, phy, base, graph = self.context()
        with self.assertRaises(FactorizationLimit):
            factorized_block(graph, base, ((0, 0), (0, 1)), 0, phy.evaluate, max_cell_assignments=1)
        result = factorized_block(graph, base, ((0, 0),), 0, lambda a, b: None)
        self.assertFalse(result.complete)
        self.assertIsNone(result.action)
        def truncated(a, b):
            return phy.evaluate(a[:1], b)
        result = factorized_block(graph, base, ((0, 0), (0, 1)), 0, truncated)
        self.assertFalse(result.complete)

    def test_pareto_keeps_floor_tradeoff_and_tiny_real_gain(self):
        keep = pareto_indices([10., 9., 8., 10., 10.+1.e-12], [1, 3, 2, 1, 1])
        self.assertEqual(set(keep), {1, 4})

    def test_likelihood_adapter_preserves_frozen_search_and_recording_is_passive(self):
        p, root, phy, base, graph = self.context()
        floor = int(np.ceil(.9*base.values[0, :, DEQUEUE].sum()))
        cfg = SearchConfig(budget_ms=None, max_rounds=3, max_candidates=4096)
        for ties in (False, True):
            checkpoint = self.checkpoint(ties)
            expected = solve(p, root, cfg, repairer=CachedJointRepair(checkpoint))
            records = []
            actual = solve(p, root, cfg, repairer=ContinuationRepair(checkpoint, floor, recorder=records.append))
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
            for key in ('score', 'floor', 'candidate_evaluations', 'accepted_updates', 'rounds'):
                self.assertEqual(actual[2][key], expected[2][key])
            self.assertGreater(len(records), 0)
            for record in records:
                self.assertEqual(record['x'].shape, (len(record['choices']), 95))
                self.assertTrue(np.isfinite(record['x']).all())
                for token in record['block'][record['depth']:]:
                    self.assertEqual(record['partial'][token], -1)

    def test_continuation_honors_fixed_prefix_and_matches_exhaustive_suffix(self):
        p, root, phy, base, graph = self.context(prgs=3)
        block = ((0, 0), (1, 0), (0, 1))
        partial = root.copy(); partial[0, 0] = 1
        submitted = []
        def evaluate(actions, source):
            self.assertTrue(np.all(actions[:, 0, 0] == 1))
            fixed = np.ones(root.shape, bool)
            for token in block:
                fixed[token] = False
            np.testing.assert_array_equal(actions[:, fixed], np.broadcast_to(root[fixed], actions[:, fixed].shape))
            submitted.extend(actions)
            return phy.evaluate(actions, source)
        result = bounded_continuation(graph, base, partial, block, 1, 0, evaluate, horizon=2, width=9)
        expected = self.exhaustive(p, partial, block[1:], 0)
        self.assertTrue(result.complete)
        self.assertAlmostEqual(result.value, expected, places=6)
        self.assertEqual(result.submitted, len(submitted))

    def test_missing_feasible_future_is_not_zero_and_truncation_is_recorded(self):
        p, root, phy, base, graph = self.context()
        result = bounded_continuation(graph, base, root, ((0, 0), (1, 0)), 0,
                                      1.e20, phy.evaluate, max_candidates=1)
        self.assertIsNone(result.value)
        self.assertFalse(result.complete)
        self.assertEqual(result.submitted, 1)

    def test_teacher_uses_real_student_prefix_and_does_not_mutate_observation(self):
        p, root, phy, base, graph = self.context(prgs=3)
        records = []
        repair = ContinuationRepair(self.checkpoint(), 0, recorder=records.append)
        repair.repair(graph, base, ((0, 0), (1, 0), (0, 1)), phy.evaluate)
        observation = records[1]
        before = observation['partial'].copy()
        label = label_own_prefix(p, observation, width=9)
        np.testing.assert_array_equal(before, observation['partial'])
        self.assertTrue(label['known'].all())
        self.assertTrue(label['complete'].all())
        self.assertEqual(label['candidate_evaluations'], label['submitted'].sum()+1)

    def test_search_budgets_cover_decomposition_and_explicit_rollouts(self):
        p, root, phy, base, graph = self.context()
        floor = int(np.ceil(.9*base.values[0, :, DEQUEUE].sum()))
        for repair in (StructuredRepair(floor),
                       ContinuationRepair(self.checkpoint(), floor, mode='rollout', width=2, shortlist=4)):
            action, values, record = solve(p, root,
                SearchConfig(budget_ms=None, max_candidates=40), repairer=repair)
            self.assertLessEqual(record['candidate_evaluations'], 40)
            self.assertGreaterEqual(record['score'], record['initial_score'])
            self.assertGreaterEqual(values[:, DEQUEUE].sum(), floor)
            np.testing.assert_allclose(values, ReferencePhy(p).evaluate(action)[0], rtol=1.e-10, atol=1.e-8)

    def test_value_ranking_requires_actual_pruning_space(self):
        for mode in ('current', 'rollout'):
            with self.assertRaises(ValueError):
                ContinuationRepair(self.checkpoint(), 0, mode=mode, width=16, shortlist=16)


if __name__ == '__main__':
    unittest.main()

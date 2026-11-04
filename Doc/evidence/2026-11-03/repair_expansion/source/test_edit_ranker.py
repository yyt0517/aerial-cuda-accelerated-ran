"""Edit-ranking inputs are causal, whole-TB, identity-free, and cheap."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from .edit_ranker import (
    EditFeatureContext, candidate_features, candidate_features_reference, EditRanker, make_model, export_model,
    FEATURE_DIM, FEATURE_NAMES, BYTE_SCALE,
)
from .physics import Problem, NativePhy, DEQUEUE
from .search import greedy_candidates
from .test_physics import fixture


class EditRankerTests(unittest.TestCase):
    def context(self, **kwargs):
        p, action = fixture(**kwargs)
        phy = NativePhy(p)
        self.addCleanup(phy.close)
        base = phy.evaluate(action)
        return p, action, phy, base, EditFeatureContext(p, phy)

    def test_no_candidate_physics_or_bler_query_and_full_logical_dequeues(self):
        p, root, phy, base, context = self.context()
        candidates = greedy_candidates(p, root)
        before = phy.stats
        floor = int(np.ceil(.9*base.values[0, :, DEQUEUE].sum()))
        with patch.object(phy, 'evaluate', side_effect=AssertionError('Hidden candidate PHY')):
            with patch('training.stageb_lns.physics.native_bler', side_effect=AssertionError('Hidden BLER lookup')):
                x, heuristic = candidate_features(p, context, base, candidates, floor=floor)
        self.assertEqual(before, phy.stats)
        self.assertEqual(x.shape, (len(candidates), FEATURE_DIM))
        self.assertEqual(heuristic.shape, (len(candidates),))
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(np.isfinite(heuristic).all())
        exact = phy.evaluate(candidates, base).values[:, :, DEQUEUE].sum(1)
        np.testing.assert_array_equal(context.last_dequeue_totals, exact)
        np.testing.assert_allclose(x[:, FEATURE_NAMES.index('new_dequeue_slack')],
                                   (exact-floor)/BYTE_SCALE, rtol=1.e-6, atol=1.e-6)

    def test_cross_prg_min_layer_lift_and_block_complement(self):
        p, root = fixture(prgs=5)
        p.caps[:] = 4
        p.caps[0, 0] = 1
        phy = NativePhy(p); self.addCleanup(phy.close)
        base = phy.evaluate(root)
        context = EditFeatureContext(p, phy)
        after = root.copy(); after[0, 0] = -1
        x, _ = candidate_features(p, context, base, np.stack((root, after)))
        exact = phy.evaluate(np.stack((root, after)), base).values[:, :, DEQUEUE].sum(1)
        np.testing.assert_array_equal(context.last_dequeue_totals, exact)
        self.assertEqual(x[0, FEATURE_NAMES.index('dequeue_delta')], 0)
        self.assertGreater(x[1, FEATURE_NAMES.index('positive_layer_delta')], 0)
        self.assertGreater(x[1, FEATURE_NAMES.index('unique_bottleneck_removals')], 0)
        # Four allocations outside the edited PRG remain in the donor TB.
        self.assertGreater(p.tbs[0, 4, 4], p.tbs[0, 5, 1])

    def test_activity_proxy_includes_unchanged_victim_ues(self):
        p, root, phy, base, context = self.context()
        after = root.copy(); after[0, 0] = -1
        x, _ = candidate_features(p, context, base, after)
        self.assertEqual(x[0, FEATURE_NAMES.index('resource_changed_ues')], np.float32(1/p.U))
        self.assertGreater(x[0, FEATURE_NAMES.index('activity_affected_victims')], 1/p.U)
        self.assertGreater(x[0, FEATURE_NAMES.index('activity_risk_gain')], 0)

    def test_feature_stream_permutation_invariance_and_empty_stream(self):
        p, root, phy, base, context = self.context()
        candidates = np.concatenate((root[None], greedy_candidates(p, root)))
        x, h = candidate_features(p, context, base, candidates)
        permutation = np.random.default_rng(17).permutation(len(candidates))
        actual, score = candidate_features(p, context, base, candidates[permutation])
        np.testing.assert_array_equal(actual, x[permutation])
        np.testing.assert_array_equal(score, h[permutation])
        empty, empty_score = candidate_features(p, context, base, candidates[:0])
        self.assertEqual(empty.shape, (0, FEATURE_DIM))
        self.assertEqual(empty_score.shape, (0,))
        self.assertEqual(h[0], 0)

    def test_invalid_neighborhood_and_stale_context_are_rejected(self):
        p, root, phy, base, context = self.context()
        after = root.copy(); after[0, 0] = -1; after[1, 1] = -1
        with self.assertRaises(ValueError):
            candidate_features(p, context, base, after)
        after = root.copy(); after[:, 0] = -1
        with self.assertRaises(ValueError):
            candidate_features(p, context, base, after)
        _, _, other_phy, other_base, _ = self.context(seed=91)
        with self.assertRaises(ValueError):
            candidate_features(p, context, other_base, root)
        with self.assertRaises(ValueError):
            candidate_features(p, context, base, root, floor=-1)

    def test_incumbent_cache_refreshes_after_verified_action_changes(self):
        p, root, phy, base, context = self.context()
        first = greedy_candidates(p, root)
        candidate_features(p, context, base, first)
        after = root.copy(); after[0, 0] = 1
        fresh = phy.evaluate(after)
        candidates = greedy_candidates(p, after)
        x, h = candidate_features(p, context, fresh, candidates)
        ref_x, ref_h = candidate_features(p, phy, fresh, candidates)
        np.testing.assert_array_equal(x, ref_x)
        np.testing.assert_array_equal(h, ref_h)
        np.testing.assert_array_equal(context.last_dequeue_totals,
            phy.evaluate(candidates, fresh).values[:, :, DEQUEUE].sum(1))

    def test_ue_relabeling_preserves_features_and_scores(self):
        p, root, phy, base, context = self.context()
        candidates = greedy_candidates(p, root)
        x, h = candidate_features(p, context, base, candidates)
        permutation = np.array([5, 0, 3, 2, 1, 4])
        inverse = np.argsort(permutation)
        q = Problem(p.h[permutation], p.w[permutation], p.noise, p.queue[permutation],
            p.serving[permutation], p.mcs[permutation], p.caps[permutation],
            p.tbs[permutation], p.weights[permutation], p.legal[:, :, permutation])
        def relabel(actions):
            result = actions.copy()
            result[result >= 0] = inverse[result[result >= 0]]
            return result
        other = NativePhy(q); self.addCleanup(other.close)
        other_base = other.evaluate(relabel(root))
        actual, score = candidate_features(q, other, other_base, relabel(candidates))
        np.testing.assert_allclose(actual, x, rtol=1.e-6, atol=1.e-7)
        np.testing.assert_allclose(score, h, rtol=1.e-10, atol=1.e-8)

    def test_zero_output_is_exact_heuristic_and_numpy_matches_torch(self):
        import torch
        torch.manual_seed(391)
        rng = np.random.default_rng(193)
        x = rng.normal(0, 1, (41, FEATURE_DIM)).astype(np.float32)
        mean = rng.normal(0, .1, FEATURE_DIM).astype(np.float32)
        scale = rng.uniform(.1, 2, FEATURE_DIM).astype(np.float32)
        heuristic = rng.normal(0, 3000, len(x))
        model = make_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'ranker.npz'
            export_model(path, model, mean, scale)
            deployed = EditRanker(path)
            self.assertEqual(deployed.parameters, 2113)
            np.testing.assert_array_equal(deployed.predict(x, heuristic), heuristic)
            with torch.no_grad():
                model[4].weight.normal_(0, .1)
                model[4].bias.fill_(.17)
            export_model(path, model, mean, scale)
            deployed = EditRanker(path)
            tx = torch.from_numpy(np.clip((x-mean)/scale, -10, 10))
            with torch.no_grad():
                expected = heuristic+model(tx).numpy()[:, 0].astype(np.float64)*BYTE_SCALE
            np.testing.assert_allclose(deployed.predict(x, heuristic), expected, rtol=2.e-6, atol=1.e-3)

    def test_compiled_features_match_independent_numpy_reference(self):
        for seed, cells, prgs in ((17, 3, 5), (71, 7, 3), (391, 2, 7)):
            p, root = fixture(seed=seed, cells=cells, prgs=prgs)
            p.weights[:] = np.random.default_rng(seed).uniform(.2, 4, p.U)
            root[:, 0] = -1
            phy = NativePhy(p); self.addCleanup(phy.close)
            base = phy.evaluate(root)
            context = EditFeatureContext(p, phy)
            self.addCleanup(context.close)
            candidates = np.concatenate((root[None], greedy_candidates(p, root)))
            floor = int(np.ceil(.9*base.values[0, :, DEQUEUE].sum()))
            actual, scores = candidate_features(p, context, base, candidates, floor=floor)
            totals = context.last_dequeue_totals.copy()
            expected, expected_scores = candidate_features_reference(p, phy, base, candidates, floor=floor)
            np.testing.assert_allclose(actual, expected, rtol=2.e-6, atol=2.e-6)
            np.testing.assert_allclose(scores, expected_scores, rtol=1.e-10, atol=1.e-8)
            np.testing.assert_array_equal(totals, phy.evaluate(candidates, base).values[:, :, DEQUEUE].sum(1))
            context.close()
            with self.assertRaises(RuntimeError):
                candidate_features(p, context, base, candidates)


if __name__ == '__main__':
    unittest.main()

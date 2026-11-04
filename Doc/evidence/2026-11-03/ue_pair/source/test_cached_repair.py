"""Decoder substitutions must preserve features, beam ties and verified search."""
import tempfile
from pathlib import Path
import unittest
import numpy as np

from .cached_repair import CachedJointRepair, CachedDeadlineRepair
from .deadline import DeadlineRepair
from .neural_repair import JointRepair
from .physics import NativePhy
from .search import ProposalGraph, SearchConfig, solve
from .test_physics import fixture


def recording(cls):
    class Recorded(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.inputs, self.outputs = [], []

        def logits(self, x):
            self.inputs.append(x.copy())
            result = super().logits(x)
            self.outputs.append(result.copy())
            return result
    return Recorded


class CachedRepairTests(unittest.TestCase):
    def models(self, version, width, ties=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        checkpoint = Path(directory.name)/'model.npz'
        rng = np.random.default_rng(193)
        d = 79+(8 if version == 2 else 0)
        np.savez(checkpoint, feature_version=version,
            mean=rng.normal(0, .1, d).astype(np.float32), scale=np.ones(d, np.float32),
            w1=rng.normal(0, .1, (64, d)).astype(np.float32), b1=np.zeros(64, np.float32),
            w2=rng.normal(0, .1, (32, 64)).astype(np.float32), b2=np.zeros(32, np.float32),
            w3=(np.zeros((1, 32)) if ties else rng.normal(0, .1, (1, 32))).astype(np.float32),
            b3=np.zeros(1, np.float32))
        classes = ((JointRepair, CachedJointRepair) if version == 1
                   else (DeadlineRepair, CachedDeadlineRepair))
        return tuple(recording(cls)(checkpoint, width) for cls in classes)

    def equal_proposals(self, models, graph, base, block):
        for model in models:
            model.inputs.clear()
            model.outputs.clear()
        expected, actual = [model.propose(graph, base, block) for model in models]
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(models[0].inputs), len(models[1].inputs))
        for reference, cached in zip(models[0].inputs, models[1].inputs):
            self.assertEqual(reference.tobytes(), cached.tobytes(), 'Feature bytes differ')
        for reference, cached in zip(models[0].outputs, models[1].outputs):
            self.assertEqual(reference.tobytes(), cached.tobytes(), 'Logit bytes differ')
        for key in ('calls', 'proposals', 'option_scores', 'logit_calls'):
            self.assertEqual(getattr(models[0], key), getattr(models[1], key))

    def test_prefix_features_logits_and_tie_order_match_reference(self):
        for version in (1, 2):
            for ties in (False, True):
                for width in (1, 16, 128):
                    with self.subTest(version=version, ties=ties, width=width):
                        p, root = fixture(seed=version+width)
                        rng = np.random.default_rng(311)
                        p.weights = rng.uniform(.1, 4, p.U)
                        p.deadline = rng.uniform(0, 1, (p.U, 8))
                        if not ties:
                            p.sinr_hint = rng.uniform(0, 100, (p.U, p.P, 4))
                        # No legal nonblank option at one token; blank UE index 0
                        # must never update its logical resources.
                        p.legal[0, 0] = False
                        root[0, 0] = -1
                        root[1, 2:] = -1
                        phy = NativePhy(p)
                        try:
                            graph, base = ProposalGraph(phy), phy.evaluate(root)
                            models = self.models(version, width, ties)
                            for block in (((0, 0),), ((0, 0), (1, 1), (0, 1), (1, 2)),
                                          ((1, 2), (0, 1), (1, 1), (0, 0))):
                                self.equal_proposals(models, graph, base, block)
                        finally:
                            phy.close()

    def test_cache_reuses_incumbent_and_invalidates_new_base_or_graph(self):
        models = self.models(2, 16)
        p, root = fixture()
        p.deadline = np.zeros((p.U, 8))
        phy = NativePhy(p)
        self.addCleanup(phy.close)
        graph, base = ProposalGraph(phy), phy.evaluate(root)
        block = ((0, 0), (0, 1), (1, 0), (1, 1))
        self.equal_proposals(models, graph, base, block)
        cached = models[1]
        counts = cached.token_builds, cached.context_builds, cached.option_builds
        self.equal_proposals(models, graph, base, block)
        self.assertEqual(counts, (cached.token_builds, cached.context_builds, cached.option_builds))
        changed = root.copy()
        changed[0, 2] = 1
        self.equal_proposals(models, graph, phy.evaluate(changed), block)
        self.assertEqual(cached.token_builds, counts[0]+1)
        self.assertEqual(cached.option_builds, counts[2])
        other, other_root = fixture(seed=918)
        other.deadline = np.ones((other.U, 8))
        other_phy = NativePhy(other)
        self.addCleanup(other_phy.close)
        self.equal_proposals(models, ProposalGraph(other_phy), other_phy.evaluate(other_root), block)
        self.assertEqual(cached.token_builds, counts[0]+2)
        self.assertEqual(cached.option_builds, counts[2]+len(block))

    def test_verified_search_action_and_work_match_without_time_cutoff(self):
        from .deadline import packet_objective
        models = self.models(2, 16)
        p, root = fixture()
        p.deadline = np.random.default_rng(19).uniform(0, 1, (p.U, 8))
        config = SearchConfig(budget_ms=None, max_candidates=100000, max_rounds=3)
        results = [solve(p, root, config, repairer=m, objective=packet_objective(p)) for m in models]
        for expected, actual in zip(results[0][:2], results[1][:2]):
            np.testing.assert_array_equal(actual, expected)
        for key in ('score', 'objective_score', 'floor', 'chosen_dequeue',
                    'candidate_evaluations', 'rounds', 'accepted_updates', 'stop_reason'):
            self.assertEqual(results[0][2][key], results[1][2][key])


if __name__ == '__main__':
    unittest.main()

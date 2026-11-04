import tempfile
from pathlib import Path
import unittest
import numpy as np

from .test_physics import fixture
from .physics import NativePhy
from .search import ProposalGraph, SearchConfig, solve
from .neural_repair import JointRepair, options, features, destroy, REPAIR_FEATURE_VERSION


class JointRepairTests(unittest.TestCase):
    def test_prefix_changes_features_and_outputs_remain_in_block(self):
        problem, root = fixture()
        phy = NativePhy(problem); self.addCleanup(phy.close)
        base = phy.evaluate(root); graph = ProposalGraph(phy)
        block = ((0,0),(0,1),(1,0),(1,1))
        partial = destroy(root, block)
        choices = options(problem, 0, 1)
        first = features(graph, base, block, partial[None], 1, choices)
        partial[0,0] = 0
        second = features(graph, base, block, partial[None], 1, choices)
        self.assertFalse(np.array_equal(first, second))
        rng = np.random.default_rng(73); d = first.shape[-1]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'repair.npz'
            np.savez(checkpoint,feature_version=REPAIR_FEATURE_VERSION,mean=np.zeros(d),scale=np.ones(d),
                w1=rng.normal(0,.1,(64,d)),b1=np.zeros(64),w2=rng.normal(0,.1,(32,64)),b2=np.zeros(32),
                w3=rng.normal(0,.1,(1,32)),b3=np.zeros(1))
            model = JointRepair(checkpoint,width=12)
            result = model.propose(graph,base,block)
            problem.actions(result)
            mask = np.ones(root.shape,bool)
            for c,p in block: mask[c,p] = False
            np.testing.assert_array_equal(result[:,mask],np.broadcast_to(root[mask],result[:,mask].shape))
            self.assertLessEqual(len(result),12)
            self.assertEqual(len(np.unique(result.reshape(len(result),-1),axis=0)),len(result))
            counts=(model.calls,model.proposals,model.option_scores,model.logit_calls)
            repeated=model.propose(graph,base,block)
            np.testing.assert_array_equal(result,repeated)
            self.assertEqual((model.calls,model.proposals,model.option_scores,model.logit_calls),
                             tuple(2*c for c in counts))
            self.assertGreater(model.option_scores,model.proposals)
            a,_,record=solve(problem,root,SearchConfig(budget_ms=None,max_candidates=128),repairer=model)
            self.assertGreaterEqual(record['score'],record['initial_score'])
            self.assertGreaterEqual(record['chosen_dequeue'],record['floor'])

    def test_warm_start_keeps_original_floor(self):
        problem, root = fixture()
        action,_,first=solve(problem,root,SearchConfig(mode='greedy',budget_ms=None,max_candidates=10000))
        _,_,second=solve(problem,root,SearchConfig(mode='greedy',budget_ms=None,max_candidates=10000),initial_action=action)
        self.assertEqual(first['floor'],second['floor'])
        self.assertGreaterEqual(second['score'],first['score'])

    def test_deterministic_no_change_scan_stops_without_spending_two_more_passes(self):
        class Keep:
            def __init__(self, deterministic):
                self.deterministic=deterministic
            def propose(self, graph, base, block):
                return base.actions.copy()
        problem, root = fixture()
        config=SearchConfig(budget_ms=None,max_candidates=10000,max_rounds=3)
        old,_,reference=solve(problem,root,config,repairer=Keep(False))
        new,_,optimized=solve(problem,root,config,repairer=Keep(True))
        np.testing.assert_array_equal(old,new)
        self.assertEqual(reference['score'],optimized['score'])
        self.assertEqual(optimized['rounds'],1)
        self.assertLess(optimized['candidate_evaluations'],reference['candidate_evaluations'])


if __name__ == '__main__':
    unittest.main()

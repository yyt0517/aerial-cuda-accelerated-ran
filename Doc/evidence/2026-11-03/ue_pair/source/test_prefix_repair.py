"""Verified prefix search must expose legal work and respect search guards."""
import tempfile
from pathlib import Path
import unittest
import numpy as np

from .prefix_repair import PrefixRepair, shortlist, coverage_order
from .physics import NativePhy, EXPECTED, DEQUEUE
from .search import ProposalGraph, SearchConfig, solve
from .deadline import packet_objective
from .test_physics import fixture


class PrefixRepairTests(unittest.TestCase):
    def model(self, mode):
        directory=tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path=Path(directory.name)/'actor.npz'
        rng=np.random.default_rng(501)
        np.savez(path,feature_version=2,mean=np.zeros(87,np.float32),scale=np.ones(87,np.float32),
            w1=rng.normal(0,.1,(64,87)).astype(np.float32),b1=np.zeros(64,np.float32),
            w2=rng.normal(0,.1,(32,64)).astype(np.float32),b2=np.zeros(32,np.float32),
            w3=rng.normal(0,.1,(1,32)).astype(np.float32),b3=np.zeros(1,np.float32))
        return PrefixRepair(path,width=4,mode=mode,prefix_budget=12,reserve=1)

    def test_shortlist_preserves_global_and_parent_local_candidates_within_cap(self):
        joint=np.random.default_rng(93).normal(size=(16,13))
        choices=np.arange(-1,12)
        result=shortlist(joint,choices,3,16,48,2)
        self.assertEqual(len(result),48)
        self.assertEqual(len(np.unique(result)),48)
        self.assertTrue(set(np.argsort(-joint.ravel(),kind='stable')[:16])<=set(result))
        for parent in range(16):
            best=np.argsort(-joint[parent],kind='stable')[:2]
            self.assertTrue(set(parent*13+best)<=set(result))

    def test_coverage_reserves_different_branches_without_losing_best_core(self):
        order=np.arange(10)
        groups=np.array([0,0,0,0,0,1,2,3,4,5])
        kept=coverage_order(order,groups,4,2)
        np.testing.assert_array_equal(kept,[0,1,5,6])
        np.testing.assert_array_equal(coverage_order(order,groups,4,0),order[:4])
        np.testing.assert_array_equal(coverage_order(order,np.zeros(10),4,2),order[:4])

    def test_each_prefix_completes_the_incumbent_and_never_changes_other_tokens(self):
        p,root=fixture()
        p.deadline=np.zeros((p.U,8));p.deadline[:,1]=1
        phy=NativePhy(p);self.addCleanup(phy.close)
        base=phy.evaluate(root);graph=ProposalGraph(phy)
        block=((0,0),(1,0),(0,1),(1,1))
        for mode in PrefixRepair.MODES:
            with self.subTest(mode=mode):
                model=self.model(mode);calls=[]
                def evaluate(actions,source):
                    depth=len(calls)+1
                    p.actions(actions)
                    untouched=np.ones(root.shape,bool)
                    for c,prg in block[:depth]:untouched[c,prg]=False
                    np.testing.assert_array_equal(actions[:,untouched],
                        np.broadcast_to(root[untouched],actions[:,untouched].shape))
                    self.assertIs(source,base)
                    self.assertLessEqual(len(actions),4 if mode=='coverage' else 12)
                    calls.append(len(actions))
                    return phy.evaluate(actions,source)
                model.repair(graph,base,block,evaluate)
                self.assertEqual(len(calls),len(block))
                self.assertEqual(model.proposals,sum(calls))
                self.assertEqual(model.prefix_candidates,sum(calls))

    def test_search_counts_all_verification_and_honors_budget_floor_and_objective(self):
        p,root=fixture()
        p.deadline=np.random.default_rng(15).uniform(0,1,(p.U,8))
        for mode in PrefixRepair.MODES:
            with self.subTest(mode=mode):
                model=self.model(mode);batches=[]
                action,values,result=solve(p,root,
                    SearchConfig(budget_ms=None,max_candidates=119,max_rounds=3,batch_size=7),
                    repairer=model,objective=packet_objective(p),trace=batches.append)
                p.actions(action)
                self.assertEqual(result['candidate_evaluations'],1+sum(len(v['actions']) for v in batches))
                self.assertLessEqual(result['candidate_evaluations'],119)
                self.assertGreaterEqual(result['score'],result['initial_score'])
                self.assertGreaterEqual(values[:,DEQUEUE].sum(),result['floor'])
                independent=NativePhy(p)
                try:np.testing.assert_array_equal(values,independent.evaluate(action).values[0])
                finally:independent.close()


if __name__=='__main__':unittest.main()

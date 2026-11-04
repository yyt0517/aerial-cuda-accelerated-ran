"""Sparse draft reuse must never accept unverified actions or bypass budgets."""
import unittest
import numpy as np
from . import test_prefix_repair
from .test_physics import fixture
from .physics import NativePhy,DEQUEUE
from .search import ProposalGraph,SearchConfig,solve
from .draft_repair import RebasedDraft,DraftRepair
from .residual import DraftCache
from .deadline import packet_objective


class DraftRepairTests(unittest.TestCase):
    def test_rebase_matches_fresh_cache_and_invalidates_geometry_on_new_graph(self):
        p,a=fixture();p.deadline=np.zeros((p.U,8));p.deadline[:,1]=1
        p.sinr_hint=np.ones((p.U,p.P,4))*10
        phy=NativePhy(p);self.addCleanup(phy.close);g=ProposalGraph(phy)
        manager=RebasedDraft();self.addCleanup(manager.close)
        first=manager.for_base(g,phy.evaluate(a));context=first.context
        next_action=a.copy();next_action[0,0]=-1;base=phy.evaluate(next_action)
        rebased=manager.for_base(g,base)
        self.assertEqual(context,rebased.context)
        fresh=DraftCache(g,base);self.addCleanup(fresh.close)
        pool=np.stack((a,next_action))
        for key,v in fresh.evaluate(pool).items():np.testing.assert_array_equal(rebased.evaluate(pool)[key],v)
        second_phy=NativePhy(p);self.addCleanup(second_phy.close)
        new=manager.for_base(ProposalGraph(second_phy),second_phy.evaluate(a))
        self.assertIsNot(new,first)

    def test_exact_quota_all_work_counted_final_action_verified_under_floor(self):
        # Reuse only the temporary model fixture, not another TestCase's tests.
        helper=test_prefix_repair.PrefixRepairTests();helper.addCleanup=self.addCleanup
        actor=helper.model('verified')
        import tempfile
        from pathlib import Path
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        checkpoint=Path(directory.name)/'actor.npz'
        np.savez(checkpoint,feature_version=2,**{k:getattr(actor,k) for k in ('mean','scale','w1','b1','w2','b2','w3','b3')})
        residual_checkpoint=Path(directory.name)/'residual.npz'
        np.savez(residual_checkpoint,version=1,mode=0,mean=np.zeros(32,np.float32),scale=np.ones(32,np.float32),
            w1=np.zeros((3,32,32),np.float32),b1=np.zeros((3,32),np.float32),
            w2=np.zeros((3,16,32),np.float32),b2=np.zeros((3,16),np.float32),
            w3=np.zeros((3,1,16),np.float32),b3=np.zeros((3,1),np.float32))
        p,a=fixture();p.deadline=np.random.default_rng(33).uniform(0,1,(p.U,8))
        outputs={}
        for mode in ('trace','spatial','residual'):
            model=DraftRepair(checkpoint,mode=mode,residual_checkpoint=residual_checkpoint,
                width=4,prefix_budget=12,exact_quota=3)
            self.addCleanup(model.close);trace=[]
            action,values,record=solve(p,a,SearchConfig(max_candidates=79,budget_ms=None,max_rounds=3,batch_size=7),
                repairer=model,objective=packet_objective(p),trace=trace.append)
            self.assertEqual(record['candidate_evaluations'],1+model.verified_candidates)
            self.assertEqual(model.proposals,model.verified_candidates)
            self.assertLessEqual(record['candidate_evaluations'],79)
            self.assertGreaterEqual(model.draft_candidates,model.verified_candidates)
            self.assertGreaterEqual(record['score'],record['initial_score'])
            self.assertGreaterEqual(values[:,DEQUEUE].sum(),record['floor'])
            independent=NativePhy(p)
            try:np.testing.assert_array_equal(values,independent.evaluate(action).values[0])
            finally:independent.close()
            outputs[mode]=(action,values,record['candidate_evaluations'])
        for x,y in zip(outputs['trace'],outputs['residual']):np.testing.assert_array_equal(x,y)


if __name__=='__main__':unittest.main()

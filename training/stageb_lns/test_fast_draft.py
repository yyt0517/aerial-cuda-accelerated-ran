"""Raw binding and omitted uncertainty must leave every mean score unchanged."""
from pathlib import Path
import unittest
import numpy as np
from .test_physics import fixture
from .physics import NativePhy
from .search import ProposalGraph
from .residual import DraftCache,ResidualValue
from .fast_draft import FastDraftCache,MeanResidualValue

class FastDraftTests(unittest.TestCase):
    def test_all_draft_arrays_and_frozen_residual_means_are_bit_identical(self):
        p,a=fixture();p.deadline=np.random.default_rng(123).uniform(0,1,(p.U,8))
        phy=NativePhy(p);self.addCleanup(phy.close);base=phy.evaluate(a);g=ProposalGraph(phy)
        plain=DraftCache(g,base);fast=FastDraftCache(g,base)
        self.addCleanup(plain.close);self.addCleanup(fast.close)
        rng=np.random.default_rng(124);actions=[a.copy()]
        for _ in range(32):
            z=a.copy()
            for _ in range(4):
                c,prg=int(rng.integers(p.C)),int(rng.integers(p.P));z[c,prg]=rng.choice([-1,2*c,2*c+1])
            actions.append(z)
        actions=np.asarray(actions);one=plain.evaluate(actions);two=fast.evaluate(actions)
        for k in one:np.testing.assert_array_equal(one[k],two[k])
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'residual.npz'
            np.savez(path,version=1,mode=0,mean=np.zeros(32,np.float32),scale=np.ones(32,np.float32),
                w1=rng.normal(0,.1,(3,32,32)).astype(np.float32),b1=np.zeros((3,32),np.float32),
                w2=rng.normal(0,.1,(3,16,32)).astype(np.float32),b2=np.zeros((3,16),np.float32),
                w3=rng.normal(0,.1,(3,1,16)).astype(np.float32),b3=np.zeros((3,1),np.float32))
            model=ResidualValue(path);mean=MeanResidualValue(path)
            np.testing.assert_array_equal(model.score(one)[0],mean.score(two)[0])
            self.assertIsNone(mean.score(two)[1])
        bad=a.copy();bad[0,0]=2
        with self.assertRaises(ValueError):fast.evaluate(bad)

if __name__=='__main__':unittest.main()

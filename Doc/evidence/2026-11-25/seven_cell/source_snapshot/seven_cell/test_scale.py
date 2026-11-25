"""Regression tests for UE bits 64..83, FIFO closure and original solver parity."""
import unittest
import numpy as np
from training.stageb_lns.test_physics import fixture
from training.stageb_lns.physics import Problem
from training.stageb_lns.bounded_dpls import solve_bounded as original
from .solver import solve_bounded
from .oracle import audit

class ScaleTests(unittest.TestCase):
    def test_original_three_cell_parity(self):
        for seed in (17,39,113):
            p,a=fixture(seed=seed)
            x=original(p,a);y=solve_bounded(p,a)
            np.testing.assert_array_equal(x[0],y[0]);np.testing.assert_array_equal(x[1],y[1])
            self.assertEqual(x[2]['rounds'],y[2]['rounds'])

    def test_eighty_four_users_including_high_bits(self):
        for seed in (19,41,83):
            p,a=fixture(cells=7,prgs=3,seed=seed)
            ids=np.array([0,11,12,23,24,35,36,47,48,63,64,71,72,83])
            U,C,P=84,7,3
            h=np.zeros((U,P,C,4,4),complex);h[ids]=p.h
            w=np.tile(np.eye(4,dtype=complex),(U,P,1,1));w[ids]=p.w
            serving=np.repeat(np.arange(C),12);serving[ids]=p.serving
            queue=np.zeros(U);queue[ids]=p.queue
            caps=np.ones((U,P),int);caps[ids]=p.caps
            tbs=np.zeros((U,P+1,5),int);tbs[ids]=p.tbs
            legal=np.zeros((C,P,U),bool);legal[:,:,ids]=p.legal
            q=Problem(h,w,p.noise,queue,serving,np.full(U,3),caps,tbs,np.ones(U),legal)
            self.assertTrue(audit(q,ids[a])['passed'])

if __name__=='__main__':unittest.main()

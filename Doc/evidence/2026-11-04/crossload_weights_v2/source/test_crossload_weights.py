"""Causal literature forms, fixed-grid identity and independent weighted oracle."""
import unittest
import numpy as np
from .crossload_native import weights_for,state_features,FEATURES,COEFFICIENTS
from .ttl_weights import objective_weights
from .test_physics import fixture
from .bounded_dpls import solve_bounded
from .persistent_dpls import solve_integer_reference

class CrossloadTests(unittest.TestCase):
    def test_new_TTL_grid_retains_old_weight_arithmetic(self):
        rng=np.random.default_rng(204);base=rng.uniform(.2,2,36);base/=base.mean();urgency=rng.random(36)
        for k in COEFFICIENTS:
            got=weights_for('ttl',base,np.ones(36),urgency,np.ones(36),k)
            expected=objective_weights(base,urgency,k)
            np.testing.assert_array_equal(got.view(np.uint64),expected.view(np.uint64))

    def test_literature_relative_weights_and_empty_finite_case(self):
        ema=np.array([2.,4.,8.]);u=np.array([0.,.5,1.]);q=np.ones(3);a=-np.log(.01)/400
        age=np.array([1.,200.,400.]);pf=1/ema
        expected=a*age*pf;expected/=expected.mean()
        np.testing.assert_array_equal(weights_for('mlwdf_pf',q,ema,u,q,0),expected)
        x=np.mean(a*age);expected=pf*np.exp((a*age-x)/(1+np.sqrt(x)));expected/=expected.mean()
        np.testing.assert_array_equal(weights_for('exp_pf',q,ema,u,q,0),expected)
        for f in ('mlwdf_pf','exp_pf'):
            w=weights_for(f,q,np.zeros(3),np.zeros(3),np.zeros(3),0)
            np.testing.assert_allclose(w,np.ones(3),rtol=2e-16,atol=0)

    def test_weighted_solver_remains_exact(self):
        for f in ('ttl','mlwdf_pf','exp_pf'):
            p,a=fixture();p.weights=weights_for(f,p.weights,np.arange(p.U)+1.,np.linspace(0,1,p.U),p.queue,8)
            r=solve_integer_reference(p,a);b=solve_bounded(p,a)
            np.testing.assert_array_equal(r[0],b[0]);np.testing.assert_array_equal(r[1],b[1])

    def test_low_dimensional_features_are_read_only(self):
        q=np.array([0.,2400.]);e=np.array([1200.,1200.]);u=np.array([0.,.5]);before=[x.copy() for x in (q,e,u)]
        x=state_features(q,e,u);self.assertEqual(x.shape,(len(FEATURES),));self.assertTrue(np.isfinite(x).all())
        for old,new in zip(before,(q,e,u)):np.testing.assert_array_equal(old,new)

if __name__=='__main__':unittest.main()

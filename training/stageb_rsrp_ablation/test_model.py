import unittest
import numpy as np
from training.stageb_lns.test_physics import fixture
from training.stageb_lns.physics import ReferencePhy
from training.stageb_lns.bounded_dpls import solve_bounded,BoundedConfig
from training.stageb_lns.persistent_dpls import solve_integer_reference
from training.stageb_rsrp.model import RsrpMeasurements,approximate_problem
from .model import STAGES,transform,check_counters


class AblationTests(unittest.TestCase):
    def test_spatial_step_preserves_each_prg_source_power(self):
        p,_=fixture();original=p.h.copy();q=transform(p,'spatial')
        cov=q.h@q.h.conj().swapaxes(-1,-2)
        raw=p.h@p.h.conj().swapaxes(-1,-2)
        for u in range(p.U):
            for c in range(p.C):
                if c==p.serving[u]:continue
                target=np.trace(raw[u,:,c],axis1=-2,axis2=-1).real[:,None,None]*np.eye(4)/4
                np.testing.assert_allclose(cov[u,:,c],target,rtol=1e-14,atol=1e-14)
        np.testing.assert_array_equal(p.h,original)

    def test_frequency_step_is_average_of_spatial_covariances(self):
        p,_=fixture();spatial=transform(p,'spatial');wide=transform(p,'frequency')
        a=spatial.h@spatial.h.conj().swapaxes(-1,-2);b=wide.h@wide.h.conj().swapaxes(-1,-2)
        for u in range(p.U):
            for c in range(p.C):
                if c!=p.serving[u]:np.testing.assert_allclose(b[u,:,c],np.broadcast_to(a[u,:,c].mean(0),(p.P,4,4)))

    def test_filtered_and_reported_are_causal_and_old_endpoint_exact(self):
        p,_=fixture();meter=RsrpMeasurements()
        for t in range(22):meter.observe((1+.05*t)*p.h,t)
        report,stamp=meter.read();self.assertEqual(stamp,20)
        last=transform(p,'reported',reported=report)
        np.testing.assert_array_equal(last.h,approximate_problem(p,report).h)
        filtered=transform(p,'filtered',filtered=meter.filtered)
        self.assertFalse(np.array_equal(last.h,filtered.h))
        for stage in STAGES:
            q=transform(p,stage,filtered=meter.filtered,reported=report)
            np.testing.assert_array_equal(q.h[np.arange(p.U),:,p.serving],p.h[np.arange(p.U),:,p.serving])

    def test_profile_observer_and_pruning_do_not_change_decisions(self):
        p,a=fixture(prgs=3)
        for stage in STAGES:
            q=transform(p,stage,filtered=np.ones((p.U,p.C)),reported=np.ones((p.U,p.C))*.9)
            plain=solve_bounded(q,a);check_counters(plain[2])
            profiled=solve_bounded(q,a,BoundedConfig(diagnostics=True))
            full=solve_integer_reference(q,a)
            np.testing.assert_array_equal(plain[0],profiled[0]);np.testing.assert_array_equal(plain[1],profiled[1])
            np.testing.assert_array_equal(plain[0],full[0]);np.testing.assert_array_equal(plain[1],full[1])
            np.testing.assert_allclose(plain[1],ReferencePhy(q).evaluate(plain[0])[0],rtol=1e-8,atol=1e-6)


if __name__=='__main__':unittest.main()

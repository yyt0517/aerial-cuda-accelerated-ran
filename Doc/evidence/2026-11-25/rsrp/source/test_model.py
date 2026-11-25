import unittest
import numpy as np
from training.stageb_lns.test_physics import fixture
from training.stageb_lns.physics import ReferencePhy, NativePhy
from training.stageb_lns.bounded_dpls import solve_bounded
from training.stageb_lns.persistent_dpls import solve_integer_reference
from .model import RsrpConfig, RsrpMeasurements, approximate_problem


class RsrpTests(unittest.TestCase):
    def test_delayed_periodic_reports_and_linear_filter(self):
        meter = RsrpMeasurements(RsrpConfig(2, 1, 3.0))
        h = np.ones((2, 3, 3, 4, 4), complex)
        meter.observe(h, 0)
        with self.assertRaises(RuntimeError): meter.read()
        meter.observe(2*h, 1)
        np.testing.assert_array_equal(meter.read()[0], np.ones((2, 3)))
        meter.observe(3*h, 2)
        self.assertEqual(meter.read()[1], 0)
        expected = 1 + meter.alpha*3
        expected += meter.alpha*(9-expected)
        meter.observe(100*h, 3)
        np.testing.assert_allclose(meter.read()[0], expected)
        self.assertEqual(meter.read()[1], 2)
        with self.assertRaises(ValueError): meter.observe(h, 5)

    def test_neighbor_phase_and_unmeasured_ports_cannot_leak_into_report(self):
        p, _ = fixture()
        a, b = RsrpMeasurements(), RsrpMeasurements()
        h = p.h.copy()
        h[..., 0] *= np.exp(1j*np.arange(4))[None, None, None, :]
        h[..., 1:] *= 19.0
        for t in range(3):
            a.observe(p.h, t); b.observe(h, t)
        np.testing.assert_allclose(a.read()[0], b.read()[0])

    def test_scalar_covariance_and_original_execution_are_unchanged(self):
        p, action = fixture()
        raw = p.h.copy()
        before = ReferencePhy(p).evaluate(action)
        rsrp = np.arange(p.U*p.C).reshape(p.U, p.C)/10+.1
        q = approximate_problem(p, rsrp)
        cov = q.h @ q.h.conj().swapaxes(-1, -2)
        for u in range(p.U):
            for c in range(p.C):
                if c != p.serving[u]:
                    np.testing.assert_allclose(cov[u, :, c], np.broadcast_to(4*rsrp[u,c]*np.eye(4), (p.P,4,4)))
        np.testing.assert_array_equal(q.h[np.arange(p.U), :, p.serving], raw[np.arange(p.U), :, p.serving])
        np.testing.assert_array_equal(p.h, raw)
        np.testing.assert_array_equal(ReferencePhy(p).evaluate(action), before)
        self.assertFalse(np.allclose(before[..., 4], ReferencePhy(q).evaluate(action)[..., 4]))
        changed = p.h.copy()
        for u in range(p.U):
            changed[u, :, np.arange(p.C)!=p.serving[u]] *= 100
        p.h = changed
        np.testing.assert_array_equal(approximate_problem(p, rsrp).h, q.h)

    def test_approximate_phy_and_bounded_search_match_independent_oracles(self):
        p, action = fixture(prgs=3)
        q = approximate_problem(p, np.full((p.U,p.C), .15))
        phy = NativePhy(q)
        try:
            np.testing.assert_allclose(phy.evaluate(action).values, ReferencePhy(q).evaluate(action), rtol=1e-10, atol=1e-8)
        finally: phy.close()
        actual = solve_bounded(q, action)
        expected = solve_integer_reference(q, action)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertTrue(actual[2]['converged'])


if __name__ == '__main__': unittest.main()

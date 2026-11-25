import unittest
from dataclasses import replace
import numpy as np
from training.stageb_lns.test_physics import fixture
from training.stageb_lns.physics import ReferencePhy
from training.stageb_lns.bounded_dpls import solve_bounded
from training.stageb_lns.persistent_dpls import solve_integer_reference
from training.stageb_rsrp.model import approximate_problem
from .model import (normalized_shapes, pack_shape, unpack_shape, decode_factor,
                    spatial_problem, SpatialReports, SpatialReceiver, neighbors, feedback_rows)


class SpatialTests(unittest.TestCase):
    def test_trace_psd_encoding_and_power_separation(self):
        p, _ = fixture()
        beta = np.full((p.U, p.C), .7)
        for granularity, shape in normalized_shapes(p.h, p.serving).items():
            encoded = pack_shape(shape)
            restored = unpack_shape(encoded)
            np.testing.assert_allclose(restored, shape, atol=2e-14)
            np.testing.assert_allclose(np.trace(restored, axis1=-2, axis2=-1), 4, atol=2e-14)
            factor = decode_factor(encoded)
            np.testing.assert_allclose(factor@factor.conj().swapaxes(-1, -2), shape, atol=2e-14)
            q = spatial_problem(p, beta, factor)
            cov = q.h@q.h.conj().swapaxes(-1, -2)
            for u in range(p.U):
                for j, c in enumerate(neighbors(p.serving, p.C)[u]):
                    target = np.broadcast_to(4*.7*shape[u, :, j], (p.P, 4, 4))
                    np.testing.assert_allclose(cov[u, :, c], target, atol=2e-14)
            np.testing.assert_array_equal(q.h[np.arange(p.U), :, p.serving], p.h[np.arange(p.U), :, p.serving])

    def test_isotropic_packet_matches_scalar_phy_and_ignores_neighbor_H(self):
        p, action = fixture(prgs=3)
        beta = np.full((p.U, p.C), .3)
        shape = np.broadcast_to(np.eye(4), (p.U, 1, p.C-1, 4, 4))
        factor = decode_factor(pack_shape(shape))
        q = spatial_problem(p, beta, factor)
        scalar = approximate_problem(p, beta)
        np.testing.assert_allclose(ReferencePhy(q).evaluate(action), ReferencePhy(scalar).evaluate(action), atol=1e-12)
        changed = p.h.copy()
        for u in range(p.U):
            for c in range(p.C):
                if c != p.serving[u]: changed[u, :, c] *= 100+19j
        np.testing.assert_array_equal(spatial_problem(replace(p, h=changed), beta, factor).h, q.h)

    def test_delay_hold_and_receiver_reuse(self):
        p, _ = fixture()
        meter = SpatialReports(p.serving, ('prg_t4', 'wideband_t4'))
        meter.observe(p.h, 0)
        with self.assertRaises(RuntimeError): meter.read('prg_t4')
        receiver = SpatialReceiver()
        for t in range(1, 8):
            h = p.h.copy(); h[..., 0, :] *= 1+t
            meter.observe(h, t)
            sampled, packet = meter.read('prg_t4')
            self.assertEqual(sampled, 0 if t <= 4 else 4)
            old = receiver.factor
            updated = receiver.accept(sampled, packet)
            self.assertEqual(updated, t in (1, 5))
            if not updated: self.assertIs(old, receiver.factor)
        np.testing.assert_array_equal(meter.read('prg_t4')[1], pack_shape(normalized_shapes(
            np.concatenate((p.h[..., :1, :]*5, p.h[..., 1:, :]), axis=-2), p.serving)['prg']))

    def test_singular_covariance_and_exact_search_reference(self):
        p, action = fixture(prgs=3)
        zero = normalized_shapes(np.zeros_like(p.h), p.serving)
        np.testing.assert_array_equal(zero['prg'], np.broadcast_to(np.eye(4), zero['prg'].shape))
        h = p.h.copy(); h[..., 1:, :] = 0
        rank_one = normalized_shapes(h, p.serving)['prg']
        factor = decode_factor(pack_shape(rank_one))
        np.testing.assert_allclose(factor@factor.conj().swapaxes(-1, -2), rank_one, atol=1e-12)
        for shape in normalized_shapes(p.h, p.serving).values():
            q = spatial_problem(p, np.full((p.U, p.C), .7), decode_factor(pack_shape(shape)))
            bounded = solve_bounded(q, action); full = solve_integer_reference(q, action)
            np.testing.assert_array_equal(bounded[0], full[0])
            np.testing.assert_array_equal(bounded[1], full[1])
            np.testing.assert_allclose(bounded[1], ReferencePhy(q).evaluate(bounded[0])[0], rtol=1e-8, atol=1e-6)

    def test_feedback_counts_only_neighbors_and_normalized_degrees(self):
        rows = {r['stage']: r for r in feedback_rows()}
        self.assertEqual(rows['reported']['payload_mbps_at_16bits'], .1152)
        self.assertAlmostEqual(rows['prg_t200']['payload_mbps_at_16bits'], 3.0528)
        self.assertAlmostEqual(rows['wideband_t200']['payload_mbps_at_16bits'], .288)
        self.assertAlmostEqual(rows['exact']['payload_mbps_at_16bits'], 626.688)


if __name__ == '__main__': unittest.main()

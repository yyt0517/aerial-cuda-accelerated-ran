"""No PHY evaluations: bound arithmetic, lookup structure and lazy certificates."""
import unittest
from unittest.mock import patch
import numpy as np
from .certified_bounds import lazy_canonical,logical_bounds,curve_structure,SCALE
from .dpls_operations import canonical_choice,compact_neighborhood
from .test_physics import fixture
from .physics import COUNT,LAYER,CAPACITY,DEQUEUE,EXPECTED


class CertificateTests(unittest.TestCase):
    def test_one_unit_tie_requires_smaller_unseen_key(self):
        exact=np.array([100,99]);keys=np.array([20,10]);upper=exact.copy()
        # The proposed100>=99+1 shortcut would incorrectly stop at key20.
        got,paid=lazy_canonical(keys,upper,[True,True],16,lambda i:int(exact[i]),known={0:100})
        self.assertEqual(got,1);self.assertEqual(paid,[1])
        got,paid=lazy_canonical(keys[::-1],upper,[True,True],16,lambda i:int(exact[i]),known={0:100})
        self.assertEqual(got,0);self.assertEqual(paid,[])

    def test_new_maximum_changes_anchored_band(self):
        exact=np.array([99,100,101]);keys=np.array([0,10,20])
        got,paid=lazy_canonical(keys,exact,np.ones(3,bool),16,lambda i:int(exact[i]),known={0:99,1:100})
        self.assertEqual(got,1);self.assertEqual(paid,[2])

    def test_threshold_certificate_and_floor_need_no_evaluations(self):
        got,paid=lazy_canonical([1,2,3],np.array([16,15,100]),[True,True,False],16,
            lambda _:self.fail('A certified candidate was evaluated'))
        self.assertEqual(got,-1);self.assertEqual(paid,[])

    def test_order_only_changes_work_not_canonical_choice(self):
        rng=np.random.default_rng(719)
        for _ in range(150):
            n=int(rng.integers(1,50));values=rng.integers(-10,40,n)
            upper=values+rng.integers(0,25,n);keys=rng.permutation(n)
            eligible=rng.random(n)>.2;known={i:int(values[i]) for i in range(n) if rng.random()<.3}
            expected=canonical_choice(values,eligible,keys,4,1)
            for priority in (upper,rng.normal(size=n),-values):
                got,paid=lazy_canonical(keys,upper,eligible,4,lambda i:int(values[i]),
                    known=known,priority=priority)
                self.assertEqual(got,expected);self.assertEqual(len(paid),len(set(paid)))
                if np.array_equal(priority,upper):
                    relevant=eligible&(upper>4)
                    if expected>=0:
                        maximum=values[eligible&(values>4)].max()
                        relevant&=upper>=maximum-1
                    self.assertLessEqual(len(paid),int(relevant.sum()))

    def test_invalid_bound_or_noninteger_exact_label_refuses_certificate(self):
        with self.assertRaises(AssertionError):lazy_canonical([1],np.array([20]),[True],16,lambda _:21)
        with self.assertRaises(TypeError):lazy_canonical([1],np.array([20]),[True],16,lambda _:17.5)
        with self.assertRaises(TypeError):lazy_canonical([1],np.array([20]),[True],16,lambda _:17,known={0:17.5})

    def test_native_lookup_structure_including_MCS3_unsorted_axis(self):
        rows=curve_structure()
        self.assertEqual([r['mcs'] for r in rows if not r['SNR_axis_strictly_descending']],[3])
        self.assertTrue(all(r['BLER_real_monotone'] and r['BLER_range_0_1'] for r in rows))

    def test_logical_capacity_bound_and_floor_without_PHY(self):
        p,root=fixture(seed=19);root[0,0]=-1
        def logical(a):
            result=np.zeros((p.U,8))
            for u in range(p.U):
                chosen=np.flatnonzero(a[p.serving[u]]==u);n=len(chosen)
                layer=int(p.caps[u,chosen].min()) if n else 0
                d=min(p.queue[u],p.tbs[u,n,layer])
                result[u,[COUNT,LAYER,CAPACITY,DEQUEUE,EXPECTED]]=[n,layer,p.tbs[u,n,layer],d,.7*d]
            return result
        values=logical(root);prg,assignment=compact_neighborhood(root,p.legal)
        with patch('numpy.linalg.inv',side_effect=AssertionError('No MMSE')):
            bounds=logical_bounds(p,root,values,prg,assignment)
        base_q=np.rint(values[:,EXPECTED]*p.weights*SCALE).astype(np.int64).sum()
        for i in range(len(prg)):
            a=root.copy();a[:,prg[i]]=assignment[i];v=logical(a)
            q=np.rint(v[:,EXPECTED]*p.weights*SCALE).astype(np.int64).sum()-base_q
            self.assertLessEqual(q,bounds['capacity_q'][i])
            self.assertEqual(v[:,DEQUEUE].sum()-values[:,DEQUEUE].sum(),bounds['delta_d'][i])


if __name__=='__main__':unittest.main()

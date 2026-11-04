"""Joint invalidation, stale-label causality and permutation-invariant ties."""
import itertools
import unittest
import numpy as np
from .dpls_operations import (compact_neighborhood, expand_edits, edit_metadata,
                              infer_legal, canonical_choice, canonical_order)
from .analyze_dpls_joint_cost import batch_union, stale_rank
from .analyze_dpls_moves import operators
from .search import greedy_candidates
from .test_physics import fixture


def frame(root, p):
    prg, assignment = compact_neighborhood(root, p.legal)
    keys, affected, footprints = edit_metadata(root, prg, assignment, p.U)
    return dict(root=root, prg=prg, assignment=assignment, keys=keys,
                affected=affected, footprints=footprints, users=p.U)


class JointDiagnosticsTests(unittest.TestCase):
    def test_compact_domain_and_keys_equal_original_stream(self):
        p, root = fixture()
        for action in (root, np.full_like(root, -1)):
            f = frame(action,p)
            np.testing.assert_array_equal(expand_edits(action,f['prg'],f['assignment']), greedy_candidates(p,action))
            np.testing.assert_array_equal(infer_legal(action,f['prg'],f['assignment'],p.U),p.legal)
            previous = operators(action,f['prg'],f['assignment'],p.U)
            for got,want in zip((f['keys'],f['affected'],f['footprints']),previous):
                np.testing.assert_array_equal(got,want)

    def test_batch_union_is_union_not_product(self):
        p, root = fixture(cells=3, prgs=3)
        f = frame(root,p)
        # Identity replacements on different cells/PRGs have disjoint UE sets.
        def find(q, values):
            return int(np.flatnonzero((f['prg']==q) & np.all(f['assignment']==values,axis=1))[0])
        a,b = find(0,[1,2,4]),find(1,[0,3,4])
        self.assertEqual(int(f['footprints'][a] & f['footprints'][b]),0)
        single = batch_union(f,p.legal,[a])
        both = batch_union(f,p.legal,[a,b])
        self.assertGreater(both['union_common_fraction'],single['union_common_fraction'])

    def test_canonical_selection_is_permutation_invariant_and_max_anchored(self):
        # a~b, b~c, a!~c: a pairwise tolerance comparator would be ambiguous.
        gain = np.array([10.,10.-.75e-6,10.-1.5e-6])
        keys = np.array([30,20,10])
        for perm in itertools.permutations(range(3)):
            perm = np.array(perm)
            i = canonical_choice(gain[perm],np.ones(3,bool),keys[perm],1.,1.e-6)
            self.assertEqual(keys[perm[i]],20)
        self.assertEqual(canonical_choice(np.array([16.-1.e-7,16.+1.e-7]),
            np.ones(2,bool),np.array([0,1]),16.),1)
        np.testing.assert_array_equal(keys[canonical_order(gain,keys)], [20,30,10])

    def test_stale_order_never_reads_new_gains(self):
        previous = dict(keys=np.array([1,2,3]),gain=np.array([3.,2.,1.]),
                        chosen=0, footprints=np.array([1,2,4],np.uint64))
        current = dict(keys=np.array([1,2,3,4]),gain=np.array([999.,999.,999.,999.]),
                       chosen=1,eligible=np.ones(4,bool),footprints=np.array([1,2,4,8],np.uint64))
        result = stale_rank(previous,current)
        self.assertEqual(result['rank_all'],2)
        current['gain'] *= -1
        self.assertEqual(stale_rank(previous,current)['rank_all'],2)
        current['chosen'] = 3
        self.assertFalse(stale_rank(previous,current)['top16_all'])
        self.assertTrue(stale_rank(previous,current)['dirty_top16_recovers'])


if __name__ == '__main__':
    unittest.main()

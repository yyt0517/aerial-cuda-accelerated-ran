"""Ordered-pool and complete decision equivalence for the cost-only backend."""
import unittest
import numpy as np
from .test_physics import fixture
from .physics import COUNT,LAYER
from .search import ProposalGraph
from .resource_conditions import Conditions,ConditionalPhy,ConditionalConfig,local_pool,logical,solve_conditional
from .resource_conditions_fast import fast_pool,ordered_unique,solve_conditional as solve_fast


class FastResourceTests(unittest.TestCase):
    def test_compiled_pool_has_identical_order_and_legality(self):
        for seed in (17,29,43):
            p,a=fixture(seed=seed);a[:,1::2]+=1;a[0,2]=-1
            phy=ConditionalPhy(p);self.addCleanup(phy.close);base=phy.evaluate(a);g=ProposalGraph(phy);n,l=logical(p,a)
            for condition in (Conditions(),Conditions(n[0],l[0]),Conditions(activity=a>=0),Conditions(n[0],l[0],a>=0)):
                condition=condition.validate(p)
                np.testing.assert_array_equal(local_pool(p,g,base,condition),fast_pool(p,g,base,condition))

    def test_byte_dedup_matches_first_original_occurrence(self):
        p,a=fixture();b=a.copy();b[1,1]=-1;c=a.copy();c[0,2]=-1
        rows=np.stack([b,a,b,c,a,c])
        _,ids=np.unique(rows.reshape(len(rows),-1),axis=0,return_index=True)
        np.testing.assert_array_equal(rows[np.sort(ids)],ordered_unique(rows))

    def test_final_values_and_actual_work_unchanged(self):
        p,a=fixture(prgs=5);a[:,1::2]+=1;n,l=logical(p,a);config=ConditionalConfig(max_candidates=73)
        for condition in (Conditions(),Conditions(n[0],l[0]),Conditions(activity=a>=0),Conditions(n[0],l[0],a>=0)):
            reference=solve_conditional(p,a,condition,config,offline_diagnostic=True)
            fast=solve_fast(p,a,condition,config,offline_diagnostic=True)
            np.testing.assert_array_equal(reference[0],fast[0]);np.testing.assert_array_equal(reference[1],fast[1])
            for key in ('candidate_evaluations','proxy_candidates','iterations','feature_roots','mmse_requests','mmse_misses'):
                self.assertEqual(reference[2][key],fast[2][key])


if __name__=='__main__':unittest.main()

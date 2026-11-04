"""Sparse approximate PHY must preserve full-TB and interference dependencies."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

from .physics import NativePhy,EXPECTED
from .search import ProposalGraph
from .test_physics import fixture
from .residual import DraftCache,terms
from .deadline import urgency,packet_objective


def reference(graph,base,actions):
    # Independent previous-round implementation, read directly from its artifact.
    path=Path(__file__).resolve().parents[2]/'deliverables/tb_process_value_20261029_v1/controls.py'
    spec=importlib.util.spec_from_file_location('old_trace_control',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.TraceDraft(graph,base).score(actions)


class ResidualCacheTests(unittest.TestCase):
    def setup_case(self,hints=True):
        p,a=fixture();p.deadline=np.random.default_rng(33).uniform(0,1,(p.U,8))
        if hints:p.sinr_hint=np.random.default_rng(34).uniform(.1,30,(p.U,p.P,4))
        phy=NativePhy(p);self.addCleanup(phy.close)
        base=phy.evaluate(a);g=ProposalGraph(phy);cache=DraftCache(g,base)
        self.addCleanup(cache.close)
        return p,a,phy,base,g,cache

    def test_reference_parity_without_hidden_exact_phy_and_stable_batching(self):
        for hints in (True,False):
            p,a,phy,base,g,cache=self.setup_case(hints)
            rng=np.random.default_rng(77);actions=[a.copy()]
            for _ in range(70):
                z=a.copy()
                for _ in range(int(rng.integers(1,5))):
                    c,pr=int(rng.integers(p.C)),int(rng.integers(p.P))
                    z[c,pr]=rng.choice([-1,2*c,2*c+1])
                actions.append(z)
            actions=np.array(actions);before=phy.stats
            with patch.object(phy,'evaluate',side_effect=AssertionError('hidden exact PHY')):
                result=cache.evaluate(actions)
            self.assertEqual(before,phy.stats)
            np.testing.assert_allclose(result['scores'][:,0],reference(g,base,actions),rtol=1.e-10,atol=1.e-8)
            np.testing.assert_array_equal(result['scores'][0],0)
            self.assertFalse(result['mask'][0].any())
            for i in range(len(actions)):
                one=cache.evaluate(actions[i:i+1])
                for key in result:np.testing.assert_array_equal(one[key][0],result[key][i])
            self.assertTrue(np.isfinite(result['features']).all())

    def test_resource_layer_change_recomputes_entire_tb_and_activity_victims(self):
        p,a,phy,base,g,cache=self.setup_case()
        changed=a.copy();changed[0,0]=-1
        out=cache.evaluate(changed)
        self.assertTrue(out['mask'][0,0])
        self.assertTrue(out['mask'][0,2])
        self.assertTrue(out['mask'][0,4])
        self.assertFalse(out['mask'][0,1])
        exact=phy.evaluate(changed,base)
        # Exact expected bytes outside the approximate dependency closure stay fixed.
        np.testing.assert_array_equal(exact.values[0,~out['mask'][0],EXPECTED],base.values[0,~out['mask'][0],EXPECTED])
        self.assertEqual(cache.stats['affected_ues'],3)
        self.assertEqual(cache.stats['affected_prgs'],14)
        e=out['raw'][0,:,1]
        np.testing.assert_allclose((terms(e,out['raw'][0])-terms(base.values[0,:,EXPECTED],out['raw'][0])).sum(),out['scores'][0,0],atol=1.e-9)

    def test_cache_refuses_foreign_graph_closed_use_and_illegal_assignments(self):
        p,a,phy,base,g,cache=self.setup_case()
        other=NativePhy(p);self.addCleanup(other.close)
        with self.assertRaises(ValueError):DraftCache(ProposalGraph(other),base)
        bad=a.copy();bad[0,0]=2
        with self.assertRaises(ValueError):cache.evaluate(bad)
        cache.close()
        with self.assertRaises(RuntimeError):cache.evaluate(a)
        with self.assertRaises(RuntimeError):_ = cache.stats


if __name__=='__main__':unittest.main()

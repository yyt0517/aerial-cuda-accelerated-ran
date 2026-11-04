import unittest
import numpy as np
from .test_physics import fixture
from .physics import NativePhy,ReferencePhy,DEQUEUE,COUNT,EXPECTED
from .dpls_operations import canonical_choice
from .persistent_dpls import (PersistentPhy,PersistentConfig,solve_persistent,solve_integer_reference,
                              q_contributions,check_contract,weak_mask)


class PersistentTests(unittest.TestCase):
    def test_compiled_phy_matches_frozen_binary_bitwise(self):
        p,draft=fixture()
        a,b=NativePhy(p),PersistentPhy(p)
        try:np.testing.assert_array_equal(a.evaluate(draft).values,b.evaluate(draft).values)
        finally:a.close();b.close()

    def test_every_candidate_matches_independent_full_integer_definition(self):
        for seed in (17,19,39,113):
            p,draft=fixture(seed=seed);trace=[]
            def save(r):trace.append({k:v.copy() if isinstance(v,np.ndarray) else v for k,v in r.items()})
            reference=solve_integer_reference(p,draft,observer=save)
            for mode in ('full','cached'):
                def audit(r):
                    expected=trace[r['iteration']-1]
                    for key in ('root','root_values','keys','values','delta_q','delta_d','selected_keys'):
                        np.testing.assert_array_equal(r[key],expected[key],err_msg=key)
                    self.assertEqual(r['score_q'],expected['score_q'])
                actual=solve_persistent(p,draft,PersistentConfig(mode),observer=audit)
                np.testing.assert_array_equal(actual[0],reference[0]);np.testing.assert_array_equal(actual[1],reference[1])
                self.assertEqual(actual[2]['score_q'],reference[2]['score_q'])
                self.assertEqual(actual[2]['rounds'],len(trace))
                self.assertEqual(actual[2]['candidate_evaluations'],actual[2]['initialization_evaluations']+actual[2]['scored_edits'])
                if mode=='cached':
                    self.assertLess(actual[2]['generated_edits'],actual[2]['sum_neighborhood_sizes'])

    def test_guarded_batch_phy_and_no_guard_evaluation_cost(self):
        p,draft=fixture(seed=39)
        for mode in ('cap4','guarded4'):
            a,v,r=solve_persistent(p,draft,PersistentConfig(mode))
            np.testing.assert_allclose(v,ReferencePhy(p).evaluate(a)[0],rtol=1.e-10,atol=1.e-8)
            self.assertGreaterEqual(v[:,DEQUEUE].sum(),r['floor'])
            self.assertEqual(r['candidate_evaluations'],1+r['scored_edits'])
            self.assertTrue(r['converged'])

    def test_quantization_ties_and_overflow_contract(self):
        values=np.zeros((2,8));values[:,6]=[.5e-6,1.5e-6]
        np.testing.assert_array_equal(q_contributions(values,np.ones(2)),[0,2])
        p,_=fixture();p.queue[:]=1.e15
        with self.assertRaises(OverflowError):check_contract(p)

    def test_guard_acceptance_and_fallback_against_single_counterfactual(self):
        p,draft=fixture(prgs=7,seed=1)
        p.weights[:]=np.random.default_rng(820).uniform(1,4,p.U)
        weak=np.array([bool((weak_mask(p)>>u)&1) for u in range(p.U)])
        def audit(r):
            if not len(r['selected_keys']):return
            best=canonical_choice(r['delta_q'],r['dequeue']+r['delta_d']>=r['floor'],
                r['keys'],(1 if r['iteration']<=2 else 16)*1_000_000,1)
            self.assertEqual(r['selected_keys'][0],r['keys'][best])
            if r['guard_rejected']:
                self.assertEqual(len(r['selected_keys']),1)
            if len(r['selected_keys'])>1:
                combined=r['root_values'].copy();used=np.zeros(p.U,dtype=bool)
                for key in r['selected_keys']:
                    v=r['values'][np.flatnonzero(r['keys']==key)[0]]
                    changed=np.any(v!=r['root_values'],axis=1)
                    self.assertFalse(np.any(used & changed))
                    combined[changed]=v[changed];used|=changed
                single=r['values'][best]
                self.assertFalse(np.any((single[:,COUNT]>0)&(combined[:,COUNT]==0)))
                self.assertTrue(np.all(combined[weak,EXPECTED]+1.e-6>=single[weak,EXPECTED]))
                self.assertGreaterEqual(combined[:,DEQUEUE].sum(),r['floor'])
        _,_,r=solve_persistent(p,draft,PersistentConfig('guarded4'),observer=audit)
        self.assertGreater(r['accepted_batches'],0)
        self.assertGreater(r['guard_rejections'],0)
        self.assertEqual(r['candidate_evaluations'],r['initialization_evaluations']+r['scored_edits'])


if __name__=='__main__':unittest.main()

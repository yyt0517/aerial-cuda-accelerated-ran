"""Certificates, spatial sufficient statistics, and bounded residual contracts."""
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import numpy as np

from .test_physics import fixture
from .physics import NativePhy,ReferencePhy,COUNT,LAYER,DEQUEUE,EXPECTED,curves,native_bler
from .search import greedy_candidates
from .ranked_dpls import solve_ranked
from .bounded_probes import ProbePhy,Predictor,certified_search,block_search,generate,local_actions


class ProbeTests(unittest.TestCase):
    def test_probability_bound_holds_over_all_native_piecewise_segments(self):
        for m,(xs,ys) in enumerate(curves()):
            knots=np.unique(np.r_[xs,-4.75])
            samples=np.r_[np.nextafter(knots,-np.inf),knots,np.nextafter(knots,np.inf),
                (knots[1:]+knots[:-1])/2,knots[0]-1,knots[-1]+1]
            values=native_bler(samples,m)
            self.assertTrue(np.all((values>=0)&(values<=1)),m)

    def test_new_predictor_matches_torch_with_clipping_and_nonzero_weights(self):
        import torch
        rng=np.random.default_rng(20261102)
        for width in (48,99,120):
            data=dict(mean=rng.normal(size=width).astype(np.float32),scale=rng.uniform(.1,2,width).astype(np.float32))
            for key,shape in [('w1',(32,width)),('b1',(32,)),('w2',(16,32)),('b2',(16,)),('w3',(1,16)),('b3',(1,))]:
                data[key]=rng.normal(0,.1,shape).astype(np.float32)
            x=rng.normal(0,20,(37,width)).astype(np.float32);prior=rng.normal(0,3000,len(x))
            z=torch.clamp((torch.tensor(x)-torch.tensor(data['mean']))/torch.tensor(data['scale']),-10,10)
            z=torch.relu(z@torch.tensor(data['w1']).T+torch.tensor(data['b1']))
            z=torch.relu(z@torch.tensor(data['w2']).T+torch.tensor(data['b2']))
            expected=(z@torch.tensor(data['w3']).T+torch.tensor(data['b3'])).numpy().ravel()*3000+prior
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'model.npz';np.savez(path,**data)
                np.testing.assert_allclose(Predictor(path).predict(x,prior),expected,rtol=3.e-5,atol=.005)

    def test_probe_kernel_matches_independent_phy(self):
        for seed in (17,39,113):
            p,root=fixture(seed=seed)
            phy=ProbePhy(p)
            try:
                actions=np.r_[root[None],greedy_candidates(p,root)]
                np.testing.assert_allclose(phy.evaluate(actions).values,
                    ReferencePhy(p).evaluate(actions),rtol=1.e-10,atol=1.e-8)
            finally:phy.close()

    def test_upper_bound_and_logical_dequeue_including_full_three_cell_edits(self):
        for seed in (17,39,113):
            p,root=fixture(seed=seed);phy=ProbePhy(p)
            try:
                base=phy.evaluate(root)
                actions=local_actions(p,root,0,[(-1,2*c,2*c+1) for c in range(p.C)])
                for spatial in (False,True):
                    before=phy.stats
                    x,upper,total,_=phy.features(base,actions,spatial)
                    if not spatial:self.assertEqual(before,phy.stats)
                    exact=phy.evaluate(actions,base).values
                    np.testing.assert_array_equal(total,exact[:,:,DEQUEUE].sum(1))
                    self.assertTrue(np.all(upper>=exact[:,:,EXPECTED]@p.weights))
                    self.assertTrue(np.isfinite(x).all())
            finally:phy.close()

    def test_spatial_features_reconstruct_exact_tb_sinr_without_bler_queries(self):
        p,root=fixture(seed=39);phy=ProbePhy(p)
        try:
            phy.prepare_bank();base=phy.evaluate(root);actions=greedy_candidates(p,root)
            before=phy.stats
            with patch.object(phy,'evaluate',side_effect=AssertionError('Candidate PHY in inputs')):
                x,_,_,_=phy.features(base,actions,True)
            self.assertEqual(before['candidate_evaluations'],phy.stats['candidate_evaluations'])
            self.assertEqual(before['mmse_misses'],phy.stats['mmse_misses'])
            exact=phy.evaluate(actions,base).values
            for i,action in enumerate(actions):
                prg=int(np.flatnonzero((action!=root).any(0))[0])
                users=[]
                def add(u):
                    if u>=0 and u not in users:users.append(int(u))
                for c in range(p.C):
                    if root[c,prg]!=action[c,prg]:add(root[c,prg]);add(action[c,prg])
                if not np.array_equal(root[:,prg]>=0,action[:,prg]>=0):
                    for c in range(p.C):add(action[c,prg])
                for slot,u in enumerate(users):
                    f=x[i,slot*12:(slot+1)*12];layers=int(exact[i,u,LAYER])
                    self.assertAlmostEqual(float(f[4])*3000,exact[i,u,DEQUEUE],places=3)
                    if layers>0:
                        gamma=np.expm1(f[8:8+layers].astype(float)*5).mean()
                        self.assertAlmostEqual(10*np.log10(max(gamma,1.e-30)),exact[i,u,4],places=4)
        finally:phy.close()

    def test_certified_search_preserves_full_trajectory_and_ties(self):
        for seed in (17,39,113):
            p,root=fixture(seed=seed)
            trace=[]
            full=solve_ranked(p,root,observer=lambda info:trace.append(info['chosen_index']))
            actual=certified_search(p,root)
            np.testing.assert_array_equal(actual[0],full[0]);np.testing.assert_array_equal(actual[1],full[1])
            self.assertEqual(actual[2]['rounds'],full[2]['rounds'])
            self.assertEqual([h['best_index'] for h in actual[2]['history']],trace)
            self.assertLessEqual(actual[2]['candidate_evaluations'],full[2]['candidate_evaluations'])

    def test_bounded_block_search_covers_all_prgs_before_deepening(self):
        p,root=fixture()
        for mode in ('random','heuristic'):
            action,values,r=block_search(p,root,mode,budget=4096)
            self.assertEqual(r['covered_prgs'],p.P)
            self.assertLessEqual(r['candidate_evaluations'],4096)
            self.assertGreaterEqual(values[:,DEQUEUE].sum(),r['floor'])
            np.testing.assert_allclose(values,ReferencePhy(p).evaluate(action)[0],rtol=1.e-10,atol=1.e-8)

    def test_scalar_certificate_needs_no_spatial_bank_and_preserves_actions(self):
        class Prior:
            def predict(self,x,prior):
                if x.shape[1]!=48:raise AssertionError('Wrong scalar inputs')
                return prior
        p,root=fixture(seed=113)
        with patch.object(ProbePhy,'prepare_bank',side_effect=AssertionError('Scalar bank work')):
            a,v,_=certified_search(p,root,'scalar',Prior())
        full=solve_ranked(p,root)
        np.testing.assert_array_equal(a,full[0]);np.testing.assert_array_equal(v,full[1])

    def test_sparse_score_layout_preserves_full_batch_ties(self):
        rng=np.random.default_rng(48)
        values=rng.normal(0,3000,(1712,36,8))
        values[1619]=values[1234]
        weights=rng.uniform(.2,4,36)
        expected=values[:,:,EXPECTED]@weights
        buffer=np.zeros_like(values)
        for index in (1619,1234):
            buffer[index]=values[index]
            actual=buffer[:,:,EXPECTED]@weights
            self.assertEqual(actual[index],expected[index])
        self.assertEqual(actual[1234],actual[1619])

    def test_residual_generator_never_calls_search_or_teacher(self):
        p,root=fixture()
        with patch('training.stageb_lns.bounded_probes.solve_ranked',side_effect=AssertionError('Online teacher')):
            a,v,r=generate(p,root,'heuristic',width=3,depth=3)
        self.assertEqual(r['full_search_calls'],0);self.assertLessEqual(r['pool_size'],4)
        self.assertLessEqual(r['candidate_evaluations'],1+3+2*3*2)
        self.assertGreaterEqual(v[:,DEQUEUE].sum(),r['floor'])
        self.assertGreaterEqual(r['score'],r['initial_score'])
        np.testing.assert_allclose(v,ReferencePhy(p).evaluate(a)[0],rtol=1.e-10,atol=1.e-8)

    def test_stale_closed_and_mult_prg_inputs_rejected(self):
        p,root=fixture();a=ProbePhy(p);b=ProbePhy(p)
        try:
            with self.assertRaises(ValueError):a.features(b.evaluate(root),root)
            action=root.copy();action[0,0]=-1;action[0,1]=-1
            with self.assertRaises(ValueError):a.features(a.evaluate(root),action)
            a.close()
            with self.assertRaises(RuntimeError):a.prepare_bank()
        finally:a.close();b.close()


if __name__=='__main__':unittest.main()

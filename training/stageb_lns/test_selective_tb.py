"""Incremental closure, protection, selective quota and independent PHY checks."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from .selective_tb import (Incremental,ServiceGuard,SelectiveConfig,solve_selective,choose_indices)
from .tb_macro import Descriptors,candidate_pools
from .bounded_probes import ProbePhy
from .physics import COUNT,LAYER,DEQUEUE,EXPECTED,ReferencePhy
from .test_physics import fixture


class SelectiveTests(unittest.TestCase):
    def context(self,p):
        phy=ProbePhy(p);self.addCleanup(phy.close);return phy

    def test_incremental_cross_prg_counts_minimum_layers_and_dequeue(self):
        p,a=fixture();a[:,1::2]+=1;p.caps[:]=4;p.caps[0,0]=1
        phy=self.context(p);base=phy.evaluate(a);d=Incremental(p,phy)
        singles,joint=candidate_pools(p,d,base,0);actions=np.concatenate([singles,joint])
        before=phy.stats.copy();f=d.features(base,actions,0)
        self.assertEqual(before,phy.stats)
        independent=ReferencePhy(p).evaluate(actions)
        np.testing.assert_array_equal(f['counts'],independent[:,:,COUNT])
        np.testing.assert_array_equal(f['layers'],np.maximum(independent[:,:,LAYER],0))
        np.testing.assert_array_equal(f['dequeue'],independent[:,:,DEQUEUE])
        self.assertTrue(np.isfinite(f['x']).all());self.assertEqual(f['x'].shape,(len(actions),60))

    def test_incremental_proxy_matches_frozen_wide_formula(self):
        for seed in (13,27,39):
            p,a=fixture(seed=seed);a[:,1::2]+=1;phy=self.context(p);base=phy.evaluate(a)
            d=Incremental(p,phy);wide=Descriptors(p,phy,56)
            actions=np.concatenate(candidate_pools(p,d,base,1));f=d.features(base,actions,123)
            proxy,total=wide.proxies(base,actions)
            np.testing.assert_allclose(f['prior'],proxy,rtol=1.e-11,atol=1.e-7)
            np.testing.assert_allclose(f['total'],total,rtol=0,atol=1.e-8)

    def test_empty_root_and_empty_batch(self):
        p,a=fixture();phy=self.context(p);base=phy.evaluate(np.full_like(a,-1));d=Incremental(p,phy)
        f=d.features(base,np.empty((0,p.C,p.P),np.int64),0)
        self.assertEqual(f['x'].shape,(0,60))
        b=base.actions[0].copy();b[0,0]=0;f=d.features(base,b,0)
        self.assertEqual(f['counts'][0,0],1);self.assertGreater(f['dequeue'][0,0],0)

    def test_anchor_prevents_compounding_allowed_service_loss(self):
        anchor=np.zeros((4,8));anchor[:,COUNT]=1;anchor[:,EXPECTED]=100
        guard=ServiceGuard(anchor);current=anchor.copy();current[:,EXPECTED]=98
        candidate=current.copy();candidate[:,EXPECTED]*=.98
        self.assertTrue(guard.against(candidate,current)[0])
        self.assertFalse(guard.safe(candidate,current)[0])
        candidate=anchor.copy();candidate[0,COUNT]=0
        self.assertFalse(guard.safe(candidate,anchor)[0])

    def test_selective_quota_can_spend_zero_one_or_two_joint_slots(self):
        joint=np.array([False]*4+[True]*3);available=np.ones(7,bool)
        for scores,wanted in [([9,8,7,6,1,0,-1],0),([9,8,7,6,10,0,-1],1),([9,8,7,6,10,11,-1],2)]:
            chosen=choose_indices(np.array(scores,float),joint,available,'selective',4)
            self.assertEqual(joint[chosen].sum(),wanted);self.assertEqual(len(set(chosen)),4)
            self.assertIn(0,chosen);self.assertIn(1,chosen)

    def test_guarded_search_reproducible_and_independently_valid(self):
        p,a=fixture(prgs=5);observed=[];config=SelectiveConfig()
        first=solve_selective(p,a,config,observer=observed.append);again=solve_selective(p,a,config)
        np.testing.assert_array_equal(first[0],again[0]);self.assertGreater(len(observed),0)
        np.testing.assert_allclose(first[1],ReferencePhy(p).evaluate(first[0])[0],rtol=1.e-10,atol=1.e-8)
        anchor=ReferencePhy(p).evaluate(a)[0];self.assertTrue(ServiceGuard.against(first[1],anchor)[0])
        self.assertLessEqual(first[2]['candidate_evaluations'],212);self.assertLessEqual(first[2]['feature_roots'],56)
        self.assertEqual(first[2]['covered_prgs'],p.P)

    def test_observer_cannot_mutate_student_and_no_teacher_is_online(self):
        p,a=fixture()
        def corrupt(v):v['root'][:]=-1;v['anchor'][:]=0;v['selected'][:]=-1
        before=solve_selective(p,a);after=solve_selective(p,a,observer=corrupt)
        np.testing.assert_array_equal(before[0],after[0]);np.testing.assert_array_equal(before[1],after[1])
        self.assertEqual(before[2]['candidate_evaluations'],after[2]['candidate_evaluations'])

    def test_small_budget_stops_before_speculative_full_search(self):
        p,a=fixture();result=solve_selective(p,a,SelectiveConfig(max_candidates=11))
        self.assertLessEqual(result[2]['candidate_evaluations'],11)
        self.assertEqual(result[2]['full_search_calls'],0)

    def test_training_target_rejects_unsafe_high_gain(self):
        import torch
        from .selective_tb_training import loss_function
        raw=torch.zeros((2,2),requires_grad=True)
        loss=loss_function(raw,torch.zeros(2),torch.tensor([10.,.2]),torch.tensor([False,True]))
        loss.backward()
        self.assertGreater(float(raw.grad[0,0]),0)
        self.assertLess(float(raw.grad[1,0]),0)
        self.assertGreater(float(raw.grad[0,1]),0)

    def test_zero_model_and_deployed_numpy_path(self):
        import torch
        from .selective_tb_training import build_model
        from .selective_tb import Ranker
        torch.manual_seed(7);model=build_model();x=np.random.default_rng(7).normal(size=(10,60)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.npz'
            np.savez(path,mean=np.zeros(60,np.float32),scale=np.ones(60,np.float32),**{k:v.detach().numpy() for k,v in (
                ('w1',model[0].weight),('b1',model[0].bias),('w2',model[2].weight),('b2',model[2].bias),
                ('w3',model[4].weight),('b3',model[4].bias))})
            r=Ranker(path);prior=np.arange(10,dtype=float)
            np.testing.assert_array_equal(r.predict(x,prior),prior-1500)
            with torch.no_grad():model[4].weight.uniform_(-.1,.1);expected=model(torch.from_numpy(x)).numpy()
            r.w3=model[4].weight.detach().numpy()
            np.testing.assert_allclose(r.outputs(x),expected,rtol=1.e-5,atol=1.e-6)


if __name__=='__main__':unittest.main()

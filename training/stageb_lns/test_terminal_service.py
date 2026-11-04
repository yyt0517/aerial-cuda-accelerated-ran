"""Root cache ordering/isolation and terminal information/budget boundaries."""
from unittest import TestCase,main
from unittest.mock import patch
import inspect
import numpy as np
from .test_physics import fixture
from .weak_service_campaign import RecordedMemo
from .directed_reorder import DirectedConfig,solve_directed
from .terminal_cached import CachedFeatures,solve_cached
from .terminal_risk import candidate_features,select_index,RiskModel,RISK_CAP
from .joint_reorder import JointFeatures
from .resource_conditions import Conditions,ConditionalPhy,logical,satisfies
from .physics import ReferencePhy,EXPECTED


class TerminalTests(TestCase):
    def context(self,seed=17):
        p,a=fixture(prgs=8,seed=seed);a[:,1::2]+=1;p.caps[:]=2
        n,l=logical(p,a);c=Conditions(n[0],l[0],a>=0).validate(p)
        phy=ConditionalPhy(p);self.addCleanup(phy.close)
        return p,a,c,phy,phy.evaluate(a)

    def test_lazy_proxy_preserves_reorder_duplicates_and_new_rows(self):
        p,a,c,phy,b=self.context();bank=phy.bank();old=JointFeatures(p,bank,c);new=CachedFeatures(p,bank,c)
        actions=old.pool(b,True,17)
        for ids in ([0,1,2,2],[2,0,4,3],[4,0]):
            for actual,expected in zip(new.proxy(b,actions[ids]),old.proxy(b,actions[ids])):np.testing.assert_array_equal(actual,expected)
        self.assertEqual(new.proxy_computed,5);self.assertEqual(new.proxy_hits,5)

    def test_proxy_values_are_keyed_by_root_and_decision(self):
        p,a,c,phy,b=self.context();bank=phy.bank();new=CachedFeatures(p,bank,c);pool=new.pool(b,True,17)
        b2=phy.evaluate(pool[0]);old=JointFeatures(p,bank,c)
        new.proxy(b,pool[1:4])
        for actual,expected in zip(new.proxy(b2,pool[1:4]),old.proxy(b2,pool[1:4])):np.testing.assert_array_equal(actual,expected)
        self.assertEqual(len(new.proxies),2);self.assertFalse(CachedFeatures(p,bank,c).proxies)

    def test_pool_cache_preserves_rng_and_configuration(self):
        p,a,c,phy,b=self.context();f=CachedFeatures(p,phy.bank(),c)
        old=f.pool(b,True,17).copy();np.testing.assert_array_equal(f.pool(b,True,17),old)
        f.pool(b,False,17);f.pool(b,True,18)
        self.assertEqual(f.pool_hits,1);self.assertEqual(f.pool_misses,3)

    def test_complete_search_queries_and_values_match(self):
        for seed in (17,23):
            p,a,c,phy,b=self.context(seed);cfg=DirectedConfig(joint_candidates=False,max_candidates=97)
            with patch('training.stageb_lns.directed_reorder.ExactMemo',RecordedMemo):
                old=solve_directed(p,a,c,cfg,offline_diagnostic=True);events=RecordedMemo.latest.events.copy();rows=dict(RecordedMemo.latest.rows)
            with patch('training.stageb_lns.terminal_cached.ExactMemo',RecordedMemo):
                new=solve_cached(p,a,c,cfg,offline_diagnostic=True);self.assertEqual(events,RecordedMemo.latest.events)
                for k,v in rows.items():np.testing.assert_array_equal(v,RecordedMemo.latest.rows[k])
            for i in (0,1,3):np.testing.assert_array_equal(old[i],new[i])
            for k in ('mmse_requests','mmse_misses','candidate_evaluations','iterations'):self.assertEqual(old[2][k],new[2][k])
            self.assertLessEqual(new[2]['proxy_gamma_array_reads'],old[2]['proxy_gamma_array_reads'])

    def test_selector_keeps_j_cap_even_when_best_risk_is_outside(self):
        j=np.array([100.,99.,97.]);i,safe=select_index(j,np.array([.1,.8,.99]),.9)
        self.assertEqual(i,1);self.assertFalse(safe);self.assertGreaterEqual(j[i],(1-RISK_CAP)*j.max())
        i,safe=select_index(j,np.array([.95,.99,.99]),.9);self.assertEqual(i,0);self.assertTrue(safe)

    def test_features_causal_and_finite_for_zero_service(self):
        p,a,c,phy,b=self.context();values=np.repeat(b.values,3,axis=0);values[:,:,EXPECTED]=0
        x,j,h,names=candidate_features(p,c,values)
        self.assertEqual(x.shape,(3,42));self.assertTrue(np.isfinite(x).all());self.assertEqual(len(names),42)
        self.assertEqual(tuple(inspect.signature(candidate_features).parameters),('p','condition','values'))

    def test_features_candidate_order_equivariance(self):
        p,a,c,phy,b=self.context();f=JointFeatures(p,phy.bank(),c);values=phy.evaluate(f.pool(b,True,17)[:7]).values
        order=np.array([3,0,6,2,5,1,4]);old=candidate_features(p,c,values);new=candidate_features(p,c,values[order])
        for i in range(3):np.testing.assert_allclose(old[i][order],new[i],rtol=1e-14,atol=1e-14)

    def test_terminal_choices_use_only_paid_archive_and_keep_conditions(self):
        p,a,c,phy,b=self.context();cfg=DirectedConfig(joint_candidates=False,max_candidates=65)
        arrays=dict(mean=np.zeros(42),scale=np.ones(42),calibration=np.array([1.,0.]))
        for i,(n,m) in enumerate(((42,32),(32,16),(16,1))):arrays[f'w{i}']=np.zeros((n,m));arrays[f'b{i}']=np.zeros(m)
        for mode in ('continuous','attainable','calibrated'):
            with patch('training.stageb_lns.terminal_cached.ExactMemo',RecordedMemo):
                action,v,r,g=solve_cached(p,a,c,cfg,mode=mode,model=RiskModel(arrays),offline_diagnostic=True)
                self.assertIn(action.tobytes(),RecordedMemo.latest.rows)
            self.assertTrue(satisfies(p,action,c)[0]);self.assertLessEqual(r['selected_j_loss_fraction'],RISK_CAP+1e-12)
            np.testing.assert_allclose(v,ReferencePhy(p).evaluate(action)[0],rtol=1e-10,atol=1e-8)

    def test_model_required_and_offline_fence(self):
        p,a,c,phy,b=self.context()
        with self.assertRaises(ValueError):solve_cached(p,a,c)
        with self.assertRaises(ValueError):solve_cached(p,a,c,DirectedConfig(),offline_diagnostic=True)


if __name__=='__main__':main()

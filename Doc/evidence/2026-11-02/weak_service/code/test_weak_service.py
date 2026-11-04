"""Frontier lifecycle, continuous metric scope and complete-query parity."""
from unittest import TestCase,main
from unittest.mock import patch
import numpy as np
from .test_physics import fixture
from .joint_reorder import nondominated,frontier_roots,service_vector,solve_joint,ReorderConfig
from .weak_service import FrontierCache,ServiceMetric,ServiceConfig,solve_service
from .resource_conditions import Conditions,logical,satisfies
from .physics import EXPECTED,DEQUEUE,ReferencePhy
from .tb_macro import ExactMemo


class TraceMemo(ExactMemo):
    latest=None
    def __init__(self,*args):
        super().__init__(*args);self.events=[self.root.actions.tobytes()];TraceMemo.latest=self
    def evaluate(self,actions,base):
        result=super().evaluate(actions,base)
        self.events.extend([base.actions.tobytes(),result[0].actions.tobytes() if result[0] is not None else b''])
        return result


class WeakServiceTests(TestCase):
    def context(self,seed=17):
        p,a=fixture(prgs=8,seed=seed);a[:,1::2]+=1;p.caps[:]=2
        n,l=logical(p,a);condition=Conditions(n[0],l[0],a>=0).validate(p)
        return p,a,condition

    def test_incremental_front_matches_reference_for_ties_and_reordering(self):
        rng=np.random.default_rng(19);obj={i:rng.integers(0,8,3).astype(float) for i in range(80)}
        cache=FrontierCache(80)
        for n in range(8,81,8):
            cache.add(list(range(n-8,n)),obj)
            for _ in range(3):
                keys=list(rng.permutation(n))[:max(1,n-5)]
                self.assertEqual(cache.front(keys),nondominated(keys,obj))

    def test_exhausted_dominator_restores_previous_dominated_candidate(self):
        obj={'a':np.array([3,3,3]),'b':np.array([2,2,2]),'c':np.array([2,2,2])};cache=FrontierCache(3)
        cache.add(list(obj),obj)
        self.assertEqual(cache.front(['a','b','c']),['a'])
        self.assertEqual(cache.front(['c','b']),['c'])

    def test_cached_near_roots_equal_reference(self):
        rng=np.random.default_rng(21);obj={i:rng.integers(0,10,3).astype(float) for i in range(60)}
        cache=FrontierCache(60);cache.add(list(obj),obj)
        for _ in range(20):
            keys=list(rng.permutation(60))[:40]
            for width in (1,2,4):
                roots,peak=cache.roots(keys,obj,width)
                self.assertEqual(roots,frontier_roots(keys,obj,width));self.assertEqual(peak,len(nondominated(keys,obj)))

    def test_batched_legacy_metrics_are_bitwise_identical(self):
        p,a,c=self.context();rng=np.random.default_rng(20);group=np.array([True,False,True,True,False,True])
        values=rng.uniform(0,3000,(32,p.U,8));metric=ServiceMetric(p,c,group,ServiceConfig())
        expected=np.stack([service_vector(v,group) for v in values])
        np.testing.assert_array_equal(metric.expected(values[:,:,EXPECTED]),expected)

    def test_quota_group_includes_zero_expected_positive_dequeue(self):
        p,a,c=self.context();p.queue[0]=0
        metric=ServiceMetric(p,c,np.zeros(p.U,bool),ServiceConfig(metric='quota_log'))
        self.assertFalse(metric.group[0]);self.assertTrue(metric.group[1:].all())
        self.assertTrue(np.isfinite(metric.expected(np.zeros((2,p.U)))).all())

    def test_normalized_metrics_are_invariant_to_ue_byte_scale(self):
        p,a,c=self.context();expected=np.arange(1,p.U+1)*17.
        for mode in ('quota_log','quota_exp'):
            m=ServiceMetric(p,c,np.zeros(p.U,bool),ServiceConfig(metric=mode));original=m.expected(expected)
            scale=np.arange(1,p.U+1)*2;p.queue*=scale;p.tbs=p.tbs*scale[:,None,None]
            n=ServiceMetric(p,c,np.zeros(p.U,bool),ServiceConfig(metric=mode))
            np.testing.assert_allclose(n.expected(expected*scale),original,rtol=1e-14)
            p.queue/=scale;p.tbs=p.tbs/scale[:,None,None]

    def test_continuous_metric_responds_below_one_byte_without_membership_jump(self):
        p,a,c=self.context()
        for mode in ('quota_log','quota_exp'):
            metric=ServiceMetric(p,c,np.zeros(p.U,bool),ServiceConfig(metric=mode))
            e=np.zeros(p.U);before=metric.expected(e);e[0]=.25;after=metric.expected(e)
            self.assertTrue(np.all(after>before));self.assertTrue(metric.group.all())

    def test_continuous_metric_rewards_balancing_at_equal_mean(self):
        p,a,c=self.context()
        for mode in ('quota_log','quota_exp'):
            metric=ServiceMetric(p,c,np.zeros(p.U,bool),ServiceConfig(metric=mode))
            unbalanced=np.array([0,1,0,1,0,1])*metric.dequeue
            balanced=.5*metric.dequeue
            self.assertGreater(metric.expected(balanced)[0],metric.expected(unbalanced)[0])
            self.assertAlmostEqual(metric.expected(balanced)[1],metric.expected(unbalanced)[1])

    def test_fast_legacy_full_query_sequence_actions_values_work_equal(self):
        p,a,c=self.context()
        with patch('training.stageb_lns.joint_reorder.ExactMemo',TraceMemo):
            old=solve_joint(p,a,c,ReorderConfig(True,True,0,73),offline_diagnostic=True)
            trace_old=TraceMemo.latest.events.copy()
        with patch('training.stageb_lns.weak_service.ExactMemo',TraceMemo):
            new=solve_service(p,a,c,ServiceConfig(max_candidates=73),offline_diagnostic=True)
            trace_new=TraceMemo.latest.events.copy()
        self.assertEqual(trace_old,trace_new)
        np.testing.assert_array_equal(old[0],new[0]);np.testing.assert_array_equal(old[1],new[1]);np.testing.assert_array_equal(old[3],new[3])
        for field in ('candidate_evaluations','ue_evaluations','mmse_requests','mmse_misses','proxy_candidates','proxy_gamma_array_reads','feature_roots','frontier_peak'):
            self.assertEqual(old[2][field],new[2][field])

    def test_new_metrics_keep_budget_floor_conditions_and_physics(self):
        p,a,c=self.context()
        for mode in ('quota_p5','quota_log','quota_exp'):
            action,v,r,g=solve_service(p,a,c,ServiceConfig(metric=mode,max_candidates=65),offline_diagnostic=True)
            self.assertLessEqual(r['candidate_evaluations'],65);self.assertLessEqual(r['feature_roots'],56)
            self.assertGreaterEqual(v[:,DEQUEUE].sum(),r['floor']);self.assertTrue(satisfies(p,action,c)[0])
            np.testing.assert_allclose(v,ReferencePhy(p).evaluate(action)[0],rtol=1e-10,atol=1e-8)
            self.assertEqual(r['score'],r['best_observed_j'])

    def test_empty_group_and_capacity_failure_are_explicit(self):
        p,a,c=self.context();p.queue[:]=0
        for mode in ('bootstrap_p5','quota_p5','quota_log','quota_exp'):
            metric=ServiceMetric(p,c,np.zeros(p.U,bool),ServiceConfig(metric=mode))
            np.testing.assert_array_equal(metric.expected(np.zeros((3,p.U))),np.zeros((3,2)))
        with self.assertRaises(ValueError):FrontierCache(1).add([1,2],{1:np.zeros(3),2:np.ones(3)})

    def test_config_and_offline_fences(self):
        with self.assertRaises(ValueError):ServiceConfig(metric='bad')
        with self.assertRaises(ValueError):ServiceConfig(log_epsilon=0)
        p,a,c=self.context()
        with self.assertRaises(ValueError):solve_service(p,a,c)


if __name__=='__main__':main()

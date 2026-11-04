"""Per-UE invariances, pair labels, confidence rejection and paid-search parity."""
from unittest import TestCase,main
from unittest.mock import patch
import inspect
import numpy as np
from . import test_terminal_service as terminal_tests
from .ue_pair_risk import ue_features,service_scores,heuristic_scores,rank_select,UEPairModel
from .ue_pair_train import make_pairs
from .ue_pair_solver import solve_ue_pair
from .terminal_cached import solve_cached
from .terminal_risk import RiskModel
from .directed_reorder import DirectedConfig
from .joint_reorder import JointFeatures
from .weak_service_campaign import RecordedMemo
from .physics import EXPECTED,ReferencePhy


class UEPairTests(TestCase):
    def context(self):return terminal_tests.TerminalTests.context(self)

    def models(self):
        arrays=dict(alpha=np.array(20.),heuristic_alpha=np.array(20.))
        for i in range(3):
            arrays.update({f'mean{i}':np.zeros(20),f'scale{i}':np.ones(20),f'w{i}_0':np.zeros((20,24)),
                f'b{i}_0':np.zeros(24),f'w{i}_1':np.zeros((24,10)),f'b{i}_1':np.zeros(10)})
        old=dict(mean=np.zeros(42),scale=np.ones(42),calibration=np.array([1.,0.]))
        for i,(a,b) in enumerate(((42,32),(32,16),(16,1))):old[f'w{i}']=np.zeros((a,b));old[f'b{i}']=np.zeros(b)
        return dict(pair=UEPairModel(arrays),aggregate=RiskModel(old))

    def test_low_confidence_returns_exact_maxj(self):
        i,r=rank_select(np.array([100.,99.]),np.array([[0.,.001]]),20.)
        self.assertEqual(i,0);self.assertTrue(r['fallback_to_maxj'])

    def test_one_disagreeing_member_blocks_replacement(self):
        scores=np.array([[0.,1.],[0.,1.],[0.,-.1]])
        i,r=rank_select(np.array([100.,99.]),scores,20.)
        self.assertEqual(i,0);self.assertLess(r['pair_confidence'],.5)

    def test_confident_change_respects_risk_cap(self):
        i,r=rank_select(np.array([100.,99.,97.]),np.array([[0.,1.,5.]]),20.)
        self.assertEqual(i,1);self.assertFalse(r['fallback_to_maxj'])

    def test_unguarded_ablation_changes_same_proposal(self):
        i,r=rank_select(np.array([100.,99.]),np.array([[0.,.001]]),20.,False)
        self.assertEqual(i,1);self.assertEqual(r['terminal_reason'],'unguarded')

    def test_pair_orientation_ties_and_exclusion(self):
        j=np.array([100.,99.,98.5,80.]);risk=np.array([1.,0.,0.,0.])
        pp=make_pairs(j,risk,17)
        self.assertGreater(len(pp),0)
        for a,b,t in pp:
            self.assertNotIn(3,(a,b));a=int(a);b=int(b)
            if t==1:self.assertLess(risk[a],risk[b])
            else:self.assertEqual(risk[a],risk[b])
        np.testing.assert_array_equal(pp,make_pairs(j,risk,17))

    def test_features_have_no_teacher_arguments_and_stay_finite(self):
        p,a,c,phy,b=self.context();vv=np.repeat(b.values,3,axis=0);vv[:,:,EXPECTED]=0
        f=ue_features(p,c,vv)
        self.assertEqual(f['context'].shape,(p.U,20));self.assertEqual(f['losses'].shape,(3,p.U,7))
        self.assertTrue(np.isfinite(heuristic_scores(f)).all());self.assertEqual(tuple(inspect.signature(ue_features).parameters),('p','condition','values'))

    def test_shared_ranker_equivariant_to_ue_permutation(self):
        p,a,c,phy,b=self.context();pool=JointFeatures(p,phy.bank(),c).pool(b,True,17)[:5]
        f=ue_features(p,c,phy.evaluate(pool).values);rng=np.random.default_rng(17);parameters=rng.normal(size=(p.U,10));order=rng.permutation(p.U)
        new=dict(f,context=f['context'][order],losses=f['losses'][:,order],r=f['r'][:,order],group=f['group'][order])
        np.testing.assert_allclose(service_scores(f,parameters),service_scores(new,parameters[order]),rtol=1e-14,atol=1e-14)

    def test_fixed_context_service_increase_cannot_worsen_learned_score(self):
        p,a,c,phy,b=self.context();vv=np.repeat(b.values,2,axis=0);vv[:,:,EXPECTED]*=.5
        f=ue_features(p,c,vv);f['r'][1]=np.minimum(f['r'][1]+.1,1)
        # Corresponding bases all decrease; context stays fixed.
        f['losses'][1]=np.maximum(f['losses'][0]-.01,0)
        params=np.random.default_rng(8).normal(size=(p.U,10));scores=service_scores(f,params)
        self.assertGreaterEqual(scores[1],scores[0])

    def test_all_selectors_preserve_full_paid_query_sequence(self):
        p,a,c,phy,b=self.context();cfg=DirectedConfig(joint_candidates=False,max_candidates=65)
        with patch('training.stageb_lns.terminal_cached.ExactMemo',RecordedMemo):
            old=solve_cached(p,a,c,cfg,offline_diagnostic=True);events=RecordedMemo.latest.events.copy();rows=dict(RecordedMemo.latest.rows)
        for mode in ('maxj','aggregate_guard','ue_heuristic_guard','ue_pair','ue_pair_guard'):
            with patch('training.stageb_lns.ue_pair_solver.ExactMemo',RecordedMemo):
                new=solve_ue_pair(p,a,c,cfg,mode=mode,model=self.models(),offline_diagnostic=True)
                self.assertEqual(events,RecordedMemo.latest.events)
                for k,v in rows.items():np.testing.assert_array_equal(v,RecordedMemo.latest.rows[k])
                self.assertIn(new[0].tobytes(),rows)
            if mode in ('maxj','aggregate_guard'):np.testing.assert_array_equal(old[0],new[0])
            np.testing.assert_allclose(new[1],ReferencePhy(p).evaluate(new[0])[0],rtol=1e-10,atol=1e-8)


if __name__=='__main__':main()

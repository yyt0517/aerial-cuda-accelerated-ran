"""CTBS-free input integrity, raw-PFQ retention, and fresh PHY reconstruction."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np

from .no_ctbs_inputs import (
    detach_raw_phy_row, problem_from_raw_row, raw_pfq_draft, raw_pfq_floor,
)
from .physics import NativePhy, DEQUEUE
from .test_physics import fixture


def row_fixture(cells=2, prgs=3):
    p, action = fixture(cells=cells, prgs=prgs)
    config = dict(mcs_lut_type=0,mcs_sinr_cap_db=25.989999771118164,
        layer_sinval_threshold=.10000000149011612,prbs_per_prg=16,
        pdsch_data_symbols=11,effective_layer_cap=4)
    # Tiny queues intentionally leave many PFQ PRGs beyond useful capacity.
    queue = np.ones(p.U, dtype=float)
    raw = dict(tti=19,snapshot_id=20,config=config,
        serving_cell=p.serving.copy(),prg_layer_cap=p.caps.copy(),mcs=p.mcs.copy())
    slots=np.arange(p.U)
    obs = dict(obs_ue_features=queue[:,None].copy(),
        action_mask_ue=np.zeros(p.U,bool),
        action_mask_cell_ue=np.zeros((cells,p.U),bool),
        action_mask_prg_cell=np.ones((cells,prgs),bool),
        teacher_action_ue_select=slots,
        teacher_action_prg_alloc=action.T.reshape(-1))
    row = dict(tti=19,seed=41,family='raw',source='synthetic',raw_phy=raw,obs=obs,
        snapshot=dict(tti=19,h=p.h.copy(),w=p.w.copy(),noise=p.noise,queue=queue),
        ema=np.linspace(600,4800,p.U))
    return row, action


class NoCtbsInputsTests(unittest.TestCase):
    def test_entire_timed_path_ignores_ctbs_and_demand_masks(self):
        row, action = row_fixture()
        class ProtectedRow(dict):
            def __getitem__(self,key):
                if key in ('ctbs','chosen','native_info','prediction'):
                    raise AssertionError('Forbidden CTBS or post-action dependency')
                return super().__getitem__(key)
        row=ProtectedRow(row)
        row['ctbs']=object()
        with (patch('training.stageb_lns.archive.initial_draft', side_effect=AssertionError('CTBS prune called')),
              patch('training.stageb_lns.archive.problem_from_row', side_effect=AssertionError('Old CTBS adapter called'))):
            p=problem_from_raw_row(row)
            actual=raw_pfq_draft(p,row['obs'])
        np.testing.assert_array_equal(actual,action)
        for u in range(p.U):
            self.assertTrue(p.legal[p.serving[u],:,u].all())
        expected=np.clip(2400./row['ema'],1.,4.)
        np.testing.assert_array_equal(p.weights,expected/expected.mean())

    def test_raw_pfq_retains_prgs_after_queue_saturation(self):
        row, action = row_fixture()
        p=problem_from_raw_row(row)
        raw=raw_pfq_draft(p,row['obs'])
        self.assertEqual((raw>=0).sum(),p.C*p.P)
        np.testing.assert_array_equal(raw,action)
        phy=NativePhy(p)
        try:
            values=phy.evaluate(raw).values[0]
            self.assertTrue((values[:,DEQUEUE] <= p.queue).all())
            total=int(values[:,DEQUEUE].sum())
            self.assertEqual(raw_pfq_floor(values),(90*total+99)//100)
        finally:phy.close()

    def test_physical_prg_availability_is_preserved_without_ue_demand_clipping(self):
        row,_=row_fixture()
        row['obs']['action_mask_prg_cell'][0,1]=False
        row['snapshot']['queue'][1]=0
        row['obs']['obs_ue_features'][1,0]=0
        p=problem_from_raw_row(row)
        self.assertFalse(p.legal[0,1].any())
        self.assertFalse(p.legal[:,:,1].any())
        self.assertTrue(p.legal[0,0,0])
        self.assertFalse(p.legal[1,:,0].any())

    def test_raw_pfq_validates_native_slot_semantics(self):
        row,_=row_fixture()
        p=problem_from_raw_row(row)
        for bad in (-2,p.U):
            obs=dict(row['obs'])
            obs['teacher_action_prg_alloc']=row['obs']['teacher_action_prg_alloc'].copy()
            obs['teacher_action_prg_alloc'][0]=bad
            with self.assertRaises(ValueError):raw_pfq_draft(p,obs)
        obs=dict(row['obs']); obs['teacher_action_ue_select']=np.full(p.U,-1)
        with self.assertRaises(ValueError):raw_pfq_draft(p,obs)

    def test_stale_phy_or_queue_observation_is_rejected(self):
        for field in ('row_tti','raw_tti','queue'):
            row,_=row_fixture()
            if field=='row_tti':row['tti']+=2
            elif field=='raw_tti':row['raw_phy']['tti']+=1
            else:row['obs']['obs_ue_features'][0,0]+=1
            with self.assertRaises(ValueError):problem_from_raw_row(row)

    def test_upcoming_decision_clock_requires_matching_snapshot_identity(self):
        row,_=row_fixture()
        row['tti']=20
        problem_from_raw_row(row)
        row['raw_phy']['snapshot_id']=21
        with self.assertRaises(ValueError):problem_from_raw_row(row)

    def test_metadata_detach_never_invokes_headroom_or_mcs_methods(self):
        row,_=row_fixture()
        raw=row.pop('raw_phy')
        carrier=SimpleNamespace(tti=raw['tti'],snapshot_id=raw['snapshot_id'],
            config=SimpleNamespace(**raw['config']),serving_cell=raw['serving_cell'],
            prg_layer_cap=raw['prg_layer_cap'],
            wb_sinr_first_layer_db=np.full(4,10.),olla_delta=np.full(4,1.),
            olla_delta_up=np.full(4,1.),olla_delta_down=np.full(4,.1),
            prior_tb_err=np.array([-1,0,1,-1]),
            reconstructed_mcs_for_ue=lambda u: (_ for _ in ()).throw(AssertionError('CTBS method called')))
        row['ctbs']=carrier
        detached=detach_raw_phy_row(row)
        self.assertNotIn('ctbs',detached)
        for name in ('seed','family','tti','source'):self.assertEqual(detached[name],row[name])
        p=problem_from_raw_row(detached)
        # MCS at effective SINR 9.0 / 9.1 / 8.0 / 9.0 under native LUT0.
        np.testing.assert_array_equal(p.mcs,[9,9,8,9])
        self.assertIsNone(p.deadline)

    def test_raw_metadata_preferred_and_invalid_ema_or_native_mcs_rejected(self):
        row,_=row_fixture()
        row['ctbs']=object()
        detached=detach_raw_phy_row(row)
        self.assertNotIn('ctbs',detached)
        np.testing.assert_array_equal(problem_from_raw_row(detached).mcs,row['raw_phy']['mcs'])
        for name,value in (('ema',np.array([np.nan]*4)),('mcs',np.array([1.5]*4))):
            changed,_=row_fixture()
            if name=='ema':changed['ema']=value
            else:changed['raw_phy']['mcs']=value
            with self.assertRaises(ValueError):problem_from_raw_row(changed)


if __name__=='__main__':unittest.main()

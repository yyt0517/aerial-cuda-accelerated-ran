"""Exact trajectory, global-floor and native-input checks for the separate library."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
from .test_physics import fixture
from .persistent_dpls import solve_integer_reference
from .bounded_dpls import solve_bounded
from .bounded_campaign import audit_round
from .tournament_dpls import solve_tournament,TournamentConfig
from .native_schedule_inputs import prepare_native_problem
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .analyze_bounded_stress import local

def clone(i):
    return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in i.items()}

def same_inputs(row):
    old=problem_from_raw_row(row);new,draft=prepare_native_problem(row)
    for key in ('h','w','noise','queue','serving','caps','mcs','tbs','legal','weights'):
        np.testing.assert_array_equal(getattr(new,key),getattr(old,key),err_msg=key)
    np.testing.assert_array_equal(draft,raw_pfq_draft(old,row['obs']))
    return new,draft

class TournamentTests(unittest.TestCase):
    def test_full_oracle_and_frozen_heap_match_including_floor_and_direct_cache(self):
        for seed,cells,prgs in ((17,3,5),(39,3,7),(113,3,5),(19,7,3)):
            p,a=fixture(cells=cells,prgs=prgs,seed=seed)
            p.weights[:]=np.random.default_rng(seed).uniform(.2,3,p.U)
            trace=[];ref=solve_integer_reference(p,a,observer=lambda i:trace.append(clone(i)))
            original=solve_bounded(p,a)
            for fast in (False,True):
                rounds=[]
                def audit(i):rounds.append(audit_round(i,trace[i['iteration']-1]))
                actual=solve_tournament(p,a,TournamentConfig(direct_gamma=fast),observer=audit)
                np.testing.assert_array_equal(actual[0],ref[0]);np.testing.assert_array_equal(actual[1],ref[1])
                self.assertEqual(len(rounds),original[2]['rounds'])
                for key in ('score_q','floor','chosen_dequeue','candidate_evaluations','ue_evaluations','mmse_requests','mmse_misses'):
                    self.assertEqual(actual[2][key],original[2][key],key)

    def test_causal_saved_native_adapter_and_mcs_boundaries(self):
        import torch
        from .archive import enable_legacy_types
        enable_legacy_types()
        source=Path('deliverables/bounded_load_stress_20261104_v1')
        runs=json.loads((source/'completion.json').read_text())['runs']
        for run in runs:
            d=local(run['directory']);item=json.loads((d/'saved_states.json').read_text())[0]
            row=torch.load(local(item['path']),map_location='cpu',weights_only=False)
            p,a=same_inputs(row)
            old=solve_bounded(p,a,warm_start=a);new=solve_tournament(p,a,TournamentConfig(direct_gamma=True),warm_start=a)
            np.testing.assert_array_equal(old[0],new[0]);np.testing.assert_array_equal(old[1],new[1])
        # Both LUTs, float32 threshold neighbors, and all three OLLA outcomes.
        from .native_schedule_inputs import mcs_lut
        for kind in (0,1):
            for direction in (-np.inf,np.inf):
                r=copy.deepcopy(row);r['raw_phy']['config']['mcs_lut_type']=kind
                lut=mcs_lut(kind).astype(np.float32);n=len(r['ema'])
                r['raw_phy']['wb_sinr_first_layer_db']=np.resize(np.nextafter(lut,np.float32(direction)),n)
                for err in (-1,0,1):
                    r['raw_phy']['prior_tb_err']=np.full(n,err)
                    same_inputs(r)
        r=copy.deepcopy(row);r['raw_phy']['tti']-=1
        with self.assertRaises(ValueError):prepare_native_problem(r)
        r=copy.deepcopy(row);r['obs']['teacher_action_prg_alloc'][0]=30000
        with self.assertRaises(ValueError):prepare_native_problem(r)

if __name__=='__main__':unittest.main()

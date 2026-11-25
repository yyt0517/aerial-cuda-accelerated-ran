"""Meaningful checks for forced initialization and same-state three-branch selection."""
import unittest
from pathlib import Path
import numpy as np
import torch
from training.stageb_lns.no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from training.stageb_lns.crossload_native import weights_for
from training.stageb_lns.physics import DEQUEUE
from .policy import select,TIE_ORDER
from .seeded_solver import solve_bounded as solve_seeded
from .audit import audit_branch

ROOT=Path(__file__).resolve().parents[2]/'deliverables/seven_cell_boundary08_20261105_v1'

class PolicyTests(unittest.TestCase):
    def test_same_state_branches_and_old_actions(self):
        for source in ('deficit','mlwdf_pf'):
            for seed in (41,42,43):
                row=torch.load(ROOT/f'campaign/{source}_s{seed}/states/tti008001.pt',map_location='cpu',weights_only=False)
                p=problem_from_raw_row(row);da=weights_for('mlwdf_pf',p.weights,row['ema'],row['hol_urgency'],p.queue,0.)
                a,v,r,b=select(p,raw_pfq_draft(p,row['obs']),da)
                branch='dpls' if source=='deficit' else 'original_da'
                np.testing.assert_array_equal(b[branch]['action'],row['selected_action'])
                np.testing.assert_array_equal(b[branch]['values'],row['selected_values'])
                self.assertEqual(r['score_q'],max(x['da_score_q'] for x in b.values()))
                self.assertEqual(r['selected_branch'],max(TIE_ORDER,key=lambda n:b[n]['da_score_q']))

    def test_forced_lower_scoring_seed_and_original_floor(self):
        row=torch.load(ROOT/'campaign/deficit_s41/states/tti008001.pt',map_location='cpu',weights_only=False)
        p=problem_from_raw_row(row);da=weights_for('mlwdf_pf',p.weights,row['ema'],row['hol_urgency'],p.queue,0.)
        _,_,_,b=select(p,raw_pfq_draft(p,row['obs']),da);p.weights=da
        raw=b['original_da']['action'];initial=b['dpls']['action'];self.assertLess(b['dpls']['da_score_q'],b['original_da']['da_score_q'])
        roots=[]
        def observe(x):
            if x['iteration']==1:roots.append((x['root'].copy(),x['floor']))
        a,v,r=solve_seeded(p,raw,warm_start=initial,observer=observe)
        np.testing.assert_array_equal(roots[0][0],initial)
        self.assertEqual(roots[0][1],(90*int(b['original_da']['values'][:,DEQUEUE].sum())+99)//100)
        x,y,_=audit_branch(p,raw,initial)
        np.testing.assert_array_equal(a,x);np.testing.assert_array_equal(v,y)

if __name__=='__main__':unittest.main()

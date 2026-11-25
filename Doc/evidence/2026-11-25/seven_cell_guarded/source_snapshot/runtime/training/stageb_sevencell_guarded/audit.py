"""Full-neighborhood oracle, objective reconstruction and final guard audit."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from training.stageb_ppo.native import write
from training.stageb_lns.no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from training.stageb_lns.persistent_dpls import q_contributions
from training.stageb_lns.physics import EXPECTED
from training.stageb_sevencell_multistart.audit import audit_branch
from .policy import make_weights,TIE_ORDER,BETA

def main(source,output):
    checks=[]
    for run in sorted(Path(source).glob('*_s*')):
        if not (run/'done.json').exists():continue
        with np.load(run/'actions.npz') as log:
            for path in sorted((run/'states').glob('*.pt')):
                row=torch.load(path,map_location='cpu',weights_only=False);p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs'])
                base=p.weights.copy();w=make_weights(base,row['hol_urgency']);scores={};service={}
                np.testing.assert_array_equal(base,row['deficit_weights']);np.testing.assert_array_equal(w,row['objective_weights'])
                for name in ('dpls','new_cold','new_from_dpls'):
                    p.weights=base if name=='dpls' else w
                    initial=row['branches']['dpls']['action'] if name=='new_from_dpls' else None
                    a,v,r=audit_branch(p,draft,initial)
                    np.testing.assert_array_equal(a,row['branches'][name]['action']);np.testing.assert_array_equal(v,row['branches'][name]['values'])
                    scores[name]=int(q_contributions(v,w).sum(dtype=np.int64));service[name]=float(v[:,EXPECTED].sum())
                    assert scores[name]==row['branches'][name]['score_q']
                    checks.append(dict(run=run.name,tti=int(row['tti']),branch=name,**r));print('AUDIT',run.name,row['tti'],name,r,flush=True)
                eligible={n:service[n]>=BETA*service['dpls'] for n in TIE_ORDER}
                for n in TIE_ORDER:assert eligible[n]==row['branches'][n]['eligible']
                winner=max((n for n in TIE_ORDER if eligible[n]),key=lambda n:scores[n])
                assert winner==row['selected_branch'] and scores[winner]>=scores['dpls']
                np.testing.assert_array_equal(row['selected_action'],row['branches'][winner]['action'])
                np.testing.assert_array_equal(row['selected_action'],log['actions'][int(row['tti'])-1])
    assert checks
    write(output,dict(passed=True,states=len(checks)//3,branch_searches=len(checks),checks=checks,
        selected_actions_match_native_execution=True,weights_reconstructed=True,
        final_guard_and_max_eligible_score_exact=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)

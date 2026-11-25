"""Independent complete-neighborhood audit including a forced seed."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from training.stageb_ppo.native import write
from training.stageb_lns.physics import NativePhy,ReferencePhy,DEQUEUE
from training.stageb_lns.persistent_dpls import q_contributions,SCALE
from training.stageb_lns.dpls_operations import compact_neighborhood,expand_edits,canonical_choice
from training.stageb_lns.bounded_campaign import audit_round
from training.stageb_lns.no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from training.stageb_sevencell.solver import solve_bounded
from .seeded_solver import solve_bounded as solve_seeded
from .policy import TIE_ORDER

def full(p,draft,observer,initial=None):
    phy=NativePhy(p)
    try:
        root=phy.evaluate(draft);floor=(90*int(root.values[0,:,DEQUEUE].sum())+99)//100
        if initial is not None:root=phy.evaluate(initial)
        assert root.values[0,:,DEQUEUE].sum()>=floor
        score=int(q_contributions(root.values[0],p.weights).sum(dtype=np.int64))
        for iteration in range(1,257):
            prg,assignment=compact_neighborhood(root.actions[0],p.legal)
            targets=np.where(assignment!=root.actions[0][:,prg].T,assignment,-2);keys=prg.copy()
            for c in range(p.C):keys=keys*(p.U+3)+targets[:,c]+2
            result=phy.evaluate(expand_edits(root.actions[0],prg,assignment),root)
            scores=q_contributions(result.values,p.weights).sum(1,dtype=np.int64);totals=result.values[:,:,DEQUEUE].sum(1).astype(np.int64)
            chosen=canonical_choice(scores-score,totals>=floor,keys,(1 if iteration<=2 else 16)*SCALE,1)
            observer(dict(iteration=iteration,root=root.actions[0].copy(),root_values=root.values[0].copy(),keys=keys,
                score_q=score,delta_q=scores-score,delta_d=totals-int(root.values[0,:,DEQUEUE].sum()),floor=floor,
                selected_keys=keys[chosen:chosen+1] if chosen>=0 else np.empty(0,np.int64)))
            if chosen>=0:root=phy.take(result,chosen);score=int(scores[chosen])
            if iteration>=3 and chosen<0:return root.actions[0].copy(),root.values[0].copy()
        raise AssertionError('Full oracle round cap')
    finally:phy.close()

def audit_branch(p,draft,initial=None):
    trace=[];a,v=full(p,draft,trace.append,initial);checks=[]
    def observe(i):checks.append(audit_round(i,trace[i['iteration']-1]))
    if initial is None:b,w,r=solve_bounded(p,draft,observer=observe)
    else:b,w,r=solve_seeded(p,draft,warm_start=initial,observer=observe)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(v,w)
    np.testing.assert_allclose(w,ReferencePhy(p).evaluate(b)[0],rtol=1e-8,atol=1e-6)
    assert r['converged'] and r['rounds']==len(trace)
    return a,v,dict(passed=True,rounds=len(trace),candidates=sum(x['candidates'] for x in checks))

def main(source,output):
    checks=[]
    for run in sorted(Path(source).glob('*_s*')):
        if not (run/'done.json').exists():continue
        with np.load(run/'actions.npz') as log:
            for path in sorted((run/'states').glob('*.pt')):
                row=torch.load(path,map_location='cpu',weights_only=False);p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs'])
                base=p.weights.copy();da=row['objective_weights'];scores={}
                for name in ('dpls','original_da','from_dpls_da'):
                    p.weights=base if name=='dpls' else da
                    initial=row['branches']['dpls']['action'] if name=='from_dpls_da' else None
                    a,v,r=audit_branch(p,draft,initial)
                    np.testing.assert_array_equal(a,row['branches'][name]['action']);np.testing.assert_array_equal(v,row['branches'][name]['values'])
                    scores[name]=int(q_contributions(v,da).sum(dtype=np.int64))
                    assert scores[name]==row['branches'][name]['da_score_q']
                    checks.append(dict(run=run.name,tti=int(row['tti']),branch=name,**r));print('AUDIT',run.name,row['tti'],name,r,flush=True)
                winner=max(TIE_ORDER,key=lambda n:scores[n]);assert winner==row['selected_branch']
                np.testing.assert_array_equal(row['selected_action'],row['branches'][winner]['action'])
                np.testing.assert_array_equal(row['selected_action'],log['actions'][int(row['tti'])-1])
    assert checks
    write(output,dict(passed=True,states=len(checks)//3,branch_searches=len(checks),checks=checks,
        selected_actions_match_native_execution=True,all_three_branch_scores_and_selection_exact=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)

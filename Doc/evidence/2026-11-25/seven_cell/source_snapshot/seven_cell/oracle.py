"""Independent full-array reference, without any sparse support-bit masks."""
import numpy as np
from training.stageb_lns.physics import NativePhy, ReferencePhy, EXPECTED, DEQUEUE
from training.stageb_lns.dpls_operations import compact_neighborhood,expand_edits,canonical_choice
from training.stageb_lns.persistent_dpls import q_contributions,SCALE
from training.stageb_lns.bounded_campaign import audit_round
from .contract import check_contract
from .solver import solve_bounded

def full(p,draft,observer):
    check_contract(p);phy=NativePhy(p)
    try:
        root=phy.evaluate(draft);score=int(q_contributions(root.values[0],p.weights).sum(dtype=np.int64))
        floor=(90*int(root.values[0,:,DEQUEUE].sum())+99)//100
        for iteration in range(1,257):
            prg,assignment=compact_neighborhood(root.actions[0],p.legal)
            targets=np.where(assignment!=root.actions[0][:,prg].T,assignment,-2)
            keys=prg.copy()
            for c in range(p.C):keys=keys*(p.U+3)+targets[:,c]+2
            assert len(np.unique(keys))==len(keys)
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

def audit(p,draft):
    trace=[];a,v=full(p,draft,trace.append);checks=[]
    def observe(i):checks.append(audit_round(i,trace[i['iteration']-1]))
    b,w,r=solve_bounded(p,draft,observer=observe)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(v,w)
    np.testing.assert_allclose(w,ReferencePhy(p).evaluate(b)[0],rtol=1e-8,atol=1e-6)
    assert r['converged'] and r['rounds']==len(trace)
    return dict(passed=True,rounds=len(trace),candidates=sum(x['candidates'] for x in checks),
        full_array_actions_values_bounds_cache_choices_and_numpy_phy=True)

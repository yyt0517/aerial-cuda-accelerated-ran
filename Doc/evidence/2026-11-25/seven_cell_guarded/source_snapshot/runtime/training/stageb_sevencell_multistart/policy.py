"""All branches see one state; choose the highest canonical DA objective."""
import numpy as np
from training.stageb_lns.physics import EXPECTED, DEQUEUE
from training.stageb_lns.persistent_dpls import q_contributions
from training.stageb_sevencell.solver import solve_bounded
from .seeded_solver import solve_bounded as solve_seeded

ARM='da_three_branch'
LABEL='DA-PLS (three branches)'
# Exact outer objective ties retain the original DA branch.
TIE_ORDER=('original_da','from_dpls_da','dpls')

def select(p,raw_pfq,da_weights):
    base=p.weights.copy();da_weights=np.ascontiguousarray(da_weights,np.float64)
    branches={}
    try:
        for name,weights in [('dpls',base),('original_da',da_weights),('from_dpls_da',da_weights)]:
            p.weights=weights
            if name=='from_dpls_da':a,v,r=solve_seeded(p,raw_pfq,warm_start=branches['dpls']['action'])
            else:a,v,r=solve_bounded(p,raw_pfq)
            assert r['converged'],(name,'round cap reached')
            da_q=int(q_contributions(v,da_weights).sum(dtype=np.int64))
            branches[name]=dict(action=a,values=v,stats=r,da_score_q=da_q,
                da_score=float(v[:,EXPECTED]@da_weights),expected_bytes=float(v[:,EXPECTED].sum()))
        assert len({b['stats']['floor'] for b in branches.values()})==1
        floor=branches['dpls']['stats']['floor']
        assert all(int(b['values'][:,DEQUEUE].sum())>=floor for b in branches.values())
        assert branches['from_dpls_da']['da_score_q']>=branches['dpls']['da_score_q']
        winner=max(TIE_ORDER,key=lambda n:branches[n]['da_score_q'])
        chosen=branches[winner]
        stats=dict(converged=True,selected_branch=winner,score_q=chosen['da_score_q'],score=chosen['da_score'],
            floor=floor,chosen_dequeue=int(chosen['values'][:,DEQUEUE].sum()),
            gain_over_original_da_q=chosen['da_score_q']-branches['original_da']['da_score_q'])
        for key in ('rounds','candidate_evaluations','accepted_moves','ue_evaluations','mmse_requests','mmse_misses'):
            stats[key]=sum(b['stats'][key] for b in branches.values())
        for name,b in branches.items():
            stats.update({name+'_score_q':b['da_score_q'],name+'_expected_bytes':b['expected_bytes'],
                name+'_rounds':b['stats']['rounds'],name+'_candidate_evaluations':b['stats']['candidate_evaluations']})
        return chosen['action'],chosen['values'],stats,branches
    finally:p.weights=base

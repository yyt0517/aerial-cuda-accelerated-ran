"""Three same-state searches; new objective for both new branches and selection."""
import numpy as np
from training.stageb_lns.physics import EXPECTED,DEQUEUE
from training.stageb_lns.persistent_dpls import q_contributions
from training.stageb_lns.ttl_weights import objective_weights
from training.stageb_sevencell.solver import solve_bounded
from training.stageb_sevencell_multistart.seeded_solver import solve_bounded as solve_seeded

ARM='da_guarded_l010_b098'
LABEL='Bounded-urgency DA (lambda=0.1, beta=0.98)'
LAMBDA=.1
BETA=.98
TIE_ORDER=('new_cold','new_from_dpls','dpls')

def make_weights(base,urgency):
    urgency=np.asarray(urgency,np.float64)
    assert np.isfinite(urgency).all() and ((urgency>=0)&(urgency<=1)).all()
    result=objective_weights(np.asarray(base,np.float64),urgency,LAMBDA)
    assert result.max()/result.min()<=4*(1+LAMBDA)+1e-12
    return result

def choose(branches):
    floor=BETA*branches['dpls']['expected_bytes']
    eligible={name:branches[name]['expected_bytes']>=floor for name in TIE_ORDER}
    assert eligible['dpls']
    winner=max((name for name in TIE_ORDER if eligible[name]),key=lambda n:branches[n]['score_q'])
    return winner,floor,eligible

def select(p,raw_pfq,new_weights):
    base=p.weights.copy();new_weights=np.ascontiguousarray(new_weights,np.float64);branches={}
    try:
        for name,weights in [('dpls',base),('new_cold',new_weights),('new_from_dpls',new_weights)]:
            p.weights=weights
            if name=='new_from_dpls':a,v,r=solve_seeded(p,raw_pfq,warm_start=branches['dpls']['action'])
            else:a,v,r=solve_bounded(p,raw_pfq)
            assert r['converged'],(name,'round cap reached')
            score_q=int(q_contributions(v,new_weights).sum(dtype=np.int64))
            branches[name]=dict(action=a,values=v,stats=r,score_q=score_q,
                score=float(v[:,EXPECTED]@new_weights),expected_bytes=float(v[:,EXPECTED].sum()))
        assert len({b['stats']['floor'] for b in branches.values()})==1
        floor=branches['dpls']['stats']['floor']
        assert all(int(b['values'][:,DEQUEUE].sum())>=floor for b in branches.values())
        assert branches['new_from_dpls']['score_q']>=branches['dpls']['score_q']
        winner,service_floor,eligible=choose(branches);chosen=branches[winner]
        unconstrained=max(TIE_ORDER,key=lambda n:branches[n]['score_q'])
        assert chosen['score_q']>=branches['dpls']['score_q']
        assert chosen['expected_bytes']>=service_floor
        stats=dict(converged=True,selected_branch=winner,score_q=chosen['score_q'],score=chosen['score'],
            floor=floor,chosen_dequeue=int(chosen['values'][:,DEQUEUE].sum()),
            service_floor=service_floor,selected_expected_bytes=chosen['expected_bytes'],
            service_guard_passed=True,unconstrained_winner=unconstrained,
            guard_changed_winner=winner!=unconstrained,
            gain_over_dpls_q=chosen['score_q']-branches['dpls']['score_q'])
        for key in ('rounds','candidate_evaluations','accepted_moves','ue_evaluations','mmse_requests','mmse_misses'):
            stats[key]=sum(b['stats'][key] for b in branches.values())
        for name,b in branches.items():
            b['eligible']=eligible[name]
            stats.update({name+'_score_q':b['score_q'],name+'_expected_bytes':b['expected_bytes'],
                name+'_eligible':eligible[name],name+'_rounds':b['stats']['rounds'],
                name+'_candidate_evaluations':b['stats']['candidate_evaluations']})
        return chosen['action'],chosen['values'],stats,branches
    finally:p.weights=base

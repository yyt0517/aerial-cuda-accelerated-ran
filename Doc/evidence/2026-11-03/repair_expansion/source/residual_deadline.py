"""Optional neural-anchor refinement, without full physical-neighborhood search.

Keep the established neural proposal path intact; apply one verified corrective
scan. The same function accepts the old decoder as a compute/control ablation.
"""
from dataclasses import replace
import time
from .quality import JOINT
from .deadline import packet_objective,components
from .search import solve


def select(problem,draft,anchor,corrector):
    started=time.perf_counter()
    seed,_,base=solve(problem,draft,JOINT,repairer=anchor)
    action,values,extra=solve(problem,draft,replace(JOINT,max_rounds=1),initial_action=seed,
        repairer=corrector,objective=packet_objective(problem))
    assert extra['score']+1.e-7>=base['score']
    return action,values,dict(extra,arm='neural_residual',baseline_neural_score=base['score'],
        residual_gain=extra['score']-base['score'],core_ms=(time.perf_counter()-started)*1000,
        candidate_evaluations=base['candidate_evaluations']+extra['candidate_evaluations'],
        anchor_ms=base['core_ms'],residual_ms=extra['core_ms'])

"""Quality-first policies: no wall-clock cutoff; explicit learned contribution."""
from dataclasses import replace
import time
from .search import SearchConfig, solve


FULL = SearchConfig(mode="greedy", budget_ms=None, max_candidates=1000000,
                    max_rounds=256, batch_size=256)
JOINT = replace(FULL, mode="graph_lns", max_rounds=3, batch_size=32)


def select(problem, draft, arm, repairer=None):
    start = time.perf_counter()
    risk=arm in ('risk_neural','risk_hybrid','risk_physical_lns','risk_physical_direct')
    objective=None
    if risk:
        from .deadline import packet_objective
        objective=packet_objective(problem)
    if arm == "greedy":
        action, values, record = solve(problem, draft, FULL)
        if not record["converged"]:
            raise RuntimeError("Physical baseline did not converge")
        return action, values, dict(record, arm=arm)
    if arm == 'risk_physical_direct':
        # Strong direct physical control: same draft, risk acceptance objective,
        # block graph and width-16 beam as the frozen neural path. Physical
        # prefixes use exact evaluation; all legal UEs remain available.
        config=replace(JOINT,repair_width=16,ue_topk=problem.U)
        action,values,record=solve(problem,draft,config,objective=objective)
        return action,values,dict(record,arm=arm)
    if arm in ("neural", "risk_neural"):
        if repairer is None:
            raise ValueError("Neural policy needs a checkpoint")
        action, values, record = solve(problem, draft, JOINT, repairer=repairer,objective=objective)
        return action, values, dict(record, arm=arm)
    if arm not in ("hybrid", "risk_hybrid", "physical_lns", "risk_physical_lns"):
        raise ValueError(arm)
    if arm in ("hybrid", "risk_hybrid") and repairer is None:
        raise ValueError("Hybrid policy needs a checkpoint")
    root, values, original = solve(problem, draft, FULL)
    if not original["converged"]:
        raise RuntimeError("Physical baseline did not converge")
    action, values, joint = solve(problem, draft, JOINT, initial_action=root,
                                  repairer=repairer if arm in ("hybrid", "risk_hybrid") else None,objective=objective)
    polish = None
    if joint["score"] > original["score"] + 1:
        action, values, polish = solve(problem, draft, FULL, initial_action=action,objective=objective)
        if not polish["converged"]:
            raise RuntimeError("Physical polishing did not converge")
    stages = [original, joint] + ([] if polish is None else [polish])
    final = stages[-1]
    assert final["score"] + 1.e-7 >= original["score"]
    record = dict(arm=arm, initial_score=original["initial_score"], score=final["score"],
        improvement=final["score"]-original["initial_score"],
        baseline_score=original["score"], joint_delta=joint["score"]-original["score"],
        polish_delta=0. if polish is None else polish["score"]-joint["score"],
        floor=original["floor"], draft_dequeue=original["draft_dequeue"],
        chosen_dequeue=final["chosen_dequeue"], core_ms=(time.perf_counter()-start)*1000,
        candidate_evaluations=sum(s["candidate_evaluations"] for s in stages),
        baseline_ms=original["core_ms"], joint_ms=joint["core_ms"],
        polish_ms=0. if polish is None else polish["core_ms"],
        stop_reason=final["stop_reason"], baseline_converged=True)
    return action, values, record

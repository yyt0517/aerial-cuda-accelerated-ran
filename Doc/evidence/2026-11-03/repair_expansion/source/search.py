"""TB-aware block selection, joint repair, and bounded anytime search.

The sparse graph guides proposals only. All actual boundary interference is
included by the verifier. Search thresholds/floor are explicit experiment inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import time
import numpy as np

from .physics import NativePhy, EXPECTED, DEQUEUE, BLER, COUNT, LAYER


@dataclass(frozen=True)
class SearchConfig:
    mode: str = "graph_lns"
    budget_ms: float | None = 10.
    max_candidates: int = 4096
    max_rounds: int = 128
    block_size: int = 4
    neighbor_cells: int = 2
    ue_topk: int = 3
    repair_width: int = 4
    batch_size: int = 16
    floor_numerator: int = 90
    floor_denominator: int = 100
    seed: int = 0
    tb_edges: bool = True

    def __post_init__(self):
        if self.mode not in ("graph_lns", "random_lns", "greedy"):
            raise ValueError("Unknown search mode")
        if self.budget_ms is not None and (not np.isfinite(self.budget_ms) or self.budget_ms < 0):
            raise ValueError("Invalid time budget")
        for field in ("max_candidates", "max_rounds", "block_size", "neighbor_cells", "ue_topk", "repair_width", "batch_size", "floor_denominator"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if not 0 <= self.floor_numerator <= self.floor_denominator:
            raise ValueError("Invalid causal floor")


class ProposalGraph:
    def __init__(self, phy):
        self.problem = p = phy.problem
        received = phy.received
        signal = received[np.arange(p.U), :, p.serving]
        denominator = 4*p.noise + received.sum(axis=-1)-signal
        # Cheap surrogate for ranking, not the physical acceptance score.
        self.quality = signal/np.maximum(denominator, 1.e-30)
        self.utility = p.weights[:, None]*np.minimum(p.queue[:, None], p.tbs[:, 1, 1][:, None])*np.log1p(self.quality)
        self.received = received

    def options(self, action, cell, prg, topk, prediction=None):
        p = self.problem
        legal = np.flatnonzero(p.legal[cell, prg])
        utility = self.utility[legal, prg]
        if prediction is not None:
            count = prediction[legal, COUNT].astype(int)
            layers = np.minimum(np.where(count > 0, prediction[legal, LAYER], 4).astype(int), p.caps[legal, prg])
            added = (action[cell, prg] != legal).astype(int)
            capacity = p.tbs[legal, np.minimum(count+added, p.P), layers]
            delta = np.minimum(p.queue[legal], capacity)-prediction[legal, DEQUEUE]
            reliability = np.where(count > 0, 1-np.maximum(prediction[legal, BLER], 0.), .5)
            # Preserve a small channel-quality term to explore unserved UEs.
            utility = p.weights[legal]*delta*reliability + .05*utility
        ranked = legal[np.argsort(-utility, kind="stable")[:topk]]
        return tuple(dict.fromkeys([int(action[cell, prg]), -1, *map(int, ranked)]))

    def blocks(self, action, prediction, config, rng):
        p = self.problem
        priority = np.zeros((p.C, p.P))
        neighbors = {}
        for c in range(p.C):
            for prg in range(p.P):
                targets = []
                for other in range(p.C):
                    u = action[other, prg]
                    if other == c or u < 0:
                        continue
                    fraction = self.received[u, prg, c]/max(4*p.noise+self.received[u, prg].sum()-self.received[u, prg, other], 1.e-30)
                    benefit = p.weights[u]*prediction[u, DEQUEUE]*max(0., prediction[u, BLER])*fraction
                    priority[c, prg] += benefit
                    targets.append((benefit, other))
                targets.sort(key=lambda x: (-x[0], x[1]))
                neighbors[c, prg] = [other for _, other in targets[:config.neighbor_cells]]
                u = action[c, prg]
                if u >= 0:
                    # Low-quality allocations and low-layer bottlenecks merit joint repair.
                    priority[c, prg] += p.weights[u]*prediction[u, DEQUEUE]*max(0., prediction[u, BLER])/(1+self.quality[u, prg])
                    if p.caps[u, prg] == prediction[u, LAYER] and prediction[u, COUNT] > 1:
                        priority[c, prg] += .05*p.weights[u]*prediction[u, DEQUEUE]
        order = np.argsort(-priority.ravel(), kind="stable")
        if config.mode == "random_lns":
            order = rng.permutation(p.C*p.P)
        seen = set()
        blocks = []
        for token in order:
            c, prg = divmod(int(token), p.P)
            block = [(c, prg)]
            block.extend((other, prg) for other in neighbors[c, prg])
            if config.tb_edges:
                for cell, local in tuple(block):
                    u = action[cell, local]
                    if u < 0:
                        continue
                    related = np.flatnonzero(action[cell] == u)
                    related = sorted(related, key=lambda x: (self.quality[u, x], int(x)))
                    block.extend((cell, int(x)) for x in related if x != local)
            block = tuple(dict.fromkeys(block))[:config.block_size]
            # Include positions on other PRGs only through a genuine TB relation.
            key = tuple(sorted(block))
            if key not in seen:
                seen.add(key)
                blocks.append(block)
        return blocks


def greedy_candidates(problem, action):
    """Frozen neighborhood order, omitting only the two duplicate root rows.

    Use one allocation as the existing accelerated implementation does. A slow
    Python candidate-copy loop would artificially disadvantage this control.
    """
    blocks, size = [], 0
    for prg in range(problem.P):
        choices = [np.flatnonzero(problem.legal[c, prg] & (np.arange(problem.U) != action[c, prg])) for c in range(problem.C)]
        for receiver in range(problem.C):
            if len(choices[receiver]):
                blocks.append((-1, receiver, prg, choices[receiver]))
                size += len(choices[receiver])
        for source in range(problem.C):
            if action[source, prg] < 0:
                continue
            blocks.append((source, -1, prg, None)); size += 1
            for receiver in range(problem.C):
                if receiver != source and len(choices[receiver]):
                    blocks.append((source, receiver, prg, choices[receiver]))
                    size += len(choices[receiver])
    result = np.empty((size, problem.C, problem.P), dtype=np.int64)
    result[:] = action
    offset = 0
    for source, receiver, prg, ues in blocks:
        end = offset+(1 if receiver < 0 else len(ues))
        if source >= 0:
            result[offset:end, source, prg] = -1
        if receiver >= 0:
            result[offset:end, receiver, prg] = ues
        offset = end
    return result


def repaired_candidates(graph, base, block, config, rng, evaluate, objective=None):
    """Bounded beam repair. Partial choices are complete feasible schedules.

    Lower-scoring partial schedules can survive in the beam, so a coordinated
    final repair need not improve after every individual variable change.
    """
    frontier = base.actions.copy()
    frontier_values = base.values
    for depth, (cell, prg) in enumerate(block):
        proposals = []
        seen = set()
        for parent, prediction in zip(frontier, frontier_values):
            for value in graph.options(parent, cell, prg, config.ue_topk, prediction):
                child = parent.copy(); child[cell, prg] = value
                key = child.tobytes()
                if key not in seen:
                    seen.add(key); proposals.append(child)
        if not proposals:
            return
        # Every verified intermediate is a complete legal candidate and may
        # update the incumbent. The evaluator enforces budget and floor.
        result = evaluate(np.stack(proposals), base)
        if result is None:
            return
        scores = (result.values[:, :, EXPECTED] @ graph.problem.weights
                  if objective is None else objective(result.values))
        order = np.argsort(-scores, kind="stable")
        width = min(config.repair_width, len(order))
        # Reserve one diverse trajectory when there are more candidates than slots.
        if width >= 2 and len(order) > width:
            order = np.r_[order[:width-1], rng.choice(order[width-1:])]
        frontier = result.actions[order[:width]]
        frontier_values = result.values[order[:width]]
        yield depth+1, len(result.actions)


def solve(problem, draft, config=SearchConfig(), *, trace=None, start_time=None, block_orderer=None,
          repairer=None, initial_action=None, objective=None):
    """Measured interval includes PHY setup, graph, generation, and verification.

    A deadline is checked between small verification batches. An indivisible
    operation can overrun; this prototype reports it rather than claiming hard RT.
    """
    start = time.perf_counter() if start_time is None else start_time
    deadline = float("inf") if config.budget_ms is None else start+config.budget_ms/1000.
    phy = NativePhy(problem)
    record = None
    try:
        incumbent = phy.evaluate(draft)
        initial_score = float(incumbent.values[0, :, EXPECTED] @ problem.weights)
        score = initial_score
        dequeue = int(incumbent.values[0, :, DEQUEUE].sum())
        floor = (config.floor_numerator*dequeue+config.floor_denominator-1)//config.floor_denominator
        initial_ms = (time.perf_counter()-start)*1000
        if initial_action is not None:
            candidate = phy.evaluate(initial_action)
            candidate_score = float(candidate.values[0, :, EXPECTED] @ problem.weights)
            if candidate.values[0, :, DEQUEUE].sum() < floor or candidate_score+1.e-7 < initial_score:
                raise ValueError("Warm start must satisfy the original draft floor and objective")
            incumbent, score = candidate, candidate_score
        utility = score if objective is None else float(objective(incumbent.values)[0])
        graph = None
        rng = np.random.default_rng(config.seed)
        evaluations, rounds, accepted = phy.stats["candidate_evaluations"], 0, 0
        stop_reason = "round_limit"
        converged = False
        policy_stopped = False

        def available():
            return evaluations < config.max_candidates and time.perf_counter() < deadline

        def evaluate(actions, base, collect=True):
            nonlocal evaluations, incumbent, score, accepted, utility
            parts = []
            for offset in range(0, len(actions), config.batch_size):
                if not available():
                    break
                batch = actions[offset:offset+min(config.batch_size, config.max_candidates-evaluations)]
                result = phy.evaluate(batch, base)
                evaluations += len(batch)
                scores = result.values[:, :, EXPECTED] @ problem.weights
                eligible = result.values[:, :, DEQUEUE].sum(axis=1) >= floor
                best = int(np.argmax(np.where(eligible, scores, -np.inf)))
                threshold = 1. if rounds <= 2 else 16.
                base_score = float(base.values[0, :, EXPECTED] @ problem.weights)
                utilities = scores if objective is None else objective(result.values)
                if objective is not None:
                    # The shaped objective never buys deadline/fairness gains by
                    # lowering the incumbent's original physical objective.
                    eligible &= (scores > score) & (scores > base_score+threshold) & (utilities > utility+1.)
                    best = int(np.argmax(np.where(eligible, utilities, -np.inf)))
                verified_in_time = time.perf_counter() <= deadline
                if verified_in_time and eligible[best] and scores[best] > score and scores[best] > base_score+threshold:
                    incumbent = phy.take(result, best)
                    score = float(scores[best]); accepted += 1
                    utility = float(utilities[best])
                if trace is not None:
                    trace(dict(round=rounds, actions=result.actions.copy(), scores=scores.copy(),
                               eligible=eligible.copy(), best_score=score,
                               verified_in_time=verified_in_time,
                               elapsed_ms=(time.perf_counter()-start)*1000))
                if collect:
                    parts.append(result)
            if not parts:
                return None
            from .physics import Evaluation
            actions = np.concatenate([r.actions for r in parts]); values = np.concatenate([r.values for r in parts])
            actions.flags.writeable = values.flags.writeable = False
            return Evaluation(actions, values, phy)

        while available() and rounds < config.max_rounds and not policy_stopped:
            rounds += 1
            root, old_score = incumbent, score
            if config.mode == "greedy":
                actions = greedy_candidates(problem, root.actions[0])
                if len(actions):
                    evaluate(actions, root, collect=False)
            else:
                if graph is None:
                    graph = ProposalGraph(phy)
                blocks = graph.blocks(root.actions[0], root.values[0], config, rng)
                if not available():
                    break
                adaptive = bool(getattr(block_orderer, "adaptive", False))
                if block_orderer is not None and not adaptive:
                    remaining_ms = None if config.budget_ms is None else max(0., (deadline-time.perf_counter())*1000)
                    blocks = block_orderer(graph, root, blocks, remaining_ms)
                pending = list(blocks)
                while pending:
                    if not available():
                        break
                    if adaptive:
                        remaining_ms = None if config.budget_ms is None else max(0., (deadline-time.perf_counter())*1000)
                        selected = block_orderer(graph, incumbent, pending, remaining_ms)
                        if not selected:
                            policy_stopped = True
                            stop_reason = "policy_stop"
                            break
                        block = selected[0]
                        if block not in pending:
                            raise ValueError("Policy selected a foreign block")
                        pending.remove(block)
                    else:
                        block = pending.pop(0)
                    # Start each block from the best solution found so far.
                    if repairer is None:
                        for _ in repaired_candidates(graph, incumbent, block, config, rng, evaluate, objective):
                            pass
                    elif hasattr(repairer, "repair"):
                        # Prefix verifiers use this same evaluator so that all
                        # work, budgets and incumbent protections are accounted.
                        repairer.repair(graph, incumbent, block, evaluate)
                    else:
                        proposals = repairer.propose(graph, incumbent, block)
                        if len(proposals):
                            # The physical verifier sees complete joint repairs only.
                            evaluate(proposals, incumbent, collect=False)
            deterministic_repair = (repairer is not None and getattr(repairer, "deterministic", False)
                                    and block_orderer is None and config.mode == "graph_lns")
            if score == old_score and (rounds >= 3 or deterministic_repair) and not policy_stopped:
                converged = config.mode == "greedy" and available()
                stop_reason = "no_improvement_in_visited_neighborhood"
                break
        if time.perf_counter() >= deadline:
            stop_reason = "time_budget"
        elif evaluations >= config.max_candidates:
            stop_reason = "candidate_budget"
        values = incumbent.values[0].copy()
        action = incumbent.actions[0].copy()
        end = time.perf_counter()
        record = dict(config=asdict(config), initial_score=initial_score, score=score,
            improvement=score-initial_score, objective_score=utility, draft_dequeue=dequeue, floor=floor,
            chosen_dequeue=float(values[:, DEQUEUE].sum()), initial_ms=initial_ms,
            core_ms=(end-start)*1000, overrun_ms=0. if config.budget_ms is None else max(0., (end-start)*1000-config.budget_ms),
            rounds=rounds, accepted_updates=accepted, stop_reason=stop_reason, converged=converged,
            **phy.stats)
        return action, values, record
    finally:
        phy.close()
        if record is not None:
            elapsed_ms = (time.perf_counter()-start)*1000
            record["core_ms"] = elapsed_ms
            record["overrun_ms"] = 0. if config.budget_ms is None else max(0., elapsed_ms-config.budget_ms)

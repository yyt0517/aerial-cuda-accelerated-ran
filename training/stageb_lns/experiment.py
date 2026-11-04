"""Versioned saved-state validation and equal-budget comparisons.

Usage: python -m training.stageb_lns.experiment --output deliverables/tb_lns_v1
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from .archive import ARCHIVE, CURRENT, FORMAL, load_rows, problem_from_row, initial_draft, tbs_table
from .physics import NativePhy, ReferencePhy, EXPECTED, DEQUEUE, FIELDS, native_bler, ROOT
from .search import SearchConfig, solve, greedy_candidates


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_sources(output):
    paths = [p for p in Path(__file__).parent.glob("*") if p.suffix in (".py", ".cpp", ".so")]
    paths += [ROOT/"cuMAC/src/cumac.h", ARCHIVE/"training/stageb_hgraph/ctbs_headroom.py",
              ARCHIVE/"training/stageb_hgraph/phy_tbs_alignment_audit.py"]
    sources = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        destination = output/"source"/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(path.read_bytes())
        sources[str(relative)] = sha(path)
    return sources


def legacy_phy(row):
    path = FORMAL / "boundary08_directed_coordination_20261008_v1/causal_phy.py"
    spec = importlib.util.spec_from_file_location("lns_legacy_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    phy = module.CausalPhy(row["ctbs"].config)
    phy.prepare(row["snapshot"], row["ctbs"])
    return phy


def validation(row, count=32):
    problem = problem_from_row(row)
    draft = initial_draft(problem, row["obs"])
    if row["tti"] >= 4001:
        saved = np.load(Path(row["source"]).with_name("drafts.npy"), mmap_mode="r")
        np.testing.assert_array_equal(draft, saved[row["tti"]-4001])
    rng = np.random.default_rng(row["seed"]*100000+row["tti"])
    actions = [draft, row["chosen"], np.full_like(draft, -1)]
    for _ in range(count):
        action = draft.copy()
        for _ in range(int(rng.integers(1, 7))):
            c, p = int(rng.integers(problem.C)), int(rng.integers(problem.P))
            action[c, p] = rng.choice(np.r_[-1, np.flatnonzero(problem.legal[c, p])])
        actions.append(action)
    actions = np.stack(actions)
    spec = importlib.util.spec_from_file_location("lns_legacy_candidates",
        FORMAL/"boundary75_search_acceleration_20261020_v1/candidates.py")
    candidate_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate_module)
    original_candidates, _ = candidate_module.candidates(draft, draft, row["obs"], problem.queue, True)
    np.testing.assert_array_equal(greedy_candidates(problem, draft), original_candidates[2:])
    phy = NativePhy(problem)
    try:
        base = phy.evaluate(draft)
        incremental = phy.evaluate(actions, base).values
        full = phy.evaluate(actions).values
        np.testing.assert_array_equal(incremental, full)
        permutation = rng.permutation(len(actions))
        np.testing.assert_array_equal(phy.evaluate(actions[permutation], base).values, full[permutation])
        chunks = np.concatenate([phy.evaluate(actions[i:i+7], base).values for i in range(0, len(actions), 7)])
        np.testing.assert_array_equal(chunks, full)
        independent = ReferencePhy(problem).evaluate(actions)
        np.testing.assert_allclose(incremental, independent, rtol=2.e-9, atol=2.e-6)
        old = legacy_phy(row)
        frozen = old.evaluate(actions)
        # Canonical native scan, applied equally to all new comparison arms.
        active = frozen["count"] > 0
        frozen["bler"][active] = native_bler(frozen["sinr_db"][active], frozen["mcs"][active])
        frozen["expected"] = frozen["dequeue"]*(1-np.where(active, frozen["bler"], 0.))
        reference = np.stack([frozen[key] for key in FIELDS], axis=-1)
        np.testing.assert_allclose(full, reference, rtol=2.e-9, atol=2.e-6)

        # Same prebuilt candidate stream, warm MMSE cache, both full and affected
        # evaluation. Includes Python validation/copy overhead in both arms.
        candidates = np.stack(list(greedy_candidates(problem, draft)))[:512]
        phy.evaluate(candidates)
        timings = {"full": [], "incremental": []}
        ue_evaluations = {}
        for repeat in range(5):
            for name in (["full", "incremental"] if repeat % 2 else ["incremental", "full"]):
                before = phy.stats["ue_evaluations"]
                begin = time.perf_counter()
                phy.evaluate(candidates, base if name == "incremental" else None)
                timings[name].append((time.perf_counter()-begin)*1000)
                ue_evaluations[name] = phy.stats["ue_evaluations"]-before
        return dict(seed=row["seed"], tti=row["tti"], actions_checked=len(actions),
                    exact_incremental_full_and_batch_invariance=True, exact_archived_draft=row["tti"] >= 4001,
                    exact_original_neighborhood_order_except_duplicate_roots=True,
                    independent_max_abs_error=float(np.abs(full-independent).max()),
                    canonical_legacy_max_abs_error=float(np.abs(full-reference).max()),
                    evaluator_ms={k:float(np.median(v)) for k,v in timings.items()},
                    evaluator_ue_counts=ue_evaluations, benchmark_candidates=len(candidates))
    finally:
        phy.close()


def run_one(row, config, block_orderer=None):
    start = time.perf_counter()
    problem = problem_from_row(row)
    draft = initial_draft(problem, row["obs"])
    action, values, result = solve(problem, draft, config, start_time=start, block_orderer=block_orderer)
    # Outside the timer: independent audit of the final action and acceptance.
    independent = ReferencePhy(problem).evaluate(action)[0]
    np.testing.assert_allclose(values, independent, rtol=2.e-9, atol=2.e-6)
    assert result["score"]+1.e-6 >= result["initial_score"]
    assert independent[:, DEQUEUE].sum() >= result["floor"]
    return action, result


def main(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    seeds = tuple(map(int, args.seeds.split(",")))
    budgets = tuple(map(float, args.budgets.split(",")))
    if not seeds or not budgets or args.repeats <= 0 or args.validation_actions < 0:
        raise ValueError("Invalid experiment dimensions")
    config = SearchConfig(max_candidates=args.max_candidates, block_size=args.block_size,
                          repair_width=args.repair_width, ue_topk=args.ue_topk)
    sources = snapshot_sources(output)
    ranker = None
    budget_ranker = None
    adaptive_ranker = None
    if args.ranker:
        from .learning import BlockRanker
        ranker = BlockRanker(args.ranker)
        (output/"block_ranker.npz").write_bytes(Path(args.ranker).read_bytes())
    if args.budget_ranker:
        from .budget_learning import BudgetRanker
        budget_ranker = BudgetRanker(args.budget_ranker)
        (output/"budget_ranker.npz").write_bytes(Path(args.budget_ranker).read_bytes())
    if args.budget_initial:
        from .budget_learning import BudgetRanker
        adaptive_ranker = BudgetRanker(args.budget_initial)
        (output/"budget_initial.npz").write_bytes(Path(args.budget_initial).read_bytes())
    arms_available = ["greedy", "graph_lns", "random_lns"] + (["learned_lns"] if ranker else []) + (["adaptive_lns"] if adaptive_ranker else []) + (["budget_rl_lns"] if budget_ranker else [])
    protocol = dict(version=1, seeds=seeds, budgets_ms=budgets, repeats=args.repeats,
        mode="fixed_state_only", score="native descending BLER scan; whole-TB weighted expected bytes",
        floor="ceil(90 * own PFQ_CTBS draft dequeue / 100)",
        source_sha256=sources, active_registry_sha256=sha(ROOT/"training/stageb_hgraph/current_algorithm.json"),
        snapshot_sha256={str(CURRENT/f"seed{s}_current_20000/states.pt"):sha(CURRENT/f"seed{s}_current_20000/states.pt") for s in seeds},
        selection="all six saved TTIs per requested seed; no outcome filtering",
        hardware=platform.platform(), processor=platform.processor(), numpy=np.__version__,
        threads={k:os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        timing="decoded snapshot to action: problem construction, CTBS prune, PHY prepare, graph, candidate generation, verification; excludes native PFQ production, IPC, pickle, cold process/compile/static TBS table initialization",
        limitations=["No live simulator or CRC rollout", "Deadline checked between indivisible operations; overruns reported", "Saved states are correlated and policy-conditioned", "Full local search is a reference, not a global optimum"],
        base_config=config.__dict__, greedy_batch_size=args.greedy_batch_size,
        ranker_sha256=sha(args.ranker) if args.ranker else None,
        budget_ranker_sha256=sha(args.budget_ranker) if args.budget_ranker else None,
        budget_initial_sha256=sha(args.budget_initial) if args.budget_initial else None,
        arms=arms_available)
    write_json(output/"protocol.json", protocol)
    rows = load_rows(seeds)
    import torch
    torch.set_num_threads(1)
    # Immutable PHY tables and dynamic-loader initialization are process setup.
    problem = problem_from_row(rows[0])
    warm = NativePhy(problem); warm.close()
    checks = []
    for row in rows:
        checks.append(validation(row, args.validation_actions))
        print("VALIDATED", row["seed"], row["tti"], flush=True)
    write_json(output/"validation.json", dict(passed=True, states=len(checks), checks=checks))
    records, actions = [], {}
    for state_index, row in enumerate(rows):
        reference_config = replace(config, mode="greedy", budget_ms=None, max_candidates=1000000, max_rounds=256, batch_size=args.greedy_batch_size)
        reference_action, reference = run_one(row, reference_config)
        if not reference["converged"]:
            raise RuntimeError("Reference failed to finish its full local neighborhood")
        records.append(dict(seed=row["seed"], tti=row["tti"], arm="full_greedy", repeat=0, **reference))
        actions[f"s{row['seed']}_t{row['tti']}_full"] = reference_action
        for budget in budgets:
            for repeat in range(args.repeats):
                arms = arms_available.copy()
                rotation = (state_index+repeat) % len(arms)
                arms = arms[rotation:]+arms[:rotation]
                for arm in arms:
                    current = replace(config, mode="graph_lns" if arm in ("learned_lns", "adaptive_lns", "budget_rl_lns") else arm, budget_ms=budget, seed=1700+repeat)
                    if arm == "greedy":
                        current = replace(current, batch_size=args.greedy_batch_size)
                    orderer = ranker if arm == "learned_lns" else budget_ranker if arm == "budget_rl_lns" else adaptive_ranker if arm == "adaptive_lns" else None
                    action, result = run_one(row, current, orderer)
                    denominator = reference["score"]-reference["initial_score"]
                    retained = (result["score"]-result["initial_score"])/denominator if denominator > 1. else None
                    records.append(dict(seed=row["seed"], tti=row["tti"], arm=arm, repeat=repeat,
                                        retained_reference_gain=retained, **result))
                    actions[f"s{row['seed']}_t{row['tti']}_{arm}_b{budget}_r{repeat}"] = action
        print("BENCHMARKED", row["seed"], row["tti"], "reference_gain", reference["improvement"], flush=True)
    for relative, digest in sources.items():
        if sha(ROOT/relative) != digest:
            raise RuntimeError(f"Source changed during experiment: {relative}")
    write_json(output/"results.json", records)
    np.savez_compressed(output/"actions.npz", **actions)
    summaries = []
    for budget in budgets:
        for arm in arms_available:
            subset = [r for r in records if r["arm"] == arm and r["config"]["budget_ms"] == budget]
            summaries.append(dict(arm=arm, budget_ms=budget, decisions=len(subset),
                mean_gain=float(np.mean([r["improvement"] for r in subset])),
                mean_retained_gain=float(np.mean([r["retained_reference_gain"] for r in subset if r["retained_reference_gain"] is not None])),
                median_ms=float(np.median([r["core_ms"] for r in subset])),
                p95_ms=float(np.percentile([r["core_ms"] for r in subset], 95)),
                max_ms=max(r["core_ms"] for r in subset),
                max_overrun_ms=max(r["overrun_ms"] for r in subset),
                mean_candidates=float(np.mean([r["candidate_evaluations"] for r in subset]))))
    write_json(output/"summary.json", dict(scope=protocol["mode"], rows=summaries))
    with (output/"summary.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="41,42,43,44,45")
    parser.add_argument("--budgets", default="0.5,2,10,50")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=100000)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--repair-width", type=int, default=4)
    parser.add_argument("--ue-topk", type=int, default=3)
    parser.add_argument("--validation-actions", type=int, default=32)
    parser.add_argument("--ranker")
    parser.add_argument("--budget-ranker")
    parser.add_argument("--greedy-batch-size", type=int, default=64)
    parser.add_argument("--budget-initial")
    main(parser.parse_args())

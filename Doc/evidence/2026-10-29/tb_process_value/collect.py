"""Regenerate missing candidate labels by frozen, bounded prefix replay."""
from dataclasses import replace
from pathlib import Path
import json
import shutil
import time
import numpy as np
import pandas as pd
import torch

from training.stageb_lns.archive import CURRENT, enable_legacy_types, problem_from_row, initial_draft
from training.stageb_lns.experiment import ROOT, sha, write_json, snapshot_sources
from training.stageb_lns.cached_repair import CachedDeadlineRepair
from training.stageb_lns.prefix_repair import shortlist
from training.stageb_lns.neural_repair import destroy
from training.stageb_lns.physics import NativePhy, DEQUEUE, EXPECTED
from training.stageb_lns.deadline import packet_objective
from training.stageb_lns.search import ProposalGraph
from training.stageb_lns.quality import JOINT
from training.stageb_lns.process_value import cheap_completion_features

HERE = Path(__file__).resolve().parent


def sources(plan):
    for split in ('train', 'selection', 'calibration', 'evaluation'):
        for seed in plan[split+'_seeds']:
            if seed <= 45:
                path = CURRENT/f'seed{seed}_current_20000/states.pt'
                warmup = 4000
            elif seed in (46, 47):
                path = ROOT/f'deliverables/tb_joint_repair_20261027_native_v2/seed{seed}_neural/states.pt'
                warmup = 2000
            elif seed == 49:
                path = ROOT/'deliverables/tb_deadline_20261028_collection_v1/seed49_neural/states.pt'
                warmup = 1000
            else:
                path = ROOT/f'deliverables/tb_deadline_20261028_validation_native_v1/seed{seed}_risk_neural/states.pt'
                warmup = 1000
            rows = [r for r in torch.load(path, weights_only=False, map_location='cpu') if r['tti'] > warmup]
            if split != 'evaluation':
                rows = [rows[i] for i in np.linspace(0, len(rows)-1, 5).astype(int)]
            yield split, seed, path, rows


def trace_block(graph, phy, model, base, block):
    p = graph.problem
    cache = model.cache_for(graph, base)
    root = destroy(base.actions[0], block)
    assigned = root[p.serving, :] == np.arange(p.U)[:, None]
    count = assigned.sum(-1)[None]
    layer = np.where(assigned, p.caps, 5).min(-1)[None]
    layer = np.where(count > 0, layer, 4)
    active = (root >= 0)[None]
    edits = np.full((1, len(block)), -1, dtype=np.int64)
    logp = np.zeros(1)
    objective = packet_objective(p)
    root_r = float(objective(base.values)[0])
    parent_r = np.array([root_r])
    batches = []
    for depth, (cell, prg) in enumerate(block):
        choices = cache.node(cell, prg)[0]
        x = cache.features(block, depth, count, layer, active)
        logits = model.logits(x)
        logits -= logits.max(-1, keepdims=True)
        logits -= np.log(np.exp(logits).sum(-1, keepdims=True))
        joint = logp[:, None]+logits
        indices = shortlist(joint, choices, int(base.actions[0, cell, prg]), 16, 48, 2)
        parent, value = np.divmod(indices, len(choices))
        prefix = edits[parent].copy()
        prefix[:, depth] = choices[value]
        complete = np.broadcast_to(base.actions[0], (len(prefix), p.C, p.P)).copy()
        for step, (c, pr) in enumerate(block[:depth+1]):
            complete[:, c, pr] = prefix[:, step]
        candidate_logp = joint.ravel()[indices]
        begin = time.perf_counter()
        cheap, proxy = cheap_completion_features(graph, base, complete)
        feature_ms = 1000*(time.perf_counter()-begin)
        meta = np.stack((candidate_logp/20, logp[parent]/20,
            np.full(len(indices), depth/3), np.full(len(indices), len(block)/4),
            (complete != base.actions[0]).sum((1, 2))/4,
            (choices[value] < 0).astype(float)), axis=-1)
        feature = np.concatenate((x[parent, value], meta, cheap), axis=1).astype(np.float32)
        result = phy.evaluate(complete, base)
        r = objective(result.values)
        batches.append(dict(x=feature, delta=r-root_r, step_delta=r-parent_r[parent],
            j=result.values[:, :, EXPECTED]@p.weights,
            dequeue=result.values[:, :, DEQUEUE].sum(1), proxy=proxy,
            logp=candidate_logp, depth=np.full(len(indices), depth), actions=complete,
            feature_ms=np.full(len(indices), feature_ms/len(indices))))
        order = np.lexsort((np.arange(len(r)), -candidate_logp, -r))[:16]
        selected = choices[value[order]]
        previous = parent[order]
        edits = prefix[order]
        count, layer, active = count[previous], layer[previous], active[previous]
        active[:, cell, prg] = selected >= 0
        rr = np.flatnonzero(selected >= 0)
        uu = selected[rr]
        count[rr, uu] += 1
        layer[rr, uu] = np.minimum(layer[rr, uu], p.caps[uu, prg])
        logp = candidate_logp[order]
        parent_r = r[order]
    merged = {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
    keys = merged['actions'].reshape(len(merged['actions']), -1)
    _, first = np.unique(keys, axis=0, return_index=True)
    unique = np.zeros(len(keys), dtype=bool)
    unique[first] = True
    merged['unique'] = unique
    merged['eligible'] = ((merged['j'] > float(base.values[0, :, EXPECTED]@p.weights)+16)
                          & (merged['delta'] > 1))
    return merged


def main():
    enable_legacy_types()
    plan = json.loads((HERE/'plan.json').read_text())
    assert sha(ROOT/plan['actor_checkpoint']) == plan['actor_sha256']
    out = HERE/'data'
    out.mkdir(exist_ok=False)
    shutil.copyfile(ROOT/plan['actor_checkpoint'], HERE/'actor_frozen.npz')
    source_hashes = snapshot_sources(out)
    write_json(out/'frozen_plan.json', dict(plan=plan, sha256=sha(HERE/'plan.json'), source_sha256=source_hashes))
    model = CachedDeadlineRepair(HERE/'actor_frozen.npz', 16)
    all_arrays, groups, states, input_hashes = [], [], [], {}
    gid = 0
    started = time.perf_counter()
    for split, seed, path, rows in sources(plan):
        input_hashes[str(path.relative_to(ROOT))] = sha(path)
        for row in rows:
            begin = time.perf_counter()
            p = problem_from_row(row)
            phy = NativePhy(p)
            try:
                draft = initial_draft(p, row['obs'])
                draft_eval = phy.evaluate(draft)
                floor = (90*int(draft_eval.values[0, :, DEQUEUE].sum())+99)//100
                graph = ProposalGraph(phy)
                local_occurrences = 0
                for origin, action in (('own_pfqdraft_ctbs', draft), ('saved_chosen', row['chosen'])):
                    base = phy.evaluate(action)
                    cfg = replace(JOINT, repair_width=16, ue_topk=p.U)
                    blocks = graph.blocks(base.actions[0], base.values[0], cfg, np.random.default_rng(0))
                    selected = list(range(min(8, len(blocks))))
                    if len(blocks) > 8:
                        selected += list(map(int, np.linspace(8, len(blocks)-1, min(4, len(blocks)-8))))
                    for bi in selected:
                        block = blocks[bi]
                        batch = trace_block(graph, phy, model, base, block)
                        n = len(batch['delta'])
                        batch['eligible'] &= batch['dequeue'] >= floor
                        # One fresh full evaluation checks incremental label integrity per root.
                        if bi == selected[0]:
                            audit = NativePhy(p)
                            try:
                                full = audit.evaluate(batch['actions'][:1])
                                inc = phy.evaluate(batch['actions'][:1], base)
                                np.testing.assert_array_equal(full.values, inc.values)
                            finally:
                                audit.close()
                        batch.pop('actions')
                        batch['group'] = np.full(n, gid, dtype=np.int32)
                        batch['seed'] = np.full(n, seed, dtype=np.int16)
                        all_arrays.append(batch)
                        groups.append(dict(group=gid, split=split, seed=seed, tti=row['tti'],
                            origin=origin, block=bi, tokens=json.dumps(block), occurrences=n,
                            unique=int(batch['unique'].sum()), root_j=float(base.values[0, :, EXPECTED]@p.weights),
                            root_r=float(packet_objective(p)(base.values)[0]), floor=floor))
                        gid += 1
                        local_occurrences += n
                states.append(dict(split=split, seed=seed, tti=row['tti'],
                    candidate_occurrences=local_occurrences, **phy.stats,
                    wall_ms=1000*(time.perf_counter()-begin), independent_audit_evaluations=2))
            finally:
                phy.close()
            print('COLLECT', split, seed, row['tti'], local_occurrences, flush=True)
            (HERE/'progress.txt').write_text(f'逐候选标签回放：{len(states)}个快照，{gid}个修复块；未使用新测试seed。\n')
    arrays = {k: np.concatenate([a[k] for a in all_arrays]) for k in all_arrays[0]}
    assert arrays['x'].shape[1] == 189
    assert all(np.isfinite(v).all() for v in arrays.values())
    np.savez_compressed(out/'candidates.npz', **arrays)
    pd.DataFrame(groups).to_csv(out/'groups.csv', index=False)
    pd.DataFrame(states).to_csv(out/'states.csv', index=False)
    for path, digest in input_hashes.items():
        assert sha(ROOT/path) == digest
    for path, digest in source_hashes.items():
        assert sha(ROOT/path) == digest
    write_json(out/'checks.json', dict(passed=True, input_sha256=input_hashes,
        snapshots=len(states), groups=len(groups), candidate_occurrences=len(arrays['delta']),
        unique_candidates=int(arrays['unique'].sum()), features=189,
        total_wall_s=time.perf_counter()-started, incremental_full_audits=2*len(states),
        inputs_exclude_counterfactual_phy=True, future_outcomes_unused=True,
        registry_sha256=sha(ROOT/'training/stageb_hgraph/current_algorithm.json')))


if __name__ == '__main__':
    main()

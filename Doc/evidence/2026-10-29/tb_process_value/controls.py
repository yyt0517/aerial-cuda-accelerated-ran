"""Stronger post-hoc physics control: root-anchored trace-ratio SINR draft.

Uses observed per-PRG hints, current root PHY, logical TB sizes and native
BLER tables. No candidate MMSE solve and no learned parameters are used.
"""
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collect
from metrics import curves
from training.stageb_lns.physics import NativePhy, COUNT, LAYER, SINR, EXPECTED, native_bler
from training.stageb_lns.search import ProposalGraph
from training.stageb_lns.quality import JOINT
from training.stageb_lns.cached_repair import CachedDeadlineRepair
from training.stageb_lns.archive import enable_legacy_types, problem_from_row, initial_draft
from training.stageb_lns.experiment import write_json, sha
from training.stageb_lns.process_value import risk_from_expected, ProcessValue
from dataclasses import replace


class TraceDraft:
    def __init__(self, graph, base):
        self.p = p = graph.problem
        self.base = base
        self.received = graph.received.copy()
        self.received[np.arange(p.U), :, p.serving] = 0
        self.original = 4*p.noise+np.einsum('cp,upc->up', base.actions[0] >= 0, self.received)
        self.signal = graph.received[np.arange(p.U), :, p.serving]
        self.root_rough_db = self.rough(base.actions)[2][0]
        self.root_r = float(risk_from_expected(p, base.values[0, :, EXPECTED]))

    def rough(self, actions):
        p = self.p
        assigned = actions[:, p.serving, :] == np.arange(p.U)[None, :, None]
        count = assigned.sum(-1)
        layers = np.where(assigned, p.caps, 5).min(-1)
        layers = np.where(count > 0, layers, 0)
        denominator = 4*p.noise+np.einsum('bcp,upc->bup', actions >= 0, self.received)
        if p.sinr_hint is None:
            gamma = self.signal[None]/denominator
        else:
            include = np.arange(4)[None, None, None] < layers[..., None, None]
            hint = (p.sinr_hint[None]*include).sum(-1)/np.maximum(layers[..., None], 1)
            gamma = hint*self.original[None]/denominator
        gamma = (gamma*assigned).sum(-1)/np.maximum(count, 1)
        db = np.where(count > 0, 10*np.log10(np.maximum(gamma, 1.e-12)), 0)
        return count, layers, db

    def score(self, actions):
        p = self.p
        count, layers, db = self.rough(actions)
        served = self.base.values[0, :, COUNT] > 0
        db[:, served] += self.base.values[0, served, SINR]-self.root_rough_db[served]
        bler = native_bler(db, p.mcs)
        dequeue = np.minimum(p.queue, p.tbs[np.arange(p.U)[None], count, layers])
        return risk_from_expected(p, dequeue*(1-bler))-self.root_r


def main():
    enable_legacy_types()
    out = HERE/'diagnostics';out.mkdir(exist_ok=False)
    plan = json.loads((HERE/'plan.json').read_text())
    write_json(out/'protocol.json', dict(kind='posthoc_strengthened_control_and_depth_conditioning',
        value_model_sha256=sha(HERE/'value_frozen.npz'), model_retraining=False,
        selection_not_reopened=True, no_new_final_seeds=True,
        reasons=['Frozen-BLER control cannot respond to interference changes',
                 'Cross-depth union ranking can confound actor cumulative likelihood with sequence length'],
        depth_audit='Same frozen model; groups split by depth of first canonical occurrence. These are diagnostic metrics, not retuned primary selection gates'))
    with np.load(HERE/'data/candidates.npz') as saved:
        arrays = {k: saved[k] for k in saved.files}
    groups = pd.read_csv(HERE/'data/groups.csv')
    model = CachedDeadlineRepair(HERE/'actor_frozen.npz', 16)
    scores, calls, timings = [], [], []
    offset = 0;gid = 0
    for split, seed, path, rows in collect.sources(plan):
        for row in rows:
            p = problem_from_row(row);phy = NativePhy(p)
            try:
                graph = ProposalGraph(phy)
                draft = initial_draft(p, row['obs'])
                for origin, action in (('own_pfqdraft_ctbs', draft), ('saved_chosen', row['chosen'])):
                    base = phy.evaluate(action)
                    control = TraceDraft(graph, base)
                    np.testing.assert_allclose(control.score(base.actions), 0, rtol=0, atol=1.e-8)
                    blocks = graph.blocks(base.actions[0], base.values[0],
                        replace(JOINT, repair_width=16, ue_topk=p.U), np.random.default_rng(0))
                    selected = list(range(min(8, len(blocks))))
                    if len(blocks) > 8:
                        selected += list(map(int, np.linspace(8, len(blocks)-1, min(4, len(blocks)-8))))
                    for bi in selected:
                        batch = collect.trace_block(graph, phy, model, base, blocks[bi])
                        n = len(batch['delta']);ids = slice(offset, offset+n)
                        np.testing.assert_array_equal(arrays['x'][ids], batch['x'])
                        np.testing.assert_array_equal(arrays['delta'][ids], batch['delta'])
                        np.testing.assert_array_equal(arrays['unique'][ids], batch['unique'])
                        begin = time.perf_counter();score = control.score(batch['actions'])
                        wall_ms = 1000*(time.perf_counter()-begin)
                        scores.append(score)
                        timings.append(dict(group=gid, split=split, seed=seed, candidates=n,
                            trace_physics_ms=wall_ms))
                        offset += n;gid += 1
                calls.append(dict(split=split, seed=seed, tti=row['tti'], **phy.stats))
            finally:
                phy.close()
            print('CONTROL', split, seed, row['tti'], flush=True)
    assert offset == len(arrays['delta']) and gid == len(groups)
    scores = np.concatenate(scores)
    np.save(out/'trace_physics_scores.npy', scores)
    pd.DataFrame(timings).to_csv(out/'trace_physics_timings.csv', index=False)
    pd.DataFrame(calls).to_csv(out/'additional_replay_cost.csv', index=False)
    z = curves(arrays, groups[groups.split != 'train'], scores, plan['budgets'], 'trace_sinr_physics')
    z.to_csv(out/'trace_curves_per_block.csv', index=False)
    z.groupby(['split', 'seed', 'method', 'budget']).mean(numeric_only=True).to_csv(out/'trace_curves_per_seed.csv')
    depth_arrays = dict(arrays, group=4*arrays['group']+arrays['depth'])
    depth_groups = []
    for g in groups[groups.split != 'train'].itertuples():
        for depth in range(4):
            if not np.any((depth_arrays['group'] == 4*g.group+depth) & arrays['unique']):
                continue
            depth_groups.append(dict(group=4*g.group+depth, split=g.split, seed=g.seed,
                tti=g.tti, origin=g.origin, depth=depth))
    dg = pd.DataFrame(depth_groups)
    prediction, _ = ProcessValue(HERE/'value_frozen.npz').predict(arrays['x'])
    ds = dict(actor_likelihood=arrays['logp'], frozen_bler_tb_physics=arrays['proxy'],
        trace_sinr_physics=scores, frozen_tb_mlp=prediction[:, 0],
        exact_oracle=np.where(arrays['eligible'], arrays['delta'], -1.e30))
    depth = pd.concat([curves(depth_arrays, dg, v, plan['budgets'], k) for k, v in ds.items()], ignore_index=True)
    depth = depth.merge(dg[['group', 'depth']], on='group', validate='many_to_one')
    depth.to_csv(out/'depth_curves_per_group.csv', index=False)
    depth.groupby(['split', 'seed', 'method', 'budget']).mean(numeric_only=True).to_csv(out/'depth_curves_per_seed.csv')
    write_json(out/'checks.json', dict(passed=True, every_replayed_feature_and_label_exact=True,
        original_candidates_npz_sha256=sha(HERE/'data/candidates.npz'),
        root_anchored_physics_identity_passed=True, frozen_value_sha256=sha(HERE/'value_frozen.npz'),
        control_uses_no_counterfactual_mmse=True, control_uses_native_bler_lookup=True,
        primary_gate_not_changed=True, additional_replay_evaluations=int(pd.DataFrame(calls).candidate_evaluations.sum())))


if __name__ == '__main__':
    main()

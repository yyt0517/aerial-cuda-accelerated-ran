"""Factor switches preserve the frozen global selector, including batch order."""
import inspect
from types import SimpleNamespace
import numpy as np

from common import ARMS, baseline, optimized, prepare_candidates


class Ablation:
    def __init__(self, arm):
        self.arm = arm
        self.config = ARMS[arm]
        self.audit = []
        self.drafts = []
        source = inspect.getsource(optimized._select)
        assert source.count('keep_dequeue=float(dequeues[0])') == 1
        source = source.replace('keep_dequeue=float(dequeues[0])',
                                'keep_dequeue=float((90*int(dequeues[0])+99)//100)')
        self.selector_source = source

    def weights(self, ema, target):
        return optimized.service_weights(ema, target) if self.config['deficit'] else np.ones(36)

    def policy(self, obs, ctbs, snapshot, builder, edges, actor, coordinator, phy):
        assert actor is None and coordinator is None
        graph = edges.build_graph(builder.build(obs))
        raw_ue = np.asarray(obs['teacher_action_ue_select']).copy()
        raw_prg = np.asarray(obs['teacher_action_prg_alloc']).copy()
        raw = baseline.r.p.decode_type0_teacher_action(raw_ue, raw_prg, n_cell=3, n_prg=17).reshape(3, 17)
        if self.config['ctbs']:
            draft, ue, prg, audit = baseline.r.replacement.pfq_draft(obs, graph, ctbs)
            pruned = audit['dropped_count']
        else:
            draft, ue, prg, pruned = raw, raw_ue, raw_prg, 0
        # Pure CTBS has no PHY search setup; diagnostic TBS uses the same bank.
        if self.config['search'] != 'none':
            phy.prepare(snapshot, ctbs)
        root = baseline.discrete_prediction(draft, ctbs, snapshot, phy.tbs)
        raw_root = baseline.discrete_prediction(raw, ctbs, snapshot, phy.tbs)
        dequeue = int(root['dequeue'].sum())
        self.audit.append(dict(tti=int(ctbs.tti) + 1, ctbs_pruned_prgs=pruned,
            raw_blank=int((raw < 0).sum()), aligned_blank=int((draft < 0).sum()),
            keep_blank=int((draft < 0).sum()), coordinator_blank_token=-1,
            draft_dequeue=dequeue, raw_pfq_dequeue=int(raw_root['dequeue'].sum()),
            causal_floor_bytes=(90*dequeue+99)//100, component_arm=self.arm))
        self.drafts.append(draft.copy())
        return draft, draft.copy(), ue, prg, None, -1

    def select(self, draft, keep, phy, obs, ctbs, weights):
        np.testing.assert_array_equal(draft, keep)
        assert not self.config['deficit'] or np.isclose(weights.mean(), 1.)
        if self.config['search'] == 'global':
            namespace = dict(optimized._globals)
            namespace['two'] = SimpleNamespace(candidates=prepare_candidates(obs, phy.snapshot['queue']),
                evaluate=optimized._evaluate, native_bler=optimized._native_bler)
            exec(compile(self.selector_source, __file__ + ':frozen_global', 'exec'), namespace)
            chosen, prediction, record, rounds = namespace['select'](draft, keep, phy, obs, ctbs, weights)
        elif self.config['search'] == 'none':
            chosen, prediction, rounds = draft.copy(), None, []
            record = dict(converged=True, cap_reached=False, rounds_evaluated=0,
                chosen_dequeue=self.audit[-1]['draft_dequeue'], candidate_evaluations=0)
        else:
            chosen, prediction, record, rounds = self.independent(draft, phy, obs, weights)
        record.update(draft_dequeue=self.audit[-1]['draft_dequeue'],
            keep_dequeue=self.audit[-1]['causal_floor_bytes'],
            candidate_evaluations=sum(x['candidates'] for x in rounds),
            component_arm=self.arm, bc_forward_calls=0, coordinator_forward_calls=0,
            external_floor_reads=0, search_kind=self.config['search'])
        assert record['chosen_dequeue'] >= record['keep_dequeue']
        return chosen, prediction, record, rounds

    def independent(self, draft, phy, obs, weights):
        """Three independent cell optimizations against the SAME frozen draft.

        Each selects on its own UEs; no neighbor score, joint edit or feedback
        from another cell's selected action is available. Simultaneous commit.
        Per-cell ceil(.9*D_cell) implies the aggregate causal floor. This is an
        explicitly defined independent control, not a unique universal one.
        """
        root_pred = optimized._evaluate(phy, draft[None])
        legal = np.asarray(obs['action_mask_cell_ue'], bool) & np.asarray(obs['action_mask_ue'], bool)[None]
        legal &= (np.asarray(phy.snapshot['queue']) > 0)[None]
        prg_legal = np.asarray(obs['action_mask_prg_cell'], bool)
        chosen = draft.copy()
        rows = []
        for cell in range(3):
            users = np.asarray(ctbs_users := (phy.serving == cell))
            root_dequeue = int(root_pred['dequeue'][0, users].sum())
            floor = (90*root_dequeue+99)//100
            incumbent = draft.copy()
            score = None
            for number in range(1, 257):
                actions = [incumbent.copy(), incumbent.copy()]
                for prg in range(17):
                    if prg_legal[cell, prg]:
                        for ue in np.flatnonzero(legal[cell]):
                            if ue != incumbent[cell, prg]:
                                action = incumbent.copy(); action[cell, prg] = ue
                                actions.append(action)
                    if incumbent[cell, prg] >= 0:
                        action = incumbent.copy(); action[cell, prg] = -1
                        actions.append(action)
                actions = np.stack(actions)
                prediction = optimized._evaluate(phy, actions)
                scores = (prediction['expected'][:, users] * weights[None, users]).sum(1)
                dequeues = prediction['dequeue'][:, users].sum(1)
                if score is None:
                    score = float(scores[0])
                eligible = dequeues >= floor
                assert eligible[0]
                best = int(np.argmax(np.where(eligible, scores, -np.inf)))
                gain = float(scores[best] - score)
                accepted = gain > (1. if number <= 2 else 16.)
                if accepted:
                    incumbent = actions[best]
                    score = float(scores[best])
                # Other cells remain exactly the initial draft throughout.
                np.testing.assert_array_equal(incumbent[np.arange(3) != cell], draft[np.arange(3) != cell])
                rows.append(dict(cell=cell, round=number, weighted_expected=score,
                    gain=gain if accepted else 0., accepted=accepted, candidates=len(actions), cell_floor=floor))
                if number >= 3 and not accepted:
                    break
            assert not rows[-1]['accepted'], 'independent 256-round cap hit'
            chosen[cell] = incumbent[cell]
        final = optimized._evaluate(phy, chosen[None])
        prediction = {key: value[0] for key, value in final.items()}
        rows.append(dict(cell=-1, round=0, weighted_expected=float((prediction['expected']*weights).sum()),
            gain=0., accepted=False, candidates=2, cell_floor=-1))  # root + combined final PHY
        record = dict(converged=True, cap_reached=False,
            rounds_evaluated=sum(row['cell'] >= 0 for row in rows),
            chosen_dequeue=int(prediction['dequeue'].sum()),
            chosen_expected=float(prediction['expected'].sum()),
            chosen_weighted_expected=float((prediction['expected']*weights).sum()))
        return chosen, prediction, record, rows

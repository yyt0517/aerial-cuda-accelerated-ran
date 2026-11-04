"""Bounded, search-free neural candidates plus the unmodified PFQ fallback."""
import time
import numpy as np
from training.stageb_lns.archive import problem_from_row
from training.stageb_lns.physics import NativePhy, EXPECTED, DEQUEUE, COUNT, BLER, SINR
from training.stageb_lns.search import ProposalGraph
from training.stageb_ppo_v2.catalog import completion_bytes
from .features import packet_features, STATE_DIM
from .generator import StudentRepair

MAX_CHOICES = 8


class FastCatalog:
    def __init__(self, checkpoint, k=8):
        assert k in (4, 8)
        self.repair = StudentRepair(checkpoint); self.k = k

    def build(self, env, teacher_action=None):
        start = time.perf_counter()
        p = problem_from_row(dict(ctbs=env.inputs, obs=env.obs, snapshot=env.snapshot, ema=np.full(36, 2400.)))
        p.weights[:] = 1.; p.deadline = None
        packet, potential = packet_features(env.fifo, env.tti+1, p.queue)
        phy = NativePhy(p)
        try:
            base = phy.evaluate(env.raw_pfq()); graph = ProposalGraph(phy)
            neural = self.repair.decode(graph, base, packet, self.k-1)
            generated = np.concatenate((base.actions, neural))
            # Do not refill duplicates with extra proposals or any hidden search.
            distinct = np.ones(self.k, bool)
            for i in range(1, self.k): distinct[i] = not any(np.array_equal(generated[i], generated[j]) for j in range(i))
            values = np.concatenate((base.values, phy.evaluate(generated[1:], base).values))
            score = values[:, :, EXPECTED].sum(-1); dq = values[:, :, DEQUEUE].sum(-1)
            floor = (90*int(dq[0])+99)//100
            eligible = distinct & (dq >= floor); assert eligible[0]
            full = completion_bytes(env.fifo, env.tti+1, p.queue, values[:, :, DEQUEUE])*(1-np.maximum(values[:, :, BLER], 0))
            # First 15 entries retain the old feature ordering, with PFQ as
            # reference. The appended six entries describe the whole age mix.
            state = np.concatenate((packet[:, :8], np.log1p(p.queue[:, None]/3000), p.mcs[:, None]/27,
                base.values[0, :, EXPECTED, None]/3000, base.values[0, :, COUNT, None]/17,
                base.values[0, :, DEQUEUE, None]/np.maximum(p.queue[:, None], 3000),
                np.maximum(base.values[0, :, BLER, None], 0), base.values[0, :, SINR, None]/30, packet[:, 8:]), -1)
            candidate = np.stack((values[:, :, EXPECTED]/3000, (values[:, :, EXPECTED]-values[0, :, EXPECTED])/3000,
                values[:, :, DEQUEUE]/3000, (values[:, :, DEQUEUE]-values[0, :, DEQUEUE])/3000,
                values[:, :, COUNT]/17, (values[:, :, COUNT]-values[0, :, COUNT])/17,
                np.maximum(values[:, :, BLER], 0), values[:, :, SINR]/30, full/3000), -1)
            summary = np.stack((score/86400, (score-score[0])/3000, dq/86400, full.sum(-1)/86400,
                (values[:, :, COUNT] > 0).sum(-1)/36, (generated < 0).sum((1, 2))/51, (np.arange(self.k) == 0).astype(float)), -1)
            padded = np.zeros((MAX_CHOICES, 36, 9), np.float32); padded[:self.k] = candidate
            summaries = np.zeros((MAX_CHOICES, 7), np.float32); summaries[:self.k] = summary
            mask = np.zeros(MAX_CHOICES, bool); mask[:self.k] = eligible
            actions = np.full((MAX_CHOICES, 3, 17), -1, np.int64); actions[:self.k] = generated
            prior = np.zeros(MAX_CHOICES, np.float32); prior[:self.k] = (score-score[eligible].max())/3000
            assert state.shape == (36, STATE_DIM)
            # Snapshot fast-path work before any training-only teacher labels.
            audit = dict(tti=env.tti+1, k=self.k, choices=int(mask.sum()), generated=self.k,
                unique=int(distinct.sum()), neural_eligible=int(eligible[1:].sum()),
                candidate_evaluations=phy.stats['candidate_evaluations'], floor_bytes=floor,
                best_expected_bytes=float(score[eligible].max()), pfq_expected_bytes=float(score[0]),
                ctbs_pruned=0, weights_min=float(p.weights.min()), weights_max=float(p.weights.max()),
                full_search_calls=0, wall_s=time.perf_counter()-start)
            result = dict(state=np.clip(state, -10, 10).astype(np.float32), candidate=np.clip(padded, -10, 10),
                summary=np.clip(summaries, -10, 10), mask=mask, prior=prior, actions=actions, potential=potential, audit=audit)
            if teacher_action is not None:
                target = phy.evaluate(teacher_action, base)
                teacher_score = float(target.values[:, :, EXPECTED].sum())
                result['label'] = self.repair.decode(graph, base, packet, 1, target=teacher_action)
                result['teacher'] = dict(expected_bytes=teacher_score,
                    best_student_over_teacher=float(score[eligible].max()/max(teacher_score, 1)),
                    assignment_match=float((generated[score.argmax()]==teacher_action).mean()),
                    teacher_action=np.asarray(teacher_action).copy())
            return result
        finally:
            phy.close()

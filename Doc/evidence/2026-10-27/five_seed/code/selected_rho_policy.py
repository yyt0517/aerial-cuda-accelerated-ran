"""No learned proposal; a causal floor from this policy's own PFQ+CTBS draft."""
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from output.pfq_independent_reuse.formal.boundary08_corrected_physics_20261020_v1 import run as corrected
from output.pfq_independent_reuse.formal.boundary75_search_acceleration_20261020_v1.candidates import prepare_candidates

OUT = Path(__file__).resolve().parent
r = corrected.r
RATIO_NUM = 100
AUDIT, DRAFTS = [], []


def selector_source():
    source = inspect.getsource(r.optimized._select)
    assignment = 'keep_dequeue=float(dequeues[0])'
    assert source.count(assignment) == 1
    source = source.replace(assignment,
        'draft_dequeue=int(dequeues[0])\n'
        '            keep_dequeue=float((RATIO_NUM*draft_dequeue+99)//100)')
    old_return = '    return chosen,prediction,record,rows'
    assert source.count(old_return) == 1
    return source.replace(old_return,
        '    record.update(draft_dequeue=draft_dequeue, floor_ratio_num=RATIO_NUM)\n'
        + old_return)


def select(draft, keep, phy, obs, ctbs, weights):
    assert RATIO_NUM in (100, 95, 90)
    np.testing.assert_array_equal(draft, keep)
    namespace = dict(r.optimized._globals)
    namespace.update(RATIO_NUM=RATIO_NUM, two=SimpleNamespace(
        candidates=prepare_candidates(obs, phy.snapshot['queue']),
        evaluate=r.optimized._evaluate, native_bler=r.optimized._native_bler))
    exec(compile(selector_source(), str(OUT / 'search_ratio.py'), 'exec'), namespace)
    chosen, prediction, record, history = namespace['select'](draft, keep, phy, obs, ctbs, weights)
    independent = AUDIT[-1]
    assert independent['tti'] == int(ctbs.tti)+1
    assert record['draft_dequeue'] == independent['draft_dequeue_independent']
    assert record['keep_dequeue'] == independent['causal_floor_bytes']
    record.update(candidate_evaluations=sum(x['candidates'] for x in history),
                  coordinator_forward_calls=0, external_floor_reads=0)
    return chosen, prediction, record, history


def policy(obs, ctbs, snapshot, builder, edges, actor, coordinator, phy):
    assert actor is None and coordinator is None
    graph = edges.build_graph(builder.build(obs))
    draft, ue, prg, audit = r.replacement.pfq_draft(obs, graph, ctbs)
    phy.prepare(snapshot, ctbs)
    # Independent discrete capacity calculation, before full-neighborhood scoring.
    root = corrected.discrete_prediction(draft, ctbs, snapshot, phy.tbs)
    dequeue = int(root['dequeue'].sum())
    floor = (RATIO_NUM*dequeue+99)//100
    assert 0 <= floor <= dequeue
    AUDIT.append(dict(tti=int(ctbs.tti)+1, ctbs_pruned_prgs=audit['dropped_count'],
                      raw_blank=int((np.asarray(obs['teacher_action_prg_alloc']) < 0).sum()),
                      aligned_blank=int((draft < 0).sum()), keep_blank=int((draft < 0).sum()),
                      coordinator_blank_token=-1, floor_ratio_num=RATIO_NUM,
                      draft_dequeue_independent=dequeue, causal_floor_bytes=floor))
    DRAFTS.append(draft.copy())
    return draft, draft.copy(), ue, prg, None, -1

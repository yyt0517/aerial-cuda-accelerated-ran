"""A single trajectory, exact top-K validation and decision-local failed edits.

All mutable caches belong to one fresh decision. A changed incumbent discards
the old candidate numbering, features, scores and failed mask. No teacher or
full-neighborhood scan is accessible inside solve().
"""
from dataclasses import dataclass
import hashlib
import time

import numpy as np

from .bounded_probes import ProbePhy
from .edit_ranker import EditFeatureContext,candidate_features
from .physics import EXPECTED,DEQUEUE
from .search import greedy_candidates
from .no_ctbs_inputs import raw_pfq_floor,problem_from_raw_row,raw_pfq_draft
from .own_trajectory_diagnostics import positive_coverage
from .own_trajectory import coverage
from .ranked_dpls_campaign import identity
from .experiment import write_json


@dataclass(frozen=True)
class Config:
    top_k: int = 2
    deduplicate: bool = True
    max_feature_roots: int = 56
    max_candidates: int = 106

    def __post_init__(self):
        if self.top_k not in (1,2,4):raise ValueError('Only fixed top1/top2/top4 are supported')
        if min(self.max_feature_roots,self.max_candidates)<1:raise ValueError('Invalid resources')


def state_key(state_id,root,failed):
    return state_id,hashlib.sha256(root.tobytes()).hexdigest(),hashlib.sha256(np.asarray(failed,np.int64).tobytes()).hexdigest()


def solve(p,draft,config=Config(),ranker=None,*,observer=None):
    phy=ProbePhy(p);context=EditFeatureContext(p,phy)
    try:
        root=phy.evaluate(draft);floor=raw_pfq_floor(root.values[0])
        initial=float(root.values[0,:,EXPECTED]@p.weights);score=initial
        cached=None;accepted=0;feature_roots=0;batches=0;stagnant=0;repeats=0;failed_skips=0
        feature_ms=network_ms=generation_ms=physical_ms=score_ms=0.;generated=0;verified_prgs=set()
        trace=[];stop='physical_budget'
        while phy.stats['candidate_evaluations']<config.max_candidates:
            if cached is None:
                if feature_roots>=config.max_feature_roots:stop='feature_budget';break
                feature_roots+=1
                stamp=time.perf_counter();actions=greedy_candidates(p,root.actions[0]);generated+=len(actions)
                generation_ms+=(time.perf_counter()-stamp)*1000
                stamp=time.perf_counter();x,prior=candidate_features(p,context,root,actions,floor=floor)
                feasible=context.last_dequeue_totals>=floor
                feature_ms+=(time.perf_counter()-stamp)*1000
                stamp=time.perf_counter();prediction=prior if ranker is None else ranker.predict(x,prior)
                if not np.isfinite(prediction).all():raise ValueError('Nonfinite ranking')
                order=np.lexsort((np.arange(len(actions)),-prediction));order=order[feasible[order]]
                network_ms+=(time.perf_counter()-stamp)*1000
                cached=dict(actions=actions,order=order,failed=np.zeros(len(actions),bool),
                    evaluated=np.zeros(len(actions),bool),
                    # Preserve full DPLS's numerical reduction shape/row index.
                    scores=np.zeros((len(actions),p.U,8),np.float64))
            actions=cached['actions'];order=cached['order'];failed=cached['failed']
            failed_before=np.flatnonzero(failed)
            if config.deduplicate:
                available=order[~failed[order]];failed_skips+=int(failed[order].sum())
            else:available=order
            if not len(available):stop='finite_domain_exhausted';break
            remaining=config.max_candidates-phy.stats['candidate_evaluations']
            selected=available[:min(config.top_k,remaining)]
            if config.deduplicate and cached['evaluated'][selected].any():raise AssertionError('Repeated exact candidate at the same root')
            repeats+=int(cached['evaluated'][selected].sum());cached['evaluated'][selected]=True
            stamp=time.perf_counter();exact=phy.evaluate(actions[selected],root)
            physical_ms+=(time.perf_counter()-stamp)*1000
            stamp=time.perf_counter();cached['scores'][selected]=exact.values
            js=(cached['scores'][:,:,EXPECTED]@p.weights)[selected]
            maximum=float(js.max());tied=selected[js==maximum];best_index=int(tied.min())
            index=int(np.flatnonzero(selected==best_index)[0])
            # Match the previous scalar generator: every accepted edit >1 byte.
            # Canonical full DPLS's later16B threshold stays in its own control.
            take=maximum>score+1.
            score_ms+=(time.perf_counter()-stamp)*1000
            batches+=1;stagnant+=not take
            for candidate in actions[selected]:verified_prgs.update(map(int,np.flatnonzero((candidate!=root.actions[0]).any(0))))
            signature=hashlib.sha256(root.actions.tobytes()).hexdigest()
            trace.append(dict(batch=batches,root_sha256=signature,selected=selected.tolist(),
                failed_before=failed_before.tolist(),accepted=bool(take),best_index=best_index if take else -1,
                best_gain=maximum-score))
            if observer is not None:
                observer(dict(root=root.actions[0].copy(),failed_before=failed_before.copy(),
                    selected=selected.copy(),root_sha256=signature,batch=batches,ordinal=batches,
                    floor=floor,accepted=bool(take),path_edits=accepted))
            if take:
                root=phy.take(exact,index);score=float(root.values[0,:,EXPECTED]@p.weights)
                accepted+=1;cached=None
            else:
                # All members of this batch failed the declared >1B test.
                failed[selected]=True
        result=dict(score=float(root.values[0,:,EXPECTED]@p.weights),initial_score=initial,floor=floor,
            top_k=config.top_k,deduplicate=config.deduplicate,feature_roots=feature_roots,
            verification_batches=batches,accepted_edits=accepted,stagnant_batches=stagnant,
            repeated_exact_candidates=repeats,failed_candidate_skips=failed_skips,
            generated_candidates=generated,verified_prgs=len(verified_prgs),
            feature_ms=feature_ms,network_ms=network_ms,generation_ms=generation_ms,physical_ms=physical_ms,
            canonical_score_ms=score_ms,stop_reason=stop,full_search_calls=0,
            trace=trace,**phy.stats)
        assert result['candidate_evaluations']<=config.max_candidates
        assert feature_roots<=config.max_feature_roots
        assert not config.deduplicate or repeats==0
        assert root.values[0,:,DEQUEUE].sum()>=floor
        return root.actions[0].copy(),root.values[0].copy(),result
    finally:context.close();phy.close()


def service_metrics(values,reference):
    return dict(**coverage(values,reference),**positive_coverage(values,reference))


def collect_own(rows,model,config,out,max_states=8):
    import torch
    groups=[];work=[];visits=[]
    for row in rows:
        p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs']);observed=[]
        action,values,student=solve(p,draft,config,model,observer=observed.append)
        unique={}
        for v in observed:unique.setdefault(state_key(identity(row),v['root'],v['failed_before']),v)
        unique=list(unique.values())
        chosen=np.unique(np.linspace(0,len(unique)-1,min(max_states,len(unique)),dtype=int)) if unique else []
        teacher=ProbePhy(p);context=EditFeatureContext(p,teacher)
        try:
            for sample in chosen:
                visit=unique[int(sample)];base=teacher.evaluate(visit['root'])
                actions=greedy_candidates(p,visit['root'])
                x,prior=candidate_features(p,context,base,actions,floor=student['floor'])
                exact=teacher.evaluate(actions,base)
                gain=exact.values[:,:,EXPECTED]@p.weights-base.values[0,:,EXPECTED]@p.weights
                eligible=exact.values[:,:,DEQUEUE].sum(1)>=student['floor']
                np.testing.assert_array_equal(eligible,context.last_dequeue_totals>=student['floor'])
                eligible[visit['failed_before']]=False
                key=state_key(identity(row),visit['root'],visit['failed_before'])
                groups.append(dict(state_id=identity(row),seed=row['seed'],iteration=visit['batch'],
                    x48=x,prior=prior,gain=gain,eligible=eligible,root=visit['root'].copy(),floor=student['floor'],
                    failed_before=visit['failed_before'].copy(),root_sha256=key[1],failed_sha256=key[2]))
                visits.append(dict(state_id=identity(row),seed=row['seed'],root_sha256=key[1],failed_sha256=key[2],
                    batch=visit['batch'],failed_candidates=len(visit['failed_before']),
                    actual_visited=True,candidate_labels=len(gain)))
            teacher_work=teacher.stats
        finally:context.close();teacher.close()
        again=solve(p,draft,config,model)
        np.testing.assert_array_equal(action,again[0])
        assert student['candidate_evaluations']==again[2]['candidate_evaluations']
        work.append(dict(state_id=identity(row),student={k:v for k,v in student.items() if k!='trace'},teacher=teacher_work,
            actual_visits=len(observed),unique_states=len(unique),sampled_states=len(chosen)))
        print('OWN_SHORTLIST',identity(row),len(unique),len(chosen),flush=True)
    torch.save(groups,out/'own_states.pt')
    write_json(out/'own_work.json',dict(work=work,visits=visits,groups=len(groups),
        candidate_labels=sum(len(g['gain']) for g in groups),post_decision_teacher=True,student_replay_identical=True))
    return groups,visits

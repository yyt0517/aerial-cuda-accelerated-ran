"""Post-selection terminal-root audit; no new online capability or training."""
import argparse
import json
from pathlib import Path

import numpy as np

from .shortlist_trajectory import Config,solve
from .bounded_probes import Predictor,ProbePhy,csv_write
from .physics import EXPECTED,DEQUEUE
from .edit_ranker import EditFeatureContext,candidate_features
from .search import greedy_candidates
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .ranked_dpls_campaign import read_rows,identity
from .experiment import write_json,sha


def main(source):
    import torch
    torch.set_num_threads(1);source=Path(source).resolve();out=source/'terminal_diagnostics';out.mkdir(exist_ok=False)
    protocol=json.loads((source/'protocol.json').read_text());selected=json.loads((source/'selection.json').read_text())
    write_json(out/'protocol.json',dict(post_selection=True,retraining=False,checkpoint_selection=False,
        selection_sha256=sha(source/'selection.json'),roles=['validation','evaluation'],
        models=['initial','trained'],top_k=[2,4],
        audit='replay fixed complete decisions; query terminal full original neighborhood independently with failed-after mask',
        improvement_threshold=16,scope='terminal one-step opportunity only; not a global bound or free online teacher'))
    initial=Predictor(protocol['initial']['checkpoint']);trained=Predictor(selected['checkpoint'])
    results=[];work=[]
    for role,seeds in [('validation',(44,45)),('evaluation',(46,47,48))]:
        rows,_=read_rows(seeds,evaluation=role=='evaluation')
        ttis=protocol['evaluation_ttis'] if role=='evaluation' else protocol['train_validation_ttis']
        rows=[r for r in rows if r['tti'] in ttis]
        phase='phase_a' if role=='validation' else 'evaluation'
        with np.load(source/f'{phase}_actions.npz') as actions_saved:
            for row in rows:
                p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs'])
                for label,model in [('initial',initial),('trained',trained)]:
                    for k in (2,4):
                        action,values,r=solve(p,draft,Config(k),model)
                        saved_key=f'{identity(row)}|{label}_{k}'
                        if saved_key in actions_saved:
                            np.testing.assert_array_equal(action,actions_saved[saved_key])
                        last=r['trace'][-1] if r['trace'] else None
                        failed=(np.r_[last['failed_before'],last['selected']].astype(np.int64)
                            if last is not None and not last['accepted'] else np.empty(0,np.int64))
                        phy=ProbePhy(p);context=EditFeatureContext(p,phy)
                        try:
                            root=phy.evaluate(action);candidates=greedy_candidates(p,action)
                            x,prior=candidate_features(p,context,root,candidates,floor=r['floor'])
                            prediction=model.predict(x,prior)
                            exact=phy.evaluate(candidates,root)
                            gain=exact.values[:,:,EXPECTED]@p.weights-root.values[0,:,EXPECTED]@p.weights
                            eligible=exact.values[:,:,DEQUEUE].sum(1)>=r['floor']
                            eligible[failed]=False
                            indices=np.flatnonzero(eligible)
                            order=indices[np.lexsort((indices,-prediction[indices]))]
                            available_gain=gain[order]
                            positive=np.flatnonzero(available_gain>16)
                            record=dict(role=role,state_id=identity(row),seed=row['seed'],arm=f'{label}_{k}',
                                stop_reason=r['stop_reason'],feature_roots=r['feature_roots'],
                                candidate_evaluations=r['candidate_evaluations'],accepted_edits=r['accepted_edits'],
                                failed_at_terminal=len(failed),remaining_legal=len(order),
                                best_gain=float(max(0,available_gain.max())) if len(order) else 0.,
                                first_positive_rank=int(positive[0]+1) if len(positive) else -1,
                                top2_best_gain=float(max(0,available_gain[:2].max())) if len(order) else 0.,
                                top4_best_gain=float(max(0,available_gain[:4].max())) if len(order) else 0.)
                            results.append(record);work.append(dict(role=role,state_id=identity(row),arm=record['arm'],
                                student_replay_scores=r['candidate_evaluations'],teacher=phy.stats))
                        finally:context.close();phy.close()
    csv_write(out/'terminal_opportunities.csv',results)
    write_json(out/'work.json',dict(replay_and_teacher_work=work,models_unchanged=True))
    print('TERMINAL AUDIT COMPLETE',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);main(p.parse_args().source)

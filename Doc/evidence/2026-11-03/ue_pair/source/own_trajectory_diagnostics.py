"""Post-selection stall/coverage audit; no training or checkpoint selection."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .own_trajectory import generate,DEEP
from .bounded_probes import Predictor,ProbePhy,csv_write,seed_mean
from .physics import EXPECTED,COUNT
from .edit_ranker import EditFeatureContext,candidate_features
from .search import greedy_candidates
from .ranked_dpls_campaign import read_rows,identity
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .experiment import write_json,sha


def positive_coverage(actual,reference):
    target=reference[:,EXPECTED];service=actual[:,EXPECTED]
    served=np.flatnonzero(target>1.)
    if not len(served):return dict(reference_positive_ues=0,positive_p5_ratio=1.,positive_weak_ratio=1.,
        all_ue_reference_p5=float(np.quantile(target,.05)),all_ue_p5_degenerate=True)
    weak=served[np.argsort(target[served],kind='stable')[:max(1,int(np.ceil(.2*len(served))))]]
    return dict(reference_positive_ues=len(served),
        positive_p5_ratio=float(np.quantile(service[served],.05)/np.quantile(target[served],.05)),
        positive_weak_ratio=float(service[weak].sum()/target[weak].sum()),
        all_ue_reference_p5=float(np.quantile(target,.05)),all_ue_p5_degenerate=bool(np.quantile(target,.05)<=1.e-9))


def main(source):
    import torch
    torch.set_num_threads(1);source=Path(source).resolve();out=source/'diagnostics';out.mkdir(exist_ok=False)
    sel=json.loads((source/'selection.json').read_text());protocol=json.loads((source/'protocol.json').read_text())
    write_json(out/'protocol.json',dict(post_selection=True,retraining=False,checkpoint_selection=False,
        selection_sha256=sha(source/'selection.json'),
        roles=['validation','evaluation'],students=['initial_deep','trained_deep'],
        stall_probe='most frequently visited failed root, one/student/state; exact full neighborhood queried only after decisions',
        k_diagnostics=[1,2,4,8],positive_coverage='P5 and bottom20% among reference expected bytes>1B',
        gates_unchanged=True,limits='posthoc development diagnostics, not new blind tests or an online speedup'))
    initial=Predictor(protocol['prior_scalar']['checkpoint']);trained=Predictor(sel['checkpoint'])
    stalls=[];work=[];coverage_rows=[]
    for role,seeds in [('validation',(44,45)),('evaluation',(46,47,48))]:
        rows,_=read_rows(seeds,evaluation=role=='evaluation')
        ttis=protocol['evaluation_ttis'] if role=='evaluation' else protocol['train_validation_ttis']
        rows=[r for r in rows if r['tti'] in ttis]
        phase='phase_a' if role=='validation' else 'evaluation'
        with np.load(source/f'{phase}_values.npz') as values:
            for row in rows:
                sid=identity(row);reference=values[f'{sid}|full']
                for key in values.files:
                    if key.startswith(sid+'|'):
                        coverage_rows.append(dict(role=role,state_id=sid,seed=row['seed'],arm=key.split('|')[-1],
                            **positive_coverage(values[key],reference)))
        for row in rows:
            p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs'])
            for name,model in [('initial_deep',initial),('trained_deep',trained)]:
                observed=[];a,v,r=generate(p,draft,DEEP,model,observer=observed.append)
                failed={}
                for visit in observed:
                    if visit['accepted']==0:failed.setdefault(visit['fingerprint'],[]).append(visit)
                if not failed:
                    stalls.append(dict(role=role,state_id=identity(row),seed=row['seed'],arm=name,has_stall=False))
                    continue
                repeated=max(failed.values(),key=len);visit=repeated[0]
                phy=ProbePhy(p);context=EditFeatureContext(p,phy)
                try:
                    base=phy.evaluate(visit['root']);actions=greedy_candidates(p,visit['root'])
                    x,prior=candidate_features(p,context,base,actions,floor=r['floor'])
                    prediction=model.predict(x,prior);exact=phy.evaluate(actions,base)
                    gain=exact.values[:,:,EXPECTED]@p.weights-base.values[0,:,EXPECTED]@p.weights
                    eligible=exact.values[:,:,3].sum(1)>=r['floor']
                    order=np.lexsort((np.arange(len(actions)),-prediction));order=order[eligible[order]]
                    best=float(gain[eligible].max()) if eligible.any() else 0.
                    record=dict(role=role,state_id=identity(row),seed=row['seed'],arm=name,
                        has_stall=True,repeated_failed_visits=len(repeated),root_sha256=visit['fingerprint'],
                        best_gain=best,missed_qualifying_edit=best>16.,student_depth=r['depth'],
                        actual_successful_edits=r['best_path_edits'])
                    for k in (1,2,4,8):record[f'top{k}_best_gain']=float(gain[order[:k]].max()) if len(order) else 0.
                    stalls.append(record);work.append(dict(role=role,state_id=identity(row),arm=name,**phy.stats))
                finally:context.close();phy.close()
    csv_write(out/'stalls.csv',stalls);csv_write(out/'positive_coverage.csv',coverage_rows)
    write_json(out/'work.json',dict(teacher=work,models_unchanged=True))
    print('DIAGNOSTICS COMPLETE',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);main(p.parse_args().source)

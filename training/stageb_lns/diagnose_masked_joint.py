"""Post-evaluation candidate coverage audit. No fitting or selection occurs."""
import argparse
import json
from pathlib import Path
import numpy as np

from .archive import problem_from_row,initial_draft
from .physics import NativePhy,EXPECTED,DEQUEUE
from .search import ProposalGraph
from .masked_joint import JointContext,MaskedJointRepair
from .masked_search import masks
from .train_masked_joint import teacher_pool
from .eval_masked_joint import evaluation_rows
from .experiment import write_json,sha


def main(args):
    import pandas as pd
    root=Path(args.training).resolve();evaluation=root/'evaluation_mask_reuse'
    out=root/'diagnosis';out.mkdir(exist_ok=False)
    checkpoint=root/'repair.npz';digest=sha(checkpoint)
    rows,_=evaluation_rows(json.loads((root/'protocol.json').read_text()))
    roots=np.load(evaluation/'escape_roots.npz')
    decoder=MaskedJointRepair(checkpoint);records=[]
    for i,row in enumerate(rows):
        p=problem_from_row(row);draft=initial_draft(p,row['obs']);action=roots[f'{i}_strict_greedy']
        phy=NativePhy(p)
        try:
            original=phy.evaluate(draft);floor=(90*int(original.values[0,:,DEQUEUE].sum())+99)//100
            base=phy.evaluate(action);base_score=float(base.values[0,:,EXPECTED]@p.weights)
            graph=ProposalGraph(phy);context=JointContext(phy,graph);visited=set()
            stages=[[np.arange(p.C*p.P)],masks(graph,base,visited=visited),masks(graph,base,visited=visited)]
            rng=np.random.default_rng(1307)
            for stage,blocks in enumerate(stages,1):
                generated,traces=decoder.propose_many(context,base,floor,blocks,rng,return_trace=True)
                for trace in traces:
                    candidates=trace['actions'];block=trace['block']
                    checked=phy.evaluate(candidates,base)
                    scores=checked.values[:,:,EXPECTED]@p.weights
                    feasible=checked.values[:,:,DEQUEUE].sum(1)>=floor
                    neural_best=max(base_score,float(scores[feasible].max())) if feasible.any() else base_score
                    teacher,cost=teacher_pool(phy,graph,base,block,floor,
                               np.random.default_rng(9401+i*10+stage),candidate_limit=8192)
                    threshold=1. if stage<=2 else 16.
                    records.append(dict(state_id=row['state_id'],seed=row['seed'],stage=stage,mask_size=len(block),
                           candidates=len(candidates),unique_candidates=len(np.unique(candidates.reshape(len(candidates),-1),axis=0)),
                           floor_violations=int((~feasible).sum()),
                           objective_improvements=int((scores>base_score+threshold).sum()),
                           acceptable_improvements=int((feasible & (scores>base_score+threshold)).sum()),
                           neural_gain=neural_best-base_score,teacher_gain=teacher[0][0]-base_score,
                           teacher_positive=teacher[0][0]>base_score+threshold,**cost))
        finally:phy.close()
        print('DIAGNOSE',i+1,len(rows),flush=True)
    frame=pd.DataFrame(records);frame.to_csv(out/'mask_coverage.csv',index=False)
    per=frame.groupby('state_id').agg(neural_best_gain=('neural_gain','max'),
         teacher_best_gain=('teacher_gain','max'),teacher_positive_masks=('teacher_positive','sum')).reset_index()
    per.to_csv(out/'per_state.csv',index=False)
    write_json(out/'summary.json',dict(checkpoint_unchanged=sha(checkpoint)==digest,
          states=len(rows),masks=len(frame),neural_positive_masks=int((frame.acceptable_improvements>0).sum()),
          teacher_positive_masks=int(frame.teacher_positive.sum()),
          states_with_teacher_improvement=int((per.teacher_positive_masks>0).sum()),
          mean_unique_of_8=float(frame.unique_candidates.mean()),floor_violation_fraction=float(frame.floor_violations.sum()/frame.candidates.sum()),
          teacher_candidate_evaluations=int(frame.candidate_evaluations.sum()),
          interpretation='post-hoc diagnosis of the same masks on strict greedy roots; larger offline teacher work is not inference acceleration; no labels feed back into training'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--training',required=True)
    main(parser.parse_args())

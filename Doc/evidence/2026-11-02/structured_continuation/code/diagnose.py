"""Post-hoc work-allocation replay, without fitting or changing any policy."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

from training.stageb_lns.archive import problem_from_row, initial_draft
from training.stageb_lns.structured_campaign import evaluation_rows, FROZEN79, CombinedRepair
from training.stageb_lns.structured_repair import StructuredRepair
from training.stageb_lns.continuation_repair import ContinuationRepair, ContinuationValue
from training.stageb_lns.cached_repair import CachedJointRepair
from training.stageb_lns.physics import NativePhy, EXPECTED, DEQUEUE
from training.stageb_lns.quality import JOINT
from training.stageb_lns.search import solve
from dataclasses import replace


class WorkRecorder:
    deterministic=True

    def __init__(self, inner):
        self.inner=inner
        self.blocks=[]

    def repair(self, graph, base, block, evaluate):
        item=dict(visit=len(self.blocks),block=[list(t) for t in block],
                  requests=0,verified=0,batches=0)
        def score(actions, source):
            item['requests']+=len(actions)
            item['batches']+=1
            result=evaluate(actions,source)
            if result is not None:item['verified']+=len(result.actions)
            return result
        if hasattr(self.inner,'repair'):
            self.inner.repair(graph,base,block,score)
        else:
            score(self.inner.propose(graph,base,block),base)
        self.blocks.append(item)


def main(args):
    source=Path(args.source).resolve()
    selection=json.loads((source/'selection.json').read_text())
    rows,_=evaluation_rows()
    # Deterministic source/seed coverage, selected after the main experiment.
    rows=[(i,r) for i,r in enumerate(rows) if (r['seed'],r['family'],r['tti']) in
          ((46,'pfq0',3001),(47,'neural',3001),(48,'greedy',3001))]
    actions=np.load(source/'actions.npz',allow_pickle=False)
    frozen=pd.read_csv(source/'per_state.csv')
    records=[]
    for row_index,row in rows:
        p=problem_from_row(row);draft=initial_draft(p,row['obs'])
        phy=NativePhy(p)
        try:floor=(90*int(phy.evaluate(draft).values[0,:,DEQUEUE].sum())+99)//100
        finally:phy.close()
        for arm in ('cached79','current','rollout','structure','learned','combined'):
            if arm=='cached79':inner=CachedJointRepair(FROZEN79)
            elif arm=='structure':inner=StructuredRepair(floor,max_blocks=8)
            else:
                inner=ContinuationRepair(FROZEN79,floor,
                    mode='learned' if arm in ('learned','combined') else arm,
                    value=ContinuationValue(selection['selected']) if arm in ('learned','combined') else None,
                    shortlist=32,horizon=2,rollout_width=2)
                if arm=='combined':inner=CombinedRepair(inner,StructuredRepair(floor,max_blocks=4))
            recorder=WorkRecorder(inner)
            action,_,record=solve(p,draft,replace(JOINT,max_candidates=4095),repairer=recorder)
            np.testing.assert_array_equal(action,actions[f'{row_index}_{arm}'])
            state=f"{row['family']}:seed{row['seed']}:tti{row['tti']}"
            expected=frozen[(frozen.state_id==state)&(frozen.arm==arm)].iloc[0]
            if abs(record['score']-expected.score)>1.e-6 or record['candidate_evaluations']+1!=expected.candidate_evaluations:
                raise AssertionError('Diagnostic instrumentation changed the frozen decision/work')
            total=sum(x['verified'] for x in recorder.blocks)
            if total+1!=record['candidate_evaluations']:
                raise AssertionError('Per-block scored work does not reconcile with native total')
            records.append(dict(state_id=state,arm=arm,score=record['score'],
                scored_blocks=sum(x['verified']>0 for x in recorder.blocks),
                total_scored=total,first_four_scored=sum(x['verified'] for x in recorder.blocks[:4]),
                stop_reason=record['stop_reason'],blocks=recorder.blocks))
    (source/'work_allocation.json').write_text(json.dumps(dict(post_hoc=True,
        no_training_or_policy_change=True,instrumentation_actions_and_work_exact=True,records=records),indent=2)+'\n')
    pd.DataFrame([{k:v for k,v in r.items() if k!='blocks'} for r in records]).to_csv(source/'work_allocation.csv',index=False)
    print(pd.DataFrame([{k:v for k,v in r.items() if k!='blocks'} for r in records]).to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    main(parser.parse_args())

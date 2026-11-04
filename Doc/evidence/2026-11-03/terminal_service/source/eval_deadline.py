"""Paired saved-state validation, with separate physical and risk objectives."""
import argparse
from pathlib import Path
import numpy as np
from .closed_data import dataset
from .archive import problem_from_row,initial_draft
from .deadline import DeadlineRepair,components
from .neural_repair import JointRepair
from .quality import select
from .experiment import write_json,sha


def main(args):
    import pandas as pd
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    rows,_=dataset(args.collection,(48,50));rows=rows[::args.stride]
    old=JointRepair(args.initial);new=DeadlineRepair(args.checkpoint);records=[]
    for i,row in enumerate(rows):
        p=problem_from_row(row);draft=initial_draft(p,row['obs'])
        for name,arm,model in [('greedy','greedy',None),('old_neural','neural',old),
                               ('new_j_only','neural',new),('risk_neural','risk_neural',new)]:
            _,values,result=select(p,draft,arm,model)
            j,u,ttl,weak=components(p,values)
            records.append(dict(result,method=name,seed=row['seed'],tti=row['tti'],source=row['source'],
                packet_proxy=float(u[0]),urgent_shortfall=float(ttl[0]),weak_expected=float(weak[0])))
        if i%10==0:print('DEADLINE_VALIDATE',i,len(rows),flush=True)
    frame=pd.DataFrame(records);frame.to_csv(output/'measurements.csv',index=False)
    summary=frame.groupby('method')[['improvement','packet_proxy','urgent_shortfall','weak_expected','core_ms']].mean()
    summary.to_csv(output/'summary.csv')
    write_json(output/'protocol.json',dict(validation_seeds=[48,50],states=len(rows),stride=args.stride,
        checkpoint_sha256=sha(args.checkpoint),initial_sha256=sha(args.initial),scope='saved validation states, not native outcomes'))
    print(summary.to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--collection',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--initial',required=True)
    p.add_argument('--stride',type=int,default=2);main(p.parse_args())

"""Validation-only selection and attribution for the joint repair checkpoint."""
import argparse
from pathlib import Path
import time
import numpy as np
from .archive import load_rows, problem_from_row, initial_draft
from .neural_repair import JointRepair
from .quality import select
from .experiment import write_json, sha, snapshot_sources


def main(args):
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    sources = snapshot_sources(output)
    checkpoint = Path(args.checkpoint).resolve()
    records = []
    for row in load_rows(tuple(map(int, args.seeds.split(',')))):
        p = problem_from_row(row); draft = initial_draft(p, row['obs'])
        for arm in ('greedy', 'neural', 'hybrid', 'physical_lns'):
            repairer = JointRepair(checkpoint, args.width)
            started = time.perf_counter()
            _, _, result = select(p, draft, arm, repairer)
            result.update(seed=row['seed'], tti=row['tti'], width=args.width,
                          total_ms=(time.perf_counter()-started)*1000,
                          neural_calls=repairer.calls, neural_proposals=repairer.proposals)
            records.append(result)
            print('VALIDATION', row['seed'], row['tti'], arm, round(result['score'],1),
                  round(result['total_ms'],1), flush=True)
    import pandas as pd
    pd.DataFrame(records).to_csv(output/'measurements.csv', index=False)
    summary = []
    frame = pd.DataFrame(records)
    base = frame[frame.arm == 'greedy'].set_index(['seed','tti'])
    for arm, group in frame.groupby('arm'):
        group = group.set_index(['seed','tti']).loc[base.index]
        summary.append(dict(arm=arm, mean_gain=float(group.improvement.mean()),
            gain_retained_ratio=float(group.improvement.sum()/base.improvement.sum()),
            mean_delta_over_greedy=float((group.score-base.score).mean()),
            improved_states=int((group.score>base.score+1).sum()),
            worse_states=int((group.score<base.score-1).sum()),
            mean_ms=float(group.total_ms.mean()), mean_candidates=float(group.candidate_evaluations.mean())))
    write_json(output/'summary.json', summary)
    write_json(output/'protocol.json', dict(seeds=args.seeds, checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint), width=args.width, source_sha256=sources,
        scope='saved-state validation; no closed-loop claim; no time cutoff'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--seeds',default='44,45')
    parser.add_argument('--width',type=int,default=16)
    main(parser.parse_args())

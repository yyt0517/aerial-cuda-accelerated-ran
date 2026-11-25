"""Serial same-state timing after quality trajectories finish.

Each source arm contributes its own saved states. Exact/approximate evaluators
receive identical queue, weights, PFQ initialization, serving CSI and native
MCS/layer state. Reports come from the matching seed/time's causal measurement
stream, whose original H is checked bit-for-bit. No concurrent trajectory time
is used as an acceleration claim.
"""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from training.stageb_ppo.native import write, sha
from training.stageb_lns.archive import enable_legacy_types
from training.stageb_lns.no_ctbs_inputs import problem_from_raw_row, raw_pfq_draft
from training.stageb_lns.bounded_dpls import solve_bounded
from .analyze import local
from .model import approximate_problem


def main(args):
    import torch
    torch.set_num_threads(1);enable_legacy_types()
    source=Path(args.source);output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    assert json.loads((source/'completion.json').read_text())['completed']
    available=sorted(os.sched_getaffinity(0));cpu=args.cpu if args.cpu is not None else available[-1]
    os.sched_setaffinity(0,{cpu})
    specs=[json.loads(p.read_text()) for p in sorted(source.glob('*/done.json'))]
    records=[];checks=[]
    for spec in specs:
        d=local(spec['directory'])
        for sample in json.loads((d/'saved_states.json').read_text()):
            path=local(sample['path']);assert sha(path)==sample['sha256']
            row=torch.load(path,map_location='cpu',weights_only=False)
            report_key=f"{spec['scene']}_r{spec['rate']:g}_rsrp_deficit_s{spec['seed']}"
            report_row=torch.load(source/report_key/'states'/path.name,map_location='cpu',weights_only=False)
            np.testing.assert_array_equal(row['snapshot']['h'],report_row['snapshot']['h'])
            report=report_row['rsrp_report']
            original_h=row['snapshot']['h'].copy()
            expected={}
            def evaluate(mode):
                p=problem_from_raw_row(row);p.weights=np.asarray(row['objective_weights']).copy()
                draft=raw_pfq_draft(p,row['obs'])
                if mode=='rsrp':p=approximate_problem(p,report)
                return solve_bounded(p,draft)
            for mode in ('exact','rsrp'):
                result=evaluate(mode);assert result[2]['converged'];expected[mode]=result
            own_mode='rsrp' if spec['arm'].startswith('rsrp_') else 'exact'
            np.testing.assert_array_equal(expected[own_mode][0],row['selected_action'])
            np.testing.assert_array_equal(expected[own_mode][1],row['selected_values'])
            for repeat in range(args.repeats):
                modes=('exact','rsrp') if repeat%2==0 else ('rsrp','exact')
                for mode in modes:
                    start=time.perf_counter();start_cpu=time.process_time()
                    result=evaluate(mode)
                    cpu_ms=(time.process_time()-start_cpu)*1000;wall_ms=(time.perf_counter()-start)*1000
                    np.testing.assert_array_equal(result[0],expected[mode][0])
                    np.testing.assert_array_equal(result[1],expected[mode][1])
                    records.append(dict(scene=spec['scene'],seed=spec['seed'],source_arm=spec['arm'],
                        objective=spec['arm'].removeprefix('rsrp_'),tti=int(row['tti']),mode=mode,repeat=repeat,
                        wall_ms=wall_ms,cpu_ms=cpu_ms,candidate_evaluations=result[2]['candidate_evaluations'],
                        mmse_misses=result[2]['mmse_misses'],ue_evaluations=result[2]['ue_evaluations']))
            np.testing.assert_array_equal(row['snapshot']['h'],original_h)
            checks.append(dict(source=str(path),sha256=sha(path),saved_decision_reproduced=True,
                changed_positions=int(np.sum(expected['exact'][0]!=expected['rsrp'][0])),passed=True))
            print('TIMED',spec['key'],row['tti'],flush=True)
    df=pd.DataFrame(records);df.to_csv(output/'timing_raw.csv',index=False)
    # Both source trajectories and three timepoints receive equal weight within seed.
    metrics=['wall_ms','cpu_ms','candidate_evaluations','mmse_misses','ue_evaluations']
    per_seed=df.groupby(['scene','seed','objective','mode'],as_index=False)[metrics].mean()
    per_seed.to_csv(output/'timing_per_seed.csv',index=False)
    summary=per_seed.groupby(['scene','objective','mode'],as_index=False)[metrics].mean()
    summary.to_csv(output/'timing_summary.csv',index=False)
    # Explicitly separate performance at each algorithm's own visited states.
    own=df[df['mode']==np.where(df.source_arm.str.startswith('rsrp_'),'rsrp','exact')]
    own_seed=own.groupby(['scene','seed','source_arm'],as_index=False)[metrics].mean()
    own_seed.to_csv(output/'own_trajectory_per_seed.csv',index=False)
    own_seed.groupby(['scene','source_arm'],as_index=False)[metrics].mean().to_csv(output/'own_trajectory_summary.csv',index=False)
    write(output/'validation.json',dict(passed=True,cpu=cpu,states=len(checks),repeats=args.repeats,checks=checks,
        scope='serial saved-state reconstruction, fixed recorded weights, PFQ decode, RSRP model conversion, context/search/destruction',
        exclusions='GPU/native PFQ, bridge, FIFO HOL scan, report measurement and transport; measurement emulation separately logged online',
        same_solver=True,closed_loop_quality_campaign_finished=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--repeats',type=int,default=5);parser.add_argument('--cpu',type=int)
    main(parser.parse_args())

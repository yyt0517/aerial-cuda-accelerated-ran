"""Separately charged native cost decomposition on completed stress inputs."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .physics import ROOT,curves
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .bounded_dpls import solve_bounded,BoundedConfig,bounded_library
from .analyze_bounded_stress import local
from .analyze_dpls_moves import write_csv,write_json,sha

def main(args):
    import torch
    from .archive import enable_legacy_types
    enable_legacy_types();os.sched_setaffinity(0,{255})
    source,out=Path(args.source).resolve(),Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    done=json.loads((source/'completion.json').read_text());assert done['completed']
    rows=[];inputs={};bounded_library();curves()
    first=local(done['runs'][0]['directory'])
    state=local(json.loads((first/'saved_states.json').read_text())[0]['path'])
    problem_from_raw_row(torch.load(state,map_location='cpu',weights_only=False))
    for run in done['runs']:
        directory=local(run['directory'])
        with np.load(directory/'actions.npz') as z:actions=z['actions'].copy()
        for item in json.loads((directory/'saved_states.json').read_text()):
            path=local(item['path']);assert sha(path)==item['sha256'];inputs[str(path)]=sha(path)
            raw=torch.load(path,map_location='cpu',weights_only=False)
            start=time.perf_counter();p=problem_from_raw_row(raw);draft=raw_pfq_draft(p,raw['obs'])
            a,v,r=solve_bounded(p,draft,BoundedConfig(diagnostics=True))
            elapsed=1000*(time.perf_counter()-start)
            np.testing.assert_array_equal(a,actions[raw['tti']-1])
            rows.append(dict(rate=run['rate'],seed=run['seed'],tti=raw['tti'],e2e_ms=elapsed,**r))
    table=pd.DataFrame(rows);summary=table.groupby('rate').mean(numeric_only=True)
    summary.to_csv(out/'summary.csv');write_csv(out/'profiles.csv',rows)
    write_json(out/'input_manifest.json',inputs)
    write_json(out/'source_manifest.json',{str(Path(__file__).relative_to(ROOT)):sha(__file__)})
    write_json(out/'completion.json',dict(completed=True,profiled_decisions=len(rows),
        profiled_candidate_evaluations=sum(r['candidate_evaluations'] for r in rows),
        new_native_runs=0,immutable_adapter_table_warmed=True,
        interpretation='separate profiling, not fair timed decisions; all actions match completed native trajectories'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);main(p.parse_args())

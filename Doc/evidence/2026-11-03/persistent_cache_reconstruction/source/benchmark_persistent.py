"""Fair native full-refresh/cache comparison plus old Python controls."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from .physics import ROOT,library,curves
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .cached_dpls import solve_canonical,CanonicalDplsConfig
from .persistent_dpls import solve_persistent,solve_integer_reference,PersistentConfig,persistent_library
from .analyze_dpls_moves import write_json,write_csv,sha

ARMS=('float_python_full','integer_python_full','full','cached','cap4','guarded4')


def main(args):
    import torch
    from .archive import enable_legacy_types
    enable_legacy_types()
    source,gate,out=map(lambda s:Path(s).resolve(),(args.source,args.gate,args.output))
    if not json.loads((gate/'completion.json').read_text())['integer_equivalence_passed']:raise ValueError('Integer gate required')
    out.mkdir(parents=True,exist_ok=False);os.sched_setaffinity(0,{args.cpu})
    paths=sorted(source.glob('*_seed*/states/tti*.pt'));rows=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
    library();persistent_library();curves()
    for row in rows:problem_from_raw_row(row)
    files=[Path(__file__),*[Path(__file__).with_name(n) for n in ('persistent_dpls.py','persistent_kernel.cpp',
        '_persistent.so','physics_kernel.cpp','physics.py','_physics.so','cached_dpls.py','dpls_operations.py')]]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files};(out/'source').mkdir()
    for p in files:shutil.copy2(p,out/'source'/p.name)
    write_json(out/'source_manifest.json',hashes)
    write_json(out/'input_manifest.json',{str(p):sha(p) for p in paths})
    write_json(out/'protocol.json',dict(arms=list(ARMS),repeats=args.repeats,cpu_affinity=[args.cpu],
        timing='single-process alternating/rotating arms; fresh metadata -> PHY/context -> search/close; immutable imports/table/library outside',
        profiling=False,observers=False,audits_inside_timing=False,history=False,
        reference='report both native full-refresh and old Python full; compile/persistent generation savings are not independent cache savings',
        source=str(source),gate=str(gate),shared_host=True))
    measurements=[];answers=[]
    def run(row,arm,profile=False):
        started=time.perf_counter();p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs'])
        warm=row['previous_action'] if row['source_arm'].startswith('warm_') else None
        if arm=='float_python_full':a,v,r=solve_canonical(p,draft,CanonicalDplsConfig('full',diagnostics=profile),warm_start=warm)
        elif arm=='integer_python_full':a,v,r=solve_integer_reference(p,draft,warm_start=warm)
        else:a,v,r=solve_persistent(p,draft,PersistentConfig(arm,diagnostics=profile),warm_start=warm)
        return (time.perf_counter()-started)*1000,(a,v,r)
    for repeat in range(args.repeats):
        order=list(range(len(rows)));order=order[::-1] if repeat%2 else order
        for state in order:
            offset=(repeat+state)%len(ARMS);arms=list(ARMS[offset:]+ARMS[:offset])
            if (repeat+state)%2:arms.reverse()
            for arm in arms:
                elapsed,result=run(rows[state],arm)
                measurements.append(dict(state=state,arm=arm,repeat=repeat,e2e_ms=elapsed,
                    candidate_evaluations=result[2]['candidate_evaluations'],mmse_misses=result[2]['mmse_misses']))
                answers.append((state,arm,result[0],result[1]))
        print('PERSISTENT_TIMED',repeat+1,flush=True)
    for state,arm,a,v in answers:
        name=arm if arm in ('full','cached','cap4','guarded4') else 'full'
        with np.load(gate/f'state{state:02d}_{name}.npz') as expected:
            # On these12 inputs, old floating and new integer full terminals coincide.
            np.testing.assert_array_equal(a,expected['action']);np.testing.assert_array_equal(v,expected['values'])
    per_state=[];summary=[]
    for state in range(len(rows)):
        vals={a:float(np.median([r['e2e_ms'] for r in measurements if r['state']==state and r['arm']==a])) for a in ARMS}
        for arm in ARMS:per_state.append(dict(state=state,arm=arm,ms=vals[arm],
            versus_native_full=vals[arm]/vals['full'],versus_float_python_full=vals[arm]/vals['float_python_full']))
    baseline_native=np.mean([r['ms'] for r in per_state if r['arm']=='full'])
    baseline_python=np.mean([r['ms'] for r in per_state if r['arm']=='float_python_full'])
    for arm in ARMS:
        vals=[r['ms'] for r in per_state if r['arm']==arm]
        summary.append(dict(arm=arm,mean_ms=float(np.mean(vals)),p50_ms=float(np.quantile(vals,.5)),
            p95_ms=float(np.quantile(vals,.95)),ratio_native_full=float(np.mean(vals)/baseline_native),
            ratio_float_python_full=float(np.mean(vals)/baseline_python)))
    profiles=[]
    for state,row in enumerate(rows):
        for arm in ('full','cached','guarded4'):
            elapsed,(_,_,r)=run(row,arm,True)
            profiles.append(dict(state=state,arm=arm,e2e_ms=elapsed,
                table_generation_ms=r['table_generation_ms'],sparse_phy_ms=r['sparse_phy_ms'],
                selection_guard_ms=r['selection_guard_ms'],first_scan_ms=r['first_scan_ms'],
                first_scan_fraction=r['first_scan_ms']/elapsed,candidate_evaluations=r['candidate_evaluations'],
                first_scan_evaluations=r['first_scan_evaluations'],mmse_misses=r['mmse_misses']))
    for name,data in [('timing_raw',measurements),('per_state',per_state),('summary',summary),('profiles',profiles)]:write_csv(out/(name+'.csv'),data)
    for name,digest in hashes.items():
        if sha(ROOT/name)!=digest:raise AssertionError('Changed source: '+name)
    write_json(out/'completion.json',dict(completed=True,timed_decisions=len(measurements),profiled_decisions=len(profiles),
        timed_candidate_evaluations=sum(r['candidate_evaluations'] for r in measurements),
        profiled_candidate_evaluations=sum(r['candidate_evaluations'] for r in profiles),
        all_outputs_match_frozen_gate=True,new_native_runs=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--gate',required=True)
    p.add_argument('--output',required=True);p.add_argument('--cpu',type=int,default=255);p.add_argument('--repeats',type=int,default=5)
    main(p.parse_args())

"""Serial paired wall timing with logging, profiling and audits off in decisions."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import time
import numpy as np
from .cached_dpls import CanonicalDplsConfig,solve_canonical
from .physics import ROOT,library,curves
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .analyze_dpls_moves import write_json,write_csv,sha


def main(args):
    import torch
    from .archive import enable_legacy_types
    enable_legacy_types()
    if args.repeats < 5:
        raise ValueError('At least five alternating timing repetitions')
    source,out=Path(args.source).resolve(),Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    cpu=max(os.sched_getaffinity(0)) if args.cpu is None else args.cpu
    os.sched_setaffinity(0,{cpu})
    paths=sorted(source.glob('*_seed*/states/tti*.pt'))
    rows=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
    library();curves()
    for row in rows:
        problem_from_raw_row(row) # immutable TBS tables/link-adaptation imports only
    files=[Path(__file__),Path(__file__).with_name('cached_dpls.py'),Path(__file__).with_name('dpls_operations.py'),
           Path(__file__).with_name('physics.py'),Path(__file__).with_name('_physics.so'),
           Path(__file__).with_name('no_ctbs_inputs.py')]
    manifest={str(p.relative_to(ROOT)):sha(p) for p in files}
    (out/'source').mkdir()
    for p in files:shutil.copy2(p,out/'source'/p.name)
    write_json(out/'source_manifest.json',manifest)
    write_json(out/'input_manifest.json',{str(p):sha(p) for p in paths})
    write_json(out/'protocol.json',dict(cpu_affinity=[cpu],repeats=args.repeats,
        source=str(source),python=platform.python_version(),
        timing='current saved raw metadata -> MCS/TBS/raw PFQ -> fresh PHY/search -> action and close; immutable imports/table/library excluded',
        decision_logging=False,decision_history=False,decision_profiling=False,decision_audit=False,
        ordering='serial one process, reverse state order on alternating repetitions; alternate full/cached per state',
        post_timing_checks='all repetitions match frozen canonical gate actions/PHY, outside every measured interval',
        profiling='separate paid run with diagnostics; first-scan/setup shares are approximate profiled shares, not timing-run instrumentation',
        host='shared host; no real-time or packet-delay claim',independent_scene_seeds=1))
    records,answers=[],[]
    def run(row,mode,diagnostics):
        begin=time.perf_counter()
        p=problem_from_raw_row(row)
        draft=raw_pfq_draft(p,row['obs'])
        warm=row['previous_action'] if row['source_arm'].startswith('warm_') else None
        answer=solve_canonical(p,draft,CanonicalDplsConfig(mode,diagnostics=diagnostics),warm_start=warm)
        elapsed=(time.perf_counter()-begin)*1000
        return elapsed,answer
    for repeat in range(args.repeats):
        order=list(range(len(rows)))
        if repeat%2:order.reverse()
        for i in order:
            modes=('full','cached') if (repeat+i)%2==0 else ('cached','full')
            for mode in modes:
                elapsed,answer=run(rows[i],mode,False)
                r=answer[2]
                records.append(dict(state=i,source_arm=rows[i]['source_arm'],tti=rows[i]['tti'],repeat=repeat,
                    mode=mode,e2e_ms=elapsed,candidate_evaluations=r['candidate_evaluations'],
                    ue_evaluations=r['ue_evaluations'],mmse_requests=r['mmse_requests'],mmse_misses=r['mmse_misses']))
                answers.append((i,answer[0],answer[1]))
        print('TIMED_REPEAT',repeat+1,flush=True)
    # No terminal checking, NumPy reference or per-round observer in timing.
    reference=Path(args.reference).resolve()
    for i,action,values in answers:
        row=rows[i]
        with np.load(reference/f"{row['source_arm']}_seed{row['seed']}_tti{row['tti']}.npz") as expected:
            np.testing.assert_array_equal(action,expected['action'])
            np.testing.assert_array_equal(values,expected['values'])
    profiles=[]
    for i,row in enumerate(rows):
        for mode in ('full','cached'):
            elapsed,answer=run(row,mode,True)
            r=answer[2]
            profiles.append(dict(state=i,source_arm=row['source_arm'],tti=row['tti'],mode=mode,
                e2e_ms=elapsed,first_scan_ms=r['history'][0]['round_ms'],setup_ms=r['setup_ms'],
                first_scan_time_fraction=r['history'][0]['round_ms']/elapsed,
                first_plus_preparation_time_fraction=(elapsed-r['core_ms']+r['setup_ms']+r['history'][0]['round_ms'])/elapsed,
                first_scan_candidates=r['history'][0]['scored'],candidate_evaluations=r['candidate_evaluations'],
                first_scan_candidate_fraction=r['history'][0]['scored']/r['candidate_evaluations'],
                generation_ms=r['generation_ms'],mapping_ms=r['mapping_ms'],reassembly_ms=r['reassembly_ms'],
                score_ms=r['score_ms'],evaluate_ms=r['evaluate_ms'],native_kernel_ms=r['native_kernel_ms'],
                mmse_misses=r['mmse_misses']))
    reduced=[]
    for i in range(len(rows)):
        full=np.median([r['e2e_ms'] for r in records if r['state']==i and r['mode']=='full'])
        cached=np.median([r['e2e_ms'] for r in records if r['state']==i and r['mode']=='cached'])
        reduced.append(dict(state=i,source_arm=rows[i]['source_arm'],tti=rows[i]['tti'],
                            full_ms=float(full),cached_ms=float(cached),ratio=float(cached/full)))
    for name,data in [('timing_raw',records),('per_state',reduced),('profiles',profiles)]:
        write_csv(out/(name+'.csv'),data)
    write_json(out/'summary.json',dict(full_mean_ms=float(np.mean([r['full_ms'] for r in reduced])),
        cached_mean_ms=float(np.mean([r['cached_ms'] for r in reduced])),
        ratio_of_means=float(np.mean([r['cached_ms'] for r in reduced])/np.mean([r['full_ms'] for r in reduced])),
        mean_state_ratio=float(np.mean([r['ratio'] for r in reduced])),
        full_p50_ms=float(np.quantile([r['full_ms'] for r in reduced],.5)),
        full_p95_ms=float(np.quantile([r['full_ms'] for r in reduced],.95)),
        cached_p50_ms=float(np.quantile([r['cached_ms'] for r in reduced],.5)),
        cached_p95_ms=float(np.quantile([r['cached_ms'] for r in reduced],.95)),
        checks='all timed results exactly equal frozen gate actions/PHY; outside timing'))
    for p,digest in manifest.items():
        if sha(ROOT/p)!=digest:raise AssertionError('Source changed during timing: '+p)
    write_json(out/'completion.json',dict(completed=True,timed_decisions=len(records),
        profiling_decisions=len(profiles),new_native_runs=0,
        timed_candidate_evaluations=sum(r['candidate_evaluations'] for r in records),
        profiling_candidate_evaluations=sum(r['candidate_evaluations'] for r in profiles),
        timed_mmse_misses=sum(r['mmse_misses'] for r in records),
        profiling_mmse_misses=sum(r['mmse_misses'] for r in profiles)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--repeats',type=int,default=5);p.add_argument('--cpu',type=int)
    main(p.parse_args())

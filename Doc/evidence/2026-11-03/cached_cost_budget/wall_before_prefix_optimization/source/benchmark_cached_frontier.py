"""Paired no-audit wall frontier for full/cache/cap4/dirty M16,32,64."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from .physics import ROOT,library,curves
from .cached_dpls import solve_canonical,CanonicalDplsConfig
from .budgeted_cached_dpls import solve_budgeted,BudgetedCacheConfig
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .analyze_dpls_moves import write_json,write_csv,sha


ARMS=('full','cached','cap4','M16','M32','M64')


def config(arm,diagnostics=False):
    if arm=='cap4':return BudgetedCacheConfig('batch',batch_cap=4,diagnostics=diagnostics)
    return BudgetedCacheConfig('dirty',dirty_m=int(arm[1:]),diagnostics=diagnostics)


def main(args):
    import torch
    from .archive import enable_legacy_types
    enable_legacy_types()
    if args.repeats<5:raise ValueError('At least5 repetitions')
    source,quality,reference,out=map(lambda x:Path(x).resolve(),(args.source,args.quality,args.reference,args.output))
    if not json.loads((quality/'completion.json').read_text())['completed']:raise ValueError('Quality campaign incomplete')
    out.mkdir(parents=True,exist_ok=False)
    cpu=max(os.sched_getaffinity(0)) if args.cpu is None else args.cpu
    os.sched_setaffinity(0,{cpu})
    paths=sorted(source.glob('*_seed*/states/tti*.pt'))
    rows=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
    library();curves()
    for row in rows:problem_from_raw_row(row)
    files=[Path(__file__),*[Path(__file__).with_name(n) for n in ('cached_dpls.py','budgeted_cached_dpls.py',
        'dpls_operations.py','physics.py','no_ctbs_inputs.py','_physics.so')]]
    sources={str(p.relative_to(ROOT)):sha(p) for p in files}
    (out/'source').mkdir()
    for p in files:shutil.copy2(p,out/'source'/p.name)
    write_json(out/'source_manifest.json',sources)
    write_json(out/'input_manifest.json',{str(p):sha(p) for p in paths})
    write_json(out/'protocol.json',dict(arms=list(ARMS),repeats=args.repeats,cpu_affinity=[cpu],
        source=str(source),quality=str(quality),reference=str(reference),
        timed_scope='raw pre-action metadata -> MCS/TBS/PFQ -> fresh PHY decision -> close; immutable warmup excluded',
        diagnostics=False,history=False,teacher_queries=False,audits=False,logging_in_decision=False,
        ordering='serial one process; rotate/reverse arms per state/repetition; alternate state order',
        post_timing_validation='all outputs exactly match frozen quality campaign; outside measured intervals',
        profiles='one separate diagnostic run per state/arm for first-scan/setup share; not instrumentation inside timing runs',
        host='shared host, single pinned CPU and BLAS thread; same exposed seed41 inputs, no deployment guarantee'))
    def run(row,arm,diagnostics):
        started=time.perf_counter()
        p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs'])
        warm=row['previous_action'] if row['source_arm'].startswith('warm_') else None
        if arm in ('full','cached'):
            result=solve_canonical(p,draft,CanonicalDplsConfig(arm,diagnostics=diagnostics),warm_start=warm)
        else:result=solve_budgeted(p,draft,config(arm,diagnostics),warm_start=warm)
        return (time.perf_counter()-started)*1000,result
    records=[];answers=[]
    for repeat in range(args.repeats):
        states=list(range(len(rows)))
        if repeat%2:states.reverse()
        for state in states:
            offset=(repeat+state)%len(ARMS)
            arms=list(ARMS[offset:]+ARMS[:offset])
            if (repeat+state)%2:arms.reverse()
            for arm in arms:
                elapsed,result=run(rows[state],arm,False)
                r=result[2]
                records.append(dict(state=state,source_arm=rows[state]['source_arm'],tti=rows[state]['tti'],
                    arm=arm,repeat=repeat,e2e_ms=elapsed,rounds=r['rounds'],candidate_evaluations=r['candidate_evaluations'],
                    mmse_misses=r['mmse_misses']))
                answers.append((state,arm,result[0],result[1]))
        print('FRONTIER_REPEAT',repeat+1,flush=True)
    for state,arm,action,values in answers:
        row=rows[state]
        expected=(reference/f"{row['source_arm']}_seed{row['seed']}_tti{row['tti']}.npz" if arm=='full'
                  else quality/f'state{state:02d}_{arm}.npz')
        with np.load(expected) as saved:
            np.testing.assert_array_equal(action,saved['action'])
            np.testing.assert_array_equal(values,saved['values'])
    profiles=[]
    for state,row in enumerate(rows):
        for arm in ARMS:
            elapsed,result=run(row,arm,True)
            r=result[2]
            if arm in ('full','cached'):
                first_ms=r['history'][0]['round_ms'];first_candidates=r['history'][0]['scored']
            else:first_ms=r['first_scan_ms'];first_candidates=r['first_scan_candidates']
            profiles.append(dict(state=state,source_arm=row['source_arm'],arm=arm,e2e_ms=elapsed,
                setup_ms=r['setup_ms'],first_scan_ms=first_ms,
                first_scan_time_fraction=first_ms/elapsed,
                first_plus_preparation_time_fraction=(elapsed-r['core_ms']+r['setup_ms']+first_ms)/elapsed,
                first_scan_candidates=first_candidates,first_scan_candidate_fraction=first_candidates/r['candidate_evaluations'],
                candidate_evaluations=r['candidate_evaluations'],mmse_misses=r['mmse_misses']))
    reduced=[]
    for state in range(len(rows)):
        baseline=np.median([r['e2e_ms'] for r in records if r['state']==state and r['arm']=='full'])
        for arm in ARMS:
            value=np.median([r['e2e_ms'] for r in records if r['state']==state and r['arm']==arm])
            reduced.append(dict(state=state,arm=arm,e2e_ms=float(value),ratio=float(value/baseline)))
    summary=[]
    full_mean=np.mean([r['e2e_ms'] for r in reduced if r['arm']=='full'])
    for arm in ARMS:
        values=[r['e2e_ms'] for r in reduced if r['arm']==arm]
        p=[r for r in profiles if r['arm']==arm]
        summary.append(dict(arm=arm,mean_ms=float(np.mean(values)),p50_ms=float(np.quantile(values,.5)),
            p95_ms=float(np.quantile(values,.95)),ratio_of_means=float(np.mean(values)/full_mean),
            mean_first_scan_profiled_time_fraction=float(np.mean([r['first_scan_time_fraction'] for r in p])),
            mean_first_plus_preparation_profiled_time_fraction=float(np.mean([r['first_plus_preparation_time_fraction'] for r in p])),
            first_scan_candidate_fraction=sum(r['first_scan_candidates'] for r in p)/sum(r['candidate_evaluations'] for r in p)))
    for name,data in [('timing_raw',records),('per_state',reduced),('profiles',profiles),('summary',summary)]:write_csv(out/(name+'.csv'),data)
    for name,digest in sources.items():
        if sha(ROOT/name)!=digest:raise AssertionError('Source changed during timing: '+name)
    write_json(out/'completion.json',dict(completed=True,timed_decisions=len(records),profiling_decisions=len(profiles),
        timed_candidate_evaluations=sum(r['candidate_evaluations'] for r in records),
        profiling_candidate_evaluations=sum(r['candidate_evaluations'] for r in profiles),
        timed_mmse_misses=sum(r['mmse_misses'] for r in records),
        profiling_mmse_misses=sum(r['mmse_misses'] for r in profiles),new_native_runs=0,
        all_timed_outputs_match_frozen_quality=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--quality',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--cpu',type=int);main(p.parse_args())

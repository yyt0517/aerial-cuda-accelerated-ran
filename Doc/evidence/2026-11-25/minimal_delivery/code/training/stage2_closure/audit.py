"""Independent full-neighborhood audit and paired serial cost on DA-PLS states."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from training.stageb_lns.archive import enable_legacy_types
from training.stageb_lns.physics import ROOT, ReferencePhy
from training.stageb_lns.no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from training.stageb_lns.crossload_native import weights_for
from training.stageb_lns.persistent_dpls import solve_integer_reference,solve_persistent,PersistentConfig
from training.stageb_lns.bounded_dpls import solve_bounded
from training.stageb_lns.bounded_campaign import audit_round
from training.stageb_ppo.native import write,sha

def problem(row):
    p=problem_from_raw_row(row)
    p.weights=weights_for('mlwdf_pf',p.weights,row['ema'],row['hol_urgency'],p.queue,0.)
    np.testing.assert_array_equal(p.weights,row['objective_weights'])
    return p,raw_pfq_draft(p,row['obs'])

def one(path,out,repeats,concurrent_quality_campaign=True):
    import torch
    row=torch.load(path,map_location='cpu',weights_only=False);p,draft=problem(row)
    trace=[]
    def capture(info):trace.append({k:v.copy() if isinstance(v,np.ndarray) else v for k,v in info.items()})
    reference=solve_integer_reference(p,draft,observer=capture)
    rounds=[]
    def check(info):
        z=audit_round(info,trace[info['iteration']-1]);rounds.append(dict(iteration=info['iteration'],**z))
    a,v,r=solve_bounded(p,draft,observer=check)
    np.testing.assert_array_equal(a,reference[0]);np.testing.assert_array_equal(v,reference[1])
    np.testing.assert_array_equal(a,row['selected_action']);np.testing.assert_array_equal(v,row['selected_values'])
    np.testing.assert_allclose(v,ReferencePhy(p).evaluate(a)[0],rtol=1e-8,atol=1e-6)
    assert r['converged'] and len(rounds)==len(trace)
    # Preserve the first actual joint edit with complete affected-UE accounting.
    case=None
    for step in trace:
        if not len(step['selected_keys']):continue
        ix=int(np.flatnonzero(step['keys']==step['selected_keys'][0])[0])
        next_step=trace[step['iteration']] if step['iteration']<len(trace) else None
        if next_step is None:continue
        before=step['root'];after=next_step['root'];changes=np.argwhere(before!=after)
        if len(changes)!=2 or len(set(changes[:,1]))!=1:continue
        if not any(before[c,g]>=0 and after[c,g]<0 for c,g in changes):continue
        old=step['root_values'];new=step['values'][ix]
        changes_ue=np.flatnonzero(np.any(old!=new,axis=1))
        case=dict(tti=int(row['tti']),iteration=step['iteration'],edits=[dict(cell=int(c),prg=int(g),old_ue=int(before[c,g]),new_ue=int(after[c,g])) for c,g in changes],
          gain_q=int(step['delta_q'][ix]),dequeue_before=int(old[:,3].sum()),dequeue_after=int(new[:,3].sum()),floor=int(step['floor']),
          ues=[dict(ue=int(u),weight=float(p.weights[u]),expected_before=float(old[u,6]),expected_after=float(new[u,6]),
                    weighted_delta=float((new[u,6]-old[u,6])*p.weights[u]),sinr_before_db=float(old[u,4]),sinr_after_db=float(new[u,4]),
                    count_before=int(old[u,0]),count_after=int(new[u,0])) for u in changes_ue])
        break
    timers=[];arms=['python_full','native_full','bounded']
    # Immutable imports and tables have been used above; all timed contexts are new.
    for repeat in range(repeats):
        order=arms[repeat%3:]+arms[:repeat%3]
        for arm in order:
            start=time.perf_counter();cpu=time.process_time();pp,dd=problem(row)
            result=(solve_integer_reference(pp,dd) if arm=='python_full' else
                    solve_persistent(pp,dd,PersistentConfig('full')) if arm=='native_full' else solve_bounded(pp,dd))
            elapsed=(time.perf_counter()-start)*1000;cpu_ms=(time.process_time()-cpu)*1000
            np.testing.assert_array_equal(result[0],a);np.testing.assert_array_equal(result[1],v)
            timers.append(dict(arm=arm,repeat=repeat,wall_ms=elapsed,cpu_ms=cpu_ms,
                               candidates=result[2]['candidate_evaluations'],ue_evaluations=result[2]['ue_evaluations'],mmse_misses=result[2]['mmse_misses']))
    result=dict(path=str(path),sha256=sha(path),tti=int(row['tti']),passed=True,
        rounds=rounds,timing=timers,case=case,full=reference[2],bounded=r,
        concurrent_quality_campaign=concurrent_quality_campaign)
    write(out,result);return result

def main(args):
    enable_legacy_types();os.sched_setaffinity(0,{args.cpu})
    root=Path(args.source);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    paths=sorted(root.glob('boundary_r*_mlwdf_pf_s*/states/tti*.pt'))
    if args.partial:paths=[p for p in paths if time.time()-p.stat().st_mtime>2]
    if not args.partial:assert len(paths)==90
    quality_active=len(list(root.glob('*/done.json')))!=220
    results=[]
    for path in paths:
        target=out/f'{path.parent.parent.name}_{path.stem}.json'
        if target.exists():
            r=json.loads(target.read_text());assert r['sha256']==sha(path)
            if not args.partial and not quality_active and r.get('concurrent_quality_campaign',True):
                archive=out/'concurrent_timing_archive';archive.mkdir(exist_ok=True)
                prior=archive/target.name
                if not prior.exists():prior.write_bytes(target.read_bytes())
                r=one(path,target,args.repeats,False)
        else:r=one(path,target,args.repeats,quality_active)
        results.append(r);print('AUDIT',path.parent.parent.name,path.stem,r['passed'],flush=True)
    rows=[];rounds=[]
    for r in results:
        d=Path(r['path']).parent.parent.name;rate=float(d.split('_')[1][1:]);seed=int(d.split('_')[-1][1:])
        for t in r['timing']:rows.append(dict(rate=rate,seed=seed,tti=r['tti'],**t))
        for z in r['rounds']:rounds.append(dict(rate=rate,seed=seed,tti=r['tti'],**z))
    if rows:
        df=pd.DataFrame(rows);df.to_csv(out/'timing_raw.csv',index=False)
        # First average repeats and sampled states within each seed, then equal seeds.
        seed=df.groupby(['rate','seed','arm'],as_index=False)[['wall_ms','cpu_ms','candidates','ue_evaluations','mmse_misses']].mean()
        seed.to_csv(out/'timing_per_seed.csv',index=False)
        summary=seed.groupby(['rate','arm'],as_index=False)[['wall_ms','cpu_ms','candidates','ue_evaluations','mmse_misses']].mean()
        summary.to_csv(out/'timing_summary.csv',index=False)
        pd.DataFrame(rounds).to_csv(out/'rounds.csv',index=False)
    write(out/'completion.json',dict(completed=not args.partial,states=len(results),passed=all(r['passed'] for r in results),
      audited_rounds=len(rounds),audited_candidates=sum(r['candidates'] for r in rounds),
      new_evaluations=sum(r['fresh'] for r in rounds),required_evaluations=sum(r['required'] for r in rounds),
      timing_scope='serial CPU pinned; saved-input reconstruction, weight generation/consistency check, PFQ decode, context/search/destruction; excludes raw observation copy, HOL FIFO scan, native PFQ computation, bridge and GPU simulator',
      repeats=args.repeats,cpu=args.cpu,
      concurrent_quality_campaign=any(r.get('concurrent_quality_campaign',True) for r in results)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--cpu',type=int,default=128)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--partial',action='store_true');main(p.parse_args())

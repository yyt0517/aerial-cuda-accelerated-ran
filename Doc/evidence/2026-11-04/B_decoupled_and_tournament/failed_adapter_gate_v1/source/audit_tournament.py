"""Separately paid bitwise/transitive gate and serial cost controls; never used by B."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from .physics import ROOT,curves
from .archive import enable_legacy_types
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .bounded_dpls import solve_bounded,bounded_library
from .tournament_dpls import solve_tournament,TournamentConfig,tournament_library
from .native_schedule_inputs import prepare_native_problem
from .persistent_dpls import solve_persistent,PersistentConfig,persistent_library
from .analyze_bounded_stress import local
from .analyze_dpls_moves import write_json,write_csv,sha

def equal(a,b,key):
    a,b=np.asarray(a),np.asarray(b)
    if a.dtype.kind in 'fc' and b.dtype==a.dtype:
        np.testing.assert_array_equal(a.view(np.uint64),b.view(np.uint64),err_msg=key)
    else:np.testing.assert_array_equal(a,b,err_msg=key)

def states():
    gate=ROOT/'deliverables/bounded_gate_wall_20261104_v1/input_manifest.json'
    items=[]
    for name,digest in sorted(json.loads(gate.read_text()).items()):
        p=local(name);assert sha(p)==digest;items.append(dict(path=str(p),rate=0.,seed=41,sha256=digest))
    stress=ROOT/'deliverables/bounded_load_stress_20261104_v1/completion.json'
    for run in json.loads(stress.read_text())['runs']:
        for item in json.loads((local(run['directory'])/'saved_states.json').read_text()):
            p=local(item['path']);assert sha(p)==item['sha256']
            items.append(dict(path=str(p),rate=run['rate'],seed=run['seed'],sha256=item['sha256']))
    assert len(items)==252
    return items

def prep(row,native=False):
    if native:p,draft=prepare_native_problem(row)
    else:p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs'])
    warm=row['previous_action'] if row.get('source_arm','').startswith('warm_') else None
    return p,draft,warm

def solve(row,arm,observer=None,profile=False):
    p,draft,warm=prep(row,arm=='native_direct')
    if arm=='full':return solve_persistent(p,draft,PersistentConfig('full'),warm_start=warm)
    if arm=='heap':return solve_bounded(p,draft,warm_start=warm,observer=observer)
    return solve_tournament(p,draft,TournamentConfig(diagnostics=profile,direct_gamma=arm!='tree'),
        warm_start=warm,observer=observer)

def main(args):
    import torch
    enable_legacy_types();os.sched_setaffinity(0,{args.cpu})
    out=Path(args.output).resolve();items=states();curves();bounded_library();tournament_library();persistent_library()
    if args.phase=='audit':
        out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
        files=[Path(__file__).with_name(n) for n in ('audit_tournament.py','tournament_kernel.cpp','tournament_dpls.py',
            'native_schedule_inputs.py','build_tournament.py','_tournament.so','test_tournament.py',
            'bounded_kernel.cpp','bounded_dpls.py','_bounded.so','physics_kernel.cpp','persistent_kernel.cpp')]
        sources={str(f.relative_to(ROOT)):sha(f) for f in files}
        for f in files:shutil.copy2(f,out/'source'/f.name)
        write_json(out/'source_manifest.json',sources);write_json(out/'states.json',items)
        write_json(out/'protocol.json',dict(states=252,
            gate='bit patterns of every round root/UE field, key, upper, exact/stale delta, D, status, selected key and final versus frozen heap; previous full-reference gates provide transitive equivalence',
            independent='four synthetic independent full-oracle paths in test_tournament, both cache modes',
            numerical='original Eigen fixed4x4 arithmetic untouched; direct cache copies original gamma; no Woodbury/batched arithmetic',
            priority='persistent perPRG local trees plus one global leaf perPRG; dirty dependency updates; global floor crossings refreshed; cheap known-exact seeding still scans metadata',
            adapter='native MCS/TBS/domain/PFQ mapping, NumPy deficit normalization and covariance/signal preprocessing retained for identical bits',
            scientific_B='separate library, never substituted into any frozen B arm',
            timing='separate serial CPU pin, four engineering arms plus full reference, rotating order, fresh adapter/context/search/close, three repeats per state, medians then mean; audit/profile disabled',
            tails='saved-state cost P95/max, not a replacement for native measured-TTI tails'))
        rows=[];ledger=[];total_rounds=total_candidates=0
        for idx,item in enumerate(items):
            row=torch.load(item['path'],map_location='cpu',weights_only=False);p,draft,warm=prep(row)
            n,nd,nw=prep(row,True)
            for key in ('h','w','noise','queue','serving','caps','mcs','tbs','legal','weights'):
                equal(getattr(p,key),getattr(n,key),key)
            equal(draft,nd,'draft');trace=[]
            def save(i):trace.append({k:v.copy() if isinstance(v,np.ndarray) else v for k,v in i.items()})
            a,v,r=solve(row,'heap',save);work={'heap':r}
            for arm in ('tree','direct','native_direct'):
                count=[0]
                def check(i):
                    ref=trace[i['iteration']-1];assert i.keys()==ref.keys()
                    for key in i:equal(i[key],ref[key],key)
                    count[0]+=1
                aa,vv,rr=solve(row,arm,check);equal(a,aa,'final action');equal(v,vv,'final UE bits')
                assert count[0]==len(trace)
                for key in ('rounds','score_q','floor','chosen_dequeue','scored_edits','candidate_evaluations',
                    'ue_evaluations','mmse_requests','mmse_misses'):
                    assert rr[key]==r[key],(arm,key,rr[key],r[key])
                work[arm]=rr
            candidates=sum(len(i['keys']) for i in trace);total_rounds+=len(trace);total_candidates+=candidates
            rows.append(dict(state=idx,rate=item['rate'],seed=item['seed'],tti=row['tti'],rounds=len(trace),
                candidates=candidates,scored=r['scored_edits'],mmse_misses=r['mmse_misses']))
            ledger.append(dict(state=idx,**work));np.savez_compressed(out/f'answer{idx:03d}.npz',action=a,values=v)
            if idx%30==0:print('TOURNAMENT_GATE',idx,len(items),flush=True)
        write_csv(out/'equivalence.csv',rows);write_json(out/'audit_work_ledger.json',ledger)
        for f,digest in sources.items():assert sha(ROOT/f)==digest
        write_json(out/'audit_completion.json',dict(completed=True,bitwise_equivalence_passed=True,states=len(items),
            rounds=total_rounds,candidates=total_candidates,new_native_runs=0,training_steps=0))
        return
    assert json.loads((out/'audit_completion.json').read_text())['bitwise_equivalence_passed']
    for f,digest in json.loads((out/'source_manifest.json').read_text()).items():assert sha(ROOT/f)==digest
    arms=('full','heap','tree','direct','native_direct');timings=[];profiles=[];work=[]
    loaded=[torch.load(i['path'],map_location='cpu',weights_only=False) for i in items]
    # Warm only immutable imports/LUT/TBS. All contexts remain decision-local.
    for arm in arms:solve(loaded[0],arm)
    for repeat in range(args.repeats):
        for idx in (range(len(items)) if repeat%2==0 else reversed(range(len(items)))):
            row=loaded[idx];offset=(repeat+idx)%len(arms)
            with np.load(out/f'answer{idx:03d}.npz') as z:a=z['action'];v=z['values']
            for arm in arms[offset:]+arms[:offset]:
                begin=time.perf_counter();aa,vv,r=solve(row,arm);elapsed=1000*(time.perf_counter()-begin)
                equal(a,aa,'timed action');equal(v,vv,'timed UE bits')
                timings.append(dict(state=idx,rate=items[idx]['rate'],seed=items[idx]['seed'],repeat=repeat,
                    arm=arm,ms=elapsed,candidates=r['candidate_evaluations']))
                work.append(dict(state=idx,repeat=repeat,arm=arm,**r))
        print('TOURNAMENT_WALL',repeat+1,flush=True)
    for idx,row in enumerate(loaded):
        begin=time.perf_counter();p,draft,warm=prep(row,True);adapt=1000*(time.perf_counter()-begin)
        aa,vv,r=solve_tournament(p,draft,TournamentConfig(diagnostics=True,direct_gamma=True),warm_start=warm)
        profiles.append(dict(state=idx,rate=items[idx]['rate'],native_adapter_ms=adapt,**r))
    table=pd.DataFrame(timings);per=table.groupby(['state','rate','seed','arm'],as_index=False).ms.median()
    summary=[]
    for rate,g in per.groupby('rate'):
        full=g[g.arm=='full'].ms.mean();heap=g[g.arm=='heap'].ms.mean()
        for arm,h in g.groupby('arm'):
            summary.append(dict(rate=rate,arm=arm,states=len(h),mean_ms=float(h.ms.mean()),
                P95_ms=float(h.ms.quantile(.95)),max_ms=float(h.ms.max()),
                mean_ratio_full=float(h.ms.mean()/full),mean_ratio_heap=float(h.ms.mean()/heap)))
    write_csv(out/'timing_raw.csv',timings);per.to_csv(out/'timing_per_state.csv',index=False)
    write_csv(out/'wall_summary.csv',summary);write_csv(out/'profiles.csv',profiles)
    write_json(out/'timing_work_ledger.json',work)
    for f,digest in json.loads((out/'source_manifest.json').read_text()).items():assert sha(ROOT/f)==digest
    for item in items:assert sha(item['path'])==item['sha256']
    write_json(out/'completion.json',dict(completed=True,bitwise_equivalence_passed=True,
        states=len(items),timed_decisions=len(timings),profiled_decisions=len(profiles),new_native_runs=0,
        deployment_wall_all_stress_passed=all(s['mean_ratio_full']<=.4 for s in summary if s['rate']>0 and s['arm']=='native_direct'),
        scientific_B_solver_unchanged=True,native_tail_not_repeated=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--phase',choices=('audit','timing'),required=True)
    p.add_argument('--cpu',type=int,default=251);p.add_argument('--repeats',type=int,default=3);main(p.parse_args())

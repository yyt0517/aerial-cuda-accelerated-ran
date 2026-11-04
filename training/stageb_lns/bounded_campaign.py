"""Paid equivalence, necessary-set audit and serial capacity-bound wall gate."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from .physics import ROOT,ReferencePhy,curves
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .persistent_dpls import solve_integer_reference,solve_persistent,PersistentConfig,persistent_library
from .bounded_dpls import solve_bounded,BoundedConfig,certificate_sets,bounded_library
from .analyze_dpls_moves import write_json,write_csv,sha

def audit_round(info,reference):
    for k in ('root','root_values','keys','delta_d','selected_keys'):
        np.testing.assert_array_equal(info[k],reference[k],err_msg=k)
    assert info['score_q']==reference['score_q'] and info['floor']==reference['floor']
    if not np.all(info['upper']>=reference['delta_q']):raise AssertionError('Candidate upper violation')
    valid=(info['status']&1)!=0
    np.testing.assert_array_equal(info['delta_q'][valid],reference['delta_q'][valid])
    required,fresh=certificate_sets(info,reference)
    np.testing.assert_array_equal(required,fresh,err_msg='UB order exceeds/misses frozen certificate required set')
    return dict(candidates=len(valid),fresh=int(fresh.sum()),required=int(required.sum()),
        known_before=int(np.sum((info['status']&4)!=0)),still_unscored=int(np.sum(~valid)))

def main(args):
    import torch
    from .archive import enable_legacy_types
    enable_legacy_types();os.sched_setaffinity(0,{args.cpu})
    source,out=Path(args.source).resolve(),Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    paths=sorted(source.glob('*_seed*/states/tti*.pt'));assert len(paths)==12
    files=[Path(__file__),*[Path(__file__).with_name(n) for n in ('bounded_dpls.py','bounded_kernel.cpp','build_bounded.py',
        '_bounded.so','persistent_dpls.py','persistent_kernel.cpp','_persistent.so','physics_kernel.cpp','physics.py','no_ctbs_inputs.py')]]
    (out/'source').mkdir();hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
    for f in files:shutil.copy2(f,out/'source'/f.name)
    write_json(out/'source_manifest.json',hashes)
    write_json(out/'input_manifest.json',{str(f):sha(f) for f in paths})
    write_json(out/'protocol.json',dict(version='capacity_bound_q1e6_v1',states=12,
        objective='unchanged perUE nearest-even1e-6B, same initialization/floor/1/16B thresholds/canonical1-unit band',
        certificate='current max m/key k: unresolved UB>=m or UB==m-1 with smaller key; no budget bypass',
        required_set='CURRENT-UNKNOWN eligible entries with UB>=true max, UB==max-1 with key<final winner, plus unknown winner; confirmation UB>threshold',
        inference='required for frozen maximum-based certificate, not all conceivable key-aware certificates',
        gate='every root/choice/upper/logicalD/cached exact delta vs newly paid independent full integer reference',
        timing='serial CPU pinned, alternating rotating order, fresh causal input/context/PFQ/search/close; observers/log/history/profiling off; immutable imports/tables warm',
        repeats=args.repeats,cpu=args.cpu,wall_gate='bounded mean <=40% same architecture native full',
        heavy_load_next='rate0.8 multiplied1.4/1.6 =>1.12/1.28; report measuredTTI P95/max and serial fixed-input paired replays',
        B_entry='only after equivalence/wall/high-load cost checks; five frozen urgency coefficients on training41--43; no ranking training'))
    inputs=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
    rows=[];rounds=[];ledger=[];answers={}
    persistent_library();bounded_library();curves()
    for state,row in enumerate(inputs):
        p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs'])
        warm=row['previous_action'] if row['source_arm'].startswith('warm_') else None
        trace=[]
        def save(i):trace.append({k:v.copy() if isinstance(v,np.ndarray) else v for k,v in i.items()})
        ref=solve_integer_reference(p,draft,warm_start=warm,observer=save)
        def audit(i):
            result=audit_round(i,trace[i['iteration']-1])
            rounds.append(dict(state=state,iteration=i['iteration'],phase='first' if i['iteration']==1 else
                ('inner' if len(i['selected_keys']) else 'confirmation'),**result))
        a,v,r=solve_bounded(p,draft,warm_start=warm,observer=audit)
        np.testing.assert_array_equal(a,ref[0]);np.testing.assert_array_equal(v,ref[1]);assert r['rounds']==len(trace) and r['converged']
        np.testing.assert_allclose(v,ReferencePhy(p).evaluate(a)[0],rtol=1.e-8,atol=1.e-6)
        answers[state]=(a.copy(),v.copy());np.savez_compressed(out/f'state{state:02d}.npz',action=a,values=v)
        ledger.append(dict(state=state,full_reference=ref[2],bounded=r))
        rows.append(dict(state=state,source_arm=row['source_arm'],tti=row['tti'],**r))
        print('BOUNDED_GATE',state,'full',ref[2]['candidate_evaluations'],'bounded',r['candidate_evaluations'],flush=True)
    timed=[];profiles=[]
    arms=('full','cached','bounded')
    for repeat in range(args.repeats):
        for state in (range(12) if repeat%2==0 else reversed(range(12))):
            row=inputs[state];offset=(state+repeat)%3;order=arms[offset:]+arms[:offset]
            for arm in order:
                begin=time.perf_counter();p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs'])
                warm=row['previous_action'] if row['source_arm'].startswith('warm_') else None
                a,v,r=(solve_bounded(p,draft,warm_start=warm) if arm=='bounded' else
                    solve_persistent(p,draft,PersistentConfig(arm),warm_start=warm))
                elapsed=1000*(time.perf_counter()-begin)
                np.testing.assert_array_equal(a,answers[state][0]);np.testing.assert_array_equal(v,answers[state][1])
                timed.append(dict(state=state,repeat=repeat,arm=arm,ms=elapsed,candidates=r['candidate_evaluations'],
                    ue_evaluations=r['ue_evaluations'],mmse_requests=r['mmse_requests'],mmse_misses=r['mmse_misses']))
        print('BOUNDED_WALL',repeat+1,flush=True)
    for state,row in enumerate(inputs):
        begin=time.perf_counter();p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs'])
        warm=row['previous_action'] if row['source_arm'].startswith('warm_') else None
        _,_,r=solve_bounded(p,draft,BoundedConfig(diagnostics=True),warm_start=warm)
        profiles.append(dict(state=state,e2e_ms=1000*(time.perf_counter()-begin),**r))
    import pandas as pd
    table=pd.DataFrame(timed);per=table.groupby(['state','arm'],as_index=False).ms.median();aggregate=[]
    fullmean=float(per[per.arm=='full'].ms.mean())
    for arm in arms:
        z=per[per.arm==arm].ms
        aggregate.append(dict(arm=arm,mean_ms=float(z.mean()),P50_ms=float(z.quantile(.5)),P95_ms=float(z.quantile(.95)),
            max_ms=float(z.max()),mean_ratio_full=float(z.mean()/fullmean)))
    bound=next(r for r in aggregate if r['arm']=='bounded')
    write_csv(out/'metrics.csv',rows);write_csv(out/'rounds.csv',rounds);write_json(out/'work_ledger.json',ledger)
    write_csv(out/'timing_raw.csv',timed);write_csv(out/'timing_per_state.csv',per.to_dict('records'))
    write_csv(out/'wall_summary.csv',aggregate);write_csv(out/'profiles.csv',profiles)
    for f,digest in hashes.items():assert sha(ROOT/f)==digest
    totalfresh=sum(r['fresh'] for r in rounds);required=sum(r['required'] for r in rounds)
    write_json(out/'completion.json',dict(completed=True,equivalence_passed=True,states=12,
        audited_rounds=len(rounds),audited_candidates=sum(r['candidates'] for r in rounds),
        bounded_scored=totalfresh,required_set=required,UB_order_over_required=totalfresh/required,
        wall_passed=bound['mean_ratio_full']<=.4,bounded_wall_ms=bound['mean_ms'],
        native_runs=0,training_steps=0,timed_decisions=len(timed)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='deliverables/anytime_dpls_20261103_smoke_v3')
    p.add_argument('--output',required=True);p.add_argument('--cpu',type=int,default=255)
    p.add_argument('--repeats',type=int,default=5);main(p.parse_args())

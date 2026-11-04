"""Twelve-state MMSE-free upper-bound screening from existing paid logs."""
import argparse
import ast
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import numpy as np
import pandas as pd
from .analyze_dpls_moves import write_json,write_csv,sha
from .physics import ROOT,NativePhy,ReferencePhy,SINR,BLER,EXPECTED
from .persistent_dpls import PersistentPhy
from .dpls_operations import compact_neighborhood
from .certified_bounds import logical_bounds,curve_structure,activity,SCALE


def summary(rows):
    result=[]
    for method in ('capacity','theory_added'):
        for phase in ('first','inner','confirmation'):
            part=[r for r in rows if r['method']==method and r['phase']==phase]
            d=dict(method=method,phase=phase,rounds=len(part))
            sums=('candidates','eligible','dirty','dirty_eligible','floor_skipped','threshold_live',
                  'dirty_threshold_live','above_best_lo','above_best_hi','dirty_above_best_lo',
                  'dirty_above_best_hi','tie_live_lo','tie_live_hi',
                  'all_above_best_lo','all_above_best_hi','all_threshold_live')
            for field in sums:
                vals=[r[field] for r in part if r[field] is not None]
                d[field]=sum(vals) if vals else None
            for field,denom in (('above_best_lo','eligible'),('above_best_hi','eligible'),
                ('dirty_above_best_lo','dirty_eligible'),('dirty_above_best_hi','dirty_eligible'),
                ('threshold_live','eligible'),('dirty_threshold_live','dirty_eligible'),
                ('tie_live_lo','eligible'),('tie_live_hi','eligible'),
                ('all_above_best_lo','candidates'),('all_above_best_hi','candidates'),
                ('all_threshold_live','candidates')):
                d[field+'_fraction']=d[field]/d[denom] if d[field] is not None and d[denom] else None
            d['zero_fresh_confirmation_certificates']=sum(r['zero_fresh_confirmation_certificate'] for r in part)
            result.append(d)
    return result


def main(args):
    import torch
    from .archive import enable_legacy_types
    from .no_ctbs_inputs import problem_from_raw_row
    enable_legacy_types()
    source,reference,integer,quant,out=map(lambda v:Path(v).resolve(),
        (args.source,args.reference,args.integer_gate,args.quantization,args.output))
    out.mkdir(parents=True,exist_ok=False)
    paths=sorted(source.glob('*_seed*/states/tti*.pt'))
    if len(paths)!=12:raise ValueError('This protocol is the fixed12-state diagnostic')
    inputs={str(p):sha(p) for p in paths}
    for p in (reference/'decisions.csv',integer/'rounds.csv',quant):inputs[str(p)]=sha(p)
    decisions=pd.read_csv(reference/'decisions.csv');ir=pd.read_csv(integer/'rounds.csv')
    sensitivity=pd.read_csv(quant)
    assert len(sensitivity)==378 and not sensitivity.choice_changed.any()
    files=[Path(__file__),Path(__file__).with_name('certified_bounds.py')]
    for name in ('physics.py','physics_kernel.cpp','dpls_operations.py','no_ctbs_inputs.py','persistent_dpls.py'):
        files.append(Path(__file__).with_name(name))
    files.append(ROOT/'cuMAC/src/cumac.h')
    (out/'source').mkdir()
    for file in files:shutil.copy2(file,out/'source'/file.name)
    sources={str(p.relative_to(ROOT)):sha(p) for p in files}
    write_json(out/'source_manifest.json',sources)
    write_json(out/'protocol.json',dict(version='certified_bound_offline_v1',
        input='same12 correlated seed41 causal states and stored full roots/per-UE exact PHY',
        methods=dict(capacity='all affected contributions bounded by nearest-even(weight*min(Q,TBS)*1e6); BLER lower0',
            theory_added='add-only interference + unchanged victim assignment: upper contribution is stored root contribution; exact-arithmetic model only'),
        dirty='existing full-cache local refresh set only; not a rollout of cache entries deferred by a lazy solver',
        integer_best='stored per-UE roots/next-root give selected integer gain b; stored quantization choice equality proves true maximum in[b,b+1]; report both endpoints',
        confirmation='no accepted move => threshold certificate UB<=threshold; true subthreshold maximum was not saved',
        fraction_denominator='exact cheap floor-feasible candidates; all-domain counts also retained',
        tie='max-anchored1 integer unit; unseen UB==known maximum-1 can win by smaller key',
        no_PHY='constructors/evaluators and numpy inverse/solve patched to raise; no MMSE/BLER counterfactual scoring',
        safeguards='no pruning deployed online; no learned ordering fitted; no wall-time or actual scoring-saving claim',
        cap4='existing seed44 failure unchanged; any future approximate-trajectory protocol first needs full canonical/random-tie native path-variation controls',
        new_PHY=0,new_native_runs=0,training_steps=0))
    rows=[];observed=[];initializers=[];ambiguous_best=0;total_candidates=0;total_dirty=0
    with ExitStack() as stack:
        for cls in (NativePhy,ReferencePhy,PersistentPhy):
            for name in ('__init__','evaluate'):
                stack.enter_context(patch.object(cls,name,side_effect=AssertionError('No new PHY permitted')))
        for name in ('inv','solve'):
            stack.enter_context(patch.object(np.linalg,name,side_effect=AssertionError('No new matrix solve permitted')))
        curves=curve_structure()
        for state,path in enumerate(paths):
            row=torch.load(path,map_location='cpu',weights_only=False);p=problem_from_raw_row(row)
            # Prove the floating capture and integer replay used the same
            # initial root: warm-score gaps dwarf any per-UE quantization
            # error, or the common integer dequeue floor rejects warm.
            decision_path=path.parent.parent/'decisions.jsonl'
            inputs[str(decision_path)]=sha(decision_path)
            launch=next(json.loads(s) for s in decision_path.open() if json.loads(s).get('tti')==row['tti'])
            ref=reference/f"{row['source_arm']}_seed{row['seed']}_tti{row['tti']}.npz"
            inputs[str(ref)]=sha(ref)
            with np.load(ref) as z:
                roots=z['roots'].copy();values=z['root_values'].copy();selected=z['chosen_keys'].copy();terminal=z['values'].copy()
            full=ir[(ir.state==state)&(ir['mode']=='full')].sort_values('iteration')
            cached=ir[(ir.state==state)&(ir['mode']=='cached')].sort_values('iteration')
            keys=np.array([(ast.literal_eval(v) or [-1])[0] for v in full.selected_keys])
            np.testing.assert_array_equal(keys,selected)
            floor=int(decisions[(decisions.source_arm==row['source_arm'])&(decisions.tti==row['tti'])]['floor'].iloc[0])
            root_j=float(values[0,:,EXPECTED]@p.weights)
            np.testing.assert_allclose(root_j,launch['search_initial_score'],rtol=0,atol=1.e-8)
            gap=None if launch.get('warm_score') is None else launch['warm_score']-launch['initial_score']
            warm_feasible=launch.get('warm_dequeue') is not None and launch['warm_dequeue']>=floor
            assert not warm_feasible or abs(gap)>p.U/SCALE+1.e-8
            initializers.append(dict(state=state,initializer=launch['initialization'],warm_score_gap_B=gap,
                warm_floor_feasible=warm_feasible,integer_initializer_stable=True))
            previous=None
            for i,(root,v,key) in enumerate(zip(roots,values,selected)):
                prg,assignment=compact_neighborhood(root,p.legal)
                b=logical_bounds(p,root,v,prg,assignment)
                n=len(prg);total_candidates+=n
                feasible=int(v[:,3].sum())+b['delta_d']>=floor
                dirty=np.ones(n,dtype=bool)
                if previous is not None:
                    old,oldprg,oldaff,oldkey=previous
                    if oldkey>=0:
                        dirty=(prg==oldprg)|(b['affected']&oldaff!=0)|~np.isin(b['keys'],old)
                    else:dirty=~np.isin(b['keys'],old)
                assert int(dirty.sum())==int(cached.iloc[i].scored)
                total_dirty+=int(dirty.sum())
                phase='first' if i==0 else ('inner' if key>=0 else 'confirmation')
                threshold=(1 if i<2 else 16)*SCALE
                chosen=int(np.flatnonzero(b['keys']==key)[0]) if key>=0 else -1
                if chosen>=0:
                    nextv=values[i+1] if i+1<len(values) else terminal
                    best_lo=int(np.rint(nextv[:,EXPECTED]*p.weights*SCALE).astype(np.int64).sum()-
                                np.rint(v[:,EXPECTED]*p.weights*SCALE).astype(np.int64).sum())
                    assert best_lo>threshold and feasible[chosen]
                    best_hi=best_lo+1
                    expected=root.copy();expected[:,prg[chosen]]=assignment[chosen]
                    if i+1<len(roots):np.testing.assert_array_equal(expected,roots[i+1])
                    previous=(b['keys'],prg[chosen],b['affected'][chosen],key)
                else:
                    best_lo=best_hi=None;previous=(b['keys'],-1,np.uint64(0),key)
                for method in ('capacity','theory_added'):
                    upper=b[method+'_q']
                    if chosen>=0:
                        assert upper[chosen]>=best_lo
                        lo=feasible&(upper>=best_hi);hi=feasible&(upper>=best_lo)
                        tlo=feasible&(upper>=best_hi-1)&(upper>threshold)
                        thi=feasible&(upper>=best_lo-1)&(upper>threshold)
                        ambiguous_best+=int(np.sum(hi&~lo))
                    else:lo=hi=tlo=thi=None
                    live=feasible&(upper>threshold)
                    rows.append(dict(state=state,source_arm=row['source_arm'],tti=row['tti'],iteration=i+1,
                        method=method,phase=phase,candidates=n,eligible=int(feasible.sum()),dirty=int(dirty.sum()),
                        dirty_eligible=int(np.sum(dirty&feasible)),floor_skipped=int(np.sum(~feasible)),
                        selected_gain_q=best_lo,maximum_q_lo=best_lo,maximum_q_hi=best_hi,
                        above_best_lo=int(lo.sum()) if lo is not None else None,
                        above_best_hi=int(hi.sum()) if hi is not None else None,
                        all_above_best_lo=int(np.sum(upper>=best_hi)) if chosen>=0 else None,
                        all_above_best_hi=int(np.sum(upper>=best_lo)) if chosen>=0 else None,
                        all_threshold_live=int(np.sum(upper>threshold)),
                        dirty_above_best_lo=int(np.sum(dirty&lo)) if lo is not None else None,
                        dirty_above_best_hi=int(np.sum(dirty&hi)) if hi is not None else None,
                        tie_live_lo=int(tlo.sum()) if tlo is not None else None,
                        tie_live_hi=int(thi.sum()) if thi is not None else None,
                        threshold_live=int(live.sum()),dirty_threshold_live=int(np.sum(dirty&live)),
                        zero_fresh_confirmation_certificate=bool(phase=='confirmation' and not np.any(dirty&live)),
                        mean_theory_monotone_victims=float(b['monotone_victims'].mean())))
                if i+1<len(roots):
                    a,bmask=activity(root),activity(roots[i+1])
                    for u in range(p.U):
                        assigned=root[p.serving[u]]==u
                        if not assigned.any() or not np.array_equal(assigned,roots[i+1][p.serving[u]]==u):continue
                        oldm=a[assigned]&~(np.uint64(1)<<np.uint64(p.serving[u]))
                        newm=bmask[assigned]&~(np.uint64(1)<<np.uint64(p.serving[u]))
                        if np.array_equal(oldm,newm):continue
                        add=bool(np.all((oldm&~newm)==0));remove=bool(np.all((newm&~oldm)==0))
                        if not (add or remove):continue
                        diff=float(values[i+1,u,SINR]-v[u,SINR])
                        violation=max(0.,diff if add else -diff)
                        observed.append(dict(state=state,iteration=i+1,ue=u,added=add,SINR_delta_db=diff,
                            monotonicity_violation_db=violation,
                            BLER_delta=float(values[i+1,u,BLER]-v[u,BLER])))
            print('BOUNDS',state,len(roots),flush=True)
    aggregate=summary(rows)
    write_csv(out/'rounds.csv',rows);write_csv(out/'summary.csv',aggregate)
    work=[]
    for method in ('capacity','theory_added'):
        part=[r for r in rows if r['method']==method]
        # Descending UB evaluates a true-max candidate before any candidate
        # whose UB is below max-1. Add confirmation UB>threshold. Count ALL
        # candidates, not just locally dirty entries: no assumed cache reuse,
        # no multiplication of cache and pruning fractions.
        upper=sum(r['tie_live_hi'] if r['phase']!='confirmation' else r['threshold_live'] for r in part)
        work.append(dict(method=method,conditional_evaluations_upper_bound=upper,
            initializations_retained=20,total_evaluations_upper_bound=upper+20,
            ratio_original_full=(upper+20)/624690,ratio_existing_cache=(upper+20)/183079,
            basis='UB-priority descending, valid bounds, exhaustive logical domain, tie-safe certificate; count all entries each round, even deferred/never-cached entries',
            interpretation='mathematical scoring-count bound on frozen path, not an executed lazy trajectory or wall-time claim',
            finite_kernel_certificate=method=='capacity'))
    write_json(out/'UB_order_work_bound.json',work)
    write_csv(out/'native_curve_structure.csv',curves);write_csv(out/'observed_monotonicity.csv',observed)
    write_csv(out/'initialization_checks.csv',initializers)
    write_json(out/'input_manifest.json',inputs)
    for filename,digest in inputs.items():assert sha(filename)==digest
    for filename,digest in sources.items():assert sha(ROOT/filename)==digest
    write_json(out/'completion.json',dict(completed=True,states=len(paths),rounds=len(rows)//2,
        candidate_bound_pairs=total_candidates,baseline_cached_dirty=total_dirty,
        integer_best_count_ambiguities=ambiguous_best,integer_initializers_proved_stable=len(initializers),
        observed_monotone_pairs=len(observed),max_observed_SINR_violation_db=max((r['monotonicity_violation_db'] for r in observed),default=0),
        new_candidate_PHY=0,new_UE_PHY=0,new_MMSE=0,new_native_runs=0,training_steps=0,
        online_pruning_enabled=False,wall_speedup_evaluated=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='deliverables/anytime_dpls_20261103_smoke_v3')
    p.add_argument('--reference',default='deliverables/canonical_cached_dpls_gate_20261103_v1')
    p.add_argument('--integer-gate',default='deliverables/persistent_integer_gate_20261103_v1')
    p.add_argument('--quantization',default='deliverables/cached_budget_quality_20261103_v1/quantization.csv')
    p.add_argument('--output',required=True);main(p.parse_args())

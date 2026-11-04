"""High-load exact oracle audit and separate serial saved-input cost controls."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .physics import ROOT,curves
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .persistent_dpls import solve_integer_reference,solve_persistent,PersistentConfig,persistent_library
from .bounded_dpls import solve_bounded,bounded_library
from .bounded_campaign import audit_round
from .analyze_dpls_moves import write_json,write_csv,sha

def local(path):
    p=Path(path)
    return ROOT/p.relative_to('/opt/nvidia/cuBB') if str(p).startswith('/opt/nvidia/cuBB/') else p

def main(args):
    import torch
    from .archive import enable_legacy_types
    enable_legacy_types();os.sched_setaffinity(0,{args.cpu})
    source,out=Path(args.source).resolve(),Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    complete=json.loads((source/'completion.json').read_text());assert complete['completed']
    protocol=json.loads((source/'protocol.json').read_text());assert protocol['mode']=='stress' and protocol['workers']==1
    write_json(out/'protocol.json',dict(source=str(source),
        thresholds='inherited unchanged from anytime_dpls_step0/next_protocol.json; per-load work mean<=.25/P95<=.40, serial wallmean/full<=.40',
        wall='all240 saved states, three serial alternating repeats, mean of per-state medians; shared host',
        native_tail='all4000 measuredTTIs per seed/load, sequential CPU255; max retained, no invented realtime deadline',
        load='baseline0.8; x1.4=1.12 x1.6=1.28; density growth versus baseline reported without hiding it'))
    pointrows=[];states=[];inputs={};auditrows=[];rounds=[];ledger=[]
    for run in complete['runs']:
        d=local(run['directory']);s=json.loads((d/'summary.json').read_text())
        assert s['native_passed'] and s['pending_cohort']==0
        for n in ('summary.json','validation.json','reward_validation.json','search.csv','saved_states.json'):
            inputs[str(d/n)]=sha(d/n)
        valid=json.loads((d/'validation.json').read_text());assert valid['passed'] and valid['exact_native_actions']==12400
        table=pd.read_csv(d/'search.csv');table=table[table.tti.between(8001,12000)]
        assert len(table)==4000
        ratio=table.candidates/table.full_candidate_count
        mean_count_ratio=float(table.candidates.mean()/table.full_candidate_count.mean())
        P95_count_ratio=float(table.candidates.quantile(.95)/table.full_candidate_count.quantile(.95))
        pointrows.append(dict(seed=s['seed'],rate=s['rate'],load=s['rate']/.8,
            wall_mean_ms=float(table.wall_ms.mean()),wall_P95_ms=float(table.wall_ms.quantile(.95)),
            wall_max_ms=float(table.wall_ms.max()),cpu_P95_ms=float(table.cpu_ms.quantile(.95)),
            cpu_max_ms=float(table.cpu_ms.max()),candidate_fraction_mean=float(ratio.mean()),
            candidate_fraction_P95=float(ratio.quantile(.95)),candidate_fraction_max=float(ratio.max()),
            mean_candidate_count_ratio=mean_count_ratio,P95_candidate_count_ratio=P95_count_ratio,
            mean_work_passed=bool(mean_count_ratio<=.25 and ratio.mean()<=.25),
            P95_work_passed=bool(P95_count_ratio<=.4 and ratio.quantile(.95)<=.4)))
        for item in json.loads((d/'saved_states.json').read_text()):
            f=local(item['path']);assert sha(f)==item['sha256'];inputs[str(f)]=sha(f)
            states.append((d,s['rate'],s['seed'],f))
    states.sort(key=lambda r:(r[1],r[2],str(r[3])))
    timings=[];answers={};persistent_library();bounded_library();curves()
    for state,(directory,rate,seed,path) in enumerate(states):
        row=torch.load(path,map_location='cpu',weights_only=False);p=problem_from_raw_row(row)
        np.testing.assert_array_equal(p.weights,row['objective_weights'])
        draft=raw_pfq_draft(p,row['obs']);trace=[]
        def save(i):trace.append({k:v.copy() if isinstance(v,np.ndarray) else v for k,v in i.items()})
        reference=solve_integer_reference(p,draft,observer=save)
        def audit(i):
            a=audit_round(i,trace[i['iteration']-1]);rounds.append(dict(state=state,rate=rate,seed=seed,iteration=i['iteration'],**a))
        a,v,r=solve_bounded(p,draft,observer=audit)
        np.testing.assert_array_equal(a,reference[0]);np.testing.assert_array_equal(v,reference[1]);assert r['converged']
        with np.load(directory/'actions.npz') as actions:np.testing.assert_array_equal(a,actions['actions'][row['tti']-1])
        answers[state]=(a,v);auditrows.append(dict(state=state,rate=rate,seed=seed,tti=row['tti'],
            rounds=r['rounds'],full_candidates=reference[2]['candidate_evaluations'],bounded_candidates=r['candidate_evaluations']))
        ledger.append(dict(state=state,full_reference=reference[2],bounded=r))
        for repeat in range(args.repeats):
            arms=('full','cached','bounded');offset=(state+repeat)%3
            for arm in arms[offset:]+arms[:offset]:
                begin=time.perf_counter();problem=problem_from_raw_row(row);raw=raw_pfq_draft(problem,row['obs'])
                aa,vv,rr=solve_bounded(problem,raw) if arm=='bounded' else solve_persistent(problem,raw,PersistentConfig(arm))
                elapsed=1000*(time.perf_counter()-begin)
                np.testing.assert_array_equal(aa,a);np.testing.assert_array_equal(vv,v)
                timings.append(dict(state=state,rate=rate,seed=seed,repeat=repeat,arm=arm,ms=elapsed,
                    candidates=rr['candidate_evaluations']))
        if state%20==0:print('STRESS_AUDIT',state,len(states),rate,flush=True)
    t=pd.DataFrame(timings);per=t.groupby(['state','rate','seed','arm'],as_index=False).ms.median()
    wall=[]
    for rate,g in per.groupby('rate'):
        full=float(g[g.arm=='full'].ms.mean())
        for arm,h in g.groupby('arm'):
            wall.append(dict(rate=rate,arm=arm,states=len(h),mean_ms=float(h.ms.mean()),P95_ms=float(h.ms.quantile(.95)),
                max_ms=float(h.ms.max()),mean_ratio_full=float(h.ms.mean()/full)))
    points=pd.DataFrame(pointrows)
    for i,row in points.iterrows():
        b=points[(points.seed==row.seed)&(points.rate==.8)].iloc[0]
        points.loc[i,'density_mean_vs_base']=row.candidate_fraction_mean/b.candidate_fraction_mean
        points.loc[i,'density_P95_vs_base']=row.candidate_fraction_P95/b.candidate_fraction_P95
    write_csv(out/'native_costs.csv',points.to_dict('records'));write_csv(out/'equivalence.csv',auditrows)
    write_csv(out/'rounds.csv',rounds);write_csv(out/'timing_raw.csv',timings)
    write_csv(out/'wall_summary.csv',wall);write_json(out/'work_ledger.json',ledger)
    write_json(out/'input_manifest.json',inputs)
    for f,digest in inputs.items():assert sha(f)==digest
    passed=bool(all(p['mean_work_passed'] and p['P95_work_passed'] for p in pointrows) and
        all(w['mean_ratio_full']<=.4 for w in wall if w['arm']=='bounded'))
    write_json(out/'completion.json',dict(completed=True,B_entry_passed=passed,equivalence_passed=True,
        audited_states=len(states),audited_rounds=len(rounds),audited_candidates=sum(r['candidates'] for r in rounds),
        UB_order_over_required=sum(r['fresh'] for r in rounds)/sum(r['required'] for r in rounds),
        measured_TTIs=len(pointrows)*4000,new_native_runs=0,training_steps=0,
        density_variation_reported=True,wall_speed_reference='separate serial saved-input controls'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cpu',type=int,default=255);p.add_argument('--repeats',type=int,default=3);main(p.parse_args())

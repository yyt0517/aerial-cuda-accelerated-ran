"""Predeclared cross-load Pareto/oracle analysis and separately paid exact audits."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from .physics import ROOT
from .crossload_native import COEFFICIENTS,RATES,run_key
from .analyze_bounded_stress import local
from .analyze_dpls_moves import write_json,write_csv,sha

def read_points(source,grid_only=False):
    protocol=json.loads((source/'protocol.json').read_text());points=[]
    for run in protocol['reused_runs']:
        points.append(dict(directory=local(run['directory']),family='ttl'))
    for spec in protocol['jobs']:
        family=spec['weight_family']
        if grid_only and family!='ttl':continue
        points.append(dict(directory=source/run_key(spec['seed'],spec['rate'],spec['coefficient'],family),family=family))
    result={};inputs={}
    for point in points:
        d=point['directory'];s=json.loads((d/'summary.json').read_text())
        assert s['native_passed'] and s['pending_cohort']==0
        for f in ('validation.json','reward_validation.json'):assert json.loads((d/f).read_text())['passed']
        pk=pd.read_csv(d/'cohort.csv.gz');assert not pk.censored.any();success=pk[pk.successfully_delivered]
        np.testing.assert_allclose([success.packet_size_bytes.sum()*8/2e6,100*pk.outcome.eq('expired').mean(),
            success.system_time_ms.quantile(.95)],[s['complete_payload_mbps'],s['expiry_pct'],s['success_delay_p95_ms']],rtol=0,atol=1e-8)
        u=pd.read_csv(d/'per_ue.csv').set_index('ue')
        key=(s['rate'],point['family'],s['coefficient'] if point['family']=='ttl' else 0.,s['seed'])
        assert key not in result
        result[key]=dict(summary=s,packets=pk,perue=u,directory=d)
        for f in ('summary.json','validation.json','reward_validation.json','cohort.csv.gz','per_ue.csv','search.csv',
            'launch.json','saved_states.json','actions.npz','traffic_offered_packets.csv'):
            inputs[str(d/f)]=sha(d/f)
    assert len(result)==(45 if grid_only else 63)
    return result,inputs

def aggregate(points,rate,k,seeds=(41,42,43),family='ttl'):
    group=[points[rate,family,k,s] for s in seeds];pk=pd.concat([p['packets'] for p in group],ignore_index=True)
    success=pk[pk.successfully_delivered]
    return dict(rate=rate,load=rate/.8,family=family,coefficient=k,
        payload_mbps=float(np.mean([p['summary']['complete_payload_mbps'] for p in group])),
        expiry_pct=float(100*pk.outcome.eq('expired').mean()),P95_ms=float(success.system_time_ms.quantile(.95)),
        worst_ue_expiry_pct=max(p['summary']['worst_ue_expiry_pct'] for p in group))

def oracle(rows):
    best=max(r['payload_mbps'] for r in rows)
    eligible=[r for r in rows if r['payload_mbps']>=.997*best]
    return min(eligible,key=lambda r:(r['worst_ue_expiry_pct'],r['expiry_pct'],r['P95_ms'],-r['payload_mbps'],r['coefficient']))

def pareto(rows):
    def vector(r):return np.array([-r['payload_mbps'],r['expiry_pct'],r['P95_ms'],r['worst_ue_expiry_pct']])
    for r in rows:
        r['pareto4D']=not any(np.all(vector(s)<=vector(r)) and np.any(vector(s)<vector(r)) for s in rows)

def main(args):
    source=Path(args.source).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    points,inputs=read_points(source,args.grid_only);rows=[];perseed=[];paired=[]
    for rate in RATES:
        grid=[aggregate(points,rate,k) for k in COEFFICIENTS];pareto(grid);rows.extend(grid)
        if not args.grid_only:
            for f in ('mlwdf_pf','exp_pf'):rows.append(aggregate(points,rate,0.,family=f))
        group=[r for r in rows if r['rate']==rate];copies=[r.copy() for r in group];pareto(copies)
        for r,s in zip(group,copies):r['pareto_all_forms']=s['pareto4D']
    for (rate,f,k,seed),p in points.items():
        s=p['summary'];perseed.append(dict(rate=rate,seed=seed,family=f,coefficient=k,payload_mbps=s['complete_payload_mbps'],
            expiry_pct=s['expiry_pct'],P95_ms=s['success_delay_p95_ms'],worst_ue_expiry_pct=s['worst_ue_expiry_pct']))
        base=points[rate,'ttl',2.,seed];b=base['summary'];u,bu=p['perue'],base['perue']
        zeros=set(u.index[u.payload_mbps==0])-set(bu.index[bu.payload_mbps==0])
        gaps=set(u.index[u.longest_no_successful_byte_ttis>=400])-set(bu.index[bu.longest_no_successful_byte_ttis>=400])
        r=dict(rate=rate,seed=seed,family=f,coefficient=k,payload_ratio=s['complete_payload_mbps']/b['complete_payload_mbps'],
            expiry_delta_pp=s['expiry_pct']-b['expiry_pct'],P95_delta_ms=s['success_delay_p95_ms']-b['success_delay_p95_ms'],
            worst_expiry_delta_pp=s['worst_ue_expiry_pct']-b['worst_ue_expiry_pct'],new_zero_UEs=len(zeros),new_TTL_gap_UEs=len(gaps))
        r['B_no_worse_vs_k2']=bool(r['payload_ratio']>=1 and r['expiry_delta_pp']<=0 and r['P95_delta_ms']<=1 and
            s['worst_ue_expiry_pct']<=max(b['worst_ue_expiry_pct']+2,5) and not zeros and not gaps)
        paired.append(r)
    selected=[];near=[];train=[]
    for rate in RATES:
        grid=[r for r in rows if r['rate']==rate and r['family']=='ttl'];chosen=oracle(grid)
        selected.append(chosen.copy());tr=[aggregate(points,rate,k,seeds=(41,42)) for k in COEFFICIENTS]
        train.append(oracle(tr));mins={key:min(r[key] for r in grid) for key in ('expiry_pct','P95_ms','worst_ue_expiry_pct')}
        for r in grid:
            seed_ok=all(points[rate,'ttl',r['coefficient'],s]['summary']['complete_payload_mbps']>=
                .994*max(points[rate,'ttl',k,s]['summary']['complete_payload_mbps'] for k in COEFFICIENTS) for s in (41,42,43))
            near.append(dict(rate=rate,coefficient=r['coefficient'],
                payload_passed=r['payload_mbps']>=.997*max(x['payload_mbps'] for x in grid) and seed_ok,
                expiry_passed=r['expiry_pct']<=mins['expiry_pct']+.1,
                P95_passed=r['P95_ms']<=mins['P95_ms']+1,
                worst_passed=r['worst_ue_expiry_pct']<=mins['worst_ue_expiry_pct']+2))
    for r in near:r['all_near']=all(r[k] for k in ('payload_passed','expiry_passed','P95_passed','worst_passed'))
    universal=[k for k in COEFFICIENTS if all(r['all_near'] for r in near if r['coefficient']==k)]
    varying=len({r['coefficient'] for r in train})>1
    residual=any(all(r['expiry_pct']>0 and r['worst_ue_expiry_pct']>0 for r in rows
        if r['rate']==rate and r['family']=='ttl') for rate in (1.12,1.28))
    entry=bool(varying or residual)
    decision=dict(completed=True,grid_points=45,all_points=len(points),train_only_oracle=train,all_seed_descriptive_oracle=selected,
        train_oracle_varies=varying,all_fixed_high_load_residual_expiry=residual,universal_near_optimal_coefficients=universal,
        learning_signal_present=entry,train_low_dimensional_policy=bool(entry and varying and not universal),
        common_near_optimal_blocks_training=True,
        interpretation='different oracle labels permit development classifier; residual loss is not a lower bound on achievable adaptive benefit; all-seed oracle never used as training label')
    write_csv(out/'aggregate.csv',rows);write_csv(out/'per_seed.csv',perseed);write_csv(out/'paired_vs_k2.csv',paired)
    write_csv(out/'oracle_all_seeds.csv',selected);write_csv(out/'oracle_train_seeds.csv',train);write_csv(out/'near_optimal.csv',near)
    write_json(out/'learning_entry.json',decision)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,(key,label) in zip(axes.flat,[('payload_mbps','Payload Mbps, seed mean'),('expiry_pct','Mature expiry %'),
        ('P95_ms','Successful packet P95 ms'),('worst_ue_expiry_pct','Worst seed/UE expiry %')]):
        for rate in RATES:
            g=sorted([r for r in rows if r['rate']==rate and r['family']=='ttl'],key=lambda r:r['coefficient'])
            ax.plot([r['coefficient'] for r in g],[r[key] for r in g],marker='o',label=f'load x{rate/.8:g}')
        ax.set_xlabel('Fixed TTL coefficient');ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0,0].legend();fig.savefig(out/'crossload_grid.png',dpi=180);fig.savefig(out/'crossload_grid.pdf');plt.close(fig)
    if args.grid_only:
        write_json(out/'input_manifest.json',inputs);write_json(out/'completion.json',dict(completed=True,grid_only=True,learning=decision));return
    # Native pair checks also apply to literature runs. No sampled prefix shortcut.
    for rate in RATES:
        for seed in (41,42,43):
            base=points[rate,'ttl',0.,seed];bd=base['directory'];off=pd.read_csv(bd/'traffic_offered_packets.csv')
            with np.load(bd/'actions.npz') as z:ba=z['actions'][:4000].copy();bh=z['input_hashes'][:4000].copy()
            for (r,f,k,s),p in points.items():
                if r!=rate or s!=seed:continue
                d=p['directory'];pd.testing.assert_frame_equal(off,pd.read_csv(d/'traffic_offered_packets.csv'))
                with np.load(d/'actions.npz') as z:
                    np.testing.assert_array_equal(ba,z['actions'][:4000]);np.testing.assert_array_equal(bh,z['input_hashes'][:4000])
                cols=['ue_id','packet_id','arrival_tti','packet_size_bytes','ttl_tti']
                pd.testing.assert_frame_equal(base['packets'][cols].sort_values(['ue_id','packet_id']).reset_index(drop=True),
                    p['packets'][cols].sort_values(['ue_id','packet_id']).reset_index(drop=True))
    import torch
    from .archive import enable_legacy_types
    from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
    from .persistent_dpls import solve_integer_reference
    from .bounded_dpls import solve_bounded
    from .bounded_campaign import audit_round
    enable_legacy_types();os.sched_setaffinity(0,{args.cpu});ledger=[];equivalence=[]
    protocol=json.loads((source/'protocol.json').read_text())
    for spec in protocol['jobs']:
        d=source/run_key(spec['seed'],spec['rate'],spec['coefficient'],spec['weight_family'])
        with np.load(d/'actions.npz') as z:actions=z['actions'].copy()
        for item in json.loads((d/'saved_states.json').read_text()):
            path=local(item['path']);assert sha(path)==item['sha256'];inputs[str(path)]=sha(path)
            row=torch.load(path,map_location='cpu',weights_only=False);p=problem_from_raw_row(row)
            p.weights=np.ascontiguousarray(row['objective_weights']);draft=raw_pfq_draft(p,row['obs']);trace=[];counts=[]
            def save(i):trace.append({k:v.copy() if isinstance(v,np.ndarray) else v for k,v in i.items()})
            ref=solve_integer_reference(p,draft,observer=save)
            def check(i):counts.append(audit_round(i,trace[i['iteration']-1]))
            a,v,r=solve_bounded(p,draft,observer=check)
            np.testing.assert_array_equal(a,ref[0]);np.testing.assert_array_equal(v,ref[1])
            np.testing.assert_array_equal(a,actions[row['tti']-1])
            ledger.append(dict(**spec,tti=row['tti'],full=ref[2],bounded=r))
            equivalence.append(dict(**spec,tti=row['tti'],rounds=len(counts),candidates=sum(x['candidates'] for x in counts),
                scored=sum(x['fresh'] for x in counts),required=sum(x['required'] for x in counts)))
        print('CROSSLOAD_WEIGHTED_ORACLE',spec,flush=True)
    write_json(out/'weighted_work_ledger.json',ledger);write_csv(out/'weighted_equivalence.csv',equivalence)
    for name,digest in inputs.items():assert sha(local(name))==digest
    write_json(out/'input_manifest.json',inputs)
    write_json(out/'completion.json',dict(completed=True,points=len(points),weighted_states=len(ledger),
        weighted_candidates=sum(x['candidates'] for x in equivalence),new_native_runs=0,gradient_steps=0,
        new_grid=True,old_results_unchanged=True,learning=decision))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--grid-only',action='store_true');p.add_argument('--cpu',type=int,default=243);main(p.parse_args())

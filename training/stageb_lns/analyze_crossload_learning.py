"""Closed-loop attribution of the predeclared small tree, without model retuning."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .analyze_crossload import read_points
from .analyze_bounded_stress import local
from .analyze_dpls_moves import write_json,write_csv,sha

def main(args):
    source=Path(args.source).resolve();learn=Path(args.learning).resolve();out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False);c=json.loads((learn/'completion.json').read_text());assert c['completed']
    if not c['trained']:
        write_json(out/'completion.json',dict(completed=True,trained=False,new_native_runs=0,reason=c['reason']));return
    while not (source/'completion.json').exists():time.sleep(5)
    points,_=read_points(source,False);entry=json.loads((learn/'entry_decision.json').read_text())
    oracle={r['rate']:r['coefficient'] for r in entry['train_only_oracle']}
    descriptive={r['rate']:r['coefficient'] for r in entry['all_seed_descriptive_oracle']}
    rows=[];inputs={};ledger=[];allpk={};choice_rows=[]
    import torch
    from .archive import enable_legacy_types
    from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
    from .bounded_dpls import solve_bounded
    from .persistent_dpls import solve_integer_reference
    from .bounded_campaign import audit_round
    enable_legacy_types();os.sched_setaffinity(0,{args.cpu})
    for run in c['runs']:
        d=local(run['directory']);s=json.loads((d/'summary.json').read_text());rate,seed=s['rate'],s['seed']
        assert s['pending_cohort']==0 and s['native_passed']
        for n in ('validation.json','reward_validation.json'):assert json.loads((d/n).read_text())['passed']
        pk=pd.read_csv(d/'cohort.csv.gz');u=pd.read_csv(d/'per_ue.csv').set_index('ue');allpk[rate,seed]=pk
        chosen=np.load(d/'chosen_coefficients.npy');assert chosen.shape==(8400,)
        measured_choices=chosen[4000:8000]
        choice_rows.append(dict(rate=rate,seed=seed,switches=int(np.sum(measured_choices[1:]!=measured_choices[:-1])),
            nonoracle_settling_TTIs=int(np.sum(chosen[:4000]!=oracle[rate])),
            nonoracle_measured_TTIs=int(np.sum(measured_choices!=oracle[rate])),
            nonoracle_followup_TTIs=int(np.sum(chosen[8000:]!=oracle[rate])),
            nonoracle_absolute_TTIs=(np.flatnonzero(chosen!=oracle[rate])+4001).tolist(),
            **{f'fraction_k{k}':float(np.mean(measured_choices==k)) for k in (0,1,2,4,8)}))
        with np.load(d/'actions.npz') as z:a=z['actions'].copy();h=z['input_hashes'].copy()
        for name,k,family in (('fixed2',2.,'ttl'),('train_oracle',oracle[rate],'ttl'),('descriptive_oracle',descriptive[rate],'ttl'),
            ('mlwdf_pf',0.,'mlwdf_pf'),('exp_pf',0.,'exp_pf')):
            b=points[rate,family,k,seed];bs=b['summary'];bu=b['perue'];bd=b['directory']
            pd.testing.assert_frame_equal(pd.read_csv(d/'traffic_offered_packets.csv'),pd.read_csv(bd/'traffic_offered_packets.csv'))
            with np.load(bd/'actions.npz') as z:
                np.testing.assert_array_equal(a[:4000],z['actions'][:4000]);np.testing.assert_array_equal(h[:4000],z['input_hashes'][:4000])
                baseline_actions=z['actions'].copy()
            if family=='ttl' and np.all(chosen==k):
                np.testing.assert_array_equal(a,baseline_actions,err_msg='constant model must reproduce fixed-policy actions')
            cols=['ue_id','packet_id','arrival_tti','packet_size_bytes','ttl_tti']
            pd.testing.assert_frame_equal(pk[cols].sort_values(['ue_id','packet_id']).reset_index(drop=True),
                b['packets'][cols].sort_values(['ue_id','packet_id']).reset_index(drop=True))
            zeros=set(u.index[u.payload_mbps==0])-set(bu.index[bu.payload_mbps==0])
            gaps=set(u.index[u.longest_no_successful_byte_ttis>=400])-set(bu.index[bu.longest_no_successful_byte_ttis>=400])
            r=dict(rate=rate,seed=seed,baseline=name,baseline_k=k,payload_mbps=s['complete_payload_mbps'],
                expiry_pct=s['expiry_pct'],P95_ms=s['success_delay_p95_ms'],worst_ue_expiry_pct=s['worst_ue_expiry_pct'],
                payload_ratio=s['complete_payload_mbps']/bs['complete_payload_mbps'],expiry_delta_pp=s['expiry_pct']-bs['expiry_pct'],
                P95_delta_ms=s['success_delay_p95_ms']-bs['success_delay_p95_ms'],
                worst_expiry_delta_pp=s['worst_ue_expiry_pct']-bs['worst_ue_expiry_pct'],new_zero_UEs=len(zeros),new_TTL_gap_UEs=len(gaps))
            r['measured_action_difference_fraction']=float(np.mean(np.any(a[8000:12000]!=baseline_actions[8000:12000],axis=(1,2))))
            r['measured_coefficient_difference_fraction']=float(np.mean(chosen[4000:8000]!=k)) if family=='ttl' else None
            r['B_no_worse']=bool(r['payload_ratio']>=1 and r['expiry_delta_pp']<=0 and r['P95_delta_ms']<=1 and
                s['worst_ue_expiry_pct']<=max(bs['worst_ue_expiry_pct']+2,5) and not zeros and not gaps)
            rows.append(r)
        for item in json.loads((d/'saved_states.json').read_text()):
            path=local(item['path']);assert sha(path)==item['sha256'];inputs[str(path)]=sha(path)
            row=torch.load(path,map_location='cpu',weights_only=False);p=problem_from_raw_row(row)
            from .ttl_weights import objective_weights
            expected=objective_weights(p.weights,row['hol_urgency'],chosen[row['tti']-4001])
            np.testing.assert_array_equal(expected.view(np.uint64),np.asarray(row['objective_weights']).view(np.uint64))
            p.weights=np.ascontiguousarray(row['objective_weights']);draft=raw_pfq_draft(p,row['obs']);trace=[]
            def save(i):trace.append({k:v.copy() if isinstance(v,np.ndarray) else v for k,v in i.items()})
            ref=solve_integer_reference(p,draft,observer=save)
            def check(i):audit_round(i,trace[i['iteration']-1])
            aa,vv,rr=solve_bounded(p,draft,observer=check)
            np.testing.assert_array_equal(aa,ref[0]);np.testing.assert_array_equal(vv,ref[1])
            np.testing.assert_array_equal(aa,a[row['tti']-1]);ledger.append(dict(rate=rate,seed=seed,tti=row['tti'],full=ref[2],bounded=rr))
        for n in ('summary.json','validation.json','reward_validation.json','cohort.csv.gz','per_ue.csv','search.csv','launch.json',
            'actions.npz','traffic_offered_packets.csv','chosen_coefficients.npy','policy_identity.json'):
            inputs[str(d/n)]=sha(d/n)
        print('ADAPTIVE_WEIGHTED_ORACLE',rate,seed,flush=True)
    write_csv(out/'paired.csv',rows);write_csv(out/'coefficient_choices.csv',choice_rows)
    write_json(out/'weighted_work_ledger.json',ledger)
    aggregate=[]
    for rate in sorted(oracle):
        pk=pd.concat([allpk[rate,s] for s in (41,42,43)],ignore_index=True);success=pk[pk.successfully_delivered]
        rr=[r for r in rows if r['rate']==rate and r['baseline']=='fixed2']
        aggregate.append(dict(rate=rate,payload_mbps=float(np.mean([r['payload_mbps'] for r in rr])),
            expiry_pct=float(100*pk.outcome.eq('expired').mean()),P95_ms=float(success.system_time_ms.quantile(.95)),
            worst_ue_expiry_pct=max(r['worst_ue_expiry_pct'] for r in rr),
            no_worse_all_seeds_vs_fixed2=all(r['B_no_worse'] for r in rr),
            no_worse_all_seeds_vs_train_oracle=all(r['B_no_worse'] for r in rows if r['rate']==rate and r['baseline']=='train_oracle'),
            no_worse_all_seeds_vs_descriptive_oracle=all(r['B_no_worse'] for r in rows if r['rate']==rate and r['baseline']=='descriptive_oracle'),
            no_worse_all_seeds_vs_mlwdf_pf=all(r['B_no_worse'] for r in rows if r['rate']==rate and r['baseline']=='mlwdf_pf'),
            no_worse_all_seeds_vs_exp_pf=all(r['B_no_worse'] for r in rows if r['rate']==rate and r['baseline']=='exp_pf')))
    write_csv(out/'aggregate.csv',aggregate);write_json(out/'input_manifest.json',inputs)
    for p,digest in inputs.items():assert sha(p)==digest
    write_json(out/'completion.json',dict(completed=True,trained=True,new_native_runs=0,weighted_states=len(ledger),
        no_worse_all_loads_vs_fixed2=all(r['no_worse_all_seeds_vs_fixed2'] for r in aggregate),
        no_worse_all_loads_vs_oracle=all(r['no_worse_all_seeds_vs_train_oracle'] for r in aggregate),
        no_worse_all_loads_vs_descriptive_oracle=all(r['no_worse_all_seeds_vs_descriptive_oracle'] for r in aggregate),
        no_worse_all_loads_vs_mlwdf_pf=all(r['no_worse_all_seeds_vs_mlwdf_pf'] for r in aggregate),
        no_worse_all_loads_vs_exp_pf=all(r['no_worse_all_seeds_vs_exp_pf'] for r in aggregate),
        strict_improvement_any_point_vs_train_oracle=any(r['payload_ratio']>1 or r['expiry_delta_pp']<0 or
            r['P95_delta_ms']<0 or r['worst_expiry_delta_pp']<0 for r in rows if r['baseline']=='train_oracle'),
        default_changed=False,retuned_after_evaluation=False,interpretation='development regime-imitation tree only, no final confirmation/generalization claim'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--learning',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cpu',type=int,default=242);main(p.parse_args())

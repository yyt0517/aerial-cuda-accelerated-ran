"""Mature-packet, per-seed attribution and four-objective TTL coefficient frontier."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .physics import ROOT
from .analyze_dpls_moves import write_json,write_csv,sha
from .analyze_bounded_stress import local

def main(args):
    source,out=Path(args.source).resolve(),Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    c=json.loads((source/'completion.json').read_text());assert c['completed'] and c['mode']=='ttl'
    runs=c['runs']+c['reused_runs'];assert len(runs)==15
    points={};inputs={};rows=[];paired=[];offered={};actions={};cohorts={};perue={}
    for run in runs:
        d=local(run['directory']);s=json.loads((d/'summary.json').read_text());seed=s['seed'];k=s['coefficient']
        assert s['native_passed'] and s['pending_cohort']==0
        for f in ('validation.json','reward_validation.json'):assert json.loads((d/f).read_text())['passed']
        pk=pd.read_csv(d/'cohort.csv.gz');assert not pk.censored.any()
        success=pk[pk.successfully_delivered]
        np.testing.assert_allclose([success.packet_size_bytes.sum()*8/2e6,100*pk.outcome.eq('expired').mean(),
            success.system_time_ms.mean(),success.system_time_ms.quantile(.95)],
            [s['complete_payload_mbps'],s['expiry_pct'],s['success_delay_mean_ms'],s['success_delay_p95_ms']],rtol=0,atol=1.e-8)
        u=pd.read_csv(d/'per_ue.csv').set_index('ue');off=pd.read_csv(d/'traffic_offered_packets.csv')
        with np.load(d/'actions.npz') as z:
            a=z['actions'].copy();h=z['input_hashes'][:4000].copy()
        points[seed,k]=dict(s,directory=str(d));cohorts[seed,k]=pk;perue[seed,k]=u;offered[seed,k]=off;actions[seed,k]=(a,h)
        rows.append(dict(seed=seed,coefficient=k,payload_mbps=s['complete_payload_mbps'],expiry_pct=s['expiry_pct'],
            P95_ms=s['success_delay_p95_ms'],mean_ms=s['success_delay_mean_ms'],worst_ue_expiry_pct=s['worst_ue_expiry_pct'],
            ue_payload_P5_mbps=s['ue_complete_payload_p5_mbps']))
        for f in ('summary.json','validation.json','reward_validation.json','per_ue.csv','cohort.csv.gz','actions.npz',
                  'traffic_offered_packets.csv','search.csv','launch.json','saved_states.json'):
            inputs[str(d/f)]=sha(d/f)
    for (seed,k),s in points.items():
        if k==0:continue
        b=points[seed,0.];u=perue[seed,k];bu=perue[seed,0.]
        pd.testing.assert_frame_equal(offered[seed,k],offered[seed,0.])
        np.testing.assert_array_equal(actions[seed,k][0][:4000],actions[seed,0.][0][:4000])
        np.testing.assert_array_equal(actions[seed,k][1],actions[seed,0.][1])
        columns=['ue_id','packet_id','arrival_tti','packet_size_bytes','ttl_tti']
        pd.testing.assert_frame_equal(cohorts[seed,k][columns].sort_values(['ue_id','packet_id']).reset_index(drop=True),
            cohorts[seed,0.][columns].sort_values(['ue_id','packet_id']).reset_index(drop=True))
        zero=set(u.index[u.payload_mbps==0])-set(bu.index[bu.payload_mbps==0])
        gap=set(u.index[u.longest_no_successful_byte_ttis>=400])-set(bu.index[bu.longest_no_successful_byte_ttis>=400])
        actual_packets=cohorts[seed,k];base_packets=cohorts[seed,0.]
        common=actual_packets[actual_packets.successfully_delivered].merge(
            base_packets[base_packets.successfully_delivered],on=['ue_id','packet_id'],suffixes=('_k','_base'),validate='one_to_one')
        r=dict(seed=seed,coefficient=k,payload_ratio=s['complete_payload_mbps']/b['complete_payload_mbps'],
            expiry_delta_pp=s['expiry_pct']-b['expiry_pct'],P95_delta_ms=s['success_delay_p95_ms']-b['success_delay_p95_ms'],
            mean_delay_delta_ms=s['success_delay_mean_ms']-b['success_delay_mean_ms'],
            worst_expiry_delta_pp=s['worst_ue_expiry_pct']-b['worst_ue_expiry_pct'],
            ue_P5_ratio=s['ue_complete_payload_p5_mbps']/b['ue_complete_payload_p5_mbps'],
            new_zero_UEs=len(zero),new_TTL_gap_UEs=len(gap),
            common_success_packets=len(common),
            common_success_mean_delay_delta_ms=float((common.system_time_ms_k-common.system_time_ms_base).mean()),
            measured_action_difference_fraction=float(np.mean(np.any(actions[seed,k][0][8000:12000]!=actions[seed,0.][0][8000:12000],axis=(1,2)))))
        r['historical_quality_passed']=bool(r['payload_ratio']>=.994 and r['expiry_delta_pp']<=.1 and
            r['P95_delta_ms']<=1 and r['mean_delay_delta_ms']<=.5 and r['ue_P5_ratio']>=.97 and
            s['worst_ue_expiry_pct']<=max(b['worst_ue_expiry_pct']+2,5) and not zero and not gap)
        r['B_no_worse_rule_passed']=bool(r['payload_ratio']>=1 and r['expiry_delta_pp']<=0 and r['P95_delta_ms']<=1 and
            s['worst_ue_expiry_pct']<=max(b['worst_ue_expiry_pct']+2,5) and not zero and not gap)
        paired.append(r)
    # Pay a full independent weighted oracle on the four saved states of each
    # newly run grid point. Reused k0 seeds41/42 were already fully audited by
    # the load stage; do not charge/repeat their40-state audits here.
    import torch
    from .archive import enable_legacy_types
    from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
    from .persistent_dpls import solve_integer_reference
    from .bounded_dpls import solve_bounded
    from .bounded_campaign import audit_round
    enable_legacy_types();weighted=[];weighted_ledger=[]
    for (seed,k),point in sorted(points.items()):
        if k==0 and seed in (41,42):continue
        directory=Path(point['directory'])
        for item in json.loads((directory/'saved_states.json').read_text()):
            path=local(item['path']);assert sha(path)==item['sha256'];inputs[str(path)]=sha(path)
            row=torch.load(path,map_location='cpu',weights_only=False)
            p=problem_from_raw_row(row);p.weights=np.ascontiguousarray(row['objective_weights'],np.float64)
            draft=raw_pfq_draft(p,row['obs']);trace=[]
            def remember(i):trace.append({key:value.copy() if isinstance(value,np.ndarray) else value for key,value in i.items()})
            reference=solve_integer_reference(p,draft,observer=remember);counts=[]
            def audit(i):counts.append(audit_round(i,trace[i['iteration']-1]))
            a,v,r=solve_bounded(p,draft,observer=audit)
            np.testing.assert_array_equal(a,reference[0]);np.testing.assert_array_equal(v,reference[1])
            np.testing.assert_array_equal(a,actions[seed,k][0][row['tti']-1])
            weighted.append(dict(seed=seed,coefficient=k,tti=row['tti'],rounds=len(counts),
                candidates=sum(c['candidates'] for c in counts),fresh=sum(c['fresh'] for c in counts),
                required=sum(c['required'] for c in counts)))
            weighted_ledger.append(dict(seed=seed,coefficient=k,tti=row['tti'],full=reference[2],bounded=r))
        print('TTL_WEIGHTED_AUDIT',seed,k,flush=True)
    write_csv(out/'weighted_equivalence.csv',weighted);write_json(out/'weighted_work_ledger.json',weighted_ledger)
    aggregate=[]
    for k in (0.,.25,.5,1.,2.):
        pk=pd.concat([cohorts[seed,k] for seed in (41,42,43)],ignore_index=True)
        success=pk[pk.successfully_delivered]
        group=[points[seed,k] for seed in (41,42,43)]
        matches=[r for r in paired if r['coefficient']==k]
        payload=np.mean([r['complete_payload_mbps'] for r in group])
        basepayload=np.mean([points[seed,0.]['complete_payload_mbps'] for seed in (41,42,43)])
        aggregate.append(dict(coefficient=k,payload_mbps=float(payload),expiry_pct=float(100*pk.outcome.eq('expired').mean()),
            P95_ms=float(success.system_time_ms.quantile(.95)),worst_ue_expiry_pct=max(r['worst_ue_expiry_pct'] for r in group),
            payload_ratio_mean=float(payload/basepayload),
            historical_quality_all_seeds=bool(k==0 or (all(r['historical_quality_passed'] for r in matches) and payload/basepayload>=.997)),
            B_no_worse_all_seeds=bool(k==0 or all(r['B_no_worse_rule_passed'] for r in matches))))
    for r in aggregate:
        a=np.array([-r['payload_mbps'],r['expiry_pct'],r['P95_ms'],r['worst_ue_expiry_pct']])
        r['pareto4D']=not any(np.all(np.array([-s['payload_mbps'],s['expiry_pct'],s['P95_ms'],s['worst_ue_expiry_pct']])<=a) and
            np.any(np.array([-s['payload_mbps'],s['expiry_pct'],s['P95_ms'],s['worst_ue_expiry_pct']])<a) for s in aggregate)
    write_csv(out/'per_seed.csv',rows);write_csv(out/'paired.csv',paired);write_csv(out/'frontier.csv',aggregate)
    write_json(out/'protocol.json',dict(source=str(source),coefficients=[0,.25,.5,1,2],seeds=[41,42,43],
        aggregation='mean seed payload; expiry and success-P95 pooled across mature packets; worst expiry=max over all seed/UE pairs',
        pareto='maximize payload; minimize expiry/P95/worstUE expiry; all four objectives, no outcome-driven scalar score',
        B_rule='perseed payload/expiry no worse, P95<=baseline+1ms, worstUE<=max(baseworst+2pp,5%), no new zero or TTL gap',
        baseline='same exact bounded solver, same initialized FIFO/EMA/native PHY; only weight urgency coefficient changes',
        confidence='exposed training/development frontier, no holdout/generalization claim; no learned policy fitted'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=plt.cm.viridis(np.linspace(.1,.9,5));fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for r,color in zip(aggregate,colors):
        marker='D' if r['pareto4D'] else 'o'
        axes[0].scatter(r['payload_mbps'],r['expiry_pct'],c=[color],s=75,marker=marker)
        axes[1].scatter(r['P95_ms'],r['worst_ue_expiry_pct'],c=[color],s=75,marker=marker)
        for ax,x,y in ((axes[0],r['payload_mbps'],r['expiry_pct']),(axes[1],r['P95_ms'],r['worst_ue_expiry_pct'])):
            ax.annotate(f" k={r['coefficient']:g}",(x,y),xytext=(4,5),textcoords='offset points',fontsize=9)
    axes[0].set_xlabel('Complete payload (Mbps), mean of 3 seeds');axes[0].set_ylabel('Mature packet expiry (%)')
    axes[1].set_xlabel('Successful packet P95 (ms), pooled');axes[1].set_ylabel('Worst seed/UE expiry (%)')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Frozen causal TTL coefficient sweep; diamonds = four-objective Pareto points')
    fig.savefig(out/'ttl_frontier.png',dpi=180);fig.savefig(out/'ttl_frontier.pdf');plt.close(fig)
    frame=pd.DataFrame(rows);fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
    for ax,(column,label) in zip(axes.flat,[('payload_mbps','Payload (Mbps)'),('expiry_pct','Expiry (%)'),
        ('P95_ms','Successful P95 (ms)'),('worst_ue_expiry_pct','Worst UE expiry (%)')]):
        for seed in (41,42,43):
            f=frame[frame.seed==seed].sort_values('coefficient');ax.plot(f.coefficient,f[column],marker='o',label=f'seed{seed}')
        ax.set_xlabel('TTL urgency coefficient');ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0,0].legend();fig.savefig(out/'ttl_per_seed.png',dpi=180);fig.savefig(out/'ttl_per_seed.pdf');plt.close(fig)
    for f,digest in inputs.items():assert sha(f)==digest
    write_json(out/'input_manifest.json',inputs)
    write_json(out/'completion.json',dict(completed=True,points=15,coefficients=5,seeds=3,
        parameter_scan_only=True,gradient_training_steps=0,ranking_training_steps=0,
        Pareto_coefficients=[r['coefficient'] for r in aggregate if r['pareto4D']],
        B_no_worse_nonzero_coefficients=[r['coefficient'] for r in aggregate if r['coefficient']>0 and r['B_no_worse_all_seeds']],
        new_native_runs=0,default_changed=False,weighted_equivalence_states=len(weighted),
        weighted_equivalence_candidates=sum(r['candidates'] for r in weighted)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);main(p.parse_args())

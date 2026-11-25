"""Read-only diagnosis of frozen seven-cell trajectories; no new policy rollout."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import torch
from scipy.signal import lfilter
from training.stage2_closure.revise_metrics import FIELDS

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'deliverables/seven_cell_boundary08_20261105_v1'
NEW=ROOT/'deliverables/seven_cell_boundary08_multistart_20261105_v1'
OUT=ROOT/'deliverables/seven_cell_delay_diagnosis_20261106_v1'
ARMS=('deficit','mlwdf_pf','da_three_branch')
FIRST,LAST=8001,28000

def directory(arm,seed):
    return (NEW if arm=='da_three_branch' else OLD)/'campaign'/f'{arm}_s{seed}'

def mean(x):return float(np.mean(x))

def trace(arm,seed):
    path=directory(arm,seed)
    raw=np.memmap(path/'metrics.bin',dtype='<f8',mode='r').reshape(28400,84,len(FIELDS))
    get=lambda n:np.asarray(raw[:LAST,:,FIELDS.index(n)])
    queue=get('pre_queue_bytes');post=get('post_queue_bytes');good=get('successful_bytes')
    np.testing.assert_array_equal(queue,get('previous_post_queue_bytes')+get('accepted_bytes')-get('expired_bytes'))
    np.testing.assert_array_equal(post,queue-get('scheduled_dequeue_bytes'))
    # Bridge feedback is float32: the online .005*feedback term rounds before
    # it is added to the float64 EMA. Preserve that operation order exactly.
    increment=(.005*good.astype(np.float32)).astype(np.float64)
    ema=np.vstack([np.zeros((1,84)),lfilter([1.],[1,-.995],increment,axis=0)[:-1]])
    terminals=pd.read_csv(path/'packet_outcome_samples.csv',usecols=['ue_id','packet_id','arrival_tti','terminal_tti'])
    terminals=terminals.sort_values(['ue_id','arrival_tti','packet_id'])
    age=np.ones_like(queue);clock=np.arange(1,LAST+1)
    for u,z in terminals.groupby('ue_id'):
        born=z.arrival_tti.to_numpy();end=z.terminal_tti.to_numpy()
        assert np.all(np.diff(end)>=0)
        idx=np.searchsorted(end,clock,side='left');assert idx.max()<len(end)
        active=queue[:,u]>0
        np.testing.assert_array_equal(born[idx]<=clock,active)
        age[:,u]=np.where(active,np.maximum(1,clock-born[idx]),1)
    base=np.clip(2400/np.maximum(ema,1e-9),1,4);base/=base.mean(axis=1,keepdims=True)
    da=age/np.maximum(ema,1e-9);da/=da.mean(axis=1,keepdims=True)
    checked=[]
    for file in sorted((path/'states').glob('*.pt')):
        r=torch.load(file,map_location='cpu',weights_only=False);i=int(r['tti'])-1
        np.testing.assert_array_equal(queue[i],r['snapshot']['queue'])
        np.testing.assert_allclose(ema[i],r['ema'],rtol=2e-13,atol=1e-9)
        np.testing.assert_allclose(age[i],np.maximum(1,400*r['hol_urgency']),rtol=0,atol=1e-12)
        np.testing.assert_allclose((base if arm=='deficit' else da)[i],r['objective_weights'],rtol=2e-13,atol=1e-12)
        checked.append(int(r['tti']))
    sl=slice(FIRST-1,LAST);q=queue[sl];a=age[sl];e=ema[sl];b=base[sl];d=da[sl]
    active=q>0;w=b if arm=='deficit' else d
    pre_count=np.ceil(queue/3000);post_count=np.ceil(post/3000)
    empty=lambda x:float(np.mean(x))
    summary=dict(arm=arm,seed=seed,active_pct=100*mean(active),queue_mean_mb=mean(post[sl].sum(1))/1e6,
        queue_packets_mean=mean(post_count[sl].sum(1)),hol_mean_ms=mean(a[active])/2,
        hol_ge100ms_pct=100*mean(a[active]>=200),hol_ge180ms_pct=100*mean(a[active]>=360),
        ema_lt600_pct=100*mean(e[active]<600),weight_p05=float(np.quantile(w[active],.05)),
        weight_p50=float(np.quantile(w[active],.5)),weight_p95=float(np.quantile(w[active],.95)),
        weight_ratio_mean=mean(w.max(1)/w.min(1)),da_ratio_mean=mean(d.max(1)/d.min(1)),
        hol_p95_p05_ratio_mean=mean(np.quantile(a,.95,axis=1)/np.quantile(a,.05,axis=1)),
        inverse_ema_p95_p05_ratio_mean=mean(np.quantile(e,.95,axis=1)/np.quantile(e,.05,axis=1)),
        scheduled_mbps=get('scheduled_dequeue_bytes')[sl].sum()*8/1e7,
        expired_remaining_mbps=get('expired_bytes')[sl].sum()*8/1e7,
        failed_mbps=get('failed_bytes')[sl].sum()*8/1e7,
        offered_mbps=get('offered_bytes')[sl].sum()*8/1e7,
        goodput_bytes_per_prg=good[sl].sum()/get('actual_prg_count')[sl].sum(),
        occupancy_pct=100*get('actual_prg_count')[sl].sum()/(20000*7*17))
    logs=np.log(np.maximum(a,1));loge=np.log(np.maximum(e,1e-9));logw=np.log(d)
    def centered_corr(x,y):
        x=x-x.mean(1,keepdims=True);y=y-y.mean(1,keepdims=True)
        return mean((x*y).sum(1)/np.sqrt((x*x).sum(1)*(y*y).sum(1)))
    summary.update(da_logweight_logage_corr_mean=centered_corr(logw,logs),
        da_logweight_loginverseema_corr_mean=centered_corr(logw,-loge))
    series=[]
    for first in range(0,LAST,200):
        s=slice(first,first+200);act=queue[s]>0
        series.append(dict(arm=arm,seed=seed,time_s=(first+100)*.0005,
            queue_mb=mean(post[s].sum(1))/1e6,queue_packets=mean(post_count[s].sum(1)),
            hol_mean_ms=mean(age[s][act])/2,hol_ge180ms_pct=100*mean(age[s][act]>=360),
            goodput_mbps=good[s].sum()*8/1e5,expiry_remaining_mbps=get('expired_bytes')[s].sum()*8/1e5))
    per=pd.read_csv(path/'per_ue.csv')
    per['arm']=arm;per['seed']=seed;per['queue_mean_kb']=post[sl].mean(0)/1000
    per['hol_mean_ms']=a.mean(0)/2;per['hol_ge180ms_pct']=100*(a>=360).mean(0)
    per['weight_mean']=w.mean(0);per['da_weight_mean']=d.mean(0);per['base_weight_mean']=b.mean(0)
    per['ema_mean']=e.mean(0);per['prgs_mean']=get('actual_prg_count')[sl].mean(0)
    per['service_bytes_per_prg']=good[sl].sum(0)/np.maximum(get('actual_prg_count')[sl].sum(0),1)
    per['offered_mbps']=get('offered_bytes')[sl].sum(0)*8/1e7
    per['mean_packets_queued']=post_count[sl].mean(0)
    print('TRACE',arm,seed,summary['queue_mean_mb'],summary['hol_mean_ms'],flush=True)
    return summary,series,per,dict(arm=arm,seed=seed,checked_saved_ttis=checked,queue_hol_ema_weights_reconstructed=True)

def packets(seed):
    frames={};summaries=[];hists=[]
    cols=['packet_id','ue_id','system_time_ms','successfully_delivered','outcome']
    for arm in ARMS:
        p=directory(arm,seed);c=pd.read_csv(p/'cohort.csv.gz',usecols=cols).sort_values('packet_id').reset_index(drop=True)
        frames[arm]=c;g=c[c.successfully_delivered];bad=c[(~c.successfully_delivered)&c.outcome.ne('expired')]
        s=dict(arm=arm,seed=seed,successful_packets=len(g),cohort_packets=len(c),mean_success_delay_ms=g.system_time_ms.mean(),
            mean_outcome_delay_ms=c.system_time_ms.mean(),failed_completion_delay_ms=bad.system_time_ms.mean())
        for limit in [10,25,50,100,150,180,195,200]:s[f'success_le{limit}ms_pct']=100*mean(g.system_time_ms<=limit)
        for quantile in [.1,.25,.5,.75,.9,.95]:s[f'success_q{int(quantile*100)}_ms']=float(g.system_time_ms.quantile(quantile))
        summaries.append(s)
        counts=np.bincount((2*g.system_time_ms.to_numpy()).astype(int),minlength=401);cdf=np.cumsum(counts)/len(g)
        for i in range(1,401):hists.append(dict(arm=arm,seed=seed,delay_ms=i/2,success_cdf=float(cdf[i])))
    comparisons=[];ue_pairs=[]
    b=frames['deficit']
    for arm in ARMS[1:]:
        a=frames[arm];np.testing.assert_array_equal(a[['packet_id','ue_id']],b[['packet_id','ue_id']])
        both=a.successfully_delivered & b.successfully_delivered
        ax=a.successfully_delivered & ~b.successfully_delivered;bx=~a.successfully_delivered & b.successfully_delivered
        comparisons.append(dict(arm=arm,seed=seed,both_success_count=int(both.sum()),
            both_fraction_of_new_success_pct=100*both.sum()/a.successfully_delivered.sum(),
            both_success_new_delay_ms=a.loc[both,'system_time_ms'].mean(),both_success_dpls_delay_ms=b.loc[both,'system_time_ms'].mean(),
            same_packet_delay_delta_ms=(a.loc[both,'system_time_ms']-b.loc[both,'system_time_ms']).mean(),
            same_packet_slower_pct=100*mean(a.loc[both,'system_time_ms']>b.loc[both,'system_time_ms']),
            new_only_success_count=int(ax.sum()),new_only_success_delay_ms=a.loc[ax,'system_time_ms'].mean(),
            dpls_only_success_count=int(bx.sum()),dpls_only_success_delay_ms=b.loc[bx,'system_time_ms'].mean()))
        for u in range(84):
            ix=both&a.ue_id.eq(u)
            ue_pairs.append(dict(arm=arm,seed=seed,ue=u,both_success_count=int(ix.sum()),
                same_packet_delay_delta_ms=(a.loc[ix,'system_time_ms']-b.loc[ix,'system_time_ms']).mean()))
    print('PACKETS',seed,flush=True)
    return summaries,hists,comparisons,ue_pairs

def guards():
    rows=[]
    order=['original_da','from_dpls_da','dpls']
    for seed in [41,42,43]:
        z=pd.read_csv(directory('da_three_branch',seed)/'search.csv.gz');z=z[z.tti.between(FIRST,LAST)]
        scores=z[[b+'_score_q' for b in order]].to_numpy();service=z[[b+'_expected_bytes' for b in order]].to_numpy()
        for beta in [0.,.9,.95,.98,1.]:
            eligible=service>=beta*service[:,2:3]-1e-8
            selected=np.where(eligible,scores,-1).argmax(1);i=np.arange(len(z));x=service[i,selected]
            rows.append(dict(seed=seed,min_expected_service_ratio=beta,
                expected_service_vs_dpls_pct=100*(x.mean()/service[:,2].mean()-1),
                da_score_vs_dpls_mean_gain_pct=100*mean(scores[i,selected]/scores[:,2]-1),
                dpls_selected_pct=100*mean(selected==2),original_da_selected_pct=100*mean(selected==0),from_dpls_da_selected_pct=100*mean(selected==1)))
    return rows

def reweight_saved():
    rows=[];details=[]
    for seed in [41,42,43]:
        for file in sorted((directory('da_three_branch',seed)/'states').glob('*.pt')):
            r=torch.load(file,map_location='cpu',weights_only=False)
            if int(r['tti'])<FIRST:continue
            base=r['deficit_weights'];h=np.asarray(r['hol_urgency']);q=np.asarray(r['snapshot']['queue'])
            names=['original_da','from_dpls_da','dpls'];v=np.array([r['branches'][n]['values'][:,6] for n in names]);svc=v.sum(1)
            candidates={'da':r['objective_weights'],'dpls':base}
            age=np.maximum(1.,400*h)
            candidates['capped_pf_times_age']=base*age
            candidates['uncapped_pf_base_plus_hol']=2400/np.maximum(r['ema'],1e-9)*(1+.25*h)
            for lam in [.05,.1,.25,.5,1.]:candidates[f'bounded_hol_{lam}']=base*(1+lam*h)
            for lam in [.25,.5]:candidates[f'bounded_queue_hol_{lam}']=base*(1+lam*h+.25*np.minimum(q/(2400*400),1))
            for name,w in candidates.items():
                w=w/w.mean();scores=np.rint(1e6*v*w).astype(np.int64).sum(1);i=int(scores.argmax())
                rows.append(dict(seed=seed,tti=int(r['tti']),objective=name,chosen=names[i],
                    expected_service_vs_dpls_pct=100*(svc[i]/svc[2]-1),weight_ratio=float(w.max()/w.min())))
            chosen=names.index(r['selected_branch']);delta=v[chosen]-v[2]
            for u in range(84):details.append(dict(seed=seed,tti=int(r['tti']),ue=u,hol_ms=max(1,400*h[u])/2,
                ema=r['ema'][u],queue_bytes=q[u],da_weight=r['objective_weights'][u],dpls_weight=base[u],
                dpls_expected_bytes=v[2,u],selected_expected_bytes=v[chosen,u],service_delta=delta[u]))
    return rows,details

def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'validation.json').exists(), 'Refuse to overwrite completed diagnosis'
    rows=[];series=[];per=[];checks=[]
    for seed in [41,42,43]:
        for arm in ARMS:
            a,b,c,d=trace(arm,seed);rows.append(a);series.extend(b);per.append(c);checks.append(d)
    pd.DataFrame(rows).to_csv(OUT/'trajectory_summary.csv',index=False)
    pd.DataFrame(series).to_csv(OUT/'queue_time_series.csv',index=False)
    pd.concat(per,ignore_index=True).to_csv(OUT/'per_ue_diagnosis.csv',index=False)
    sums=[];hist=[];pairs=[];ue=[]
    for seed in [41,42,43]:
        a,b,c,d=packets(seed);sums.extend(a);hist.extend(b);pairs.extend(c);ue.extend(d)
    for name,data in [('packet_delay_summary',sums),('delay_cdf',hist),('matched_packets',pairs),('matched_packets_per_ue',ue),('same_state_service_guards',guards())]:
        pd.DataFrame(data).to_csv(OUT/(name+'.csv'),index=False)
    a,b=reweight_saved();pd.DataFrame(a).to_csv(OUT/'saved_candidate_reweight.csv',index=False)
    pd.DataFrame(b).to_csv(OUT/'saved_service_transfers.csv',index=False)
    (OUT/'validation.json').write_text(json.dumps(dict(passed=True,trace_runs=9,checks=checks,matched_cohort_ids_exact=True,
        scope='Retrospective diagnosis and same-state re-ranking only; no changed objective closed-loop run.'),indent=2)+'\n')
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    print('COMPLETE',OUT,flush=True)

if __name__=='__main__':main()

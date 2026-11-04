"""Read-only terminal risk, candidate-oracle and equivalent-cache audit."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from .terminal_service_campaign import NAMES
from .terminal_risk import candidate_features,RiskModel,RISK_CAP
from .ranked_dpls_campaign import read_rows,identity
from .no_ctbs_inputs import problem_from_raw_row
from .resource_conditions import satisfies
from .resource_condition_campaign import compressed_condition,REGISTRY
from .shortlist_trajectory import service_metrics
from .physics import ROOT,EXPECTED,DEQUEUE
from .bounded_probes import csv_write
from .report_joint_reorder import read,mean
from .experiment import write_json,sha

PAIRS=[('cached_maxj','existing2'),('continuous','cached_maxj'),('attainable','cached_maxj'),('calibrated','cached_maxj'),('calibrated','attainable')]


def audit_phase(out,phase,model):
    rows,_=read_rows((44,45) if phase=='validation' else (46,47,48),evaluation=phase=='evaluation')
    ttis=(4001,8001,18000) if phase=='validation' else (3001,4001,5000);rows=[r for r in rows if r['tti'] in ttis]
    raw=read(out/f'{phase}_measurements.csv');state=read(out/f'{phase}_per_state.csv');summary=read(out/f'{phase}_summary.csv')
    values=np.load(out/f'{phase}_values.npz');actions=np.load(out/f'{phase}_actions.npz')
    observed=np.load(out/f'{phase}_observed_values.npz');visited=np.load(out/f'{phase}_observed_actions.npz')
    stats=[];selection=[];perue=[];predictions={};paired=[]
    index={(r['state_id'],r['arm']):r for r in state}
    for r in state:
        rr=[v for v in raw if (v['state_id'],v['arm'])==(r['state_id'],r['arm'])]
        assert len(rr)==3 and float(r['e2e_ms'])==float(np.median([float(v['e2e_ms']) for v in rr]))
        if r['arm']!='full':assert len({v['query_sha256'] for v in rr})==len({v['archive_sha256'] for v in rr})==1
    for r in summary:
        rr=[v for v in state if v['arm']==r['arm']]
        q=all(float(v['ratio'])>=.995 for v in rr)
        c=all(int(v['newly_unserved_ues'])==0 and min(float(v['positive_p5_ratio']),float(v['positive_weak_ratio']))>=.98 for v in rr)
        assert (r['quality_pass']=='True')==q and (r['coverage_pass']=='True')==c
        assert np.isclose(mean(rr,'score'),float(r['mean_j'])) and np.isclose(mean(rr,'e2e_ms'),float(r['mean_ms']))
    for row in rows:
        sid=identity(row);p=problem_from_raw_row(row);ref=values[f'{sid}|full'];c=compressed_condition(p,actions[f'{sid}|full'],ref,'joint212')
        vv=observed[f'{sid}|cached_maxj'];aa=visited[f'{sid}|cached_maxj'];floor=float(index[sid,'cached_maxj']['floor'])
        keep=satisfies(p,aa,c)&(vv[:,:,DEQUEUE].sum(1)>=floor);vv=vv[keep];aa=aa[keep];memoids=np.flatnonzero(keep)
        x,j,h,names=candidate_features(p,c,vv);prob=model.predict(x);labels=[service_metrics(v,ref) for v in vv]
        cov=np.array([v['newly_unserved_ues']==0 and v['positive_p5_ratio']>=.98 and v['positive_weak_ratio']>=.98 for v in labels])
        jr=j/float(ref[:,EXPECTED]@p.weights);joint=cov&(jr>=.995);allowed=j+1.e-8>=(1-RISK_CAP)*j.max();good=joint&allowed;safe=prob>=.9
        group=(vv[:,:,EXPECTED].max(0)>1)&(c.counts>0);target=ref[:,EXPECTED]>1.
        clipped=(abs((x-model.arrays['mean'])/model.arrays['scale'])>8).any(1)
        stats.append(dict(state_id=sid,seed=row['seed'],candidates=len(vv),joint_candidates=int(joint.sum()),allowed_candidates=int(allowed.sum()),
            allowed_joint_candidates=int(good.sum()),oracle_best_j_ratio=float(jr[good].max()) if good.any() else None,
            max_probability_joint=float(prob[good].max()) if good.any() else None,max_probability_bad=float(prob[allowed&~cov].max()) if (allowed&~cov).any() else None,
            safe_candidates=int((allowed&safe).sum()),safe_bad_candidates=int((allowed&safe&~cov).sum()),
            brier=float(np.mean((prob-cov)**2)),allowed_brier=float(np.mean((prob[allowed]-cov[allowed])**2)),
            attainable_extra_ues=int((group&~target).sum()),attainable_missing_ues=int((target&~group).sum()),
            saturated_feature_candidates=int(clipped.sum())))
        predictions[sid]=np.c_[memoids,jr,h,prob,cov,allowed,joint]
        for arm in NAMES[1:]:
            selected=actions[f'{sid}|{arm}'];ii=np.flatnonzero(np.all(aa==selected,axis=(1,2)));assert len(ii)==1;i=int(ii[0])
            r=index[sid,arm]
            selection.append(dict(state_id=sid,seed=row['seed'],arm=arm,memo_index=int(memoids[i]),probability=float(prob[i]),heuristic=float(h[i]),
                j_ratio=float(jr[i]),p5_ratio=float(labels[i]['positive_p5_ratio']),weak_ratio=float(labels[i]['positive_weak_ratio']),
                coverage_pass=bool(cov[i]),joint_pass=bool(joint[i]),internal_j_loss=float(1-j[i]/j.max()),
                terminal_safe_found=r.get('terminal_safe_found',''),maxj_probability=float(prob[np.argmax(j)])))
        for u in np.flatnonzero(group|target):
            d=vv[0,u,DEQUEUE]
            perue.append(dict(state_id=sid,ue=int(u),reference_positive=bool(target[u]),reference_expected=float(ref[u,EXPECTED]),quota=float(d),
                archive_max=float(vv[:,u,EXPECTED].max()),archive_mean=float(vv[:,u,EXPECTED].mean()),
                **{arm:float(values[f'{sid}|{arm}'][u,EXPECTED]) for arm in NAMES[1:]}))
        for new,old in PAIRS:
            a=index[sid,new];b=index[sid,old]
            paired.append(dict(state_id=sid,seed=row['seed'],new=new,old=old,j_delta=float(a['score'])-float(b['score']),
                delta_pp=100*(float(a['ratio'])-float(b['ratio'])),ms_delta=float(a['e2e_ms'])-float(b['e2e_ms']),
                p5_delta=float(a['positive_p5_ratio'])-float(b['positive_p5_ratio'])))
    aggregate=[]
    for new,old in PAIRS:
        rr=[r for r in paired if (r['new'],r['old'])==(new,old)]
        aggregate.append(dict(new=new,old=old,j_wins=sum(r['j_delta']>1 for r in rr),j_losses=sum(r['j_delta']<-1 for r in rr),j_ties=sum(abs(r['j_delta'])<=1 for r in rr),
            mean_j_delta=mean(rr,'j_delta'),mean_ms_delta=mean(rr,'ms_delta'),mean_j_delta_pp=mean(rr,'delta_pp')))
    work=[]
    fields=('bank_ms','matching_ms','generation_ms','proxy_ms','service_ms','proxy_candidates','proposal_attempts','proxy_gamma_array_reads',
        'seed_gamma_array_reads','feature_roots','iterations','candidate_evaluations','mmse_requests','mmse_misses')
    extra=('terminal_ms','pool_cache_hits','pool_cache_misses','proxy_cache_hits','proxy_computed','cache_payload_bytes')
    for arm in NAMES[1:]:
        rr=[r for r in state if r['arm']==arm]
        work.append(dict(arm=arm,**{k:mean(rr,k) for k in fields},**({k:mean(rr,k) for k in extra} if arm!='existing2' else {})))
    for name,data in [('risk_diagnostics',stats),('terminal_diagnostics',selection),('ue_diagnostics',perue),('paired',paired),('paired_summary',aggregate)]:csv_write(out/f'{phase}_{name}.csv',data)
    write_json(out/f'{phase}_work.json',work);np.savez_compressed(out/f'{phase}_candidate_predictions.npz',**predictions)
    write_json(out/f'{phase}_diagnostic_schema.json',dict(columns=['memo_index','J/fullJ','attainable_score','calibrated_probability','coverage_label','inside2pct_cap','joint_label'],
        posthoc=True,additional_native_scores=0,retraining=False,threshold_tuning=False))
    return dict(summary=summary,paired=aggregate,work=work,risk=stats,terminal=selection)


def plot(out,val,dev):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for ax,phase,data in zip(axes,('Validation: 12 exposed states','Development: 9 exposed states'),(val,dev)):
        rr={r['arm']:r for r in data['summary']};full=float(rr['full']['mean_ms'])
        for i,arm in enumerate(NAMES[1:]):
            r=rr[arm];ms=float(r['mean_ms']);passing=r['conditional_target_pass']=='True'
            ax.barh(i,ms,color='#237a57' if passing else '#ad5d38',alpha=.85,height=.6)
            label=f"{ms:.2f} ms  |  J {float(r['mean_relative_j_pct']):+.3f}%  |  P5 min {float(r['worst_positive_p5']):.3f}"
            ax.text(2,i,label,va='center',fontsize=8,color='white')
        ax.axvline(.8*full,color='black',ls='--',lw=1)
        ax.text(.8*full,.99,'80% full cost',transform=ax.get_xaxis_transform(),ha='right',va='top',fontsize=8)
        ax.set_yticks(range(5),NAMES[1:]);ax.invert_yaxis();ax.set_xlim(0,max(float(rr[k]['mean_ms']) for k in NAMES[1:])*1.28)
        ax.set(title=phase,xlabel='Complete execution ms (teacher excluded)');ax.grid(axis='x',alpha=.15)
    fig.suptitle('Cache saves cost; calibrated selector has no independent quality / coverage gain')
    fig.savefig(out/'quality_cost.png',dpi=170);fig.savefig(out/'quality_cost.pdf');plt.close(fig)


def main(source):
    out=Path(source).resolve();assert json.loads((out/'completion.json').read_text())['completed']
    model=RiskModel.load(out/'risk_model.npz');val=audit_phase(out,'validation',model);dev=audit_phase(out,'evaluation',model)
    protocol=json.loads((out/'protocol.json').read_text());manifest=json.loads((out/'source_manifest.json').read_text())
    for name,digest in manifest.items():assert sha(out/'source'/name)==sha(ROOT/'training/stageb_lns'/name)==digest
    assert sha(REGISTRY)==protocol['registry_sha256'] and sha(ROOT/'training/stageb_lns/physics_kernel.cpp')==protocol['physics_source_sha256']
    assert sha(out/'risk_model.npz')==json.loads((out/'model_frozen.json').read_text())['sha256']
    tests=Path('/tmp/terminal_service_tests_20261103.log').read_text();assert 'Ran 74 tests' in tests and '\nOK\n' in tests
    shutil.copyfile('/tmp/terminal_service_tests_20261103.log',out/'tests.log');shutil.copyfile('/tmp/terminal_service_execution_20261103.log',out/'execution.log')
    ledger={}
    for phase in ('training','validation','evaluation'):
        rr=read(out/f'{phase}_teacher_work.csv');ledger[phase]=dict(decisions=len(rr),exact_scores=sum(int(r['candidate_evaluations']) for r in rr),teacher_ms=sum(float(r['teacher_ms']) for r in rr))
    ledger['total']={k:sum(v[k] for v in ledger.values()) for k in ('decisions','exact_scores','teacher_ms')};write_json(out/'teacher_ledger.json',ledger)
    write_json(out/'posthoc_audit.json',dict(passed=True,additional_native_scores=0,model_unchanged=True,source_registry_physics_valid=True,
        timing_and_gates_recomputed=True,candidate_labels_offline_only=True,selection_trace_recovered=True))
    plot(out,val,dev)
    print(json.dumps(dict(validation=val['summary'],development=dev['summary'],validation_risk=val['risk'],ledger=ledger),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);main(p.parse_args().source)

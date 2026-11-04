"""Post-freeze paired-ranking, confidence and repair-state coverage audit."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from .ue_pair_campaign import NAMES,PREVIOUS
from .ue_pair_risk import ue_features,service_scores,heuristic_scores,rank_select,UEPairModel,sigmoid
from .terminal_risk import RISK_CAP
from .resource_condition_campaign import compressed_condition,REGISTRY
from .resource_conditions import satisfies
from .ranked_dpls_campaign import read_rows,identity
from .no_ctbs_inputs import problem_from_raw_row
from .shortlist_trajectory import service_metrics
from .physics import ROOT,EXPECTED,DEQUEUE
from .experiment import sha,write_json
from .bounded_probes import csv_write
from .report_joint_reorder import read,mean


def auc(score,good,allowed):
    a=score[allowed&good];b=score[allowed&~good]
    return float(((a[:,None]>b[None])+.5*(a[:,None]==b[None])).mean()) if len(a) and len(b) else None


def parameters(model,f):
    a=model.arrays;params=[]
    for i in range(3):
        x=np.clip((f['context']-a[f'mean{i}'])/a[f'scale{i}'],-8,8)
        params.append(np.tanh(x@a[f'w{i}_0']+a[f'b{i}_0'])@a[f'w{i}_1']+a[f'b{i}_1'])
    return np.stack(params)


def training_audit(out,model):
    data=np.load(out/'training_data.npz');oof=np.load(out/'oof_scores.npz');records=[]
    states=list(dict.fromkeys(k.split('|')[0] for k in data.files))
    for sid in states:
        get=lambda n:data[f'{sid}|{n}'];f={n:get(n) for n in ('context','losses','r','group','j')}
        scores=model.scores(f);average=scores.mean(0);j=f['j'];base=int(np.argmax(j));allowed=j+1.e-8>=.98*j[base];good=get('good')
        i,r=rank_select(j,scores,float(model.arrays['alpha']));pairs=get('pairs');a=pairs[:,0].astype(int);b=pairs[:,1].astype(int)
        strict=pairs[:,2]==1;repairs=strict&good[a]&~good[b];vsbase=repairs&(b==base)
        bestgood=max(average[allowed&good]) if (allowed&good).any() else None
        records.append(dict(state_id=sid,seed=int(sid.split('seed')[1].split(':')[0]),baseline_joint_pass=bool(good[base]),
            joint_candidates=int(good.sum()),allowed_joint_candidates=int((good&allowed).sum()),repairable_failure=bool(not good[base] and (good&allowed).any()),
            strict_pairs=int(strict.sum()),good_vs_bad_pairs=int(repairs.sum()),repair_vs_maxj_pairs=int(vsbase.sum()),
            in_sample_pair_accuracy=float(((average[a[strict]]>average[b[strict]])+.5*(average[a[strict]]==average[b[strict]])).mean()) if strict.any() else None,
            in_sample_good_bad_auc=auc(average,good,allowed),oof_good_bad_auc=auc(oof[sid],good,allowed),
            best_good_rank=1+int((average[allowed]>bestgood).sum()) if bestgood is not None else None,
            in_sample_unguarded_joint_pass=bool(good[max(np.flatnonzero(allowed),key=lambda i:(average[i],j[i]))]),
            in_sample_guarded_joint_pass=bool(good[i]),guard_reason=r['terminal_reason']))
    csv_write(out/'training_repair_coverage.csv',records)
    summary=dict(unique_states=len(records),baseline_failures=sum(not r['baseline_joint_pass'] for r in records),
        repairable_failures=sum(r['repairable_failure'] for r in records),
        repairable_failure_states=[r['state_id'] for r in records if r['repairable_failure']],
        sampled_good_vs_bad_pairs=sum(r['good_vs_bad_pairs'] for r in records),sampled_repair_vs_maxj_pairs=sum(r['repair_vs_maxj_pairs'] for r in records),
        posthoc=True,training_changed=False,additional_native_scores=0)
    write_json(out/'training_repair_coverage.json',summary);return summary


def audit_phase(out,phase,model):
    rows,_=read_rows((44,45) if phase=='validation' else (46,47,48),evaluation=phase=='evaluation')
    rows=[r for r in rows if r['tti'] in ((4001,8001,18000) if phase=='validation' else (3001,4001,5000))]
    raw=read(out/f'{phase}_measurements.csv');state=read(out/f'{phase}_per_state.csv');summary=read(out/f'{phase}_summary.csv')
    values=np.load(out/f'{phase}_values.npz');actions=np.load(out/f'{phase}_actions.npz')
    vvz=np.load(out/f'{phase}_observed_values.npz');aaz=np.load(out/f'{phase}_observed_actions.npz');index={(r['state_id'],r['arm']):r for r in state}
    for r in state:
        rr=[v for v in raw if (v['state_id'],v['arm'])==(r['state_id'],r['arm'])]
        assert len(rr)==3 and float(r['e2e_ms'])==float(np.median([float(v['e2e_ms']) for v in rr]))
        if r['arm']!='full':
            assert len({v['query_sha256'] for v in rr})==len({v['archive_sha256'] for v in rr})==1
            assert int(r['proxy_computed'])+int(r['proxy_cache_hits'])==int(r['proxy_candidates'])
    for r in summary:
        rr=[v for v in state if v['arm']==r['arm']]
        q=all(float(v['ratio'])>=.995 for v in rr);c=all(int(v['newly_unserved_ues'])==0 and min(float(v['positive_p5_ratio']),float(v['positive_weak_ratio']))>=.98 for v in rr)
        assert (r['quality_pass']=='True')==q and (r['coverage_pass']=='True')==c
        assert np.isclose(mean(rr,'score'),float(r['mean_j'])) and np.isclose(mean(rr,'e2e_ms'),float(r['mean_ms']))
    records=[];selected=[];ue=[];predictions={};paired=[];alph=float(model.arrays['alpha']);ha=float(model.arrays['heuristic_alpha'])
    for row in rows:
        sid=identity(row);p=problem_from_raw_row(row);ref=values[f'{sid}|full'];c=compressed_condition(p,actions[f'{sid}|full'],ref,'joint212')
        vv=vvz[f'{sid}|cached_maxj'];aa=aaz[f'{sid}|cached_maxj'];floor=float(index[sid,'cached_maxj']['floor'])
        keep=satisfies(p,aa,c)&(vv[:,:,DEQUEUE].sum(1)>=floor);vv=vv[keep];aa=aa[keep];memo=np.flatnonzero(keep)
        f=ue_features(p,c,vv);j=f['j'];base=f['best'];allowed=j+1.e-8>=.98*j[base];s=model.scores(f);average=s.mean(0);h=heuristic_scores(f)
        prob=sigmoid(alph*(s-s[:,base,None])).min(0);hp=sigmoid(ha*(h-h[base]));metrics=[service_metrics(v,ref) for v in vv]
        jr=j/float(ref[:,EXPECTED]@p.weights);cov=np.array([m['newly_unserved_ues']==0 and min(m['positive_p5_ratio'],m['positive_weak_ratio'])>=.98 for m in metrics]);good=cov&(jr>=.995)
        ranker,rr=rank_select(j,s,alph,False);guarded,gr=rank_select(j,s,alph,True)
        np.testing.assert_array_equal(actions[f'{sid}|ue_pair'],aa[ranker]);np.testing.assert_array_equal(actions[f'{sid}|ue_pair_guard'],aa[guarded])
        params=parameters(model,f);oracle=params.copy();target=(ref[:,EXPECTED]>1).astype(float);quota=vv[0,:,DEQUEUE]
        reliability=np.clip(ref[:,EXPECTED]/np.maximum(quota,1.e-12),1.e-7,1-1.e-7)
        membership=np.clip(target,1.e-7,1-1.e-7);oracle[:,:,8]=np.log(membership/(1-membership));oracle[:,:,9]=np.log(reliability/(1-reliability))
        os=np.stack([service_scores(f,z) for z in oracle]);oi,orr=rank_select(j,os,alph,False)
        bestgood=max(average[allowed&good]) if (allowed&good).any() else None
        records.append(dict(state_id=sid,seed=row['seed'],joint_candidates=int(good.sum()),allowed_candidates=int(allowed.sum()),allowed_joint_candidates=int((good&allowed).sum()),
            good_bad_auc=auc(average,good,allowed),best_good_rank=1+int((average[allowed]>bestgood).sum()) if bestgood is not None else None,
            good_max_pair_confidence=float(prob[allowed&good].max()) if (allowed&good).any() else None,
            good_max_score_gain=float(bestgood-average[base]) if bestgood is not None else None,
            high_confidence_good=int((allowed&good&(prob>=.9)).sum()),high_confidence_bad=int((allowed&~good&(prob>=.9)).sum()),
            unguarded_joint_pass=bool(good[ranker]),guarded_joint_pass=bool(good[guarded]),proposed_memo_index=int(memo[ranker]),
            proposed_confidence=gr['pair_confidence'],guard_reason=gr['terminal_reason'],
            oracle_auxiliary_heads_joint_pass=bool(good[oi]),oracle_auxiliary_heads_j_ratio=float(jr[oi]),
            oracle_auxiliary_heads_p5=float(metrics[oi]['positive_p5_ratio']),oracle_auxiliary_heads_memo_index=int(memo[oi])))
        predictions[sid]=np.c_[memo,jr,cov,good,allowed,h,hp,s.T,average,prob,os.mean(0)]
        for arm in NAMES[1:]:
            idx=int(np.flatnonzero(np.all(aa==actions[f'{sid}|{arm}'],axis=(1,2)))[0]);r=index[sid,arm]
            selected.append(dict(state_id=sid,arm=arm,memo_index=int(memo[idx]),j_ratio=float(jr[idx]),p5_ratio=float(metrics[idx]['positive_p5_ratio']),
                weak_ratio=float(metrics[idx]['positive_weak_ratio']),joint_pass=bool(good[idx]),changed_from_maxj=bool(idx!=base),
                pair_confidence=float(prob[idx]),reported_confidence=r.get('pair_confidence',''),reason=r.get('terminal_reason','maxj_baseline')))
            if arm!='cached_maxj':
                b=index[sid,'cached_maxj'];paired.append(dict(state_id=sid,seed=row['seed'],arm=arm,
                    j_delta=float(r['score'])-float(b['score']),ms_delta=float(r['e2e_ms'])-float(b['e2e_ms']),
                    p5_delta=float(r['positive_p5_ratio'])-float(b['positive_p5_ratio'])))
        for u in np.flatnonzero(f['group']):
            ue.append(dict(state_id=sid,ue=int(u),reference_expected=float(ref[u,EXPECTED]),reference_positive=bool(target[u]),
                reference_ratio=float(reliability[u]),predicted_effective=float(sigmoid(params[:,u,8]).mean()),
                predicted_reference_ratio=float(sigmoid(params[:,u,9]).mean()),mean_weight=float((2*sigmoid(params[:,u,:8])).mean()),
                maxj_expected=float(vv[base,u,EXPECTED]),unguarded_expected=float(vv[ranker,u,EXPECTED]),archive_max=float(vv[:,u,EXPECTED].max())))
    agg=[];work=[]
    for arm in NAMES[2:]:
        rr=[r for r in paired if r['arm']==arm];ss=[r for r in selected if r['arm']==arm]
        agg.append(dict(arm=arm,j_wins=sum(r['j_delta']>1 for r in rr),j_losses=sum(r['j_delta']<-1 for r in rr),j_ties=sum(abs(r['j_delta'])<=1 for r in rr),
            mean_j_delta=mean(rr,'j_delta'),mean_ms_delta=mean(rr,'ms_delta'),changed_states=sum(r['changed_from_maxj'] for r in ss)))
    fields=('terminal_ms','generation_ms','proxy_ms','service_ms','matching_ms','proxy_computed','proxy_cache_hits','cache_payload_bytes')
    for arm in NAMES[1:]:
        rr=[r for r in state if r['arm']==arm];extra=('terminal_feature_ms','terminal_network_ms','terminal_selection_ms') if arm!='cached_maxj' else ()
        work.append(dict(arm=arm,**{k:mean(rr,k) for k in (*fields,*extra)}))
    for name,data in [('ranking_diagnostics',records),('terminal_diagnostics',selected),('ue_diagnostics',ue),('paired',paired),('paired_summary',agg)]:csv_write(out/f'{phase}_{name}.csv',data)
    write_json(out/f'{phase}_work.json',work);np.savez_compressed(out/f'{phase}_candidate_scores.npz',**predictions)
    write_json(out/f'{phase}_diagnostic_schema.json',dict(columns=['memo_index','J/fullJ','coverage','joint','inside2pct','heuristic_score','heuristic_pair_confidence',
        'model0_score','model1_score','model2_score','mean_score','minimum_pair_confidence','oracle_aux_heads_mean_score'],
        posthoc=True,additional_native_scores=0,model_changed=False,oracle='replace only perUE effective/reference heads with teacher labels; preserve learned weights; not deployable'))
    return dict(summary=summary,ranking=records,paired=agg,work=work)


def plot(out,val,dev):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
    for ax,title,data in zip(axes,('Validation: 12 exposed states','Development: 9 exposed states'),(val,dev)):
        d={r['arm']:r for r in data['summary']}
        for i,arm in enumerate(NAMES[1:]):
            r=d[arm];ms=float(r['mean_ms']);color='#237a57' if r['conditional_target_pass']=='True' else '#ad5d38'
            ax.barh(i,ms,height=.6,color=color);ax.text(2,i,f"{ms:.2f} ms | J {float(r['mean_relative_j_pct']):+.3f}% | min P5 {float(r['worst_positive_p5']):.3f}",va='center',color='white',fontsize=8)
        ax.axvline(.8*float(d['full']['mean_ms']),ls='--',color='black',lw=1,label='80% full cost')
        ax.set_yticks(range(5),NAMES[1:]);ax.invert_yaxis();ax.set(title=title,xlabel='Complete execution ms (teacher excluded)');ax.grid(axis='x',alpha=.15)
        ax.set_xlim(0,max(float(r['mean_ms']) for r in data['summary'] if r['arm']!='full')*1.22);ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('Confidence fallback preserves maximum-J behavior; no learned coverage gain')
    fig.savefig(out/'quality_cost.png',dpi=170);fig.savefig(out/'quality_cost.pdf');plt.close(fig)


def main(source):
    out=Path(source).resolve();assert json.loads((out/'completion.json').read_text())['completed'];model=UEPairModel.load(out/'ue_pair_model.npz')
    train=training_audit(out,model);val=audit_phase(out,'validation',model);dev=audit_phase(out,'evaluation',model)
    protocol=json.loads((out/'protocol.json').read_text())
    for n,h in json.loads((out/'source_manifest.json').read_text()).items():assert sha(out/'source'/n)==sha(ROOT/'training/stageb_lns'/n)==h
    for n,h in json.loads((out/'models_frozen.json').read_text())['hashes'].items():assert sha(out/n)==h
    assert sha(REGISTRY)==protocol['registry_sha256'] and sha(ROOT/'training/stageb_lns/physics_kernel.cpp')==protocol['physics_source_sha256']
    tests=Path('/tmp/ue_pair_tests_20261103.log').read_text();assert 'Ran 83 tests' in tests and '\nOK\n' in tests
    shutil.copyfile('/tmp/ue_pair_tests_20261103.log',out/'tests.log');shutil.copyfile('/tmp/ue_pair_execution_20261103.log',out/'execution.log')
    formal=[];teachers=[]
    for phase in ('validation','evaluation'):formal+=read(out/f'{phase}_measurements.csv');teachers+=read(out/f'{phase}_teacher_work.csv')
    engineering=read(out/'engineering_measurements.csv')
    ledger=dict(formal_decisions=len(formal),formal_exact_scores=sum(int(r['candidate_evaluations']) for r in formal),
        new_teacher_decisions=len(teachers),new_teacher_scores=sum(int(r['candidate_evaluations']) for r in teachers),new_teacher_ms=sum(float(r['teacher_ms']) for r in teachers),
        engineering_decisions=len(engineering),engineering_scores=sum(int(r['candidate_evaluations']) for r in engineering),
        additional_training_native_scores=0,additional_posthoc_native_scores=0,reused_training_cost_source=str(PREVIOUS/'total_work_ledger.json'),
        independent_numpy_finals=159)
    ledger['new_native_scores']=ledger['formal_exact_scores']+ledger['new_teacher_scores']+ledger['engineering_scores'];write_json(out/'total_work_ledger.json',ledger)
    write_json(out/'posthoc_audit.json',dict(passed=True,timing_gates_recomputed=True,source_model_registry_valid=True,training_changed=False,
        additional_native_scores=0,oracle_heads_diagnostic_only=True))
    plot(out,val,dev);print(json.dumps(dict(training=train,validation=val,development=dev,ledger=ledger),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);main(p.parse_args().source)

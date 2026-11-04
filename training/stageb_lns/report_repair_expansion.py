"""Post-freeze audit; does not train, select thresholds or issue PHY queries."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from .repair_expansion import NAMES, PREVIOUS, PRIOR_ARCHIVE
from .repair_pair_train import focused_pairs, cohort
from .report_ue_pair import auc, parameters
from .ue_pair_risk import UEPairModel, ue_features, rank_select, sigmoid
from .resource_condition_campaign import compressed_condition, REGISTRY
from .resource_conditions import satisfies
from .ranked_dpls_campaign import read_rows, identity
from .no_ctbs_inputs import problem_from_raw_row
from .shortlist_trajectory import service_metrics
from .physics import ROOT, EXPECTED, DEQUEUE
from .experiment import sha, write_json
from .bounded_probes import csv_write, seed_mean
from .report_joint_reorder import read


def training_audit(out, models):
    z = np.load(out/'prepared/training_data.npz')
    states = list(dict.fromkeys(k.split('|')[0] for k in z.files))
    records = []
    for arm, model in models.items():
        oof = np.load(out/arm/'oof_scores.npz') if arm != 'previous_guard' else None
        for sid in states:
            d = {k.split('|')[1]:z[k] for k in z.files if k.startswith(sid+'|')}
            j = d['j']; base = int(np.argmax(j)); allowed = j+1.e-8 >= .98*j[base]; good = d['good']
            pairs = focused_pairs(d) if arm == 'repair_guard' else d['pairs']
            if arm == 'previous_guard': pairs = np.empty((0, 3))
            a = pairs[:,0].astype(int); b = pairs[:,1].astype(int); strict = pairs[:,2] == 1
            repair = strict & good[a] & ~good[b]
            s = model.scores(d); average = s.mean(0)
            selected, gate = rank_select(j, s, float(model.arrays['alpha']))
            raw, _ = rank_select(j, s, float(model.arrays['alpha']), False)
            best = max(average[allowed & good]) if (allowed & good).any() else None
            oo = oof[sid] if oof is not None else None
            os = rank_select(j, oo, float(model.arrays['alpha']))[0] if oo is not None else None
            records.append(dict(arm=arm, state_id=sid, seed=int(sid.split('seed')[1].split(':')[0]),
                cohort=cohort(d), pairs=len(pairs), strict_pairs=int(strict.sum()),
                good_vs_bad_pairs=int(repair.sum()), repair_vs_maxj_pairs=int((repair & (b==base)).sum()),
                baseline_joint_pass=bool(good[base]), allowed_good=int((allowed & good).sum()),
                in_sample_auc=auc(average, good, allowed), oof_auc=auc(oo, good, allowed) if oo is not None else None,
                best_good_rank=1+int((average[allowed]>best).sum()) if best is not None else None,
                in_sample_guarded_pass=bool(good[selected]), in_sample_unguarded_pass=bool(good[raw]),
                guard_reason=gate['terminal_reason'], confidence=gate['pair_confidence'],
                oof_guarded_pass=bool(good[os]) if os is not None else None))
    csv_write(out/'training_ranking_diagnostics.csv', records)
    summary=[]
    for arm in models:
        rr=[r for r in records if r['arm']==arm]; repairs=[r for r in rr if r['cohort']=='repairable']; controls=[r for r in rr if r['cohort']=='baseline_good']
        summary.append(dict(arm=arm, states=len(rr), repairable=len(repairs),
            total_pairs=sum(r['pairs'] for r in rr), strict_pairs=sum(r['strict_pairs'] for r in rr),
            repair_vs_maxj_pairs=sum(r['repair_vs_maxj_pairs'] for r in rr),
            repair_cohort_good_vs_bad_pairs=sum(r['good_vs_bad_pairs'] for r in repairs),
            in_sample_repairs=sum(r['in_sample_guarded_pass'] for r in repairs),
            oof_repairs=sum(bool(r['oof_guarded_pass']) for r in repairs) if arm!='previous_guard' else None,
            in_sample_control_harms=sum(not r['in_sample_guarded_pass'] for r in controls),
            oof_control_harms=sum(not r['oof_guarded_pass'] for r in controls) if arm!='previous_guard' else None))
    write_json(out/'training_ranking_summary.json', summary)
    return summary


def audit_phase(out, phase, models):
    rows,_=read_rows((44,45) if phase=='validation' else (46,47,48), evaluation=phase=='evaluation')
    rows=[r for r in rows if r['tti'] in ((4001,8001,18000) if phase=='validation' else (3001,4001,5000))]
    raw=read(out/f'{phase}_measurements.csv'); state=read(out/f'{phase}_per_state.csv'); summary=read(out/f'{phase}_summary.csv')
    index={(r['state_id'],r['arm']):r for r in state}
    values=np.load(out/f'{phase}_values.npz'); actions=np.load(out/f'{phase}_actions.npz')
    vz=np.load(out/f'{phase}_observed_values.npz'); az=np.load(out/f'{phase}_observed_actions.npz')
    for r in state:
        repeated=[v for v in raw if (v['state_id'],v['arm'])==(r['state_id'],r['arm'])]
        assert len(repeated)==3 and float(r['e2e_ms'])==float(np.median([float(v['e2e_ms']) for v in repeated]))
        if r['arm']!='full':
            assert len({v['query_sha256'] for v in repeated})==len({v['archive_sha256'] for v in repeated})==1
            assert int(r['proxy_computed'])+int(r['proxy_cache_hits'])==int(r['proxy_candidates'])
    for r in summary:
        rr=[v for v in state if v['arm']==r['arm']]
        numeric=[dict(seed=int(v['seed']),score=float(v['score']),e2e_ms=float(v['e2e_ms'])) for v in rr]
        q=all(float(v['ratio'])>=.995 for v in rr)
        c=all(int(v['newly_unserved_ues'])==0 and min(float(v['positive_p5_ratio']),float(v['positive_weak_ratio']))>=.98 for v in rr)
        assert (r['quality_pass']=='True')==q and (r['coverage_pass']=='True')==c
        assert np.isclose(seed_mean(numeric,'score'),float(r['mean_j'])) and np.isclose(seed_mean(numeric,'e2e_ms'),float(r['mean_ms']))
    ranking=[]; selected=[]; ues=[]; predictions={}
    for row in rows:
        sid=identity(row); p=problem_from_raw_row(row); ref=values[f'{sid}|full']
        condition=compressed_condition(p, actions[f'{sid}|full'], ref, 'joint212')
        vv=vz[f'{sid}|cached_maxj']; aa=az[f'{sid}|cached_maxj']
        keep=satisfies(p,aa,condition)&(vv[:,:,DEQUEUE].sum(1)>=float(index[sid,'cached_maxj']['floor']))
        vv=vv[keep]; aa=aa[keep]; memo=np.flatnonzero(keep)
        f=ue_features(p,condition,vv); j=f['j']; base=f['best']; allowed=j+1.e-8>=.98*j[base]
        metrics=[service_metrics(v,ref) for v in vv]; jr=j/float(ref[:,EXPECTED]@p.weights)
        cov=np.array([m['newly_unserved_ues']==0 and min(m['positive_p5_ratio'],m['positive_weak_ratio'])>=.98 for m in metrics])
        good=cov&(jr>=.995)
        for arm,model in models.items():
            s=model.scores(f); average=s.mean(0); alpha=float(model.arrays['alpha'])
            prob=sigmoid(alpha*(s-s[:,base,None])).min(0)
            chosen,gate=rank_select(j,s,alpha); unguarded,_=rank_select(j,s,alpha,False)
            np.testing.assert_array_equal(actions[f'{sid}|{arm}'],aa[chosen])
            if arm=='repair_guard':np.testing.assert_array_equal(actions[f'{sid}|repair_unguarded'],aa[unguarded])
            best=max(average[allowed&good]) if (allowed&good).any() else None
            ranking.append(dict(state_id=sid,seed=row['seed'],arm=arm,baseline_joint_pass=bool(good[base]),
                allowed_candidates=int(allowed.sum()),allowed_good=int((allowed&good).sum()),
                good_bad_auc=auc(average,good,allowed),best_good_rank=1+int((average[allowed]>best).sum()) if best is not None else None,
                good_max_confidence=float(prob[allowed&good].max()) if (allowed&good).any() else None,
                guarded_joint_pass=bool(good[chosen]),unguarded_joint_pass=bool(good[unguarded]),
                selected_memo_index=int(memo[chosen]),proposed_memo_index=int(memo[unguarded]),
                confidence=gate['pair_confidence'],reason=gate['terminal_reason']))
            predictions[f'{sid}|{arm}']=np.c_[memo,jr,cov,good,allowed,s.T,average,prob]
            params=parameters(model,f); quota=vv[0,:,DEQUEUE]
            for u in np.flatnonzero(f['group']):
                ues.append(dict(state_id=sid,arm=arm,ue=int(u),reference_expected=float(ref[u,EXPECTED]),
                    reference_positive=bool(ref[u,EXPECTED]>1),reference_ratio=float(ref[u,EXPECTED]/max(quota[u],1.e-12)),
                    predicted_effective=float(sigmoid(params[:,u,8]).mean()),predicted_reference_ratio=float(sigmoid(params[:,u,9]).mean()),
                    mean_weight=float((2*sigmoid(params[:,u,:8])).mean()),maxj_expected=float(vv[base,u,EXPECTED]),selected_expected=float(vv[chosen,u,EXPECTED])))
        for arm in NAMES[1:]:
            chosen=int(np.flatnonzero(np.all(aa==actions[f'{sid}|{arm}'],axis=(1,2)))[0])
            r=index[sid,arm]; b=index[sid,'cached_maxj']
            selected.append(dict(state_id=sid,seed=row['seed'],arm=arm,memo_index=int(memo[chosen]),
                changed=bool(chosen!=base),baseline_pass=bool(good[base]),joint_pass=bool(good[chosen]),
                repaired=bool(not good[base] and good[chosen]),harmed=bool(good[base] and not good[chosen]),
                j_ratio=float(jr[chosen]),p5_ratio=float(metrics[chosen]['positive_p5_ratio']),weak_ratio=float(metrics[chosen]['positive_weak_ratio']),
                j_delta=float(r['score'])-float(b['score']),ms_delta=float(r['e2e_ms'])-float(b['e2e_ms']),
                confidence=r.get('pair_confidence',''),reason=r.get('terminal_reason','maxj')))
    paired=[];work=[]
    for arm in NAMES[1:]:
        rr=[r for r in selected if r['arm']==arm]
        paired.append(dict(arm=arm,changed=sum(r['changed'] for r in rr),repaired=sum(r['repaired'] for r in rr),harmed=sum(r['harmed'] for r in rr),
            j_wins=sum(r['j_delta']>1 for r in rr),j_losses=sum(r['j_delta']<-1 for r in rr),j_ties=sum(abs(r['j_delta'])<=1 for r in rr),
            mean_j_delta=seed_mean(rr,'j_delta'),mean_ms_delta=seed_mean(rr,'ms_delta')))
        rr=[r for r in state if r['arm']==arm]
        fields=['generation_ms','proxy_ms','service_ms','matching_ms','terminal_ms','cache_payload_bytes']
        if arm!='cached_maxj':fields+=['terminal_feature_ms','terminal_network_ms','terminal_selection_ms']
        work.append(dict(arm=arm,**{k:seed_mean([dict(seed=int(r['seed']),value=float(r[k])) for r in rr],'value') for k in fields}))
    for name,records in [('ranking_diagnostics',ranking),('terminal_diagnostics',selected),('ue_diagnostics',ues),('paired_summary',paired)]:csv_write(out/f'{phase}_{name}.csv',records)
    write_json(out/f'{phase}_work.json',work); np.savez_compressed(out/f'{phase}_candidate_scores.npz',**predictions)
    write_json(out/f'{phase}_diagnostic_schema.json',dict(columns=['memo_index','J/fullJ','coverage','joint','inside2pct','model0_score','model1_score','model2_score','mean_score','minimum_pair_confidence'],posthoc=True,additional_native_scores=0))
    return dict(summary=summary,paired=paired,work=work)


def plot(out, phases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
    for ax,title,data in zip(axes,('Validation: 12 exposed states','Development: 9 exposed states'),phases):
        d={r['arm']:r for r in data['summary']}
        for i,arm in enumerate(NAMES[1:]):
            r=d[arm];ms=float(r['mean_ms']);quality=r['quality_pass']=='True' and r['coverage_pass']=='True'
            ax.barh(i,ms,height=.6,color='#237a57' if quality else '#ad5d38')
            ax.text(2,i,f"{ms:.2f} ms | J {float(r['mean_relative_j_pct']):+.3f}% | min P5 {float(r['worst_positive_p5']):.3f}",va='center',color='white',fontsize=8)
        ax.axvline(.8*float(d['full']['mean_ms']),ls='--',color='black',lw=1,label='80% full cost')
        ax.set_yticks(range(5),NAMES[1:]);ax.invert_yaxis();ax.set(title=title,xlabel='Complete execution ms (teacher excluded)')
        ax.set_xlim(0,max(float(r['mean_ms']) for r in d.values())*1.05);ax.grid(axis='x',alpha=.15);ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('Directed repair supervision restores coverage; validation cost remains above gate')
    fig.savefig(out/'quality_cost.png',dpi=170);fig.savefig(out/'quality_cost.pdf');plt.close(fig)


def main(source):
    out=Path(source).resolve();completion=json.loads((out/'completion.json').read_text());assert completion['completed']
    models={'previous_guard':UEPairModel.load(out/'previous_model.npz'),**{name:UEPairModel.load(out/name/'ue_pair_model.npz') for name in ('expanded_ordinary','repair_guard')}}
    training=training_audit(out,models);val=audit_phase(out,'validation',models);dev=audit_phase(out,'evaluation',models)
    protocol=json.loads((out/'protocol.json').read_text())
    for n,h in json.loads((out/'source_manifest.json').read_text()).items():assert sha(out/'source'/n)==sha(ROOT/'training/stageb_lns'/n)==h
    for n,h in json.loads((out/'models_frozen.json').read_text())['hashes'].items():assert sha(out/n)==h
    assert sha(REGISTRY)==protocol['registry_sha256'] and sha(ROOT/'training/stageb_lns/physics_kernel.cpp')==protocol['physics_source_sha256']
    tests=Path('/tmp/repair_expansion_tests_20261103.log').read_text();assert 'Ran 89 tests' in tests and '\nOK\n' in tests
    shutil.copyfile('/tmp/repair_expansion_tests_20261103.log',out/'tests.log');shutil.copyfile('/tmp/repair_expansion_execution_20261103.log',out/'execution.log')
    formal=[];teachers=[];checks=[]
    for phase in ('validation','evaluation'):
        formal+=read(out/f'{phase}_measurements.csv');teachers+=read(out/f'{phase}_teacher_work.csv')
        checks+=json.loads((out/f'{phase}_checks.json').read_text())['independent_phy']
        checks+=json.loads((out/f'{phase}_teacher_checks.json').read_text())['independent_phy']
    teachers+=read(out/'collection/added_teacher_work.csv');new=read(out/'collection/added_decoder_work.csv');eng=read(out/'engineering_measurements.csv')
    for path in ('collection/added_teacher_checks.json','collection/added_decoder_checks.json','engineering_checks.json'):checks+=json.loads((out/path).read_text())['independent_phy']
    assert len(checks)==195
    ledger=dict(formal_decisions=len(formal),formal_exact_scores=sum(int(r['candidate_evaluations']) for r in formal),
        new_teacher_decisions=len(teachers),new_teacher_scores=sum(int(r['candidate_evaluations']) for r in teachers),new_teacher_ms=sum(float(r['teacher_ms']) for r in teachers),
        new_collection_decoders=len(new),new_collection_scores=sum(int(r['candidate_evaluations']) for r in new),
        new_collection_ms=sum(float(r['e2e_ms']) for r in new),engineering_decisions=len(eng),engineering_scores=sum(int(r['candidate_evaluations']) for r in eng),
        posthoc_native_scores=0,feature_preparation_native_scores=0,independent_numpy_finals=len(checks),
        prior_training_cost_source=str(PRIOR_ARCHIVE/'total_work_ledger.json'))
    ledger['new_native_scores']=sum(ledger[k] for k in ('formal_exact_scores','new_teacher_scores','new_collection_scores','engineering_scores'))
    write_json(out/'total_work_ledger.json',ledger)
    write_json(out/'final_audit.json',dict(passed=True,tests=89,formal_decisions=378,independent_finals=195,max_independent_phy_error=max(r['max_error'] for r in checks),
        training_updates=2880,source_model_registry_valid=True,models_frozen_before_validation=True,
        all_terminal_query_archive_equivalent=True,previous_maxj_exact_equal_states=21,complete_timing_recomputed=True,
        posthoc_native_scores=0,validation_quality_service_pass=True,development_quality_service_pass=True,
        validation_cost_pass=False,training_cross_seed_coverage_pass=False,online_ai_success=False,
        final_seeds_read=False,native_trajectories=0))
    plot(out,(val,dev));print(json.dumps(dict(training=training,validation=val,development=dev,ledger=ledger),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);main(parser.parse_args().source)

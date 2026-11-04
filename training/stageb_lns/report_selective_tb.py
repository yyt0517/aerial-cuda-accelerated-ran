"""Audit frozen experiment, separate engineering/learning, diagnose guard scope."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .selective_tb import Ranker,ServiceGuard,choose_indices
from .selective_tb_campaign import CONFIGS,NAMES
from .physics import ROOT,ReferencePhy,EXPECTED,COUNT,DEQUEUE
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .ranked_dpls_campaign import read_rows,identity
from .shortlist_trajectory import service_metrics
from .bounded_probes import csv_write
from .experiment import sha,write_json

PAIRS=[('increment_fixed_unprotected','old_joint212'),
    ('increment_selective_unprotected','increment_fixed_unprotected'),
    ('increment_fixed_guard','increment_fixed_unprotected'),
    ('increment_selective_guard','increment_selective_unprotected'),
    ('increment_selective_guard','increment_fixed_guard'),
    ('increment_selective_guard','increment_single_guard'),
    ('trained_guard','increment_selective_guard'),
    ('trained_gain_only_guard','trained_guard'),('trained_guard','legacy_ai212')]


def read(path):return list(csv.DictReader(Path(path).open()))
def avg(rows,key):return float(np.mean([float(r[key]) for r in rows]))
def mean(rows,key):return float(np.mean([avg([r for r in rows if r['seed']==s],key) for s in sorted({r['seed'] for r in rows})]))


def audit_phase(source,phase):
    raw=read(source/f'{phase}_measurements.csv');state=read(source/f'{phase}_per_state.csv')
    summary=read(source/f'{phase}_summary.csv');index={(r['state_id'],r['arm']):r for r in state}
    for r in state:
        rr=[v for v in raw if v['state_id']==r['state_id'] and v['arm']==r['arm']]
        assert len(rr)==3 and np.isclose(float(r['e2e_ms']),np.median([float(v['e2e_ms']) for v in rr]))
        if r['arm']!='full':assert int(r['candidate_evaluations'])<=212 and int(r['feature_roots'])<=56
        if r['arm'] in CONFIGS and CONFIGS[r['arm']].protect:assert r['anchor_guard_pass']=='True'
    for r in summary:
        rr=[v for v in state if v['arm']==r['arm']]
        assert np.isclose(mean(rr,'score'),float(r['mean_j'])) and np.isclose(mean(rr,'e2e_ms'),float(r['mean_ms']))
        q=all(float(v['ratio'])>=.995 for v in rr)
        c=all(int(v['newly_unserved_ues'])==0 and float(v['positive_p5_ratio'])>=.98 and float(v['positive_weak_ratio'])>=.98 for v in rr)
        cost=float(r['time_ratio'])<=.8
        assert (r['quality_pass']=='True')==q and (r['coverage_pass']=='True')==c
        assert (r['joint_pass']=='True')==(q and c and cost)
    paired=[]
    for sid in dict.fromkeys(r['state_id'] for r in state):
        for new,old in PAIRS:
            a=index[sid,new];b=index[sid,old]
            paired.append(dict(state_id=sid,seed=a['seed'],new=new,old=old,
                j_delta=float(a['score'])-float(b['score']),j_delta_pct=100*(float(a['score'])/float(b['score'])-1),
                delta_pp=100*(float(a['ratio'])-float(b['ratio'])),ms_delta=float(a['e2e_ms'])-float(b['e2e_ms']),
                newly_unserved_delta=int(a['newly_unserved_ues'])-int(b['newly_unserved_ues']),
                positive_p5_delta=float(a['positive_p5_ratio'])-float(b['positive_p5_ratio'])))
    csv_write(source/f'{phase}_paired.csv',paired);paired_summary=[];seed_rows=[]
    for new,old in PAIRS:
        rr=[r for r in paired if r['new']==new and r['old']==old]
        paired_summary.append(dict(new=new,old=old,wins=sum(r['j_delta']>1 for r in rr),losses=sum(r['j_delta']<-1 for r in rr),
            ties=sum(abs(r['j_delta'])<=1 for r in rr),mean_j_delta=mean(rr,'j_delta'),mean_j_delta_pct=mean(rr,'j_delta_pct'),
            mean_delta_pp=mean(rr,'delta_pp'),mean_ms_delta=mean(rr,'ms_delta'),
            mean_newly_unserved_delta=mean(rr,'newly_unserved_delta'),mean_positive_p5_delta=mean(rr,'positive_p5_delta')))
        for seed in sorted({r['seed'] for r in rr}):
            vv=[r for r in rr if r['seed']==seed]
            seed_rows.append(dict(new=new,old=old,seed=seed,j_delta=avg(vv,'j_delta'),j_delta_pct=avg(vv,'j_delta_pct')))
    csv_write(source/f'{phase}_paired_summary.csv',paired_summary);csv_write(source/f'{phase}_paired_per_seed.csv',seed_rows)
    work=[]
    for arm in CONFIGS:
        rr=[r for r in state if r['arm']==arm]
        work.append(dict(arm=arm,**{key:mean(rr,key) for key in ('candidate_evaluations','feature_roots','proxy_candidates',
            'feature_ms','candidate_generation_ms','network_ms','physical_ms','memo_hits','selected_joint','joint_accepted','guard_rejects')},
            stop_counts={s:sum(r['stop_reason']==s for r in rr) for s in sorted({r['stop_reason'] for r in rr})}))
    write_json(source/f'{phase}_work.json',work)
    with np.load(source/f'{phase}_actions.npz') as actions:
        for sid in dict.fromkeys(r['state_id'] for r in state):
            np.testing.assert_array_equal(actions[f'{sid}|old_joint212'],actions[f'{sid}|increment_fixed_unprotected'])
    return dict(summary=summary,state=state,paired=paired_summary,seeds=seed_rows,raw_decisions=len(raw))


def anchor_diagnosis(source):
    records=[];start=time.perf_counter()
    for phase,seeds in [('validation',(44,45)),('evaluation',(46,47,48))]:
        rows,_=read_rows(seeds,evaluation=phase=='evaluation');ttis=(4001,8001,18000) if phase=='validation' else (3001,4001,5000)
        values=np.load(source/f'{phase}_values.npz')
        for row in rows:
            if row['tti'] not in ttis:continue
            p=problem_from_raw_row(row);sid=identity(row);anchor=ReferencePhy(p).evaluate(raw_pfq_draft(p,row['obs']))[0]
            full=values[f'{sid}|full'];trained=values[f'{sid}|trained_guard'];drop=(anchor[:,COUNT]>0)&(full[:,COUNT]==0)
            m=service_metrics(full,anchor)
            records.append(dict(phase=phase,state_id=sid,seed=row['seed'],full_passes_pfq_anchor=bool(ServiceGuard.against(full,anchor)[0]),
                pfq_allocated=int((anchor[:,COUNT]>0).sum()),full_allocated=int((full[:,COUNT]>0).sum()),
                full_drops_pfq_allocated=int(drop.sum()),full_drops_pfq_positive=int((drop&(anchor[:,EXPECTED]>1)).sum()),
                full_drops_pfq_nearzero=int((drop&(anchor[:,EXPECTED]<=1)).sum()),
                full_p5_vs_pfq=m['positive_p5_ratio'],full_weak_vs_pfq=m['positive_weak_ratio'],
                pfq_j=float(anchor[:,EXPECTED]@p.weights),full_j=float(full[:,EXPECTED]@p.weights),
                trained_j=float(trained[:,EXPECTED]@p.weights)))
    csv_write(source/'posthoc_anchor_scope.csv',records)
    write_json(source/'posthoc_anchor_scope.json',dict(posthoc=True,policy_selection_changed=False,
        purpose='diagnose additional guard scope, not alter acceptance gates or prove infeasibility',
        full_anchor_passes=sum(r['full_passes_pfq_anchor'] for r in records),states=len(records),
        numpy_pfq_scores=len(records),elapsed_s=time.perf_counter()-start,
        finding='all allocated PFQ UEs are a stronger protection set than full-DPLS allocation/service reference',
        limitation='full reference violating a constraint is not proof no other near-full-J schedule satisfies it'))
    return records


def audit_training(source,model):
    import torch
    groups=torch.load(source/'training_groups.pt',weights_only=False);data=json.loads((source/'training_data.json').read_text())
    all_x=np.concatenate([g['x'][g['available']] for g in groups if g['available'].any()])
    np.testing.assert_array_equal(model.mean,all_x.mean(0).astype(np.float32))
    np.testing.assert_array_equal(model.scale,np.maximum(all_x.std(0),.05).astype(np.float32))
    rows,_=read_rows((41,42,43));by_id={identity(r):r for r in rows};checks=[]
    for seed in (41,42,43):
        g=next(v for v in groups if v['seed']==seed);p=problem_from_raw_row(by_id[g['state_id']]);ref=ReferencePhy(p)
        base=ref.evaluate(g['root'])[0];anchor=ref.evaluate(raw_pfq_draft(p,by_id[g['state_id']]['obs']))[0]
        exact=ref.evaluate(g['actions']);np.testing.assert_allclose(g['anchor'],anchor,rtol=2.e-9,atol=2.e-6)
        gain=np.array([float(v[:,EXPECTED]@p.weights) for v in exact])-float(base[:,EXPECTED]@p.weights)
        np.testing.assert_allclose(g['gain'],gain,rtol=2.e-9,atol=2.e-6)
        safe=ServiceGuard(anchor).safe(exact,base)&(exact[:,:,DEQUEUE].sum(1)>=g['floor'])
        np.testing.assert_array_equal(g['safe'],safe)
        available=(exact[:,:,DEQUEUE].sum(1)>=g['floor'])&~((base[None,:,COUNT]>0)&(exact[:,:,COUNT]==0)).any(1)
        available &= np.array([a.tobytes() not in g['failed'] for a in g['actions']],bool)
        np.testing.assert_array_equal(g['available'],available)
        checks.append(dict(state_id=g['state_id'],seed=seed,candidate_checks=len(g['gain']),max_gain_error=float(abs(g['gain']-gain).max())))
    fit=[]
    for g in groups:
        assert int(g['seed']) in (41,42,43)
        available=g['available'];eligible=g['safe']&available
        best=max(0.,float(g['gain'][eligible].max())) if eligible.any() else 0.
        for name,scores in [('heuristic',g['prior']),('trained',model.predict(g['x'],g['prior']))]:
            selected=choose_indices(scores,g['joint'],available,'selective',4)
            safe_selected=selected[g['safe'][selected]]
            found=max(0.,float(g['gain'][safe_selected].max())) if len(safe_selected) else 0.
            fit.append(dict(state_id=g['state_id'],seed=g['seed'],ordinal=g['ordinal'],model=name,best_safe_gain=best,
                achieved_safe_gain=found,regret=best-found,near_optimal=found>=best-16,positive_found=found>1))
    csv_write(source/'training_fit_diagnostics.csv',fit)
    write_json(source/'training_label_audit.json',dict(passed=True,normalization_training_only=True,
        sampled_group_per_seed=checks,training_states_only=True,actual_visit_keys_preserved=True,
        fit_scope='72 correlated bootstrap visit groups; held-in fit diagnostic, not generalization or final policy',
        total_independent_candidate_checks=sum(r['candidate_checks'] for r in checks)))
    return checks


def plot(val,dev,source):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#263442','#963c4b','#8d8d8d','#337eb1','#2a9f8a','#a86d26','#ce9742','#6b5da8','#c7464e','#d37991']
    labels=['Full DPLS','Legacy AI','Old wide joint','Incremental fixed','Incremental selective','Guarded singles',
        'Guarded fixed','Guarded selective','Trained guarded','Gain-head-only']
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for col,(data,title) in enumerate([(val,'Validation: 12 historical states'),(dev,'Development: 9 historical states')]):
        for i,r in enumerate(data['summary']):
            for row,field in [(0,'mean_relative_j_pct'),(1,'mean_positive_p5')]:
                x=float(r['time_ratio']);y=float(r[field]);axes[row,col].scatter(x,y,s=55,color=colors[i],marker='*' if i==0 else 'o',label=labels[i] if row==0 else None)
                axes[row,col].annotate(str(i),(x,y),xytext=(4,5 if i%2==0 else -10),textcoords='offset points',fontsize=8)
        for row in (0,1):
            axes[row,col].axvline(.8,color='#963c4b',linestyle=':',alpha=.7);axes[row,col].grid(alpha=.2)
            axes[row,col].set_xlabel('Complete mean time / full DPLS')
        axes[0,col].axhline(-.5,color='#963c4b',linestyle=':',alpha=.7);axes[1,col].axhline(.98,color='#963c4b',linestyle=':',alpha=.7)
        axes[0,col].set_title(title);axes[0,col].set_ylabel('Mean J change vs full (%)');axes[1,col].set_ylabel('Mean effective-UE P5 service ratio')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,[f'{i}: {s}' for i,s in enumerate(labels)],loc='lower center',ncol=5,frameon=False,fontsize=8,bbox_to_anchor=(.5,.025))
    fig.suptitle('Incremental TB features and protected selective ranking: score cap 212',fontsize=13)
    fig.text(.5,.012,'Engineering equivalence and learned contribution are separate. Gates apply per state; plotted means do not establish success.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.10,1,.95));fig.savefig(source/'quality_cost.png',dpi=160);fig.savefig(source/'quality_cost.pdf');plt.close(fig)


def main(source):
    source=Path(source).resolve();complete=json.loads((source/'completion.json').read_text());assert complete['completed']
    protocol=json.loads((source/'protocol.json').read_text());selection=json.loads((source/'selection.json').read_text())
    model_info=json.loads((source/'model_selection.json').read_text());model=Ranker(model_info['checkpoint'])
    val=audit_phase(source,'validation');dev=audit_phase(source,'evaluation');scope=anchor_diagnosis(source);training_checks=audit_training(source,model)
    baseline=Ranker(source/'selective_e00.npz');changed=any(not np.array_equal(getattr(model,k),getattr(baseline,k)) for k in ('w1','b1','w2','b2','w3','b3'))
    learning=next(r for r in dev['paired'] if r['new']=='trained_guard' and r['old']=='increment_selective_guard')
    seed_learning=[r for r in dev['seeds'] if r['new']=='trained_guard' and r['old']=='increment_selective_guard']
    trained=next(r for r in dev['summary'] if r['arm']=='trained_guard')
    directional=changed and learning['wins']>=6 and all(r['j_delta']>0 for r in seed_learning)
    engineering=read(source/'engineering_features.csv')
    by_arm={r['arm']:r for r in dev['summary']}
    new_time=float(by_arm['increment_fixed_unprotected']['mean_ms']);old_time=float(by_arm['old_joint212']['mean_ms'])
    gates=dict(selected_on_validation=selection,selected_model=model_info,
        quality_coverage_cost={r['arm']:{k:r[k]=='True' for k in ('quality_pass','coverage_pass','cost_pass','joint_pass')} for r in dev['summary'] if r['arm']!='full'},
        engineering=dict(all21_final_actions_identical=True,incremental_proxy_equivalent=True,
            microbenchmark_mean_incremental_ms=avg(engineering,'incremental_ms'),microbenchmark_mean_wide_ms=avg(engineering,'wide_ms'),
            complete_development_old_ms=old_time,complete_development_new_ms=new_time,complete_time_reduction_pct=100*(1-new_time/old_time),ai_credit=False),
        learning=dict(selected_weights_changed=changed,development_pair=learning,per_seed=seed_learning,
            directional_gate=directional,joint_learning_gate=directional and trained['joint_pass']=='True'),
        guard_scope=dict(full_reference_anchor_passes=sum(r['full_passes_pfq_anchor'] for r in scope),states=len(scope),
            diagnosis='locking every allocated PFQ UE imposes a stronger objective; no infeasibility theorem'),
        next_path=dict(selected='align causal service-set protection with terminal quality/coverage target before further distillation',
            implementation_status='chosen next bounded probe, not implemented',
            retain=['incremental descriptor engine','raw PFQ90% floor','single/joint candidate pools','complete charging','unchanged final quality/service/cost gates'],
            next_probe='training-only full-DPLS terminal allocation/effective-service labels for a causal UE service-set proposer; '
                'compare fixed-allocation lock with selective service-set protection and bounded terminal repair; first measure set recall and conditional quality-cost headroom',
            no_free_teacher='full terminal sets are offline labels only, never online masks or development oracle actions',
            defer=['more training on current locked-set student','PPO','finite continuation value','default registry replacement']))
    write_json(source/'gate_results.json',gates)
    for name,digest in json.loads((source/'source_manifest.json').read_text()).items():
        assert sha(source/'source'/name)==digest and sha(ROOT/'training/stageb_lns'/name)==digest
    assert sha(protocol['old_model'])==protocol['old_model_sha256']
    assert sha(ROOT/'training/stageb_hgraph/current_algorithm.json')==protocol['registry_sha256']
    assert sha(model.path)==model_info['sha256']
    tests=Path('/tmp/selective_tb_tests_20261102.log').read_text();assert 'Ran 63 tests' in tests and '\nOK\n' in tests
    shutil.copyfile('/tmp/selective_tb_tests_20261102.log',source/'tests.log');shutil.copyfile('/tmp/selective_tb_execution_20261102.log',source/'execution.log')
    write_json(source/'final_audit.json',dict(passed=True,tests=63,formal_decisions=val['raw_decisions']+dev['raw_decisions'],
        independent_final_phy=complete['formal_independent_final_audits'],
        training_independent_candidate_checks=sum(r['candidate_checks'] for r in training_checks),
        whole_tb_engineering_batches=16,summary_and_paired_metrics_recomputed=True,
        all21_final_engineering_actions_identical=True,source_model_and_registry_hashes_valid=True,
        native_trajectories=0,final_seeds_read=False))
    plot(val,dev,source)
    print(json.dumps(dict(gates=gates,validation=val['summary'],development=dev['summary'],pairs=dev['paired']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);main(parser.parse_args().source)

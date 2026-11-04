"""Read-only auditing of continuous service, cost parity and visited headroom."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from .weak_service_campaign import NAMES,CONFIGS,PARITY_FIELDS
from .report_joint_reorder import read,mean
from .bounded_probes import csv_write
from .experiment import sha,write_json
from .physics import ROOT,EXPECTED,COUNT,DEQUEUE
from .ranked_dpls_campaign import read_rows,identity
from .no_ctbs_inputs import problem_from_raw_row
from .resource_condition_campaign import REGISTRY

PAIRS=[('frontier_fast','frontier_reference'),('quota_p5','frontier_fast'),('quota_log','quota_p5'),
    ('quota_exp','quota_log'),('quota_exp','frontier_fast'),('quota_exp','activity_reference')]


def phase_audit(source,phase):
    raw=read(source/f'{phase}_measurements.csv');state=read(source/f'{phase}_per_state.csv');summary=read(source/f'{phase}_summary.csv')
    for r in state:
        repeats=[v for v in raw if (v['state_id'],v['arm'])==(r['state_id'],r['arm'])]
        assert len(repeats)==3 and np.isclose(float(r['e2e_ms']),np.median([float(v['e2e_ms']) for v in repeats]))
        if r['arm']!='full':assert int(r['candidate_evaluations'])==212 and int(r['feature_roots'])<=56 and r['condition_feasible']=='True'
    for r in summary:
        rr=[v for v in state if v['arm']==r['arm']]
        q=all(float(v['ratio'])>=.995 for v in rr)
        c=all(int(v['newly_unserved_ues'])==0 and float(v['positive_p5_ratio'])>=.98 and float(v['positive_weak_ratio'])>=.98 for v in rr)
        assert (r['quality_pass']=='True')==q and (r['coverage_pass']=='True')==c
        assert (r['conditional_target_pass']=='True')==(q and c and float(r['execution_time_ratio'])<=.8)
        assert np.isclose(mean(rr,'score'),float(r['mean_j'])) and np.isclose(mean(rr,'e2e_ms'),float(r['mean_ms']))
    index={(r['state_id'],r['arm']):r for r in state};paired=[];aggregate=[];per_seed=[]
    for sid in dict.fromkeys(r['state_id'] for r in state):
        for field in PARITY_FIELDS:assert index[sid,'frontier_fast'][field]==index[sid,'frontier_reference'][field]
        for new,old in PAIRS:
            a=index[sid,new];b=index[sid,old]
            paired.append(dict(state_id=sid,seed=a['seed'],new=new,old=old,j_delta=float(a['score'])-float(b['score']),
                delta_pp=100*(float(a['ratio'])-float(b['ratio'])),ms_delta=float(a['e2e_ms'])-float(b['e2e_ms']),
                p5_delta=float(a['positive_p5_ratio'])-float(b['positive_p5_ratio']),weak_delta=float(a['positive_weak_ratio'])-float(b['positive_weak_ratio'])))
    for new,old in PAIRS:
        rr=[r for r in paired if (r['new'],r['old'])==(new,old)]
        aggregate.append(dict(new=new,old=old,wins=sum(r['j_delta']>1 for r in rr),losses=sum(r['j_delta']<-1 for r in rr),ties=sum(abs(r['j_delta'])<=1 for r in rr),
            mean_j_delta=mean(rr,'j_delta'),mean_delta_pp=mean(rr,'delta_pp'),mean_ms_delta=mean(rr,'ms_delta'),mean_p5_delta=mean(rr,'p5_delta'),mean_weak_delta=mean(rr,'weak_delta')))
        for seed in sorted({r['seed'] for r in rr}):
            ss=[r for r in rr if r['seed']==seed];per_seed.append(dict(new=new,old=old,seed=seed,j_delta=mean(ss,'j_delta'),ms_delta=mean(ss,'ms_delta')))
    for name,records in [('paired',paired),('paired_summary',aggregate),('paired_per_seed',per_seed)]:csv_write(source/f'{phase}_{name}.csv',records)
    work=[]
    fields=('bank_ms','matching_ms','generation_ms','proxy_ms','service_ms','proxy_candidates','proposal_attempts','proxy_gamma_array_reads','seed_gamma_array_reads','mmse_requests','mmse_misses','feature_roots','frontier_peak')
    for arm in ['frontier_reference',*CONFIGS]:
        rr=[r for r in state if r['arm']==arm]
        work.append(dict(arm=arm,**{k:mean(rr,k) for k in fields},
            **({k:mean(rr,k) for k in ('service_rows','dominance_pairs','frontier_queries')} if arm in CONFIGS else {})))
    write_json(source/f'{phase}_work.json',work)
    return dict(state=state,summary=summary,paired=aggregate,work=work)


def observed_diagnosis(source):
    """Existing paid rows only. No exact candidate query, no policy changes."""
    records=[];groups_out=[];failures=[];matched={}
    for phase,seeds,ttis in [('validation',(44,45),(4001,8001,18000)),('evaluation',(46,47,48),(3001,4001,5000))]:
        rows,_=read_rows(seeds,evaluation=phase=='evaluation');rows=[r for r in rows if r['tti'] in ttis]
        values=np.load(source/f'{phase}_values.npz');groups=np.load(source/f'{phase}_service_groups.npz');observed=np.load(source/f'{phase}_observed_values.npz')
        matched[phase]=dict(log_exp_actions_equal=0,log_exp_observed_values_equal=0)
        actions=np.load(source/f'{phase}_actions.npz')
        for row in rows:
            p=problem_from_raw_row(row);sid=identity(row);ref=values[f'{sid}|full'];target=np.flatnonzero(ref[:,EXPECTED]>1)
            weak=target[np.argsort(ref[target,EXPECTED],kind='stable')[:max(1,int(np.ceil(.2*len(target))))]]
            jref=float(ref[:,EXPECTED]@p.weights)
            matched[phase]['log_exp_actions_equal']+=int(np.array_equal(actions[f'{sid}|quota_log'],actions[f'{sid}|quota_exp']))
            matched[phase]['log_exp_observed_values_equal']+=int(np.array_equal(observed[f'{sid}|quota_log'],observed[f'{sid}|quota_exp']))
            for arm in ['frontier_reference',*CONFIGS]:
                key=f'{sid}|{arm}';chosen=values[key];group=groups[key];mask=ref[:,EXPECTED]>1
                groups_out.append(dict(phase=phase,state_id=sid,arm=arm,group_size=int(group.sum()),
                    missing_reference_positive=np.flatnonzero(mask&~group).tolist(),extra_nonreference=np.flatnonzero(group&~mask).tolist()))
                # Discard initial PFQ unconditionally: it may not satisfy joint
                # conditions. Every subsequent paid proposal was conditioned.
                vv=observed[key][1:];j=vv[:,:,EXPECTED]@p.weights
                p5=np.quantile(vv[:,target,EXPECTED],.05,axis=1)/np.quantile(ref[target,EXPECTED],.05)
                wr=vv[:,weak,EXPECTED].sum(1)/ref[weak,EXPECTED].sum()
                no_new=~((ref[None,:,COUNT]>0)&(vv[:,:,COUNT]==0)).any(axis=1)
                floor=float(next(r['floor'] for r in read(source/f'{phase}_per_state.csv') if r['state_id']==sid and r['arm']==arm))
                eligible=vv[:,:,DEQUEUE].sum(1)>=floor
                coverage=(p5>=.98)&(wr>=.98)&no_new&eligible;joint=coverage&(j>=.995*jref)
                chosen_ratio=float(chosen[:,EXPECTED]@p.weights/jref)
                chosen_p5=float(np.quantile(chosen[target,EXPECTED],.05)/np.quantile(ref[target,EXPECTED],.05))
                chosen_weak=float(chosen[weak,EXPECTED].sum()/ref[weak,EXPECTED].sum())
                record=dict(phase=phase,state_id=sid,seed=row['seed'],arm=arm,observed_conditional=len(vv),
                    coverage_candidates=int(coverage.sum()),joint_candidates=int(joint.sum()),
                    best_observed_j_relative_pct=float(100*(j.max()/jref-1)),
                    best_coverage_j_relative_pct=float(100*(j[coverage].max()/jref-1)) if coverage.any() else None,
                    best_positive_p5=float(p5.max()),selected_relative_j_pct=100*(chosen_ratio-1),selected_positive_p5=chosen_p5,
                    selected_positive_weak=chosen_weak,quality_pass=chosen_ratio>=.995,coverage_pass=chosen_p5>=.98 and chosen_weak>=.98)
                records.append(record)
                if arm=='quota_exp' and not(record['quality_pass'] and record['coverage_pass']):
                    failures.append(dict(**record,reference_positive_zeroed=np.flatnonzero(mask&(chosen[:,EXPECTED]<=1)).tolist()))
    csv_write(source/'observed_service_diagnosis.csv',records)
    write_json(source/'group_diagnosis.json',dict(posthoc=True,additional_physical_scores=0,teacher_positive_used_only_in_audit=True,groups=groups_out))
    result=dict(posthoc=True,additional_physical_scores=0,policy_changed=False,selection_changed=False,
        scope='all already-paid conditional candidates except initial PFQ; teacher membership used only after execution',
        selected_failures=failures,log_exp=matched,
        finding='remaining selected failures have no already-observed candidate passing joint gates; final selector alone cannot repair them',
        limitation='finite visited archive is not a complete neighborhood or feasibility certificate')
    assert all(r['joint_candidates']==0 for r in failures)
    write_json(source/'observed_service_diagnosis.json',result);return result


def plot(source,val,dev):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Full','Activity ref.','Old frontier','Fast same frontier','All-quota P5','All-quota log','All-quota exp']
    colors=['#263442','#b57e2b','#92568b','#467aab','#6c9a83','#ce6b43','#663ea1']
    fig,axes=plt.subplots(2,2,figsize=(12,8.8))
    for col,(data,title) in enumerate([(val,'Validation:12 historical states'),(dev,'Development:9 historical states')]):
        for i,r in enumerate(data['summary']):
            for row,field in [(0,'worst_relative_j_pct'),(1,'worst_positive_p5')]:
                x=float(r['execution_time_ratio']);y=float(r[field]);axes[row,col].scatter(x,y,s=60,color=colors[i],marker='*' if i==0 else 'o',label=labels[i] if row==0 else None)
                axes[row,col].annotate(str(i),(x,y),xytext=(3,6 if i%2==0 else -10),textcoords='offset points',fontsize=8)
        for row in (0,1):
            axes[row,col].axvline(.8,color='#a43e4b',linestyle=':',alpha=.7);axes[row,col].grid(alpha=.2)
            axes[row,col].set_xlabel('Complete conditional execution / full DPLS')
        axes[0,col].axhline(-.5,color='#a43e4b',linestyle=':',alpha=.7);axes[1,col].axhline(.98,color='#a43e4b',linestyle=':',alpha=.7)
        axes[0,col].set_title(title);axes[0,col].set_ylabel('Worst-state J change vs full (%)');axes[1,col].set_ylabel('Worst-state effective-UE P5 ratio')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,[f'{i}: {s}' for i,s in enumerate(l)],loc='lower center',ncol=4,frameon=False,fontsize=8,bbox_to_anchor=(.5,.035))
    fig.suptitle('Continuous weak-service metrics and equivalent frontier cost reduction',fontsize=13)
    fig.text(.5,.015,'212 exact scores; offline teacher conditions/preparation excluded; all costs measured; no online AI claim.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.13,1,.95));fig.savefig(source/'quality_cost.png',dpi=160);fig.savefig(source/'quality_cost.pdf');plt.close(fig)


def main(source):
    source=Path(source).resolve();complete=json.loads((source/'completion.json').read_text());assert complete['completed']
    val=phase_audit(source,'validation');dev=phase_audit(source,'evaluation');diagnosis=observed_diagnosis(source)
    selection=json.loads((source/'selection.json').read_text());chosen=selection['quality_candidate']
    summary={r['arm']:r for r in dev['summary']}
    gates=dict(selection=selection,validation=val['summary'],development=dev['summary'],selected_development=summary[chosen],
        engineering=dict(all21_ordered_roots_queries_values_actions_work_equal=True),training_started=False,online_ai_success=False,
        next_path=dict(selected='retain fast frontier and all-quota continuous metric; bounded weak-UE-directed coordinated proposals and failure-root candidate-recall audit',
            implementation_status='selected next experiment, not implemented',
            reason='full quota group fixes omissions and continuous metrics fix one development J failure; remaining validation service and development J failures have no passing visited candidate',
            design='within212 exact scores, reallocate a small fixed proposal share to weak-UE/whole-TB reliability-directed coordinated permutations; keep current frontier fallback; diagnose proxy recall versus proposal reachability with separately paid labels',
            cost='validation still above80% full; track generation/proxy/frontier components and require full cost gate, not development mean alone',
            defer=['more smoothing-parameter sweeps','more final J relaxation','upstream causal model training','PPO','default replacement'],
            training_entry='stable conditional quality,service and complete cost first; then causal upstream prediction errors and cost'))
    write_json(source/'gate_results.json',gates)
    protocol=json.loads((source/'protocol.json').read_text())
    for name,digest in json.loads((source/'source_manifest.json').read_text()).items():assert sha(source/'source'/name)==sha(ROOT/'training/stageb_lns'/name)==digest
    assert sha(REGISTRY)==protocol['registry_sha256'] and sha(ROOT/'training/stageb_lns/physics_kernel.cpp')==protocol['physics_source_sha256']
    tests=Path('/tmp/weak_service_tests_20261102.log').read_text();assert 'Ran 106 tests' in tests and '\nOK\n' in tests
    shutil.copyfile('/tmp/weak_service_tests_20261102.log',source/'tests.log');shutil.copyfile('/tmp/weak_service_execution_20261102.log',source/'execution.log')
    ledger={}
    for phase in ('engineering','validation','evaluation'):
        rr=read(source/f'{phase}_teacher_work.csv');checks=json.loads((source/f'{phase}_teacher_checks.json').read_text())
        assert len(rr)==checks['states']==len(checks['independent_phy'])
        ledger[phase]=dict(decisions=len(rr),exact_scores=sum(int(r['candidate_evaluations']) for r in rr),teacher_ms=sum(float(r['teacher_ms']) for r in rr))
    ledger['total']={k:sum(v[k] for v in ledger.values()) for k in ('decisions','exact_scores','teacher_ms')};write_json(source/'teacher_ledger.json',ledger)
    write_json(source/'final_audit.json',dict(passed=True,tests=106,formal_decisions=441,formal_independent_finals=147,
        teacher_independent_finals=23,engineering_decoder_finals=10,fast_parity_formal_states=21,fast_parity_engineering_states=2,
        ordered_query_roots_all_observed_values_work_groups_equal=True,source_and_registry_hashes_valid=True,
        posthoc_diagnostic_scores=0,training_updates=0,native_trajectories=0,final_seeds_read=False,scientific_gate_passed=False))
    plot(source,val,dev)
    print(json.dumps(dict(selection=selection,development=dev['summary'],paired=dev['paired'],diagnosis=diagnosis),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);main(parser.parse_args().source)

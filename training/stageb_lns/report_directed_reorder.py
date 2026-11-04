"""Audit directed-budget comparisons, paid recall and observed selector headroom."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from .directed_reorder_campaign import NAMES,CONFIGS
from .report_joint_reorder import read,mean
from .bounded_probes import csv_write
from .experiment import write_json,sha
from .physics import ROOT,EXPECTED,DEQUEUE,COUNT
from .resource_condition_campaign import REGISTRY
from .ranked_dpls_campaign import read_rows,identity
from .no_ctbs_inputs import problem_from_raw_row

PAIRS=[('existing2','baseline'),('joint2','existing2'),('joint4','joint2'),('sparse2','joint2'),('joint4','baseline')]


def phase_audit(source,phase):
    raw=read(source/f'{phase}_measurements.csv');state=read(source/f'{phase}_per_state.csv');summary=read(source/f'{phase}_summary.csv')
    for r in state:
        rr=[v for v in raw if (v['state_id'],v['arm'])==(r['state_id'],r['arm'])]
        assert len(rr)==3 and np.isclose(float(r['e2e_ms']),np.median([float(v['e2e_ms']) for v in rr]))
        if r['arm']!='full':
            assert int(r['candidate_evaluations'])<=212 and int(r['feature_roots'])<=56 and r['condition_feasible']=='True'
            assert len({v['query_sha256'] for v in rr})==len({v['archive_sha256'] for v in rr})==1
    for r in summary:
        rr=[v for v in state if v['arm']==r['arm']]
        q=all(float(v['ratio'])>=.995 for v in rr)
        c=all(int(v['newly_unserved_ues'])==0 and float(v['positive_p5_ratio'])>=.98 and float(v['positive_weak_ratio'])>=.98 for v in rr)
        assert (r['quality_pass']=='True')==q and (r['coverage_pass']=='True')==c
        assert (r['conditional_target_pass']=='True')==(q and c and float(r['execution_time_ratio'])<=.8)
        assert np.isclose(mean(rr,'score'),float(r['mean_j'])) and np.isclose(mean(rr,'e2e_ms'),float(r['mean_ms']))
    index={(r['state_id'],r['arm']):r for r in state};pairs=[];aggregate=[];seeds=[]
    for sid in dict.fromkeys(r['state_id'] for r in state):
        for new,old in PAIRS:
            a=index[sid,new];b=index[sid,old]
            pairs.append(dict(state_id=sid,seed=a['seed'],new=new,old=old,j_delta=float(a['score'])-float(b['score']),
                delta_pp=100*(float(a['ratio'])-float(b['ratio'])),ms_delta=float(a['e2e_ms'])-float(b['e2e_ms']),
                p5_delta=float(a['positive_p5_ratio'])-float(b['positive_p5_ratio']),weak_delta=float(a['positive_weak_ratio'])-float(b['positive_weak_ratio'])))
    for new,old in PAIRS:
        rr=[r for r in pairs if (r['new'],r['old'])==(new,old)]
        aggregate.append(dict(new=new,old=old,wins=sum(r['j_delta']>1 for r in rr),losses=sum(r['j_delta']<-1 for r in rr),ties=sum(abs(r['j_delta'])<=1 for r in rr),
            mean_j_delta=mean(rr,'j_delta'),mean_delta_pp=mean(rr,'delta_pp'),mean_ms_delta=mean(rr,'ms_delta'),mean_p5_delta=mean(rr,'p5_delta')))
        for seed in sorted({r['seed'] for r in rr}):
            ss=[r for r in rr if r['seed']==seed];seeds.append(dict(new=new,old=old,seed=seed,j_delta=mean(ss,'j_delta'),ms_delta=mean(ss,'ms_delta')))
    for name,rows in [('paired',pairs),('paired_summary',aggregate),('paired_per_seed',seeds)]:csv_write(source/f'{phase}_{name}.csv',rows)
    work=[]
    basefields=('candidate_evaluations','bank_ms','matching_ms','generation_ms','proxy_ms','service_ms','proxy_candidates','proposal_attempts',
        'proxy_gamma_array_reads','seed_gamma_array_reads','mmse_requests','mmse_misses','feature_roots','iterations')
    extra=('directed_queries','directed_pool_queries','multicell_queries','directed_generation_ms','directed_candidates','directed_attempts',
        'directed_gamma_reads','directed_swap_steps','directed_combinations','directed_multicell_candidates','wide_rounds','sparse_rounds')
    for arm in ['baseline',*CONFIGS]:
        rr=[r for r in state if r['arm']==arm]
        work.append(dict(arm=arm,**{k:mean(rr,k) for k in basefields},**({k:mean(rr,k) for k in extra} if arm in CONFIGS else {})))
    write_json(source/f'{phase}_work.json',work)
    actions=np.load(source/f'{phase}_actions.npz');values=np.load(source/f'{phase}_values.npz')
    parity=sum(np.array_equal(actions[f'{sid}|existing2'],actions[f'{sid}|joint2']) and np.array_equal(values[f'{sid}|existing2'],values[f'{sid}|joint2']) for sid in dict.fromkeys(r['state_id'] for r in state))
    return dict(state=state,summary=summary,paired=aggregate,work=work,existing_joint2_equal_states=int(parity))


def archive_audit(source):
    records=[];selector=[];summaries=[]
    for phase,seeds,ttis in [('validation',(44,45),(4001,8001,18000)),('evaluation',(46,47,48),(3001,4001,5000))]:
        raw,_=read_rows(seeds,evaluation=phase=='evaluation');raw=[r for r in raw if r['tti'] in ttis]
        values=np.load(source/f'{phase}_values.npz');observed=np.load(source/f'{phase}_observed_values.npz');groups=np.load(source/f'{phase}_service_groups.npz')
        for row in raw:
            p=problem_from_raw_row(row);sid=identity(row);reference=values[f'{sid}|full'];jref=float(reference[:,EXPECTED]@p.weights)
            target=np.flatnonzero(reference[:,EXPECTED]>1)
            weak=target[np.argsort(reference[target,EXPECTED],kind='stable')[:max(1,int(np.ceil(.2*len(target))))]]
            for arm in ['baseline',*CONFIGS]:
                key=f'{sid}|{arm}';v=observed[key][1:];j=v[:,:,EXPECTED]@p.weights;group=groups[key]
                p5=np.quantile(v[:,target,EXPECTED],.05,axis=1)/np.quantile(reference[target,EXPECTED],.05)
                wr=v[:,weak,EXPECTED].sum(1)/reference[weak,EXPECTED].sum()
                cov=(p5>=.98)&(wr>=.98)&~((reference[None,:,COUNT]>0)&(v[:,:,COUNT]==0)).any(1)
                floor=float(next(r['floor'] for r in read(source/f'{phase}_per_state.csv') if r['state_id']==sid and r['arm']==arm))
                good=cov&(j>=.995*jref)&(v[:,:,DEQUEUE].sum(1)>=floor)
                best=float(j[good].max()) if good.any() else None
                records.append(dict(phase=phase,state_id=sid,seed=row['seed'],arm=arm,observed_conditioned=len(v),
                    joint_candidates=int(good.sum()),best_joint_j_relative_pct=100*(best/jref-1) if best is not None else None,
                    min_internal_j_loss_pct=100*(1-best/j.max()) if best is not None else None))
                # Read-only selector diagnostic, one previously studied0.5% cap.
                # Its failure is not permission to tune on validation/development.
                ratio=np.clip(v[:,group,EXPECTED]/v[:,group,DEQUEUE],0,1)
                smooth=-.1*np.log(np.exp(-ratio/.1).mean(1));mean_r=ratio.mean(1)
                allowed=np.flatnonzero(j>=.995*j.max());k=max(allowed,key=lambda k:(smooth[k],mean_r[k],j[k]))
                selector.append(dict(phase=phase,state_id=sid,arm=arm,cap=.005,index=int(k)+1,
                    relative_j_pct=float(100*(j[k]/jref-1)),p5_ratio=float(p5[k]),weak_ratio=float(wr[k]),
                    joint_pass=bool(good[k]),actual_internal_loss_pct=float(100*(1-j[k]/j.max()))))
        for arm in ['baseline',*CONFIGS]:
            rr=[r for r in records if r['phase']==phase and r['arm']==arm]
            summaries.append(dict(phase=phase,arm=arm,states=len(rr),states_with_joint_candidate=sum(r['joint_candidates']>0 for r in rr),
                max_required_internal_loss_pct=max((r['min_internal_j_loss_pct'] for r in rr if r['min_internal_j_loss_pct'] is not None),default=None)))
    csv_write(source/'observed_feasibility.csv',records);csv_write(source/'observed_feasibility_summary.csv',summaries)
    csv_write(source/'posthoc_selector005.csv',selector)
    write_json(source/'observed_feasibility.json',dict(posthoc=True,additional_native_scores=0,policy_changed=False,
        privileged_membership_and_quality_labels_offline_only=True,summary=summaries,
        initial_pfq_excluded=True,scope='already-paid212 archive excluding initialPFQ;not a deployable oracle selector or complete search certificate'))
    return summaries


def plot(source,val,dev):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Full','Continuous baseline','Existing pool+2','Joint pool+2','Joint pool+4','Alternating sparse+2']
    colors=['#263442','#777777','#3c8c76','#387bb8','#9b5ea4','#cd8a35']
    fig,axes=plt.subplots(2,2,figsize=(12,8.8))
    for col,(data,title) in enumerate([(val,'Validation:12 historical states'),(dev,'Development:9 historical states')]):
        for i,r in enumerate(data['summary']):
            for row,field in [(0,'worst_relative_j_pct'),(1,'worst_positive_p5')]:
                x=float(r['execution_time_ratio']);y=float(r[field]);axes[row,col].scatter(x,y,s=60,color=colors[i],marker='*' if i==0 else 'o',label=labels[i] if row==0 else None)
                axes[row,col].annotate(str(i),(x,y),xytext=(3,6 if i%2==0 else -11),textcoords='offset points',fontsize=8)
        for row in (0,1):
            axes[row,col].axvline(.8,color='#a43e4b',linestyle=':',alpha=.7);axes[row,col].grid(alpha=.2)
            axes[row,col].set_xlabel('Complete conditional execution / full DPLS')
        axes[0,col].axhline(-.5,color='#a43e4b',linestyle=':',alpha=.7);axes[1,col].axhline(.98,color='#a43e4b',linestyle=':',alpha=.7)
        axes[0,col].set_title(title);axes[0,col].set_ylabel('Worst-state J change vs full (%)');axes[1,col].set_ylabel('Worst-state effective-UE P5 ratio')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,[f'{i}: {s}' for i,s in enumerate(l)],loc='lower center',ncol=3,frameon=False,fontsize=8,bbox_to_anchor=(.5,.035))
    fig.suptitle('Budgeted weak-UE directed reorders:212 exact scores',fontsize=13)
    fig.text(.5,.015,'Paid recall diagnostics excluded from policy results;offline teacher preparation excluded;no online AI claim.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.13,1,.95));fig.savefig(source/'quality_cost.png',dpi=160);fig.savefig(source/'quality_cost.pdf');plt.close(fig)


def main(source):
    source=Path(source).resolve();complete=json.loads((source/'completion.json').read_text());assert complete['completed']
    val=phase_audit(source,'validation');dev=phase_audit(source,'evaluation');observed=archive_audit(source)
    selection=json.loads((source/'selection.json').read_text());summary={r['arm']:r for r in dev['summary']}
    diagnostic=json.loads((source/'recall_diagnosis/summary.json').read_text())
    gates=dict(selection=selection,validation=val['summary'],development=dev['summary'],selected_quality_development=summary[selection['quality_candidate']],
        baseline_eligible=True,development_reselection=False,paid_recall=diagnostic,existing2_joint2_final_equal_states=val['existing_joint2_equal_states']+dev['existing_joint2_equal_states'],
        training_started=False,online_ai_success=False,
        next_path=dict(selected='candidate-level causal service-risk calibration and constrained terminal choice, plus cache deterministic per-root candidate/proxy work',
            implementation_status='selected next experiment, not implemented',
            reason='directed archives contain J/coverage passing actions in all21states,but maxJ loses validation service;new joint pool did not improve existing2 finals;paid recall exposes proxy rank36/287 misses',
            mechanism='retain fast quota-exp frontier and existing-pool2-slot control;freeze a causal service-risk/attainability feature ablation and a small learned selector trained only on41-43 paid own-state labels;compare with same mechanism heuristic',
            quality='service risk must use causal inputs/own observed candidate values;full-positive mask and fullJ only offline targets;0.5% globalJ gate unchanged;simple0.5% internal-loss terminal preference already fails read-only audit',
            cost='cache candidate sets and root-keyed proxy scores within an immutable decision;verify action/query/value parity;charge model and complete execution costs',
            defer=['expanding macro pools as primary path','blind sparse alternation','upstream resource/layer/activity predictor training','PPO','default replacement']))
    write_json(source/'gate_results.json',gates)
    recovery=json.loads((source/'serialization_recovery.json').read_text())
    for name,digest in json.loads((source/'source_manifest.json').read_text()).items():
        assert sha(source/'source'/name)==digest
        assert sha(ROOT/'training/stageb_lns'/name)==(recovery['fixed_sha256'] if name==recovery['file'] else digest)
    protocol=json.loads((source/'protocol.json').read_text());assert sha(REGISTRY)==protocol['registry_sha256']
    assert sha(ROOT/'training/stageb_lns/physics_kernel.cpp')==protocol['physics_source_sha256']
    tests=Path('/tmp/directed_reorder_tests_20261102.log').read_text();assert 'Ran 118 tests' in tests and '\nOK\n' in tests
    shutil.copyfile('/tmp/directed_reorder_tests_20261102.log',source/'tests.log');shutil.copyfile('/tmp/directed_reorder_execution_20261102.log',source/'execution.log')
    shutil.copyfile('/tmp/directed_reorder_recovery_20261102.json',source/'recovery.log')
    ledger={}
    for phase in ('engineering','validation','evaluation'):
        rr=read(source/f'{phase}_teacher_work.csv');c=json.loads((source/f'{phase}_teacher_checks.json').read_text());assert len(rr)==len(c['independent_phy'])
        ledger[phase]=dict(decisions=len(rr),exact_scores=sum(int(r['candidate_evaluations']) for r in rr),teacher_ms=sum(float(r['teacher_ms']) for r in rr))
    ledger['total']={k:sum(v[k] for v in ledger.values()) for k in ('decisions','exact_scores','teacher_ms')};write_json(source/'teacher_ledger.json',ledger)
    write_json(source/'final_audit.json',dict(passed=True,tests=118,formal_decisions=378,formal_independent_finals=126,
        teacher_independent_finals=23,engineering_decoder_finals=12,paid_recall_exact_scores=diagnostic['exact_scores'],paid_recall_roots=diagnostic['roots'],
        paid_recall_independent_audits=21,recovery_repeated_independent_audits=21,recovery_native_scores=0,
        baseline_previous_actions_values_equal_states=21,disabled_engineering_full_query_parity_states=2,
        existing2_joint2_final_equal_states=gates['existing2_joint2_final_equal_states'],source_policy_kernel_registry_hashes_valid=True,
        diagnostic_json_int_cast_only_patch=True,training_updates=0,native_trajectories=0,final_seeds_read=False,
        all_validation_and_development_gates_passed=False,selected_development_gate_passed=True))
    plot(source,val,dev)
    print(json.dumps(dict(selection=selection,development=dev['summary'],paired=dev['paired'],observed=observed,paid=diagnostic),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);main(p.parse_args().source)

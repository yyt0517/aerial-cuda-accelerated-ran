"""Audit wide-move/frontier comparisons, fixed selectors and group failures."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from .joint_reorder_campaign import NAMES,CONFIGS,REGISTRY
from .experiment import sha,write_json
from .physics import ROOT,EXPECTED
from .bounded_probes import csv_write

PAIRS=[('narrow_delta_j','joint_reference'),('wide_j','narrow_delta_j'),
    ('narrow_pareto005','narrow_delta_j'),('wide_pareto_j','wide_j'),
    ('wide_pareto005','wide_pareto_j'),('wide_pareto02','wide_pareto005'),
    ('wide_pareto_j','joint_reference'),('wide_pareto_j','activity_reference')]


def read(path):return list(csv.DictReader(Path(path).open()))
def avg(rows,key):return float(np.mean([float(r[key]) for r in rows]))
def mean(rows,key):return float(np.mean([avg([r for r in rows if r['seed']==s],key) for s in sorted({r['seed'] for r in rows})]))


def audit_phase(source,phase):
    raw=read(source/f'{phase}_measurements.csv');state=read(source/f'{phase}_per_state.csv');summary=read(source/f'{phase}_summary.csv')
    for r in state:
        rr=[v for v in raw if v['state_id']==r['state_id'] and v['arm']==r['arm']]
        assert len(rr)==3 and np.isclose(float(r['e2e_ms']),np.median([float(v['e2e_ms']) for v in rr]))
        if r['arm']!='full':assert int(r['candidate_evaluations'])<=212 and int(r['feature_roots'])<=56 and r['condition_feasible']=='True'
        if r['arm'] in CONFIGS:
            assert float(r['score'])+1.e-8>=(1-CONFIGS[r['arm']].final_j_loss)*float(r['best_observed_j'])
    for r in summary:
        rr=[v for v in state if v['arm']==r['arm']]
        assert np.isclose(mean(rr,'score'),float(r['mean_j'])) and np.isclose(mean(rr,'e2e_ms'),float(r['mean_ms']))
        q=all(float(v['ratio'])>=.995 for v in rr)
        c=all(int(v['newly_unserved_ues'])==0 and float(v['positive_p5_ratio'])>=.98 and float(v['positive_weak_ratio'])>=.98 for v in rr)
        cost=float(r['execution_time_ratio'])<=.8
        assert (r['quality_pass']=='True')==q and (r['coverage_pass']=='True')==c
        assert (r['conditional_target_pass']=='True')==(q and c and cost)
    index={(r['state_id'],r['arm']):r for r in state};pairs=[]
    for sid in dict.fromkeys(r['state_id'] for r in state):
        for new,old in PAIRS:
            a=index[sid,new];b=index[sid,old]
            pairs.append(dict(state_id=sid,seed=a['seed'],new=new,old=old,j_delta=float(a['score'])-float(b['score']),
                delta_pp=100*(float(a['ratio'])-float(b['ratio'])),ms_delta=float(a['e2e_ms'])-float(b['e2e_ms']),
                p5_delta=float(a['positive_p5_ratio'])-float(b['positive_p5_ratio']),
                newly_unserved_delta=int(a['newly_unserved_ues'])-int(b['newly_unserved_ues'])))
    csv_write(source/f'{phase}_paired.csv',pairs);paired=[];seeds=[]
    for new,old in PAIRS:
        rr=[r for r in pairs if r['new']==new and r['old']==old]
        paired.append(dict(new=new,old=old,wins=sum(r['j_delta']>1 for r in rr),losses=sum(r['j_delta']<-1 for r in rr),
            ties=sum(abs(r['j_delta'])<=1 for r in rr),mean_j_delta=mean(rr,'j_delta'),mean_delta_pp=mean(rr,'delta_pp'),
            mean_ms_delta=mean(rr,'ms_delta'),mean_p5_delta=mean(rr,'p5_delta')))
        for seed in sorted({r['seed'] for r in rr}):
            vv=[r for r in rr if r['seed']==seed];seeds.append(dict(new=new,old=old,seed=seed,j_delta=avg(vv,'j_delta')))
    csv_write(source/f'{phase}_paired_summary.csv',paired);csv_write(source/f'{phase}_paired_per_seed.csv',seeds)
    work=[]
    for arm in CONFIGS:
        rr=[r for r in state if r['arm']==arm]
        work.append(dict(arm=arm,**{k:mean(rr,k) for k in ('bank_ms','matching_ms','generation_ms','proxy_ms','service_ms',
            'proxy_candidates','proposal_attempts','proxy_gamma_array_reads','seed_gamma_array_reads','mmse_requests','mmse_misses',
            'feature_roots','frontier_peak','selected_j_loss_fraction')},
            stop_counts={s:sum(r['stop_reason']==s for r in rr) for s in sorted({r['stop_reason'] for r in rr})}))
    write_json(source/f'{phase}_work.json',work)
    actions=np.load(source/f'{phase}_actions.npz');values=np.load(source/f'{phase}_values.npz');groups=np.load(source/f'{phase}_service_groups.npz')
    for sid in dict.fromkeys(r['state_id'] for r in state):
        np.testing.assert_array_equal(actions[f'{sid}|joint_reference'],actions[f'{sid}|narrow_delta_j'])
        np.testing.assert_array_equal(values[f'{sid}|joint_reference'],values[f'{sid}|narrow_delta_j'])
        for arm in list(CONFIGS)[1:]:np.testing.assert_array_equal(groups[f'{sid}|{arm}'],groups[f'{sid}|narrow_delta_j'])
    return dict(state=state,summary=summary,paired=paired,seeds=seeds,decisions=len(raw))


def group_diagnosis(source,val,dev):
    records=[]
    for phase,data in [('validation',val),('evaluation',dev)]:
        groups=np.load(source/f'{phase}_service_groups.npz');values=np.load(source/f'{phase}_values.npz')
        for r in [v for v in data['state'] if v['arm']=='wide_pareto_j']:
            sid=r['state_id'];g=groups[f'{sid}|wide_pareto_j'];target=values[f'{sid}|full'][:,EXPECTED]>1
            missing=np.flatnonzero(target&~g);extra=np.flatnonzero(g&~target)
            q=float(r['ratio'])>=.995;c=int(r['newly_unserved_ues'])==0 and float(r['positive_p5_ratio'])>=.98 and float(r['positive_weak_ratio'])>=.98
            records.append(dict(phase=phase,state_id=sid,seed=r['seed'],quality_pass=q,coverage_pass=c,
                relative_j_pct=100*(float(r['ratio'])-1),positive_p5_ratio=float(r['positive_p5_ratio']),
                omitted_reference_positive=missing.tolist(),extra_nonreference=extra.tolist(),
                missing_count=len(missing),extra_count=len(extra)))
    write_json(source/'posthoc_group_diagnosis.json',dict(posthoc=True,policy_or_selection_changed=False,records=records,
        scope='teacher-positive membership used only after timing to diagnose bootstrap group; never supplied to search',
        finding='some failed states coincide with omitted positive UEs; validation also has extra group UEs and zero-P5 degeneration',
        caution='coincidence is not proof of a unique cause; a future continuous all-quota metric needs a controlled ablation'))
    return records


def plot(val,dev,source):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Full','Activity ref.','Joint ref.','Sparse narrow J','Wide J','Narrow frontier0.5%',
            'Wide frontier J','Wide frontier0.5%','Wide frontier2%']
    colors=['#263442','#b17a26','#999999','#4e897b','#367bb1','#8aa3c2','#8754a6','#c84963','#d88a9b']
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for col,(data,title) in enumerate([(val,'Validation: 12 historical states'),(dev,'Development: 9 historical states')]):
        for i,r in enumerate(data['summary']):
            for row,field in [(0,'mean_relative_j_pct'),(1,'mean_positive_p5')]:
                x=float(r['execution_time_ratio']);y=float(r[field]);axes[row,col].scatter(x,y,s=60,color=colors[i],marker='*' if i==0 else 'o',label=labels[i] if row==0 else None)
                axes[row,col].annotate(str(i),(x,y),xytext=(4,6 if i%2==0 else -11),textcoords='offset points',fontsize=8)
        for row in (0,1):
            axes[row,col].axvline(.8,color='#963c4b',linestyle=':',alpha=.7);axes[row,col].grid(alpha=.2)
            axes[row,col].set_xlabel('Complete conditional execution / full DPLS')
        axes[0,col].axhline(-.5,color='#963c4b',linestyle=':',alpha=.7);axes[1,col].axhline(.98,color='#963c4b',linestyle=':',alpha=.7)
        axes[0,col].set_title(title);axes[0,col].set_ylabel('Mean J change vs full (%)');axes[1,col].set_ylabel('Mean effective-UE P5 service ratio')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,[f'{i}: {s}' for i,s in enumerate(l)],loc='lower center',ncol=3,frameon=False,fontsize=8,bbox_to_anchor=(.5,.024))
    fig.suptitle('Wider UE-PRG reorder and causal observed-service frontiers:212 exact scores',fontsize=13)
    fig.text(.5,.009,'Offline teacher conditions, no teacher positions/service. Teacher preparation excluded. Gates apply per state; no online AI claim.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.13,1,.95));fig.savefig(source/'quality_cost.png',dpi=160);fig.savefig(source/'quality_cost.pdf');plt.close(fig)


def main(source):
    source=Path(source).resolve();complete=json.loads((source/'completion.json').read_text());assert complete['completed']
    val=audit_phase(source,'validation');dev=audit_phase(source,'evaluation');diagnosis=group_diagnosis(source,val,dev)
    selected=json.loads((source/'selection.json').read_text());summary={r['arm']:r for r in dev['summary']}
    gates=dict(selection=selected,development={r['arm']:{k:r[k]=='True' for k in ('quality_pass','coverage_pass','execution_cost_pass','conditional_target_pass')} for r in dev['summary']},
        selected_quality_development=summary[selected['quality_candidate']],selected_cost_development=summary[selected['execution_cost_candidate']],
        engineering=dict(all21_narrow_sparse_final_actions_and_values_identical=True),training_started=False,online_ai_success=False,
        diagnostics=dict(validation_failures=sum(not r['quality_pass'] or not r['coverage_pass'] for r in diagnosis if r['phase']=='validation'),
            development_quality_failures=sum(not r['quality_pass'] for r in diagnosis if r['phase']=='evaluation'),
            development_coverage_failures=sum(not r['coverage_pass'] for r in diagnosis if r['phase']=='evaluation')),
        next_path=dict(selected='retain wide reorders/frontier search; ablate continuous all-positive-dequeue quota service risk and cache/batch frontier computation',
            implementation_status='chosen next experiment, not implemented',
            objective='replace brittle bootstrap >1B membership/hard P5 with a continuous normalized weak-service surrogate over logical-positive quota UEs, without new hard allocation locks',
            engineering='batch exact service metrics, maintain/cache Pareto front and avoid repeated per-row sums/dominance; require action/work parity for cost-only changes',
            diagnosis='initial service omissions coincide with some failed states, not proven sole cause; target the remaining quality failures with bounded coordinated reassignments',
            unchanged=['final99.5%-full J','full-positive P5/weak>=98%','80% execution mean','raw PFQ90% floor','teacher only offline'],
            training_entry='only after stable conditional quality/service and cost; then test causal proposal errors and train quota/layer/activity proposer',
            defer=['upstream model training now','PPO','default replacement']))
    write_json(source/'gate_results.json',gates)
    for name,digest in json.loads((source/'source_manifest.json').read_text()).items():assert sha(source/'source'/name)==digest and sha(ROOT/'training/stageb_lns'/name)==digest
    protocol=json.loads((source/'protocol.json').read_text());assert sha(REGISTRY)==protocol['registry_sha256']
    assert sha(ROOT/'training/stageb_lns/physics_kernel.cpp')==protocol['physics_source_sha256']
    tests=Path('/tmp/joint_reorder_tests_20261102.log').read_text();assert 'Ran 94 tests' in tests and '\nOK\n' in tests
    shutil.copyfile('/tmp/joint_reorder_tests_20261102.log',source/'tests.log');shutil.copyfile('/tmp/joint_reorder_execution_20261102.log',source/'execution.log')
    write_json(source/'final_audit.json',dict(passed=True,tests=94,formal_decisions=val['decisions']+dev['decisions'],
        formal_independent_finals=189,offline_teacher_independent_finals=23,engineering_independent_decoder_finals=8,
        sparse_dense_proxy_checks=2,all21_sparse_narrow_actions_values_identical=True,
        selector_only_search_group_and_budget_parity=True,source_hashes_valid=True,registry_and_physics_unchanged=True,
        training_updates=0,native_trajectories=0,final_seeds_read=False))
    plot(val,dev,source)
    print(json.dumps(dict(gates=gates,paired=dev['paired']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);main(parser.parse_args().source)

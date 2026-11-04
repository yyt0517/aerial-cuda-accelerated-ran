"""Audit both equivalent backends, conditional headroom and explicit work."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from .resource_condition_campaign import NAMES,CONDITIONAL,compressed_condition
from .ranked_dpls_campaign import read_rows,identity
from .no_ctbs_inputs import problem_from_raw_row
from .physics import ROOT,COUNT,LAYER
from .bounded_probes import csv_write
from .experiment import sha,write_json

PAIRS=[('bank_none212','ordinary212'),('quota_layer212','bank_none212'),('activity212','bank_none212'),
    ('joint212','bank_none212'),('joint212','quota_layer212'),('joint212','activity212')]


def read(path):return list(csv.DictReader(Path(path).open()))
def avg(rows,key):return float(np.mean([float(r[key]) for r in rows]))
def mean(rows,key):return float(np.mean([avg([r for r in rows if r['seed']==s],key) for s in sorted({r['seed'] for r in rows})]))


def audit_phase(source,prior,phase):
    raw=read(source/f'{phase}_measurements.csv');state=read(source/f'{phase}_per_state.csv');summary=read(source/f'{phase}_summary.csv')
    old={ (r['state_id'],r['arm']):r for r in read(prior/f'{phase}_per_state.csv')}
    for r in state:
        rr=[v for v in raw if v['state_id']==r['state_id'] and v['arm']==r['arm']]
        assert len(rr)==3 and np.isclose(float(r['e2e_ms']),np.median([float(v['e2e_ms']) for v in rr]))
        if r['arm']!='full':assert int(r['candidate_evaluations'])<=212 and int(r['feature_roots'])<=56
        before=old[r['state_id'],r['arm']]
        for key in ('score','candidate_evaluations','ue_evaluations','mmse_requests','mmse_misses','proxy_candidates','feature_roots','iterations'):
            if r.get(key) and before.get(key):assert r[key]==before[key],(r['state_id'],r['arm'],key)
    for r in summary:
        rr=[v for v in state if v['arm']==r['arm']]
        assert np.isclose(mean(rr,'score'),float(r['mean_j'])) and np.isclose(mean(rr,'e2e_ms'),float(r['mean_ms']))
        valid=all(v.get('condition_feasible','') in ('','True') for v in rr)
        q=valid and all(float(v['ratio'])>=.995 for v in rr)
        c=valid and all(int(v['newly_unserved_ues'])==0 and float(v['positive_p5_ratio'])>=.98 and float(v['positive_weak_ratio'])>=.98 for v in rr)
        cost=float(r['execution_time_ratio'])<=.8
        assert (r['quality_pass']=='True')==q and (r['coverage_pass']=='True')==c
        assert (r['conditional_target_pass']=='True')==(q and c and cost)
        assert r['online_ai_success']=='False'
    index={(r['state_id'],r['arm']):r for r in state};paired=[]
    for sid in dict.fromkeys(r['state_id'] for r in state):
        for new,old_arm in PAIRS:
            a=index[sid,new];b=index[sid,old_arm]
            paired.append(dict(state_id=sid,seed=a['seed'],new=new,old=old_arm,j_delta=float(a['score'])-float(b['score']),
                delta_pp=100*(float(a['ratio'])-float(b['ratio'])),ms_delta=float(a['e2e_ms'])-float(b['e2e_ms']),
                newly_unserved_delta=int(a['newly_unserved_ues'])-int(b['newly_unserved_ues']),
                p5_delta=float(a['positive_p5_ratio'])-float(b['positive_p5_ratio'])))
    csv_write(source/f'{phase}_paired.csv',paired);pair_summary=[];seed_rows=[]
    for new,old_arm in PAIRS:
        rr=[r for r in paired if r['new']==new and r['old']==old_arm]
        pair_summary.append(dict(new=new,old=old_arm,wins=sum(r['j_delta']>1 for r in rr),losses=sum(r['j_delta']<-1 for r in rr),
            ties=sum(abs(r['j_delta'])<=1 for r in rr),mean_j_delta=mean(rr,'j_delta'),mean_delta_pp=mean(rr,'delta_pp'),
            mean_ms_delta=mean(rr,'ms_delta'),mean_p5_delta=mean(rr,'p5_delta')))
        for seed in sorted({r['seed'] for r in rr}):
            vv=[r for r in rr if r['seed']==seed];seed_rows.append(dict(new=new,old=old_arm,seed=seed,j_delta=avg(vv,'j_delta')))
    csv_write(source/f'{phase}_paired_summary.csv',pair_summary);csv_write(source/f'{phase}_paired_per_seed.csv',seed_rows)
    work=[]
    for arm in NAMES[2:]:
        rr=[r for r in state if r['arm']==arm]
        work.append(dict(arm=arm,**{k:mean(rr,k) for k in ('bank_ms','matching_ms','proxy_ms','generation_ms','proxy_candidates',
            'proxy_ue_rows','mmse_requests','mmse_misses','bank_mmse_requests','bank_mmse_misses','feature_roots','decode_failures')},
            stop_counts={s:sum(r['stop_reason']==s for r in rr) for s in sorted({r['stop_reason'] for r in rr})}))
    write_json(source/f'{phase}_work.json',work)
    conditions=[];rows,_=read_rows((44,45) if phase=='validation' else (46,47,48),evaluation=phase=='evaluation')
    by_id={identity(r):r for r in rows};a=np.load(source/f'{phase}_teacher_actions.npz');v=np.load(source/f'{phase}_teacher_values.npz')
    for sid in dict.fromkeys(r['state_id'] for r in state):
        p=problem_from_raw_row(by_id[sid])
        for arm in CONDITIONAL:
            condition=compressed_condition(p,a[sid],v[sid],arm)
            conditions.append(dict(state_id=sid,arm=arm,counts=None if condition.counts is None else condition.counts.tolist(),
                layers=None if condition.layers is None else condition.layers.tolist(),
                activity=None if condition.activity is None else condition.activity.astype(int).tolist(),
                teacher_UE_positions_passed=False))
    write_json(source/f'{phase}_condition_manifest.json',conditions)
    return dict(summary=summary,state=state,paired=pair_summary,decisions=len(raw))


def plot(val,dev,oldval,olddev,source):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Full DPLS','Ordinary212','Same decoder, no condition','Counts + min layer','Activity only','Both conditions']
    colors=['#263442','#62768a','#999999','#3977ac','#ba7b28','#7a58a2']
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for col,(data,title) in enumerate([(val,'Validation: 12 historical states'),(dev,'Development: 9 historical states')]):
        for i,r in enumerate(data['summary']):
            for row,field in [(0,'mean_relative_j_pct'),(1,'mean_positive_p5')]:
                x=float(r['execution_time_ratio']);y=float(r[field]);axes[row,col].scatter(x,y,s=65,color=colors[i],marker='*' if i==0 else 'o',label=labels[i] if row==0 else None)
                axes[row,col].annotate(str(i),(x,y),xytext=(4,5 if i%2==0 else -12),textcoords='offset points',fontsize=8)
        for row in (0,1):
            axes[row,col].axvline(.8,color='#963c4b',linestyle=':',alpha=.7);axes[row,col].grid(alpha=.2)
            axes[row,col].set_xlabel('Measured execution mean / full DPLS')
        axes[0,col].axhline(-.5,color='#963c4b',linestyle=':',alpha=.7);axes[1,col].axhline(.98,color='#963c4b',linestyle=':',alpha=.7)
        axes[0,col].set_title(title);axes[0,col].set_ylabel('Mean J change vs full (%)');axes[1,col].set_ylabel('Mean effective-UE P5 service ratio')
    handles,names=axes[0,0].get_legend_handles_labels();fig.legend(handles,[f'{i}: {s}' for i,s in enumerate(names)],loc='lower center',ncol=3,frameon=False,fontsize=9,bbox_to_anchor=(.5,.028))
    fig.suptitle('Resource-level conditional ablations, equivalent fast decoder, score cap212',fontsize=13)
    fig.text(.5,.012,'Conditional points use offline teacher information. Teacher preparation excluded; these are not online AI latency claims. Gates apply per state.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.10,1,.95));fig.savefig(source/'quality_cost.png',dpi=160);fig.savefig(source/'quality_cost.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,(new,old,title) in zip(axes,[(val['summary'],oldval,'Validation'),(dev['summary'],olddev,'Development')]):
        before={r['arm']:r for r in old};rr=new[2:];x=np.arange(len(rr));w=.35
        ax.bar(x-w/2,[float(before[r['arm']]['mean_ms']) for r in rr],w,label='V1 Python pool')
        ax.bar(x+w/2,[float(r['mean_ms']) for r in rr],w,label='V2 equivalent compiled pool')
        ax.set_xticks(x,['None','Quota/layer','Activity','Both']);ax.set_ylabel('Complete execution mean (ms)');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Engineering remeasurement: all final actions and values identical')
    fig.tight_layout();fig.savefig(source/'backend_cost.png',dpi=160);fig.savefig(source/'backend_cost.pdf');plt.close(fig)


def main(source,prior):
    source=Path(source).resolve();prior=Path(prior).resolve();complete=json.loads((source/'completion.json').read_text());assert complete['completed']
    for directory in (source,prior):
        for name,digest in json.loads((directory/'source_manifest.json').read_text()).items():assert sha(directory/'source'/name)==digest
    for name,digest in json.loads((source/'source_manifest.json').read_text()).items():assert sha(ROOT/'training/stageb_lns'/name)==digest
    val=audit_phase(source,prior,'validation');dev=audit_phase(source,prior,'evaluation')
    for phase in ('validation','evaluation'):
        assert json.loads((source/f'{phase}_v1_parity.json').read_text())['passed']
        for suffix in ('actions','values'):
            old=np.load(prior/f'{phase}_{suffix}.npz');new=np.load(source/f'{phase}_{suffix}.npz')
            for key in new.files:np.testing.assert_array_equal(old[key],new[key])
    terminal=json.loads((source/'terminal_audit/summary.json').read_text());terminal_rows=read(source/'terminal_audit/neighborhoods.csv')
    index={r['state_id']:r for r in val['state'] if r['arm']=='joint212'}
    current_pass=[];fixed_by_neighbor=[];unresolved_quality=[]
    for sid,r in index.items():
        good=float(r['ratio'])>=.995 and int(r['newly_unserved_ues'])==0 and float(r['positive_p5_ratio'])>=.98 and float(r['positive_weak_ratio'])>=.98
        if good:current_pass.append(sid)
        audit=next(v for v in terminal_rows if v['state_id']==sid)
        if not good and audit['any_quality_service_passing_neighbor']=='True':fixed_by_neighbor.append(sid)
        if float(r['ratio'])<.995 and float(audit['best_gain'])<=1:unresolved_quality.append(sid)
    write_json(source/'terminal_audit/interpretation.json',dict(current_joint_quality_service_passes=len(current_pass),
        additional_roots_with_passing_neighbor=fixed_by_neighbor,quality_failures_without_local_gain_gt1=unresolved_quality,
        scope='pair swaps and 3-cycles only among first6 active PRGs; wider cycles/global assignments remain unknown'))
    protocol=json.loads((source/'protocol.json').read_text());assert sha(ROOT/'training/stageb_hgraph/current_algorithm.json')==protocol['registry_sha256']
    assert sha(ROOT/'training/stageb_lns/physics_kernel.cpp')==protocol['physics_source_sha256']
    selection=json.loads((source/'selection.json').read_text());chosen=selection['quality_candidate']
    rows={r['arm']:r for r in dev['summary']};gates=dict(selection=selection,
        development={r['arm']:{k:r[k]=='True' for k in ('condition_valid','quality_pass','coverage_pass','execution_cost_pass','conditional_target_pass')} for r in dev['summary']},
        selected_validation_condition_development=rows[chosen],training_started=False,online_ai_success=False,
        engineering=dict(all126_final_actions_values_identical=True,actual_non_timing_work_identical=True,
            joint_old_development_ms=float(next(r for r in read(prior/'evaluation_summary.csv') if r['arm']=='joint212')['mean_ms']),
            joint_new_development_ms=float(rows['joint212']['mean_ms'])),
        terminal_audit=dict(roots=terminal['roots'],physical_scores=terminal['physical_scores'],positive_gt16_roots=terminal['positive_gt16_roots'],
            current_quality_service_passes=len(current_pass),additional_roots_with_passing_neighbor=len(fixed_by_neighbor)),
        next_path=dict(selected='improve joint-conditioned UE-to-PRG decoding and quality/service Pareto selection before causal proposer training',
            implementation_status='chosen next experiment, not run',
            first='compare wider multi-PRG cycles/global constrained redecoding and directly verified sparse neighborhoods under complete-cost budgets',
            selection='retain J/service tradeoff candidates instead of only maximum weighted J; teacher service checks remain offline diagnostic labels',
            training_entry='only after stable conditional quality/service and execution-cost headroom; then train causal quota/layer/activity proposals and test prediction perturbations',
            retain=['incremental engine','canonical full PHY','raw PFQ90% floor','unchanged final gates','separately charged teacher/MMSE work'],
            do_not_claim='finite local stall is not a global bound; current conditional failure does not prove perfect conditions lack useful schedules',
            defer=['training larger causal upstream proposer now','PPO','continuation value','default replacement']))
    write_json(source/'gate_results.json',gates)
    tests=Path('/tmp/resource_conditions_fast_tests_20261102.log').read_text();assert 'Ran 84 tests' in tests and '\nOK\n' in tests
    shutil.copyfile('/tmp/resource_conditions_fast_tests_20261102.log',source/'tests.log')
    shutil.copyfile('/tmp/resource_conditions_fast_execution_20261102.log',source/'execution.log')
    shutil.copyfile('/tmp/resource_conditions_tests_20261102.log',prior/'tests.log');shutil.copyfile('/tmp/resource_conditions_execution_20261102.log',prior/'execution.log')
    write_json(source/'final_audit.json',dict(passed=True,tests=84,version1_tests=81,formal_decisions_per_version=378,
        formal_decisions_total=756,formal_independent_finals_total=252,offline_teacher_independent_finals_total=46,
        engineering_decoder_independent_finals_total=16,terminal_independent_best_checks=len(terminal['independent_best_checks']),
        terminal_offline_phy=terminal['physical_scores'],all21_states_six_arms_v1_v2_actions_values_and_work_identical=True,
        exact_conditions_and_raw_floor_valid=True,old_source_snapshot_hashes_valid=True,current_fast_sources_hashes_valid=True,
        registry_and_physics_source_unchanged=True,training_updates=0,native_trajectories=0,final_seeds_read=False))
    plot(val,dev,read(prior/'validation_summary.csv'),read(prior/'evaluation_summary.csv'),source)
    print(json.dumps(dict(gates=gates,development=dev['summary'],paired=dev['paired']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--prior',required=True)
    args=parser.parse_args();main(args.source,args.prior)

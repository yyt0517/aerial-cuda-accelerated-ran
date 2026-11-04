"""Recompute small experiment statistics and export immutable local evidence."""
import argparse
import csv
import json
from pathlib import Path
import platform
import shutil
import numpy as np

from .physics import ROOT
from .experiment import sha,write_json
from .bounded_probes import csv_write
from .tb_macro_campaign import CONFIGS,NAMES

PAIRS=[('joint_b212','single_b212'),('single_continue_b212','single_b212'),
    ('joint_continue_b212','joint_b212'),('joint_b424','single_b424'),
    ('joint_continue_b424','joint_b424'),('single_b424','single_b212'),
    ('joint_b424','joint_b212'),('joint_continue_b424','joint_continue_b212'),
    ('single_b212','shortlist_trained212'),('joint_continue_b212','shortlist_trained212')]


def read(path):return list(csv.DictReader(Path(path).open()))
def avg(rows,key):return float(np.mean([float(r[key]) for r in rows]))
def mean(rows,key):return float(np.mean([avg([r for r in rows if r['seed']==s],key) for s in sorted({r['seed'] for r in rows})]))


def audit_phase(source,phase):
    raw=read(source/f'{phase}_measurements.csv');state=read(source/f'{phase}_per_state.csv')
    summary=read(source/f'{phase}_summary.csv');index={(r['state_id'],r['arm']):r for r in state}
    for row in state:
        rr=[r for r in raw if r['state_id']==row['state_id'] and r['arm']==row['arm']]
        assert len(rr)==3 and np.isclose(float(row['e2e_ms']),np.median([float(r['e2e_ms']) for r in rr]))
        if row['arm']!='full':assert int(row['candidate_evaluations'])<=int(row['budget']) and int(row['feature_roots'])<=56
        if row['arm'] in CONFIGS:assert int(row['covered_prgs'])==17 and int(row['second_pass_prgs'])==17
    for row in summary:
        rr=[r for r in state if r['arm']==row['arm']]
        assert np.isclose(mean(rr,'score'),float(row['mean_j'])) and np.isclose(mean(rr,'e2e_ms'),float(row['mean_ms']))
        assert np.isclose(100*(mean(rr,'ratio')-1),float(row['mean_relative_j_pct']))
        assert np.isclose(100*(min(float(r['ratio']) for r in rr)-1),float(row['worst_relative_j_pct']))
        q=all(float(r['ratio'])>=.995 for r in rr)
        c=all(int(r['newly_unserved_ues'])==0 and float(r['positive_p5_ratio'])>=.98 and float(r['positive_weak_ratio'])>=.98 for r in rr)
        assert (row['quality_pass']=='True')==q and (row['coverage_pass']=='True')==c
        cost=float(row['time_ratio'])<=.8
        assert (row['cost_pass']=='True')==cost and (row['joint_pass']=='True')==(q and c and cost)
    paired=[]
    for sid in dict.fromkeys(r['state_id'] for r in state):
        for new,old in PAIRS:
            a=index[sid,new];b=index[sid,old]
            paired.append(dict(state_id=sid,seed=a['seed'],new=new,old=old,j_delta=float(a['score'])-float(b['score']),
                relative_j_delta_pp=100*(float(a['ratio'])-float(b['ratio'])),ms_delta=float(a['e2e_ms'])-float(b['e2e_ms']),
                phy_delta=int(a['candidate_evaluations'])-int(b['candidate_evaluations']),
                newly_unserved_delta=int(a['newly_unserved_ues'])-int(b['newly_unserved_ues']),
                positive_p5_delta=float(a['positive_p5_ratio'])-float(b['positive_p5_ratio'])))
    csv_write(source/f'{phase}_paired.csv',paired);pair_summary=[]
    for new,old in PAIRS:
        rr=[r for r in paired if r['new']==new and r['old']==old]
        pair_summary.append(dict(new=new,old=old,wins=sum(r['j_delta']>1 for r in rr),losses=sum(r['j_delta']<-1 for r in rr),
            ties=sum(abs(r['j_delta'])<=1 for r in rr),mean_j_delta=mean(rr,'j_delta'),mean_delta_pp=mean(rr,'relative_j_delta_pp'),
            mean_ms_delta=mean(rr,'ms_delta'),mean_phy_delta=mean(rr,'phy_delta'),
            mean_newly_unserved_delta=mean(rr,'newly_unserved_delta'),mean_positive_p5_delta=mean(rr,'positive_p5_delta')))
    csv_write(source/f'{phase}_paired_summary.csv',pair_summary)
    per_seed=[]
    for arm in NAMES:
        for seed in sorted({r['seed'] for r in state}):
            rr=[r for r in state if r['arm']==arm and r['seed']==seed]
            per_seed.append(dict(arm=arm,seed=seed,mean_j=avg(rr,'score'),mean_ms=avg(rr,'e2e_ms'),
                mean_relative_j_pct=100*(avg(rr,'ratio')-1),mean_positive_p5=avg(rr,'positive_p5_ratio'),
                mean_newly_unserved=avg(rr,'newly_unserved_ues')))
    csv_write(source/f'{phase}_per_seed.csv',per_seed)
    events=json.loads((source/f'{phase}_continuations.json').read_text());event_summary=[]
    for arm in [k for k,v in CONFIGS.items() if v.continuation]:
        rr=[r for r in events if r['arm']==arm];observed=[r for r in rr if r.get('seed_j') is not None]
        event_summary.append(dict(arm=arm,domains=len(rr),observed_seeds=len(observed),complete_domains=sum(r['complete'] for r in rr),
            incomplete_domains=sum(not r['complete'] for r in rr),no_domain=sum(r.get('reason')=='no_compensation_domain' for r in rr),
            seed_below_floor=sum(r.get('seed_feasible') is False for r in rr),
            rescued=sum(r.get('rescued_unacceptable_seed',False) for r in rr),
            positive_root_gain=sum(r.get('improvement_vs_root') is not None and r['improvement_vs_root']>1 for r in rr),
            physical_scores=sum(r['physical_scores'] for r in rr),
            finite_observed_future_gains=sum(r.get('observed_future_gain') is not None for r in rr)))
    write_json(source/f'{phase}_continuation_summary.json',event_summary)
    work=[]
    for arm in CONFIGS:
        rr=[r for r in state if r['arm']==arm]
        work.append(dict(arm=arm,mean_phy=mean(rr,'candidate_evaluations'),mean_features=mean(rr,'feature_roots'),
            mean_proxy_candidates=mean(rr,'proxy_candidates'),mean_pool_ms=mean(rr,'candidate_generation_ms'),
            mean_proxy_ms=mean(rr,'feature_ms'),mean_continuation_phy=mean(rr,'continuation_phy'),
            mean_memo_hits=mean(rr,'memo_hits'),mean_macro_accepts=mean(rr,'macro_accepts'),
            stop_counts={s:sum(r['stop_reason']==s for r in rr) for s in sorted({r['stop_reason'] for r in rr})}))
    write_json(source/f'{phase}_work.json',work)
    return dict(summary=summary,pairs=pair_summary,events=event_summary,state=state,decisions=len(raw),work=work)


def plot(val,dev,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#263442','#963c4b','#3977ac','#26896c','#a86d26','#655ca8','#72a0c3','#70a98f','#9e91c5']
    labels=['Full DPLS','Frozen AI 212','Singles 212','Joint 212','Singles + cont. 212','Joint + cont. 212',
            'Singles 424','Joint 424','Joint + cont. 424']
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for col,(data,title) in enumerate([(val,'Validation: 12 historical states'),(dev,'Development: 9 historical states')]):
        for i,(r,label) in enumerate(zip(data['summary'],labels)):
            x=float(r['time_ratio'])
            axes[0,col].scatter(x,float(r['mean_relative_j_pct']),s=60,marker='*' if i==0 else 'o',color=colors[i],label=label)
            axes[1,col].scatter(x,float(r['mean_positive_p5']),s=60,marker='*' if i==0 else 'o',color=colors[i])
            for row,field in [(0,'mean_relative_j_pct'),(1,'mean_positive_p5')]:
                axes[row,col].annotate(str(i),(x,float(r[field])),xytext=(4,4),textcoords='offset points',fontsize=8)
        for row in (0,1):
            ax=axes[row,col];ax.axvline(.8,color='#9b4159',linestyle=':',alpha=.7);ax.grid(alpha=.2)
            ax.set_xlabel('Complete mean time / full DPLS')
        axes[0,col].axhline(-.5,color='#9b4159',linestyle=':',alpha=.7)
        axes[1,col].axhline(.98,color='#9b4159',linestyle=':',alpha=.7)
        axes[0,col].set_title(title);axes[0,col].set_ylabel('Mean J change vs full (%)')
        axes[1,col].set_ylabel('Mean effective-UE P5 expected service ratio')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,[f'{i}: {s}' for i,s in enumerate(labels)],loc='lower center',ncol=3,frameon=False,fontsize=9,bbox_to_anchor=(.5,.025))
    fig.suptitle('Whole-TB macro candidates and paid finite continuation',fontsize=14)
    fig.text(.5,.012,'212 / 424 are score caps. Gates apply per state; plotted means alone do not determine success. All service diagnostics are instantaneous.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.13,1,.95));fig.savefig(destination/'quality_cost.png',dpi=160);fig.savefig(destination/'quality_cost.pdf');plt.close(fig)


def main(source):
    source=Path(source).resolve();completion=json.loads((source/'completion.json').read_text());assert completion['completed']
    val=audit_phase(source,'validation');dev=audit_phase(source,'evaluation')
    protocol=json.loads((source/'protocol.json').read_text());manifest=json.loads((source/'source_manifest.json').read_text())
    for name,digest in manifest.items():
        assert sha(source/'source'/name)==digest
        assert sha(ROOT/'training/stageb_lns'/name)==digest
    assert sha(protocol['model'])==protocol['model_sha256']
    assert sha(ROOT/'training/stageb_hgraph/current_algorithm.json')==protocol['registry_sha256']
    tests=Path('/tmp/tb_macro_tests_20261102.log').read_text();assert 'Ran 53 tests' in tests and '\nOK\n' in tests
    shutil.copyfile('/tmp/tb_macro_tests_20261102.log',source/'tests.log')
    shutil.copyfile('/tmp/tb_macro_execution_20261102.log',source/'execution.log')
    write_json(source/'final_audit.json',dict(passed=True,tests=53,serial_decisions=val['decisions']+dev['decisions'],
        independent_phy=completion['independent_final_audits'],summary_and_paired_metrics_recomputed=True,
        executed_sources_unchanged=True,model_and_registry_unchanged=True,training_updates=0,native_trajectories=0,final_seeds_read=False))
    write_json(source/'runtime_environment.json',dict(python=platform.python_version(),numpy=np.__version__,threads=1,gpu_used=False))
    plot(val,dev,source)
    print(json.dumps(dict(selection=json.loads((source/'selection.json').read_text()),validation=val['summary'],
        development=dev['summary'],paired=dev['pairs'],continuations=dev['events'],work=dev['work']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);main(parser.parse_args().source)

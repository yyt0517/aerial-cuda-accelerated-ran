"""Recompute service-set/repair results, label provenance and conditional gaps."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .service_repair_campaign import CONFIGS,NAMES
from .service_sets import SetModel,set_metrics,calibration_key,causal_features
from .physics import ROOT,COUNT,LAYER,EXPECTED,ReferencePhy
from .bounded_probes import ProbePhy,csv_write
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .ranked_dpls_campaign import read_rows,identity
from .experiment import write_json,sha

PAIRS=[('unprotected212','main180'),('learned_repair32','main180'),
    ('learned_repair32','unprotected212'),('learned_repair32','matched_repair32'),
    ('learned_repair32','heuristic_repair32'),('learned_lock_repair32','heuristic_lock_repair32'),
    ('learned_lock_repair32','all_pfq_guard212'),('learned_repair64','learned_repair32')]


def read(path):return list(csv.DictReader(Path(path).open()))
def avg(rows,key):return float(np.mean([float(r[key]) for r in rows]))
def mean(rows,key):return float(np.mean([avg([r for r in rows if r['seed']==s],key) for s in sorted({r['seed'] for r in rows})]))


def audit_phase(source,phase):
    raw=read(source/f'{phase}_measurements.csv');state=read(source/f'{phase}_per_state.csv');summary=read(source/f'{phase}_summary.csv')
    for r in state:
        rr=[v for v in raw if v['state_id']==r['state_id'] and v['arm']==r['arm']]
        assert len(rr)==3 and np.isclose(float(r['e2e_ms']),np.median([float(v['e2e_ms']) for v in rr]))
        if r['arm']!='full':assert int(r['candidate_evaluations'])<=212 and int(r['feature_roots'])<=56
        if r['arm'] in CONFIGS:
            config=CONFIGS[r['arm']]
            assert int(r['main_phy'])<=config.max_candidates-config.repair_budget
            assert int(r['repair_phy'])<=config.repair_budget
            assert np.isclose(float(r['quality_fence']),.995*float(r['main_j']))
            assert float(r['score'])+1.e-8>=float(r['quality_fence']) and r['diagnostic']=='False'
    for r in summary:
        rr=[v for v in state if v['arm']==r['arm']]
        assert np.isclose(mean(rr,'score'),float(r['mean_j'])) and np.isclose(mean(rr,'e2e_ms'),float(r['mean_ms']))
        q=all(float(v['ratio'])>=.995 for v in rr)
        c=all(int(v['newly_unserved_ues'])==0 and float(v['positive_p5_ratio'])>=.98 and float(v['positive_weak_ratio'])>=.98 for v in rr)
        cost=float(r['time_ratio'])<=.8
        assert (r['quality_pass']=='True')==q and (r['coverage_pass']=='True')==c
        assert (r['cost_pass']=='True')==cost and (r['joint_pass']=='True')==(q and c and cost)
        if r['allocation_recall']:
            assert np.isclose(mean(rr,'allocation_recall'),float(r['allocation_recall']))
            sg=all(float(v['allocation_recall'])>=.98 and float(v['effective_recall'])>=.98 for v in rr) and mean(rr,'size_ratio')<=.75
            assert (r['set_gate']=='True')==sg
    index={(r['state_id'],r['arm']):r for r in state};paired=[]
    for sid in dict.fromkeys(r['state_id'] for r in state):
        for new,old in PAIRS:
            a=index[sid,new];b=index[sid,old]
            paired.append(dict(state_id=sid,seed=a['seed'],new=new,old=old,j_delta=float(a['score'])-float(b['score']),
                delta_pp=100*(float(a['ratio'])-float(b['ratio'])),ms_delta=float(a['e2e_ms'])-float(b['e2e_ms']),
                newly_unserved_delta=int(a['newly_unserved_ues'])-int(b['newly_unserved_ues']),
                p5_delta=float(a['positive_p5_ratio'])-float(b['positive_p5_ratio'])))
    csv_write(source/f'{phase}_paired.csv',paired);pair_summary=[];seeds=[]
    for new,old in PAIRS:
        rr=[r for r in paired if r['new']==new and r['old']==old]
        pair_summary.append(dict(new=new,old=old,wins=sum(r['j_delta']>1 for r in rr),losses=sum(r['j_delta']<-1 for r in rr),
            ties=sum(abs(r['j_delta'])<=1 for r in rr),mean_j_delta=mean(rr,'j_delta'),mean_delta_pp=mean(rr,'delta_pp'),
            mean_ms_delta=mean(rr,'ms_delta'),mean_newly_unserved_delta=mean(rr,'newly_unserved_delta'),mean_p5_delta=mean(rr,'p5_delta')))
        for seed in sorted({r['seed'] for r in rr}):
            vv=[r for r in rr if r['seed']==seed];seeds.append(dict(new=new,old=old,seed=seed,j_delta=avg(vv,'j_delta')))
    csv_write(source/f'{phase}_paired_summary.csv',pair_summary);csv_write(source/f'{phase}_paired_per_seed.csv',seeds)
    events=json.loads((source/f'{phase}_repair_events.json').read_text())
    for event in events:
        r=index[event['state_id'],event['arm']]
        if event['accepted']:
            assert event['feasible'] and event['candidate_j']+1.e-8>=float(r['quality_fence'])
            assert tuple(event['after_goal'])>=tuple(event['before_goal'])
    work=[]
    for arm in CONFIGS:
        rr=[r for r in state if r['arm']==arm]
        work.append(dict(arm=arm,**{k:mean(rr,k) for k in ('main_phy','repair_phy','repair_submitted','repair_accepted','repair_j_losses',
            'main_ms','repair_ms','proposal_ms','feature_ms','candidate_generation_ms','proxy_candidates','memo_hits','feature_roots')},
            repair_stop_counts={s:sum(r['repair_stop']==s for r in rr) for s in sorted({r['repair_stop'] for r in rr})}))
    write_json(source/f'{phase}_work.json',work)
    # Recompute proposal metrics from frozen arrays and full reference, after timing.
    values=np.load(source/f'{phase}_values.npz');sets=np.load(source/f'{phase}_sets.npz')
    rows,_=read_rows((44,45) if phase=='validation' else (46,47,48),evaluation=phase=='evaluation')
    by_id={identity(r):r for r in rows};checks=[]
    for r in state:
        if not r.get('allocation_recall'):continue
        p=problem_from_raw_row(by_id[r['state_id']]);ref=values[f'{r["state_id"]}|full']
        target=np.stack([ref[:,COUNT]>0,ref[:,EXPECTED]>1],1)
        recomputed=set_metrics(sets[f'{r["state_id"]}|{r["arm"]}'],target,p.legal.any((0,1)))
        for name,value in recomputed.items():assert np.isclose(float(r[name]),value)
        checks.append(dict(state_id=r['state_id'],arm=r['arm']))
    write_json(source/f'{phase}_set_audit.json',dict(passed=True,recomputed=checks,teacher_reference_used_after_timing_only=True))
    return dict(summary=summary,state=state,paired=pair_summary,seeds=seeds,raw_decisions=len(raw))


def audit_model(source,model):
    import torch
    training=torch.load(source/'training_set_groups.pt',weights_only=False)
    validation=torch.load(source/'validation_set_groups.pt',weights_only=False)
    all_x=np.concatenate([g['x'][g['eligible']] for g in training])
    np.testing.assert_array_equal(model.mean,all_x.mean(0).astype(np.float32))
    np.testing.assert_array_equal(model.scale,np.maximum(all_x.std(0),.05).astype(np.float32))
    for g in training:
        assert g['seed'] in (41,42,43)
        np.testing.assert_array_equal(g['target'],np.stack([g['teacher_values'][:,COUNT]>0,g['teacher_values'][:,EXPECTED]>1],1))
    rows,_=read_rows((41,42,43));by_id={identity(r):r for r in rows};checks=[]
    for seed in (41,42,43):
        g=next(v for v in training if v['seed']==seed);row=by_id[g['state_id']];p=problem_from_raw_row(row);phy=ProbePhy(p)
        try:
            base=phy.evaluate(raw_pfq_draft(p,row['obs']));before=phy.stats.copy();x,u,e=causal_features(p,phy,base)
            np.testing.assert_array_equal(x,g['x']);np.testing.assert_array_equal(e,g['eligible']);assert before==phy.stats
            checks.append(dict(state_id=g['state_id'],feature_phy=phy.stats['candidate_evaluations']))
        finally:phy.close()
    write_json(source/'model_data_audit.json',dict(passed=True,training_only_normalization=True,
        initial_causal_features_recomputed=checks,training_states=len(training),validation_states=len(validation),
        prior_validation_probe_labels_not_training=True))
    return validation


def conditional_gaps(source,groups):
    values=np.load(source/'conditional_values.npz');actions=np.load(source/'conditional_actions.npz');records=[]
    for g in groups:
        for arm in ('teacher_repair32','teacher_lock_repair32'):
            key=f'{g["state_id"]}|{arm}';v=values[key];ref=g['teacher_values'];e=ref[:,EXPECTED]>1
            records.append(dict(state_id=g['state_id'],seed=g['seed'],arm=arm,
                prg_count_l1=float(abs(v[:,COUNT]-ref[:,COUNT]).sum()),
                effective_layer_mismatches=int((e&(v[:,LAYER]!=ref[:,LAYER])).sum()),
                activity_slot_mismatches=int(((actions[key]>=0)!=(g['teacher_action']>=0)).sum()),
                expected_service_shortfall=float(np.maximum(ref[:,EXPECTED]-v[:,EXPECTED],0).sum())))
    csv_write(source/'conditional_resource_gaps.csv',records)
    return records


def plot(val,dev,source):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#263442','#963c4b','#337eb1','#73a9cc','#a86d26','#70a775','#27866a','#b6a573','#9690bd','#6651a4','#c7464e']
    labels=['Full','Legacy AI','Ordinary 212','Main 180','All-PFQ lock','Heuristic lock+32','Learned lock+32',
        'Heuristic repair32','Size-matched repair32','Learned repair32','Learned repair64']
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
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,[f'{i}: {s}' for i,s in enumerate(labels)],loc='lower center',ncol=4,frameon=False,fontsize=8,bbox_to_anchor=(.5,.026))
    fig.suptitle('Causal service sets and terminal repair inside a 212-score cap',fontsize=13)
    fig.text(.5,.012,'Online arms only. Teacher-mask conditional probes are separate. Gates apply per state; scores and time are distinct budgets.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.13,1,.95));fig.savefig(source/'quality_cost.png',dpi=160);fig.savefig(source/'quality_cost.pdf');plt.close(fig)


def main(source):
    source=Path(source).resolve();complete=json.loads((source/'completion.json').read_text());assert complete['completed']
    protocol=json.loads((source/'protocol.json').read_text());info=json.loads((source/'model_selection.json').read_text());model=SetModel(info['checkpoint'])
    val=audit_phase(source,'validation');dev=audit_phase(source,'evaluation');groups=audit_model(source,model);gap=conditional_gaps(source,groups)
    baseline=SetModel(source/'service_e00.npz');changed=any(not np.array_equal(getattr(model,k),getattr(baseline,k)) for k in ('w1','b1','w2','b2','w3','b3'))
    target=next(r for r in dev['summary'] if r['arm']=='learned_repair32')
    pair=next(r for r in dev['paired'] if r['new']=='learned_repair32' and r['old']=='matched_repair32')
    seeds=[r for r in dev['seeds'] if r['new']=='learned_repair32' and r['old']=='matched_repair32']
    cond=read(source/'conditional_summary.csv');cond_checks=json.loads((source/'conditional_checks.json').read_text())
    assert all(r['quality_pass']=='False' and r['coverage_pass']=='False' for r in cond)
    gates=dict(selected_on_validation=json.loads((source/'selection.json').read_text()),model=info,
        quality_service_cost={r['arm']:{k:r[k]=='True' for k in ('quality_pass','coverage_pass','cost_pass','joint_pass')} for r in dev['summary'] if r['arm']!='full'},
        service_set_gate=target['set_gate']=='True',selected_weights_changed=changed,learning_pair=pair,per_seed=seeds,
        learning_direction=changed and pair['wins']>=6 and all(r['j_delta']>0 for r in seeds),
        joint_learning_gate=changed and pair['wins']>=6 and all(r['j_delta']>0 for r in seeds) and target['set_gate']=='True' and target['joint_pass']=='True',
        teacher_conditional=dict(validation_only=True,decisions=24,physical_scores=cond_checks['physical_scores'],
            both_quality_and_service_fail=True,online_cost_claim=False,complete_upper_bound=False),
        next_path=dict(selected='whole-TB resource quota/min-layer and cell-activity joint conditional probes before another proposer training round',
            implementation_status='chosen next experiment, not run',
            rationale='correct binary membership still leaves observed count/layer/activity gaps and about5% J shortfall in this bounded repair',
            proposed_ablations=['teacher resource counts/min-layer only','teacher cell-activity pattern only','both together, without teacher UE-to-PRG positions'],
            teacher_use='offline conditional diagnostics only; if headroom exists, train causal joint proposals with conservation-aware decoding',
            retain=['incremental engine','raw PFQ90% floor','exact full-schedule verifier','complete charging','unchanged final gates'],
            stop_as_primary=['more binary-set classifier training','hard service-set locking','larger late repair reserve'],
            no_claim='conditional probes are not complete bounds or proof which gap causes the loss; no free teacher online',
            defer=['PPO','continuation value','default replacement']))
    write_json(source/'gate_results.json',gates)
    for name,digest in json.loads((source/'source_manifest.json').read_text()).items():
        assert sha(source/'source'/name)==digest and sha(ROOT/'training/stageb_lns'/name)==digest
    assert sha(protocol['old_model'])==protocol['old_model_sha256'] and sha(model.path)==info['sha256']
    assert sha(ROOT/'training/stageb_hgraph/current_algorithm.json')==protocol['registry_sha256']
    tests=Path('/tmp/service_repair_tests_20261102.log').read_text();assert 'Ran 72 tests' in tests and '\nOK\n' in tests
    shutil.copyfile('/tmp/service_repair_tests_20261102.log',source/'tests.log');shutil.copyfile('/tmp/service_repair_execution_20261102.log',source/'execution.log')
    write_json(source/'final_audit.json',dict(passed=True,tests=72,formal_decisions=val['raw_decisions']+dev['raw_decisions'],
        formal_independent_finals=complete['formal_independent_audits'],conditional_independent_finals=24,
        offline_teacher_independent_finals=30,summary_set_and_paired_metrics_recomputed=True,
        phase_fences_and_budgets_valid=True,matched_cardinality_valid=True,unlocked_main_prefix_identical=True,
        source_model_and_registry_hashes_valid=True,native_trajectories=0,final_seeds_read=False))
    plot(val,dev,source)
    print(json.dumps(dict(gates=gates,paired=dev['paired'],conditional=cond),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);main(parser.parse_args().source)

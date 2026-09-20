"""Fixed-reference snapshots and paired reports for the reproduction campaign."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import heapq
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import numpy as np
import torch

from .dpa_ablation_experiment import METRICS, read_csv, sha, write_json
from .ctbs_headroom import compare_ctbs_reference_labels
from .edge_generator import EdgeGeneratorConfig, SparseEdgeGenerator
from .feature_snapshot import FeatureSnapshotBuilder


def uniform_snapshot_tti(tti):
    return 1001 <= tti <= 4000 and (tti-1001)%30==0


def aggregate_differences(matrix):
    means=[statistics.mean(row) for row in matrix]
    return {'paired_differences':matrix,'training_seed_means':means,'mean':statistics.mean(means),
            'n_training_replicates':len(means),'range':[min(means),max(means)],
            'design':'crossed training seeds and scene seeds; not independent per-packet replicates'}


def endpoint_assessment(full_rows,ablated_rows,arm):
    """Descriptive pre-specified effect/protection screen, not a significance test."""
    low_expiry=all(r['expiry_drop_rate']<.001 for matrix in (full_rows,ablated_rows) for row in matrix for r in row)
    metric=('successful_bytes_per_active_prg' if arm=='no_reuse_attention' else
            'expiry_drop_rate' if arm=='no_reward' and not low_expiry else 'ue_expiry_aware_rate_p10_mbps')
    threshold=.0005 if metric=='expiry_drop_rate' else (.03 if arm=='no_reuse_attention' else .05)
    effects=[];violations=[]
    for i,(full_row,ablated_row) in enumerate(zip(full_rows,ablated_rows)):
        row=[]
        for j,(f,a) in enumerate(zip(full_row,ablated_row)):
            if metric=='expiry_drop_rate':row.append(a[metric]-f[metric])
            else:row.append((f[metric]-a[metric])/a[metric] if a[metric]>0 else None)
            checks={'goodput':f['goodput_mbps']>=.99*a['goodput_mbps'],
                    'expiry':f['expiry_drop_rate']<=a['expiry_drop_rate']+.0005,
                    'mean_delay':f['outcome_delay_mean_ms']<=a['outcome_delay_mean_ms']+max(.05*a['outcome_delay_mean_ms'],.5),
                    'p95_delay':f['outcome_delay_p95_ms']<=a['outcome_delay_p95_ms']+max(.05*a['outcome_delay_p95_ms'],.5),
                    'ue_p10':f['ue_expiry_aware_rate_p10_mbps']>=.95*a['ue_expiry_aware_rate_p10_mbps'],
                    'backlog':f['end_backlog_bytes']<=a['end_backlog_bytes']+.01*a['offered_bytes']}
            if not all(checks.values()):violations.append({'training_index':i,'scene_index':j,'failed':[k for k,v in checks.items() if not v]})
        effects.append(row)
    summary=aggregate_differences(effects) if all(x is not None for row in effects for x in row) else None
    effect_met=summary is not None and summary['mean']>=threshold
    same_direction=summary is not None and all(x>0 for x in summary['training_seed_means'])
    return {'metric':metric,'units':'absolute rate reduction' if metric=='expiry_drop_rate' else 'relative gain',
            'reward_low_expiry_fallback':arm=='no_reward' and low_expiry,'effect':summary,'threshold':threshold,
            'engineering_effect_met':effect_met,'positive_for_all_training_seeds':same_direction,
            'protection_passed_all_pairs':not violations,'protection_violations':violations,
            'classification':('performance_tradeoff' if effect_met and violations else
                              'consistent_descriptive_gain' if effect_met and same_direction else 'no_stable_engineering_gain'),
            'statistical_significance_established':False}


class ReferenceCapture:
    """PFQ-only, pre-action sampling. Keeps 100 regular and 10 pressure records."""
    def __init__(self, directory, client):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=False)
        self.builder=FeatureSnapshotBuilder()
        self.edges=SparseEdgeGenerator(EdgeGeneratorConfig(
            candidate_mode='all_demand_ues',include_pfq_score_anchor=False,
            include_candidate_risk_features=False))
        self.runner=SimpleNamespace(env=client,args=SimpleNamespace(independent_headroom_accounting='ctbs-v1'))
        self.uniform={}; self.pressure=[]; self._pending=None
        self.ctbs_audit={'steps':0,'mcs_mismatch_count':0,'layer_mismatch_count':0,
                         'tbs_mismatch_count':0,'reference_label_deletion_equal':True}

    def observe(self, tti, obs, info, action_ue, action_prg):
        from .online_ppo import _ctbs_inputs_for_rollout_step
        # Request every TTI, including warmup, so optional collection has a fixed handshake schedule.
        inputs=_ctbs_inputs_for_rollout_step(self.runner,info)
        graph=self.edges.build_graph(self.builder.build(obs))
        self._pending=(int(tti),inputs,graph,np.array(action_ue,copy=True))
        if not 1001<=tti<=4000: return
        features=np.asarray(obs['obs_ue_features'])
        pressure=float(np.max(features[:,12]))
        regular=uniform_snapshot_tti(tti)
        high=len(self.pressure)<10 or (pressure,-tti)>self.pressure[0][:2]
        if not (regular or high): return
        record={'tti':tti,'bridge_tti':int(info['tti']),'graph':graph,'ctbs_inputs':inputs,
                'teacher_action_ue':np.array(action_ue,copy=True),'teacher_action_prg':np.array(action_prg,copy=True),
                'pressure':pressure,'ici_proxy':float(graph.snapshot.prg_features.values[:,7].mean()),
                'reference':'native PFQ','timing':'pre-action; no future PHY labels'}
        if regular:self.uniform[tti]=record
        if high:
            item=(pressure,-tti,record)
            if len(self.pressure)<10:heapq.heappush(self.pressure,item)
            else:heapq.heapreplace(self.pressure,item)

    def after_step(self, action_ue, action_prg, next_info):
        if self._pending is None:
            raise RuntimeError('mechanism capture post-step has no matching pre-action state')
        tti,inputs,graph,pre_action_ue=self._pending; self._pending=None
        if not np.array_equal(np.asarray(action_ue),pre_action_ue):
            raise RuntimeError('mechanism capture pre/post action UE mapping changed')
        required=('actual_ue_mcs','actual_ue_layer','actual_ue_tbs_capacity_bytes')
        missing=[key for key in required if key not in next_info]
        if missing:raise RuntimeError('mechanism CTBS post-action labels missing: '+', '.join(missing))
        slots=np.asarray(action_prg); selected=np.asarray(action_ue)
        expected=(graph.snapshot.static_meta.n_prg,graph.snapshot.static_meta.n_cell)
        if slots.size!=expected[0]*expected[1]:
            raise RuntimeError('native PFQ slot action has unexpected PRG*cell length')
        slots=slots.reshape(expected)
        chosen=np.full(slots.shape,-1,dtype=np.int64)
        valid=slots>=0
        if np.any(slots[valid]>=selected.shape[0]):raise RuntimeError('native PFQ slot index out of range')
        chosen[valid]=selected[slots[valid]]
        parity=compare_ctbs_reference_labels(
            inputs,chosen_ue_per_prg=chosen.transpose().reshape(-1).tolist(),
            actual_mcs=next_info['actual_ue_mcs'],actual_layer=next_info['actual_ue_layer'],
            actual_tbs_capacity_bytes=next_info['actual_ue_tbs_capacity_bytes'],
            n_cell=int(graph.snapshot.static_meta.n_cell),n_prg=int(graph.snapshot.static_meta.n_prg))
        self.ctbs_audit['steps']+=1
        for key in ('mcs_mismatch_count','layer_mismatch_count','tbs_mismatch_count'):
            self.ctbs_audit[key]+=int(getattr(parity,key))
        self.ctbs_audit['reference_label_deletion_equal'] &= bool(parity.reference_label_deletion_equal)
        if not parity.all_exact:
            raise RuntimeError('mechanism CTBS causal PHY/TBS parity failed: '
                               f'mcs={parity.mcs_mismatch_count} layer={parity.layer_mismatch_count} '
                               f'tbs={parity.tbs_mismatch_count} '
                               f'deletion={int(parity.reference_label_deletion_equal)}')

    def finalize(self):
        if self._pending is not None:raise ValueError('mechanism capture lacks final post-action parity')
        if self.ctbs_audit['steps']!=4000:raise ValueError('mechanism CTBS audit does not cover 4000 TTI')
        if len(self.uniform)!=100: raise ValueError('reference trajectory lacks 100 uniform snapshots')
        selected=dict(self.uniform)
        pressure_ttis={item[2]['tti'] for item in self.pressure}
        selected.update({item[2]['tti']:item[2] for item in self.pressure})
        records=[]
        for tti,record in sorted(selected.items()):
            path=self.directory/f'tti_{tti:04d}.pt'; torch.save(record,path)
            records.append({'tti':tti,'file':path.name,'sha256':sha(path),
                            'uniform':tti in self.uniform,'pressure_supplement':tti in pressure_ttis,
                            'pressure':record['pressure'],'ici_proxy':record['ici_proxy']})
        ici=[r['ici_proxy'] for r in self.uniform.values()]
        manifest={'records':records,'uniform_count':100,'pressure_count':10,'deduplicated_count':len(records),
                  'pressure_rule':'max UE pre-action log1p drain pressure; earliest TTI breaks ties',
                  'graph_schema':{'candidate_mode':'all_demand_ues','include_pfq_score_anchor':False,
                                  'include_candidate_risk_features':False,'ue_prg_edge_feature_dim':5},
                  'ctbs_post_action_audit':self.ctbs_audit,
                  'ici_quantiles_30_70':np.percentile(ici,[30,70]).tolist(),
                  'stratification_population':'fixed uniformly sampled PFQ states; pressure supplements separate'}
        write_json(self.directory/'manifest.json',manifest)


def _model(path,device,ablation=None):
    from .eval_checkpoint_online import _load_actor_checkpoint
    actor,_=_load_actor_checkpoint(str(path),device)
    if ablation: actor.cfg=replace(actor.cfg,ablation=ablation)
    actor.eval(); return actor


def _conditional_outputs(actor,graph,packed,trace):
    from .independent_policy import _run_policy
    replay=_run_policy(actor,graph,packed,mode='greedy',stored_trace=trace,collect_numerical_trace=True)
    valid=replay.valid_at_step
    logits=replay.raw_logits_at_step
    if not torch.equal(valid,trace.valid_at_step.to(valid.device)):
        raise ValueError('same-prefix mask changed')
    return logits,logits.masked_fill(~valid,-torch.inf).argmax(-1),valid


def compare_snapshot(record,actors):
    from .online_ppo import _graph_to_device
    from .graph_ops import pack_prg_candidates
    from .independent_policy import _run_policy,resolve_full_candidate_width,select_independent_action
    device=next(actors['801/full'].parameters()).device
    graph=_graph_to_device(record['graph'],device)
    packed=pack_prg_candidates(graph,max_candidates=resolve_full_candidate_width(graph),fixed_width=True)
    base=actors['801/full']; prefix='PFQ'
    with torch.no_grad():
        try:
            trace=_run_policy(base,graph,packed,mode='greedy',
                              teacher_actions=(record['teacher_action_ue'],record['teacher_action_prg']),
                              ctbs_headroom_inputs=record['ctbs_inputs'])
        except ValueError as e:
            if 'teacher' not in str(e): raise
            prefix='Full801 fallback: '+str(e)
            trace=_run_policy(base,graph,packed,mode='greedy',ctbs_headroom_inputs=record['ctbs_inputs'])
        outputs={name:_conditional_outputs(actor,graph,packed,trace) for name,actor in actors.items()}
        independent={}
        for name,actor in actors.items():
            own,decoded=select_independent_action(actor,graph,packed,mode='greedy',collect_diagnostics=True,
                                                  ctbs_headroom_inputs=record['ctbs_inputs'])
            diag=own.ctbs_diagnostics
            predicted=[0.0]*graph.snapshot.static_meta.n_ue
            if diag is not None:
                for ue,selected,capacity in zip(diag.selected_ue.tolist(),diag.selected_nonblank.tolist(),
                                                 diag.cumulative_tbs_after_bytes.tolist()):
                    if selected and ue>=0:predicted[ue]=max(predicted[ue],float(capacity))
            independent[name]={'legal_joint_chosen_ue_matrix':decoded.chosen_ue_per_prg.reshape(
                                  graph.snapshot.static_meta.n_cell,graph.snapshot.static_meta.n_prg).cpu().tolist(),
                               'predicted_ctbs_service_bytes_by_ue':predicted,
                               'prediction_not_actual_phy':True}
    comparisons={}
    for name,(logits,choice,valid) in outputs.items():
        seed=name.split('/')[0]; reference=outputs[seed+'/full']
        delta=(choice!=reference[1])
        c,p=graph.snapshot.static_meta.n_cell,graph.snapshot.static_meta.n_prg
        ids=packed.ue_ids.reshape(c,p,-1).permute(1,0,2)
        ids=torch.cat((ids,torch.full((*ids.shape[:2],1),-1,device=ids.device,dtype=ids.dtype)),dim=-1)
        before=ids.gather(2,reference[1].unsqueeze(-1)).squeeze(-1)
        after=ids.gather(2,choice.unsqueeze(-1)).squeeze(-1)
        blank=packed.blank_class_index
        comparisons[name]={'effective_logit_max_abs_difference':float((logits-reference[0])[valid].abs().max()),
                           'conditional_argmax_changed_fraction':float(delta.float().mean()),
                           'conditional_blank_changed_fraction':float(((choice==blank)!=(reference[1]==blank)).float().mean()),
                           'empty_prefix_first_group_changed_fraction':float(delta[0].float().mean()),
                           'involved_ue_ids':sorted(x for x in set(before[delta].tolist()+after[delta].tolist()) if x>=0),
                           'conditional_ue_matrix':after.transpose(0,1).cpu().tolist()}
    return {'prefix':prefix,'comparisons':comparisons,'independently_decoded':independent,
            'matrix_semantics':'each conditional choice uses the same reference prefix; not a jointly executed schedule',
            'causal_service_claim':False}


def finish_campaign(campaign,models,results):
    root=campaign.root; campaign.status['phase']='P4'; campaign.progress('P4/paired_summary')
    paired={}
    metrics=(*METRICS,'max_backlogged_zero_dequeue_streak_tti',
             'max_backlogged_schedulable_zero_dequeue_streak_tti')
    for condition in ('standard','deadline_pressure'):
        paired[condition]={}
        for arm in ('no_reuse_attention','no_dimensionless','no_reward'):
            paired[condition][arm]={}
            for metric in metrics:
                matrix=[]; relative=[]
                for train_seed in (801,802,803):
                    row=[]; rr=[]
                    for scene in (211,212,213,214,215):
                        full=results[f'test/{condition}/{scene}/{train_seed}/full'][metric]
                        ablated=results[f'test/{condition}/{scene}/{train_seed}/{arm}'][metric]
                        row.append(full-ablated); rr.append((full-ablated)/ablated if ablated else None)
                    matrix.append(row); relative.append(rr)
                paired[condition][arm][metric]={**aggregate_differences(matrix),'relative_paired':relative,
                    'relative_summary':aggregate_differences(relative) if all(x is not None for r in relative for x in r) else None}
    write_json(root/'paired_results.json',paired)
    endpoints={}
    for arm in ('no_reuse_attention','no_dimensionless','no_reward'):
        condition='standard' if arm=='no_reuse_attention' else 'deadline_pressure'
        full=[[results[f'test/{condition}/{scene}/{seed}/full'] for scene in range(211,216)] for seed in (801,802,803)]
        ablated=[[results[f'test/{condition}/{scene}/{seed}/{arm}'] for scene in range(211,216)] for seed in (801,802,803)]
        endpoints[arm]={'condition':condition,**endpoint_assessment(full,ablated,arm)}
    write_json(root/'primary_endpoints.json',endpoints)
    actors={name:_model(Path(m['checkpoint']),torch.device('cuda')) for name,m in models.items()}
    for seed in (801,802,803):
        for ablation in ('no_reuse_attention','no_dimensionless'):
            actors[f'{seed}/frozen_{ablation}']=_model(Path(models[f'{seed}/full']['checkpoint']),torch.device('cuda'),ablation)
    rows=[]; cases={}
    for condition in ('standard','deadline_pressure'):
        for scene in (211,212,213,214,215):
            campaign.progress(f'P4/mechanisms/{condition}/{scene}')
            directory=root/f'test/{condition}/{scene}/pfq/snapshots'
            manifest=json.loads((directory/'manifest.json').read_text())
            lo,hi=manifest['ici_quantiles_30_70']
            for entry in manifest['records']:
                path=directory/entry['file']
                if sha(path)!=entry['sha256']:raise ValueError('reference snapshot changed')
                record=torch.load(path,map_location='cpu',weights_only=False)
                comparison=compare_snapshot(record,actors)
                row={'condition':condition,'scene_seed':scene,**entry,**comparison,
                     'interference_stratum':'high' if entry['ici_proxy']>=hi else ('low' if entry['ici_proxy']<=lo else 'middle')}
                rows.append(row)
                for component,eligible in (('no_reuse_attention',entry['ici_proxy']>=hi),('no_reward',entry['pressure_supplement'])):
                    if component not in cases and eligible and comparison['comparisons'][f'801/{component}']['conditional_argmax_changed_fraction']>0:
                        cases[component]={**row,'ue_features':record['graph'].snapshot.ue_features.values.tolist(),
                                          'ue_feature_names':record['graph'].snapshot.ue_features.names}
            write_json(root/'mechanism_results_partial.json',{'rows':rows,'cases':cases})
    write_json(root/'mechanism_results.json',{'rows':rows,'cases':cases,'counterfactual_phy_executed':False,
                'interpretation':'conditional decision sensitivity, not actual or long-term causal service benefit'})
    mechanism_summary={}
    for condition in ('standard','deadline_pressure'):
        for population in ('uniform','pressure_supplement'):
            for stratum in ('all','low','high'):
                for prefix in ('PFQ','Full801 fallback'):
                    for name in actors:
                        by_scene=[]
                        for scene in range(211,216):
                            selected=[r['comparisons'][name] for r in rows if r['condition']==condition and
                                      r['scene_seed']==scene and r[population] and r['prefix'].startswith(prefix) and
                                      (stratum=='all' or r['interference_stratum']==stratum)]
                            if selected:by_scene.append({'scene':scene,'n_states':len(selected),
                                **{key:statistics.mean(r[key] for r in selected) for key in
                                   ('conditional_argmax_changed_fraction','conditional_blank_changed_fraction',
                                    'empty_prefix_first_group_changed_fraction','effective_logit_max_abs_difference')}})
                        mechanism_summary[f'{condition}/{population}/{stratum}/{prefix}/{name}']=by_scene
    write_json(root/'mechanism_summary.json',{'groups':mechanism_summary,
                 'note':'uniform and pressure supplements are separate, potentially overlapping populations; state counts are not independent training replicates'})
    # Measured inference latency on exactly the same fixed graph, device, precision and batch.
    from .graph_ops import pack_prg_candidates
    from .independent_policy import select_independent_action,resolve_full_candidate_width
    from .online_ppo import _graph_to_device
    graph=_graph_to_device(record['graph'],torch.device('cuda'))
    packed=pack_prg_candidates(graph,max_candidates=resolve_full_candidate_width(graph),fixed_width=True)
    timings={}
    with torch.no_grad():
        for name,actor in actors.items():
            for _ in range(3):select_independent_action(actor,graph,packed,mode='greedy',ctbs_headroom_inputs=record['ctbs_inputs'])
            torch.cuda.synchronize(); t=time.perf_counter()
            for _ in range(10):select_independent_action(actor,graph,packed,mode='greedy',ctbs_headroom_inputs=record['ctbs_inputs'])
            torch.cuda.synchronize();timings[name]=(time.perf_counter()-t)*100
    write_json(root/'inference_ms.json',{'mean_ms_per_tti':timings,'device':torch.cuda.get_device_name(),
                 'batch':1,'precision':'float32','warmup':3,'repetitions':10})
    _plots_and_report(root,models,results,paired,cases)
    write_json(root/'final_report.json',{'status':'complete','models':models,'runs':results,'paired':paired,
         'mechanisms':'mechanism_results.json','inference':'inference_ms.json','primary_endpoints':endpoints,
         'limitations':['three training replicates','conditional logits are not executed counterfactual service',
                        'reward removal includes weak-UE fairness shaping','no graph/CTBS/sequential-decoder ablation']})
    campaign.status.update(status='complete',current_job=None)


def _plots_and_report(root,models,results,paired,cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    endpoints=json.loads((root/'primary_endpoints.json').read_text())
    colors={'full':'black','no_reuse_attention':'tab:blue','no_dimensionless':'tab:orange','no_reward':'tab:green'}
    refs61=json.loads((root/'p0_reference_results.json').read_text())['references']['61']
    learning_numeric={}
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for name,model in models.items():
        xs=[];ys=[[],[],[]]; numeric=[]
        for stage,offset in (('S0',0),('S1',0),('S2',2048),('S3',53248)):
            for row in model['stages'][stage]['candidates']:
                xs.append(offset+row['iteration']*512)
                gate=row.get('gate',acceptance(row['metrics'],refs61['old_dpa'],refs61['pfq']))
                numeric.append({'stage':stage,'iteration':row['iteration'],'effective_transition':xs[-1],
                                'expiry_drop_rate':row['metrics']['expiry_drop_rate'],
                                'end_backlog_bytes':row['metrics']['end_backlog_bytes'],'gate_passed':gate['passed']})
                for j,m in enumerate(('outcome_delay_p95_ms','ue_expiry_aware_rate_p10_mbps','goodput_mbps')):ys[j].append(row['metrics'][m])
        arm=name.split('/')[1]
        for ax,y in zip(axes,ys):ax.plot(xs,y,color=colors[arm],alpha=.5,label=name)
        learning_numeric[name]=numeric
    write_json(root/'learning_gate_curves.json',learning_numeric)
    for ax,title in zip(axes,('Argmax P95 (ms)','Argmax UE P10 (Mbps)','Argmax goodput (Mbps)')):
        for x in (2048,53248):ax.axvline(x,color='gray',linestyle=':')
        ax.set(title=title,xlabel='PPO effective transitions (BC at zero)')
    axes[-1].legend(fontsize=6);fig.tight_layout();fig.savefig(root/'learning_curves.png');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for i,condition in enumerate(('standard','deadline_pressure')):
        for j,arm in enumerate(('no_reuse_attention','no_dimensionless','no_reward')):
            ax=axes[i,j]
            metric=endpoints[arm]['metric']
            values=paired[condition][arm][metric]['training_seed_means']
            ax.scatter([801,802,803],values)
            ax.axhline(0,color='gray');ax.set(title=condition+' / '+arm,xticks=[801,802,803],ylabel=metric+' (Full minus ablation)')
    fig.tight_layout();fig.savefig(root/'paired_endpoints.png');plt.close(fig)
    if cases:
        fig,axes=plt.subplots(len(cases),2,figsize=(12,3*len(cases)),squeeze=False)
        for i,(arm,case) in enumerate(cases.items()):
            for j,name in enumerate(('801/full','801/'+arm)):
                axes[i,j].imshow(case['independently_decoded'][name]['legal_joint_chosen_ue_matrix'],aspect='auto',vmin=-1,vmax=35)
                names=case['ue_feature_names']; features=np.asarray(case['ue_features'])
                details=[]
                for ue in case['comparisons'][name]['involved_ue_ids']:
                    details.append(f'UE{ue}: q={features[ue,names.index("buffer_bytes")]:.0f}, hol={features[ue,names.index("hol_delay_ms")]:.1f}, ttl={features[ue,names.index("ttl_slack_ms")]:.1f}')
                axes[i,j].set(title=f'{name}: seed{case["scene_seed"]}, TTI{case["tti"]}\n'+('; '.join(details[:3])),xlabel='local PRG',ylabel='cell')
        fig.suptitle('Legal independently decoded schedules; CTBS service is predicted, not actual PHY');fig.tight_layout();fig.savefig(root/'conditional_cases.png');plt.close(fig)
    timings=json.loads((root/'inference_ms.json').read_text())['mean_ms_per_tti']
    lines=['# DPA 训练复现与组件消融结果','',
           '保留全部正式运行。配对差值为 Full − 消融；时延/过期越低越好。',
           '训练重复单位为 3 个 seed；5 个场景是交叉测试，不按 15 次独立训练解释。',
           '奖励删除也移除了 expiry-aware 弱 UE 公平性塑形，不能将差异全部归因于期限信息。',
           '', '## 预先指定的主终点', '']
    for arm,row in endpoints.items():
        lines.append(f'- {arm} / {row["condition"]}：{row["metric"]}；描述性分类 `{row["classification"]}`；工程阈值达到={row["engineering_effect_met"]}，三训练 seed 同向={row["positive_for_all_training_seeds"]}，全部配对保护门通过={row["protection_passed_all_pairs"]}。')
    lines+=['', '以上为工程效应与保护门的描述性筛查，不是统计显著性声明。详见 [主终点](primary_endpoints.json) 和 [完整配对结果](paired_results.json)。',
           '','## Table 1: all service outcomes','',
           '| run | goodput Mbps | BLER | expired bytes / offered bytes | mean ms | P95 ms | UE P10 Mbps | PRG util | Blank | efficiency | backlog bytes |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in results.items():lines.append(f'| {name} | {m["goodput_mbps"]:.4f} | {m["tb_bler"]:.6f} | {m["expired_bytes"]} / {m["offered_bytes"]} ({100*m["expiry_drop_rate"]:.5f}%) | {m["outcome_delay_mean_ms"]:.4f} | {m["outcome_delay_p95_ms"]:.4f} | {m["ue_expiry_aware_rate_p10_mbps"]:.4f} | {m["prg_utilization_ratio"]:.6f} | {1-m["prg_utilization_ratio"]:.6f} | {m["successful_bytes_per_active_prg"]:.3f} | {m["end_backlog_bytes"]} |')
    lines+=['','## Table 2: lifecycle budgets','',
            '| model | charged PPO samples | actual optimizer updates | final selection qualified | inference ms/TTI |',
            '|---|---:|---:|---|---:|']
    for name,model in models.items():
        updates=0
        gates=[]
        for stage in ('S1','S2','S3'):
            path=Path(model['stages'][stage]['selected_checkpoint'])
            folder=path.parent.parent if path.parent.name=='candidate_checkpoints' else path.parent
            updates+=sum(int(float(r['num_updates'])) for r in read_csv(folder/'online_ppo_metrics.csv'))
        for stage in ('S0','S1','S2','S3'):
            gates.extend((stage,r['iteration'],bool(r.get('gate',acceptance(r['metrics'],refs61['old_dpa'],refs61['pfq']))['passed']))
                         for r in model['stages'][stage]['candidates'])
        qualified=sum(g[2] for g in gates); first=next((f'{s}@{i}' for s,i,p in gates if p),'none')
        first_index=next((i for i,g in enumerate(gates) if g[2]),None)
        persists=first_index is not None and all(g[2] for g in gates[first_index:])
        lines.append(f'| {name} | 57344 | {updates} | {model["stages"]["S3"]["selected"]["gate"]["passed"]} ({qualified}/32; first={first}; all later={persists}) | {timings[name]:.3f} |')
    lines+=['','## Interpretation limits','',
            'Use paired_results.json to inspect all three seed directions and engineering effect sizes; no significance claim is inferred from one favorable mean.',
            'Inspect goodput, expiry, delay, weak-UE and backlog costs before attributing an efficiency gain.',
            'Frozen-delete comparisons are in mechanism_results.json and do not substitute for retrained component comparisons.',
            'Case selection used the first qualifying fixed-reference state; missing cases mean no qualifying divergence was found.',
            'No simulator state clone was used. Conditional action matrices are not executed counterfactual schedules.',
            'Figures: learning_curves.png, paired_endpoints.png, conditional_cases.png (when a qualifying case exists).']
    (root/'report.md').write_text('\n'.join(lines)+'\n')

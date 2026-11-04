"""Full-cohort attribution, paired seed intervals and predeclared contrasts."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t, ttest_1samp

from common import ARMS, OUT, REPO, SEEDS, load_file, path_for, sha, write
from validate import validate

METRICS = ['complete_payload_mbps','goodput_mbps','complete_success_pct','expiry_pct',
    'success_delay_mean_ms','success_delay_p95_ms','outcome_mean_ms','outcome_p99_ms',
    'ue_complete_payload_p5_mbps','ue_goodput_p5_mbps','ue_goodput_jain','tb_bler_pct','prg_utilization']
LABELS = {'pfq':'PFQ','ctbs':'PFQ+CTBS','raw_global_equal':'无CTBS·全局·等权',
    'raw_global_deficit':'无CTBS·全局·缺口权重','ctbs_global_equal':'CTBS·全局·等权',
    'full':'完整DPLS','ctbs_independent_equal':'CTBS·独立小区·等权',
    'ctbs_independent_deficit':'CTBS·独立小区·缺口权重'}


def summary(values):
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    n=len(values);mean=float(values.mean())
    sd=float(values.std(ddof=1)) if n>1 else float('nan')
    margin=float(student_t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else float('nan')
    if n>1 and sd==0:
        p=1. if mean==0 else 0.
    else:
        p=float(ttest_1samp(values,0).pvalue) if n>1 else float('nan')
    return dict(n=n,mean=mean,sample_sd=sd,ci95_low=mean-margin,ci95_high=mean+margin,
                positive_seeds=int((values>0).sum()),negative_seeds=int((values<0).sum()),p_nominal=p)


def extract_one(arm,seed):
    (OUT/'checks').mkdir(exist_ok=True)
    validation=validate(arm,seed)
    helper=load_file('ablation_long_metrics',REPO/'deliverables/boundary08_steady_state_20261021/analyze.py')
    path=path_for(arm,seed)
    execution=json.loads((path/'execution.json').read_text())
    launch=json.loads((path/'launch.json').read_text())
    # Keep native arm identity for the extractor's strict provenance checks.
    captured={}
    original=helper.old.packets_from_fifo
    def capture(service,offered,outcomes,total):
        packets=original(service,offered,outcomes,total)
        captured.update(service=service,packets=packets)
        return packets
    helper.old.packets_from_fifo=capture
    rows,ue,stability,provenance=helper.extract(path,seed,execution['arm'],launch['source_sha256'],
        json.loads((OUT/'protocol.json').read_text())['simulator_sha256'])
    for row in rows:row['arm']=arm
    for row in ue:row['arm']=arm
    stability['arm']=arm;provenance['arm']=arm
    packets=captured['packets']
    cohort=packets[packets.arrival_tti.between(8001,28000)].copy()
    assert not cohort.censored.any()
    cohort.to_csv(OUT/'packet_cohorts'/f'{arm}_seed{seed}.csv.gz',index=False,
        compression=dict(method='gzip',compresslevel=3,mtime=0))
    per=pd.DataFrame(ue).set_index('ue_id')
    for label,data in [('success_delay',cohort[cohort.successfully_delivered]),('outcome',cohort)]:
        group=data.groupby('ue_id').system_time_ms
        for quantile in [.5,.95,.99]:per[f'{label}_p{int(quantile*100)}_ms']=group.quantile(quantile)
    d=pd.read_csv(path/'decisions.csv').query('8001<=tti<=28000')
    if 'candidate_evaluations' in d:
        candidate_mean=float(d.candidate_evaluations.mean())
    elif ARMS[arm]['search']!='none':
        candidate_mean=float(pd.read_csv(path/'rounds.csv').query('8001<=tti<=28000').groupby('tti').candidates.sum().mean())
    else:candidate_mean=0.
    cost=dict(arm=arm,seed=seed,mean_shared_host_decision_ms=float(d.decision_wall_s.mean()*1000),
        mean_candidates=candidate_mean,mean_rounds=float(d.rounds_evaluated.mean()) if 'rounds_evaluated' in d else 0.,
        native_elapsed_s=execution['elapsed_native_s'])
    if not ARMS[arm]['reused']:
        audit=pd.read_csv(path/'alignment_audit.csv').query('8001<=tti<=28000')
        cost.update(mean_ctbs_pruned_prgs=float(audit.ctbs_pruned_prgs.mean()),
            draft_vs_raw_dequeue_differing_decisions=int(audit.draft_dequeue.ne(audit.raw_pfq_dequeue).sum()),
            mean_draft_minus_raw_dequeue=float((audit.draft_dequeue-audit.raw_pfq_dequeue).mean()))
    (OUT/'analysis_parts'/f'{arm}_seed{seed}.json').write_text(json.dumps(dict(rows=rows,ue=per.reset_index().to_dict('records'),
        stability=stability,provenance=provenance,cost=cost),ensure_ascii=False,allow_nan=True)+'\n')
    print('COHORT_ANALYZED',arm,seed,rows[0]['complete_payload_mbps'],flush=True)
    return arm,seed


def attribution(frame,protocol):
    indexed=frame.set_index(['seed','arm'])
    contrasts=[];per_seed=[]
    for added,reference in protocol['contrasts']:
        for metric in METRICS:
            values=[]
            for seed in SEEDS:
                a=float(indexed.loc[(seed,added),metric]);b=float(indexed.loc[(seed,reference),metric])
                delta=a-b;values.append(delta)
                per_seed.append(dict(seed=seed,added=added,reference=reference,metric=metric,
                    added_value=a,reference_value=b,delta=delta))
            row=dict(added=added,reference=reference,metric=metric,**summary(values))
            row['gain_of_means_pct']=float(100*(frame[frame.arm==added][metric].mean()/frame[frame.arm==reference][metric].mean()-1)) if frame[frame.arm==reference][metric].mean()!=0 else np.nan
            contrasts.append(row)
    contrasts=pd.DataFrame(contrasts)
    contrasts['p_holm_primary']=np.nan
    primary=contrasts[contrasts.metric=='complete_payload_mbps'].sort_values('p_nominal')
    adjusted=np.maximum.accumulate(primary.p_nominal.to_numpy()*(len(primary)-np.arange(len(primary))))
    contrasts.loc[primary.index,'p_holm_primary']=np.minimum(1,adjusted)
    pd.DataFrame(per_seed).to_csv(OUT/'contrast_per_seed.csv',index=False)
    contrasts.to_csv(OUT/'contrasts.csv',index=False)
    def arm_for(enabled):
        c,g,w=[key in enabled for key in ['CTBS','GLOBAL_SEARCH','DEFICIT']]
        if not g:return 'ctbs' if c else 'pfq'
        if c:return 'full' if w else 'ctbs_global_equal'
        return 'raw_global_deficit' if w else 'raw_global_equal'
    shapley=[];interactions=[]
    factors=['CTBS','GLOBAL_SEARCH','DEFICIT']
    for seed in SEEDS:
        for metric in METRICS:
            values={arm:float(indexed.loc[(seed,arm),metric]) for arm in ARMS}
            contributions=dict.fromkeys(factors,0.)
            for order in itertools.permutations(factors):
                enabled=set()
                for factor in order:
                    before=values[arm_for(enabled)]
                    enabled.add(factor)
                    contributions[factor]+=(values[arm_for(enabled)]-before)/6
            total=values['full']-values['pfq']
            np.testing.assert_allclose(sum(contributions.values()),total,rtol=0,atol=1e-9)
            for factor,contribution in contributions.items():
                shapley.append(dict(seed=seed,metric=metric,factor=factor,contribution=contribution,total_gain=total))
            interactions.append(dict(seed=seed,metric=metric,
                ctbs_x_search_equal=(values['ctbs_global_equal']-values['raw_global_equal'])-(values['ctbs']-values['pfq']),
                ctbs_x_deficit_under_search=(values['full']-values['ctbs_global_equal'])-(values['raw_global_deficit']-values['raw_global_equal']),
                deficit_given_no_ctbs_search=values['raw_global_deficit']-values['raw_global_equal'],
                deficit_given_ctbs_search=values['full']-values['ctbs_global_equal']))
    shapley=pd.DataFrame(shapley);shapley.to_csv(OUT/'shapley_per_seed.csv',index=False)
    result=[]
    for (metric,factor),group in shapley.groupby(['metric','factor'],sort=False):
        row=dict(metric=metric,factor=factor,**summary(group.contribution))
        row['share_of_mean_total_gain_pct']=float(100*group.contribution.mean()/group.total_gain.mean()) if group.total_gain.mean()!=0 else np.nan
        result.append(row)
    pd.DataFrame(result).to_csv(OUT/'shapley_summary.csv',index=False)
    interactions=pd.DataFrame(interactions);interactions.to_csv(OUT/'interaction_per_seed.csv',index=False)
    result=[]
    for metric,group in interactions.groupby('metric',sort=False):
        for name in interactions.columns.drop(['seed','metric']):
            result.append(dict(metric=metric,interaction=name,**summary(group[name])))
    pd.DataFrame(result).to_csv(OUT/'interaction_summary.csv',index=False)


def make_report(frame,stability):
    primary=pd.read_csv(OUT/'contrasts.csv').query("metric=='complete_payload_mbps'")
    shapley=pd.read_csv(OUT/'shapley_summary.csv').query("metric=='complete_payload_mbps'")
    lines=['DPLS逐项消融：CTBS、跨小区协调与服务缺口权重','2026-10-29',
        '', '实验完成：boundary/rate0.8，5 seed（41–45），8组；30条新增原生轨迹＋10条已核验参照。',
        '每条4000TTI共同PFQ预热＋4000稳定化＋20000测量（10秒）＋400随访。按seed等权；完整到达包cohort无截尾。',
        '', '全部组结果（均值±seed样本标准差）：',
        'arm | 完整包payload Mbps | 分段goodput Mbps | 整包成功% | TTL过期% | 成功平均ms | 成功P95ms | 全包终态均值ms | UE完整包P5 Mbps']
    table_metrics=['complete_payload_mbps','goodput_mbps','complete_success_pct','expiry_pct','success_delay_mean_ms','success_delay_p95_ms','outcome_mean_ms','ue_complete_payload_p5_mbps']
    for arm in ARMS:
        group=frame[frame.arm==arm]
        lines.append(LABELS[arm]+' | '+' | '.join(f'{group[m].mean():.4f}±{group[m].std(ddof=1):.4f}' for m in table_metrics))
    lines+=['','逐项增量/移除：完整包payload Mbps；区间为名义95%配对t区间；p为11项主要对照的Holm校正。']
    for _,r in primary.iterrows():
        lines.append(f'{LABELS[r.added]} − {LABELS[r.reference]}: {r["mean"]:+.4f} [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}], 正向{r.positive_seeds}/5, Holm p={r.p_holm_primary:.6g}')
    def contrast(a,b):return primary[(primary.added==a)&(primary.reference==b)].iloc[0]
    ctbs=contrast('ctbs','pfq');local=contrast('ctbs_independent_equal','ctbs')
    coordination=contrast('ctbs_global_equal','ctbs_independent_equal');weights=contrast('full','ctbs_global_equal')
    lines+=['','同一条解释路径（每一步都是独立闭环策略的均值差，可相加还原完整DPLS−PFQ）：',
        f'PFQ → CTBS: {ctbs["mean"]:+.4f} Mbps；',
        f'CTBS → 小区独立等权物理优化: {local["mean"]:+.4f} Mbps；',
        f'独立等权 → 全局等权协调搜索: {coordination["mean"]:+.4f} Mbps；',
        f'全局等权 → 缺口加权完整DPLS: {weights["mean"]:+.4f} Mbps。',
        '该路径的贡献依赖加入顺序；全局搜索差异包括邻居收益、联合编辑和共享出队预算，不单独归因某一项。',
        '', '三模块Shapley分摊（遍历6种加入顺序，包含模块交互）：']
    for _,r in shapley.iterrows():
        lines.append(f'{r.factor}: {r["mean"]:+.4f} Mbps [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}]；占均值总增益 {r.share_of_mean_total_gain_pct:.2f}%')
    largest=shapley.sort_values('mean',ascending=False).iloc[0]
    lines += [f'按这一事前声明的分摊约定，平均贡献最大为 {largest.factor}。此处GLOBAL_SEARCH包含物理优化；纯协调的额外收益应看上面的独立小区对照。',
        'Shapley比例是分摊约定，不是唯一因果贡献；可为负或超过100%，不能强行裁剪。',
        '', '缺口权重的收益/代价（完整DPLS−CTBS全局等权；以下为探索性多指标，区间未作多重比较校正）：']
    all_contrasts=pd.read_csv(OUT/'contrasts.csv')
    for metric in ['complete_payload_mbps','ue_complete_payload_p5_mbps','ue_goodput_jain','expiry_pct','success_delay_mean_ms','outcome_mean_ms']:
        r=all_contrasts[(all_contrasts.added=='full')&(all_contrasts.reference=='ctbs_global_equal')&(all_contrasts.metric==metric)].iloc[0]
        lines.append(f'{metric}: {r["mean"]:+.6f} [{r.ci95_low:+.6f}, {r.ci95_high:+.6f}]')
    weak=pd.read_csv(OUT/'ue_effects_per_seed.csv')
    sub=weak[(weak.added=='full')&(weak.reference=='ctbs_global_equal')]
    lines.append('各seed中完整包payload下降超过2%的UE数量（full相对全局等权）：'+', '.join(f'{int(r.seed)}:{int(r.payload_dropped_gt2pct_ues)}/36' for _,r in sub.iterrows()))
    lines += [
        '', '解释边界：',
        '1. CTBS关闭时仍保持相同90%因果下界规则，但草案改变可能使下界值改变；逐状态差异见search_costs.csv。不能把CTBS作用限定为一个纯物理算子。',
        '2. 缺口权重没有搜索时不改变PFQ或CTBS，所以2×2×2逻辑组合折叠为6个不同策略；独立小区对照另加两组。',
        '3. 不同模块会改变后续队列、OLLA/MCS和EMA；同seed只要求外生信道/流量相同，不要求分叉后的完整状态相同。',
        '4. 平均时延仅成功包条件分布；须联合完整成功率、TTL、全包终态和UE指标，不能只看其中一项。',
        '5. seed已参与开发，不是新盲测；本轮只针对boundary，不能直接推广为uniform的贡献比例。',
        '6. 保留冻结legacy BLER查表和全部局部邻域，不改有效算法注册表。共享主机并发耗时仅作描述，不是部署实时性或公平速度比较。',
        f'7. 有限窗口稳定性诊断通过 {int(stability.practical_stability_passed.sum())}/{len(stability)} 条；未通过者保留且不称稳态；TTL可通过丢包稳定队列。',
        '8. 置信区间以seed为单位；包、UE和1秒子窗均不作为独立重复。主要payload对照作Holm校正，其余多指标为探索性。',
        '', '证据：protocol.json、preflight.json、native_campaign_complete.json、validation.json；',
        'per_seed_metrics.csv、per_ue_metrics.csv、window_metrics.csv、stability.csv、contrasts.csv、contrast_per_seed.csv；',
        'shapley_summary.csv、shapley_per_seed.csv、interaction_summary.csv、interaction_per_seed.csv、search_costs.csv；',
        'matched_ue_deltas.csv、ue_effects_per_seed.csv；',
        'checks/、packet_cohorts/、native/、logs/、comparison.png/pdf、manifest.json。']
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')


def ue_effects(per,protocol):
    per=per.set_index(['seed','arm','ue_id'])
    details=[];summaries=[]
    for added,reference in protocol['contrasts']:
        for seed in SEEDS:
            a=per.loc[(seed,added)].sort_index()
            b=per.loc[(seed,reference)].sort_index()
            np.testing.assert_array_equal(a.index,b.index)
            assert len(a)==36
            relative=100*(a.complete_payload_mbps-b.complete_payload_mbps)/b.complete_payload_mbps.where(b.complete_payload_mbps>0)
            for ue in a.index:
                row=dict(seed=seed,added=added,reference=reference,ue_id=int(ue),
                    relative_complete_payload_pct=float(relative.loc[ue]))
                for metric in ['complete_payload_mbps','fixed_window_goodput_mbps','success_delay_mean_ms','queue_failure_pct','outcome_mean_ms','complete_success_fraction']:
                    row[metric+'_added']=float(a.loc[ue,metric])
                    row[metric+'_reference']=float(b.loc[ue,metric])
                    row[metric+'_delta']=float(a.loc[ue,metric]-b.loc[ue,metric])
                details.append(row)
            summaries.append(dict(seed=seed,added=added,reference=reference,
                payload_improved_ues=int(a.complete_payload_mbps.gt(b.complete_payload_mbps).sum()),
                payload_dropped_ues=int(a.complete_payload_mbps.lt(b.complete_payload_mbps).sum()),
                payload_dropped_gt2pct_ues=int(relative.lt(-2).sum()),
                relative_payload_defined_ues=int(relative.notna().sum()),
                worst_relative_payload_pct=float(relative.min()),
                success_delay_improved_ues=int(a.success_delay_mean_ms.lt(b.success_delay_mean_ms).sum()),
                success_delay_worsened_ues=int(a.success_delay_mean_ms.gt(b.success_delay_mean_ms).sum())))
    pd.DataFrame(details).to_csv(OUT/'matched_ue_deltas.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'ue_effects_per_seed.csv',index=False)


def plot(frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['PFQ','CTBS','Raw/global/equal','Raw/global/deficit','CTBS/global/equal','Full DPLS','CTBS/independent/equal','CTBS/independent/deficit']
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    for ax,metric,title in zip(axes.ravel(),['complete_payload_mbps','success_delay_mean_ms','expiry_pct','ue_complete_payload_p5_mbps'],['Complete packet payload (Mbps)','Successful packet mean delay (ms)','Packet expiry (%)','UE complete payload P5 (Mbps)']):
        means=[frame[frame.arm==arm][metric].mean() for arm in ARMS]
        sds=[frame[frame.arm==arm][metric].std(ddof=1) for arm in ARMS]
        ax.barh(np.arange(len(ARMS)),means,xerr=sds,color=['#a5acb4' if arm!='full' else '#167d8d' for arm in ARMS],capsize=3)
        ax.set_yticks(np.arange(len(ARMS)),names,fontsize=8);ax.invert_yaxis()
        ax.set_title(title);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Boundary component ablation | 5 paired seeds | 10 s measurement\nBars: seed mean; whiskers: sample SD',fontsize=13)
    fig.savefig(OUT/'comparison.png',dpi=160);fig.savefig(OUT/'comparison.pdf');plt.close(fig)


def main():
    assert json.loads((OUT/'native_campaign_complete.json').read_text())['passed']
    protocol=json.loads((OUT/'protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():assert sha(OUT/name)==digest
    assert sha(REPO/'training/stageb_hgraph/current_algorithm.json')==protocol['registry_sha256']
    for reference in protocol['reference'].values():
        for name,digest in reference['files'].items():assert sha(Path(reference['path'])/name)==digest
    for name in ['analysis_parts','packet_cohorts','checks']:(OUT/name).mkdir(exist_ok=True)
    jobs=[(arm,seed) for arm in ARMS for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=5) as pool:
        tasks=[pool.submit(extract_one,*job) for job in jobs]
        for task in as_completed(tasks):task.result()
    parts=[json.loads((OUT/'analysis_parts'/f'{arm}_seed{seed}.json').read_text()) for arm,seed in jobs]
    rows=pd.DataFrame([r for part in parts for r in part['rows']])
    frame=rows[rows.window=='measurement'].copy()
    assert len(frame)==40 and not frame.duplicated(['seed','arm']).any()
    frame.to_csv(OUT/'per_seed_metrics.csv',index=False)
    rows[rows.window!='measurement'].to_csv(OUT/'window_metrics.csv',index=False)
    per=pd.DataFrame([r for part in parts for r in part['ue']])
    per.to_csv(OUT/'per_ue_metrics.csv',index=False)
    stability=pd.DataFrame([part['stability'] for part in parts]);stability.to_csv(OUT/'stability.csv',index=False)
    pd.DataFrame([part['cost'] for part in parts]).to_csv(OUT/'search_costs.csv',index=False)
    frame.groupby('arm',sort=False)[METRICS].agg(['mean','std']).to_csv(OUT/'mean_std.csv')
    for seed in SEEDS:
        group=[part['provenance'] for part in parts if part['provenance']['seed']==seed]
        for key in ['first4000_service_sha256','mature_cohort_identity_sha256']:
            assert len({p[key] for p in group})==1
    # Check reused reports against freshly reconstructed raw metrics.
    old=pd.read_csv(REPO/'deliverables/boundary08_rho090_five_seed_20261027/per_seed_metrics.csv')
    for arm,oldarm in [('pfq','pfq0'),('full','current_rho090')]:
        for seed in SEEDS:
            actual=frame[(frame.arm==arm)&(frame.seed==seed)].iloc[0]
            expected=old[(old.arm==oldarm)&(old.seed==seed)].iloc[0]
            for metric in METRICS:np.testing.assert_allclose(actual[metric],expected[metric],rtol=0,atol=1e-9)
    attribution(frame,protocol)
    ue_effects(per,protocol)
    make_report(frame,stability);plot(frame)
    validation=dict(passed=True,all_40_trajectories_checked=True,new_runs=30,reused_runs=10,
        no_censoring=True,matched_cohort_and_warmup_all_seeds=True,primary_contrasts=11,
        current_registry_unchanged=True,source_protocol_unchanged=True,
        practical_stability_passes=int(stability.practical_stability_passed.sum()),
        sources=[part['provenance'] for part in parts])
    write(OUT/'validation.json',validation)
    files=[p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json']
    write(OUT/'manifest.json',dict(files={p.name:sha(p) for p in files},
        protocol_sha256=sha(OUT/'protocol.json'),report_sha256=sha(OUT/'report.txt')))
    print('ABLATION_COMPLETE_VALIDATED',flush=True)


if __name__=='__main__':main()

"""Read-only follow-up: delivery protection versus latency under deficit weights."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[1]
SOURCE=REPO/'Doc/evidence/2026-10-29/dpls_component_ablation'
COHORTS=REPO/'deliverables/dpls_component_ablation_20261029_v1/packet_cohorts'
EQUAL='ctbs_global_equal'
WEIGHTED='full'
SEEDS=list(range(41,46))
METRICS=['goodput_mbps','complete_payload_mbps','complete_success_pct','expiry_pct',
 'residual_crc_failure_pct','total_failure_pct','success_delay_mean_ms','success_delay_p50_ms',
 'success_delay_p95_ms','success_delay_p99_ms','outcome_mean_ms','outcome_p95_ms','outcome_p99_ms',
 'success_within_1ms_pct','success_within_5ms_pct','ue_goodput_p5_mbps','ue_goodput_p10_mbps',
 'ue_complete_payload_p5_mbps','ue_complete_payload_p10_mbps','ue_goodput_jain',
 'min_ue_complete_success_pct','worst_ue_queue_failure_pct','mean_backlog_bytes',
 'prg_utilization','successful_bytes_per_prg','tb_bler_pct','sinr_tx_mean_db','mcs_tx_mean',
 'layer_tx_mean','tx_samples','mac_dequeue_mbps','failed_byte_mbps',
 'ue_min_complete_payload_mbps','ue_complete_payload_jain']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired(a,b):
    delta=np.asarray(a)-np.asarray(b)
    assert len(delta)==5 and np.isfinite(delta).all()
    mean=float(delta.mean());sd=float(delta.std(ddof=1));half=float(t.ppf(.975,4)*sd/np.sqrt(5))
    am,bm=float(np.mean(a)),float(np.mean(b))
    return dict(equal_mean=bm,weighted_mean=am,mean_delta=mean,delta_sd=sd,
        ci95_low=mean-half,ci95_high=mean+half,
        relative_of_means_pct=100*(am/bm-1) if bm else None,
        positive_seeds=int((delta>1e-9).sum()),negative_seeds=int((delta< -1e-9).sum()))


def report(summary,weak,ue_changes,cdf,common,counts,cost):
    s=summary.set_index('metric')
    lines=['缺口权重是否值得保留：详细对比','2026-10-29',
        '', '判断：有保留价值，主要是保护最差UE的完整交付、降低TTL过期；不宜把它视为所有业务都默认启用的无代价增强。',
        '若优先平均/P95时延，当前等权全局搜索更合适；若优先避免严重欠服务、减少200ms TTL过期，缺口加权有明确收益。',
        '本轮只读取已完成实验，不运行新策略、不调参、不更改默认算法。',
        '', '比较对象：同一CTBS＋全局物理搜索、同一rho=.90下界规则，仅将全1权重换成因果服务缺口权重。',
        'boundary/rate0.8、seeds41–45；每seed共同PFQ预热4000、稳定化4000、测量20000TTI=10秒、随访400TTI。',
        '以下为逐seed等权均值；差值=加权−等权，95%配对t区间以seed为单位。除了原预声明payload结果，本轮多指标均为事后探索、区间未作多重比较校正。',
        '', '指标 | 等权 | 缺口加权 | 差值 | 95%配对区间']
    for metric in METRICS:
        r=s.loc[metric]
        lines.append(f'{metric} | {r.equal_mean:.6f} | {r.weighted_mean:.6f} | {r.mean_delta:+.6f} | [{r.ci95_low:+.6f}, {r.ci95_high:+.6f}]')
    lines+=['','少数严重欠服务UE的改善：',
        '系统P5吞吐近似不变，不能推断最差一个UE也没有受益。36个UE的5%分位位于最差与第三差UE之间，可能遗漏单个极端落后者的恢复。',
        '各seed按等权组整包成功率选原最差UE，再匹配同一UE（此选择为事后描述，不是预声明盲测亚组）：',
        'seed | UE | 整包成功率 等权→加权 % | TTL过期 等权→加权 % | 完整包吞吐 等权→加权 Mbps | 成功均值 等权→加权 ms']
    for _,r in weak.iterrows():
        lines.append(f'{int(r.seed)} | {int(r.ue_id)} | {r.success_equal_pct:.3f}→{r.success_weighted_pct:.3f} | {r.ttl_equal_pct:.3f}→{r.ttl_weighted_pct:.3f} | {r.payload_equal_mbps:.4f}→{r.payload_weighted_mbps:.4f} | {r.delay_equal_ms:.3f}→{r.delay_weighted_ms:.3f}')
    lines += [f'原最差UE平均吞吐增量 {weak.payload_delta_mbps.mean():.6f} Mbps；系统净增量 {s.loc["complete_payload_mbps","mean_delta"]:.6f} Mbps。最差UE恢复是主要收益来源，其余UE总体有小幅资源让出。',
        '', '180个(seed,UE)配对的变化计数（描述性；不能当180次独立重复）：']
    for _,r in ue_changes.iterrows():
        lines.append(f'{r.metric}: 增加{int(r.increased)}、减少{int(r.decreased)}、不变{int(r.unchanged)}')
    lines += ['', f'两个策略相同成熟到达包总数 {counts["cohort_packets"]}；加权净增加完整成功 {counts["success_delta"]}，净减少TTL过期 {-counts["expiry_delta"]}，但CRC残留整包失败净增加 {counts["crc_failure_delta"]}。TTL减少不是全部转化为完整成功。',
        'CRC残留整包失败与TB BLER不同：前者以完整到达包为分母，后者以已发送TB为分母；本轮TB BLER变化区间跨零。',
        '', '同一到达包在截止时间前完成完整交付的比例（所有到达包作分母，包含失败）：',
        'deadline ms | 等权 % | 加权 % | 加权−等权 pp | 95%配对区间']
    for ms,g in cdf.groupby('deadline_ms',sort=True):
        equal=g.sort_values('seed').equal_pct.to_numpy();weighted=g.sort_values('seed').weighted_pct.to_numpy()
        r=paired(weighted,equal)
        lines.append(f'{ms:g} | {r["equal_mean"]:.4f} | {r["weighted_mean"]:.4f} | {r["mean_delta"]:+.4f} | [{r["ci95_low"]:+.4f}, {r["ci95_high"]:+.4f}]')
    lines += ['', '共同成功包的配对时延（同一包在两臂均完整成功，仍是事后条件子集，不是随机因果主层）：',
        'seed | 共同成功包数 | 等权平均 ms | 加权平均 ms | 差值 ms']
    for _,r in common.iterrows():
        lines.append(f'{int(r.seed)} | {int(r.common_success)} | {r.common_equal_mean_ms:.6f} | {r.common_weighted_mean_ms:.6f} | {r.common_mean_delta_ms:+.6f}')
    r=paired(common.common_weighted_mean_ms,common.common_equal_mean_ms)
    lines.append(f'逐seed等权配对均值差 {r["mean_delta"]:+.6f}ms，区间[{r["ci95_low"]:+.6f},{r["ci95_high"]:+.6f}]。全包终态均值也变差；时延损失不能完全用成功包组成变化解释。')
    lines += ['', '资源与计算：']
    for _,r in cost.iterrows():
        lines.append(f'{r.metric}: 等权{r.equal_mean:.6f}，加权{r.weighted_mean:.6f}，变化{r.relative_of_means_pct:+.3f}%')
    lines += ['共享主机/MPS并发环境及复用参照计时不同，不能用上述墙钟比较宣称更快或更慢；候选数/轮数是实际算法工作量。',
        '', '机制解释与保留建议：',
        '权重=clip(2400/max(成功服务EMA,1e-9),1,4)，再均值归一化；EMA=.995*旧值+.005*实际成功字节。它优先补偿历史服务不足，没有直接优化包年龄或平均时延。',
        '本轮最差UE明显恢复，其他UE很多只有很小吞吐下降，但等待成本更广泛；PRG占用和总积压上升、每PRG成功字节下降，与上述取舍相容。这些观察不是唯一机制证明。',
        '建议保留实现、开关和论文消融，定位为交付保障/低过期选项；以平均/P95时延为优先时建议等权作为候选默认。当前任务不切换任何默认。',
        '未来可研究仅对严重缺口UE触发、减弱权重或加入年龄保护；这些都是待验证假设，不能据本轮宣称已找到兼顾方案。',
        '不应用本次开发seed反复调参后再称最终盲测；311–315未使用。结果限于此boundary负载与TTL、关闭HARQ及冻结历史BLER评分口径。',
        '', '证据：metric_comparison.csv、per_seed_selected.csv、weak_ue_pairs.csv、ue_change_counts.csv、deadline_delivery.csv、common_success.csv、outcome_transitions.csv、resource_search_cost.csv、checks.json、manifest.json；数据来自上一轮冻结消融，未新增轨迹。']
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')


def main():
    assert not (OUT/'report.txt').exists(),'Refuse overwrite; use a new output version.'
    source_manifest=json.loads((SOURCE/'archive_manifest.json').read_text())
    used=['per_seed_metrics.csv','per_ue_metrics.csv','contrasts.csv','search_costs.csv','validation.json','protocol.json']
    for name in used:assert sha(SOURCE/name)==source_manifest['files'][name],name
    assert json.loads((SOURCE/'validation.json').read_text())['passed']
    x=pd.read_csv(SOURCE/'per_seed_metrics.csv')
    a=x[x.arm==WEIGHTED].set_index('seed').sort_index();b=x[x.arm==EQUAL].set_index('seed').sort_index()
    u=pd.read_csv(SOURCE/'per_ue_metrics.csv')
    ua=u[u.arm==WEIGHTED].set_index(['seed','ue_id']).sort_index()
    ub=u[u.arm==EQUAL].set_index(['seed','ue_id']).sort_index()
    np.testing.assert_array_equal(ua.index,ub.index)
    for frame,arm in [(a,WEIGHTED),(b,EQUAL)]:
        for seed in SEEDS:
            q=u[u.arm.eq(arm)&u.seed.eq(seed)].complete_payload_mbps.to_numpy()
            frame.loc[seed,'ue_min_complete_payload_mbps']=q.min()
            frame.loc[seed,'ue_complete_payload_jain']=q.sum()**2/(36*(q*q).sum())
    summary=pd.DataFrame([dict(metric=m,**paired(a[m],b[m])) for m in METRICS])
    summary.to_csv(OUT/'metric_comparison.csv',index=False)
    pd.concat([b.assign(arm=EQUAL),a.assign(arm=WEIGHTED)]).reset_index().to_csv(OUT/'per_seed_selected.csv',index=False)
    # Cross-check the original primary contrast without changing its family.
    old=pd.read_csv(SOURCE/'contrasts.csv')
    old=old[old.added.eq(WEIGHTED)&old.reference.eq(EQUAL)].set_index('metric')
    for m in set(old.index)&set(METRICS):
        row=summary.set_index('metric').loc[m]
        np.testing.assert_allclose([row.mean_delta,row.ci95_low,row.ci95_high],old.loc[m,['mean','ci95_low','ci95_high']].to_numpy(dtype=float),atol=1e-9,rtol=0)
    weak=[]
    for seed in SEEDS:
        bb,aa=ub.loc[seed],ua.loc[seed]
        ue=int(bb.complete_success_fraction.idxmin());v,w=bb.loc[ue],aa.loc[ue]
        weak.append(dict(seed=seed,ue_id=ue,success_equal_pct=100*v.complete_success_fraction,
            success_weighted_pct=100*w.complete_success_fraction,ttl_equal_pct=v.queue_failure_pct,
            ttl_weighted_pct=w.queue_failure_pct,payload_equal_mbps=v.complete_payload_mbps,
            payload_weighted_mbps=w.complete_payload_mbps,payload_delta_mbps=w.complete_payload_mbps-v.complete_payload_mbps,
            delay_equal_ms=v.success_delay_mean_ms,delay_weighted_ms=w.success_delay_mean_ms))
    weak=pd.DataFrame(weak);weak.to_csv(OUT/'weak_ue_pairs.csv',index=False)
    changes=[]
    for m in ['complete_payload_mbps','success_delay_mean_ms','success_delay_p95_ms','outcome_mean_ms','queue_failure_pct','complete_success_fraction']:
        d=ua[m]-ub[m]
        changes.append(dict(metric=m,increased=int((d>1e-9).sum()),decreased=int((d< -1e-9).sum()),unchanged=int((d.abs()<=1e-9).sum()),positive_delta_sum=float(d[d>1e-9].sum()),negative_delta_sum=float(d[d< -1e-9].sum())))
    changes=pd.DataFrame(changes);changes.to_csv(OUT/'ue_change_counts.csv',index=False)
    cdf=[];common=[];transitions=[];packet_hashes={}
    for seed in SEEDS:
        frames={}
        for arm,metrics in [(EQUAL,b),(WEIGHTED,a)]:
            path=COHORTS/f'{arm}_seed{seed}.csv.gz';packet_hashes[str(path)]=sha(path)
            f=pd.read_csv(path)
            assert not f.censored.any() and f.successfully_delivered.dtype==bool
            assert len(f)==metrics.loc[seed,'cohort_packets']
            assert int(f.successfully_delivered.sum())==metrics.loc[seed,'complete_success']
            np.testing.assert_allclose(f.loc[f.successfully_delivered,'system_time_ms'].mean(),metrics.loc[seed,'success_delay_mean_ms'],atol=1e-9,rtol=0)
            frames[arm]=f
        bf,af=frames[EQUAL],frames[WEIGHTED]
        merged=bf.merge(af,on=['ue_id','packet_id'],suffixes=('_equal','_weighted'),validate='one_to_one')
        assert len(merged)==len(bf)==len(af)
        for name in ['arrival_tti','packet_size_bytes','ttl_tti']:
            np.testing.assert_array_equal(merged[name+'_equal'],merged[name+'_weighted'])
        for ms in [.5,1,2,5,10,20,50,100,150,190,200]:
            eq=100*(bf.successfully_delivered & bf.system_time_ms.le(ms)).mean()
            wt=100*(af.successfully_delivered & af.system_time_ms.le(ms)).mean()
            cdf.append(dict(seed=seed,deadline_ms=ms,equal_pct=eq,weighted_pct=wt,delta_pp=wt-eq))
        both=merged.successfully_delivered_equal & merged.successfully_delivered_weighted
        common.append(dict(seed=seed,common_success=int(both.sum()),
            gained_success=int((~merged.successfully_delivered_equal&merged.successfully_delivered_weighted).sum()),
            lost_success=int((merged.successfully_delivered_equal&~merged.successfully_delivered_weighted).sum()),
            common_equal_mean_ms=float(merged.loc[both,'system_time_ms_equal'].mean()),
            common_weighted_mean_ms=float(merged.loc[both,'system_time_ms_weighted'].mean()),
            common_mean_delta_ms=float((merged.loc[both,'system_time_ms_weighted']-merged.loc[both,'system_time_ms_equal']).mean())))
        for label in ['equal','weighted']:
            merged['state_'+label]=np.where(merged['successfully_delivered_'+label],'success',np.where(merged['outcome_'+label].eq('expired'),'expired','crc_failure'))
        for (before,after),group in merged.groupby(['state_equal','state_weighted']):
            transitions.append(dict(seed=seed,equal_outcome=before,weighted_outcome=after,packets=len(group)))
        print('PAIRED_COHORT_CHECKED',seed,'common_success',int(both.sum()),flush=True)
    cdf=pd.DataFrame(cdf);common=pd.DataFrame(common)
    cdf.to_csv(OUT/'deadline_delivery.csv',index=False);common.to_csv(OUT/'common_success.csv',index=False)
    pd.DataFrame(transitions).to_csv(OUT/'outcome_transitions.csv',index=False)
    counts=dict(cohort_packets=int(a.cohort_packets.sum()),success_delta=int((a.complete_success-b.complete_success).sum()),
        expiry_delta=int((a.expired_packets-b.expired_packets).sum()),crc_failure_delta=int((a.residual_crc_failure_packets-b.residual_crc_failure_packets).sum()))
    assert counts['success_delta']==-counts['expiry_delta']-counts['crc_failure_delta']
    cost=pd.read_csv(SOURCE/'search_costs.csv')
    ca=cost[cost.arm.eq(WEIGHTED)].set_index('seed').sort_index();cb=cost[cost.arm.eq(EQUAL)].set_index('seed').sort_index()
    cost=pd.DataFrame([dict(metric=m,**paired(ca[m],cb[m])) for m in ['mean_candidates','mean_rounds','mean_shared_host_decision_ms']])
    cost.to_csv(OUT/'resource_search_cost.csv',index=False)
    report(summary,weak,changes,cdf,common,counts,cost)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for name,label,color in [('equal_pct','Equal weights','#64748b'),('weighted_pct','Deficit weights','#147d92')]:
        v=cdf.groupby('deadline_ms')[name].mean();axes[0].plot(v.index,v.values,marker='o',label=label,color=color)
    axes[0].set_xscale('log');axes[0].set_xlabel('Delivery deadline (ms, log scale)');axes[0].set_ylabel('All arrivals delivered by deadline (%)');axes[0].grid(alpha=.2);axes[0].legend()
    axes[1].plot(SEEDS,weak.success_equal_pct,marker='o',label='Equal weights',color='#64748b')
    axes[1].plot(SEEDS,weak.success_weighted_pct,marker='o',label='Deficit weights',color='#147d92')
    axes[1].set_xlabel('Seed');axes[1].set_ylabel('Complete success of baseline worst UE (%)');axes[1].set_xticks(SEEDS);axes[1].grid(alpha=.2);axes[1].legend()
    fig.suptitle('Deficit weights: delivery protection versus latency | boundary, 5 paired seeds')
    fig.savefig(OUT/'tradeoff.png',dpi=160);fig.savefig(OUT/'tradeoff.pdf');plt.close(fig)
    checks=dict(passed=True,new_native_runs=0,source_manifest_verified=True,original_primary_contrast_reproduced=True,
        ten_complete_cohorts_checked=True,matching_packet_identities=True,no_censoring=True,
        packet_success_failure_count_conservation=True,counts=counts,
        source_files={name:sha(SOURCE/name) for name in used},packet_cohorts_sha256=packet_hashes,
        limitations=['Post-hoc exploratory metrics; nominal unadjusted intervals.','Seed is independent unit n=5; UE and packet counts are descriptive.','Weak UE selected from equal-arm outcomes; no unseen subgroup validation.','Both-success packet subset is post-treatment conditional, not an identified causal principal stratum.','Frozen boundary/rate0.8, no HARQ; not a new blind test.','Concurrent timing is not a controlled comparison.'])
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    (OUT/'manifest.json').write_text(json.dumps(dict(files={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}),indent=2)+'\n')
    print('DEFICIT_WEIGHT_FOLLOWUP_COMPLETE',flush=True)


if __name__=='__main__':main()

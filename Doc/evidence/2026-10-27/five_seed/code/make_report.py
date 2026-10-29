"""Export all metrics, preserve raw detail, and reconcile the project evidence."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .run import OUT,REFERENCE,REPO,REGISTRY,baseline


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4<<20),b''):value.update(chunk)
    return value.hexdigest()


def plot_summary(frame,windows,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for axis,(metric,label) in zip(axes.flat,[('goodput_mbps','Segment goodput (Mbps)'),
        ('complete_payload_mbps','Complete packet payload (Mbps)'),('expiry_pct','Packet expiry (%)'),
        ('success_delay_p95_ms','Successful packet P95 (ms)')]):
        for seed in range(41,46):
            pair=frame[frame.seed==seed].set_index('arm')
            axis.plot([0,1],[pair.loc['pfq0',metric],pair.loc['current_rho090',metric]],marker='o',alpha=.8,label=f'seed{seed}')
        axis.set_xticks([0,1],['PFQ','rho=0.90']);axis.set_ylabel(label);axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Boundary08: five matched seeds, 10-second measurement')
    fig.tight_layout();fig.savefig(destination/'comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    seconds=windows[windows.window.str.fullmatch(r'second_\d{2}')].copy()
    seconds['second']=seconds.window.str[-2:].astype(int)
    for arm,color in [('pfq0','#8c6d31'),('current_rho090','#2166ac')]:
        for _,data in seconds[seconds.arm==arm].groupby('seed'):
            data=data.sort_values('second')
            axes[0].plot(data.second,data.goodput_mbps,color=color,alpha=.15)
            axes[1].plot(data.second,data.mean_backlog_bytes/1e6,color=color,alpha=.15)
        means=seconds[seconds.arm==arm].groupby('second').mean(numeric_only=True)
        axes[0].plot(means.index,means.goodput_mbps,color=color,linewidth=2,label=arm)
        axes[1].plot(means.index,means.mean_backlog_bytes/1e6,color=color,linewidth=2,label=arm)
    axes[0].set_ylabel('Segment goodput (Mbps)');axes[1].set_ylabel('Mean queue (MB)')
    axes[1].set_xlabel('Nonoverlapping second within measurement')
    for axis in axes:axis.legend();axis.grid(alpha=.2)
    fig.tight_layout();fig.savefig(destination/'window_diagnostics.png',dpi=170);plt.close(fig)


def main():
    validation=json.loads((OUT/'validation.json').read_text());assert validation['passed']
    protocol=json.loads((OUT/'protocol.json').read_text())
    frame=pd.read_csv(OUT/'per_seed_metrics.csv')
    means=pd.read_csv(OUT/'mean_std.csv')
    ci=pd.read_csv(OUT/'paired_mean_ci.csv');primary=ci[ci.window=='measurement']
    stable=pd.read_csv(OUT/'stability.csv');costs=pd.read_csv(OUT/'search_and_floor_diagnostics.csv')
    by_metric=primary.set_index('metric')
    positive=all(by_metric.loc[key,'ci95_low']>0 for key in ['goodput_mbps','complete_payload_mbps'])
    lines=['最新智能算法ρ=0.90：与PFQ相同的五seed长窗口原生实验（2026-10-27）','',
        '结论：'+('本窗口中两类吞吐的平均配对增益95%区间均为正。' if positive else '本窗口未能同时建立两类吞吐的正向配对区间。'),
        '算法ID：'+protocol['algorithm_id'],
        '新算法为PFQ→CTBS裁剪→因果服务缺口权重→完整物理局部搜索，下界ceil(0.90*本臂当前草案总出队量)。',
        '不加载/推理BC或定向协调器，不构造定向学习特征，不读取外部下界序列；比例固定不再调节。',
        '本次新增ρ0.90原生轨迹5条，共142000TTI、5112000条UE×TTI记录；复用已核验PFQ5条同条件轨迹。',
        'seeds41–45；共同PFQ预热1–4000，各策略稳定化4001–8000；测量8001–28000共10秒，随访至28400。',
        '同PFQ的连续信道时钟genFastFadingGpu(t)、实际关联SVD、显式3dB阴影、同一步fresh输入；3小区/36UE、17PRG、boundary0.8、3000B、TTL200ms、HARQ关闭。','',
        '主要指标：seed等权均值±样本SD，独立单位n=5',
        '指标 | PFQ | 最新ρ0.90']
    metrics=[('actual_offered_mbps','实际输入Mbps'),('goodput_mbps','分段goodput Mbps'),
        ('complete_payload_mbps','完整包payload Mbps'),('complete_success_pct','整包成功率%'),
        ('success_delay_mean_ms','成功包平均时延ms'),('success_delay_p50_ms','成功包P50ms'),
        ('success_delay_p95_ms','成功包P95ms'),('success_delay_p99_ms','成功包P99ms'),
        ('outcome_mean_ms','全包平均终态时延ms'),('outcome_p99_ms','全包P99终态时延ms'),
        ('expiry_pct','过期率%'),('residual_crc_failure_pct','残留CRC整包失败率%'),
        ('total_failure_pct','总整包未交付率%'),('tb_bler_pct','TB BLER%'),
        ('ue_goodput_p5_mbps','UE goodput P5 Mbps'),('ue_goodput_p10_mbps','UE goodput P10 Mbps'),
        ('ue_goodput_jain','UE goodput Jain'),('sinr_tx_mean_db','实际TX SINR均值dB'),
        ('mcs_tx_mean','实际TX MCS均值'),('prg_utilization','cell-PRG占用比例'),
        ('mean_backlog_bytes','平均队列bytes')]
    def summary(arm,key):
        row=means[(means.arm==arm)&(means.metric==key)].iloc[0]
        return f'{row["mean"]:.6f} ± {row.sample_std:.6f}'
    for key,label in metrics:lines.append(label+' | '+summary('pfq0',key)+' | '+summary('current_rho090',key))
    lines.extend(['','配对效应：ρ0.90−PFQ，名义95%t区间df4，未作多重比较校正'])
    for key in ['goodput_mbps','complete_payload_mbps','complete_success_pct','expiry_pct','success_delay_mean_ms','ue_goodput_p5_mbps']:
        row=by_metric.loc[key]
        lines.append(f"{key}：均差{row.mean_delta:+.6f}，95%CI [{row.ci95_low:.6f},{row.ci95_high:.6f}]；正/负/零seed={int(row.positive_seeds)}/{int(row.negative_seeds)}/{int(row.equal_seeds)}；均值之比增益{row.gain_of_means_pct:+.6f}%。")
    lines.extend(['','逐seed完整测量窗','seed | PFQ goodput | rho090 goodput | PFQ完整包 | rho090完整包 | PFQ过期% | rho090过期%'])
    for seed in range(41,46):
        pair=frame[frame.seed==seed].set_index('arm');a=pair.loc['current_rho090'];b=pair.loc['pfq0']
        lines.append(f'{seed} | {b.goodput_mbps:.6f} | {a.goodput_mbps:.6f} | {b.complete_payload_mbps:.6f} | {a.complete_payload_mbps:.6f} | {b.expiry_pct:.6f} | {a.expiry_pct:.6f}')
    lines.extend(['','实用稳定性与积压贡献',
        f"通过实际稳定性诊断：{int(stable.practical_stability_passed.sum())}/{len(stable)}条轨迹。", 
        '沿用原PFQ协议：前后5秒goodput差绝对值≤全窗5%；净队列变化及队列线性趋势速率绝对值≤实际输入1%。通过不证明渐近稳态或容量；TTL过期可使过载队列保持有界。'])
    for _,row in stable.iterrows():
        lines.append(f"seed{int(row.seed)}/{row.arm}：后5s−前5s={row.goodput_second_minus_first_half_pct_of_full:+.6f}%全窗；净队列={row.net_queue_drift_pct_of_offered:+.6f}%输入；队列趋势={row.queue_linear_trend_pct_of_offered:+.6f}%输入；通过={row.practical_stability_passed}。")
    balance=pd.read_csv(OUT/'goodput_gain_balance.csv')
    lines.append('分段增益均值的守恒分解(Mbps)：'+', '.join(f'{key}={balance[key].mean():+.6f}' for key in
        ['goodput_gain_mbps','accepted_contribution_mbps','expiry_contribution_mbps','failed_byte_contribution_mbps','backlog_release_contribution_mbps']))
    matched=pd.read_csv(OUT/'matched_ue_deltas.csv')
    lines.extend(['','逐UE与尾部结果'])
    for key in ['fixed_window_goodput_mbps','complete_payload_mbps','complete_success_fraction','success_delay_mean_ms','queue_failure_pct']:
        d=matched[key+'_delta'];lines.append(f'{key}：180个匹配(seed,UE)中正/负/零={int((d>0).sum())}/{int((d<0).sum())}/{int((d==0).sum())}。对时延/失败率，负值表示改善。')
    lines.extend(['不能由总体均值推出所有UE或所有指标均改善。详见matched_ue_deltas.csv和完整包级cohort。','',
        '下界和计算工作量',f"ρ0.90测量期平均轮数={costs.mean_rounds.mean():.6f}，每决策平均候选数={costs.mean_candidates.mean():.6f}，最大轮数={int(costs.max_rounds.max())}。",
        '并行共享主机墙钟只作诊断，不宣称实时调度或受控提速；完整搜索只指既定局部邻域，不是全局最优。','',
        '有效性与解释边界',
        '五条新轨迹逐TTI动作/MCS/层数/TBS/出队、逐决策独立当前草案下界、完整收敛及模型调用为0核验通过。',
        '全28400TTI外生H/W指纹、业务/拓扑及共同4000TTI PFQ前缀/4001份完整输入CTBS配对通过；FIFO/逐行及跨TTI字节守恒和成熟cohort无截尾通过。',
        '新轨迹30个保存状态独立核验SVD/接收器/输入物理量；首个信道跨界递推单独通过。物理量独立重算范围是这些检查点，不是全部TTI。',
        'seed41参与过短窗比例选型，因此本组五seed不构成完全未见过的盲测；仅一个负载/场景和10秒有限窗口。',
        '短窗选型使用旧周期时钟，本轮匹配PFQ连续时钟；新旧绝对数值变化不能单独归因于策略。',
        '本轮保持ρ0.90及旧冻结MCS3评分语义，不用五seed结果重调比例；没有把本次版本变化称为纯学习模块收益。',
        'goodput按服务时隙成功分段统计；完整包按测量期到达且全部字节成功的成熟cohort统计。',
        '总整包未交付=过期+残留CRC失败，TB BLER分母不同。成功时延有条件选择；SINR/MCS仅按实际TX样本，样本组成可因策略变化。','',
        '详细工件',
        'per_seed_metrics.csv、mean_std.csv、paired_differences.csv、paired_mean_ci.csv：完整系统指标和配对效应。',
        'per_ue_metrics.csv、matched_ue_deltas.csv：逐UE吞吐、资源、成功/失败、mean/p50/p95/p99时延与退化明细。',
        'window_metrics.csv、stability.csv、goodput_gain_balance.csv：十个1秒窗、两个5秒半窗、稳定性和积压贡献。',
        'packet_cohorts/*.csv.gz、deadline_delivery_cdf.csv：全部成熟测量到达包终态/字节/时延及及时完整交付分布。',
        'raw_runs/：每条新旧轨迹的原生metrics.bin.gz、动作/输入指纹及启动/验证身份；新算法另保留决策/每轮搜索/草案/下界审计。',
        'raw_index.json、metric_definitions.json、manifest.json：原始二进制字段、解压后SHA-256、来源和完整性。',
        'current_algorithm.json及code/：当前版本注册和本次实际实现/分析脚本快照。',
        '原实验目录：'+str(OUT)])
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    detail=[]
    for _,row in primary.iterrows():
        key=row.metric
        detail.append(f'{key} | PFQ {summary("pfq0",key)} | rho090 {summary("current_rho090",key)} | delta {row.mean_delta:+.9g} [{row.ci95_low:.9g},{row.ci95_high:.9g}] | gain% {row.gain_of_means_pct:.9g}')
    (OUT/'detailed_metrics.txt').write_text('\n'.join(detail)+'\n')
    destination=REPO/'deliverables/boundary08_rho090_five_seed_20261027'
    destination.mkdir(exist_ok=False)
    exports=['report.txt','detailed_metrics.txt','protocol.json','per_seed_metrics.csv','mean_std.csv','per_ue_metrics.csv',
        'paired_differences.csv','paired_mean_ci.csv','window_metrics.csv','stability.csv','goodput_gain_balance.csv',
        'matched_ue_deltas.csv','deadline_delivery_cdf.csv','search_and_floor_diagnostics.csv','metric_definitions.json',
        'validation.json','native_campaign_complete.json','campaign_launch.json','campaign_status.json','native_build_manifest.json']
    for name in exports:shutil.copy2(OUT/name,destination/name)
    shutil.copy2(REGISTRY,destination/'current_algorithm.json')
    shutil.copytree(OUT/'packet_cohorts',destination/'packet_cohorts')
    (destination/'code').mkdir()
    for path in OUT.glob('*.py'):shutil.copy2(path,destination/'code'/path.name)
    shutil.copy2(Path(baseline.__file__),destination/'code/reference_long_runner.py')
    policy_source=OUT.parent/'boundary08_causal_floor_ratio_seed41_20261027_v1/policy.py'
    shutil.copy2(policy_source,destination/'code/selected_rho_policy.py')
    shutil.copy2(policy_source.with_name('search_ratio.py'),destination/'code/search_ratio.py')
    shutil.copy2(REPO/'cuMAC/scripts/run_stageB_current_algorithm.sh',destination/'code/run_stageB_current_algorithm.sh')
    raw_index=[]
    for seed in range(41,46):
        for arm,source in [('pfq0',REFERENCE/f'seed{seed}_pfq0_20000'),('current_rho090',OUT/f'seed{seed}_current_20000')]:
            target=destination/'raw_runs'/f'seed{seed}_{arm}';target.mkdir(parents=True)
            with (source/'metrics.bin').open('rb') as incoming,(target/'metrics.bin.gz').open('wb') as raw_out:
                with gzip.GzipFile(filename='',mode='wb',fileobj=raw_out,compresslevel=3,mtime=0) as zipped:
                    shutil.copyfileobj(incoming,zipped,4<<20)
            compressed_tables=[]
            for name in ['traffic_offered_packets.csv','packet_outcome_samples.csv']:
                archived=target/(name+'.gz')
                with (source/name).open('rb') as incoming,archived.open('wb') as raw_out:
                    with gzip.GzipFile(filename='',mode='wb',fileobj=raw_out,compresslevel=3,mtime=0) as zipped:
                        shutil.copyfileobj(incoming,zipped,4<<20)
                compressed_tables.append(dict(file=str(archived.relative_to(destination)),uncompressed_sha256=sha(source/name)))
            for name in ['prefix.npz','states.pt','weights.npy','launch.json','execution.json','validation.json','physical_validation.json',
                         'causal_floor_validation.json','paired_validation.json','input_validation.json',
                         'decisions.csv','rounds.csv','alignment_audit.csv','drafts.npy','ue_topology.csv','snr.txt','sim.log']:
                if (source/name).exists():shutil.copy2(source/name,target/name)
            raw_index.append(dict(seed=seed,arm=arm,source=str(source),compressed_metrics=str((target/'metrics.bin.gz').relative_to(destination)),
                uncompressed_metrics_sha256=sha(source/'metrics.bin'),shape=[28400,36,19],dtype='float64 little endian',
                compressed_tables=compressed_tables,
                source_fields=json.loads((OUT/'metric_definitions.json').read_text())['source_fields']))
    (destination/'raw_index.json').write_text(json.dumps(raw_index,indent=2)+'\n')
    plot_summary(frame,pd.read_csv(OUT/'window_metrics.csv'),destination)
    entries=[]
    for path in sorted(destination.rglob('*')):
        if path.is_file():entries.append(dict(file=str(path.relative_to(destination)),bytes=path.stat().st_size,sha256=sha(path)))
    (destination/'manifest.json').write_text(json.dumps(dict(files=entries,source=str(OUT)),indent=2)+'\n')
    evidence=REPO/'Doc/evidence/2026-10-27';manifest=json.loads((evidence/'manifest.json').read_text())
    for name in ['per_seed_metrics.csv','mean_std.csv','per_ue_metrics.csv','paired_mean_ci.csv','stability.csv',
                 'goodput_gain_balance.csv','search_and_floor_diagnostics.csv','metric_definitions.json','validation.json','report.txt']:
        source=OUT/name;target=evidence/'five_seed'/name;shutil.copy2(source,target)
        manifest['files']=[x for x in manifest['files'] if x['file']!=str(target.relative_to(evidence))]
        manifest['files'].append(dict(file=str(target.relative_to(evidence)),source=str(source),bytes=target.stat().st_size,sha256=sha(target)))
    code_evidence=evidence/'five_seed/code';code_evidence.mkdir(exist_ok=False)
    for name in ['run.py','validate.py','analyze.py','make_report.py','independent_review.py',
                 'selected_rho_policy.py','search_ratio.py','reference_long_runner.py']:
        source=destination/'code'/name;target=code_evidence/name;shutil.copy2(source,target)
        manifest['files'].append(dict(file=str(target.relative_to(evidence)),source=str(source),bytes=target.stat().st_size,sha256=sha(target)))
    (evidence/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    update_project_docs(frame,means,by_metric,validation)
    print('RHO090_FIVE_SEED_REPORT_EXPORTED',destination,flush=True)


def update_project_docs(frame,means,intervals,validation):
    path=REPO/'Doc/stageB_current_progress.md';text=path.read_text()
    first='### 五seed长窗口对照\n'
    start=text.index(first);end=text.index('\n## 1.',start)
    lines=[first,'与PFQ相同协议的五seed验证已完成：seeds41–45，共同PFQ预热4000TTI、各策略稳定化4000TTI，测量8001–28000共10秒，随访至28400。ρ=0.90未重新调节；新增智能算法五条原生轨迹，复用已核验的PFQ五条轨迹。','',
        '| 指标（五seed均值±样本SD） | PFQ | 最新ρ0.90 |','|---|---:|---:|']
    for metric,label in [('goodput_mbps','分段goodput Mbps'),('complete_payload_mbps','完整包payload Mbps'),
        ('complete_success_pct','整包成功率%'),('expiry_pct','过期率%'),('success_delay_mean_ms','成功包平均时延ms'),
        ('success_delay_p95_ms','成功包P95ms'),('ue_goodput_p5_mbps','UE goodput P5 Mbps')]:
        cells=[]
        for arm in ['pfq0','current_rho090']:
            x=means[(means.arm==arm)&(means.metric==metric)].iloc[0];cells.append(f'{x["mean"]:.2f} ± {x.sample_std:.2f}')
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines.append('')
    for metric,label in [('goodput_mbps','分段goodput'),('complete_payload_mbps','完整包payload')]:
        x=intervals.loc[metric]
        lines.append(f'{label}相对PFQ：均值之比增益{x.gain_of_means_pct:+.2f}%；配对差{x.mean_delta:+.2f}Mbps，名义95%t区间[{x.ci95_low:.2f},{x.ci95_high:.2f}]，正向{int(x.positive_seeds)}/5seed。')
    lines.extend(['',f"全部原生/FIFO/配对与因果下界核验通过；实际稳定性诊断全部通过={validation['all_practical_stability_checks_passed']}。稳定性通过不证明渐近稳态或容量；seed41参与过短窗比例选择，不能称全新盲测。",
        '本轮使用与PFQ一致的连续信道时钟。单seed选择使用旧周期时钟短窗，新旧绝对数值差不能单独归因于算法；历史含协调器current不改名。',
        '详细指标保留在[完整报告](../deliverables/boundary08_rho090_five_seed_20261027/report.txt)、[逐seed指标](evidence/2026-10-27/five_seed/per_seed_metrics.csv)、[逐UE指标](evidence/2026-10-27/five_seed/per_ue_metrics.csv)、[配对区间](evidence/2026-10-27/five_seed/paired_mean_ci.csv)及本地deliverables的包级cohort/原生逐TTI档案。',''])
    text=text[:start]+'\n'.join(lines)+text[end:]
    text=text.replace('单seed比例选型通过；与PFQ相同协议的五seed长窗口实验正在执行。','单seed比例选型及与PFQ相同协议的五seed长窗口实验均已完成，详细结果和证据见下。')
    path.write_text(text)
    path=REPO/'Doc/evidence/2026-10-27/README.md'
    text=path.read_text().replace('与PFQ一致的长窗口五seed协议；原生运行与最终指标逐步补齐，未完成时不宣称多seed收益',
        '与PFQ一致的长窗口五seed协议、完整系统/逐UE指标、配对区间、稳定性及核验证据；已完成')
    path.write_text(text)
    path=REPO/'Doc/current_stageB_effective_configuration.md'
    text=path.read_text().replace('完整口径见[指标字典](evidence/2026-10-21/corrected/metric_definitions.json)。',
        '完整口径见[当前指标字典](evidence/2026-10-27/five_seed/metric_definitions.json)。')
    path.write_text(text)


if __name__=='__main__':main()

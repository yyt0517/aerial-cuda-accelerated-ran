"""Self-contained Chinese report, scientific figures and a compact evidence bundle."""
import argparse
import html
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import pandas as pd
from training.stageb_ppo.native import ROOT, write, sha
from .model import STAGES, SPATIAL_STAGES, PERIODS, setting

ALGORITHMS={'deficit':'DPLS','mlwdf_pf':'DA-PLS'}
COLORS={'prg':'#2166ac','wideband':'#c45f22'}


def table(df):
    return df.to_html(index=False,border=0,escape=True,float_format=lambda x:f'{x:.3f}',na_rep='—')


def add_settings(df):
    df=df.copy()
    df['granularity']=[setting(s)[0] if s in SPATIAL_STAGES else s for s in df.stage]
    df['period_ms']=[setting(s)[1]*.5 if s in SPATIAL_STAGES else np.nan for s in df.stage]
    return df


def figures(root, q, quality, timing, mechanism, recovery, feedback):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','figure.facecolor':'white'})
    periods=np.array(PERIODS)*.5
    quality=add_settings(quality);q=add_settings(q);timing=add_settings(timing);mechanism=add_settings(mechanism)
    metrics=[('payload_mbps','Whole-packet throughput (Mbps)'),('delay_p95_ms','Successful-packet P95 (ms)'),
             ('delay_p99_ms','Successful-packet P99 (ms)'),('expiry_pct','Expired packets (%)')]
    def style(ax, label):
        ax.set_xscale('log');ax.set_xticks(periods,[f'{x:g}' for x in periods]);ax.grid(alpha=.2)
        ax.set_xlabel('Covariance period (ms)');ax.set_ylabel(label)
    def curves(ax, data, metric, baselines=True):
        for granularity in COLORS:
            p=data[data.granularity==granularity].sort_values('period_ms')
            ax.plot(p.period_ms,p[metric],'o-',lw=2,color=COLORS[granularity],label=granularity)
        if baselines:
            for stage,color,line,label in [('exact','#36764a','--','Original CSI'),('reported','#68727e',':','Scalar RSRP')]:
                ax.axhline(data.loc[data.stage==stage,metric].iloc[0],color=color,ls=line,lw=1.5,label=label)
    for scene,label in [('boundary','Boundary08'),('uniform','Uniform')]:
        fig,axes=plt.subplots(2,4,figsize=(17,8),layout='constrained')
        for row,(objective,algorithm) in enumerate(ALGORITHMS.items()):
            part=quality[(quality.scene==scene)&(quality.objective==objective)]
            raw=q[(q.scene==scene)&(q.objective==objective)]
            for column,(metric,ylabel) in enumerate(metrics):
                ax=axes[row,column];curves(ax,part,metric)
                for granularity in COLORS:
                    for seed,g in raw[raw.granularity==granularity].groupby('seed'):
                        g=g.sort_values('period_ms');ax.plot(g.period_ms,g[metric],color=COLORS[granularity],alpha=.16,lw=.8)
                style(ax,ylabel);ax.set_title(algorithm)
                if metric!='payload_mbps':ax.set_ylim(bottom=0)
                if row==0 and column==0:ax.legend(fontsize=8)
        fig.suptitle(f'{label} | original-channel execution | means and three paired seed traces',fontsize=14)
        fig.savefig(root/f'quality_{scene}.png',dpi=165);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for (scene,objective),ax in zip([('boundary','deficit'),('boundary','mlwdf_pf'),('uniform','deficit'),('uniform','mlwdf_pf')],axes.flat):
        g=recovery[(recovery.scene==scene)&(recovery.objective==objective)&(recovery.metric=='payload_mbps')]
        g=add_settings(g)
        curves(ax,g,'gap_recovered_pct',False);style(ax,'Throughput gap recovered (%)')
        ax.axhline(0,color='#68727e',ls=':',label='Scalar RSRP')
        ax.axhline(100,color='#36764a',ls='--',label='Original CSI')
        ax.set_title(f'{scene} / {ALGORITHMS[objective]}')
    axes[0,0].legend(fontsize=9)
    fig.suptitle('Recovery of original-to-scalar throughput gap | ratio of seed means; not clipped',fontsize=14)
    fig.savefig(root/'recovery.png',dpi=165);plt.close(fig)
    fig,axes=plt.subplots(4,3,figsize=(16,14),layout='constrained')
    for row,(scene,objective) in enumerate([('boundary','deficit'),('boundary','mlwdf_pf'),('uniform','deficit'),('uniform','mlwdf_pf')]):
        t=timing[(timing.scene==scene)&(timing.objective==objective)]
        m=mechanism[(mechanism.scene==scene)&(mechanism.objective==objective)]
        curves(axes[row,0],t,'amortized_cpu_ms');style(axes[row,0],'Amortized decision CPU (ms)')
        curves(axes[row,1],m,'scored_edits');style(axes[row,1],'Fresh candidate scores / decision')
        g=recovery[(recovery.scene==scene)&(recovery.objective==objective)&(recovery.metric=='payload_mbps')]
        g=add_settings(g).merge(feedback[['stage','payload_mbps_at_16bits']],on='stage',validate='one_to_one')
        ax=axes[row,2]
        for granularity in COLORS:
            p=g[g.granularity==granularity].sort_values('period_ms')
            ax.plot(p.payload_mbps_at_16bits,p.gap_recovered_pct,'o-',color=COLORS[granularity],label=granularity)
            for v in p.itertuples():ax.annotate(f'{v.period_ms:g}',(v.payload_mbps_at_16bits,v.gap_recovered_pct),fontsize=7,xytext=(4,4),textcoords='offset points')
        ax.set_xscale('log');ax.grid(alpha=.2);ax.set_xlabel('Hypothetical 16-bit payload (Mbps)')
        ax.set_ylabel('Throughput gap recovered (%)');ax.axhline(100,color='#36764a',ls='--',lw=1)
        for ax in axes[row]:ax.set_title(f'{scene} / {ALGORITHMS[objective]}')
        if row==0:axes[row,0].legend(fontsize=8)
    fig.suptitle('Matched-state CPU / scores | feedback-performance tradeoff; annotations = period (ms)',fontsize=14)
    fig.savefig(root/'cost_feedback.png',dpi=165);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for (scene,objective),ax in zip([('boundary','deficit'),('boundary','mlwdf_pf'),('uniform','deficit'),('uniform','mlwdf_pf')],axes.flat):
        t=timing[(timing.scene==scene)&(timing.objective==objective)]
        for granularity in COLORS:
            g=t[t.granularity==granularity].sort_values('period_ms')
            ax.plot(g.period_ms,g.cached_cpu_ms,'o-',color=COLORS[granularity],label=granularity+' cached')
            ax.plot(g.period_ms,g.refresh_cpu_ms,'s--',color=COLORS[granularity],label=granularity+' refresh')
        style(ax,'Decision CPU (ms)');ax.set_title(f'{scene} / {ALGORITHMS[objective]}')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Cache-hit versus new spatial-report processing | fresh PHY/search context in both',fontsize=14)
    fig.savefig(root/'timing_refresh.png',dpi=165);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(13,13),layout='constrained')
    for row,(scene,objective) in enumerate([('boundary','deficit'),('boundary','mlwdf_pf'),('uniform','deficit'),('uniform','mlwdf_pf')]):
        m=mechanism[(mechanism.scene==scene)&(mechanism.objective==objective)]
        for column,(metric,label) in enumerate([('normalized_shape_relative_error','Normalized spatial-shape error'),('root_sinr_bias_db','Same-action SINR bias (dB)')]):
            curves(axes[row,column],m,metric);style(axes[row,column],label)
            axes[row,column].set_title(f'{scene} / {ALGORITHMS[objective]}')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Spatial aging and frequency averaging | same saved states and same PFQ action',fontsize=14)
    fig.savefig(root/'shape_age.png',dpi=165);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(14,13),layout='constrained')
    for row,(scene,objective) in enumerate([('boundary','deficit'),('boundary','mlwdf_pf'),('uniform','deficit'),('uniform','mlwdf_pf')]):
        m=mechanism[(mechanism.scene==scene)&(mechanism.objective==objective)]
        for column,(metric,label) in enumerate([
            ('selected_true_utility_vs_exact_pct','Selected true utility vs original action (%)'),
            ('selected_predicted_utility_error_pct','Selected utility prediction error (%)')]):
            curves(axes[row,column],m,metric);style(axes[row,column],label)
            axes[row,column].set_title(f'{scene} / {ALGORITHMS[objective]}')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Chosen actions evaluated on original channel | same queues, weights and MCS/layers',fontsize=14)
    fig.savefig(root/'action_truth.png',dpi=165);plt.close(fig)


def main(args):
    root=Path(args.root).resolve();analysis=root/'analysis';replay=root/'replay'
    for file,field in [('campaign/completion.json','completed'),('analysis/completion.json','passed'),
        ('analysis/pairing.json','passed'),('analysis/spatial_pairing.json','passed'),
        ('replay/validation.json','passed'),('smoke_analysis/completion.json','passed'),
        ('smoke_endpoint_equivalence.json','passed'),('replay_smoke/validation.json','passed')]:
        assert json.loads((root/file).read_text())[field],file
    q=pd.read_csv(analysis/'per_seed.csv');assert len(q)==144 and set(q.seed)=={41,42,43}
    quality=pd.read_csv(analysis/'means.csv');recovery=pd.read_csv(analysis/'recovery.csv')
    feedback=pd.read_csv(analysis/'feedback.csv');paired=pd.read_csv(analysis/'paired.csv')
    timing=pd.read_csv(replay/'timing_summary.csv');own=pd.read_csv(replay/'own_timing_summary.csv')
    mechanism=pd.read_csv(replay/'mechanism_summary.csv');validation=json.loads((replay/'validation.json').read_text())
    assert validation['matched_states']==72 and validation['own_states']==432
    figures(root,q,quality,timing,mechanism,recovery,feedback)
    complete=add_settings(quality)
    recovery_wide=recovery.pivot(index=['scene','objective','stage'],columns='metric',values='gap_recovered_pct')
    recovery_wide=recovery_wide.rename(columns={c:c+'_gap_recovered_pct' for c in recovery_wide.columns}).reset_index()
    complete=complete.merge(recovery_wide,on=['scene','objective','stage'],how='left',validate='one_to_one')
    complete=complete.merge(timing[['scene','objective','stage','cached_cpu_ms','refresh_cpu_ms','amortized_cpu_ms']],
        on=['scene','objective','stage'],validate='one_to_one')
    complete=complete.merge(feedback[['stage','payload_mbps_at_16bits','payload_mbps_at_32bits','float64_serialized_mbps']],
        on='stage',validate='many_to_one')
    for metric in ['amortized_cpu_ms','payload_mbps']:
        base=complete[complete.stage=='reported'][['scene','objective',metric]].rename(columns={metric:metric+'_scalar'})
        complete=complete.merge(base,on=['scene','objective'],validate='many_to_one')
        complete[metric+'_change_vs_scalar_pct']=100*(complete[metric]/complete[metric+'_scalar']-1)
    complete=complete.sort_values(['scene','objective','granularity','period_ms'])
    complete.to_csv(root/'comparison.csv',index=False)
    new=complete[complete.stage.isin(SPATIAL_STAGES)]
    best=new.loc[new.groupby(['scene','objective','granularity']).payload_mbps.idxmax()].copy()
    best.to_csv(root/'best_observed_mean.csv',index=False)
    baseline=complete[complete.stage.isin(['exact','reported'])]
    baseline.to_csv(root/'baselines.csv',index=False)
    lines=[]
    for v in best.itertuples():
        lines.append(f'{v.scene} / {ALGORITHMS[v.objective]} / {v.granularity}：扫描内吞吐均值最高点为 {v.period_ms:g} ms，'
            f'完整包吞吐 {v.payload_mbps:.2f} Mbps，恢复原版与纯 RSRP 差距的 {v.payload_mbps_gap_recovered_pct:.1f}%；'
            f'P95 {v.delay_p95_ms:.2f} ms，P99 {v.delay_p99_ms:.2f} ms，过期率 {v.expiry_pct:.3f}%。')
    columns=['scene','objective','granularity','period_ms','payload_mbps','payload_mbps_gap_recovered_pct',
        'delay_p95_ms','delay_p99_ms','expiry_pct','amortized_cpu_ms','amortized_cpu_ms_change_vs_scalar_pct',
        'payload_mbps_at_16bits']
    displayed=new[columns].copy();displayed.objective=displayed.objective.map(ALGORITHMS)
    best_display=best[columns].copy();best_display.objective=best_display.objective.map(ALGORITHMS)
    feedback_display=feedback[['stage','period_ms','rsrp_real_parameters_per_s','covariance_real_parameters_per_s',
        'payload_mbps_at_16bits','payload_mbps_at_32bits','float64_serialized_mbps']]
    diagnostic_columns=['scene','objective','stage','covariance_age_tti','normalized_shape_relative_error',
        'centered_shape_similarity','root_sinr_bias_db','root_sinr_mae_db','scored_edits','rounds',
        'sum_neighborhood_sizes','first_slack_weighted_bytes','first_scored','first_required',
        'first_paid_to_required','first_paid_fraction','mmse_misses']
    error_columns=['scene','objective','stage','actual_tb_bler','native_mcs_mean','full_sinr_bias_db','full_sinr_var_db2','full_sinr_mae_db']
    full_error=q[q.full_window_prediction].groupby(['scene','objective','stage'],as_index=False)[error_columns[3:]].mean()
    full_error.to_csv(root/'full_window_prediction_summary.csv',index=False)
    ue=pd.read_csv(analysis/'per_ue.csv');assert len(ue)==144*36
    ue['ues_with_expiry']=(ue.expired_packets>0).astype(int)
    ue['ues_p95_above_100ms']=(ue.delay_p95_ms>100).astype(int)
    ue['ues_without_success']=ue.delay_p95_ms.isna().astype(int)
    ue_counts=ue.groupby(['scene','objective','stage','seed'],as_index=False)[
        ['ues_with_expiry','ues_p95_above_100ms','ues_without_success']].sum()
    ue_counts.to_csv(root/'ue_tail_counts.csv',index=False)
    html_text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RSRP＋慢更新空间协方差</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;color:#203047;max-width:1350px;margin:40px auto;padding:0 24px}}
h1,h2{{line-height:1.3}}h2{{margin-top:34px}}table{{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}}
th,td{{border-bottom:1px solid #dbe2eb;padding:8px;text-align:right;white-space:nowrap}}th{{background:#edf3f8}}
img{{width:100%;height:auto}}pre{{background:#edf3f8;padding:16px;overflow:auto}}.note{{padding:16px;background:#f2f6fa;border-left:4px solid #2166ac}}
a{{color:#166b98}}details{{margin:20px 0}}summary{{cursor:pointer;font-weight:bold}}</style><body>
<h1>邻区 RSRP＋慢更新归一化空间协方差</h1>
<p>2026-11-10 · 三小区 Boundary08 / Uniform · DPLS / DA-PLS · seeds 41–43</p>
<p class="note">144 条正式轨迹：120 条新增，24 条已验证原版／纯 RSRP 端点复用。每条测量 10 秒。
两种频域粒度 × 五个更新周期；执行端始终使用原始瞬时信道，服务 CSI、MCS／层数／OLLA 和搜索器保持原协议。</p>
<h2>模型与对照</h2>
<pre>C = H Hᴴ
R = 4C / trace(C)，trace(R) = 4
估计邻区干扰协方差 Ĉ = 4 × beta_RSRP × R_reported
纯 RSRP 对应 R = I₄</pre>
<p>逐 PRG 方案保留每个 UE／邻区／PRG 的方向结构；宽带方案先对 17 个 PRG 的 C 求平均，再归一化，一份矩阵用于全部 PRG。
更新周期为 0.5、2、10、50、100 ms，均有 0.5 ms 传输延迟。因此最快点也不是零延迟 oracle。</p>
<p>RSRP 仍采用原来的单参考端口功率、100 ms 因果 EMA、10 ms 上报和 0.5 ms 传输。
空间协方差不另加时间滤波：本轮隔离慢速采样／保持造成的老化，而不是同时改变空间统计窗口。
归一化矩阵是额外测量信息，不能从 RSRP 标量推导出来；这里假设能获得理想的分邻区、全端口空间测量。</p>
<p>恢复方向结构不恢复逐 PRG 瞬时总功率；宽带 RSRP 的端口、滤波和上报误差仍在。
因此本轮恢复比例不是对上一轮“空间步骤损失份额”的机械逆运算，两者还存在交互和闭环轨迹变化。</p>
<h2>吞吐、尾时延和过期恢复</h2>
<p>以下列出扫描内吞吐均值最高的点，用于快速定位曲线；它不是经独立验证或多重比较校正的“最优周期”。</p>
<ul>{''.join('<li>'+html.escape(s)+'</li>' for s in lines)}</ul>
{table(best_display)}
<p>吞吐恢复比例 = (新方案 − 纯 RSRP) / (原版 − 纯 RSRP)。时延／过期率改用相反方向，恢复比例同样以原版到纯 RSRP 的差距为分母。
采用三个 seed 等权均值之比，不裁剪到 0～100%；负值表示更差，超过 100% 表示本次均值超过原版。原始差距为零时记为“—”，不硬算恢复率。</p>
<img src="recovery.png" alt="两种协方差粒度随周期变化的吞吐差距恢复比例">
<img src="quality_boundary.png" alt="Boundary08 吞吐、P95、P99、过期率">
<img src="quality_uniform.png" alt="Uniform 吞吐、P95、P99、过期率">
<p>粗线为三 seed 均值，细线为逐 seed 轨迹；水平线是原版和纯 RSRP 均值。P95/P99 只统计成功包。
包 TTL 为 200 ms，成功包尾时延接近上限时，需要同时看过期率，不能把尾时延趋平解释为服务没有继续恶化。</p>
<details><summary>全部周期的性能、反馈与 CPU 对照</summary>{table(displayed)}</details>
<details><summary>吞吐、P95、P99 与过期率各自恢复比例</summary>
{table(new[['scene','objective','granularity','period_ms','payload_mbps_gap_recovered_pct','delay_p95_ms_gap_recovered_pct','delay_p99_ms_gap_recovered_pct','expiry_pct_gap_recovered_pct']])}</details>
<details><summary>原版／纯 RSRP 端点</summary>{table(baseline[['scene','objective','stage','payload_mbps','delay_p95_ms','delay_p99_ms','expiry_pct','amortized_cpu_ms']])}</details>
<details><summary>完整包、MAC 成功字节与过期以外的未交付</summary>
<p>完整包吞吐采用测量期到达包并随访至终态；MAC 成功字节吞吐统计测量 TTI 内的成功字节，二者的边界包口径也不同。
HARQ 关闭时，失败字节仍出队，所以零过期不等于所有包都完整交付。</p>
{table(complete[['scene','objective','stage','payload_mbps','goodput_mbps','expiry_pct','other_nondelivery_pct','actual_tb_bler']])}</details>
<details><summary>Boundary08 的逐 UE 尾部与过期范围</summary>
<p>下表逐 seed 统计 36 个 UE 中有过期包的数量、成功包 P95 超过 100 ms 的数量，以及没有成功包而无法定义该分位数的数量。
100 ms 是描述性阈值。吞吐恢复率不能替代各 UE 的排队改善；没有成功包的 UE 也不能被当作低时延 UE。</p>
{table(ue_counts[(ue_counts.scene=='boundary') & ue_counts.stage.isin(['exact','reported','prg_t1','prg_t200'])])}</details>
<details><summary>相对两个端点的配对区间</summary><p>nominal 95% 配对 t 区间，n=3，无多重比较修正。区间跨零不表示等价；恢复比例本身没有据此构造比值置信区间。</p>
{table(paired[paired.metric.isin(['payload_mbps','delay_p95_ms','delay_p99_ms','expiry_pct'])])}</details>
<h2>反馈量</h2>
<p>只计 36 UE × 2 邻区 = 72 条邻区链路的有效载荷；服务 CSI 沿用原接口，不计入增量。
4×4 Hermitian 矩阵有 16 个实自由度，固定 trace 后剩 15 个：三个实对角元和六个复非对角元。
逐 PRG 每次发送 72×17 份，宽带每次发送 72 份，另保留每 10 ms 的 72 个 RSRP 标量。</p>
{table(feedback_display)}
<p class="note">16／32 bit 列是每个实参数采用该位宽时的理论有效载荷预算，性能仿真没有量化。
实际模型采用 float64 参数，另列对应载荷；未计报头、时间戳、链路标识、编码、导频资源或测量成本，因此不能直接当作标准无线反馈开销。
反馈传输占用的无线资源也未从吞吐结果中扣除。</p>
<p>统计的是调度器需要的邻区信息。仿真内部为兼容执行端仍保留原始信道和完整测量数组，不把这些内部内存数据误算成新方案的空口反馈。</p>
<p>原版干扰计算可以直接使用逐 PRG 的完整协方差，按 16 个实参数／矩阵计；当前原始复 H 接口为 32 个实参数／矩阵，数据量还会翻倍。
此处同时提供两种表示口径，避免只用较大的复信道接口夸大压缩收益；两者都不是考虑时间压缩等手段后的信息论下界。</p>
<h2>调度 CPU 与更新成本</h2>
<img src="cost_feedback.png" alt="调度 CPU、候选评分和反馈恢复权衡">
<p>72 个共同状态来自最快逐 PRG 方案：每个场景／算法／seed 取六个分层并扰动的报告相位。
所有 12 种模型接收相同队列、权重、原始 PFQ 动作和服务 CSI；每个模型串行重复 12 次、循环轮换顺序、固定一个 CPU，关闭审计和 profiling。</p>
<p>分别测缓存命中和新报告到达两种路径。新报告路径包含解包和 PSD 因子分解；旧报告复用已缓存因子。</p>
<pre>估计摊销 CPU = cached × (1 − 1/K) + refresh / K，K 以 TTI 计</pre>
<p>这是抽样状态上的受控摊销估计，不能当作全窗口在线平均时延或 P95 保证。共同状态来自一种策略访问的队列状态，
因此另列全部 432 个常规保存状态的各策略自身轨迹计时。两种比较回答不同问题。</p>
<img src="timing_refresh.png" alt="缓存命中与报告刷新时的 CPU 成本">
<details><summary>同状态串行 CPU 全表</summary>{table(timing)}</details>
<details><summary>各策略自身轨迹的串行 CPU</summary>{table(own)}</details>
<p>CPU 包括状态重建、保存权重、PFQ 解码、空间模型适配和搜索；不含原生信道生成、GPU PFQ、桥接、HOL 计算、测量模拟与传输。
原生执行二进制未变，本轮未重复测量信道生成墙钟。并发闭环计时保留在逐 seed 原始表中，不用于加速比。</p>
<p>为沿用已核验的 H 输入 PHY，协方差在报告到达时分解成因子，然后按 RSRP 缩放；这不是对邻区物理复信道的估计。
矩阵分解成本已计入更新路径。直接接收协方差的专用接口可能有不同成本，本实验未实现或宣称该优化。</p>
<h2>空间信息为什么能恢复或失效</h2>
<img src="shape_age.png" alt="归一化方向误差及相同动作 SINR 偏差">
<p>归一化形状误差相对原始逐 PRG 的 R 计算；宽带方案因此同时包含频域方向平均和报告老化。
centered_shape_similarity 是去掉 I 后的矩阵方向相似度；纯 RSRP 的零方向向量无定义。
SINR 偏差在相同原始 PFQ 动作上计算，避免各方案选择不同动作造成偏差。原版长期平均空间统计接近各向同性，并不保证瞬时空间方向能被慢报告保留。</p>
<details><summary>同状态机制数据</summary>{table(mechanism[diagnostic_columns])}</details>
<img src="action_truth.png" alt="选中动作的原始信道加权期望收益及其预测误差">
<details><summary>选中动作在原始信道上的期望收益</summary>
<p>各模型先在相同队列、权重、MCS／层数下选动作，再用独立 NumPy PHY 和原始信道计算该动作的期望成功字节及加权收益，计算在计时区间外。
selected_true_utility_vs_exact_pct 比较该动作与原版选中动作的真实加权期望收益，负值表示较低；原版是局部搜索参照，不是全局最优上界。
selected_predicted_utility_error_pct 是该动作的预测收益相对其真实期望收益的误差，正值才表示这项加权目标被高估。
百分比先逐状态计算，再按 seed 等权平均。这比跨不同动作直接比较平均 SINR 更接近调度选择本身，但仍是保存状态上的单步期望收益，不是闭环吞吐。</p>
{table(mechanism[['scene','objective','stage','selected_true_expected_bytes','selected_true_utility_vs_exact_pct','selected_predicted_utility_error_pct']])}</details>
<p>平均 SINR 误差与最终选中动作的收益误差需要分开看。搜索在近似目标上挑选动作，整体 SINR 平均误差接近零，仍可能伴随被选动作的加权收益被高估。
这里各模型的 MCS／层数固定相同，直接反映的是动作选择及收益估计的差异；闭环中的 MCS 仍可能随后通过原生反馈调整。</p>
<p>候选上界不做该候选的干扰精评，而是以可出队字节 d 替代预计成功字节 e = d × (1 − BLER)。
对受候选影响的 UE，上界与精评收益的差为 Σ[Q(wd) − Q(we)]，Q 是搜索器的整数化函数。
因此输入从复信道变成标量，并不自动减少候选评分：估计 BLER、收益排序、访问轮数及该差值共同决定必须精评多少候选。
表中的 first_slack_weighted_bytes 是首轮可行候选的平均上界松弛；first_paid_fraction 是首轮精评数／可行候选数，
first_paid_to_required 才是精评数／当前证书必评数。scored_edits 与 sum_neighborhood_sizes 覆盖整次搜索，后者按访问轮次累计候选出现次数。</p>
<p>所有上界与精评都使用同一份近似信息。邻区瞬时原始信道不直接用于候选评分，只进入测量模拟、执行与对照核验；服务 CSI 始终保留。
旧协方差也可能让近似目标更容易剪枝，却产生更差的实际调度；评分减少和证书成立不能单独证明物理性能恢复。
CPU 还包括每次空间报告到达时的解包与因子分解，必须结合缓存命中、刷新和按周期摊销的三列判断。</p>
<p>独立审计覆盖 {validation['certificate_rounds']:,} 个访问轮次、{validation['candidate_occurrences']:,} 次候选出现。
实际新精评 {validation['fresh_paid']:,} 次，当前剪枝证书必评 {validation['required']:,} 次，上界违规 {validation['bound_violations']}。
每个 witness 的第一个诊断相位审计所有访问轮次，其余相位审计完整首轮。证书针对相应近似目标，不证明它等于原始物理目标或达到全局最优。</p>
<p>新增轨迹记录了全部测量 TTI 的已选动作预测 SINR／BLER；复用的历史端点没有这份完整预测序列，因此下表只列新轨迹。
逐 UE 实际 BLER、MCS、过期数和完整包吞吐覆盖所有 144 条轨迹，另有上述同状态比较覆盖全部 12 个模型。</p>
<details><summary>新轨迹的完整窗口预测误差</summary><p>误差为估计减原生执行 SINR；均值为正才表示平均偏乐观。方差单位 dB²。</p>{table(full_error)}</details>
<h2>核验、边界与下载</h2>
<p>5 项单元检查、48 条短程原生轨迹、8 个历史端点逐值复现均通过。各正式轨迹均检查原始 H 未修改、搜索收敛和候选计数守恒。
432 个常规保存状态通过独立 NumPy PHY 与原生执行核验。新轨迹的拓扑、业务、预热、信道／RSRP 流与端点配对；同 seed 下空间报告与 witness 以及另一目标函数的完整报告流一致。</p>
<p>另在全部 72 个共同状态核对选中动作的原始信道期望成功字节与原生执行记录，最大绝对差为 {validation['chosen_action_native_max_expected_bytes_error']:.6g} 字节。</p>
<p>每条轨迹 2 秒共同预热、2 秒策略稳定、10 秒测量和 0.2 秒随访；测量期到达包均有终态。HARQ 关闭，失败字节仍出队。
归一化空间报告是理想测量，没有量化噪声、波束／端口校准误差或额外测量噪声；三个 seed 的结果属于探索性证据。</p>
<ul><li><a href="comparison.csv">完整周期对照</a> · <a href="analysis/recovery.csv">吞吐／尾时延／过期率恢复比例</a> · <a href="analysis/feedback.csv">反馈量预算</a></li>
<li><a href="analysis/per_seed.csv">逐 seed 指标</a> · <a href="analysis/paired.csv">配对区间</a> · <a href="analysis/per_ue.csv">逐 UE 指标与新轨迹全窗 SINR 误差</a> · <a href="ue_tail_counts.csv">逐 UE 尾部／过期数量</a></li>
<li><a href="search_traces/index.csv">逐 TTI 候选／精评／在线计时记录索引</a>（完整原始表保存在报告包的 search_traces 目录；按索引中的测量起止 TTI 筛选，在线并发计时不替代受控串行 CPU 比较）</li>
<li><a href="replay/timing_raw.csv">缓存／更新串行计时</a> · <a href="replay/own_timing_summary.csv">自身轨迹计时</a> · <a href="replay/paired_intervals.csv">CPU 配对区间</a></li>
<li><a href="replay/mechanism_raw.csv">逐状态机制数据</a> · <a href="replay/common_root_errors.csv">同动作逐 UE 误差</a> · <a href="replay/certificate_rounds.csv">逐轮证书审计</a></li>
<li><a href="campaign/protocol.json">冻结协议</a> · <a href="analysis/spatial_pairing.json">空间报告配对</a> · <a href="replay/validation.json">重放与选中动作收益核验</a></li>
<li><a href="source/README.rst">实现与复现说明</a> · <a href="experiment_bundle.zip">报告、数据与复现源码包</a></li></ul></body></html>'''
    (root/'report.html').write_text(html_text,encoding='utf-8')
    source=root/'source';source.mkdir(exist_ok=True)
    for path in Path(__file__).parent.iterdir():
        if path.suffix in ('.py','.rst'):shutil.copy2(path,source/path.name)
    traces=root/'search_traces';traces.mkdir(exist_ok=True);trace_index=[]
    for v in q.itertuples():
        native=Path(v.native_directory);src=native/'search.csv.gz';dst=traces/(v.key+'.csv.gz')
        shutil.copy2(src,dst);digest=sha(dst);assert digest==sha(src)
        launch=json.loads((native/'launch.json').read_text())
        trace_index.append(dict(key=v.key,scene=v.scene,objective=v.objective,stage=v.stage,seed=v.seed,
            path=str(dst.relative_to(root)),sha256=digest,
            measurement_first_tti=launch['measurement_first_tti'],measurement_last_tti=launch['measurement_last_tti']))
    pd.DataFrame(trace_index).to_csv(traces/'index.csv',index=False)
    protocol=json.loads((root/'campaign/protocol.json').read_text())
    for name,digest in protocol['sources'].items():assert sha(ROOT/name)==digest,name
    artifacts=[root/'report.html',root/'comparison.csv',root/'best_observed_mean.csv',root/'full_window_prediction_summary.csv',root/'ue_tail_counts.csv']
    artifacts+=list(root.glob('*.png'))+[analysis/n for n in ('per_seed.csv','paired.csv','recovery.csv','feedback.csv','per_ue.csv','pairing.json','spatial_pairing.json')]
    artifacts+=[replay/n for n in ('validation.json','timing_summary.csv','own_timing_summary.csv','mechanism_summary.csv','certificate_rounds.csv')]
    artifacts+=list(traces.glob('*.csv.gz'))+[traces/'index.csv']
    write(root/'result_manifest.json',dict(completed=True,runs=144,new_runs=120,reused_runs=24,seeds=[41,42,43],
        periods_ms=[p*.5 for p in PERIODS],granularities=['prg','wideband'],
        artifacts={str(p.relative_to(root)):sha(p) for p in artifacts},
        source_sha256={p.name:sha(p) for p in source.iterdir() if p.is_file()}))
    include=[p for p in root.iterdir() if p.suffix in ('.html','.png','.csv','.json') and p.name!='postprocessing_status.json']
    include+=list(source.glob('*'))+[p for p in replay.iterdir() if p.is_file()]+[p for p in analysis.iterdir() if p.is_file()]
    include+=list(traces.iterdir())
    include+=[root/'campaign/protocol.json',root/'campaign/completion.json',root/'replay_smoke/validation.json',
        root/'unit_tests.log',root/'smoke_v1/completion.json',root/'smoke_analysis/completion.json',
        root/'smoke_analysis/pairing.json',root/'smoke_analysis/spatial_pairing.json']
    with zipfile.ZipFile(root/'experiment_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in include:
            if path.is_file():z.write(path,str(path.relative_to(root)))
    print(best_display.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);main(parser.parse_args())

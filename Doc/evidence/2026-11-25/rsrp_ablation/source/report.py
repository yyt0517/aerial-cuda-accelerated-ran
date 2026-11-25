"""Standalone figures, tables and evidence report for ordered ablation."""
import argparse
import html
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import zipfile
import numpy as np
import pandas as pd
from training.stageb_ppo.native import ROOT,write,sha
from .model import STAGES,LABELS,FACTORS

ALGORITHMS={'deficit':'DPLS','mlwdf_pf':'DA-PLS'}
SHORT=['A Exact','B Spatial','C Frequency','D Pilot','E Filter','F Report']
COLORS={'deficit':'#2166ac','mlwdf_pf':'#c45f22'}


def table(df):
    return df.to_html(index=False,border=0,escape=True,float_format=lambda x:f'{x:.3f}')


def figures(root,quality,timing,mechanism,ue=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'figure.facecolor':'white','axes.titleweight':'bold'})
    groups=[('boundary','Boundary08'),('uniform','Uniform')]
    metrics=[('payload_mbps','Whole-packet throughput (Mbps)'),('delay_p95_ms','Successful-packet P95 (ms)'),
             ('delay_p99_ms','Successful-packet P99 (ms)'),('expiry_pct','Expired packets (%)')]
    fig,axes=plt.subplots(2,4,figsize=(17,8),layout='constrained')
    for r,(scene,label) in enumerate(groups):
        for c,(metric,title) in enumerate(metrics):
            ax=axes[r,c]
            for objective,name in ALGORITHMS.items():
                g=quality[(quality.scene==scene)&(quality.objective==objective)]
                means=g.groupby('stage')[metric].mean().reindex(STAGES)
                ax.plot(range(6),means,'o-',label=name,c=COLORS[objective],lw=2)
                for seed,part in g.groupby('seed'):
                    ax.plot(range(6),part.set_index('stage')[metric].reindex(STAGES),c=COLORS[objective],alpha=.18,lw=.8)
            ax.set_xticks(range(6),list('ABCDEF'));ax.grid(alpha=.2);ax.set_ylabel(title);ax.set_title(label)
            if metric!='payload_mbps':ax.set_ylim(bottom=0)
            if r==0 and c==0:ax.legend()
    nseeds=quality.seed.nunique()
    fig.suptitle(f'Ordered closed-loop ablation | original-channel execution | means and {nseeds} paired seed traces',fontsize=14)
    fig.supxlabel('A Exact  →  B Remove spatial structure  →  C Average PRGs  →  D Single pilot port  →  E EMA  →  F Periodic delayed report')
    fig.savefig(root/'quality.png',dpi=165);plt.close(fig)
    metrics=[(timing,'cpu_ms','CPU time / decision (ms)'),
        (mechanism,'scored_edits','Fresh candidate scores / decision'),
        (mechanism,'first_slack_weighted_bytes','First-round bound slack (weighted bytes)'),
        (mechanism,'first_paid_fraction','First-round paid / floor-eligible')]
    fig,axes=plt.subplots(2,4,figsize=(17,8),layout='constrained')
    for r,(scene,label) in enumerate(groups):
        for c,(data,metric,title) in enumerate(metrics):
            ax=axes[r,c]
            for objective,name in ALGORITHMS.items():
                g=data[(data.scene==scene)&(data.objective==objective)].set_index('stage').reindex(STAGES)
                ax.plot(range(6),g[metric],'o-',label=name,c=COLORS[objective],lw=2)
            ax.set_xticks(range(6),list('ABCDEF'));ax.set_title(label);ax.set_ylabel(title);ax.grid(alpha=.2);ax.set_ylim(bottom=0)
            if r==0 and c==0:ax.legend()
    fig.suptitle(f'Identical saved states for all six models | {24*nseeds} endpoint states | serial timing; oracle audit untimed',fontsize=14)
    fig.supxlabel('A Exact  →  B Spatial  →  C Frequency  →  D Pilot  →  E Filter  →  F Report')
    fig.savefig(root/'mechanism.png',dpi=165);plt.close(fig)
    if ue is not None:
        metrics=[('actual_tb_bler','Actual TB BLER (%)',100.),('native_mcs_mean','Native MCS',1.),
            ('used_olla_p50_db','OLLA median and P05/P95 (dB)',1.),('selected_sinr_error_mean_db','Sampled SINR estimate - original (dB)',1.)]
        fig,axes=plt.subplots(2,4,figsize=(17,8),layout='constrained')
        for r,(scene,label) in enumerate(groups):
            for c,(metric,title,scale) in enumerate(metrics):
                ax=axes[r,c]
                for objective,name in ALGORITHMS.items():
                    g=ue[(ue.scene==scene)&(ue.objective==objective)].set_index('stage').reindex(STAGES)
                    ax.plot(range(6),scale*g[metric],'o-',label=name,c=COLORS[objective],lw=2)
                    if metric=='used_olla_p50_db':
                        ax.fill_between(range(6),g.used_olla_p05_db,g.used_olla_p95_db,color=COLORS[objective],alpha=.12)
                ax.set_xticks(range(6),list('ABCDEF'));ax.set_title(label);ax.set_ylabel(title);ax.grid(alpha=.2)
                if r==0 and c==0:ax.legend()
                if metric=='actual_tb_bler':ax.set_ylim(0,10.5)
                if metric=='used_olla_p50_db' and scene=='uniform':
                    ax.text(.03,.04,'Unclipped OLLA; MCS capped at 27',transform=ax.transAxes,fontsize=8)
                if metric=='selected_sinr_error_mean_db':ax.axhline(0,c='#666',ls=':',lw=1)
        fig.suptitle('Native BLER / MCS / OLLA: full measurement window | SINR error: saved states only',fontsize=14)
        fig.supxlabel('A Exact  →  B Spatial  →  C Frequency  →  D Pilot  →  E Filter  →  F Report | statistics averaged equally across seeds')
        fig.savefig(root/'ue_diagnostics.png',dpi=165);plt.close(fig)


def main(args):
    root=Path(args.root).resolve();analysis=root/'analysis';replay=root/'replay'
    for path,field in [(root/'campaign/completion.json','completed'),(analysis/'completion.json','completed'),
        (analysis/'pairing.json','passed'),(replay/'validation.json','passed'),
        (root/'smoke_analysis/completion.json','completed'),(root/'smoke_endpoint_equivalence.json','passed')]:
        assert json.loads(path.read_text())[field]
    q=pd.read_csv(analysis/'per_seed.csv');assert len(q)==72 and set(q.seed)=={41,42,43}
    if not (root/'ue_diagnostics/validation.json').exists():
        from .ue_diagnostics import main as diagnose
        diagnose(SimpleNamespace(root=str(root),workers=4))
    assert json.loads((root/'ue_diagnostics/validation.json').read_text())['passed']
    while not (root/'native_timing/validation.json').exists():
        print('WAITING for separate native timing probe',flush=True);time.sleep(30)
    assert json.loads((root/'native_timing/validation.json').read_text())['passed']
    ued=pd.read_csv(root/'ue_diagnostics/summary.csv')
    per_ue=pd.read_csv(root/'ue_diagnostics/per_ue.csv')
    saturation=pd.read_csv(root/'ue_diagnostics/olla_saturation_per_ue.csv')
    per_ue=per_ue.merge(saturation[['key','ue_id','scheduled_samples','mcs27_fraction','used_olla_below_minus20_fraction']],
        on=['key','ue_id'],validate='one_to_one')
    ue_distributions=[]
    for identity,part in per_ue.groupby(['scene','seed','objective','stage']):
        record=dict(zip(['scene','seed','objective','stage'],identity))
        record.update(ue_count=len(part),ues_with_expiry=int((part.expired_packets>0).sum()),
            expired_packets=int(part.expired_packets.sum()),
            transmitted_tb_per_tti=float(part.tb_tx_count.sum()/20000),
            successful_tb_per_tti=float((part.tb_tx_count-part.tb_err_count).sum()/20000),
            scheduled_mcs27_pct=float(100*(part.mcs27_fraction*part.scheduled_samples).sum()/part.scheduled_samples.sum()),
            scheduled_olla_below_minus20_pct=float(100*(part.used_olla_below_minus20_fraction*part.scheduled_samples).sum()/part.scheduled_samples.sum()),
            maximum_ue_expiry_pct=float(part.expired_pct.max()))
        for quantile in (.05,.5,.95):
            record[f'ue_bler_p{int(quantile*100):02d}_pct']=100*part.actual_tb_bler.quantile(quantile)
        ue_distributions.append(record)
    ue_distributions=pd.DataFrame(ue_distributions)
    ue_distributions.to_csv(root/'ue_diagnostics/ue_distribution_per_seed.csv',index=False)
    ue_distributions=ue_distributions.drop(columns='seed').groupby(['scene','objective','stage'],as_index=False).mean()
    ue_distributions.to_csv(root/'ue_diagnostics/ue_distribution_summary.csv',index=False)
    hotspot_columns=['scene','seed','objective','ue_id','expired_packets','expired_pct',
        'actual_tb_bler','native_mcs_mean','scheduled_tti_pct','whole_packet_mbps']
    hotspots=per_ue[(per_ue.stage=='reported')&(per_ue.expired_packets>0)].sort_values('expired_packets',ascending=False)
    hotspots=hotspots.groupby(['scene','objective']).head(3)[hotspot_columns]
    controls=per_ue[per_ue.stage=='exact'][hotspot_columns]
    hotspots=hotspots.merge(controls,on=['scene','seed','objective','ue_id'],suffixes=('_reported','_exact'),validate='one_to_one')
    hotspots.to_csv(root/'ue_diagnostics/expiry_hotspots.csv',index=False)
    native_timing=pd.read_csv(root/'native_timing/summary.csv')
    t=pd.read_csv(replay/'timing_summary.csv');m=pd.read_csv(replay/'mechanism_summary.csv')
    counters=pd.read_csv(replay/'cost_decomposition.csv');profile=pd.read_csv(replay/'profile_summary.csv')
    own=pd.read_csv(replay/'own_timing_summary.csv')
    channel=pd.read_csv(root/'channel_information_summary.csv')
    assert json.loads((root/'channel_information_validation.json').read_text())['passed']
    certificate=json.loads((replay/'validation.json').read_text())
    metrics=['payload_mbps','goodput_mbps','delay_mean_ms','delay_p95_ms','delay_p99_ms','outcome_p95_ms',
        'expiry_pct','success_pct','other_nondelivery_pct','queue_mean_mb']
    means=q.groupby(['scene','objective','stage'],as_index=False)[metrics].mean()
    means['stage']=pd.Categorical(means.stage,STAGES,ordered=True)
    means=means.sort_values(['scene','objective','stage']);means.to_csv(root/'quality_summary.csv',index=False)
    comparisons=[];adjacent=[];statements=[];cost_statements=[];bound_statements=[]
    paired=pd.read_csv(analysis/'paired.csv')
    for scene in ('boundary','uniform'):
        for objective in ALGORITHMS:
            g=means[(means.scene==scene)&(means.objective==objective)].set_index('stage')
            tg=t[(t.scene==scene)&(t.objective==objective)].set_index('stage')
            mg=m[(m.scene==scene)&(m.objective==objective)].set_index('stage')
            a,b=g.loc['reported'],g.loc['exact'];ta,tb=tg.loc['reported'],tg.loc['exact'];ma,mb=mg.loc['reported'],mg.loc['exact']
            comparisons.append(dict(scene=scene,algorithm=ALGORITHMS[objective],
                throughput_change_pct=100*(a.payload_mbps/b.payload_mbps-1),
                p95_original_ms=b.delay_p95_ms,p95_reported_ms=a.delay_p95_ms,
                p99_original_ms=b.delay_p99_ms,p99_reported_ms=a.delay_p99_ms,
                expiry_original_pct=b.expiry_pct,expiry_reported_pct=a.expiry_pct,
                serial_cpu_original_ms=tb.cpu_ms,serial_cpu_reported_ms=ta.cpu_ms,
                serial_cpu_change_pct=100*(ta.cpu_ms/tb.cpu_ms-1),
                paid_scoring_change_pct=100*(ma.scored_edits/mb.scored_edits-1),
                first_round_paid_original=mb.first_scored,first_round_paid_reported=ma.first_scored,
                first_slack_original=mb.first_slack_weighted_bytes,first_slack_reported=ma.first_slack_weighted_bytes))
            factor=counters[(counters.scene==scene)&(counters.objective==objective)&(counters.stage=='reported')].iloc[0]
            own_cost=q[(q.scene==scene)&(q.objective==objective)].groupby('stage').scored_edits_mean.mean()
            cost_statements.append(f"{scene} / {ALGORITHMS[objective]}：同状态精评数为原版的 {factor.paid_scoring_ratio:.3f} 倍，"
                f"可记账为轮数 {factor.rounds_ratio:.3f} × 每轮邻域 {factor.neighborhood_per_round_ratio:.3f} × 精评比例 {factor.paid_fraction_ratio:.3f}。"
                f"自身闭环轨迹的每 TTI 平均精评数则是 {own_cost['exact']:.1f}→{own_cost['reported']:.1f}，"
                f"变化 {100*(own_cost['reported']/own_cost['exact']-1):+.2f}%。")
            bound_statements.append(f"{scene} / {ALGORITHMS[objective]}：同根 SINR 误差均值 {ma.root_sinr_bias_db:+.3f} dB；"
                f"首轮上界间隙 {mb.first_slack_weighted_bytes:.1f}→{ma.first_slack_weighted_bytes:.1f} 加权字节，"
                f"增加 {100*(ma.first_slack_weighted_bytes/mb.first_slack_weighted_bytes-1):.1f}%；"
                f"上界标准差变化 {100*(ma.first_upper_std/mb.first_upper_std-1):+.1f}%，"
                f"精确增益标准差变化 {100*(ma.first_gain_std/mb.first_gain_std-1):+.1f}%。")
            for i,stage in enumerate(STAGES[1:],1):
                prev=STAGES[i-1];a,before=g.loc[stage],g.loc[prev]
                item=dict(scene=scene,algorithm=ALGORITHMS[objective],factor=FACTORS[i-1],stage=stage,control=prev,
                    throughput_delta_mbps=a.payload_mbps-before.payload_mbps,
                    throughput_delta_pct_of_exact=100*(a.payload_mbps-before.payload_mbps)/b.payload_mbps,
                    p95_delta_ms=a.delay_p95_ms-before.delay_p95_ms,p99_delta_ms=a.delay_p99_ms-before.delay_p99_ms,
                    expiry_delta_pp=a.expiry_pct-before.expiry_pct,
                    serial_cpu_delta_ms=tg.loc[stage,'cpu_ms']-tg.loc[prev,'cpu_ms'])
                ci=paired[(paired.scene==scene)&(paired.objective==objective)&(paired.stage==stage)&
                          (paired.control==prev)&(paired.metric=='payload_mbps')].iloc[0]
                item.update(throughput_ci_low=ci.ci_low,throughput_ci_high=ci.ci_high)
                adjacent.append(item)
            changes=[g.loc[STAGES[i],'payload_mbps']-g.loc[STAGES[i-1],'payload_mbps'] for i in range(1,6)]
            statements.append(f"{scene} / {ALGORITHMS[objective]}：按五个相邻步骤，完整包吞吐变化依次为 "
                + '、'.join(f'{x:+.2f}' for x in changes)+' Mbps。')
    comparisons=pd.DataFrame(comparisons);adjacent=pd.DataFrame(adjacent)
    comparisons.to_csv(root/'endpoint_comparison.csv',index=False);adjacent.to_csv(root/'adjacent_effects.csv',index=False)
    figures(root,q,t,m,ued)
    display=means.copy();display['objective']=display.objective.map(ALGORITHMS);display['stage']=display.stage.map(LABELS)
    costcols=['scene','objective','stage','rounds','scored_edits','sum_neighborhood_sizes','known_exact_reuses',
              'floor_pruned','threshold_pruned','certificate_heap_pruned','mmse_requests','mmse_misses']
    firstcols=['scene','objective','stage','root_sinr_bias_db','first_root_failure_fraction','first_slack_weighted_bytes',
        'first_upper_weighted_bytes','first_best_gain_weighted_bytes','first_scored','first_eligible','first_paid_fraction']
    dispersioncols=['scene','objective','stage','first_upper_std','first_gain_std','first_prg_max_upper_std',
        'first_prg_max_gain_std','first_unique_upper_fraction']
    onlinecols=['rounds_mean','scored_edits_mean','candidate_evaluations_mean','neighborhood_per_round',
        'scored_edits_fraction','known_exact_reuses_fraction','floor_pruned_fraction','threshold_pruned_fraction',
        'certificate_heap_pruned_fraction','mmse_misses_mean','mmse_hit_fraction','queue_mean_mb']
    online=q.groupby(['scene','objective','stage'],as_index=False)[onlinecols].mean()
    online.to_csv(root/'closed_loop_search_summary.csv',index=False)
    scheduling=[]
    for row in q.itertuples():
        d=Path(row.native_directory);launch=json.loads((d/'launch.json').read_text())
        first,last=launch['measurement_first_tti'],launch['measurement_last_tti']
        with np.load(d/'actions.npz') as z:actions=z['actions'][first-1:last]
        assert actions.shape==(last-first+1,3,17)
        active=(actions>=0).sum(axis=1)
        scheduling.append(dict(scene=row.scene,seed=row.seed,objective=row.objective,stage=row.stage,
            mean_active_cells_per_prg=float(active.mean()),
            all_three_active_pct=float(100*(active==3).mean()),
            two_active_pct=float(100*(active==2).mean()),one_active_pct=float(100*(active==1).mean()),
            idle_pct=float(100*(active==0).mean())))
    scheduling=pd.DataFrame(scheduling);scheduling.to_csv(root/'scheduling_per_seed.csv',index=False)
    scheduling=scheduling.groupby(['scene','objective','stage'],as_index=False).mean(numeric_only=True).drop(columns='seed')
    scheduling.to_csv(root/'scheduling_summary.csv',index=False)
    intervals=paired[paired.metric.isin(['payload_mbps','delay_p95_ms','delay_p99_ms','expiry_pct'])&
        paired.apply(lambda row:row.stage!='exact' and row.control==STAGES[STAGES.index(row.stage)-1],axis=1)]
    stages=''.join(f'<tr><td>{chr(65+i)}</td><td>{html.escape(LABELS[s])}</td><td>{html.escape(text)}</td></tr>'
        for i,(s,text) in enumerate(zip(STAGES,[
        '原版瞬时分源 H Hᴴ；执行和评价模型一致',
        '每 UE／PRG／邻区保留 trace，改成 trace(H Hᴴ)/4 × I₄',
        '再把该功率在 PRG 之间平均；仍用全部 TX 端口',
        '改成单参考端口的当前宽带功率；单独隔离端口采样差异',
        '加入 100 ms 因果 EMA，每 TTI 立即可用',
        '再加入 10 ms 周期上报和 0.5 ms 传输延迟；等于原 RSRP 版本'])))
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RSRP 消融与候选评分开销</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;color:#203047;max-width:1350px;margin:40px auto;padding:0 24px}}
h1,h2{{line-height:1.3}}h2{{margin-top:34px}}table{{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}}
th,td{{border-bottom:1px solid #dbe2eb;padding:8px;text-align:right;white-space:nowrap}}th{{background:#edf3f8}}
img{{width:100%;height:auto}}pre{{background:#edf3f8;padding:16px;overflow:auto}}.note{{padding:16px;background:#f2f6fa;border-left:4px solid #2166ac}}
a{{color:#166b98}}details{{margin:20px 0}}summary{{cursor:pointer;font-weight:bold}}</style><body>
<h1>RSRP 消融：哪些信息造成损失，为什么候选评分变多</h1>
<p>2026-11-10 · 三小区 Boundary08 / Uniform · DPLS / DA-PLS · seeds 41–43</p>
<p class="note">72 条正式轨迹：48 条新增闭环消融，24 条已验证端点复用。各条测量 10 秒，执行端始终使用原始瞬时信道。
采用逐级消融；相邻差值是给定加入顺序下的条件效果，不是顺序无关的独立贡献。正负变化均保留。</p>
<p>代码核对：原版输入是含路径损耗／阴影及逐 TTI 相关 Rayleigh 快衰落的复信道 H，干扰使用逐 UE／PRG／邻区的 H Hᴴ，
并非只有大尺度增益。冻结原生源中的 rho=0.9938、rhoPrime=0.0786 及随机递推代码已收录在代码证据文件中。</p>
<h2>逐级实验</h2><table><tr><th>阶段</th><th>评价端</th><th>相对前一步的变化</th></tr>{stages}</table>
<p>额外拆出单端口采样，避免把原 RSRP 实现使用 TX 端口 0、而原协方差使用全部端口带来的差异误归到时间滤波。
最后一步包含周期保持和传输延迟，不能把该差值全部归因于 0.5 ms 延迟。空间、频域步骤使用原始信道构造理想控制量，不是可直接部署的测量接口。</p>
<h2>闭环性能</h2><ul>{''.join('<li>'+html.escape(s)+'</li>' for s in statements)}</ul>
<img src="quality.png" alt="六阶段吞吐、P95、P99、过期率；粗线为均值，细线为各 seed">
<p>相邻吞吐差值中的正数表示改善；时延与过期率差值中的正数表示变差。百分比均以同算法原版吞吐为分母；它们可沿本路径相加，但不代表通用、可交换的损失份额。</p>
<p>本场景包 TTL 为 200 ms。成功包 P95/P99 接近该上限后，进一步的服务恶化可能主要表现为过期率增加，不能仅凭尾时延趋平判断“没有额外损失”。</p>
{table(adjacent)}<p>下表保留最终 RSRP 与原版端点对照；计算开销来自新的同状态串行复测，而不是混用并发运行时间。</p>{table(comparisons)}
<details><summary>全部质量指标</summary>{table(display)}</details>
<details><summary>同一批信道上丢失的信息</summary><p>对全部 UE／邻区等权计算。空间误差为归一化协方差 Frobenius 误差；其余为相邻功率近似的 dB 差。
这些不同量纲的误差用于描述信息变化，不能直接换算为吞吐损失份额。</p>{table(channel)}</details>
<p>空间消融保留了每 UE／PRG／邻区的瞬时干扰总功率，却把各接收方向的功率填成相同值。
共同信道样本中，单邻区协方差最弱特征值约为平均特征值的 0.062 倍，最强约为 2.45 倍；各向同性替代会填高原先较弱的干扰方向，改变 MMSE 抑制收益。
这给出了空间误差导致 SINR 偏差的具体机制，但偏差仍取决于服务信号方向，不能推导每个 UE／动作都必然偏保守。
阶段 B 尚无任何频域或时间平均，因此它相对 A 的损失不能归因于 100 ms 滤波。</p>
<p>滤波的条件效果可以为正：单参考端口的瞬时功率并不等于全部发射端口的当前平均功率。
在共同信道样本上，以后者为参照，单端口即时功率 MAE 约 0.34 dB，滤波后约 0.27 dB；平滑可降低端口采样波动，同时带来时间滞后。
这只解释可能的抵消机制，净吞吐和尾时延效果以闭环相邻差值为准。</p>
<details><summary>相邻差值的 nominal 95% 配对 t 区间</summary><p>n=3，无多重比较修正；区间跨零不表示等价。</p>{table(intervals)}</details>
<h2>为什么近似模型可能需要更多候选评分</h2>
<p class="note">对 {certificate['certificate_rounds']} 个保存状态搜索轮次、{certificate['certificate_candidate_occurrences']:,} 次候选出现作独立完整枚举：
精评 {certificate['certificate_fresh_paid']:,} 次，当前剪枝证书要求的新评分 {certificate['certificate_required']:,} 次，比值为 1；上界违反次数为 {certificate['bound_violations']}。
每个被审计 TTI 的比值和完整在线逐 TTI 候选计数均可下载。这里的“必评”针对当前固定最大值证书，不能当作所有可能算法的信息论下界；0/0 记为无定义。</p>
<p>搜索器并不为每个候选重新穷举全部 UE／PRG／干扰组合。它先用容量上界剪枝，只对无法排除的候选增量评分，并复用不受影响的候选值和 UE／PRG／邻区活动掩码的物理缓存。
这里沿用同一个搜索器和矩阵内核；输入从协方差变成标量，并不会自动移除矩阵计算。</p>
<pre>Q(x) = round-to-even(x × 10⁶)；A 为这次编辑影响的 UE 集合
d = 本次可出队字节；e = d × (1 − BLER)；w 为固定业务权重

UB   = Σ_A [ Q(w·d_candidate) − Q(w·e_root) ]
gain = Σ_A [ Q(w·e_candidate) − Q(w·e_root) ]
UB − gain = Σ_A [ Q(w·d_candidate) − Q(w·e_candidate) ]</pre>
<p>因此，上界松弛量等于受影响候选 UE 的预测加权失败字节（含原实现的整数舍入）。BLER 预测变高时，即使逻辑容量不变，也可能扩大这个间隙，让上界不足以证明候选无用，迫使搜索器付费评分。
这不是“RSRP 测量本身更难计算”，而是评价模型改变了目标值与剪枝证据。是否发生、幅度多少，以以下同根候选的完整枚举数据为准。</p>
<p>上界和精评使用同一评价模型的根节点期望字节；上界的候选容量来自同一 MCS、层数、队列和 TBS，利用 e≤d 保证可采纳。
执行端原始 H 只用于动作执行，不混入该证书。因此这里验证的是近似目标的正确搜索，不是证明近似得分等于真实物理收益。
邻区功率频域平均也不消除服务链路频选性；上界和精确增益的标准差、跨 PRG 最大值的标准差及不同上界占比保存在机制原始表中，用于直接检验候选是否变得难以区分。</p>
<p>现有容量上界在原版中就不查询候选逐 PRG SINR；同根替换模型主要通过每 UE 的根节点期望字节改变上界。
因此“RSRP 没有频选性，所以原先精细的上界变平”并不是这段实现的直接运算链条，需要和实际松弛度、不同上界比例区分。</p>
<ul>{''.join('<li>'+html.escape(s)+'</li>' for s in bound_statements)}</ul>
<p>本次四组首轮精确增益的离散程度确有下降，但上界标准差没有下降；更直接的证据是上界间隙扩大、精评比例增加。
同状态每轮邻域规模略有减少，搜索轮数也有升有降，所以评分次数增加主要体现在候选被迫进入精评的比例上。
这说明剪枝变弱，但完整枚举未发现上界不可采纳或选错该近似目标下最佳操作。</p>
<img src="mechanism.png" alt="同状态串行耗时、评分次数和首轮上界松弛度">
<p>所有六个模型在同一原始 PFQ 根节点、同一权重、同一邻域和同一出队下限可行集合上重放。
首轮用于同根机制比较；每个后续访问轮次也作独立全数组枚举，逐候选核对上界、精确增益、松弛量、最优操作和证书所需的新评分集合。不能用队列不同解释同根比较中的差异。</p>
<p>first_root_failure_fraction 是原始 PFQ 动作的加权预测失败字节比例，既不是未加权 TB BLER，也不是最终选定动作的实测 CRC 失败率。</p>
{table(m[firstcols])}
<details><summary>上界与精确增益的离散程度</summary><p>标准差单位均为加权字节；PRG 列先取每 PRG 的最大值再跨 PRG 求标准差。不同上界占比使用原生整数精度，样本和候选可行集合保持配对。</p>{table(m[dispersioncols])}</details>
<h2>轮数、候选规模与剪枝的开销分解</h2>
<ul>{''.join('<li>'+html.escape(s)+'</li>' for s in cost_statements)}</ul>
<pre>每次决策平均新评分次数
 = 平均搜索轮数 × 每轮平均邻域规模 × 邻域中实际新评分比例</pre>
<p>下面三项比值相乘严格等于评分次数比值，均相对同状态原版。它们是开销记账分解，不是三个独立因果干预。
接受不同操作还会让后续根节点、改动涉及的 UE 集合和缓存失效范围不同；闭环中的积压又会改变合法候选和业务权重，因此另列在线自身轨迹统计。</p>
{table(counters)}
<details><summary>同状态完整搜索计数</summary>{table(m[costcols])}</details>
<details><summary>闭环自身轨迹搜索统计</summary><p>每个 seed 内先汇总，再跨 seed 等权平均。</p>{table(online)}</details>
<details><summary>闭环资源复用统计</summary><p>统计每 PRG 上同时有分配的活动小区数，反映各模型最终选择的资源复用程度；不单独解释为空口效率。</p>{table(scheduling)}</details>
<details><summary>串行耗时与独立 profiling</summary><p>同状态对照：</p>{table(t)}
<p>各阶段自身轨迹保存状态的串行耗时，包含闭环积压、权重及搜索路径变化：</p>{table(own)}
<p>独立 profiling：first_scan_ms 是首轮的重叠统计，core_ms 是搜索总计，不能与其内部各项相加。
原生细分项为插桩墙钟，带 cpu_ms 的三项为宿主进程 CPU；主耗时比较始终采用关闭插桩的串行测量。</p>{table(profile)}</details>
<h2>核对乐观估计、MCS、OLLA 和逐 UE 结局</h2>
<p>当前近似只替换邻区干扰，不滤掉服务信道的瞬时衰落。MCS 由原生宽带 SINR 和 OLLA 产生；相同状态下替换干扰评价模型不会改变 MCS。
不同闭环轨迹会通过调度及 CRC 历史间接改变 OLLA/MCS，这与“直接拿近似 SINR 选更高 MCS”不同。失败字节在 HARQ 关闭时仍会出队；部分包剩余字节仍可能排队或过期，但已失败的字节不会重传。
所以吞吐和尾时延同升本身不能确定估计方向，应分别看真实 CRC、成功 TB 载荷、资源复用、队列与有符号预测误差。</p>
<p>只滤邻区干扰时，邻区链路深衰落而报告仍接近均值，倾向于高估干扰、低估 SINR；干扰变强时才可能反向。
空间各向同性误差和频域平均会叠加在此之上，具体方向需逐 UE、逐动作核对。</p>
<p>本次最终 RSRP 与原版的四组端点均值中，完整包吞吐和成功 TB 字节吞吐均下降，并未出现吞吐与尾时延同时上升。
Boundary 的每 UE BLER 分布也没有明显整体抬升，而部分 UE 的过期率显著增加；因此不能把主要恶化归因为更多 CRC 失败。
OLLA 的反馈调节会让长期 BLER 接近其平衡值，所以 BLER 接近本身也不能证明干扰估计准确，仍需结合有符号 SINR 误差和服务量。
逐 UE 分布和过期最多 UE 的同 seed 对照如下。</p>
<p>例如 Boundary 的 DPLS，成功 TB 平均载荷约 4484→4989 字节，但成功 TB 数约 18.02→14.69 个／TTI，成功字节吞吐仍为 1292.44→1171.22 Mbps。
较大的成功 TB 不能单独推出单位时间吞吐更高。上面的端点结论也不排除某两个相邻消融阶段出现小幅吞吐与尾时延同升。</p>
<details><summary>每 UE BLER 分布与过期覆盖范围</summary><p>先在每个 seed 的 36 UE 上取分位数，再对三个 seed 等权平均；UE 数与过期数也是每 seed 均值。</p>{table(ue_distributions)}</details>
<details><summary>最终 RSRP 中过期最多 UE 与原版配对</summary><p>各场景／算法至多取三个 UE-seed；只列有过期包的组合。BLER 列是比例，expired_pct 列是百分数。</p>{table(hotspots)}</details>
<p>下面的原生 BLER/MCS/OLLA 和过期数覆盖全部测量 TTI。OLLA 以原生 float32 递推重建，并在全部 216 个保存状态核对 36 UE 的偏移和历史 CRC，逐值一致。
偏移采用代码约定：MCS 输入 SINR 减去 delta，因此 delta 更负意味着更激进的修正。
SINR 误差定义为估计减原始物理值；均值为正才是平均偏乐观，方差单位 dB²。这些误差来自保存状态，不能称为完整时间序列误差。</p>
<p>逐 UE 的 SINR 方差使用 ddof=1；样本少于两个时留空，样本数一并报告。下表汇总方差是各 seed 保存样本的描述性方差（ddof=0）再等权平均。</p>
<p>原生 OLLA 没有偏移限幅，MCS 却有 27 的上限；部分高质量 UE 即使 MCS 已封顶仍持续减小 delta，导致 Uniform 的偏移出现很大的负值。
这是沿用实现的饱和/漂移现象，不能把所有负偏移都解释为等量的有效 MCS 提升。
<a href="ue_diagnostics/olla_saturation_per_ue.csv">逐 UE 的 MCS 封顶比例、偏移范围及低于 −20 dB 的比例</a>供核对。</p>
<p>Uniform 的端点实际 BLER 略升，但平均 MCS 下降；原版约 23% 的 TB 已在 MCS 27，最终 RSRP 约为 10%。
这种封顶、反馈和服务样本组成的变化，说明不能把聚合 BLER 上升直接解释为“近似 SINR 选了更高 MCS”。分布表同时列出按调度 TB 加权的封顶比例。</p>
{table(ued)}
<img src="ue_diagnostics.png" alt="实际 BLER、原生 MCS、OLLA 分布及有符号 SINR 误差">
<p><a href="ue_diagnostics/per_ue.csv">逐 UE 完整窗口 BLER、MCS、OLLA 分位数、过期数和尾时延</a> ·
<a href="ue_diagnostics/sinr_per_ue.csv">逐 UE 有符号 SINR 误差均值、方差和样本数</a> ·
<a href="ue_diagnostics/sinr_samples.csv">逐样本误差</a> · <a href="ue_diagnostics/per_tti_candidates.csv.gz">全部 144 万个测量 TTI 的候选与精评计数</a>。
generated_edits 含重新生成条目，sum_neighborhood_sizes 是各访问轮次的候选出现次数，scored_edits 为实际新精评数；不要把前两者误当成证书必评数。
每条轨迹还保留完整 OLLA 分布数组。</p>
<p>另有<a href="replay/common_root_sinr_per_ue.csv">完全相同状态、MCS 与原始 PFQ 动作上的逐 UE 误差均值/方差</a>及
<a href="replay/common_root_sinr_errors.csv">逐样本记录</a>，可避免六阶段选择不同动作造成的抽样差异。
应特别比较滤波阶段 E 与单端口当前功率阶段 D，而非仅凭最终 F 对原版 A 的均值来判定滤波的方向。</p>
<h2>调度与仿真分别计时</h2>
<p>主要 CPU 比较来自离线同状态重放，完全不生成瞬时信道，也不调用原生仿真器，因此其 CPU 增量不能归因于信道生成。
另建只加时钟的独立原生计时探针，在现有同步点前后分别记录信道生成和 SVD，并保留原生 PFQ、调度评分和桥接往返计时。
八条串行短程探针与未插桩轨迹的动作、全部原生指标、信道/报告哈希逐值一致。该探针不替换正式质量实验的冻结二进制。</p>
{table(native_timing)}
<p>native_roundtrip_wall_ms 包括下一状态仿真、桥接和宿主检查；host_wrapper_cpu_ms 只含宿主 CPU；channel_generation_wall_ms 和 svd_wall_ms 含原有 GPU 同步等待。
它们是不同范围，不能直接相加；探针使用全部 480 个热身后 TTI，是共享 GPU 上的短程诊断。正式串行搜索耗时仍以上面的 72 同状态对照为主。</p>
<p>探针的 scheduler_cpu_ms 沿用完整在线策略循环范围，包含原始输入提取、HOL／权重计算和调度；离线重放使用保存权重，且不含 HOL 扫描。
两张表还使用不同状态与运行环境，不能将探针 CPU 时间与离线 CPU 时间相除作为加速比。</p>
<p>独立 profiling 还把输入重建、评价模型转换和搜索拆开；合成对角 H 的转换属于调度端计算，与原生随机信道生成不同。
中间 oracle 阶段的转换包含从 H 提取功率，E/F 阶段的滤波或上报值已给定；测量模拟耗时另记，不能把 oracle 转换时间当作真实空口反馈开销。</p>
<h2>边界与核验</h2>
<p>RSRP 是仿真功率代理，采用线性功率、参考/数据端口已校准假设，未模拟测量噪声、量化及波束不匹配；不宣称是标准上报。
服务链路 CSI、原生 PFQ、MCS/层数和 OLLA 保留，HARQ 关闭。各向同性替代同时丢失空间相关和方向间不同功率；它不只是把非对角元置零。</p>
<p>每条轨迹 2 秒共同预热、2 秒策略稳定、10 秒测量、0.2 秒随访。完整包吞吐按测量期到达包的最终完整交付计算；P95/P99 只针对成功包，过期率和其它未交付率另列。
测量期到达包均已有终态。三 seed 为探索性证据，不足以保证所有信道和负载下同样结论。</p>
<p>原始信道逐步未被修改；216 个正式保存状态由独立 NumPy PHY 核验执行 SINR/BLER。相同 seed 的拓扑、到达业务、共同预热动作和保存信道一致；新轨迹与 RSRP 端点的完整信道及上报哈希流一致。
24 条短程闭环通过，旧原版和 RSRP 短程端点复现。72 个端点状态 × 6 种模型的全部访问轮次通过全枚举和证书核验。</p>
<p>耗时是在完整质量实验结束后固定一个 CPU、每状态每模型重复 6 次、轮换顺序测量；关闭 observer/profiling。
包含状态重建、PFQ 解码、近似转换和完整搜索，不含 GPU PFQ、桥接、HOL、测量生成与传输。独立 profiling 用于归因，在线并发耗时不用于加速比。</p>
<p>沿用的保存时刻为 TTI 8001、18001、28000，对应报告年龄为 20、20、19 TTI，处于周期保持较久的相位。
同状态计时和首轮诊断是这批稀疏状态的受控结果，不代表全上报相位的实时保证；闭环质量及在线候选计数覆盖全部 20000 个测量 TTI。</p>
<h2>数据与复现</h2><ul>
<li><a href="adjacent_effects.csv">相邻效果 CSV</a> · <a href="quality_summary.csv">六阶段均值 CSV</a> · <a href="endpoint_comparison.csv">端点对照</a></li>
<li><a href="analysis/per_seed.csv">逐 seed 指标</a> · <a href="analysis/paired.csv">完整配对区间</a> · <a href="analysis/pairing.json">输入配对核验</a></li>
<li><a href="replay/mechanism_raw.csv">逐状态机制数据</a> · <a href="replay/rounds_raw.csv">逐轮搜索记录</a> · <a href="replay/timing_raw.csv">串行计时</a> · <a href="replay/validation.json">全枚举核验</a></li>
<li><a href="replay/paired_intervals.csv">耗时与机制计数的 seed 配对区间</a> · <a href="scheduling_per_seed.csv">逐 seed 资源复用</a></li>
<li><a href="replay/certificate_per_tti.csv">审计 TTI 的精评/证书必评数</a> · <a href="replay/certificate_rounds.csv">逐轮上界核验</a> · <a href="native_timing/validation.json">仿真计时探针核验</a></li>
<li><a href="mechanism_code_audit.json">上界、MCS/OLLA 与失败字节出队的代码证据</a></li>
<li><a href="campaign/protocol.json">冻结协议</a> · <a href="source/README.rst">说明与复现命令</a> · <a href="experiment_bundle.zip">报告和复现源码包</a></li></ul></body></html>'''
    (root/'report.html').write_text(body,encoding='utf-8')
    source=root/'source';source.mkdir(exist_ok=True)
    for path in Path(__file__).parent.iterdir():
        if path.suffix in ('.py','.rst'):shutil.copy2(path,source/path.name)
    protocol=json.loads((root/'campaign/protocol.json').read_text())
    for name,digest in protocol['sources'].items():assert sha(ROOT/name)==digest,name
    artifacts=[root/'report.html',root/'quality.png',root/'mechanism.png',root/'adjacent_effects.csv',
        root/'endpoint_comparison.csv',root/'quality_summary.csv',analysis/'per_seed.csv',analysis/'paired.csv',
        analysis/'pairing.json',replay/'validation.json',replay/'cost_decomposition.csv']
    artifacts += [root/'ue_diagnostics.png',root/'ue_diagnostics/validation.json',root/'ue_diagnostics/per_ue.csv',
        root/'ue_diagnostics/sinr_per_ue.csv',root/'ue_diagnostics/per_tti_candidates.csv.gz',
        root/'native_timing/validation.json',root/'native_timing/summary.csv',replay/'certificate_per_tti.csv']
    artifacts += [root/'mechanism_code_audit.json',root/'channel_information_summary.csv',
        root/'ue_diagnostics/expiry_hotspots.csv',root/'ue_diagnostics/ue_distribution_summary.csv',
        root/'ue_diagnostics/olla_saturation_per_ue.csv',replay/'common_root_sinr_per_ue.csv',
        root/'native_timing/build/manifest.json']
    write(root/'result_manifest.json',dict(completed=True,runs=72,new_runs=48,reused_runs=24,seeds=[41,42,43],
        interpretation='ordered conditional ablation, not order-independent effects',
        artifacts={str(p.relative_to(root)):sha(p) for p in artifacts},
        source_sha256={p.name:sha(p) for p in source.iterdir() if p.is_file()}))
    # Compact review package: native logs and packet cohorts stay in the workspace.
    include=[p for p in root.iterdir() if p.suffix in ('.html','.png','.csv','.json','.log')
        and p.name not in ('postprocessing_status.json','report.log','postprocess.log')
        and not p.name.startswith(('interim_','partial_'))]
    include+=list(source.glob('*'))+list(replay.glob('*'))
    include+=[p for p in (root/'ue_diagnostics').glob('*') if p.is_file()]
    include+=[p for p in (root/'native_timing').glob('*') if p.is_file()]
    include+=[root/'native_timing/build/manifest.json',root/'native_timing/build/instrumented_main.cpp']
    include+=[p for p in analysis.glob('*') if p.is_file()]
    include+=[root/'campaign/protocol.json',root/'campaign/completion.json']
    with zipfile.ZipFile(root/'experiment_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in include:
            if p.is_file():z.write(p,str(p.relative_to(root)))
    print(adjacent.to_string(index=False));print(comparisons.to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);main(parser.parse_args())

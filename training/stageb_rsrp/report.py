"""Static research figures and a self-contained HTML evidence report."""
import argparse
import html
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from training.stageb_ppo.native import ROOT, write, sha

LABELS={'deficit':'DPLS','mlwdf_pf':'DA-PLS','rsrp_deficit':'DPLS + RSRP','rsrp_mlwdf_pf':'DA-PLS + RSRP'}
ORDER=list(LABELS)
COLORS=['#3e638f','#268878','#9cafd0','#98c6b7']


def table(frame):
    return frame.to_html(index=False,border=0,escape=True,float_format=lambda x:f'{x:.4f}')


def figures(root, df):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','axes.titleweight':'bold'})
    metrics=[('payload_mbps','Whole-packet throughput (Mbps)'),('delay_p95_ms','Successful-packet P95 (ms)'),
             ('expiry_pct','Expired packets (%)')]
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,scene in enumerate(['boundary','uniform']):
        for col,(metric,label) in enumerate(metrics):
            ax=axes[row,col];subset=df[df.scene==scene]
            means=[subset[subset.arm==a][metric].mean() for a in ORDER]
            ax.bar(range(4),means,color=COLORS,width=.65,zorder=2)
            for i,arm in enumerate(ORDER):
                values=subset[subset.arm==arm].sort_values('seed')[metric].to_numpy()
                ax.scatter(i+np.linspace(-.13,.13,len(values)),values,c='#293241',s=18,zorder=3)
            ax.set_xticks(range(4),['DPLS','DA-PLS','DPLS\nRSRP','DA-PLS\nRSRP'])
            ax.set_ylabel(label);ax.set_title(('Boundary08' if scene=='boundary' else 'Uniform')+' | '+label.split(' (')[0])
            ax.grid(axis='y',alpha=.2,zorder=0);ax.set_ylim(bottom=0)
    fig.suptitle('Original-channel execution | 3 paired seeds | 10 s measurement per run',fontsize=14)
    fig.savefig(root/'quality.png',dpi=170);plt.close(fig)
    timing=pd.read_csv(root/'timing/timing_per_seed.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,scene in zip(axes,['boundary','uniform']):
        means=[]
        for objective,mode in [('deficit','exact'),('mlwdf_pf','exact'),('deficit','rsrp'),('mlwdf_pf','rsrp')]:
            values=timing[(timing.scene==scene)&(timing.objective==objective)&(timing['mode']==mode)].cpu_ms
            means.append(values.mean())
        ax.bar(range(4),means,color=COLORS,width=.65)
        ax.set_xticks(range(4),['DPLS','DA-PLS','DPLS\nRSRP','DA-PLS\nRSRP'])
        ax.set_title('Boundary08' if scene=='boundary' else 'Uniform');ax.set_ylabel('CPU ms / decision (same saved states)')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Serial CPU timing; exact and RSRP evaluated on identical inputs',fontsize=13)
    fig.savefig(root/'timing.png',dpi=170);plt.close(fig)


def main(args):
    root=Path(args.root).resolve();analysis=root/'analysis';timing=root/'timing'
    for path,field in [(analysis/'completion.json','completed'),(analysis/'pairing.json','passed'),
                       (timing/'validation.json','passed'),(root/'channel_audit.json','passed')]:
        assert json.loads(path.read_text())[field]
    df=pd.read_csv(analysis/'per_seed.csv')
    assert len(df)==24 and set(df.seed)=={41,42,43}
    cols=['payload_mbps','goodput_mbps','delay_mean_ms','delay_p95_ms','delay_p99_ms','outcome_p95_ms',
          'expiry_pct','success_pct','other_nondelivery_pct','sample_sinr_bias_db','sample_sinr_mae_db']
    means=df.groupby(['scene','arm'],as_index=False)[cols].mean()
    tc=pd.read_csv(timing/'timing_summary.csv')
    changes=[]
    for scene in ['boundary','uniform']:
        for objective in ['deficit','mlwdf_pf']:
            a=means[(means.scene==scene)&(means.arm=='rsrp_'+objective)].iloc[0]
            b=means[(means.scene==scene)&(means.arm==objective)].iloc[0]
            t=tc[(tc.scene==scene)&(tc.objective==objective)].set_index('mode')
            changes.append(dict(scene=scene,algorithm=LABELS[objective],
                throughput_change_pct=100*(a.payload_mbps/b.payload_mbps-1),
                p95_original_ms=b.delay_p95_ms,p95_rsrp_ms=a.delay_p95_ms,
                expiry_original_pct=b.expiry_pct,expiry_rsrp_pct=a.expiry_pct,
                expiry_change_pp=a.expiry_pct-b.expiry_pct,
                serial_cpu_original_ms=t.loc['exact','cpu_ms'],serial_cpu_rsrp_ms=t.loc['rsrp','cpu_ms'],
                serial_cpu_change_pct=100*(t.loc['rsrp','cpu_ms']/t.loc['exact','cpu_ms']-1)))
    changes=pd.DataFrame(changes);changes.to_csv(root/'comparison.csv',index=False)
    means.to_csv(root/'quality_summary.csv',index=False)
    figures(root,df)
    show=means.copy();show['arm']=show.arm.map(LABELS)
    paired=pd.read_csv(analysis/'paired.csv')
    paired=paired[(paired.arm=='rsrp_'+paired.control)&paired.metric.isin(['payload_mbps','delay_p95_ms','expiry_pct'])]
    own=pd.read_csv(timing/'own_trajectory_summary.csv')
    online=df.groupby(['scene','arm'],as_index=False)[['wall_ms_mean','cpu_ms_mean','mmse_misses_mean','candidate_evaluations_mean']].mean()
    diagnostic_html=''
    if (root/'diagnostics/summary.csv').exists():
        assert json.loads((root/'diagnostics/validation.json').read_text())['passed']
        diagnostic=pd.read_csv(root/'diagnostics/summary.csv')
        diagnostic=diagnostic[['scene','arm','spatial_only_sinr_bias_db','spatial_only_sinr_mae_db',
                               'wideband_sinr_bias_db','wideband_sinr_mae_db',
                               'reported_rsrp_sinr_bias_db','reported_rsrp_sinr_mae_db']]
        diagnostic_html='<h2>保存状态中的误差机制</h2><p>固定同一已选动作，分别只去掉干扰空间方向、再去掉频域差异、以及使用实际延迟滤波的 RSRP 代理。前两项使用理想原始信道信息，只是离线诊断，并非另外两种闭环实验。误差相对原始 MMSE-IRC；不同列的误差不能相减解释为可加的因果贡献。</p>'+table(diagnostic)
    statement=[]
    for row in changes.itertuples():
        statement.append(f"{row.scene} 的 {row.algorithm}：完整包吞吐变化 {row.throughput_change_pct:+.2f}%；"
            f"成功包 P95 {row.p95_original_ms:.2f}→{row.p95_rsrp_ms:.2f} ms；"
            f"过期率 {row.expiry_original_pct:.3f}%→{row.expiry_rsrp_pct:.3f}%；"
            f"同状态串行 CPU 耗时变化 {row.serial_cpu_change_pct:+.2f}%。")
    # Standard references identify measurement semantics, not proof of this proxy's accuracy.
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>邻区 RSRP 干扰近似对照实验</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;color:#203047;max-width:1200px;margin:40px auto;padding:0 24px}}
h1,h2{{line-height:1.3}}h2{{margin-top:36px}}table{{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}}
th,td{{border-bottom:1px solid #dbe2eb;padding:9px;text-align:right;white-space:nowrap}}th{{background:#edf3f8}}
img{{width:100%;height:auto}}code{{background:#edf3f8;padding:2px 4px}}.note{{padding:16px;background:#f3f6fa;border-left:4px solid #268878}}
a{{color:#166b98}}</style><body>
<h1>邻区 RSRP 干扰近似：三小区配对实验</h1><p>2026-11-09 · Boundary08 / Uniform · seeds 41–43</p>
<p class="note"><b>代码核对结论：</b>现有 DPLS 与 DA-PLS 使用包含路径损耗、阴影衰落和逐次更新 Rayleigh 小尺度衰落的瞬时复数信道，
并按分源协方差 H Hᴴ 计算候选动作下的 MMSE-IRC SINR。它们并非只使用大尺度增益。</p>
<h2>结果</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in statement)}</ul>
<p>以下是逐 seed 指标的等权均值；吞吐指测量窗口到达、最终完整成功交付的包载荷。正的耗时变化表示更慢。</p>
{table(changes)}<img src="quality.png" alt="两场景吞吐、尾时延和过期率；点表示各 seed">
<h2>实现与实验边界</h2>
<p>仿真器没有标准邻区 RSRP 上报，本实验使用理想参考信号功率代理：取 TX 端口 0 的 |H|²，在 17 个 PRG 和 4 个接收分支上作线性平均；
以 100 ms 时间常数作因果 EMA，每 10 ms 上报，延迟 0.5 ms。调度评价器只接收每 UE、每邻区一个标量 β，以 Ĉ=4βI₄ 表示该源在全部 PRG 上的干扰。
β 使用仿真器已缩放的功率单位，不能当成 dBm；假设参考信号与数据端口功率可准确换算，未加入测量噪声、量化和波束不匹配。</p>
<p>服务链路瞬时 CSI、原生 PFQ 初选、MCS/OLLA、层数限制和业务权重保留。执行端始终使用原始 H，按原始 PHY 产生 SINR、BLER、CRC 与队列更新。
本实验隔离的是邻区反事实干扰评价精度，不是一个已移除全部瞬时信息或完成真实反馈接口的部署版本。</p>
<p>RSRP 是参考信号功率测量，不能恢复完整空间协方差；代理的具体接收分支平均方式也不宣称符合标准 RSRP 上报规则。
参考：<a href="https://www.etsi.org/deliver/etsi_ts/138200_138299/138215/18.02.00_60/ts_138215v180200p.pdf">3GPP TS 38.215 §5.1.1–5.1.2</a>。</p>
<h2>评测口径与全部质量指标</h2>
<p>每条轨迹包含 2 s 共同 PFQ 预热、2 s 策略稳定、10 s 测量和 0.2 s 随访；所有测量期到达包均已有终态。
原始两算法复用同版本已核验的 12 条轨迹，新运行 12 条 RSRP 轨迹。保存哈希、配对验证和原生执行检查，不重写旧实验。
P95/P99 仅针对成功包；outcome P95 包含过期和 CRC 损坏结局，避免只看成功包而忽略丢包。
过期率与 CRC 等其它未交付比例分列。三 seed 仅用于探索性判断，不能据此证明普遍等效或不劣。</p>
{table(show)}{diagnostic_html}
<h2>配对差值和不确定性</h2><p>RSRP 版本减对应原始版本；nominal 95% 配对 t 区间，n=3，未作多重比较修正。
吞吐单位 Mbps，时延单位 ms，过期率差值单位百分点。区间跨零不表示已经证明两者等价。</p>{table(paired)}
<h2>计算开销</h2><img src="timing.png" alt="相同保存状态上的串行 CPU 耗时">
<p>同一搜索器和同一矩阵内核用于两版本，未额外开发标量专用 MMSE 内核。每个保存状态都执行精确与近似两种评价，
交替顺序重复 5 次、固定一个 CPU；两条轨迹的状态等权纳入。计时包含状态重建、PFQ 解码、近似模型转换和物理搜索，
不含 GPU 原生 PFQ、桥接、HOL 扫描和测量传输。源状态的权重固定，因此这是干扰评价替换的受控耗时比较。</p>
{table(tc)}<p>各算法自身轨迹的保存状态耗时如下，用于补充说明闭环状态变化对计算量的影响：</p>{table(own)}
<p>以下在线耗时含并发争用，仅作描述，不能直接相除作为加速比：</p>{table(online)}
<h2>证据与复现</h2><p>单元测试覆盖报告延迟、滤波因果性、信息隔离、协方差归一化和近似搜索的独立全枚举对照。
八条短闭环试验先通过；正式状态再用独立 NumPy 物理模型核对调度预测和原始执行。
原始信道、预热动作、到达业务及拓扑配对通过；精确与近似版本的物理预测差异属于实验结果，不被误记为执行错误。</p>
<ul><li><a href="channel_audit.json">信道来源核查</a></li><li><a href="campaign/protocol.json">冻结协议与源码哈希</a></li>
<li><a href="analysis/per_seed.csv">逐 seed 指标</a> · <a href="analysis/paired.csv">全部配对差值</a> · <a href="analysis/pairing.json">配对核验</a></li>
<li><a href="timing/timing_raw.csv">原始串行计时</a> · <a href="timing/validation.json">计时与决策复现核验</a></li>
<li><a href="comparison.csv">主要对照表 CSV</a> · <a href="source/README.rst">实现说明与复现命令</a></li></ul></body></html>'''
    (root/'report.html').write_text(body,encoding='utf-8')
    source=root/'source';source.mkdir(exist_ok=True)
    for path in Path(__file__).parent.iterdir():
        if path.suffix in ('.py','.rst'):shutil.copy2(path,source/path.name)
    write(root/'result_manifest.json',dict(completed=True,runs=24,new_runs=12,reused_runs=12,
        seeds=[41,42,43],code_audit='instantaneous H, not large-scale-only',
        artifacts={str(p.relative_to(root)):sha(p) for p in [root/'report.html',root/'quality.png',root/'timing.png',root/'comparison.csv',
           analysis/'per_seed.csv',analysis/'paired.csv',analysis/'pairing.json',timing/'validation.json']},
        source_sha256={p.name:sha(p) for p in source.iterdir() if p.is_file()}))
    print(changes.to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);main(parser.parse_args())

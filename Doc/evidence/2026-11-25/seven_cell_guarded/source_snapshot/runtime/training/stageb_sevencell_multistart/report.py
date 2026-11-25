"""Four-policy comparison using unchanged packet measurement conventions."""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .policy import ARM

def main(root):
    root=Path(root);out=root/'analysis';df=pd.read_csv(out/'per_seed.csv');branch=pd.read_csv(out/'branch_selection.csv')
    assert len(df)==12 and set(df.seed)=={41,42,43}
    order=['pfq','deficit','mlwdf_pf',ARM];labels=['PFQ','DPLS','Original DA-PLS','Three-branch DA']
    metrics=[('payload_mbps','Complete-packet payload (Mbps)'),('delay_mean_ms','Successful-packet mean delay (ms)'),
        ('delay_p95_ms','Successful-packet P95 (ms)'),('expiry_pct','Expired packets (%)')]
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,(metric,title) in zip(axes.flat,metrics):
        z=[df[df.arm==arm][metric].to_numpy() for arm in order]
        ax.bar(range(4),[x.mean() for x in z],color=['#7a8793','#2978b5','#e18835','#2c9870'],yerr=[x.std(ddof=1) for x in z],capsize=4,alpha=.85)
        for i,x in enumerate(z):ax.scatter(np.full(3,i)+[-.12,0,.12],x,s=20,c='black',zorder=4)
        ax.set_xticks(range(4),labels,fontsize=9);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Seven-cell Boundary08 / 84 UEs / seeds 41, 42, 43\nMean ± sample SD; 10 s measured per seed')
    fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)
    mean=df.groupby('arm').mean(numeric_only=True)
    text=['七小区三分支DA-PLS闭环实验','='*40,'',
        '7小区、84UE、seeds 41/42/43；4000共同PFQ预热＋4000稳定＋20000测量＋400随访，测量10秒。',
        '场景、物理模型、业务、权重、邻域及单分支停止规则均沿用原七小区boundary08。',
        '每个时隙在同一当前状态分别运行：DPLS、原DA-PLS、从DPLS终点起步的DA。',
        '第三分支强制采用DPLS起点；三者共用由原始PFQ计算的90%计划出队下界。',
        '最终按DA整数目标sum round_even(1e6*w*expected_service)最大值择优；完全相同分数优先原DA、再DPLS起点DA、最后DPLS。',
        '仅执行最终动作并更新同一条轨迹的队列、历史成功服务和链路状态；没有复用其它轨迹的动作或未来信息。',
        '新跑3条闭环轨迹，原PFQ/DPLS/DA九条结果复用；逐seed匹配拓扑、业务、全部信道、共同预热及首个策略输入。','',
        '逐seed等权均值±样本标准差：','指标 | PFQ | DPLS | 原DA-PLS | 三分支DA']
    names={'payload_mbps':'完整包吞吐(Mbps)','goodput_mbps':'成功字节吞吐(Mbps)','delay_mean_ms':'成功包平均时延(ms)',
        'delay_p95_ms':'成功包P95(ms)','expiry_pct':'过期率(%)','outcome_mean_ms':'全部结局平均时延(ms)',
        'other_nondelivery_pct':'CRC损坏未交付(%)','ue_payload_p5_mbps':'UE吞吐P5(Mbps)','jain_payload':'吞吐Jain公平性',
        'worst_ue_expiry_pct':'最差UE过期率(%)','occupancy_pct':'PRG占用率(%)','inr_pooled_db':'已调度链路INR(dB)'}
    for m,label in names.items():text.append(label+' | '+' | '.join(f'{df[df.arm==a][m].mean():.4f} ± {df[df.arm==a][m].std():.4f}' for a in order))
    text+=['','三分支方案相对原结果（均值之比/均值之差）']
    for a,label in zip(order[:3],labels[:3]):
        x=mean.loc[ARM];b=mean.loc[a]
        text.append(f'对{label}：完整包吞吐{100*(x.payload_mbps/b.payload_mbps-1):+.3f}%；成功包平均时延{x.delay_mean_ms-b.delay_mean_ms:+.3f}ms；P95{x.delay_p95_ms-b.delay_p95_ms:+.3f}ms；过期率{x.expiry_pct-b.expiry_pct:+.3f}个百分点。')
    text+=['','正式测量窗口分支选择']
    for _,b in branch.iterrows():text.append(f'seed{int(b.seed)}：原DA {b.original_da_selected_pct:.3f}%，DPLS起点DA {b.from_dpls_da_selected_pct:.3f}%，DPLS {b.dpls_selected_pct:.3f}%；DA目标严格改善时隙{b.da_score_strict_improvement_pct:.3f}%。')
    text+=['','统计口径与限制',
        '完整包吞吐按测量窗口到达的包追踪至终态，仅计全部分段成功的完整包；400TTI随访，所有cohort无截尾。',
        '成功包时延只包含完整成功交付包；过期率不包含其它CRC损坏未交付，故须同时观察吞吐和过期。',
        'DA分数不劣于三分支的保证只针对同一状态的单步目标；不等于长期吞吐、时延或过期率必然改善。',
        '本实验仅3个seed。paired.csv提供未作多重比较校正的名义95%配对t区间；保存状态及TTI不视为独立seed。',
        '运行墙钟来自本轮环境，不与原并发环境直接计算公平加速比；三分支增加搜索成本。',
        '原七小区结果保留在deliverables/seven_cell_boundary08_20261105_v1。']
    (out/'report.txt').write_text('\n'.join(text)+'\n');print(out/'report.txt')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)

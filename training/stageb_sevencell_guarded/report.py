"""Four-policy comparison, paired differences and service protection diagnostics."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .policy import ARM

def main(root):
    out=Path(root)/'analysis';df=pd.read_csv(out/'per_seed.csv');branch=pd.read_csv(out/'branch_selection.csv')
    assert len(df)==12 and set(df.seed)=={41,42,43}
    order=['deficit','mlwdf_pf','da_three_branch',ARM]
    labels=['DPLS','Original DA','Original three-branch DA','New DA (0.1, 0.98)']
    metrics=[('payload_mbps','Complete-packet payload (Mbps)'),('delay_mean_ms','Successful-packet mean delay (ms)'),
             ('delay_p95_ms','Successful-packet P95 delay (ms)'),('expiry_pct','Expired packets (%)')]
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ax,(metric,title) in zip(axes.flat,metrics):
        values=[df[df.arm.eq(a)][metric].to_numpy() for a in order]
        ax.bar(range(4),[v.mean() for v in values],yerr=[v.std(ddof=1) for v in values],capsize=4,
               color=['#2879b4','#dc8535','#60a99a','#7953ac'],alpha=.85)
        for i,v in enumerate(values):ax.scatter(np.full(3,i)+[-.12,0,.12],v,s=20,c='black',zorder=4)
        ax.set_xticks(range(4),labels,fontsize=8);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Seven-cell Boundary08 / 84 UEs / seeds 41, 42, 43\nMean ± sample SD; 10 s measurement per seed')
    fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)
    means=df.groupby('arm').mean(numeric_only=True)
    lines=['七小区有界紧急度DA：lambda=0.1，beta=0.98','='*50,'',
        '同一七小区/84UE boundary08，seeds41/42/43；4000共同PFQ预热＋4000稳定＋20000正式测量＋400随访。每seed测量10秒。',
        '新权重：归一化[clip(2400/EMA,1,4)*(1+0.1*HOL_urgency)]，HOL_urgency沿用原因果FIFO定义。',
        '三分支：原权重DPLS；PFQ起步的新目标搜索；强制从DPLS终点起步的新目标搜索。后两支的全部搜索及最终排序均使用新权重。',
        '三个终点均保留原始PFQ的90%计划出队下界；最终先筛选总预期成功服务>=0.98*同状态DPLS总预期成功服务的候选，再按新目标择优。',
        '新目标为sum round_even(1e6*w_new*expected_success_bytes)。精确同分优先新目标冷启动、DPLS起点新目标、DPLS。',
        'beta保护仅放在最终候选过滤，未改各分支邻域、剪枝及停止规则；DPLS始终是可行候选。只执行选中动作并推进一条闭环轨迹。',
        '原DPLS、原DA、原三分支DA共9条轨迹作为冻结对照复用。逐seed验证拓扑、业务、全程信道、共同预热动作/原生指标和首个策略输入。','',
        '三个seed等权均值±样本标准差：','指标 | DPLS | 原DA | 原三分支DA | 新DA']
    names={'payload_mbps':'完整包吞吐Mbps','goodput_mbps':'成功字节吞吐Mbps','delay_mean_ms':'成功包平均时延ms',
        'delay_p50_ms':'成功包P50 ms','delay_p95_ms':'成功包P95 ms','success_le10ms_pct':'成功包10ms内交付比例%',
        'expiry_pct':'过期率%','other_nondelivery_pct':'CRC损坏未交付%','queue_mean_mb':'平均积压MB',
        'outcome_mean_ms':'全部包结局平均时延ms','ue_payload_p5_mbps':'UE吞吐P5 Mbps','jain_payload':'吞吐Jain公平性',
        'worst_ue_expiry_pct':'最差UE过期率%','occupancy_pct':'PRG占用率%','inr_pooled_db':'已调度链路INR dB'}
    for m,label in names.items():lines.append(label+' | '+' | '.join(f'{df[df.arm.eq(a)][m].mean():.4f} ± {df[df.arm.eq(a)][m].std():.4f}' for a in order))
    lines+=['','新方案相对原结果（均值之比/均值之差）']
    for a,label in zip(order[:3],labels[:3]):
        x=means.loc[ARM];b=means.loc[a]
        lines.append(f'对{label}：完整包吞吐{100*(x.payload_mbps/b.payload_mbps-1):+.3f}%；成功包平均时延{x.delay_mean_ms-b.delay_mean_ms:+.3f}ms；P95{x.delay_p95_ms-b.delay_p95_ms:+.3f}ms；过期率{x.expiry_pct-b.expiry_pct:+.3f}个百分点。')
    lines+=['','正式测量窗口分支与保护线']
    for x in branch.itertuples():
        lines.append(f'seed{x.seed}：新冷启动{x.new_cold_selected_pct:.3f}%，DPLS起点新搜索{x.new_from_dpls_selected_pct:.3f}%，DPLS{x.dpls_selected_pct:.3f}%；保护线改变选择{x.guard_changed_winner_pct:.3f}%；最低服务比{x.min_expected_service_ratio:.8f}；服务均值相对当前DPLS{x.selected_vs_dpls_expected_service_pct:+.3f}%。')
    lines+=['','口径与限制',
        '完整包吞吐：测量窗口到达的包追踪至终态，仅计全部分段成功的完整交付；400TTI随访，无cohort截尾。',
        '成功包时延仅包含完整成功交付包；过期率不包含其它CRC损坏未交付。三项指标须联合解释。',
        '每个策略时隙保证选中动作的预期成功服务>=同状态DPLS的98%，且新目标分数>=同状态DPLS；不意味着在另一条独立DPLS轨迹上有长期指标保证。',
        '只有3个seed。paired.csv列出名义95%配对t区间（未作多重比较校正）；时隙与保存状态不能当作更多独立seed。',
        '共享运行环境下墙钟成本仅为本轮描述，不与历史运行直接计算公平加速比。',
        '详细结果、每seed指标、分支频率及全部验证记录保存在本目录与campaign/。']
    (out/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(out/'report.txt',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)

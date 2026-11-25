"""Standalone plots and a concise Chinese experiment report."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .campaign import ARMS,LABELS

METRICS=[('payload_mbps','Complete-packet payload (Mbps)'),('expiry_pct','Expired packets (%)'),
 ('delay_mean_ms','Successful-packet mean delay (ms)'),('delay_p95_ms','Successful-packet P95 (ms)'),
 ('ue_payload_p5_mbps','UE payload P5 (Mbps)'),('inr_pooled_db','Scheduled-link INR (dB)')]

def main(root):
    root=Path(root);out=root/'analysis';df=pd.read_csv(out/'per_seed.csv');protocol=json.loads((root/'campaign/protocol.json').read_text())
    assert len(df)==9 and not protocol['smoke'];means=df.groupby('arm').mean(numeric_only=True)
    fig,axes=plt.subplots(2,3,figsize=(12.8,7.6),layout='constrained');colors=['#7a8793','#2978b5','#e18835']
    for ax,(metric,label) in zip(axes.flat,METRICS):
        z=[df[df.arm==arm][metric].to_numpy() for arm in ARMS]
        ax.bar(range(3),[x.mean() for x in z],color=colors,width=.58,alpha=.85,yerr=[x.std(ddof=1) for x in z],capsize=4)
        for i,x in enumerate(z):ax.scatter(np.full(len(x),i)+np.linspace(-.12,.12,len(x)),x,s=20,c='black',zorder=4)
        ax.set_xticks(range(3),[LABELS[a] for a in ARMS]);ax.set_title(label,fontsize=11);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('7 cells / 84 UEs · Boundary08 · seeds 41, 42, 43\nMean ± sample SD; dots are individual seeds',fontsize=14)
    fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,seed in zip(axes,protocol['seeds']):
        path=root/'campaign'/f'pfq_s{seed}';top=pd.read_csv(path/'ue_topology.csv');bs=top[top.node_type=='bs'];ue=top[top.node_type=='ue'].sort_values('ue_id');serving=np.load(path/'serving.npy')
        ax.scatter(ue.x_m,ue.y_m,c=serving,cmap='tab10',vmin=0,vmax=9,s=15)
        ax.scatter(bs.x_m,bs.y_m,c=bs.cell_id,cmap='tab10',vmin=0,vmax=9,s=90,marker='^',edgecolors='black')
        for _,b in bs.iterrows():ax.text(b.x_m+25,b.y_m+25,str(int(b.cell_id)),fontsize=9)
        for _,u in ue.iterrows():
            b=bs[bs.cell_id==serving[int(u.ue_id)]].iloc[0]
            ax.plot([b.x_m,u.x_m],[b.y_m,u.y_m],color='gray',alpha=.12,lw=.4)
        ax.set_aspect('equal');ax.set_title(f'Seed {seed}');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    fig.suptitle('Seven-cell boundary placement; colors/lines show actual strongest-cell association')
    fig.savefig(out/'topology.png',dpi=180);fig.savefig(out/'topology.pdf');plt.close(fig)
    text=['七小区 boundary08 同频干扰对比实验','='*44,'',
        'PFQ、DPLS（用户请求中的 DLPS，缺口权重版本）、DA-PLS（M-LWDF-PF 时延感知权重）。',
        'seeds 41、42、43；每条4000 TTI共同PFQ预热＋4000稳定＋20000测量＋400随访，测量10秒。',
        '7小区（中心＋六邻区），每小区名义生成12 UE，总84 UE；实际服务关联仍按strongest。',
        '站间距1000m、半径参数500m、17PRG×16PRB/小区、4T4R、30kHz、0.5ms TTI、同频复用。',
        '每UE到达率0.8包/TTI、3000B/包、TTL200ms；理论总负载3225.6Mbps。',
        '边界布点沿相邻公共Voronoi边，切向范围±280m、向本小区内缩0–20m；中央每邻边2UE，外围每邻边4UE。',
        '其余沿用三小区boundary08：全向、阴影3dB、连续Rayleigh、修正后PHY、固定MCS/OLLA、无HARQ。',
        '所有搜索从原始PFQ出发，共用90%计划出队下界、相同邻域、阈值和规范tie；没有训练或重新调参。','',
        '下表为逐seed等权均值±样本标准差；P95先在seed内求值。','']
    names={'payload_mbps':'完整包吞吐(Mbps)','goodput_mbps':'成功字节吞吐(Mbps)','payload_per_cell_mbps':'每小区完整包吞吐(Mbps)',
        'expiry_pct':'过期率(%)','other_nondelivery_pct':'CRC损坏未交付(%)','delay_mean_ms':'成功包平均时延(ms)',
        'delay_p95_ms':'成功包P95(ms)','outcome_mean_ms':'全部结局平均时延(ms)','outcome_p95_ms':'全部结局P95(ms)',
        'ue_payload_p5_mbps':'UE吞吐P5(Mbps)','jain_payload':'完整包吞吐Jain公平性','worst_ue_expiry_pct':'最差UE过期率(%)',
        'inr_pooled_db':'已调度链路INR(dB)','sinr_mean_db':'已发送TB平均SINR(dB)','tb_error_pct':'TB错误率(%)','occupancy_pct':'PRG占用率(%)'}
    text.append('指标 | PFQ | DPLS | DA-PLS')
    for m,label in names.items():
        text.append(label+' | '+' | '.join(f'{df[df.arm==a][m].mean():.4f} ± {df[df.arm==a][m].std(ddof=1):.4f}' for a in ARMS))
    text+=['','相对变化（均值之比）']
    for a,b in [('deficit','pfq'),('mlwdf_pf','pfq'),('mlwdf_pf','deficit')]:
        delta=100*(means.loc[a,'payload_mbps']/means.loc[b,'payload_mbps']-1)
        text.append(f'{LABELS[a]} 相对 {LABELS[b]}：完整包吞吐 {delta:+.2f}%；过期率 {means.loc[a,"expiry_pct"]-means.loc[b,"expiry_pct"]:+.3f} 个百分点；成功包P95 {means.loc[a,"delay_p95_ms"]-means.loc[b,"delay_p95_ms"]:+.3f} ms。')
    text+=['','口径与限制',
        '完整包吞吐只计算测量窗口内到达、所有分段成功并最终完整交付的包；400 TTI随访保证到达集合无截尾。',
        'outcome时延包含CRC损坏的MAC完成包与自然过期包；成功包时延只含完整成功包。',
        'INR按冻结PHY中的activity-only协方差H H^H，以实际PRG活动计算，除以4×噪声；先在seed内按已调度UE-PRG池化，再转dB。',
        '该口径不是额外引入干扰UE预编码/层数的新模型；同时改变用户组成和静默率会影响池化INR。',
        '九条轨迹逐seed核验相同拓扑、业务、全程原始信道和共同PFQ前缀；逐TTI核验动作、MCS/层/TBS/出队。',
        '独立NumPy物理评价对照原生传输；另付费完整枚举对照七小区保存状态，验证上界、缓存、每轮选择和终态。',
        '仅3个seed，paired.csv给出未作多重比较校正的名义95%配对t区间；结论是本场景的规模扩展检查。',
        '并发且共享GPU，运行墙钟不用于宣称公平加速比或实时部署达标。',
        '三小区旧结果仅作历史背景；小区增多会自然改变邻接关系与干扰强度，不能假设拓扑几何完全相同。','',
        '文件：per_seed.csv、aggregate.csv、paired.csv、comparison.png/pdf、topology.png/pdf；原始日志在campaign/。']
    (out/'report.txt').write_text('\n'.join(text)+'\n')
    print(out/'report.txt')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)

"""Publication figures, equal seed weights; never pool unequal packet cohorts."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .figures import plt,COLORS,save,select_case

NAMES={'pfq':'PFQ','rr':'RR','rrq':'RRQ','pf':'PF','equal':'等权','deficit':'缺口 κ0','ttl1':'κ1','ttl2':'κ2','exp_pf':'EXP-PF','mlwdf_pf':'DA-PLS'}

def main(args):
    root=Path(args.root);out=root/'figures';out.mkdir(exist_ok=True)
    agg=pd.read_csv(root/'analysis/aggregate.csv');per=pd.read_csv(root/'analysis/per_seed.csv')
    agg=agg[(agg.subset=='41-50')&(agg.n==10)]
    def bars(ax,scene,rate,arms,metric,title,ylabel):
        z=agg[(agg.scene==scene)&(agg.rate==rate)&(agg.metric==metric)].set_index('arm')
        if not set(arms)<=set(z.index):raise ValueError('Incomplete ten-seed figure inputs')
        means=z.loc[arms,'mean'].to_numpy();half=(z.loc[arms,'ci_high']-z.loc[arms,'mean']).to_numpy()
        ax.bar(np.arange(len(arms)),means,yerr=half,capsize=3,color=[COLORS[0] if a=='pfq' else COLORS[2] if a=='mlwdf_pf' else COLORS[1] for a in arms])
        ax.set_xticks(np.arange(len(arms)),[NAMES[a] for a in arms]);ax.set(title=title,ylabel=ylabel);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    arms=['pfq','deficit','mlwdf_pf']
    fig,axes=plt.subplots(2,2,figsize=(9,6),constrained_layout=True)
    for i,(scene,rate,title) in enumerate([('boundary',.8,'Boundary08'),('uniform',1.,'uniform / rate1.0')]):
        bars(axes[i,0],scene,rate,arms,'payload_mbps',title+'：完整包吞吐','Mbps')
        bars(axes[i,1],scene,rate,arms,'delay_p95_ms',title+'：成功包P95','ms')
    fig.suptitle('10个seed等权均值；误差线为均值95%区间',fontsize=12);save(fig,out,'main_results')
    if not args.core:
        arms=['equal','deficit','ttl1','ttl2','exp_pf','mlwdf_pf'];fig,axes=plt.subplots(1,3,figsize=(11,3.6),constrained_layout=True)
        for ax,m,t,y in zip(axes,['payload_mbps','delay_p95_ms','worst_ue_expiry_pct'],['整包吞吐','成功包P95','最差UE过期率'],['Mbps','ms','%']):
            bars(ax,'boundary',.8,arms,m,t,y);ax.tick_params(axis='x',rotation=35)
        save(fig,out,'weight_ablation')
        fig,axes=plt.subplots(1,3,figsize=(10,3.4),constrained_layout=True)
        for ax,metric,title in zip(axes,['payload_mbps','expiry_pct','delay_p95_ms'],['完整包吞吐 / Mbps','自然过期率 / %','成功包P95 / ms']):
            for arm,color in zip(['pfq','deficit','mlwdf_pf'],COLORS[:3]):
                z=agg[(agg.scene=='boundary')&(agg.arm==arm)&(agg.metric==metric)].sort_values('rate')
                assert len(z)==3
                ax.errorbar(z.rate/.8,z['mean'],yerr=z.ci_high-z['mean'],marker='o',capsize=3,label=NAMES[arm],color=color)
            ax.set(xlabel='相对Boundary08负载',ylabel=title,xticks=[1,1.4,1.6]);ax.grid(alpha=.2)
        axes[0].legend();save(fig,out,'load_sensitivity')
    main=per[(per.scene=='boundary')&(per.rate==.8)&per.arm.isin(['pfq','mlwdf_pf'])]
    assert len(main)==20
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
    for arm,color in [('pfq',COLORS[0]),('mlwdf_pf',COLORS[2])]:
        keys=main[main.arm==arm].key.tolist()
        for ax,field,title in zip(axes,['phy_sinr_db_cpu','mcs'],['实际发送SINR / dB','实际发送MCS']):
            hs=[pd.read_csv(root/'analysis'/k/f'{field}_hist.csv') for k in keys]
            mass=np.mean([h.mass.to_numpy() for h in hs],axis=0);h=hs[0]
            x=(h.left+h.right)/2 if field=='mcs' else h.right
            ax.step(x,np.cumsum(mass),where='post',color=color,label=NAMES[arm]);ax.set(xlabel=title,ylabel='seed等权累计分布',ylim=(0,1));ax.grid(alpha=.2)
    axes[0].legend();save(fig,out,'phy_distributions')
    fig,ax=plt.subplots(figsize=(7,3.5),constrained_layout=True);left=np.zeros(2)
    for n,label in enumerate(['全部空置','单小区活动','双小区活动','三小区活动']):
        values=[float(main[main.arm==arm][f'reuse_{n}_pct'].mean()) for arm in ['pfq','mlwdf_pf']]
        ax.barh([0,1],values,left=left,label=label,color=COLORS[n]);left+=values
    ax.set(yticks=[0,1],yticklabels=['PFQ','DA-PLS'],xlabel='同PRG上的活动小区数比例 / %',xlim=(0,100));ax.legend(loc='upper center',bbox_to_anchor=(.5,1.25),ncol=4,fontsize=8)
    save(fig,out,'reuse_distribution')
    fig,ax=plt.subplots(figsize=(8,3.6),constrained_layout=True)
    for arm,color in [('pfq',COLORS[0]),('mlwdf_pf',COLORS[2])]:
        frames=[pd.read_csv(root/'analysis'/k/'per_ue.csv') for k in main[main.arm==arm].key]
        values=np.sort(np.concatenate([f.expiry_pct.to_numpy() for f in frames]))
        ax.plot(values,np.arange(1,len(values)+1)/len(values),label=NAMES[arm],color=color)
    ax.set(xlabel='单UE自然过期率 / %',ylabel='累计UE比例',title='10 seed × 36 UE；每个seed与UE均等权');ax.grid(alpha=.2);ax.legend();save(fig,out,'ue_expiry')
    plot_case(root,out)

def plot_case(root,out):
    source=select_case(root)
    if source:
        case=source['case'];fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
        ax=axes[0];ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
        ax.text(.5,.92,f'同PRG联合移动：TTI {case["tti"]}，轮次 {case["iteration"]}',ha='center',weight='bold')
        for i,e in enumerate(case['edits']):
            y=.65-i*.25;before='静默' if e['old_ue']<0 else f'UE {e["old_ue"]}';after='静默' if e['new_ue']<0 else f'UE {e["new_ue"]}'
            ax.text(.02,y,f'小区 {e["cell"]} / PRG {e["prg"]}',va='center')
            ax.text(.48,y,before,ha='center',va='center',bbox=dict(boxstyle='round,pad=.4',fc='#E7EFF5',ec=COLORS[0]))
            ax.annotate('',xy=(.78,y),xytext=(.60,y),arrowprops=dict(arrowstyle='->',color=COLORS[1]))
            ax.text(.90,y,after,ha='center',va='center',bbox=dict(boxstyle='round,pad=.4',fc='#E7F2EE',ec=COLORS[2]))
        ax.text(.5,.13,f'计划出队：{case["dequeue_before"]:,} → {case["dequeue_after"]:,} B\n可行下界：{case["floor"]:,} B',ha='center',va='center')
        ax=axes[1];ues=case['ues'];values=[u['weighted_delta'] for u in ues]
        ax.bar([f'UE {u["ue"]}' for u in ues],values,color=[COLORS[2] if x>=0 else COLORS[3] for x in values]);ax.axhline(0,color='#697782',lw=.8)
        ax.set(ylabel='完整TB加权预期服务变化 / B',title=f'簇级净增益：{case["gain_q"]/1e6:,.3f} B');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.suptitle('同一动作前状态的反事实物理评价；UE与小区编号从0开始',fontsize=11)
        save(fig,out,'joint_move_case')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--core',action='store_true');main(p.parse_args())

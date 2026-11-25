"""Six publication figures plus the retained case, all exactly 6 inches wide."""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/report_fig_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
from fontTools.ttLib import TTCollection
BASE=Path(__file__).resolve().parents[1];DATA=BASE/'data';OUT=BASE/'figures';OUT.mkdir(exist_ok=True)
# Song-style fallback used by the local document renderer for the source's 宋体.
fontfile=BASE/'source/NotoSerifCJKsc-Regular.ttf'
if not fontfile.exists():TTCollection('/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc').fonts[2].save(fontfile)
fm.fontManager.addfont(str(fontfile));font=fm.FontProperties(fname=str(fontfile)).get_name()
plt.rcParams.update({'font.family':font,'font.size':9.5,'axes.titlesize':10,'axes.labelsize':9.5,'xtick.labelsize':9.5,'ytick.labelsize':9.5,'legend.fontsize':9.5,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#727B82','axes.linewidth':.6,'grid.color':'#DDE2E5','grid.linewidth':.6,'figure.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42,'svg.fonttype':'none'})
COL={'rr':'#CED2D5','rrq':'#8F979F','pf':'#91B9D3','pfq':'#284F73','deficit':'#CA914F','mlwdf_pf':'#008A79'}
NAME={'rr':'RR','rrq':'RRQ','pf':'PF','pfq':'PFQ','deficit':'DPLS','mlwdf_pf':'DA-PLS'}
BINCOL=['#238B65','#99CAA0','#F1D879','#E99A52','#A52335']
BINS=['le5','5_20','20_100','100_200','expired']
BINLABEL=['≤5 ms','5–20 ms','20–100 ms','100–200 ms','过期']
SCENES=[('boundary','Boundary08'),('uniform','Uniform')]
per=pd.read_csv(DATA/'per_seed_metrics.csv');means=per.groupby(['scene','arm']).mean(numeric_only=True)
binmean=pd.read_csv(DATA/'delay_bins_mean.csv');rankmean=pd.read_csv(DATA/'ue_rank_mean.csv')
manifest={}
def save(fig,name,conclusion):
    fig.canvas.draw()
    assert abs(fig.get_figwidth()-6)<1e-9
    text_min=min(t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text())
    assert text_min>=9.5,(name,text_min)
    # No tight bounding box: keep exported size exactly 6 inches.
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=300)
    manifest[name]={'width_inches':6,'height_inches':fig.get_figheight(),'minimum_font_pt':text_min,'conclusion':conclusion}
    plt.close(fig)
def clean(ax,axis='y'):
    ax.grid(axis=axis,zorder=0);ax.set_axisbelow(True)
    ax.tick_params(length=2.5,pad=3)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3))
def value(scene,arm,key):return float(means.loc[(scene,arm),key])
def main_metrics(arms,name,title):
    fig,axes=plt.subplots(2,3,figsize=(6,4.65))
    fig.subplots_adjust(left=.09,right=.985,bottom=.11,top=.80,wspace=.37,hspace=.64)
    fig.suptitle(title,y=.99,fontsize=11.5)
    fig.text(.5,.925,'柱：10个seed平均  ·  点：seed 41–50各次结果',ha='center',fontsize=9.5,color='#5F666C')
    metrics=[('payload_mbps','完整包吞吐（Mbps）'),('expiry_pct','过期率（%）'),('delay_mean_ms','成功包平均时延（ms）')]
    for i,(scene,slabel) in enumerate(SCENES):
        top=axes[i,0].get_position().y1
        fig.text(.09,top+.055,slabel,fontsize=10.5,color='#202E38')
        for j,(key,label) in enumerate(metrics):
            ax=axes[i,j];vals=[value(scene,a,key) for a in arms]
            ax.bar(range(len(arms)),vals,color=[COL[a] for a in arms],width=.69,edgecolor='white',linewidth=.5,zorder=2)
            maximum=0
            for n,(a,v) in enumerate(zip(arms,vals)):
                seeds=per[(per.scene==scene)&(per.arm==a)].sort_values('seed')[key].to_numpy()
                maximum=max(maximum,float(seeds.max()))
                ax.scatter(n+np.linspace(-.23,.23,10),seeds,s=9,facecolors='white',edgecolors='#3D474F',linewidths=.5,zorder=3)
                if key=='expiry_pct':
                    txt=f'{v:.3f}%' if (len(arms)==3 or v<1) else f'{v:.1f}'
                elif key=='delay_mean_ms':txt=f'{v:.2f}' if len(arms)==3 else f'{v:.1f}'
                else:txt=f'{v:.1f}'
                ax.annotate(txt,(n,v),xytext=(0,17 if key=='expiry_pct' and a=='mlwdf_pf' else 5),textcoords='offset points',ha='center',va='bottom',fontsize=9.5,color=COL[a] if key=='expiry_pct' and a=='mlwdf_pf' else '#14222C',arrowprops=dict(arrowstyle='-',color=COL[a],lw=.6,shrinkA=2,shrinkB=3) if key=='expiry_pct' and a=='mlwdf_pf' else None,zorder=5,bbox=dict(fc='white',ec='none',alpha=.85,pad=.1))
            ax.set_ylim(0,maximum*1.14 if maximum else 1)
            ax.set_xlim(-.55,len(arms)-.45)
            ax.set_xticks(range(len(arms)),[NAME[a] for a in arms])
            ax.set_title(label,pad=7)
            clean(ax)
    fig.text(.5,.015,'各场景使用独立纵轴范围；精确数值见正文表格。',ha='center',fontsize=9.5,color='#5F666C')
    save(fig,name,title)

def delay_bars(arms,name,title):
    fig,axes=plt.subplots(2,1,figsize=(6,4.55 if len(arms)==4 else 4.0))
    fig.subplots_adjust(left=.135,right=.80,top=.80,bottom=.20,hspace=.52)
    fig.suptitle(title,y=.99,fontsize=11.5)
    fig.legend(handles=[Patch(fc=c,label=l) for c,l in zip(BINCOL,BINLABEL)],loc='upper center',bbox_to_anchor=(.50,.93),ncol=5,frameon=False,columnspacing=.8,handlelength=1.1,handletextpad=.35)
    for ax,(scene,sl) in zip(axes,SCENES):
        ax.set_title(sl,loc='left',pad=8)
        y=np.arange(len(arms));left=np.zeros(len(arms))
        for k,(b,c) in enumerate(zip(BINS,BINCOL)):
            vals=[float(binmean[(binmean.scene==scene)&(binmean.arm==a)&(binmean.bin==b)].pct.iloc[0]) for a in arms]
            ax.barh(y,vals,left=left,height=.65,color=c,edgecolor='white',linewidth=.3,zorder=2)
            for n,v in enumerate(vals):
                if v>=11:
                    ax.text(left[n]+v/2,n,('≈100%' if 99.95<v<100 else f'{v:.1f}%'),ha='center',va='center',color='white' if k in [0,4] else '#253038',fontsize=9.5)
            left+=vals
        np.testing.assert_allclose(left,100,atol=1e-10)
        for n,a in enumerate(arms):
            expired=value(scene,a,'expiry_pct')
            ax.text(1.035,n,f'{expired:.3f}%',transform=ax.get_yaxis_transform(),va='center',ha='left',color='#8E2131' if expired else '#007D6F',fontsize=9.5)
        ax.text(1.035,1.04,'过期率',transform=ax.transAxes,ha='left',va='bottom',fontsize=9.5)
        ax.set(xlim=(0,100),yticks=y,yticklabels=[NAME[a] for a in arms],xticks=[0,25,50,75,100])
        ax.set_ylim(len(arms)-.5,-.5);ax.grid(axis='x',zorder=0);ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    axes[1].set_xlabel('包占比（%）')
    fig.text(.5,.025,'每个seed先统计包占比，再对10个seed等权平均。',ha='center',fontsize=9.5,color='#5F666C')
    save(fig,name,title)

def ue_rank():
    fig,axes=plt.subplots(2,1,figsize=(6,4.4),sharex=True)
    fig.subplots_adjust(left=.11,right=.97,top=.80,bottom=.18,hspace=.42)
    fig.suptitle('Boundary08：DA-PLS使全部UE过期率归零',y=.99,fontsize=11.5)
    fig.text(.5,.915,'10个seed内分别排序，再按名次等权平均；两幅纵轴不同',ha='center',fontsize=9.5,color='#5F666C')
    x=np.arange(1,37)
    pfq=rankmean[(rankmean.scene=='boundary')&(rankmean.arm=='pfq')].sort_values('rank').expiry_pct.to_numpy()
    dpls=rankmean[(rankmean.scene=='boundary')&(rankmean.arm=='deficit')].sort_values('rank').expiry_pct.to_numpy()
    da=rankmean[(rankmean.scene=='boundary')&(rankmean.arm=='mlwdf_pf')].sort_values('rank').expiry_pct.to_numpy()
    axes[0].scatter(x,pfq,s=22,c=COL['pfq'],label='PFQ',zorder=3)
    axes[0].set_ylim(0,pfq.max()*1.08);axes[0].legend(loc='upper left',frameon=False)
    axes[0].text(.50,.93,f'最差名次 {pfq[-1]:.3f}%',transform=axes[0].transAxes,ha='center',va='top',color=COL['pfq'],fontsize=9.5)
    axes[1].scatter(x-.13,dpls,s=24,c=COL['deficit'],marker='o',label='DPLS',zorder=3,clip_on=False)
    axes[1].scatter(x+.13,da,s=26,facecolors='white',edgecolors=COL['mlwdf_pf'],marker='s',linewidths=1.2,label='DA-PLS',zorder=4,clip_on=False)
    axes[1].set_ylim(0,dpls.max()*1.09);axes[1].legend(loc='upper left',ncol=2,frameon=False)
    axes[1].annotate(f'最差名次 {dpls[-1]:.3f}%',(36,dpls[-1]),(-7,-12),textcoords='offset points',ha='right',color='#986124',fontsize=9.5)
    axes[1].text(2,3.2,'DA-PLS：10 × 36个UE结果均为0.000%',color=COL['mlwdf_pf'],fontsize=9.5)
    for ax in axes:
        ax.set_ylabel('过期率（%）');ax.set_xlim(.3,36.7);clean(ax)
    axes[1].set_xticks([1,6,12,18,24,30,36]);axes[1].set_xlabel('UE过期率由低到高的排序名次')
    fig.text(.5,.035,'排序名次不代表固定UE编号；重合零值仅在横向轻微错开。',ha='center',fontsize=9.5,color='#5F666C')
    save(fig,'D_逐UE过期率','平均值之外，DA-PLS在本实验的全部UE上均无过期')

def tradeoff():
    fig=plt.figure(figsize=(6,5.2));gs=fig.add_gridspec(3,2,left=.14,right=.97,bottom=.09,top=.77,wspace=.51,hspace=1.28)
    fig.suptitle('Boundary08：更多PRG换来及时服务',y=.99,fontsize=11.5)
    fig.text(.30,.91,'代价',ha='center',fontsize=10.5);fig.text(.77,.91,'收益',ha='center',fontsize=10.5)
    fig.legend(handles=[Patch(fc=COL[a],label=NAME[a]) for a in ['deficit','mlwdf_pf']],loc='upper center',bbox_to_anchor=(.53,.89),ncol=2,frameon=False)
    def pair(ax,key,title,unit,dec=3):
        vals=[value('boundary',a,key) for a in ['deficit','mlwdf_pf']]
        ax.barh([1,0],vals,color=[COL[a] for a in ['deficit','mlwdf_pf']],height=.48,zorder=2)
        top=max(vals)*1.10;ax.set_xlim(0,top)
        for y,v in zip([1,0],vals):
            label=f'{v:.{dec}f}'+('%' if unit=='%' else '')
            if v>top*.23:ax.text(v-top*.025,y,label,ha='right',va='center',fontsize=9.5,color='white')
            else:
                ax.scatter([v],[y],s=16,c=COL['mlwdf_pf'],zorder=3,clip_on=False)
                ax.text(v+top*.035,y,label,ha='left',va='center',fontsize=9.5,color='#16392F')
        ax.set(yticks=[1,0],yticklabels=['DPLS','DA-PLS'],ylim=(-.5,1.5))
        ax.set_title(title,loc='left',pad=8);ax.set_xlabel(unit,labelpad=1);ax.xaxis.set_major_locator(MaxNLocator(3));ax.grid(axis='x',zorder=0);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)
    pair(fig.add_subplot(gs[0,0]),'prg_occupancy_pct','PRG占用率  +7.97个百分点','%',2)
    pair(fig.add_subplot(gs[1,0]),'sinr_tx_mean_db','平均SINR  −2.242 dB','dB')
    ax=fig.add_subplot(gs[0,1]);v=[value('boundary',a,'served_mbps') for a in ['deficit','mlwdf_pf']];inp=value('boundary','mlwdf_pf','offered_mbps')
    ax.axvline(inp,ls='--',lw=1,color='#596268',zorder=1)
    for y,a,w in zip([1,0],['deficit','mlwdf_pf'],v):
        ax.scatter(w,y,s=32,color=COL[a],zorder=3)
        ax.annotate(f'{w:.3f}',(w,y),(5,6) if y==1 else (-5,6),textcoords='offset points',ha='left' if y==1 else 'right',fontsize=9.5,color='#233139')
    ax.set(xlim=(1374,1383),ylim=(-.6,1.65),yticks=[1,0],yticklabels=['DPLS','DA-PLS'],xticks=[1374,1378,1382])
    ax.set_title(f'出队速率；虚线为输入\n输入 {inp:.3f} Mbps',loc='left',pad=5,fontsize=9.5)
    ax.set_xlabel('Mbps（局部范围）',labelpad=1);ax.tick_params(axis='y',length=0);ax.grid(axis='x',zorder=0)
    pair(fig.add_subplot(gs[1,1]),'queue_mean_mb','平均簇队列  −86.35%','MB')
    pair(fig.add_subplot(gs[2,1]),'worst_ue_expiry_pct','最差UE过期率归零','%')
    ax=fig.add_subplot(gs[2,0]);ax.axis('off')
    ax.text(0,.65,'出队与输入的缺口\n7.598 → 0.044 Mbps',ha='left',va='center',fontsize=10,color='#007D6F',linespacing=1.7)
    fig.text(.5,.015,'10个seed平均；输入为测量窗口实测到达速率。',ha='center',fontsize=9.5,color='#5F666C')
    save(fig,'E_代价与收益','链路质量略降，但出队速率接近输入、队列和最差UE风险下降')

def case_figure():
    case=json.loads((DATA/'joint_case.json').read_text())['case']
    fig,axes=plt.subplots(1,2,figsize=(6,3.65),gridspec_kw={'width_ratios':[1.06,1]})
    fig.subplots_adjust(left=.025,right=.98,top=.77,bottom=.22,wspace=.42)
    fig.suptitle('静默＋邻区换UE：同一输入下的联合调整',y=.99,fontsize=11)
    fig.text(.5,.90,'TTI 18001，第1轮；UE、小区与PRG编号从0开始',ha='center',fontsize=9.5,color='#5F666C')
    ax=axes[0];ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    for i,e in enumerate(case['edits']):
        y=.83-i*.34;before=f'UE {e["old_ue"]}';after='静默' if e['new_ue']<0 else f'UE {e["new_ue"]}'
        ax.text(.0,y+.115,f'小区{e["cell"]} / PRG {e["prg"]}',fontsize=9.5)
        ax.text(.19,y,before,ha='center',va='center',bbox=dict(boxstyle='round,pad=.3',fc='#F2F3F4',ec='#8F979F'),fontsize=9.5)
        ax.annotate('',xy=(.66,y),xytext=(.41,y),arrowprops=dict(arrowstyle='->',color='#56626A'))
        ax.text(.84,y,after,ha='center',va='center',bbox=dict(boxstyle='round,pad=.3',fc='#EAF5F1',ec=COL['mlwdf_pf']),fontsize=9.5)
    ax.text(.02,.13,f'计划出队：{case["dequeue_before"]:,} → {case["dequeue_after"]:,} B\n下界：{case["floor"]:,} B（满足）',ha='left',va='center',fontsize=9.5,linespacing=1.6)
    ax=axes[1];ues=case['ues'];vals=[u['weighted_delta'] for u in ues]
    ax.bar(range(4),vals,color=['#8F979F' if v<0 else COL['mlwdf_pf'] for v in vals],width=.64,zorder=2)
    for i,v in enumerate(vals):
        if v==0:ax.scatter(i,0,marker='_',s=100,color='#32424E',zorder=4)
        ax.annotate(f'{v:+.3f}' if v else '0.000',(i,v),(0,5 if v>=0 else -5),textcoords='offset points',ha='center',va='bottom' if v>=0 else 'top',fontsize=9.5,rotation=0)
    ax.set(xticks=range(4),xticklabels=[f'UE\n{u["ue"]}' for u in ues],ylim=(-1900,9400),yticks=[0,4000,8000])
    ax.axhline(0,color='#78838B',lw=.6);ax.set_ylabel('加权预期服务变化（B）',labelpad=2);ax.grid(axis='y',zorder=0);ax.set_axisbelow(True)
    ax.set_title('净增益 +13,136.148 B',fontsize=10,pad=9)
    fig.text(.5,.06,'UE 28调整前后预期服务均为0；零值用横线与数值明确标出。',ha='center',fontsize=9.5,color='#5F666C')
    save(fig,'案例_联合调整','联合调整的净增益为正；显式标注UE 28的零变化')

if __name__=='__main__':
    main_metrics(['rr','rrq','pf','pfq'],'A_传统基线核心指标','PFQ是最强传统基线，但边界场景仍有31.3%过期')
    delay_bars(['rr','rrq','pf','pfq'],'B7_传统基线时延分档','传统基线：边界场景中大量包等待至期限')
    main_metrics(['pfq','deficit','mlwdf_pf'],'C_协同算法核心指标','DA-PLS消除过期；Uniform下与DPLS基本持平')
    ue_rank();tradeoff()
    delay_bars(['pfq','deficit','mlwdf_pf'],'B9_协同算法时延分档','DA-PLS将时延压向短时延区间，并消除过期')
    case_figure()
    (OUT/'figure_manifest.json').write_text(json.dumps(dict(font=font,font_note='Source body uses 宋体; local renderer substitutes Noto Serif CJK SC. Figures use the identical Song-style local fallback.',algorithm_colors=COL,figures=manifest),ensure_ascii=False,indent=2)+'\n')
    print('Saved',len(manifest),'figures to',OUT)

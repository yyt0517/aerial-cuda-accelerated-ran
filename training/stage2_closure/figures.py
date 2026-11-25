"""Vector-friendly technical figures for the revised Word report."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib import font_manager
import numpy as np

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),
 'axes.unicode_minus':False,'font.size':10,'savefig.bbox':'tight','figure.dpi':160})
COLORS=['#165D86','#DD8B28','#238577','#AE3B4D','#6D619D']

def select_case(root):
    cases=[json.loads(p.read_text()) for p in sorted((Path(root)/'audit').glob('boundary_r0.8_mlwdf_pf_s*_tti*.json'))]
    cases=[c for c in cases if c.get('case')]
    tradeoffs=[c for c in cases if any(u['weighted_delta']<0 for u in c['case']['ues'])]
    return (tradeoffs or cases or [None])[0]

def box(ax,x,y,w,h,text,color=COLORS[0]):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',edgecolor=color,facecolor='#F0F6F9',lw=1.4))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)

def arrow(ax,start,end):ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color='#425667',lw=1.4))

def save(fig,out,name):
    fig.savefig(out/f'{name}.png',dpi=220);fig.savefig(out/f'{name}.pdf');plt.close(fig)

def diagrams(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    f,a=plt.subplots(figsize=(8.8,3.2));a.axis('off');a.set(xlim=(0,1),ylim=(0,1))
    box(a,.02,.52,.25,.33,'动作前状态\n队列 / HOL / 服务 EMA\n簇内信道 / 固定 MCS')
    box(a,.37,.52,.25,.33,'DA-PLS 决策\n时延权重 + 物理评价\n精确上界剪枝搜索')
    box(a,.72,.52,.25,.33,'原生 MAC / PHY 执行\nType-0 分配\nMMSE-IRC / BLER')
    arrow(a,(.27,.685),(.37,.685));arrow(a,(.62,.685),(.72,.685))
    box(a,.37,.05,.25,.25,'成功服务 / 分段失败\n队列终态 / OLLA 更新')
    arrow(a,(.85,.52),(.62,.175));arrow(a,(.37,.175),(.145,.52));save(f,out,'closed_loop')
    f,a=plt.subplots(figsize=(8.8,2.7));a.axis('off');a.set(xlim=(-.5,17.5),ylim=(-.8,3.3))
    for c in range(3):
        for p in range(17):
            a.add_patch(Rectangle((p,c),.9,.72,facecolor=COLORS[(p+c)%3],alpha=.22,edgecolor='white'))
            a.text(p+.45,c+.36,str(p),ha='center',va='center',fontsize=8)
        a.text(-.25,c+.36,f'小区 {c+1}',ha='right',va='center')
    a.text(8.5,3.05,'每小区：17 个项目 PRG × 16 PRB = 272 PRB',ha='center',weight='bold')
    a.text(8.5,-.55,'相同编号表示相同频域位置；每格选择一个 UE 或静默',ha='center')
    save(f,out,'resource_grid')
    f,a=plt.subplots(figsize=(8.8,2.5));a.axis('off');a.set(xlim=(0,1),ylim=(0,1))
    names=['异构图 + PPO\n学习路线探索','分源物理建模\n完整 TB 净收益','缺口权重 DPLS\n联合邻域搜索','DA-PLS\n时延权重 + 精确剪枝']
    for i,n in enumerate(names):
        box(a,.015+i*.25,.31,.215,.43,n,COLORS[i%4])
        if i<3:arrow(a,(.23+i*.25,.525),(.265+i*.25,.525))
    a.text(.5,.08,'路线选择以修复后的服务质量与计算成本证据为依据',ha='center')
    save(f,out,'evolution')
    f,a=plt.subplots(figsize=(8.8,4));a.axis('off');a.set(xlim=(0,1),ylim=(0,1))
    items=[(.03,.73,.26,.20,'原始 PFQ 初选\n确定本时隙出队下界'),(.37,.73,.26,.20,'构建联合编辑邻域\n计算廉价容量上界'),
      (.71,.73,.26,.20,'按上界排序\n复用仍有效的精确值'),(.71,.36,.26,.20,'证书尚未满足\n精评必要候选的完整 TB'),
      (.37,.36,.26,.20,'规范最大增益移动\n更新根方案与依赖失效'),(.03,.36,.26,.20,'无超过门槛的可行改进\n输出分配，原生执行')]
    for x,y,w,h,s in items:box(a,x,y,w,h,s)
    arrow(a,(.29,.83),(.37,.83));arrow(a,(.63,.83),(.71,.83));arrow(a,(.84,.73),(.84,.56))
    arrow(a,(.71,.46),(.63,.46));arrow(a,(.5,.56),(.5,.73));arrow(a,(.37,.46),(.29,.46))
    a.text(.5,.12,'所有剪枝必须有上界证书；不设置提前截断预算。\n等价于规范完整局部搜索，不是全局最优证明。',ha='center')
    save(f,out,'solver')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);diagrams(p.parse_args().output)

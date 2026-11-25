"""Preserve the original scenario narrative and report a complete KPI set per configuration."""
import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt

from . import build_report as base
from .figures import plt, COLORS, save, select_case
from .plot_results import plot_case

ENVIRONMENTS=[('boundary',.8,'Boundary08主负载'),('uniform',1.,'Uniform均匀布点'),
              ('boundary',1.12,'Boundary高负载×1.4'),('boundary',1.28,'Boundary高负载×1.6')]
LABEL={'rr':'RR','rrq':'RRQ','pf':'PF','pfq':'PFQ','deficit':'DPLS（缺口）','mlwdf_pf':'DA-PLS',
       'equal':'等权DPLS','ttl1':'κ-TTL（1）','ttl2':'κ-TTL（2）','exp_pf':'EXP-PF'}

class Writer(base.Writer):
    def p(self,s,style=None):
        return super().p(re.sub(r'(?<![0-9A-Za-z])-(?=\d)','−',s),style)
    def table(self,title,headers,rows):
        clean=lambda s:re.sub(r'(?<![0-9A-Za-z])-(?=\d)','−',str(s))
        t=super().table(title,headers,[[clean(v) for v in row] for row in rows])
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.line_spacing=1.1
                    p.paragraph_format.space_before=Pt(2);p.paragraph_format.space_after=Pt(2)
        return t

def chapter6(w):
    w.h('同频场景搭建',1)
    w.h('场景总体架构')
    w.p('本报告针对同频簇级协同调度问题，构建统一的下行多小区仿真实验环境。当前阶段暂不引入移动性切换控制，核心研究维度保持为三个方面：分析簇内小区之间的同频干扰耦合；考察频域资源位图化分配后的小区间协同空间；为传统基线与物理建模智能调度提供共同、可复现的基础验证平台。')
    w.p('当前场景由三个相邻站点构成等边三角形协调簇，小区半径为500 m，相邻站点间距为1000 m。三个站点之间同时存在服务链路和跨小区干扰链路，形成一个封闭的同频协调范围。该拓扑保留了联合调度所需的干扰耦合，同时将全局资源组合规模控制在便于逐时隙精确评价的范围内。')
    w.p('在网络拓扑与资源配置上，场景遵循单站点、单小区的映射原则，并采用复用因子为1的同频复用机制。三个小区全部参与调度、终端承载与KPI统计，使用相同载频、资源组编号和时隙时钟。本轮采用全向发射模式，不通过外围扇区指向中心的额外定向构造放大干扰。')
    w.p('这一设置保持原文“聚焦簇内同频干扰与频域协同”的研究边界：不引入簇外干扰小区、切换控制或动态负载均衡。研究对象是簇级MAC算法及其闭环服务效果，实际覆盖规划、跨簇协调和测量反馈开销需要另行验证。')
    w.h('终端分布与网络关联策略')
    w.p('在小区与UE配置方面，每个名义小区生成12个UE，协调簇共计36个业务活跃终端，所有终端都进入系统统计。终端高度为1.5 m，生成区域受配置的服务距离和Voronoi几何约束限制。Uniform采用区域内均匀布点；Boundary将终端生成位置集中在相邻小区的边界带附近，以提供不同的空间分布条件。')
    w.p('初始化时采用strongest-cell association，依据接收信号功率确定服务小区；运行期间保持静态关联，不执行handover。因而“每个名义小区生成12个UE”不等于每个小区固定服务12个UE。各seed实际关联结果从原生执行动作和状态记录核对，小区goodput、MCS和谱效均按实际服务关联归集。')
    w.p('通过固定关联和移动性控制边界，可以将主要观察对象保持在调度策略对频域分配、干扰、队列服务与过期风险的影响上。配对比较中，同一环境、负载和seed的算法共用拓扑与业务到达；不同策略产生的队列、MCS和历史服务状态则由各自执行反馈自然演化。')
    w.h('无线参数与天线配置')
    w.p('系统工作于2.5 GHz载频，子载波间隔为30 kHz，基站高度25 m，终端高度1.5 m。基站基础发射功率配置为49 dBm，基站与终端均采用4T4R，服务链路使用SVD预编码，接收端采用MMSE-IRC。原生噪声底为−98.9 dBm。')
    w.p('为保证不同资源分配算法在横向比较中的功率条件一致，基站功率按4根发射天线及17个PRG均匀归一化。搜索可以将指定PRG发射关闭，但不会把释放的功率重新分配到其它PRG；本阶段不联合优化连续功率或多小区波束矩阵。全向模式采用冻结实现的统一天线增益，不混用原稿中的定向扇区假设。')
    w.table('表 6-1 无线与节点配置',['参数','本轮设置'],[
      ['小区／UE','3个协调小区；每名义小区生成12个UE，共36个'],['小区半径／站点间距','500 m／1000 m'],
      ['基站／终端高度','25 m／1.5 m'],['载频／SCS','2.5 GHz／30 kHz'],['基站功率／噪声底','49 dBm／−98.9 dBm'],
      ['收发天线','4T4R'],['服务预编码／接收','SVD／MMSE-IRC'],['基础发射方向图','本轮冻结路径的全向模式'],
      ['最小生成距离','30 m'],['阴影标准差','3 dB；自然对数参数按3·ln(10)/10换算']])
    w.h('时频资源建模与信道传播机制')
    w.p('系统配置272个PRB，对应97.92 MHz的有效带宽。频域划分为17个项目PRG，每个PRG包含16个PRB；调度采用Type-0风格的离散位图分配，允许同一UE获得不连续的资源组。本文沿用项目代码的PRG术语，17×16是当前仿真粒度，不将其直接等同于任意NR带宽下的标准RBG配置。')
    w.p('不同小区的同编号PRG对应相同频域位置，同时发射时形成同频干扰。候选动作既可以改变服务UE，也可以将小区在该位置静默；一次编辑可能同时改变本地完整传输块的容量与邻区的接收状态。因此，频域分配按完整TB及跨小区活动状态评价，而不是简单累加互不相关的PRG速率。')
    w.p('大尺度传播按冻结代码中的三维距离损耗表达式与对数正态阴影实现：PL=32.4+20log₁₀(fGHz)+30log₁₀(d3D,m)。原生配置保留UMa场景标记，但本报告以实际执行表达式为准，不据此宣称实现了完整标准模型的全部条件分支。小尺度衰落采用时间相关Rayleigh，递推相关系数为0.9938，实部和虚部创新系数为0.0786。')
    w.p('在不引入簇外干扰源的条件下，簇内同频干扰来自另外两个小区的活动位置。每个小区既是其关联UE的期望信号源，也是邻区UE的干扰源。冻结物理模型保留分源协方差，按PRG是否发射决定干扰贡献；本轮没有启用CDL剖面、独立CFO或对应时延扩展参数。')
    w.table('表 6-2 时频资源与信道传播核心参数',['类别','参数','设定值'],[
      ['时频','TTI','0.5 ms'],['时频','PRB／PRG','272 PRB；17 PRG×16 PRB'],['时频','有效带宽','97.92 MHz/cell'],
      ['时频','分配与复用','Type-0位图式分配；Reuse-1'],['信道','小尺度模型','时间相关Rayleigh'],
      ['信道','递推系数','0.9938；实／虚部创新系数0.0786'],['信道','阴影','3 dB，已修正单位换算'],
      ['干扰','来源与模型','簇内同频小区；activity-only分源协方差'],['运行','单轨迹总长度','28400 TTI＝14.2 s，详见第7章窗口划分']])
    w.h('调度机制与业务流模型')
    w.p('MAC层采用RR、RRQ、PF和PFQ构成传统基线，具体可复现规则在第7章统一说明。DA-PLS与其它搜索对照从本臂当前状态的原始PFQ动作出发，输出UE—PRG映射及静默决策；随后由原生端执行，更新队列、CRC结果和链路自适应。')
    w.p('36个UE各承载一条同参数下行业务流，包长固定为3000 byte，逐UE独立生成Poisson到达。λ表示每UE每TTI的平均到达包数，同一时隙可以生成0个、1个或多个包。Boundary主负载取λ=0.8，Uniform取λ=1.0，Boundary高负载取λ=1.12和1.28。该业务模型包含有限分组、排队和截止时间，不等同于无限缓存的full-buffer模型。')
    w.p('包TTL为400 TTI，即200 ms；保留自然过期，不执行主动丢包。HARQ关闭，CRC失败分段仍从MAC队列移除，成功字节另行记账。只有一个包的所有分段都成功且未过期，才计为完整包交付；MAC完成、分段goodput与完整包吞吐是不同指标。')
    w.h('Uniform与协作区场景的设计动机')
    w.p('在统一的三小区拓扑规则、信道模型和资源约束下，本文设置Uniform与Boundary两类UE分布，对传统基线与智能调度开展对照。两类场景分别观察分散分布和边界集中分布下的表现，分析算法收益对空间分布及小区间耦合关系的依赖，避免依据单一场景评价性能。')
    w.p('Uniform作为常规空间分布下的基本评测条件，比较吞吐、成功包及outcome时延、可靠性、公平性和小区谱效，检查收益是否伴随资源占用或弱用户代价。Uniform同样存在小区间干扰，不能等同于无干扰或没有协同需求。')
    w.p('Boundary将UE向小区边界和覆盖交叠区集中，用于考察跨小区耦合对接收SINR、可用容量、解码结果和队列服务的影响。场景的作用是检验协同调度能否改善资源复用与用户服务的权衡，不预设智能算法必然优于传统方法。')
    w.p('改变UE分布也会影响关联、期望信号强度及各小区业务量；本轮两个主场景的λ还分别为0.8与1.0。因此跨场景绝对数值不是只改变布点的单因素实验。正式结论首先来自每个环境配置内的同seed配对比较，高负载倍数则统一以Boundary的λ=0.8为基准。')
    w.h('场景与智能调度验证目标的对应关系')
    w.p('后续评测采用覆盖吞吐、交付、时延、用户公平、队列和资源效率的统一指标集。第7章逐环境报告传统基线，第9章以同一口径逐环境报告PFQ、缺口权重DPLS和DA-PLS，并给出权重消融。计算成本与搜索精确性作为独立问题，集中在第10章叙述。')
    w.p('时延采用同一个成熟到达包集合：测量窗口内到达的包均追踪至完成或自然过期。成功包时延只包含完整成功交付的包；outcome时延包含所有MAC完成包（含CRC损坏包）和过期包，两者都报告平均值与P95。过期包的终止时长按TTL计为200 ms。未终止包不记为零，本轮随访保证该到达集合不存在截尾。')
    w.p('吞吐与用户体验同时报告完整包吞吐、分段goodput、出队速率、UE平均速率及P5、Jain指数和最差UE过期率；可靠性报告完整成功率、过期率、其它未交付比例和TB错误率。物理与资源指标包括实际发送SINR、MCS、层数、PRG占用、每已用PRG成功字节、活动小区数分布及各小区谱效。测量期平均和起止队列补充解释积压变化，指标定义及统计分母在第7章明确。')

class Quality:
    def __init__(self,w,root,chapter):
        self.w=w;self.root=root;self.chapter=chapter;self.table_no=0;self.figure_no=0
        self.data=pd.read_csv(root/'analysis/aggregate.csv');self.cells=pd.read_csv(root/'analysis/cell_aggregate.csv')
    def table(self,title,headers,rows):
        self.table_no+=1;self.w.table(f'表 {self.chapter}-{self.table_no} {title}',headers,rows)
    def image(self,name,title):
        self.figure_no+=1;self.w.image(name,f'图 {self.chapter}-{self.figure_no} {title}')
    def values(self,scene,rate,arms):return base.complete_means(self.root,scene,rate,arms)
    def metrics(self,title,scene,rate,arms,headers,metrics):
        rows=base.metric_rows(self.root,scene,rate,arms,metrics)
        assert len(rows)==len(arms),(title,scene,rate)
        self.table(title,['算法']+headers,rows)
    def environment(self,scene,rate,title,arms,prefix,plots=True,cells=True):
        w=self.w;v=self.values(scene,rate,arms);assert v
        w.h(title)
        w.p(f'本节配置为{title}，λ={rate:g}包/UE/TTI，理论输入速率{rate*1728:.2f} Mbps。'+('比较' if len(arms)>1 else '报告')+'、'.join(LABEL[a] for a in arms)+'；每个算法均使用seed 41–50的10次完整轨迹。以下各表为先按seed求值、再等权汇总的结果，成功包与outcome时延使用相同的成熟到达集合。')
        w.h('系统吞吐与服务能力',3)
        self.metrics(title+'吞吐与服务量',scene,rate,arms,['输入Mbps','出队Mbps','分段goodput Mbps','整包Mbps'],['offered_mbps','served_mbps','goodput_mbps','payload_mbps'])
        best=max(arms,key=lambda a:v[a,'payload_mbps'])
        rank='完整包吞吐均值最高，为' if len(arms)>1 else '完整包吞吐均值为'
        w.p(f'本配置中，{LABEL[best]}的{rank}{v[best,"payload_mbps"]:.3f} Mbps；对应分段goodput为{v[best,"goodput_mbps"]:.3f} Mbps。出队量含CRC失败分段，分段goodput又可能属于最终未完整交付的包，因此三者分别列示，不以出队或分段服务替代整包交付。')
        w.h('成功包与outcome时延及交付可靠性',3)
        self.metrics(title+'两类包时延（ms）',scene,rate,arms,['成功包平均','成功包P95','outcome平均','outcome P95'],['delay_mean_ms','delay_p95_ms','outcome_mean_ms','outcome_p95_ms'])
        self.metrics(title+'交付结局与TB错误率（%）',scene,rate,arms,['完整成功','自然过期','其它未交付','TB错误率'],['success_pct','expiry_pct','other_nondelivery_pct','tb_error_pct'])
        focus='pfq' if 'pfq' in arms else arms[-1]
        w.p(f'{LABEL[focus]}的成功包平均时延为{v[focus,"delay_mean_ms"]:.3f} ms，outcome平均时延为{v[focus,"outcome_mean_ms"]:.3f} ms；相应P95分别为{v[focus,"delay_p95_ms"]:.3f}和{v[focus,"outcome_p95_ms"]:.3f} ms。前者描述成功交付者的等待，后者将完成与过期都计入终止驻留时间。不能把因大量丢包造成的成功样本筛选误认为整体低时延。')
        w.p(f'本配置中{LABEL[focus]}自然过期率为{v[focus,"expiry_pct"]:.3f}%，其它未交付为{v[focus,"other_nondelivery_pct"]:.3f}%。完整成功、自然过期及其它未交付三类互斥且覆盖全部成熟包；TB错误率则按发送TB计数，统计对象不同，不能与包级比例直接相加。')
        w.h('用户公平性、弱用户与队列积压',3)
        self.metrics(title+'用户分布与公平性',scene,rate,arms,['整包UE P5 Mbps','goodput UE P5 Mbps','整包Jain','goodput Jain','最差UE过期%'],['ue_payload_p5_mbps','ue_goodput_p5_mbps','jain_payload','jain_goodput','worst_ue_expiry_pct'])
        self.metrics(title+'平均用户速率与队列',scene,rate,arms,['UE整包均值 Mbps','UE goodput均值 Mbps','队列均值MB','期初队列MB','期末队列MB'],['ue_payload_mean_mbps','ue_goodput_mean_mbps','queue_mean_mb','queue_start_mb','queue_end_mb'])
        w.p(f'{LABEL[focus]}的UE整包速率P5为{v[focus,"ue_payload_p5_mbps"]:.3f} Mbps，最差UE过期率为{v[focus,"worst_ue_expiry_pct"]:.3f}%。P5、Jain与最差值均先在每个seed的36个UE上计算，包含零速率UE；不能从系统吞吐较高直接推出弱UE改善。队列使用十进制MB，均值为测量期各TTI的簇总剩余字节均值，起止值分别对应测量开始前和测量结束后的状态。')
        w.h('物理传输状态、资源复用与小区谱效',3)
        self.metrics(title+'实际发送与资源效率',scene,rate,arms,['SINR dB','MCS','层数','PRG占用%','成功B/已用PRG'],['sinr_tx_mean_db','mcs_tx_mean','layer_tx_mean','prg_occupancy_pct','bytes_per_used_prg'])
        self.metrics(title+'同PRG活动小区数分布（%）',scene,rate,arms,['零小区','单小区','双小区','三小区'],['reuse_0_pct','reuse_1_pct','reuse_2_pct','reuse_3_pct'])
        self.metrics(title+'goodput谱效（bit/s/Hz）',scene,rate,arms,['簇平均小区谱效','最差小区谱效'],['cluster_goodput_spectral_efficiency','worst_cell_goodput_spectral_efficiency'])
        if cells:
            rows=[]
            for arm in arms:
                c=self.cells[(self.cells.scene==scene)&(self.cells.rate==rate)&(self.cells.arm==arm)].sort_values('cell')
                assert len(c)==3
                for r in c.itertuples():rows.append([LABEL[arm],int(r.cell),f'{r.goodput_mbps:.3f}',f'{r.mcs_tx_mean:.3f}',f'{r.layer_tx_mean:.3f}',f'{r.spectral_efficiency:.3f}'])
            self.table(title+'各小区实际传输状态',['算法','小区','goodput Mbps','MCS','层数','谱效'],rows)
        w.p(f'{LABEL[focus]}的实际发送SINR均值为{v[focus,"sinr_tx_mean_db"]:.3f} dB，MCS为{v[focus,"mcs_tx_mean"]:.3f}，PRG占用为{v[focus,"prg_occupancy_pct"]:.3f}%，每个已用PRG对应{v[focus,"bytes_per_used_prg"]:.3f}成功byte。发送样本的SINR、MCS和层数受用户选择影响，应与交付和公平性一起解释；资源空置也可能源于无业务需求，不能全部归因于主动静默。')
        w.p('小区谱效以该小区goodput除以97.92 MHz计算；簇平均以簇总goodput除以3×97.92 MHz。最差小区谱效在每个seed先取三个小区的最小值，再跨seed平均，因此通常不等于上表三个小区均值的最小值。小区编号为0、1、2。')
        if plots:
            self.image(prefix+'_overview',title+'吞吐、交付与两类时延对比')
            self.image(prefix+'_delays',title+'成功包与outcome时延累计分布')
            self.image(prefix+'_phy',title+'实际发送SINR与MCS累计分布')

def chapter7(w,root):
    q=Quality(w,root,7)
    w.h('同频场景传统基线评测',1)
    w.h('评测目标与数据使用原则')
    w.p('第6章给出了同频组网、资源和业务模型。本章对修正后的传统基线进行系统评测，重点回答：RR、PF及其队列感知版本在不同环境配置中提供多少完整服务；成功包和全部结局的等待代价是否一致；收益与弱用户、公平性、积压、物理状态及资源占用之间存在何种权衡。第8章据此引出协同算法，第9章按同一指标口径检验其增益。')
    w.p('本章逐环境完整报告非计算类指标，不将Boundary的物理或公平性结果代替Uniform，也不以主负载结果代替过载结果。所有数字来自本轮同版本、同协议的修正后轨迹；原稿不同拓扑、窗口和修正前批次的数值不混入对照。')
    q.table('本章环境配置与对照范围',['环境','λ','理论输入Mbps','已运行对照'],[
      ['Boundary08主负载','0.8','1382.40','RR、RRQ、PF、PFQ；各10seed'],['Uniform均匀布点','1.0','1728.00','RR、RRQ、PF、PFQ；各10seed'],
      ['Boundary高负载×1.4','1.12','1935.36','PFQ，10seed；搜索对照见第9章'],['Boundary高负载×1.6','1.28','2211.84','PFQ，10seed；搜索对照见第9章']])
    w.h('实验协议、实现语义与统计口径')
    w.h('共同预热与完整观察窗口',3)
    q.table('每条正式轨迹的时序',['阶段','TTI','时长','用途'],[
      ['共同PFQ预热','1–4000','2 s','同seed共同动作与输入'],['策略稳定','4001–8000','2 s','承接队列、EMA及链路状态'],
      ['正式测量','8001–28000','10 s','统计连续服务；选取本期到达包'],['终态随访','28001–28400','0.2 s','跟踪测量期到达包至完成或过期']])
    w.p('预热结束不清空队列，策略切换不重置信道、EMA或OLLA。每个配置的算法共用同seed拓扑、Poisson到达及4000 TTI预热输入和动作；各策略随后产生的状态分化属于算法闭环效果。本轮成熟到达集合全部终止，无截尾包。')
    w.h('四种传统基线的可复现规则',3)
    q.table('基线规则与队列处理',['算法','分配规则','队列及历史服务'],[
      ['RR','逐小区轮转正队列关联UE，逐PRG分配','跳过空队列；不按队列大小截断已分PRG'],
      ['RRQ','沿用RR次序；TBS覆盖当前队列后停止继续分配','按当前MCS、PRG数及最小层数判断需求'],
      ['PF','逐PRG按瞬时物理速率／历史成功服务EMA选择','不作队列容量截断'],['PFQ','执行冻结原生queue-aware PF动作','需求余量0 byte；为后续搜索提供可行初选']])
    w.p('RR、RRQ和PF为本报告明确规定的可复现对照，不声称与历史原生二进制逐位相同。RR与RRQ使用同一轮转机制，适合检查需求限制的作用；PF与PFQ还存在具体优先级和实现差异，不能把其全部差值归因于唯一的queue-aware开关。')
    w.h('指标定义与统计口径',3)
    q.table('指标口径与分母',['指标组','定义'],[
      ['输入／出队／分段goodput','测量期对应字节×8／10 s；分别为生成、MAC移除与CRC成功分段'],
      ['完整包吞吐','成熟到达集合中完整成功包payload×8／10 s'],['成功包平均／P95','完整成功包的交付时延；按每seed计算'],
      ['outcome平均／P95','同一到达集合中所有完成及自然过期包的终止驻留时间；含CRC损坏后完成'],
      ['成功／过期／其它未交付','同一成熟集合的互斥三种结局；比例之和100%'],['TB错误率','错误TB数／发送TB数；不是包过期率'],
      ['UE均值／P5／Jain／最差UE','每seed先按36个UE求值，保留零速率UE'],['队列','测量期平均簇总队列，以及测量期初和期末队列；MB=10⁶ byte'],
      ['SINR／MCS／层数','实际分配资源并发送的TB样本'],['资源与复用','cell-PRG占用／全部可用cell-PRG；同PRG活动小区数分布'],
      ['谱效','小区goodput／97.92 MHz；最差小区先在seed内取最小值']])
    w.p('所有平均实验均先计算每个seed的指标，再对seed 41–50等权平均；P95、Jain及最差值也先在seed内部计算，不将不同seed的包合池。图中误差线为seed均值的Student t 95%区间；配对区间以同seed差值计算，未作多重比较校正。未对区间施加物理边界截断，故均值区间偶尔可越过0；它不表示出现负时延。')
    w.p('outcome不是给成功包时延简单增加罚项，而是改变统计对象：CRC损坏后完成的包也在其中。部分未完整交付包可能较早终止，因此outcome平均值不保证大于成功包平均值；两类时延应与三种交付结局一起阅读。')
    for i,(scene,rate,title) in enumerate(ENVIRONMENTS):
        arms=['rr','rrq','pf','pfq'] if i<2 else ['pfq']
        q.environment(scene,rate,title,arms,f'baseline_{i}',plots=i<2)
        if i<2:
            v=q.values(scene,rate,arms)
            w.p(f'本环境RR→RRQ的整包吞吐为{v["rr","payload_mbps"]:.3f}→{v["rrq","payload_mbps"]:.3f} Mbps，过期率为{v["rr","expiry_pct"]:.3f}%→{v["rrq","expiry_pct"]:.3f}%；PF→PFQ的整包吞吐为{v["pf","payload_mbps"]:.3f}→{v["pfq","payload_mbps"]:.3f} Mbps。需要将这一比较与实现语义、完整交付及资源占用共同解释。')
            if scene=='boundary':
                w.p(f'Boundary主负载下，RR和RRQ都保持接近满资源占用，完整包吞吐相差不足1%，outcome平均时延分别为{v["rr","outcome_mean_ms"]:.3f}与{v["rrq","outcome_mean_ms"]:.3f} ms。仅增加队列需求截断没有明显改变该配置的服务状态。PFQ虽取得更高总吞吐，其UE整包速率P5为{v["pfq","ue_payload_p5_mbps"]:.3f} Mbps，低于PF的{v["pf","ue_payload_p5_mbps"]:.3f} Mbps；总吞吐排序与弱用户排序因此需要分别观察。')
            else:
                w.p(f'Uniform下，RR到RRQ的outcome平均时延从{v["rr","outcome_mean_ms"]:.3f}降到{v["rrq","outcome_mean_ms"]:.3f} ms，平均队列从{v["rr","queue_mean_mb"]:.3f}降到{v["rrq","queue_mean_mb"]:.3f} MB，同时PRG占用下降。该结果说明本配置中需求限制与更有效的资源使用相伴。PF的平均MCS为{v["pf","mcs_tx_mean"]:.3f}，高于PFQ的{v["pfq","mcs_tx_mean"]:.3f}，但整包吞吐更低、过期更多，不能以较高MCS单独判定调度更好。')
    w.h('配对结果与负载敏感性')
    pairs=pd.read_csv(root/'analysis/baseline_paired.csv');rows=[]
    for scene,rate,title in ENVIRONMENTS[:2]:
        for metric,name in [('payload_mbps','整包Mbps'),('delay_mean_ms','成功平均ms'),('delay_p95_ms','成功P95 ms'),('outcome_mean_ms','outcome平均ms'),('outcome_p95_ms','outcome P95 ms'),('expiry_pct','过期百分点')]:
            r=pairs[(pairs.scene==scene)&(pairs.rate==rate)&(pairs.baseline=='rrq')&(pairs.metric==metric)].iloc[0]
            rows.append([title,name,f'{r.delta:+.3f}',f'[{r.ci_low:+.3f}, {r.ci_high:+.3f}]'])
    q.table('PFQ减RRQ的同seed配对差值',['环境','指标','均值差','95%配对区间'],rows)
    q.image('baseline_load','Boundary三档负载下PFQ的服务、结局与队列变化')
    vv=[q.values('boundary',r,['pfq']) for r in [.8,1.12,1.28]]
    w.p('三档负载下PFQ的完整包吞吐依次为'+'、'.join(f'{v["pfq","payload_mbps"]:.3f}' for v in vv)+' Mbps，过期率依次为'+'、'.join(f'{v["pfq","expiry_pct"]:.3f}%' for v in vv)+'；outcome平均时延依次为'+'、'.join(f'{v["pfq","outcome_mean_ms"]:.3f}' for v in vv)+' ms。输入增加的同时，更多业务等待至期限，完整成功交付量反而降低。')
    w.p('Boundary中增加输入负载并不保证完整包吞吐增加。应同时观察出队、分段goodput与整包交付：接近时限的积压、失败分段和过期共同决定最终交付。负载图只连接本轮已运行的0.8、1.12和1.28三个固定负载点，不作为未运行的细粒度负载扫描或动态切换结果。')
    w.h('本章结论与协同算法的评测要求')
    w.p('两种主环境中，PFQ均取得四种传统对照中最高的完整包吞吐均值，但这一排序不替代时延、过期、公平性及资源指标。Uniform与Boundary分别报告完整指标后可以看到，算法收益与环境条件相关；跨环境几何和到达率同时变化，不能将差异单独归因于干扰。')
    w.p('后续协同方法应在相同输入和统计口径下，明确回答其完整包收益是否伴随成功包与outcome时延改善、是否降低弱UE过期、是否改变实际物理状态及同频复用。第8章给出以分源物理评价和时延权重解决这一问题的方法，第9章逐环境验证效果，计算代价与精确性独立在第10章评估。')

def chapter9(w,root):
    q=Quality(w,root,9);w.h('DA-PLS仿真评估与机制分析',1)
    w.h('对照设计与统一指标')
    w.p('本章在第7章相同的四个环境配置中，比较PFQ、缺口权重DPLS和DA-PLS。每个环境分别报告吞吐、两类平均/P95时延、交付结局、用户公平、队列、物理传输与资源效率；不以一个场景的指标代替另一个场景。所有搜索臂使用同一原始PFQ初选、90%计划出队下界、完整TB评价及精确内层，隔离权重形式在闭环中的作用。')
    w.p('正式矩阵合计220个配置—seed点，其中210点本轮重跑，10条PFQ轨迹经同版本、同协议验证后复用。41–43参与过本次权重选型；44–50另作稳健性子集，但在项目历史中有暴露，不称完全未见盲测。全部逐seed结果及两种seed子集的区间随结果表提供。')
    arms=['pfq','deficit','mlwdf_pf'];pair=pd.read_csv(root/'analysis/weight_paired.csv')
    for i,(scene,rate,title) in enumerate(ENVIRONMENTS):
        q.environment(scene,rate,title,arms,f'comparison_{i}',plots=False)
        q.image(f'comparison_{i}_overview',title+'协同算法的服务与两类时延')
        v=q.values(scene,rate,arms);delta=(v['mlwdf_pf','payload_mbps']/v['pfq','payload_mbps']-1)*100
        w.p(f'本环境DA-PLS相对PFQ的完整包吞吐变化为{delta:+.2f}%。相对同内层的缺口权重，吞吐变化为{100*(v["mlwdf_pf","payload_mbps"]/v["deficit","payload_mbps"]-1):+.2f}%，过期率相差{v["mlwdf_pf","expiry_pct"]-v["deficit","expiry_pct"]:+.3f}个百分点。这一区分分别衡量与原生基线的总体差异，以及相同物理搜索下的权重差异。')
        rows=[]
        for metric,name in [('payload_mbps','整包Mbps'),('delay_mean_ms','成功平均ms'),('delay_p95_ms','成功P95 ms'),('outcome_mean_ms','outcome平均ms'),('outcome_p95_ms','outcome P95 ms'),('expiry_pct','过期百分点'),('worst_ue_expiry_pct','最差UE过期百分点')]:
            r=pair[(pair.scene==scene)&(pair.rate==rate)&(pair.subset=='41-50')&(pair.metric==metric)].iloc[0]
            rows.append([name,f'{r.delta:+.3f}',f'[{r.ci_low:+.3f}, {r.ci_high:+.3f}]'])
        q.table(title+'：DA-PLS减缺口权重的配对差值',['指标','均值差','95%配对区间'],rows)
        if rate>1:
            w.p('该过载配置中，缺口权重的整包吞吐和整体过期率优于M-LWDF-PF，配对区间也支持这一方向。两种成功包P95均接近TTL，不能据此把过载状态解释为低时延服务。本文如实保留该边界；没有执行或宣称验证动态权重切换。')
        elif scene=='uniform':
            w.p('Uniform中M-LWDF-PF相对缺口权重的整包吞吐略低，两者均无自然过期；相对PFQ的成功包P95和过期率配对区间跨0。因此不能将Boundary主负载的权重优势直接推广为所有布点下的优势。')
    w.h('同一内层下的权重消融')
    w.p('权重消融固定Boundary/rate0.8。缺口权重为clip[2400/max(EMA,10⁻⁹),1,4]后均值归一化；2400 B/TTI在全部负载下固定。κ-TTL在此基础上乘[1+κ(1−slack/400)]，κ取1、2。M-LWDF-PF及EXP-PF使用未截断的PF分母；本组比较完整权重形式，不将差异全部归因于单一紧迫度曲线。')
    ablations=['equal','deficit','ttl1','ttl2','exp_pf','mlwdf_pf']
    # A complete metric set also applies to the ablation configuration.
    q.environment('boundary',.8,'Boundary08权重消融配置',ablations,'ablation',plots=False,cells=False)
    q.image('ablation_overview','不同权重的整包交付、成功包与outcome时延')
    w.p('主负载六种权重中，M-LWDF-PF的整包吞吐均值最高、成功包P95最低。κ-TTL及EXP-PF同样没有观测到自然过期，故仅以“是否零过期”不足以区分方案。配对区间及全部指标详见结果表，不将公平性、高负载和计算成本合并为一个“全面最优”的结论。')
    w.h('分源干扰协调的机制证据')
    for i,title in [(0,'Boundary08'),(1,'Uniform')]:
        q.image(f'comparison_{i}_phy',title+'的实际发送SINR与MCS分布')
        q.image(f'comparison_{i}_delays',title+'的成功包及outcome时延分布')
    q.image('comparison_load','三个Boundary负载的适用边界')
    q.image('joint_move_case','真实保存状态中的静默与邻区换UE联合移动')
    case=select_case(root)
    if case:
        c=case['case'];w.p(f'联合移动案例选自{Path(case["path"]).parent.parent.name}的TTI {c["tti"]}。计划出队由{c["dequeue_before"]}变为{c["dequeue_after"]} byte，仍高于{c["floor"]} byte下界；净整数加权增益为{c["gain_q"]/1e6:.6f} byte。案例优先选择首个同时含本地代价与邻区收益的已核验联合编辑，用于解释一次决策，不估计其发生频率。')
        q.table('联合移动的受益与代价',['UE','权重','原预期B','新预期B','加权变化B'],[[x['ue'],f'{x["weight"]:.4f}',f'{x["expected_before"]:.3f}',f'{x["expected_after"]:.3f}',f'{x["weighted_delta"]:+.3f}'] for x in c['ues']])
    w.p('以上质量结果共同限定二阶段算法的适用范围。分布图受被选择的发送UE影响，不能仅凭平均SINR/MCS上升声称所有链路改善；逐UE、交付结局和资源复用指标提供互补证据。计算成本与精确性在下一章独立报告。')

ORIGINAL_CONCLUSION=base.chapter10
def chapter10(w,root):
    w.h('计算成本与精确性验证',1)
    w.h('验证对象、输入与统计方式')
    w.p('本章与前面的业务质量评测分开。精确性验证回答剪枝是否保持规范完整搜索的选择序列与终态；计算成本验证回答相同输入上的工程开销降低多少。两者不替代成功包/outcome时延，也不将局部搜索等价性解释为全局最优。')
    a=json.loads((root/'analysis/completion.json').read_text());c=json.loads((root/'audit/completion.json').read_text())
    w.p(f'平台侧完成{a["prefix_checks"]}个共同预热检查及{a["physical_states"]}个保存动作的完整TB物理核验。SINR最大绝对差为{a["physical_sinr_max_abs_db"]:.3e} dB，BLER最大差为{a["physical_bler_max_abs"]:.3e}，分别小于0.005 dB与0.0005；动作和字节按整数精确检查。浮点物理容差与下述搜索逐位等价分开验收。')
    w.h('逐轮精确性与证书核验')
    w.table('表 10-1 搜索精确性核验',['项目','结果'],[['保存状态',c['states']],['逐轮检查',c['audited_rounds']],['检查候选',c['audited_candidates']],['新增精评',c['new_evaluations']],['证书必评集合',c['required_evaluations']],['新增／必评比','1.000'],['根、缓存、上界、选择、终态','全部通过']])
    w.p('每个Boundary负载取10个seed，每条轨迹保存测量开始、中间和末尾三个输入。独立规范完整枚举逐轮核对上界可采纳性、缓存、出队可行性、规范赢家和终态。新增／必评比对应当前证书规则，不是所有可能算法的信息论下界；停止门槛下的局部稳定也不等于全局最优。')
    w.h('同输入串行计算成本')
    timings=pd.read_csv(root/'audit/timing_summary.csv');rows=[]
    for rate,g in timings.groupby('rate'):
        g=g.set_index('arm');b=g.loc['bounded'];f=g.loc['native_full'];p=g.loc['python_full']
        rows.append([f'×{rate/.8:g}',f'{p.wall_ms:.3f}',f'{f.wall_ms:.3f}',f'{b.wall_ms:.3f}',f'{100*b.wall_ms/f.wall_ms:.2f}%',f'{100*b.candidates/f.candidates:.2f}%',f'{100*b.mmse_misses/f.mmse_misses:.2f}%'])
    w.table('表 10-2 保存状态串行成本（ms；相对native完整搜索）',['负载','Python完整','native完整','精确剪枝','耗时占比','候选占比','MMSE占比'],rows)
    w.p('最终计时在220个质量实验全部结束后执行，固定单CPU核，三种实现轮换顺序，每个状态重复三次。先平均同seed的重复与三个状态，再等权平均seed。早期并发计时单独归档，沿用同一批最终复测结果，不因本次文稿重排改变计时数据。')
    w.p('计时包含保存输入重构、权重生成及一致性检查、PFQ动作解码、物理上下文创建、搜索和销毁；不含原始观测复制、HOL的FIFO扫描、原生PFQ计算、仿真推进和跨进程通信。因此它是内层同输入比较，不是完整调度端到端时延。')
    w.h('连续运行成本与实时性边界')
    cc=pd.read_csv(root/'analysis/continuous_cost.csv');rows=[]
    for rate in [.8,1.12,1.28]:
        group=cc[(cc.scene=='boundary')&(cc.rate==rate)&(cc.arm=='mlwdf_pf')].set_index('metric')
        for metric,name in [('wall_ms','墙钟/ms'),('cpu_ms','进程CPU/ms'),('candidate_evaluations','候选精评数'),('ue_evaluations','UE评价数'),('mmse_misses','MMSE miss')]:
            r=group.loc[metric];rows.append([f'×{rate/.8:g}',name,f'{r.seed_mean:.3f}',f'{r.seed_p95_mean:.3f}',f'{r.observed_max:.3f}'])
    w.table('表 10-3 连续测量窗口成本',['负载','指标','seed均值的均值','seed P95的均值','观测最大值'],rows)
    w.p('连续记录包括观测、权重与HOL整理、搜索和三个保存点归档I/O，保留离群值。质量仿真并发运行，墙钟含主机争用，不能作为公平加速比分母。保存状态的串行剪枝耗时约6.28–7.02 ms，高于0.5 ms TTI；当前完成的是算法研究验证，生产实时部署仍需进一步工程优化。')
    ORIGINAL_CONCLUSION(w,root)

def make_figures(root):
    out=root/'figures';out.mkdir(exist_ok=True);agg=pd.read_csv(root/'analysis/aggregate.csv');per=pd.read_csv(root/'analysis/per_seed.csv')
    agg=agg[(agg.subset=='41-50')&(agg.n==10)]
    jobs=[]
    for i,(scene,rate,title) in enumerate(ENVIRONMENTS):
        if i<2:jobs.append((scene,rate,title,['rr','rrq','pf','pfq'],f'baseline_{i}'))
        jobs.append((scene,rate,title,['pfq','deficit','mlwdf_pf'],f'comparison_{i}'))
    jobs.append(('boundary',.8,'Boundary08权重消融',['equal','deficit','ttl1','ttl2','exp_pf','mlwdf_pf'],'ablation'))
    for scene,rate,title,arms,name in jobs:
        z=agg[(agg.scene==scene)&(agg.rate==rate)]
        fig,axes=plt.subplots(2,3,figsize=(10,6),constrained_layout=True)
        for ax,metric,label in zip(axes.flat,['payload_mbps','expiry_pct','delay_mean_ms','outcome_mean_ms','delay_p95_ms','outcome_p95_ms'],['完整包吞吐 / Mbps','自然过期率 / %','成功包平均 / ms','outcome平均 / ms','成功包P95 / ms','outcome P95 / ms']):
            a=z[z.metric==metric].set_index('arm').loc[arms]
            ax.bar(np.arange(len(arms)),a['mean'],yerr=a.ci_high-a['mean'],capsize=2,color=[COLORS[i%len(COLORS)] for i in range(len(arms))])
            ax.set_xticks(np.arange(len(arms)),[LABEL[x] for x in arms],rotation=25,fontsize=8);ax.set_title(label,fontsize=10);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.suptitle(title+'：10 seed等权均值；误差线为95%区间',fontsize=12);save(fig,out,name+'_overview')
        if name=='ablation':continue
        for kind in ['delays','phy']:
            fig,axes=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
            for ai,arm in enumerate(arms):
                keys=per[(per.scene==scene)&(per.rate==rate)&(per.arm==arm)].key.tolist();assert len(keys)==10
                for j,ax in enumerate(axes):
                    if kind=='delays':
                        frames=[pd.read_csv(root/'analysis'/k/'delay_hist.csv') for k in keys];field=['success_mass','outcome_mass'][j]
                        x=frames[0].delay_ms;mass=np.mean([f[field].to_numpy() for f in frames],axis=0);label=['成功包时延 / ms','outcome时延 / ms'][j]
                    else:
                        field=['phy_sinr_db_cpu','mcs'][j];frames=[pd.read_csv(root/'analysis'/k/(field+'_hist.csv')) for k in keys]
                        x=frames[0].right if j==0 else (frames[0].left+frames[0].right)/2
                        mass=np.mean([f.mass.to_numpy() for f in frames],axis=0);label=['实际发送SINR / dB','实际发送MCS'][j]
                    ax.step(x,np.cumsum(mass),where='post',label=LABEL[arm],color=COLORS[ai%len(COLORS)])
                    ax.set(xlabel=label,ylabel='seed等权累计分布',ylim=(0,1.01));ax.grid(alpha=.2)
            axes[0].legend(fontsize=8);save(fig,out,name+'_'+kind)
    for tag,arms in [('baseline',['pfq']),('comparison',['pfq','deficit','mlwdf_pf'])]:
        fig,axes=plt.subplots(2,3,figsize=(10,6),constrained_layout=True)
        for ax,metric,title in zip(axes.flat,['payload_mbps','expiry_pct','queue_mean_mb','delay_mean_ms','outcome_mean_ms','outcome_p95_ms'],['整包吞吐 / Mbps','自然过期率 / %','平均队列 / MB','成功包平均 / ms','outcome平均 / ms','outcome P95 / ms']):
            for i,arm in enumerate(arms):
                z=agg[(agg.scene=='boundary')&(agg.arm==arm)&(agg.metric==metric)].sort_values('rate');assert len(z)==3
                ax.errorbar(z.rate/.8,z['mean'],yerr=z.ci_high-z['mean'],marker='o',capsize=2,label=LABEL[arm],color=COLORS[i])
            ax.set(xlabel='相对Boundary08负载',ylabel=title,xticks=[1,1.4,1.6]);ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8);save(fig,out,tag+'_load')
    plot_case(root,out)

def main(args):
    root=Path(args.root).resolve();make_figures(root)
    base.Writer=Writer;base.LABEL=LABEL
    base.REPLACE[28]='本文第2、3章介绍资源、约束与传统方法，第4章保留AI方法综述，第5章说明平台与修正，第6章沿场景总体架构、终端、无线参数、时频信道和业务模型展开，第7章系统评测传统基线，第8章给出DA-PLS方法，第9章逐环境报告算法质量与消融，第10章独立评价计算成本与精确性，第11章总结适用范围。'
    base.REPLACE[121]='正式评测逐环境报告完整包吞吐、分段goodput、成功包和outcome的平均/P95时延、过期与其它未交付、公平性、弱UE、队列及物理资源指标。计算成本与精确性在后续独立章节报告。'
    base.chapter6=chapter6;base.chapter7=chapter7;base.chapter9=chapter9;base.chapter10=chapter10
    base.main(SimpleNamespace(root=str(root),core=False,draft=False))
    generated=root/'技术研究报告_二阶段同频收尾_DA-PLS.docx';d=Document(generated)
    for p in d.paragraphs:
        if p.text=='2026年11月4日':p.text='2026年11月5日 · 评测扩充修订版'
        if p.text.startswith('已完成搜索轨迹的'):
            p._element.getparent().remove(p._element)
    for table in d.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text=='HOL delay、TTL slack、p90/p95/p99 delay':
                    cell.text='HOL等待、TTL余量；成功包与outcome平均时延及P95'
    target=root/'技术研究报告_二阶段同频收尾_评测扩充版.docx';d.save(target);generated.unlink()
    text='\n'.join(p.text for p in d.paragraphs)+'\n'+'\n'.join(c.text for t in d.tables for row in t.rows for c in row.cells)
    assert 'P99' not in text and 'p99' not in text and 'CTBS' not in text
    (root/'report_text_audit.txt').write_text(text)
    print('REVISED_REPORT_READY',target,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args())

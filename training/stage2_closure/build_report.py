"""Revise the supplied report in Word, retaining its foundation and survey."""
import argparse
from copy import deepcopy
import json
import re
from pathlib import Path
import sys
sys.path.insert(0,'/tmp/stage2_report_python')
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.table import Table
import pandas as pd
from .figures import diagrams,select_case

ROOT=Path(__file__).resolve().parents[2]
ATTACH=Path('/home/oai-mac/.codex/attachments/464f2cd8-a289-43e2-a3dd-5b0428e87cd1')
SOURCE=ROOT/'deliverables/stage2_closure_20261104/source_documents/原技术研究报告.docx'
if not SOURCE.exists():SOURCE=next(ATTACH.glob('2.*docx'))
LABEL={'pfq':'PFQ','pf':'PF','rr':'RR','rrq':'RRQ','equal':'等权 DPLS','deficit':'缺口权重 DPLS（κ=0）',
       'ttl1':'κ-TTL（κ=1）','ttl2':'κ-TTL（κ=2）','exp_pf':'EXP-PF + DPLS','mlwdf_pf':'DA-PLS'}

REPLACE={
191:'本章综述无线资源分配中的AI方法，依次讨论强化学习与多智能体学习、知识增强与模型驱动方法、图结构化资源分配，并区分文献中的信息条件和工程边界。综述用于说明前期探索及设计取舍；二阶段最终路线是显式物理建模、时延感知权重与精确局部搜索，异构图和PPO不再作为在线主算法。',
279:'领域先验可以进入学习方法的状态、网络结构、目标和可行性约束，也可以直接进入优化器或搜索器。本报告采用后一种形式：在候选生成和物理评价阶段落实关联、队列、资源及出队约束，输出已经通过可行性检查的分配。',
287:'本文吸收知识增强中的规则表达思想，将服务关联、资源掩码、物理参数和队列状态转化为可计算约束与评价量。二阶段不构建知识图谱或神经网络，不以相关综述代替最终算法的实际实现。',
300:'图神经网络（Graph Neural Network, GNN）是一类处理图结构数据的神经网络，图卷积网络（Graph Convolutional Network, GCN）是其中的一类。图表示可以容纳不同数量的实体及其连接关系；无线网络中的基站、用户、链路和频域资源之间的服务、干扰及竞争关系，为此类方法提供了建模对象。具体模型是否具有置换等变性或跨规模泛化能力，仍取决于架构和验证条件[2,3]。',
329:'前期异构图原型曾将频域资源显式建模为节点，用于描述Type-0位图式分配中的实体关系。其Cell、UE和PRG三类节点如下；这一表示属于路线演进背景，不是DA-PLS的在线网络结构：',
182:'MILP/MINLP可为资源分配提供优化基准和约束模板。在求解规模与时间允许时，可报告经过求解器证实的最优解；提前停止时应区分可行解值、有效界和最优性间隙，不能把一个近似可行解直接称为最大化问题的上界。松弛、分解和启发式初解可以辅助求解，但本报告的DPLS终态不借此宣称全局最优。',
226:'PPO通过与环境交互采样，再对替代目标进行多轮小批量更新，在原论文的基准任务中体现了样本效率、实现简洁性与计算时间之间的折中[6]。这种经验表现不构成任意无线场景下训练稳定性或最优性的保证。本项目将PPO保留为前期原型与方法综述的一部分，最终在线算法采用物理搜索。',
231:'在无线资源管理中，MARL可用于多小区频谱分配、功率控制和用户调度。例如，Zhang和Guo的多小区方案由各小区利用局部信息进行联合频谱、功率与调度决策，在随机到达业务下优化分组时延[7]。它为分布式协同提供了一种路径；信息条件和执行预算应结合具体实现比较。',
45:'RR、PF及其队列感知版本RRQ、PFQ，以及本文DA-PLS，均在统一的位图式资源表达上输出可执行分配。算法差异主要来自状态利用、优先级计算、跨小区物理建模和搜索机制，不改变基本帧结构。',
272:'K-DDPG的知识辅助思想说明，领域知识可以直接体现为状态、目标和约束，而不必依赖完整知识图谱[23]。在本阶段，原始PFQ用于初选，物理模型用于评价候选，显式资源约束和出队下界用于限定可行集合；不执行离线预训练或PPO微调。',
273:'本文将知识先验落实为可计算对象：UE—小区归属转化为合法掩码；HOL与历史服务生成业务权重；分源信道生成干扰协方差；当前资源集合决定完整TB容量。所有对象直接进入精确搜索。',
280:'知识增强可以直接作用于搜索空间和物理评价：关联与资源可用性限定合法编辑，固定MCS下的TBS给出容量上界，当前队列约束计划出队。本文采用这些显式对象，不依靠执行后的动作改写替代搜索中的可行性检查。',
292:'联邦学习可用于未来多区域模型训练、跨场景迁移和隐私保护，但本阶段DA-PLS没有训练模型，因此不使用联邦更新。相关综述保留为后续研究背景。',
296:'强离散的PRG调度不必被展开成可训练网络。知识／模型驱动也可以直接形成物理决策器：用明确模型评价候选，以可采纳上界控制计算量。本文选择这一方向；其边界是依赖模型和观测精度、局部搜索限制及实际计算预算。',
325:'用户—小区异构图适合表达服务关联和用户差异，是前期原型的状态表示方式。DA-PLS则直接用每UE队列、历史服务及物理链路参数表达这些差异，不保留在线异构图网络。',
21:'同频密集组网的关键矛盾是跨小区在相同时频位置复用造成的干扰外部性。二阶段将优化范围明确为时隙内的 UE—PRG 联合映射和发射／静默协调；空域采用既定服务链路 SVD 预编码，不宣称实现联合波束零陷或连续功率优化。',
26:'簇级协同方案必须明确其信息条件。本报告采用集中式研究设置，假设能够在动作前获得簇内服务及跨小区信道，进而形成分源干扰协方差。该设置允许直接评价反事实调度动作；实际系统中的测量、反馈、同步、信道老化和误差代价仍需另行评估。本文不将仿真器可见的完整信道等同于已经具备工程部署接口。',
27:'二阶段技术路线收敛为 DA-PLS（Delay-Aware Physical Local Search）：用 M-LWDF-PF 形式的时延感知权重表达业务紧迫度，以完整传输块物理评价衡量协同净收益，通过上界剪枝的精确 DPLS 内层完成联合调度。算法不含训练模型；其智能性来自物理知识建模、经典启发式搜索与逐时隙反馈。',
28:'本文第2、3章介绍资源、约束和传统方法，第4章保留 AI 方法综述并说明路线选择，第5章记录仿真平台与两项单位缺陷修正，第6章统一场景及统计协议，第7章重评传统基线，第8章给出 DA-PLS 方法与等价性论证，第9章报告新仿真、消融和复杂度，第10章总结二阶段交付边界及后续工作。',
31:'本章统一下行 MAC 调度的状态、资源、动作和反馈语义，为传统基线与 DA-PLS 建立共同系统模型。本文以三个小区、36个 UE、每小区17个项目级 PRG 为研究规模，标准概念与仿真中的抽象粒度分别说明。',
53:'跨小区关系包括服务关联、同频资源复用和来自不同邻区的干扰协方差。DA-PLS 显式保留干扰源身份，并在候选动作下重新计算受影响 UE 的物理收益；图结构学习仅作为第4章综述及第8章路线演进的历史研究方向。',
56:'每个调度周期内的主要决策是在每个（小区，PRG）位置选择一个具有正队列的关联 UE，或选择静默。实现中的 UE slot 只是原生接口编码，不构成必须先行裁剪候选集合的算法层次。',
57:'候选集合由当前服务小区、正队列和可用 PRG 掩码决定。DA-PLS 从原始 PFQ 分配出发，在整个合法局部邻域中搜索；不额外引入候选预算或学习筛选。',
58:'频域映射可以包含多个不连续资源组。静默是否有益必须同时计算本小区让出资源的代价与邻区完整 TB 的受益，并满足本时隙总出队下界。',
59:'本时隙候选评价固定已由 OLLA 选定的 MCS；UE 获得的 PRG 集合及其最小可用层数共同决定 TBS。资源分配会影响实际接收质量及下一时隙反馈，但不在同一轮搜索中反复修改 MCS。',
64:'图2-2给出本报告的状态—动作—反馈闭环。动作前读取队列、HOL、历史成功服务 EMA 和簇内信道，形成权重及物理模型；DA-PLS 输出合法分配；原生端执行并更新成功字节、分段失败、FIFO 终态与 OLLA。正式轨迹逐 TTI 核对实际动作、MCS、层数、TBS 和队列扣减。',
65:'队列状态按 MAC 出队字节更新；成功交付还需结合 CRC 和整包生命周期单独统计。忽略溢出时，令 A 为本周期到达字节，X 为自然过期移除字节，D 为调度出队字节，则队列守恒为：',
66:'Q(t+1) = max{Q(t) + A(t) − X(t) − D(t), 0}。',
67:'在本实验关闭 HARQ 的设置下，CRC 失败的分段仍从 MAC 队列移除，不能把 D 直接等同于成功服务 S。一个包只有所有分段均成功且未过期，才计入完整包吞吐。',
77:'项目在每小区272个 PRB 上采用17个频域调度组，每组16个 PRB。每个时隙联合决定三个小区共51个位置的 UE 映射或静默，以控制离散组合规模。',
78:'术语上，本文沿用项目代码中的 PRG 表示频域调度组；其粒度是仿真抽象，不将17×16这一设置直接宣称为任意 NR 带宽下的标准 Type-0 RBG 配置。Type-0 在本文强调的是可非连续的位图式分配语义。',
81:'图2-3展示272 PRB按16 PRB聚合为17个项目 PRG。相同 PRG 编号在三个小区对应相同频域位置，因而邻区的发射活动产生同频干扰耦合。',
86:'若小区 c 有 Uc 个合法 UE，则联合动作空间至多为 ∏c(Uc+1)^17。DA-PLS 不枚举全局组合，而以原始 PFQ 为可行起点，通过局部编辑和可采纳上界削减实际物理评分次数。',
90:'空间层和服务链路 SVD 预编码属于统一物理环境。本文利用它们评价完整 TB，但不以联合波束矩阵或独立层选择作为搜索动作。',
94:'同一发射位置采用统一功率配置；算法可通过静默关闭该位置的发射，静默后不把功率重新分配到其它位置。因此二阶段实现的是时频联合与功率开关级协调，连续功率和联合波束优化属于后续工作。',
101:'上述标量表达用于说明一般同频干扰关系。本报告实际物理评价器使用4×4协方差和 MMSE-IRC；邻区干扰贡献由该 PRG 是否发射决定，不随邻区具体服务 UE 的身份改变，这一 activity-only 简化在第8章明确。',
107:'同一 PRG 上静默一个小区会改变其它接收端的干扰协方差。若仅替换该小区服务 UE 而仍发射，activity-only 模型中的邻区干扰项保持不变；但新服务 UE 的期望信号和本地收益会改变。完整联合移动需要同时评价两种效应。',
120:'上式仅描述一般多目标结构，不是本文最终实现的奖励函数。DA-PLS 实际优化目标是时延权重下的预期成功服务字节，并以完整包吞吐、过期、尾时延、公平性及资源占用检验其闭环后果。',
121:'正式评测统一报告完整包吞吐、分段 goodput、成熟包过期率、成功包均时/P95/P99、UE速率P5、Jain公平性、最差UE过期率、实际发送SINR/MCS、资源占用及计算开销。每项指标先在单个 seed 内计算，再跨 seed 等权汇总。',
183:'本课题的 Boundary08 规模为3小区、36 UE、每小区17 PRG。即使只保留离散 UE 映射和静默，全局组合空间仍较大。本报告采用物理局部搜索，不将其终态宣称为全局整数规划最优解。',
186:'传统方法提供本报告的基线与理论来源：PFQ 提供强工程对照及可行初选；M-LWDF/PF 提供时延和历史服务的权重形式；局部搜索与上界剪枝提供可解释的协同决策机制。传统规则和物理搜索可直接组成无需训练的智能调度方法。',
187:'场景同时保留强干扰边界布点和均匀布点，以检查收益是否依赖特定几何与负载。场景设计服务于可检验的干扰协调机制，不以人为放大收益或预设学习方法优于基线为目的。',
200:'公开研究采用的信息条件各不相同。本文将有限遥测条件下的学习方法与完整簇内信道条件下的物理搜索分别讨论，避免将前者的部署条件与后者的仿真收益混为一谈。',
203:'本课题的模型驱动定位与可观测边界',
204:'本报告采用簇内集中式调度，假设动作前可获得服务及跨小区信道或等价分源协方差。DA-PLS 用该信息直接构建物理反事实评价器，属于知识／模型驱动与经典启发式搜索的组合。',
205:'决策输入包括当前关联、正队列、队首包等待时间、已成功服务 EMA、当前 OLLA/MCS、各 PRG 可用层数、服务预编码和分源信道。所有输入必须对应同一动作前时点；不得使用本次动作之后的实际 CRC、未来到达或未来信道。',
206:'实际系统可研究通过测量和簇内信息交换构造上述输入，但本报告不把这种设想写成已经实现的接口，也不声称不需要跨小区信道。信道估计误差、反馈开销和老化鲁棒性属于后续验证。',
207:'DA-PLS 不含神经网络和训练步骤。HOL 与服务 EMA 的逐时隙更新属于闭环反馈；精确上界和规范搜索提供结果一致性保证，均不等同于数据驱动学习或全局最优保证。',
240:'学习调度原型常采用“UE选择—PRG映射”两级动作，以降低组合动作的输出维数。本文将这一结构作为历史路线进行分析；最终 DA-PLS 直接在合法 UE—PRG 分配上做局部编辑，原生 slot 编码仅用于接口传输。',
337:'将 Cell、UE、PRG 组织为异构实体能够表达细粒度关系，是前期学习路线的建模探索。最终方法以分源物理信道和联合候选评价直接表达这些关系，不保留异构图网络作为在线主算法。',
340:'前期异构图研究促成了对服务关系、资源位置及干扰源身份的区分。DA-PLS 进一步把这些关系落到可计算的分源协方差与完整 TB 收益，判断一个静默或联合编辑是否带来簇级净增益。',
}

class Writer:
    def __init__(self,doc,fig):self.d=doc;self.fig=fig
    def p(self,s,style=None):return self.d.add_paragraph(s,style)
    def h(self,s,level=2):return self.d.add_heading(s,level)
    def eq(self,s,number):
        p=self.d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        math=OxmlElement('m:oMath');r=OxmlElement('m:r');t=OxmlElement('m:t');t.text=s;r.append(t);math.append(r);p._p.append(math);p.add_run('    ('+number+')')
    def table(self,title,headers,rows):
        caption=self.p(title,'Caption');caption.paragraph_format.keep_with_next=True
        t=self.d.add_table(rows=1,cols=len(headers));t.style='Table Grid'
        for c,s in zip(t.rows[0].cells,headers):c.text=str(s)
        props=t.rows[0]._tr.get_or_add_trPr();repeat=OxmlElement('w:tblHeader');props.append(repeat)
        for row in rows:
            for c,s in zip(t.add_row().cells,row):c.text=str(s)
        for i,row in enumerate(t.rows):
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
            for c in row.cells:
                for p in c.paragraphs:
                    p.alignment=WD_ALIGN_PARAGRAPH.LEFT
                    if len(rows)<=5:p.paragraph_format.keep_with_next=i<len(rows)
                    p.paragraph_format.space_after=Pt(3);p.paragraph_format.space_before=Pt(3)
                    for r in p.runs:r.font.size=Pt(9);r.bold=i==0
                if i==0:
                    sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E7EFF5');c._tc.get_or_add_tcPr().append(sh)
        self.p('');return t
    def image(self,name,caption):
        p=self.p('');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
        p.add_run().add_picture(str(self.fig/f'{name}.png'),width=Cm(16.2))
        cap=self.p(caption,'Caption');cap.paragraph_format.keep_with_next=False

def foundation(doc,original,fig):
    body=doc._element.body
    for e in list(body):
        if not e.tag.endswith('}sectPr'):body.remove(e)
    w=Writer(doc,fig)
    for _ in range(3):w.p('')
    p=w.p('基于AI技术的多维协同资源分配研究','Title');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p=w.p('技术研究报告','Title');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p=w.p('二阶段同频收尾 · DA-PLS','Subtitle');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    w.p('M-LWDF-PF 时延感知权重与上界剪枝的精确物理局部搜索').alignment=WD_ALIGN_PARAGRAPH.CENTER
    for _ in range(4):w.p('')
    w.p('2026年11月4日').alignment=WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break();w.p('修订摘要','Title')
    w.p('二阶段同频算法采用DA-PLS：M-LWDF-PF时延感知权重＋上界剪枝的精确DPLS内层。前期异构图与PPO转为路线演进背景，最终在线决策不含神经网络或训练环节。新方法以分源干扰协方差、MMSE-IRC和完整传输块评价计算联合移动的净收益，属于物理知识建模与经典搜索方法。')
    w.p('本次统一为3小区、36 UE、17 PRG×16 PRB的同频场景，保留Boundary08与uniform对照。队列bit/byte和阴影标准差单位修正前的性能数字全部撤回；正式结果采用固定版本、seed 41–50、2秒共同PFQ预热＋2秒稳定＋10秒测量＋0.2秒随访，按成熟到达包统计完整成功交付，并给出逐seed配对区间。')
    root=fig.parent
    for scene,rate in [('boundary',.8),('uniform',1.)]:
        v=complete_means(root,scene,rate,['pfq','mlwdf_pf'])
        if v:
            w.p(f'{scene}十seed结果：PFQ→DA-PLS的完整包吞吐为{v["pfq","payload_mbps"]:.3f}→{v["mlwdf_pf","payload_mbps"]:.3f} Mbps，成功包P95为{v["pfq","delay_p95_ms"]:.3f}→{v["mlwdf_pf","delay_p95_ms"]:.3f} ms，过期率为{v["pfq","expiry_pct"]:.3f}%→{v["mlwdf_pf","expiry_pct"]:.3f}%。')
    v=complete_means(root,'boundary',.8,['deficit','mlwdf_pf'])
    if v:
        w.p(f'在相同精确内层下，Boundary08的缺口权重→M-LWDF-PF完整包吞吐为{v["deficit","payload_mbps"]:.3f}→{v["mlwdf_pf","payload_mbps"]:.3f} Mbps，成功包P95为{v["deficit","delay_p95_ms"]:.3f}→{v["mlwdf_pf","delay_p95_ms"]:.3f} ms。权重效果与物理协同效果分别对照，不把两者混为同一个收益。')
    pair_path=root/'analysis/paired.csv'
    if pair_path.exists() and pair_path.stat().st_size>1:
        pairs=pd.read_csv(pair_path)
        u=pairs[(pairs.scene=='uniform')&(pairs.arm=='mlwdf_pf')&(pairs.subset=='41-50')&(pairs.n==10)&pairs.metric.isin(['delay_p95_ms','expiry_pct'])]
        if len(u)==2 and ((u.ci_low<=0)&(u.ci_high>=0)).all():
            w.p('uniform中，相对PFQ的成功包P95及过期率配对区间均跨0，均值改善尚不足以建立稳定优势；具体区间见第9章。')
    heavy=[]
    for rate in [1.12,1.28]:
        h=complete_means(root,'boundary',rate,['deficit','mlwdf_pf'])
        if h:
            delta=100*(h['mlwdf_pf','payload_mbps']/h['deficit','payload_mbps']-1)
            expiry=h['mlwdf_pf','expiry_pct']-h['deficit','expiry_pct']
            heavy.append(f'×{rate/.8:g}负载的整包吞吐变化为{delta:+.2f}%，过期率相差{expiry:+.3f}个百分点')
    if len(heavy)==2:w.p('高负载下DA-PLS相对缺口权重：'+'；'.join(heavy)+'。主负载结论不能直接外推到过载情形。')
    w.p('精确性指剪枝与规范完整邻域搜索的选择序列及终态一致，得到的是既定邻域和停止门槛下的局部稳定解。单核计算成本独立评测，不把相对降本称为满足0.5 ms实时调度；集中式簇内信道可得、activity-only干扰、固定MCS和发射／静默两档功率是当前研究边界。零过期也不等于零丢包，整包CRC损坏单列。')
    status=root/'analysis/completion.json'
    if not status.exists() or not json.loads(status.read_text())['completed']:
        w.p('版本范围：本版优先提供已完成的核心证据；未齐备的传统基线、消融及高负载结果在相应位置明确标注，后续补齐，不使用修正前数值占位。')
    doc.add_page_break();w.p('目录','Title')
    p=w.p('')
    def fieldpart(node):
        run=OxmlElement('w:r');run.append(node);p._p.append(run)
    begin=OxmlElement('w:fldChar');begin.set(qn('w:fldCharType'),'begin');fieldpart(begin)
    text=OxmlElement('w:instrText');text.set(qn('xml:space'),'preserve');text.text=' TOC \\o "1-2" \\h \\z \\u ';fieldpart(text)
    separate=OxmlElement('w:fldChar');separate.set(qn('w:fldCharType'),'separate');fieldpart(separate)
    cache=OxmlElement('w:t');cache.text='目录将在文档渲染时更新。';fieldpart(cache)
    end=OxmlElement('w:fldChar');end.set(qn('w:fldCharType'),'end');fieldpart(end)
    doc.add_page_break()
    for i in range(16,349):
        e=deepcopy(original[i])
        # Original REF fields point to bookmarks in the removed bibliography.
        # Keep their cached, verified citation numbers as ordinary text; the new
        # TOC and page-number fields remain live because they are created apart.
        for field in list(e.iter()):
            if field.tag in (qn('w:fldChar'),qn('w:instrText')):
                field.getparent().remove(field)
            elif field.tag==qn('w:fldSimple'):
                parent=field.getparent();offset=parent.index(field)
                for child in list(field):parent.insert(offset,child);offset+=1
                parent.remove(field)
        if i==62:w.image('closed_loop','图 2-2 下行调度的状态—动作—反馈闭环');continue
        if i==63:continue
        if i==79:w.image('resource_grid','图 2-3 项目 PRG 与 Type-0 位图式映射');continue
        if i==80:continue
        body.insert(len(body)-1,e)
        if e.tag.endswith('}p'):
            p=Paragraph(e,doc)
            if i in REPLACE:p.text=REPLACE[i]
            # Broken reference fields are removed, keeping the correct bibliography reference.
            if 'Error: Reference source not found' in p.text:
                assert i==216, f'Unresolved source reference at paragraph {i}'
                p.text=p.text.replace('Error: Reference source not found','[4]')
            if i==153:
                w.p('M-LWDF-PF 的优先级形式为 aᵤDᵤrᵤ,p/R̄ᵤ，其中 aᵤ=−ln(δᵤ)/Tᵤ。EXP-PF 的实时业务形式可写为 exp[(aᵤDᵤ−x̄)/(1+√x̄)]·rᵤ,p/R̄ᵤ，x̄ 为活动业务 aᵤDᵤ 的均值[14][35]。第8章将其时延权重移植到联合预期字节目标，不宣称完整复现原论文的调度器或最优性定理。')
            if i==161:
                w.h('局部搜索、分支定界与可采纳上界',3)
                w.p('局部搜索从一个可行分配出发，在约定邻域内逐轮接受改进。分支定界与最佳优先搜索进一步利用可采纳上界，排除不可能优于当前已知候选的分支[36]。本报告只在每轮局部候选集中使用上界证书；它保证剪枝不改变规范完整搜索的选择，不把局部邻域的精确选择扩大为全局最优。')
        elif e.tag.endswith('}tbl'):
            t=Table(e,doc)
            if i==96:
                for row in t.rows:
                    if row.cells[0].text=='RBG/项目PRG':row.cells[1].text='16 PRB/组，共17组'
                    if row.cells[0].text=='Slot/TTI':row.cells[2].text='状态提取、动作执行和服务反馈周期'
                    if row.cells[0].text=='发射功率':row.cells[2].text='统一功率，允许 PRG 发射／静默'
            if i==202:
                vals=['保留状态—动作—反馈的设计经验；不作为本阶段主算法','采用完整交付、TTL和尾时延作为评价指标','显式计算分源干扰与簇级净收益','采用物理模型和结构先验直接评价候选','复用原生仿真与外部决策接口；不宣称实时部署','作为后续异频资源维度扩展参考']
                for row,val in zip(t.rows[1:],vals):row.cells[-1].text=val
    return w

def chapter4(w):
    w.h('主流方法对比与本课题定位')
    w.p('本章的学习方法综述为路线选择提供背景，最终交付采用模型驱动的物理搜索。是否包含训练模型、是否依赖跨小区信道以及是否满足实时预算是三个独立问题，应分别说明。')
    w.table('表 4-6 主流路线与二阶段方案的关系',['路线','主要优势','本报告中的作用与边界'],[
      ['监督／模仿学习','推理流程紧凑，可模仿专家','已探索方向；不计为最终主算法'],
      ['强化学习／多智能体学习','可以优化跨时隙目标','保留原型和机制认识；修正前数字不作效果证据'],
      ['图结构学习','显式组织异构实体和拓扑','提供关系建模启发；最终在线路径无图网络'],
      ['物理建模＋精确剪枝搜索（本文）','无需训练，决策可解释，剪枝质量可证','依赖簇内信道；仅保留局部搜索等价性；计算仍需评估']])
    w.h('当前研究范围与可观测信息边界')
    w.p('研究范围是下行、三个小区、Reuse-1、项目 PRG 级资源映射、队列反馈及发射／静默协调。功率只取既定发射或关闭两档，空间维采用服务链路 SVD，不联合优化多小区波束。异频、载波激活和负载切换策略不属于本阶段已完成算法。')
    w.table('表 4-7 DA-PLS 输入及因果边界',['信息','使用方式','限制'],[
      ['队列、HOL等待、已成功服务EMA','生成 M-LWDF-PF 权重','只读动作前队列与历史反馈'],
      ['簇内信道／分源干扰协方差','计算动作相关 MMSE-IRC 与 BLER','集中式强观测假设；真实采集开销未验证'],
      ['OLLA、MCS、可用空间层','确定当前完整 TB 容量','本次搜索固定 MCS'],
      ['实际成功字节和 CRC','执行后更新EMA、统计整包生命周期','不提前作为本次候选输入']])
    w.h('原生仿真与物理决策闭环')
    w.p('C++/CUDA 端负责信道、业务、原生 MAC/PHY 执行及反馈，Python 负责因果状态整理与实验调度，C++ 内层负责精确物理搜索。每一步输出明确的 UE—PRG 分配，执行侧核对编号、归属、资源唯一性及物理参数。没有 PPO 更新、策略网络或训练 checkpoint。')
    w.p('“闭环自适应”仅指服务 EMA、队首等待和链路自适应随执行反馈更新。二阶段的验收重点是算法研究结果、修正后可复现实验与复杂度证据；达到0.5 ms调度截止时间仍须独立验证。')

def chapter5(w,root):
    w.h('仿真环境与平台问题修正',1)
    w.h('仿真平台与运行身份')
    w.p('仿真基于 NVIDIA Aerial CUDA-Accelerated RAN 的 cuMAC 多小区下行示例及项目研究接口。cuMAC承担 UE 选择、资源映射和链路自适应，CUDA执行信道及相关物理计算；报告使用固定研究仿真器和库文件，避免把当前工作区中其它尚未提交的改动混入结果。')
    w.table('表 5-1 本次重跑环境及版本',['项目','取值'],[
      ['宿主系统','Ubuntu 22.04 / x86_64；Python 3.10'],['CPU','Intel Xeon Platinum 8454H；串行成本固定一个逻辑核'],['本次可用 GPU','NVIDIA A800 80GB PCIe（1张）'],
      ['原生仿真器 SHA-256','a68f48668bc623c0bc793c452f1b0cd5d8943929eb26e8eefd25f776d180b615'],
      ['libcumac.so SHA-256','7b414f13851e50da73c91db3fc1342017113c8180a04148606f24a5adcc08eea'],
      ['内层','capacity_bound_q1e6_v1；运行库与输入哈希随实验归档'],
      ['计时区分','并发闭环运行用于质量；同输入单线程串行回放用于相对复杂度']])
    w.h('安装配置与复现流程')
    w.p('基础安装步骤依次为：确认 NVIDIA 驱动和 CUDA 可用；准备 Aerial 容器及其运行时依赖；进入既有容器并启用项目 Python 环境；核对冻结仿真器、库文件和源码清单；先运行短轨迹及行为测试，再运行正式实验。安装命令应依据实际机器及镜像版本执行，本报告不将示例安装版本作为算法参数。')
    w.p('本次实验入口为 training.stage2_closure.campaign，统计入口为 training.stage2_closure.analyze；原始记录位于 deliverables/stage2_closure_20261104。每条轨迹保存启动参数、输入哈希、实际动作、原生 UE×TTI 指标、逐包到达和终态、验证记录及保存状态。旧结果不覆盖；失败启动和中断轨迹独立保留，不进入指标表。')
    if (root/'execution_priority.json').exists():
        w.p('为先交付核心对比，运行期间曾暂缓本批次的部分扩展轨迹，将资源优先提供给尚未完成的PFQ和缺口权重对照；随后恢复扩展轨迹。调度只影响宿主机的实际运行先后，信道、到达、EMA、TTL与测量窗口均按逻辑TTI推进，不据墙钟暂停时间增加仿真时延。执行事件另行归档，连续墙钟不用于跨臂公平加速比较。')
    w.h('平台单位缺陷及其影响')
    w.p('本节修正清单仅包含队列扣减 bit/byte 不一致和阴影衰落标准差单位不一致两项。它们会改变后续调度状态和物理分布，因此不能通过对旧吞吐数字乘常数完成校正。修正前报告中的效果表和图全部撤回，不再用于本报告的性能结论。')
    w.table('表 5-2 两项平台修正及核验方法',['问题','错误及后果','修正与验证'],[
      ['队列扣减单位[37]','TBS以bit返回，队列以byte存储；直接相减最多造成8倍扣减','先把TBS除以8转为byte，再取min(Q,TBS)；逐TTI核对出队及FIFO守恒'],
      ['阴影标准差单位[38]','3 dB直接传给对数正态分布的自然对数标准差，实际约13.03 dB','使用σln=3·ln(10)/10；独立采样统计及原生运行日志中的转换参数双重核对']])
    w.eq('D = min{Q, TBS_bit / 8}；S = D · (1 − TB_error)','5-1')
    w.p('例如984 bit的容量对应123 byte；若队列充足，应扣减123 byte而非984 byte。成功字节又必须乘本次传输成功标志。正式验证同时核对 Qpre=Qprev+A−X、Qpost=Qpre−D 和 D=S+F；MAC出队完成不自动构成整包成功。')
    w.eq('σ_dB = (10 / ln 10) σ_ln；σ_ln = σ_dB · ln 10 / 10','5-2')
    path=root/'e0/shadow_statistics.csv'
    if path.exists():
        z=pd.read_csv(path);rows=[]
        for case,name in [('legacy','旧参数'),('corrected','修正参数')]:
            q=z[z['case']==case];rows.append([name,'41–50；各100万次',f'{q.sample_std_db.mean():.6f}',f'{q.sample_std_db.min():.6f}–{q.sample_std_db.max():.6f}',f'{q.theoretical_std_db.mean():.6f}'])
        w.table('表 5-3 阴影标准差独立重采样（单位dB）',['实现','重复设置','实测标准差均值','逐seed范围','理论值'],rows)
    w.p('表5-3是对采样表达式的独立统计复现，不是修正前后PFQ吞吐的单因素反事实实验。既有历史轨迹含有不同版本因素，不能将差值全部归因于某一项缺陷。修复后的物理动作一致性、字节守恒和相同种子配对验证为本报告结果的有效性依据。')
    path=root/'analysis/completion.json'
    if path.exists():
        c=json.loads(path.read_text())
        if c.get('physical_states',0):
            w.p(f'已完成搜索轨迹的{c["physical_states"]}个保存状态另做完整TB物理核验：所选动作的双精度评价与独立NumPy MMSE实现一致，再与原生实际发送记录比较。SINR最大绝对差为{c["physical_sinr_max_abs_db"]:.3e} dB，BLER最大绝对差为{c["physical_bler_max_abs"]:.3e}；分别小于预设0.005 dB和0.0005容差。动作、PRG数、层数和出队字节另按整数精确核对，不将浮点容差与搜索逐位等价混为一谈。')

def chapter6(w):
    w.h('同频场景与统一实验协议',1)
    w.h('Boundary08主场景与uniform对照')
    w.p('主场景Boundary08采用三个同频小区、36个全边界布点UE，服务小区按最强接收链路关联。uniform对照保持三个小区及36个UE，使用均匀布点。主场景到达率为0.8包/UE/TTI；uniform为1.0。由于几何和负载同时改变，跨场景比较解释为综合稳健性检查，不作为只改变布点的单因素消融。')
    w.table('表 6-1 统一系统与业务参数',['参数','设置'],[
      ['小区 / UE','3 / 36；按实际最强链路关联'],['小区半径 / 站点间距','500 m / 1000 m'],
      ['载频 / 原生噪声底','2.5 GHz / −98.9 dBm'],['基站功率','49 dBm；按4发射天线及17 PRG均分的固定基准'],
      ['下行与复用','DL、Reuse-1、固定基础发射功率；允许PRG静默'],['收发天线 / 预编码','4T4R / 服务链路SVD'],
      ['频域粒度','每小区17 PRG × 16 PRB = 272 PRB'],['信道 / 接收','时间相关Rayleigh / MMSE-IRC'],
      ['信道递推参数','相关系数0.9938、实／虚部创新系数0.0786；每次信道更新按冻结实现使用'],
      ['阴影','显式3 dB；自然对数参数正确换算'],['TTI / TTL','0.5 ms / 200 ms（400 TTI）'],
      ['业务','泊松到达；3000 byte/包；同一场景同seed各臂共同到达序列'],
      ['到达率','Boundary：0.8；uniform：1.0；高负载Boundary：1.12、1.28'],
      ['HARQ / 主动丢包','均关闭；保留自然TTL过期和分段CRC失败'],
      ['链路自适应','OLLA；一次搜索固定当前MCS'],['初选与下界','原始PFQ；本臂当前总出队量90%，向上取整']])
    w.p('配置输入速率为 λ×36×3000×8/0.0005：Boundary主负载为1382.4 Mbps，uniform为1728 Mbps；×1.4和×1.6负载分别为1935.36和2211.84 Mbps。它们是期望到达配置，实际到达量按日志统计。高负载倍数均以Boundary的0.8为基准，不以uniform的1.0为基准。')
    w.h('共同前缀、稳定窗口与成熟到达cohort')
    w.table('表 6-2 每条正式轨迹的时序',['阶段','TTI范围','时长','作用'],[
      ['共同PFQ预热','1–4000','2 s','所有策略执行相同原始PFQ'],['各策略稳定','4001–8000','2 s','切换目标策略；不重置状态'],
      ['正式测量','8001–28000','10 s','业务服务及计算指标；选取到达cohort'],['随访','28001–28400','0.2 s','追踪测量期到达包至终态']])
    w.p('队列、EMA、OLLA与连续信道时钟跨阶段保持连续。配对要求同seed、同场景、同负载的拓扑、全部到达及4000 TTI预热动作和输入完全一致。分叉后的队列、MCS与服务EMA允许随策略不同而分化。完整包指标只统计测量窗口内到达的包，随访结束不得存在截尾包。')
    w.h('指标定义与统计单位')
    w.table('表 6-3 主要指标及分母',['指标','定义'],[
      ['完整包吞吐','测量期到达且所有分段成功的整包payload bit总量 / 10 s'],
      ['分段goodput','测量窗内实际成功分段byte总量 ×8 / 10 s'],
      ['过期率','成熟到达cohort中自然TTL过期包数 / 全部到达包数'],
      ['成功包均时/P95/P99','仅完整成功包；(终态TTI−到达TTI+1)×0.5 ms'],
      ['其它未交付','未完整成功且非自然过期的包比例；本设置主要为CRC损坏'],
      ['UE P5 / Jain / 最差UE','先求各UE速率或过期率，再计算单seed的P5、公平指数或最大过期率'],
      ['SINR/MCS','实际发送TB对应的物理接收SINR/MCS分布'],
      ['资源占用','分配的cell-PRG数 / (20000×3×17)；空置不全等于主动静默'],
      ['每已用PRG成功字节','测量窗成功字节总量 / 实际分配cell-PRG数'],
      ['计算成本','每TTI候选评分、UE评价及MMSE计数；墙钟均值/P95/最大值']])
    w.p('所有正式效果表使用seed 41–50共10次独立重复，先在每个seed内计算指标，再等权求均值和样本标准差。特别是P95、P99和最差UE指标不把各seed包合池后再计算。配对95%区间为 d̄±t0.975,n−1·sd/√n，其中d为同seed方法减PFQ的差值；不把TTI或包当独立重复，不作多重比较校正。')
    w.p('M-LWDF-PF已在41–43上选型，主表另列44–50均值作为未参与本次权重选择的稳健性子集。44–50在项目历史其它探索中存在暴露，因此该子集不称完全未见的最终盲测。本文不进行新的权重调参、负载切换或训练。')

def metric_rows(root,scene,rate,arms,metrics,subset='41-50'):
    path=root/'analysis/aggregate.csv'
    if not path.exists():return []
    data=pd.read_csv(path);rows=[]
    required=10 if subset=='41-50' else 7
    for arm in arms:
        z=data[(data.scene==scene)&(data.rate==rate)&(data.arm==arm)&(data.subset==subset)].set_index('metric')
        if not all(m in z.index and z.loc[m,'n']==required for m in metrics):continue
        values=[]
        for metric in metrics:
            value=z.loc[metric,'mean']
            values.append(f'{value:.4f}' if metric.startswith('jain') else f'{value:.2e}' if 0<abs(value)<.0005 else f'{value:.3f}')
        rows.append([LABEL[arm]]+values)
    return rows

def result_table(w,root,title,scene,rate,arms,headers,metrics,subset='41-50'):
    rows=metric_rows(root,scene,rate,arms,metrics,subset)
    if len(rows)==len(arms):w.table(title,headers,rows)
    else:w.p(title+'：正式10-seed重跑尚未完整通过，本版暂不填入结果，后续补齐。')

def complete_means(root,scene,rate,arms,subset='41-50'):
    path=root/'analysis/aggregate.csv'
    if not path.exists():return None
    df=pd.read_csv(path);n=10 if subset=='41-50' else 7
    df=df[(df.scene==scene)&(df.rate==rate)&(df.subset==subset)&(df.n==n)]
    if not set(arms)<=set(df.arm):return None
    return {(r.arm,r.metric):r.mean for r in df.itertuples(index=False)}

def chapter7(w,root):
    w.h('同频场景传统基线重新评测',1)
    w.h('对照算法与实现语义')
    w.p('所有传统基线运行于相同修复后的原生环境。为消除旧RR实现依赖墙钟随机数的不可复现因素，本轮明确采用可复现的RR、RRQ和PF规则；它们是本报告定义的传统对照实现，不声称与NVIDIA历史基线二进制逐位同一。PFQ沿用冻结环境导出的原始工程调度动作。')
    w.table('表 7-1 本轮基线规则',['算法','选择规则','队列及历史服务处理'],[
      ['RR','逐小区轮转所有正队列关联UE；按PRG轮流分配','跳过空队列；不按队列大小限制已分配PRG'],
      ['RRQ','相同轮转次序；达到队列容量需求后停止给该UE继续分配','按当前MCS及所分PRG最小层数的TBS覆盖Q判断；非精确全局打包'],
      ['PF','逐PRG最大化 Σ层log₂(1+γ)/max(成功服务EMA,10⁻⁹)','同一活动掩码和固定层数信息；无按队列大小的资源截断'],
      ['PFQ','直接执行冻结原生环境的queue-aware PF初选','需求余量为0 byte；作为主性能对照与所有搜索臂的初选']])
    w.p('RR与RRQ的配对用于检查需求限制的作用；PF与PFQ还包含具体实现差异，因此不能把二者差值全部解释为唯一的队列感知因子。所有新规则的原生执行动作、MCS、层数、TBS和出队字节均逐TTI核验。')
    w.h('Boundary08主场景结果')
    result_table(w,root,'表 7-2 Boundary08传统基线（10个seed等权均值）','boundary',.8,['rr','rrq','pf','pfq'],
      ['算法','整包Mbps','过期%','成功均时ms','P95 ms','UE goodput P5'],
      ['payload_mbps','expiry_pct','delay_mean_ms','delay_p95_ms','ue_goodput_p5_mbps'])
    result_table(w,root,'表 7-3 Boundary08资源效率与物理传输状态','boundary',.8,['rr','rrq','pf','pfq'],
      ['算法','goodput Mbps','SINR dB','MCS','占用%','成功B/PRG'],
      ['goodput_mbps','sinr_tx_mean_db','mcs_tx_mean','prg_occupancy_pct','bytes_per_used_prg'])
    v=complete_means(root,'boundary',.8,['rr','rrq','pf','pfq'])
    if v:
        w.p(f'在相同轮转规则下加入队列需求限制，RR到RRQ的完整包吞吐为{v["rr","payload_mbps"]:.3f}→{v["rrq","payload_mbps"]:.3f} Mbps，资源占用为{v["rr","prg_occupancy_pct"]:.3f}%→{v["rrq","prg_occupancy_pct"]:.3f}%。PF与PFQ的完整包吞吐分别为{v["pf","payload_mbps"]:.3f}和{v["pfq","payload_mbps"]:.3f} Mbps；该组还包含具体优先级及实现差异，需与第7.1节定义一起解释。')
        if abs(v['rrq','payload_mbps']/v['rr','payload_mbps']-1)<.01 and min(v['rr','prg_occupancy_pct'],v['rrq','prg_occupancy_pct'])>=99.9:
            w.p('在本Boundary负载下，RR与RRQ均接近满资源占用，整包吞吐差距不足1%；单独增加队列容量限制未带来明显的吞吐均值变化。该结论只适用于当前配置，不排除较轻负载下需求限制的作用，也不将其等同于跨小区干扰协调。')
    w.p('完整包吞吐作为主指标，分段goodput作为服务过程指标。两者可能出现不同变化方向，因为分段服务可能最终归属于CRC损坏或过期的包。成功包时延也具有条件筛选效应：必须与整包交付、自然过期和其它未交付比例共同解释。')
    w.h('uniform场景结果与基线启示')
    result_table(w,root,'表 7-4 uniform/rate1.0传统基线（10个seed等权均值）','uniform',1.,['rr','rrq','pf','pfq'],
      ['算法','整包Mbps','过期%','成功均时ms','P95 ms','UE goodput P5'],
      ['payload_mbps','expiry_pct','delay_mean_ms','delay_p95_ms','ue_goodput_p5_mbps'])
    v=complete_means(root,'uniform',1.,['rr','rrq'])
    if v:
        w.p(f'uniform场景中，RR到RRQ的整包吞吐为{v["rr","payload_mbps"]:.3f}→{v["rrq","payload_mbps"]:.3f} Mbps，过期率为{v["rr","expiry_pct"]:.3f}%→{v["rrq","expiry_pct"]:.3f}%，资源占用为{v["rr","prg_occupancy_pct"]:.3f}%→{v["rrq","prg_occupancy_pct"]:.3f}%。两个主场景的布点与负载同时不同，不能将相对收益的差异全部归因于布点变化。')
    w.p('两个场景分别给表，不把不同负载、不同布点的包混池。基线结果为后续协同算法提供可检验的出发点：队列感知可以减少无效占用，但逐小区规则本身不直接求解“本小区静默是否足以补偿本地服务代价”的簇级净收益。该问题由第8章物理评价与联合邻域显式处理。')

def chapter8(w):
    w.h('时延感知物理建模协同调度算法DA-PLS',1)
    w.h('技术路线演进与设计动机')
    w.p('前期以Cell—UE—PRG异构图和PPO探索联合调度，随后开展分源干扰表达、物理反事实评价与学习加速。二阶段采用证据驱动的路线收敛：把可解释的物理评价器直接用于决策，并将计算优化限制在保持规范搜索结果一致的范围内。原先的神经联合修复方案不作为本阶段最终在线主算法。')
    w.table('表 8-1 技术路线演进与保留认识',['阶段','探索内容','导向最终方案的认识'],[
      ['异构图＋PPO','关系建模、两级离散动作、长期奖励','服务次序可改善；但不能仅凭时延收益证明干扰协调，执行动作改写也影响训练目标'],
      ['复用注意力与定向协调','识别谁造成干扰、谁应受保护','聚合干扰可能丢失源身份；需要分源反事实'],
      ['物理局部搜索','完整TB评价，换UE、静默和联合移动','直接评价本地代价与邻区收益；计算量成为主要问题'],
      ['学习加速尝试','神经修复、服务选择、预算与排序','减少候选数不必然减少总时延，近似轨迹可能损害弱UE服务'],
      ['DA-PLS','M-LWDF-PF权重与精确上界剪枝','以明确业务权重改善服务；以可证明等价的剪枝削减物理评分']])
    w.image('evolution','图 8-1 从学习原型到DA-PLS的路线演进')
    w.p('上述历史探索用于说明设计取舍，修正前环境的定量收益不作为本报告性能证据。本文也不作“PPO普遍无效”的结论：在当时的动作和奖励设计下，服务次序与干扰协调并未被稳定地同时解决。最终方法吸收了分源干扰、完整TB耦合和净收益评价三项关键认识。')
    w.p('同频场景需要回答三个具体问题：哪个邻区对哪个UE形成干扰，当前应优先保护谁，以及让出一个资源位置能否增加簇级净收益。如果两条干扰信道互换后聚合特征保持不变，关闭不同干扰源的效果却可能互换。DA-PLS以分源协方差保留这一可辨识性。')
    w.h('动作、可行集合与优化目标')
    w.p('在时隙t，令A(c,p)表示小区c的PRG p上服务的UE，或静默符号∅。每个位置至多一个UE，UE必须与该小区关联、具有正队列并满足资源可用性。记Q为动作前队列byte数，dᵤ(A)=min{Qᵤ,TBSᵤ(A)}，eᵤ(A)=dᵤ(A)[1−BLERᵤ(A)]。')
    w.eq('max A∈Fₜ  Jₜ(A) = Σᵤ wᵤ,ₜ · min{Qᵤ,ₜ, TBSᵤ(A)} · [1 − BLERᵤ(A)]','8-1')
    w.eq('D(A) = Σᵤ dᵤ(A) ≥ ⌈0.90 · D(A⁰PFQ)⌉','8-2')
    w.p('A⁰PFQ是本臂、本时隙当前状态下的原始PFQ初选。下界每个TTI重算，不读取参照策略未来服务，不采用跨臂外部下界。初选天然可行。该约束限制的是计划出队量，既不是实际成功吞吐的90%保证，也不是跨时隙QoS保证。')
    w.p('TBS由当前MCS、分配PRG数及跨PRG最小可用层数共同决定。修改一个位置可能改变整块的层数、容量和BLER，因此不能把若干独立PRG速率相加代替完整TB评价。')
    w.h('显式物理评价器')
    w.p('记zᵦ,p为邻区b在PRG p是否发射，Cᵦ→ᵤ,p为该干扰源对UE u的协方差贡献，Sᵤ,p为服务链路信道与固定SVD预编码的乘积。接收端按MMSE-IRC计算：')
    w.eq('Rᵤ,p(A) = σ²I + Σᵦ≠c(u) zᵦ,p(A) Cᵦ→ᵤ,p','8-3')
    w.eq('γᵤ,p,l = 1 / [(I + Sᴴᵤ,p R⁻¹ᵤ,p Sᵤ,p)⁻¹]ll − 1','8-4')
    w.p('按冻结原生口径对已分PRG及有效层的线性SINR取平均，再转换为TB级dB值，并用同一MCS的原生BLER表插值。一次候选编辑重新评价所有受影响UE的完整TB，既包含让出资源的本地代价，也包含相同PRG上邻区UE的干扰变化。独立物理核验区分浮点容差验证与搜索的整数逐位一致验证。')
    w.p('本报告采用activity-only干扰模型：在一个PRG上，邻区发射贡献由其活动状态决定，服务UE身份不改变邻区协方差。因此三个小区时，每个(UE,PRG)只有最多2²=4种干扰掩码，整个时隙最多36×17×4=2448个MMSE键。该简化不等同于面向真实调度波束的全部干扰模型[39]。')
    w.h('M-LWDF-PF时延感知权重')
    w.eq('a = −ln(0.01)/400；ageᵤ,ₜ = max{1, 400 − slackᵤ,ₜ}','8-5')
    w.eq('ŵᵤ,ₜ = a · ageᵤ,ₜ / max{s̄ᵤ,ₜ, 10⁻⁹}；wᵤ,ₜ = ŵᵤ,ₜ / meanᵥ(ŵᵥ,ₜ)','8-6')
    w.eq('s̄ᵤ,ₜ₊₁ = 0.995 s̄ᵤ,ₜ + 0.005 Sᵤ,ₜ','8-7')
    w.p('age以TTI计，使用队首存活包的等待状态；新到达包采用一个时隙的年龄下限。s̄以每TTI实际成功服务byte为单位，初值为0并从共同预热开始累计。所有权重正且在当前搜索内固定，均值归一化使1 byte和16 byte阈值具有统一尺度。空队列仍可有有限权重，但服务项为0。')
    w.p('HOL读取不再次执行FIFO到期更新，避免破坏原有过期记账。M-LWDF-PF采用经典M-LWDF与比例公平的权重形式[14]；同一a在均值归一化后会抵消，δ=0.01是形式参数，不代表本算法已经保证1%丢包率。本文不把该权重本身作为新发明，也不移植原文的最优性保证。')
    w.h('联合邻域与规范局部搜索')
    w.table('表 8-2 每个PRG上的三类编辑',['移动','示意','评价重点'],[
      ['换UE或激活','(a,b,c) → (a′,b,c)；a可为∅','不同UE服务收益及整TB变化；允许重新激活静默位置'],
      ['单小区静默','(a,b,c) → (∅,b,c)','让出代价与邻区受保护收益'],
      ['静默＋邻区换UE／激活','(a,b,c) → (∅,b′,c)；b可为∅','同时保护合适受益者；捕获两步单独操作不易接受的联合改进']])
    w.p('每轮在全部可行编辑中选择最大整数增益；前两轮要求严格超过1 byte，之后要求严格超过16 byte。最大值一个定点单位以内视为规范平局带，按编辑操作的规范键选择最小者。至少完成规定确认轮，直到不存在超过相应阈值的可行移动。256轮是异常防护上限，触及上限判为验证失败，不作为预算终止输出。')
    w.p('准确地说，终态是既定邻域及停止阈值下的局部稳定解。较小的正增益可能低于16 byte阈值，较远的多PRG联合改变也可能更优；因此“精确内层”仅指与规范完整邻域搜索一致，不意味着全局最优或所有正增益都被消除。')
    w.h('精确加速与上界证书')
    w.p('第一步，使用持久稀疏编辑表。候选仅保存一个PRG上的编辑和规范键，跨轮复用。接受移动后重建该PRG邻域，并将共享受影响UE的精确评价及上界标记失效；其它候选保持有效。未评分候选在无关根变化后仍保持未评分，不为缓存一致性额外触发物理评价。')
    w.p('第二步，将每个UE贡献按最近偶数舍入量化到10⁻⁶加权byte，并以int64求和。候选得分由根得分加受影响UE的整数差得到。量化前的乘法次序、规范键、阈值和输入精度固定；先检查整数溢出界，避免将不同浮点归约顺序误称为逐位等价。')
    w.eq('q(x) = round_even(10⁶ x)；Jq(A) = Σᵤ q(wᵤ eᵤ(A))','8-8')
    w.p('第三步，不调用MMSE即计算候选容量上界。固定MCS时，队列、PRG计数与跨PRG最小层数足以得到dᵤ(A′)。对受影响UE集合Uaff，定义整数增益上界：')
    w.eq('U(A′) = Σᵤ∈Uaff [q(wᵤ dᵤ(A′)) − q(wᵤ eᵤ(A))]','8-9')
    w.p('命题1（上界可采纳）。在正权重、固定MCS、BLER∈[0,1]、一致量化规则和无整数溢出的条件下，U(A′)≥Jq(A′)−Jq(A)。证明：对任一受影响UE，eᵤ(A′)=dᵤ(A′)(1−BLERᵤ(A′))≤dᵤ(A′)。正数乘法与最近偶数取整保持非严格单调性，故q(wᵤeᵤ(A′))≤q(wᵤdᵤ(A′))。逐UE减去相同根贡献并相加即得不等式；未受影响项完全抵消。该证明不依赖“干扰越少必然容量越大”的额外假设。')
    w.p('第四步，每轮从当前有效上界重建堆，按上界降序、规范键升序排列。有效精确缓存首先提供已知最大增益m。对尚未评分且出队可行的候选，只要上界U≥m，或U=m−1且规范键小于当前赢家，就继续精评；上界不超过改善阈值的候选直接排除。当所有未知候选均不能改变规范赢家时，获得停止证书。若尚无超过门槛的候选，则必须排除所有可能超过门槛的未知项。')
    w.p('命题2（选择序列与终态等价）。假定两实现具有相同根、候选全集、出队下界、整数目标、改善门槛及一个单位平局规则，且精确缓存失效正确、无预算提前停止。由命题1，任何被跳过候选的真实增益不超过其上界；证书排除了其超越已知最大值或在平局带以更小规范键获选的可能。因此本轮赢家与完整枚举一致。接受相同赢家后两者得到相同根，按轮归纳可得全部根、选择序列和终态一致。该结论不把某种证书的必评集合解释为所有可能算法的信息论下界。')
    w.image('solver','图 8-2 DA-PLS的精确剪枝搜索流程')
    w.h('算法步骤与复杂度')
    w.table('表 8-3 DA-PLS每TTI执行步骤',['步骤','操作'],[
      ['1','读取同一动作前状态；以历史成功服务与HOL生成并归一化M-LWDF-PF权重。'],
      ['2','解码原始PFQ初选；按当前状态计算D₀及⌈0.9D₀⌉。'],
      ['3','建立三类候选编辑、影响依赖和廉价容量上界；保留有效精确缓存。'],
      ['4','按上界最佳优先精评，直到足以证明本轮规范赢家或无合格移动。'],
      ['5','有合格移动则更新分配和依赖失效，返回步骤3；否则按确认规则停止。'],
      ['6','输出分配；核验原生执行，按实际成功字节更新EMA，推进队列与OLLA。']])
    w.p('令C为小区数、P为PRG数、Uc为单小区候选UE数、R为接受轮数。三类邻域的规模量级为O(P·C²·max Uc)。完整枚举每轮均评价候选的受影响完整TB；剪枝版将物理评分数减为有证书需求的K≤N，同时仍承担上界维护、依赖失效与堆构建成本。最坏情况下上界全部不紧，K可等于N，因此不声称改善最坏情况复杂度阶。')
    w.p('缓存的物理键最多U·P·2^(C−1)，在本场景为2448；高负载时许多候选共享这些键，减少候选评分不等于同比减少矩阵运算。复杂度报告必须同时给出候选数、UE评价数、MMSE miss和墙钟，并以同架构完整搜索为主要分母。Python完整枚举与C++内层的比较另列，以区分工程实现收益和剪枝收益。')
    w.h('适用边界')
    w.p('DA-PLS依赖集中式簇内信道和activity-only干扰模型，输出既定阈值下的局部稳定分配。本阶段实现了物理模型、闭环权重与精确剪枝的组合，不包含学习模型训练。零过期不等于零丢包，出队下界不保证成功服务下界，时延感知权重不保证所有负载都最优。高负载需同时观察缺口权重对照；不在本报告中引入未验证的负载切换。')
    w.p('研究实现的单线程CPU搜索成本与0.5 ms的TTI预算分别报告。即使相对完整搜索有明显降本，也不能据此称为实时调度器或可直接部署产品。二阶段就绪指方法、实现和修正后证据可审查；工程实时性、非理想信道和更大规模仍是后续条件。')

def chapter9(w,root):
    w.h('DA-PLS仿真评估与机制分析',1)
    w.h('实验矩阵与证据使用原则')
    w.table('表 9-1 二阶段重跑矩阵',['编号','内容','本轮设置'],[
      ['E0','两项单位修正验证','独立阴影采样；队列/TBS/FIFO逐TTI一致性'],
      ['E1','传统基线','Boundary08与uniform；RR/RRQ/PF/PFQ，各10seed'],
      ['E3','主对比','两场景PFQ、缺口权重DPLS、DA-PLS；同协议各10seed'],
      ['E4','权重消融','Boundary08：等权、缺口、κ1、κ2、EXP-PF、M-LWDF-PF'],
      ['E5','负载敏感性','Boundary×1.0/1.4/1.6；PFQ、缺口、DA-PLS，各10seed'],
      ['E7','等价性与复杂度','各负载M-LWDF权重保存状态；独立完整枚举核验及串行计时'],
      ['E8','物理机制','实际发送SINR/MCS、资源复用、整包过期分布、真实联合编辑案例']])
    w.p('E1、E3、E4和E5重叠的实验点只运行一次，共220个配置—seed点。只复用已验证的同版本、同协议PFQ长窗口轨迹；原带额外初选裁剪的DPLS结果不进入本报告正式对照。所有搜索臂均以原始PFQ为初选，使用相同精确内层与90%出队下界。混合布点、0.4–1.0细负载扫描和邻域／ρ扫描不列为本轮已完成实验。')
    w.h('主场景与uniform的配对收益')
    cols=['算法','整包Mbps','过期%','成功均时ms','P95 ms','P99 ms']
    mets=['payload_mbps','expiry_pct','delay_mean_ms','delay_p95_ms','delay_p99_ms']
    arms=['pfq','deficit','mlwdf_pf']
    result_table(w,root,'表 9-2 Boundary08主对比（seed41–50等权均值）','boundary',.8,arms,cols,mets)
    result_table(w,root,'表 9-3 Boundary08稳健性子集（seed44–50等权均值）','boundary',.8,arms,cols,mets,'44-50')
    result_table(w,root,'表 9-4 uniform/rate1.0主对比（seed41–50等权均值）','uniform',1.,arms,cols,mets)
    result_table(w,root,'表 9-5 uniform稳健性子集（seed44–50等权均值）','uniform',1.,arms,cols,mets,'44-50')
    path=root/'analysis/paired.csv'
    if path.exists() and path.stat().st_size>1:
        df=pd.read_csv(path);rows=[]
        for scene,rate in [('boundary',.8),('uniform',1.)]:
            z=df[(df.scene==scene)&(df.rate==rate)&(df.arm=='mlwdf_pf')&(df.subset=='41-50')&(df.n==10)]
            names={'payload_mbps':'整包吞吐/Mbps','expiry_pct':'过期率/百分点','delay_mean_ms':'成功均时/ms','delay_p95_ms':'成功P95/ms','ue_payload_p5_mbps':'UE整包速率P5/Mbps'}
            for metric,name in names.items():
                q=z[z.metric==metric]
                if len(q):
                    a=q.iloc[0];rows.append([scene,name,f'{a.delta:+.3f}',f'[{a.ci_low:+.3f},\n{a.ci_high:+.3f}]',f'{a.relative_change_pct:+.2f}%' if pd.notna(a.relative_change_pct) else '基线为0，不定义'])
        if rows:w.table('表 9-6 DA-PLS减PFQ的逐seed配对差值与95%区间',['场景','指标','均值差','95%配对区间','相对均值变化'],rows)
    w.p('相对提升以两臂跨seed均值之比计算；配对区间以同seed差值计算，二者不是同一个统计量。区间跨0时不声称已建立稳定优势。成功包尾时延须结合交付率解释；特别保留其它非过期未交付比例，避免把“零过期”写成“零丢包”。完整逐seed和逐UE表随数据包提供。')
    for scene,rate in [('boundary',.8),('uniform',1.)]:
        v=complete_means(root,scene,rate,['pfq','mlwdf_pf'])
        if v:
            delta=100*(v['mlwdf_pf','payload_mbps']/v['pfq','payload_mbps']-1)
            w.p(f'{scene}中DA-PLS相对PFQ的整包吞吐均值变化为{delta:+.2f}%；成功包P95由{v["pfq","delay_p95_ms"]:.3f}变为{v["mlwdf_pf","delay_p95_ms"]:.3f} ms，最差UE过期率由{v["pfq","worst_ue_expiry_pct"]:.3f}%变为{v["mlwdf_pf","worst_ue_expiry_pct"]:.3f}%。这几个量分别反映有效交付、已成功包的尾部等待和弱UE风险，不能相互替代。')
            if v['mlwdf_pf','expiry_pct']==0:w.p(f'{scene}的DA-PLS在本轮测量到达cohort内未观测到自然过期；仍有{v["mlwdf_pf","other_nondelivery_pct"]:.3f}%其它未交付，不能外推为任意负载下的零过期或零丢包保证。')
        baseline_arms=['rr','rrq','pf','pfq']
        all_v=complete_means(root,scene,rate,baseline_arms+['mlwdf_pf'])
        if all_v:
            best=max(baseline_arms,key=lambda a:all_v[a,'payload_mbps'])
            delta=100*(all_v['mlwdf_pf','payload_mbps']/all_v[best,'payload_mbps']-1)
            w.p(f'{scene}的四种传统基线中，{LABEL[best]}按10seed整包吞吐均值排名最高，为{all_v[best,"payload_mbps"]:.3f} Mbps；DA-PLS为{all_v["mlwdf_pf","payload_mbps"]:.3f} Mbps，相对变化{delta:+.2f}%。该排名只针对整包吞吐均值，时延、过期及公平性指标分别列示。')
    path=root/'analysis/weight_paired.csv'
    if path.exists() and path.stat().st_size>1:
        wp=pd.read_csv(path)
        for scene,rate in [('boundary',.8),('uniform',1.)]:
            rows=wp[(wp.scene==scene)&(wp.rate==rate)&(wp.subset=='41-50')&(wp.n==10)].set_index('metric')
            if {'payload_mbps','delay_p95_ms','expiry_pct'}<=set(rows.index):
                parts=[]
                for metric,name,unit in [('payload_mbps','整包吞吐','Mbps'),('delay_p95_ms','成功包P95','ms'),('expiry_pct','过期率','个百分点')]:
                    r=rows.loc[metric];parts.append(f'{name}差值{r.delta:+.3f} {unit}（95%配对区间[{r.ci_low:+.3f}, {r.ci_high:+.3f}]）')
                w.p(f'隔离同一精确内层下的权重作用，{scene}中DA-PLS相对缺口权重DPLS：'+'；'.join(parts)+'。该比较同时固定物理评价、初选、邻域和剪枝规则，衡量完整权重形式在闭环中的影响。')
                if scene=='uniform' and rows.loc['payload_mbps'].ci_high<0:
                    w.p(f'uniform下，M-LWDF-PF整包吞吐比缺口权重低{abs(rows.loc["payload_mbps"].relative_change_pct):.2f}%，两者均未观测到过期，成功包P95的差值区间跨0。因此主负载Boundary的权重优势不能推广为所有用户布点下都占优。')
    w.h('权重消融与负载敏感性')
    result_table(w,root,'表 9-7 同一精确内层的权重消融（Boundary08，10seed）','boundary',.8,
      ['equal','deficit','ttl1','ttl2','exp_pf','mlwdf_pf'],
      ['权重','整包Mbps','过期%','P95 ms','最差UE过期%','Jain'],
      ['payload_mbps','expiry_pct','delay_p95_ms','worst_ue_expiry_pct','jain_payload'])
    w.p('缺口权重为clip[2400/max(EMA,10⁻⁹),1,4]后均值归一化，参考服务量2400 B/TTI在全部负载下固定，不随到达率调参。κ-TTL在此基础上乘[1+κ(1−slack/400)]，本轮固定κ=1、2。等权取w=1。M-LWDF-PF和EXP-PF使用未截断的PF分母，故文献权重与κ族差异不只来自紧迫度曲线；本消融衡量完整权重形式的闭环效果，不把所有收益归因于单一函数形状。')
    alt=['equal','deficit','ttl1','ttl2','exp_pf'];av=complete_means(root,'boundary',.8,alt+['mlwdf_pf'])
    if av and all(av['mlwdf_pf','payload_mbps']>av[a,'payload_mbps'] and av['mlwdf_pf','delay_p95_ms']<av[a,'delay_p95_ms'] for a in alt) and all(av[a,'expiry_pct']==0 for a in ['ttl1','ttl2','exp_pf','mlwdf_pf']):
        w.p('在Boundary/rate0.8所测六种权重中，M-LWDF-PF的整包吞吐最高、成功包P95最低。κ=1、κ=2及EXP-PF同样未观测到过期包，因此仅比较是否零过期不足以区分这些方案；公平性指标及高负载结果仍须分别考察。')
    apath=root/'analysis/ablation_paired.csv'
    if apath.exists() and apath.stat().st_size>1:
        ad=pd.read_csv(apath)
        for baseline in ['equal','ttl1','ttl2','exp_pf']:
            z=ad[(ad.scene=='boundary')&(ad.rate==.8)&(ad.baseline==baseline)&(ad.subset=='41-50')&(ad.n==10)].set_index('metric')
            if {'payload_mbps','delay_p95_ms'}<=set(z.index):
                a=z.loc['payload_mbps'];b=z.loc['delay_p95_ms']
                w.p(f'DA-PLS相对{LABEL[baseline]}：整包吞吐差值{a.delta:+.3f} Mbps（95%配对区间[{a.ci_low:+.3f}, {a.ci_high:+.3f}]）；成功包P95差值{b.delta:+.3f} ms（95%配对区间[{b.ci_low:+.3f}, {b.ci_high:+.3f}]）。完整逐seed差值、过期及弱UE指标见随附权重配对表；本组比较未作多重比较校正。')
    for i,rate in enumerate([1.12,1.28]):
        result_table(w,root,f'表 9-{8+i} Boundary高负载×{rate/.8:g}（10seed）','boundary',rate,arms,
          ['算法','整包Mbps','过期%','P95 ms','最差UE过期%'],['payload_mbps','expiry_pct','delay_p95_ms','worst_ue_expiry_pct'])
        v=complete_means(root,'boundary',rate,['deficit','mlwdf_pf'])
        if v:
            w.p(f'负载×{rate/.8:g}时，缺口权重与M-LWDF-PF的完整包吞吐分别为{v["deficit","payload_mbps"]:.3f}、{v["mlwdf_pf","payload_mbps"]:.3f} Mbps；过期率分别为{v["deficit","expiry_pct"]:.3f}%、{v["mlwdf_pf","expiry_pct"]:.3f}%；最差UE过期率分别为{v["deficit","worst_ue_expiry_pct"]:.3f}%、{v["mlwdf_pf","worst_ue_expiry_pct"]:.3f}%。因此该负载下的评价须按交付、时延和弱UE风险分别判断，不能沿用主负载的排序。')
            weight_path=root/'analysis/weight_paired.csv'
            if weight_path.exists():
                weight_data=pd.read_csv(weight_path)
                q=weight_data[(weight_data.scene=='boundary')&(weight_data.rate==rate)&(weight_data.subset=='41-50')&(weight_data.n==10)].set_index('metric')
                if {'payload_mbps','expiry_pct','worst_ue_expiry_pct'}<=set(q.index):
                    parts=[]
                    for metric,name,unit in [('payload_mbps','整包吞吐','Mbps'),('expiry_pct','过期率','个百分点'),('worst_ue_expiry_pct','最差UE过期率','个百分点')]:
                        r=q.loc[metric];parts.append(f'{name}{r.delta:+.3f} {unit}（95%配对区间[{r.ci_low:+.3f}, {r.ci_high:+.3f}]）')
                    w.p('该负载下DA-PLS减缺口权重的同seed差值为：'+'；'.join(parts)+'。区间未作多重比较校正。')
    w.p('负载敏感性用于限定适用范围。主负载下获得更低时延的权重，不必在过载下同时最大化整包吞吐并最小化最差UE风险；当缺口权重更好时如实保留该结果。本轮各条轨迹负载固定，没有负载切换实验，因此不能推断动态切换已经有效。')
    w.h('精确等价性与计算成本')
    completion=root/'audit/completion.json'
    if completion.exists():
        c=json.loads(completion.read_text())
        w.p(f'本轮已核验{c["states"]}个M-LWDF权重保存状态、{c["audited_rounds"]}轮和{c["audited_candidates"]}个候选。逐轮检查根、候选上界、精确缓存增益、出队差、规范选择和终态；新增精评数为{c["new_evaluations"]}，该证书的必评集合计数为{c["required_evaluations"]}。逐轮证书计数不含初始化根评价，成本表的工作量计数包含该项。只有全部状态通过才作为最终等价性证据。')
        w.p('本版保存状态计时在全部220个质量实验结束后重新执行，计时期间本批质量仿真已停止；早期并发期间的计时另行归档。' if not c.get('concurrent_quality_campaign',True) else '本版保存状态的三种搜索在同一CPU核串行计时，但本批其它质量轨迹仍在并发运行，存在主机争用；完整版本将在全部质量实验结束后统一复测。')
        tpath=root/'audit/timing_summary.csv'
        if tpath.exists():
            t=pd.read_csv(tpath);rows=[]
            for rate,g in t.groupby('rate'):
                g=g.set_index('arm')
                if set(g.index)=={'python_full','native_full','bounded'}:
                    b=g.loc['bounded'];f=g.loc['native_full'];p=g.loc['python_full']
                    rows.append([f'×{rate/.8:g}',f'{p.wall_ms:.3f}',f'{f.wall_ms:.3f}',f'{b.wall_ms:.3f}',f'{100*b.wall_ms/f.wall_ms:.2f}%',f'{100*b.candidates/f.candidates:.2f}%',f'{100*b.mmse_misses/f.mmse_misses:.2f}%'])
            if rows:w.table('表 9-10 同输入串行成本（ms；相对同架构native完整搜索）',['负载','Python完整','native完整','精确剪枝','墙钟占比','候选占比','MMSE占比'],rows)
    else:w.p('M-LWDF权重的保存状态正在随正式轨迹生成；最终计时表只使用本轮同权重输入，不借用旧缺口权重的34%–46%成本曲线。')
    w.p('串行计时固定CPU核，交替轮转执行顺序；预先加载不变表和库，计时包含保存输入的重构、权重生成及一致性检查、PFQ动作解码、物理上下文创建、搜索和销毁，关闭审计观察器。它不包含原始观测复制、HOL的FIFO扫描、原生PFQ初选计算、仿真推进或跨进程传输，因此不构成DA-PLS相对PFQ的完整调度时延比。Python完整枚举也使用本轮规范目标，避免把历史不同实现的181.7 ms与新结果直接拼接。')
    w.p('每条轨迹取测量窗开始、中间和末尾三个状态，每状态轮换计时三次；先平均每个seed的重复和状态，再等权平均seed。该固定时间点样本用于相同输入上的算法比较，不代表完整连续轨迹的成本分布；连续P95和最大值另列。')
    w.p('逐TTI连续日志保留墙钟和进程CPU时间的均值、P95与最大值，范围含观测复制、权重与HOL整理、搜索，以及三个保存点的研究归档I/O。质量实验并发运行，墙钟还含主机争用，不能用它计算公平加速比；离群值及保存点不删除。若搜索均值或尾部高于0.5 ms，仅能给出研究算法的复杂度结论，不能称满足实时调度截止时间。')
    path=root/'analysis/continuous_cost.csv'
    if path.exists() and path.stat().st_size>1:
        cc=pd.read_csv(path);rows=[]
        names={'wall_ms':'墙钟/ms','cpu_ms':'进程CPU/ms','candidate_evaluations':'候选精评数','ue_evaluations':'UE评价数','mmse_misses':'MMSE miss'}
        for rate in [.8,1.12,1.28]:
            group=cc[(cc.scene=='boundary')&(cc.rate==rate)&(cc.arm=='mlwdf_pf')&(cc.n==10)].set_index('metric')
            for metric,name in names.items():
                if metric in group.index:
                    r=group.loc[metric];rows.append([f'×{rate/.8:g}',name,f'{r.seed_mean:.3f}',f'{r.seed_p95_mean:.3f}',f'{r.observed_max:.3f}'])
        if rows:w.table('表 9-11 DA-PLS连续测量窗口成本（并发运行，非公平加速比分母）',['负载','指标','逐seed均值的均值','逐seed P95的均值','全部观测最大值'],rows)
    w.h('物理机制与逐UE服务分布')
    result_table(w,root,'表 9-12 Boundary08物理状态与资源效率（10seed）','boundary',.8,['pfq','mlwdf_pf'],
      ['算法','SINR dB','MCS','占用%','成功B/PRG','其它未交付%'],
      ['sinr_tx_mean_db','mcs_tx_mean','prg_occupancy_pct','bytes_per_used_prg','other_nondelivery_pct'])
    result_table(w,root,'表 9-13 同PRG上活动小区数的分布（%）','boundary',.8,['pfq','mlwdf_pf'],
      ['算法','零小区','单小区','双小区','三小区'],['reuse_0_pct','reuse_1_pct','reuse_2_pct','reuse_3_pct'])
    for name,caption in [('main_results','图 9-1 两场景完整包吞吐与时延对比'),('weight_ablation','图 9-2 同一内层的权重消融'),
      ('load_sensitivity','图 9-3 高负载适用边界'),('phy_distributions','图 9-4 实际发送SINR与MCS分布'),
      ('reuse_distribution','图 9-5 单／双／三小区复用与资源空置'),('ue_expiry','图 9-6 逐UE过期率分布'),
      ('joint_move_case','图 9-7 真实保存状态中的静默与邻区换UE联合移动')]:
        if (w.fig/f'{name}.png').exists():w.image(name,caption)
    case=select_case(root)
    if case:
        w.p('案例按文件顺序优先选取首个同时含正、负UE收益的已核验联合移动，用于展示本地代价与邻区收益的权衡；它不用于估计此类移动的发生频率或平均收益。')
        c=case['case'];w.p(f'真实联合移动案例取自{Path(case["path"]).parent.parent.name}，TTI {c["tti"]}、第{c["iteration"]}轮。编辑为'+ '；'.join(f'小区{x["cell"]} PRG{x["prg"]}: UE{x["old_ue"]}→'+('静默' if x['new_ue']<0 else f'UE{x["new_ue"]}') for x in c['edits'])+f'。整数加权净增益为{c["gain_q"]/1e6:.6f} byte，出队{c["dequeue_before"]}→{c["dequeue_after"]} byte，下界{c["floor"]} byte。')
        w.table('表 9-14 真实联合移动的完整TB受益与代价',['UE','权重','原预期B','新预期B','加权变化B'],[
          [x['ue'],f'{x["weight"]:.4f}',f'{x["expected_before"]:.3f}',f'{x["expected_after"]:.3f}',f'{x["weighted_delta"]:+.3f}'] for x in c['ues']])
    w.p('这些证据互相补充：物理分布变化说明所选传输的接收状态，复用分布说明资源活动模式，联合编辑案例解释一次决策的净收益，逐UE分布用于识别收益是否以少数弱UE恶化为代价。发送样本的SINR/MCS分布也受用户选择影响，因此不把总体分布变化单独解释为所有链路都改善的因果证明。')

def chapter10(w,root):
    w.h('结论与后续工作',1)
    w.p('二阶段同频算法收敛为DA-PLS：M-LWDF-PF时延感知权重、分源MMSE-IRC与完整TB评价、换UE／静默／联合编辑邻域，以及容量上界驱动的精确剪枝。该路线保留原生队列与链路反馈，直接优化当前加权预期成功服务，不含异构图网络、PPO或其它学习组件。')
    path=root/'analysis/aggregate.csv'
    if path.exists():
        z=pd.read_csv(path)
        for scene,rate in [('boundary',.8),('uniform',1.)]:
            a=z[(z.scene==scene)&(z.rate==rate)&(z.subset=='41-50')&(z.n==10)]
            if set(a.arm)>={'pfq','mlwdf_pf'}:
                def v(arm,metric):return float(a[(a.arm==arm)&(a.metric==metric)]['mean'].iloc[0])
                w.p(f'{scene}/rate{rate:g}十seed配对中，完整包吞吐由{v("pfq","payload_mbps"):.3f}变为{v("mlwdf_pf","payload_mbps"):.3f} Mbps；成功包均时由{v("pfq","delay_mean_ms"):.3f}变为{v("mlwdf_pf","delay_mean_ms"):.3f} ms；过期率由{v("pfq","expiry_pct"):.3f}%变为{v("mlwdf_pf","expiry_pct"):.3f}%。差值区间、44–50子集及逐UE限制见第9章。')
    heavy=[]
    for rate in [1.12,1.28]:
        v=complete_means(root,'boundary',rate,['deficit','mlwdf_pf'])
        if v:
            delta=100*(v['mlwdf_pf','payload_mbps']/v['deficit','payload_mbps']-1)
            expiry=v['mlwdf_pf','expiry_pct']-v['deficit','expiry_pct']
            heavy.append(f'×{rate/.8:g}负载下整包吞吐相差{delta:+.2f}%，过期率相差{expiry:+.3f}个百分点')
    if len(heavy)==2:
        w.p('高负载检验中，M-LWDF-PF相对缺口权重：'+'；'.join(heavy)+'。这限定了主负载权重收益的外推范围；本轮未执行动态负载切换，不能据此宣称切换策略已经验证。')
    w.p('本报告将仿真有效性与调度效果分开验收。队列bit/byte扣减和阴影标准差单位修正后，旧结果全部退出正式证据链；新结果以冻结版本、共同前缀、成熟到达cohort、完整包CRC以及逐seed统计为依据。上界可采纳与等价性证明支撑的是搜索质量保持，实际开销由同输入串行计时支撑。')
    w.p('可交付内容包括：DA-PLS方法与精确内层实现、统一场景协议、两项缺陷的修正说明及验证、正式原始日志、逐seed/逐UE统计、配对区间、等价性与成本核验、修订技术报告。算法研究就绪与生产实时部署就绪分别判断；本报告不将“无需训练”写成没有场景迁移风险。')
    w.p('后续工作集中在四点：非理想CSI与跨小区测量开销；满足0.5 ms截止时间的工程实现；过载时业务目标和权重切换的独立验证；异频／载波聚合中的激活、测量和功率开销建模。后两项均不在本阶段结果中预先宣称完成。')

def finalize(doc,root,draft):
    for section in doc.sections:
        section.page_width=Cm(21);section.page_height=Cm(29.7)
        section.top_margin=Cm(2.1);section.bottom_margin=Cm(2.1);section.left_margin=Cm(2.3);section.right_margin=Cm(2.3)
        section.header_distance=Cm(.8);section.footer_distance=Cm(.8)
        hp=section.header.paragraphs[0];hp.text='多维协同资源分配研究 · 二阶段同频技术报告';hp.alignment=WD_ALIGN_PARAGRAPH.RIGHT
        for r in hp.runs:r.font.size=Pt(8);r.font.color.rgb=RGBColor.from_string('697782')
        fp=section.footer.paragraphs[0];fp.text='';fp.alignment=WD_ALIGN_PARAGRAPH.CENTER
        fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');fp._p.append(fld)
    for name in ['Normal','Body Text']:
        st=doc.styles[name];st.font.name='Noto Serif CJK SC';st.font.size=Pt(10.5)
        st._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Noto Serif CJK SC')
        st.paragraph_format.line_spacing=1.35;st.paragraph_format.space_after=Pt(5)
    for level in range(1,4):
        st=doc.styles[f'Heading {level}'];st.font.name='Noto Sans CJK SC';st.font.size=Pt([18,14,11.5][level-1]);st.font.bold=True
        st._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Noto Sans CJK SC')
        pp=st._element.get_or_add_pPr()
        for child in list(pp):
            if child.tag==qn('w:numPr'):pp.remove(child)
        st.paragraph_format.space_before=Pt(14);st.paragraph_format.space_after=Pt(7);st.paragraph_format.keep_with_next=True
    chapter=section=sub=0
    for p in doc.paragraphs:
        if p.style and p.style.name.startswith('Heading'):
            level=int(p.style.name[-1]);text=p.text.strip()
            props=p._p.get_or_add_pPr()
            for ch in list(props):
                if ch.tag==qn('w:numPr'):props.remove(ch)
            if text in ['目录','参考文献']:continue
            if level==1:chapter+=1;section=sub=0;p.text=f'{chapter}  {text}';p.paragraph_format.page_break_before=True
            elif level==2:section+=1;sub=0;p.text=f'{chapter}.{section}  {text}'
            else:sub+=1;p.text=f'{chapter}.{section}.{sub}  {text}'
        # Remove old explicit fonts and numbering artifacts, while retaining bold/italics and equations.
        if p.style and not p.style.name.startswith(('Title','Subtitle')):
            for r in p.runs:
                r.font.name='Noto Serif CJK SC';r._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Noto Serif CJK SC')
                if not p.style.name.startswith('Heading'):r.font.size=Pt(10.5)
    for node in list(doc._element.iter()):
        if node.tag in (qn('w:commentRangeStart'),qn('w:commentRangeEnd'),qn('w:commentReference')):
            node.getparent().remove(node)
    update=OxmlElement('w:updateFields');update.set(qn('w:val'),'true');doc.settings.element.append(update)
    doc.core_properties.title='基于AI技术的多维协同资源分配研究技术研究报告（二阶段同频收尾）'
    doc.core_properties.subject='DA-PLS；M-LWDF-PF权重；精确上界剪枝；修正后仿真'
    doc.core_properties.comments='修订原报告；修正前结果不沿用；采用M-LWDF-PF与精确剪枝物理搜索。'
    if draft:
        p=doc.paragraphs[5];p.add_run('\n修订工作稿：正式仿真与结果核验进行中').font.color.rgb=RGBColor.from_string('A05A15')

def main(args):
    root=Path(args.root).resolve();fig=root/'figures';diagrams(fig)
    doc=Document(SOURCE);original=list(doc._element.body);refs=[deepcopy(e) for e in original[899:933]]
    for style in Document().styles:
        if style.name not in doc.styles:doc.styles.element.append(deepcopy(style.element))
    w=foundation(doc,original,fig);chapter4(w);chapter5(w,root);chapter6(w);chapter7(w,root);chapter8(w);chapter9(w,root);chapter10(w,root)
    w.h('参考文献',1).paragraph_format.page_break_before=True
    for i,e in enumerate(refs,1):
        p=Paragraph(e,doc);text=p.text.strip();w.p(f'[{i}] {text}')
    extra=[
      'E. M. Ang, K. Wee, Y. H. Pang, and K. K. Phang, “A performance analysis on packet scheduling schemes based on an exponential rule for real-time traffic in LTE,” EURASIP Journal on Wireless Communications and Networking, 2015:201. https://doi.org/10.1186/s13638-015-0429-8',
      'A. H. Land and A. G. Doig, “An Automatic Method of Solving Discrete Programming Problems,” Econometrica, 28(3), 497–520, 1960. https://doi.org/10.2307/1910129',
      'NVIDIA Aerial repository, Issue #67: TBS bits used as bytes when updating UE buffers. https://github.com/NVIDIA/aerial-cuda-accelerated-ran/issues/67 （核查日期：2026-11-04）',
      'NVIDIA Aerial repository, Issue #71: Shadow fading standard deviation in dB passed directly to std::lognormal_distribution. https://github.com/NVIDIA/aerial-cuda-accelerated-ran/issues/71 （核查日期：2026-11-04）',
      '3GPP, TR 36.829 V11.1.0 (2012-12), Enhanced performance requirement for LTE User Equipment (UE), Release 11, §4.3 MMSE IRC receiver. https://www.3gpp.org/ftp/Specs/archive/36_series/36.829/ （接收机背景；本文activity-only假设以冻结代码口径为准）']
    for i,s in enumerate(extra,35):w.p(f'[{i}] {s}')
    if args.core:
        per=pd.read_csv(root/'analysis/per_seed.csv')
        for scene,rate in [('boundary',.8),('uniform',1.)]:
            for arm in ['pfq','deficit','mlwdf_pf']:
                seeds=per[(per.scene==scene)&(per.rate==rate)&(per.arm==arm)].seed.tolist()
                assert sorted(seeds)==list(range(41,51)), (scene,arm,seeds)
        proofs=[json.loads(p.read_text()) for p in (root/'audit').glob('boundary_r0.8_mlwdf_pf_s*_tti*.json')]
        assert len(proofs)==30 and all(p['passed'] for p in proofs)
        checks=pd.read_csv(root/'analysis/prefix_validation.csv')
        assert len(checks[(checks.rate<=1)&checks.arm.isin(['pfq','deficit','mlwdf_pf'])])==60
    elif not args.draft:
        for directory in ['analysis','audit']:
            assert json.loads((root/directory/'completion.json').read_text())['completed']
        assert json.loads((root/'analysis/completion.json').read_text())['points']==220
    finalize(doc,root,args.draft)
    if args.core:
        p=doc.paragraphs[5];p.add_run('\n核心对比版：主对比已核验；传统基线扩展、权重消融及高负载结果续补').font.color.rgb=RGBColor.from_string('A05A15')
    target=root/('技术研究报告_二阶段同频收尾_核心对比版.docx' if args.core else '技术研究报告_二阶段同频收尾_修订工作稿.docx' if args.draft else '技术研究报告_二阶段同频收尾_DA-PLS.docx')
    doc.save(target)
    text='\n'.join(p.text for p in doc.paragraphs)+'\n'+'\n'.join(c.text for t in doc.tables for row in t.rows for c in row.cells)
    (root/'report_text_audit.txt').write_text(text)
    print(target)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--draft',action='store_true');p.add_argument('--core',action='store_true');main(p.parse_args())

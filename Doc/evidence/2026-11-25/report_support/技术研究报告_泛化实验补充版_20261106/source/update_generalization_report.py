"""Add evidence-backed generalization results to the completed figure revision."""
import json,sys,hashlib,csv
from pathlib import Path
from copy import deepcopy
from zipfile import ZipFile
sys.path.insert(0,'/tmp/stage2_report_python')
import numpy as np
import pandas as pd
from lxml import etree
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches,Pt,Twips
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
SRC=ROOT/'outputs/技术研究报告_图表精简版_20261106/技术研究报告_二阶段同频_图表精简版.docx'
EXP=ROOT/'deliverables/stage2_generalization_20261106'
AN=EXP/'analysis'
DEST=OUT/'技术研究报告_二阶段同频_泛化实验补充版.docx'
ARMS=['pfq','deficit','mlwdf_pf'];LABELS=dict(pfq='PFQ',deficit='DPLS',mlwdf_pf='DA-PLS')
SUBGROUPS=[
 ('G1','placement_group','boundary','G1 边缘用户'),('G1','placement_group','uniform','G1 随机用户'),
 ('G2','traffic_group','small_high','G2 小包高频'),('G2','traffic_group','large_low','G2 大包低频'),
 ('G3','cell_load','heavy_cell','G3 重载小区'),('G3','cell_load','light_cells','G3 轻载小区'),
 ('G4','placement_group','boundary','G4 边缘用户'),('G4','placement_group','uniform','G4 随机用户'),
 ('G4','traffic_group','small_high','G4 小包高频'),('G4','traffic_group','large_low','G4 大包低频'),
 ('G4','cell_load','heavy_cell','G4 重载小区'),('G4','cell_load','light_cells','G4 轻载小区'),
 ('G4','heavy_cell_traffic','large_low','G4 重载区大包'),
]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(p,text):
    # Keep paragraph properties and first-run typography; no crossrefs are removed.
    old=next((r for r in p.runs if r.text),None)
    rp=deepcopy(old._r.rPr) if old is not None and old._r.rPr is not None else None
    for child in list(p._p):
        if child.tag!=qn('w:pPr'):p._p.remove(child)
    r=p.add_run(text)
    if rp is not None:r._r.insert(0,rp)
    return p

def no_grid(p):
    pp=p._p.get_or_add_pPr();snap=pp.find(qn('w:snapToGrid'))
    if snap is None:snap=OxmlElement('w:snapToGrid');pp.append(snap)
    snap.set(qn('w:val'),'false')

def insert(anchor,text,template):
    el=deepcopy(template._p)
    for c in list(el):
        if c.tag!=qn('w:pPr'):el.remove(c)
    anchor._p.addprevious(el)
    from docx.text.paragraph import Paragraph
    p=Paragraph(el,anchor._parent);r=p.add_run(text)
    old=next((r for r in template.runs if r.text),None)
    if old is not None and old._r.rPr is not None:r._r.insert(0,deepcopy(old._r.rPr))
    if template.style.name=='Normal' and anchor.text=='本章小结':
        p.paragraph_format.line_spacing=Pt(23.4);no_grid(p)
    return p

def insert_after(anchor,text,template):
    p=insert(anchor,text,template);anchor._p.addnext(p._p);return p

def write_csv(name,rows):
    p=OUT/'data'/name
    with p.open('w',newline='',encoding='utf-8-sig') as f:csv.writer(f).writerows(rows)

def table(doc,anchor,headers,rows,widths,template,bold_col=None,bold_rows=None):
    t=doc.add_table(rows=1,cols=len(headers));t.style=template.style;t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    tp=t._tbl.tblPr
    for key in ['tblW','tblInd','tblCellMar']:
        old=tp.find(qn('w:'+key))
        if old is not None:tp.remove(old)
        src=template._tbl.tblPr.find(qn('w:'+key))
        if src is not None:tp.append(deepcopy(src))
    # Use the same overall width and grid-border style as existing tables.
    grid=t._tbl.tblGrid
    for el in list(grid):grid.remove(el)
    for w in widths:
        el=OxmlElement('w:gridCol');el.set(qn('w:w'),str(w));grid.append(el)
    for _ in rows:t.add_row()
    for i,values in enumerate([headers]+rows):
        row=t.rows[i];trp=row._tr.get_or_add_trPr()
        el=OxmlElement('w:cantSplit');trp.append(el)
        if i==0:
            el=OxmlElement('w:tblHeader');trp.append(el)
        for j,value in enumerate(values):
            c=row.cells[j];c.width=Twips(widths[j]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            c.text=str(value);p=c.paragraphs[0];p.style='Normal';p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            pf=p.paragraph_format;pf.first_line_indent=Pt(0);pf.left_indent=Pt(0);pf.right_indent=Pt(0)
            pf.space_before=Pt(0);pf.space_after=Pt(0);pf.line_spacing=1.15;pf.keep_with_next=True
            no_grid(p)
            for r in p.runs:
                r.font.name='Times New Roman';r.font.size=Pt(9.5);r.bold=(i==0 or j==bold_col or (bold_rows and i-1 in bold_rows))
                r._r.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'宋体')
    anchor._p.addprevious(t._tbl)
    return t

def note(anchor,text,template):
    p=insert(anchor,text,template);pf=p.paragraph_format;pf.keep_with_next=False;pf.first_line_indent=Pt(0);pf.line_spacing=1.15;pf.space_before=Pt(3);pf.space_after=Pt(6)
    no_grid(p)
    for r in p.runs:r.font.size=Pt(9.5)
    return p

def main():
    val=json.loads((AN/'validation.json').read_text());assert val['passed'] and val['complete'] and val['runs']==60
    protocol=json.loads((EXP/'protocol.json').read_text());assert protocol['seeds']==[51,52,53,54,55]
    df=pd.read_csv(AN/'per_seed.csv');assert len(df)==60
    means=df.groupby(['scenario','arm']).mean(numeric_only=True)
    groups=pd.read_csv(AN/'groups_mean.csv').set_index(['scenario','arm','dimension','group'])
    words=json.loads((OUT/'source/result_narrative.json').read_text())
    doc=Document(SRC);old_tables=[[[c.text for c in r.cells] for r in t.rows] for t in doc.tables]
    p=doc.paragraphs
    body=p[528];heading=p[578];caption=p[577];tabtemplate=doc.tables[27]
    # Clarify which original settings belong only to the original two environments.
    plain(p[17],p[17].text.replace('相对前序物理搜索DPLS，DA-PLS在相同搜索方法上取得更好的业务表现。','在Boundary08主负载中，DA-PLS相对前序物理搜索DPLS在相同搜索方法上取得更好的业务表现。'))
    plain(p[595],p[595].text.replace('最终计时在220个质量实验全部结束后执行','原主评测的计时在220个质量实验全部结束后执行'))
    plain(p[427],p[427].text.replace('36个UE各承载一条同参数下行业务流','Boundary08与Uniform主场景中，36个UE各承载一条同参数下行业务流')+' 第9.6节在相同物理条件下扩展混合布点、异构包长／到达频率及小区负载不平衡。')
    plain(p[435],p[435].text.replace('第10章报告','第9.6节另以四类复杂场景检验泛化；第10章报告'))
    plain(p[440],p[440].text.replace('第9章的协同算法沿用本章全部设置与指标','第9.2–9.5节的协同算法沿用本章全部设置与指标；第9.6节保留时序和统计口径，扩展空间与业务配置'))
    plain(p[446],p[446].text.replace('所有指标先在每个seed内计算，再对seed 41–50等权平均','本章及第9.2–9.5节的指标先在每个seed内计算，再对seed 41–50等权平均'))
    plain(p[527],p[527].text.replace('本章在第7章相同的两个环境中比较','本章先在第7章相同的两个环境中比较'))
    plain(p[528],p[528].text+' 第9.6节另采用seed 51–55，对四类复杂场景及三种算法运行60次正式仿真，不与原10个seed混合平均。')
    plain(p[531],p[531].text.replace('是同频干扰最强、对协同要求最高的场景','用于考察用户集中于边界时的同频干扰与协同需求'))
    plain(p[551],p[551].text.replace('本报告所有方法中最高的链路质量','Boundary08对比中最高的链路质量'))
    plain(p[579],p[579].text.replace('在干扰最强的Boundary08中','在Boundary08中'))
    # Keep the original summary heading; insertion makes it section 9.7 automatically.
    anchor=p[578]
    new_heading=insert(anchor,'复杂业务与空间分布下的泛化评测',heading)
    new_heading.paragraph_format.page_break_before=True
    anchor.paragraph_format.page_break_before=True
    insert(anchor,'本节检验混合空间分布、异构业务与小区负载不平衡下的适应性。每个名义小区生成6个Uniform随机用户、6个Boundary边缘用户，三小区共36个UE；按最强信号关联并保持固定。“随机”指生成方式，不等同于小区中心，四类配置见表9-5。',body)
    insert(anchor,'表 9-5 复杂同频泛化场景配置',caption)
    scenarios=[
        ['G1 混合布点','边缘18＋随机18','全部3000 B；0.8包/TTI','各UE等平均字节输入'],
        ['G2 异构业务','同G1','1500 B×18；12000 B×18','同G1'],
        ['G3 小区不平衡','同G1','全部3000 B；频率按小区分配','实际服务小区输入2∶1∶1'],
        ['G4 叠加复杂性','同G1','1500 B×18；12000 B×18','实际服务小区输入2∶1∶1'],
    ]
    headers=['场景','用户布点','包长与到达','负载配置']
    table(doc,anchor,headers,scenarios,[1550,1500,2856,2300],tabtemplate)
    note(anchor,'注：四场景理论簇输入均为1382.40 Mbps，TTL均为200 ms。G1/G2的每UE平均输入相同，实际关联人数不同会使小区总输入不同；G3/G4才显式控制小区输入比。',body)
    write_csv('表9-5_场景配置.csv',[headers]+scenarios)
    insert(anchor,'G2的小包／大包到达率为1.6／0.2包/TTI（3200／400包/s），每UE平均输入均为38.4 Mbps，包型在两种布点组内各占一半。G3/G4按实际关联向重载小区分配691.20 Mbps，另两区各345.60 Mbps，各区内部按UE等分字节输入；G4再按包长反向调整到达频率。重载小区依seed轮换为0、1、2、0、1。',body)
    insert(anchor,'每场景比较PFQ、DPLS和DA-PLS，使用新seed 51–55，共60次。单次依次为4000 TTI共同PFQ预热、4000 TTI稳定期、20000 TTI测量及400 TTI延长观察，测量10 s。同场景同seed共用位置、逐包到达与预热状态；指标先在seed内计算，再对5个seed等权平均。',body)
    insert(anchor,'物理模型、90%计划出队下界和精确搜索不变。DPLS仅把缺口分子由统一的2400 B/TTI改为逐UE配置的平均字节到达量，保留[1,4]裁剪及均值归一化；DA-PLS权重不变，均不按场景调参。关闭扩展时已通过原场景512 TTI一致性核验。',body)
    new_caption=insert(anchor,'表 9-6 复杂同频场景的核心结果（seed 51–55均值）',caption)
    new_caption.paragraph_format.page_break_before=True
    headers=['场景','算法','完整包吞吐\n（Mbps）','过期率\n（%）','字节过期率\n（%）','成功包\n平均时延\n（ms）','最差UE\n过期率\n（%）']
    rows=[]
    for s in ['G1','G2','G3','G4']:
        for a in ARMS:
            r=means.loc[(s,a)]
            rows.append([s,LABELS[a]]+[f'{r[m]:.3f}' for m in ['payload_mbps','expiry_pct','expiry_byte_pct','delay_mean_ms','worst_ue_expiry_pct']])
    table(doc,anchor,headers,rows,[620,1050,1450,1120,1240,1410,1166],tabtemplate,bold_rows={2,5,8,11})
    note(anchor,'注：粗体标识DA-PLS。过期率按包数计算；字节过期率为过期包完整载荷占输入载荷的比例，包含已部分传输的过期包。最差UE先在seed内取最大值。吞吐与成功包时延仅计全部分段CRC成功且未过期的包；本表DPLS和DA-PLS的0.000均对应未观测到过期。',body)
    write_csv('表9-6_核心结果.csv',[headers]+rows)
    for text in words['core_paragraphs']:insert(anchor,text,body)
    new_caption=insert(anchor,'表 9-7 重点用户分组的业务质量（seed 51–55均值）',caption)
    new_caption.paragraph_format.page_break_before=True
    headers=['场景与分组','PFQ\n过期率（%）','DA-PLS\n过期率（%）','DA-PLS成功包\n均时延（ms）','DA-PLS完整\n成功率（%）']
    rows=[]
    for s,dim,g,label in SUBGROUPS:
        da=groups.loc[(s,'mlwdf_pf',dim,g)]
        rows.append([label,f"{groups.loc[(s,'pfq',dim,g),'expiry_pct']:.3f}",f"{da.expiry_pct:.3f}",f"{da.delay_mean_ms:.3f}",f"{da.success_pct:.3f}"])
    table(doc,anchor,headers,rows,[2206,1450,1450,1550,1650],tabtemplate,bold_col=2)
    note(anchor,'注：三算法整体比较见表9-6。分组过期率与完整成功率以本组到达包为分母，均时延仅含成功包；均为5个seed等权均值。重载／轻载按实际服务小区划分；“重载区大包”为两类条件的交集。各行可重叠，不能相加。完整成功率还受CRC失败影响。',body)
    write_csv('表9-7_重点用户分组.csv',[headers]+rows)
    for text in words['group_paragraphs']:insert(anchor,text,body)
    note(anchor,'核验：'+words['validation_paragraph'],body)
    note(anchor,'范围：结论限于这四类配置、5个新seed、固定总输入与200 ms截止时间。用户静止且关联固定；移动、不同TTL、相关突发到达及更强过载仍需单独评测。本次并发运行只统计业务质量，计算成本沿用第10章串行计时。',body)
    # Propagate the actual result, including limitations, to the chapter summary and conclusion.
    insert_after(p[579],words['summary_paragraph'],body)
    insert_after(p[601],'复杂场景泛化。'+words['conclusion_paragraph'],body)
    insert_after(p[17],words['abstract_paragraph'],body)
    plain(p[603],p[603].text.replace('全部实验以统一的代码版本','原主评测与本次场景扩展均冻结代码和配置，以共同的物理与调度实现'))
    # Native Word formulas and all old table values must remain unchanged.
    new_tables=[[[c.text for c in r.cells] for r in t.rows] for t in doc.tables]
    assert len(new_tables)==len(old_tables)+3
    assert new_tables[:28]==old_tables[:28] and new_tables[31:]==old_tables[28:]
    doc.save(DEST)
    ns={'m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
    oldxml=etree.fromstring(ZipFile(SRC).read('word/document.xml'));newxml=etree.fromstring(ZipFile(DEST).read('word/document.xml'))
    before=oldxml.xpath('//m:oMath',namespaces=ns);after=newxml.xpath('//m:oMath',namespaces=ns)
    assert len(before)==len(after)==72
    for a,b in zip(before,after):assert etree.tostring(a,method='c14n')==etree.tostring(b,method='c14n')
    manifest=dict(source=str(SRC),source_sha256=sha(SRC),output=str(DEST),source_tables_preserved=len(old_tables),added_tables=['9-5','9-6','9-7'],native_math_preserved=72,experiment_validation=val,protocol_sha256=sha(EXP/'protocol.json'),per_seed_sha256=sha(AN/'per_seed.csv'),narrative_sha256=sha(OUT/'source/result_narrative.json'))
    (OUT/'核验/文档生成核验.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(DEST)
if __name__=='__main__':main()

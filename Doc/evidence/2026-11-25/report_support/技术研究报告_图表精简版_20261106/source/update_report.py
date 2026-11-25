"""Revise only the supplied report, preserving unrelated content and formulas."""
from pathlib import Path
from copy import deepcopy
from lxml import etree
import sys,re,json,hashlib
sys.path.insert(0,'/tmp/stage2_report_python')
from docx import Document
from docx.shared import Inches,Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
BASE=Path(__file__).resolve().parents[1]
SOURCE=Path('/home/oai-mac/.codex/attachments/2f3e26fc-a491-4e56-a9e0-fbf5e3fbc2c6/技术研究报告_二阶段同频.docx')
OUT=BASE/'技术研究报告_二阶段同频_图表精简版.docx'
doc=Document(SOURCE);ps=list(doc.paragraphs)
assert len(ps)==656 and ps[559].text.startswith('积压并没有')
original_tables=[[ [c.text for c in r.cells] for r in t.rows] for t in doc.tables]

def settext(p,text):
    rp=deepcopy(p.runs[0]._r.rPr) if p.runs and p.runs[0]._r.rPr is not None else None
    for child in list(p._p):
        if child.tag not in [qn('w:pPr'),qn('w:bookmarkStart'),qn('w:bookmarkEnd')]:p._p.remove(child)
    r=p.add_run(text)
    if rp is not None:r._r.insert(0,rp)

def append_text(p,text):
    r=p.add_run(text)
    r.font.name='Times New Roman';r.font.size=Pt(12);r._r.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体')

def new_after(p,style='Normal',text=None):
    el=OxmlElement('w:p');p._p.addnext(el);new=Paragraph(el,p._parent);new.style=style
    if text is not None:
        r=new.add_run(text);r.font.name='Times New Roman';r.font.size=Pt(10 if style=='Caption' else 12)
        r._r.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体')
    return new

inserted=[]
def add_figure(after,stem,caption,note):
    p=new_after(after);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    fmt=p.paragraph_format;fmt.left_indent=Inches(-.116);fmt.right_indent=Inches(-.116)
    fmt.space_before=Pt(6);fmt.space_after=Pt(0);fmt.keep_with_next=True;fmt.keep_together=True;fmt.line_spacing=1
    shape=p.add_run().add_picture(str(BASE/'figures'/f'{stem}.png'),width=Inches(6))
    shape._inline.docPr.set('descr',caption)
    cap=new_after(p,'Caption',caption);cap.alignment=WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.keep_with_next=True;cap.paragraph_format.keep_together=True
    cap.paragraph_format.space_before=Pt(3);cap.paragraph_format.space_after=Pt(3)
    n=new_after(cap,'Normal','注：'+note);n.paragraph_format.first_line_indent=Pt(0)
    n.paragraph_format.line_spacing=1.05;n.paragraph_format.space_after=Pt(6);n.paragraph_format.keep_together=True
    for r in n.runs:r.font.size=Pt(9.5)
    inserted.append(dict(caption=caption,figure=stem,note=note,width_inches=6))
    return n

# Remove the 12 old results plus the case image, which is replaced below.
removed=[]
for i,p in enumerate(ps):
    if re.match(r'^图\s*[79]-\d+\s',p.text):
        prev=ps[i-1];assert prev._p.xpath('.//w:drawing')
        removed.append(p.text)
        for item in [prev,p]:item._p.getparent().remove(item._p)
assert len(removed)==13

settext(ps[457],'从表7-4可以得到四点观察；两种场景的核心指标与完结包时延分档统一见7.3节末的图7-1、图7-2。')
append_text(ps[461],' 各算法的SINR与MCS均值见表7-4，不再另列分布曲线。')
append_text(ps[473],' 相关物理传输指标见表7-5。')
append_text(ps[446],' 图中以柱高表示均值，并用10个seed的小点表示波动；颜色在第7章与第9章保持一致，DA-PLS统一使用青绿色。时延分档先在每个seed内计算包占比，再等权平均，不直接合并不同seed的包。')
append_text(ps[474],' 图7-1集中对比三项核心指标，图7-2进一步区分快速完成、长时间等待和过期，精确数值仍由表7-4、表7-5给出。')
cap='图 7-1 两种场景的传统基线核心指标'
note='seed 41–50共10个seed平均；柱为均值，小点为每个seed的结果。上行为Boundary08，下行为Uniform。仅保留完整包吞吐、过期率和成功包平均时延，P95等指标见表7-4、表7-5。'
p=add_figure(ps[474],'A_传统基线核心指标',cap,note)
notes_b='seed 41–50共10个seed平均。分母为测量期到达并已全部终止的包；前四档包含所有MAC完成包（含CRC损坏后完成），不等同于完整成功交付。分档依次为≤5、(5,20]、(20,100]、(100,200] ms；过期包单独计入深红档，与前四档互斥。细小色段可能难以辨认，右侧列出过期率；标签按显示精度取舍。'
add_figure(p,'B7_传统基线时延分档','图 7-2 两种场景的传统基线完结包时延分档',notes_b)

append_text(ps[552],' 图9-1将两个场景的三项核心指标并列展示：DA-PLS在Boundary08下优于DPLS，而Uniform下两者基本持平。')
add_figure(ps[552],'C_协同算法核心指标','图 9-1 两种场景的协同算法核心指标','seed 41–50共10个seed平均；柱为均值，小点为每个seed的结果。近零过期率仍直接标值，0.000%对应本次实验未观测到过期；各场景采用独立纵轴范围。完整精确指标见表9-1、表9-2。')
append_text(ps[559],' 图9-2按过期率排序显示36个UE的分布：DPLS的过期集中在少数末位UE，DA-PLS在10个seed的全部36个UE上均未观测到过期。不同seed的用户拓扑不同，先在每个seed、每种算法内排序，再对同一名次求均值；第36名因此对应表中的“逐seed最差UE过期率均值”，而非固定UE编号的跨seed平均。')
add_figure(ps[559],'D_逐UE过期率','图 9-2 Boundary08下按过期率排序的逐UE结果','10个seed平均。各seed、各算法独立对36个UE由低到高排序，再平均同一名次；横轴是排序名次，不用于跨算法识别同一UE。上下图纵轴范围不同；下图重合零值仅沿横向轻微错开，不改变过期率。DPLS末位为13.706%，DA-PLS全部为0.000%。')
append_text(ps[565],' 图9-3把资源与链路方面的代价、以及服务与积压方面的收益并列呈现，说明更高SINR并非业务表现更好的充分条件。')
add_figure(ps[565],'E_代价与收益','图 9-3 Boundary08下DA-PLS相对DPLS的代价与收益','seed 41–50共10个seed平均。左侧为PRG占用率与平均SINR，右侧为出队速率、平均簇队列与最差UE过期率。出队速率采用局部范围的点图，虚线为相同业务到达下的实测输入1382.487 Mbps；理论输入为1382.40 Mbps。其余条形图均从0起。')
settext(ps[569],'时延分档与联合调整案例')
settext(ps[570],'图9-4给出两个场景的完结包时延分档。Boundary08下，PFQ有41.589%的包在100–200 ms内完成、31.344%的包过期；DA-PLS有93.249%的包在5 ms内完成，且未观测到过期。Uniform下，DPLS与DA-PLS均无过期，分别有99.923%与约100%的包在5 ms内完成。这里的完成含CRC损坏后完成，完整成功率仍见表9-1、表9-2；SINR与MCS的精确均值也由这些表格给出，其代价与收益关系见图9-3。')
add_figure(ps[570],'B9_协同算法时延分档','图 9-4 两种场景的协同算法完结包时延分档',notes_b)
settext(ps[579],'图9-5与表9-4拆解了一次真实存档调度输入中的“静默＋邻区换UE”联合调整，案例取自Boundary08场景DA-PLS的seed 43、TTI 18001。小区2在PRG 0上将UE 28改为静默，小区1在同一PRG上将UE 13换为UE 18。该调整使计划出队由87291变为85656 byte，仍高于78562 byte的下界；UE 13承担让出代价（加权变化−495.678 byte），UE 16和UE 18分别获得+5367.444和+8264.382 byte的加权服务增益。UE 28调整前后的预期服务均为0，因此加权变化为0.000 byte。净整数加权增益为13136.147515 byte，体现了第8章所述“让出值不值”的定量判断。')
add_figure(ps[579],'案例_联合调整','图 9-5 真实存档调度输入中的静默与邻区换UE联合调整','采用相同动作前输入比较调整前后完整TB的预期服务，柱上直接标出加权变化；UE 28的零值以横线与0.000标出。')
# Terminology consistency: exact numbers remain unchanged.
for t in doc._element.xpath('.//w:t'):
    if t.text:t.text=t.text.replace('自然过期率','过期率').replace('完结包时延（outcome时延）','完结包时延')
for ti in [18,19,24,25,26]:
    for row in doc.tables[ti].rows:
        if row.cells[0].text=='理论输入':settext(row.cells[0].paragraphs[0],'实测输入')
for idx in [556,558,565,567]:
    for r in ps[idx].runs:r.text=r.text.replace('理论输入','实测输入').replace('与理论输入','与实测输入')
# Add definition immediately after the existing statistics paragraph.
append_text(ps[446],' 表中的实测输入按测量窗口生成的字节数计算，与按到达率参数计算的理论输入略有差异。')
# Mark fields for update, retaining all original native equations and numbering.
settings=doc.settings.element
upd=settings.find(qn('w:updateFields'))
if upd is None:upd=OxmlElement('w:updateFields');settings.append(upd)
upd.set(qn('w:val'),'true')
doc.save(OUT)
# Audit content, table figures and equations before document-renderer field updates.
reopened=Document(OUT)
newtables=[[ [c.text for c in r.cells] for r in t.rows] for t in reopened.tables]
normalize=lambda s:s.replace('自然过期率','过期率').replace('理论输入','实测输入') if s=='理论输入' else s.replace('自然过期率','过期率')
assert [[list(map(normalize,r)) for r in t] for t in original_tables]==newtables
captions=[p.text for p in reopened.paragraphs if re.match(r'^图\s*[79]-\d+\s',p.text)]
assert len(captions)==7 and [re.search(r'[79]-\d+',x).group() for x in captions]==['7-1','7-2','9-1','9-2','9-3','9-4','9-5']
oldmath=Document(SOURCE)._element.xpath('.//m:oMath');newmath=reopened._element.xpath('.//m:oMath')
assert [etree.tostring(x) for x in oldmath]==[etree.tostring(x) for x in newmath]
body='\n'.join(p.text for p in reopened.paragraphs)
assert not re.search(r'图\s*(?:7-[3-6]|9-6)\b',body)
assert len(reopened.inline_shapes)==15
report=dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),output=str(OUT),old_result_figures=12,new_result_figures=6,case_retained=True,case_renumbering='9-7 -> 9-5',tables_preserved=len(newtables),native_math_objects_preserved=len(newmath),figures=inserted,removed_captions=removed,notes=['No new simulations. 120 runs, seed 41–50.','Input rate label corrected from theoretical to measured; values unchanged.','Rank-wise UE means preserve mean-of-within-seed-worst convention.','Uniform DA-PLS and DPLS remain described as comparable.','Source uses 宋体; local PDF renderer substitutes Noto Serif CJK SC; new plots use same fallback.'])
(BASE/'修订核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(OUT)

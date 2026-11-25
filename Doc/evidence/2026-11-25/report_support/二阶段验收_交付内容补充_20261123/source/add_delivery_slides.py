from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from copy import deepcopy
from lxml import etree as E
import json,hashlib,re,uuid
ROOT=Path('/data/aerial-cuda-accelerated-ran')
SRC=Path('/home/oai-mac/.codex/attachments/07b9d654-0525-4ada-ae60-0d74f21889a1/AI-MAC项目交流-二阶段验收.pptx')
OUT=ROOT/'outputs/二阶段验收_交付内容补充_20261123'
DEST=OUT/'AI-MAC项目交流-二阶段验收-补充交付包与研究报告.pptx'
NS={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships','p14':'http://schemas.microsoft.com/office/powerpoint/2010/main'}
REL='http://schemas.openxmlformats.org/package/2006/relationships'
EMU=914400;GREEN='445437';TITLE='34432D';BODY='202820';GRAY='5B6558';PALE='F1F4ED';LINE='BBC6B3'
def el(tag,parent=None,**attrs):
 pre,name=tag.split(':');n=E.Element('{'+NS[pre]+'}'+name,{k:str(v) for k,v in attrs.items()})
 if parent is not None:parent.append(n)
 return n
def xml(n):return E.tostring(n,xml_declaration=True,encoding='UTF-8',standalone=True)
class Slide:
 def __init__(self,source,title):
  self.root=deepcopy(source);self.tree=self.root.find('p:cSld/p:spTree',NS);self.sid=100
  for c in list(self.tree):
   if E.QName(c).localname in ['nvGrpSpPr','grpSpPr','pic']:continue
   vals=c.xpath('.//a:t/text()',namespaces=NS)
   if any(t in ['课题概况','当前进展','后续安排','问题交流'] for t in vals):continue
   self.tree.remove(c)
  # Clone the original title shape, retaining font family, size, and position.
  title_shape=next(s for s in source.findall('.//p:sp',NS) if '总结' in s.xpath('.//a:t/text()',namespaces=NS))
  s=deepcopy(title_shape);tx=s.find('p:txBody',NS);p=tx.find('a:p',NS)
  for other in list(tx.findall('a:p',NS))[1:]:tx.remove(other)
  p.find('.//a:t',NS).text=title
  self.tree.append(s)
 def box(self,name,x,y,w,h,fill=None,line=None):
  self.sid+=1;s=el('p:sp',self.tree);nv=el('p:nvSpPr',s)
  el('p:cNvPr',nv,id=self.sid,name=name);el('p:cNvSpPr',nv,txBox='1');el('p:nvPr',nv)
  pr=el('p:spPr',s);xf=el('a:xfrm',pr);el('a:off',xf,x=round(x*EMU),y=round(y*EMU));el('a:ext',xf,cx=round(w*EMU),cy=round(h*EMU));el('a:avLst',el('a:prstGeom',pr,prst='rect'))
  if fill:el('a:srgbClr',el('a:solidFill',pr),val=fill)
  else:el('a:noFill',pr)
  ln=el('a:ln',pr,w='6350')
  if line:el('a:srgbClr',el('a:solidFill',ln),val=line)
  else:el('a:noFill',ln)
  return s
 def text(self,name,x,y,w,h,lines,size=16,color=BODY,bold=False,align='l',font='Arial',fill=None):
  s=self.box(name,x,y,w,h,fill);tx=el('p:txBody',s)
  bp=el('a:bodyPr',tx,wrap='square',lIns='0',rIns='0',tIns='0',bIns='0',anchor='t');el('a:noAutofit',bp);el('a:lstStyle',tx)
  for line in lines.split('\n'):
   p=el('a:p',tx);pp=el('a:pPr',p,algn=align);el('a:spcPct',el('a:lnSpc',pp),val='112000');el('a:spcPts',el('a:spcBef',pp),val='0');el('a:spcPts',el('a:spcAft',pp),val='0');el('a:buNone',pp)
   r=el('a:r',p);pr=el('a:rPr',r,lang='zh-CN',altLang='en-US',sz=round(size*100),b='1' if bold else '0',dirty='0');el('a:srgbClr',el('a:solidFill',pr),val=color)
   el('a:latin',pr,typeface=font);el('a:ea',pr,typeface='微软雅黑');el('a:cs',pr,typeface='Arial');el('a:t',r).text=line
   el('a:endParaRPr',p,lang='zh-CN',sz=round(size*100))
  return s
 def picture(self,name,rid,x,y,w,h):
  self.sid+=1;p=el('p:pic',self.tree);nv=el('p:nvPicPr',p);el('p:cNvPr',nv,id=self.sid,name=name,descr='用户提供的二阶段研究报告首页');el('a:picLocks',el('p:cNvPicPr',nv),noChangeAspect='1');el('p:nvPr',nv)
  bf=el('p:blipFill',p);b=el('a:blip',bf);b.set('{'+NS['r']+'}embed',rid);el('a:fillRect',el('a:stretch',bf))
  pr=el('p:spPr',p);xf=el('a:xfrm',pr);el('a:off',xf,x=round(x*EMU),y=round(y*EMU));el('a:ext',xf,cx=round(w*EMU),cy=round(h*EMU));el('a:avLst',el('a:prstGeom',pr,prst='rect'));el('a:srgbClr',el('a:solidFill',el('a:ln',pr,w='6350')),val=LINE)
with ZipFile(SRC) as z:
 base=E.fromstring(z.read('ppt/slides/slide22.xml'))
 a=Slide(base,'二阶段实验代码与结果交付包')
 a.text('交付概述',.73,1.82,10.5,.34,'源码、配置、结果与核查工具已按研究报告完成最小化整理',16,GREEN)
 a.text('压缩包体积',11.3,1.82,1.25,.34,'约 5.8 MB',16,GREEN,True,align='r')
 a.text('目录栏目',.73,2.38,6.30,.34,'目录架构',18,TITLE,True)
 a.text('实验覆盖栏目',7.42,2.38,5.18,.34,'实验覆盖',18,TITLE,True)
 a.box('目录标题线',.73,2.80,6.30,.022,GREEN);a.box('实验标题线',7.42,2.80,5.18,.022,GREEN)
 a.box('两栏分隔线',7.18,2.98,.009,3.36,'D4DCCF')
 rows=[('交付包/',''),('├─ code/','算法实现与实验脚本'),('├─ native/','冻结 cuMAC 与仿真源码'),('├─ protocols/','实验协议与运行配置'),('├─ results/','结果数据与搜索审计'),('│  ├─ main/','主场景'),('│  ├─ generalization/','泛化场景'),('│  ├─ input_simplification/','输入简化'),('│  └─ search_audit/','搜索耗时与等价性'),('├─ sample/','离线调度样例'),('├─ validation/','验证记录'),('└─ provenance/','文件来源与版本身份')]
 for i,(code,desc) in enumerate(rows):
  y=2.99+i*.275
  a.text('目录_'+str(i),.84,y,3.52,.24,code,12.6,GREEN if i==0 else BODY,i==0,font='Consolas')
  if desc:a.text('目录说明_'+str(i),4.40,y,2.57,.24,desc,12.6,GRAY)
 data=[('传统基线与 DA-PLS 主场景','120 组'),('G1–G4 复杂场景泛化','60 组'),('RSRP 输入简化及宽带对照','36 组'),('搜索耗时与等价性审计','90 个状态')]
 for i,(name,num) in enumerate(data):
  y=3.00+i*.40
  a.text('实验_'+str(i),7.43,y,3.90,.31,name,15.2)
  a.text('数量_'+str(i),11.44,y,1.15,.31,num,15.2,GREEN,True,align='r')
 a.text('复用说明',7.43,4.68,5.18,.24,'部分结果复用既有实验，各项不直接相加。',10.5,GRAY)
 a.text('验证标题',7.43,5.12,5.18,.31,'交付与验证状态',17,TITLE,True)
 a.text('核查通过1',7.43,5.57,5.18,.30,'• 696 项报告数值复核通过',14.5)
 a.text('核查通过2',7.43,5.98,5.18,.49,'• 离线编译、搜索一致性及文件校验通过',14.5)
 a.box('使用入口底色',.73,6.64,11.87,.35,PALE)
 a.text('使用入口',.87,6.70,11.55,.24,'使用入口：README.md  ·  结果速览.csv  ·  verify_results.py  ·  replay_sample.py',11.5,GREEN)
 a.text('复现边界',.73,7.10,11.87,.24,'用于交付审阅与离线核查；完整仿真仍需 Aerial/CUDA 环境及外部依赖，未附全量原始轨迹。',10.5,GRAY)
 b=Slide(base,'二阶段技术研究报告')
 b.text('报告全名',.73,1.82,11.88,.38,'《基于 AI 技术的多维协同资源分配研究技术研究报告（二阶段）》',15.5,GREEN)
 b.picture('研究报告实际首页','rIdReportCover',.88,2.44,2.79,3.946)
 b.text('报告封面说明',.83,6.48,2.90,.27,'研究报告（二阶段修订版）',12,GREEN,align='ctr')
 b.text('报告结构标题',4.08,2.40,8.50,.35,'报告结构：11 章正文＋附录 A',18,TITLE,True)
 b.box('报告结构线',4.08,2.83,8.50,.022,GREEN)
 report=[('研究基础｜第 1–4 章','研究背景、NR 调度模型，以及传统与 AI 调度方法。'),('平台与基线｜第 5–7 章','Aerial 缺陷修正、统一仿真场景与评测协议、传统基线。'),('核心方法与验证｜第 8–9 章','DA-PLS 与精确剪枝；主场景、机制分析及 G1–G4 泛化。'),('输入简化与总结｜第 10–11 章、附录 A','RSRP／空间协方差验证、研究结论及逐小区传输状态。')]
 for i,(head,body) in enumerate(report):
  y=3.02+i*.79
  b.text('章节标题_'+str(i),4.08,y,8.50,.31,head,16,TITLE,True)
  b.text('章节内容_'+str(i),4.08,y+.38,8.50,.32,body,14.8)
 b.box('报告成果带',.73,6.93,11.87,.022,GREEN)
 b.text('报告验收支撑',.73,7.05,11.87,.28,'验收支撑：统一说明研究方法、实验协议与评测结论，关键数据与交付源码和结果文件对应。',12,GREEN)
 # Dedicated claim block remains distinct from the chapter outline.
 b.text('报告核心结论',4.08,6.33,8.50,.49,'核心结论：Boundary08 完整包吞吐较 PFQ 提升 67.78%，\n成功包平均时延由 121.577 ms 降至 2.624 ms。',13.8,GREEN,True)
 updates={}
 titles=['二阶段实验代码与结果交付包','二阶段技术研究报告']
 notes=[
  '本页介绍按二阶段最终研究报告整理的最小源码与结果交付包。主场景120条结果、泛化60条、输入简化36条；输入简化包含复用实验，不能将这些条目相加称为独立仿真次数。搜索审计含90个存档状态。包内结果通过696项报告数值复核、504项聚合数据复核；样例从包内源码完成离线编译，24轮38376个候选审计通过。完整GPU端到端仿真仍需原Aerial/CUDA和目录依赖，未附全量逐TTI/逐包原始轨迹。约5.8MB为ZIP十进制大小。目录仅展示主要项，results还含platform。',
  '封面缩略图来自用户提供的二阶段修订报告原文件。报告共11章正文及附录A：1–4章为背景、基础、传统与AI方法；5–7章为平台修正、场景和传统基线；8–9章为DA-PLS、精确剪枝、主场景及泛化评测；10章为输入简化，11章总结，附录A为逐小区传输状态。Boundary08指标为seed41–50的等权均值：完整包吞吐749.497→1257.499Mbps，成功包平均时延121.577→2.624ms。核心数字与交付包结果一致；不将逻辑精确性等同于全局最优，也不把搜索耗时等同于端到端实时部署时延。'
 ]
 for index,slide in zip([26,27],[a,b]):
  updates[f'ppt/slides/slide{index}.xml']=xml(slide.root)
  rels=E.fromstring(z.read('ppt/slides/_rels/slide22.xml.rels'))
  for r in rels:
   if r.get('Type').endswith('/notesSlide'):r.set('Target',f'../notesSlides/notesSlide{index}.xml')
  if index==27:E.SubElement(rels,'{'+REL+'}Relationship',Id='rIdReportCover',Type=NS['r']+'/image',Target='../media/report_delivery_cover.png')
  updates[f'ppt/slides/_rels/slide{index}.xml.rels']=xml(rels)
  note=E.fromstring(z.read('ppt/notesSlides/notesSlide22.xml'))
  sp=note.xpath('//p:sp[p:nvSpPr/p:nvPr/p:ph[@type="body"]]',namespaces=NS)[0];tx=sp.find('p:txBody',NS)
  for pp in list(tx.findall('a:p',NS)):tx.remove(pp)
  pp=el('a:p',tx);r=el('a:r',pp);el('a:rPr',r,lang='zh-CN',sz='1200');el('a:t',r).text=notes[index-26]
  for fld in note.findall('.//a:fld',NS):fld.set('id','{'+str(uuid.uuid4()).upper()+'}')
  updates[f'ppt/notesSlides/notesSlide{index}.xml']=xml(note)
  nr=E.fromstring(z.read('ppt/notesSlides/_rels/notesSlide22.xml.rels'))
  for r in nr:
   if r.get('Type').endswith('/slide'):r.set('Target',f'../slides/slide{index}.xml')
  updates[f'ppt/notesSlides/_rels/notesSlide{index}.xml.rels']=xml(nr)
 updates['ppt/media/report_delivery_cover.png']=(OUT/'preview/报告首页.png').read_bytes()
 pres=E.fromstring(z.read('ppt/presentation.xml'));lst=pres.find('p:sldIdLst',NS);oldids=[x.get('id') for x in lst]
 newids=[str(max(map(int,oldids))+i) for i in [1,2]]
 rels=E.fromstring(z.read('ppt/_rels/presentation.xml.rels'));nextrel=max(int(r.get('Id')[3:]) for r in rels if r.get('Id').startswith('rId'))+1
 for k,num in enumerate([26,27]):
  rid='rId'+str(nextrel+k);E.SubElement(rels,'{'+REL+'}Relationship',Id=rid,Type=NS['r']+'/slide',Target=f'slides/slide{num}.xml')
  s=el('p:sldId',id=newids[k]);s.set('{'+NS['r']+'}id',rid);lst.insert(23+k,s)
 for section in pres.findall('.//p14:sldIdLst',NS):
  for i,s in enumerate(list(section)):
   if s.get('id')==oldids[22]:
    for k in range(2):section.insert(i+1+k,el('p14:sldId',id=newids[k]))
 updates['ppt/presentation.xml']=xml(pres);updates['ppt/_rels/presentation.xml.rels']=xml(rels)
 ct=E.fromstring(z.read('[Content_Types].xml'));ctns=E.QName(ct).namespace
 for i in [26,27]:
  E.SubElement(ct,'{'+ctns+'}Override',PartName=f'/ppt/slides/slide{i}.xml',ContentType='application/vnd.openxmlformats-officedocument.presentationml.slide+xml')
  E.SubElement(ct,'{'+ctns+'}Override',PartName=f'/ppt/notesSlides/notesSlide{i}.xml',ContentType='application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml')
 updates['[Content_Types].xml']=xml(ct)
 app=E.fromstring(z.read('docProps/app.xml'));an=E.QName(app).namespace;vt='http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes'
 app.find('{'+an+'}Slides').text='27';app.find('{'+an+'}Notes').text='27'
 vector=app.find('{'+an+'}HeadingPairs/{'+vt+'}vector');variants=list(vector)
 for i,v in enumerate(variants):
  lp=v.find('{'+vt+'}lpstr')
  if lp is not None and lp.text=='幻灯片标题':variants[i+1].find('{'+vt+'}i4').text='27'
 vec=app.find('{'+an+'}TitlesOfParts/{'+vt+'}vector');start=len(vec)-25
 for k,t in enumerate(titles):
  n=E.Element('{'+vt+'}lpstr');n.text=t;vec.insert(start+23+k,n)
 vec.set('size',str(len(vec)));updates['docProps/app.xml']=xml(app)
 with ZipFile(DEST,'w',ZIP_DEFLATED,compresslevel=6) as out:
  for info in z.infolist():out.writestr(info,updates.pop(info.filename,z.read(info.filename)))
  for name,data in updates.items():out.writestr(name,data)
# Original slide XML and relationships remain byte-for-byte identical.
with ZipFile(SRC) as old,ZipFile(DEST) as new:
 assert new.testzip() is None
 for name in old.namelist():
  if name not in ['ppt/presentation.xml','ppt/_rels/presentation.xml.rels','[Content_Types].xml','docProps/app.xml']:assert old.read(name)==new.read(name),name
 p=E.fromstring(new.read('ppt/presentation.xml'));r=E.fromstring(new.read('ppt/_rels/presentation.xml.rels'));m={x.get('Id'):x.get('Target') for x in r};order=[m[x.get('{'+NS['r']+'}id')] for x in p.find('p:sldIdLst',NS)]
 assert order[22:27]==['slides/slide23.xml','slides/slide26.xml','slides/slide27.xml','slides/slide24.xml','slides/slide25.xml']
 for num in [26,27]:
  x=E.fromstring(new.read(f'ppt/slides/slide{num}.xml'))
  for xf in x.findall('.//a:xfrm',NS):
   pos=xf.find('a:off',NS);size=xf.find('a:ext',NS)
   if pos is not None and size is not None:
    assert int(pos.get('x'))>=0 and int(pos.get('y'))>=0
    assert int(pos.get('x'))+int(size.get('cx'))<=12192000
    assert int(pos.get('y'))+int(size.get('cy'))<=6858000
 result={'source_slide_count':25,'output_slide_count':27,'inserted_positions':[24,25],'inserted_titles':titles,'original_parts_preserved':True,'zip_integrity':True,'shape_bounds_valid':True,'source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(DEST.read_bytes()).hexdigest()}
 (OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(DEST);print(json.dumps(result,ensure_ascii=False))

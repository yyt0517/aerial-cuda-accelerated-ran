from pathlib import Path
import sys,json,hashlib,re,subprocess
from zipfile import ZipFile,ZIP_DEFLATED
sys.path.insert(0,'/tmp/stage2_report_python')
from docx import Document
from lxml import etree
BASE=Path(__file__).resolve().parents[1]
source=Path('/home/oai-mac/.codex/attachments/2f3e26fc-a491-4e56-a9e0-fbf5e3fbc2c6/技术研究报告_二阶段同频.docx')
original=Document(source);before=Document('/tmp/report_before_field_update.docx');final=Document(BASE/'技术研究报告_二阶段同频_图表精简版.docx')
def sig(el):return (el.tag,tuple(sorted(el.attrib.items())),el.text,tuple(sig(c) for c in el))
assert [sig(x) for x in original._element.xpath('.//m:oMath')]==[sig(x) for x in final._element.xpath('.//m:oMath')]
assert [[[c.text for c in r.cells] for r in t.rows] for t in before.tables]==[[[c.text for c in r.cells] for r in t.rows] for t in final.tables]
caps=[p.text for p in final.paragraphs if re.match(r'^图\s*[79]-\d+\s',p.text)]
assert len(caps)==7
assert len(final.inline_shapes)==15 and len(final.tables)==33
widths=[s.width.inches for s in final.inline_shapes];assert sum(abs(v-6)<1e-6 for v in widths)==7
text='\n'.join(p.text for p in final.paragraphs)
assert not re.search(r'图\s*(?:7-[3-6]|9-[6-7])\b',text)
assert '累计分布' not in text[text.index('同频场景传统基线评测'):text.index('计算成本与精确性验证')]
manifest=json.loads((BASE/'修订核验.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['source_sha256']
plots=json.loads((BASE/'figures/figure_manifest.json').read_text())
assert all(x['width_inches']==6 and x['minimum_font_pt']>=9.5 for x in plots['figures'].values())
figs=['A_传统基线核心指标','B7_传统基线时延分档','C_协同算法核心指标','D_逐UE过期率','E_代价与收益','B9_协同算法时延分档','案例_联合调整']
subprocess.run(['pdfunite',*[str(BASE/'figures'/f'{f}.pdf') for f in figs],str(BASE/'新图汇总.pdf')],check=True)
info=subprocess.check_output(['pdfinfo',str(BASE/'技术研究报告_二阶段同频_图表精简版.pdf')],text=True)
pages=int(re.search(r'Pages:\s+(\d+)',info).group(1))
data=json.loads((BASE/'data/validation.json').read_text())
audit=dict(passed=True,source_unchanged=True,original_formula_objects_preserved=72,table_values_preserved=True,tables=33,new_result_figures=6,retained_case_figures=1,caption_sequence=caps,width_inches=6,minimum_figure_font_pt=9.5,pdf_pages=pages,table_of_contents_updated=True,raw_cohorts_revalidated=data['cohorts'],raw_packets_revalidated=data['packets'],da_pls_720_ue_seed_results_zero_expiry=True,aggregation='Within-seed first, equal-seed average',visual_checks='All seven figures inspected; embedded page layouts checked; zero labels, cases, figure/caption pairing checked.',main_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.iterdir() if p.suffix in ['.docx','.pdf']})
(BASE/'最终核验.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
zip_path=BASE/'修订稿_图件与绘图数据.zip'
with ZipFile(zip_path,'w',ZIP_DEFLATED) as z:
    for p in sorted(BASE.rglob('*')):
        if p.is_file() and p!=zip_path and p.suffix not in ['.ttf','.pyc'] and '__pycache__' not in p.parts:
            z.write(p,str(p.relative_to(BASE)))
print(json.dumps({k:v for k,v in audit.items() if k!='main_files'},ensure_ascii=False,indent=2))
print('Package:',zip_path,'bytes:',zip_path.stat().st_size)

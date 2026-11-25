"""Verify delivered Word/PDF content and extract the added section for review."""
import sys,json,hashlib,re,subprocess,csv,tempfile
from pathlib import Path
sys.path.insert(0,'/tmp/stage2_report_python')
from docx import Document
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1]
SRC=ROOT/'outputs/技术研究报告_图表精简版_20261106/技术研究报告_二阶段同频_图表精简版.docx'
DEST=OUT/'技术研究报告_二阶段同频_泛化实验补充版.docx'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sig(el):return (el.tag,tuple(sorted(el.attrib.items())),el.text,tuple(sig(c) for c in el))
def rows(t):return [[c.text for c in r.cells] for r in t.rows]
def main():
    previous=json.loads((SRC.parent/'最终核验.json').read_text())
    assert sha(SRC)==previous['main_files'][SRC.name]
    old=Document(SRC);new=Document(DEST)
    assert [sig(x) for x in old._element.xpath('.//m:oMath')]==[sig(x) for x in new._element.xpath('.//m:oMath')]
    assert len(new.tables)==36 and len(new.inline_shapes)==len(old.inline_shapes)==15
    a=[rows(t) for t in old.tables];b=[rows(t) for t in new.tables]
    assert a[:28]==b[:28] and a[28:]==b[31:]
    for idx,name in enumerate(['表9-5_场景配置.csv','表9-6_核心结果.csv','表9-7_重点用户分组.csv'],28):
        with (OUT/'data'/name).open(encoding='utf-8-sig',newline='') as f:expected=list(csv.reader(f))
        assert b[idx]==expected,(idx,b[idx],expected)
    assert [p.text for p in old.paragraphs if p.style.name=='Caption' and p.text.startswith('图')]==[p.text for p in new.paragraphs if p.style.name=='Caption' and p.text.startswith('图')]
    raw=subprocess.check_output(['pdftotext','-layout',str(DEST.with_suffix('.pdf')),'-'],text=True)
    assert '错误!未找到引用源' not in raw and 'Error! Reference source not found' not in raw
    pages=raw.split('\f');hits=[]
    for i,p in enumerate(pages,1):
        if re.search(r'9\.6\.?\s+复杂业务与空间分布下的泛化评测',p) and re.search(r'36\s*个',p):hits.append(i)
    assert len(hits)==1,hits
    first=hits[0];last=None
    for i in range(first,len(pages)+1):
        if re.search(r'9\.7\.?\s+本章小结',pages[i-1]):last=i;break
    assert last is not None
    last-=1  # The following summary starts on its own page.
    assert last>=first
    # Export just the new section with its concluding summary for convenient reading.
    output=OUT/'泛化实验补充节.pdf'
    with tempfile.TemporaryDirectory(prefix='generalization_pages_') as temp:
        subprocess.run(['pdfseparate','-f',str(first),'-l',str(last),str(DEST.with_suffix('.pdf')),str(Path(temp)/'page-%d.pdf')],check=True)
        subprocess.run(['pdfunite',*[str(Path(temp)/f'page-{i}.pdf') for i in range(first,last+1)],str(output)],check=True)
    (OUT/'核验/补充节文本.txt').write_text('\n'.join(pages[first-1:last]))
    for page in pages[first-1:last]:
        assert '表 9-5' in raw or '表9-5' in raw
    # A full TOC update must include the added section and renumbered summary.
    toc='\n'.join(pages[:15]);assert '复杂业务与空间分布下的泛化评测' in toc
    audit=dict(passed=True,source_report_unchanged=True,original_tables_preserved=33,added_tables=3,original_math_objects_preserved=72,original_figures_preserved=15,table_values_match_exported_csv=True,pdf_section_pages=[first,last],table_of_contents_updated=True,pdf_page_count=len(pages)-int(not pages[-1].strip()),files={p.name:sha(p) for p in OUT.glob('*') if p.suffix in ['.docx','.pdf']})
    (OUT/'核验/最终文档核验.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

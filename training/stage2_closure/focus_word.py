"""Narrow the reviewed Word report without rebuilding unaffected chapters."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


def replace_text(paragraph, text):
    nodes = list(paragraph._p.iter(qn('w:t')))
    if nodes:
        nodes[0].text = text
        for node in nodes[1:]:
            node.text = ''
    else:
        paragraph.add_run(text)


def main(source, target):
    doc = Document(source)
    body = doc.element.body
    removed = []

    def paragraph_start(prefix):
        found = [p for p in doc.paragraphs if p.text.startswith(prefix)]
        assert len(found) == 1, (prefix, len(found))
        return found[0]

    def remove_range(first, stop):
        node = paragraph_start(first)._p
        end = paragraph_start(stop)._p
        while node is not end:
            nxt = node.getnext()
            body.remove(node)
            node = nxt
        removed.append(first)

    remove_range('7.5  Boundary高负载', '7.8  本章结论')
    remove_range('9.4  Boundary高负载', '9.8  分源干扰')

    # Delete caption and associated object, preserving neighboring analysis.
    for p in list(doc.paragraphs):
        if p.style.name == 'Caption' and ('同PRG活动小区数分布' in p.text or '配对差值' in p.text):
            node = p._p.getnext()
            assert node.tag == qn('w:tbl'), p.text
            body.remove(node)
            removed.append(p.text)
            body.remove(p._p)
        elif p.style.name == 'Caption' and '三个Boundary负载的适用边界' in p.text:
            node = p._p.getprevious()
            assert node.findall('.//' + qn('w:drawing')), p.text
            body.remove(node)
            removed.append(p.text)
            body.remove(p._p)

    # Main environment list and metric definitions follow the reduced scope.
    for table in doc.tables:
        for row in list(table.rows):
            text = ' '.join(c.text for c in row.cells)
            if 'Boundary高负载' in text:
                table._tbl.remove(row._tr)
            elif '同PRG活动小区数分布' in text:
                row.cells[0].text = '资源占用与效率'
                row.cells[1].text = 'cell-PRG占用／全部可用cell-PRG；每已用PRG的成功字节数'

    for p in list(doc.paragraphs):
        text = p.text
        if text.startswith(('高负载下DA-PLS相对缺口权重', '高负载检验中，M-LWDF-PF')):
            body.remove(p._p)
            continue
        text = text.replace('2026年11月5日 · 评测扩充修订版', '2026年11月5日 · 主场景评测修订版')
        text = text.replace('并给出逐seed配对区间', '逐seed计算后等权汇总')
        text = text.replace('第9章逐环境报告算法质量与消融', '第9章逐环境报告算法质量与机制分析')
        text = text.replace('Boundary主负载取λ=0.8，Uniform取λ=1.0，Boundary高负载取λ=1.12和1.28。', 'Boundary主负载取λ=0.8，Uniform取λ=1.0。')
        text = text.replace('正式结论首先来自每个环境配置内的同seed配对比较，高负载倍数则统一以Boundary的λ=0.8为基准。', '正式结论来自每个环境配置内使用相同seed的算法比较。')
        text = text.replace('并给出权重消融。', '。').replace('DA-PLS，。', 'DA-PLS。')
        text = text.replace('、活动小区数分布及各小区谱效', '及各小区谱效')
        text = text.replace('，也不以主负载结果代替过载结果', '')
        text = text.replace('；配对区间以同seed差值计算，未作多重比较校正', '，未作多重比较校正')
        text = text.replace('相同的四个环境配置', '相同的Boundary08主负载与Uniform两个环境配置')
        text = text.replace('物理传输状态、资源复用与小区谱效', '物理传输状态、资源效率与小区谱效')
        text = text.replace('逐UE、交付结局和资源复用指标提供互补证据', '逐UE、交付结局和资源效率指标提供互补证据')
        text = text.replace('高负载时许多候选共享这些键', '许多候选共享这些键')
        text = text.replace('高负载需同时观察缺口权重对照；不在本报告中引入未验证的负载切换。', '当前业务质量结论限于Boundary08与Uniform两个配置，不据此推断其它业务负载下的表现。')
        text = text.replace('十seed配对中', '十seed等权均值中')
        text = text.replace('差值区间、44–50子集及逐UE限制见第9章。', '两类时延、完整交付及逐UE指标见第9章。')
        text = text.replace('逐seed/逐UE统计、配对区间、等价性与成本核验', '逐seed/逐UE统计、等价性与成本核验')
        if text.startswith('uniform中，相对PFQ的成功包P95'):
            text = 'Uniform的均值结果显示DA-PLS相对PFQ有所改善，但不能据此判定每个seed都获得相同收益；场景结论应结合完整交付、时延和用户分布阅读。'
        elif text.startswith('正式矩阵合计220个配置'):
            text = '本版业务质量评测覆盖Boundary08与Uniform，每个环境包含RR、RRQ、PF、PFQ、缺口权重DPLS及DA-PLS六种方法，各使用seed 41–50，共120个不同的配置—seed点。第7章与本章的PFQ结果共用相同轨迹。41–43参与过本次权重选型，44–50在项目历史中也有暴露，因此本轮结果不称完全未见盲测。'
        elif text.startswith('Uniform中M-LWDF-PF相对缺口权重'):
            text = 'Uniform中M-LWDF-PF相对缺口权重的整包吞吐均值略低，两者均未观测到自然过期。Boundary08的权重收益不能直接推广为所有布点下的优势；本节以各项指标的实际均值和分布共同判断服务效果。'
        if text != p.text:
            replace_text(p, text)

    # Renumber the remaining headings and caption sequences in the edited chapters.
    chapter = None
    counters = {}
    caption_map = {}
    for p in doc.paragraphs:
        if p.style.name == 'Heading 1':
            match = re.match(r'(\d+)\s+', p.text)
            chapter = int(match[1]) if match else None
            counters = {2: 0, 3: 0}
        elif chapter in (7, 9) and p.style.name in ('Heading 2', 'Heading 3'):
            level = int(p.style.name[-1])
            counters[level] += 1
            if level == 2:
                counters[3] = 0
            suffix = re.sub(r'^\d+(?:\.\d+)*\s+', '', p.text)
            number = f'{chapter}.{counters[2]}' + (f'.{counters[3]}' if level == 3 else '')
            replace_text(p, number + '  ' + suffix)
        elif p.style.name == 'Caption':
            match = re.match(r'([表图])\s+([79])-(\d+)\s+(.*)', p.text)
            if match:
                kind, chap, _, suffix = match.groups()
                key = (kind, chap)
                counters[key] = counters.get(key, 0) + 1
                old = f'{kind} {chap}-{match[3]}'
                new = f'{kind} {chap}-{counters[key]}'
                caption_map[old] = new
                replace_text(p, new + ' ' + suffix)

    for p in doc.paragraphs:
        if p.style.name != 'Caption':
            new = re.sub(r'[表图]\s+[79]-\d+(?!\d)', lambda m: caption_map.get(m[0], m[0]), p.text)
            if new != p.text:
                replace_text(p, new)

    target.parent.mkdir(parents=True, exist_ok=True)
    doc.save(target)
    text = '\n'.join(p.text for p in doc.paragraphs)
    text += '\n' + '\n'.join(c.text for t in doc.tables for r in t.rows for c in r.cells)
    assert not any(s in text for s in ['权重消融', '配对区间', '配对差值', '活动小区数分布', 'Boundary高负载', 'p99', 'P99', 'CTBS'])
    (target.parent / '正文核验.txt').write_text(text)
    (target.parent / '修订记录.json').write_text(json.dumps({
        'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'target': str(target), 'removed': removed, 'captions': caption_map,
        'quality_scope': ['Boundary08/rate0.8', 'Uniform/rate1.0'],
        'quality_experiment_points': 120, 'pdf_generated': False,
        'chapter10': 'Separate computation and exactness audit retained from previous revision',
    }, ensure_ascii=False, indent=2) + '\n')
    print(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    main(args.source.resolve(), args.target.resolve())

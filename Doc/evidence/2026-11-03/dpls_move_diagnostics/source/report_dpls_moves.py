"""Archive the four logged-move diagnostics, with scope and numerical limits."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from .analyze_dpls_moves import sha, write_json, write_csv, distribution


def number(row, key):
    return float(row[key])


def average(rows, key):
    return float(np.mean([number(row, key) for row in rows])) if rows else None


def main(args):
    source, out = Path(args.source).resolve(), Path(args.output).resolve()
    completed = json.loads((source/'completion.json').read_text())
    if not completed['completed']:
        raise ValueError('Incomplete analysis')
    out.mkdir(parents=True, exist_ok=False)
    for name in ('protocol.json', 'completion.json', 'input_manifest.json', 'summary.json', 'rounds.csv', 'transitions.csv'):
        shutil.copy2(source/name, out/name)
    (out/'source').mkdir()
    shutil.copy2(source/'analyze_dpls_moves.py', out/'source/analyze_dpls_moves.py')
    for name in ('report_dpls_moves.py', 'test_dpls_move_analysis.py'):
        shutil.copy2(Path(__file__).with_name(name), out/'source'/name)
    if args.tests:
        shutil.copy2(args.tests, out/'tests.txt')
    rows = list(csv.DictReader((source/'rounds.csv').open()))
    summary = json.loads((source/'summary.json').read_text())
    manifest = json.loads((source/'input_manifest.json').read_text())
    native = Path(json.loads((source/'protocol.json').read_text())['source'])
    additional = []
    for arm, stats in summary.items():
        subset = [r for r in rows if r['arm'] == arm]
        accepted = [r for r in subset if r['accepted_moves'] == '1']
        late = [r for r in accepted if int(r['round']) > 5]
        with_input = [r for r in accepted if r['proxy_available'] == 'True']
        item = dict(arm=arm, accepted_rounds=len(accepted),
            accepted_round_fraction=len(accepted)/len(subset),
            parallel_positive_accepted_lower=average(accepted, 'parallel_lower'),
            parallel_positive_accepted_upper=average(accepted, 'parallel_upper'),
            parallel_greedy_accepted=average(accepted, 'parallel_greedy'),
            parallel_positive_after5_lower=average(late, 'parallel_lower'),
            parallel_positive_after5_upper=average(late, 'parallel_upper'),
            parallel_qualifying_after5_lower=average(late, 'qualifying_parallel_lower'),
            saved_input_accepted_rounds=len(with_input),
            saved_input_structural_lower=average(with_input, 'parallel_lower'),
            saved_input_floor_lower=average(with_input, 'parallel_floor_lower'),
            saved_input_floor_upper=average(with_input, 'parallel_floor_upper'),
            saved_input_floor_greedy=average(with_input, 'parallel_floor_greedy'))
        # Additional descriptive checks use only saved action arrays/JSON gains.
        activity_changed = accepted_count = 0
        for trace in sorted((native/f'{arm}_seed41/traces').glob('*.npz')):
            with np.load(trace) as z:
                for i, chosen in enumerate(z['chosen_index']):
                    if chosen < 0:
                        continue
                    at = z['offsets'][i]+chosen
                    before = z['roots'][i, :, z['prg'][at]] >= 0
                    after = z['assignment'][at] >= 0
                    activity_changed += int(np.any(before != after))
                    accepted_count += 1
        item['accepted_moves_changing_activity_fraction'] = activity_changed/max(1, accepted_count)
        if arm.endswith('full'):
            decisions = [json.loads(line) for line in (native/f'{arm}_seed41/decisions.jsonl').read_text().splitlines()]
            retained = []
            for d in decisions:
                total = d['score']-d['initial_score']
                initial_gain = d['search_initial_score']-d['initial_score']
                prefix = initial_gain+sum(h['gain'] for h in d['history'][:5])
                retained.append(prefix/total if total > 1.e-9 else 1.)
            item['own_full_trajectory_first5_gain_retention'] = distribution(retained)
        additional.append(item)
    write_json(out/'additional_summary.json', additional)
    write_csv(out/'four_metrics.csv', [dict(arm=a, rounds=s['rounds'], accepted=s['accepted_moves'],
        changed_fraction=s['changed_fraction_candidate_weighted'], lazy_refresh_ratio=s['lazy_refresh_ratio'],
        parallel_lower=next(v['parallel_positive_accepted_lower'] for v in additional if v['arm'] == a),
        parallel_upper=next(v['parallel_positive_accepted_upper'] for v in additional if v['arm'] == a),
        sir_rank_p50=s['sir_tb_rank_eligible']['distribution']['p50'],
        sir_rank_p95=s['sir_tb_rank_eligible']['distribution']['p95'],
        sir_top16=s['sir_tb_rank_eligible']['recall']['16']) for a,s in summary.items()])
    mismatch = [r for r in rows if r['cached_chosen_matches'] != 'True']
    write_json(out/'numerical_ties.json', mismatch)
    lines = ['DPLS逐轮日志的四项诊断：惰性重评、并行移动与廉价SIR排序',
        '2026-11-03；仅分析已有日志，不新增原生轨迹、候选PHY评分或MMSE', '',
        '结论：先做跨轮精确依赖缓存与保留共同出队下界的多移动原型。候选排序提前做诊断，但当前廉价SIR的top16召回不足1%，不能直接作为硬短名单。',
        '热启动单独3–8倍降本的预期撤回；本报告的候选数量推算不是墙钟加速承诺。', '',
        '样本与口径',
        '仅使用已完成v3的seed41、4001–4032TTI：cold_full/warm_full/warm_n3各32个连续TTI，合计96个相关瞬态状态。三个臂有不同内生队列，跨臂统计不是同输入质量对照。',
        '全部96条轨迹共1952轮；保存了完整输入的12个TTI才计算逻辑TB下界与SIR，其中接受动作251个。无独立seed泛化或成熟包级验收。', '',
        '四项核心统计',
        'arm | 接受数/轮数 | 接受后公共候选ΔJ变化% | 有接受轮的无冲突正Δ移动数下界–上界 | SIR名次P50/P95 | top16召回%']
    for arm, s in summary.items():
        a = next(r for r in additional if r['arm'] == arm)
        rank = s['sir_tb_rank_eligible']
        lines.append(f"{arm} | {s['accepted_moves']}/{s['rounds']} | {100*s['changed_fraction_candidate_weighted']:.2f} | "
                     f"{a['parallel_positive_accepted_lower']:.2f}–{a['parallel_positive_accepted_upper']:.2f} | "
                     f"{rank['distribution']['p50']:.1f}/{rank['distribution']['p95']:.1f} | {100*rank['recall']['16']:.2f}")
    lines.extend(['', '1. 每轮实际接受几个移动',
        '实际每轮只有0或1个移动，这是当前固定根、单次argmax实现的结构限制。一个移动内部可能同时静默一小区并给邻区换UE，两个编辑槽位仍只算一个联合移动。',
        'full的未接受轮是各TTI的停止轮；warm_n3最后一轮接受后没有下一轮全邻域，不能拿它补算跨轮ΔJ变化。', '',
        '2. 接受后下一轮ΔJ真正变化的候选比例',
        '用(PRG, 被覆盖小区的目标UE/静默)定义编辑操作，未编辑小区用独立哨兵；在相邻轮取相同操作的交集。候选行号和完整方案都不能直接跨轮对齐。',
        '先减去各轮根J，再以绝对差>1e−6字节定义真实变化，排除整轮根J平移和浮点抵消噪声。新增/消失操作单独计数，新增者必须重评。',
        '跨轮失效集合为“同PRG，或候选受影响UE与接受移动受影响UE有交集”。活动变化时，受影响集包括该PRG所有活动物理邻居，不只包括被换入/换出的UE。',
        '只有上述充分依赖集合之外才允许复用候选ΔJ；出队可行性须随当前根D和候选ΔD廉价重算，不能缓存旧eligible布尔值。',
        'arm | 公共候选ΔJ变化% | 含每TTI首扫的依赖刷新候选/full% | 依赖外>1e−6B变化数 | 原浮点选择一致轮数 | 加并列保护后'] )
    for arm, s in summary.items():
        lines.append(f"{arm} | {100*s['changed_fraction_candidate_weighted']:.2f} | {100*s['lazy_refresh_ratio']:.2f} | "
                     f"{s['outside_dependency_changed']}/{s['outside_dependency_compared']} | {s['cached_choice_matches']}/{s['rounds']} | {s['guarded_choice_matches']}/{s['rounds']}")
    lines.extend(['依赖外最大数值漂移为5.82e−11B。未经保护的ΔJ缓存在warm_full/tti4025/第34轮出现1次选项差异：原索引1038与缓存索引1446的当前精确增益都为24.83565094716323B，约2.91e−11B历史舍入误差破坏原始索引平局规则。',
        '因此“受影响候选集合正确”还不等于“已实现逐步一致”。应缓存受影响UE的精确值并按原布局重新组装/归约，或对接近最大值及阈值的候选精确回退。',
        '以1e−6B邻域内候选的已记录新分数模拟付费回退，1952/1952轮索引一致；额外候选刷新1142个。这是日志内计数，不是新PHY实验，也不构成任意输入浮点误差上界证明。',
        '28.74%/34.02%是冷/热full在原轨迹上“候选重评数量”的保留比例，含每TTI首扫；未包含真实缓存维护、候选生成、内存复制等成本，不能倒数当墙钟提速。', '',
        '3. 同轮可并行的正Δ移动数',
        '正Δ以>1e−6B且单项满足日志floor定义；另外单列超过原1/16B接受阈值的子集。冲突规则是PRG相同或受影响UE相交，既覆盖同PRG干扰，也覆盖跨PRG整TB层/资源耦合。',
        '用整数集合打包最大化数量，限时未证最优者保留上下界，不把贪心数称最大值。构造性贪心保留原最大J移动为第一项；最大数量解则不保证包含该首项。',
        'arm | 第5轮后、有接受轮的正Δ最大数量 | 同样后段超过原阈值的最大数量下界 | 全部有接受轮的构造性贪心数量'] )
    for a in additional:
        if a['parallel_positive_after5_lower'] is not None:
            lines.append(f"{a['arm']} | {a['parallel_positive_after5_lower']:.2f}–{a['parallel_positive_after5_upper']:.2f} | "
                         f"{a['parallel_qualifying_after5_lower']:.2f} | {a['parallel_greedy_accepted']:.2f}")
    lines.extend(['结构独立保证当前模型下单时隙J增益可加；多个各自达floor的动作仍可能共同跌破floor。对保存输入的12状态，用资源数/最小层/TBS/队列精确计算ΔD，并增加 ΣΔD≥floor−D_root。该步骤没有调用PHY。',
        'arm | 保存输入的接受轮数 | 同子集结构最大数下界 | 加共同floor后的最大数下界–上界 | 保留原首项且逐次守floor的贪心数'] )
    for a in additional:
        lines.append(f"{a['arm']} | {a['saved_input_accepted_rounds']} | {a['saved_input_structural_lower']:.2f} | "
                     f"{a['saved_input_floor_lower']:.2f}–{a['saved_input_floor_upper']:.2f} | {a['saved_input_floor_greedy']:.2f}")
    lines.extend(['这支持多移动原型具有压缩长尾的空间，但不能把并行数量直接当轮数/墙钟加速倍数；一次提交多个移动会改变之后的贪心轨迹，仍须单独验收终态与服务。', '',
        '4. 被接受移动在廉价SIR排序下的名次',
        '主代理预先定义为 Σw·min(Q,TBS(count,min_layer))·log(1+已分配PRG平均trace-SIR)，SIR=||H_serving||F²/(4noise+Σ活动邻区||H_neighbor||F²)。只做标量比值和逻辑TB，不运行MMSE或BLER。',
        '报告的是在当前floor可行候选中的稳定名次（并列用原索引）；全部候选名次及次要单PRG SIR代理也保留在rounds.csv/summary.json。没有根据结果挑代理。',
        '主代理top16召回：cold_full0/130、warm_full1/109、warm_n3 0/12，合计1/251=0.40%。因此不能直接把这个SIR的top16作为精确验证短名单。',
        '这只否定本轮明确的廉价SIR代理，不是否定其他物理代理或学习排序。排序应提前为离线诊断任务，先提升精确最佳移动召回，再考虑上线截断。', '',
        '下一步顺序',
        '先做“跨轮依赖缓存＋原浮点平局/阈值保护”的逐步等价实现，和现有C++单候选内增量明确区分；以本批完整轨迹每轮选择及PHY值一致作为工程门。',
        '然后做守共同floor的多移动原型，保留单移动对照与终态服务验收。原C提前做召回对照，不直接启用当前SIR的硬top16。只继承t−1活动模式、UE重新选可作为单独初始化臂，90%活动重合不等于可冻结该模式。',
        '完成这些诊断/等价核验后，再安排2秒开发评估；本轮不启动该评估或任何新训练。', '',
        '补充口径：旧表只有warm_b160/warm_b212完全相同，cold两者仍略有提高；它说明额外前缀候选在那些状态没有改进所选结果，不单独证明排序是全部瓶颈。',
        '5轮59.09%来自warm_n3来源固定输入回放，24.25轮来自warm_full自己的原生轨迹，两者不能直接相减构成同轨迹长尾曲线。additional_summary.json另算了各full自身轨迹的前5轮增益保留。', '',
        f'原始轨迹：{native}', f'离线分析：{source}',
        '代码中封锁NativePhy.evaluate与ReferencePhy.evaluate，发生新评分会立即报错；completion记录新原生/候选PHY/MMSE均为0。',
        '输入SHA、全部轮级/转移级CSV、源码与测试一起归档，旧step0报告保留。'])
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    # A compact figure makes the two independent engineering opportunities visible.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms = ['cold_full', 'warm_full']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(arms))
    axes[0].bar(x-.18, [100*summary[a]['changed_fraction_candidate_weighted'] for a in arms], .36, label='Delta actually changed')
    axes[0].bar(x+.18, [100*summary[a]['lazy_refresh_ratio'] for a in arms], .36, label='Dependency refresh incl. first scan')
    axes[0].set(ylabel='Candidate fraction (%)', xticks=x, xticklabels=arms, ylim=(0,100))
    axes[0].legend(fontsize=8)
    late = [next(r for r in additional if r['arm'] == a) for a in arms]
    axes[1].bar(x, [r['parallel_positive_after5_lower'] for r in late], color=['C0','C1'])
    axes[1].set(ylabel='Independent positive moves after round 5', xticks=x, xticklabels=arms)
    axes[1].text(.5,.95,'Structural packing; shared floor checked only on saved-input subset', transform=axes[1].transAxes,
                 ha='center', va='top', fontsize=7, wrap=True)
    fig.suptitle('Logged transient seed41 diagnostics; no new PHY or closed-loop run', fontsize=11)
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(out/f'move_diagnostics.{extension}', dpi=160)
    plt.close(fig)
    write_json(out/'archive_manifest.json', {str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tests')
    main(parser.parse_args())

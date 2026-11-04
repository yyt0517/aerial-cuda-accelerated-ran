"""Small reproducible evidence archive for native DPLS traces and replay."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from .physics import ROOT
from .experiment import write_json, sha


def csv_rows(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    source = Path(args.source).resolve()
    replay = Path(args.replay).resolve() if args.replay else None
    out = Path(args.destination).resolve()
    completed = json.loads((source/'completion.json').read_text())
    if not completed['completed']:
        raise ValueError('Native run did not complete')
    out.mkdir(parents=True, exist_ok=False)
    for name in ('protocol.json', 'completion.json', 'source_manifest.json'):
        shutil.copy2(source/name, out/name)
    decisions, rounds, checks, summaries = [], [], [], []
    manifest = {}
    for folder in sorted(source.glob('*_seed*')):
        if not folder.is_dir():
            continue
        summary = json.loads((folder/'summary.json').read_text())
        summaries.append(summary)
        native = json.loads((folder/'validation.json').read_text())
        checks.append(dict(arm=summary['arm'], seed=summary['seed'], **native))
        for line in (folder/'decisions.jsonl').read_text().splitlines():
            record = json.loads(line)
            if not record['measured']:
                continue
            identity = {k: record[k] for k in ('arm', 'seed', 'tti')}
            rounds.extend(dict(identity, **row) for row in record['history'])
            decisions.append({k: v for k, v in record.items() if k != 'history'})
        for path in [folder/'launch.json', folder/'decisions.jsonl', folder/'summary.json',
                     folder/'validation.json', *folder.glob('traces/*.npz'), *folder.glob('states/*.pt')]:
            manifest[str(path)] = sha(path)
    csv_rows(out/'decisions.csv', decisions)
    csv_rows(out/'rounds.csv', rounds)
    write_json(out/'native_checks.json', checks)
    write_json(out/'summaries.json', summaries)
    source_names = ('ranked_dpls.py', 'physics.py', 'physics_kernel.cpp', 'search.py', 'no_ctbs_inputs.py',
                    'dpls_trace.py', 'anytime_dpls_campaign.py', 'test_anytime_dpls.py')
    for name in source_names:
        original = source/'source/training/stageb_lns'/name
        target = out/'source'/name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(original, target)
    shutil.copy2(Path(__file__).with_name('anytime_dpls_README.rst'), out/'README.rst')
    shutil.copy2(__file__, out/'source'/Path(__file__).name)
    if args.tests:
        shutil.copy2(args.tests, out/'tests.txt')
    failures = []
    for name in args.failed_runs:
        folder = Path(name).resolve()
        failure = dict(path=str(folder), completed=False)
        if (folder/'failure.json').exists():
            failure.update(json.loads((folder/'failure.json').read_text()))
        failure['native_checks'] = {str(p.relative_to(folder)): json.loads(p.read_text())
                                    for p in folder.glob('*_seed*/validation.json')}
        failures.append(failure)
    write_json(out/'retained_failures.json', failures)
    pairs = []
    if replay:
        result = json.loads((replay/'completion.json').read_text())
        if not result['completed']:
            raise ValueError('Replay did not complete')
        rows = [json.loads(s) for s in (replay/'replay.jsonl').read_text().splitlines()]
        by_state = {}
        for row in rows:
            by_state.setdefault(row['source'], {}).setdefault(row['arm'], []).append(row)
        for state, arms in by_state.items():
            reference = arms['cold_full']
            for arm, repeats in arms.items():
                if len({r['score'] for r in repeats}) != 1:
                    raise ValueError('Repeated deterministic decisions disagree')
                row = repeats[0]
                baseline = reference[0]
                full_gain = baseline['score']-baseline['initial_score']
                pairs.append(dict(arm=arm, seed=row['seed'], tti=row['tti'], source=state,
                    predecessor_arm=row['previous_action_source_arm'],
                    score_ratio=row['score']/baseline['score'] if baseline['score'] else 1.,
                    gain_retention=(row['score']-row['initial_score'])/full_gain if full_gain > 1.e-9 else 1.,
                    phy_ratio=row['candidate_evaluations']/baseline['candidate_evaluations'],
                    time_ratio=float(np.median([r['e2e_ms'] for r in repeats])/
                                     np.median([r['e2e_ms'] for r in reference])),
                    candidate_evaluations=row['candidate_evaluations'], rounds=row['rounds']))
        csv_rows(out/'paired_replay.csv', pairs)
        for name in ('protocol.json', 'completion.json', 'input_manifest.json'):
            shutil.copy2(replay/name, out/f'replay_{name}')
        manifest[str(replay/'replay.jsonl')] = sha(replay/'replay.jsonl')
    write_json(out/'external_artifact_manifest.json', manifest)
    lines = ['DPLS第0步插桩与因果热启动物理对照：工程实现及短闭环核验', '2026-11-03', '',
        '本轮完成日志、可截断冷/热启动、总PHY预算限制及原生采集/固定输入回放入口。未训练新模型。',
        '协议：无CTBS、因果缺口权重、canonical原生BLER；不是等权服务PPO协议，也不与旧legacy BLER完整DPLS计时混排。',
        '现有C++核已经有受影响UE整TB重算和MMSE缓存。候选内MCS固定，跨TTI重新构造PHY；不能把已有增量评分当成本轮新贡献。', '',
        '热启动只读取本臂实际执行的t−1动作，按当前合法性修正，精评后与当前原始PFQ择优。两次初始化精评都收费。',
        '支持cold/warm_n1,n2,n3,n5与到收敛；cold/warm_b106,b160,b212总预算包括初始化。部分扫描不称收敛。',
        '日志保留所有候选ΔJ、可行性、动作、被接受索引、每轮计数与墙钟；NPZ使用单PRG压缩表示。', '',
        '实际执行的短闭环（逐臂独立重放同一PFQ预热；均未设稳定化、成熟cohort随访不足）：',
        'arm | seed | 测量TTI | 轮数均值 | PHY候选均值 | 槽位重合% | 活动分配重合% | 活动模式重合%']
    for row in summaries:
        s = row['search']
        means = [s[k]['mean'] for k in ('rounds', 'candidate_evaluations', 'slot_agreement',
                                      'active_assignment_jaccard', 'activity_agreement')]
        lines.append(f"{row['arm']} | {row['seed']} | {s['decisions']} | {means[0]:.2f} | {means[1]:.2f} | "
                     f'{100*means[2]:.2f} | {100*means[3]:.2f} | {100*means[4]:.2f}')
    lines.extend(['', '重合率来自连续实际执行动作，首个搜索TTI的前驱是PFQ预热动作。含切换瞬态，不能据此判定稳定态重合率或A的继续门槛。',
        '三种重合率分别报告；活动模式稳定不代表UE分配稳定。没有以稀疏归档状态冒充相邻TTI。',
        'native_checks.json记录逐TTI原生动作/MCS/层/TBS/出队核验；采样终态另以独立NumPy PHY核验。共同PFQ前缀与外生到达另作精确检查。'])
    if pairs:
        # Equal state/seed weights; first reduce each state's serial repetitions.
        arm_summary = []
        for arm in dict.fromkeys(p['arm'] for p in pairs):
            subset = [p for p in pairs if p['arm'] == arm]
            item = dict(arm=arm)
            for key in ('score_ratio', 'gain_retention', 'phy_ratio', 'time_ratio'):
                item[key] = float(np.mean([np.mean([p[key] for p in subset if p['seed'] == seed])
                                          for seed in sorted({p['seed'] for p in subset})]))
            arm_summary.append(item)
        csv_rows(out/'replay_summary.csv', arm_summary)
        lines.extend(['', '固定输入回放（每状态串行交替计时取中位，再seed内/seed间等权）：',
                      'arm | J/full% | 收敛增益保留% | PHY/full% | 墙钟/full%'])
        for row in arm_summary:
            lines.append(row['arm']+' | '+' | '.join(f'{100*row[k]:.2f}' for k in
                ('score_ratio', 'gain_retention', 'phy_ratio', 'time_ratio')))
        lines.append('回放读取所标明source arm实际t−1动作；不是所有热启动臂自己的闭环历史，亦不证明包级服务质量。')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for arm in ('cold', 'warm'):
            selected = [r for r in arm_summary if r['arm'].startswith(arm+'_n')]
            selected.sort(key=lambda r: r['phy_ratio'])
            for ax, key, ylabel in zip(axes, ('gain_retention', 'score_ratio'),
                                      ('Converged gain retained', 'J / full DPLS')):
                ax.plot([r['phy_ratio'] for r in selected], [r[key] for r in selected], 'o-', label=arm)
                ax.set(xlabel='Complete PHY candidate evaluations / full', ylabel=ylabel)
                ax.grid(alpha=.25)
                ax.legend()
        fig.suptitle('Fixed-input diagnostic; source-arm predecessor; no packet-quality gate')
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            fig.savefig(out/f'frontier.{extension}', dpi=160)
        plt.close(fig)
    lines.extend(['', '当前尚未完成：41–43各2秒的第0步采集；44–45各2秒、含稳定化与成熟cohort的A前沿及21状态J/服务检查；统一质量/弱UE/成本门槛。',
        '所有短闭环/回放数字仅用于工程诊断，不称3–8倍加速或已经达到A的≤40%PHY门槛，不进入新seed确认。',
        '计时分解见summaries.json：native_kernel_ms、validation_ms、output_copy_ms、observer_ms。其余时间含NumPy、分配、Python和包装开销，不称纯Python占比。',
        '数据复制与日志压缩写盘分别计时，带日志原生墙钟不作为公平部署速度结论。', '',
        '下一门：先完成A的同协议闭环质量与成本前沿；E超过固定轮数/固定Δ阈值后才保留；F使用本臂t−1条件；B先扫低维时延系数。',
        '服务PPO不再续训；repair_guard不扫置信度、不扩网络。C/D保留为条件启动项。',
        '同池物理贪心≈DPLS支持“现有候选池即时收益有限”，测得的PPO+0.23%不是所有可学习策略收益的理论上界。', '',
        '联调失败版本按原样保留，见retained_failures.json。v1在更新新实验launch元数据时用了只准新建的写接口；v2在原生轨迹完成后未筛掉horizon后的到达记录，精确外生流量检查报shape不一致。',
        '两处脚本问题已修正；本报告只使用重新运行并完整通过检查的v3，不把v2的原生局部通过替代整轮完成。',
        f'完整数据：{source}', f'固定输入回放：{replay}' if replay else '固定输入回放：未提供',
        '大体积逐候选NPZ和原生数据保留在deliverables，SHA见external_artifact_manifest.json；本目录仅保存小型可审查证据。'])
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    write_json(out/'gates.json', dict(engineering_passed=all(c['passed'] for c in checks),
        full_step0_complete=False, A_quality_gate_evaluated=False, A_cost_gate_evaluated=False,
        weak_ue_gate_evaluated=False, learned_component_added=False, registry_changed=False))
    write_json(out/'next_protocol.json', dict(
        baseline='freeze same no-CTBS causal-deficit canonical-BLER full DPLS; no cross-protocol ratios',
        development=dict(seeds=[44, 45], measured_ttis=4000, mature_cohort=True,
                         historical_snapshots=21, role='exposed development only'),
        confirmation=dict(seeds='five fresh seeds avoiding exposed41--48; freeze once before outcomes',
                          measured_seconds=10, scenario='boundary/rate0.8 plus a higher-load robustness arm'),
        quality=dict(payload_paired_mean_min_ratio=.997, payload_each_seed_min_ratio=.994,
                     expiry_max_increase_pp=.10, success_mean_delay_max_increase_ms=.5,
                     success_p95_delay_max_increase_ms=1., weak_ue_payload_p5_min_ratio=.97,
                     worst_ue_expiry_limit='max(full worst +2 percentage points, 5%)',
                     newly_persistently_unserved_ues=0),
        final_cost=dict(phy_mean_max_ratio=.25, phy_p95_max_ratio=.40,
                        wall_mean_max_ratio=.40, wall_quantiles=['p50', 'p95', 'maximum']),
        A=dict(continue_if='all development quality gates and PHY mean <=40%',
               controls='cold/warm rounds1,2,3,5/full plus equal candidate budgets106/160/212'),
        E=dict(gain_retention_min=.97, rounds_reduction_vs_best_heuristic_min=.20,
               controls=['fixed rounds', 'fixed delta threshold'], worst_tti_tail_fraction=.05),
        F=dict(arms=['teacher conditions', 'own executed t-1 conditions', 'no conditions'],
               continue_if='J/service passing count >=half teacher arm and clearly exceeds no-condition arm',
               predecessor='true consecutive TTI, not previous sparse saved state; teacher work charged separately'),
        B=dict(initial_urgency_values=5, low_dimensional_parameters='2--6', training_seeds=[41,42,43],
               continue_if='payload/expiry not worse and P95 <=full+1ms'),
        C=dict(start_only_if='A/E quality or <=25% PHY target still unmet', k=16,
               best_action_recall_min=.90, learned_recall_advantage_min_pp=5),
        D=dict(start_only_if='A/E quality or cost gap persists',
               repair_round_reduction_vs_t_minus_1_min=.30),
        freezes=['no service PPO continuation', 'no repair_guard confidence sweep or larger network'],
        status='predeclared next experiments; none of these scientific gates certified by the short smoke'))
    write_json(out/'archive_manifest.json', {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--replay')
    parser.add_argument('--destination', required=True)
    parser.add_argument('--tests')
    parser.add_argument('--failed-runs', nargs='*', default=[])
    main(parser.parse_args())

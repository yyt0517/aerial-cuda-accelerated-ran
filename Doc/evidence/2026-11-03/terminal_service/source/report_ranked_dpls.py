"""Small, reproducible evidence archive for the CTBS-free DPLS ranking pilot.

Only completed source artifacts are read. The source experiment is never
modified, and independent contributions are compared at the same top-K.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SMALL_ARTIFACTS = (
    'protocol.json', 'source_manifest.json', 'input_manifest.json',
    'evaluation_input_manifest.json', 'runtime_environment.json',
    'training_round0.json', 'training_round1.json',
    'collection_round0.json', 'collection_round1.json', 'selection.json',
    'tests.log', 'measurements.csv', 'per_state.csv', 'summary.csv',
    'evaluation_checks.json', 'completion.json', 'preflight.json',
)
NEW_SOURCES = (
    'no_ctbs_inputs.py', 'test_no_ctbs_inputs.py',
    'edit_ranker.py', 'test_edit_ranker.py', 'ranked_dpls.py',
    'test_ranked_dpls.py', 'ranked_dpls_campaign.py', 'report_ranked_dpls.py',
    'edit_features.cpp', 'build_edit_ranker.py', 'ranked_dpls_README.rst',
)


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def kind(arm):
    label = str(arm).lower()
    if label.startswith('full'):
        return 'full'
    if label.startswith('heur'):
        return 'heuristic'
    if label.startswith('ai'):
        return 'ai'
    return label


def seed_equal(frame, column):
    return float(frame.groupby('seed')[column].mean().mean())


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def completed_metadata(source):
    completion = read_json(source/'completion.json', {})
    if not (completion.get('completed') is True or completion.get('passed') is True
            or completion.get('status') in ('complete', 'completed')):
        raise ValueError('Only a completed experiment may be summarized')
    return completion


def selected_checkpoint(selection, source):
    """Find only a selected checkpoint, never an arbitrary training candidate."""
    for key in ('selected', 'selected_checkpoint', 'checkpoint', 'model'):
        value = selection.get(key)
        if isinstance(value, dict):
            found = selected_checkpoint(value, source)
            if found is not None:
                return found
        if isinstance(value, str) and value.endswith('.npz'):
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = source/candidate
            if candidate.exists():
                return candidate
    return None


def zero_residual(selection, source):
    path = selected_checkpoint(selection, source)
    if path is None:
        value = selection.get('zero_residual')
        return dict(verified=False, zero_residual=value if isinstance(value, bool) else None)
    with np.load(path, allow_pickle=False) as weights:
        if not {'w3', 'b3'}.issubset(weights.files):
            return dict(verified=False, zero_residual=None, checkpoint=str(path))
        zero = (not np.any(weights['w3']) and not np.any(weights['b3']))
        if 'residual_scale' in weights and float(weights['residual_scale']) == 0:
            zero = True
    return dict(verified=True, zero_residual=bool(zero), checkpoint=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def comparison_tables(per):
    needed = {'state_id', 'seed', 'arm', 'score', 'e2e_ms'}
    if not needed.issubset(per.columns):
        raise ValueError(f'Missing per-state columns: {sorted(needed-set(per.columns))}')
    per = per.copy()
    if 'top_k' not in per:
        per['top_k'] = 0
    per['top_k'] = per['top_k'].fillna(0).astype(int)
    per['kind'] = per.arm.map(kind)
    full = per[per.kind == 'full'].copy()
    if full.empty:
        raise ValueError('A full-neighborhood control is required')
    # A full scan does not depend on K. Multiple repeated full labels may be
    # present; their numerical result must agree before costs are aggregated.
    if (full.groupby('state_id').score.agg(lambda x: x.max()-x.min()) > 1.e-6).any():
        raise ValueError('Duplicate full controls disagree about J')
    baseline = full.groupby('state_id').median(numeric_only=True)
    results, paired, policy_groups = [], [], {}
    optional = ('candidate_evaluations', 'ue_evaluations', 'mmse_misses',
                'fallback_fraction', 'rounds', 'feature_ms',
                'inference_and_sort_ms', 'final_scan_candidates')
    for (arm, top_k), frame in per.groupby(['arm', 'top_k'], sort=True):
        if frame.state_id.duplicated().any():
            raise ValueError(f'Duplicate per-state rows for {arm}, K={top_k}')
        group = frame.set_index('state_id')
        if set(group.index) != set(baseline.index):
            raise ValueError(f'Unpaired state set for {arm}, K={top_k}')
        reference = baseline.loc[group.index]
        delta = group.score-reference.score
        relative = group.score/reference.score-1
        cost_relative = group.e2e_ms/reference.e2e_ms-1
        item = dict(arm=str(arm), kind=kind(arm), top_k=int(top_k), states=len(group),
            seeds=int(group.seed.nunique()), score=seed_equal(group, 'score'),
            mean_ms=seed_equal(group, 'e2e_ms'),
            p95_state_median_ms=float(group.e2e_ms.quantile(.95)),
            mean_relative_j_pct=float(relative.groupby(group.seed).mean().mean()*100),
            worst_relative_j_pct=float(relative.min()*100),
            mean_paired_cost_change_pct=float(cost_relative.groupby(group.seed).mean().mean()*100),
            ratio_of_mean_cost_change_pct=(seed_equal(group, 'e2e_ms')/seed_equal(reference, 'e2e_ms')-1)*100,
            wins=int((delta > 1).sum()), losses=int((delta < -1).sum()))
        for column in optional:
            if column in group:
                item[column] = finite_number(seed_equal(group, column))
        results.append(item)
        policy_groups[(kind(arm), int(top_k), str(arm))] = group
        for state in group.index:
            paired.append(dict(state_id=state, seed=int(group.loc[state, 'seed']),
                arm=str(arm), top_k=int(top_k), j_delta=float(delta.loc[state]),
                j_relative_pct=float(relative.loc[state]*100),
                ms_delta=float(group.loc[state, 'e2e_ms']-reference.loc[state, 'e2e_ms']),
                ms_relative_pct=float(cost_relative.loc[state]*100)))
    learned_contribution = []
    for (family, top_k, arm), group in policy_groups.items():
        if family != 'ai':
            continue
        controls = [(name, control) for (other, k, name), control in policy_groups.items()
                    if other == 'heuristic' and k == top_k]
        for control_name, control in controls:
            control = control.loc[group.index]
            delta = group.score-control.score
            rel = group.score/control.score-1
            learned_contribution.append(dict(ai_arm=arm, heuristic_arm=control_name,
                top_k=top_k, states=len(group), seeds=int(group.seed.nunique()),
                mean_j_delta=seed_equal(group, 'score')-seed_equal(control, 'score'),
                mean_relative_j_pct=float(rel.groupby(group.seed).mean().mean()*100),
                worst_relative_j_pct=float(rel.min()*100),
                mean_ms_delta=seed_equal(group, 'e2e_ms')-seed_equal(control, 'e2e_ms'),
                ratio_of_mean_cost_change_pct=(seed_equal(group, 'e2e_ms')/seed_equal(control, 'e2e_ms')-1)*100,
                wins=int((delta > 1).sum()), losses=int((delta < -1).sum()),
                ties=int((np.abs(delta) <= 1).sum())))
    return pd.DataFrame(results), pd.DataFrame(paired), pd.DataFrame(learned_contribution)


def plot_results(summary, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'full': '#36454f', 'heuristic': '#c57c20', 'ai': '#2166ac'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = [f"{r.arm}"+(f" / K{r.top_k}" if r.kind != 'full' else '')
              for r in summary.itertuples()]
    for index, row in enumerate(summary.itertuples()):
        color = colors.get(row.kind, '#777777')
        axes[0].scatter(row.mean_ms, row.score, s=55, color=color)
        axes[0].annotate(labels[index], (row.mean_ms, row.score), xytext=(5, 7 if index % 2 else -14),
                         textcoords='offset points', fontsize=8)
    axes[0].set_xlabel('Complete saved-state decision time (ms)')
    axes[0].set_ylabel('Weighted expected bytes J')
    axes[0].margins(x=.25, y=.25)
    axes[0].grid(alpha=.2)
    values = summary.mean_ms.to_numpy()
    axes[1].barh(np.arange(len(summary)), values,
                 color=[colors.get(k, '#777777') for k in summary.kind])
    axes[1].set_yticks(np.arange(len(summary)), labels)
    axes[1].set_xlabel('Complete saved-state decision time (ms)')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=.2)
    fig.tight_layout()
    fig.savefig(destination/'quality_cost.png', dpi=170)
    fig.savefig(destination/'quality_cost.pdf')
    plt.close(fig)


def freeze_source_copies(source, destination):
    code = destination/'code'
    code.mkdir(exist_ok=True)
    manifest = read_json(source/'source_manifest.json', {})
    copied = []
    for name in NEW_SOURCES:
        relative = Path('training/stageb_lns')/name
        candidates = (source/'source'/relative, source/'frozen_source'/relative,
                      source/'source'/name)
        frozen = next((p for p in candidates if p.exists()), None)
        if frozen is None:
            live = ROOT/relative
            expected = manifest.get(str(relative), manifest.get(str(live)))
            if isinstance(expected, dict):
                expected = expected.get('sha256')
            if live.exists() and expected == hashlib.sha256(live.read_bytes()).hexdigest():
                frozen = live
        if frozen is not None:
            shutil.copy2(frozen, code/name)
            copied.append(name)
    return copied


def main(args):
    source, destination = Path(args.source).resolve(), Path(args.destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError('The evidence destination must be outside the source experiment')
    completion = completed_metadata(source)
    protocol = read_json(source/'protocol.json', {})
    checks = read_json(source/'evaluation_checks.json', {})
    selection = read_json(source/'selection.json', {})
    destination.mkdir(parents=True, exist_ok=True)
    copied_artifacts = []
    for name in SMALL_ARTIFACTS:
        path = source/name
        if path.exists():
            shutil.copy2(path, destination/name)
            copied_artifacts.append(name)
    per = pd.read_csv(source/'per_state.csv')
    summary, paired, contribution = comparison_tables(per)
    summary.to_csv(destination/'recomputed_summary.csv', index=False)
    paired.to_csv(destination/'paired_vs_full.csv', index=False)
    contribution.to_csv(destination/'ai_vs_heuristic.csv', index=False)
    numeric = per.select_dtypes(include='number').columns.difference(['seed', 'top_k'])
    keys = ['seed', 'arm']+(['top_k'] if 'top_k' in per else [])
    per.groupby(keys)[numeric].mean().reset_index().to_csv(destination/'per_seed.csv', index=False)
    zero = zero_residual(selection, source)
    (destination/'selected_model_diagnostics.json').write_text(json.dumps(zero, indent=2, allow_nan=False)+'\n')
    plot_results(summary, destination)
    rows = []
    for row in summary.itertuples():
        label = row.arm+(f'/K{row.top_k}' if row.kind != 'full' else '')
        candidates = getattr(row, 'candidate_evaluations', np.nan)
        rows.append(f'{label:20s} | {row.score:11.3f} | {row.mean_ms:9.3f} | '
                    f'{row.p95_state_median_ms:9.3f} | {candidates:10.1f} | '
                    f'{row.mean_relative_j_pct:+8.3f}% | {row.worst_relative_j_pct:+8.3f}% | '
                    f'{row.ratio_of_mean_cost_change_pct:+8.3f}%')
    independent = []
    for row in contribution.itertuples():
        independent.append(f'{row.ai_arm} − {row.heuristic_arm}, K={row.top_k}: '
            f'J均值差{row.mean_j_delta:+.3f}，逐状态相对J均值{row.mean_relative_j_pct:+.3f}%；'
            f'完整耗时均值差{row.mean_ms_delta:+.3f}ms，均值之比{row.ratio_of_mean_cost_change_pct:+.3f}%；'
            f'{row.wins}胜/{row.losses}负/{row.ties}平。')
    zero_text = ('选定值头的输出残差为零，已读取权重核验。AI臂此时沿用物理启发式排序；'
                 '不能把相同结果或网络执行称为学习增益。' if zero.get('zero_residual') is True
                 else '选定模型是否为零残差见selected_model_diagnostics.json；非零权重本身也不能证明独立收益。')
    engineering = '通过' if checks.get('passed') is True else '未确认通过'
    scientific = '通过' if checks.get('scientific_gate_passed') is True else '未通过'
    audits = checks.get('independent_audits', [])
    audit_count = len(audits) if isinstance(audits, list) else audits
    repetition = (int(pd.read_csv(source/'measurements.csv').repetition.nunique())
                  if (source/'measurements.csv').exists() else None)
    report = f'''无CTBS：AI候选排序加速DPLS第一轮实验
日期：{protocol.get('date', '见protocol.json')}

结论：完整结果、模型选型和核验已完成归档。工程核验：{engineering}；预声明科学门槛：{scientific}。
本报告只陈述保存状态上的质量与完整决策耗时，不宣称已获得场景泛化、原生吞吐/TTL/包时延或部署实时性改善。

算法及去CTBS范围：
三条核心路径为完整邻域DPLS、物理启发式top-K、AI残差排序top-K。若附加多个K，分别保留，不混池计算学习贡献。
原始PFQ仅解码，不做CTBS剪裁、累计TBS headroom或需求上限掩码；教师、训练、推理和所有对照共用该原始草案。
每状态下界为ceil(0.90×本臂当前原始PFQ总出队字节)。保持因果服务缺口权重，未读取其他策略轨迹作为下界。
所有正队列服务UE保留；UE/cellUE需求掩码不用于裁剪搜索域。仅物理关联及原生prgMsk资源可用性约束继续生效。
归档载体名ctbs只保存历史pre-action物理诊断；计时输入适配使用中性raw_phy，通过通用原生CPU OLLA/MCS规则重建。
状态生成历史及训练/选型/评估角色以protocol.json、input_manifest.json为准；历史开发暴露不会因本轮拆分而变成盲测。

物理及停止口径：
三臂共享canonical原生BLER倒序查表及C++受影响UE物理核，保持MCS3非单调行语义。历史legacy评分不可混作同一加速基线。
每轮固定一个incumbent，最多接受一次编辑；前两轮增益阈值1字节，之后16字节。
top-K未发现可接受改善时，仅评分未评估的原邻域补集，不重复计算短名单。
最终完整邻域检查只确认满足原停止条件；不同候选路径可以收敛到不同局部解，不保证与full逐动作等价。

配对完整搜索结果：
每个状态先取重复计时中位数，再在seed内及seed间等权。J为加权期望成功字节，不是Mbps。
本轮状态数{per.state_id.nunique()}，独立重复单位为seed（{per.seed.nunique()}个）；同seed状态/UE/候选与{repetition}次计时重复不是独立无线场景。
P95为这些状态中位耗时的95%分位，不是部署延迟或调度时限保证。
arm/K                |           J | 完整均时ms | 状态P95ms | 精确候选 | 相对full J | 最差状态J | 均时变化
{chr(10).join(rows)}

AI独立贡献（仅同K、同状态、同物理/缓存/回退实现比较）：
{chr(10).join(independent) if independent else '没有可配对的同K启发式控制，不得宣称已隔离AI贡献。'}
{zero_text}
即便某臂快于full，也需与物理启发式对照，避免把相同筛选机制、缓存或工程优化归为AI收益。

计时与工作量：
完整耗时必须包含状态数值重建、原始PFQ、PHY/特征缓存建立、候选生成、模型推理和排序、精确验证、回退/最终全扫及销毁。
模型加载、归档读取与不可变TBS表首次建立可在外，具体计时边界以冻结protocol为准。教师标签查询/训练另计，不属于免费在线能力。
候选数、受影响UE、MMSE miss、回退比例、feature_ms、inference_and_sort_ms均为诊断；候选减少比例不能直接当成耗时减少比例。
paired_vs_full.csv与ai_vs_heuristic.csv从相同state_id重新计算；均值之比与逐状态比率的均值分别记录，不混用。

选型、核验与限制：
模型和K必须在读取本轮评估结果前冻结；选型原文保存在selection.json，不按本报告结果重新挑选。
独立终态审计数：{audit_count}。完整检查及科学门槛保持evaluation_checks.json原始含义，completion.json保持执行完成的含义。
有限状态质量门槛与闭环业务非劣结论分开。仍需新原生实验验证完整包payload、TTL、成功平均/P95、UE P5及退化UE数量。
本轮不能证明输入已达上限、继续价值无效或更大网络必然有效；负结果不删除，不把暴露状态重新命名为未见最终集。

完整本地实验：{source}
重建入口：python -m training.stageb_lns.report_ranked_dpls --source <完成的实验目录> --destination <独立证据目录>
源清单可能包含未归档的模型/数据/源码；archive_manifest.json仅覆盖本目录小型归档。
'''
    (destination/'report.txt').write_text(report)
    sources = freeze_source_copies(source, destination)
    (destination/'README.rst').write_text('''CTBS-free DPLS ranking evidence
==============================

Read report.txt, protocol.json and evaluation_checks.json together. The archive
contains small paired results and frozen new-source snapshots, not all models,
teacher data or native trajectories. Original full experiment artifacts remain
at the source path in report.txt. Historical development states are not blind
deployment evidence. Source manifests refer to the original experiment;
archive_manifest.json covers this small archive. Re-run the report module into
an independent destination; source artifacts are read-only.
''')
    (destination/'report_provenance.json').write_text(json.dumps(dict(
        source=str(source), copied_artifacts=copied_artifacts, copied_new_sources=sources,
        full_control_pairing='state_id; duplicate full-K labels require equal J',
        ai_contribution_pairing='same state_id and same top_k only',
        source_completed=completion), indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    manifest = {str(path.relative_to(destination)):dict(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
        for path in sorted(destination.rglob('*')) if path.is_file()
        and path.name != 'archive_manifest.json' and '__pycache__' not in path.parts}
    (destination/'archive_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', required=True)
    main(parser.parse_args())

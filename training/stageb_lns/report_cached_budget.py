"""Wall-clock, cap4, terminal service and realized dirty-budget regret evidence."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from .analyze_dpls_moves import write_json,write_csv,sha


def read(path):
    return list(csv.DictReader(Path(path).open()))


def load(path):
    return json.loads(Path(path).read_text())


def copy_run(source,target):
    target.mkdir()
    for p in source.iterdir():
        if p.is_file() and p.suffix in ('.json','.csv','.npz'):
            shutil.copy2(p,target/p.name)
    if (source/'source').exists():
        (target/'source').mkdir()
        for p in (source/'source').iterdir():
            if p.suffix in ('.py','.cpp'):shutil.copy2(p,target/'source'/p.name)


def main(args):
    initial,quality,wall,previous,reference=map(lambda s:Path(s).resolve(),
        (args.initial_wall,args.quality,args.wall,args.previous_wall,args.reference))
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    for name,p in [('initial_wall',initial),('quality',quality),('wall',wall),('wall_before_prefix_optimization',previous)]:
        if not load(p/'completion.json')['completed']:raise ValueError('Incomplete input run')
        copy_run(p,out/name)
    shutil.copy2(__file__,out/Path(__file__).name)
    if args.tests:shutil.copy2(args.tests,out/'tests.txt')
    metrics=read(quality/'metrics.csv');regret=read(quality/'regret.csv');rounds=read(quality/'rounds.csv')
    times={r['arm']:r for r in read(wall/'summary.csv')}
    ref=read(reference/'decisions.csv')
    baseline_candidates=sum(int(r['full_candidate_evaluations']) for r in ref)
    summaries=[]
    for arm in ('cached','cap4','M16','M32','M64'):
        rows=[r for r in metrics if r['arm']==arm]
        accepted=[r for r in regret if r['arm']==arm and r['accepted']=='True']
        retention=np.array([float(r['gain_retention']) for r in accepted])
        regular=[r for r in rounds if r['arm']==arm and int(r['iteration'])>1 and int(r['moves'])>0 and int(r['fallback_candidates'])==0]
        online=sum(int(r['candidate_evaluations']) for r in rows)
        fallback=sum(int(r['fallback_candidates']) for r in rows)
        summary=dict(arm=arm,states=len(rows),J_mean_ratio=float(np.mean([float(r['J_ratio']) for r in rows])),
            J_min_ratio=min(float(r['J_ratio']) for r in rows),J_pass_count=sum(float(r['J_ratio'])>=.995 for r in rows),
            snapshot_joint_pass_count=sum(r['snapshot_quality_service_passed']=='True' for r in rows),
            newly_unallocated_state_ue_pairs=sum(int(r['newly_unallocated']) for r in rows),
            newly_ineffective_state_ue_pairs=sum(int(r['newly_ineffective']) for r in rows),
            worst_positive_P5_ratio=min(float(r['positive_p5_ratio']) for r in rows),
            worst_weak_sum_ratio=min(float(r['positive_weak_ratio']) for r in rows),
            rounds=sum(int(r['rounds']) for r in rows),online_candidates=online,
            candidate_ratio=online/baseline_candidates,fallback_candidates=fallback,
            fallback_fraction=fallback/online,
            first_scan_candidate_fraction=sum(int(r['first_scan_candidates']) for r in rows)/online,
            mean_regular_fresh_candidates=float(np.mean([int(r['scored']) for r in regular])),
            mean_all_fresh_candidates_per_round=online/sum(int(r['rounds']) for r in rows),
            gain_retention_mean=float(retention.mean()),gain_retention_p05=float(np.quantile(retention,.05)),
            gain_retention_min=float(retention.min()),
            weighted_gain_retention=sum(float(r['selected_delta']) for r in accepted)/sum(float(r['true_maximum_delta']) for r in accepted),
            regret_mean_bytes=float(np.mean([float(r['regret_bytes']) for r in accepted])),
            wall_mean_ms=float(times[arm]['mean_ms']),wall_p95_ms=float(times[arm]['p95_ms']),
            wall_ratio=float(times[arm]['ratio_of_means']),
            first_scan_profiled_time_fraction=float(times[arm]['mean_first_scan_profiled_time_fraction']),
            first_plus_preparation_profiled_time_fraction=float(times[arm]['mean_first_plus_preparation_profiled_time_fraction']))
        summaries.append(summary)
    write_json(out/'summary.json',summaries);write_csv(out/'summary.csv',summaries)
    # Correct the v1 schema's name, retaining its raw metadata for provenance.
    schema=load(quality/'label_schema.json')
    shutil.copy2(quality/'label_schema.json',out/'quality/label_schema_raw.json')
    corrected=dict(schema)
    corrected.pop('independent_states',None)
    corrected.update(captured_inputs=12,independent_states_not_established=True,
                     note='12 input rows include correlated/shared-warmup histories; not12 independent wireless scenes')
    (out/'quality/label_schema.json').unlink()
    write_json(out/'quality/label_schema.json',corrected)
    write_json(out/'quality/metadata_correction.json',dict(field='independent_states',
        correction='v1 generator used12 as captured input count; rename captured_inputs, independence not established',
        numeric_results_or_labels_changed=False,raw_metadata='label_schema_raw.json'))
    quant=read(quality/'quantization.csv')
    initial_s=load(initial/'summary.json')
    first_profile=read(initial/'profiles.csv')
    cached_p=[r for r in first_profile if r['mode']=='cached']
    profiling=dict(generation_ms=float(np.mean([float(r['generation_ms']) for r in cached_p])),
        reassembly_ms=float(np.mean([float(r['reassembly_ms']) for r in cached_p])),
        score_ms=float(np.mean([float(r['score_ms']) for r in cached_p])),
        e2e_ms=float(np.mean([float(r['e2e_ms']) for r in cached_p])))
    write_json(out/'profiling_summary.json',profiling)
    ledger={name:load(p/'completion.json') for name,p in [('initial_wall',initial),('quality',quality),('wall_v1',previous),('wall_v2',wall)]}
    write_json(out/'work_ledger.json',ledger)
    write_json(out/'gates.json',dict(engineering_tests_passed=True,quality_decisions_all_converged=True,
        all_timing_outputs_and_work_match_frozen_quality=True,native_closed_loop_started=False,
        advanced_arm_promoted=False,quantized_incremental_version_implemented=False,
        learned_ranker_trained=False,registry_changed=False,
        limitation='Only captured-state J/expected-service gate; no mature packet/native weak-UE or new-seed conclusion'))
    text=['精确缓存墙钟、cap4与脏候选M16/32/64：实际轨迹与质量代价',
        '2026-11-03；仅已有12个保存输入，均seed41、相关且含共享预热历史；不新增原生闭环',
        '协议：无CTBS、因果缺口权重、canonical原生BLER；原始PFQ出队90%下界，现行浮点规范平局。', '',
        '结论：精确缓存保持原终态，正式六臂计时墙钟为full的67.31%，没有达到40%墙钟目标。cap4虽降到33.82%，但快照J/服务只过4/12；M16/32/64只过1/4/1，均不晋级。',
        '本轮实现并测量了cap4与M轨迹；没有训练模型，没有实施新的整数量化目标或整数增量评分基线。', '',
        '1. 无日志、无审计墙钟',
        '同一进程、CPU255固定亲和、单BLAS线程；每输入每臂5次串行旋转/交替，先取状态中位数再对12状态等权。计时内关闭逐轮记录、profiling、observer与审计；源码/库加载和不可变TBS首次构造在外。',
        '计时含当前元数据到MCS/TBS/PFQ重建、新PHY上下文、搜索和释放。所有定时动作/PHY在计时结束后才核对，全部等于冻结质量轨迹。正常PHY合法性/有限值校验仍是算法成本。',
        'arm | 墙钟均值ms | 状态中位数P95 ms | 墙钟/full% | 在线PHY/full% | 终态J均比/最差比 | 快照J/服务通过数']
    for arm in ('full','cached','cap4','M16','M32','M64'):
        t=times[arm]
        if arm=='full':
            text.append(f"full | {float(t['mean_ms']):.3f} | {float(t['p95_ms']):.3f} | 100.00 | 100.00 | 1/1 | 12/12（参照）")
        else:
            r=next(r for r in summaries if r['arm']==arm)
            text.append(f"{arm} | {r['wall_mean_ms']:.3f} | {r['wall_p95_ms']:.3f} | {100*r['wall_ratio']:.2f} | "
                        f"{100*r['candidate_ratio']:.2f} | {r['J_mean_ratio']:.6f}/{r['J_min_ratio']:.6f} | {r['snapshot_joint_pass_count']}/12")
    text.extend(['', f"两臂初测另保存于initial_wall：full{initial_s['full_mean_ms']:.3f}ms、cached{initial_s['cached_mean_ms']:.3f}ms，均值之比{100*initial_s['ratio_of_means']:.2f}%。正式六臂以wall/v2的同批full作参照，不跨会话混算速度。",
        f"初测的独立profile中，缓存版候选生成约{profiling['generation_ms']:.2f}ms，完整布局复制{profiling['reassembly_ms']:.2f}ms、归约/选择{profiling['score_ms']:.2f}ms。后两项合计约{profiling['reassembly_ms']+profiling['score_ms']:.2f}ms，确实吃掉了评分节省；候选生成同样是大项。",
        'M版第一次六臂计时为约228–231ms：为取前M仍完整遍历全部ε分组。现改成在前M生成后停止，严格等于原完整排序前缀；配置、动作/PHY、轮数及评分计数不变。旧计时与旧源码完整保存在wall_before_prefix_optimization。',
        '当前M版约105–113ms，仍不能从约12%候选评分推导约12%墙钟；候选重建、全布局复制/归约及多轮搜索仍在。没有部署实时性或原生包时延结论。', '',
        '2. cap=4：按并集刷新、守共同floor、联合精验',
        '每轮先保留规范规则最大J移动，再按旧规范ε顺序挑最多3个不同PRG、受影响UE不相交的可接受移动，逐次检查合并后总出队floor。一次接受多于1个时再做一次联合PHY评分，计入在线成本。',
        '下一轮按全部已接受移动的依赖并集失效；所有候选仍使用现行浮点目标与规范平局规则。末轮必须完整确认无超过原1/16B阈值的可行编辑。',
        'cap4平均J不差不代表服务不差：12状态累计新增未分配(state,UE)对14个，最差positive-P5比为0、弱服务和比为0.65297；J/服务联合只有4/12通过。',
        '其在线评分为完整基线约24.82%，墙钟约33.82%；速度部分有收益，但完整质量/弱UE门槛没有通过，不进入2秒或新seed确认。', '',
        '3. 脏候选M：实际运行到经确认的局部收敛',
        '每TTI首扫完整；已知clean沿用精确缓存，novel全部精评，dirty-known只评旧精确ΔJ前M。排序不读取当前教师输出，也不使用免费的当前floor标签；只对已精验候选知道当前可行性。',
        '未评的脏条目跨轮持续无效，直到实际重评；旧标签可能多于一轮。短名单没有可接受增益时，补评剩余脏候选再决定是否停止，不能用短名单停冒充收敛。',
        '悔恨审计只在决策结束后由独立PHY补评当时未评分动作，标签不回流策略。比值R=所选ΔJ/真实最大ΔJ，另报绝对悔恨ΔJ_max−ΔJ_selected。以下仅接受轮；停止轮单列检查，没有假收敛。',
        'arm | R均值/P5/最小值 | 按ΔJ_max加权R | 普通接受轮新精评均数 | 含首扫/回退的每轮均数 | 回退占在线评分%'] )
    for r in summaries:
        if r['arm'].startswith('M'):
            text.append(f"{r['arm']} | {r['gain_retention_mean']:.4f}/{r['gain_retention_p05']:.4f}/{r['gain_retention_min']:.4f} | "
                        f"{r['weighted_gain_retention']:.4f} | {r['mean_regular_fresh_candidates']:.2f} | "
                        f"{r['mean_all_fresh_candidates_per_round']:.2f} | {100*r['fallback_fraction']:.2f}")
    text.extend(['普通轮确实约20/36/68个新精评，但经完整确认收敛后，全程每轮约176–191个；首扫和回退不能免费。M16与M32总评分接近，回退次数/轨迹抵消了较小M的部分节省。',
        'J均值可高于原局部最优，但最差状态均下降约1.6–2.1%，且弱UE失败更严重；不按均值或这12状态事后挑出“最好M”。',
        '快照服务门：J>=full99.5%，不得新增当前未分配UE；在full期望服务>1B的UE内，positive-P5和最弱20%服务和各>=98%。这些只是单状态期望字节，不是成熟包payload/过期/P95或长期未服务结论。', '',
        '4. 首扫固定成本：按前两档完成后的各实际分支分别报告',
        '首扫时间来自另外一次关闭审计、开启轻量diagnostics的分解运行；不是主计时内偷偷开启profile。首轮时间含该轮选择/接受；“含准备”还包括MCS/PFQ/PHY上下文与初始化。',
        'arm | 首扫占在线候选% | 首轮占profile墙钟% | 准备+首轮占profile墙钟%'] )
    for arm,t in times.items():
        text.append(f"{arm} | {100*float(t['first_scan_candidate_fraction']):.2f} | "
                    f"{100*float(t['mean_first_scan_profiled_time_fraction']):.2f} | {100*float(t['mean_first_plus_preparation_profiled_time_fraction']):.2f}")
    text.extend(['M的首扫已占约25–27%在线评分，但当前只占约8%墙钟；全邻域构建/复制/排序等共同开销还没有消掉。因此“首扫是固定成本”成立，但暂非当前墙钟的最大部分。', '',
        '5. 整数增量与第一项学习组件的边界',
        f"只用精确缓存本来算出的值做了量化敏感度检查：每UE的w·E按1e−6B四舍五入至int64再相加，在{len(quant)}个已访问根上选择差异{sum(r['choice_changed']=='True' for r in quant)}次，最大总J误差{max(float(r['max_absolute_J_quantization_error']) for r in quant):.9g}B。没有额外PHY查询。",
        '整数加法可在新量化目标内实现严格增量，但并不自动精确等于旧浮点目标；误差可超过当前1e−6B平局带。仍需一版明确定义舍入、溢出边界和阈值单位的新规范基线及完整等价门。本轮尚未实现整数增量版，不把敏感度检查称为等价通过或性能收益。',
        '小M确有质量损失，学习任务因此明确为“旧ΔJ＋受影响UE变化→当前ΔJ”。已保存161700条精确缓存新算出的dirty-known标签，8维特征见quality/label_schema.json，额外标签PHY成本为0。未训练或选择模型。',
        '这些免费标签来自精确缓存访问的轨迹；若补标M自身访问轨迹上的未评候选，仍须像本轮悔恨审计一样另付PHY，不能称免费。',
        '全部标签来自12个相关保存输入、1个已暴露seed，含共享预热/轨迹重复；不能把161700条标签当独立场景。原v1元数据曾将输入行数命名independent_states，归档保留raw并显式订正为captured_inputs，数值与标签未改。',
        '学习排序首先要胜过旧ΔJ的留出悔恨/终态J/服务曲线；即使排序改善，也不能把总出队floor视作弱UE服务保护。本轮cap4/M不晋级，原有效注册表不变。', '',
        '账本与复现',
        'work_ledger.json分列初测、质量运行、悔恨审计、优化前后墙钟和额外profile的实际新评分；离线曲线/标签不读取新原生轨迹。质量审计次数不混入在线评分或计时。',
        f"质量运行在线评分{ledger['quality']['online_candidate_evaluations']}次，事后悔恨审计{ledger['quality']['audit_candidate_evaluations']}次；另有{ledger['quality']['numpy_terminal_audits']}次NumPy终态核验。",
        '各工作目录source/及source_manifest.json对应其实际版本，不用后来优化代码冒充原质量数据的源码；定时版本逐项核对冻结动作/PHY、轮数和评分量。',
        '原生闭环、2秒开发、新seed确认和生产部署均未启动。'])
    (out/'report.txt').write_text('\n'.join(text)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m=[next(r for r in summaries if r['arm']==a) for a in ('M16','M32','M64')]
    fig,axes=plt.subplots(1,3,figsize=(13,4.2))
    x=np.array([16,32,64])
    axes[0].plot(x,[r['gain_retention_mean'] for r in m],'o-',label='Mean delta retained')
    axes[0].plot(x,[r['gain_retention_p05'] for r in m],'o--',label='Worst5% threshold')
    axes[0].set(xlabel='Dirty-known shortlist M',ylabel='Selected delta / true maximum',ylim=(0,1.05));axes[0].legend(fontsize=8)
    axes[1].plot(x,[r['J_mean_ratio'] for r in m],'o-',label='Mean terminal J')
    axes[1].plot(x,[r['J_min_ratio'] for r in m],'o--',label='Worst terminal J')
    axes[1].axhline(.995,color='grey',linestyle=':');axes[1].set(xlabel='M',ylabel='Terminal J / full');axes[1].legend(fontsize=8)
    axes[2].plot(x,[r['candidate_ratio'] for r in m],'o-',label='Online PHY / full')
    axes[2].plot(x,[r['wall_ratio'] for r in m],'o-',label='Unlogged wall / full')
    axes[2].set(xlabel='M',ylabel='Cost ratio');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.25)
    fig.suptitle('12 exposed correlated saved inputs; shared floor; service joint gate passes M16/32/64:1/4/1 of12',fontsize=10)
    fig.tight_layout()
    for extension in ('png','pdf'):fig.savefig(out/f'dirty_budget_curve.{extension}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.3))
    a=['cached','cap4','M16','M32','M64'];z=np.arange(len(a))
    axes[0].bar(z,[100*float(times[t]['ratio_of_means']) for t in a]);axes[0].axhline(40,color='grey',linestyle=':')
    axes[0].set(xticks=z,xticklabels=a,ylabel='Unlogged wall / full (%)')
    axes[1].bar(z-.18,[100*float(times[t]['first_scan_candidate_fraction']) for t in a],.36,label='First-scan PHY share')
    axes[1].bar(z+.18,[100*float(times[t]['mean_first_scan_profiled_time_fraction']) for t in a],.36,label='First-round profiled wall share')
    axes[1].set(xticks=z,xticklabels=a,ylabel='Fraction (%)');axes[1].legend(fontsize=8)
    fig.suptitle('Serial paired timing; no audits/history/profiling inside primary timed decisions',fontsize=10)
    fig.tight_layout()
    for extension in ('png','pdf'):fig.savefig(out/f'wall_and_first_scan.{extension}',dpi=160)
    plt.close(fig)
    write_json(out/'archive_manifest.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--initial-wall',required=True);p.add_argument('--quality',required=True)
    p.add_argument('--wall',required=True);p.add_argument('--previous-wall',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--output',required=True);p.add_argument('--tests');main(p.parse_args())

"""Archive completed calibration, integer equivalence, wall and native evidence.

No scheduler, simulator, PHY evaluator, fitting or candidate selection is run.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from .physics import ROOT
from .analyze_dpls_moves import sha,write_json,write_csv


def main(args):
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    roots={
        'calibration':ROOT/'deliverables/full_service_calibration_20261103_v1',
        'envelope':ROOT/'deliverables/full_service_envelope_20261103_v2',
        'integer_gate':ROOT/'deliverables/persistent_integer_gate_20261103_v1',
        'wall':ROOT/'deliverables/persistent_wall_20261103_v1',
        'native':ROOT/'deliverables/persistent_native_development_20261103_v1',
        'native_analysis':ROOT/'deliverables/persistent_native_analysis_20261103_v2',
    }
    source_checks=0;input_checks=0
    for name,root in roots.items():
        if not json.loads((root/'completion.json').read_text())['completed']:raise AssertionError(name)
        for manifest,kind in (('source_manifest.json','source'),('input_manifest.json','input')):
            if not (root/manifest).exists():continue
            for filename,digest in json.loads((root/manifest).read_text()).items():
                file=Path(filename)
                if not file.is_absolute():file=ROOT/file
                if str(file).startswith('/opt/nvidia/cuBB/'):
                    file=ROOT/file.relative_to('/opt/nvidia/cuBB')
                if sha(file)!=digest:raise AssertionError(f'{name}:{filename} hash changed')
                if kind=='source':source_checks+=1
                else:input_checks+=1
    for name,root in roots.items():
        if name=='native':continue
        shutil.copytree(root,out/name)
    native=roots['native'];dest=out/'native';dest.mkdir()
    for f in native.iterdir():
        if f.is_file():shutil.copy2(f,dest/f.name)
    shutil.copytree(native/'source',dest/'source')
    native_inputs={};native_steps=0;cohort_rows=0;per_ue_checks=0
    for seed in (44,45):
        for mode in ('full','guarded4'):
            d=native/f'{mode}_seed{seed}';a=dest/d.name;a.mkdir()
            for name in ('summary.json','validation.json','reward_validation.json','per_ue.csv',
                         'search.csv','launch.json','cohort.csv.gz'):
                shutil.copy2(d/name,a/name)
            for f in d.iterdir():
                if f.is_file():native_inputs[str(f.relative_to(ROOT))]=sha(f)
            val=json.loads((d/'validation.json').read_text())
            assert val['passed'] and val['sources_unchanged']
            assert val['exact_native_actions']==val['fresh_input_checks']==val['exact_mcs_layer_tbs_dequeue']==12400
            native_steps+=val['exact_native_actions']
            launch=json.loads((d/'launch.json').read_text())
            for filename,digest in launch['source_sha256'].items():
                assert sha(ROOT/Path(filename).relative_to('/opt/nvidia/cuBB'))==digest
            assert sha(ROOT/Path(launch['command'][0]).relative_to('/opt/nvidia/cuBB'))==launch['simulator_sha256']
            assert launch['current_teacher_calls']==0 and not launch['ctbs_action_clipping']
            # Recompute archived scientific metrics without the online adapter
            # or its metrics()/packets() functions. The frozen binary schema is
            # 19 little-endian doubles: tti, UE, ... successful_bytes at index10.
            c=pd.read_csv(a/'cohort.csv.gz');s=json.loads((a/'summary.json').read_text())
            assert not c.censored.any() and c.arrival_tti.between(8001,12000).all()
            success=c[c.successfully_delivered];expired=c.outcome.eq('expired')
            ue_payload=np.bincount(success.ue_id.to_numpy(),weights=success.packet_size_bytes.to_numpy(),minlength=36)*8/2e6
            ue_expiry=100*np.bincount(c.loc[expired,'ue_id'],minlength=36)/np.bincount(c.ue_id,minlength=36)
            np.testing.assert_allclose([ue_payload.sum(),100*expired.mean(),success.system_time_ms.mean(),
                success.system_time_ms.quantile(.95),np.quantile(ue_payload,.05),ue_expiry.max()],
                [s[k] for k in ('complete_payload_mbps','expiry_pct','success_delay_mean_ms',
                               'success_delay_p95_ms','ue_complete_payload_p5_mbps','worst_ue_expiry_pct')],rtol=0,atol=1.e-9)
            per=pd.read_csv(a/'per_ue.csv').sort_values('ue')
            np.testing.assert_allclose(per.payload_mbps,ue_payload,rtol=0,atol=1.e-9)
            np.testing.assert_allclose(per.expiry_pct,ue_expiry,rtol=0,atol=1.e-9)
            data=np.memmap(d/'metrics.bin',dtype='<f8',mode='r',shape=(12400,36,19))
            np.testing.assert_array_equal(data[:,0,0],np.arange(1,12401))
            np.testing.assert_array_equal(data[0,:,1],np.arange(36))
            flags=data[8000:12000,:,10]>0;longest=np.zeros(36,np.int64);run=longest.copy()
            for t in flags:
                run=np.where(t,0,run+1);longest=np.maximum(longest,run)
            np.testing.assert_array_equal(per.longest_no_successful_byte_ttis,longest)
            np.testing.assert_allclose(data[8000:12000,:,10].sum()*8/2e6,s['goodput_mbps'],rtol=0,atol=1.e-9)
            search=pd.read_csv(a/'search.csv');m=search[search.tti.between(8001,12000)]
            assert len(m)==4000
            np.testing.assert_allclose([m.candidates.mean(),m.candidates.quantile(.95)],
                [s['mean_candidate_evaluations'],s['p95_candidate_evaluations']],rtol=0,atol=1.e-9)
            cohort_rows+=len(c);per_ue_checks+=len(per)
    write_json(out/'native_raw_manifest.json',native_inputs)
    assert native_steps==49600

    pairs=pd.read_csv(roots['calibration']/'pairs.csv')
    distinct=pd.read_csv(roots['envelope']/'distinct_full.csv')
    reps={(int(r.state),r.variant) for r in distinct.itertuples()}
    keep=[(int(r.state),r.actual) in reps and (int(r.state),r.reference) in reps for r in pairs.itertuples()]
    unique=pairs[keep];unique.to_csv(out/'distinct_full_pairs.csv',index=False)
    cal=dict(raw_cold_warm_pairs=24,raw_cold_warm_passes=6,
        raw_all_pairs=len(pairs),raw_old_gate_passes=int(pairs.old_gate_passed.sum()),
        distinct_patterns=len(distinct),distinct_directed_pairs=len(unique),
        distinct_old_gate_passes=int(unique.old_gate_passed.sum()),
        distinct_J_close_pairs=int(unique.J_close.sum()),
        distinct_J_close_service_passes=int(unique[unique.J_close].service_passed.sum()))
    write_json(out/'calibration_interpretation.json',cal)
    gate=pd.read_csv(roots['integer_gate']/'metrics.csv')
    aggregate=[]
    full_evals=int(gate[gate['mode']=='full'].candidate_evaluations.sum())
    for mode,g in gate.groupby('mode',sort=False):
        aggregate.append(dict(mode=mode,old_single_passes=int(g.old_single_gate_passed.sum()),
            envelope_passes=int(g.empirical_envelope_passed.sum()),rounds=int(g.rounds.sum()),
            candidate_evaluations=int(g.candidate_evaluations.sum()),PHY_ratio=float(g.candidate_evaluations.sum()/full_evals),
            generated_edits=int(g.generated_edits.sum()),accepted_batches=int(g.accepted_batches.sum()),
            guard_rejections=int(g.guard_rejections.sum()),minimum_old_J_ratio=float(g.old_J_ratio.min())))
    write_csv(out/'snapshot_summary.csv',aggregate)
    summary=pd.read_csv(roots['wall']/'summary.csv').set_index('arm')
    profile=pd.read_csv(roots['wall']/'profiles.csv').groupby('arm').mean(numeric_only=True)
    profile.to_csv(out/'profile_summary.csv')
    cache=profile.loc['cached']
    bound=dict(cache_profile_ms=float(cache.e2e_ms),cache_PHY_ms=float(cache.sparse_phy_ms),
        remaining_non_PHY_ms=float(cache.e2e_ms-cache.sparse_phy_ms),
        deleting_all_PHY_further_speedup_upper_bound=float(cache.e2e_ms/(cache.e2e_ms-cache.sparse_phy_ms)),
        first_scan_wall_fraction_mean=float(cache.first_scan_fraction),
        interpretation='very loose Amdahl bound: unrealistically deletes all PHY including mandatory first scan; excludes ranking/verification overhead; not an M forecast',
        M_paused=True,learning_paused=True)
    write_json(out/'ranking_upper_bound.json',bound)

    labels=ROOT/'deliverables/cached_budget_quality_20261103_v1/exact_dirty_labels.npz'
    previous=json.loads((ROOT/'Doc/evidence/2026-11-03/cached_cost_budget/archive_manifest.json').read_text())
    assert sha(labels)==previous['quality/exact_dirty_labels.npz']
    registry=ROOT/'training/stageb_hgraph/current_algorithm.json'
    review=json.loads((ROOT/'Doc/evidence/2026-11-03/ai_scheduling_review/recommended_implementation.json').read_text())
    # The historical implementation review stores its original registry hash.
    def find_registry(obj):
        if isinstance(obj,dict):
            if obj.get('path','').endswith('current_algorithm.json') and 'sha256' in obj:return obj['sha256']
            for k,v in obj.items():
                if k.endswith('current_algorithm.json') and isinstance(v,str) and len(v)==64:return v
                result=find_registry(v)
                if result:return result
        if isinstance(obj,list):
            for v in obj:
                result=find_registry(v)
                if result:return result
        return None
    registry_before=find_registry(review)
    if not registry_before:raise AssertionError('Historical registry digest not located')
    assert sha(registry)==registry_before
    shutil.copy2(args.tests,out/'tests.txt')
    assert 'Ran 20 tests' in Path(args.tests).read_text() and '\nOK\n' in Path(args.tests).read_text()
    source=out/'source';source.mkdir()
    for f in Path(__file__).parent.iterdir():
        if f.name.startswith(('persistent','service_gate_calibration','calibrated_envelope','benchmark_persistent',
                               'build_persistent','summarize_persistent','report_persistent','test_persistent')):
            if f.is_file():shutil.copy2(f,source/f.name)
    shutil.copy2(Path(__file__).with_name('persistent_README.rst'),out/'README.rst')
    write_json(out/'source_manifest.json',{str(f.relative_to(ROOT)):sha(f) for f in Path(__file__).parent.iterdir()
        if f.name in {a.name for a in source.iterdir()}})
    dependencies=set()
    for seed in (44,45):
        for mode in ('full','guarded4'):
            launch=json.loads((native/f'{mode}_seed{seed}'/'launch.json').read_text())
            dependencies.update(str(Path(f).relative_to('/opt/nvidia/cuBB'))
                for f in launch['source_sha256'] if f.endswith('.py'))
    dependencies.update('training/stageb_lns/'+name for name in (
        'physics.py','physics_kernel.cpp','_physics.so','archive.py','no_ctbs_inputs.py',
        'dpls_operations.py','ranked_dpls.py','cached_dpls.py','cached_budget_campaign.py','analyze_dpls_moves.py'))
    for filename in sorted(dependencies):
        target=out/'dependencies'/filename;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/filename,target)
    write_json(out/'dependency_manifest.json',{f:sha(ROOT/f) for f in sorted(dependencies)})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    arms=['float_python_full','integer_python_full','full','cached','guarded4']
    names=['Float Python full','Integer Python full','Native full','Exact cache','Guarded cap4']
    colors=['#adb5bd','#adb5bd','#495057','#087f8c','#d17a22']
    axes[0].barh(names,summary.loc[arms,'mean_ms'],color=colors)
    axes[0].invert_yaxis();axes[0].set_xlabel('Serial wall time (ms), 12 saved inputs')
    for i,a in enumerate(arms):axes[0].text(summary.loc[a,'mean_ms']+1,i,f'{summary.loc[a,"mean_ms"]:.2f}',va='center',fontsize=9)
    axes[0].set_xlim(0,210)
    sub=['full','cached','guarded4'];x=np.arange(3)
    snap={r['mode']:r for r in aggregate}
    axes[1].bar(x-.17,[snap[a]['PHY_ratio'] for a in sub],.34,label='Snapshot PHY / native full',color='#087f8c')
    axes[1].bar(x+.17,summary.loc[sub,'ratio_native_full'],.34,label='Serial wall / native full',color='#d17a22')
    axes[1].set_xticks(x,['Native full','Exact cache','Guarded cap4']);axes[1].set_ylim(0,1.13)
    axes[1].set_ylabel('Remaining fraction');axes[1].legend(fontsize=8,loc='upper right')
    fig.savefig(out/'wall_and_work.png',dpi=180);fig.savefig(out/'wall_and_work.pdf');plt.close(fig)

    paired=pd.read_csv(roots['native_analysis']/'paired.csv')
    ng=json.loads((roots['native_analysis']/'gates.json').read_text())
    guard=[r for r in aggregate if r['mode']=='guarded4'][0]
    report=f'''持久编辑表精确缓存、full服务门校准与带守卫cap4
2026-11-03；12个相关保存输入为seed41开发状态；44/45为已暴露开发seed。

结论
精确缓存已完成架构重构与新整数规范等价门。公平串行墙钟：同架构native full {summary.loc['full','mean_ms']:.2f}ms，cache {summary.loc['cached','mean_ms']:.2f}ms（{100*summary.loc['cached','ratio_native_full']:.2f}%）；不是原先预测的35–40%。原Python浮点full {summary.loc['float_python_full','mean_ms']:.2f}ms，cache为其{100*summary.loc['cached','ratio_float_python_full']:.2f}%，这包含编译/候选架构收益，不能全部归于缓存或AI。
带服务守卫cap4通过12/12经验包络筛选，按事前条件启动两个seed的2秒原生闭环；seed44的UE24过期率12.0554%→14.8253%（+2.7699pp）超过+2pp门，seed45通过。因此本配置不晋级、不改默认；M和学习修正器继续暂停。

1. full自己的服务门校准
在同12状态上付费重跑8个full变体：source canonical、冷/热canonical、两组固定随机平局冷/热、source打乱枚举。全部完整收敛；随机只在max锚定1e-6B平局带内选择。source打乱枚举逐位等于source canonical，证明规范键消除了枚举行号依赖。
旧单解门：冷/热互比6/24通过；六变体全配对194/360通过。随机平局大多得到重复终态，不能把这些配对当独立样本。按逐UE COUNT和量化1e-6B的E去重，只有22种服务模式，每状态1–3种；不同模式有向互比0/22通过，J差≤0.5%的0/4也未通过。旧门衡量相对某一局部解的服务一致性，失败不能直接等同长期弱UE损失；同时也不能证明这些差异无害。
原始逐变体leave-one-out为72/72，因重复终态虚高；去重后只有4/19，3个单模式状态无法做留一，12状态的全队列P5均为0，P5指标在此不提供区分力。6变体不足以估计概率分布/置信区间。
因此在新缓存/cap4评估前冻结v2描述性经验包络：J≥full变体最小J的99.5%，当前正队列UE未分配数≤full最大值、有效服务数≥full最小值、全队列P5和因果缺口最高20%UE服务和≥full最小值的98%。它是逐指标经验范围，并非一个真实full终态，更非公平性证书；只用于进入开发闭环。旧单解门仍并列保留，没有把历史失败改写为成功。原始v1分位数诊断一并保留，正式筛选以envelope/protocol.json v2为准。

2. 持久编辑表与新整数规范
候选为跨轮持久的单PRG编辑操作，规范键编码PRG和每小区overwrite（-2不改，-1清空）。只重建已提交移动涉及的PRG；其它PRG保留编辑表，按受影响UE依赖并集标脏。每候选只保存受影响UE的稀疏PHY结果、int64 ΔJ和整数ΔD；评分时改一个scratch根的PRG并恢复，不复制完整方案，不按全UE重归约。每轮仍扫描O(候选数)元数据指针。
q_u=nearbyint((E_u*w_u)*1e6)，nearest-even逐UE量化后int64求和；候选分数=根分数+受影响UE贡献差之和。阈值前两轮严格>1e6、以后严格>16e6；全局最大整数ΔJ锚定1单位平局带，再取规范键最小。整数队列、正权重、sum ceil(queue*w*1e6)+1<INT64_MAX/4、UE+PRG位掩码/规范键边界均检查，超出契约失败。
新库_persistent.so使用独立Context并复用未改动physics_kernel.cpp；原_physics.so不变。审计observer才构建完整候选值矩阵，正式墙钟和原生运行关闭observer/日志/history/profiling。
独立Python完整方案/全整数归约参考付费重跑。native full和cache各378轮624670个候选全部8字段PHY、ΔJ、ΔD、根、选择逐位一致：总756轮1249340个候选比较。12个终态恰与旧浮点终态相同，但新量化目标仍单独版本化，不宣称对任意浮点输入普遍等价。
full/cache在线评分624690/183079（29.31%）；持久表各累计生成55951个编辑，避免每轮重造624670个完整候选。cap4用全部移动依赖并集更新，不乘缓存与轮数节省。48终态NumPy独立核验、批量根额外499次精确审计另计费。

3. cap4守卫与快照结果
每批最多4个不同PRG、受影响UE不相交的正Δ移动，逐个检查共同PFQ90%出队floor。批次相对同根最佳单移动一旦新增未分配UE，或任一因果缺口最高20%正队列UE的期望服务下降>1e-6B，就整批退回最佳单移动。守卫只合并已评分稀疏payload，无额外PHY；不读当前教师终态。
未守卫cap4：旧单解门4/12、经验包络7/12；137轮、154939次评分、101个批次。
守卫cap4：旧单解门{guard['old_single_passes']}/12、经验包络{guard['envelope_passes']}/12；{guard['rounds']}轮、{guard['candidate_evaluations']}次评分、{guard['accepted_batches']}个批次、{guard['guard_rejections']}次批次回退，最差终态J/full={guard['minimum_old_J_ratio']:.8f}。
公平墙钟guarded4 {summary.loc['guarded4','mean_ms']:.2f}ms，full的{100*summary.loc['guarded4','ratio_native_full']:.2f}%；它未比精确cache进一步加速，保守守卫和批次选择成本抵消了评分节省。

4. 公平墙钟、首扫与排序的收益上限
同12输入、CPU255单进程/单BLAS线程，各臂5次旋转/反转交替计时，先取每状态中位数再等权；不含外部审计/不可变导入/静态表和库首次加载，含每次因果问题/PFQ构建、Context和搜索/销毁。共享主机计时，不作为部署时延。另做36次独立profile。
原浮点Python full {summary.loc['float_python_full','mean_ms']:.2f}ms；整数Python full {summary.loc['integer_python_full','mean_ms']:.2f}ms；同持久架构native full {summary.loc['full','mean_ms']:.2f}ms；cache {summary.loc['cached','mean_ms']:.2f}ms；cap4 {summary.loc['cap4','mean_ms']:.2f}ms；guarded4 {summary.loc['guarded4','mean_ms']:.2f}ms。C++完整基线也享受相同候选/稀疏架构，必须同时报告。
cache独立profile：表生成/失效{cache.table_generation_ms:.3f}ms、PHY {cache.sparse_phy_ms:.3f}ms、选择{cache.selection_guard_ms:.3f}ms、完整{cache.e2e_ms:.3f}ms；PHY重新成为主要开销。每TTI首扫平均占墙钟{100*cache.first_scan_fraction:.2f}%，guarded为{100*profile.loc['guarded4','first_scan_fraction']:.2f}%，不能沿用旧Python的8%。首扫时间是上述总时间的子集，不额外相加。
暂不运行M/训练。把cache的所有PHY（包括必须的首扫）都凭空删除，仍有{bound['remaining_non_PHY_ms']:.3f}ms，进一步加速的极宽松Amdahl上限约{bound['deleting_all_PHY_further_speedup_upper_bound']:.2f}倍；真实排序必须保留首扫/精验并支付排序成本，会明显更小。此数不是M收益预测，既有M轨迹不能移植成本后当成已执行结果。161700条标签SHA保持不变。

5. 两seed、2秒成熟cohort原生闭环
配置在结果前冻结：boundary/rate0.8、无CTBS裁剪、因果缺口权重、两臂同整数目标、冷raw PFQ初始化；当前教师调用0。使用已有连续信道仿真器/原生FIFO，不修改物理/业务。NativeEnv仍导出历史教师审计字段，策略不使用；闭环总进程耗时不当作策略速度。
44/45 × native full/guarded4共4轨迹，每条4000TTI共同PFQ预热+4000TTI稳定+4000TTI（2s）测量+400TTI到期追踪，合计49600步；另2条16步工程smoke共32步。两种子均为历史暴露开发输入，不是盲测。4轨迹并发，公平墙钟只引用独立串行回放。
逐步动作/MCS/layer/TBS/出队/新鲜输入全部核验；4000步共同PFQ前缀与既有输入/动作逐位一致，外部offered packets和配对测量cohort的UE/packet/arrival/size/TTL严格一致。按8001–12000到达包统计并追踪成熟，pending=0。
seed | payload比 | 过期差pp | 平均成功时延差ms | P95差ms | UE payload P5比 | PHY均值比/P95比
'''
    for r in paired.itertuples():
        report+=f'{r.seed} | {r.payload_ratio:.8f} | {r.expiry_delta_pp:+.6f} | {r.mean_delay_delta_ms:+.6f} | {r.p95_delay_delta_ms:+.3f} | {r.ue_payload_P5_ratio:.8f} | {r.mean_PHY_ratio:.6f}/{r.P95_PHY_ratio:.6f}\n'
    report+=f'''
两seed平均payload比{ng['payload_paired_mean_ratio']:.8f}；无新增零payload UE或400TTI连续无成功字节UE。seed44 UE24的完整包payload 29.748→28.980Mbps（-2.58%），过期率+2.7699pp失败；当步E服务守卫不能保证动态队列/包截止期。seed45最差UE过期率9.3642%→6.2280%。质量检查1/2 seed通过，总判定失败；不得平均掉seed44。共同成功包的配对时延另列，不能将不同成功包集合的均值变化直接称逐包改善。
PHY均值约35%/34%仍高于25%目标；P95约34%通过40%目标。缓存独立相对同架构full的墙钟46.6%仍未达到更强40%对照目标；对旧Python full的工程大幅降本单列，不据此宣称缓存/AI独立达标。没有启动最终5新seed×10秒确认，没有重新选择阈值/种子，没有更改有效算法注册表。

6. 保留与下一步
保留精确cache作为新的强物理研究对照。cap4守卫保留为已运行负结果，不晋级；若继续此分支，需另行冻结能反映跨时隙服务债务/TTL的保护协议，然后再验证，不能根据44/45结果调门后称原门通过。M和学习修正器继续暂停，标签只留档。先基于新profile明确需要保留的首扫/收敛精验成本，再决定排序是否值得实施。
20项定向测试通过；测试包含守卫接受/拒绝、相对单移动的服务约束及无额外PHY、整数ties/overflow、独立全候选等价、原PHY回归。校准、等价、计时/剖析、原生工作分别计费，work_ledger.json不将审计算免费。完整输入/源码/原生大文件SHA见各manifest；archive_manifest.json覆盖本归档。
'''
    (out/'report.txt').write_text(report)
    completions={name:json.loads((root/'completion.json').read_text()) for name,root in roots.items()}
    native_search=sum(int(pd.read_csv(native/f'{mode}_seed{seed}'/'search.csv').candidates.sum())
        for mode in ('full','guarded4') for seed in (44,45))
    write_json(out/'work_ledger.json',dict(calibration=completions['calibration'],integer_gate=completions['integer_gate'],
        benchmark=completions['wall'],native=dict(runs=4,steps=49600,warmup_per_run=4000,
            policy_decisions_per_run=8400,policy_candidate_evaluations=native_search),
        smoke=json.loads((ROOT/'deliverables/persistent_native_smoke_20261103_v1/completion.json').read_text()),
        envelope_and_report_new_PHY=0,training_steps=0,new_M_decisions=0))
    write_json(out/'final_audit.json',dict(completed=True,tests=20,source_hash_checks=source_checks,input_hash_checks=input_checks,
        integer_equivalence_passed=True,native_exact_checked_steps=native_steps,labels_sha256=sha(labels),
        independently_recomputed_cohort_rows=cohort_rows,independently_recomputed_per_ue_rows=per_ue_checks,
        independently_recomputed_payload_expiry_delay_gap_and_cost=True,
        default_registry_sha256=sha(registry),default_registry_matches_historical_review=True,
        calibration_deduplicated=True,full_probability_distribution_claimed=False,
        native_quality_passed=ng['all_native_quality_passed'],native_PHY_cost_passed=ng['all_native_PHY_cost_passed'],
        matched_architecture_cache_wall_ratio=float(summary.loc['cached','ratio_native_full']),
        promotion=False,M_paused=True,learning_paused=True))
    write_json(out/'completion.json',dict(completed=True,new_PHY_evaluations=0,new_native_runs=0,
        scope='archive/analysis only; scientific quality and cost results separate from completion'))
    write_json(out/'archive_manifest.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--tests',default='/tmp/persistent_reconstruction_20261103_tests.txt');main(p.parse_args())

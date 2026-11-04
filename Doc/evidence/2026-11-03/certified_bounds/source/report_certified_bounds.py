"""Archive the completed zero-PHY certificate diagnostic; no scoring or fit."""
import argparse
import json
from pathlib import Path
import shutil
import pandas as pd
from .physics import ROOT
from .analyze_dpls_moves import sha,write_json


def main(args):
    source=ROOT/'deliverables/certified_bounds_offline_20261103_v4'
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    completion=json.loads((source/'completion.json').read_text())
    assert completion['completed'] and completion['new_candidate_PHY']==completion['new_MMSE']==0
    for f in source.iterdir():
        if f.is_dir():shutil.copytree(f,out/f.name)
        else:shutil.copy2(f,out/f.name)
    for name in ('input_manifest.json','source_manifest.json'):
        for file,digest in json.loads((source/name).read_text()).items():
            path=Path(file);path=path if path.is_absolute() else ROOT/path
            assert sha(path)==digest
    tests=Path(args.tests).read_text();assert 'Ran 7 tests' in tests and '\nOK\n' in tests
    (out/'tests.txt').write_text(tests)
    for name in ('certified_bounds.py','analyze_certified_bounds.py','test_certified_bounds.py',
                 'certified_bounds_README.rst','report_certified_bounds.py'):
        shutil.copy2(Path(__file__).with_name(name),out/'source'/name)
    shutil.copy2(Path(__file__).with_name('certified_bounds_README.rst'),out/'README.rst')
    logs=out/'canonical_logs';logs.mkdir()
    canonical=ROOT/'deliverables/canonical_cached_dpls_gate_20261103_v1'
    for f in canonical.glob('*.npz'):shutil.copy2(f,logs/f.name)
    for name in ('decisions.csv','rounds.csv','completion.json','protocol.json'):
        shutil.copy2(canonical/name,logs/name)
    shutil.copy2(ROOT/'deliverables/persistent_integer_gate_20261103_v1/rounds.csv',out/'integer_rounds.csv')
    shutil.copy2(ROOT/'deliverables/cached_budget_quality_20261103_v1/quantization.csv',out/'quantization.csv')
    previous=ROOT/'Doc/evidence/2026-11-03/persistent_cache_reconstruction'
    frozen=json.loads((previous/'final_audit.json').read_text())
    labels=ROOT/'deliverables/cached_budget_quality_20261103_v1/exact_dirty_labels.npz'
    registry=ROOT/'training/stageb_hgraph/current_algorithm.json'
    assert sha(labels)==frozen['labels_sha256'] and sha(registry)==frozen['default_registry_sha256']
    prior_manifest=json.loads((previous/'archive_manifest.json').read_text())
    for filename,digest in prior_manifest.items():assert sha(previous/filename)==digest
    for filename,digest in json.loads((previous/'integer_gate/source_manifest.json').read_text()).items():
        assert sha(ROOT/filename)==digest
    write_json(out/'preserved_previous_evidence.json',dict(labels_sha256=sha(labels),labels=161700,
        registry_sha256=sha(registry),previous_archive_files_unchanged=len(prior_manifest),
        cap4_seed44_failed_original_gate=True,new_native_controls_run=False))
    summary=pd.read_csv(source/'summary.csv').set_index(['method','phase'])
    work=json.loads((source/'UB_order_work_bound.json').read_text())
    write_json(out/'next_protocol.json',dict(
        status='design frozen for next paid implementation gate; not executed this turn',
        numerical='unchanged persistent_q1e6_v1 objective/strict thresholds/max-anchored1-unit canonical key',
        first_bound='capacity only; positive weights and native BLER[0,1], same per-UE nearest-even operations',
        tighter_bound='theory only; defer Eigen finite-error certificate; little incremental gain here',
        representation='separate versioned native library, sparse persistent edit table; only dirty PRG/UE bounds refreshed',
        heap='version every entry; all novel edits present; invalid old bounds cannot terminate a round',
        ranking_controls=['upper bound descending','stale exact delta, novel fallback upper bound'],
        learned_ordering='deferred until engineering/wall gate; existing161700 correlated labels may rank only, never prune',
        certificate='unseen UB>threshold and (UB>=known max or UB==known max-1 with smaller key) still needs evaluation',
        exact_cache='reuse only current valid exact payload; skipped entries stay unscored/dirty, never become clean by root advance',
        equivalence='pay for independent full integer baseline, compare every root/choice/terminal; full candidate audit outside timing',
        costs='serial alternating all12 states; no observer/log/history/profiling inside timing; report bound/heap/UE/MMSE requests/misses/first scan',
        early_stop='no time/evaluation budget may bypass a certificate',
        future_approximate_trajectory='before another approximate service gate, run paired canonical-full/random-tie-full native controls with same initialization/offered traffic and fixed random seeds; record tie opportunities and per-UE expiry/path variation; freeze a prospective protocol, never retroactively change cap4 failure',
        native_controls='not launched; a tie-free/identical-path control is noninformative about broader local-optimum variability'))
    report='''MMSE无调用的ΔJ上界与规范最佳优先惰性精评：零新增PHY离线实验
2026-11-03；同12个相关seed41开发输入、378轮、624670个候选。

结论
BLER下界=0的基础界已经足够紧，值得进入单独版本的等价实现门。本轮完成上界分析和通用惰性选择证书参考，没有修改在线缓存核，没有运行新PHY、原生闭环或训练。精确缓存相对新整数full的轨迹等价结论保留；不将其无限外推为旧浮点目标的普遍等价。

1. 逐候选基础界及数值规范
MCS在本次决策内固定。编辑后的PRG数、跨PRG最小layer、TBS与D=min(Q,TBS)全部由逻辑操作得到，floor也不需要PHY。正权重且原生BLER位于[0,1]，故候选每UE的实际E<=D。
定义q(E)=round_even((E*w)*1e6)，上界U=Σ受影响UE[q(D_candidate)-q(E_root)]；未受影响贡献直接消去。这沿用同一浮点乘法顺序/逐UE舍入，正乘法和nearest-even舍入保持单调，整数加减不重归约浮点J。基础界不依赖干扰单调性，不调用MMSE；整型溢出/维度沿用原规范契约。
本轮logical_bounds是稠密离线统计实现，并非新的在线全候选×UE开销。后续集成须沿用持久编辑表的稀疏UE支持集和增量根贡献，不能将稠密参考直接称加速实现。

2. 核内干扰单调性：可证明的部分与数值边界
当前activity-only核：R=noise*I+Σ活动干扰Hc Hc^H，noise>0；自身信号S固定。增加干扰有R2≽R1，R2^-1≼R1^-1；E=(I+S^H R^-1 S)^-1满足E2≽E1，故逐层gamma=max(0,1/E_ll-1)不增。受害UE自身PRG集合/layer/MCS固定时，平均gamma和SINR同样不增。
逐原生查表分支核验28行：BLER实际查表函数均单调不增、范围[0,1]。MCS3只是SNR横轴3.0/3.25乱序；不能对表排序或直接用排序插值代替原生扫描。可达分支的区间/斜率/跳变已检查。早先“非单调MCS3行”的注释不意味着实际BLER(SINR)函数必然非单调。
因此纯增加干扰且受害UE自身配置不变时，其E上界可收为根E（整数贡献增益上界0）。移除干扰、改变自身PRG/layer、同时增删不可比较干扰集合时，仍用基础容量界。已有根日志提供43个单调受害UE转移，最大SINR违例0dB；没有新增MMSE验证。
以上矩阵序证明是精确算术性质，43个样本不构成Eigen LDLT/inverse浮点误差证书。因此收紧版theory_added只列理论诊断，不部署为有限精度等价剪枝条件。基础容量界无需解决这个额外数值问题。

3. 规范平局与停止证书
现行选择是：先找所有floor可行、严格超过1/16B阈值的最大整数ΔJ=m，再从[m-1,m]取编辑规范键最小者k。
未评候选若UB>=m，或者UB=m-1且键小于k，仍可能影响选择，必须继续处理；UB<=阈值可直接排除。无合格已知移动时，所有剩余可行候选UB<=阈值才是收敛证书。
反例：已知(key20,ΔJ100)，未评(key10,UB99)，虽然100>=99+1，未评候选若精确值99仍会胜出。简单“最佳值>=最大剩余上界+1”不保证当前规范argmax。无键信息的充分条件应为m>最大剩余UB+1（整数至少差2）；实现参考采用上面的键感知条件。
lazy_canonical只让优先级决定访问顺序，证书只使用当前可证上界/当前有效精确值。旧ΔJ和模型输出不能变成精确缓存，未精评条目不能随轮次推进自动标clean；不能截断预算后绕过证书。任意排序都保持规范终态的前提是这些条件完整成立。

4. 日志完整性与最大值区间
保存了全部根的逐UE精确PHY，但未保存每个候选的整数分数。利用初始化大间隔/floor拒绝证明整数初始化稳定；既有378轮量化选择变化为0，且新整数full的选择键序列逐轮一致。由下一根/终态逐UE贡献得到所选增益b，规范平局保证真实最大值在[b,b+1]，不能假造一个点值。
以下区间分别用b+1/b统计UB>=真实最大值；分母为逻辑floor可行候选。两个方法合计328个候选计数落在1单位不确定边界。确认轮不知真实的亚阈值最大值，明确改报UB>接受阈值的证书压力。

方法 | 首扫12轮：UB>=最大值 | 轮内354轮：UB>=最大值 | 确认12轮：UB>阈值
'''
    for method in ('capacity','theory_added'):
        f=summary.loc[method,'first'];m=summary.loc[method,'inner'];c=summary.loc[method,'confirmation']
        report+=f"{method} | {100*f.above_best_lo_fraction:.4f}–{100*f.above_best_hi_fraction:.4f}% | {100*m.above_best_lo_fraction:.4f}–{100*m.above_best_hi_fraction:.4f}% | {100*c.threshold_live_fraction:.4f}%\n"
    f=summary.loc['capacity','first'];m=summary.loc['capacity','inner'];c=summary.loc['capacity','confirmation']
    report+=f"\n不做floor过滤的全部候选，基础界三类比例为：首扫{100*f.all_above_best_lo_fraction:.4f}–{100*f.all_above_best_hi_fraction:.4f}%，轮内{100*m.all_above_best_lo_fraction:.4f}–{100*m.all_above_best_hi_fraction:.4f}%，确认{100*c.all_threshold_live_fraction:.4f}%。原始表同时保留全域和可行域分母，不把逻辑floor收益混算为上界收益。\n"
    report+='''
基础界首扫252–257/19254；轮内22049–22208/580175；确认1253/19752。普通精确cache原本本轮需刷新集合中，首扫同比例，轮内4698–4773/156472（3.0025–3.0504%），确认251/5029（4.9911%）。这些dirty统计只描述原精确cache已填满缓存的局部根，不假设惰性搜索此前跳过的候选已经拥有精确标签。
12个确认根均不能仅靠基础界零精评停止，收紧界也不能。基础界本身可排除确认轮约93.66%的全部可行候选，但剩余上界高的条目仍须精评后再给证书。

5. 上界优先的条件工作量界，非实测加速
按UB递减时，真正最大值候选的UB至少为最大值，所以任何UB<最大值-1的候选被访问前，真正最大值已被评过/已知并能给出剪枝。接受轮保留所有floor可行、严格超过阈值且UB>=b-1的候选；确认轮保留UB>阈值。这给出无需完整精确标签的精评次数上限。
采用全部候选计数，不乘cache比例、不假设跳过候选未来可免费复用，也不假设未知新编辑已有标签。每轮上界已包含平局带；初始化20次评分保留。
'''
    for r in work:
        report+=f"{r['method']}: {r['conditional_evaluations_upper_bound']}+20={r['total_evaluations_upper_bound']}次，full的{100*r['ratio_original_full']:.4f}%，现有cache的{100*r['ratio_existing_cache']:.4f}%。\n"
    report+='''这不是已经运行的惰性轨迹、MMSE次数或墙钟比例。上界计算/队列维护也要计费，PHY候选数不等于独立MMSE miss数。上界排序的工作量保证不自动适用于旧ΔJ/学习排序；它们在同一证书下质量等价，但可能更快也可能更慢。
单调性收紧只把上界23738降为23448，增益很小，优先实现数值义务更少的基础界。原cache墙钟的74%为PHY相关开销仍是已知profile；不能用13%评分比例直接乘74%来声称实际时间。

6. 后续执行门与学习位置
下一门先将基础界＋上界优先队列集成到单独版本的C++持久表，付费跑独立完整整数对照的逐根/逐选择/终态等价门，再做无日志/审计的公平串行墙钟。旧ΔJ只作访问顺序对照，首扫/novel项回退上界；本轮不报告它们未实际执行的评分收益。下一协议同时要求所有脏/新条目被界覆盖、堆条目按代际失效、已跳过条目保持未精评状态。
161700条标签仅保留，未拟合模型；学习只能决定顺序，不能改界、floor、候选全集、阈值或停止条件。先过基础工程与成本门，再评旧ΔJ/上界/模型三种排序的净成本。相关seed41标签不能充当独立泛化证据。
守卫cap4在seed44的+2.7699pp失败按原门保留。如果以后评估改变轨迹的方法，先在相同初始化/业务输入下做规范full与固定随机平局full闭环对，记录逐UE过期率和真正出现的平局/轨迹分歧，冻结前瞻协议；没有平局或路径相同的对照不能证明不同局部解波动小。本轮没有启动这些闭环，不回改旧门。

7. 核验与付费边界
7项零PHY测试通过，含1单位边界、新最大值改变平局带、floor/阈值零精评、150组×3种任意排序与规范argmax一致、错误界拒绝、原生BLER分支与纯逻辑TBS/ΔD。624670个逻辑候选界，0新candidate/UE PHY、0MMSE、0原生轨迹、0训练步。原161700标签、默认注册表、上轮290个归档文件SHA全部保持。原始大状态/初始化JSONL按input_manifest保留来源，紧凑根日志已复制归档。
'''
    (out/'report.txt').write_text(report)
    sourcefiles=[Path(__file__).with_name(n) for n in ('certified_bounds.py','analyze_certified_bounds.py',
        'test_certified_bounds.py','certified_bounds_README.rst','report_certified_bounds.py')]
    write_json(out/'new_source_manifest.json',{str(f.relative_to(ROOT)):sha(f) for f in sourcefiles})
    write_json(out/'final_audit.json',dict(completed=True,tests=7,random_priority_cases=450,
        logical_candidate_bounds=624670,new_candidate_PHY=0,new_UE_PHY=0,new_MMSE=0,new_native_runs=0,
        training_steps=0,online_kernel_unchanged=True,previous_archive_unchanged=True,
        labels_sha256=sha(labels),registry_sha256=sha(registry),
        capacity_work_bound=work[0]['total_evaluations_upper_bound'],
        executed_lazy_equivalence_gate=False,wall_speedup_evaluated=False))
    write_json(out/'archive_manifest.json',{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--tests',default='/tmp/certified_bounds_20261103_tests.txt');main(p.parse_args())

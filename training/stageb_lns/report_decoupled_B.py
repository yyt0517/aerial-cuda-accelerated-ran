"""Archive completed B and engineering evidence without modifying old results."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from .physics import ROOT
from .analyze_bounded_stress import local
from .analyze_dpls_moves import write_json,sha

def main(args):
    campaign=ROOT/'deliverables/B_ttl_decoupled_20261104_v1'
    frontier=ROOT/'deliverables/B_ttl_frontier_20261104_v1'
    engineering=ROOT/'deliverables/tournament_gate_wall_20261104_v2'
    activity=ROOT/'deliverables/activity_mmse_coverage_20261104_v1'
    b=json.loads((campaign/'completion.json').read_text());f=json.loads((frontier/'completion.json').read_text())
    e=json.loads((engineering/'completion.json').read_text())
    assert b['completed'] and f['completed'] and e['completed']
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    for name,src in (('frontier',frontier),('engineering',engineering),('activity',activity)):
        shutil.copytree(src,out/name)
    (out/'B_campaign').mkdir()
    for path in campaign.glob('*.json'):shutil.copy2(path,out/'B_campaign'/path.name)
    shutil.copytree(campaign/'source',out/'B_campaign/source')
    native=out/'native';native.mkdir();raw={};cohort_rows=0
    counts=dict(new_runs=b['new_native_runs'],new_native_steps=b['new_native_steps'],
        reused_runs=2,reused_native_steps_not_new=24800,policy_decisions=0,
        candidate_evaluations=0,UE_evaluations=0,MMSE_requests=0,MMSE_misses=0)
    for run in b['runs']+b['reused_runs']:
        d=local(run['directory']);dest=native/d.name;dest.mkdir()
        for name in ('summary.json','validation.json','reward_validation.json','launch.json','saved_states.json',
            'search.csv','per_ue.csv','cohort.csv.gz'):
            shutil.copy2(d/name,dest/name)
        for path in d.iterdir():
            if path.is_file():raw[str(path.relative_to(ROOT))]=sha(path)
        val=json.loads((d/'validation.json').read_text())
        assert val['passed'] and val['sources_unchanged']
        assert val['exact_native_actions']==val['fresh_input_checks']==val['exact_mcs_layer_tbs_dequeue']==12400
        assert json.loads((d/'reward_validation.json').read_text())['passed']
        launch=json.loads((d/'launch.json').read_text())
        assert launch['current_teacher_calls']==0 and not launch['ctbs_action_clipping']
        for path,digest in launch['source_sha256'].items():assert sha(local(path))==digest
        assert sha(local(launch['command'][0]))==launch['simulator_sha256']
        cohort_rows+=len(pd.read_csv(d/'cohort.csv.gz'))
        if run in b['runs']:
            s=pd.read_csv(d/'search.csv');counts['policy_decisions']+=len(s)
            for col,key in (('candidates','candidate_evaluations'),('ue_evaluations','UE_evaluations'),
                ('mmse_requests','MMSE_requests'),('mmse_misses','MMSE_misses')):counts[key]+=int(s[col].sum())
    write_json(out/'native_raw_manifest.json',raw)
    source=out/'source';source.mkdir();files=[]
    for name in ('report_decoupled_B.py','ttl_decoupled_campaign.py','analyze_ttl_frontier.py','analyze_activity_mmse.py',
        'tournament_kernel.cpp','tournament_dpls.py','build_tournament.py','_tournament.so','native_schedule_inputs.py',
        'test_tournament.py','audit_tournament.py','decoupled_B_README.rst'):
        path=Path(__file__).with_name(name);shutil.copy2(path,source/name);files.append(path)
    write_json(out/'source_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in files})
    shutil.copy2(Path(__file__).with_name('decoupled_B_README.rst'),out/'README.rst')
    shutil.copy2(args.tests,out/'tests.txt');assert 'Ran 13 tests' in Path(args.tests).read_text() and '\nOK\n' in Path(args.tests).read_text()
    for name in ('B_ttl_decoupled_20261104_v1.log','B_ttl_frontier_20261104_v1.log','tournament_audit_20261104_v1.log',
        'tournament_audit_20261104_v2.log','tournament_wall_20261104_v2.log'):
        path=Path('/tmp')/name
        if path.exists():shutil.copy2(path,out/name)
    # Retain the failed native-bool-interface attempt, including its source.
    failed=ROOT/'deliverables/tournament_gate_wall_20261104_v1'
    shutil.copytree(failed,out/'failed_adapter_gate_v1')
    records=json.loads((engineering/'audit_work_ledger.json').read_text())
    partial=[dict(state=r['state'],**{a:r[a] for a in ('heap','tree','direct','native_direct') if r['state']<4 or a!='native_direct'})
        for r in records[:5]]
    write_json(out/'failed_adapter_gate_v1/accounting.json',dict(
        reason='uint8 legal mask failed bool in-place warm repair at fifth state; fixed as bool view, old repair unchanged',
        completed_states=4,paid_replays=19,partial_work_reconstructed_from_identical_deterministic_v2_counts=partial,
        attempted_native_fifth_state_PHY=0,certified=False))
    prototype=ROOT/'deliverables/tournament_prototype_20261104_v1'
    if prototype.exists():shutil.copytree(prototype,out/'early_tree_prototype')
    previous_files=0
    for directory in (ROOT/'Doc/evidence/2026-11-03/persistent_cache_reconstruction',
        ROOT/'Doc/evidence/2026-11-03/certified_bounds',ROOT/'Doc/evidence/2026-11-04/bounded_inner_load'):
        for name,digest in json.loads((directory/'archive_manifest.json').read_text()).items():
            assert sha(directory/name)==digest;previous_files+=1
    old=ROOT/'Doc/evidence/2026-11-04/bounded_inner_load'
    for name,digest in json.loads((old/'source_manifest.json').read_text()).items():assert sha(ROOT/name)==digest
    labels=ROOT/'deliverables/cached_budget_quality_20261103_v1/exact_dirty_labels.npz'
    registry=ROOT/'training/stageb_hgraph/current_algorithm.json'
    frozen=json.loads((old/'final_audit.json').read_text())
    assert sha(labels)==frozen['labels_sha256'] and sha(registry)==frozen['registry_sha256']
    for path,digest in json.loads((campaign/'source_manifest.json').read_text()).items():assert sha(ROOT/path)==digest
    for path,digest in json.loads((frontier/'input_manifest.json').read_text()).items():assert sha(local(path))==digest
    weighted=json.loads((frontier/'weighted_work_ledger.json').read_text())
    audit=json.loads((engineering/'audit_completion.json').read_text())
    summary=pd.read_csv(engineering/'wall_summary.csv');points=pd.read_csv(frontier/'frontier.csv')
    paired=pd.read_csv(frontier/'paired.csv');bank=pd.read_csv(activity/'coverage_by_load.csv')
    write_json(out/'work_ledger.json',dict(native_new_only=counts,
        reused='two prior exactk0 controls, no new runtime/PHY charge',
        weighted_oracles=weighted,engineering_audit=json.loads((engineering/'audit_work_ledger.json').read_text()),
        engineering_timing=json.loads((engineering/'timing_work_ledger.json').read_text()),
        engineering_profile=pd.read_csv(engineering/'profiles.csv').to_dict('records'),
        failed_adapter='19 partial replays, accounting.json retained; not silently dropped',
        exploratory='prototype/simplified three-state timing probes: elapsed summaries retained, exact replay/PHY counts not retained; excluded from exhaustive counted-work totals and formal speed evidence; synthetic tests separately charged',
        ranking_training_steps=0,gradient_training_steps=0,activity_audit_new_PHY=0))
    # Static scientific plots are standalone exportable artifacts.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for arm,label in (('heap','Frozen heap'),('tree','Tournament'),('direct','Tournament+direct gamma'),('native_direct','Native input+direct+tree')):
        g=summary[(summary.rate>0)&(summary.arm==arm)].sort_values('rate')
        axes[0].plot(g.rate/.8,100*g.mean_ratio_full,marker='o',label=label)
    axes[0].axhline(40,color='gray',ls=':');axes[0].set_xlabel('Load relative to rate0.8')
    axes[0].set_ylabel('Serial mean wall/full (%)');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    x=np.arange(3);axes[1].bar(x-.17,100*bank.full_bank_coverage,.34,label='Full',color='#666')
    axes[1].bar(x+.17,100*bank.bounded_bank_coverage,.34,label='Bounded',color='#087f8c')
    axes[1].set_xticks(x,['x1','x1.4','x1.6']);axes[1].set_ylabel('MMSE tuple bank coverage (% of2448/TTI)')
    axes[1].legend();fig.savefig(out/'engineering_and_MMSE.png',dpi=180);fig.savefig(out/'engineering_and_MMSE.pdf');plt.close(fig)
    text='''方法B与部署墙钟解耦：冻结TTL前沿及独立等价工程
2026-11-04；开发seed41/42/43，rate0.8；工程另含×1.4/×1.6已保存输入。

1. 本轮授权与冻结
用户明确同意科学比较B不再等待40%部署墙钟门。只改入口：旧高负载43.49%/48.57%失败与40%目标不变，旧报告不改。κ=0/.25/.5/1/2、seed41–43、4000PFQ预热+4000稳定+4000测量+400追踪、权重和服务门全部沿原方案。
所有科学臂仍用同一个冻结_bounded.so。因果缺口权重乘(1+κ*(1−HOLslack/400))后均值归一，κ0直接保留原位值；只读FIFO、跳过将过期的队头，不二次before()。不读未来结局、无当前教师调用/CTBS裁剪。部署优化独立库从未替换中途内层，因此科学差异只归因于这个权重族。
13条新增原生轨迹161200步，复用κ0 seed41/42旧24800步，构成15点；全部成熟cohort pending=0、共同PFQ4000步/输入、offered业务和测量cohort成员一致，原生动作/MCS/layer/TBS/dequeue/FIFO核验通过。并发采集墙钟不作公平速度证据。

2. 冻结系数前沿
聚合payload为三seed均值；expiry与成功包P95按成熟包合池；最差UE为所有(seed,UE)最大值。P95只针对成功包；另报共同成功包的时延差。完整逐seed/配对/逐UE/cohort保留，不按聚合掩盖个别seed。
κ | payload Mbps | expiry % | success P95 ms | worst(seed,UE) expiry % | 4D Pareto | B全seed不劣
'''
    for r in points.itertuples():text+=f'{r.coefficient:g} | {r.payload_mbps:.3f} | {r.expiry_pct:.6f} | {r.P95_ms:.3f} | {r.worst_ue_expiry_pct:.6f} | {r.pareto4D} | {r.B_no_worse_all_seeds}\n'
    text+=f'''\n四目标非支配系数={f['Pareto_coefficients']}。非零系数B全seed不劣={f['B_no_worse_nonzero_coefficients']}。固定网格所有点保留，未改门/补扫/挑seed。B逐seed要求payload与expiry不劣、P95≤κ0+1ms、最差UE≤max(原最差+2pp,5%)且不新增零服务或TTL长间隔UE。历史质量宽容门另列，不能把Pareto点等同晋级。
'''
    text+='κ2只是本冻结网格的开发集强对照，位于网格边界，不称全局最优；后续状态相关学习需同时比较κ0和固定κ2。当前κ≥.5已将这些有限窗口的过期清零，学习的额外余量尚未证明，不根据本轮结果立即补扫或扩网。\n'
    for r in paired.itertuples():text+=f'seed{r.seed} κ{r.coefficient:g}: payload比{100*r.payload_ratio:.4f}%，expiry差{r.expiry_delta_pp:+.5f}pp，P95差{r.P95_delta_ms:+.3f}ms，最差UE差{r.worst_expiry_delta_pp:+.5f}pp；B={r.B_no_worse_rule_passed}\n'
    text+=f'''\n每新增点4个保存状态，共{f['weighted_equivalence_states']}个加权状态、{f['weighted_equivalence_candidates']}个完整候选另付独立full整数oracle，根/选择/上界/终态通过；κ0复用状态旧审计不重复计费。全前沿与逐seed曲线见frontier/ttl_frontier.png/pdf、ttl_per_seed.png/pdf。
这只是开发集系数扫描，梯度训练/排序训练均0，没有状态相关学习策略或泛化结论，未启动最终新seed确认，不能称AI已改善服务。下阶段学习仍须和同一精确内层、固定权重族及逐seed服务门比较。

3. MMSE问题数上限
当前activity-only核去除受害UE自身cell bit后，C=3每(UE,PRG)最多4个mask；36×17×4=2448键/TTI。旧50400个策略TTI均不超过此上限，240个full/bounded配对输入也满足。现有日志只有总miss，能验证键数覆盖，不能还原每对UE/PRG的4种分布；相同协方差的碰巧合并还可能让实际不同矩阵数更少。
'''
    for r in bank.itertuples():text+=f'×{r.rate/.8:g}: full覆盖bank{100*r.full_bank_coverage:.3f}%，bounded{100*r.bounded_bank_coverage:.3f}%，bounded/full miss={100*r.bounded_over_full_misses:.3f}%。\n'
    text+='''零新增PHY验证支持高负载小候选集合已覆盖大部分独立MMSE键。原核本来就是Eigen固定4×4；本轮未做批量矩阵算术或Woodbury数值替换。

4. 独立等价工程与公平墙钟
跨轮持久每PRG本地锦标赛、全局每PRG一个叶子，按受影响UE依赖只刷新脏界；全局dequeue变化时还更新跨floor边界的可行性，不能仅凭PHY依赖失效。已知精确最大值仍扫描轻量元数据，不宣称所有全域遍历消失。早期双树维护原型没有净收益，保留；最终仅未知UB维护树。
直接索引Γ表只复制原Context::gamma算出的4个double，每次新miss仍用原Eigen表达式，重复键跳过关联容器查找；C>8回退原缓存。native适配迁移MCS/TBS/合法域和PFQ槽映射；权重归一及H协方差/信号仍用原NumPy算术，Context原本已native，没有声称整个1.9ms都被迁移。
'''
    text+=f'原12+负载240状态共{audit["rounds"]}轮、{audit["candidates"]}候选，tree/direct/native_direct三臂每轮根/全部UE字段的64位模式、键/UB/ΔD/缓存ΔJ/状态/选择和终态均与冻结heap一致，所有候选/UE/MMSE请求/miss计数一致。旧完整oracle门提供传递等价，另四合成路径直接对独立full。13项测试通过，覆盖全局floor、warm repair、MCS两套LUT/阈值邻域及三种OLLA结果。\n'
    text+='''首轮审计在第5个warm状态遇到uint8合法mask与旧bool原地运算不兼容，已改新适配器的bool视图；旧repair未改，失败源/日志/19次部分回放成本保留。重跑252全部通过，没有把异常当质量失败或跳掉warm状态。
公平串行CPU255，252状态×五臂×3旋转重复，输入/PFQ/Context/搜索/销毁计时，observer/日志/审计/profile关，静态表预热；每状态中位数后均值。原生B运行和分析均已结束后才计时；共享主机干扰与离群点保留。以下P95/max是保存状态中位数分布，不能替代旧连续测量TTI P95/max。
负载(rate0为原12状态) | arm | mean/P95/max ms | wall/full | wall/heap
'''
    for r in summary.itertuples():text+=f'{r.rate:g} | {r.arm} | {r.mean_ms:.3f}/{r.P95_ms:.3f}/{r.max_ms:.3f} | {100*r.mean_ratio_full:.3f}% | {100*r.mean_ratio_heap:.3f}%\n'
    text+='\n锦标赛没有净墙钟收益，直接Γ表也未带来明显收益；native适配贡献主要节省，约0.4ms，并未省掉原估计的1.7ms。×1.6同次heap9.545ms→native_direct9.138ms，full19.703ms，仍46.38%；×1.4仍41.23%。树维护代价抵消堆重建节省，因此不将1.15ms旧profile直接当作可删除成本，也不默认晋级工程版。独立profile含额外计时开销，不能与关profile的正式墙钟相减解释每项净收益。\n'
    text+=f'''\n新native_direct全部负载≤40%保存状态均时门={e['deployment_wall_all_stress_passed']}。未新增工程版连续原生尾分布，不把本轮保存状态速度升级成部署验证；原连续尾部报告仍在bounded_inner_load。本轮成本账本分科学轨迹、加权full oracle、工程等价/计时/profile与失败尝试，不把MMSE审计或对照成本算免费。早期3状态探索仅保留耗时摘要，未保留完整精评计数，不纳入可穷尽工作总数或正式速度结论。

5. 保留
原161700标签、有效默认注册表及{previous_files}个既有归档文件SHA核验不变，旧源码/库哈希也不变。cap4 seed44按原门失败的结论不变。没有默认替换、Woodbury、排序模型、新M或最终确认seed。原生大日志/动作/业务/状态保留在deliverables，完整SHA与小型成熟cohort/逐UE/成本/协议/源码在本归档。复现见README.rst。
'''
    (out/'report.txt').write_text(text)
    write_json(out/'final_audit.json',dict(completed=True,tests=13,new_native_steps=161200,
        reused_steps_not_new=24800,quality_points=15,mature_cohort_rows=cohort_rows,
        previous_files_unchanged=previous_files,old_solver_unchanged=True,scientific_solver_frozen=True,
        labels_sha256=sha(labels),registry_sha256=sha(registry),default_changed=False,
        B_wall_decoupled_by_user=True,old_deployment_failure_unchanged=True,
        weighted_oracle_states=52,engineering_bitwise_states=252,
        new_saved_state_wall_gate=e['deployment_wall_all_stress_passed'],new_continuous_tail=False,
        B_nonzero_no_worse=f['B_no_worse_nonzero_coefficients'],ranking_training_steps=0,gradient_training_steps=0))
    write_json(out/'archive_manifest.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--tests',default='/tmp/B_tournament_tests_20261104.txt');main(p.parse_args())

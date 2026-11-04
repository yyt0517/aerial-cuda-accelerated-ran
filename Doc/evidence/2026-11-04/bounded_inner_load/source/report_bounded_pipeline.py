"""Archive the completed bounded-inner/load/TTL pipeline; never run experiments."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from .physics import ROOT
from .analyze_dpls_moves import write_json,write_csv,sha
from .analyze_bounded_stress import local

def main(args):
    roots=dict(gate=ROOT/'deliverables/bounded_gate_wall_20261104_v1',
        stress=ROOT/'deliverables/bounded_load_stress_20261104_v1',
        stress_analysis=ROOT/'deliverables/bounded_load_stress_analysis_20261104_v1',
        ttl=ROOT/'deliverables/bounded_ttl_scan_20261104_v1',
        frontier=ROOT/'deliverables/bounded_ttl_analysis_20261104_v1')
    roots['profile']=ROOT/'deliverables/bounded_load_profile_20261104_v2'
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    gate=json.loads((roots['gate']/'completion.json').read_text())
    stress=json.loads((roots['stress_analysis']/'completion.json').read_text())
    assert gate['completed'] and stress['completed']
    has_B=stress['B_entry_passed']
    if has_B:assert json.loads((roots['frontier']/'completion.json').read_text())['completed']
    for key in ('gate','stress_analysis',*(['frontier'] if has_B else [])):
        shutil.copytree(roots[key],out/key)
    shutil.copytree(roots['profile'],out/'load_profile')
    shutil.copytree(ROOT/'deliverables/bounded_load_profile_20261104_v1',out/'profile_before_immutable_warmup')
    for key in ('stress',*(['ttl'] if has_B else [])):
        dest=out/key;dest.mkdir()
        for f in roots[key].glob('*.json'):shutil.copy2(f,dest/f.name)
        shutil.copytree(roots[key]/'source',dest/'source')
    shutil.copytree(ROOT/'deliverables/bounded_B_plan_20261104_v1',out/'B_plan')
    native_dirs={};native_counts={};raw_inputs={}
    for key in ('stress',*(['ttl'] if has_B else [])):
        c=json.loads((roots[key]/'completion.json').read_text())
        native_counts[key]=dict(runs=c['new_native_runs'],steps=c['new_native_steps'])
        searches=[pd.read_csv(local(r['directory'])/'search.csv') for r in c['runs']]
        native_counts[key].update(policy_decisions=sum(len(t) for t in searches),
            candidate_evaluations=int(sum(t.candidates.sum() for t in searches)),
            UE_evaluations=int(sum(t.ue_evaluations.sum() for t in searches)),
            MMSE_requests=int(sum(t.mmse_requests.sum() for t in searches)),
            MMSE_misses=int(sum(t.mmse_misses.sum() for t in searches)))
        for r in c['runs']+c['reused_runs']:
            d=local(r['directory']);native_dirs[d.name]=d
    native=out/'native';native.mkdir();cohort_rows=0;native_steps=0
    for name,d in sorted(native_dirs.items()):
        dest=native/name;dest.mkdir()
        for f in ('summary.json','validation.json','reward_validation.json','launch.json','search.csv','per_ue.csv','cohort.csv.gz','saved_states.json'):
            shutil.copy2(d/f,dest/f)
        for f in d.iterdir():
            if f.is_file():raw_inputs[str(f.relative_to(ROOT))]=sha(f)
        val=json.loads((d/'validation.json').read_text());assert val['passed'] and val['sources_unchanged']
        assert val['exact_native_actions']==val['fresh_input_checks']==val['exact_mcs_layer_tbs_dequeue']==12400
        native_steps+=val['exact_native_actions'];cohort_rows+=len(pd.read_csv(d/'cohort.csv.gz'))
        launch=json.loads((d/'launch.json').read_text());assert launch['current_teacher_calls']==0 and not launch['ctbs_action_clipping']
        for f,digest in launch['source_sha256'].items():assert sha(local(f))==digest
        assert sha(local(launch['command'][0]))==launch['simulator_sha256']
    write_json(out/'native_raw_manifest.json',raw_inputs)
    costs=[]
    stresscomplete=json.loads((roots['stress']/'completion.json').read_text())
    for rate in (.8,1.12,1.28):
        measured=[]
        for r in stresscomplete['runs']:
            if r['rate']==rate:
                f=pd.read_csv(local(r['directory'])/'search.csv');measured.append(f[f.tti.between(8001,12000)])
        f=pd.concat(measured);fraction=f.candidates/f.full_candidate_count
        costs.append(dict(rate=rate,load=rate/.8,TTIs=len(f),wall_mean_ms=float(f.wall_ms.mean()),
            wall_P95_ms=float(f.wall_ms.quantile(.95)),wall_max_ms=float(f.wall_ms.max()),
            cpu_P95_ms=float(f.cpu_ms.quantile(.95)),cpu_max_ms=float(f.cpu_ms.max()),
            candidate_work_ratio=float(f.candidates.sum()/f.full_candidate_count.sum()),
            candidate_fraction_P95=float(fraction.quantile(.95)),candidate_fraction_max=float(fraction.max())))
    write_csv(out/'cost_by_load.csv',costs)
    previous=ROOT/'Doc/evidence/2026-11-03/persistent_cache_reconstruction'
    freeze=json.loads((previous/'final_audit.json').read_text())
    labels=ROOT/'deliverables/cached_budget_quality_20261103_v1/exact_dirty_labels.npz'
    registry=ROOT/'training/stageb_hgraph/current_algorithm.json'
    assert sha(labels)==freeze['labels_sha256'] and sha(registry)==freeze['default_registry_sha256']
    for f,digest in json.loads((previous/'integer_gate/source_manifest.json').read_text()).items():assert sha(ROOT/f)==digest
    old_files=0
    for directory in (previous,ROOT/'Doc/evidence/2026-11-03/certified_bounds'):
        for f,digest in json.loads((directory/'archive_manifest.json').read_text()).items():assert sha(directory/f)==digest;old_files+=1
    source=out/'source';source.mkdir()
    sourcefiles=[]
    for n in ('bounded_kernel.cpp','bounded_dpls.py','build_bounded.py','_bounded.so','bounded_campaign.py',
        'bounded_native.py','rate_native.py','ttl_weights.py','test_bounded_dpls.py','analyze_bounded_stress.py',
        'analyze_ttl_frontier.py','bounded_README.rst','report_bounded_pipeline.py'):
        f=Path(__file__).with_name(n);shutil.copy2(f,source/n);sourcefiles.append(f)
    f=Path(__file__).with_name('profile_bounded_stress.py');shutil.copy2(f,source/f.name);sourcefiles.append(f)
    shutil.copy2(Path(__file__).with_name('bounded_README.rst'),out/'README.rst')
    write_json(out/'source_manifest.json',{str(f.relative_to(ROOT)):sha(f) for f in sourcefiles})
    shutil.copy2(args.tests,out/'tests.txt');assert 'Ran 11 tests' in Path(args.tests).read_text() and '\nOK\n' in Path(args.tests).read_text()
    wall=pd.read_csv(roots['gate']/'wall_summary.csv').set_index('arm')
    profile=pd.read_csv(roots['gate']/'profiles.csv').mean(numeric_only=True)
    load_profile=pd.read_csv(roots['profile']/'summary.csv').set_index('rate')
    load_wall=pd.read_csv(roots['stress_analysis']/'wall_summary.csv')
    equivalent=pd.read_csv(roots['stress_analysis']/'equivalence.csv').set_index('state')
    accounting=[]
    for r in json.loads((roots['stress_analysis']/'work_ledger.json').read_text()):
        f,b=r['full_reference'],r['bounded']
        accounting.append(dict(rate=float(equivalent.loc[r['state'],'rate']),
            full_candidates=f['candidate_evaluations'],bounded_candidates=b['candidate_evaluations'],
            full_UE=f['ue_evaluations'],bounded_UE=b['ue_evaluations'],
            full_MMSE_misses=f['mmse_misses'],bounded_MMSE_misses=b['mmse_misses']))
    physics=pd.DataFrame(accounting).groupby('rate').sum()
    physics['candidate_ratio']=physics.bounded_candidates/physics.full_candidates
    physics['UE_ratio']=physics.bounded_UE/physics.full_UE
    physics['MMSE_miss_ratio']=physics.bounded_MMSE_misses/physics.full_MMSE_misses
    physics.to_csv(out/'PHY_accounting_by_load.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    rates=(.8,1.12,1.28)
    for rate,color in zip(rates,('#087f8c','#d17a22','#7752a3')):
        values=[]
        for run in stresscomplete['runs']:
            if run['rate']==rate:
                t=pd.read_csv(local(run['directory'])/'search.csv')
                values.extend(t[t.tti.between(8001,12000)].wall_ms.tolist())
        values=np.sort(values)
        axes[0].semilogy(values,np.arange(len(values),0,-1)/len(values),label=f'load x{rate/.8:g}',color=color)
    axes[0].axhline(.05,color='#777',ls=':',lw=1)
    axes[0].set_xlabel('Per-TTI policy wall time (ms), all measured TTIs')
    axes[0].set_ylabel('Fraction at or above cost');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    x=np.arange(3)
    ratios=[physics.loc[rate,'candidate_ratio'] for rate in rates]
    misses=[physics.loc[rate,'MMSE_miss_ratio'] for rate in rates]
    walls=[float(load_wall[(load_wall.rate==rate)&(load_wall.arm=='bounded')].mean_ratio_full.iloc[0]) for rate in rates]
    axes[1].bar(x-.24,ratios,.24,label='Candidate evaluations/full',color='#087f8c')
    axes[1].bar(x,misses,.24,label='Independent MMSE/full',color='#6c757d')
    axes[1].bar(x+.24,walls,.24,label='Serial mean wall/full',color='#d17a22')
    axes[1].axhline(.4,color='#777',ls=':',lw=1,label='Mean-wall gate40%')
    axes[1].set_xticks(x,['x1','x1.4','x1.6']);axes[1].set_ylabel('Remaining fraction, same240 inputs')
    axes[1].set_ylim(0,1.05);axes[1].legend(fontsize=7)
    fig.savefig(out/'cost_tail_and_work.png',dpi=180);fig.savefig(out/'cost_tail_and_work.pdf');plt.close(fig)
    rnd=pd.read_csv(roots['gate']/'rounds.csv');phases=rnd.groupby('phase')[['fresh','required']].sum()
    report=f'''基础容量界精确惰性内层：等价、必评集合与高负载成本门
2026-11-04；保存输入seed41，高负载seed41/42；方法B预定41–43，因入口失败未执行。未使用最终确认seed。

结论
基础容量界已接入独立C++持久编辑求解器，12状态378轮624670候选的根/选择/上界/ΔD/有效精确缓存全过等价门。新精评{gate['bounded_scored']}次，当前证书所需集合{gate['required_set']}，UB序精评/必评={gate['UB_order_over_required']:.6f}。无排序模型训练。
公平串行保存状态墙钟：同架构full {wall.loc['full','mean_ms']:.3f}ms，cache {wall.loc['cached','mean_ms']:.3f}ms，bounded {wall.loc['bounded','mean_ms']:.3f}ms（full的{100*wall.loc['bounded','mean_ratio_full']:.2f}%）。负载门通过={has_B}，下文保留全部每TTI P95/最大值及负载变化，原生采集吞吐不当作调度速度。

1. 实现、等价与必评集合
上界只用BLER≥0：逐UE贡献round_even((w*min(Q,TBS))*1e6)，减去受影响UE根贡献后整数求和。自身count/min-layer/TBS和候选ΔD不调用MMSE。根维护排除各PRG后的最小layer，编辑只遍历稀疏受影响UE。旧persistent_kernel.cpp、physics_kernel.cpp及原二进制均不变，新库_bounded.so自己分配/释放Context。
编辑和界跨轮保留；变更PRG重建，共享UE依赖的精确值/界失效，从未精评条目持续标未评。堆逐轮用当前界重建，不携带旧轮节点；轮内按UB降序、键升序。有效精确缓存先提供已知最大值m和赢家键k；UB≥m或UB=m−1且键更小的未知候选继续精评，UB≤阈值直接排除。没有预算旁路。
付费独立完整整数参考对全部候选给出离线oracle，核对所有界均≥精确ΔJ、逻辑ΔD逐位相同，以及根/选择/终态逐位相同。每个当前有效精确值核对；未评条目的INT64_MIN标记从不当作标签。终态另作12次独立NumPy物理核验。审计不在正式计时中。
必评集合扣除本轮开始时有效精确缓存：可行且UB≥真实最大值，或UB=最大值−1且键小于最终赢家，再纳入必要的未评赢家；确认轮则是未知可行UB>阈值。这是当前最大值证书的所需集合，不能据此宣称所有可能键感知证明的信息论最优。
'''
    for phase,r in phases.iterrows():report+=f"{phase}: 新精评{int(r.fresh)}/必评{int(r.required)}={r.fresh/r.required if r.required else 1:.6f}\n"
    report+=f'''
计入20次初始化，实际7383次候选评分（full624690、原cache183079）。旧23738是未假设缓存复用的上限，实际更低来自有效缓存；不将两种节省比例相乘。当前UB排序达到该所需集合，161700条旧标签不为排序训练。任何有限优先级仍需同一证书才保证等价，但不能保证成本优于UB顺序。
独立profile平均：初始化{profile.native_setup_ms:.3f}ms、编辑生成/依赖维护{profile.table_generation_ms:.3f}ms、界更新{profile.bound_update_ms:.3f}ms、堆/证书{profile.heap_certificate_ms:.3f}ms、精确PHY {profile.sparse_phy_ms:.3f}ms；这些分项不含全部Python输入成本，不能当端到端总和。首扫、UE评分、MMSE请求/miss另列。候选数不等于MMSE逆矩阵次数。

2. 真实负载与成本尾部
倍数以rate0.8为基准：1.12=×1.4、1.28=×1.6。每负载41/42两seed，单进程单CPU255顺序运行；每条4000原生PFQ预热+4000稳定+4000测量+400TTL追踪，合计6条74400步。所有4000测量TTI的计时含当前输入适配、PFQ、Context建立/搜索/销毁，不含事后审计、状态落盘和env.step的GPU/原生FIFO校验。共享主机，保留所有计时离群点及process-time对照，无实时部署截止期承诺。
load | TTIs | mean/P95/max wall ms | 候选评分/full工作量 | 逐TTI评分比例P95/max
'''
    for r in costs:report+=f"×{r['load']:g} | {r['TTIs']} | {r['wall_mean_ms']:.3f}/{r['wall_P95_ms']:.3f}/{r['wall_max_ms']:.3f} | {100*r['candidate_work_ratio']:.3f}% | {100*r['candidate_fraction_P95']:.3f}%/{100*r['candidate_fraction_max']:.3f}%\n"
    report+='\n同输入、各80个状态×3次串行重复，完整墙钟对照：\n'
    for rate in (.8,1.12,1.28):
        z=load_wall[load_wall.rate==rate].set_index('arm')
        report+=f"rate{rate:g}: full {z.loc['full','mean_ms']:.3f}ms，cache {z.loc['cached','mean_ms']:.3f}ms，bounded {z.loc['bounded','mean_ms']:.3f}ms；bounded/full={100*z.loc['bounded','mean_ratio_full']:.3f}%。\n"
    report+='\n另付240次独立剖析；以下为native子项均值，不在公平计时中混算：\n'
    for rate,z in load_profile.iterrows():
        report+=f"rate{rate:g}: 生成/依赖维护{z.table_generation_ms:.3f}ms，界更新{z.bound_update_ms:.3f}ms，堆/证书{z.heap_certificate_ms:.3f}ms，精确PHY {z.sparse_phy_ms:.3f}ms。\n"
    report+='''高负载下评分集合变大，精确PHY与堆处理开销随之增加；排序已经等于当前证书所需集合，继续训练排序不能补足墙钟缺口。后续工程应降低单次精确PHY/元数据处理成本，再重过同一门，不用改变目标轨迹或放宽门槛来宣布成功。剖析v1首个端到端样本含不可变TBS表冷构建，v2在外预热后重跑；两版与额外成本保留，这不改变正式计时或原生最大值。
'''
    report+='\n同240输入独立oracle的PHY子工作量进一步说明评分数不能替代MMSE数：\n'
    for rate,r in physics.iterrows():
        report+=f"rate{rate:g}: candidate/full={100*r.candidate_ratio:.3f}%，UE精评/full={100*r.UE_ratio:.3f}%，独立MMSE miss/full={100*r.MMSE_miss_ratio:.3f}%。\n"
    report+='×1.6仍需完整搜索约81.6%的独立MMSE计算；大量候选可剪掉，并不等于相同幅度减少共享物理子计算。这是高负载墙钟收益收缩的重要成本边界。\n'
    report+=f'''
所有240个保存TTI另付费完整整数oracle，逐轮界与选择等价检查；与实际原生动作也逐位一致。再对同输入full/cache/bounded各3次旋转串行计时，以每状态中位数比较每负载墙钟。连续轨迹计时P95/最大值与稀疏保存状态的成对速度比是不同统计量，各自保留，不互换。
继承原门：各点评分均值≤full25%、P95≤40%，各负载同输入bounded墙钟均值≤full40%；本次B入口={has_B}。相对基准的界松紧变化见stress_analysis/native_costs.csv，达到绝对门不等于证明比例严格不随负载变化。无额外阈值或种子挑选。

3. 方法B：同一个精确内层的因果权重扫描入口
'''
    if has_B:
        frontier=pd.read_csv(roots['frontier']/'frontier.csv');fc=json.loads((roots['frontier']/'completion.json').read_text())
        report+='''入口过门后运行五个预先冻结系数κ=0/.25/.5/1/2、seed41/42/43。自身因果服务缺口权重乘以(1+κ*(1−HOL剩余TTL/400))，再归一到均值1；κ=0直接保留原权重位值。MCS、候选域、floor、内层/阈值/平局、仿真器与外部业务均相同。只改变权重，因此这里的差异归于该权重族，不称已训练AI收益。
HOL来自自身历史FIFO与当前正队列的只读视图，跳过本TTI将过期的旧头，当前新到达包有完整400TTI。特征不二次调用before()，避免清掉过期奖励；不读未来包结局。三条共同PFQ前缀、offered包与成熟测量cohort逐项配对一致。κ0复用stress的41/42，另新增13条161200步，构成15个质量点；所有点保留，没有结果后调网格/门槛。
每条测量窗口8001–12000到达包追踪至成熟，pending=0；总payload按完整成功包，过期按完整包，P95为成功包系统时延。聚合payload为三seed平均、expiry/P95按成熟包合池，最差UE为所有(seed,UE)的最大过期率。逐seed表与相对κ0服务门并列，不用聚合平均遮盖单seed损失。
κ | payload Mbps | expiry% | P95 ms | worstUE expiry% | 四目标Pareto | B逐seed不劣门
'''
        for r in frontier.itertuples():report+=f"{r.coefficient:g} | {r.payload_mbps:.3f} | {r.expiry_pct:.5f} | {r.P95_ms:.3f} | {r.worst_ue_expiry_pct:.5f} | {r.pareto4D} | {r.B_no_worse_all_seeds}\n"
        report+=f'''
四目标非支配系数={fc['Pareto_coefficients']}；非零系数逐seed满足B不劣门={fc['B_no_worse_nonzero_coefficients']}。B门保留payload/过期不劣、P95≤κ0+1ms，并保留最差UE/零服务保护；原历史宽容质量门另外报告。参数扫描不是梯度训练或泛化确认，不能从这组训练seed前沿宣称最终晋级。
每个新点4个保存状态，合计{fc['weighted_equivalence_states']}个加权状态、{fc['weighted_equivalence_candidates']}个候选另付完整整数oracle，界/根/选择/终态核对，验证不同权重仍保持精确内层；复用点的审计不重复收费。绘制frontier/ttl_frontier.png/pdf与逐seed四指标曲线。
'''
    else:report+='负载墙钟入口未过，依冻结顺序未启动正式TTL系数扫描。只保留κ=0/.25/.5/1/2、seed41–43的可审查方案和运行代码，没有前沿或训练结果。\n'
    report+='''
4. 边界与保留
11项定向测试通过（包括新库物理逐位一致、独立完整整数路径/上界/证书、7cell小例、因果FIFO边界和κ0精确控制）；另3条16步工程smoke共48步，其中κ1只验证接口/原生闭合，不形成成熟cohort质量点。训练/评估均为已暴露开发seed，梯度拟合和排序训练步数为0。有效默认注册表、161700旧标签及此前336个归档文件SHA保持。守卫cap4在seed44的原门失败不改；未来改变轨迹的方案仍应先做规范full/随机平局full的路径波动对照。B方案单独冻结，入口判定不放宽，不追溯修改旧结论。
工作账本分开保存新完整oracle、定时重复、剖析和原生策略调用；不把oracle或控制采集算免费。原生大日志与全部输入状态保留在deliverables，归档保存成熟cohort、逐UE/逐TTI成本、源码/二进制/协议及SHA。连续成本尾分布与候选/MMSE/墙钟对照见cost_tail_and_work.png/pdf；没有生成B前沿图。完整随机种子确认和默认替换没有执行。
'''
    (out/'report.txt').write_text(report)
    write_json(out/'work_ledger.json',dict(native=native_counts,native_steps=native_steps,
        gate_oracles=json.loads((roots['gate']/'work_ledger.json').read_text()),
        stress_oracles=json.loads((roots['stress_analysis']/'work_ledger.json').read_text()),
        profile_v1=json.loads((ROOT/'deliverables/bounded_load_profile_20261104_v1/completion.json').read_text()),
        profile_v2=json.loads((roots['profile']/'completion.json').read_text()),
        TTL_oracles=json.loads((roots['frontier']/'weighted_work_ledger.json').read_text()) if has_B else [],
        smoke=json.loads((ROOT/'deliverables/bounded_native_smoke_20261104_v1/completion.json').read_text()),
        timed_candidate_evaluations={name:{arm:int(value) for arm,value in
            pd.read_csv(roots[name]/'timing_raw.csv').groupby('arm').candidates.sum().items()}
            for name in ('gate','stress_analysis')},
        grid_gradient_steps=0,ranking_training_steps=0))
    write_json(out/'final_audit.json',dict(completed=True,tests=11,previous_files_unchanged=old_files,
        native_checked_steps=native_steps,mature_cohort_rows=cohort_rows,labels_sha256=sha(labels),
        registry_sha256=sha(registry),default_changed=False,ranking_training_steps=0,
        bounded_equivalence=True,bounded_wall_passed=gate['wall_passed'],stress_B_entry=has_B,
        UB_over_required=gate['UB_order_over_required'],TTL_frontier_completed=has_B))
    write_json(out/'archive_manifest.json',{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--tests',default='/tmp/bounded_20261104_tests.txt');main(p.parse_args())

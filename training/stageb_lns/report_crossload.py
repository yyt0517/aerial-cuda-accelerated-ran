"""Archive the completed NEW cross-load experiment and terminal engineering result."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from .physics import ROOT
from .analyze_bounded_stress import local
from .analyze_dpls_moves import write_json,write_csv,sha

def data(path):return json.loads(Path(path).read_text())

def main(args):
    campaign=ROOT/'deliverables/crossload_weights_20261104_v2';analysis=ROOT/'deliverables/crossload_analysis_20261104_v2'
    grid=ROOT/'deliverables/crossload_grid_analysis_20261104_v2';learning=ROOT/'deliverables/crossload_learning_20261104_v1'
    la=ROOT/'deliverables/crossload_learning_analysis_20261104_v1';gamma=ROOT/'deliverables/gamma_soa_final_20261104_v1'
    c,a,l,s=data(campaign/'completion.json'),data(analysis/'completion.json'),data(learning/'completion.json'),data(gamma/'completion.json')
    assert all(x['completed'] for x in (c,a,l,s)) and data(la/'completion.json')['completed']
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    for name,path in (('analysis',analysis),('grid_analysis',grid),('learning_analysis',la),('SoA_final',gamma)):
        shutil.copytree(path,out/name)
    for name,path in (('campaign',campaign),('learning',learning)):
        dest=out/name;dest.mkdir()
        for f in path.iterdir():
            if f.is_file():shutil.copy2(f,dest/f.name)
        if (path/'source').exists():shutil.copytree(path/'source',dest/'source')
    shutil.copytree(ROOT/'deliverables/crossload_weights_smoke_20261104_v1/source',out/'smoke_source')
    shutil.copy2(ROOT/'deliverables/crossload_weights_smoke_20261104_v1/completion.json',out/'smoke_completion.json')
    native=out/'native';native.mkdir();runs=c['runs']+c['reused_runs']+l.get('runs',[]);raw={};cohort_rows=0
    counts={name:dict(runs=0,steps=0,policy_decisions=0,candidate_evaluations=0,UE_evaluations=0,MMSE_requests=0,MMSE_misses=0)
        for name in ('new_fixed_and_literature','new_adaptive','reused_not_new')}
    for run in runs:
        d=local(run['directory']);dest=native/d.name;dest.mkdir()
        for name in ('summary.json','validation.json','reward_validation.json','launch.json','saved_states.json','per_ue.csv','cohort.csv.gz'):
            shutil.copy2(d/name,dest/name)
        with (d/'search.csv').open('rb') as source,gzip.open(dest/'search.csv.gz','wb') as target:shutil.copyfileobj(source,target)
        for f in ('policy_identity.json','chosen_coefficients.npy'):
            if (d/f).exists():shutil.copy2(d/f,dest/f)
        for f in d.iterdir():
            if f.is_file():raw[str(f.relative_to(ROOT))]=sha(f)
        val=data(d/'validation.json');assert val['passed'] and val['sources_unchanged']
        assert val['exact_native_actions']==val['fresh_input_checks']==val['exact_mcs_layer_tbs_dequeue']==12400
        assert data(d/'reward_validation.json')['passed']
        launch=data(d/'launch.json');assert launch['current_teacher_calls']==0 and not launch['ctbs_action_clipping']
        for f,digest in launch['source_sha256'].items():assert sha(local(f))==digest
        assert sha(local(launch['command'][0]))==launch['simulator_sha256']
        cohort_rows+=len(pd.read_csv(d/'cohort.csv.gz'))
        key='new_adaptive' if run in l.get('runs',[]) else ('new_fixed_and_literature' if run in c['runs'] else 'reused_not_new')
        work=counts[key];work['runs']+=1;work['steps']+=12400;t=pd.read_csv(d/'search.csv');work['policy_decisions']+=len(t)
        for col,k in (('candidates','candidate_evaluations'),('ue_evaluations','UE_evaluations'),('mmse_requests','MMSE_requests'),('mmse_misses','MMSE_misses')):
            work[k]+=int(t[col].sum())
    write_json(out/'native_raw_manifest.json',raw)
    files=[Path(__file__).with_name(n) for n in ('crossload_campaign.py','crossload_native.py','crossload_learning.py',
        'analyze_crossload.py','analyze_crossload_learning.py','report_crossload.py','crossload_README.rst','crossload_requirements.txt','test_crossload_weights.py',
        'gamma_soa.cpp','gamma_bridge.cpp','build_gamma_soa.py','audit_gamma_soa.py','_gamma_soa.so')]
    (out/'source').mkdir()
    for f in files:shutil.copy2(f,out/'source'/f.name)
    write_json(out/'source_manifest.json',{str(f.relative_to(ROOT)):sha(f) for f in files})
    shutil.copy2(Path(__file__).with_name('crossload_README.rst'),out/'README.rst')
    shutil.copy2(args.tests,out/'tests.txt');assert 'Ran 17 tests' in Path(args.tests).read_text() and '\nOK\n' in Path(args.tests).read_text()
    for name in ('crossload_weights_20261104_v2.log','crossload_analysis_20261104_v2.log','crossload_followups_20261104.log',
        'crossload_learning_analysis_20261104_v1.log','gamma_soa_gate_20261104_v1.log','gamma_soa_build_20261104.log','crossload_dependency_install_20261104.log'):
        f=Path('/tmp')/name
        if f.exists():shutil.copy2(f,out/name)
    prior_files=0
    for name in ('2026-11-03/persistent_cache_reconstruction','2026-11-03/certified_bounds',
        '2026-11-04/bounded_inner_load','2026-11-04/B_decoupled_and_tournament'):
        d=ROOT/'Doc/evidence'/name
        for f,digest in data(d/'archive_manifest.json').items():assert sha(d/f)==digest;prior_files+=1
    old=ROOT/'Doc/evidence/2026-11-04/B_decoupled_and_tournament'
    for f,digest in data(old/'source_manifest.json').items():assert sha(ROOT/f)==digest
    for f,digest in data(campaign/'source_manifest.json').items():assert sha(ROOT/f)==digest
    for f,digest in data(campaign/'input_manifest.json').items():assert sha(local(f))==digest
    for f,digest in data(analysis/'input_manifest.json').items():assert sha(local(f))==digest
    if l['trained']:
        for f,digest in data(learning/'source_manifest.json').items():assert sha(ROOT/f)==digest
        for f,digest in data(learning/'input_manifest.json').items():assert sha(local(f))==digest
    labels=ROOT/'deliverables/cached_budget_quality_20261103_v1/exact_dirty_labels.npz'
    registry=ROOT/'training/stageb_hgraph/current_algorithm.json';frozen=data(old/'final_audit.json')
    assert sha(labels)==frozen['labels_sha256'] and sha(registry)==frozen['registry_sha256']
    decision=data(grid/'learning_entry.json');aggregate=pd.read_csv(analysis/'aggregate.csv')
    oracles=pd.read_csv(analysis/'oracle_all_seeds.csv');train=pd.read_csv(analysis/'oracle_train_seeds.csv')
    near=pd.read_csv(analysis/'near_optimal.csv');lc=data(la/'completion.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    series={}
    for family,label in (('mlwdf_pf','M-LWDF-PF form'),('exp_pf','EXP-PF form')):
        series[label]=aggregate[aggregate.family==family].sort_values('rate')
    series['Fixed k2']=aggregate[(aggregate.family=='ttl')&(aggregate.coefficient==2)].sort_values('rate')
    rows=[]
    for r in train.itertuples():rows.append(aggregate[(aggregate.rate==r.rate)&(aggregate.family=='ttl')&(aggregate.coefficient==r.coefficient)].iloc[0])
    series['Per-load train oracle']=pd.DataFrame(rows).sort_values('rate')
    if l['trained']:series['Learned depth2 tree']=pd.read_csv(la/'aggregate.csv').sort_values('rate')
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,(key,label) in zip(axes.flat,[('payload_mbps','Payload Mbps, mean seed'),('expiry_pct','Mature expiry %'),
        ('P95_ms','Successful P95 ms'),('worst_ue_expiry_pct','Worst seed/UE expiry %')]):
        for name,table in series.items():ax.plot(table.rate/.8,table[key],marker='o',label=name)
        ax.set_xlabel('Load relative to rate0.8');ax.set_ylabel(label);ax.set_xticks([1,1.4,1.6]);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.savefig(out/'baseline_comparison.png',dpi=180);fig.savefig(out/'baseline_comparison.pdf');plt.close(fig)
    text='''新一版跨负载TTL网格、文献权重与条件学习；最后一次SoA工程门
2026-11-04；仅开发seed41/42/43。独立新协议，不是对旧κ网格补扫，旧报告不改。

1. 冻结与执行
κ={0,1,2,4,8} × 负载×1/×1.4/×1.6(rate0.8/1.12/1.28) × seed41–43，共45点。另有M-LWDF-PF/EXP-PF形式各9点，共63固定/文献质量点。13条完全相同旧轨迹核验复用，新跑32条κ＋18条文献，共50条620000原生步；复用161200步不重复算新增。
每条4000共同PFQ预热＋4000稳定＋4000测量＋400TTL追踪，成熟cohort无pending；共同offered业务、PFQ动作/输入前缀、cohort成员、原生动作/MCS/layer/TBS/dequeue和FIFO校验通过。科学臂全用同一个冻结_bounded.so、相同floor/初始化/整数目标/阈值/规范键，只改权重；工程库不替换。并发采集墙钟不作速度证据。

2. 文献定义与边界
M-LWDF由Andrews等2001年论文给出；EXP-PF形式见Biernacki/Tutschku与Ang等的实验。原始链接及参数见README.rst。这里移植的是权重形式，仍用本项目联合期望字节局部求解器，不称原论文RB调度器复现或继承其最优性保证，不将TTL加权写作新贡献。
TTL400TTI、PLR目标.01，a=−ln(.01)/400，HOL等待age=max(1,400−slack)。M-LWDF-PF权重=(a*age)/EMA；EXP-PF=exp((a*age−x)/(1+sqrt(x)))/EMA，x为活动UE的a*age均值，最终均值归一。EMA沿原成功服务.995/.005更新，分母下限1e−9。fresh/empty一槽age下限保持正值，空队列服务项为0。文献臂用未截断PF分母，κ族沿原clipped deficit，二者差异不能只归因紧迫项形状；未拟合论文参数。
EXP-PF的斜率本身随全局HOL统计变化，因此它已经是无需训练的状态相关基线；学习不能只优于固定κ就忽略这条对照。

3. 全部固定网格与文献点
payload为seed均值；expiry和成功包P95合池；最差UE过期率取所有(seed,UE)最大值。成功P95不含未成功包，逐seed与cohort保留，不以均值掩盖个别seed。
load | family | κ | payload Mbps | expiry % | success P95 ms | worstUE expiry % | Pareto among all forms
'''
    for r in aggregate.itertuples():text+=f'×{r.load:g} | {r.family} | {r.coefficient:g} | {r.payload_mbps:.3f} | {r.expiry_pct:.6f} | {r.P95_ms:.3f} | {r.worst_ue_expiry_pct:.6f} | {r.pareto_all_forms}\n'
    text+='''
4. 最优κ是否随负载变化
事前单一oracle规则：先保留payload均值≥本负载最佳均值99.7%的κ，再按最差UE过期率、总体过期率、P95、−payload、κ字典序选取；全部四目标Pareto集合另报。oracle只在这五κ/此网格/此指标优先级内，不称全局最优。
load | all3seed descriptive oracle | train41/42-only oracle | 4D Pareto κ
'''
    for r in oracles.itertuples():
        tr=train[train.rate==r.rate].iloc[0];g=aggregate[(aggregate.rate==r.rate)&(aggregate.family=='ttl')]
        front=g[g.pareto4D==True].coefficient.tolist()
        text+=f'×{r.load:g} | {r.coefficient:g} | {tr.coefficient:g} | {front}\n'
    text+=f'''\n训练集oracle随负载不同={decision['train_oracle_varies']}；高负载存在所有固定κ均剩余过期={decision['all_fixed_high_load_residual_expiry']}；所有负载共同近优κ={decision['universal_near_optimal_coefficients']}。
近优是事前逐坐标规则：payload均值≥最佳99.7%且逐seed≥99.4%；expiry≤最小+.1pp，P95≤最小+1ms，最差UE≤最小+2pp。各坐标可能冲突，无共同近优点不等于能实现那个理想向量。负载间最优不同或残余损失只提供开发信号，不证明状态自适应一定有效，也不证明剩余过期可由学习消除。
每条轨迹内负载固定，本轮没有负载切换实验；工况间最优不同不等于已经识别同一工况内逐TTI的最优κ。全部负载若有共同近优固定κ，也会阻止拟合。
本次三seed聚合四指标上，×1的κ8支配其余κ，×1.6的κ0支配其余κ；×1.4存在Pareto取舍。两端的偏好变化并非只由99.7%筛选制造，但聚合支配关系不外推为每个seed都支配或其它业务普遍成立。
'''
    text+='\n5. 条件低维学习\n'
    if l['trained']:
        training=data(learning/'training.json');pairs=pd.read_csv(la/'paired.csv');adapt=pd.read_csv(la/'aggregate.csv')
        text+=f'''45点固定网格完成后才启动一次预声明的深度2决策树，min_leaf256，无超参扫。标签仅来自41/42每负载固定oracle；训练数据仅新固定轨迹的4001–8000稳定期，按负载均衡，共{training['rows']}行，{training['nodes']}节点/{training['leaves']}叶；seed43不参与标签/拟合。6个输入为队列均值/P95、活动比例、HOL紧迫均值/P95和历史成功服务EMA均值，没有真实负载、seed、未来结局或oracle索引输入。固定oracle标签是负载工况标签，不是自适应队列状态的逐TTI最优动作标签；只称工况模仿。
新增9条111600原生步，以同窗口/同内层闭环比较固定κ2、训练集固定oracle；三seed描述性oracle另作参照但从不回流标签。seed43已暴露，仅开发检查，不称最终未见泛化。模型冻结后不重新拟合/挑运行。
首次拟合后曾因NumPy int64元数据序列化失败中断，修复记录层后复用同一policy.joblib，SHA与fit_resume.json核对，未二次fit、无失败前新增原生步；旧代码/部分元数据/失败日志保留。
load | learned payload Mbps | expiry % | P95 ms | worstUE % | allseed no-worse vsκ2 | vs train oracle | vs descriptive oracle
'''
        for r in adapt.itertuples():text+=f'×{r.rate/.8:g} | {r.payload_mbps:.3f} | {r.expiry_pct:.6f} | {r.P95_ms:.3f} | {r.worst_ue_expiry_pct:.6f} | {r.no_worse_all_seeds_vs_fixed2} | {r.no_worse_all_seeds_vs_train_oracle} | {r.no_worse_all_seeds_vs_descriptive_oracle}\n'
        text+=f'''\n全部负载通过κ2不劣门={lc['no_worse_all_loads_vs_fixed2']}；通过训练oracle不劣门={lc['no_worse_all_loads_vs_oracle']}；通过描述性oracle不劣门={lc['no_worse_all_loads_vs_descriptive_oracle']}。对M-LWDF-PF/EXP-PF全部负载不劣={lc['no_worse_all_loads_vs_mlwdf_pf']}/{lc['no_worse_all_loads_vs_exp_pf']}。逐seed门沿旧B：payload/expiry不劣、P95≤控制+1ms、最差UE≤max(控制+2pp,5%)，无新增零服务/TTL长间隔UE。失败逐项留在learning_analysis/paired.csv，不用平均替代。训练行高度相关，不当独立轨迹重复，不以分类准确率证明服务收益。
'''
        phases=[];expected_k={r['rate']:r['coefficient'] for r in decision['train_only_oracle']}
        for run in l['runs']:
            path=local(run['directory'])/'chosen_coefficients.npy';values=np.load(path);k=expected_k[run['rate']]
            phases.append(dict(rate=run['rate'],seed=run['seed'],source_sha256=sha(path),
                settling_nonoracle=int(np.sum(values[:4000]!=k)),measured_nonoracle=int(np.sum(values[4000:8000]!=k)),
                followup_nonoracle=int(np.sum(values[8000:]!=k)),measured_switches=int(np.sum(values[4001:8000]!=values[4000:7999])),
                nonoracle_TTIs=(np.flatnonzero(values!=k)+4001).tolist()))
        write_csv(out/'coefficient_choice_phases.csv',phases)
        assert sum(r['measured_switches']+r['measured_nonoracle'] for r in phases)==0
        assert sum(r['settling_nonoracle'] for r in phases)==14 and sum(r['followup_nonoracle'] for r in phases)==0
        text+='\n机制检查：全部测量窗口均无κ切换。8/9条全8400策略TTI都选择对应训练oracle，动作轨迹与固定臂逐位相同；唯一例外为×1.6 seed43，在稳定期TTI5842–5877之间14次选择κ2，其余选择κ0，测量期4000步全κ0。该点payload较固定κ0约+3.50%、过期约−2.675pp，收益在扰动后延续。它是一处开发信号，当前无法区分有用状态反馈与搜索/服务路径扰动；没有同量非学习扰动对照，不能称已学到持续动态调权。跨负载选择的动机已成立，但稳健学习新增贡献仍未证明，且低负载M-LWDF形式是更强对照。\n'
    else:text+=f"未训练：{l['reason']}。只有高负载残余损失而没有不同工况标签，不拟合恒定分类器冒充学习增益。\n"
    text+=f'''\n6. 最后SoA/SIMD尝试与停止工程线
独立库以SoA布局跨8个键执行AVX-512，TTI完整2448键(含未请求键)全部计算。实现pivoted LDLT、三角求解、伴随乘积和余子式求逆；无Woodbury/倒数近似，FMA收缩及fast-math关。原Eigen参考编译单元保留旧架构标志。显式递推/公式不等于复现Eigen内部packet及归约顺序，必须逐位门验证。
252状态2467584个Γ标量中，{s['mismatch_scalars']}个不逐位一致，全部{s['mismatch_states']}状态有差异；最大绝对差{s['max_abs']:.12g}、最大相对差{s['max_rel']:.12g}。生成{s['AVX512_instruction_lines']}行AVX512相关汇编，FMA指令{s['FMA_instructions']}，不是未向量化空跑。新增{s['new_MMSE_problems']}个MMSE问题(参考/候选两份全表)，未进入求解器或任何科学臂，未用容差替代位级门，也没有新的等价/速度结论。
本实现尝试失败，不证明所有SoA实现无效。按用户本轮止损要求，40%工程追逐到此结束，不继续树/Woodbury/数值规范变更。保留已有精确native适配＋直接Γ＋树在×1/×1.4/×1.6的33.90%/41.23%/46.38%同输入均时曲线，分母为同架构native full，输入来自旧κ0缺口权重轨迹，不保证本轮各权重臂成本；旧连续TTI P95/最大值仍有效，未新增SoA部署尾分布。

7. 审计、成本与保留
50个新增固定/文献点各4状态，共{a['weighted_states']}个加权状态、{a['weighted_candidates']}候选，另付独立完整整数oracle核验界/根/缓存/选择/终态，与原生实际动作一致。条件学习若启动，另{data(la/'completion.json').get('weighted_states',0)}个保存状态付同样oracle。复用点的旧核验不重复算新增。
17项定向测试通过；候选/UE/MMSE账本按新原生、复用、oracle和SIMD测试分列，树拟合另记行数/节点/一次fit，梯度训练和旧排序标签拟合均0。正式策略调用/审计不当免费，单元测试微例不纳入正式轨迹工作总数。
'''
    text+=f'旧标签、默认注册表、旧源/库和{prior_files}个既有归档文件SHA不变；cap4 seed44原失败不改，没有默认替换或最终确认seed。大动作/业务/原生bin/输入状态保留于deliverables，SHA随native_raw_manifest及各input_manifest；本归档search.csv.gz为原search.csv无损压缩。\n'
    (out/'report.txt').write_text(text)
    write_json(out/'work_ledger.json',dict(native=counts,smoke=data(out/'smoke_completion.json'),
        weighted_oracles=data(analysis/'weighted_work_ledger.json'),adaptive_oracles=data(la/'weighted_work_ledger.json') if l['trained'] else [],
        SoA=s,tree_training=data(learning/'training.json') if l['trained'] else dict(fit_calls=0),
        gradients=0,ranking_label_training=0,tests='17 small behavioral cases, not included in formal native totals'))
    write_json(out/'final_audit.json',dict(completed=True,tests=17,quality_points=len(runs),
        new_native_steps=c['new_native_steps']+l['new_native_steps'],reused_steps_not_new=161200,mature_cohort_rows=cohort_rows,
        previous_files_unchanged=prior_files,labels_sha256=sha(labels),registry_sha256=sha(registry),old_solver_unchanged=True,
        default_changed=False,learning_trained=l['trained'],learning_all_no_worse_gates=lc.get('no_worse_all_loads_vs_oracle',False),
        learning_any_strict_improvement=lc.get('strict_improvement_any_point_vs_train_oracle',False),
        SoA_bitwise_passed=False,engineering_stopped=True,engineering_cost_curve=[.3390066000374841,.4122752089726274,.4637749941500488],
        final_confirmation_seeds_used=False))
    write_json(out/'archive_retry.json',dict(previous_partial='deliverables/crossload_report_partial_20261104_v1',
        reason='diagnostic reader was started before extra phase columns were added; supplement phases directly from immutable npy',
        new_native_steps_due_to_retry=0,new_PHY_due_to_retry=0,refitted=False))
    write_json(out/'archive_manifest.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--tests',default='/tmp/crossload_all_tests_20261104.txt');main(p.parse_args())

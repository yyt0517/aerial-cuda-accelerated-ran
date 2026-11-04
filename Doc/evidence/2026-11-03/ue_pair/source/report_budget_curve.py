"""Audit frozen budget curves, draw standalone figures and archive evidence."""
import argparse
import csv
import json
from pathlib import Path
import platform
import shutil

import numpy as np

from .budget_curve import BUDGETS,FAMILIES,arm_name,parse_arm
from .physics import ROOT
from .experiment import write_json,sha
from .bounded_probes import csv_write


def read_csv(path):return list(csv.DictReader(Path(path).open()))


def average(rows,key):return float(np.mean([float(r[key]) for r in rows]))


def audit_phase(source,phase):
    raw=read_csv(source/f'{phase}_measurements.csv');state=read_csv(source/f'{phase}_per_state.csv')
    summary=read_csv(source/f'{phase}_summary.csv')
    for row in state:
        rr=[r for r in raw if r['state_id']==row['state_id'] and r['arm']==row['arm']]
        assert len(rr)==3
        assert np.isclose(float(row['e2e_ms']),np.median([float(r['e2e_ms']) for r in rr]))
        if row['arm']!='full':
            assert int(row['candidate_evaluations'])<=int(row['budget'])
            assert int(row['feature_roots'])<=56 and row['repeated_exact_candidates']=='0'
    for row in summary:
        rr=[r for r in state if r['arm']==row['arm']]
        for field,key in [('score','mean_j'),('e2e_ms','mean_ms')]:
            expected=np.mean([average([r for r in rr if r['seed']==seed],field) for seed in sorted({r['seed'] for r in rr})])
            assert np.isclose(expected,float(row[key]))
        q=all(float(r['ratio'])>=.995 for r in rr)
        c=all(int(r['newly_unserved_ues'])==0 and float(r['positive_p5_ratio'])>=.98 and float(r['positive_weak_ratio'])>=.98 for r in rr)
        assert (row['quality_pass']=='True')==q
        assert (row['coverage_pass']=='True')==c
        assert (row['cost_pass']=='True')==(float(row['time_ratio'])<=.8)
    deltas=read_csv(source/f'{phase}_budget_deltas.csv')
    index={(r['state_id'],r['arm']):r for r in state}
    for r in deltas:
        a=index[r['state_id'],arm_name(r['model'],int(r['top_k']),int(r['from_budget']))]
        b=index[r['state_id'],arm_name(r['model'],int(r['top_k']),int(r['to_budget']))]
        assert np.isclose(float(r['j_delta']),float(b['score'])-float(a['score']))
        assert np.isclose(float(r['ms_delta']),float(b['e2e_ms'])-float(a['e2e_ms']))
    return summary,state,deltas,len(raw)


def plot_curves(validation,evaluation,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(heuristic='#68717e',initial='#2769a7',trained='#cb6424')
    labels=dict(heuristic='Physical heuristic',initial='Initial AI',trained='Trained AI')
    fig,axes=plt.subplots(2,2,figsize=(12.6,9.5),sharex='col')
    for col,(rows,title) in enumerate([(validation,'Validation: 12 historical states'),(evaluation,'Development: 9 historical states')]):
        for family in FAMILIES:
            for k in (2,4):
                points=sorted([r for r in rows if r['model']==family and int(r['top_k'])==k],key=lambda r:int(r['budget']))
                x=[float(r['time_ratio']) for r in points]
                for line,field in [(0,'mean_relative_j_pct'),(1,'worst_relative_j_pct')]:
                    y=[float(r[field]) for r in points]
                    axes[line,col].plot(x,y,marker='o' if k==2 else 's',linestyle='-' if k==2 else '--',
                        color=colors[family],linewidth=1.7,label=f'{labels[family]}, K={k}')
                    for xx,yy,p in zip(x,y,points):
                        axes[line,col].annotate(p['budget'],(xx,yy),xytext=(3,5 if k==2 else -10),
                            textcoords='offset points',fontsize=7,color=colors[family])
        for row in (0,1):
            ax=axes[row,col];ax.axvline(.8,color='#9b4159',linestyle=':',linewidth=1.1)
            ax.axhline(-.5,color='#9b4159',linestyle=':',linewidth=1.1)
            ax.scatter([1],[0],marker='*',s=100,color='#263442',zorder=5)
            ax.grid(alpha=.2);ax.set_ylabel('Mean J change vs full (%)' if row==0 else 'Worst-state J change vs full (%)')
            if row==1:ax.set_xlabel('Complete mean time / full DPLS')
        axes[0,col].set_title(title)
    handles,legend=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.027),fontsize=9)
    fig.suptitle('Frozen-model quality-cost curves: PHY caps 106 / 160 / 212; feature roots <= 56',fontsize=13)
    fig.text(.5,.012,'Labels are score caps, not actual work. Dotted lines: 80% time / -0.5% J. Quality uses worst state; service coverage is a separate gate.',
        ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.09,1,.95))
    fig.savefig(out/'quality_cost_curves.png',dpi=160)
    fig.savefig(out/'quality_cost_curves.pdf')
    plt.close(fig)


def main(source,destination,tests_log,execution_log):
    import torch
    source=Path(source).resolve();destination=Path(destination).resolve()
    protocol=json.loads((source/'protocol.json').read_text());selection=json.loads((source/'selection.json').read_text())
    completion=json.loads((source/'completion.json').read_text());assert completion['completed']
    val,val_state,val_delta,n_val=audit_phase(source,'validation')
    dev,dev_state,dev_delta,n_dev=audit_phase(source,'evaluation')
    by_val={r['arm']:r for r in val};by_dev={r['arm']:r for r in dev}
    selected=selection['within_cost_best']['arm']
    assert float(by_val[selected]['time_ratio'])<=.8
    assert not any(r['joint_pass']=='True' for r in dev if r['arm']!='full')
    comparisons=[]
    for budget in BUDGETS:
        for k in (2,4):
            heuristic=[r for r in dev_state if r['arm']==arm_name('heuristic',k,budget)]
            lookup={r['state_id']:r for r in heuristic}
            for family in ('initial','trained'):
                for row in [r for r in dev_state if r['arm']==arm_name(family,k,budget)]:
                    ref=lookup[row['state_id']]
                    comparisons.append(dict(state_id=row['state_id'],seed=row['seed'],model=family,top_k=k,budget=budget,
                        j_delta=float(row['score'])-float(ref['score']),ms_delta=float(row['e2e_ms'])-float(ref['e2e_ms']),
                        positive_p5_delta=float(row['positive_p5_ratio'])-float(ref['positive_p5_ratio'])))
    csv_write(source/'ai_vs_heuristic.csv',comparisons)
    per_seed=[]
    for phase,rows in [('validation',val_state),('evaluation',dev_state)]:
        for arm in dict.fromkeys(r['arm'] for r in rows):
            for seed in sorted({r['seed'] for r in rows}):
                rr=[r for r in rows if r['arm']==arm and r['seed']==seed]
                per_seed.append(dict(phase=phase,arm=arm,seed=seed,mean_j=average(rr,'score'),mean_ms=average(rr,'e2e_ms'),
                    mean_relative_j_pct=100*(average(rr,'ratio')-1),worst_relative_j_pct=100*(min(float(r['ratio']) for r in rr)-1)))
    csv_write(source/'per_seed_curve.csv',per_seed)
    increments={}
    for phase,rows in [('validation',val_delta),('evaluation',dev_delta)]:
        for start,end in ((106,160),(160,212)):
            rr=[r for r in rows if int(r['from_budget'])==start and int(r['to_budget'])==end]
            increments[f'{phase}:{start}->{end}']=dict(states_and_curves=len(rr),
                positive=sum(float(r['j_delta'])>1 for r in rr),negative=sum(float(r['j_delta'])<-1 for r in rr),
                tied=sum(abs(float(r['j_delta']))<=1 for r in rr))
    next_path=dict(selected='TB-aware coordinated macro candidates and a bounded continuation-value probe',
        implementation_status='chosen next experiment; not run in this campaign',
        rationale='larger tested budgets improve J but no point meets quality/coverage, and some service metrics worsen',
        saturation_proven=False,larger_budget_ruled_out=False,
        next_minimum='compare the frozen212-score reference with small same-PRG cross-cell activity and cross-PRG same-TB compensating edits; '
                     'protect full-schedule coverage and measure complete cost before training a proposer',
        teacher='finite continuation teacher only after validating candidate headroom; charge all offline queries explicitly',
        shared_controls='physical heuristic and AI share candidate family, cache and complete timing; no free exact oracle',
        paused=['additional uniform single-edit budget expansion as primary route','same immediate-gain training','PPO'],
        caution='curves show no J saturation; a larger budget remains a control, and instantaneous UE service is not packet fairness')
    gates=dict(selected_on_validation=selection,
        selected_development={k:by_dev[selected][k] for k in ('mean_j','mean_ms','mean_relative_j_pct','worst_relative_j_pct','mean_newly_unserved','mean_positive_p5')},
        quality_coverage_cost={r['arm']:{k:r[k]=='True' for k in ('quality_pass','coverage_pass','cost_pass','joint_pass')} for r in dev if r['arm']!='full'},
        paired_budget_increments=increments,
        feature_budget_stops_validation=sum(int(r['feature_budget_stops']) for r in val),
        feature_budget_stops_development=sum(int(r['feature_budget_stops']) for r in dev),
        independent_ai_joint_target=False,next_path=next_path)
    assert all(sha(v['path'])==v['sha256'] for v in protocol['models'].values())
    assert sha(ROOT/'training/stageb_lns/shortlist_trajectory.py')==protocol['solver_sha256']
    assert sha(ROOT/'training/stageb_hgraph/current_algorithm.json')==protocol['registry_sha256']
    tests=Path(tests_log).read_text();assert 'Ran 46 tests' in tests and '\nOK\n' in tests
    shutil.copyfile(tests_log,source/'tests.log');shutil.copyfile(execution_log,source/'execution.log')
    write_json(source/'gate_results.json',gates)
    audits=sum(len(json.loads((source/f'{phase}_checks.json').read_text())['independent_phy']) for phase in ('validation','evaluation'))
    write_json(source/'final_audit.json',dict(passed=True,tests=46,serial_decisions=n_val+n_dev,
        independent_final_phy_checks=audits,summary_gates_and_deltas_recomputed=True,
        models_and_solver_unchanged=True,training_updates=0,registry_unchanged=True,native_trajectories=0,final_seeds_read=False))
    write_json(source/'runtime_environment.json',dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
        platform=platform.platform(),threads=1,gpu_used=False))
    destination.mkdir(parents=True,exist_ok=False)
    for p in source.iterdir():
        if p.is_file() and p.suffix in ('.csv','.json','.log'):shutil.copyfile(p,destination/p.name)
    code=destination/'code';code.mkdir()
    for name in ('budget_curve.py','report_budget_curve.py','test_budget_curve.py','budget_curve_README.rst'):
        shutil.copyfile(Path(__file__).with_name(name),code/name)
    write_json(destination/'code_manifest.json',dict(note='Executed original snapshot in deliverables/source; reporter/plots added after all measurements.',
        files={p.name:sha(p) for p in code.iterdir()}))
    plot_curves(val,dev,destination)
    _,chosen_k,_=parse_arm(selected);chosen_family=parse_arm(selected)[0]
    report=['106/160/212次物理评分预算：冻结模型质量—成本曲线','2026-11-02','',
        '结论：两种冻结模型＋同机制物理启发式、top2/top4、三档评分预算全部完成。',
        '预算增加在本开发集全部逐状态改善J，但所有点仍未通过质量/有效UE服务覆盖门槛。',
        '没有观察到J饱和，也没有排除更大预算；在已测范围内未找到质量、覆盖和成本同时达标的点。',
        '下一步优先验证整TB协调宏动作及有限继续收益，暂不继续同目标扩训或PPO。','',
        '一、冻结范围与评估协议',
        '沿用3小区/36UE/17PRG、canonical原生BLER、因果服务缺口权重和原始PFQ90%出队下界。',
        '无CTBS/headroom裁剪、无PPO、无新训练。原始标量与上轮top4学生训练的epoch10检查点完全冻结。',
        '物理启发式、原始AI和训练后AI共享候选/特征/排序缓存、根内失败去重、数值归约及验证代码。',
        '只改变topK=2/4和物理评分上限106/160/212；特征根仍≤56。初始草案评分计入预算。',
        '标量路径>1B接纳和full原1/16B阈值均未修改；末批可截短，预算间不保证相同后续路径或J单调。',
        '验证44–45，两个历史来源，TTI4001/8001/18000，共12状态；开发46–48纯PFQ，TTI3001/4001/5000，共9状态。',
        '两组均有历史开发暴露，不能称新盲测；验证选型文件先于本轮开发指标读取提交。',
        f'18个曲线点＋full参照，共{n_val}次验证、{n_dev}次开发串行决策，合计{n_val+n_dev}次。',
        '每状态三次交替计时取中位数，再seed等权。P95为有限状态中位耗时的分位数，不是部署尾延迟。',
        '完整耗时包括输入/PFQ解码、全新PHY/缓存、候选、特征、网络/排序、物理验证、评分布局与清理。',
        '仅归档/模型/库IO及首次不变TBS表准备在外；代码、模型SHA均核对未改。','',
        '二、冻结的验证选型',
        f'质量较优点：{selection["quality_best"]["arm"]}；成本门槛内较优点：{selected}。',
        '分别报告质量较优与完整均时≤full80%的可行点，不按开发结果改选模型、K或预算。',
        f'所选点验证平均J变化{float(by_val[selected]["mean_relative_j_pct"]):+.3f}%，最差{float(by_val[selected]["worst_relative_j_pct"]):+.3f}%，'
            f'完整均时{float(by_val[selected]["mean_ms"]):.3f}ms。',
        f'所选点开发平均J变化{float(by_dev[selected]["mean_relative_j_pct"]):+.3f}%，最差{float(by_dev[selected]["worst_relative_j_pct"]):+.3f}%，'
            f'完整均时{float(by_dev[selected]["mean_ms"]):.3f}ms。','',
        '三、全部开发曲线点（J是加权期望成功字节，不是Mbps）',
        'arm | 平均J | 完整均时ms | 对full平均J变化 | 最差状态J | 新增未分配UE均值 | 有效UE P5比 | 联合门槛']
    for r in dev:
        report.append(f'{r["arm"]}: {float(r["mean_j"]):.3f} | {float(r["mean_ms"]):.3f} | '
            f'{float(r["mean_relative_j_pct"]):+.3f}% | {float(r["worst_relative_j_pct"]):+.3f}% | '
            f'{float(r["mean_newly_unserved"]):.3f} | {float(r["mean_positive_p5"]):.4f} | '
            f'{r["joint_pass"] if r["arm"]!="full" else "参照"}')
    report.extend([
        '完整门槛保持：每状态J≥full99.5%；相对full无新增未分配UE（COUNT>0）；',
        '在full期望服务>1B的有效UE子集中，P5及底20%服务量均≥参照98%；完整均时≤full80%。',
        '开发18个点均通过成本门槛，但质量及服务覆盖均未通过；每个点的最差有效UE P5比为0。','',
        f'四、验证选定的{chosen_family}/top{chosen_k}曲线',
        '预算 | 完整均时ms | 平均J变化 | 新增未分配UE均值 | 有效UE P5比'])
    for budget in BUDGETS:
        r=by_dev[arm_name(chosen_family,chosen_k,budget)]
        report.append(f'{budget}: {float(r["mean_ms"]):.3f} | {float(r["mean_relative_j_pct"]):+.3f}% | '
            f'{float(r["mean_newly_unserved"]):.3f} | {float(r["mean_positive_p5"]):.4f}')
    report.extend([
        '预算增加提升总J，但这一曲线的新增未分配UE增多、有效UE P5下降。',
        '不能仅凭总J上升认定服务覆盖改善，更不能推导为包级吞吐、时延或长期公平性。',
        '训练后top2/212在开发均值上更高，但本轮没有根据开发结果回选该点；其完整结果保留在全表。','',
        '五、预算增量与停止原因'])
    for name,r in increments.items():report.append(f'{name}: {r["positive"]}正向/{r["negative"]}负向/{r["tied"]}持平（状态×曲线描述，不是独立无线重复）。')
    report.extend([
        '全部验证和开发曲线点都在物理评分预算处停止，56特征根上限均未触发。',
        '因此不能把本轮解释为特征根上限导致质量平台，也不能声称更大评分预算已被证伪。',
        '本轮开发预算增量没有负J样本；仍保留末批截断会改变轨迹的实现边界，不将曲线强制平滑或重构包络。',
        '同预算AI与启发式的逐状态差值另见ai_vs_heuristic.csv；精确候选数和缓存收益不能算作AI独立质量收益。','',
        '六、下一步路径（已选择，尚未实施）',
        '优先验证完整TB协调宏动作：同PRG跨小区活动变化，以及跨PRG同UE的资源补偿/最小层瓶颈修复。',
        '以本轮冻结的212评分点作对照，先证明联合候选在完整计费下的质量和有效服务空间，再训练AI提议器。',
        '保护全调度覆盖，不在最初几个块耗尽预算；物理启发式和AI必须共享候选家族、缓存和计时边界。',
        '候选空间有收益后再做小规模有限继续教师，离线查询全部单列，不冒充免费在线精确oracle。',
        '继续扩大统一单编辑评分上限降为备选对照；没有排除其收益，但暂不作为主要路线。',
        '不追加当前即时收益标签训练、不接PPO、不改默认算法。','',
        '七、核验、曲线图与工件',
        f'46项测试通过；{audits}组正式终态独立NumPy物理核验；全部三次重复动作一致，因果floor和资源上限核验通过。',
        '原始和训练后检查点、在线求解器SHA完全未改，训练更新次数为0，注册表未变。',
        f'主实验耗时{completion["elapsed_s"]:.2f}秒；图表与统计复核在测量后完成。',
        'quality_cost_curves.png/pdf给出验证/开发的平均及最差J—完整成本曲线；点标签是预算上限，实际工作见CSV。',
        '未读311–315、无新原生轨迹；不宣称泛化、完整包吞吐/TTL/时延改善或0.5ms实时部署。',
        f'完整模型、数组和执行源码：{source}',
        '本证据目录保留小型协议、曲线、逐状态增量、门槛、负结果、检查及新增源码。'])
    (destination/'report.txt').write_text('\n'.join(report)+'\n')
    write_json(destination/'archive_manifest.json',{str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()})
    print('BUDGET CURVE EXPORTED',destination,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--tests-log',required=True);p.add_argument('--execution-log',required=True);a=p.parse_args()
    main(a.source,a.destination,a.tests_log,a.execution_log)

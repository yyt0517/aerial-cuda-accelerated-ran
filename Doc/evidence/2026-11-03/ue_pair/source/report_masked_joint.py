"""Package both successful engineering checks and the failed quality result."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np

from .experiment import write_json,sha,snapshot_sources


def representative_rows(frame,arm):
    import pandas as pd
    return pd.DataFrame([g.loc[g.e2e_ms.sort_values().index[len(g)//2]]
                         for _,g in frame[frame.arm==arm].groupby('state_id')])


def timing_parts(frame):
    r=representative_rows(frame,'masked_joint').mean(numeric_only=True)
    parts=dict(context=r.conversion_ms+r.phy_initialization_ms+r.feature_setup_ms,
               masks=r.mask_ms,neural=r.model_encoder_ms+r.model_head_ms+r.model_decoder_ms,
               verification=r.verification_ms)
    parts['other']=r.e2e_ms-sum(parts.values())
    return parts


def main(args):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(args.training).resolve();evaluation=root/'evaluation_mask_reuse'
    test_log=Path(args.tests).resolve()
    test_text=test_log.read_text()
    if 'Ran 25 tests' not in test_text or not test_text.rstrip().endswith('OK'):
        raise ValueError('The final 25-test suite must pass before packaging')
    evidence=Path(args.evidence).resolve();evidence.mkdir(parents=True,exist_ok=False)
    train=json.loads((root/'training.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    diag=json.loads((root/'diagnosis/summary.json').read_text())
    summary=pd.read_csv(evaluation/'summary.csv').set_index('arm')
    escape=pd.read_csv(evaluation/'escape_summary.csv')
    before=pd.read_csv(root/'evaluation/measurements.csv');after=pd.read_csv(evaluation/'measurements.csv')
    old=pd.read_csv(root/'evaluation/per_state.csv').set_index(['state_id','arm'])
    new=pd.read_csv(evaluation/'per_state.csv').set_index(['state_id','arm'])
    checks=['score','candidate_evaluations','ue_evaluations','mmse_requests','mmse_misses']
    np.testing.assert_allclose(new.loc[old.index,checks],old[checks],rtol=0,atol=1.e-7)
    a=np.load(root/'evaluation/actions.npz');b=np.load(evaluation/'actions.npz')
    for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    old_escape=pd.read_csv(root/'evaluation/escape.csv').set_index(['state_id','root','arm'])
    new_escape=pd.read_csv(evaluation/'escape.csv').set_index(['state_id','root','arm'])
    np.testing.assert_allclose(new_escape.loc[old_escape.index,['score','candidates']],
                               old_escape[['score','candidates']],rtol=0,atol=1.e-7)
    parts_before,parts_after=timing_parts(before),timing_parts(after)
    before_time=representative_rows(before,'masked_joint').e2e_ms.mean()
    n=summary.loc['masked_joint'];n79=summary.loc['neural79_cached'];phy=summary.loc['physical4']
    caps=summary.loc['neural79_cap73']
    speed79=float(n79.mean_e2e_ms/n.mean_e2e_ms);speedphy=float(phy.mean_e2e_ms/n.mean_e2e_ms)
    assessment=dict(engineering_checks_passed=True,tests_passed=25,quality_and_speed_improvement_demonstrated=False,
             strict_local_escapes=int(escape[(escape.root=='strict_greedy')&(escape.arm=='masked_joint')].escapes.iloc[0]),
             physical_beam_stop_escapes=int(escape[(escape.root=='physical4_stop')&(escape.arm=='masked_joint')].escapes.iloc[0]),
             actual_speed_ratio_vs_79=speed79,actual_speed_ratio_vs_physical4=speedphy,
             relative_J_vs_physical4_pct=float(n.mean_relative_J_pct),
             optimization_identical_actions_and_work=True,optimization_identical_escape_results=True,
             timing_before_ms=float(before_time),timing_after_ms=float(n.mean_e2e_ms),
             timing_parts_before=parts_before,timing_parts_after=parts_after,
             baseline_79_unchanged=sha(Path(protocol['frozen79']))==protocol['frozen79_sha256'],
             no_active_algorithm_replacement=True,scope='27 historical frozen states, three seeds; no new native closed-loop rollout')
    write_json(root/'assessment.json',assessment)
    display=summary[['score','mean_relative_J_pct','mean_e2e_ms','p95_state_median_ms','candidates','wins','losses']]
    report=f'''MACSIM/MaskCO启发的独立联合修复第一版，2026-10-30

结论：实现、训练和冻结状态验证完成；本轮没有实现“接近或超过强物理局部搜索，同时降低完整推理耗时”。新方法快，但质量明显退化，不能替换现有算法。所有实验使用独立入口，未修改当前算法注册表。

从相同PFQ+CTBS草案出发，新方法完整推理均时{n.mean_e2e_ms:.3f}ms；缓存版79维修复为{n79.mean_e2e_ms:.3f}ms；强4位置物理beam为{phy.mean_e2e_ms:.3f}ms。对应耗时比分别为{speed79:.2f}倍和{speedphy:.2f}倍。但是，新方法J相对强物理beam逐状态平均下降{-n.mean_relative_J_pct:.2f}%，27个状态全部更差。这个耗时下降不能表述成保持同等质量的加速。

完整直接求解对比：score为加权期望成功字节J，不是Mbps或实际交付；相对差逐状态计算，再按seed等权。每个状态每方法串行测量3次，先取中位数，再按seed等权。p95是27个状态中位耗时的95%分位，不是部署时延保证。
{display.to_string(float_format=lambda x:f'{x:.3f}')}

同名义73候选预算下，79维对照J为{caps.score:.3f}、耗时{caps.mean_e2e_ms:.3f}ms；新方法J为{n.score:.3f}、耗时{n.mean_e2e_ms:.3f}ms。新方法实际平均只送验{n.candidates:.2f}个去重候选，79维对照送验73个。因此这是相同候选上限的比较，不是相同实际FLOPs；也没有显示新模型具有更好的质量/成本组合。

跳出搜索停点：
{escape.to_string(index=False,float_format=lambda x:f'{x:.3f}')}

strict_greedy起点已经对原greedy_candidates完整邻域作精确检查：所有满足原出队下界的邻居增益不超过1e-7。新方法0/27改善，79维11/27改善，16位置物理beam21/27改善。physical4_stop从严格greedy解继续运行全合法UE、宽16的4位置物理beam，最多32轮，均因访问邻域没有改进而停止；它不是完整联合邻域的穷举最优证明。该起点新方法0/27改善，79维1/27改善，16位置物理beam7/27改善。准备起点的物理搜索成本单列，未用于直接推理加速数字。

输入、模型与训练：
3cell/36UE/17PRG；12维UE、17维位置、89维候选边，后者包含81维空间协方差/期望信号描述。静态边编码每决策一次，当前方案变化后重新编码动态状态。宽度32、两轮关系聚合、4个联合候选头；每头1个贪心和1个采样候选。轻量C++前缀解码只更新TB计数、层数、TBS查表和粗略干扰，完整候选由原精确PHY验收。模型{train['parameters']}参数；没有把原79维模型权重冒充联合新模型。

训练seeds41--43，共36个current/PFQ历史状态；验证44--45，共24个状态。40轮掩码重建，2轮学生自身状态采样/物理教师增强，各10轮训练，另有72次Best-of-8策略梯度更新。共{train['groups']}个根状态组、{train['cases']}个条件修复样本。样本、相邻快照和组存在依赖，不能当作1008个独立无线场景。验证按固定推理预算的真实物理J选型，选定epoch={train['best_epoch']}。最后一批Best-of-8更新降低验证J，没有采用其权重；早一批更新仍参与了后续训练。此实验不是多TTI PPO，也没有优化实际包时延。

测试seeds46--48，PFQ、neural、greedy三类源轨迹，TTI3001/4001/5000，共27个状态；预先写入protocol.json。它们在项目历史实验中已被观察，属于历史冻结状态验证。测试阶段没有调模型或候选预算；训练数据量、结构、损失和搜索控制都与79维旧实验不同，因此是整条算法路径的对比，不是单一组件的因果消融。

推理：1次草案评估+8个全局候选+2轮×4块×8候选，上限73；warm start另计一次根方案评估。每轮只接受一个最佳完整候选；原90%草案下界固定。所有方法使用相同J与精确物理核，强物理4/16均为全合法UE、beam16、最多3轮。冻结状态转换、初始化、特征、网络、生成、PHY、销毁全部计时；排除归档读取、一次性TBS表生成、模型/库加载、IPC和模拟器。CPU单线程计时，不据此宣称达到0.5ms调度周期。

实际成本：
新方法平均生成72个候选，去重及去掉原方案后平均送验{n.candidates:.2f}个；每决策受影响UE重算{n.ue_evaluations:.2f}次，MMSE misses {n.mmse_misses:.2f}次。79维分别为{n79.candidates:.2f}、{n79.ue_evaluations:.2f}、{n79.mmse_misses:.2f}。大块候选影响更多UE，因此候选减少比例不等于PHY算量减少比例。受影响UE重算和MMSE缓存原本已经存在，不能算成本轮新增贡献。

首轮完整耗时{before_time:.3f}ms，其中选块{parts_before['masks']:.3f}ms，原因是4/8/16/8尺寸重复计算同一干扰优先级。仅复用该确定性计算后，耗时{n.mean_e2e_ms:.3f}ms、选块{parts_after['masks']:.3f}ms。全部27状态、8条直接求解臂的最终动作/PHY工作量和所有继续搜索结果与首轮完全一致；模型、候选与验收没有改变。前后两轮完整测量及旧源码均保留，未挑选有利单次耗时。
最终代表性中位运行的平均分段ms：{json.dumps(parts_after,ensure_ascii=False)}。context含输入转换/PHY初始化/静态特征；neural含编码/联合头/轻量解码；other含动态特征、候选打包、去重和销毁等。

冻结后的候选覆盖诊断：
在27个strict_greedy起点上，对新方法实际选择的全局和局部掩码另做较强离线物理搜索。共{diag['masks']}个掩码，学生有可接受改进的掩码为{diag['neural_positive_masks']}；物理教师为{diag['teacher_positive_masks']}，覆盖{diag['states_with_teacher_improvement']}个状态。每掩码8个学生候选平均仅{diag['mean_unique_of_8']:.2f}个互异，出队下界不满足比例{diag['floor_violation_fraction']*100:.2f}%。教师另用了{diag['teacher_candidate_evaluations']}次精确候选评估，此开销只属于事后诊断，不计作新方法的推理能力，也没有把这些标签回流训练。该诊断能区分“选中掩码内没有已知改善”和“存在已知改善但学生没有生成”。当前证据不支持把退化唯一归因于数据量、网络容量或线性TB校正中的某一个因素。

工程核验与边界：
最终25项测试全部通过。Torch训练和NumPy部署logit一致到测试容差；编译解码与独立参考逐动作一致；掩码外不变、UE多PRG复用、空合法集、前缀无未来目标泄漏、固定原草案下界、完整候选上限及确定性重复均通过。直接与继续搜索的最终PHY均经独立NumPy参考检查。没有新增原生闭环轨迹，不能把J或主机耗时变化换算成实际吞吐、TTL或包时延收益。

后续若继续，优先解决有限候选集合的质量和联合相关性，使用保持相同状态/预算的生成器对照；当前结果不支持直接增加部署范围。科学结果为未达到双目标，工程实现与核验通过，两者分开报告。

复现说明：training/stageb_lns/masked_joint_README.rst。
完整数据、教师成本、训练历史、模型、源码、逐决策记录和候选诊断：deliverables/tb_masked_joint_20261030_v1/。
冻结模型SHA256：{sha(root/'repair.npz')}。
'''
    (root/'report.txt').write_text(report)

    colors={'masked_joint':'#126c9a','masked_no_global':'#6b9eb8','neural79_cached':'#c97913',
            'neural79_cap73':'#dba352','greedy':'#23875d','physical4':'#ba4040',
            'physical16':'#7854a3','physical4_cap73':'#9b7777'}
    labels={'masked_joint':'New masked','masked_no_global':'No global pass','neural79_cached':'79-input cached',
            'neural79_cap73':'79-input / cap 73','greedy':'Greedy','physical4':'Physical / 4 slots',
            'physical16':'Physical / 16 slots','physical4_cap73':'Physical / cap 73'}
    offsets={'masked_joint':(12,-22),'masked_no_global':(-22,-40),'neural79_cached':(-85,-22),
             'neural79_cap73':(12,12),'greedy':(-35,14),'physical4':(-88,-28),
             'physical16':(-80,12),'physical4_cap73':(10,10)}
    fig,axes=plt.subplots(1,2,figsize=(12.8,4.9),layout='constrained')
    for arm,row in summary.iterrows():
        x=row.mean_e2e_ms;y=row.score/phy.score*100
        axes[0].scatter(x,y,s=64,color=colors[arm],zorder=3)
        axes[0].annotate(labels[arm],(x,y),xytext=offsets[arm],textcoords='offset points',fontsize=8.5,
                         arrowprops=dict(arrowstyle='-',color=colors[arm],lw=.6))
    axes[0].set(xscale='log',xlabel='Complete CPU decision time (ms, log scale)',
                ylabel='Mean J / mean physical-4 J (%)',ylim=(25,111),xlim=(3,1600),
                title='Faster inference with a large quality loss')
    axes[0].axhline(100,color='#bbbbbb',lw=.8,ls='--');axes[0].grid(alpha=.15)
    arms=['masked_joint','neural79_cached','physical16'];x=np.arange(2)
    for j,arm in enumerate(arms):
        heights=[int(escape[(escape.root==r)&(escape.arm==arm)].escapes.iloc[0])
                  for r in ('strict_greedy','physical4_stop')]
        bars=axes[1].bar(x+(j-1)*.24,heights,width=.22,label=labels[arm],color=colors[arm])
        axes[1].bar_label(bars,padding=3)
    axes[1].set(xticks=x,xticklabels=['Certified greedy\nlocal optima','Physical beam\nstop points'],
                ylim=(0,29),ylabel='States with an accepted improvement / 27',
                title='The new model did not escape either starting set')
    axes[1].legend(frameon=False,fontsize=8.5);axes[1].grid(axis='y',alpha=.15)
    fig.savefig(root/'quality_cost.png',dpi=180);fig.savefig(root/'quality_cost.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.4),layout='constrained')
    palette=['#b2c8d7','#dc914b','#397eaa','#5f9d72','#a9a9a9']
    left=np.zeros(2)
    for color,name in zip(palette,parts_before):
        values=np.array([parts_before[name],parts_after[name]])
        bars=ax.barh([1,0],values,left=left,label=name,color=color)
        for bar,val in zip(bars,values):
            if val>.8:ax.text(bar.get_x()+val/2,bar.get_y()+bar.get_height()/2,f'{val:.2f}',ha='center',va='center',fontsize=9)
        left+=values
    ax.set(yticks=[1,0],yticklabels=['Before mask reuse','After mask reuse'],xlabel='Complete decision time (ms)',
           title='Identical model, actions and exact physical work')
    ax.legend(loc='lower center',bbox_to_anchor=(.5,-.42),ncol=5,frameon=False,fontsize=9)
    fig.savefig(root/'timing_breakdown.png',dpi=180);fig.savefig(root/'timing_breakdown.pdf');plt.close(fig)

    shutil.copy2(test_log,root/'tests.log')
    files={'report.txt':root/'report.txt','assessment.json':root/'assessment.json','tests.log':root/'tests.log',
           'protocol.json':evaluation/'protocol.json','training.json':root/'training.json',
           'summary.csv':evaluation/'summary.csv','escape_summary.csv':evaluation/'escape_summary.csv',
           'per_state.csv':evaluation/'per_state.csv','checks.json':evaluation/'checks.json',
           'certificates.json':evaluation/'certificates.json','mask_reuse_parity.json':root/'mask_reuse_parity.json',
           'diagnosis_summary.json':root/'diagnosis/summary.json',
           'reproduction.rst':Path(__file__).with_name('masked_joint_README.rst'),
           'quality_cost.png':root/'quality_cost.png','quality_cost.pdf':root/'quality_cost.pdf',
           'timing_breakdown.png':root/'timing_breakdown.png','timing_breakdown.pdf':root/'timing_breakdown.pdf'}
    for name,path in files.items():shutil.copy2(path,evidence/name)
    frozen=root/'frozen_source';frozen.mkdir(exist_ok=False)
    sources=snapshot_sources(frozen);write_json(frozen/'manifest.json',sources)
    write_json(evidence/'manifest.json',dict(files={name:sha(evidence/name) for name in files},
               checkpoint_sha256=sha(root/'repair.npz'),source_manifest_sha256=sha(frozen/'manifest.json')))
    print(json.dumps(assessment,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--training',required=True);parser.add_argument('--evidence',required=True)
    parser.add_argument('--tests',required=True)
    main(parser.parse_args())

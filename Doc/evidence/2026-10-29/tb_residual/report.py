"""Report cache gains and all frozen-pool/full-search regressions together."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent

def main():
    checks=json.loads((HERE/'checks.json').read_text());assert checks['passed']
    plan=json.loads((HERE/'plan.json').read_text());col=json.loads((HERE/'data/checks.json').read_text())
    rs=json.loads((HERE/'residual_training/selected.json').read_text());ps=json.loads((HERE/'proposer_training/selected.json').read_text())
    c=pd.read_csv(HERE/'evaluation/curves_per_seed.csv');eval10=c[(c.split=='evaluation')&(c.budget==.1)]
    proposal=pd.read_csv(HERE/'evaluation/proposal_per_seed.csv')
    b=pd.read_csv(HERE/'benchmark/block_costs.csv');bm=b.groupby('method').ms.mean()
    setup=pd.read_csv(HERE/'benchmark/setups.csv');timings=pd.read_csv(HERE/'benchmark/timings.csv')
    once=timings[(timings.method=='sparse_cpp')&(timings.repeat==0)]
    opt=pd.read_csv(HERE/'optimized/means.csv');replay=pd.read_csv(HERE/'replay/means.csv')
    st=pd.read_csv(HERE/'data/states.csv');gr=pd.read_csv(HERE/'data/groups.csv')
    errs=pd.read_csv(HERE/'diagnostics/self_search_summary.csv')
    mae=pd.read_csv(HERE/'evaluation/per_ue_mae.csv')
    cal=pd.read_csv(HERE/'evaluation/empirical_threshold_seeds.csv')
    # Paired root quality, descriptive only: the six states are temporally correlated.
    paired=pd.read_csv(HERE/'optimized/paired_states.csv')
    base=paired[paired.method=='verified_original'].set_index(['seed','tti'])
    comparisons=[]
    for method,quota in [('trace_original',8),('spatial_original',16),('residual_original',8),('residual_new_actor',8)]:
        z=paired[(paired.method==method)&(paired.quota==quota)&paired.fast].set_index(['seed','tti'])
        delta=z.risk-base.risk
        comparisons.append(dict(method=method,quota=quota,mean_risk_delta=float(delta.mean()),
            wins=int((delta>1).sum()),losses=int((delta< -1).sum()),ties=int((delta.abs()<=1).sum()),
            mean_cost_delta_ms=float((z.serial_ms-base.serial_ms).mean())))
    pd.DataFrame(comparisons).to_csv(HERE/'paired_vs_verified.csv',index=False)
    lines=[
        '2026-10-29 近似物理＋神经残差＋增量缓存＋集合外正例：本轮实现和对照完成。',
        '',
        '结论：增量缓存有效，神经残差和提议器在冻结候选池上有小幅到中等增益，但没有稳定转化为实际搜索质量/总耗时优势；95%召回门槛未过，不切换当前算法、不启动311–315闭环。平均成功包时延及P95仍是原生阶段必过条件，本轮没有新原生轨迹，不能声称已守住新策略的平均时延。',
        '',
        '本轮完成的四项实现：',
        '1. draft_kernel.cpp/residual.py：当前已验证root锚定的trace SINR草稿，以及按信号方向投影协方差的空间草稿；原生BLER查表、整TB的count/min-layer/TBS，均不求候选MMSE。预存PRG活动掩码下的标量比值；候选只重算资源变化UE与其干扰活动变化受害UE。graph/root更换时刷新相应缓存，geometry在同graph的root更新中复用；root相关gamma目前仍整表刷新，其成本计入。',
        '2. UE级残差头：输入32维因果局部物理摘要，32/16隐藏层，每成员1601参数、三成员4803参数；修正预计成功字节后截断到[0,dequeue]，再汇总原风险R。精确PHY和原J/R/floor仍独立终审。',
        '3. 集合外正例提议器：同一教师前缀下枚举全部合法下一步，不再只记录旧48项。用高收益集合的listwise分布、跨深度可微父前缀概率及原策略KL微调87维actor；保留合法掩码与原标准化，未增加模型参数或固定类别配额。',
        '4. draft_repair.py/fast_draft.py：实验搜索先廉价评分48项，在每深度精确验证quota4/8/16；可保留尚未验证的16个前缀，接受动作必须来自精确回调。另修复ctypes数组转换与重复项、跳过均值策略不使用的标准差，保持权重、动作、物理值和计数一致。未改默认解码器、原有实验或算法注册表。',
        '',
        '数据与事前协议：',
        '同上一轮58个快照、两个root（自己的PFQ+CTBS草案/保存的chosen）、每root12个预定图块；训练41/42/43/46/47/49，选型44，经验校准45，开发评估48/50。各角色在本轮分开，但44/45/48/50均有历史开发暴露；不是盲测或独立新校准样本。101–106、311–315均不读入本轮训练或评估。',
        f'全合法逐步教师仍为宽16的有界beam，不是完整组合穷举：{col["blocks"]}块、{col["groups"]}个同深度组、{col["candidate_occurrences"]}次候选、{col["sparse_pairs"]}个受影响UE样本。旧48项外存在{col["outside_positives"]}个可接受正增益出现，其中训练seed为{int(gr[gr.split=="train"].outside_positives.sum())}；不是独立机会数，也不是此前15个漏选的复现计数。',
        f'初次标签精确PHY共{int(st.candidate_evaluations.sum())}次，含setup；采集整体{col["wall_s"]:.3f}s，全部精确改变都包含于稀疏依赖集合。另有完整搜索、核验和诊断的工作量，分别见对应states/runs.csv，不计作免费部署数据。',
        '残差最多均匀抽样500000条训练UE样本，每种模式各30epoch、三成员；只以seed44同深度10%预算召回选模式/epoch。保留epoch0恒等基线。最终trace残差epoch20，选型召回78.37%→86.35%；空间草稿自身85.76%，说明还须与更强无学习草稿比较。提议器20epoch，选择20；选型48项召回79.35%→79.89%，熵0.955→1.055，未出现本轮选型中的熵崩塌。',
        '',
        '冻结完整合法候选池：',
        '每组为同一深度、同一组固定教师父前缀的所有合法下一步，均用root后缀补全；高收益集合为满足精确J>rootJ+16、R>rootR+1及自身PFQ草案90%floor的候选中收益最高10%（含并列）。无正机会组的召回为NA。名义10%预算因小组取整实际约11.37%/11.57%；所有方法同组同K，先组平均再seed等权。新候选总体比上一轮48-shortlist池大，不与上一轮52.54%直接作增益差。',
        eval10[['seed','method','actual_fraction','recall','any_high_yield','regret']].to_string(index=False,float_format=lambda x:f'{x:.6f}'),
        '神经残差对其trace底座召回增加4.18/7.94pp；相对新增空间草稿仅增加0.14/1.90pp。两seed仍未达到95%，精确oracle受并列及取整影响的均值上限也只有97.99%/97.45%。只看“至少一个命中”会高估完整高收益集合覆盖；不能换指标宣称过关。',
        '',
        '短名单覆盖与集合外召回：',
        proposal[proposal.split=='evaluation'][['seed','method','recall','outside_recall','any_high_yield']].to_string(index=False),
        '旧actor→新actor的48项高收益召回：81.19%→81.88%、81.53%→82.04%，各增加0.69/0.51pp。新actor找回原漏选高收益集合的22.08%/24.49%，但同时有替换损失，总召回增量较小。原短名单在共同教师父前缀下定义，不是原生已知2个漏选的直接恢复证明；验证seed正例未进入训练。',
        '原actor+残差在10%预算召回73.56%/77.90%，新actor+残差74.40%/78.32%，仍显著低于不经48项短名单的残差82.29%/87.30%；候选覆盖依然限制收益。',
        '',
        '局部预测和经验校准：',
        mae.to_string(index=False),
        '评估教师池上的逐UE预计成功字节MAE反而升高：418.58→468.34、304.47→355.96；排序召回改善不能当作物理校准全面改善。',
        cal.to_string(index=False),
        '经验阈值校准召回98.21%；评估95.54%/98.62%，仍保留约41.99%/46.96%候选，未达到省80%的目标。按seed保守单位n=1不能给出非平凡2%分布无关保证。在线实验仅固定quota按均值排序，未使用未证实的保形风险保证。',
        '',
        '同候选批次的成本归因：',
        bm.to_string(),
        f'单位为ms/修复块（按原4个深度组累加，各组3次交替串行中位数）。同C++、相同标量/BLER规则与输出格式：全UE重算{bm.all_ue_cpp:.3f}→受影响UE重算{bm.sparse_cpp:.3f}，独立降低{100*(1-bm.sparse_cpp/bm.all_ue_cpp):.3f}%。只需处理的UE对占全量{100*once.affected_pairs.sum()/once.full_pairs.sum():.3f}%。',
        f'完整Python trace草稿{bm.full_trace_py:.3f}→增量C++草稿加残差头{bm.sparse_cpp_residual:.3f}，降低{100*(1-bm.sparse_cpp_residual/bm.full_trace_py):.3f}%；这包含C++移植，不能全归给缓存。此项未含actor提议/精确终审；root初始化均值{setup.setup_ms.mean():.3f}ms、同graph重置{setup.rebase_ms.mean():.3f}ms另列。两种空间/trace结果与独立Python实现一致，全UE/稀疏C++受影响特征逐字节一致。',
        '全UE重算是基准专用库：只去掉同一源码的affected skip，复用同样的gamma缓存/数据布局/查表，用于隔离候选UE闭包节省；不是另一个部署物理模型。',
        '',
        '实际改变搜索树的完整回放：',
        '六个事前固定状态：seed48/50，TTI2001/3001/4000；最大3轮，所有quota方法共用actor cap48、beam16。每quota上限相同不等于实际精确次数或FLOPs完全相同。所有root/graph/cache/特征/actor/head/精确验收及销毁都计入计时；模型加载在外。表为同一优化会话的串行3次中位数、先状态后seed等权，不能与早期并发耗时混算。',
        opt[['method','quota','fast','risk','candidates','serial_ms']].to_string(index=False,float_format=lambda x:f'{x:.6f}'),
        '同方法fast前后全部最终动作、PHY数组、精确计数一致；trace quota8耗时326.986→319.406ms（-2.32%），残差旧actor410.043→385.525ms（-5.98%），残差新actor432.700→407.949ms（-5.72%）。这项保行为优化不改变质量。',
        '物理trace quota8的R均值87337.747，相对verified87422.604约-0.097%；候选3977对19334，减少79.43%，但完整耗时319.406对320.180ms，基本持平，远非省80%主机耗时。相对直接物理耗时553.745ms更低，但只是六个保存状态的点结果，不是闭环吞吐/时延非劣证据。',
        '残差quota8旧/新actor R分别84546.193/83559.856，均低于trace quota8与verified，并且耗时更高。完整quota曲线（包括4/16的全部结果）另见replay/means.csv；未只挑最好点隐藏退化。',
        pd.DataFrame(comparisons).to_string(index=False),
        'trace8虽均值接近verified，六状态只有1胜5负，均值被一个较大正增益拉高。残差旧/新actor同样各1胜5负。状态相关且只有两个开发seed，不做独立样本的显著性声明；更不能推断原生平均包时延。',
        '',
        '为何局部指标提高仍搜索退化：',
        errs.to_string(index=False),
        '在残差策略自身选中并送精确验证的相同候选上，残差预测的R增量MAE比trace更低（1328→1300、1239→1124），正偏也降低（930→410、713→330）；结果不是简单的“残差过估更多”。不同策略访问的样本不同，不能拿trace策略与残差策略两行的总体误差作无偏比较。诊断重放的最终分数和精确计数与未插桩版本一致，标签只分析、不回流训练。',
        '可证实的是：本轮当前补全价值更准/短名单召回更好，没有带来稳定的最终解改善。一个待检验解释是，当前incumbent补全分数并不代表有限继续搜索后的潜力；纠正局部物理误差会重新排前缀、压掉需要后续组合才能获益的分支。还可能包含自身搜索分布偏移、48项覆盖损失和actor替换损失，本轮不能把退化唯一归因于其中之一。',
        '',
        '后续优先级与边界：',
        '保留已验证的增量内核和绑定降本，不部署当前残差/新actor。先在训练seed采集自身搜索访问的前缀，并对少量继续搜索做显式计费，建立“继续后的增益/过程优势”标签；当前物理残差用于校正一步PHY，前缀未来价值另建目标，不能混称为PAV。未展开候选的未来值是缺失/删失标签，不能当0。',
        '提议器继续报告找回与替换损失，增加有向干扰/跨PRG组合条件，不能只增加固定多样性配额或预算STOP RL。以后每轮先冻结训练/选型/经验校准设计，完整搜索过关后才运行原生新seed。',
        '原生晋级条件不变：相对明确冻结的ρ=.90物理对照，逐seed完整包payload≥99.5%、TTL增量≤.1pp、平均/P95完整包时延不高于对照、UE P5不低于对照，并事前声明UE payload下降>2%的数量上限。当前有效J物理搜索与同风险R物理搜索分别报告；不以本轮R点值代替交付或时延。保留boundary UE75%、配置业务负载68.75%的不同口径。311–315继续留作未读最终测试，不拿来调参。',
        '',
        '核验与复现：',
        '36项测试通过；替换临时测试数据后的受影响测试单独重跑通过。源码、原始状态、模型、选型、oracle预算上界、所有PHY依赖闭包、独立近似参考、完整搜索动作/物理验收及计数核验通过。技术checks.passed=true与scientific_gate_passed=false分开，未修改当前算法注册表。',
        '容器c_aerial_oai2，目录/opt/nvidia/cuBB，.venv/bin/python，PYTHONPATH=/opt/nvidia/cuBB，OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1。先-m training.stageb_lns.build_draft，再在本目录依次collect.py、train_residual.py、train_proposer.py、evaluate.py、replay.py、benchmark.py、optimize.py、diagnose.py、audit.py、report.py。所有性能测量应在其他训练/回放结束后串行运行。输出目录排他创建；复现复制脚本与plan到新版本目录，不覆盖现有原始证据。',
        '全部候选、UE样本、模型、源码快照和profiler留在本地deliverables/tb_residual_20261029_v1。与PAV/迭代价值引导研究的联系是设计推断；上一轮已核对文献见相邻tb_process_value_20261029_v1/research.txt。'
    ]
    (HERE/'report.txt').write_text('\n'.join(lines)+'\n')
    (HERE/'progress.txt').write_text('本轮完成：近似物理C++增量缓存、UE残差头、集合外正例提议器；冻结池、完整搜索、成本分解、保行为绑定优化和自身前缀误差诊断均完成。\n缓存独立降本37.83%；残差召回82.29%/87.30%，未达95%；实际搜索未出现稳定神经收益。\n36项测试及来源审计通过；平均包时延未做新原生验证；不切换算法，不运行311–315。\n')
    # Scientific artifacts: include both the positive local metric and full-search failure.
    fig,axes=plt.subplots(2,2,figsize=(11.5,8.2),constrained_layout=True)
    avg=c[c.split=='evaluation'].groupby(['method','budget']).mean(numeric_only=True).reset_index()
    labels={'trace_draft':'Trace draft','spatial_draft':'Spatial draft','residual':'Residual value','exact_oracle':'Exact oracle'}
    for method,label in labels.items():
        z=avg[avg.method==method];axes[0,0].plot(100*z.actual_fraction,100*z.recall,'o-',label=label)
    axes[0,0].axhline(95,ls='--',color='gray');axes[0,0].set_xlabel('Exact candidates / full legal pool (%)')
    axes[0,0].set_ylabel('High-yield recall (%)');axes[0,0].legend(fontsize=8);axes[0,0].grid(alpha=.2)
    methods=['full_trace_py','all_ue_cpp','sparse_cpp','sparse_cpp_residual']
    axes[0,1].bar(range(4),[bm[m] for m in methods],color=['#777777','#4477aa','#228833','#cc6677'])
    axes[0,1].set_xticks(range(4),['Python full','C++ all UE','C++ sparse','Sparse + residual'],rotation=12)
    axes[0,1].set_ylabel('Approximation/feature time per block (ms)')
    for method,color,label in [('trace_original','#228833','Trace'),('spatial_original','#4477aa','Spatial'),('residual_original','#cc6677','Residual'),('residual_new_actor','#aa3377','New actor + residual')]:
        z=replay[replay.method==method].sort_values('quota');axes[1,0].plot(z.candidates,z.risk,'o-',color=color,label=label)
    verified=replay[replay.method=='verified_original'].iloc[0]
    axes[1,0].axhline(verified.risk,ls='--',color='gray',label='Verified reference')
    axes[1,0].set_xlabel('Realized exact PHY candidates / solve');axes[1,0].set_ylabel('Final physical-risk score R')
    axes[1,0].legend(fontsize=8);axes[1,0].grid(alpha=.2)
    bars=opt[(~opt.fast)&opt.method.isin(['cached_original','verified_original','physical_direct'])].copy()
    bars=pd.concat((bars,opt[opt.fast]),ignore_index=True)
    order=['cached_original','verified_original','physical_direct','trace_original','spatial_original','residual_original','residual_new_actor']
    bars['order']=pd.Categorical(bars.method,categories=order,ordered=True)
    bars=bars.sort_values('order')
    names=['Cached','Verified','Physical','Trace','Spatial','Residual','Residual + new']
    axes[1,1].barh(range(len(bars)),bars.serial_ms,color='#4477aa')
    axes[1,1].set_yticks(range(len(bars)),names);axes[1,1].set_xlabel('Complete serial search (ms)')
    fig.suptitle('Residual/cache iteration: development evidence; no native-delay/promotion claim')
    fig.savefig(HERE/'results.png',dpi=180);fig.savefig(HERE/'results.pdf');plt.close(fig)
    print('REPORT WRITTEN',flush=True)

if __name__=='__main__':main()

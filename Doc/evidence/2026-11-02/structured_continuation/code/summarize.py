"""Rebuild the small evidence archive from a completed, frozen pilot."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np
import pandas as pd


def main(args):
    source = Path(args.source).resolve()
    destination = Path(args.destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    completion = json.loads((source/'completion.json').read_text())
    assert completion['completed'] and completion['protected_files_unchanged']
    per = pd.read_csv(source/'per_state.csv')
    summary = pd.read_csv(source/'summary.csv').set_index('arm')
    checks = json.loads((source/'evaluation_checks.json').read_text())
    selection = json.loads((source/'selection.json').read_text())
    decomposition = json.loads((source/'decomposition_checks.json').read_text())
    files = ['protocol.json','source_manifest.json','input_manifest.json','evaluation_input_manifest.json',
             'runtime_environment.json','decomposition_checks.json','collection_round0.json','collection_round1.json',
             'training_round0.json','training_round1.json','selection.json','measurements.csv','per_state.csv',
             'summary.csv','evaluation_checks.json','completion.json']
    files += [name for name in ('work_allocation.json','work_allocation.csv') if (source/name).exists()]
    for name in files:
        shutil.copy2(source/name,destination/name)
    by_seed = per.groupby(['seed','arm']).mean(numeric_only=True).reset_index()
    by_seed.to_csv(destination/'per_seed.csv',index=False)
    reference = per[per.arm=='cached79'].set_index('state_id')
    differences=[]
    for arm,group in per.groupby('arm'):
        group=group.set_index('state_id'); ref=reference.loc[group.index]
        for state in group.index:
            differences.append(dict(state_id=state,seed=int(group.loc[state,'seed']),arm=arm,
                j_delta=float(group.loc[state,'score']-ref.loc[state,'score']),
                ms_delta=float(group.loc[state,'e2e_ms']-ref.loc[state,'e2e_ms'])))
    pd.DataFrame(differences).to_csv(destination/'paired_vs_cached.csv',index=False)
    import torch
    rounds=[torch.load(source/f'own_prefix_round{i}.pt',map_location='cpu',weights_only=False) for i in (0,1)]
    def key(g):
        return (g['state_id'],g['depth'],tuple(g['block']),g['base_action'].tobytes(),g['partial'].tobytes())
    seen={key(g) for g in rounds[0]}
    diagnostics=[]
    for i,groups in enumerate(rounds):
        for depth in range(4):
            selected=[g for g in groups if g['depth']==depth]
            targets=np.concatenate([g['target'][g['known'] & g['complete']] for g in selected])
            diagnostics.append(dict(round=i,depth=depth,groups=len(selected),known_targets=len(targets),
                positive_target_pct=float((targets>0).mean()*100),mean_observed_gain_bytes=float(targets.mean()*3000)))
    pd.DataFrame(diagnostics).to_csv(destination/'teacher_targets.csv',index=False)
    first=json.loads((source/'collection_round0.json').read_text())
    second=json.loads((source/'collection_round1.json').read_text())
    teacher=sum(x['teacher_cost']['candidate_evaluations'] for x in (first,second))
    observations=dict(round0_groups=len(rounds[0]),round1_groups=len(rounds[1]),
        round1_prefixes_already_seen=sum(key(g) in seen for g in rounds[1]),
        independently_visited_training_states=18,teacher_candidate_evaluations=teacher,
        selected_checkpoint=selection['selected'],selected_epoch_zero=Path(selection['selected']).stem.endswith('_e00'))
    (destination/'prefix_diagnostics.json').write_text(json.dumps(observations,indent=2)+'\n')
    current=per[per.arm=='current'].set_index('state_id')
    cached=reference.loc[current.index]
    delta=current.score-cached.score
    current_j_pct=(current.score.mean()/cached.score.mean()-1)*100
    current_ms_pct=(current.e2e_ms.mean()/cached.e2e_ms.mean()-1)*100
    # Standalone scientific plot; all points use the same current timing session.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax=plt.subplots(figsize=(8,5))
    for arm,row in summary.iterrows():
        ax.scatter(row['mean_ms'],row['score'],s=45)
        offset={'combined':(-8,-16),'learned':(6,8),'physical16':(-8,-18)}.get(arm,(4,5))
        ax.annotate(arm,(row['mean_ms'],row['score']),xytext=offset,
                    ha='right' if arm=='physical16' else 'left',textcoords='offset points',fontsize=8)
    ax.set_xlabel('Complete saved-state decision time (ms)')
    ax.set_ylabel('Weighted expected bytes J')
    ax.margins(x=.12,y=.12);ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(destination/'quality_cost.png',dpi=160)
    fig.savefig(destination/'quality_cost.pdf');plt.close(fig)
    rows=decomposition['blocks']
    full=sum(x['cartesian_candidates'] for x in rows)
    factored=sum(x['factored_work']['candidate_evaluations'] for x in rows)
    table=[]
    for arm in ('cached79','greedy','physical4','physical16','structure','current','rollout','learned','combined'):
        r=summary.loc[arm]
        table.append(f"{arm:12s} | {r['score']:11.3f} | {r['relative_j_pct']:+8.3f}% | {r['mean_ms']:9.3f} | {r['candidates']:10.1f} | {r['worst_relative_j_pct']:+8.3f}%")
    status='通过' if checks['scientific_gate_passed'] else '未通过'
    text=f'''结构化联合动作、继续价值与自身前缀教师纠错：首轮离线实验
2026-11-02

结论：独立入口、两轮采集/训练、27状态×9臂×3次串行对照和核验已完成。预声明质量与成本联合门槛：{status}。
冻结79维模型和缓存解码器、有效算法注册表保持不变。没有新原生轨迹，没有读取311–315，不形成场景泛化或包级时延结论。

实现与正确性：
1. 固定活动模式后分解小区UE分配，保留局部(J,D)非支配前沿再满足全局下界；完整TB包含块外PRG。
2. 3个训练seed的9个预定单PRG真实小块，全合法组合oracle与分解最优J一致，最大误差{max(abs(x['oracle_j']-x['decomposed_j']) for x in rows):.3g}。
   精确候选评估{full}→{factored}，减少{(1-factored/full)*100:.3f}%。这不是同等比例的完整决策降时延证明。
   合成测试另覆盖跨PRG、空合法集、无可行解、Pareto全局下界权衡及预算截断；只有完整求解的小块才可称块内精确。
3. 继续值是冻结CSI/MCS下，宽4、最多后续2位置、全合法有限物理beam观察到的最好可行J，相对验证根归一化。
   它不是最优后缀、过程优势或跨TTI整包价值。未知可行继续值保留missing；不填0。在线rollout宽2，物理调用全部进入solve预算。
4. 新增95输入值头（79原特征＋16因果后缀/TB/下界描述），32/16隐藏层，3617参数；原7233参数actor不更新。

数据与选型：
训练41–43、验证44–45，current/PFQ两个来源，TTI4001/8001/18000，训练18个历史状态。
每次学生完成决策后，在独立PHY上查询实际访问的前缀：每深度第一条保留父前缀、前4个访问块。
两轮分别{len(rounds[0])}/{len(rounds[1])}组，完整包/UE/前缀均不当独立无线场景；教师共{teacher}次精确候选评估，另有正确性/选型/回放开销。
第二轮有{observations['round1_prefixes_already_seen']}/{len(rounds[1])}组前缀已在首轮出现。
每轮20epoch，0/5/10/20按验证seed等权完整搜索J选型；epoch0值头全0，明确退回原概率排序。
最终选型：{selection['selected']}。
若选中epoch0，只证明训练版未超过原排序，不能把该路径称为学习增益。新的95输入构造和网络仍计入learned臂耗时。

历史冻结状态对照：
46–48三个已开发暴露seed，PFQ/neural/greedy三来源，TTI3001/4001/5000，共27状态。
训练与选型完成后才读取这一轮评估文件；历史暴露事实不变，不称盲测。
按状态取3次中位数，再seed等权。J为加权期望成功字节，不是Mbps。相对差先逐状态计算。
arm          |           J | 对physical4相对J | 完整ms | 精确候选 | 最差状态相对J
{chr(10).join(table)}

current相对缓存79维的均值之比：J {current_j_pct:+.3f}%，完整耗时{current_ms_pct:+.3f}%；逐状态{int((delta>1).sum())}胜{int((delta< -1).sum())}负。
这一均值改善不足以称稳定优势；普通贪心在本轮仍同时更快且平均J更高。所有新路径未通过联合晋级。
learned与缓存79维的最终动作/物理值一致，增加耗时来自额外特征与值头，不能将零值回退当成学习成功。

事后预算分布诊断：
在按来源/seed覆盖固定选出的3个历史状态上，原缓存版遍历125–132个有评分的块，rollout只覆盖3–4个，结构枚举覆盖5–7个。
rollout在前4个块耗尽4094次可编辑候选评分；这支持“继续搜索挤占全调度覆盖”的解释，不证明继续价值本身无用。
组合臂交错消耗预算，不能保证完整复现原缓存搜索的全部块；其当步J单调验收不等于逐状态不差于独立缓存基线。
该诊断逐动作、最终J和实际PHY工作量均与冻结正式决策一致；诊断成本不混入串行计时，也没有用于重训或重新选型。

预算与解释：
神经/结构/继续值/组合臂共用4096次实际完整调度评分上限；其中公共因果floor准备计1次，solve最多4095次。
物理greedy/4/16是上限100万的强质量参考，不能把相同模型或束宽称为等算量比较。
current/rollout短名单32大于保留束16，保证排序有真实剪枝空间；其中中间完整方案也可验收，所以差异包含中间候选暴露。
structure最多尝试8个四位置块；combined先值头提议，再最多4个精确块，共享同一全局预算。预算截断不伪称精确块最优。
计时含状态转换、新PHY/graph/cache、floor准备、特征/网络/提议/终审与清理；模型/库加载、归档IO和不变TBS表首次构造在外。
独立NumPy终态核验在计时全部结束后进行，教师查询单独计费。CPU研究流程耗时不等于0.5ms生产调度能力。

科学门槛与下一步：
本轮门槛为每个历史评估状态J≥完整physical4的99.5%，且seed等权平均决策耗时不高于缓存79维；结果{status}。
工程核验通过不等于科学质量通过。保留小块分解及其正确性证据；只有真实完整搜索收益和成本同时改善后才进入新场景/原生验证。
后续先保护缓存搜索的全调度覆盖，为少量有歧义的前缀分配继续搜索预算；再研究活动模式排序与惰性局部求解，避免首几个块耗尽总预算。
值头研究优先检查对原好前缀的替换损失、缺失目标与近最优覆盖。首轮选中零值头使第二轮全部前缀重复，未提供新的学生分布证据。
保持当前训练/选型/最终测试隔离，不据本轮评估状态调到通过。27项相关测试通过；243组正式终态独立NumPy核验通过。
仍需原生完整包payload、TTL、成功平均/P95、UE P5及退化UE数门槛；不能用当步J推导长期指标。

准备批次说明：
v1在评估数据读取和选型提交之前主动停止：发现current/rollout短名单与保留束均为16，排序没有真正剪枝空间。
v1原产物及aborted.json保留；修正为32>16并补回归检查后重启v2。本报告只使用v2完整批次，不覆盖或混入准备批次结果。

复现：training/stageb_lns/structured_README.rst。
完整数据/权重/源码/动作：{source}。
归档仅包含小型结果与新增源码；源清单包含未归档大工件的哈希。没有修改历史冻结报告。
'''
    (destination/'report.txt').write_text(text)
    code=destination/'code'
    code.mkdir(exist_ok=True)
    root=Path(__file__).resolve().parents[5]
    for name in ('structured_repair.py','continuation_repair.py','structured_campaign.py','test_structured_continuation.py','structured_README.rst'):
        shutil.copy2(source/'source/training/stageb_lns'/name,code/name) if (source/'source/training/stageb_lns'/name).exists() else shutil.copy2(root/'training/stageb_lns'/name,code/name)
    manifest={str(p.relative_to(destination)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
              for p in sorted(destination.rglob('*')) if p.is_file() and p.name!='archive_manifest.json' and '__pycache__' not in p.parts}
    (destination/'archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(text)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--destination',required=True)
    main(parser.parse_args())

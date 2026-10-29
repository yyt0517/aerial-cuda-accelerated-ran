"""Publish complete positive/negative pilot results with the stronger control."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main():
    selected = json.loads((HERE/'training/selected.json').read_text())
    checks = json.loads((HERE/'checks.json').read_text())
    assert checks['passed'] and not checks['scientific_gate_passed']
    macro = pd.read_csv(HERE/'evaluation/curves_macro.csv')
    per_seed = pd.read_csv(HERE/'evaluation/curves_per_seed.csv')
    trace = pd.read_csv(HERE/'diagnostics/trace_curves_per_seed.csv')
    depth = pd.read_csv(HERE/'diagnostics/depth_curves_per_seed.csv')
    table = macro[(macro.split == 'evaluation') & (macro.budget == .1)].copy()
    tr = trace[(trace.split == 'evaluation') & (trace.budget == .1)].groupby('method').mean(numeric_only=True).reset_index()
    table = pd.concat((table, tr), ignore_index=True)
    cols = ['method', 'actual_fraction', 'recall', 'any_high_yield', 'retained_gain_fraction', 'regret']
    ds = depth[(depth.split == 'evaluation') & (depth.budget == .1)].groupby('method').mean(numeric_only=True).reset_index()
    screens = pd.read_csv(HERE/'evaluation/threshold_per_seed.csv')
    err = pd.read_csv(HERE/'evaluation/prediction_errors.csv')
    timing = pd.read_csv(HERE/'evaluation/head_timings.csv').groupby('group').head_only_ms.median()
    st = pd.read_csv(HERE/'data/states.csv')
    cal = json.loads((HERE/'evaluation/calibration.json').read_text())
    source = json.loads((HERE/'data/checks.json').read_text())
    extra = pd.read_csv(HERE/'diagnostics/additional_replay_cost.csv')
    a = per_seed[(per_seed.split == 'evaluation') & (per_seed.method == 'tb_features_mlp') & (per_seed.budget == .1)]
    lines = [
        '2026-10-29 过程价值蒸馏与物理草稿对照：离线试验完成，质量门槛未通过。',
        '',
        '主要结论：约10%候选评估预算下，冻结神经价值头的高收益候选召回率均值52.54%，seed48/50分别50.53%/54.55%，远低于预设95%。补强的无学习近似PHY草稿召回69.02%。尚无“等质量省90%精确评估”的证据，未启动新seed闭环、未注册或替换当前有效ρ=.90物理搜索。',
        '',
        '本轮完成的是用户方案的最小离线实验：构造过程标签、蒸馏轻量价值头、校准、绘制召回—精确评估曲线并核查物理近似对照。覆盖提议器的重训、逐UE否决、难度路由、自演化及311–315闭环均未宣称完成。',
        '',
        '数据与冻结：',
        '训练41/42/43/46/47/49，每seed5个快照；选型44，共5个；经验校准45，共5个；离线评估48/50，每seed9个。58个快照各取自身PFQ+CTBS草案和保存的chosen两个root，各取8个高优先块加4个均匀分布余块。1392个块、214508个候选出现，按块内完整动作去重后156492个。训练去重80940、选型13428、校准13465、评估48659。',
        '48/50已反复用于开发，不能称盲测；101–106未用于本轮训练/调参；311–315未读取或运行。旧15个漏选及其2个未恢复例不混入训练，本轮没有证明解决这2个短名单漏选。',
        '旧归档只有状态和计数，缺少完整逐候选输入/标签。本轮冻结actor回放补采，不把标签当作免费已有数据。教师width16、shortlist48、per_parent2，按精确的incumbent后缀补全分数保留前缀；只覆盖这个有限提议集合，不是完整物理动作空间。',
        f'初次标签采集含setup与增量核验共{int(st.candidate_evaluations.sum())}次精确PHY，加{int(st.independent_audit_evaluations.sum())}次独立全量核验，合计{int(st.candidate_evaluations.sum()+st.independent_audit_evaluations.sum())}次；整体{source["total_wall_s"]:.3f}s。事后补强对照重建候选另用{int(extra.candidate_evaluations.sum())}次，全部输入特征与标签逐元素精确复现。',
        '',
        '模型与标签：',
        '主标签ΔR=精确R(当前前缀+root后缀)−精确R(root)；辅助标签为相对parent同式补全的分差。没有把精确parent分数、待评候选PHY值、未来逐包结果输入模型。该标签是一步PHY风险代理，不是PAV定义的未来成功优势。',
        '两种事前固定架构：actor87维+6维因果元信息；再补96维廉价整TB/活动摘要。MLP48/24，三成员集合，分别60epoch；只用seed44选择架构和epoch。',
        f'最终选择{selected["architecture"]}，epoch{selected["epoch"]}，输入{selected["features"]}维，每成员{selected["parameters_per_member"]}参数；3成员合计{3*selected["parameters_per_member"]}。训练总墙钟{selected["wall_s"]:.3f}s。冻结后才校准和评估，未据48/50结果重训。',
        '',
        '主要冻结池指标：',
        '每个块合并各深度已提议的完整动作，再按动作去重。高收益集合=同时满足原J、R、因果floor验收的正增益候选中，增益最高10%（含阈值并列）的集合；无正机会块的召回记NA，不能伪装成功。召回是该集合中被送精确PHY的比例；any_high_yield是至少保住一个的块比例；retained_gain_fraction是精确终审后的最大可接受增益保留率。先块平均再seed等权；样本不是独立TTI试验。',
        '预算采用ceil(.1*N)，因取整实际平均10.586%。表中所有比例为0–1；regret为原R的期望字节加权单位，不是包时延。',
        table[cols].to_string(index=False, float_format=lambda x:f'{x:.6f}'),
        '',
        '逐seed主要门槛：',
        a[['seed', 'actual_fraction', 'recall', 'any_high_yield', 'regret']].to_string(index=False),
        '两seed均未通过95%召回。主模型均值在20%/50%/80%预算时召回74.92%/95.34%/99.07%；“50%预算均值达到95%”不等于“10%预算逐seed达到95%”，也不是闭环质量通过。',
        '',
        '公平性诊断与强物理草稿：',
        '冻结BLER/TBS对照不随干扰变化，过弱。事后补充“root精确SINR锚定+因果PRG提示+活动协方差trace比例+候选整TB尺寸+原生BLER查表”的无学习草稿，不做候选MMSE。它的高收益集合召回69.02%、至少一项命中95.12%；并非所有指标占优，跨深度池的平均原R regret为216.335，神经头为183.074。不能只引用弱对照宣称神经独立优势。',
        '跨深度累计logp有长度偏置，合并池中actor的12.05%不能当作同深度前缀排序基线。下面将同一冻结模型按“首次出现深度”分组，仍用原块内动作去重、不重新训练或选型。这是事后诊断；小组取整使名义10%实际平均12.682%。',
        ds[['method', 'actual_fraction', 'recall', 'any_high_yield', 'regret']].to_string(index=False, float_format=lambda x:f'{x:.6f}'),
        '同深度神经头召回51.60%，actor45.36%，强物理草稿74.64%；神经头对actor有约6.24pp增量，但落后强物理草稿约23.04pp。当前数据支持“需要物理辅助/更充分的局部特征”，不支持独立学习价值头已经优于近似物理。',
        '',
        '校准与迁移：',
        f'固定sigma=集合标准差+200，seed45按高收益漏检率经验校准λ={cal["empirical_value_lambda"]:.6f}。该经验阈值不同于CRC正式保证。',
        screens[['split', 'seed', 'method', 'actual_fraction', 'recall', 'any_high_yield', 'regret']].to_string(index=False),
        '经验校准召回98.09%，到48/50降为94.06%/90.32%，还需精确检查48.11%/51.94%的候选。未达到“α=2%时减少80%精确评估”。把seed作为保守单位时n=1，98% split-conformal阈值为无穷，正式保守退路是全部验证；不能拿同seed数千个相关候选充当数千个独立校准样本。',
        '',
        '成本范围：',
        f'额外96维特征在采集期按块平均{err.cheap_features_collection_ms.mean():.3f}ms；头推理仅已构造的去重特征，块中位数{timing.median():.3f}ms，块中位数分布P99={timing.quantile(.99):.3f}ms。不是全TTI/P99实时延迟。额外特征仍整批重建，尚未实现受影响UE/PRG增量，特征成本已成为主要问题。强物理草稿的Python BLER查表也有显著额外成本，不能只拿精确PHY次数代表总计算量。',
        '精确教师产生整棵候选树的成本没有包含在筛选曲线横轴；故曲线只是冻结池条件下的必要离线筛选实验，不能乘19200宣称已降到约2000次/TTI。改成学习引导树后还会发生分布改变。',
        '',
        '决定与下一轮：',
        '本轮按事前规则不进入闭环，不修改当前算法；不放宽95%门槛，不用最终seed补救。先在训练seed补齐有向受害关系、活动掩码、层数瓶颈及局部信道/协方差方向信息，复用近似PHY结果训练残差头或排序头；增量缓存必须按graph/root失效，所有构造和查表成本计入总耗时。',
        '提议器另用训练seed的完整物理邻域数据补足集合外正例，优化固定K召回与Best-of-N目标；只重排现有48项无法找回短名单之外的组合。固定4槽覆盖、更多预算/STOP RL不作为本轮新增方案。',
        '后续需要重新预声明训练/选型/校准轮次，在未读最终seed前冻结模型、经验或正式校准口径、逐UE回退阈值。候选池门槛通过后，先做学习实际改变前缀树的等预算回放，再做原生新seed闭环。当前有效J物理搜索与同风险R直接物理搜索应分别保留为对照，不混称同一ρ=.90基线。',
        '原生计划保留311–315、每seed至少5秒和负载×1.0/1.2/1.4/1.6；逐seed完整包payload≥物理99.5%、TTL增量≤.1pp、平均及P95成功包时延不高于对照、UE P5不低于对照、UE payload下降>2%的数量上限需在开跑前写死。逐seed点估计门槛与带配对区间的统计非劣效结论分开。保留boundary UE占比75%与配置业务负载占比68.75%两个口径。',
        '逐UE终审只能对当前预计交付等因果代理作硬约束，不能承诺所有UE长期吞吐/P95；难度路由和上一TTI热启动须重新核验变化依赖；自演化不得反复把固定最终测试集转为训练再称其盲测。',
        '',
        '保证修订与文献：',
        '精确终审保证当前模型下的验收合法性；学习漏筛会改变解与未来服务，不能自动继承投机解码的不变分布保证。PAV未来优势、CRC期望风险、RBR训练奖励与逐UE硬约束的区别，及联网核对的19篇原列文献/3篇2026年相关研究，见research.txt。',
        '',
        '核验与复现：',
        '30项单元测试通过；116对增量/独立全PHY检查精确一致；补强对照重放逐候选特征/标签精确一致；源码快照、原始输入、模型、选型、原算法注册表SHA256检查通过。checks.json的passed=true仅指完整性；scientific_gate_passed=false明确记录科学门槛失败。',
        '容器c_aerial_oai2，工作目录/opt/nvidia/cuBB；OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1，PYTHONPATH=/opt/nvidia/cuBB，使用.venv/bin/python。顺序collect.py、train.py、evaluate.py、controls.py、audit.py、report.py。脚本输出目录排他创建，复现实验请复制脚本/plan到新版本目录，不覆盖原始证据。'
    ]
    (HERE/'report.txt').write_text('\n'.join(lines)+'\n')
    (HERE/'progress.txt').write_text('已完成：58快照、1392块、214508候选出现的标签回放；价值头训练/选型；经验校准；强近似PHY及同深度对照；30项单元测试和来源审计。\n离线质量门槛未通过：10%预算神经召回50.53%/54.55%，强近似PHY均值69.02%。\n未启动311–315闭环；未改当前有效算法。下一轮优先局部物理特征缓存与物理残差学习。\n')
    # Final figure includes the stronger, independently labelled physics control.
    block = pd.read_csv(HERE/'evaluation/curves_per_seed.csv')
    block = pd.concat((block, trace), ignore_index=True)
    block = block[block.split == 'evaluation'].groupby(['method', 'budget']).mean(numeric_only=True).reset_index()
    dp = depth[depth.split == 'evaluation'].groupby(['method', 'budget']).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row, (frame, label) in enumerate(((block, 'Whole-block union'), (dp, 'Conditioned on depth (post-hoc)'))):
        for method, name in [('actor_likelihood', 'Actor likelihood'),
            ('tb_features_mlp' if row == 0 else 'frozen_tb_mlp', 'Frozen neural value'),
            ('trace_sinr_physics', 'Root-anchored approximate PHY'), ('exact_oracle', 'Exact oracle')]:
            z = frame[frame.method == method]
            for col, metric in enumerate(('recall', 'any_high_yield')):
                axes[row, col].plot(z.actual_fraction*100, z[metric]*100, 'o-', label=name)
        for col in range(2):
            ax = axes[row, col];ax.axhline(95, color='gray', ls='--', lw=.8)
            ax.axvline(10, color='gray', ls='--', lw=.8)
            ax.set_title(label);ax.set_ylim(0, 102);ax.grid(alpha=.2)
            ax.set_xlabel('Exact evaluations / unique candidates (%)')
            ax.set_ylabel('High-yield recall (%)' if col == 0 else 'Any high-yield retained (%)')
    axes[0, 1].legend(fontsize=8, loc='lower right')
    fig.suptitle('Offline process-value pilot; development seeds48/50; no closed-loop claim')
    fig.savefig(HERE/'recall_cost_controls.png', dpi=180);fig.savefig(HERE/'recall_cost_controls.pdf')
    plt.close(fig)
    print('REPORT WRITTEN', flush=True)


if __name__ == '__main__':
    main()

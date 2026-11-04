"""Keep zero-PHY log diagnostics separate from the paid canonical cache gate."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from .analyze_dpls_moves import write_json,write_csv,sha


def read_csv(path):
    return list(csv.DictReader(Path(path).open()))


def archive_manifest(out):
    write_json(out/'archive_manifest.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})


def main(args):
    offline,gate = Path(args.offline).resolve(),Path(args.gate).resolve()
    out,paid = Path(args.offline_evidence).resolve(),Path(args.gate_evidence).resolve()
    for folder in (out,paid):
        folder.mkdir(parents=True,exist_ok=False)
        (folder/'source').mkdir()
    for name in ('protocol.json','completion.json','input_manifest.json','summary.json',
                 'union.csv','within.csv','cross_tti.csv','work_model.csv'):
        shutil.copy2(offline/name,out/name)
    for name in ('analyze_dpls_joint_cost.py','dpls_operations.py'):
        shutil.copy2(offline/name,out/'source'/name)
    for name in ('protocol.json','completion.json','source_manifest.json','input_manifest.json','decisions.csv','rounds.csv'):
        shutil.copy2(gate/name,paid/name)
    for p in (gate/'source').iterdir():
        if p.suffix in ('.py','.cpp'):
            shutil.copy2(p,paid/'source'/p.name)
    for p in gate.glob('*.npz'):
        shutil.copy2(p,paid/p.name)
    for folder in (out,paid):
        shutil.copy2(__file__,folder/'source'/Path(__file__).name)
        if args.tests:
            shutil.copy2(args.tests,folder/'tests.txt')
    for name in ('test_dpls_joint_cost.py','test_cached_dpls.py'):
        shutil.copy2(Path(__file__).with_name(name),paid/'source'/name)
    summary = json.loads((offline/'summary.json').read_text())
    within = read_csv(offline/'within.csv')
    models = read_csv(offline/'work_model.csv')
    recall_sweep = []
    for arm in summary:
        rows = [r for r in within if r['arm']==arm]
        for m in (16,32,64,128):
            recovered = sum(r['chosen_dirty']=='False' or r['chosen_new']=='True' or
                            (r['dirty_known_rank'] and int(r['dirty_known_rank'])<=m) for r in rows)
            fresh = np.mean([int(r['new_eligible'])+min(m,int(r['dirty_known_candidates'])) for r in rows])
            recall_sweep.append(dict(arm=arm,M=m,best_recovered=recovered/len(rows),mean_fresh_candidates=float(fresh)))
    write_csv(out/'dirty_budget_recall.csv',recall_sweep)
    certified = []
    for arm in summary:
        for cap in ('1','2','4','8','all'):
            rows = [r for r in models if r['arm']==arm and r['cap']==cap and r['floor_certified']=='True']
            certified.append(dict(arm=arm,cap=cap,states=len(rows),
                projected_combined_candidate_ratio=sum(float(r['projected_refresh']) for r in rows)/sum(float(r['baseline_candidates']) for r in rows)))
    write_csv(out/'floor_certified_work_model.csv',certified)
    lines = ['批量依赖并集与旧精确ΔJ排序：零新增PHY的日志分析',
        '2026-11-03；本报告只读既有96条相关瞬态轨迹，不新增PHY/MMSE/原生运行', '',
        '1. 缓存与批量接受的节省不能相乘',
        '每轮按ΔJ排序构造互不冲突、各自超过原1/16B阈值的移动集合，并保留原接受动作作首项。12个保存输入的状态同时守共同floor；其余仅作结构打包，不称floor通过。',
        '合并后重新生成逻辑邻域，按规范编辑操作匹配公共候选。依赖并集覆盖=与任一移动同PRG或受影响UE相交的公共候选数/公共候选数；新增操作全部计入刷新。此过程不调用PHY。',
        'arm | cap | 实际平均k | 公共候选依赖并集% | 联合候选工作量/full% | 错误相乘估算%']
    for arm,s in summary.items():
        for cap in ('1','2','4','8','all'):
            r=s[cap]
            lines.append(f"{arm} | {cap} | {r['mean_k']:.2f} | {100*r['union_common_fraction']:.2f} | "
                         f"{100*r['projected_combined_candidate_ratio']:.2f} | {100*r['incorrect_product_ratio']:.2f}")
    lines.extend(['', '联合工作模型逐TTI计算：W=N_first + Σ(F_union(r)/k(r))，等效轮数R=1+Σ(1/k(r))；只对有下一轮记录的接受转移求和。',
        'N_first保留每TTI完整首扫；F_union包含合并后新增候选与公共候选的依赖并集。k=1时精确恢复上一轮诊断的28.74%/34.02%冷/热full刷新比。',
        '错误乘法是“单移动缓存比例×仅批量完整重扫工作比例”。整包情况下其7.91%/12.01%明显小于联合18.37%/22.88%。',
        '这只是原日志各根上的局部“轮数×刷新量”速率估算：合并轨迹没有运行，之后出现的收益、轮数、候选域及服务结果均未知；它不是严格性能上界、实测PHY账本或墙钟提速。',
        'floor_certified_work_model.csv另列仅12状态的共同floor子集，不能将全样本结构估算称为守floor算法结果。', '',
        '2. 上一轮精确ΔJ作为排序对照',
        '目标是日志记录的接受操作；不把旧路径当成新ε规则的完整路径。排序只读取上一轮分数，未见操作没有旧标签、按未召回处理；规范键仅用于稳定平局。',
        'floor过滤使用日志可行性代替在线可廉价计算的count/min-layer/TBS检查；无过滤的召回也单列。',
        'arm | 可预测接受轮 | 全候选top16召回% | floor过滤top16召回% | 脏且已知赢家top16召回% | clean免费/novel全评/脏top16后的赢家恢复% | 新精评数均值'] )
    for arm,s in summary.items():
        r=s['within']
        lines.append(f"{arm} | {r['count']} | {100*r['top16_all']:.2f} | {100*r['top16_floor']:.2f} | "
                     f"{100*r['dirty_known_top16']:.2f} | {100*r['dirty_top16_best_recovered']:.2f} | {r['mean_fresh_candidates']:.2f}")
    lines.extend(['轮内旧精确ΔJ显著强于此前SIR，但top16并非近100%：旧次优动作可在受影响UE变化后跃升。dirty-only策略保留全部clean精确值，并单独收费全评novel操作，不能把它叫作总预算16。',
        'dirty_budget_recall.csv列M=16/32/64/128的召回与新评估数。该策略未运行完整预算轨迹，也没有通过终态/服务门槛，当前仅是离线排序对照。', '',
        '3. 上一TTI终轮精确ΔJ对下一TTI首扫的排序',
        'arm | 相邻TTI对 | 赢家操作在前TTI有标签 | top16召回% | 标签位于前TTI实际执行终态？'] )
    for arm,s in summary.items():
        r=s['cross_tti']
        lines.append(f"{arm} | {r['count']} | {r['known_winner']} | {100*r['top16_floor']:.2f} | "+
                     ('是' if arm.endswith('_full') else '否，最后一次接受前的根，仅作另列因果旧标签对照'))
    lines.extend(['cold/warm完整轨迹合计0/62；其中有旧标签的赢家也未进入top16。跨TTI的信道、队列、MCS与根解变化，不能从轮内约75%不变推出跨TTI好排序。',
        'warm_n3的2/31来自最后一次已评分根，而非执行终态，不能混入终态标签结果。首个采样TTI没有t−1日志，全部排除。', '',
        '4. 平局规则与新基线',
        '新规则采用全局最大值锚定：先筛选当前floor可行且ΔJ严格超过原阈值的候选，取真正最大ΔJ，在距最大值≤ε=1e−6B的候选中选规范键最小者。',
        '规范键是PRG及各小区被覆盖目标（未编辑哨兵−2、静默−1、UE编号）。两两“差≤ε”比较不传递，所以不使用遍历式比较器。',
        '新完整/缓存模式同用canonical_dependency_v1；旧索引基线、历史证据和有效注册表不改写。旧日志排序分析与新基线重跑严格分账。',
        f'新增评分的基线/缓存等价门见：{paid}/report.txt', '',
        '结论：先过精确缓存门，再实现守floor多移动；批量节省按依赖并集估算。排序首先采用轮内旧精确ΔJ，跨TTI旧终轮ΔJ暂不能作为首扫强先验。',
        f'完整离线输出：{offline}；原始数据SHA见input_manifest.json。'])
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    complete=json.loads((gate/'completion.json').read_text())
    rows=read_csv(gate/'decisions.csv')
    ratio=complete['new_cached_candidate_evaluations']/complete['new_full_candidate_evaluations']
    text=['规范平局基线＋精确跨轮缓存：付费冻结状态等价门',
        '2026-11-03；新版本canonical_dependency_v1，ε=1e−6B', '',
        '工程门通过：12个既有保存输入，各重跑一次新完整基线和一次缓存版，均搜索到收敛。无新原生闭环。',
        f"逐轮核验{complete['rounds']}轮、{complete['bitwise_candidate_phy_audits']}个完整候选：全部UE的8个PHY字段、J/ΔJ、当前floor可行性、根动作/根PHY、规范键选择及终态动作逐位一致。",
        '缓存保存上次精评的受影响UE精确PHY值；每轮从当前根复制未受影响UE，再填回可复用候选的受影响值，对脏/新增候选调用PHY，最后按完整候选布局重新归约J与D。',
        '不复用旧全局ΔJ、不累加已舍入增益、不沿用旧eligible。该实现因此不依赖原候选行号的平局，也不需要用经验误差边界声称精确。', '',
        '新增付费工作账本（与零PHY日志分析分开）',
        f"完整基线：{complete['new_full_candidate_evaluations']}次完整候选评分；缓存版：{complete['new_cached_candidate_evaluations']}次，比例{100*ratio:.2f}%。两者均包含PFQ和热启动初始化。",
        f"Native MMSE miss：完整{complete['new_full_mmse_misses']}、缓存{complete['new_cached_mmse_misses']}；另有{complete['numpy_terminal_audits']}次独立NumPy终态审计。",
        '候选评分少不代表MMSE miss同比减少：原物理核已经缓存MMSE，跨轮缓存减少的是重复候选/受影响UE计算与既有MMSE缓存查询。',
        '测试夹具上的评分属于代码核验，不混入此12状态研究账本。', '',
        'source arm | 保存输入数 | 核验轮数 | 缓存/完整候选评分%']
    arm_summary=[]
    for arm in dict.fromkeys(r['source_arm'] for r in rows):
        selected=[r for r in rows if r['source_arm']==arm]
        f=sum(int(r['full_candidate_evaluations']) for r in selected)
        c=sum(int(r['cached_candidate_evaluations']) for r in selected)
        item=dict(arm=arm,states=len(selected),rounds=sum(int(r['audited_rounds']) for r in selected),candidate_ratio=c/f)
        arm_summary.append(item)
        text.append(f"{arm} | {item['states']} | {item['rounds']} | {100*item['candidate_ratio']:.2f}")
    write_json(paid/'summary.json',arm_summary)
    historical=[r for r in rows if r['historical_full_terminal_changed']]
    changed=sum(r['historical_full_terminal_changed']=='True' for r in historical)
    text.extend(['', f'新完整基线相对历史完整基线可比较的{len(historical)}个保存输入，终态动作改变{changed}个；仍按新版本独立计费与冻结，不把历史基线视作已免费重跑。',
        'warm_n3来源输入也运行完整收敛的新基线，但它的原历史终态是截断结果，不与之冒充同算法比较。',
        '当前只证明这些输入上的工程等价。相邻TTI/同seed候选不独立；没有包级质量、新seed泛化、墙钟成本或部署实时性结论。',
        '本轮计时包含完整候选数组的存储/审计，不能把decisions.csv的原始计时当公平在线加速；wall_speed_gate_evaluated=false。',
        '所有新规则在共享dpls_operations.py中，旧ranked_dpls及注册默认保留。多移动接受尚未执行，轮内旧ΔJ预算排序仍只作离线召回对照。',
        f'完整工作目录：{gate}', '核验及源/输入SHA、逐轮CSV、新基线根/终态NPZ一起归档；_physics.so留在完整工作目录，SHA有记录。'])
    (paid/'report.txt').write_text('\n'.join(text)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.4))
    for arm in ('cold_full','warm_full'):
        s=summary[arm]
        caps=['1','2','4','8','all']
        axes[0].plot(range(5),[100*s[k]['union_common_fraction'] for k in caps],'o-',label=arm)
        axes[1].plot(range(5),[100*s[k]['projected_combined_candidate_ratio'] for k in caps],'o-',label=arm+' union model')
        axes[1].plot(range(5),[100*s[k]['incorrect_product_ratio'] for k in caps],':',label=arm+' incorrect product')
    for ax in axes:
        ax.set_xticks(range(5),['1','2','4','8','all'])
        ax.set_xlabel('Maximum moves in the logged-root greedy pack')
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    axes[0].set_ylabel('Dirty common candidates (%)')
    axes[1].set_ylabel('Projected candidate work / full (%)')
    fig.suptitle('Local logged-root projection; shared floor certified only on saved-input subset',fontsize=10)
    fig.tight_layout()
    for extension in ('png','pdf'):
        fig.savefig(out/f'joint_cost.{extension}',dpi=160)
    plt.close(fig)
    archive_manifest(out)
    archive_manifest(paid)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--offline',required=True)
    parser.add_argument('--gate',required=True)
    parser.add_argument('--offline-evidence',required=True)
    parser.add_argument('--gate-evidence',required=True)
    parser.add_argument('--tests')
    main(parser.parse_args())

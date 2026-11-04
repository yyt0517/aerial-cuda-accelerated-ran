"""Post-run checks and small evidence export; never trains or selects models."""
import argparse
import csv
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np

from .bounded_probes import (ROOT, Predictor, ProbePhy, certified_search, read_rows,
    identity, problem_from_raw_row, raw_pfq_draft, solve_ranked, write_json, sha)


def read_csv(path):
    return list(csv.DictReader(Path(path).open()))


def main(source,destination,tests_log):
    import torch
    torch.set_num_threads(1)
    source=Path(source).resolve();destination=Path(destination).resolve()
    protocol=json.loads((source/'protocol.json').read_text())
    selection=json.loads((source/'selection.json').read_text())
    if not (source/'completion.json').exists():raise ValueError('Incomplete pilot')
    start=time.perf_counter();full_work={};trace_audit=[];counts={}
    for role,seeds in [('training',(41,42,43)),('validation',(44,45)),('evaluation',(46,47,48))]:
        rows,_=read_rows(seeds,evaluation=role=='evaluation')
        ttis=protocol['evaluation_ttis'] if role=='evaluation' else protocol['training_ttis']
        rows=[r for r in rows if r['tti'] in ttis]
        work={k:0 for k in ('candidate_evaluations','ue_evaluations','mmse_requests','mmse_misses')}
        for row in rows:
            p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs']);trace=[]
            action,values,r=solve_ranked(p,draft,observer=lambda info:trace.append(info['chosen_index']))
            for key in work:work[key]+=r[key]
            if role=='evaluation':
                for mode,model in [('bound',None),('scalar',Predictor(selection['scalar']['checkpoint'])),
                                   ('learned',Predictor(selection['spatial']['checkpoint']))]:
                    a,v,record=certified_search(p,draft,mode,model)
                    np.testing.assert_array_equal(action,a)
                    np.testing.assert_allclose(values,v,rtol=2.e-9,atol=2.e-6)
                    assert trace==[h['best_index'] for h in record['history']]
                    trace_audit.append(dict(state_id=identity(row),arm=mode,rounds=len(trace),
                        original_edit_indices_identical=True,final_action_identical=True))
        full_work[role]=dict(states=len(rows),**work)
        groups=torch.load(source/f'{role}_groups.pt',weights_only=False)
        counts[role]=dict(states=len(rows),roots=len(groups),candidate_labels=sum(len(g['gain']) for g in groups),
            additional_root_label_candidate_evaluations=len(rows)+len(groups)+sum(len(g['gain']) for g in groups),
            distinct_roots=len({(g['state_id'],g['root'].tobytes()) for g in groups}),
            gamma_bank_mmse_misses=len(rows)*36*17*4)
    original_checks=json.loads((source/'checks.json').read_text())
    teacher=dict(full_root_generation=full_work,root_labels=counts,
        training_and_validation_block_candidate_labels=original_checks['offline_block_teacher_candidates'],
        block_root_candidate_evaluations=counts['training']['roots']+counts['validation']['roots'],
        accounting='Deterministic replay of frozen full roots plus actual saved label lengths; not original teacher timing. '
                   'These offline calls are not charged as online inference and not hidden as free online capability.',
        audit_replay_elapsed_s=time.perf_counter()-start)
    write_json(source/'offline_teacher_work.json',teacher)
    write_json(source/'trace_audit.json',dict(passed=True,checks=trace_audit))
    summary=read_csv(source/'summary.csv');by_arm={r['arm']:r for r in summary}
    completion=json.loads((source/'completion.json').read_text())
    prior=source.with_name('three_bounded_ai_probes_20261102_v1')
    prior_selection=json.loads((prior/'selection.json').read_text())
    replay_note=dict(preparation=str(prior),preparation_evaluation_exposed=True,
        scalar_tie_mismatch='seed48/tti5000, edits at rounds3/4 exchanged; final action identical',
        numerical_cause='identical PHY rows accumulated differently by short strided NumPy matvec batches',
        correction='same full-neighborhood score reduction shape and original row indices, cost fully included',
        unchanged_selected_weights={k:selection[k]['sha256']==prior_selection[k]['sha256'] for k in ('scalar','spatial','block')},
        data_hyperparameters_and_budgets_unchanged=True,preparation_results_not_pooled=True)
    assert all(replay_note['unchanged_selected_weights'].values())
    write_json(source/'preparation_note.json',replay_note)
    ranks=read_csv(source/'ranking_per_seed.csv')
    feature_cost=read_csv(source/'evaluation_feature_cost.csv')
    bank_ms=float(np.mean([float(r['bank_ms']) for r in feature_cost]))
    spatial_ms=float(np.mean([float(r['spatial_feature_ms']) for r in feature_cost]))
    mean_rank={arm:{k:float(np.mean([float(r[k]) for r in ranks if r['arm']==arm]))
        for k in ('regret','near','retained')} for arm in ('heuristic','scalar','spatial')}
    per_state=read_csv(source/'per_state.csv');states=sorted({r['state_id'] for r in per_state})
    indexed={(r['state_id'],r['arm']):r for r in per_state}
    gains=np.array([float(indexed[s,'gen_scalar']['score'])-float(indexed[s,'gen_heuristic']['score']) for s in states])
    spatial_gate=all(next(float(r['regret']) for r in ranks if r['seed']==str(s) and r['arm']=='spatial')<
        next(float(r['regret']) for r in ranks if r['seed']==str(s) and r['arm']=='scalar') and
        next(float(r['near']) for r in ranks if r['seed']==str(s) and r['arm']=='spatial')>=
        next(float(r['near']) for r in ranks if r['seed']==str(s) and r['arm']=='scalar') for s in (46,47,48))
    gates=dict(input_spatial_gate=spatial_gate,scalar_vs_heuristic_regret_reduction_pct=
        100*(1-mean_rank['scalar']['regret']/mean_rank['heuristic']['regret']),ranking=mean_rank,
        quality_cost={r['arm']:dict(quality=r['quality_pass']=='True',cost=r['cost_pass']=='True',
            joint=r['joint_pass']=='True') for r in summary if r['arm']!='full'},
        generator_scalar_vs_heuristic=dict(wins=int((gains>1).sum()),losses=int((gains<-1).sum()),
            ties=int((abs(gains)<=1).sum()),mean_j_delta=float(gains.mean()),
            ratio_of_mean_gain_pct=100*(float(by_arm['gen_scalar']['mean_j'])/float(by_arm['gen_heuristic']['mean_j'])-1)),
        ai_joint_target_passed=False,default_registry_changed=False,
        next_path='Scalar residual generator with deeper single trajectories and own-prefix teacher correction; '
                  'certified physics is its offline teacher and quality reference, not an established AI speedup.',
        paused_paths=['full spatial candidate feature bank','learned block ordering','PPO integration'],
        limits='9 old development states, 3 radio seeds; no native/cohort/real-time/generalization conclusion')
    assert not any(v['joint'] for k,v in gates['quality_cost'].items() if k!='cert_bound')
    assert sha(ROOT/'training/stageb_hgraph/current_algorithm.json')==protocol['registry_sha256']
    write_json(source/'gate_results.json',gates)
    test_text=Path(tests_log).read_text()
    assert '\nOK\n' in test_text and 'Ran 24 tests' in test_text
    shutil.copyfile(tests_log,source/'tests.log')
    write_json(source/'runtime_environment.json',dict(python=platform.python_version(),numpy=np.__version__,
        torch=torch.__version__,platform=platform.platform(),threads=1,gpu_used=False,
        tests=24,tests_sha256=sha(source/'tests.log')))
    destination.mkdir(parents=True,exist_ok=False)
    selected=['protocol.json','selection.json','summary.csv','per_state.csv','measurements.csv',
        'ranking.csv','ranking_per_seed.csv','feature_cost.csv','evaluation_feature_cost.csv',
        'input_manifest.json','evaluation_input_manifest.json','checks.json','completion.json',
        'source_manifest.json','scalar_training.json','spatial_training.json','block_training.json',
        'offline_teacher_work.json','trace_audit.json','gate_results.json','runtime_environment.json','tests.log',
        'preparation_note.json']
    for name in selected:shutil.copyfile(source/name,destination/name)
    shutil.copytree(source/'scalar_attribution',destination/'scalar_attribution')
    code=destination/'code';code.mkdir()
    for name in ('bounded_probes.py','bounded_probe_followup.py','audit_bounded_probes.py',
                 'probe_kernel.cpp','test_bounded_probes.py','bounded_probes_README.rst'):
        shutil.copyfile(Path(__file__).with_name(name),code/name)
    write_json(destination/'code_manifest.json',dict(
        note='Original run source hashes are in source_manifest.json; original full snapshot stays in deliverables/source. '
        'Exported code adds final reporting/tests/README; no experiment or training semantics changed.',
        files={p.name:sha(p) for p in code.iterdir()}))
    follow=read_csv(source/'scalar_attribution/summary.csv')
    report=['三个规模受控的AI调度实验：结果与路径选择','2026-11-02','',
        '结论：三个实验及固定模型成本归因全部完成。AI质量与成本联合目标仍未通过。',
        f'物理上界剪枝在9/9开发状态复现完整DPLS动作并降低均时{100*(1-float(by_arm["cert_bound"]["time_ratio"])):.2f}%；这是物理/工程收益，不是AI收益。',
        '物理剪枝也未达到预声明的20%降时门槛；全部新路径均未通过质量与成本联合门槛。',
        '下一步选择：轻量标量残差生成器＋更深单轨迹修复＋自身访问状态/前缀教师纠错。',
        '上界剪枝作为该路线的离线教师与保质参照；暂缓全候选空间输入、学习块排序和PPO。','',
        '一、冻结范围与配对协议',
        '3小区、36UE、17PRG，沿用保存状态当前4T4R/SVD/Type-0 PHY；canonical原生BLER，保留MCS3非单调行。',
        '全部教师/训练/推理/对照无CTBS和需求headroom裁剪，保持因果服务缺口权重。',
        '与本轮相同目标的无CTBS完整DPLS比较，不与等权PPO或旧legacy评分混池。',
        '下界ceil(0.90×本状态原始PFQ物理出队量)，所有方法使用相同原始草案。',
        '训练41–43、选型44–45，来源为无CTBS缺口搜索与纯PFQ，固定TTI4001/8001/18000。',
        '每状态只取完整DPLS的第1/5/13轮根；短搜索没有第13轮时不补造样本。',
        f'训练18状态/{counts["training"]["roots"]}根/{counts["training"]["candidate_labels"]}候选标签；选型12状态/36根/55802标签。',
        '初次v1模型选型先于该轮开发指标；v2已暴露v1开发结果，仅修正浮点评分布局。',
        'v2未改变数据/训练/选型规则，48/120/99输入分别选epoch10/20/30，三个模型权重SHA与v1完全一致。',
        '开发46–48，仅纯PFQ来源，固定TTI3001/4001/5000，共9状态、27搜索根；均有历史开发暴露。',
        '九臂×三次串行计时=243次正式决策；固定模型归因另108次，共351次。重复计时与候选不作独立无线重复。',
        '同状态先取三次中位数，再seed等权；P95是9个状态中位耗时分位数，不是生产尾延迟。','',
        '二、实验1：标量与空间输入',
        '48输入原标量特征；120输入追加6个受影响UE槽×12个完整TB/逐层SINR特征。',
        '空间特征使用真实活动模式MMSE gamma和完整TB聚合，但不查询候选BLER或候选评分。',
        '每状态36×17×4=2448个MMSE条目从当前H/W重建；缓存建立和特征处理均计费，不能称为免费物理信息。',
        '比较同一组完整邻域标签上的regret@8与近优命中；近优容差max(16B,0.5%最大正增益)。',
        'arm | 平均regret@8/B | 近优命中 | 正收益保留比例']
    for arm in ('heuristic','scalar','spatial'):
        r=mean_rank[arm];report.append(f'{arm}: {r["regret"]:.3f} | {r["near"]*100:.2f}% | {r["retained"]*100:.2f}%')
    report.extend([
        '三个seed的scalar regret为192.568/117.096/68.910B，spatial为176.545/85.981/109.613B。',
        '空间输入在seed48退化，未通过逐seed改善门槛；不能认定输入遗漏是主要瓶颈，也不能证明空间信息无价值。',
        '标量模型相对启发式在三个seed均降低regret，平均降幅约82%；这是冻结邻域单步排序信号。',
        f'空间仅在2/3 seed改善，且每根额外特征均时约{spatial_ms:.2f}ms，缓存建立约{bank_ms:.2f}ms/状态。','',
        '三、实验2：协调块选择与上界剪枝',
        '块选择采用每小区当前UE/至多两个替代UE/blank的单PRG子域；完整笛卡尔组合精确验证。',
        '随机/物理风险/学习排序共享4096次实际候选上限、最多4扫；全部9状态覆盖17/17 PRG。',
        '有限子域块内枚举不等同于全合法单PRG最优，更不等同于完整调度最优。',
        '证书方法用未受影响UE的精确J＋受影响UE的加权出队上界；概率范围由所有原生BLER分段测试验证。',
        'AI只影响评分顺序，不影响证书；上界等于最优值时仍评分，保留原索引平局规则和1/16B阈值。',
        '每个开发状态的逐轮编辑索引、轮数和最终动作在bound/scalar/spatial三种证书路径均与full一致。','',
        '主实验完整质量/成本（J是加权期望成功字节，不是Mbps）：',
        'arm | 平均J | 完整均时ms | 最差状态J变化 | 相对full均时 | 质量/成本联合门槛'])
    for r in summary:
        report.append(f'{r["arm"]}: {float(r["mean_j"]):.3f} | {float(r["mean_ms"]):.3f} | '
            f'{float(r["worst_relative_j_pct"]):+.3f}% | {float(r["time_ratio"]):.4f} | {r["joint_pass"]}')
    report.extend([
        '质量门槛：逐状态J≥full的99.5%、因果floor和独立PHY核验通过；成本门槛：均时≤full的80%。',
        'full是参照臂，其cost_pass=False仅因不可能相对自身提速20%，不是参照算法质量失败。',
        f'物理证书均时{float(by_arm["full"]["mean_ms"]):.3f}→{float(by_arm["cert_bound"]["mean_ms"]):.3f}ms，精确候选74094→2428（约-96.72%），最终动作9/9一致。',
        '学习块排序比同实现物理启发式平均J更低，9状态4胜5负；随机排序的平均J反而最高。',
        '块路径最差状态仍下降5.57%–9.22%，未过质量门槛；覆盖已保住，排序与多步相互作用仍未解决。','',
        '四、实验3：不接PPO的生成器',
        '从未剪裁PFQ保留动作出发，最多7条路径、每条最多8次编辑；最多8个完整方案含PFQ。',
        '每路径首步最多7个提议，此后最多2个提议；精确J选取不同完整动作，不声称服务向量多样性。',
        '候选逐步精确验证并保护父方案与因果floor；没有在线完整搜索、教师或PPO调用。',
        '只比较同预算启发式、标量模型、空间模型；没有实现新的多解蒸馏或自身前缀纠错训练。',
        f'标量模型比启发式平均J提高{gates["generator_scalar_vs_heuristic"]["ratio_of_mean_gain_pct"]:.3f}%，7胜2负；三个seed平均均提高。',
        f'标量模型对full的逐状态相对J均值仍为-16.24%，最差-18.96%；{float(by_arm["gen_scalar"]["mean_ms"]):.2f}ms，质量失败。',
        f'空间模型相对标量3胜6负、均值只提高约101B，却耗时{float(by_arm["gen_spatial"]["mean_ms"]):.2f}ms，质量/成本均失败。',
        '保留PFQ和单步J单调不保证达到DPLS；有限深度/宽度与自身访问分布差异仍需验证。',
        '每状态预期服务下降>2%的UE计数仅是即时服务诊断，不是闭环UE吞吐或公平性结论。','',
        '五、固定模型耗时归因（主实验之后独立冻结协议；不重训、不重新选型）',
        'arm | 同会话完整均时ms | 相对同会话full | 精确候选 | MMSE miss'])
    for r in follow:
        report.append(f'{r["arm"]}: {float(r["mean_ms"]):.3f} | {float(r["time_ratio"]):.4f} | '
            f'{float(r["candidate_evaluations"]):.1f} | {float(r["mmse_misses"]):.1f}')
    report.extend([
        '标量排序加入证书搜索后仍慢于物理上界排序，且精确候选2727多于2428。',
        '因此单步regret改善没有转化为证书搜索的端到端AI加速。补充结果不覆盖或替换主实验计时。','',
        '六、下一步路径（本轮已选择，下一阶段尚未实施）',
        '1. 保留证书物理搜索作质量参照与离线教师；保质检查通过，约16%降时尚未达20%门槛，不晋级默认。',
        '2. AI主线选标量残差生成器：7/9状态改善且三seed均值改善，先修复距full约16%的质量缺口。',
        '3. 下一最小实验对照7条×8步浅束与1条×32步深轨迹，冻结总成本；先测可达收益与UE服务覆盖。',
        '4. 在训练seed采集生成器实际访问的根/前缀，离线教师提供有限继续修复及完整方案收益标签；未知继续值保留missing。',
        '5. 避免每轮为全部候选重算48维输入：共享UE/PRG编码，只刷新受影响闭包；同优化必须共享给启发式对照。',
        '6. 先独立保住候选质量、最差状态和弱UE服务，再接跨TTI/PPO；不据本轮J推导整包改善。',
        '暂缓：更大空间输入、现有学习块排序、扩大浅束候选和PPO续训。下一方案仍需独立预声明门槛。','',
        '七、核验和边界',
        '24项测试通过；117组正式/归因终态独立NumPy核验；三次重复的动作全部一致。',
        '27组开发状态×证书排序的逐轮原始编辑索引核验通过；所有原生BLER分段概率范围核验通过。',
        f'{completion["elapsed_s"]:.2f}秒为v2主实验运行时长，v1、补充归因、额外审计和编译测试另计；不是在线决策耗时。',
        '准备v1的标量证书排序在一个状态交换了两个数学等分编辑；数值归约布局修正后完整复跑为v2。',
        'v1原数据和初次核验失败日志保留，不与v2混池；约30%的首轮降时不能替代v2正式16%的结果。',
        '教师完整根生成、额外标签和块枚举计数见offline_teacher_work.json；未把教师查询藏进快速推理。',
        '未读取311–315，未运行新原生轨迹，未更换有效算法注册表；不宣称泛化、整包吞吐/TTL/时延改善或0.5ms实时能力。',
        f'完整本地实验：{source}',
        '原始执行源码快照见本地source/；本目录code/补入标量归因与最终测试，版本说明见code_manifest.json。'])
    (destination/'report.txt').write_text('\n'.join(report)+'\n')
    write_json(destination/'archive_manifest.json',{str(p.relative_to(destination)):sha(p)
        for p in destination.rglob('*') if p.is_file()})
    print('EXPORTED',destination,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--tests-log',required=True);a=p.parse_args();main(a.source,a.destination,a.tests_log)

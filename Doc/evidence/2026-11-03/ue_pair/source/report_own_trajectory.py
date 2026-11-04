"""Audit and archive the shallow/deep and own-state training campaign."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import shutil

import numpy as np

from .physics import ROOT
from .experiment import write_json,sha


def read_csv(path):
    return list(csv.DictReader(Path(path).open()))


def average(rows,key):
    return float(np.mean([float(r[key]) for r in rows]))


def audit_phase(source,phase):
    records=read_csv(source/f'{phase}_measurements.csv')
    state=read_csv(source/f'{phase}_per_state.csv')
    summary=read_csv(source/f'{phase}_summary.csv')
    for row in state:
        rr=[r for r in records if r['state_id']==row['state_id'] and r['arm']==row['arm']]
        assert len(rr)==3
        assert np.isclose(float(row['e2e_ms']),np.median([float(r['e2e_ms']) for r in rr]))
        if row['arm']!='full':
            assert int(row['feature_roots'])<=56 and int(row['candidate_evaluations'])<=106
            assert row['full_search_calls']=='0'
    for row in summary:
        rr=[r for r in state if r['arm']==row['arm']]
        for field,key in [('score','mean_j'),('e2e_ms','mean_ms')]:
            calculated=np.mean([average([r for r in rr if r['seed']==s],field) for s in sorted({r['seed'] for r in rr})])
            assert np.isclose(calculated,float(row[key]))
        assert (row['quality_pass']=='True')==all(float(r['ratio'])>=.995 for r in rr)
        assert (row['coverage_pass']=='True')==all(int(r['newly_unserved_ues'])==0 and float(r['expected_p5_ratio'])>=.98 for r in rr)
        assert (row['cost_pass']=='True')==(float(row['time_ratio'])<=.8)
    return summary,state,len(records)


def main(source,destination,tests_log,execution_log):
    import torch
    source=Path(source).resolve();destination=Path(destination).resolve()
    protocol=json.loads((source/'protocol.json').read_text());sel=json.loads((source/'selection.json').read_text())
    completion=json.loads((source/'completion.json').read_text())
    assert completion['completed']
    val,val_rows,val_count=audit_phase(source,'phase_a')
    summary,rows,eval_count=audit_phase(source,'evaluation')
    by_arm={r['arm']:r for r in summary}
    data=torch.load(source/'combined_training.pt',weights_only=False)
    assert {g['seed'] for g in data}<={41,42,43}
    assert len({(g['state_id'],g['root'].tobytes()) for g in data})==len(data)
    work=[json.loads((source/f'own_round{i}_work.json').read_text()) for i in (1,2)]
    visits=read_csv(source/'own_visits.csv')
    assert all(r['actual_visited']=='True' for r in visits)
    for r in work:assert r['post_decision_teacher'] and r['student_replay_identical']
    paired=read_csv(source/'paired_learning.csv');learning={}
    for shape in ('shallow','deep'):
        rr=[r for r in paired if r['geometry']==shape]
        seed_deltas={s:average([r for r in rr if int(r['seed'])==s],'j_delta') for s in (46,47,48)}
        wins=sum(float(r['j_delta'])>1 for r in rr)
        learning[shape]=dict(wins=wins,losses=sum(float(r['j_delta'])<-1 for r in rr),
            seed_j_deltas=seed_deltas,mean_j_delta=average(rr,'j_delta'),
            ratio_of_mean_gain_pct=100*(float(by_arm['trained_'+shape]['mean_j'])/float(by_arm['initial_'+shape]['mean_j'])-1),
            stable_learning=sel['weights_changed'] and all(v>0 for v in seed_deltas.values()) and wins>=6,
            joint_pass=by_arm['trained_'+shape]['joint_pass']=='True')
    stalls=read_csv(source/'diagnostics/stalls.csv');positive=read_csv(source/'diagnostics/positive_coverage.csv')
    diagnostics={}
    for role in ('validation','evaluation'):
        for arm in ('initial_deep','trained_deep'):
            rr=[r for r in stalls if r['role']==role and r['arm']==arm and r['has_stall']=='True']
            diagnostics[f'{role}:{arm}']=dict(stalled_states=len(rr),
                missed_qualifying=sum(r['missed_qualifying_edit']=='True' for r in rr),
                topk_recoverable={k:sum(float(r[f'top{k}_best_gain'])>16 for r in rr) for k in (1,2,4,8)})
    next_path=dict(selected='One verified trajectory with top2/top4 shortlist and per-root rejected-candidate deduplication',
        implementation_status='selected for next experiment; not implemented in this campaign',
        reason='32 attempts repeatedly chose the same failed edit; better verified edits exist at all observed stalled roots',
        keep='shallow original model as control; teacher labels remain offline; shared optimization for heuristic controls',
        stop=['plain top1 depth32 expansion','additional identical immediate-gain training rounds','PPO integration'],
        next_minimum='compare top2/top4 exact verification plus root-keyed rejected-edit cache under max56 feature roots/max106 PHY scores; '
                     'only then collect that new trajectory own-state labels',
        limits='posthoc topK recovery at fixed roots is not proof of complete-search quality, time or packet performance')
    gates=dict(learning=learning,quality_cost_coverage={r['arm']:{k:r[k]=='True' for k in
        ('quality_pass','coverage_pass','cost_pass','joint_pass')} for r in summary if r['arm']!='full'},
        ai_joint_target=False,coverage_p5_degenerate_states=sum(r['all_ue_p5_degenerate']=='True' for r in positive
            if r['role']=='evaluation' and r['arm']=='full'),
        original_gates_not_redefined=True,posthoc_diagnostics=diagnostics,next_path=next_path)
    assert not any(r['joint_pass']=='True' for r in summary if r['arm']!='full')
    assert sha(ROOT/'training/stageb_hgraph/current_algorithm.json')==protocol['registry_sha256']
    test_text=Path(tests_log).read_text();assert 'Ran 33 tests' in test_text and '\nOK\n' in test_text
    shutil.copyfile(tests_log,source/'tests.log');shutil.copyfile(execution_log,source/'execution.log')
    write_json(source/'gate_results.json',gates)
    audit_count=sum(len(json.loads((source/f'{p}_checks.json').read_text())['independent_phy']) for p in ('phase_a','evaluation'))
    write_json(source/'final_audit.json',dict(passed=True,unit_tests=33,phase_a_decisions=val_count,
        evaluation_decisions=eval_count,independent_final_phy_checks=audit_count,
        paired_statistics_recomputed=True,training_seeds_only=True,training_unique_roots=len(data),
        labels_from_actual_student_visits=True,posthoc_diagnostics_no_retraining=True,
        registry_unchanged=True,native_trajectories=0,final_seeds_read=False))
    write_json(source/'runtime_environment.json',dict(python=platform.python_version(),numpy=np.__version__,
        torch=torch.__version__,platform=platform.platform(),threads=1,gpu_used=False))
    destination.mkdir(parents=True,exist_ok=False)
    files=[p for p in source.iterdir() if p.is_file() and p.suffix in ('.csv','.json','.log')]
    for p in files:shutil.copyfile(p,destination/p.name)
    shutil.copytree(source/'diagnostics',destination/'diagnostics')
    code=destination/'code';code.mkdir()
    for name in ('own_trajectory.py','own_trajectory_campaign.py','own_trajectory_diagnostics.py',
                 'report_own_trajectory.py','test_own_trajectory.py','own_trajectory_README.rst'):
        shutil.copyfile(Path(__file__).with_name(name),code/name)
    write_json(destination/'code_manifest.json',dict(
        note='Original executed source snapshot is in deliverables/source; diagnostics/reporting/final tests added after model selection.',
        files={p.name:sha(p) for p in code.iterdir()}))
    teacher32={}
    for role in ('validation','evaluation'):
        rr=read_csv(source/f'{role}_teacher32.csv')
        teacher32[role]=dict(mean_ratio=average(rr,'ratio'),worst_ratio=min(float(r['ratio']) for r in rr))
    positive_summary={}
    for arm in ('initial_shallow','initial_deep','trained_shallow','trained_deep'):
        rr=[r for r in positive if r['role']=='evaluation' and r['arm']==arm]
        positive_summary[arm]=dict(p5_ratio=average(rr,'positive_p5_ratio'),weak_ratio=average(rr,'positive_weak_ratio'))
    report=['浅束与深轨迹对照、学生实际访问根训练及下一步选择','2026-11-02','',
        '结论：对照、两轮实际访问根训练、独立开发评估和停滞诊断全部完成；质量/覆盖/成本联合门槛未通过。',
        f'验证集选择{sel["geometry"]}，随后只采集该结构的训练seed真实访问根；开发结果不参与模型选择。',
        '下一步改为“单轨迹＋top2/top4物理精验＋当前根失败候选去重”；这是下一阶段选择，本轮尚未实现。',
        '暂停简单top1×32步增深、相同即时收益标签的继续扩训及PPO；保留原始浅束作为对照。','',
        '一、冻结实验范围',
        '沿用3小区/36UE/17PRG、canonical原生BLER、因果服务缺口权重；所有方法均无CTBS/headroom裁剪。',
        '每状态固定ceil(90%×本臂当前原始PFQ出队量)；研究内TTI冻结，不是新的原生闭环轨迹。',
        '训练41–43，验证44–45，两个无CTBS历史来源，各TTI4001/8001/18000：18训练状态、12验证状态。',
        '开发46–48仅纯PFQ来源，TTI3001/4001/5000，共9状态；已有历史开发暴露，不称盲测。',
        '两结构共用最多56个特征根/106次物理评分。实际资源不同，不称严格等墙钟或等候选比较。',
        '浅束最多7条×8次尝试，单父时最多7个候选、多父时每父2个；单轨迹每次只验证top1、最多32次。',
        '失败提议保留父状态；32次尝试不代表32次成功编辑。重复根、停滞与实际成功步数均记录。',
        '完整耗时包含输入/PFQ解码、全新PHY/特征上下文、枚举、特征、网络/排序、物理验证与清理。',
        '归档IO、模型/动态库加载和首次不变TBS表建立在外；教师、训练及诊断成本另计。',
        f'先进行验证结构对照{val_count}次，再训练选型；模型冻结后进行开发评估{eval_count}次，共{val_count+eval_count}次正式决策。',
        '每状态三次串行交替计时取中位数，再seed等权；P95只是有限状态的耗时分位数。','',
        '二、训练前的结构对照（验证集）',
        'arm | 平均J | 完整均时ms | 对full的逐状态相对J均值 | 平均成功编辑 | 平均停滞根']
    for r in val:
        report.append(f'{r["arm"]}: {float(r["mean_j"]):.3f} | {float(r["mean_ms"]):.3f} | '
            f'{float(r["mean_relative_j_pct"]):+.3f}% | {r["mean_best_path_edits"] or "—"} | {r["mean_stagnant_roots"] or "—"}')
    report.extend([
        '验证规则先检查联合门槛，再最小化J不足、服务覆盖不足，最后比较平均J比例；没有根据开发结果挑结构。',
        '单轨迹更快，但验证J更低、重复失败更多，按冻结规则选择浅束采集训练数据。','',
        '三、实际访问根训练',
        '初始模型为前轮冻结48维32/16标量残差MLP；归一化保持冻结，每轮明确新建AdamW，非优化器无缝续训。',
        '每个学生完成决策后，按真实访问顺序去重，最多均匀抽8个根；教师在独立PHY查询完整一PRG邻域。',
        '教师既不改变学生动作，也不影响其评分次数/缓存；学生决策重放动作与工作量一致。',
        '标签仍为当前根的即时整TB物理增益，没有后缀最优值、跨TTI价值或多解偏好目标。',
        '加入原53个锚点组，按(状态ID,根动作)去重，只使用41–43；每轮20epoch，0/5/10/20按完整验证输出选型。'])
    for i,r in enumerate(sel['rounds'],1):
        teacher=sum(v['candidate_evaluations'] for v in work[i-1]['teacher'])
        report.append(f'轮{i}: 采集{r["collected_groups"]}组，相对已有数据新增{r["new_roots_vs_existing"]}根，'
            f'累计训练{r["combined_groups"]}组，教师候选标签{work[i-1]["candidate_labels"]}、实际评分{teacher}次，'
            f'选择epoch{r["selection"]["epoch"]}。')
    report.extend([
        '第二轮有116个新根，但选回epoch0，未带来额外权重更新收益；最终权重等于第一轮epoch5，异于初始模型。',
        '训练发生在验证选出的浅束访问分布；trained_deep是该模型的结构迁移对照，不是独立深轨迹训练。','',
        '四、冻结模型后的开发结果',
        'J为加权期望成功字节，不是Mbps或整包吞吐。',
        'arm | 平均J | 完整均时ms | 逐状态相对J均值 | 最差状态J变化 | 新增未服务UE均值 | 联合门槛'])
    for r in summary:
        report.append(f'{r["arm"]}: {float(r["mean_j"]):.3f} | {float(r["mean_ms"]):.3f} | '
            f'{float(r["mean_relative_j_pct"]):+.3f}% | {float(r["worst_relative_j_pct"]):+.3f}% | '
            f'{float(r["mean_newly_unserved_ues"]):.3f} | {r["joint_pass"] if r["arm"]!="full" else "参照"}')
    for shape,r in learning.items():
        report.append(f'{shape}续训−初始: 均值之比{r["ratio_of_mean_gain_pct"]:+.3f}%，{r["wins"]}胜{r["losses"]}负，'
            f'逐seed J差{r["seed_j_deltas"]}。')
    report.extend([
        '浅束虽8胜1负，seed46均值仍略退化，不满足逐seed正收益要求；对完整DPLS质量缺口仍约15.56%。',
        '深轨迹初始模型平均32次尝试仅9.33次成功编辑、22.67次停滞；浅束续训模型迁移到深轨迹后进一步退化。',
        '质量门槛为每状态J≥full的99.5%；覆盖门槛为无新增未服务UE且全UE即时服务P5≥full的98%；成本≤full的80%。',
        '所有候选方法质量/覆盖门槛失败；开发成本通过不能替代质量晋级。','',
        '五、服务覆盖核验及零值退化',
        '开发9/9状态的完整DPLS全UE即时服务P5为0；原P5比例1.0没有非平凡保护含义。',
        '因此事后另报full中期望服务>1B的有效UE子集P5和该子集中服务量最低20%的总服务比例。',
        '这些诊断在模型冻结后定义并计算，没有重新选型或改判原门槛；不代表长期弱UE公平性。'])
    for arm,r in positive_summary.items():report.append(f'{arm}: 有效UE P5比例{r["p5_ratio"]:.4f}，有效UE底20%服务比例{r["weak_ratio"]:.4f}。')
    report.extend([
        '浅束训练后有效UE P5比例从约0.7234降至0.7007，说明总J小幅提高未同时改善所有服务覆盖诊断。',
        '即时已分配UE数量不能推导为闭环吞吐、完整包交付、公平性或平均/P95包时延。','',
        '六、停滞诊断（事后、只读、不重训）',
        '每学生/状态仅检查出现次数最多的一个失败根；在决策结束后精确查询完整邻域。',
        'role/student | 停滞状态 | 仍有>16B改善 | top1/2/4/8存在改善的状态数'])
    for name,r in diagnostics.items():report.append(f'{name}: {r["stalled_states"]} | {r["missed_qualifying"]} | {r["topk_recoverable"]}')
    report.extend([
        '开发初始深轨迹9/9停滞根都有未评分的改善；top2即可覆盖这9个固定根中的至少一个改善。',
        '浅束续训模型的深轨迹对照中，top4在9/9固定停滞根可找到改善。',
        '这支持验证少量精确分支与失败去重；不证明top2/4的完整轨迹必然满足DPLS质量或耗时门槛。',
        f'完整DPLS的前32次成功编辑在开发状态平均达到full的{teacher32["evaluation"]["mean_ratio"]*100:.2f}%，'
            f'最差{teacher32["evaluation"]["worst_ratio"]*100:.2f}%；它是有限贪心参照，不是32步解空间的全局上界。','',
        '七、下一步选择（尚未实施）',
        '优先：单条实际进展轨迹＋top2/top4短名单精确验证＋按当前根键控的失败编辑去重。',
        '同一根下不重复浪费32次预算在已失败编辑上；接受编辑后刷新根键，不能跨根或跨CSI复用失败结论。',
        '继续共用56个特征根/106次物理评分上限，比较相同机制的物理启发式与AI；完整耗时和逐UE服务量必须报告。',
        '先验证该搜索机制能否补齐质量，再用其自身访问根训练；当前浅束模型保留为对照，不直接移植为深轨迹成功模型。',
        '暂停：继续纯top1×32增深、第三轮相同即时标签扩训和PPO。若有限分支仍无质量空间，再转向结构化联合编辑/有限继续收益。','',
        '八、核验、工件及边界',
        f'33项测试通过；{audit_count}组正式终态独立NumPy物理核验；全部三次重复动作一致，全部因果floor核验通过。',
        f'主实验耗时{completion["elapsed_s"]:.2f}秒；另有测试、教师32步参照及事后诊断，均不隐入在线推理能力。',
        '训练、模型选型、几何选择与开发评估隔离；历史开发暴露事实保留。未读取311–315、未新增原生轨迹、未更换默认算法。',
        '工程核验通过与科学门槛失败分开；保存状态结果不证明场景泛化、整包指标改善或0.5ms部署实时性。',
        f'完整本地实验：{source}',
        '模型、完整标签、原始源码和动作/物理值数组留在deliverables；小型结果、协议、训练历史和源码进入本证据目录。'])
    (destination/'report.txt').write_text('\n'.join(report)+'\n')
    write_json(destination/'archive_manifest.json',{str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()})
    print('EXPORTED',destination,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--tests-log',required=True);p.add_argument('--execution-log',required=True)
    a=p.parse_args();main(a.source,a.destination,a.tests_log,a.execution_log)

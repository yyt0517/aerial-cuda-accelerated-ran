"""Recompute gates, verify actual rejected-mask labels and export small evidence."""
import argparse
import csv
import json
from pathlib import Path
import platform
import shutil

import numpy as np

from .physics import ROOT
from .experiment import write_json,sha
from .shortlist_trajectory import state_key


def read_csv(path):return list(csv.DictReader(Path(path).open()))


def avg(rows,key):return float(np.mean([float(r[key]) for r in rows]))


def audit_phase(source,phase):
    raw=read_csv(source/f'{phase}_measurements.csv');state=read_csv(source/f'{phase}_per_state.csv')
    summary=read_csv(source/f'{phase}_summary.csv')
    for r in state:
        matches=[v for v in raw if v['state_id']==r['state_id'] and v['arm']==r['arm']]
        assert len(matches)==3 and np.isclose(float(r['e2e_ms']),np.median([float(v['e2e_ms']) for v in matches]))
        if r['arm'] not in ('full','legacy_top1'):
            assert int(r['candidate_evaluations'])<=106 and int(r['feature_roots'])<=56
            assert r['full_search_calls']=='0'
            if not r['arm'].startswith('repeat_'):assert r['repeated_exact_candidates']=='0'
    for row in summary:
        rr=[r for r in state if r['arm']==row['arm']]
        for field,target in [('score','mean_j'),('e2e_ms','mean_ms')]:
            computed=np.mean([avg([r for r in rr if r['seed']==s],field) for s in sorted({r['seed'] for r in rr})])
            assert np.isclose(computed,float(row[target]))
        q=all(float(r['ratio'])>=.995 for r in rr)
        c=all(int(r['newly_unserved_ues'])==0 and float(r['positive_p5_ratio'])>=.98 and float(r['positive_weak_ratio'])>=.98 for r in rr)
        assert (row['quality_pass']=='True')==q
        assert (row['coverage_pass']=='True')==c
        assert (row['cost_pass']=='True')==(float(row['time_ratio'])<=.8)
    return summary,state,len(raw)


def main(source,destination,tests_log,execution_log):
    import torch
    source=Path(source).resolve();destination=Path(destination).resolve()
    protocol=json.loads((source/'protocol.json').read_text());sel=json.loads((source/'selection.json').read_text())
    completion=json.loads((source/'completion.json').read_text());assert completion['completed']
    phase_a,val_rows,n_val=audit_phase(source,'phase_a')
    summary,rows,n_eval=audit_phase(source,'evaluation');by_arm={r['arm']:r for r in summary}
    own=torch.load(source/'own_states.pt',weights_only=False)
    work=json.loads((source/'own_work.json').read_text())
    visits={(v['state_id'],v['root_sha256'],v['failed_sha256']) for v in work['visits']}
    for g in own:
        key=state_key(g['state_id'],g['root'],g['failed_before'])
        assert key in visits and key[1]==g['root_sha256'] and key[2]==g['failed_sha256']
        assert g['seed'] in (41,42,43)
        assert not g['eligible'][g['failed_before']].any()
    combined=torch.load(source/'combined_training.pt',weights_only=False)
    assert all(g['seed'] in (41,42,43) for g in combined)
    assert len({state_key(g['state_id'],g['root'],g.get('failed_before',np.empty(0,np.int64))) for g in combined})==len(combined)
    assert work['student_replay_identical'] and work['post_decision_teacher']
    paired=read_csv(source/'paired_learning.csv');learning={}
    for k in (2,4):
        rr=[r for r in paired if int(r['top_k'])==k]
        deltas={s:avg([r for r in rr if int(r['seed'])==s],'j_delta') for s in (46,47,48)}
        wins=sum(float(r['j_delta'])>1 for r in rr);losses=sum(float(r['j_delta'])<-1 for r in rr)
        learning[k]=dict(wins=wins,losses=losses,seed_j_deltas=deltas,
            mean_j_delta=avg(rr,'j_delta'),ratio_of_mean_gain_pct=100*(float(by_arm[f'trained_{k}']['mean_j'])/float(by_arm[f'initial_{k}']['mean_j'])-1),
            stable_learning=sel['weights_changed'] and wins>=6 and all(v>0 for v in deltas.values()),
            joint_pass=by_arm[f'trained_{k}']['joint_pass']=='True')
    terminal=read_csv(source/'terminal_diagnostics/terminal_opportunities.csv');terminal_summary={}
    for role in ('validation','evaluation'):
        for arm in ('initial_2','initial_4','trained_2','trained_4'):
            rr=[r for r in terminal if r['role']==role and r['arm']==arm]
            terminal_summary[f'{role}:{arm}']=dict(states=len(rr),remaining_gain_gt16=sum(float(r['best_gain'])>16 for r in rr),
                top4_gain_gt16=sum(float(r['top4_best_gain'])>16 for r in rr),
                median_first_positive_rank=float(np.median([int(r['first_positive_rank']) for r in rr])),mean_best_gain=avg(rr,'best_gain'))
    selected_k=sel['top_k']
    next_path=dict(selected='Freeze models and measure PHY-budget/quality curves before any further training',
        implementation_status='chosen next experiment, not run in this campaign',
        grid=dict(top_k=[2,4],max_candidates=[106,160,212],max_feature_roots=56),
        reason='all development paths exhaust106 scores and every terminal initial/trained state retains >16B improvement',
        gates='select only on validation, retain all-state99.5% J, positive-UE service and <=80% full cost criteria',
        learning='original/trained AI and matching heuristic share every cache/negative-mask mechanism; no new fine tuning yet',
        conditional='if extra budget cannot reach quality under full cost gate, investigate score/regret and structured joint edits/continuation targets',
        limitations='terminal opportunities do not prove that larger budgets recover full DPLS or improve packet metrics')
    gates=dict(learning=learning,quality_coverage_cost={r['arm']:{k:r[k]=='True' for k in
        ('quality_pass','coverage_pass','cost_pass','joint_pass')} for r in summary if r['arm']!='full'},
        selected_k_before_training=selected_k,final_training_epoch=sel['training_selection']['epoch'],
        independent_ai_joint_target=False,terminal=terminal_summary,next_path=next_path)
    assert not any(r['joint_pass']=='True' for r in summary if r['arm']!='full')
    assert sha(ROOT/'training/stageb_hgraph/current_algorithm.json')==protocol['registry_sha256']
    tests=Path(tests_log).read_text();assert 'Ran 41 tests' in tests and '\nOK\n' in tests
    shutil.copyfile(tests_log,source/'tests.log');shutil.copyfile(execution_log,source/'execution.log')
    write_json(source/'gate_results.json',gates)
    phy_audits=sum(len(json.loads((source/f'{phase}_checks.json').read_text())['independent_phy']) for phase in ('phase_a','evaluation'))
    physical_roots=len({(g['state_id'],g['root_sha256']) for g in own})
    write_json(source/'final_audit.json',dict(passed=True,tests=41,formal_decisions=n_val+n_eval,
        independent_final_phy_checks=phy_audits,actual_own_states=len(own),unique_physical_own_roots=physical_roots,
        mask_provenance_and_training_seeds_verified=True,summary_gates_recomputed=True,
        posthoc_terminal_diagnostics_no_retraining=True,registry_unchanged=True,native_trajectories=0,final_seeds_read=False))
    write_json(source/'runtime_environment.json',dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
        platform=platform.platform(),threads=1,gpu_used=False))
    destination.mkdir(parents=True,exist_ok=False)
    for p in source.iterdir():
        if p.is_file() and p.suffix in ('.csv','.json','.log'):shutil.copyfile(p,destination/p.name)
    shutil.copytree(source/'terminal_diagnostics',destination/'terminal_diagnostics')
    code=destination/'code';code.mkdir()
    for name in ('shortlist_trajectory.py','shortlist_campaign.py','shortlist_diagnostics.py','report_shortlist.py',
                 'test_shortlist_trajectory.py','shortlist_README.rst'):
        shutil.copyfile(Path(__file__).with_name(name),code/name)
    write_json(destination/'code_manifest.json',dict(note='Executed source snapshot stays in deliverables/source; terminal diagnostics and reporter added after final selection.',
        files={p.name:sha(p) for p in code.iterdir()}))
    model_gain=learning[selected_k]
    report=['单轨迹top2/top4精验、根内失败去重与自身状态训练','2026-11-02','',
        '结论：机制对照、一轮自身状态训练及冻结模型开发评估完成；所有新路径质量/服务覆盖联合门槛仍未通过。',
        f'验证集选择top{selected_k}，训练选择epoch{sel["training_selection"]["epoch"]}；非所选K的训练后结果仅作迁移对照。',
        '根内重复评分已消除，预算利用改善，但不等同于AI独立收益或完整调度成功。',
        '下一阶段：冻结模型，先测106/160/212次PHY预算的质量—成本曲线，最多56个特征根；暂不追加训练。','',
        '一、冻结范围与成本口径',
        '沿用3小区、36UE、17PRG与canonical原生BLER；当前因果服务缺口权重及本状态原始PFQ的90%出队下界。',
        '教师、训练、推理、启发式及消融对照均无CTBS/headroom裁剪，无PPO、主动丢包或在线完整DPLS调用。',
        '初始检查点是前轮原始48维标量残差模型；上轮浅束专用续训检查点不作为本轮起点。',
        '所有新臂共用候选/特征/排序的同根缓存；根改变即销毁失败索引和评分布局，新决策/CSI没有旧缓存。',
        'top2/top4批量精确验证，在符合原始floor的候选中接纳增益>1B者；这是原标量路径规则。',
        'full参照保留其原有1/16B阈值。不同路径不保证相同局部解，未以宽松阈值重新定义full。',
        '每臂最多56个特征根、106次实际PHY评分（含草案）；同K重复短名单臂也有特征缓存，唯独不跳过失败项。',
        '数值评分使用原完整邻域布局/行位置；缓冲构建、聚合与销毁均在完整计时内。',
        '完整耗时含输入重建/PFQ解码、新PHY/缓存、候选、特征、模型/排序、验证、数值归约与清理。',
        '归档/模型/库IO及首次不变TBS表准备在外；离线教师、训练和事后诊断单独记录。',
        '训练41–43、验证44–45，各两个无CTBS来源、固定TTI4001/8001/18000；18训练状态、12验证状态。',
        '开发46–48仅纯PFQ来源、TTI3001/4001/5000，共9状态；历史暴露保留，不称盲测。',
        f'验证机制对照{n_val}次、最终开发{n_eval}次，共{n_val+n_eval}次串行决策；每状态三次取中位数，再seed等权。','',
        '二、训练前的验证机制对照',
        'arm | 平均J | 完整均时ms | 逐状态相对J均值 | 重复精确评分均值 | 有效UE P5比']
    for r in phase_a:report.append(f'{r["arm"]}: {float(r["mean_j"]):.3f} | {float(r["mean_ms"]):.3f} | '
        f'{float(r["mean_relative_j_pct"]):+.3f}% | {r["mean_repeated_exact"] or "—"} | {float(r["mean_positive_p5"]):.4f}')
    report.extend([
        '按联合门槛、J不足、有效服务覆盖不足、平均J比例的冻结顺序选K，未根据开发结果回选。',
        'top4的验证质量/成本点优于top2，故只对top4机制做一轮限额自身状态训练；首轮尚未过最终质量门槛。',
        '这里选择训练是预声明的诊断轮次，不将机制正确性当作质量晋级。','',
        '三、实际访问状态训练',
        '一个状态含保存状态ID、当前动作根和失败前缀掩码；同物理根的不同拒绝前缀不冒充独立无线场景。',
        '学生完成决策后最多均匀抽8个真实访问状态/TTI，在独立PHY上查询完整邻域；可选标签排除学生已失败索引。',
        f'采集{len(own)}组（{physical_roots}个不同物理根），候选标签{work["candidate_labels"]}，实际教师评分'
            f'{sum(r["teacher"]["candidate_evaluations"] for r in work["work"])}次。',
        f'与原53锚点去重后{sel["training_groups"]}组，新增{sel["new_states"]}状态键；全部只来自41–43。',
        '继承标量权重和冻结归一化，明确新建AdamW；20epoch、0/5/10/20按完整验证轨迹选型。',
        '教师不影响学生动作/工作量，离线教师查询后重放学生的动作和评分次数完全一致。',
        '标签仍是即时完整TB增益，没有跨TTI回报或最优继续价值监督。','',
        '四、冻结模型后的开发结果（J为加权期望成功字节，非Mbps）',
        'arm | 平均J | 完整均时ms | 对full的逐状态相对J均值 | 最差状态J | 新增未服务UE均值 | 有效UE P5比 | 联合门槛'])
    for r in summary:report.append(f'{r["arm"]}: {float(r["mean_j"]):.3f} | {float(r["mean_ms"]):.3f} | '
        f'{float(r["mean_relative_j_pct"]):+.3f}% | {float(r["worst_relative_j_pct"]):+.3f}% | '
        f'{float(r["mean_newly_unserved"]):.3f} | {float(r["mean_positive_p5"]):.4f} | '
        f'{r["joint_pass"] if r["arm"]!="full" else "参照"}')
    report.extend([
        '质量门槛：每状态J≥full的99.5%。服务覆盖：相对full无新增未分配UE（COUNT>0口径），',
        'P5与底20%服务量另限定在full期望服务>1B的有效UE子集，均须≥参照98%。成本：seed等权完整均时≤full的80%。',
        '有效UE指标避免之前全UE P5=0的退化；这些只是瞬时服务门槛，不是长期弱UE公平性或整包交付结论。',
        '所有新路径均通过成本门槛但未通过质量/覆盖；不能以明显降时替代完整调度质量。',
        '重复消融与去重共享特征缓存；去重消除失败重复属于工程收益。AI只与相同K、缓存、去重和评分实现的启发式比较。'])
    for k,r in learning.items():report.append(f'训练后top{k}−初始同K: 均值之比{r["ratio_of_mean_gain_pct"]:+.3f}%，'
        f'{r["wins"]}胜{r["losses"]}负，逐seed J差{r["seed_j_deltas"]}。')
    report.extend([
        f'所选top{selected_k}续训在验证集改善，但开发均值变化{model_gain["ratio_of_mean_gain_pct"]:+.3f}%，学习没有稳定迁移。',
        'top2训练后结果是top4学生数据上的迁移对照；不能看到其开发均值更好后追改本轮所选K。',
        '两个K都未满足逐seed正收益与完整联合门槛；本轮不晋级模型或默认算法。','',
        '五、终态剩余机会诊断（模型冻结后、只读）',
        '所有开发新臂均耗尽106次物理评分；用独立PHY检查终态完整原邻域，并排除该终态已经失败的候选。',
        '终态独立查询和重放不计为免费在线能力，也未用于重训或重新选型。',
        'role/arm | 状态数 | 仍有>16B改善 | top4中有改善 | 首个正收益排名中位数 | 最好一步增益均值'])
    for name,r in terminal_summary.items():report.append(f'{name}: {r["states"]} | {r["remaining_gain_gt16"]} | '
        f'{r["top4_gain_gt16"]} | {r["median_first_positive_rank"]:.1f} | {r["mean_best_gain"]:.3f}B')
    report.extend([
        '开发初始与训练后的top2/top4共36个终态，全部仍有>16B改善。预算是可验证限制之一，排序也仍会遗漏高收益候选。',
        '单步剩余机会不证明增加预算一定达到full；不同局部解、服务覆盖与计算成本仍要一起验证。','',
        '六、下一步路径（本轮已选择，尚未运行）',
        '1. 冻结原始和本轮训练模型，先测top2/top4×PHY预算106/160/212的完整质量—成本曲线；特征根仍≤56。',
        '2. 仅按验证集选择预算/K，保持99.5%J、有效服务覆盖与80%完整耗时门槛，不借开发结果追选配置。',
        '3. AI和启发式共享缓存/失败去重/数值评分；同时记录每种预算下实际有效编辑、终态剩余收益和服务退化UE。',
        '4. 暂停追加相同即时标签训练及PPO；若预算增加仍无法在成本门槛内保质，再研究联合编辑或有限继续收益目标。',
        '5. 当前top4模型与top2迁移结果均只作研究工件，默认调度器保持冻结版本。','',
        '七、核验、工件与边界',
        f'41项测试通过；{phy_audits}组正式终态独立NumPy物理核验；三次重复动作一致，全部因果floor通过。',
        '已独立核对所有训练状态的根/失败掩码哈希、实际访问来源和可选标签；所有去重臂根内重复精确评分为0。',
        f'主实验耗时{completion["elapsed_s"]:.2f}秒；测试、额外终态重放/教师诊断另计。',
        '未读311–315，未跑新原生轨迹，默认算法未改；无场景泛化、整包吞吐/TTL/时延改善或0.5ms实时性结论。',
        f'完整工件：{source}',
        '完整标签、检查点、动作/物理值数组与执行源码留在deliverables；本目录保留小型协议、结果、负结果及新增源码。'])
    (destination/'report.txt').write_text('\n'.join(report)+'\n')
    write_json(destination/'archive_manifest.json',{str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()})
    print('SHORTLIST EXPORTED',destination,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--tests-log',required=True);p.add_argument('--execution-log',required=True);a=p.parse_args()
    main(a.source,a.destination,a.tests_log,a.execution_log)

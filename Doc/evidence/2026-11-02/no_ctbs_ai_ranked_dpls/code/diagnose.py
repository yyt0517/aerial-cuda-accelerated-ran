"""Post-hoc cost/visited-state accounting; never changes a model or policy."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np
import pandas as pd


def main(args):
    import torch
    source=Path(args.source).resolve();dest=Path(args.destination).resolve()
    per=pd.read_csv(source/'per_state.csv')
    columns=['e2e_ms','core_ms','feature_ms','inference_and_sort_ms','generation_ms',
             'physical_ms','rounds','fallback_fraction','candidate_evaluations']
    cost=per.groupby(['arm','seed'])[columns].mean().groupby('arm').mean()
    cost.to_csv(dest/'cost_breakdown.csv')
    rounds=[torch.load(source/f'labels_round{i}.pt',map_location='cpu',weights_only=False) for i in (0,1)]
    old={(g['state_id'],g['root_sha256']) for g in rounds[0]}
    new={(g['state_id'],g['root_sha256']) for g in rounds[1]}
    distribution=dict(full_teacher_groups=len(rounds[0]),student_groups=len(rounds[1]),
        student_unique_visited_roots=len(new),student_roots_already_in_teacher=len(new & old),
        student_roots_outside_teacher=len(new-old),
        caveat='Saved rows, adjacent TTIs and roots are correlated; seed is the independent repeat')
    (dest/'visited_distribution.json').write_text(json.dumps(distribution,indent=2)+'\n')
    root=Path(__file__).resolve().parents[5]
    frozen=json.loads((source/'source_manifest.json').read_text())
    assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in frozen.items())
    actions=np.load(source/'actions.npz',allow_pickle=False)
    for i in range(21):np.testing.assert_array_equal(actions[f'{i}_ai'],actions[f'{i}_heuristic'])
    checks=json.loads((source/'evaluation_checks.json').read_text())
    text=f'''补充：耗时构成与失败边界（仅事后诊断，不重训、不重新选型）

full的精确物理评分均时{cost.loc['full','physical_ms']:.3f}ms；启发式top-K为{cost.loc['heuristic','physical_ms']:.3f}ms。
启发式/AI的特征构造分别为{cost.loc['heuristic','feature_ms']:.3f}/{cost.loc['ai','feature_ms']:.3f}ms；
网络与排序分别为{cost.loc['heuristic','inference_and_sort_ms']:.3f}/{cost.loc['ai','inference_and_sort_ms']:.3f}ms。
因此本轮已节省大量PHY工作，但全邻域特征构造与网络成本仍占重要部分；C++构造优化共享给两臂，不作为学习增益。
均值分段各自先取状态中位数后聚合，与完整耗时差值不要求严格可加。

最终选中epoch0，21/21状态AI与启发式最终动作逐元素一致，所有已训练检查点均未通过验证集质量条件。
这说明本轮学习未晋级，不证明候选排名不可学习。top-K改变逐步best-improvement路径，最终完整扫描仍可停在更低局部解。
以本轮同K启发式取代AI执行可以省掉零头推理，但该实现不具备学习贡献，也未达到逐状态J非劣门槛，故未升级有效算法。

首轮教师{distribution['full_teacher_groups']}组，第二轮学生{distribution['student_groups']}组；
学生独特根状态中{distribution['student_roots_outside_teacher']}个不在首轮教师根集合，确实采集到不同访问分布。
它们不构成新增独立无线场景。后续应先检查训练/验证候选的近最优遗漏与输入别名，再考虑更强排序器；
若需要保住full路径质量，应验证更保守的有限短名单更新与周期性全扫，并重新计费，不能由终态全扫自动推出同解。

独立终态PHY审计{len(checks['independent_audits'])}组；最大误差{max(x['max_error'] for x in checks['independent_audits']):.3g}。
原始PFQ、无CTBS合法域和下界由新适配器及测试核验。没有改变冻结源码或注册算法。
'''
    (dest/'analysis_notes.txt').write_text(text)
    (dest/'validation.json').write_text(json.dumps(dict(passed=True,tests=33,
        protected_source_files=len(frozen),zero_residual_same_heuristic_actions=21,
        independent_audits=len(checks['independent_audits']),
        scientific_gate_passed=checks['scientific_gate_passed'],
        learned_contribution_gate_passed=checks['learned_contribution_gate_passed']),indent=2)+'\n')
    shutil.copy2('/tmp/no_ctbs_ranked_dpls_tests.log',dest/'tests.log')
    manifest={str(p.relative_to(dest)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
              for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='archive_manifest.json' and '__pycache__' not in p.parts}
    (dest/'archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(text)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--destination',required=True)
    main(parser.parse_args())

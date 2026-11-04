"""Audit all frozen trajectories before reporting neural/physical comparisons."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment import ROOT, sha, write_json


PAIRS=(('risk_neural','legacy_current'),
       ('risk_neural','physical_lns'),
       ('risk_neural','risk_physical_lns'),
       ('risk_neural','risk_physical_direct'),
       ('risk_physical_lns','physical_lns'))
METRICS=('goodput_mbps','full_payload_mbps','expiry_pct','ue_goodput_p5_mbps',
         'success_delay_mean_ms','success_delay_p95_ms','candidates_mean',
         'decision_mean_ms','neural_option_scores_mean','neural_logit_calls_mean')


def paired(frame,seeds):
    assert len(seeds)==3,'The frozen descriptive t interval is specified for three seeds'
    indexed=frame.set_index(['seed','arm']);rows=[]
    # Analytic inverse of the df=2 Student-t CDF at p=.975.
    tcrit=np.sqrt(2*.95**2/(1-.95**2))
    for arm,reference in PAIRS:
        a=indexed.xs(arm,level='arm').loc[seeds]
        b=indexed.xs(reference,level='arm').loc[seeds]
        for metric in METRICS:
            delta=a[metric]-b[metric];half=tcrit*delta.std(ddof=1)/np.sqrt(len(seeds))
            rows.append(dict(treatment=arm,reference=reference,metric=metric,
                treatment_mean=float(a[metric].mean()),reference_mean=float(b[metric].mean()),
                delta_mean=float(delta.mean()),nominal_ci95_low=float(delta.mean()-half),
                nominal_ci95_high=float(delta.mean()+half),
                mean_ratio=None if b[metric].mean()==0 else float(a[metric].mean()/b[metric].mean()),
                positive_seeds=int((delta>1.e-10).sum()),negative_seeds=int((delta<-1.e-10).sum()),
                unchanged_seeds=int((delta.abs()<=1.e-10).sum())))
    return pd.DataFrame(rows)


def summarize(output):
    output=Path(output).resolve();protocol=json.loads((output/'protocol.json').read_text())
    completion=json.loads((output/'completion.json').read_text());assert completion['passed']
    expected={(s,a) for s in protocol['seeds'] for a in protocol['arms']}
    assert len(completion['runs'])==len(expected)
    assert {(r['seed'],r['arm']) for r in completion['runs']}==expected
    assert all(r['returncode']==0 for r in completion['runs'])
    assert sha(output/'plan_frozen.json')==protocol['plan_sha256']
    assert sha(output/'repair_frozen.npz')==protocol['checkpoint_sha256']
    assert sha(ROOT/'training/stageb_hgraph/current_algorithm.json')==protocol['active_registry_sha256']
    rows=[];ue_rows=[];audit=[];manifest={}
    for seed in protocol['seeds']:
        reference=output/f'seed{seed}_legacy_current'
        ref_launch=json.loads((reference/'launch.json').read_text())
        with np.load(reference/'prefix.npz') as saved:
            prefix={key:saved[key].copy() for key in saved.files}
        for arm in protocol['arms']:
            directory=output/f'seed{seed}_{arm}'
            launch=json.loads((directory/'launch.json').read_text())
            execution=json.loads((directory/'execution.json').read_text())
            metrics=json.loads((directory/'metrics.json').read_text())
            assert execution['passed'] and execution['all_tti_action_bytes_exact'] and execution['causal_fresh_inputs_verified']
            assert metrics['censored_packets']==0
            assert launch['source_sha256']==protocol['source_sha256']
            for name,digest in launch['source_sha256'].items():
                assert sha(directory/'source'/name)==digest
            for key in ('warmup','settling','measured','followup','width'):
                assert launch[key]==protocol[key]
            assert launch['checkpoint_sha256']==protocol['checkpoint_sha256']
            assert launch['checkpoint_used_for_inference']==(arm=='risk_neural')
            for key in ('simulator_sha256','libcumac_sha256'):
                assert launch[key]==ref_launch[key]
            assert sha(directory/'traffic_offered_packets.csv')==sha(reference/'traffic_offered_packets.csv')
            with np.load(directory/'prefix.npz') as saved:
                for key in ('input_hashes','actions','goodput'):
                    np.testing.assert_array_equal(saved[key][:protocol['warmup']],prefix[key][:protocol['warmup']])
                for key in ('channel_fingerprints','precoder_fingerprints','fresh_input_tti'):
                    np.testing.assert_array_equal(saved[key],prefix[key])
            decisions=pd.read_csv(directory/'decisions.csv')
            np.testing.assert_array_equal(decisions.tti,np.arange(protocol['warmup']+1,launch['horizon']+1))
            floor=(90*decisions.draft_dequeue.to_numpy(dtype=np.int64)+99)//100
            assert (decisions.chosen_dequeue>=floor).all()
            if 'baseline_score' in decisions:
                assert (decisions.score>=decisions.baseline_score-1.e-7).all()
            for key,execution_key in (('calls','neural_calls'),('proposals','neural_proposals'),
                                      ('option_scores','neural_option_scores'),('logit_calls','neural_logit_calls')):
                assert int(decisions['neural_'+key].sum())==execution[execution_key]
            if arm!='risk_neural':
                assert (decisions.neural_option_scores==0).all()
            first,last=launch['measurement_first_tti'],launch['measurement_last_tti']
            measured=decisions[decisions.tti.between(first,last)]
            assert len(measured)==protocol['measured']
            row=dict(metrics,candidates_mean=float(measured.candidate_evaluations.mean()),
                     decision_mean_ms=float(measured.decision_wall_ms.mean()),
                     neural_option_scores_mean=float(measured.neural_option_scores.mean()),
                     neural_logit_calls_mean=float(measured.neural_logit_calls.mean()))
            data=np.fromfile(directory/'metrics.bin',dtype='<f8').reshape(launch['horizon'],36,19)
            rates=data[first-1:last,:,10].sum(axis=0)*8/(protocol['measured']*.0005)/1.e6
            np.testing.assert_allclose(rates.sum(),metrics['goodput_mbps'],rtol=0,atol=1.e-8)
            np.testing.assert_allclose(np.quantile(rates,.05),metrics['ue_goodput_p5_mbps'],rtol=0,atol=1.e-8)
            for ue,rate in enumerate(rates):
                ue_rows.append(dict(seed=seed,arm=arm,ue_id=ue,goodput_mbps=float(rate)))
            rows.append(row)
            audit.append(dict(seed=seed,arm=arm,ttis=launch['horizon'],
                matured_packets=metrics['cohort_packets'],samples=len(execution['physical_samples']),
                max_sinr_error_db=max(s['sinr_max_abs_db'] for s in execution['physical_samples']),
                max_bler_error=max(s['bler_max_abs'] for s in execution['physical_samples'])))
            manifest[directory.name]={name:sha(directory/name) for name in
                ('launch.json','execution.json','metrics.json','prefix.npz','decisions.csv','metrics.bin','packets.csv')}
    results=output/'summary';results.mkdir(exist_ok=False)
    frame=pd.DataFrame(rows).sort_values(['seed','arm']);frame.to_csv(results/'per_seed.csv',index=False)
    means=frame.groupby('arm')[list(METRICS)].mean();means.to_csv(results/'means.csv')
    frame.groupby('arm')[list(METRICS)].std().to_csv(results/'std.csv')
    comparisons=paired(frame,protocol['seeds']);comparisons.to_csv(results/'paired_comparisons.csv',index=False)
    pd.DataFrame(ue_rows).to_csv(results/'per_ue_goodput.csv',index=False)
    checks=dict(passed=True,runs=len(rows),native_ttis=sum(r['ttis'] for r in audit),
        physical_samples=sum(r['samples'] for r in audit),matured_cohort_packets=sum(r['matured_packets'] for r in audit),
        audit=audit,interpretation='Existing per-run native action/FIFO/PHY checks plus source/model/pairing/floor/counter and summary checks; not a new simulator rerun')
    write_json(results/'checks.json',checks);write_json(results/'input_manifest.json',manifest)
    lines=['风险感知直接神经修复：新seed与物理块修复配对实验',
        datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        f'{len(rows)}条轨迹全部完成并通过核验；每条测量10000 TTI（5秒），400 TTI TTL随访。',
        '均值按seed等权。模型冻结，seeds104–106未用于训练、模型选择或前一轮101–103报告。',
        '', '策略 | goodput Mbps | 整包payload Mbps | TTL过期率% | UE P5 Mbps | 平均成功时延ms | 成功P95 ms | 物理候选/决策 | 神经选项打分/决策 | 并发决策ms']
    for arm in protocol['arms']:
        r=means.loc[arm]
        lines.append(f'{arm} | {r.goodput_mbps:.3f} | {r.full_payload_mbps:.3f} | {r.expiry_pct:.4f} | '
            f'{r.ue_goodput_p5_mbps:.3f} | {r.success_delay_mean_ms:.3f} | {r.success_delay_p95_ms:.3f} | '
            f'{r.candidates_mean:.1f} | {r.neural_option_scores_mean:.1f} | {r.decision_mean_ms:.3f}')
    lines+=['','配对变化与名义95% t区间（3个seed，df=2）']
    for arm,reference in PAIRS:
        lines.append(f'{arm} − {reference}')
        for metric in METRICS:
            r=comparisons[(comparisons.treatment==arm)&(comparisons.reference==reference)&(comparisons.metric==metric)].iloc[0]
            lines.append(f'  {metric}: {r.delta_mean:+.6f} [{r.nominal_ci95_low:+.6f},{r.nominal_ci95_high:+.6f}]，'
                         f'正差{r.positive_seeds}/3，负差{r.negative_seeds}/3，相同{r.unchanged_seeds}/3。')
    lines+=['','解释与限制',
        'risk_neural从PFQ+CTBS草案直接以冻结网络提出完整块候选；risk_physical_direct从同一类草案出发，用物理评估扩展前缀。两者使用相同风险验收目标、原物理目标保护、出队下界、块图、最多3轮和宽度16，均允许所有合法UE；前缀处理、束选择及随机多样性不同，实际候选数/算量不强制相等。',
        'physical_lns保留完整物理贪心＋物理块修复＋条件精修；risk_physical_lns保留同一未加风险的贪心起点，在块束排序、块验收及条件精修中加入与新神经路径相同的风险目标。它们不是没有搜索的基线。',
        'legacy_current保留历史DPLS的BLER插值，新物理及神经路径使用同一规范化原生曲线。因此与legacy_current的差值是整体路径比较，候选生成机制的判断优先看使用相同物理评分的新对照。',
        'neural_option_scores是网络对束内候选选项的打分数量，不能与昂贵的完整物理候选评估等同，也不能据此推算FLOP。物理候选减少不保证总计算或墙钟减少。',
        '时延同时报告成功包均值和P95；TTL、两类吞吐和P5需共同判断。不能从单一均值推出全面优势。',
        '仅3个seed和有限窗口，不宣称普遍泛化、渐近稳态、统计显著或实时部署能力。区间按seed配对，UE/TTI/包不是独立重复。',
        '当前有效算法注册表未变；本轮不根据结果改模型、追加seed或改门槛。',
        '详细证据见summary/{per_seed,means,std,paired_comparisons}.csv及checks.json。']
    report=output/'final_report.txt'
    with report.open('x') as stream:stream.write('\n'.join(lines)+'\n')
    write_json(output/'report_completion.json',dict(passed=True,report_sha256=sha(report)))
    print('REPORT_READY',report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    summarize(parser.parse_args().campaign)

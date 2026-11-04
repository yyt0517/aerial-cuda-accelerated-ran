"""Frozen BC/PPO endpoints, paired development evaluation and complete cohorts."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .model import JointActor, observation, native_action
from .native import ABLATION, REFERENCE, NativeEnv, legacy, packets, sha, write


def metrics(service, cohort, first, last, seed, arm):
    s=service[service.tti.between(first,last)]
    pk=cohort[cohort.arrival_tti.between(first,last)]
    assert len(s)==36*(last-first+1) and len(pk)>0 and not pk.censored.any()
    success=pk[pk.successfully_delivered]
    seconds=(last-first+1)*.0005
    per=success.groupby('ue_id').packet_size_bytes.sum().reindex(range(36),fill_value=0)*8/seconds/1.e6
    return dict(seed=seed,arm=arm,first_tti=first,last_tti=last,measurement_seconds=seconds,
        goodput_mbps=float(s.successful_bytes.sum()*8/seconds/1.e6),
        complete_payload_mbps=float(per.sum()),complete_success_pct=float(100*pk.successfully_delivered.mean()),
        expiry_pct=float(100*pk.outcome.eq('expired').mean()),
        success_delay_mean_ms=float(success.system_time_ms.mean()) if len(success) else None,
        success_delay_p95_ms=float(success.system_time_ms.quantile(.95)) if len(success) else None,
        ue_complete_payload_p5_mbps=float(per.quantile(.05)),
        prg_utilization=float(s.actual_prg_count.sum()/((last-first+1)*51)),
        cohort_packets=len(pk),pending_cohort=0)


def evaluate_one(root, arm, seed, cfg):
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled=False
    root=Path(root)
    checkpoint=root/('bc.pt' if arm=='bc' else 'ppo_final.pt')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    actor=JointActor();actor.load_state_dict(state['actor']);actor.eval()
    warmup,settling,measured,followup=[cfg[f'validation_{k}'] for k in ['warmup','settling','measured','followup']]
    env=NativeEnv(root/f'eval_{arm}_seed{seed}',seed,warmup+settling+measured+followup)
    elapsed=[]
    try:
        for _ in range(env.horizon):
            if env.tti<warmup:
                action=env.raw_pfq()
            else:
                start=time.monotonic()
                x,mask=observation(env.obs,env.inputs)
                actions,_,_=actor.sample(x,mask,greedy=True)
                action=native_action(actions)
                elapsed.append(time.monotonic()-start)
            env.step(action)
            if env.tti%2000==0:
                print('EVAL',arm,seed,env.tti,flush=True)
        env.close()
    except BaseException:
        if not env.log.closed:env.close(validate=False)
        raise
    service,pk=packets(env.out)
    first,last=warmup+settling+1,warmup+settling+measured
    cohort=pk[pk.arrival_tti.between(first,last)]
    cohort.to_csv(env.out/'cohort.csv.gz',index=False)
    # Check the entire common prefix, not merely launch settings.
    ref=REFERENCE/f'seed{seed}_pfq0_20000'
    before=np.load(ref/'prefix.npz')
    after=np.load(env.out/'actions.npz')
    np.testing.assert_array_equal(before['input_hashes'][:warmup],after['input_hashes'][:warmup])
    np.testing.assert_array_equal(before['actions'][:warmup].reshape(-1,3,17),after['actions'][:warmup])
    offered=pd.read_csv(env.out/'traffic_offered_packets.csv')
    expected=pd.read_csv(ref/'traffic_offered_packets.csv')
    pd.testing.assert_frame_equal(offered[offered.arrival_tti<=env.horizon].reset_index(drop=True),
        expected[expected.arrival_tti<=env.horizon].reset_index(drop=True))
    result=metrics(service,pk,first,last,seed,arm)
    result.update(checkpoint_sha256=sha(checkpoint),native_validation=True,common_pfq_prefix_exact=warmup,
        mean_shared_host_policy_ms=1000*float(np.mean(elapsed)))
    write(env.out/'evaluation.json',result)
    return result


def evaluate_campaign(root,cfg):
    root=Path(root)
    write(root/'status.json',dict(stage='evaluation',seeds=cfg['validation_seeds'],arms=['bc','ppo']))
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(evaluate_one,str(root),arm,seed,cfg) for arm in ['bc','ppo'] for seed in cfg['validation_seeds']]
        for future in as_completed(jobs):
            rows.append(future.result())
            write(root/'evaluation_progress.json',rows)
    b=legacy()
    first=cfg['validation_warmup']+cfg['validation_settling']+1
    last=first+cfg['validation_measured']-1
    for seed in cfg['validation_seeds']:
        for arm,folder in [('pfq',REFERENCE/f'seed{seed}_pfq0_20000'),
                ('raw_global_equal',ABLATION/f'native/raw_global_equal/seed{seed}_current_20000')]:
            service=pd.DataFrame(np.fromfile(folder/'metrics.bin',dtype='<f8').reshape(-1,len(b.FIELDS)),columns=b.FIELDS)
            cohort=pd.read_csv(ABLATION/f'packet_cohorts/{arm}_seed{seed}.csv.gz')
            rows.append(metrics(service,cohort,first,last,seed,arm))
    table=pd.DataFrame(rows).sort_values(['seed','arm'])
    table.to_csv(root/'evaluation.csv',index=False)
    gates=[]
    for seed in cfg['validation_seeds']:
        a=table[(table.seed==seed)&(table.arm=='ppo')].iloc[0]
        reference=table[(table.seed==seed)&(table.arm=='raw_global_equal')].iloc[0]
        bc=table[(table.seed==seed)&(table.arm=='bc')].iloc[0]
        tests=dict(payload=a.complete_payload_mbps>=.995*reference.complete_payload_mbps,
            expiry=a.expiry_pct<=reference.expiry_pct+.1,
            mean_delay=a.success_delay_mean_ms<=reference.success_delay_mean_ms,
            p95_delay=a.success_delay_p95_ms<=reference.success_delay_p95_ms,
            ue_p5=a.ue_complete_payload_p5_mbps>=reference.ue_complete_payload_p5_mbps)
        gates.append(dict(seed=seed,checks={k:bool(v) for k,v in tests.items()},passed=bool(all(tests.values())),
            ppo_minus_bc_payload_mbps=float(a.complete_payload_mbps-bc.complete_payload_mbps)))
    result=dict(completed=True,quality_passed=all(x['passed'] for x in gates),seeds=gates,
        promoted=False,selection='fixed final checkpoint',measurement_seconds=cfg['validation_measured']*.0005,
        limits=cfg['limits'])
    write(root/'evaluation_gate.json',result)
    lines=['无 CTBS、等权 PPO 重训练结果', '',
        '模型：新初始化的联合自回归 GRU actor；先以无 CTBS、等权 DPLS 热启动，再进行真实在线 PPO。',
        '训练 seeds 41–43；开发验证 44–45。未使用 311–315。',
        '验证：4000 TTI PFQ 预热＋4000 稳定化＋4000 测量＋400 随访。整包 cohort 无截尾。',
        '对照：同一测量窗口的冻结 PFQ、无 CTBS 等权 DPLS；另评估 BC 起点，以区分 PPO 增益。',
        '',table[['seed','arm','complete_payload_mbps','expiry_pct','success_delay_mean_ms','success_delay_p95_ms']].to_string(index=False),
        '',f'逐 seed 质量门槛通过：{result["quality_passed"]}。没有替换当前默认算法。',
        '这是 2 秒测量的开发验证，不能替代 10 秒长窗口、多场景及新 seed 验证。']
    (root/'report.txt').write_text('\n'.join(lines)+'\n')
    write(root/'status.json',dict(stage='complete',quality_passed=result['quality_passed'],promoted=False))

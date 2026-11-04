"""Paired mature-cohort development checks, without retuning configuration."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .analyze_dpls_moves import write_json,write_csv,sha


def main(args):
    source,out=Path(args.source).resolve(),Path(args.output).resolve()
    if not json.loads((source/'completion.json').read_text())['completed']:raise ValueError('Native campaign incomplete')
    protocol=json.loads((source/'protocol.json').read_text())
    quality=json.loads((source/'quality_protocol.json').read_text())
    if protocol['measured']!=4000 or protocol['followup']<400:raise ValueError('Mature two-second protocol required')
    out.mkdir(parents=True,exist_ok=False)
    comparisons=[];metrics=[];checks=[]
    inputs={str(source/n):sha(source/n) for n in ('protocol.json','quality_protocol.json','completion.json','source_manifest.json')}
    for seed in protocol['seeds']:
        full_dir=source/f'full_seed{seed}';guard_dir=source/f'guarded4_seed{seed}'
        f=json.loads((full_dir/'summary.json').read_text());g=json.loads((guard_dir/'summary.json').read_text())
        if f['pending_cohort'] or g['pending_cohort']:raise AssertionError('Unresolved arrival cohort')
        for d in (full_dir,guard_dir):
            for name in ('validation.json','reward_validation.json'):
                if not json.loads((d/name).read_text())['passed']:raise AssertionError('Native audit failed')
        metrics.extend([dict(mode='full',**f),dict(mode='guarded4',**g)])
        fu=pd.read_csv(full_dir/'per_ue.csv').set_index('ue');gu=pd.read_csv(guard_dir/'per_ue.csv').set_index('ue')
        fpk=pd.read_csv(full_dir/'cohort.csv.gz');gpk=pd.read_csv(guard_dir/'cohort.csv.gz')
        offered=['ue_id','packet_id','arrival_tti','packet_size_bytes','ttl_tti']
        a=fpk[offered].sort_values(['ue_id','packet_id']).reset_index(drop=True)
        b=gpk[offered].sort_values(['ue_id','packet_id']).reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b)
        both=fpk[fpk.successfully_delivered].merge(gpk[gpk.successfully_delivered],on=['ue_id','packet_id'],
            suffixes=('_full','_guard'),validate='one_to_one')
        delta=both.system_time_ms_guard-both.system_time_ms_full
        newzero=set(gu.index[gu.payload_mbps==0])-set(fu.index[fu.payload_mbps==0])
        newgap=set(gu.index[gu.longest_no_successful_byte_ttis>=400])-set(fu.index[fu.longest_no_successful_byte_ttis>=400])
        r=dict(seed=seed,payload_ratio=g['complete_payload_mbps']/f['complete_payload_mbps'],
            payload_delta_mbps=g['complete_payload_mbps']-f['complete_payload_mbps'],
            expiry_delta_pp=g['expiry_pct']-f['expiry_pct'],
            mean_delay_delta_ms=g['success_delay_mean_ms']-f['success_delay_mean_ms'],
            p95_delay_delta_ms=g['success_delay_p95_ms']-f['success_delay_p95_ms'],
            ue_payload_P5_ratio=g['ue_complete_payload_p5_mbps']/f['ue_complete_payload_p5_mbps'],
            worst_expiry_full=f['worst_ue_expiry_pct'],worst_expiry_guard=g['worst_ue_expiry_pct'],
            new_zero_payload_ues=len(newzero),new_TTL_long_service_gap_ues=len(newgap),
            common_success_packets=len(both),common_success_mean_delay_delta_ms=float(delta.mean()),
            mean_PHY_ratio=g['mean_candidate_evaluations']/f['mean_candidate_evaluations'],
            P95_PHY_ratio=g['p95_candidate_evaluations']/f['p95_candidate_evaluations'])
        c=dict(payload_each_seed=r['payload_ratio']>=quality['payload_each_seed_min_ratio'],expiry=r['expiry_delta_pp']<=quality['expiry_max_increase_pp'],
            mean_delay=r['mean_delay_delta_ms']<=quality['mean_success_delay_max_increase_ms'],p95_delay=r['p95_delay_delta_ms']<=quality['p95_success_delay_max_increase_ms'],
            weak_P5=r['ue_payload_P5_ratio']>=quality['ue_payload_P5_min_ratio'],
            worst_ue_expiry=g['worst_ue_expiry_pct']<=max(f['worst_ue_expiry_pct']+2.,5.),
            no_new_zero_payload=not newzero,no_new_TTL_long_service_gap=not newgap)
        checks.append(dict(seed=seed,checks={k:bool(v) for k,v in c.items()},quality_passed=bool(all(c.values())),
            cost_mean_PHY_passed=r['mean_PHY_ratio']<=.25,cost_P95_PHY_passed=r['P95_PHY_ratio']<=.40))
        comparisons.append(r)
        for d in (full_dir,guard_dir):
            for name in ('summary.json','validation.json','reward_validation.json','per_ue.csv','cohort.csv.gz','search.csv'):
                inputs[str(d/name)]=sha(d/name)
    write_csv(out/'native_metrics.csv',metrics);write_csv(out/'paired.csv',comparisons)
    write_json(out/'gates.json',dict(seeds=checks,payload_paired_mean_ratio=float(np.mean([r['payload_ratio'] for r in comparisons])),
        payload_paired_mean_passed=bool(np.mean([r['payload_ratio'] for r in comparisons])>=quality['payload_paired_mean_min_ratio']),
        all_native_quality_passed=bool(all(r['quality_passed'] for r in checks) and np.mean([r['payload_ratio'] for r in comparisons])>=quality['payload_paired_mean_min_ratio']),
        all_native_PHY_cost_passed=all(r['cost_mean_PHY_passed'] and r['cost_P95_PHY_passed'] for r in checks),
        final_promotion=False,interpretation='exposed44/45 development; not five-new-seed10s confirmation'))
    write_json(out/'input_manifest.json',inputs)
    write_json(out/'completion.json',dict(completed=True,new_native_runs=0,new_phy_evaluations=0,
        analysis='only completed native outputs; no parameter/checkpoint/seed reselection'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);main(p.parse_args())

"""Audit pairing and aggregate completed native joint-repair campaigns."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .experiment import write_json,sha


def summarize(primary, compact, output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    primary,compact=Path(primary).resolve(),Path(compact).resolve()
    checks=[];rows=[];runs={};inputs={}
    core=('physics.py','physics_kernel.cpp','_physics.so','search.py','neural_repair.py','quality.py','native_repair.py')
    reference_sources=None
    for campaign,suffix in ((primary,''),(compact,'27')):
        completed=json.loads((campaign/'completion.json').read_text())
        assert completed['passed'] and all(r['returncode']==0 for r in completed['runs'])
        protocol=json.loads((campaign/'protocol.json').read_text())
        assert protocol['seeds']==[46,47,48]
        for out in sorted(p for p in campaign.glob('seed*') if p.is_dir()):
            launch=json.loads((out/'launch.json').read_text())
            execution=json.loads((out/'execution.json').read_text())
            metrics=json.loads((out/'metrics.json').read_text())
            seed=launch['seed'];arm=launch['arm']+suffix
            assert execution['passed'] and execution['all_tti_action_bytes_exact'] and execution['causal_fresh_inputs_verified']
            assert launch['horizon']==5400 and launch['warmup']==2000 and launch['settling']==1000
            assert launch['measured']==2000 and launch['followup']==400 and metrics['censored_packets']==0
            assert launch['checkpoint_sha256']==protocol['checkpoint_sha256']
            assert launch['source_sha256']=={k:sha(out/'source'/k) for k in launch['source_sha256']}
            source={f:launch['source_sha256']['training/stageb_lns/'+f] for f in core}
            if reference_sources is None:
                reference_sources=source
            assert source==reference_sources
            runs[seed,arm]=(out,launch)
            inputs[str(out)]=dict(launch_sha256=sha(out/'launch.json'),execution_sha256=sha(out/'execution.json'),
                                 metrics_sha256=sha(out/'metrics.json'),prefix_sha256=sha(out/'prefix.npz'))
            decisions=pd.read_csv(out/'decisions.csv')
            assert len(decisions)==3400 and np.array_equal(decisions.tti,np.arange(2001,5401))
            measured=decisions[decisions.tti.between(3001,5000)]
            row=dict(metrics,arm=arm,decision_mean_ms=float(measured.decision_wall_ms.mean()),
                     decision_p95_ms=float(measured.decision_wall_ms.quantile(.95)))
            if 'candidate_evaluations' in measured:
                row['candidate_evaluations_mean']=float(measured.candidate_evaluations.mean())
            if 'joint_delta' in measured:
                assert (decisions.score >= decisions.baseline_score-1.e-7).all()
                assert (decisions.chosen_dequeue >= decisions.floor).all()
                row.update(joint_delta_mean=float(measured.joint_delta.mean()),
                    polish_delta_mean=float(measured.polish_delta.mean()),
                    joint_improved_pct=float(100*(measured.joint_delta>1).mean()))
            elif 'floor' in measured:
                assert (decisions.chosen_dequeue >= decisions.floor).all()
            rows.append(row)
            checks.append(dict(seed=seed,arm=arm,horizon=launch['horizon'],
                mature_cohort_packets=metrics['cohort_packets'],
                physical_samples=len(execution['physical_samples']),
                sinr_max_abs_db=max(s['sinr_max_abs_db'] for s in execution['physical_samples']),
                bler_max_abs=max(s['bler_max_abs'] for s in execution['physical_samples'])))
    pairing=[]
    for seed in (46,47,48):
        base,reference=runs[seed,'pfq0']
        with np.load(base/'prefix.npz') as data:
            initial={k:data[k].copy() for k in data.files}
        traffic_sha=sha(base/'traffic_offered_packets.csv')
        for (s,arm),(out,launch) in runs.items():
            if s != seed:
                continue
            assert launch['simulator_sha256']==reference['simulator_sha256']
            assert launch['libcumac_sha256']==reference['libcumac_sha256']
            assert traffic_sha==sha(out/'traffic_offered_packets.csv')
            with np.load(out/'prefix.npz') as data:
                for key in ('actions','goodput','input_hashes'):
                    np.testing.assert_array_equal(data[key][:2000],initial[key][:2000])
                for key in ('channel_fingerprints','precoder_fingerprints','fresh_input_tti'):
                    np.testing.assert_array_equal(data[key],initial[key])
            pairing.append(dict(seed=seed,arm=arm,warmup_exact=True,full_horizon_channel_precoder_exact=True,
                                offered_traffic_file_exact=True))
    frame=pd.DataFrame(rows).sort_values(['seed','arm'])
    frame.to_csv(output/'native_metrics.csv',index=False)
    columns=['goodput_mbps','full_payload_mbps','complete_success_pct','expiry_pct','ue_goodput_jain',
             'ue_goodput_p5_mbps','success_delay_mean_ms','success_delay_p95_ms',
             'candidate_evaluations_mean','decision_mean_ms','joint_delta_mean','joint_improved_pct','polish_delta_mean']
    summary=frame.groupby('arm')[columns].mean()
    for metric in ('goodput_mbps','full_payload_mbps'):
        summary[metric+'_std']=frame.groupby('arm')[metric].std()
        summary[metric+'_pct_vs_current']=100*(summary[metric]/summary.loc['legacy_current',metric]-1)
        summary[metric+'_gain_retained_vs_current']=(summary[metric]-summary.loc['pfq0',metric])/(summary.loc['legacy_current',metric]-summary.loc['pfq0',metric])
    summary.to_csv(output/'native_summary.csv')
    paired=[]
    for arm in sorted(frame.arm.unique()):
        for seed in (46,47,48):
            a=frame[(frame.arm==arm)&(frame.seed==seed)].iloc[0]
            b=frame[(frame.arm=='legacy_current')&(frame.seed==seed)].iloc[0]
            paired.append(dict(seed=seed,arm=arm,goodput_delta_mbps=float(a.goodput_mbps-b.goodput_mbps),
                full_payload_delta_mbps=float(a.full_payload_mbps-b.full_payload_mbps),
                goodput_relative_pct=float(100*(a.goodput_mbps/b.goodput_mbps-1)),
                full_payload_relative_pct=float(100*(a.full_payload_mbps/b.full_payload_mbps-1))))
    pd.DataFrame(paired).to_csv(output/'paired_differences.csv',index=False)
    write_json(output/'checks.json',dict(passed=True,runs=len(rows),native_ttis=sum(c['horizon'] for c in checks),
        physical_samples=sum(c['physical_samples'] for c in checks),
        sinr_max_abs_db=max(c['sinr_max_abs_db'] for c in checks),bler_max_abs=max(c['bler_max_abs'] for c in checks),
        matured_measurement_packets=sum(c['mature_cohort_packets'] for c in checks),
        checks=checks,pairing=pairing,core_source_sha256=reference_sources))
    write_json(output/'input_manifest.json',inputs)
    print(summary[['goodput_mbps','full_payload_mbps','complete_success_pct','expiry_pct','ue_goodput_jain']].to_string())


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--primary',required=True)
    parser.add_argument('--compact',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();summarize(args.primary,args.compact,args.output)

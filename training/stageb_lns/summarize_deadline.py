"""Audit matching native trajectories and apply predeclared research gates."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .closed_data import outcomes
from .experiment import write_json,sha


def main(args):
    campaign=Path(args.campaign).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((campaign/'protocol.json').read_text())
    assert json.loads((campaign/'completion.json').read_text())['passed']
    plan=json.loads(Path(args.plan).read_text());rows=[];runs={};audit=[];manifest={}
    for directory in sorted(p for p in campaign.glob('seed*') if p.is_dir()):
        launch=json.loads((directory/'launch.json').read_text());execution=json.loads((directory/'execution.json').read_text())
        metrics=json.loads((directory/'metrics.json').read_text());assert execution['passed'] and metrics['censored_packets']==0
        assert execution['all_tti_action_bytes_exact'] and execution['causal_fresh_inputs_verified']
        for name,digest in launch['source_sha256'].items():assert sha(directory/'source'/name)==digest
        expected=protocol['checkpoint_sha256'] if launch['arm'].startswith('risk_') else protocol['initial_sha256']
        assert launch['checkpoint_sha256']==expected
        decisions=pd.read_csv(directory/'decisions.csv');first=launch['measurement_first_tti'];last=launch['measurement_last_tti']
        assert len(decisions)==launch['horizon']-launch['warmup']
        measured=decisions[decisions.tti.between(first,last)]
        if 'floor' in decisions:assert (decisions.chosen_dequeue>=decisions.floor).all()
        if 'baseline_score' in decisions:assert (decisions.score>=decisions.baseline_score-1.e-7).all()
        if 'baseline_neural_score' in decisions:assert (decisions.score>=decisions.baseline_neural_score-1.e-7).all()
        if launch['arm'] in ('risk_residual','neural_extra'):assert launch['anchor_sha256']==protocol['initial_sha256']
        observed=outcomes(directory)
        row=dict(metrics,mean_native_reward=float(observed['reward'][first-1:last].mean()),
            weak_deficit_cost=float((np.maximum(1-observed['ema'][first-1:last]/2400.,0)**2).mean()),
            decision_mean_ms=float(measured.decision_wall_ms.mean()),
            candidates_mean=float(measured.candidate_evaluations.mean()))
        if 'joint_delta' in measured:
            row.update(joint_delta_mean=float(measured.joint_delta.mean()),polish_delta_mean=float(measured.polish_delta.mean()))
        rows.append(row);runs[launch['seed'],launch['arm']]=(directory,launch)
        audit.append(dict(seed=launch['seed'],arm=launch['arm'],ttis=launch['horizon'],
            samples=len(execution['physical_samples']),matured_packets=metrics['cohort_packets'],
            sinr_max=max(s['sinr_max_abs_db'] for s in execution['physical_samples']),
            bler_max=max(s['bler_max_abs'] for s in execution['physical_samples'])))
        manifest[str(directory)]={name:sha(directory/name) for name in ('launch.json','execution.json','metrics.json','prefix.npz')}
    paired=[]
    for seed in protocol['seeds']:
        reference,ref_launch=runs[seed,'greedy']
        with np.load(reference/'prefix.npz') as data:base={k:data[k].copy() for k in data.files}
        for arm in protocol['arms']:
            directory,launch=runs[seed,arm]
            assert launch['simulator_sha256']==ref_launch['simulator_sha256']
            assert launch['libcumac_sha256']==ref_launch['libcumac_sha256']
            assert sha(directory/'traffic_offered_packets.csv')==sha(reference/'traffic_offered_packets.csv')
            with np.load(directory/'prefix.npz') as data:
                for key in ('input_hashes','actions','goodput'):
                    np.testing.assert_array_equal(data[key][:launch['warmup']],base[key][:launch['warmup']])
                for key in ('channel_fingerprints','precoder_fingerprints','fresh_input_tti'):
                    np.testing.assert_array_equal(data[key],base[key])
            paired.append(dict(seed=seed,arm=arm,passed=True))
    frame=pd.DataFrame(rows).sort_values(['seed','arm']);frame.to_csv(out/'per_seed.csv',index=False)
    columns=['goodput_mbps','full_payload_mbps','expiry_pct','complete_success_pct','ue_goodput_p5_mbps',
        'ue_goodput_jain','success_delay_mean_ms','success_delay_p95_ms','mean_native_reward','weak_deficit_cost',
        'candidates_mean','decision_mean_ms']
    mean=frame.groupby('arm')[columns].mean();mean.to_csv(out/'means.csv')
    frame.groupby('arm')[columns].std().to_csv(out/'std.csv')
    gates={}
    comparisons=[('risk_neural','pure_neural_gate'),('risk_hybrid','guarded_gate')]
    if 'risk_residual' in mean.index:comparisons.append(('risk_residual','pure_neural_gate'))
    for arm,key in comparisons:
        gate=plan[key];a=mean.loc[arm];b=mean.loc[gate['reference']]
        details=dict(goodput_ratio=float(a.goodput_mbps/b.goodput_mbps),payload_ratio=float(a.full_payload_mbps/b.full_payload_mbps),
            expiry_increase_pp=float(a.expiry_pct-b.expiry_pct),p5_ratio=float(a.ue_goodput_p5_mbps/b.ue_goodput_p5_mbps))
        passed=(details['goodput_ratio']>=gate['mean_goodput_ratio_min'] and details['payload_ratio']>=gate['mean_full_payload_ratio_min']
            and details['expiry_increase_pp']<=gate['expiry_increase_pp_max'] and details['p5_ratio']>=gate['p5_ratio_min'])
        gates[arm]=dict(passed=bool(passed),reference=gate['reference'],thresholds=gate,observed=details)
    write_json(out/'gates.json',gates)
    write_json(out/'checks.json',dict(passed=True,runs=len(rows),native_ttis=sum(a['ttis'] for a in audit),
        physical_samples=sum(a['samples'] for a in audit),matured_packets=sum(a['matured_packets'] for a in audit),
        max_sinr_error_db=max(a['sinr_max'] for a in audit),max_bler_error=max(a['bler_max'] for a in audit),audit=audit,pairing=paired))
    write_json(out/'input_manifest.json',manifest)
    print(mean.to_string());print(json.dumps(gates,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--output',required=True)
    p.add_argument('--plan',required=True);main(p.parse_args())

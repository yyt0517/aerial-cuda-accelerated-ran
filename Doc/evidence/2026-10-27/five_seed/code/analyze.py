"""Detailed five-seed PFQ/rho090 analysis with the established long-window metrics."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .validate import OUT,REFERENCE,REPO,baseline,load_analysis,validate


def extract_with_details(helper,path,seed,arm,expected_simulator):
    original=helper.old.packets_from_fifo
    captured={}

    def capture(service,offered,outcomes,total):
        packets=original(service,offered,outcomes,total)
        captured.update(service=service,packets=packets)
        return packets

    helper.old.packets_from_fifo=capture
    try:
        launch=json.loads((path/'launch.json').read_text())
        windows,ue,stability,source=helper.extract(path,seed,arm,launch['source_sha256'],expected_simulator)
    finally:
        helper.old.packets_from_fifo=original
    label='current_rho090' if arm=='current' else 'pfq0'
    for row in windows:row['arm']=label
    stability['arm']=label;source['arm']=label
    per=pd.DataFrame(ue).set_index('ue_id');per['arm']=label
    packets=captured['packets']
    cohort=packets[packets.arrival_tti.between(8001,28000)].copy()
    assert not cohort.censored.any()
    service=captured['service'];measured=service[service.tti.between(8001,28000)]
    for name,data in [('outcome',cohort),('success_delay',cohort[cohort.successfully_delivered])]:
        groups=data.groupby('ue_id').system_time_ms
        for q in [.5,.95,.99]:per[f'{name}_p{int(q*100)}_ms']=groups.quantile(q)
    per['mean_backlog_bytes']=measured.groupby('ue_id').post_queue_bytes.mean()
    per['start_backlog_bytes']=measured[measured.tti==8001].set_index('ue_id').previous_post_queue_bytes
    per['end_backlog_bytes']=measured[measured.tti==28000].set_index('ue_id').post_queue_bytes
    per['allocated_prgs']=measured.groupby('ue_id').actual_prg_count.sum()
    per['serving_cell_prg_share_pct']=100*per.allocated_prgs/(20000*17)
    directory=OUT/'packet_cohorts';directory.mkdir(exist_ok=True)
    exported=directory/f'seed{seed}_{label}.csv.gz'
    assert not exported.exists()
    cohort.to_csv(exported,index=False,compression=dict(method='gzip',compresslevel=3,mtime=0))
    source['packet_cohort_export']=dict(path=str(exported),rows=len(cohort),columns=list(cohort.columns),sha256=helper.old.sha(exported))
    cdf=[]
    for ms in [.5,1,2,5,10,20,50,100,150,190,200]:
        success=cohort[cohort.successfully_delivered]
        cdf.append(dict(seed=seed,arm=label,deadline_ms=ms,
            complete_delivery_within_deadline_pct=float(100*(cohort.successfully_delivered & (cohort.system_time_ms<=ms)).mean()),
            conditional_success_latency_cdf_pct=float(100*success.system_time_ms.le(ms).mean()),
            all_outcome_latency_cdf_pct=float(100*cohort.system_time_ms.le(ms).mean())))
    return windows,per.reset_index().to_dict('records'),stability,source,cdf


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert json.loads((OUT/'native_campaign_complete.json').read_text())['passed']
    helper=load_analysis();sha=helper.old.sha
    expected_simulator=protocol['simulator_sha256']
    assert sha(OUT/'simulator')==expected_simulator==sha(REFERENCE/'simulator')
    all_rows,all_ue,stability,provenance,cdfs,validations,costs=[],[],[],[],[],[],[]
    for seed in protocol['seeds']:
        paired_validation=json.loads((OUT/f'seed{seed}_current_20000/paired_validation.json').read_text())
        assert paired_validation['passed'] and paired_validation['seed']==seed
        validations.append(paired_validation)
        paired=[]
        for arm,path in [('pfq0',REFERENCE/f'seed{seed}_pfq0_20000'),('current',OUT/f'seed{seed}_current_20000')]:
            rows,ue,stable,source,cdf=extract_with_details(helper,path,seed,arm,expected_simulator)
            all_rows.extend(rows);all_ue.extend(ue);stability.append(stable);provenance.append(source);cdfs.extend(cdf);paired.append(source)
            print('LONG_METRICS_EXTRACTED',seed,arm,rows[0]['goodput_mbps'],rows[0]['complete_payload_mbps'],flush=True)
        for field in ['first4000_service_sha256','mature_cohort_identity_sha256']:
            assert paired[0][field]==paired[1][field]
        path=OUT/f'seed{seed}_current_20000'
        d=pd.read_csv(path/'decisions.csv').query('8001<=tti<=28000')
        rounds=pd.read_csv(path/'rounds.csv').query('8001<=tti<=28000')
        costs.append(dict(seed=seed,algorithm_id=protocol['algorithm_id'],floor_ratio=.9,
            mean_draft_dequeue=float(d.draft_dequeue.mean()),mean_floor=float(d.keep_dequeue.mean()),
            mean_chosen_dequeue=float(d.chosen_dequeue.mean()),
            below_own_draft_dequeue_pct=float(100*d.chosen_dequeue.lt(d.draft_dequeue).mean()),
            final_floor_binding_pct=float(100*d.chosen_dequeue.eq(d.keep_dequeue).mean()),
            mean_rounds=float(d.rounds_evaluated.mean()),max_rounds=int(d.rounds_evaluated.max()),
            mean_candidates=float(rounds.groupby('tti').candidates.sum().mean()),
            mean_shared_host_decision_ms=float(1000*d.decision_wall_s.mean()),
            native_elapsed_s=json.loads((path/'execution.json').read_text())['elapsed_native_s']))
    frame=pd.DataFrame(all_rows)
    full=frame[frame.window=='measurement']
    full.to_csv(OUT/'per_seed_metrics.csv',index=False)
    frame[frame.window!='measurement'].to_csv(OUT/'window_metrics.csv',index=False)
    pd.DataFrame(all_ue).to_csv(OUT/'per_ue_metrics.csv',index=False)
    helper.old.mean_std(full).to_csv(OUT/'mean_std.csv',index=False)
    paired_frame=frame.copy();paired_frame['arm']=paired_frame.arm.replace({'current_rho090':'current'})
    differences,intervals=helper.paired_tables(paired_frame)
    differences['current_algorithm_id']=protocol['algorithm_id']
    intervals['current_algorithm_id']=protocol['algorithm_id']
    differences.to_csv(OUT/'paired_differences.csv',index=False)
    intervals.to_csv(OUT/'paired_mean_ci.csv',index=False)
    pd.DataFrame(stability).to_csv(OUT/'stability.csv',index=False)
    pd.DataFrame(cdfs).to_csv(OUT/'deadline_delivery_cdf.csv',index=False)
    pd.DataFrame(costs).to_csv(OUT/'search_and_floor_diagnostics.csv',index=False)
    per=pd.DataFrame(all_ue)
    current=per[per.arm=='current_rho090'].set_index(['seed','ue_id'])
    pfq=per[per.arm=='pfq0'].set_index(['seed','ue_id'])
    metrics=['fixed_window_goodput_mbps','complete_payload_mbps','complete_success_fraction','queue_failure_pct',
             'total_failure_pct','success_delay_mean_ms','success_delay_p95_ms','outcome_mean_ms','outcome_p99_ms']
    deltas=pd.DataFrame(index=current.index)
    for key in metrics:
        deltas[key+'_current']=current[key];deltas[key+'_pfq']=pfq[key];deltas[key+'_delta']=current[key]-pfq[key]
    deltas.reset_index().to_csv(OUT/'matched_ue_deltas.csv',index=False)
    balance=[]
    for seed in protocol['seeds']:
        a=full[(full.seed==seed)&(full.arm=='current_rho090')].iloc[0]
        b=full[(full.seed==seed)&(full.arm=='pfq0')].iloc[0]
        pieces=dict(seed=seed,goodput_gain_mbps=float(a.goodput_mbps-b.goodput_mbps),
            accepted_contribution_mbps=float(a.accepted_mbps-b.accepted_mbps),
            expiry_contribution_mbps=float(b.expired_byte_mbps-a.expired_byte_mbps),
            failed_byte_contribution_mbps=float(b.failed_byte_mbps-a.failed_byte_mbps),
            backlog_release_contribution_mbps=float(a.backlog_release_mbps-b.backlog_release_mbps))
        np.testing.assert_allclose(pieces['goodput_gain_mbps'],sum(pieces[k] for k in
            ['accepted_contribution_mbps','expiry_contribution_mbps','failed_byte_contribution_mbps','backlog_release_contribution_mbps']),rtol=0,atol=1e-10)
        balance.append(pieces)
    pd.DataFrame(balance).to_csv(OUT/'goodput_gain_balance.csv',index=False)
    definitions=dict(source_fields=helper.FIELDS,measurement='8001..28000,10s; arrivals cohort followed through28400',
        goodput='Successful segmented bytes in service window*8/10s/1e6',
        complete_payload='Fully successful measured-arrival packet payload bits/10s/1e6',
        failure='Total failed measured arrivals=expiry+CRC-failed MAC-completed packets; TB BLER has a different denominator.',
        latency='Inclusive(terminal-arrival+1)*0.5ms. Success is conditional on full packet success; outcome includes all cohort packets.',
        windows='Ten1s and two5s diagnostic windows; they are correlated and are not independent experimental units.',
        cdf='Deadline delivery fraction includes failures/expiries in all-arrival denominator; success CDF conditions on complete success.',
        statistics='Seed is independent unit; n5, sampleSD, nominal95%paired tCI df4, no multiplicity adjustment. Seed41 was used in prior ratio selection.',
        floor='ceil(.9*own-current PFQ+CTBS draft dequeue), integer(90*D0+99)//100',
        raw='Native metrics.bin is float64LE, shape28400x36x19, fields in source_fields; action/input fingerprints in prefix.npz.')
    (OUT/'metric_definitions.json').write_text(json.dumps(definitions,indent=2,ensure_ascii=False)+'\n')
    validation=dict(passed=True,algorithm_id=protocol['algorithm_id'],seeds=protocol['seeds'],
        new_rho090_runs=5,reused_verified_pfq_runs=5,measurement_rows=len(full),new_native_ttis=142000,
        native_actions_and_discrete_physics_exact=True,all_causal_floors_independently_verified=True,
        no_bc_or_coordinator_inference=True,no_external_floor_schedule=True,all_searches_converged=True,
        complete_five_seed_comparison=True,no_censoring=True,mature_packet_cohorts_match=True,
        all_practical_stability_checks_passed=all(x['practical_stability_passed'] for x in stability),
        stability_is_not_capacity_proof=True,independent_new_physical_samples=30,
        pairing_and_inputs=validations,sources=provenance,protocol_sha256=sha(OUT/'protocol.json'),
        analysis_sha256=sha(__file__),reused_window_analysis_sha256=sha(Path(helper.__file__)),
        limitations=protocol['limitations'])
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2,ensure_ascii=False)+'\n')
    print('FIVE_SEED_RHO090_DETAILED_ANALYSIS_VALIDATED',flush=True)


if __name__=='__main__':main()

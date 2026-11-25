"""Matched packet-outcome and UE-macro review of the restored directed version."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / 'boundary08_directed_attention_20261008_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def per_ue_metrics(packets):
    rows = []
    for ue, p in packets.groupby('ue_id'):
        completed = p[p.outcome == 'mac_completed']
        success = p[p.successfully_delivered]
        assert len(completed) and len(success)
        rows.append(dict(
            ue_id=int(ue), packets=len(p), mac_completed=len(completed), complete_success=len(success),
            outcome_mean_ms=float(p.system_time_ms.mean()),
            mac_completion_mean_ms=float(completed.system_time_ms.mean()),
            success_delay_mean_ms=float(success.system_time_ms.mean()),
            mac_packet_rate_mbps=float(completed.packet_size_bytes.sum() * 8 / (completed.system_time_ms.sum() * 1000)),
            success_packet_rate_mbps=float(success.packet_size_bytes.sum() * 8 / (success.system_time_ms.sum() * 1000)),
            success_byte_outcome_rate_mbps=float(p.successful_bytes.sum() * 8 / (p.system_time_ms.sum() * 1000)),
            complete_success_fraction=float(p.successfully_delivered.mean()),
            expiry_fraction=float((p.outcome == 'expired').mean()),
        ))
    result = pd.DataFrame(rows).set_index('ue_id')
    assert np.array_equal(result.index, np.arange(36))
    return result


def main():
    OUT.mkdir(exist_ok=True)
    frozen = json.loads((SOURCE / 'final_verification.json').read_text())
    for name, digest in frozen['artifacts_sha256'].items():
        assert sha(SOURCE / name) == digest, name
    selection = json.loads((OUT.parent / 'boundary08_goodput_restore_20261009_v1/selection.json').read_text())
    assert sha(selection['checkpoint']) == selection['checkpoint_sha256']
    old_metrics = pd.read_csv(SOURCE / 'comparison.csv').set_index('arm')
    rows, ue_rows, inputs = [], [], {}
    identity = None
    for arm in ['bc', 'pfq', 'no_directed', 'full']:
        path = SOURCE / arm
        packets = pd.read_csv(path / 'packet_lifecycle.csv')
        cohort = packets[packets.arrival_tti.between(4001, 24000)].sort_values(['ue_id', 'packet_id']).reset_index(drop=True)
        assert len(cohort) == 575687 and not cohort.censored.any()
        assert not cohort.duplicated(['ue_id', 'packet_id']).any()
        ids = cohort[['ue_id', 'packet_id', 'arrival_tti', 'packet_size_bytes', 'ttl_tti']]
        if identity is None:
            identity = ids
        else:
            pd.testing.assert_frame_equal(ids, identity, check_exact=True)
        saved_cohort = pd.read_csv(path / 'measured_arrival_cohort.csv').sort_values(['ue_id', 'packet_id']).reset_index(drop=True)
        pd.testing.assert_frame_equal(cohort, saved_cohort, check_exact=True)
        assert np.array_equal(cohort.packet_size_bytes, cohort.successful_bytes + cohort.failed_dequeue_bytes + cohort.remaining_bytes)
        assert np.array_equal(cohort.successfully_delivered, (cohort.outcome == 'mac_completed') & (cohort.failed_dequeue_bytes == 0))
        # Check outcome times and queue-terminal semantics directly against native records.
        terminal = pd.read_csv(path / 'packet_outcome_samples.csv')
        joined = cohort.merge(terminal, on=['ue_id', 'packet_id', 'arrival_tti', 'packet_size_bytes'],
                              suffixes=('', '_native'), validate='one_to_one')
        assert len(joined) == len(cohort)
        assert np.array_equal(joined.terminal_tti, joined.terminal_tti_native)
        assert np.array_equal(joined.system_time_ms, joined.system_time_ms_native)
        assert np.array_equal(joined.remaining_bytes, joined.remaining_bytes_at_terminal)
        assert np.array_equal(joined.outcome.replace({'mac_completed': 'delivered'}), joined.outcome_native)
        expected_time = np.where(cohort.outcome == 'expired', cohort.ttl_tti * .5,
                                 (cohort.terminal_tti - cohort.arrival_tti + 1) * .5)
        assert np.array_equal(cohort.system_time_ms, expected_time)
        assert cohort.system_time_ms.between(.5, 200).all()
        per = per_ue_metrics(cohort)
        terminal_window = packets[packets.terminal_tti.between(4001, 24000)]
        terminal_per = per_ue_metrics(terminal_window)
        service = pd.read_csv(path / 'service_balance_trace.csv', usecols=[
            'tti', 'ue_id', 'serving_cell', 'successful_bytes', 'failed_bytes', 'scheduled_dequeue_bytes',
            'expired_bytes', 'actual_prg_count', 'post_queue_bytes'])
        s = service[service.tti.between(4001, 24000)]
        assert len(s) == 720000 and not s.duplicated(['tti', 'ue_id']).any()
        assert np.array_equal(s.successful_bytes + s.failed_bytes, s.scheduled_dequeue_bytes)
        rates = s.groupby('ue_id').successful_bytes.sum() * 8 / 10 / 1.e6
        per['fixed_window_goodput_mbps'] = rates
        per['cell_id'] = s.groupby('ue_id').serving_cell.first()
        per['arm'] = arm
        ue_rows.append(per.reset_index())
        completed = cohort[cohort.outcome == 'mac_completed']
        success = cohort[cohort.successfully_delivered]
        row = dict(
            arm=arm, cohort_packets=len(cohort), complete_success=len(success), expired_packets=int((cohort.outcome == 'expired').sum()),
            complete_success_pct=float(100 * cohort.successfully_delivered.mean()),
            expiry_pct=float(100 * (cohort.outcome == 'expired').mean()),
            goodput_mbps=float(rates.sum()), ue_mean_goodput_mbps=float(rates.mean()),
            ue_goodput_p5_mbps=float(rates.quantile(.05)), ue_goodput_p10_mbps=float(rates.quantile(.1)),
            ue_goodput_jain=float(rates.sum() ** 2 / (36 * (rates ** 2).sum())),
            outcome_mean_ms=float(cohort.system_time_ms.mean()), outcome_p50_ms=float(cohort.system_time_ms.quantile(.5)),
            outcome_p95_ms=float(cohort.system_time_ms.quantile(.95)), outcome_p99_ms=float(cohort.system_time_ms.quantile(.99)),
            ue_macro_outcome_mean_ms=float(per.outcome_mean_ms.mean()),
            success_delay_mean_ms=float(success.system_time_ms.mean()), success_delay_p95_ms=float(success.system_time_ms.quantile(.95)),
            pooled_mac_packet_rate_mbps=float(completed.packet_size_bytes.sum() * 8 / (completed.system_time_ms.sum() * 1000)),
            ue_macro_mac_packet_rate_mbps=float(per.mac_packet_rate_mbps.mean()),
            ue_macro_success_packet_rate_mbps=float(per.success_packet_rate_mbps.mean()),
            ue_macro_success_byte_outcome_rate_mbps=float(per.success_byte_outcome_rate_mbps.mean()),
            legacy_terminal_packets=len(terminal_window), legacy_terminal_outcome_mean_ms=float(terminal_window.system_time_ms.mean()),
            legacy_terminal_ue_macro_mac_packet_rate_mbps=float(terminal_per.mac_packet_rate_mbps.mean()),
            success_within_1ms_pct=float(100 * (cohort.successfully_delivered & (cohort.system_time_ms <= 1)).mean()),
            success_within_5ms_pct=float(100 * (cohort.successfully_delivered & (cohort.system_time_ms <= 5)).mean()),
            prg_utilization=float(s.actual_prg_count.sum() / (20000 * 51)),
            success_bytes_per_prg=float(s.successful_bytes.sum() / s.actual_prg_count.sum()),
            end_backlog_bytes=int(s[s.tti == 24000].post_queue_bytes.sum()),
        )
        for new, old in [('goodput_mbps', 'goodput_mbps'), ('outcome_mean_ms', 'cohort_outcome_delay_mean_ms'),
                         ('success_delay_mean_ms', 'cohort_success_delay_mean_ms'),
                         ('legacy_terminal_outcome_mean_ms', 'mac_completion_or_expiry_outcome_delay_ms_mean')]:
            assert abs(row[new] - old_metrics.loc[arm, old]) < 1.e-9, (arm, new)
        assert abs(row['ue_mean_goodput_mbps'] * 36 - row['goodput_mbps']) < 1.e-9
        rows.append(row)
        inputs[arm] = {name: sha(path / name) for name in [
            'packet_lifecycle.csv', 'measured_arrival_cohort.csv', 'packet_outcome_samples.csv', 'service_balance_trace.csv']}
    metrics = pd.DataFrame(rows).set_index('arm')
    ues = pd.concat(ue_rows, ignore_index=True)
    comparisons = {}
    for baseline in ['pfq', 'bc', 'no_directed']:
        delta = metrics.loc['full'] - metrics.loc[baseline]
        relative = 100 * delta / metrics.loc[baseline]
        a = ues[ues.arm == 'full'].set_index('ue_id')
        b = ues[ues.arm == baseline].set_index('ue_id')
        changes = pd.DataFrame(dict(
            fixed_window_goodput_delta_mbps=a.fixed_window_goodput_mbps - b.fixed_window_goodput_mbps,
            mac_packet_rate_delta_mbps=a.mac_packet_rate_mbps - b.mac_packet_rate_mbps,
            outcome_mean_delta_ms=a.outcome_mean_ms - b.outcome_mean_ms,
            success_byte_outcome_rate_delta_mbps=a.success_byte_outcome_rate_mbps - b.success_byte_outcome_rate_mbps,
        ))
        changes.to_csv(OUT / f'per_ue_full_minus_{baseline}.csv')
        comparisons[baseline] = dict(delta=delta.to_dict(), relative_pct=relative.to_dict(),
            ue_goodput_improved=int((changes.fixed_window_goodput_delta_mbps > 0).sum()),
            ue_mac_packet_rate_improved=int((changes.mac_packet_rate_delta_mbps > 0).sum()),
            ue_outcome_mean_improved=int((changes.outcome_mean_delta_ms < 0).sum()),
            largest_macro_rate_losses=changes.nsmallest(5, 'mac_packet_rate_delta_mbps').reset_index().to_dict('records'))
    metrics.to_csv(OUT / 'comparison.csv')
    ues.to_csv(OUT / 'per_ue.csv', index=False)
    result = dict(
        source_version=SOURCE.name, checkpoint_sha256=selection['checkpoint_sha256'],
        validation=dict(frozen_original_artifacts_match=True, common_cohort_packets=575687, ue_count=36,
                        no_censoring=True, native_terminal_times_exact=True, byte_accounting_exact=True,
                        original_metrics_reproduced=True),
        definitions=dict(
            outcome_delay='Time from arrival until MAC dequeue completion or expiry, including PHY-failed packets; all common-cohort packets remain in the denominator.',
            ue_mean_goodput='Successful bytes in the common 10-second observation window per UE, then equal mean over all36UE; exactly network goodput/36.',
            ue_macro_mac_packet_rate='Mean over36UE of sum(completed packet bits)/sum(completed packet system-time seconds), in Mbps. MAC completion includes PHY-failed bytes; this is the historical UE-macro packet-effective-service rate formula applied to a common arrival cohort.',
            ue_macro_success_packet_rate='Same per-UE ratio, restricted to fully successfully delivered packets; conditioned on a policy-dependent success population.',
            ue_macro_success_byte_outcome_rate='Mean over36UE of cohort successful bits/sum(all cohort outcome times); includes partial successes and retains failures/expiries in the time denominator. Not wall-clock throughput.',
            legacy_terminal='Packets terminating at TTI4001..24000; denominator population differs across policies. Reported separately for historical comparability.'),
        full_vs=comparisons, input_sha256=inputs,
        scope='Saved seed43 four-arm trajectories only; no new model, thresholds, native simulation or training. Descriptive findings, not independent-repeat significance.')
    (OUT / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(metrics[['goodput_mbps', 'outcome_mean_ms', 'outcome_p50_ms', 'outcome_p95_ms',
                   'ue_mean_goodput_mbps', 'ue_macro_mac_packet_rate_mbps', 'ue_macro_success_byte_outcome_rate_mbps']].to_string(), flush=True)
    print(json.dumps({b: {k: v for k, v in value.items() if k not in ['delta', 'relative_pct']} for b, value in comparisons.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()

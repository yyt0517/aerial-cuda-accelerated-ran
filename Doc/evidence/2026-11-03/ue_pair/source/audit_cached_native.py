"""Verify every native action, packet outcome and delay against a frozen run."""
import argparse
import json
from pathlib import Path
import numpy as np

from .experiment import sha, write_json


def audit(reference, cached, output):
    import pandas as pd
    from .native_repair import native_modules
    native, _ = native_modules()
    reference, cached = Path(reference), Path(cached)
    launches = [json.loads((p/'launch.json').read_text()) for p in (reference, cached)]
    for key in ('seed', 'arm', 'horizon', 'warmup', 'settling', 'measured', 'followup',
                'width', 'checkpoint_sha256', 'simulator_sha256', 'libcumac_sha256'):
        assert launches[0][key] == launches[1][key], key
    assert launches[0].get('decoder', 'reference') == 'reference'
    assert launches[1]['decoder'] == 'cached'
    executions = [json.loads((directory/'execution.json').read_text()) for directory in (reference, cached)]
    assert all(value['passed'] for value in executions)
    with np.load(reference/'prefix.npz') as left, np.load(cached/'prefix.npz') as right:
        assert set(left.files) == set(right.files)
        for key in left.files:
            np.testing.assert_array_equal(left[key], right[key], err_msg=key)
    # Runtime diagnostics, if added to the native ABI, require explicit exclusion.
    shape = launches[0]['horizon'], 36, len(native.FIELDS)
    left = np.fromfile(reference/'metrics.bin', dtype='<f8').reshape(shape)
    right = np.fromfile(cached/'metrics.bin', dtype='<f8').reshape(shape)
    np.testing.assert_array_equal(left, right)
    packets = [pd.read_csv(p/'packets.csv') for p in (reference, cached)]
    pd.testing.assert_frame_equal(*packets, check_exact=True)
    metrics = [json.loads((p/'metrics.json').read_text()) for p in (reference, cached)]
    assert metrics[0] == metrics[1]
    assert metrics[1]['censored_packets'] == 0
    decisions = [pd.read_csv(p/'decisions.csv') for p in (reference, cached)]
    from .benchmark_cached_repair import SEARCH_FIELDS
    for key in (*SEARCH_FIELDS, 'tti', 'ue_evaluations', 'mmse_requests', 'mmse_misses'):
        np.testing.assert_array_equal(decisions[0][key].values, decisions[1][key].values, err_msg=key)
    # Early frozen references predate per-decision neural counters. Preserve
    # that evidence boundary rather than fabricating missing measurements.
    per_decision_counters, unavailable_per_decision = [], []
    total_counters, unavailable_totals = [], []
    for key in ('neural_calls', 'neural_proposals', 'neural_option_scores', 'neural_logit_calls'):
        if all(key in frame for frame in decisions):
            np.testing.assert_array_equal(decisions[0][key].values, decisions[1][key].values, err_msg=key)
            per_decision_counters.append(key)
        else:
            unavailable_per_decision.append(key)
        if all(key in value for value in executions):
            assert executions[0][key] == executions[1][key], key
            total_counters.append(key)
        else:
            unavailable_totals.append(key)
    assert 'neural_calls' in total_counters and 'neural_proposals' in total_counters
    filenames = ('launch.json', 'execution.json', 'prefix.npz', 'metrics.bin', 'metrics.json',
                 'decisions.csv', 'packets.csv', 'traffic_offered_packets.csv')
    result = dict(passed=True, seed=launches[0]['seed'], horizon=launches[0]['horizon'],
        reference=str(reference), cached=str(cached),
        all_native_inputs_actions_bytes_and_packets_exact=True,
        all_recorded_search_work_exact=True, metrics=metrics[1],
        exact_per_decision_network_counters=per_decision_counters,
        exact_total_network_counters=total_counters,
        unavailable_per_decision_network_counters=unavailable_per_decision,
        unavailable_total_network_counters=unavailable_totals,
        success_delay_mean_delta_ms=metrics[1]['success_delay_mean_ms']-metrics[0]['success_delay_mean_ms'],
        outcome_mean_delta_ms=metrics[1]['outcome_mean_ms']-metrics[0]['outcome_mean_ms'],
        evidence_sha256={str(p/name):sha(p/name) for p in (reference, cached) for name in filenames},
        interpretation='Known validation seed, same traffic/model/search; equivalence, not new quality evidence')
    write_json(Path(output), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True)
    parser.add_argument('--cached', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit(args.reference, args.cached, args.output)

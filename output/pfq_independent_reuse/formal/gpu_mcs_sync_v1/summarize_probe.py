"""Fail-closed paired evidence checks; no training or simulation."""
import csv
import hashlib
import json
from itertools import zip_longest
from pathlib import Path

root = Path(__file__).resolve().parent
reports = {p: json.loads((root/p/'report.json').read_text()) for p in ('before','after')}
result = dict(status='complete', reports=reports, checks={})
for policy in ('pf','rr'):
    a,b = [reports[p]['arms'][policy] for p in ('before','after')]
    assert a['replay']['input_sha256'] == b['replay']['input_sha256'], 'kernel inputs differ'
    assert b['replay']['count_mismatches'] == b['replay']['mcs_mismatches'] == b['replay']['variable_slots'] == 0
    assert a['service']['metrics'] == b['service']['metrics'], 'CPU metrics changed'
    for key in ('offered_identity','topology_sha256','channel_fingerprint_sha256'):
        assert a['service'][key] == b['service'][key], key
    changed = 0
    count = 0
    with (root/'before'/policy/'service_balance_trace.csv').open() as fa, (root/'after'/policy/'service_balance_trace.csv').open() as fb:
        for ra,rb in zip_longest(csv.DictReader(fa),csv.DictReader(fb)):
            assert ra is not None and rb is not None, 'trace lengths differ'
            for key in ra:
                if key not in ('phy_sinr_db_gpu','predicted_bler_gpu'):
                    assert ra[key] == rb[key], (policy,count,key,ra[key],rb[key])
            changed += ra['predicted_bler_gpu'] != rb['predicted_bler_gpu']
            count += 1
    assert count == 5000 * 36
    result['checks'][policy] = dict(kernel_inputs_identical=True, cpu_metrics_identical=True,
        cpu_service_trace_identical_rows=count, changed_gpu_predicted_bler_rows=changed,
        unit_checked_rows=a['service']['unit_checked_rows']+b['service']['unit_checked_rows'])
result['limits'] = [
    'Fixed replay uses the first real WB-SINR TTI; prior TB errors are -1, so nonzero OLLA update history is not covered.',
    'CQI barrier restored symmetrically but CQI path was not exercised.',
    'GPU improvement means count/MCS correctness and replay stability, not a measured GPU-only goodput gain.',
    'Same one seed, DL3cell36UE PF/RR; no training resumed or new DPA performance claim.',
    'Instrumentation is identical in both versions and default-off; observed race frequency is not a population estimate.']
(root/'final_report.json').write_text(json.dumps(result,indent=2)+'\n')
(root/'status.json').write_text(json.dumps(dict(status='complete',
    question='GPU fixed-input count/MCS synchronization and CPU service isolation',
    training_resumed=False, checks=result['checks']),indent=2)+'\n')
print(json.dumps(result['checks'],indent=2))

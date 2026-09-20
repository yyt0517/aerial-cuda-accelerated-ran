"""Aggregate the completed, scoped slot-index experiment; no simulation/training."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parent
reports = {p: json.loads((root / p / 'report.json').read_text())
           for p in ('before', 'after', 'controlled_before', 'controlled_after')}
result = {'scope': 'seed61 DL Type0 PF/RR PHY slot indexing; CPU-service long comparison',
          'reports': reports, 'paired_identity': {}, 'mcs': {}}
for policy in ('pf', 'rr'):
    a, b = (reports[p]['arms'][policy] for p in ('before', 'after'))
    result['paired_identity'][policy] = {}
    for key in ('offered_identity', 'topology_sha256', 'channel_fingerprint_sha256'):
        assert a[key] == b[key], (policy, key)
        result['paired_identity'][policy][key] = a[key]
    for phase in ('before', 'after'):
        hist = Counter()
        with (root / phase / f'{policy}_5000/service_balance_trace.csv').open() as stream:
            for row in csv.DictReader(stream):
                if int(row['tti']) > 1000 and int(row['tb_tx_count']) > 0:
                    hist[int(row['mcs'])] += int(row['tb_tx_count'])
        count = sum(hist.values())
        result['mcs'][f'{phase}/{policy}'] = dict(tb_count=count, histogram=dict(hist),
            mcs0_fraction=hist[0]/count, mean_mcs=sum(k*v for k,v in hist.items())/count)
result['limitations'] = [
    'Controlled single-TTI test forces identical CPU-selected MCS/layers on both PHY paths; '
    'it does not validate native GPU link adaptation.',
    'Initial before/after invariance records verify CPU canonical inputs only; '
    'use controlled_before/controlled_after for the complete CPU+GPU physical-input claim.',
    'Native GPU MCS shared-counter race suspected and not fixed in this experiment.',
    'UL index fixes are symmetric but not runtime-tested; one DL seed is not generalization.',
    'Scheduled raw TBS sum is not sustainable achievable network capacity.',
    'All runs already use corrected TBS bits-to-bytes; no old BC data or training resumed.']
lib = root.parents[3] / 'build.phy_slot_v3/cuMAC/src/libcumac.so'
if lib.exists():
    result['common_libcumac_sha256'] = hashlib.sha256(lib.read_bytes()).hexdigest()
(root / 'final_report.json').write_text(json.dumps(result, indent=2) + '\n')

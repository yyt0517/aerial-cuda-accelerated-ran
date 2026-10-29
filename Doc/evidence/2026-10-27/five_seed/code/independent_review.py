"""Independent arithmetic/archive checks against the completed exported evidence."""
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from .run import OUT,REPO


def sha_stream(stream):
    digest=hashlib.sha256();size=0
    for data in iter(lambda:stream.read(4<<20),b''):
        digest.update(data);size+=len(data)
    return digest.hexdigest(),size


def main():
    root=REPO/'deliverables/boundary08_rho090_five_seed_20261027'
    full=pd.read_csv(root/'per_seed_metrics.csv')
    intervals=pd.read_csv(root/'paired_mean_ci.csv').query('window == "measurement"').set_index('metric')
    means=pd.read_csv(root/'mean_std.csv')
    assert sorted(full.seed.unique())==[41,42,43,44,45] and len(full)==10
    assert full.groupby(['seed','arm']).size().eq(1).all()
    historical=REPO/'deliverables/boundary08_steady_state_20261021/per_seed_metrics.csv'
    saved=pd.read_csv(historical).query('arm == "pfq0"').sort_values('seed')
    recomputed=full.query('arm == "pfq0"').sort_values('seed')
    common=saved.select_dtypes(include=[np.number]).columns.intersection(recomputed.columns)
    np.testing.assert_allclose(saved[common],recomputed[common],rtol=1e-12,atol=1e-8,equal_nan=True)
    per_ue=pd.read_csv(root/'per_ue_metrics.csv')
    assert len(per_ue)==360 and per_ue.groupby(['seed','arm']).size().eq(36).all()
    grouped=per_ue.groupby(['seed','arm'])[['fixed_window_goodput_mbps','complete_payload_mbps']].sum()
    for _,row in full.iterrows():
        np.testing.assert_allclose(grouped.loc[(int(row.seed),row.arm)].to_numpy(),
            [row.goodput_mbps,row.complete_payload_mbps],rtol=0,atol=1e-8)
    cdf=pd.read_csv(root/'deadline_delivery_cdf.csv').query('deadline_ms == 200').set_index(['seed','arm'])
    for _,row in full.iterrows():
        entry=cdf.loc[(int(row.seed),row.arm)]
        np.testing.assert_allclose([entry.complete_delivery_within_deadline_pct,entry.conditional_success_latency_cdf_pct],
            [row.complete_success_pct,100.0],rtol=0,atol=1e-8)
    primary=['goodput_mbps','complete_payload_mbps','complete_success_pct','expiry_pct','total_failure_pct',
             'success_delay_mean_ms','success_delay_p95_ms','ue_goodput_p5_mbps']
    for metric in primary:
        baseline=full[full.arm=='pfq0'].sort_values('seed')[metric].tolist()
        current=full[full.arm=='current_rho090'].sort_values('seed')[metric].tolist()
        differences=[a-b for a,b in zip(current,baseline)]
        average=statistics.mean(differences)
        deviation=statistics.stdev(differences)
        # Fixed two-sided t(0.975,4) critical value; no reuse of the report helper.
        width=2.7764451051977987*deviation/math.sqrt(5)
        reported=intervals.loc[metric]
        np.testing.assert_allclose([average,average-width,average+width],
            [reported.mean_delta,reported.ci95_low,reported.ci95_high],rtol=0,atol=1e-8)
        for arm,data in [('pfq0',baseline),('current_rho090',current)]:
            row=means[(means.arm==arm)&(means.metric==metric)].iloc[0]
            np.testing.assert_allclose([statistics.mean(data),statistics.stdev(data)],
                [row['mean'],row.sample_std],rtol=0,atol=1e-8)
    packet_rows=0
    for _,row in full.iterrows():
        path=root/'packet_cohorts'/f'seed{int(row.seed)}_{row.arm}.csv.gz'
        packets=pd.read_csv(path,dtype={'successfully_delivered':bool,'censored':bool})
        assert len(packets)==int(row.cohort_packets)
        assert not packets.censored.any() and packets.arrival_tti.between(8001,28000).all()
        assert not packets.duplicated(['ue_id','packet_id']).any()
        delivered=packets[packets.successfully_delivered]
        payload=float(delivered.packet_size_bytes.sum())*8/10/1e6
        success=100*len(delivered)/len(packets)
        expiry=100*packets.outcome.eq('expired').mean()
        np.testing.assert_allclose([payload,success,expiry],
            [row.complete_payload_mbps,row.complete_success_pct,row.expiry_pct],rtol=0,atol=1e-8)
        packet_rows+=len(packets)
    for entry in json.loads((root/'raw_index.json').read_text()):
        with gzip.open(root/entry['compressed_metrics'],'rb') as stream:
            digest,size=sha_stream(stream)
        assert digest==entry['uncompressed_metrics_sha256']
        assert size==28400*36*19*8
        for table in entry['compressed_tables']:
            with gzip.open(root/table['file'],'rb') as stream:digest,_=sha_stream(stream)
            assert digest==table['uncompressed_sha256']
    for entry in json.loads((root/'manifest.json').read_text())['files']:
        with (root/entry['file']).open('rb') as stream:digest,size=sha_stream(stream)
        assert digest==entry['sha256'] and size==entry['bytes']
    evidence=REPO/'Doc/evidence/2026-10-27'
    for entry in json.loads((evidence/'manifest.json').read_text())['files']:
        with (evidence/entry['file']).open('rb') as stream:digest,size=sha_stream(stream)
        assert digest==entry['sha256'] and size==entry['bytes']
    result=dict(passed=True,primary_metrics_recomputed=len(primary),packet_cohort_rows_checked=packet_rows,
        recomputed_pfq_metrics_match_original_five_seed_report=True,per_ue_totals_and_deadline_cdf_consistent=True,
        original_pfq_summary_sha256=hashlib.sha256(historical.read_bytes()).hexdigest(),
        raw_archives_decompressed_and_verified=10,all_delivery_and_project_evidence_hashes_valid=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'independent_review.json').write_text(json.dumps(result,indent=2)+'\n')
    manifest=json.loads((root/'manifest.json').read_text())
    path=root/'independent_review.json'
    manifest['files']=[entry for entry in manifest['files'] if entry['file']!=path.name]
    manifest['files'].append(dict(file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    target=evidence/'five_seed/independent_review.json'
    target.write_bytes(path.read_bytes())
    project_manifest=json.loads((evidence/'manifest.json').read_text())
    project_manifest['files']=[entry for entry in project_manifest['files'] if entry['file']!=str(target.relative_to(evidence))]
    project_manifest['files'].append(dict(file=str(target.relative_to(evidence)),source=str(path),bytes=target.stat().st_size,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    (evidence/'manifest.json').write_text(json.dumps(project_manifest,indent=2)+'\n')
    print('INDEPENDENT_FIVE_SEED_REVIEW_PASSED',result,flush=True)


if __name__=='__main__':main()

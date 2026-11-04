"""Freeze the protocol before inspecting ANY new component outcomes."""
import json
from common import ARMS, OUT, PROMOTED, REFERENCE, SEEDS, sha, write
from run import runner_source


def main():
    assert not (OUT / 'protocol.json').exists()
    (OUT / 'native_run_generated.py').write_text(runner_source())
    reference = {}
    for arm, root, label in [('pfq', REFERENCE, 'pfq0'), ('full', PROMOTED, 'current')]:
        for seed in SEEDS:
            path = root / f'seed{seed}_{label}_20000'
            assert json.loads((path / 'validation.json').read_text())['passed']
            reference[f'{arm}/{seed}'] = dict(path=str(path), files={name: sha(path / name) for name in
                ['launch.json', 'execution.json', 'validation.json', 'prefix.npz', 'metrics.bin',
                 'traffic_offered_packets.csv', 'packet_outcome_samples.csv', 'ue_topology.csv']})
    protocol = dict(date='2026-10-29', objective='Attribute the established boundary DPLS gains to CTBS, global physical search, and deficit weighting; separately compare independent cell optimization.',
        scene='boundary / rate0.8 / 3cells / 36UE / 17PRGx16PRB', seeds=SEEDS, arms=ARMS,
        warmup=4000, settling=4000, measured=20000, followup=400, horizon=28400,
        measurement_first=8001, measurement_last=28000, measurement_seconds=10,
        ttl_ttis=400, tti_ms=.5, common_native_pfq_warmup=True,
        target_success_bytes_per_ue_tti=2400, ema_decay=.995, ema_initial=0,
        weight='clip(target/max(EMA,1e-9),1,4), then mean-normalize; all-one control.',
        floor='Every search uses ceil(.90 * its own current draft dequeue). CTBS changes draft AND any downstream floor; raw-vs-pruned dequeue differences are audited, not assumed zero.',
        global_search='Exactly frozen legacy scorer, ordered full neighborhood, argmax tie breaking, >1B first2 then >16B, complete convergence, 256 cap is failure.',
        independent_control='Each cell optimizes only its own weighted expected bytes, with other cells FIXED to initial CTBS draft; independent per-cell ceil(.9*Dcell); all three final actions commit simultaneously. This control also removes shared floor-budget pooling and joint edits; no claim to isolate those submechanisms.',
        factorial='2x2x2 CTBS C / global-search G / weights W; W has no behavioral effect when G=0, so 8 logical cells collapse to6 unique policies. Two extra independent controls distinguish global search from within-cell optimization.',
        primary_metric='complete_payload_mbps', secondary_metrics=['goodput_mbps','expiry_pct','complete_success_pct','success_delay_mean_ms','success_delay_p95_ms','outcome_mean_ms','ue_complete_payload_p5_mbps','ue_goodput_jain','tb_bler_pct','prg_utilization'],
        contrasts=[['ctbs','pfq'],['ctbs_global_equal','ctbs'],['full','ctbs_global_equal'],['full','raw_global_deficit'],['raw_global_equal','pfq'],['raw_global_deficit','raw_global_equal'],['ctbs_global_equal','raw_global_equal'],['ctbs_independent_equal','ctbs'],['ctbs_global_equal','ctbs_independent_equal'],['ctbs_independent_deficit','ctbs_independent_equal'],['full','ctbs_independent_deficit']],
        attribution='Report direct additions/removals, CTBSxsearch interaction at W=0, CTBSxweight interaction under G=1, and per-seed Shapley over all6 component orders; Shapley is an allocation convention, not unique causal contribution.',
        statistics='Seed is experimental unit n5, equal-seed mean/sampleSD, paired nominal95%t intervals df4; Holm correction across11 predeclared primary contrasts. Windows/UE/packets are not independent replicates.',
        no_tuning=True, stop='Exactly30 new full native trajectories; retain unfavorable outcomes. No new seeds or shortened measurement. Smoke/preflight only establish correctness.',
        limits=['Existing development seeds, not blind tests.','Boundary-only attribution; uniform not evaluated in this campaign.','Frozen legacy BLER interpolation is preserved.','No HARQ; full finite mature packet cohorts.','Concurrent shared-host timing is descriptive, not deployment latency or a controlled speed comparison.','Factorial G includes physical optimization as well as coordination; independent controls address this distinction.','No claim of global optimality or asymptotic stability.'],
        simulator_sha256=sha(REFERENCE/'simulator'), reference=reference,
        registry_sha256=sha(OUT.parents[1]/'training/stageb_hgraph/current_algorithm.json'),
        source_sha256={name:sha(OUT/name) for name in ['common.py','policy.py','run.py','native_run_generated.py']})
    write(OUT / 'protocol.json', protocol)
    print('PROTOCOL_FROZEN', sha(OUT / 'protocol.json'))


if __name__ == '__main__':
    main()

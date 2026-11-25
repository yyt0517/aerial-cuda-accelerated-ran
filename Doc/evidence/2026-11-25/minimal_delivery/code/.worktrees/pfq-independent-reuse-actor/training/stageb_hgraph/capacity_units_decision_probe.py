"""Aggregation for the single-seed, corrected-units five-policy decision probe."""
from collections import defaultdict

import numpy as np

from .dpa_ablation_experiment import read_csv, sha
from .dpa_reproduction_experiment import offered_identity
from .dpa_v1_historical_comparison import WindowSpec
from .pfq_dpa_aligned_comparison import summarize_aligned_window
from .phy_tbs_alignment_audit import determine_tbs_bytes, MCS_CODE_RATES, MCS_QAM_ORDERS


def check_capacity_row(row):
    bits, capacity, dequeue = (int(row[k]) for k in
        ("tbs_capacity_bits", "tbs_capacity_bytes", "scheduled_dequeue_bytes"))
    if not (0 <= 8 * dequeue <= bits and capacity == bits // 8):
        raise ValueError(f"raw bits/dequeue bound failed: {row}")
    if not 0 <= int(row["successful_bytes"]) <= dequeue:
        raise ValueError("successful bytes exceed dequeue")
    count, mcs, layer = (int(row[k]) for k in ("actual_prg_count", "mcs", "layer"))
    if count == 0:
        if bits or capacity or dequeue:
            raise ValueError("unallocated UE has nonzero capacity/service")
        return
    # Independent literal bit-table checks; not obtained from the Python mirror.
    if count == 1 and mcs == 0 and bits != {1: 504, 2: 984, 3: 1480, 4: 2024}[layer]:
        raise ValueError("literal TBS bits check failed")
    if count == 1 and mcs == 4 and layer == 4 and bits != 9992:
        raise ValueError("literal large TBS bits check failed")
    predicted = determine_tbs_bytes(count * 16, 11, layer,
                                   MCS_CODE_RATES[mcs] / 1024., MCS_QAM_ORDERS[mcs])
    if capacity != predicted:
        raise ValueError("C++/Python physical-byte parity failed")


def canonical_actions(rows, *, total_tti, n_cell=3, n_prg=17):
    actions = np.full((total_tti, n_cell * n_prg), -1, dtype=np.int32)
    seen = set()
    for row in rows:
        t, ue, cell = (int(row[k]) for k in ("tti", "ue_id", "serving_cell"))
        if (t, ue) in seen:
            raise ValueError("duplicate UE/TTI row")
        seen.add((t, ue))
        if not (1 <= t <= total_tti and 0 <= cell < n_cell):
            raise ValueError("invalid TTI/cell")
        mask = int(str(row["actual_prg_mask"]), 0)
        if mask < 0 or mask >= 1 << n_prg:
            raise ValueError("invalid PRG bitmap")
        for prg in range(n_prg):
            if mask & (1 << prg):
                index = cell * n_prg + prg
                if actions[t-1, index] != -1:
                    raise ValueError("duplicate allocation")
                actions[t-1, index] = ue
    return actions


def compare_actions(pfq, dpa, *, warmup):
    if pfq.shape != dpa.shape or pfq.ndim != 2 or not 0 <= warmup < len(pfq):
        raise ValueError("unaligned action traces")
    equal = pfq == dpa
    divergence = np.flatnonzero(~equal.all(axis=1))
    measured_divergence = np.flatnonzero(~equal[warmup:].all(axis=1))
    return {
        "first_divergence_tti": int(divergence[0] + 1) if len(divergence) else None,
        "first_measured_divergence_tti": int(measured_divergence[0] + warmup + 1) if len(measured_divergence) else None,
        "full_token_agreement": float(equal.mean()),
        "measured_token_agreement": float(equal[warmup:].mean()),
        "measured_whole_tti_agreement": float(equal[warmup:].all(axis=1).mean()),
        "semantics": "Executed UE/Blank per (cell,PRG), including matching Blanks. Independent trajectories; after first divergence this is NOT a same-state policy comparison.",
    }


def summarize_arm(out, *, warmup, measured):
    total = warmup + measured
    service = read_csv(out / "service_balance_trace.csv")
    if len(service) != total * 36 or {(int(r['tti']), int(r['ue_id'])) for r in service} != {
            (t, u) for t in range(1, total + 1) for u in range(36)}:
        raise ValueError("incomplete/duplicate service trace")
    for row in service:
        check_capacity_row(row)
    actions = canonical_actions(service, total_tti=total)
    np.save(out / "canonical_actions.npy", actions)
    outcomes = read_csv(out / "packet_outcome_samples.csv")
    metrics = summarize_aligned_window(service_rows=service, outcome_rows=outcomes,
        decision_rows=[], window=WindowSpec("measured", warmup + 1, total))
    selected = [r for r in service if int(r["tti"]) > warmup]
    duration = measured * .0005
    metrics['offered_mbps'] = sum(int(r['offered_bytes']) for r in selected) * 8 / duration / 1e6
    metrics['scheduled_phy_capacity_mbps'] = sum(int(r['tbs_capacity_bits']) for r in selected) / duration / 1e6
    metrics['offered_to_scheduled_phy_capacity'] = (metrics['offered_mbps'] /
        metrics['scheduled_phy_capacity_mbps']) if metrics['scheduled_phy_capacity_mbps'] else None
    queue_by_tti = defaultdict(int)
    served_by_tti = defaultdict(int)
    for row in selected:
        t = int(row['tti'])
        queue_by_tti[t] += int(row['post_queue_bytes'])
        served_by_tti[t] += int(int(row['scheduled_dequeue_bytes']) > 0)
    metrics['total_backlog_mean_bytes'] = float(np.mean(list(queue_by_tti.values())))
    metrics['total_backlog_p95_bytes'] = float(np.percentile(list(queue_by_tti.values()), 95))
    metrics['per_ue_backlog_mean_bytes'] = float(np.mean([int(r['post_queue_bytes']) for r in selected]))
    metrics['per_ue_backlog_p95_bytes'] = float(np.percentile([int(r['post_queue_bytes']) for r in selected], 95))
    metrics['served_ue_per_tti_mean'] = float(np.mean(list(served_by_tti.values())))
    delays = [float(r['system_time_ms']) for r in outcomes
              if warmup < int(r['terminal_tti']) <= total and r['outcome'] == 'delivered']
    metrics['completion_delay_mean_ms'] = float(np.mean(delays)) if delays else None
    metrics['completion_delay_p95_ms'] = float(np.percentile(delays, 95)) if delays else None
    metrics['completion_count'] = len(delays)
    metrics['capacity_semantics'] = 'Sum of selected raw PHY TBS before BLER/demand clipping, not an optimal/sustainable achievable capacity.'
    channel = read_csv(out / 'service_balance_trace.csv.channel.csv')
    channel = [r for r in channel if int(r['tti']) <= total]
    if [int(r['tti']) for r in channel] != list(range(1, total + 1)):
        raise ValueError('incomplete channel fingerprint trace')
    # Bridge may prepare TTI total+1 before CLOSE; do not compare that unused step.
    import hashlib
    channel_sha = hashlib.sha256('\n'.join(r['raw_cfr_fnv1a64'] for r in channel).encode()).hexdigest()
    return {'metrics': metrics, 'unit_checked_rows': len(service),
            'literal_checked_rows': sum(int(r['actual_prg_count']) == 1 and int(r['mcs']) == 0 for r in service),
            'offered_identity': offered_identity(out)[0], 'topology_sha256': sha(out/'ue_topology.csv'),
            'channel_fingerprint_sha256': channel_sha}

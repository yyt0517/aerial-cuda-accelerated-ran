"""Recompute the paired Boundary M-LWDF-PF/PFQ report without native rollout.

The input traces are local research artifacts. This command performs no PHY
evaluation, training, or new timing experiment and never overwrites an output.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIRST, LAST, SECONDS = 8001, 12000, 2.0
SEEDS = (41, 42, 43)
PROPOSED = "M-LWDF-PF + bounded DPLS"
FIELDS = (
    "tti", "ue_id", "offered_bytes", "accepted_bytes", "rejected_bytes",
    "expired_bytes", "previous_post_queue_bytes", "pre_queue_bytes",
    "scheduled_dequeue_bytes", "post_queue_bytes", "successful_bytes",
    "failed_bytes", "actual_prg_count", "mcs", "layer", "phy_sinr_db_cpu",
    "predicted_bler_cpu", "tb_tx_count", "tb_err_count",
)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_cohort(pk, raw, seed, arm):
    assert len(pk) and not pk.censored.any()
    assert pk.arrival_tti.between(FIRST, LAST).all()
    good = pk[pk.successfully_delivered]
    per = good.groupby("ue_id").packet_size_bytes.sum().reindex(range(36), fill_value=0) * 8 / SECONDS / 1e6
    expiry = pk.assign(expired=pk.outcome.eq("expired")).groupby("ue_id").expired.mean() * 100
    data = np.memmap(raw / "metrics.bin", mode="r", dtype="<f8").reshape(-1, 36, len(FIELDS))
    service = data[FIRST - 1:LAST]
    idx = {key: i for i, key in enumerate(FIELDS)}
    np.testing.assert_array_equal(service[:, :, idx["tti"]], np.broadcast_to(np.arange(FIRST, LAST + 1)[:, None], (4000, 36)))
    np.testing.assert_array_equal(service[:, :, idx["ue_id"]], np.broadcast_to(np.arange(36), (4000, 36)))
    flags = service[:, :, idx["successful_bytes"]] > 0
    per_ue = []
    for ue in range(36):
        successes = np.flatnonzero(np.r_[True, flags[:, ue], True])
        per_ue.append(dict(seed=seed, arm=arm, ue=ue, payload_mbps=float(per.loc[ue]),
            expiry_pct=float(expiry.loc[ue]), success_mean_ms=float(good[good.ue_id == ue].system_time_ms.mean()),
            longest_gap_tti=int(np.max(np.diff(successes) - 1))))
    row = dict(seed=seed, arm=arm, cohort_packets=len(pk), payload_mbps=float(per.sum()),
        goodput_mbps=float(service[:, :, idx["successful_bytes"]].sum() * 8 / SECONDS / 1e6),
        success_pct=float(100 * pk.successfully_delivered.mean()), expiry_pct=float(100 * pk.outcome.eq("expired").mean()),
        other_nondelivery_pct=float(100 * ((~pk.successfully_delivered) & (~pk.outcome.eq("expired"))).mean()),
        mean_ms=float(good.system_time_ms.mean()), P95_ms=float(good.system_time_ms.quantile(.95)),
        P99_ms=float(good.system_time_ms.quantile(.99)), terminal_mean_ms=float(pk.system_time_ms.mean()),
        terminal_P99_ms=float(pk.system_time_ms.quantile(.99)), ue_p5_mbps=float(per.quantile(.05)),
        ue_min_mbps=float(per.min()), jain=float(per.sum() ** 2 / (36 * (per ** 2).sum())),
        worst_ue_expiry_pct=float(expiry.max()), zero_UEs=int((per == 0).sum()),
        long_gap_UEs=sum(item["longest_gap_tti"] >= 400 for item in per_ue),
        PRG_utilization_pct=float(100 * service[:, :, idx["actual_prg_count"]].sum() / (4000 * 51)),
        tb_error_pct=float(100 * service[:, :, idx["tb_err_count"]].sum() / service[:, :, idx["tb_tx_count"]].sum()))
    return row, per_ue


def main(args):
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    inputs = {Path(__file__).resolve()}
    build = args.reference.parent / "boundary08_fast_multiseed_20261010_v1/build.py"
    definition = next(stmt.value for stmt in ast.parse(build.read_text()).body
        if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FIELDS" for t in stmt.targets))
    assert isinstance(definition, ast.Call) and tuple(kw.arg for kw in definition.keywords) == FIELDS
    inputs.add(build)
    rows, ues, cohorts, searches, paired = [], [], {}, [], []
    for seed in SEEDS:
        baseline = args.baseline_cohorts / f"pfq_seed{seed}.csv.gz"
        proposed = args.proposed / f"r0p8_mlwdf_pf_seed{seed}"
        inputs.update((baseline, proposed / "cohort.csv.gz", proposed / "summary.json", proposed / "search.csv.gz"))
        bp = pd.read_csv(baseline)
        bp = bp[bp.arrival_tti.between(FIRST, LAST)].reset_index(drop=True)
        mp = pd.read_csv(proposed / "cohort.csv.gz")
        keys = ["ue_id", "packet_id", "arrival_tti", "packet_size_bytes", "ttl_tti"]
        assert not bp.duplicated(["ue_id", "packet_id"]).any()
        assert not mp.duplicated(["ue_id", "packet_id"]).any()
        pd.testing.assert_frame_equal(bp[keys].sort_values(keys).reset_index(drop=True), mp[keys].sort_values(keys).reset_index(drop=True))
        for arm, pk, raw in (("PFQ", bp, args.reference / f"seed{seed}_pfq0_20000"),
                (PROPOSED, mp, args.proposed_raw / proposed.name)):
            inputs.add(raw / "metrics.bin")
            row, per_ue = evaluate_cohort(pk, raw, seed, arm)
            if arm == PROPOSED:
                archived = json.loads((proposed / "summary.json").read_text())
                for metric, original in (("payload_mbps", "complete_payload_mbps"), ("goodput_mbps", "goodput_mbps"),
                        ("mean_ms", "success_delay_mean_ms"), ("P95_ms", "success_delay_p95_ms"),
                        ("ue_p5_mbps", "ue_complete_payload_p5_mbps")):
                    np.testing.assert_allclose(row[metric], archived[original], rtol=0, atol=1e-8)
            rows.append(row)
            ues.extend(per_ue)
            cohorts[seed, arm] = pk
        search = pd.read_csv(proposed / "search.csv.gz")
        search = search[search.tti.between(FIRST, LAST)]
        assert len(search) == 4000 and search.full_candidate_count.gt(0).all()
        searches.append(search)
        common = bp[bp.successfully_delivered].merge(mp[mp.successfully_delivered],
            on=["ue_id", "packet_id"], suffixes=("_PFQ", "_MLWDF"), validate="one_to_one")
        delta = common.system_time_ms_MLWDF - common.system_time_ms_PFQ
        paired.append(dict(seed=seed, count=len(common), mean_delta_ms=float(delta.mean()),
            faster=int((delta < 0).sum()), equal=int((delta == 0).sum()), slower=int((delta > 0).sum())))
    per_seed = pd.DataFrame(rows)
    aggregate = []
    for arm, group in per_seed.groupby("arm", sort=False):
        pk = pd.concat([cohorts[seed, arm] for seed in SEEDS])
        good = pk[pk.successfully_delivered]
        aggregate.append(dict(arm=arm, **{key: float(group[key].mean()) for key in
            ("payload_mbps", "goodput_mbps", "ue_p5_mbps", "ue_min_mbps", "jain", "PRG_utilization_pct", "tb_error_pct")},
            cohort_packets=len(pk), successful_packets=len(good), success_pct=float(100 * pk.successfully_delivered.mean()),
            expiry_pct=float(100 * pk.outcome.eq("expired").mean()),
            other_nondelivery_pct=float(100 * ((~pk.successfully_delivered) & (~pk.outcome.eq("expired"))).mean()),
            mean_ms=float(good.system_time_ms.mean()), P95_ms=float(good.system_time_ms.quantile(.95)),
            P99_ms=float(good.system_time_ms.quantile(.99)), terminal_mean_ms=float(pk.system_time_ms.mean()),
            terminal_P99_ms=float(pk.system_time_ms.quantile(.99)), worst_ue_expiry_pct=float(group.worst_ue_expiry_pct.max()),
            zero_UEs=int(group.zero_UEs.sum()), long_gap_UEs=int(group.long_gap_UEs.sum())))
    base, proposed = aggregate
    comparison = [dict(metric=key, PFQ=base[key], M_LWDF_PF_bounded=proposed[key], delta=proposed[key] - base[key],
        relative_change_pct=100 * (proposed[key] / base[key] - 1) if base[key] else None) for key in base if key != "arm"]
    search = pd.concat(searches)
    cost = dict(quality_run_wall_mean_ms=float(search.wall_ms.mean()), quality_run_wall_P95_ms=float(search.wall_ms.quantile(.95)),
        quality_run_wall_max_ms=float(search.wall_ms.max()), candidate_work_ratio=float(search.candidates.sum() / search.full_candidate_count.sum()),
        candidate_ratio_P95=float((search.candidates / search.full_candidate_count).quantile(.95)),
        candidates_mean=float(search.candidates.mean()), mmse_misses_mean=float(search.mmse_misses.mean()))
    ue_table = pd.DataFrame(ues)
    bu = ue_table[ue_table.arm == "PFQ"].set_index(["seed", "ue"])
    mu = ue_table[ue_table.arm == PROPOSED].set_index(["seed", "ue"])
    validation = dict(matching_cohort_fields=keys, matched_cohort_packets=base["cohort_packets"], all_cohorts_uncensored=True,
        proposed_metrics_equal_archived_summaries_atol=1e-8, service_rows_per_seed=4000 * 36,
        matched_seed_UEs=len(bu), higher_payload_UEs=int((mu.payload_mbps > bu.payload_mbps).sum()),
        lower_success_mean_UEs=int((mu.success_mean_ms < bu.success_mean_ms).sum()))
    engineering = pd.read_csv(args.engineering_wall)
    inputs.add(args.engineering_wall)
    timing = []
    for rate in (.8, 1.12, 1.28):
        by_arm = engineering[engineering.rate == rate].set_index("arm")
        full = float(by_arm.loc["full", "mean_ms"])
        for arm in ("heap", "native_direct"):
            elapsed = float(by_arm.loc[arm, "mean_ms"])
            timing.append(dict(rate=rate, load_multiplier=rate / .8, arm=arm, full_mean_ms=full, mean_ms=elapsed,
                reduction_pct=100 * (1 - elapsed / full), speedup=full / elapsed,
                input_weights="old k0 deficit weights; not M-LWDF-PF", statistic="mean of saved-state median wall times"))
    result = dict(protocol=dict(seeds=list(SEEDS), first_tti=FIRST, last_tti=LAST, TTI_ms=.5, measurement_seconds=SECONDS,
        followup_TTIs=400, scenario="Boundary/rate0.8", aggregation="Throughputs, UE P5, Jain, PRG and TB error: seed means; packet rates and delays: pooled; worst UE: maximum over seed,UE",
        timing="Concurrent quality-run wall times are descriptive only. No fair speed comparison to PFQ or full under M-LWDF-PF weights.",
        new_native_steps=0, new_PHY_evaluations=0, new_training_steps=0), aggregate=aggregate, comparison=comparison,
        per_seed=rows, quality_run_cost=cost, common_success_pairing=paired, validation=validation, engineering_timing=timing)
    def label(path):
        try:
            return str(path.resolve().relative_to(ROOT))
        except ValueError:
            return str(path.resolve())
    manifest = {label(path): sha(path) for path in sorted(inputs)}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "comparison.json", result)
    write_json(args.output / "input_manifest.json", manifest)
    per_seed.to_csv(args.output / "per_seed.csv", index=False)
    ue_table.to_csv(args.output / "per_ue.csv", index=False)
    pd.DataFrame(comparison).to_csv(args.output / "comparison.csv", index=False)
    pd.DataFrame(timing).to_csv(args.output / "engineering_timing.csv", index=False)
    pd.DataFrame(paired).to_csv(args.output / "common_success_pairing.csv", index=False)
    print(json.dumps(dict(output=str(args.output), validation=validation, payload_gain_pct=comparison[0]["relative_change_pct"]), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    formal = ROOT / ".worktrees/pfq-independent-reuse-actor/output/pfq_independent_reuse/formal"
    parser.add_argument("--reference", type=Path, default=formal / "boundary08_steady_state_20261021_v1")
    parser.add_argument("--baseline-cohorts", type=Path, default=ROOT / "deliverables/dpls_component_ablation_20261029_v1/packet_cohorts")
    parser.add_argument("--proposed", type=Path, default=ROOT / "Doc/evidence/2026-11-04/crossload_weights_v2/native")
    parser.add_argument("--proposed-raw", type=Path, default=ROOT / "deliverables/crossload_weights_20261104_v2")
    parser.add_argument("--engineering-wall", type=Path, default=ROOT / "Doc/evidence/2026-11-04/B_decoupled_and_tournament/engineering/wall_summary.csv")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())

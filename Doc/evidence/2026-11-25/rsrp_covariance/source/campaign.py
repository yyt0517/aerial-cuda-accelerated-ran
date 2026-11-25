"""Paired native runs with frozen exact/scalar baselines and spatial periods."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback
import numpy as np
from training.stageb_ppo.native import ROOT, write, sha, EXPECTED_SIMULATOR
from training.stage2_closure.native import ReportNativeEnv
from training.stageb_lns.no_ctbs_inputs import detach_raw_phy_row, problem_from_raw_row, raw_pfq_draft
from training.stageb_lns.bounded_dpls import solve_bounded
from training.stageb_lns.ttl_weights import hol_urgency
from training.stageb_lns.crossload_native import weights_for
from training.stageb_rsrp.model import RsrpConfig, RsrpMeasurements, approximate_problem
from training.stageb_rsrp.campaign import source_files as prior_sources
from training.stageb_rsrp_ablation.campaign import reuse
from training.stageb_rsrp_ablation.model import check_counters
from .model import (STAGES, SPATIAL_STAGES, PERIODS, SpatialReports, SpatialReceiver,
                    spatial_problem, setting, diagnostic_times)

CONFIG = RsrpConfig()


def key(job):
    return f"{job['scene']}_r{job['rate']:g}_{job['stage']}_{job['objective']}_s{job['seed']}"


def sources():
    return prior_sources() + [ROOT/'training/stageb_rsrp_ablation'/n for n in ('campaign.py', 'model.py')] + [
        Path(__file__).with_name(n) for n in ('__init__.py', 'model.py', 'campaign.py')]


def run_one(root, job, smoke=False):
    import torch
    import pandas as pd
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    out = Path(root)/key(job)
    if out.exists():
        raise FileExistsError(out)
    if job['stage'] in ('exact', 'reported') and not smoke:
        return reuse(out, job)
    warm, settle, measure, follow = (32, 16, 64, 400) if smoke else (4000, 4000, 20000, 400)
    first, last = warm+settle+1, warm+settle+measure
    env = ReportNativeEnv(out, job['seed'], last+follow, job['rate'],
                         'uniform' if job['scene'] == 'uniform' else 'coop_boundary')
    launch = json.loads((out/'launch.json').read_text())
    launch.update(job, common_pfq_warmup=warm, settling=settle, measured=measure, followup=follow,
        measurement_first_tti=first, measurement_last_tti=last, rsrp_config=asdict(CONFIG),
        spatial=dict(normalization='trace R = 4; C_hat = 4 beta_RSRP R', delay_tti=1,
                     temporal_smoothing=False, normalized_real_parameters=15,
                     period_tti=setting(job['stage'])[1] if job['stage'] in SPATIAL_STAGES else None),
        execution='frozen original native channel/PFQ/MCS/layers/OLLA/traffic/search')
    write(out/'launch.json', launch)
    (out/'states').mkdir(); (out/'diagnostic_states').mkdir()
    witness = job['stage'] == 'prg_t1'
    stages = SPATIAL_STAGES if witness else ((job['stage'],) if job['stage'] in SPATIAL_STAGES else ())
    spatial = SpatialReports(np.asarray(env.inputs.serving_cell), stages)
    receiver = SpatialReceiver()
    meter = RsrpMeasurements(CONFIG)
    ema = np.zeros(36)
    records, states, diagnostics, channels, reports, covariance_hashes = [], [], [], [], [], []
    predicted = np.empty((env.horizon-warm, 36, 2), dtype=np.float64)
    frozen = {str(p): sha(p) for p in sources()}
    saved = {first, first+measure//2, last}
    diagnostic_saved = set(diagnostic_times(job['seed'], smoke)) if witness else set()
    try:
        for _ in range(env.horizon):
            t = env.tti+1
            original = env.snapshot['h']
            digest = hashlib.sha256(original.tobytes()).hexdigest(); channels.append(digest)
            stamp = time.perf_counter()
            meter.observe(original, env.tti)
            spatial.observe(original, env.tti)
            measurement_ms = (time.perf_counter()-stamp)*1000
            if t <= warm:
                if job['stage'] in spatial.available:
                    sampled, packet = spatial.read(job['stage'])
                    receiver.accept(sampled, packet)
                action = env.raw_pfq()
            else:
                begin = time.perf_counter(); cpu = time.process_time()
                row = detach_raw_phy_row(dict(tti=t, obs=env.obs, snapshot=env.snapshot, ema=ema.copy(), ctbs=env.inputs))
                p = problem_from_raw_row(row)
                urgency, slack = hol_urgency(env.fifo, t, p.queue)
                if job['objective'] == 'mlwdf_pf':
                    p.weights = weights_for('mlwdf_pf', p.weights, ema, urgency, p.queue, 0.)
                report, report_tti = meter.read()
                reports.append(hashlib.sha256(report.tobytes()).hexdigest())
                draft = raw_pfq_draft(p, row['obs'])
                refresh, age, decode_cpu_ms = False, -1, 0.
                if job['stage'] == 'exact':
                    q = p
                elif job['stage'] == 'reported':
                    q = approximate_problem(p, report)
                else:
                    sampled, packet = spatial.read(job['stage'])
                    decode_start = time.process_time()
                    refresh = receiver.accept(sampled, packet)
                    decode_cpu_ms = (time.process_time()-decode_start)*1000
                    age = env.tti-sampled
                    assert 1 <= age <= setting(job['stage'])[1]
                    assert refresh == ((env.tti-1) % setting(job['stage'])[1] == 0)
                    q = spatial_problem(p, report, receiver.factor)
                    covariance_hashes.append(hashlib.sha256(np.int64(sampled).tobytes()+packet.tobytes()).hexdigest())
                action, values, info = solve_bounded(q, draft)
                assert info['converged']; check_counters(info)
                records.append(dict(tti=t, wall_ms=(time.perf_counter()-begin)*1000,
                    cpu_ms=(time.process_time()-cpu)*1000, measurement_emulator_ms=measurement_ms,
                    covariance_decode_cpu_ms=decode_cpu_ms, covariance_refresh=int(refresh),
                    covariance_age_tti=age, rsrp_age_tti=env.tti-report_tti,
                    **{k: v for k, v in info.items() if k not in ('core_ms', 'version')}))
                predicted[t-warm-1] = values[:, 4:6]
                if t in saved or t in diagnostic_saved:
                    row.update(objective_weights=p.weights.copy(), hol_urgency=urgency, hol_slack=slack,
                        selected_action=action.copy(), selected_values=values.copy(), stage=job['stage'],
                        rsrp_report=report.copy(), rsrp_filtered=meter.filtered.copy(), rsrp_report_tti=report_tti,
                        rsrp_config=asdict(CONFIG), spatial_reports=spatial.snapshot())
                    if t in saved:
                        path = out/'states'/f'tti{t:06d}.pt'; torch.save(row, path)
                        states.append(dict(path=str(path), sha256=sha(path)))
                    if t in diagnostic_saved:
                        path = out/'diagnostic_states'/f'tti{t:06d}.pt'; torch.save(row, path)
                        diagnostics.append(dict(path=str(path), sha256=sha(path)))
            assert hashlib.sha256(original.tobytes()).hexdigest() == digest
            env.step(action)
            ema = .995*ema+.005*np.asarray(env.info['actual_ue_goodput_bytes'])
            if t % 4000 == 0:
                print('PROGRESS', key(job), t, env.horizon, flush=True)
        env.close()
    except BaseException:
        if not env.log.closed: env.close(validate=False)
        raise
    for path, digest in frozen.items():
        assert sha(path) == digest, path
    pd.DataFrame(records).to_csv(out/'search.csv.gz', index=False)
    np.savez_compressed(out/'selected_predictions.npz', tti=np.arange(warm+1, env.horizon+1),
                        sinr_db=predicted[:, :, 0], bler=predicted[:, :, 1])
    np.save(out/'channel_hashes.npy', np.asarray(channels))
    np.save(out/'report_hashes.npy', np.asarray(reports))
    np.save(out/'covariance_hashes.npy', np.asarray(covariance_hashes))
    write(out/'saved_states.json', states); write(out/'saved_diagnostic_states.json', diagnostics)
    write(out/'spatial_validation.json', dict(passed=True, original_H_unchanged=env.horizon,
        candidate_accounting_checks=len(records), source_sha256=frozen,
        transport_delay_tti=1, update_period_and_cache_checked=True,
        measurements='ideal full-port spatial estimate; unquantized 15-real Hermitian reports, neighbor links only',
        witness_all_period_measurements=witness, witness_measurement_cost_excluded_from_scheduler=True))
    result = dict(**job, key=key(job), reused=False, directory=str(out), native_passed=True, smoke=smoke)
    write(out/'done.json', result)
    return result


def main(args):
    root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=False)
    seeds = list(map(int, args.seeds.split(',')))
    stages = args.stages.split(',') if args.stages else STAGES
    assert set(stages) <= set(STAGES)
    jobs = [dict(scene=scene, rate=rate, stage=stage, objective=objective, seed=seed)
        for stage in stages for scene, rate in [('boundary', .8), ('uniform', 1.)]
        for objective in ('deficit', 'mlwdf_pf') for seed in (seeds[:1] if args.smoke else seeds)]
    write(root/'protocol.json', dict(version='rsrp_normalized_spatial_covariance_v1', stages=list(stages),
        periods_tti=list(PERIODS), periods_ms=[k*.5 for k in PERIODS], jobs=jobs, seeds=seeds,
        smoke=args.smoke, windows=[32, 16, 64, 400] if args.smoke else [4000, 4000, 20000, 400],
        rsrp=asdict(CONFIG), covariance_delay_tti=1, covariance_temporal_filter=False,
        covariance_normalization='R = 4 C / trace C; C_hat = 4 beta_RSRP R',
        wideband='average full-port covariance over PRGs BEFORE trace normalization',
        feedback='15 independent real shape parameters per neighbor matrix, scalar RSRP separately; no quantization in PHY',
        witness='prg_t1 records all periods, with six stratified/dithered report phases for matched-state timing',
        simulator=EXPECTED_SIMULATOR, workers=args.workers,
        sources={str(p.relative_to(ROOT)): sha(p) for p in sources()}))
    completed, failed = [], []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
        pending = {pool.submit(run_one, str(root), job, args.smoke): job for job in jobs}
        for future in as_completed(pending):
            try:
                completed.append(future.result())
                print('DONE', key(pending[future]), len(completed), '/', len(jobs), flush=True)
            except BaseException:
                failed.append(dict(job=pending[future], error=traceback.format_exc()))
                print('FAILED', failed[-1], flush=True)
            write(root/'progress_campaign.json', dict(completed=completed, failed=failed, total=len(jobs)))
    write(root/'completion.json', dict(completed=not failed, runs=completed, failed=failed, total=len(jobs)))
    if failed: raise RuntimeError(f'{len(failed)} failed runs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True); parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--seeds', default='41,42,43'); parser.add_argument('--stages')
    parser.add_argument('--smoke', action='store_true'); main(parser.parse_args())

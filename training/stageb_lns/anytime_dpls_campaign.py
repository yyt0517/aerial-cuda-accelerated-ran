"""Step-0 native instrumentation and causal cold/warm truncated DPLS controls.

Runs are serial, append no data to old experiments, and never query a current
full teacher to initialize a cheaper arm. Sparse saved states are not treated
as consecutive TTIs. Use replay for isolated timing on captured real inputs.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil
import time
import numpy as np

from .physics import ROOT, ReferencePhy
from .no_ctbs_inputs import detach_raw_phy_row, problem_from_raw_row, raw_pfq_draft
from .ranked_dpls import RankedDplsConfig, solve_ranked
from .dpls_trace import DplsTrace, solution_overlap
from .experiment import write_json, sha


def arm_config(arm):
    match = re.fullmatch(r'(cold|warm)_(full|n([1-9][0-9]*)|b([1-9][0-9]*))', arm)
    if match is None:
        raise ValueError('Arm must be cold/warm_full, cold/warm_nN or cold/warm_bB')
    return RankedDplsConfig(max_rounds=int(match[3]) if match[3] else 256,
                            max_candidates=int(match[4]) if match[4] else None,
                            profile=True), match[1] == 'warm'


def quantiles(values):
    values = np.asarray(values, dtype=float)
    return dict(mean=float(values.mean()), p05=float(np.quantile(values, .05)),
                p50=float(np.quantile(values, .5)), p95=float(np.quantile(values, .95)),
                maximum=float(values.max()))


def summarize(records):
    if not records:
        return dict(decisions=0)
    keys = ('rounds', 'candidate_evaluations', 'ue_evaluations', 'mmse_requests', 'mmse_misses',
            'e2e_ms', 'core_ms', 'generation_ms', 'physical_ms', 'native_kernel_ms',
            'validation_ms', 'output_copy_ms', 'observer_ms', 'trace_write_ms',
            'slot_agreement', 'active_assignment_jaccard', 'activity_agreement')
    result = dict(decisions=len(records), converged=sum(r['converged'] for r in records),
                  warm_selected=sum(r['initialization'] == 'previous_executed' for r in records))
    for key in keys:
        values = [r[key] for r in records if r.get(key) is not None]
        if values:
            result[key] = quantiles(values)
    # Includes NumPy setup/reductions, allocations, copies, ctypes overhead and
    # Python. It cannot be interpreted as an exclusive Python CPU profile.
    result['other_ms'] = quantiles([r['e2e_ms']-r['native_kernel_ms']-r['validation_ms']-
                                    r['output_copy_ms']-r['observer_ms'] for r in records])
    return result


def native_one(output, arm, seed, args):
    import torch
    from training.stageb_ppo.native import NativeEnv, packets, REFERENCE, write
    config, warm = arm_config(arm)
    out = output/f'{arm}_seed{seed}'
    env = NativeEnv(out, seed, args.warmup+args.settling+args.measured+args.followup)
    trace_dir = out/'traces'
    trace_dir.mkdir()
    (out/'states').mkdir()
    ema = np.zeros(36)
    previous = None
    records = []
    first = args.warmup+args.settling+1
    last = first+args.measured-1
    # NativeEnv's shared transport was originally made for equal-weight PPO.
    # This arm retains the registered causal deficit EMA and objective.
    launch = json.loads((out/'launch.json').read_text())
    launch.update(business_weights=True, scheduling_objective='canonical_native_bler_causal_deficit',
                  dpls_config=asdict(config), warm_start=warm)
    try:
        write(out/'launch.json', launch)
        with (out/'decisions.jsonl').open('x') as log:
            for _ in range(env.horizon):
                tti = env.tti+1
                measured = first <= tti <= last
                row = None
                if env.tti < args.warmup:
                    action = env.raw_pfq()
                else:
                    trace = DplsTrace() if measured and args.trace_every and (tti-first)%args.trace_every == 0 else None
                    started = time.perf_counter()
                    row = detach_raw_phy_row(dict(tti=tti, obs=env.obs, snapshot=env.snapshot,
                        ema=ema.copy(), ctbs=env.inputs))
                    problem = problem_from_raw_row(row)
                    draft = raw_pfq_draft(problem, row['obs'])
                    action, values, record = solve_ranked(problem, draft, config,
                        warm_start=previous if warm else None, observer=trace)
                    record['e2e_ms'] = (time.perf_counter()-started)*1000
                    record.update(tti=tti, seed=seed, arm=arm, measured=measured)
                    record.update(solution_overlap(previous, action) if previous is not None else {})
                    record['trace_write_ms'] = 0.
                    if trace is not None:
                        started = time.perf_counter()
                        trace.write(trace_dir/f'tti{tti:06d}.npz', action=action, values=values)
                        record['trace_write_ms'] = (time.perf_counter()-started)*1000
                    if measured and args.save_every and (tti-first)%args.save_every == 0:
                        # Previous arm-executed action is causal; current full
                        # decision is deliberately excluded from replay input.
                        captured = dict(row, previous_action=None if previous is None else previous.copy(),
                                        previous_tti=tti-1, seed=seed, source_arm=arm)
                        torch.save(captured, out/'states'/f'tti{tti:06d}.pt')
                        np.testing.assert_allclose(values, ReferencePhy(problem).evaluate(action)[0],
                                                   rtol=1.e-8, atol=1.e-6)
                    if arm.endswith('_full') and not record['converged']:
                        raise RuntimeError('Full DPLS reached its round cap')
                    log.write(json.dumps(record, allow_nan=False)+'\n')
                    log.flush()
                    if measured:
                        records.append(record)
                env.step(action)
                # Use this arm's actual successful bytes, including warmup.
                ema = .995*ema+.005*np.asarray(env.info['actual_ue_goodput_bytes'])
                previous = action.copy()
                if tti%1000 == 0:
                    print('NATIVE', arm, seed, tti, flush=True)
        env.close()
    except BaseException:
        if not env.log.closed:
            env.close(validate=False)
        raise
    # Exact native-action / MCS / layer / TBS / dequeue validation is in
    # NativeEnv. Here also verify common exogenous arrivals and warmup.
    import pandas as pd
    reference = REFERENCE/f'seed{seed}_pfq0_20000'
    with np.load(reference/'prefix.npz') as before, np.load(out/'actions.npz') as after:
        np.testing.assert_array_equal(before['input_hashes'][:args.warmup], after['input_hashes'][:args.warmup])
        np.testing.assert_array_equal(before['actions'][:args.warmup].reshape(-1, 3, 17), after['actions'][:args.warmup])
    offered = pd.read_csv(out/'traffic_offered_packets.csv')
    expected = pd.read_csv(reference/'traffic_offered_packets.csv')
    pd.testing.assert_frame_equal(offered[offered.arrival_tti <= env.horizon].reset_index(drop=True),
        expected[expected.arrival_tti <= env.horizon].reset_index(drop=True))
    result = dict(seed=seed, arm=arm, first_tti=first, last_tti=last, search=summarize(records),
                  native_validation=True, common_pfq_prefix_exact=args.warmup,
                  timing='serial native run on shared host; logging is enabled; use replay for paired cost')
    if args.followup >= 400:
        from training.stageb_ppo.evaluate import metrics
        service, pk = packets(out)
        cohort = pk[pk.arrival_tti.between(first, last)]
        cohort.to_csv(out/'cohort.csv.gz', index=False)
        result['packet_metrics'] = metrics(service, pk, first, last, seed, arm)
    else:
        result['packet_metrics'] = None
        result['packet_limit'] = 'Short smoke without mature cohort; no payload/expiry/delay gate evaluated'
    write_json(out/'summary.json', result)
    return result


def replay(output, args):
    """Same-input engineering diagnostic, not a closed-loop warm-arm outcome.

    Warm inputs come from the recorded source arm's t-1 executed action.
    Their provenance is explicit; no t-1 full solution is called a cheap arm's
    own history. Online warm-arm evaluation uses native_one instead.
    """
    import torch
    from .archive import enable_legacy_types
    enable_legacy_types()
    paths = sorted(Path(args.replay).resolve().glob('**/states/tti*.pt'))
    if not paths:
        raise ValueError('No captured states found')
    results = []
    for path in paths:
        row = torch.load(path, map_location='cpu', weights_only=False)
        if row['previous_tti'] != row['tti']-1:
            raise ValueError('Replay predecessor is not consecutive')
        # Warm only immutable library/TBS initialization before paired timing.
        problem = problem_from_raw_row(row)
        draft = raw_pfq_draft(problem, row['obs'])
        solve_ranked(problem, draft, RankedDplsConfig(max_candidates=1))
        for repeat in range(args.repeats):
            order = args.arms if repeat%2 == 0 else list(reversed(args.arms))
            for arm in order:
                config, warm = arm_config(arm)
                started = time.perf_counter()
                p = problem_from_raw_row(row)
                action, values, record = solve_ranked(p, raw_pfq_draft(p, row['obs']), config,
                    warm_start=row['previous_action'] if warm else None)
                record['e2e_ms'] = (time.perf_counter()-started)*1000
                np.testing.assert_allclose(values, ReferencePhy(p).evaluate(action)[0], rtol=1.e-8, atol=1.e-6)
                record.update(arm=arm, repeat=repeat, seed=row['seed'], tti=row['tti'], source=str(path),
                              previous_action_source_arm=row['source_arm'])
                results.append(record)
    with (output/'replay.jsonl').open('x') as log:
        for record in results:
            log.write(json.dumps(record, allow_nan=False)+'\n')
    write_json(output/'input_manifest.json', {str(p): sha(p) for p in paths})
    return dict(decisions=len(results), exact_terminal_audits=len(results),
                limitation='Warm replay uses source-arm history; only a fixed-input cost/quality diagnostic')


def main(args):
    if (any(getattr(args, k) < 0 for k in ('warmup', 'settling', 'followup', 'trace_every', 'save_every'))
            or args.measured < 1 or args.repeats < 1):
        raise ValueError('Invalid run lengths or capture cadence')
    args.arms = args.arms.split(',')
    args.seeds = [int(s) for s in args.seeds.split(',')]
    if len(set(args.arms)) != len(args.arms) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Duplicate arms/seeds')
    for arm in args.arms:
        arm_config(arm)
    if not set(args.seeds) <= {41, 42, 43, 44, 45}:
        raise ValueError('Development capture only: supported native reference seeds are 41--45')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    registry = ROOT/'training/stageb_hgraph/current_algorithm.json'
    registry_sha = sha(registry)
    files = list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.cpp'))
    files += list((ROOT/'training/stageb_ppo').glob('*.py'))
    files += [Path(__file__).with_name('_physics.so')]
    sources = {str(path.relative_to(ROOT)): sha(path) for path in files}
    for name in sources:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    write_json(output/'source_manifest.json', sources)
    write_json(output/'protocol.json', dict(vars(args),
        scope='step0/A controls; no learned component, no promotion',
        objective='canonical native BLER, no CTBS, causal deficit EMA(.995), own raw PFQ ceil90% floor',
        overlap='exact slot, active-slot assignment Jaccard, and activity-only agreement; adjacent executed actions',
        counters='all scored complete candidates incl PFQ/warm initialization; affected UE and MMSE requests/misses separate',
        traces='all candidate exact J/delta, eligibility, compressed actions, chosen action; no extra scoring',
        residual_time='other_ms includes NumPy/C/allocations and Python; not an exclusive Python profile',
        phase_gate='Smoke/capture does not establish unified quality/weak-UE or speed gates',
        registry_sha256=registry_sha))
    try:
        if args.replay:
            completion = replay(output, args)
        else:
            completion = dict(runs=[native_one(output, arm, seed, args) for seed in args.seeds for arm in args.arms])
    except BaseException as error:
        write_json(output/'failure.json', dict(completed=False, error=repr(error)))
        raise
    for name, digest in sources.items():
        if sha(ROOT/name) != digest:
            raise RuntimeError(f'Source changed during run: {name}')
    if sha(registry) != registry_sha:
        raise RuntimeError('Active registry changed during run')
    write_json(output/'completion.json', dict(completion, completed=True, registry_unchanged=True))
    print('COMPLETE', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', default='41,42,43')
    parser.add_argument('--arms', default='cold_full')
    parser.add_argument('--warmup', type=int, default=4000)
    parser.add_argument('--settling', type=int, default=4000)
    parser.add_argument('--measured', type=int, default=4000)
    parser.add_argument('--followup', type=int, default=400)
    parser.add_argument('--trace-every', type=int, default=1)
    parser.add_argument('--save-every', type=int, default=128)
    parser.add_argument('--replay', help='Captured native campaign directory; no simulator launched')
    parser.add_argument('--repeats', type=int, default=3)
    main(parser.parse_args())

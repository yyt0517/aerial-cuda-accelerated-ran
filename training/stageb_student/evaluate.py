"""Paired closed-loop development evaluation, with no hidden fast fallback."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from training.stageb_ppo.native import ROOT, NativeEnv, REFERENCE, packets, sha, write
from training.stageb_ppo.evaluate import metrics
from .catalog import FastCatalog
from .full_catalog import FullCatalog
from .model import ServiceActor

ARMS = ['fast4_greedy','fast8_greedy','fast4_ppo','fast8_ppo','full_shared_ppo']


def evaluate_one(root, arm, seed, cfg):
    root = Path(root).resolve(); torch.set_num_threads(1); torch.backends.mkldnn.enabled = False
    checkpoint = root/'ppo/ppo_final.pt'; actor = ServiceActor()
    actor.load_state_dict(torch.load(checkpoint, weights_only=False)['actor']); actor.eval()
    fast = arm.startswith('fast'); k = int(arm[4]) if fast else 16
    catalog = FastCatalog(root/'ppo/generator.npz', k) if fast else FullCatalog()
    warm, settle, measured, follow = [cfg[f'validation_{x}'] for x in ['warmup','settling','measured','followup']]
    env = NativeEnv(root/f'eval_{arm}_seed{seed}', seed, warm+settle+measured+follow)
    if fast:
        def forbid(*args, **kwargs): raise AssertionError('Full teacher called in fast evaluation')
        env.expert = forbid
    audits = []; times = []
    try:
        for _ in range(env.horizon):
            if env.tti < warm: action = env.raw_pfq()
            else:
                start = time.perf_counter(); row = catalog.build(env)
                greedy = int(np.where(row['mask'], row['prior'], -1.e9).argmax())
                choice = greedy if arm.endswith('greedy') else actor.sample(row, greedy=True)[0]
                action = row['actions'][choice]; elapsed = time.perf_counter()-start
                audits.append(dict(row['audit'], selected=choice, physical_greedy=greedy,
                    differs_from_greedy=choice != greedy, policy_ms=1000*elapsed))
                times.append(elapsed)
            env.step(action)
            if env.tti % 2000 == 0: print('EVAL', arm, seed, env.tti, flush=True)
        env.close()
    except BaseException:
        if not env.log.closed: env.close(validate=False)
        raise
    service, pk = packets(env.out); first, last = warm+settle+1, warm+settle+measured
    cohort = pk[pk.arrival_tti.between(first, last)]; assert not cohort.censored.any()
    cohort.to_csv(env.out/'cohort.csv.gz', index=False)
    pd.DataFrame(audits).to_csv(env.out/'decision_audit.csv', index=False)
    ref = REFERENCE/f'seed{seed}_pfq0_20000'; before = np.load(ref/'prefix.npz'); after = np.load(env.out/'actions.npz')
    np.testing.assert_array_equal(before['input_hashes'][:warm], after['input_hashes'][:warm])
    np.testing.assert_array_equal(before['actions'][:warm].reshape(-1,3,17), after['actions'][:warm])
    offered = pd.read_csv(env.out/'traffic_offered_packets.csv'); expected = pd.read_csv(ref/'traffic_offered_packets.csv')
    pd.testing.assert_frame_equal(offered[offered.arrival_tti <= env.horizon].reset_index(drop=True),
        expected[expected.arrival_tti <= env.horizon].reset_index(drop=True))
    if fast:
        assert all(a['full_search_calls'] == 0 and a['candidate_evaluations'] == k for a in audits)
    assert all(a['ctbs_pruned'] == 0 and a['weights_min'] == a['weights_max'] == 1 for a in audits)
    result = metrics(service, pk, first, last, seed, arm)
    result.update(checkpoint_sha256=sha(checkpoint), generator_sha256=sha(root/'ppo/generator.npz'),
        native_validation=True, common_pfq_prefix_exact=warm,
        full_search_calls=sum(a['full_search_calls'] for a in audits),
        mean_shared_host_policy_ms=1000*float(np.mean(times)), p95_shared_host_policy_ms=1000*float(np.quantile(times,.95)),
        mean_choices=float(np.mean([a['choices'] for a in audits])),
        differs_from_greedy_pct=100*float(np.mean([a['differs_from_greedy'] for a in audits])))
    write(env.out/'evaluation.json', result); return result


def main(root, workers=3):
    root = Path(root).resolve(); cfg = json.loads((root/'protocol.json').read_text())
    sources = json.loads((root/'source_manifest.json').read_text())
    for p, digest in sources.items(): assert sha(ROOT/p) == digest, p
    tasks = [(a,s) for a in ARMS for s in cfg['validation_seeds'] if not (root/f'eval_{a}_seed{s}/evaluation.json').exists()]
    # Expensive full controls start first; cost comparisons use the later solo benchmark.
    tasks.sort(key=lambda x: not x[0].startswith('full'))
    if tasks:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
            futures = [pool.submit(evaluate_one, str(root), a, s, cfg) for a,s in tasks]
            for future in as_completed(futures): print(json.dumps(future.result()), flush=True)
    rows = [json.loads((root/f'eval_{a}_seed{s}/evaluation.json').read_text()) for a in ARMS for s in cfg['validation_seeds']]
    original = ROOT/'deliverables/ppo_diagnosis_continue_20261030_v1/minibatch_continuation/evaluation.csv'
    old = pd.read_csv(original); old = old[old.arm.isin(['ppo','raw_global_equal'])].copy()
    old['arm'] = old.arm.replace({'ppo':'full_original_ppo','raw_global_equal':'equal_dpls'})
    table = pd.concat([pd.DataFrame(rows),old]).sort_values(['seed','arm'])
    table.to_csv(root/'comparison.csv', index=False)
    columns = ['complete_payload_mbps','expiry_pct','success_delay_mean_ms','success_delay_p95_ms','ue_complete_payload_p5_mbps']
    table.groupby('arm')[columns].mean().to_csv(root/'comparison_means.csv')
    gates = []
    for seed in cfg['validation_seeds']:
        subset = table[table.seed == seed].set_index('arm'); original = subset.loc['full_original_ppo']; dpls = subset.loc['equal_dpls']; control = subset.loc['full_shared_ppo']
        for arm in ['fast4_ppo','fast8_ppo']:
            a = subset.loc[arm]
            checks = dict(payload_original_retained=a.complete_payload_mbps >= original.complete_payload_mbps,
                expiry_original_retained=a.expiry_pct <= original.expiry_pct,
                p95_no_worse_than_equal_dpls=a.success_delay_p95_ms <= dpls.success_delay_p95_ms,
                payload_full_shared_retained=a.complete_payload_mbps >= control.complete_payload_mbps,
                expiry_full_shared_retained=a.expiry_pct <= control.expiry_pct,
                no_full_search=a.full_search_calls == 0)
            gates.append(dict(seed=seed, arm=arm, checks={k:bool(v) for k,v in checks.items()},
                strict_quality_passed=bool(all(checks[k] for k in list(checks)[:3]))))
    write(root/'quality_gate.json', dict(rows=gates, promoted=False))
    for p, digest in sources.items(): assert sha(ROOT/p) == digest, p


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True); parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args(); main(args.root, args.workers)

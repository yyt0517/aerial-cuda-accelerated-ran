"""Audit decoder equivalence and time paired, serial, saved-state searches.

Use selection seeds only for cost development. This is a host diagnostic with
the complete search in its timing scope, not a deployment latency benchmark.
"""
import argparse
import cProfile
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import pstats
import time

import numpy as np

from .archive import enable_legacy_types, problem_from_row, initial_draft
from .cached_repair import CachedDeadlineRepair
from .deadline import DeadlineRepair
from .experiment import ROOT, sha, snapshot_sources, write_json
from .quality import select


COUNTERS = ('calls', 'proposals', 'option_scores', 'logit_calls')
CACHE_COUNTERS = ('token_builds', 'context_builds', 'option_builds', 'prefix_initializations')
SEARCH_FIELDS = ('score', 'objective_score', 'floor', 'chosen_dequeue', 'rounds',
                 'accepted_updates', 'candidate_evaluations', 'stop_reason')


def update_digest(digest, values):
    array = np.asarray(values)
    digest.update(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())


def audited(cls):
    class Audited(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.feature_hash = hashlib.sha256()
            self.logit_hash = hashlib.sha256()
            self.proposal_hash = hashlib.sha256()

        def logits(self, x):
            update_digest(self.feature_hash, x)
            result = super().logits(x)
            update_digest(self.logit_hash, result)
            return result

        def propose(self, graph, base, block):
            result = super().propose(graph, base, block)
            update_digest(self.proposal_hash, result)
            return result
    return Audited


def main(args):
    import pandas as pd
    import torch
    enable_legacy_types()
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot_sources(output)
    files = [Path(p).resolve() for p in args.states]
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in files}
    launches = [json.loads(p.with_name('launch.json').read_text()) for p in files]
    if any(launch['seed'] not in (44, 45, 48, 50) for launch in launches):
        raise ValueError('Cost development is restricted to established selection seeds')
    if any(launch['checkpoint_sha256'] != sha(checkpoint) or launch['arm'] != 'risk_neural'
           for launch in launches):
        raise ValueError('Saved-state baseline must use the same frozen risk-neural model')
    protocol = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        source_sha256=sources, input_sha256=input_hashes, repeats=args.repeats,
        width=16, time_cutoff_ms=None, seed_roles='selection states; no quality retuning',
        equivalence='feature/logit/proposal byte digests, final actions/PHY, search work',
        timing='paired alternating serial searches; no profiler or digests in timed calls',
        scope='complete select including PHY setup, cache initialization, graph and verification',
        machine=platform.platform(), numpy=np.__version__, torch=torch.__version__,
        threads={key:os.environ.get(key) for key in
                 ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        active_registry_sha256=sha(ROOT/'training/stageb_hgraph/current_algorithm.json'))
    write_json(output/'protocol.json', protocol)
    audit_rows, timing_rows = [], []
    for path, launch in zip(files, launches):
        states = torch.load(path, map_location='cpu', weights_only=False)
        for row in states:
            if row['tti'] <= launch['warmup']:
                continue
            problem = problem_from_row(row)
            draft = initial_draft(problem, row['obs'])
            models = [audited(cls)(checkpoint, 16) for cls in (DeadlineRepair, CachedDeadlineRepair)]
            results = [select(problem, draft, 'risk_neural', m) for m in models]
            for left, right in zip(results[0][:2], results[1][:2]):
                if left.tobytes() != right.tobytes():
                    raise AssertionError('Final action or physical values changed')
            np.testing.assert_array_equal(results[0][0], row['chosen'])
            for key in SEARCH_FIELDS:
                assert results[0][2][key] == results[1][2][key], key
            check = dict(seed=launch['seed'], tti=row['tti'], source=str(path.relative_to(ROOT)))
            for key in ('feature_hash', 'logit_hash', 'proposal_hash'):
                values = [getattr(m, key).hexdigest() for m in models]
                assert values[0] == values[1], (check, key)
                check[key] = values[0]
            for key in COUNTERS:
                assert getattr(models[0], key) == getattr(models[1], key), key
                check[key] = getattr(models[0], key)
            check.update({key:getattr(models[1], key) for key in CACHE_COUNTERS})
            check.update({key:results[0][2][key] for key in SEARCH_FIELDS})
            check['saved_native_action_exact'] = True
            audit_rows.append(check)
            # Distinct, uninstrumented instances for timing; caches reset on every solve.
            reference, cached = DeadlineRepair(checkpoint, 16), CachedDeadlineRepair(checkpoint, 16)
            for model in (reference, cached):
                select(problem, draft, 'risk_neural', model)
            for repeat in range(args.repeats):
                order = (('reference', reference), ('cached', cached))
                if repeat % 2:
                    order = order[::-1]
                for position, (name, model) in enumerate(order):
                    begin = time.perf_counter()
                    action, values, record = select(problem, draft, 'risk_neural', model)
                    elapsed_ms = (time.perf_counter()-begin)*1000
                    np.testing.assert_array_equal(action, results[0][0])
                    np.testing.assert_array_equal(values, results[0][1])
                    timing_rows.append(dict(seed=launch['seed'], tti=row['tti'], repeat=repeat,
                        position=position, decoder=name, elapsed_ms=elapsed_ms))
            print('AUDITED', launch['seed'], row['tti'], check['calls'],
                  check['token_builds'], check['option_builds'], flush=True)
    assert audit_rows, 'No policy states found'
    pd.DataFrame(audit_rows).to_csv(output/'equivalence.csv', index=False)
    timings = pd.DataFrame(timing_rows)
    timings.to_csv(output/'timings.csv', index=False)
    medians = timings.groupby(['seed', 'tti', 'decoder']).elapsed_ms.median().unstack()
    medians['reduction_pct'] = 100*(1-medians.cached/medians.reference)
    medians.to_csv(output/'paired_state_medians.csv')
    seed_means = medians.groupby('seed')[['reference', 'cached']].mean()
    seed_means['reduction_pct'] = 100*(1-seed_means.cached/seed_means.reference)
    seed_means.to_csv(output/'per_seed_cost.csv')
    # Profiles are separate from paired timing and use one specified audit state.
    for name, cls in (('reference', DeadlineRepair), ('cached', CachedDeadlineRepair)):
        profiler = cProfile.Profile()
        profiler.enable()
        select(problem, draft, 'risk_neural', cls(checkpoint, 16))
        profiler.disable()
        profiler.dump_stats(str(output/(name+'.prof')))
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(30)
        (output/(name+'_profile.txt')).write_text(stream.getvalue())
    for relative, digest in sources.items():
        assert sha(ROOT/relative) == digest, ('Source changed', relative)
    for relative, digest in input_hashes.items():
        assert sha(ROOT/relative) == digest, ('Input changed', relative)
    assert sha(checkpoint) == protocol['checkpoint_sha256']
    assert sha(ROOT/'training/stageb_hgraph/current_algorithm.json') == protocol['active_registry_sha256']
    summary = dict(passed=True, states=len(audit_rows), seeds=sorted(seed_means.index.tolist()),
        all_feature_logit_proposal_bytes_exact=True, all_saved_native_actions_exact=True,
        all_search_work_exact=True,
        reference_ms=float(seed_means.reference.mean()), cached_ms=float(seed_means.cached.mean()),
        reduction_pct=float(100*(1-seed_means.cached.mean()/seed_means.reference.mean())),
        timing_scope=protocol['scope'], profiles_state=dict(seed=launch['seed'], tti=row['tti']),
        packet_delay='Saved-state equivalence alone does not measure closed-loop packet delays')
    write_json(output/'checks.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--states', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('Use at least three alternating timing repetitions')
    main(args)

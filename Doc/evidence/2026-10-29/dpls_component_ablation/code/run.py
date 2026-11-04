"""Use the exact promoted native loop, changing only component callbacks."""
import argparse
import inspect
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

from common import ARMS, OUT, REFERENCE, baseline, path_for, sha, write
from policy import Ablation


def runner_source():
    source = inspect.getsource(baseline.run)
    before = 'if args.arm in ("full", "current"):'
    assert source.count(before) == 1
    return source.replace(before, 'if False:  # No learned modules in component ablation.')


def run(arm, seed, smoke=False):
    destination = path_for(arm, seed, smoke)
    if not smoke:
        assert not ARMS[arm]['reused'], 'Main PFQ/full trajectories are reused, never overwritten.'
    assert not destination.exists(), destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    protocol = OUT / 'protocol.json'
    assert protocol.exists()
    study = Ablation(arm)
    proxy = SimpleNamespace(**vars(baseline.r))
    proxy.replacement = SimpleNamespace(**vars(baseline.r.replacement))
    proxy.replacement.policy = study.policy
    proxy.replacement.AUDIT = study.audit
    proxy.optimized = SimpleNamespace(**vars(baseline.r.optimized))
    proxy.optimized.service_weights = study.weights
    proxy.optimized.select = study.select
    def forbidden(*args, **kwargs):
        raise AssertionError('No learned module may be constructed.')
    proxy.DirectedReuseCoordinator = forbidden
    generated = OUT / 'native_run_generated.py'
    assert generated.read_text() == runner_source()
    for name in ['simulator', 'native_build_manifest.json']:
        target = destination.parent / name
        if not target.exists():
            target.symlink_to(REFERENCE / name)
    def sources():
        result = baseline.sources()
        for name in ['common.py', 'policy.py', 'run.py', 'native_run_generated.py', 'protocol.json']:
            result[str(OUT / name)] = sha(OUT / name)
        return result
    namespace = dict(vars(baseline), OUT=destination.parent, r=proxy, sources=sources)
    exec(compile(generated.read_text(), str(generated), 'exec'), namespace)
    namespace['run'](argparse.Namespace(arm='current', seed=seed, gpu=0, smoke=smoke,
                                      component_arm=arm))
    np.save(destination / 'drafts.npy', np.asarray(study.drafts))
    audit = pd.DataFrame(study.audit)
    decisions = pd.read_csv(destination / 'decisions.csv')
    np.testing.assert_array_equal(decisions.tti, audit.tti)
    np.testing.assert_array_equal(decisions.draft_dequeue, audit.draft_dequeue)
    np.testing.assert_array_equal(decisions.keep_dequeue, audit.causal_floor_bytes)
    assert (decisions.chosen_dequeue >= decisions.keep_dequeue).all()
    assert decisions.converged.all() and not decisions.cap_reached.any()
    write(destination / 'component_validation.json', dict(passed=True, arm=arm, seed=seed,
        config=ARMS[arm], decisions=len(decisions), causal_floor_all_decisions_exact=True,
        no_bc_or_coordinator=True, full_convergence=True,
        ctbs_raw_dequeue_difference_decisions=int(audit.draft_dequeue.ne(audit.raw_pfq_dequeue).sum())))
    check = __import__('json').loads((destination / 'validation.json').read_text())
    check['full_search_decisions'] = len(decisions) if ARMS[arm]['search'] != 'none' else 0
    check['component_arm'] = arm
    write(destination / 'validation.json', check)
    print('COMPONENT_NATIVE_VALIDATED', arm, seed, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int, choices=range(41, 46), required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.arm, args.seed, args.smoke)

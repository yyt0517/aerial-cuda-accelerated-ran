"""Active rho=0.90 policy under the existing continuous-channel five-seed protocol."""
import argparse
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'

import numpy as np
import pandas as pd

from output.pfq_independent_reuse.formal.boundary08_steady_state_20261021_v1 import run as baseline
from output.pfq_independent_reuse.formal.boundary08_causal_floor_ratio_seed41_20261027_v1 import policy

OUT=Path(__file__).resolve().parent
REFERENCE=baseline.OUT
REPO=OUT.parents[5]
REGISTRY=REPO/'training/stageb_hgraph/current_algorithm.json'


def runner_source():
    source=inspect.getsource(baseline.run)
    old='if args.arm in ("full", "current"):'
    assert source.count(old)==1
    return source.replace(old,'if False:  # Active rho090 path has no BC/coordinator model.')


def run(seed,gpu=0,smoke=False):
    registry=json.loads(REGISTRY.read_text())
    assert registry['floor']['numerator']==90 and registry['floor']['denominator']==100
    assert registry['runtime']['module']==__package__+'.run'
    assert not registry['directed_coordinator_required'] and registry['runtime']['continuous_channel_time']
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert seed in protocol['seeds'] and protocol['floor_ratio_num']==90
    policy.RATIO_NUM=90
    policy.AUDIT.clear();policy.DRAFTS.clear()
    generated=OUT/'native_run_generated.py'
    assert generated.read_text()==runner_source()
    proxy=SimpleNamespace(**vars(baseline.r))
    proxy.replacement=SimpleNamespace(**vars(baseline.r.replacement))
    proxy.replacement.policy=policy.policy
    proxy.replacement.AUDIT=policy.AUDIT
    proxy.optimized=SimpleNamespace(**vars(baseline.r.optimized))
    proxy.optimized.select=policy.select

    def forbidden(*args,**kwargs):
        raise AssertionError('BC/coordinator construction is forbidden in active rho090 path')

    proxy.DirectedReuseCoordinator=forbidden

    def sources():
        files=baseline.sources()
        for path in [Path(__file__),generated,Path(policy.__file__),Path(policy.__file__).with_name('search_ratio.py'),
                     REGISTRY,OUT/'protocol.json']:
            files[str(path)]=baseline.r.sha(path)
        return files

    namespace=dict(vars(baseline),OUT=OUT,r=proxy,sources=sources)
    exec(compile(runner_source(),str(generated),'exec'),namespace)
    namespace['run'](argparse.Namespace(arm='current',seed=seed,gpu=gpu,smoke=smoke,floor_ratio_num=90))
    directory=OUT/(f'smoke_seed{seed}_current_136' if smoke else f'seed{seed}_current_20000')
    decisions=pd.read_csv(directory/'decisions.csv')
    audit=pd.DataFrame(policy.AUDIT)
    np.save(directory/'drafts.npy',np.asarray(policy.DRAFTS))
    np.testing.assert_array_equal(decisions.tti,audit.tti)
    np.testing.assert_array_equal(decisions.draft_dequeue,audit.draft_dequeue_independent)
    np.testing.assert_array_equal(decisions.keep_dequeue,audit.causal_floor_bytes)
    np.testing.assert_array_equal(decisions.keep_dequeue,(90*decisions.draft_dequeue.astype(np.int64)+99)//100)
    assert (decisions.chosen_dequeue>=decisions.keep_dequeue).all()
    assert (decisions.keep_dequeue<=decisions.draft_dequeue).all()
    assert decisions.converged.all() and not decisions.cap_reached.any()
    assert decisions.coordinator_forward_calls.eq(0).all() and decisions.external_floor_reads.eq(0).all()
    baseline.r.write(directory/'causal_floor_validation.json',dict(
        passed=True,algorithm_id=registry['algorithm_id'],floor_ratio_num=90,denominator=100,
        decisions=len(decisions),independent_current_draft_floor_exact=True,initial_draft_always_feasible=True,
        bc_model_loads=0,coordinator_model_loads=0,coordinator_forward_calls=0,directed_feature_calls=0,
        external_floor_reads=0,all_searches_converged=True,continuous_channel_simulator=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=range(41,46),required=True)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    run(args.seed,args.gpu,args.smoke)

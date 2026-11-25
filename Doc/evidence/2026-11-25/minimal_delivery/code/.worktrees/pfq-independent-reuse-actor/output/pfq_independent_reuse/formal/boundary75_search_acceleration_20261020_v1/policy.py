"""Stage-3 search entrypoint; frozen selection logic, optimized computation.

Construct OptimizedPhy(config) (stage=2, threads=1), then call
phy.prepare(snapshot, ctbs), service_weights(ema_success_bytes, target_bytes),
and select(draft, keep, phy, obs, ctbs, weights). Preparation remains per TTI.
Candidate legality is prepared once per select call and reused across rounds.
"""
from types import FunctionType, SimpleNamespace
from output.pfq_independent_reuse.formal.boundary75_teacher_weight_20261014_v1 import policy as _frozen
from .phy import OptimizedPhy
from .candidates import prepare_candidates

service_weights = _frozen.service_weights
_select = _frozen.select
_globals = dict(_select.__globals__)
_evaluate = _frozen.two.evaluate
_native_bler = _frozen.two.native_bler


def select(draft, keep, phy, obs, ctbs, weights):
    namespace = dict(_globals)
    namespace['two'] = SimpleNamespace(
        candidates=prepare_candidates(obs, phy.snapshot['queue']),
        evaluate=_evaluate, native_bler=_native_bler)
    # Reuse the exact frozen code object without changing shared module globals.
    search = FunctionType(_select.__code__, namespace, _select.__name__,
                          _select.__defaults__, _select.__closure__)
    return search(draft, keep, phy, obs, ctbs, weights)

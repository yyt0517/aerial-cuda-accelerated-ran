"""Immutable, native closed-loop component ablation of the promoted DPLS."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
ROOT = REPO / '.worktrees/pfq-independent-reuse-actor'
sys.path.insert(0, str(ROOT))
FORMAL = ROOT / 'output/pfq_independent_reuse/formal'
REFERENCE = FORMAL / 'boundary08_steady_state_20261021_v1'
PROMOTED = FORMAL / 'boundary08_rho090_five_seed_20261027_v1'
from output.pfq_independent_reuse.formal.boundary08_steady_state_20261021_v1 import run as baseline
from output.pfq_independent_reuse.formal.boundary75_search_acceleration_20261020_v1 import policy as optimized
from output.pfq_independent_reuse.formal.boundary75_search_acceleration_20261020_v1.candidates import prepare_candidates

ARMS = {
    'pfq': dict(ctbs=False, search='none', deficit=False, reused=True),
    'ctbs': dict(ctbs=True, search='none', deficit=False, reused=False),
    'raw_global_equal': dict(ctbs=False, search='global', deficit=False, reused=False),
    'raw_global_deficit': dict(ctbs=False, search='global', deficit=True, reused=False),
    'ctbs_global_equal': dict(ctbs=True, search='global', deficit=False, reused=False),
    'full': dict(ctbs=True, search='global', deficit=True, reused=True),
    'ctbs_independent_equal': dict(ctbs=True, search='independent', deficit=False, reused=False),
    'ctbs_independent_deficit': dict(ctbs=True, search='independent', deficit=True, reused=False),
}
SEEDS = list(range(41, 46))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def path_for(arm, seed, smoke=False):
    if not smoke and ARMS[arm]['reused']:
        root, label = (REFERENCE, 'pfq0') if arm == 'pfq' else (PROMOTED, 'current')
    else:
        root, label = OUT / ('smoke' if smoke else 'native') / arm, 'current'
    return root / (f'smoke_seed{seed}_{label}_136' if smoke else f'seed{seed}_{label}_20000')

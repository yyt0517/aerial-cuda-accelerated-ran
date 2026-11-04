"""Read existing causal snapshots without importing or modifying frozen runners."""
from functools import lru_cache
from pathlib import Path
import numpy as np

from .physics import Problem, ROOT

ARCHIVE = ROOT / ".worktrees/pfq-independent-reuse-actor"
FORMAL = ARCHIVE / "output/pfq_independent_reuse/formal"
CURRENT = FORMAL / "boundary08_rho090_five_seed_20261027_v1"


def enable_legacy_types():
    # Replay runs in its own process. Pickled CTBS types and their dependencies
    # must all come from the same archive, not the older main-tree bridge types.
    import training.stageb_hgraph
    path = str(ARCHIVE / "training/stageb_hgraph")
    if path not in training.stageb_hgraph.__path__:
        training.stageb_hgraph.__path__.insert(0, path)


def load_rows(seeds=(41, 42, 43, 44, 45)):
    enable_legacy_types()
    import torch
    rows = []
    for seed in seeds:
        path = CURRENT / f"seed{seed}_current_20000/states.pt"
        # Repository-owned archived artifacts, never arbitrary external pickle input.
        saved = torch.load(path, map_location="cpu", weights_only=False)
        rows.extend(dict(row, seed=seed, source=str(path)) for row in saved)
    return rows


@lru_cache(maxsize=8)
def tbs_table(prbs_per_prg, data_symbols, n_prg):
    enable_legacy_types()
    from training.stageb_hgraph.phy_tbs_alignment_audit import determine_tbs_bytes, MCS_CODE_RATES, MCS_QAM_ORDERS
    result = np.zeros((28, n_prg+1, 5), dtype=np.int64)
    for m in range(28):
        for n in range(1, n_prg+1):
            for layers in range(1, 5):
                result[m, n, layers] = determine_tbs_bytes(n*prbs_per_prg, data_symbols,
                    layers, MCS_CODE_RATES[m]/1024., MCS_QAM_ORDERS[m])
    result.flags.writeable = False
    return result


def problem_from_row(row):
    from .deadline import causal_features
    ctbs, snapshot, obs = row["ctbs"], row["snapshot"], row["obs"]
    if snapshot["tti"] != ctbs.tti:
        raise ValueError("Stale CSI/CTBS pair")
    h = snapshot["h"]
    n_ue, n_prg, n_cell = h.shape[:3]
    mcs = np.array([ctbs.reconstructed_mcs_for_ue(u) for u in range(n_ue)])
    tbs = tbs_table(ctbs.config.prbs_per_prg, ctbs.config.pdsch_data_symbols, n_prg)[mcs]
    weights = np.clip(2400./np.maximum(row["ema"], 1.e-9), 1., 4.)
    weights /= weights.mean()
    legal = (np.asarray(obs["action_mask_cell_ue"], bool)[:, None, :]
             & np.asarray(obs["action_mask_prg_cell"], bool)[:, :, None]
             & np.asarray(obs["action_mask_ue"], bool)[None, None, :]
             & (snapshot["queue"][None, None, :] > 0))
    return Problem(h, snapshot["w"], snapshot["noise"], snapshot["queue"],
                   np.array(ctbs.serving_cell), mcs, np.array(ctbs.prg_layer_cap), tbs, weights, legal,
                   sinr_hint=obs.get("post_eq_sinr"), deadline=causal_features(obs, row["ema"]))


def initial_draft(problem, obs):
    """Native PFQ slot decoding and the unchanged ascending-prefix CTBS prune."""
    slots = np.asarray(obs["teacher_action_ue_select"], dtype=np.int64)
    raw = np.asarray(obs["teacher_action_prg_alloc"], dtype=np.int64).reshape(problem.P, problem.C).T
    if ((raw < -1) | (raw >= len(slots))).any():
        raise ValueError("Invalid native PFQ slot")
    action = np.full(raw.shape, -1, dtype=np.int64)
    action[raw >= 0] = slots[raw[raw >= 0]]
    problem.actions(action)
    count = np.zeros(problem.U, dtype=int)
    layers = np.full(problem.U, 4, dtype=int)
    for p in range(problem.P):
        for c in range(problem.C):
            u = action[c, p]
            if u < 0:
                continue
            if count[u] and problem.tbs[u, count[u], layers[u]] >= problem.queue[u]:
                action[c, p] = -1
            else:
                count[u] += 1
                layers[u] = min(layers[u], problem.caps[u, p])
    return action

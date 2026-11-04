"""Raw-PFQ inputs for the CTBS-free candidate-ranking experiment.

No CTBS scheduler, demand headroom state, action pruning, or reconstructed-MCS
method is called. Old archives transport causal link-adaptation metadata in a
field named ``ctbs``. ``detach_raw_phy_row`` extracts only those scalar/array
diagnostics into a neutral ``raw_phy`` mapping and removes the old field before
any timed decision. MCS is then reconstructed with the ordinary native CPU
OLLA/MCS rule inside ``problem_from_raw_row``. This is link adaptation, not CTBS.

The UE/cell demand masks are deliberately not used to restrict positive-queue
UEs. Association comes from physical metadata; the native PRG availability mask
is retained (ReplayWriter::buildActionMask reads it from cellGrpPrms.prgMsk).
Only the table's immutable first construction may be warmed outside timing.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from .archive import enable_legacy_types, tbs_table
from .physics import Problem, DEQUEUE


CONFIG_FIELDS = (
    'mcs_lut_type', 'mcs_sinr_cap_db', 'layer_sinval_threshold',
    'prbs_per_prg', 'pdsch_data_symbols', 'effective_layer_cap',
)
MCS_INPUT_FIELDS = (
    'wb_sinr_first_layer_db', 'olla_delta', 'olla_delta_up',
    'olla_delta_down', 'prior_tb_err',
)


def detach_raw_phy_row(row):
    """Unpack only raw, pre-action PHY metadata; preserve row identity fields.

    The returned row has no ``ctbs`` key. Numeric MCS reconstruction and TBS
    indexing are intentionally deferred to the timed input adapter. Actual
    chosen actions, native post-action labels, and CRC outcomes are never read.
    New capture paths can supply ``raw_phy`` directly and need no legacy field.
    """
    result = {key: value for key, value in row.items() if key != 'ctbs'}
    if 'raw_phy' in row:
        result['raw_phy'] = dict(row['raw_phy'])
        return result
    if 'ctbs' not in row:
        raise ValueError('Missing causal raw_phy metadata; detach the legacy archive before replay')
    carrier = row['ctbs']
    config = {name: getattr(carrier.config, name) for name in CONFIG_FIELDS}
    raw = dict(tti=int(carrier.tti), snapshot_id=int(carrier.snapshot_id), config=config,
               serving_cell=np.array(carrier.serving_cell, dtype=np.int64, copy=True),
               prg_layer_cap=np.array(carrier.prg_layer_cap, dtype=np.int64, copy=True),
               source='legacy_pre_action_diagnostics_only')
    for name in MCS_INPUT_FIELDS:
        raw[name] = np.array(getattr(carrier, name), copy=True)
    result['raw_phy'] = raw
    return result


def _integers(value, shape, name):
    values = np.asarray(value)
    if values.shape != shape or not np.isfinite(values).all() or np.any(values != np.floor(values)):
        raise ValueError(f'Invalid {name}: expected finite integers of shape {shape}')
    return values.astype(np.int64, copy=False)


def problem_from_raw_row(row):
    """Build current J inputs without consulting any CTBS/teacher action state.

    ``raw_phy`` may contain a fresh, already selected ``mcs`` vector. Otherwise
    it must provide the pre-action WB-SINR/CPU-OLLA diagnostics used by native
    mcsSelSol. Positive queue is the only per-UE demand eligibility condition.
    The original causal deficit weights are retained; deadline shaping is off.
    """
    snapshot, obs = row['snapshot'], row['obs']
    raw = row['raw_phy']
    if not isinstance(raw, Mapping):
        raise ValueError('raw_phy must be a metadata mapping')
    tti = int(snapshot['tti'])
    if int(raw['tti']) != tti:
        raise ValueError('Stale raw PHY / snapshot TTI')
    snapshot_id = int(raw.get('snapshot_id', 1))
    if snapshot_id <= 0:
        raise ValueError('Invalid pre-action snapshot identity')
    # Older current-state archives label the pre-action clock itself. Native
    # decision archives label the upcoming TTI: row_tti == snapshot_id == tti+1.
    if 'tti' in row:
        row_tti = int(row['tti'])
        upcoming = row_tti == tti+1 and row_tti == snapshot_id
        if row_tti != tti and not upcoming:
            raise ValueError('Stale row decision TTI / pre-action snapshot identity')
    h = np.asarray(snapshot['h'])
    if h.ndim != 5 or h.shape[-2:] != (4, 4):
        raise ValueError('Expected the current 4T4R channel tensor')
    n_ue, n_prg, n_cell = h.shape[:3]
    queue = np.asarray(snapshot['queue'], dtype=np.float64)
    ue_features = np.asarray(obs['obs_ue_features'])
    if ue_features.ndim != 2 or ue_features.shape[0] != n_ue or ue_features.shape[1] < 1:
        raise ValueError('Invalid fresh UE observation')
    if not np.array_equal(ue_features[:, 0], queue):
        raise ValueError('Fresh observation queue does not match the causal snapshot')
    serving = _integers(raw['serving_cell'], (n_ue,), 'serving_cell')
    caps = _integers(raw['prg_layer_cap'], (n_ue, n_prg), 'prg_layer_cap')
    config = raw['config']
    config_values = [config[name] if isinstance(config, Mapping) else getattr(config, name)
                     for name in CONFIG_FIELDS]
    enable_legacy_types()
    from training.stageb_hgraph.phy_tbs_alignment_audit import PhyTbsConfig, reconstruct_mcs
    config = PhyTbsConfig.from_vector(config_values)
    if 'mcs' in raw:
        mcs = _integers(raw['mcs'], (n_ue,), 'mcs')
    else:
        vectors = {name: np.asarray(raw[name]) for name in MCS_INPUT_FIELDS}
        if any(v.shape != (n_ue,) or not np.isfinite(v).all() for v in vectors.values()):
            raise ValueError('Invalid causal native MCS diagnostic vectors')
        errors = _integers(vectors['prior_tb_err'], (n_ue,), 'prior_tb_err')
        mcs = np.array([reconstruct_mcs(
            wb_sinr_first_layer_db=float(vectors['wb_sinr_first_layer_db'][u]),
            olla_delta=float(vectors['olla_delta'][u]),
            olla_delta_up=float(vectors['olla_delta_up'][u]),
            olla_delta_down=float(vectors['olla_delta_down'][u]),
            prior_tb_err=int(errors[u]), config=config) for u in range(n_ue)], dtype=np.int64)
    if ((mcs < 0) | (mcs > 27)).any():
        raise ValueError('MCS outside native range')
    table = tbs_table(config.prbs_per_prg, config.pdsch_data_symbols, n_prg)[mcs]
    ema = np.asarray(row['ema'], dtype=np.float64)
    if ema.shape != (n_ue,) or not np.isfinite(ema).all() or (ema < 0).any():
        raise ValueError('Invalid causal successful-service EMA')
    weights = np.clip(2400./np.maximum(ema, 1.e-9), 1., 4.)
    weights /= weights.mean()
    prg_available = np.asarray(obs.get('action_mask_prg_cell', np.ones((n_cell, n_prg))), bool)
    if prg_available.shape != (n_cell, n_prg):
        raise ValueError('Invalid physical PRG availability shape')
    # Ignore action_mask_ue and action_mask_cell_ue values: they must not carry
    # CTBS/remaining-demand restrictions into this experiment's search domain.
    association = np.arange(n_cell)[:, None, None] == serving[None, None, :]
    legal = association & prg_available[:, :, None] & (queue[None, None, :] > 0)
    return Problem(h, snapshot['w'], snapshot['noise'], queue, serving, mcs, caps,
                   table, weights, legal, sinr_hint=obs.get('post_eq_sinr'), deadline=None)


def raw_pfq_draft(problem, obs):
    """Decode native PFQ slot indices, retaining every legal allocated PRG.

    No queue/TBS comparison, ascending-prefix pruning, or headroom condition is
    applied. TBS/dequeue saturation remains exclusively in the physical model.
    """
    slots = _integers(np.asarray(obs['teacher_action_ue_select']),
                      np.asarray(obs['teacher_action_ue_select']).shape, 'PFQ UE slots')
    if slots.ndim != 1 or ((slots < -1) | (slots >= problem.U)).any():
        raise ValueError('Invalid PFQ UE slot mapping')
    raw_values = np.asarray(obs['teacher_action_prg_alloc'])
    raw = _integers(raw_values, (problem.C*problem.P,), 'PFQ PRG slots').reshape(problem.P, problem.C).T
    if ((raw < -1) | (raw >= len(slots))).any():
        raise ValueError('PFQ allocation references an invalid slot')
    action = np.full((problem.C, problem.P), -1, np.int64)
    active = raw >= 0
    action[active] = slots[raw[active]]
    if (action[active] < 0).any():
        raise ValueError('PFQ allocation references an empty UE slot')
    problem.actions(action)
    return action


def raw_pfq_floor(values):
    """Ceil(90% of this state's raw PFQ dequeue); no other trajectory is read."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] <= DEQUEUE:
        raise ValueError('Expected one complete PFQ PHY evaluation')
    dequeue = values[:, DEQUEUE].sum()
    if not np.isfinite(dequeue) or dequeue < 0 or dequeue != int(dequeue):
        raise ValueError('PFQ dequeue must be nonnegative integral bytes')
    return (90*int(dequeue)+99)//100

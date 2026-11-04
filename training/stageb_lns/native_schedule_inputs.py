"""Native MCS/domain/PFQ preparation, retaining NumPy PHY preprocessing bits."""
import ctypes
from functools import lru_cache
import numpy as np
from .physics import Problem
from .archive import enable_legacy_types,tbs_table
from .tournament_dpls import tournament_library,solve_tournament,TournamentConfig

class PreparedProblem:
    actions=Problem.actions

@lru_cache(maxsize=1)
def prepare_function():
    lib=tournament_library()
    i=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    d=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    u=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
    lib.ts_prepare.argtypes=[ctypes.c_int]*4+[ctypes.c_double,d,d,i,ctypes.c_int,i,i,d,i,u,i,i,i,i,u,i]
    lib.ts_prepare.restype=ctypes.c_int
    return lib.ts_prepare

@lru_cache(maxsize=2)
def mcs_lut(kind):
    enable_legacy_types()
    from training.stageb_hgraph.phy_tbs_alignment_audit import _MCS_LUT_0,_MCS_LUT_1
    if kind not in (0,1):raise ValueError('Unsupported MCS LUT')
    return np.array(_MCS_LUT_0 if kind==0 else _MCS_LUT_1,dtype=np.float32).astype(np.float64)

def integers(a):
    a=np.asarray(a)
    if a.dtype.kind not in ('i','u') and (not np.isfinite(a).all() or not np.array_equal(a,np.floor(a))):
        raise ValueError('Integer metadata required')
    return np.ascontiguousarray(a,np.int64)

def prepare_native_problem(row,*,weights=None):
    snapshot,obs,raw=row['snapshot'],row['obs'],row['raw_phy']
    clock=int(snapshot['tti']);sid=int(raw.get('snapshot_id',1))
    if int(raw['tti'])!=clock or sid<=0:raise ValueError('Stale PHY metadata')
    if 'tti' in row and int(row['tti'])!=clock and not(int(row['tti'])==clock+1==sid):raise ValueError('Stale decision')
    p=PreparedProblem();p.h=np.ascontiguousarray(snapshot['h'],np.complex128)
    if p.h.ndim!=5 or p.h.shape[-2:]!=(4,4):raise ValueError('4T4R inputs required')
    p.U,p.P,p.C=p.h.shape[:3];p.w=np.ascontiguousarray(snapshot['w'],np.complex128)
    p.noise=float(snapshot['noise']);p.queue=np.ascontiguousarray(snapshot['queue'],np.float64)
    if p.w.shape!=(p.U,p.P,4,4) or p.queue.shape!=(p.U,) or not np.isfinite(p.h).all() or not np.isfinite(p.w).all():raise ValueError('Invalid PHY shape/values')
    if not np.isfinite(p.noise) or p.noise<=0:raise ValueError('Invalid noise')
    if not np.array_equal(np.asarray(obs['obs_ue_features'])[:,0],p.queue):raise ValueError('Stale queue observation')
    p.serving=integers(raw['serving_cell']);p.caps=integers(raw['prg_layer_cap'])
    if p.serving.shape!=(p.U,) or p.caps.shape!=(p.U,p.P):raise ValueError('Invalid metadata dimensions')
    cfg=raw['config'];get=lambda key:cfg[key] if isinstance(cfg,dict) else getattr(cfg,key)
    table=tbs_table(get('prbs_per_prg'),get('pdsch_data_symbols'),p.P)
    provided='mcs' in raw;supplied=integers(raw['mcs']) if provided else np.zeros(p.U,np.int64)
    if supplied.shape!=(p.U,):raise ValueError('Invalid supplied MCS')
    fields=('wb_sinr_first_layer_db','olla_delta','olla_delta_up','olla_delta_down','prior_tb_err')
    metadata=np.zeros((5,p.U),np.float64) if provided else np.ascontiguousarray([raw[k] for k in fields],np.float64)
    if metadata.shape!=(5,p.U):raise ValueError('Invalid MCS metadata shape')
    available=np.ascontiguousarray(obs.get('action_mask_prg_cell',np.ones((p.C,p.P))),np.uint8)
    slots=integers(obs['teacher_action_ue_select']);allocation=integers(obs['teacher_action_prg_alloc'])
    if available.shape!=(p.C,p.P) or slots.ndim!=1 or allocation.shape!=(p.C*p.P,):raise ValueError('Invalid PFQ shape')
    p.mcs=np.empty(p.U,np.int64);p.tbs=np.empty((p.U,p.P+1,5),np.int64)
    p.legal=np.empty((p.C,p.P,p.U),np.uint8);draft=np.empty((p.C,p.P),np.int64)
    result=prepare_function()(p.U,p.C,p.P,len(slots),get('mcs_sinr_cap_db'),metadata,mcs_lut(int(get('mcs_lut_type'))),
        supplied,int(provided),p.serving,p.caps,p.queue,table,available,slots,allocation,p.mcs,p.tbs,p.legal,draft)
    if result:raise ValueError(tournament_library().ts_error().decode())
    p.legal=p.legal.view(np.bool_)
    ema=np.asarray(row['ema'],np.float64)
    if ema.shape!=(p.U,) or not np.isfinite(ema).all() or (ema<0).any():raise ValueError('Invalid EMA')
    if weights is None:
        p.weights=np.clip(2400./np.maximum(ema,1.e-9),1.,4.);p.weights/=p.weights.mean()
    else:p.weights=np.ascontiguousarray(weights,np.float64)
    p.deadline=None;p.sinr_hint=obs.get('post_eq_sinr')
    return p,draft

def solve_native_row(row,config=TournamentConfig(direct_gamma=True),*,warm_start=None,weights=None,observer=None):
    p,draft=prepare_native_problem(row,weights=weights)
    return solve_tournament(p,draft,config,warm_start=warm_start,observer=observer)

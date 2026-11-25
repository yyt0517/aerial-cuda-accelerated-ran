"""Exact sparse capacity-bound DPLS, independent versioned native library."""
import ctypes
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import time
import weakref
import numpy as np
from training.stageb_lns.physics import NativePhy,curves,EXPECTED
from .contract import check_contract
from training.stageb_lns.ranked_dpls import repair_warm_start

I=ctypes.POINTER(ctypes.c_int64);D=ctypes.POINTER(ctypes.c_double)
CALLBACK=ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_void_p,I,I,D,I,I,I,I,I,I)

@lru_cache(maxsize=1)
def bounded_library():
    lib=ctypes.CDLL(str(Path(__file__).with_name('_bounded128.so')))
    i=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    d=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    u=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
    lib.lns_create.argtypes=[ctypes.c_int]*3+[ctypes.c_double]+[i]*4+[d]*3+[i,d,d]
    lib.lns_create.restype=ctypes.c_void_p
    lib.lns_destroy.argtypes=[ctypes.c_void_p];lib.lns_destroy.restype=None
    lib.lns_evaluate.argtypes=[ctypes.c_void_p,ctypes.c_int,i,ctypes.c_void_p,ctypes.c_void_p,d]
    lib.lns_evaluate.restype=ctypes.c_int
    lib.lns_stats.argtypes=[ctypes.c_void_p,i];lib.lns_stats.restype=None
    lib.lb_solve.argtypes=[ctypes.c_void_p,u,d,i,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,
        i,d,i,i,d,ctypes.c_void_p,ctypes.c_void_p]
    lib.lb_solve.restype=ctypes.c_int;lib.lb_error.restype=ctypes.c_char_p
    return lib

class BoundedPhy(NativePhy):
    def __init__(self,p):
        self.problem=p;self.profile=False;self.lib=bounded_library()
        self.covariance=np.ascontiguousarray(p.h@p.h.conj().swapaxes(-1,-2))
        self.signal=np.ascontiguousarray(p.h[np.arange(p.U),:,p.serving]@p.w)
        offsets=np.r_[0,np.cumsum([len(x) for x,_ in curves()])].astype(np.int64)
        xs=np.concatenate([x for x,_ in curves()]);ys=np.concatenate([y for _,y in curves()])
        self.context=self.lib.lns_create(p.U,p.C,p.P,p.noise,p.serving,p.mcs,p.caps,p.tbs,p.queue,
            self.covariance.view(np.float64),self.signal.view(np.float64),offsets,xs,ys)
        if not self.context:raise RuntimeError('Could not allocate bounded PHY')
        self._release=weakref.finalize(self,self.lib.lns_destroy,self.context)

@dataclass(frozen=True)
class BoundedConfig:
    max_rounds:int=256
    diagnostics:bool=False
    def __post_init__(self):
        if self.max_rounds<1:raise ValueError('Invalid round cap')

def solve_bounded(p,raw_pfq,config=BoundedConfig(),*,warm_start=None,observer=None):
    start=time.perf_counter();check_contract(p)
    raw=p.actions(raw_pfq)[0];warm=repair_warm_start(p,warm_start) if warm_start is not None else None
    legal=np.ascontiguousarray(p.legal,np.uint8);phy=BoundedPhy(p)
    action=np.empty((p.C,p.P),np.int64);values=np.empty((p.U,8),np.float64)
    stats=np.empty(18,np.int64);more=np.empty(8,np.int64);times=np.empty(8,np.float64)
    failures=[];callback=None
    if observer is not None:
        def observe(_,m_ptr,a_ptr,v_ptr,k_ptr,u_ptr,g_ptr,d_ptr,s_ptr,selected_ptr):
            try:
                m=np.ctypeslib.as_array(m_ptr,(16,));n=int(m[1]);chosen=int(m[5])
                def arr(ptr,shape,dtype=np.int64):
                    a=np.ctypeslib.as_array(ptr,shape) if np.prod(shape) else np.empty(shape,dtype)
                    a.flags.writeable=False;return a
                observer(dict(iteration=int(m[0]),root=arr(a_ptr,(p.C*p.P,)).reshape(p.C,p.P),
                    root_values=arr(v_ptr,(p.U*8,),np.float64).reshape(p.U,8),keys=arr(k_ptr,(n,)),
                    upper=arr(u_ptr,(n,)),delta_q=arr(g_ptr,(n,)),delta_d=arr(d_ptr,(n,)),
                    status=arr(s_ptr,(n,)),selected_keys=arr(selected_ptr,(chosen,)),
                    score_q=int(m[2]),dequeue=int(m[3]),floor=int(m[4]),scored=int(m[6]),
                    known_before=int(m[7]),threshold=int(m[11]),maximum_known=int(m[12])))
                return 0
            except BaseException as e:failures.append(e);return 1
        callback=CALLBACK(observe)
    try:
        code=phy.lib.lb_solve(phy.context,legal,p.weights,raw,
            None if warm is None else ctypes.c_void_p(warm.ctypes.data),config.max_rounds,int(config.diagnostics),
            action,values,stats,more,times,None if callback is None else ctypes.cast(callback,ctypes.c_void_p),None)
        if failures:raise failures[0]
        if code:raise RuntimeError(phy.lib.lb_error().decode())
        record=dict(version='capacity_bound_q1e6_v1',rounds=int(stats[0]),accepted_moves=int(stats[1]),
            generated_edits=int(stats[2]),regenerated_prgs=int(stats[3]),scored_edits=int(stats[4]),
            sum_neighborhood_sizes=int(stats[5]),first_scan_evaluations=int(stats[8]),
            initialization_evaluations=int(stats[9]),converged=bool(stats[10]),score_q=int(stats[11]),
            floor=int(stats[12]),chosen_dequeue=int(stats[13]),bound_updates=int(more[0]),heap_pushes=int(more[1]),
            certificate_heap_pruned=int(more[2]),known_exact_reuses=int(more[3]),floor_pruned=int(more[4]),
            threshold_pruned=int(more[5]),score=float(values[:,EXPECTED]@p.weights),**phy.stats)
        if config.diagnostics:record.update(native_setup_ms=float(times[0]),table_generation_ms=float(times[1]),
            sparse_phy_ms=float(times[2]),heap_certificate_ms=float(times[3]),bound_update_ms=float(times[4]),
            first_scan_ms=float(times[6]))
    finally:phy.close()
    record['core_ms']=(time.perf_counter()-start)*1000
    return action,values,record

def certificate_sets(info,reference):
    """Offline oracle audit; never called by the timed/online solver.

    This is the required set for the frozen maximum-based certificate, not an
    information-theoretic lower bound over all possible key-aware proofs.
    Current exact cache hits are excluded from new evaluation requirements.
    """
    keys=info['keys'];eligible=info['dequeue']+info['delta_d']>=info['floor'];threshold=info['threshold']
    before=(info['status']&4)!=0;fresh=(info['status']&2)!=0
    gain=reference['delta_q'];pool=np.flatnonzero(eligible&(gain>threshold))
    if len(pool):
        maximum=int(gain[pool].max())
        tied=pool[gain[pool]>=maximum-1];winner=int(tied[np.argmin(keys[tied])])
        required=(~before)&eligible&(info['upper']>threshold)&((info['upper']>=maximum)|
            ((info['upper']==maximum-1)&(keys<keys[winner])))
        if not before[winner]:required[winner]=True
    else:required=(~before)&eligible&(info['upper']>threshold)
    return required,fresh

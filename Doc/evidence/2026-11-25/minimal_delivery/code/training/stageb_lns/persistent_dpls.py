"""Sparse persistent edit-table solver and independent integer full baseline."""
from dataclasses import dataclass
import ctypes
from functools import lru_cache
from pathlib import Path
import time
import weakref
import numpy as np
from .physics import NativePhy,Evaluation,curves,EXPECTED,DEQUEUE
from .dpls_operations import compact_neighborhood,expand_edits,edit_metadata,canonical_choice
from .ranked_dpls import repair_warm_start

SCALE=1_000_000
MODES={'full':0,'cached':1,'cap4':2,'guarded4':3}
I=ctypes.POINTER(ctypes.c_int64)
D=ctypes.POINTER(ctypes.c_double)
CALLBACK=ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_void_p,I,I,D,I,I,I,I,D,I)


@lru_cache(maxsize=1)
def persistent_library():
    lib=ctypes.CDLL(str(Path(__file__).with_name('_persistent.so')))
    i=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    d=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    u=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
    lib.lns_create.argtypes=[ctypes.c_int]*3+[ctypes.c_double]+[i]*4+[d]*3+[i,d,d]
    lib.lns_create.restype=ctypes.c_void_p
    lib.lns_destroy.argtypes=[ctypes.c_void_p];lib.lns_destroy.restype=None
    lib.lns_evaluate.argtypes=[ctypes.c_void_p,ctypes.c_int,i,ctypes.c_void_p,ctypes.c_void_p,d]
    lib.lns_evaluate.restype=ctypes.c_int
    lib.lns_stats.argtypes=[ctypes.c_void_p,i];lib.lns_stats.restype=None
    lib.pd_solve.argtypes=[ctypes.c_void_p,u,d,i,ctypes.c_void_p,ctypes.c_int,ctypes.c_uint64,
                           ctypes.c_int,ctypes.c_int,i,d,i,d,ctypes.c_void_p,ctypes.c_void_p]
    lib.pd_solve.restype=ctypes.c_int
    lib.pd_error.restype=ctypes.c_char_p
    return lib


class PersistentPhy(NativePhy):
    def __init__(self,p):
        self.problem=p;self.profile=False;self.lib=persistent_library()
        self.covariance=np.ascontiguousarray(p.h@p.h.conj().swapaxes(-1,-2))
        self.signal=np.ascontiguousarray(p.h[np.arange(p.U),:,p.serving]@p.w)
        offsets=np.r_[0,np.cumsum([len(x) for x,_ in curves()])].astype(np.int64)
        xs=np.concatenate([x for x,_ in curves()]);ys=np.concatenate([y for _,y in curves()])
        self.context=self.lib.lns_create(p.U,p.C,p.P,p.noise,p.serving,p.mcs,p.caps,p.tbs,p.queue,
            self.covariance.view(np.float64),self.signal.view(np.float64),offsets,xs,ys)
        if not self.context:raise RuntimeError('Could not allocate persistent PHY')
        self._release=weakref.finalize(self,self.lib.lns_destroy,self.context)


@dataclass(frozen=True)
class PersistentConfig:
    mode: str='cached'
    max_rounds: int=256
    diagnostics: bool=False

    def __post_init__(self):
        if self.mode not in MODES or self.max_rounds<1:raise ValueError('Invalid persistent config')


def weak_mask(p):
    active=np.flatnonzero(p.queue>0)
    selected=active[np.lexsort((active,-p.weights[active]))[:max(1,int(np.ceil(.2*len(active))))]]
    return sum(1<<int(u) for u in selected)


def check_contract(p):
    if p.U+p.P>63 or p.P*(p.U+3)**p.C>=2**63:raise ValueError('Unsupported sparse-mask/key dimensions')
    if not np.array_equal(p.queue,np.floor(p.queue)):raise ValueError('Integer byte queues required')
    bound=sum(int(np.ceil(q*w*SCALE))+1 for q,w in zip(p.queue,p.weights))
    if bound>=np.iinfo(np.int64).max//4:raise OverflowError('Integer objective bound is unsafe')


def q_contributions(values,weights):
    return np.rint((values[...,EXPECTED]*weights)*SCALE).astype(np.int64)


def solve_persistent(p,raw_pfq,config=PersistentConfig(),*,warm_start=None,observer=None):
    begin=time.perf_counter();check_contract(p)
    raw=p.actions(raw_pfq)[0]
    warm=repair_warm_start(p,warm_start) if warm_start is not None else None
    legal=np.ascontiguousarray(p.legal,np.uint8)
    phy=PersistentPhy(p)
    action=np.empty((p.C,p.P),np.int64);values=np.empty((p.U,8),np.float64)
    stats=np.empty(18,np.int64);times=np.empty(8,np.float64)
    failures=[];history=[]
    callback=None
    if observer is not None:
        def observe(_,meta_ptr,root_ptr,root_values_ptr,keys_ptr,prg_ptr,dj_ptr,dd_ptr,dense_ptr,selected_ptr):
            try:
                meta=np.ctypeslib.as_array(meta_ptr,(12,));n=int(meta[1]);k=int(meta[5])
                def array(ptr,shape,dtype):
                    result=np.ctypeslib.as_array(ptr,shape) if np.prod(shape) else np.empty(shape,dtype)
                    result.flags.writeable=False
                    return result
                item=dict(iteration=int(meta[0]),root=array(root_ptr,(p.C*p.P,),np.int64).reshape(p.C,p.P),
                    root_values=array(root_values_ptr,(p.U*8,),np.float64).reshape(p.U,8),
                    keys=array(keys_ptr,(n,),np.int64),prg=array(prg_ptr,(n,),np.int64),
                    delta_q=array(dj_ptr,(n,),np.int64),delta_d=array(dd_ptr,(n,),np.int64),
                    values=array(dense_ptr,(n*p.U*8,),np.float64).reshape(n,p.U,8),
                    selected_keys=array(selected_ptr,(k,),np.int64),
                    score_q=int(meta[2]),dequeue=int(meta[3]),floor=int(meta[4]),
                    scored=int(meta[6]),proposed=int(meta[7]),guard_rejected=bool(meta[8]))
                observer(item)
                history.append(dict(iteration=item['iteration'],candidates=n,scored=item['scored'],
                    selected_keys=item['selected_keys'].tolist(),guard_rejected=item['guard_rejected']))
                return 0
            except BaseException as error:
                failures.append(error);return 1
        callback=CALLBACK(observe)
    try:
        result=phy.lib.pd_solve(phy.context,legal,p.weights,raw,
            None if warm is None else ctypes.c_void_p(warm.ctypes.data),MODES[config.mode],weak_mask(p),
            config.max_rounds,int(config.diagnostics),action,values,stats,times,
            None if callback is None else ctypes.cast(callback,ctypes.c_void_p),None)
        if failures:raise failures[0]
        if result:raise RuntimeError(phy.lib.pd_error().decode())
        record=dict(version='persistent_q1e6_v1',mode=config.mode,scale=SCALE,epsilon_q=1,
            rounds=int(stats[0]),accepted_moves=int(stats[1]),generated_edits=int(stats[2]),
            regenerated_prgs=int(stats[3]),scored_edits=int(stats[4]),
            sum_neighborhood_sizes=int(stats[5]),guard_rejections=int(stats[6]),
            accepted_batches=int(stats[7]),first_scan_evaluations=int(stats[8]),
            initialization_evaluations=int(stats[9]),converged=bool(stats[10]),score_q=int(stats[11]),
            floor=int(stats[12]),chosen_dequeue=int(stats[13]),reused_edits=int(stats[14]),
            score=float(values[:,EXPECTED]@p.weights),history=history,**phy.stats)
        if config.diagnostics:
            record.update(native_setup_ms=float(times[0]),table_generation_ms=float(times[1]),
                sparse_phy_ms=float(times[2]),selection_guard_ms=float(times[3]),first_scan_ms=float(times[6]))
    finally:phy.close()
    record['core_ms']=(time.perf_counter()-begin)*1000
    return action,values,record


def solve_integer_reference(p,draft,*,warm_start=None,max_rounds=256,observer=None):
    """Independent full-array/full-reduction definition of the new integer goal."""
    check_contract(p);phy=NativePhy(p)
    try:
        root=phy.evaluate(draft);score=int(q_contributions(root.values[0],p.weights).sum(dtype=np.int64))
        floor=(90*int(root.values[0,:,DEQUEUE].sum())+99)//100
        if warm_start is not None:
            other=phy.evaluate(repair_warm_start(p,warm_start))
            j=int(q_contributions(other.values[0],p.weights).sum(dtype=np.int64))
            if other.values[0,:,DEQUEUE].sum()>=floor and j>score:root,score=other,j
        for iteration in range(1,max_rounds+1):
            prg,assignment=compact_neighborhood(root.actions[0],p.legal)
            keys,_,_=edit_metadata(root.actions[0],prg,assignment,p.U)
            result=phy.evaluate(expand_edits(root.actions[0],prg,assignment),root)
            scores=q_contributions(result.values,p.weights).sum(1,dtype=np.int64)
            totals=result.values[:,:,DEQUEUE].sum(1).astype(np.int64)
            chosen=canonical_choice(scores-score,totals>=floor,keys,(1 if iteration<=2 else 16)*SCALE,1)
            if observer is not None:
                observer(dict(iteration=iteration,root=root.actions[0],root_values=root.values[0],keys=keys,
                    values=result.values,score_q=score,delta_q=scores-score,
                    delta_d=totals-int(root.values[0,:,DEQUEUE].sum()),floor=floor,
                    selected_keys=keys[chosen:chosen+1] if chosen>=0 else np.empty(0,np.int64)))
            if chosen>=0:root=phy.take(result,chosen);score=int(scores[chosen])
            if iteration>=3 and chosen<0:break
        else:raise AssertionError('Integer reference cap reached')
        return root.actions[0].copy(),root.values[0].copy(),dict(score_q=score,rounds=iteration,converged=True,**phy.stats)
    finally:phy.close()

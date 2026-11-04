"""Equivalent enumeration/deduplication backend for the frozen physical search."""
import ctypes
from functools import lru_cache
from pathlib import Path
import subprocess
import numpy as np
from .resource_conditions import solve_conditional as reference_solve
from .resource_conditions import Conditions,ConditionalConfig
from .physics import COUNT,LAYER,DEQUEUE,BLER


def build_pool():
    d=Path(__file__).parent
    subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-ffp-contract=off','-shared','-fPIC',
        str(d/'resource_pool_kernel.cpp'),'-o',str(d/'_resource_pool.so')],check=True)


@lru_cache(maxsize=1)
def pool_library():
    lib=ctypes.CDLL(str(Path(__file__).with_name('_resource_pool.so')))
    i=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS');d=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    b=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
    lib.resource_pool.argtypes=[ctypes.c_int]*3+[i,d,d,b,d]+[ctypes.c_int]*2+[b,ctypes.c_int,i]
    lib.resource_pool.restype=ctypes.c_int;return lib


def fast_pool(p,graph,base,condition):
    quota=condition.counts is not None;fixed=condition.activity is not None
    if quota:utility=np.zeros((p.U,p.P))
    else:
        v=base.values[0];count=v[:,COUNT].astype(int)
        layer=np.minimum(np.where(count>0,v[:,LAYER],4).astype(int)[:,None],p.caps)
        added=(base.actions[0,p.serving,:]!=np.arange(p.U)[:,None]).astype(int)
        capacity=p.tbs[np.arange(p.U)[:,None],np.minimum(count[:,None]+added,p.P),layer]
        delta=np.minimum(p.queue[:,None],capacity)-v[:,DEQUEUE,None]
        reliability=np.where(count>0,1-np.maximum(v[:,BLER],0),.5)
        utility=p.weights[:,None]*delta*reliability[:,None]+.05*graph.utility
    capacity=p.C*(p.P*5+p.P*(p.P-1)//2+40)
    out=np.empty((capacity,p.C,p.P),np.int64)
    active=np.ascontiguousarray(condition.activity,np.uint8) if fixed else np.zeros((p.C,p.P),np.uint8)
    n=pool_library().resource_pool(p.U,p.C,p.P,base.actions[0],base.values[0],p.weights,
        np.ascontiguousarray(p.legal,np.uint8),np.ascontiguousarray(utility),int(quota),int(fixed),active,capacity,out)
    if n<0:raise RuntimeError('Compiled resource pool failed')
    return out[:n].copy()


def ordered_unique(actions):
    seen=set();indices=[]
    for j,a in enumerate(actions):
        key=a.tobytes()
        if key not in seen:seen.add(key);indices.append(j)
    return actions[indices]


def solve_conditional(p,draft,condition=Conditions(),config=ConditionalConfig(),*,offline_diagnostic=False):
    a,v,r=reference_solve(p,draft,condition,config,offline_diagnostic=offline_diagnostic,
        pool_builder=fast_pool,deduplicator=ordered_unique)
    r['enumeration_backend']='compiled ordered pool + first-occurrence byte dedup'
    return a,v,r


if __name__=='__main__':build_pool()

"""Sequential search acceleration experiment; frozen score and audit behavior."""
import ctypes
import weakref
from pathlib import Path
import numpy as np
from output.pfq_independent_reuse.formal.boundary08_directed_coordination_20261008_v1.causal_phy import CausalPhy,SNR,BLER

lib=ctypes.CDLL(str(Path(__file__).with_name('physics_kernel.so')))
i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
early=ctypes.CDLL(str(Path(__file__).with_name('physics_kernel_stage1.so')))
early.physical.argtypes=[ctypes.c_int,i64,i64,i64,f64,f64,i64,i64,f64,f64]
early.physical.restype=None
lib.create_context.argtypes=[]
lib.create_context.restype=ctypes.c_void_p
lib.destroy_context.argtypes=[ctypes.c_void_p]
lib.destroy_context.restype=None
lib.reset_context.argtypes=[ctypes.c_void_p]
lib.reset_context.restype=None
lib.context_stats.argtypes=[ctypes.c_void_p,i64]
lib.context_stats.restype=None
lib.physical_cached.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,i64,i64,i64,f64,f64,i64,i64,f64,f64]
lib.physical_cached.restype=None

class OptimizedPhy(CausalPhy):
    def __init__(self,config,stage=2,threads=1):
        assert stage in (1,2,4) and threads>=1
        assert stage==4 or threads==1
        super().__init__(config)
        self.stage=stage
        self.threads=threads
        self.audit_calls=0
        self._context=lib.create_context()
        self._release=weakref.finalize(self,lib.destroy_context,self._context)

    @property
    def metrics(self):
        stats=np.empty(4,dtype=np.int64)
        lib.context_stats(self._context,stats)
        return dict(stage=self.stage, threads=self.threads, audit_calls=self.audit_calls,
            cache_hits=int(stats[0]),cache_misses=int(stats[1]),inactive_skips=int(stats[2]),
            actual_workers=int(stats[3]))

    def prepare(self,snapshot,ctbs):
        super().prepare(snapshot,ctbs)
        self.serving=np.ascontiguousarray(self.serving,dtype=np.int64)
        self.caps=np.ascontiguousarray(self.caps,dtype=np.int64)
        self.bank=np.ascontiguousarray(self.bank,dtype=np.float64)
        self.received=np.ascontiguousarray(self.received,dtype=np.float64)
        lib.reset_context(self._context)

    def evaluate(self,actions):
        actions=np.asarray(actions)
        if actions.ndim==2:actions=actions[None]
        assert actions.shape[1:]==(3,17)
        actions=np.ascontiguousarray(actions,dtype=np.int64)
        shape=(len(actions),36)
        count=np.empty(shape,dtype=np.int64);layers=np.empty_like(count)
        linear=np.empty(shape);interference=np.empty(shape)
        if self.stage==1:
            early.physical(len(actions),actions,self.serving,self.caps,self.bank,self.received,count,layers,linear,interference)
        else:
            lib.physical_cached(self._context,len(actions),self.threads,actions,self.serving,self.caps,
                self.bank,self.received,count,layers,linear,interference)
        capacity=self.tbs[self.mcs[None,:],count,layers]
        dequeue=np.minimum(capacity,self.snapshot['queue'][None,:])
        db=10*np.log10(np.maximum(linear,1.e-30))
        clipped=np.maximum(db,-4.75)
        bler=np.zeros_like(db)
        for m in np.unique(self.mcs):
            use=self.mcs==m
            bler[:,use]=np.interp(clipped[:,use],SNR[m][::-1],BLER[m][::-1],left=1.,right=0.)
        expected=dequeue*(1-bler)
        prediction=dict(count=count,layer=np.where(count>0,layers,-1),mcs=np.where(count>0,self.mcs[None],-1),
            capacity=capacity,dequeue=dequeue,sinr_db=np.where(count>0,db,-999.),
            bler=np.where(count>0,bler,-1.),expected=expected,interference=interference)
        if self.snapshot['tti'] in [4000,4001,4002] or self.snapshot['tti']%1000==0:
            reference=CausalPhy.evaluate(self,actions)
            for key in prediction:
                np.testing.assert_array_equal(prediction[key],reference[key],err_msg=key)
            self.audit_calls+=1
        return prediction

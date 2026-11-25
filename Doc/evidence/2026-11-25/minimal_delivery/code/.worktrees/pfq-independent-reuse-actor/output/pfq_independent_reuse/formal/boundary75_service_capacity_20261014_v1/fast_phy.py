"""Experimental native reduction; frozen candidates and NumPy BLER lookup."""
import ctypes
from pathlib import Path
import numpy as np
from output.pfq_independent_reuse.formal.boundary08_directed_coordination_20261008_v1.causal_phy import CausalPhy,SNR,BLER

lib=ctypes.CDLL(str(Path(__file__).with_name('physics_kernel.so')))
i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
lib.physical.argtypes=[ctypes.c_int,i64,i64,i64,f64,f64,i64,i64,f64,f64]
lib.physical.restype=None

class FastPhy(CausalPhy):
    def prepare(self,snapshot,ctbs):
        super().prepare(snapshot,ctbs)
        self.serving=np.ascontiguousarray(self.serving,dtype=np.int64)
        self.caps=np.ascontiguousarray(self.caps,dtype=np.int64)
        self.bank=np.ascontiguousarray(self.bank,dtype=np.float64)
        self.received=np.ascontiguousarray(self.received,dtype=np.float64)

    def evaluate(self,actions):
        actions=np.asarray(actions)
        if actions.ndim==2:actions=actions[None]
        assert actions.shape[1:]==(3,17)
        actions=np.ascontiguousarray(actions,dtype=np.int64)
        shape=(len(actions),36)
        count=np.empty(shape,dtype=np.int64);layers=np.empty_like(count)
        linear=np.empty(shape);interference=np.empty(shape)
        lib.physical(len(actions),actions,self.serving,self.caps,self.bank,self.received,count,layers,linear,interference)
        capacity=self.tbs[self.mcs[None,:],count,layers]
        dequeue=np.minimum(capacity,self.snapshot['queue'][None,:])
        db=10*np.log10(np.maximum(linear,1.e-30))
        clipped=np.maximum(db,-4.75)
        bler=np.zeros_like(db)
        for m in np.unique(self.mcs):
            use=self.mcs==m
            bler[:,use]=np.interp(clipped[:,use],SNR[m][::-1],BLER[m][::-1],left=1.,right=0.)
        expected=dequeue*(1-bler)
        return dict(count=count,layer=np.where(count>0,layers,-1),mcs=np.where(count>0,self.mcs[None],-1),
            capacity=capacity,dequeue=dequeue,sinr_db=np.where(count>0,db,-999.),
            bler=np.where(count>0,bler,-1.),expected=expected,interference=interference)

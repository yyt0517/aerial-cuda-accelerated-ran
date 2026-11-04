"""Sparse approximate-PHY cache and UE-wise neural residual corrections.

One cache owns one graph and verified root. All counterfactual calculations
are trace/projection ratios and table lookups, never exact candidate MMSE.
Only changed-resource UEs and their activity-affected victims are recomputed.
"""
import ctypes
from functools import lru_cache
from pathlib import Path
import weakref
import numpy as np

from .physics import curves
from .deadline import urgency

@lru_cache(maxsize=1)
def library():
    lib=ctypes.CDLL(str(Path(__file__).with_name('_draft.so')))
    i=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    f=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    x=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    m=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
    lib.draft_create.argtypes=[ctypes.c_int]*3+[ctypes.c_double]+[i]*3+[f]*7+[i,f,f]
    lib.draft_create.restype=ctypes.c_void_p
    lib.draft_destroy.argtypes=[ctypes.c_void_p];lib.draft_destroy.restype=None
    lib.draft_root.argtypes=[ctypes.c_void_p,i,f];lib.draft_root.restype=ctypes.c_int
    lib.draft_eval.argtypes=[ctypes.c_void_p,ctypes.c_int,i,i,f,x,f,m];lib.draft_eval.restype=ctypes.c_int
    lib.draft_stats.argtypes=[ctypes.c_void_p,i];lib.draft_stats.restype=None
    return lib


class DraftCache:
    def __init__(self, graph, base):
        p=graph.problem
        if base.owner.problem is not p or graph.received is not base.owner.received:
            raise ValueError('Draft root belongs to a different graph/PHY')
        if not 1<=p.C<=8:
            raise ValueError('Approximate activity cache currently supports 1--8 cells')
        self.graph,self.base,self.problem=graph,base,p
        self.lib=library()
        # Reuse the exact verifier's already prepared causal H/W covariance;
        # this projection performs no inverse or candidate verification.
        signal=base.owner.signal
        norm=np.maximum(np.abs(signal)**2,0).sum(-2)
        projected=np.einsum('upil,upcij,upjl->upcl',signal.conj(),base.owner.covariance,signal).real
        projected=np.maximum(projected/np.maximum(norm[:,:,None,:],1.e-30),0)
        if p.sinr_hint is not None:
            hint=p.sinr_hint
        else:
            received=graph.received.copy();received[np.arange(p.U),:,p.serving]=0
            root_denom=4*p.noise+np.einsum('cp,upc->up',base.actions[0]>=0,received)
            hint=np.repeat((graph.received[np.arange(p.U),:,p.serving]/root_denom)[:,:,None],4,axis=2)
        self._arrays=[np.ascontiguousarray(v,dtype=np.float64) for v in
            (p.queue,p.weights,urgency(p),p.deadline[:,5],graph.received,hint,projected)]
        # Seven arrays above? queue/weight/urgency/weak/received/hint/projection.
        offsets=np.r_[0,np.cumsum([len(xs) for xs,_ in curves()])].astype(np.int64)
        xs=np.concatenate([xs for xs,_ in curves()]);ys=np.concatenate([ys for _,ys in curves()])
        self.context=self.lib.draft_create(p.U,p.C,p.P,p.noise,p.serving,p.caps,p.tbs,
            *self._arrays,offsets,xs,ys)
        if not self.context:raise RuntimeError('Could not create approximate PHY cache')
        self._release=weakref.finalize(self,self.lib.draft_destroy,self.context)
        if self.lib.draft_root(self.context,base.actions,base.values):
            self.close();raise RuntimeError('Could not initialize approximate root')

    def evaluate(self, actions):
        if not self._release.alive:raise RuntimeError('Draft cache is closed')
        p=self.problem;actions=p.actions(actions);n=len(actions)
        score=np.empty((n,2),np.float64);features=np.empty((n,p.U,32),np.float32)
        raw=np.empty((n,p.U,7),np.float64);mask=np.empty((n,p.U),np.uint8)
        if self.lib.draft_eval(self.context,n,actions,p.mcs,score,features,raw,mask):
            raise RuntimeError('Approximate PHY failed')
        return dict(scores=score,features=features,raw=raw,mask=mask.astype(bool))

    @property
    def stats(self):
        if not self._release.alive:raise RuntimeError('Draft cache is closed')
        values=np.empty(4,np.int64);self.lib.draft_stats(self.context,values)
        return dict(zip(('candidates','affected_ues','affected_prgs','root_builds'),map(int,values)))

    def close(self):self._release()


def terms(expected, raw):
    return (expected*(raw[...,3]+.5*raw[...,4])-.5*np.maximum(raw[...,6]-expected,0)*raw[...,4]
            +.5*np.minimum(expected,raw[...,6])*raw[...,5])


class ResidualValue:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as z:
            if int(z['version'])!=1:raise ValueError('Unknown residual version')
            self.mode=int(z['mode'])
            for k in ('mean','scale','w1','b1','w2','b2','w3','b3'):
                setattr(self,k,z[k].copy())

    def predict_rows(self,x):
        x=np.clip((x-self.mean)/self.scale,-10,10)
        out=[]
        for j in range(len(self.w1)):
            h=np.maximum(x@self.w1[j].T+self.b1[j],0)
            h=np.maximum(h@self.w2[j].T+self.b2[j],0)
            out.append((h@self.w3[j].T+self.b3[j])[:,0])
        return np.asarray(out)

    def score(self,draft):
        b,u=np.nonzero(draft['mask'])
        base=draft['scores'][:,self.mode]
        if not len(b):return base.copy(),np.zeros(len(base))
        raw=draft['raw'][b,u];approx=raw[:,1+self.mode]
        normalizer=np.maximum(raw[:,0],2400.)
        pred=self.predict_rows(draft['features'][b,u])
        outputs=[]
        for residual in pred:
            expected=np.clip(approx+residual*normalizer,0,raw[:,0])
            score=base.copy()
            np.add.at(score,b,terms(expected,raw)-terms(approx,raw))
            outputs.append(score)
        outputs=np.asarray(outputs)
        return outputs.mean(0),outputs.std(0)

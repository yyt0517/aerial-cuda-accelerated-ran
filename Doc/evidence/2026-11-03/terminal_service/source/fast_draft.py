"""Behavior-preserving binding/mean-only optimization of sparse draft search.

Addresses are passed only after the same public action validation and private
contiguous output allocation. Neural weights, row order and floating arithmetic
are unchanged. Mean-only search does not claim to compute uncertainty.
"""
import ctypes
from functools import lru_cache
from pathlib import Path
import numpy as np
from .residual import DraftCache,ResidualValue,terms
from .draft_repair import DraftRepair,RebasedDraft

@lru_cache(maxsize=1)
def raw_evaluator():
    lib=ctypes.CDLL(str(Path(__file__).with_name('_draft.so')))
    func=lib.draft_eval
    func.argtypes=[ctypes.c_void_p,ctypes.c_int]+[ctypes.c_void_p]*6
    func.restype=ctypes.c_int
    return lib,func

class FastDraftCache(DraftCache):
    def evaluate(self,actions):
        if not self._release.alive:raise RuntimeError('Draft cache is closed')
        p=self.problem;actions=p.actions(actions);n=len(actions)
        score=np.empty((n,2),np.float64);features=np.empty((n,p.U,32),np.float32)
        raw=np.empty((n,p.U,7),np.float64);mask=np.empty((n,p.U),np.uint8)
        _,func=raw_evaluator()
        if func(self.context,n,actions.ctypes.data,p.mcs.ctypes.data,score.ctypes.data,
                features.ctypes.data,raw.ctypes.data,mask.ctypes.data):
            raise RuntimeError('Approximate PHY failed')
        return dict(scores=score,features=features,raw=raw,mask=mask.astype(bool))

class FastRebasedDraft(RebasedDraft):
    def for_base(self,graph,base):
        old=self.cache
        if old is None or old.graph is not graph or (old.base is not base and graph.problem.sinr_hint is None):
            if old is not None:old.close()
            self.cache=FastDraftCache(graph,base)
        elif old.base is not base:
            if base.owner is not old.base.owner:raise ValueError('Foreign rebase')
            if old.lib.draft_root(old.context,base.actions,base.values):raise RuntimeError('Draft rebase failed')
            old.base=base
        return self.cache

class MeanResidualValue(ResidualValue):
    def score(self,draft):
        b,u=np.nonzero(draft['mask']);base=draft['scores'][:,self.mode]
        if not len(b):return base.copy(),None
        raw=draft['raw'][b,u];approx=raw[:,1+self.mode]
        normalizer=np.maximum(raw[:,0],2400.)
        predictions=self.predict_rows(draft['features'][b,u]);before=terms(approx,raw)
        outputs=[]
        for residual in predictions:
            expected=np.clip(approx+residual*normalizer,0,raw[:,0])
            score=base.copy();np.add.at(score,b,terms(expected,raw)-before)
            outputs.append(score)
        return np.asarray(outputs).mean(0),None

class FastDraftRepair(DraftRepair):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.drafts=FastRebasedDraft()
        if self.mode=='residual':self.residual=MeanResidualValue(kwargs['residual_checkpoint'])

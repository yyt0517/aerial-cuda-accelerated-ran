"""Incremental logical descriptors and protected selective exact shortlists."""
from dataclasses import dataclass
import ctypes
from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess
import time
import numpy as np

from .bounded_probes import ProbePhy
from .physics import COUNT,LAYER,DEQUEUE,EXPECTED,BLER
from .tb_macro import ExactMemo,candidate_pools,prg_priority
from .search import ProposalGraph
from .no_ctbs_inputs import raw_pfq_floor


def build_kernel():
    d=Path(__file__).parent
    subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-ffp-contract=off','-shared','-fPIC',
        str(d/'selective_tb_kernel.cpp'),'-o',str(d/'_selective_tb.so')],check=True)


@lru_cache(maxsize=1)
def feature_library():
    lib=ctypes.CDLL(str(Path(__file__).with_name('_selective_tb.so')))
    i=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    d=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    f=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    lib.selective_tb_features.argtypes=[ctypes.c_int]*4+[i]*6+[d]*5+[ctypes.c_double]+[i,i,d,d,d,f]
    lib.selective_tb_features.restype=ctypes.c_int
    return lib


class Incremental:
    def __init__(self,p,phy,limit=56):
        self.p=p;self.phy=phy;self.limit=limit;self.rows={};self.graph=ProposalGraph(phy)
        self.pool_ms=0.;self.feature_ms=0.;self.proxy_candidates=0;self.lib=feature_library()
        self.received=phy.received.copy();self.received[np.arange(p.U),:,p.serving]=0.

    def get(self,base):
        key=base.actions[0].tobytes()
        if key not in self.rows:
            if len(self.rows)>=self.limit:return None
            start=time.perf_counter();p=self.p;root=base.actions[0]
            hist=np.zeros((p.U,5),np.int64);cells,prgs=np.nonzero(root>=0);users=root[cells,prgs]
            np.add.at(hist,(users,p.caps[users,prgs]),1)
            denominator=4*p.noise+np.einsum('cp,upc->up',root>=0,self.received)
            self.rows[key]=(hist,np.ascontiguousarray(denominator))
            self.feature_ms+=(time.perf_counter()-start)*1000
        return self.rows[key]

    def features(self,base,actions,floor,joint=False):
        data=self.get(base)
        if data is None:return None
        start=time.perf_counter();p=self.p;a=p.actions(actions);n=len(a)
        counts=np.empty((n,p.U),np.int64);layers=np.empty_like(counts);dequeue=np.empty((n,p.U))
        proxy=np.empty(n);total=np.empty(n);x=np.empty((n,60),np.float32)
        code=self.lib.selective_tb_features(n,p.U,p.C,p.P,a,base.actions[0],p.serving,p.caps,data[0],p.tbs,
            p.queue,p.weights,base.values[0],self.received,data[1],floor,counts,layers,dequeue,proxy,total,x)
        if code:raise ValueError(f'Unsupported incremental candidate: {code}')
        x[:,11]=joint
        self.proxy_candidates+=n;self.feature_ms+=(time.perf_counter()-start)*1000
        return dict(x=x,prior=proxy,total=total,counts=counts,layers=layers,dequeue=dequeue)


class ServiceGuard:
    """Fixed-anchor plus current-root protection; never consult full-DPLS labels."""
    def __init__(self,anchor):self.anchor=anchor.copy()

    @staticmethod
    def against(values,reference):
        v=np.asarray(values)
        if v.ndim==2:v=v[None]
        ok=~((reference[None,:,COUNT]>0)&(v[:,:,COUNT]==0)).any(1)
        served=np.flatnonzero(reference[:,EXPECTED]>1.)
        if len(served):
            target=reference[served,EXPECTED];weak=served[np.argsort(target,kind='stable')[:max(1,int(np.ceil(.2*len(served))))]]
            ok &= np.quantile(v[:,served,EXPECTED],.05,axis=1)+1.e-9>=.98*np.quantile(target,.05)
            ok &= v[:,weak,EXPECTED].sum(1)+1.e-9>=.98*reference[weak,EXPECTED].sum()
        return ok

    def safe(self,values,current):return self.against(values,self.anchor)&self.against(values,current)


class Ranker:
    def __init__(self,path):
        self.path=Path(path)
        with np.load(path) as z:
            for k in ('mean','scale','w1','b1','w2','b2','w3','b3'):setattr(self,k,z[k])

    def outputs(self,x):
        z=np.clip((x-self.mean)/self.scale,-10,10)
        z=np.maximum(z@self.w1.T+self.b1,0);z=np.maximum(z@self.w2.T+self.b2,0)
        return z@self.w3.T+self.b3

    def predict(self,x,prior,gain_only=False):
        out=self.outputs(x);score=prior+3000*out[:,0]
        if not gain_only:score-=3000/(1+np.exp(np.clip(out[:,1],-30,30)))
        return score


@dataclass(frozen=True)
class SelectiveConfig:
    mode: str='selective'
    protect: bool=True
    gain_only: bool=False
    max_candidates: int=212
    max_feature_roots: int=56
    max_sweeps: int=5

    def __post_init__(self):
        if self.mode not in ('single','fixed','selective') or min(self.max_candidates,self.max_feature_roots,self.max_sweeps)<1:
            raise ValueError('Invalid selective search configuration')


def ordering(scores,indices):
    # Differences below 1e-7 B are immaterial to the >1B execution threshold.
    return indices[np.argsort(-np.round(scores[indices],7),kind='stable')]


def choose_indices(scores,joint,available,mode,quota):
    single=ordering(scores,np.flatnonzero(available&~joint));macro=ordering(scores,np.flatnonzero(available&joint))
    if mode=='single':return single[:quota]
    selected=list(single[:min(2,quota)])
    if mode=='fixed':selected+=list(macro[:max(0,quota-len(selected))])
    used=np.zeros(len(scores),bool);used[selected]=True
    rest=ordering(scores,np.flatnonzero(available&~used))
    return np.array([*selected,*rest[:max(0,quota-len(selected))]],np.int64)


def solve_selective(p,draft,config=SelectiveConfig(),ranker=None,*,observer=None):
    phy=ProbePhy(p)
    try:
        memo=ExactMemo(phy,draft,config.max_candidates);root=memo.root;anchor=root.values[0].copy()
        floor=raw_pfq_floor(anchor);guard=ServiceGuard(anchor);d=Incremental(p,phy,config.max_feature_roots)
        failed=set();covered=set();second=set();accepted=joint_accepted=0;selected_joint=0;guard_rejects=0
        network_ms=physical_ms=0.;ordinal=0;stop='sweep_limit';first_sweep=None
        for sweep in range(config.max_sweeps):
            progress=accepted;order=prg_priority(p,d,root)
            for prg in order:
                if memo.remaining<=0:stop='physical_budget';break
                if d.get(root) is None:stop='feature_budget';break
                singles,macros=candidate_pools(p,d,root,int(prg));actions=np.concatenate([singles,macros])
                joint=np.arange(len(actions))>=len(singles)
                f=d.features(root,actions,floor);f['x'][:,11]=joint
                available=(f['total']>=floor)&np.array([a.tobytes() not in failed for a in actions],bool)
                if config.protect:available &= ~((root.values[0,None,:,COUNT]>0)&(f['counts']==0)).any(1)
                begin=time.perf_counter()
                scores=(f['prior'] if ranker is None or sweep==0 else ranker.predict(f['x'],f['prior'],config.gain_only))
                mode='single' if sweep==0 else config.mode
                chosen=choose_indices(scores,joint,available,mode,min(4,memo.remaining))
                network_ms+=(time.perf_counter()-begin)*1000;ordinal+=1
                if observer is not None:
                    observer(dict(root=root.actions[0].copy(),prg=int(prg),sweep=sweep,ordinal=ordinal,
                        failed=tuple(sorted(failed)),pool_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
                        selected=chosen.copy(),floor=floor,anchor=anchor.copy()))
                old_key=root.actions[0].tobytes()
                if len(chosen):
                    begin=time.perf_counter();exact,_=memo.evaluate(actions[chosen],root);physical_ms+=(time.perf_counter()-begin)*1000
                    selected_joint+=int(joint[chosen].sum());failed.update(a.tobytes() for a in actions[chosen])
                    js=np.array([float(v[:,EXPECTED]@p.weights) for v in exact.values]);ok=exact.values[:,:,DEQUEUE].sum(1)>=floor
                    if config.protect:
                        safe=guard.safe(exact.values,root.values[0]);guard_rejects+=int((ok&~safe).sum());ok &= safe
                    if ok.any():
                        best=int(np.argmax(np.where(ok,js,-np.inf)))
                        if js[best]>float(root.values[0,:,EXPECTED]@p.weights)+1:
                            joint_accepted+=int(joint[chosen[best]]);root=phy.take(exact,best);accepted+=1
                if old_key!=root.actions[0].tobytes():failed=set()
                covered.add(int(prg))
                if sweep==1:second.add(int(prg))
            if sweep==0:first_sweep=root.actions[0].copy()
            if stop in ('physical_budget','feature_budget'):break
            if sweep>=1 and progress==accepted:stop='proxy_domain_stalled';break
        v=root.values[0];r=dict(score=float(v[:,EXPECTED]@p.weights),floor=floor,feature_roots=len(d.rows),
            covered_prgs=len(covered),second_pass_prgs=len(second),accepted_edits=accepted,joint_accepted=joint_accepted,
            selected_joint=selected_joint,guard_rejects=guard_rejects,proxy_candidates=d.proxy_candidates,
            feature_ms=d.feature_ms,candidate_generation_ms=d.pool_ms,network_ms=network_ms,physical_ms=physical_ms,
            memo_hits=memo.cache_hits,stop_reason=stop,full_search_calls=0,**phy.stats)
        assert r['candidate_evaluations']<=config.max_candidates and r['feature_roots']<=config.max_feature_roots
        assert v[:,DEQUEUE].sum()>=floor
        if config.protect:assert guard.against(v,anchor)[0]
        return root.actions[0].copy(),v.copy(),r,first_sweep
    finally:phy.close()


if __name__=='__main__':build_kernel()

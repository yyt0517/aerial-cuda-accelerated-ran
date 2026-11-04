"""Reuse frozen neural repair and exact PHY to expose service alternatives.

PPO selects a verified complete action; no policy action is modified after
sampling. Equal DPLS is always available. No CTBS prune or deficit weights.
"""
from pathlib import Path
import time
import numpy as np
from training.stageb_ppo.native import ROOT,teacher,sha
from training.stageb_lns.archive import problem_from_row
from training.stageb_lns.physics import NativePhy,EXPECTED,DEQUEUE,COUNT,LAYER,BLER,SINR,CAPACITY
from training.stageb_lns.search import ProposalGraph,SearchConfig,greedy_candidates
from training.stageb_lns.cached_repair import CachedJointRepair
from .state import packet_state

REPAIR=ROOT/'deliverables/tb_joint_repair_20261027_train_v1/repair.npz'
MAX_CHOICES=16
STATE_DIM=15
ACTION_DIM=9
SUMMARY_DIM=7


def completion_bytes(fifo,tti,queue,dequeues):
    """Clean complete packet payload eligible for the current TB CRC."""
    out=np.zeros_like(dequeues,dtype=float)
    for u,q in enumerate(fifo.queues):
        active=[p for p in q if tti-p[0]+1<=fifo.ttl]
        arrived=int(queue[u])-sum(p[1] for p in active)
        assert arrived>=0 and arrived%fifo.size==0
        active += [[tti,fifo.size,False] for _ in range(arrived//fifo.size)]
        used=0
        for _,remaining,damaged in active:
            used+=remaining
            if not damaged:out[:,u]+=(dequeues[:,u]>=used)*fifo.size
    return out


class Catalog:
    def __init__(self,retention=.98):
        self.study=teacher()
        self.repair=CachedJointRepair(REPAIR,width=8)
        self.retention=retention
        self.repair_sha256=sha(REPAIR)

    def build(self,env):
        start=time.monotonic()
        root_action=env.expert(self.study)
        p=problem_from_row(dict(ctbs=env.inputs,obs=env.obs,snapshot=env.snapshot,ema=np.full(36,2400.)))
        p.weights[:]=1.;p.deadline=None
        np.testing.assert_array_equal(p.weights,np.ones(36))
        packet,phi=packet_state(env.fifo,env.tti+1,p.queue)
        with_phy=NativePhy(p)
        try:
            root=with_phy.evaluate(root_action)
            raw=with_phy.evaluate(env.raw_pfq())
            floor=(90*int(raw.values[0,:,DEQUEUE].sum())+99)//100
            # The frozen native search's root must meet its own raw-PFQ floor.
            assert root.values[0,:,DEQUEUE].sum()>=floor
            graph=ProposalGraph(with_phy)
            blocks=graph.blocks(root_action,root.values[0],SearchConfig(block_size=4),np.random.default_rng(0))[:2]
            neural=[self.repair.propose(graph,root,block) for block in blocks]
            physical=greedy_candidates(p,root_action)
            generated=np.concatenate([root.actions,physical,*neural],axis=0)
            origins=np.r_[np.array([0]),np.ones(len(physical),int),np.full(sum(map(len,neural)),2,int)]
            unique=[];labels=[];seen={}
            for action,origin in zip(generated,origins):
                key=action.tobytes()
                if key not in seen:
                    seen[key]=len(unique);unique.append(action);labels.append({int(origin)})
                else:labels[seen[key]].add(int(origin))
            result=with_phy.evaluate(np.stack(unique),root)
            values=result.values
            score=values[:,:,EXPECTED].sum(-1)
            dequeue=values[:,:,DEQUEUE].sum(-1)
            eligible=(dequeue>=floor)&(score+1.e-7>=self.retention*score[0])
            assert eligible[0]
            clean=completion_bytes(env.fifo,env.tti+1,p.queue,values[:,:,DEQUEUE])
            full=clean*(1-np.maximum(values[:,:,BLER],0))
            pool=np.flatnonzero(eligible)
            selected=[0]
            def add(index):
                if int(index) not in selected and len(selected)<MAX_CHOICES:selected.append(int(index))
            add(pool[np.argmax(score[pool])])
            add(pool[np.argmax(full[pool].sum(-1))])
            add(pool[np.argmax(dequeue[pool])])
            n_pool=np.array([i for i in pool if 2 in labels[i]],dtype=int)
            if len(n_pool):add(n_pool[np.argmax(score[n_pool])])
            # Cover distinct per-UE service allocations, all UEs equally scaled.
            vectors=values[:,:,EXPECTED]/3000.
            while len(selected)<min(MAX_CHOICES,len(pool)):
                distance=((vectors[pool,None]-vectors[np.array(selected)][None])**2).sum(-1).min(-1)
                distance[np.isin(pool,selected)]=-1
                add(pool[np.argmax(distance)])
            selected=np.array(selected)
            v=values[selected];n=len(selected)
            state=np.concatenate((packet,np.log1p(p.queue[:,None]/3000),p.mcs[:,None]/27,
                root.values[0,:,EXPECTED,None]/3000,root.values[0,:,COUNT,None]/17,
                root.values[0,:,DEQUEUE,None]/np.maximum(p.queue[:,None],3000),
                np.maximum(root.values[0,:,BLER,None],0),root.values[0,:,SINR,None]/30),axis=-1)
            action_features=np.stack((v[:,:,EXPECTED]/3000,(v[:,:,EXPECTED]-values[0,:,EXPECTED])/3000,
                v[:,:,DEQUEUE]/3000,(v[:,:,DEQUEUE]-values[0,:,DEQUEUE])/3000,
                v[:,:,COUNT]/17,(v[:,:,COUNT]-values[0,:,COUNT])/17,
                np.maximum(v[:,:,BLER],0),v[:,:,SINR]/30,full[selected]/3000),axis=-1)
            summary=np.stack((score[selected]/86400,(score[selected]-score[0])/3000,
                dequeue[selected]/86400,full[selected].sum(-1)/86400,
                (v[:,:,COUNT]>0).sum(-1)/36,(result.actions[selected]<0).sum((1,2))/51,
                (selected==0).astype(float)),-1)
            padded=np.zeros((MAX_CHOICES,36,ACTION_DIM),np.float32);padded[:n]=action_features
            sums=np.zeros((MAX_CHOICES,SUMMARY_DIM),np.float32);sums[:n]=summary
            mask=np.arange(MAX_CHOICES)<n
            actions=np.full((MAX_CHOICES,3,17),-1,np.int64);actions[:n]=result.actions[selected]
            assert state.shape==(36,STATE_DIM)
            # Fixed physical prior; the trainable residual learns future value.
            prior=np.zeros(MAX_CHOICES,np.float32);prior[:n]=(score[selected]-score[selected].max())/3000
            return dict(state=np.clip(state,-10,10).astype(np.float32),candidate=np.clip(padded,-10,10),
                summary=np.clip(sums,-10,10),mask=mask,prior=prior,actions=actions,potential=phi,
                audit=dict(tti=env.tti+1,choices=n,candidate_evaluations=with_phy.stats['candidate_evaluations'],
                    floor_bytes=floor,root_expected_bytes=float(score[0]),
                    min_catalog_expected_ratio=float(score[selected].min()/score[0]) if score[0]>0 else 1.,
                    neural_proposals=sum(map(len,neural)),neural_eligible=int(len(n_pool)),
                    neural_catalog=sum(2 in labels[i] for i in selected),
                    ctbs_pruned=0,weights_min=1.,weights_max=1.,wall_s=time.monotonic()-start))
        finally:with_phy.close()

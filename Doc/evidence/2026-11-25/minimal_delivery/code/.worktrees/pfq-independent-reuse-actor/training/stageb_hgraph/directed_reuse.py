"""Candidate-conditioned directed reuse branch for the three-cell research path.

The frozen independent actor supplies a complete joint draft. This branch
scores one additional blank, using only pre-action physical/link state.
No BLER table, expected-success label, or realized outcome is consumed here.
"""
import math
import numpy as np
import torch
from torch import nn
from .phy_tbs_alignment_audit import _MCS_LUT_0, _MCS_LUT_1

CONTEXT_NAMES=('mcs_over_27','asinh_olla_over_5','prior_tb_error','log1p_queue_packets',
    'hol_over_ttl','slack_over_ttl','log1p_avg_rate_over_30','recent_deficit',
    'prg_count_over_17','layer_over_4','dequeue_packets','tb_sinr_over_20','asinh_mcs_margin_over_5')
SOURCE_NAMES=CONTEXT_NAMES+('after_prg_count_over_17','after_layer_over_4','after_dequeue_packets',
    'after_tb_sinr_over_20','dequeue_cost_packets','own_prg_sinr_over_20','cell_active_fraction')
EDGE_NAMES=('log1p_source_interference_noise_over_10','prg_removal_gain_over_20',
    'tb_removal_gain_over_20','source_interference_share')

def build_directed_features(chosen,ctbs,queue,bank,received,noise,ue_features,tbs):
    """Fixed [51 sources, 2 scheduled neighboring victims] tensors, without labels.

    bank[UE,PRG,activity_mask,layer] is estimated pre-action MMSE SINR.
    received[UE,PRG,source_cell] is the estimated receive covariance trace.
    Queue is current; age/history retain the bridge observation timestamp.
    """
    chosen=np.asarray(chosen,dtype=np.int64)
    queue=np.asarray(queue);ue_features=np.asarray(ue_features)
    serving=np.asarray(ctbs.serving_cell);caps=np.asarray(ctbs.prg_layer_cap)
    assert chosen.shape==(3,17) and bank.shape==(36,17,8,4) and received.shape==(36,17,3)
    assert queue.shape==(36,) and ue_features.shape[0]==36 and noise>0
    assert np.isfinite(queue).all() and np.isfinite(bank).all() and (queue>=0).all()
    assert ((chosen>=-1)&(chosen<36)).all()
    for cell in range(3): assert (serving[chosen[cell][chosen[cell]>=0]]==cell).all()
    assigned=chosen[serving,:]==np.arange(36)[:,None]
    count=assigned.sum(axis=1)
    layer=np.where(count>0,np.where(assigned,caps,5).min(axis=1),1)
    mcs=np.asarray([ctbs.reconstructed_mcs_for_ue(u) for u in range(36)])
    dequeue=np.minimum(tbs[mcs,count,layer],queue)
    masks=((chosen>=0)*(1<<np.arange(3))[:,None]).sum(axis=0)
    gamma=bank[np.arange(36)[:,None],np.arange(17)[None,:],masks[None,:]]
    used_layer=np.arange(4)[None,:]<layer[:,None]
    sinr_sum=(gamma*assigned[:,:,None]*used_layer[:,None,:]).sum(axis=(1,2))
    linear=sinr_sum/np.maximum(count*layer,1)
    db=10*np.log10(np.maximum(linear,1.e-30))
    delta=np.asarray(ctbs.olla_delta,dtype=np.float32).copy()
    prior=np.asarray(ctbs.prior_tb_err)
    delta+=np.where(prior==1,ctbs.olla_delta_up,np.where(prior==0,-np.asarray(ctbs.olla_delta_down),0)).astype(np.float32)
    lut=np.asarray(_MCS_LUT_0 if ctbs.config.mcs_lut_type==0 else _MCS_LUT_1)
    margin=np.minimum(ctbs.wb_sinr_first_layer_db,ctbs.config.mcs_sinr_cap_db)-delta-lut[mcs]
    context=np.stack((mcs/27,np.arcsinh(np.asarray(ctbs.olla_delta)/5),prior,
        np.log1p(queue/3000),np.clip(ue_features[:,8]/200,0,2),
        np.where(ue_features[:,9]>=0,np.clip(ue_features[:,9]/200,0,2),1),
        np.log1p(np.maximum(ue_features[:,1],0)/30),np.clip(ue_features[:,11],-10,10),
        count/17,layer/4,dequeue/3000,db/20,np.arcsinh(margin/5)),axis=1)
    source=np.zeros((51,20),dtype=np.float32)
    victims=np.zeros((51,2,13),dtype=np.float32)
    edges=np.zeros((51,2,4),dtype=np.float32)
    valid=chosen.reshape(-1)>=0;victim_mask=np.zeros((51,2),dtype=bool)
    for token in np.flatnonzero(valid):
        cell,p=divmod(int(token),17);u=int(chosen[cell,p])
        remaining=assigned[u].copy();remaining[p]=False
        nc=int(count[u]-1);nl=int(caps[u,remaining].min()) if nc else 0
        nd=min(float(tbs[mcs[u],nc,max(nl,1)]),float(queue[u]))
        after_db=float(10*np.log10(max(float(gamma[u,remaining,:nl].mean()),1.e-30))) if nc else 0.
        own_prg_db=float(10*np.log10(max(float(gamma[u,p,:layer[u]].mean()),1.e-30)))
        source[token]=np.r_[context[u],[nc/17,nl/4,nd/3000,after_db/20,
            (dequeue[u]-nd)/3000,own_prg_db/20,np.mean(chosen[cell]>=0)]]
        for j,neighbor in enumerate(c for c in range(3) if c!=cell):
            victim=int(chosen[neighbor,p])
            if victim<0 or queue[victim]<=0: continue
            victim_mask[token,j]=True;victims[token,j]=context[victim]
            layers=int(layer[victim]);before=gamma[victim,p,:layers]
            after=bank[victim,p,int(masks[p])^(1<<cell),:layers]
            prg_gain=10*np.log10(max(float(after.mean()),1.e-30)/max(float(before.mean()),1.e-30))
            after_linear=(sinr_sum[victim]+after.sum()-before.sum())/(count[victim]*layers)
            tb_gain=10*np.log10(max(float(after_linear),1.e-30))-db[victim]
            active=((int(masks[p])>>np.arange(3))&1)*(np.arange(3)!=serving[victim])
            interference=float(np.sum(received[victim,p]*active))
            power=float(received[victim,p,cell])
            edges[token,j]=[np.log1p(power/(4*noise))/10,prg_gain/20,tb_gain/20,
                power/(interference+4*noise)]
    result=dict(source=source,victims=victims,edges=edges,valid=valid,victim_mask=victim_mask)
    assert all(np.isfinite(v).all() for v in result.values())
    return result

class DirectedReuseCoordinator(nn.Module):
    """Scores in units of 3000 expected successful bytes; class zero keeps draft."""
    def __init__(self,hidden=64,heads=4,no_directed=False):
        super().__init__()
        assert hidden%heads==0
        self.hidden=hidden;self.heads=heads;self.no_directed=no_directed
        self.source_encoder=nn.Sequential(nn.Linear(20,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.victim_encoder=nn.Sequential(nn.Linear(13,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.query=nn.Linear(hidden,hidden);self.key=nn.Linear(hidden,hidden)
        self.value=nn.Linear(hidden,hidden)
        self.edge_bias=nn.Linear(4,heads,bias=False)
        self.edge_value=nn.Linear(4,hidden,bias=False)
        self.scorer=nn.Sequential(nn.Linear(hidden*3+4,96),nn.GELU(),nn.Linear(96,32),nn.GELU(),nn.Linear(32,1))

    def forward(self,source,victims,edges,valid,victim_mask):
        if self.no_directed: edges=torch.zeros_like(edges)
        mask=victim_mask[...,None]
        victims=victims.masked_fill(~mask,0);edges=edges.masked_fill(~mask,0)
        s=self.source_encoder(source);v=self.victim_encoder(victims)
        b,t,k,_=v.shape;d=self.hidden//self.heads
        q=self.query(s).reshape(b,t,self.heads,d)
        key=self.key(v).reshape(b,t,k,self.heads,d)
        value=self.value(v)+self.edge_value(edges)
        logits=torch.einsum('bthd,btkhd->bthk',q,key)/math.sqrt(d)
        logits=logits+self.edge_bias(edges).permute(0,1,3,2)
        attn=torch.softmax(logits.masked_fill(~victim_mask[:,:,None,:],-1.e4),dim=-1)
        attn=attn*victim_mask[:,:,None,:]
        context=torch.einsum('bthk,btkhd->bthd',attn,value.reshape(b,t,k,self.heads,d)).reshape(b,t,self.hidden)
        absolute=(value*mask).sum(dim=2)
        raw_edges=edges.sum(dim=2)
        predicted=self.scorer(torch.cat((s,context,absolute,raw_edges),dim=-1)).squeeze(-1)
        predicted=predicted.masked_fill(~valid,-1.e9)
        return torch.cat((torch.zeros_like(predicted[:,:1]),predicted),dim=1)

def choose_blank(chosen,scores,valid):
    scores=np.asarray(scores).reshape(52).copy()
    assert np.isfinite(scores).all()
    scores[0]=0.;scores[1:][~np.asarray(valid,dtype=bool)]=-1.e9
    index=int(np.argmax(scores));token=index-1 if scores[index]>1/3000 else -1
    action=np.asarray(chosen).copy()
    if token>=0:
        assert action.reshape(-1)[token]>=0
        action.reshape(-1)[token]=-1
    return action,token

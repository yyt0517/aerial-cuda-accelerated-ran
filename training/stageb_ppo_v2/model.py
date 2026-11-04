"""A small service-choice residual over frozen repair and physical proposals."""
import torch
from torch import nn
from .catalog import STATE_DIM,ACTION_DIM,SUMMARY_DIM


class ServiceActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.ue=nn.Sequential(nn.Linear(STATE_DIM,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU())
        self.joint=nn.Sequential(nn.Linear(32+ACTION_DIM,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(64+64+SUMMARY_DIM,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,state,candidate,summary,mask,prior):
        ue=self.ue(state)
        shared=torch.cat((ue.mean(1),ue.amax(1)),-1)
        expanded=ue[:,None].expand(-1,candidate.shape[1],-1,-1)
        local=self.joint(torch.cat((expanded,candidate),-1))
        pooled=torch.cat((local.mean(2),local.amax(2),shared[:,None].expand(-1,candidate.shape[1],-1),summary),-1)
        residual=self.head(pooled).squeeze(-1)
        return (prior+residual).masked_fill(~mask,-1.e9).log_softmax(-1)

    @torch.no_grad()
    def sample(self,features,*,greedy=False,generator=None):
        inputs=[torch.as_tensor(features[k])[None] for k in ['state','candidate','summary','mask','prior']]
        logp=self(*inputs)[0]
        action=logp.argmax() if greedy else torch.multinomial(logp.exp(),1,generator=generator)[0]
        return int(action),float(logp[action]),logp


class Value(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(STATE_DIM*2+SUMMARY_DIM),nn.Linear(STATE_DIM*2+SUMMARY_DIM,128),
            nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,state,summary,mask):
        opportunity=summary.masked_fill(~mask[...,None],-1.e9).amax(1)
        x=torch.cat((state.mean(1),state.amax(1),opportunity),-1)
        return self.net(x).squeeze(-1)

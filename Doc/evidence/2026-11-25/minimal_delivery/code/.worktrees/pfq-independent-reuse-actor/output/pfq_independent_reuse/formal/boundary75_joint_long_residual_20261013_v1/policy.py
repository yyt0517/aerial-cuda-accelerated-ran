"""Small residual over the existing causal whole-allocation search."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from output.pfq_independent_reuse.formal.boundary75_open_main_train_20261012_v1 import old_control as runner
from output.pfq_independent_reuse.formal.boundary75_old_joint_probe_20261013_v1.selector import candidates,select
OUT=Path(__file__).resolve().parent
P=json.loads((OUT/'protocol.json').read_text())
old,p,c,prior=runner.old,runner.p,runner.c,runner.prior
K=P['shortlist_nonkeep']+1

def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,tmp);tmp.replace(path)

def static_features(obs,ctbs,snapshot):
    u=np.asarray(obs['obs_ue_features']);q=snapshot['queue']
    x=np.stack([np.asarray(ctbs.wb_sinr_first_layer_db)/20,
        np.log1p(q/3000),np.clip(u[:,8]/200,0,2),np.where(u[:,9]>=0,np.clip(u[:,9]/200,0,2),1),
        np.log1p(np.maximum(u[:,1],0)/30),np.clip(u[:,11],-10,10)/10,
        np.arcsinh(np.asarray(ctbs.olla_delta)/5),np.asarray(ctbs.prior_tb_err)],axis=-1).astype(np.float32)
    assert x.shape==(36,8) and np.isfinite(x).all()
    return x

def value_state(obs,ctbs,snapshot):
    # MCS for actor comes from the actual causal PHY prediction; value uses no candidate.
    return static_features(obs,ctbs,snapshot).reshape(-1)

def materialize(draft,old_action,phy,obs,ctbs):
    actions,meta=candidates(draft,old_action,obs,phy.snapshot['queue'],True)
    chunks=[phy.evaluate(actions[i:i+256]) for i in range(0,len(actions),256)]
    pred={k:np.concatenate([d[k] for d in chunks]) for k in chunks[0]}
    totals=pred['expected'].sum(1);gain=totals-totals[0]
    deq=pred['dequeue'].sum(1);eligible=deq>=deq[0]
    base=int(np.argmax(np.where(eligible,gain,-np.inf))) if gain[eligible].max()>1 else 0
    order=np.argsort(-np.where(eligible,gain,-np.inf),kind='stable')
    ids=[0];seen={actions[0].tobytes()}
    for i in order:
        if not eligible[i]:break
        key=actions[i].tobytes()
        if key not in seen:ids.append(int(i));seen.add(key)
        if len(ids)==K:break
    assert any(np.array_equal(actions[base],actions[i]) for i in ids)
    features=np.zeros((K,36,13),np.float32);gains=np.zeros(K,np.float64);mask=np.zeros(K,bool)
    static=static_features(obs,ctbs,phy.snapshot);static[:,0]=phy.mcs/27
    selected={k:v[ids] for k,v in pred.items()}
    dynamic=np.stack([selected['count']/17,selected['layer']/4,selected['dequeue']/3000,
        selected['expected']/3000,np.where(selected['count']>0,selected['sinr_db'],0)/20],axis=-1)
    features[:len(ids)]=np.concatenate([np.broadcast_to(static,(len(ids),36,8)),dynamic],axis=-1)
    gains[:len(ids)]=gain[ids];mask[:len(ids)]=True
    assert np.isfinite(features).all() and np.isfinite(gains).all() and gains[0]==0 and mask[0]
    return dict(features=features,gain=gains,mask=mask,ids=ids,actions=actions[ids],pred=selected,
        base_action=actions[base],full_candidates=len(actions),eligible_candidates=int(eligible.sum()))

class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(13,32),nn.GELU(),nn.Linear(32,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        raw=(self.net(x).squeeze(-1)*x[...,10]).sum(-1)
        return P['residual_bound_bytes']*torch.tanh(raw-raw[...,:1])

class Value(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(36*8,64),nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x).squeeze(-1)

def logits(model,x,gain,mask):
    return ((gain+model(x))/P['temperature_bytes']).masked_fill(~mask,-1.e9)

def decide(model,d,*,sample=False,generator=None):
    with torch.no_grad():
        x=torch.from_numpy(d['features']);delta=model(x)
        score=d['gain']+delta.numpy()
        score[~d['mask']]=-np.inf
        if sample:
            logit=logits(model,x,torch.from_numpy(d['gain'].astype(np.float32)),torch.from_numpy(d['mask']))
            probs=logit.softmax(-1);index=int(torch.multinomial(probs,1,generator=generator))
            logp=float(logit.log_softmax(-1)[index])
        else:
            index=int(score.argmax()) if score.max()>1 else 0;probs=None;logp=None
    assert d['mask'][index] and delta[0]==0 and torch.isfinite(delta).all()
    return index,delta.numpy(),probs,logp

def check_initial():
    torch.set_num_threads(1);torch.manual_seed(P['initialization_seed']);model=Residual().eval()
    refs=[OUT.parent/'boundary75_independent_keep_longrank_20261013_v1'/f'collect_s{s}/root_{cat}.pt'
        for s,cat in [(101,'keep'),(102,'positive'),(103,'keep'),(104,'positive')]]
    rows=[]
    for file in refs:
        root=torch.load(file,weights_only=False)
        phy=old.CausalPhy(root['ctbs'].config);phy.prepare(root['snapshot'],root['ctbs'])
        d=materialize(root['draft'],root['old'],phy,root['obs'],root['ctbs'])
        chosen,delta,_,_=decide(model,d)
        actual,info=select(root['draft'],root['old'],phy,root['obs'],True)
        assert np.array_equal(d['actions'][chosen],actual) and np.array_equal(d['base_action'],actual)
        assert np.count_nonzero(delta)==0
        for action in d['actions']:prior.encode(action,root['obs'])
        rows.append(dict(root=str(file),full_candidates=d['full_candidates'],shortlist=len(d['ids']),keep=chosen==0))
    write(OUT/'initial_verification.json',dict(passed=True,zero_residual_exact_search_actions=True,roots=rows,
        residual_parameters=sum(v.numel() for v in model.parameters()),value_parameters=sum(v.numel() for v in Value().parameters())))
    print('ZERO_RESIDUAL_SEARCH_EQUIVALENCE_VERIFIED',flush=True)

if __name__=='__main__':check_initial()

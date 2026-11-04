"""Closed-loop dataset aggregation and deadline/weak-UE preference distillation.

Observed delayed rewards prioritize correction of failed behavior states. Local
counterfactual preferences use a causal, explicitly approximate risk objective;
they are not claimed to be observed long-horizon counterfactual returns or PPO.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
from .archive import problem_from_row,initial_draft
from .closed_data import dataset
from .deadline import deadline_features,components
from .neural_repair import options,destroy
from .physics import NativePhy,EXPECTED,DEQUEUE
from .search import SearchConfig,ProposalGraph,repaired_candidates,solve
from .quality import FULL
from .learning import block_features
from .experiment import write_json,sha,snapshot_sources,ROOT


def collect(rows,blocks_per_root):
    cases=[];meta=[]
    cfg=SearchConfig(budget_ms=None,ue_topk=36,repair_width=16,max_candidates=1000000,max_rounds=256)
    for ri,row in enumerate(rows):
        p=problem_from_row(row);draft=initial_draft(p,row['obs'])
        teacher,_,_=solve(p,draft,FULL)
        phy=NativePhy(p)
        try:
            graph=ProposalGraph(phy);original=phy.evaluate(draft)
            floor=(90*int(original.values[0,:,DEQUEUE].sum())+99)//100
            for root_index,root in enumerate((draft,row['chosen'],teacher)):
                root=np.asarray(root,dtype=np.int64)
                base=phy.evaluate(root);raw,utility,*_=components(p,base.values)
                if base.values[0,:,DEQUEUE].sum()<floor:continue
                rng=np.random.default_rng(row['seed']*100000+row['tti']+root_index)
                blocks=graph.blocks(root,base.values[0],cfg,rng)
                count=min(blocks_per_root,len(blocks));primary=count//2
                indices=list(range(primary))+rng.choice(np.arange(primary,len(blocks)),count-primary,replace=False).tolist()
                for bi in indices:
                    block=blocks[bi]
                    pool={root.tobytes():(float(raw[0]),float(utility[0]),True,root.copy())}
                    def add(result):
                        j,u,*_=components(p,result.values)
                        feasible=(result.values[:,:,DEQUEUE].sum(1)>=floor)&(j>=raw[0]-1.e-7)
                        for i,a in enumerate(result.actions):pool[a.tobytes()]=(float(j[i]),float(u[i]),bool(feasible[i]),a.copy())
                    def evaluate(actions,source):
                        result=phy.evaluate(actions,source);add(result);return result
                    guided=root.copy()
                    for c,prg in block:guided[c,prg]=teacher[c,prg]
                    add(phy.evaluate(guided,base))
                    for _ in repaired_candidates(graph,base,block,cfg,rng,evaluate):pass
                    ranked=sorted(pool.values(),key=lambda a:-(a[1] if a[2] else -1.e12))
                    choices=[ranked[0],ranked[min(2,len(ranked)-1)],ranked[min(len(ranked)-1,max(3,len(ranked)//2))]]
                    context=block_features(graph,base,[block])[0]
                    paths=[];masks=[];targets=[]
                    for j,u,valid,target in choices:
                        partial=destroy(root,block);xx=[];mm=[];yy=[]
                        for depth,(c,prg) in enumerate(block):
                            legal=options(p,c,prg)
                            x=deadline_features(graph,base,block,partial[None],depth,legal,context)[0]
                            padded=np.zeros((p.U+1,87),np.float32);padded[:len(legal)]=x
                            mask=np.zeros(p.U+1,bool);mask[:len(legal)]=True
                            xx.append(padded);mm.append(mask);yy.append(int(np.flatnonzero(legal==target[c,prg])[0]))
                            partial[c,prg]=target[c,prg]
                        np.testing.assert_array_equal(partial,target)
                        while len(xx)<4:
                            xx.append(np.zeros_like(xx[0]));mask=np.zeros_like(mm[0]);mask[0]=True
                            mm.append(mask);yy.append(-100)
                        paths.append(xx);masks.append(mm);targets.append(yy)
                    rewards=np.array([u-utility[0] if valid else -3000. for j,u,valid,a in choices],np.float32)
                    valid=np.array([a[2] for a in choices]);logits=np.where(valid,rewards/3000.,-1.e9)
                    q=np.exp(logits-logits.max());q/=q.sum()
                    observed=row['native_return_components'].mean(0)
                    priority=1+min(3.,max(0.,-row['native_reward'])+2*observed[1]+.5*observed[2])
                    weight=priority*(.25+min(4.,max(0.,float(rewards[0]))/3000.))
                    cases.append(dict(x=np.asarray(paths,np.float32),mask=np.asarray(masks),labels=np.asarray(targets),
                        q=q,rewards=rewards,weight=weight))
                    meta.append(dict(seed=row['seed'],tti=row['tti'],source=row['source'],root_index=root_index,
                        native_reward=row['native_reward'],native_expiry_return=float(observed[1]),
                        native_weak_cost=float(observed[2]),teacher_gain=float(rewards[0]),case_priority=priority))
        finally:phy.close()
        if ri%10==0:print('DEADLINE_LABELS',ri,len(rows),len(cases),flush=True)
    packed={k:np.asarray([x[k] for x in cases]) for k in cases[0]}
    return packed,meta


def main(args):
    import torch
    import pandas as pd
    torch.set_num_threads(1);torch.manual_seed(2801)
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    sources=snapshot_sources(out)
    train,train_sources=dataset(args.collection,(46,47,49))
    valid,valid_sources=dataset(args.collection,(48,50))
    write_json(out/'protocol.json',dict(training_seeds=[41,42,43,46,47,49],validation_seeds=[44,45,48,50],
        final_test_seeds=[101,102,103],initial_sha256=sha(args.initial),training_states=len(train),validation_states=len(valid),
        training_sources=train_sources,validation_sources=valid_sources,blocks_per_root=args.blocks,
        source_sha256=sources,feature_version=2,features=87,teacher_topk=36,teacher_beam=16,
        return_horizon_tti=400,gamma=.9975,
        observed_reward='mean(goodput/2400 - 2*expired_full_packet_payload/2400 - 0.5*post_service_deficit_squared)',
        local_proxy='J + 0.5*urgent_expected - 0.5*urgent_service_shortfall + 0.5*weak_capped_expected',
        loss='failure-priority weighted [0.25*best complete-sequence NLL + reward-softened candidate preference CE]',
        selection='validation physically scored candidate preference; not imitation NLL',
        limitation='Observed reward reweights corrective imitation cases. Counterfactual preferences are local risk proxies, not observed long-horizon returns.'))
    a,ma=collect(train,args.blocks);b,mb=collect(valid,args.blocks)
    pd.DataFrame(ma).to_csv(out/'training_cases.csv',index=False);pd.DataFrame(mb).to_csv(out/'validation_cases.csv',index=False)
    np.savez_compressed(out/'training.npz',**a);np.savez_compressed(out/'validation.npz',**b)
    with np.load(args.initial) as old:
        mean=np.zeros(87,np.float32);scale=np.ones(87,np.float32);mean[:79]=old['mean'];scale[:79]=old['scale']
        flat=a['x'][a['mask']&(a['labels']!=-100)[...,None]][:,-8:]
        mean[-8:]=flat.mean(0);scale[-8:]=np.maximum(flat.std(0),.1)
        model=torch.nn.Sequential(torch.nn.Linear(87,64),torch.nn.ReLU(),torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
        with torch.no_grad():
            model[0].weight.zero_();model[0].weight[:,:79].copy_(torch.from_numpy(old['w1']))
            for layer,key in ((0,'b1'),(2,'b2'),(4,'b3')):model[layer].bias.copy_(torch.from_numpy(old[key]))
            model[2].weight.copy_(torch.from_numpy(old['w2']));model[4].weight.copy_(torch.from_numpy(old['w3']))
    device=torch.device(args.device);model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    def tensors(data):
        return tuple(torch.as_tensor(v,device=device) for v in (
            np.clip((data['x']-mean)/scale,-10,10),data['mask'],data['labels'],data['q'],data['rewards'],data['weight'].astype(np.float32)))
    ta,tb=tensors(a),tensors(b);rng=np.random.default_rng(2801);history=[];best=-np.inf
    def forward(data,index):
        x,mask,target,q,rewards,weight=(v[index] for v in data)
        logits=model(x).squeeze(-1).masked_fill(~mask,-1.e9)
        lp=torch.log_softmax(logits,dim=-1).gather(-1,target.clamp_min(0)[...,None])[...,0]
        seq=(lp*(target!=-100)).sum(-1)
        ce=-seq[:,0];preference=-(q*torch.log_softmax(seq,dim=-1)).sum(-1)
        loss=((.25*ce+preference)*weight).sum()/weight.sum().clamp_min(1.e-8)
        selected=rewards.gather(1,seq.argmax(1)[:,None])[:,0]
        return loss,selected
    for epoch in range(args.epochs):
        permutation=rng.permutation(len(a['x']));losses=[]
        for start in range(0,len(permutation),64):
            ix=torch.as_tensor(permutation[start:start+64],device=device)
            loss,_=forward(ta,ix);optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        with torch.no_grad():
            selected=torch.cat([forward(tb,slice(i,i+64))[1] for i in range(0,len(b['x']),64)])
            score=float(selected.mean())
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation_selected_proxy_gain=score))
        if score>best:
            best=score;best_epoch=epoch
            np.savez(out/'repair.npz',feature_version=2,mean=mean,scale=scale,
                w1=model[0].weight.detach().cpu().numpy(),b1=model[0].bias.detach().cpu().numpy(),
                w2=model[2].weight.detach().cpu().numpy(),b2=model[2].bias.detach().cpu().numpy(),
                w3=model[4].weight.detach().cpu().numpy(),b3=model[4].bias.detach().cpu().numpy())
        if epoch%5==0:print('DEADLINE_TRAIN',history[-1],flush=True)
    for relative,digest in sources.items():
        if sha(ROOT/relative)!=digest:raise RuntimeError('Training source changed: '+relative)
    write_json(out/'training.json',dict(training_states=len(train),validation_states=len(valid),
        training_cases=len(ma),validation_cases=len(mb),parameters=sum(p.numel() for p in model.parameters()),
        best_epoch=best_epoch,best_validation_proxy_gain=best,history=history,checkpoint_sha256=sha(out/'repair.npz')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--collection',required=True)
    p.add_argument('--initial',required=True);p.add_argument('--blocks',type=int,default=8)
    p.add_argument('--epochs',type=int,default=60);p.add_argument('--device',default='cuda');main(p.parse_args())

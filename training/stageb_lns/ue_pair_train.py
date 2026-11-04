"""Deduplicated own-archive training, within-state pairs and seed-held-out scale."""
import hashlib
import time
import numpy as np
from .physics import EXPECTED,DEQUEUE
from .resource_conditions import satisfies
from .resource_condition_campaign import compressed_condition
from .no_ctbs_inputs import problem_from_raw_row
from .ranked_dpls_campaign import identity
from .shortlist_trajectory import service_metrics
from .terminal_risk import RISK_CAP
from .ue_pair_risk import ue_features,service_scores,heuristic_scores,UEPairModel,CONTEXT_NAMES,rank_select
from .experiment import write_json,sha
from .bounded_probes import csv_write


def make_pairs(j,risk,seed):
    allowed=np.flatnonzero(j+1.e-8>=(1-RISK_CAP)*j.max());a,b=np.triu_indices(len(allowed),1);a=allowed[a];b=allowed[b]
    delta=risk[b]-risk[a];strict=abs(delta)>.005;ties=abs(delta)<=1.e-4;rng=np.random.default_rng(seed);result=[]
    for mask,target in ((strict,1.),(ties,.5)):
        ids=np.flatnonzero(mask);base=int(np.argmax(j));priority=ids[(a[ids]==base)|(b[ids]==base)]
        rest=np.setdiff1d(ids,priority,assume_unique=True);count=max(0,1024-len(priority))
        pick=np.r_[priority,rng.choice(rest,min(count,len(rest)),replace=False)];aa=a[pick].copy();bb=b[pick].copy()
        if target==1.:
            flip=risk[aa]>risk[bb];aa[flip],bb[flip]=bb[flip].copy(),aa[flip].copy()
        result.extend(zip(aa.tolist(),bb.tolist(),[target]*len(pick)))
    return np.asarray(result,dtype=float).reshape(-1,3)


def prepare(rows,prior,out):
    values=np.load(prior/'training_observed_values.npz');actions=np.load(prior/'training_observed_actions.npz')
    tv=np.load(prior/'training_teacher_values.npz');ta=np.load(prior/'training_teacher_actions.npz')
    data=[];seen={};duplicates=[];pairs=[];arrays={}
    for row in rows:
        assert row['seed'] in (41,42,43)
        sid=identity(row);p=problem_from_raw_row(row);ref=tv[sid];c=compressed_condition(p,ta[sid],ref,'joint212');vv=values[sid];aa=actions[sid]
        # Immutable inputs plus all archive/reference bytes; identity is excluded.
        digest=hashlib.sha256()
        for x in (p.h,p.w,p.queue,p.serving,p.mcs,p.caps,p.tbs,p.weights,p.legal,aa,vv,ref):digest.update(x.tobytes())
        digest.update(np.array([p.noise]).tobytes());key=digest.hexdigest()
        if key in seen:
            assert seen[key]['seed']==row['seed'];duplicates.append(dict(duplicate=sid,kept=seen[key]['state_id'],sha256=key));continue
        seen[key]=dict(state_id=sid,seed=row['seed'])
        # Prior raw root supplies exactly the unchanged own-PFQ floor.
        from .no_ctbs_inputs import raw_pfq_floor
        floor=raw_pfq_floor(vv[0]);keep=satisfies(p,aa,c)&(vv[:,:,DEQUEUE].sum(1)>=floor);vv=vv[keep]
        f=ue_features(p,c,vv);metrics=[service_metrics(v,ref) for v in vv];jr=f['j']/float(ref[:,EXPECTED]@p.weights)
        p5=np.array([m['positive_p5_ratio'] for m in metrics]);weak=np.array([m['positive_weak_ratio'] for m in metrics])
        risk=10*np.maximum(.995-jr,0)+np.maximum(.98-p5,0)+np.maximum(.98-weak,0)
        good=(jr>=.995)&(p5>=.98)&(weak>=.98)
        d=vv[0,:,DEQUEUE];target=np.clip(ref[:,EXPECTED]/np.maximum(d,1.e-12),0,1);effective=(ref[:,EXPECTED]>1).astype(float)
        item=dict(**f,state_id=sid,seed=row['seed'],risk=risk,good=good,j_ratio=jr,p5=p5,weak=weak,
            target=target,effective=effective,memo_index=np.flatnonzero(keep),fingerprint=key)
        item['pairs']=make_pairs(f['j'],risk,20261103+len(data));data.append(item)
        for a,b,t in item['pairs']:pairs.append(dict(state_id=sid,seed=row['seed'],preferred=int(a),other=int(b),target=float(t)))
        for name in ('context','losses','r','group','j','risk','good','j_ratio','p5','weak','target','effective','memo_index','pairs'):
            arrays[f'{sid}|{name}']=item[name]
    np.savez_compressed(out/'training_data.npz',**arrays);csv_write(out/'training_pairs.csv',pairs)
    files=('training_observed_values.npz','training_observed_actions.npz','training_teacher_values.npz','training_teacher_actions.npz','training_input_manifest.json','source_manifest.json')
    write_json(out/'training_reuse.json',dict(source=str(prior),files={f:sha(prior/f) for f in files},
        original_states=len(rows),unique_states=len(data),duplicates=duplicates,candidates=sum(len(x['j']) for x in data),
        pairs=len(pairs),strict_pairs=sum(x['target']==1 for x in pairs),tie_pairs=sum(x['target']==.5 for x in pairs),
        additional_training_native_scores=0,reuse_reason='terminal choices do not alter cached search; all visited archive remains on-policy for this frozen search',
        prior_teacher_cost_not_free_or_online=True))
    write_json(out/'ue_context_names.json',CONTEXT_NAMES)
    return data


def fit(data,initialization):
    import torch
    torch.set_num_threads(1);torch.manual_seed(initialization)
    x=np.stack([d['context'] for d in data]);g=np.stack([d['group'] for d in data]);active=x[g]
    mean=active.mean(0);scale=np.maximum(active.std(0),.05)
    tensor=lambda a:torch.tensor(a,dtype=torch.float64)
    xx=tensor(np.clip((x-mean)/scale,-8,8));group=tensor(g);r=tensor(np.stack([d['r'] for d in data]))
    loss=tensor(np.stack([d['losses'] for d in data]));effective=tensor(np.stack([d['effective'] for d in data]));target=tensor(np.stack([d['target'] for d in data]))
    n=r.shape[1];left=[];right=[];yy=[];ww=[]
    for s,d in enumerate(data):
        pair=d['pairs'];left.extend((s*n+pair[:,0]).astype(int));right.extend((s*n+pair[:,1]).astype(int))
        yy.extend(pair[:,2]);ww.extend([1/max(len(pair),1)/len(data)]*len(pair))
    left=torch.tensor(left,dtype=torch.long);right=torch.tensor(right,dtype=torch.long);yy=tensor(yy);ww=tensor(ww)
    net=torch.nn.Sequential(torch.nn.Linear(x.shape[-1],24),torch.nn.Tanh(),torch.nn.Linear(24,10)).double()
    opt=torch.optim.Adam(net.parameters(),lr=.01,weight_decay=.001)
    history=[]
    for epoch in range(240):
        opt.zero_grad();z=net(xx);weights=2*torch.sigmoid(z[:,:,:8]);eff=torch.sigmoid(z[:,:,8]);ref=torch.sigmoid(z[:,:,9])
        bases=torch.cat([loss,torch.relu(ref[:,None]-r)[:,:,:,None]],dim=-1)
        contributions=(bases*weights[:,None]).mean(-1)*(.1+.9*eff[:,None])*group[:,None]
        scores=-contributions.sum(-1)/group.sum(-1).clamp_min(1)[:,None];flat=scores.ravel()
        rank=(torch.nn.functional.binary_cross_entropy_with_logits(20*(flat[left]-flat[right]),yy,reduction='none')*ww).sum()
        auxiliary=((torch.nn.functional.binary_cross_entropy_with_logits(z[:,:,8],effective,reduction='none')+(ref-target)**2)*group).sum()/group.sum().clamp_min(1)
        objective=rank+.2*auxiliary;objective.backward();opt.step()
        if epoch in (0,239):history.append(dict(epoch=epoch+1,rank_loss=float(rank.detach()),auxiliary_loss=float(auxiliary.detach())))
    arrays=dict(mean=mean,scale=scale,w0=net[0].weight.detach().numpy().T.copy(),b0=net[0].bias.detach().numpy().copy(),
        w1=net[2].weight.detach().numpy().T.copy(),b1=net[2].bias.detach().numpy().copy())
    return arrays,history


def predict(arrays,item):
    a=arrays;z=np.tanh(np.clip((item['context']-a['mean'])/a['scale'],-8,8)@a['w0']+a['b0'])
    return service_scores(item,z@a['w1']+a['b1'])


def calibrate(data,scores):
    import torch
    delta=[];target=[];weight=[]
    for d,s in zip(data,scores):
        pp=d['pairs'];a=pp[:,0].astype(int);b=pp[:,1].astype(int)
        delta.extend(s[a]-s[b]);target.extend(pp[:,2]);weight.extend([1/max(len(pp),1)/len(data)]*len(pp))
    dd=torch.tensor(delta,dtype=torch.float64);yy=torch.tensor(target,dtype=torch.float64);ww=torch.tensor(weight,dtype=torch.float64)
    raw=torch.tensor(3.,dtype=torch.float64,requires_grad=True);opt=torch.optim.LBFGS([raw],lr=.5,max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();a=torch.nn.functional.softplus(raw);loss=(torch.nn.functional.binary_cross_entropy_with_logits(a*dd,yy,reduction='none')*ww).sum()+1.e-5*a*a
        loss.backward();return loss
    opt.step(closure);return float(torch.nn.functional.softplus(raw).detach())


def train(data,out):
    assert {d['seed'] for d in data}=={41,42,43}
    started=time.perf_counter();oof=[None]*len(data);folds=[];foldarrays={}
    for hold in (41,42,43):
        trainset=[d for d in data if d['seed']!=hold];model,history=fit(trainset,20261103)
        for i,d in enumerate(data):
            if d['seed']==hold:oof[i]=predict(model,d)
        folds.append(dict(held_seed=hold,training_states=[d['state_id'] for d in trainset],history=history))
        foldarrays.update({f'{hold}_{k}':v for k,v in model.items()})
    alpha=calibrate(data,oof);heuristic_alpha=calibrate(data,[heuristic_scores(d) for d in data])
    arrays=dict(alpha=np.array(alpha),heuristic_alpha=np.array(heuristic_alpha));final=[]
    for i,initialization in enumerate((20261103,20261104,20261105)):
        model,history=fit(data,initialization);final.append(dict(initialization=initialization,history=history))
        for k,v in model.items():
            name=f'{k}{i}' if k in ('mean','scale') else f'{k[0]}{i}_{k[1]}'
            arrays[name]=v
    np.savez_compressed(out/'ue_pair_model.npz',**arrays);np.savez_compressed(out/'oof_models.npz',**foldarrays)
    np.savez_compressed(out/'oof_scores.npz',**{d['state_id']:s for d,s in zip(data,oof)})
    checks=[];selected=[]
    for d,s in zip(data,oof):
        pair=d['pairs'];strict=pair[:,2]==1;a=pair[strict,0].astype(int);b=pair[strict,1].astype(int)
        checks.append(dict(state_id=d['state_id'],seed=d['seed'],strict_pairs=len(a),pair_accuracy=float(np.mean((s[a]>s[b])+.5*(s[a]==s[b]))) if len(a) else None))
        for guard in (False,True):
            i,r=rank_select(d['j'],s,alpha,guard);selected.append(dict(state_id=d['state_id'],guard=guard,**r,joint_pass=bool(d['good'][i]),
                j_ratio=float(d['j_ratio'][i]),p5_ratio=float(d['p5'][i]),weak_ratio=float(d['weak'][i])))
    csv_write(out/'training_oof_pair_accuracy.csv',checks);csv_write(out/'training_oof_selection.csv',selected)
    write_json(out/'training_summary.json',dict(unique_states=len(data),candidates=sum(len(d['j']) for d in data),
        architecture=[20,24,10],shared_per_ue=True,monotone_candidate_loss=True,epochs_per_fit=240,model_fits=6,gradient_steps=1440,
        auxiliary='0.2*(UE reference-positive BCE + reference E/D MSE), training-only full targets',
        risk_label='10*relu(.995-J/fullJ)+relu(.98-P5ratio)+relu(.98-fixedweakratio)',
        pair_rule='within each state own2%Jwindow;strict risk gap>.005, ties<=.0001;up to1024 each, prioritize maximumJ comparisons;ambiguous ignored',
        alpha=alpha,heuristic_alpha=heuristic_alpha,threshold=.9,folds=folds,final_models=final,
        confidence='minimum calibrated pair-win probability across3 initializations versus same maximumJ incumbent; not an absolute feasibility guarantee',
        calibration_diagnostic_not_independent=True,elapsed_s=time.perf_counter()-started,validation_read=False,development_read=False))
    return UEPairModel(arrays)

"""Expanded-data controlled training: ordinary versus repair-directed pairs."""
import time
import numpy as np
from .physics import EXPECTED
from .ue_pair_train import predict
from .ue_pair_risk import heuristic_scores,rank_select,UEPairModel
from .experiment import write_json
from .bounded_probes import csv_write


def padded(data,name):
    n=max(len(d[name]) for d in data);shape=(len(data),n,*data[0][name].shape[1:]);out=np.zeros(shape)
    for i,d in enumerate(data):out[i,:len(d[name])]=d[name]
    return out


def cohort(d):
    b=int(np.argmax(d['j']));allowed=d['j']+1.e-8>=.98*d['j'][b]
    if d['good'][b]:return 'baseline_good'
    if (d['good']&allowed).any():return 'repairable'
    return 'unrepairable'


def cohort_weights(data):
    labels=[cohort(d) for d in data];available=[k for k in ('repairable','baseline_good') if k in labels]
    if not available:return np.full(len(data),1/len(data))
    return np.array([1/len(available)/labels.count(k) if k in available else 0. for k in labels])


def auxiliary_weights(data):
    out=[]
    for d in data:
        group=d['group'];b=int(np.argmax(d['j']))
        missed=group&(d['effective']>0)&(d['r'][b]<.98*d['target'])
        rest=group&~missed;w=np.zeros(len(group));parts=[q for q in (missed,rest) if q.any()]
        for q in parts:w[q]=1/len(parts)/q.sum()/len(data)
        out.append(w)
    return np.stack(out)


def focused_pairs(d):
    j=d['j'];base=int(np.argmax(j));allowed=np.flatnonzero(j+1.e-8>=.98*j[base]);good=allowed[d['good'][allowed]]
    bad=allowed[~d['good'][allowed]];pairs=[]
    if not d['good'][base] and len(good):
        # Keep every qualified repair against the failed incumbent, then against
        # at most7 other highest-J bad candidates. No ordinary bad/bad pairs.
        negatives=[base]+[int(i) for i in sorted(bad,key=lambda i:j[i],reverse=True) if i!=base][:7]
        pairs=[(int(a),int(b),1.) for a in good for b in negatives]
    elif d['good'][base]:
        pairs=[(base,int(b),1.) for b in bad]
        pairs.extend((base,int(a),.5) for a in good if a!=base)
    # No feasible repair: no pair loss; retain UE auxiliary supervision only.
    return np.asarray(pairs,dtype=float).reshape(-1,3)


def set_pairs(data,focused,out):
    result=[];records=[]
    for d in data:
        v=dict(d);v['pairs']=focused_pairs(d) if focused else d['pairs'].copy();result.append(v)
        for a,b,t in v['pairs']:records.append(dict(state_id=d['state_id'],seed=d['seed'],cohort=cohort(d),preferred=int(a),other=int(b),target=float(t)))
    csv_write(out/'pairs.csv',records)
    write_json(out/'data_role.json',dict(focused=focused,states=[dict(state_id=d['state_id'],seed=d['seed'],cohort=cohort(d),pairs=len(d['pairs'])) for d in result],
        total_pairs=len(records),no_validation_or_development_labels=True))
    return result


def fit(data,initialization,focused=False):
    import torch
    torch.set_num_threads(1);torch.manual_seed(initialization)
    x=np.stack([d['context'] for d in data]);g=np.stack([d['group'] for d in data]);active=x[g]
    mean=active.mean(0);scale=np.maximum(active.std(0),.05)
    tensor=lambda a:torch.tensor(a,dtype=torch.float64)
    xx=tensor(np.clip((x-mean)/scale,-8,8));group=tensor(g);r=tensor(padded(data,'r'))
    loss=tensor(padded(data,'losses'));effective=tensor(np.stack([d['effective'] for d in data]));target=tensor(np.stack([d['target'] for d in data]))
    state_weights=cohort_weights(data) if focused else np.full(len(data),1/len(data))
    aux_weights=auxiliary_weights(data) if focused else g/max(g.sum(),1)
    aux_weights=tensor(aux_weights)
    n=r.shape[1];left=[];right=[];yy=[];ww=[]
    for s,d in enumerate(data):
        pair=d['pairs'];left.extend((s*n+pair[:,0]).astype(int));right.extend((s*n+pair[:,1]).astype(int))
        yy.extend(pair[:,2]);ww.extend([state_weights[s]/max(len(pair),1)]*len(pair))
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
        auxiliary=((torch.nn.functional.binary_cross_entropy_with_logits(z[:,:,8],effective,reduction='none')+(ref-target)**2)*aux_weights).sum()
        objective=rank+.2*auxiliary;objective.backward();opt.step()
        if epoch in (0,239):history.append(dict(epoch=epoch+1,rank_loss=float(rank.detach()),auxiliary_loss=float(auxiliary.detach())))
    arrays=dict(mean=mean,scale=scale,w0=net[0].weight.detach().numpy().T.copy(),b0=net[0].bias.detach().numpy().copy(),
        w1=net[2].weight.detach().numpy().T.copy(),b1=net[2].bias.detach().numpy().copy())
    return arrays,history

def calibrate(data,scores,focused=False):
    import torch
    delta=[];target=[];weight=[]
    sw=cohort_weights(data) if focused else np.full(len(data),1/len(data))
    for i,(d,s) in enumerate(zip(data,scores)):
        pp=d['pairs'];a=pp[:,0].astype(int);b=pp[:,1].astype(int)
        delta.extend(s[a]-s[b]);target.extend(pp[:,2]);weight.extend([sw[i]/max(len(pp),1)]*len(pp))
    dd=torch.tensor(delta,dtype=torch.float64);yy=torch.tensor(target,dtype=torch.float64);ww=torch.tensor(weight,dtype=torch.float64)
    raw=torch.tensor(3.,dtype=torch.float64,requires_grad=True);opt=torch.optim.LBFGS([raw],lr=.5,max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();a=torch.nn.functional.softplus(raw);loss=(torch.nn.functional.binary_cross_entropy_with_logits(a*dd,yy,reduction='none')*ww).sum()+1.e-5*a*a
        loss.backward();return loss
    opt.step(closure);return float(torch.nn.functional.softplus(raw).detach())

def train(data,out,focused=False):
    assert {d['seed'] for d in data}=={41,42,43}
    started=time.perf_counter();oof=[None]*len(data);folds=[];foldarrays={}
    for hold in (41,42,43):
        trainset=[d for d in data if d['seed']!=hold];model,history=fit(trainset,20261103,focused)
        for i,d in enumerate(data):
            if d['seed']==hold:oof[i]=predict(model,d)
        folds.append(dict(held_seed=hold,training_states=[d['state_id'] for d in trainset],history=history))
        foldarrays.update({f'{hold}_{k}':v for k,v in model.items()})
    alpha=calibrate(data,oof,focused);heuristic_alpha=calibrate(data,[heuristic_scores(d) for d in data],focused)
    arrays=dict(alpha=np.array(alpha),heuristic_alpha=np.array(heuristic_alpha));final=[]
    for i,initialization in enumerate((20261103,20261104,20261105)):
        model,history=fit(data,initialization,focused);final.append(dict(initialization=initialization,history=history))
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
    write_json(out/'training_summary.json',dict(focused=focused,unique_states=len(data),candidates=sum(len(d['j']) for d in data),
        architecture=[20,24,10],shared_per_ue=True,monotone_candidate_loss=True,epochs_per_fit=240,model_fits=6,gradient_steps=1440,
        auxiliary='0.2*(UE reference-positive BCE + reference E/D MSE), training-only full targets',
        risk_label='10*relu(.995-J/fullJ)+relu(.98-P5ratio)+relu(.98-fixedweakratio)',
        pair_rule='focused repair anchors and nonharmful controls' if focused else 'original ordinary pair sampling',
        state_weights=cohort_weights(data).tolist() if focused else [1/len(data)]*len(data),
        alpha=alpha,heuristic_alpha=heuristic_alpha,threshold=.9,folds=folds,final_models=final,
        confidence='minimum calibrated pair-win probability across3 initializations versus same maximumJ incumbent; not an absolute feasibility guarantee',
        calibration_diagnostic_not_independent=True,elapsed_s=time.perf_counter()-started,validation_read=False,development_read=False))
    return UEPairModel(arrays)

"""Input ablations on archived train/validation only; never select on native tests.

Compact models are zero-padded into the fixed inference interface for controlled
quality comparisons. This is not a claim that feature extraction became cheaper.
"""
import argparse
from pathlib import Path
import numpy as np
from .experiment import write_json,sha
from .archive import load_rows,problem_from_row,initial_draft
from .neural_repair import JointRepair,REPAIR_FEATURE_VERSION
from .quality import select


def main(args):
    import torch
    import pandas as pd
    torch.set_num_threads(1)
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    source=Path(args.training).resolve()
    a=dict(np.load(source/'training.npz'));b=dict(np.load(source/'validation.npz'))
    schema={
        'local27':np.arange(52,79),
        'directed31':np.r_[np.arange(47,51),np.arange(52,79)],
        'compact14':np.array([52,53,54,55,58,61,64,67,68,71,73,74,75,77]),
    }
    write_json(output/'protocol.json',dict(training_sha256=sha(source/'training.npz'),
        validation_sha256=sha(source/'validation.npz'),training_seeds=[41,42,43],validation_seeds=[44,45],
        feature_indices={k:v.tolist() for k,v in schema.items()},
        checkpoint_selection='validation sequence NLL only',native_policy_unchanged=True,
        padding='Zero-pad first-layer weights into 79-dimensional inference interface; no speed claim'))
    metadata=[];measurements=[]
    device=torch.device(args.device)
    for name,indices in schema.items():
        torch.manual_seed(3711);rng=np.random.default_rng(3711)
        flat=a['x'][a['mask'] & (a['labels'] != -100)[:,:,None]][:,indices]
        mean=flat.mean(0);scale=np.maximum(flat.std(0),.1)
        model=torch.nn.Sequential(torch.nn.Linear(len(indices),64),torch.nn.ReLU(),
            torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Linear(32,1)).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        def tensor(data):
            return (torch.from_numpy(np.clip((data['x'][...,indices]-mean)/scale,-10,10)).to(device),
                    torch.from_numpy(data['mask']).to(device),torch.from_numpy(data['labels']).to(device),
                    torch.from_numpy(data['weight']).to(device))
        train,valid=tensor(a),tensor(b)
        def loss(data,index):
            x,mask,target,weight=(v[index] for v in data)
            logits=model(x).squeeze(-1).masked_fill(~mask,-1.e9)
            ce=torch.nn.functional.cross_entropy(logits.flatten(0,1),target.flatten(),reduction='none').reshape(target.shape).sum(1)
            return (ce*weight).sum()/weight.sum().clamp_min(1.e-8)
        best=float('inf');history=[];checkpoint=output/(name+'.npz')
        for epoch in range(args.epochs):
            permutation=rng.permutation(len(a['x']))
            for start in range(0,len(permutation),128):
                current=loss(train,torch.as_tensor(permutation[start:start+128],device=device))
                optimizer.zero_grad();current.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
            with torch.no_grad():
                metric=float(np.mean([float(loss(valid,slice(i,i+128))) for i in range(0,len(b['x']),128)]))
            history.append(metric)
            if metric < best:
                best=metric;best_epoch=epoch
                full_mean=np.zeros(79,np.float32);full_scale=np.ones(79,np.float32)
                full_mean[indices]=mean;full_scale[indices]=scale
                w1=np.zeros((64,79),np.float32);w1[:,indices]=model[0].weight.detach().cpu().numpy()
                np.savez(checkpoint,feature_version=REPAIR_FEATURE_VERSION,mean=full_mean,scale=full_scale,
                    w1=w1,b1=model[0].bias.detach().cpu().numpy(),w2=model[2].weight.detach().cpu().numpy(),
                    b2=model[2].bias.detach().cpu().numpy(),w3=model[4].weight.detach().cpu().numpy(),
                    b3=model[4].bias.detach().cpu().numpy(),feature_indices=indices)
        metadata.append(dict(name=name,features=len(indices),trained_parameters=sum(p.numel() for p in model.parameters()),
            best_epoch=best_epoch,best_validation_joint_nll=best,history=history,checkpoint_sha256=sha(checkpoint)))
        repairer=JointRepair(checkpoint)
        for row in load_rows((44,45)):
            p=problem_from_row(row);draft=initial_draft(p,row['obs'])
            _,_,result=select(p,draft,'neural',repairer)
            measurements.append(dict(result,name=name,seed=row['seed'],tti=row['tti']))
        print('ABLATION',name,best,flush=True)
    write_json(output/'training.json',metadata)
    pd.DataFrame(measurements).to_csv(output/'measurements.csv',index=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--training',required=True);parser.add_argument('--device',default='cpu')
    parser.add_argument('--epochs',type=int,default=60)
    main(parser.parse_args())

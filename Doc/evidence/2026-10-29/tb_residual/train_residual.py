"""Fit sparse UE expected-byte residuals; select on seed44 only."""
from pathlib import Path
import sys,json,shutil,time
import numpy as np
import pandas as pd
import torch
from torch import nn

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from common import load,targets,metrics,residual_scores
from training.stageb_lns.residual import ResidualValue
from training.stageb_lns.experiment import sha,write_json

def export(path,models,mean,scale,mode):
    data=dict(version=1,mode=mode,mean=mean,scale=scale)
    for key,layer,attr in [('w1',0,'weight'),('b1',0,'bias'),('w2',2,'weight'),('b2',2,'bias'),('w3',4,'weight'),('b3',4,'bias')]:
        data[key]=np.stack([getattr(m[layer],attr).detach().numpy() for m in models])
    np.savez(path,**data)

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    plan=json.loads((HERE/'plan.json').read_text());out=HERE/'residual_training';out.mkdir(exist_ok=False)
    a,p,g=load();target=targets(a,g);selection=g[g.split=='selection']
    ti=np.flatnonzero(np.isin(p['seed'],plan['train_seeds']))
    rng=np.random.default_rng(8300)
    if len(ti)>500000:ti=np.sort(rng.choice(ti,500000,replace=False))
    si=np.flatnonzero(np.isin(p['seed'],plan['selection_seeds']))
    # Compact selection views keep candidate ids local for sparse score updates.
    csel=np.flatnonzero(np.isin(a['seed'],plan['selection_seeds']))
    mapping=np.full(len(a['delta']),-1,np.int32);mapping[csel]=np.arange(len(csel))
    sa={k:v[csel] for k,v in a.items()};sp={k:v[si] for k,v in p.items()}
    sp['candidate']=mapping[sp['candidate']]
    sg=selection.copy();sg['start']=mapping[sg.start.to_numpy()];sg['end']=mapping[selection.end.to_numpy()-1]+1
    starget=target[csel]
    raw=p['x'][ti];mean=raw.mean(0).astype(np.float32);scale=np.maximum(raw.std(0),1.e-3).astype(np.float32)
    x=torch.from_numpy(np.clip((raw-mean)/scale,-10,10));sx=torch.from_numpy(np.clip((p['x'][si]-mean)/scale,-10,10))
    normalizer=np.maximum(p['raw'][ti,0],2400.)
    sensitivity=torch.from_numpy((normalizer/2400.).astype(np.float32))
    weight=torch.from_numpy((p['raw'][ti,3]+.5*p['raw'][ti,4]+.5*p['raw'][ti,5]).astype(np.float32))
    weight/=weight.mean()
    write_json(out/'protocol.json',dict(plan_sha256=sha(HERE/'plan.json'),train_pairs=len(ti),selection_pairs=len(si),
        sample_seed=8300,loss='SmoothL1 per affected UE expected-byte residual in units2400, weighted by causal risk sensitivity',
        selection_seed=44,calibration_evaluation_labels_used=False,epoch0='identity residual baseline'))
    history=[];best_key=None;best_path=None;best_record=None;started=time.perf_counter()
    for mode in (0,1):
        y=torch.from_numpy(((p['exact_expected'][ti]-p['raw'][ti,1+mode])/normalizer).astype(np.float32))
        models=[];optimizers=[];generators=[]
        for seed in plan['residual_model']['seeds']:
            torch.manual_seed(seed)
            m=nn.Sequential(nn.Linear(32,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
            nn.init.zeros_(m[4].weight);nn.init.zeros_(m[4].bias)
            models.append(m);optimizers.append(torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001))
            generators.append(torch.Generator().manual_seed(seed+100))
        for epoch in range(31):
            losses=[]
            if epoch:
                for m,opt,gen in zip(models,optimizers,generators):
                    total=0.
                    for ids in torch.randperm(len(x),generator=gen).split(4096):
                        prediction=m(x[ids])[:,0]
                        loss=(nn.functional.smooth_l1_loss((prediction-y[ids])*sensitivity[ids],torch.zeros(len(ids)),reduction='none')*weight[ids]).mean()
                        opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())*len(ids)
                    losses.append(total/len(x))
            if epoch not in plan['residual_model']['checkpoints']:continue
            with torch.no_grad():pred=np.stack([m(sx).numpy()[:,0] for m in models])
            score,_=residual_scores(sa,sp,pred,mode)
            df=metrics(sa,sg,score,[.1],'residual',starget)
            recall=float(df.recall.mean());regret=float(df.regret.mean())
            path=out/f'mode{mode}_epoch{epoch:02d}.npz';export(path,models,mean,scale,mode)
            np.testing.assert_allclose(ResidualValue(path).predict_rows(p['x'][si]),pred,rtol=2.e-5,atol=1.e-5)
            record=dict(mode=mode,epoch=epoch,selection_recall=recall,selection_regret=regret,
                train_loss=float(np.mean(losses)) if losses else 0.,checkpoint=path.name,
                parameters_per_member=sum(v.numel() for v in models[0].parameters()))
            history.append(record);key=(recall,-regret,-epoch)
            if best_key is None or key>best_key:best_key,best_path,best_record=key,path,record
            pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            print('RESIDUAL',mode,epoch,'selection_recall',round(recall,6),'regret',round(regret,3),flush=True)
            (HERE/'progress.txt').write_text(f'神经残差训练：mode{mode},epoch{epoch}/30；仅seed44选型。\n')
    shutil.copyfile(best_path,HERE/'residual_frozen.npz')
    write_json(out/'selected.json',dict(**best_record,sha256=sha(HERE/'residual_frozen.npz'),
        wall_s=time.perf_counter()-started,selection_seed=44,calibration_evaluation_labels_used=False,
        numpy_torch_parity_passed=True,data_sha256=sha(HERE/'data/pairs.npz')))

if __name__=='__main__':main()

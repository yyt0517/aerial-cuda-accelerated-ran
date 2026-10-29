"""Expert iteration over full legal extensions, including outside-shortlist positives."""
from pathlib import Path
import sys,json,time,shutil
import numpy as np
import pandas as pd
import torch
from torch import nn

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from common import load,targets,actor_joint,actor_coverage
from training.stageb_lns.cached_repair import CachedDeadlineRepair
from training.stageb_lns.experiment import sha,write_json

def export(path,model,mean,scale):
    data=dict(feature_version=2,mean=mean,scale=scale)
    for key,layer,attr in [('w1',0,'weight'),('b1',0,'bias'),('w2',2,'weight'),('b2',2,'bias'),('w3',4,'weight'),('b3',4,'bias')]:
        data[key]=getattr(model[layer],attr).detach().numpy()
    np.savez(path,**data)

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    plan=json.loads((HERE/'plan.json').read_text());out=HERE/'proposer_training';out.mkdir(exist_ok=False)
    a,_,g=load();high=targets(a,g)
    with np.load(HERE/'actor_original.npz') as z:old={k:z[k] for k in z.files}
    torch.manual_seed(8401);rng=np.random.default_rng(8401)
    model=nn.Sequential(nn.Linear(87,len(old['b1'])),nn.ReLU(),nn.Linear(len(old['b1']),len(old['b2'])),nn.ReLU(),nn.Linear(len(old['b2']),1))
    for key,layer,attr in [('w1',0,'weight'),('b1',0,'bias'),('w2',2,'weight'),('b2',2,'bias'),('w3',4,'weight'),('b3',4,'bias')]:
        getattr(model[layer],attr).data.copy_(torch.from_numpy(old[key]))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
    # Train blocks preserve all parent links so ranking gradients reach earlier prefixes.
    blocks=[]
    for bi,frame in g[g.split=='train'].groupby('block',sort=True):
        start,end=int(frame.start.iloc[0]),int(frame.end.iloc[-1])
        x=np.clip((a['actor_x'][start:end]-old['mean'])/old['scale'],-10,10)
        parent=a['parent'][start:end].copy();parent[parent>=0]-=start
        chunks=[]
        for row in frame.itertuples():
            lo,hi=row.start-start,row.end-start
            target=high[row.start:row.end].astype(np.float32)
            # Uniform distribution over the high-yield set, rather than a top1 label.
            if target.sum():target/=target.sum()
            par=a['parent'][row.start:row.end]
            prev=np.zeros(len(par));valid=par>=0;prev[valid]=a['old_logp'][par[valid]]
            old_cond=(a['old_logp'][row.start:row.end]-prev).astype(np.float32)
            chunks.append((lo,hi,row.parents,row.options,torch.from_numpy(target),torch.from_numpy(old_cond)))
        blocks.append((torch.from_numpy(x),parent,chunks))
    selection=g[g.split=='selection'];sel_ids=np.flatnonzero(np.isin(a['seed'],plan['selection_seeds']))
    write_json(out/'protocol.json',dict(plan_sha256=sha(HERE/'plan.json'),training_blocks=len(blocks),
        loss='Listwise uniform high-yield coverage target at each depth with differentiable parent-prefix logprobs; .1 KL(original conditional||new)',
        original_feature_normalization_frozen=True,train_seed_positive_replay_only=True,selection_seed=44,
        final_or_evaluation_labels_used_for_training=False))
    history=[];best_key=None;best_path=None;best_record=None;started=time.perf_counter()
    for epoch in range(21):
        losses=[]
        if epoch:
            for idx in rng.permutation(len(blocks)):
                x,parent,chunks=blocks[idx];raw=model(x)[:,0];joints=[];parts=[]
                for lo,hi,n,m,target,old_cond in chunks:
                    cond=torch.log_softmax(raw[lo:hi].reshape(n,m),dim=-1).ravel()
                    if lo:
                        previous=torch.cat(joints)
                        joint=cond+previous[torch.from_numpy(parent[lo:hi]).long()]
                    else:joint=cond
                    joints.append(joint)
                    kl=(old_cond.exp()*(old_cond-cond)).sum()/n
                    ranking=-(target*torch.log_softmax(joint,dim=0)).sum() if target.sum()>0 else raw.sum()*0
                    parts.append(ranking+.1*kl)
                loss=torch.stack(parts).mean();optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        if epoch not in plan['proposer_training']['checkpoints']:continue
        path=out/f'actor_epoch{epoch:02d}.npz';export(path,model,old['mean'],old['scale'])
        actor=CachedDeadlineRepair(path,16)
        logits=np.zeros(len(a['delta']),np.float32);logits[sel_ids]=actor.logits(a['actor_x'][sel_ids])
        joints=actor_joint(a,selection,logits)
        cover=actor_coverage(a,selection,joints,'new_actor',high)
        recall=float(cover.recall.mean());outside=float(cover.outside_recall.mean())
        # Mean categorical entropy records whether coverage came from probability collapse.
        entropy=[]
        for row in selection.itertuples():
            z=logits[row.start:row.end].reshape(row.parents,row.options).copy()
            z-=z.max(-1,keepdims=True);z-=np.log(np.exp(z).sum(-1,keepdims=True))
            entropy.append(float(-(np.exp(z)*z).sum(-1).mean()))
        record=dict(epoch=epoch,selection_recall_at48=recall,outside_recall=outside,
            entropy=float(np.mean(entropy)),train_loss=float(np.mean(losses)) if losses else 0.,checkpoint=path.name)
        history.append(record);key=(recall,outside,-epoch)
        if best_key is None or key>best_key:best_key,best_path,best_record=key,path,record
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        print('PROPOSER',epoch,'selection_recall48',round(recall,6),'outside',round(outside,6),'entropy',round(record['entropy'],4),flush=True)
        (HERE/'proposal_progress.txt').write_text(f'集合外正例提议器：epoch{epoch}/20，仅seed44选型。\n')
    shutil.copyfile(best_path,HERE/'actor_frozen.npz')
    write_json(out/'selected.json',dict(**best_record,sha256=sha(HERE/'actor_frozen.npz'),wall_s=time.perf_counter()-started,
        selection_seed=44,calibration_evaluation_labels_used_for_training=False,
        outside_positive_labels_from_train_seeds_only=True))

if __name__=='__main__':main()

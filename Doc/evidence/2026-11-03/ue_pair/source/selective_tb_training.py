"""One bounded own-visit distillation round with separately charged teachers."""
import hashlib
import time
import numpy as np

from .selective_tb import Incremental,ServiceGuard,SelectiveConfig,Ranker,solve_selective
from .tb_macro import candidate_pools
from .bounded_probes import ProbePhy
from .physics import DEQUEUE,COUNT,EXPECTED
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .ranked_dpls_campaign import identity
from .experiment import write_json


def collect(rows,out,max_visits=4):
    import torch
    groups=[];work=[];visits=[];started=time.perf_counter()
    for row in rows:
        p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs']);observed=[]
        a,v,student,_=solve_selective(p,draft,observer=observed.append)
        unique={}
        for visit in observed:
            if visit['sweep']==0:continue
            key=(visit['root'].tobytes(),visit['prg'],visit['failed'])
            unique.setdefault(key,visit)
        chosen=list(unique.values()) if unique else observed
        indices=np.unique(np.linspace(0,len(chosen)-1,min(len(chosen),max_visits),dtype=int))
        phy=ProbePhy(p);d=Incremental(p,phy);teacher_start=time.perf_counter()
        try:
            anchor=phy.evaluate(draft).values[0];guard=ServiceGuard(anchor)
            for index in indices:
                visit=chosen[int(index)];base=phy.evaluate(visit['root']);np.testing.assert_array_equal(anchor,visit['anchor'])
                pools=candidate_pools(p,d,base,visit['prg']);actions=np.concatenate(pools)
                assert hashlib.sha256(actions.tobytes()).hexdigest()==visit['pool_sha256']
                joint=np.arange(len(actions))>=len(pools[0]);f=d.features(base,actions,visit['floor']);f['x'][:,11]=joint
                available=(f['total']>=visit['floor'])&~((base.values[0,None,:,COUNT]>0)&(f['counts']==0)).any(1)
                available &= np.array([action.tobytes() not in visit['failed'] for action in actions],bool)
                exact=phy.evaluate(actions,base)
                np.testing.assert_allclose(f['dequeue'],exact.values[:,:,DEQUEUE],rtol=0,atol=1.e-8)
                gain=np.array([float(values[:,EXPECTED]@p.weights) for values in exact.values])-float(base.values[0,:,EXPECTED]@p.weights)
                safe=guard.safe(exact.values,base.values[0])&(exact.values[:,:,DEQUEUE].sum(1)>=visit['floor'])
                root_sha=hashlib.sha256(visit['root'].tobytes()).hexdigest()
                failed_sha=hashlib.sha256(b''.join(visit['failed'])).hexdigest()
                groups.append(dict(state_id=identity(row),seed=row['seed'],root=visit['root'],prg=visit['prg'],
                    root_sha256=root_sha,failed_sha256=failed_sha,ordinal=visit['ordinal'],sweep=visit['sweep'],
                    failed=visit['failed'],floor=visit['floor'],anchor=anchor.copy(),actions=actions,
                    x=f['x'],prior=f['prior'],available=available,safe=safe,gain=gain,joint=joint))
                visits.append(dict(state_id=identity(row),root_sha256=root_sha,failed_sha256=failed_sha,
                    prg=visit['prg'],ordinal=visit['ordinal'],sweep=visit['sweep'],pool_sha256=visit['pool_sha256'],
                    candidates=len(actions),available=int(available.sum()),safe_available=int((safe&available).sum())))
            teacher=dict(**phy.stats,elapsed_ms=(time.perf_counter()-teacher_start)*1000)
        finally:phy.close()
        again=solve_selective(p,draft)
        np.testing.assert_array_equal(a,again[0]);np.testing.assert_array_equal(v,again[1])
        assert student['candidate_evaluations']==again[2]['candidate_evaluations']
        work.append(dict(state_id=identity(row),student=student,replay_scores=again[2]['candidate_evaluations'],teacher=teacher))
        print('SELECTIVE_OWN',identity(row),'groups',len(indices),flush=True)
    torch.save(groups,out/'training_groups.pt')
    write_json(out/'training_data.json',dict(training_seeds=[41,42,43],groups=len(groups),visits=visits,
        labels=sum(len(g['gain']) for g in groups),work=work,student_replay_identical=True,
        teacher_post_decision=True,teacher_scores=sum(w['teacher']['candidate_evaluations'] for w in work),
        collection_elapsed_s=time.perf_counter()-started))
    return groups


def build_model():
    import torch
    model=torch.nn.Sequential(torch.nn.Linear(60,32),torch.nn.ReLU(),torch.nn.Linear(32,16),
        torch.nn.ReLU(),torch.nn.Linear(16,2))
    torch.nn.init.zeros_(model[4].weight);torch.nn.init.zeros_(model[4].bias)
    return model


def loss_function(raw,prior,gain,safe):
    import torch
    score=prior+raw[:,0]-(1-torch.sigmoid(raw[:,1]))
    logits=torch.cat([score,score.new_zeros(1)])/.15
    target=torch.cat([torch.where(safe,gain,gain.new_full(gain.shape,-1.e6)),gain.new_zeros(1)])/.15
    rank=-(torch.softmax(target,0)*torch.log_softmax(logits,0)).sum()
    classification=torch.nn.functional.binary_cross_entropy_with_logits(raw[:,1],safe.float())
    regression=torch.nn.functional.smooth_l1_loss(prior+raw[:,0],gain)
    return rank+.25*classification+.1*regression


def fit(groups,out,validate):
    import torch
    started=time.perf_counter();torch.manual_seed(20261102);model=build_model()
    all_x=np.concatenate([g['x'][g['available']] for g in groups if g['available'].any()])
    mean=all_x.mean(0).astype(np.float32);scale=np.maximum(all_x.std(0),.05).astype(np.float32)
    arrays=[]
    for g in groups:
        mask=g['available']
        if not mask.any():continue
        x=np.clip((g['x'][mask]-mean)/scale,-10,10)
        arrays.append((torch.from_numpy(x),torch.tensor(g['prior'][mask]/3000,dtype=torch.float32),
            torch.tensor(g['gain'][mask]/3000,dtype=torch.float32),torch.from_numpy(g['safe'][mask])))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001);rng=np.random.default_rng(20261102)
    best=None;history=[];steps=0
    for epoch in range(21):
        losses=[]
        if epoch:
            for index in rng.permutation(len(arrays)):
                x,prior,gain,safe=arrays[index];loss=loss_function(model(x),prior,gain,safe)
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step();steps+=1;losses.append(float(loss.detach()))
        if epoch in (0,5,10,20):
            path=out/f'selective_e{epoch:02d}.npz'
            np.savez(path,mean=mean,scale=scale,**{k:v.detach().numpy() for k,v in (
                ('w1',model[0].weight),('b1',model[0].bias),('w2',model[2].weight),('b2',model[2].bias),
                ('w3',model[4].weight),('b3',model[4].bias))})
            ranker=Ranker(path)
            for x,prior,_,_ in arrays[:2]:
                with torch.no_grad():expected=model(x).numpy()
                np.testing.assert_allclose(ranker.outputs(x.numpy()*scale+mean),expected,rtol=1.e-4,atol=5.e-5)
            key,metric=validate(ranker,epoch)
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)) if losses else None,**metric))
            if best is None or key>best[0]:best=(key,ranker,epoch)
            print('SELECTIVE_FIT',epoch,metric,flush=True)
    write_json(out/'training_history.json',dict(groups=len(arrays),epochs=20,optimizer_steps=steps,history=history,
        selected_epoch=best[2],elapsed_s=time.perf_counter()-started,normalization_training_only=True,
        target='safe near-optimal listwise rank with retained-root option + feasibility BCE + residual-gain regression',
        checkpoint_selection='validation trajectory J/service only; complete-cost gate checked in formal three-repeat benchmark'))
    return best[1],best[2]

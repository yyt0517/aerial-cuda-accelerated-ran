"""Bounded terminal-set distillation; no development input before selection."""
import time
import numpy as np
from .service_sets import (FEATURE_NAMES,SetModel,causal_features,decode_probabilities,
    heuristic_masks,set_metrics,calibration_key)
from .bounded_probes import ProbePhy,csv_write
from .physics import COUNT,EXPECTED,ReferencePhy
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .ranked_dpls import solve_ranked
from .ranked_dpls_campaign import identity
from .experiment import write_json,sha


def label_states(rows,out,phase):
    groups=[];work=[];start=time.perf_counter();checks=[]
    for row in rows:
        p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs']);phy=ProbePhy(p)
        try:
            base=phy.evaluate(draft);before=phy.stats.copy();x,utility,eligible=causal_features(p,phy,base)
            assert phy.stats==before;feature_scores=phy.stats['candidate_evaluations']
        finally:phy.close()
        teacher_start=time.perf_counter();a,v,r=solve_ranked(p,draft);teacher_ms=(time.perf_counter()-teacher_start)*1000
        independent=ReferencePhy(p).evaluate(a)[0];np.testing.assert_allclose(v,independent,rtol=2.e-9,atol=2.e-6)
        target=np.stack([v[:,COUNT]>0,v[:,EXPECTED]>1],1)
        groups.append(dict(state_id=identity(row),seed=row['seed'],x=x,utility=utility,eligible=eligible,
            serving=p.serving.copy(),target=target,teacher_action=a,teacher_values=v,teacher_j=float(v[:,EXPECTED]@p.weights)))
        work.append(dict(state_id=identity(row),initial_feature_phy=feature_scores,teacher_ms=teacher_ms,**r))
        checks.append(dict(state_id=identity(row),max_error=float(abs(v-independent).max())))
        print('SERVICE_LABEL',phase,identity(row),flush=True)
    import torch
    torch.save(groups,out/f'{phase}_set_groups.pt')
    write_json(out/f'{phase}_teacher_work.json',dict(states=len(rows),ue_labels=sum(int(g['eligible'].sum()) for g in groups),
        work=work,independent_phy=checks,elapsed_s=time.perf_counter()-start,
        feature_phy=sum(r['initial_feature_phy'] for r in work),teacher_phy=sum(r['candidate_evaluations'] for r in work),
        labels_offline_only=True))
    return groups


def calibration(model,groups,problems,thresholds):
    result=[]
    for g,p in zip(groups,problems):
        prob=model.predict(g['x']);masks=decode_probabilities(p,prob,g['utility'],g['eligible'],thresholds)
        result.append(dict(state_id=g['state_id'],seed=g['seed'],**set_metrics(masks,g['target'],g['eligible'])))
    return result


def fit(training,validation,validation_rows,out):
    import torch
    torch.manual_seed(20261102);start=time.perf_counter()
    model=torch.nn.Sequential(torch.nn.Linear(len(FEATURE_NAMES),24),torch.nn.ReLU(),torch.nn.Linear(24,12),
        torch.nn.ReLU(),torch.nn.Linear(12,2));torch.nn.init.zeros_(model[4].weight);torch.nn.init.zeros_(model[4].bias)
    all_x=np.concatenate([g['x'][g['eligible']] for g in training]);mean=all_x.mean(0).astype(np.float32)
    scale=np.maximum(all_x.std(0),.05).astype(np.float32);arrays=[]
    for g in training:
        mask=g['eligible'];arrays.append((torch.from_numpy(np.clip((g['x'][mask]-mean)/scale,-10,10)),
            torch.tensor(g['target'][mask],dtype=torch.float32)))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001);rng=np.random.default_rng(20261102)
    problems=[problem_from_raw_row(row) for row in validation_rows];best=None;history=[];grid=[];steps=0
    for epoch in range(61):
        losses=[]
        if epoch:
            for index in rng.permutation(len(arrays)):
                x,y=arrays[index];logits=model(x)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)+.1*((torch.sigmoid(logits).mean(0)-y.mean(0))**2).mean()
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                losses.append(float(loss.detach()));steps+=1
        if epoch not in (0,10,30,60):continue
        path=out/f'service_e{epoch:02d}.npz'
        np.savez(path,mean=mean,scale=scale,**{name:value.detach().numpy() for name,value in (
            ('w1',model[0].weight),('b1',model[0].bias),('w2',model[2].weight),('b2',model[2].bias),('w3',model[4].weight),('b3',model[4].bias))})
        deployed=SetModel(path)
        for x,_ in arrays[:2]:
            with torch.no_grad():expected=model(x).numpy()
            np.testing.assert_allclose(deployed.logits(x.numpy()*scale+mean),expected,rtol=1.e-4,atol=5.e-5)
        for ta in (.3,.5,.7):
            for te in (.3,.5,.7):
                rr=calibration(deployed,validation,problems,(ta,te));key=calibration_key(rr)
                grid.append(dict(epoch=epoch,allocation_threshold=ta,effective_threshold=te,
                    mean_allocation_recall=float(np.mean([r['allocation_recall'] for r in rr])),
                    min_allocation_recall=min(r['allocation_recall'] for r in rr),
                    mean_effective_recall=float(np.mean([r['effective_recall'] for r in rr])),
                    min_effective_recall=min(r['effective_recall'] for r in rr),
                    size_ratio=float(np.mean([r['size_ratio'] for r in rr])),calibration_loss=-key[1],joint_set_gate=bool(key[0])))
                if best is None or key>best[0]:best=(key,deployed,(ta,te),epoch,rr)
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)) if losses else None))
        print('SERVICE_FIT',epoch,history[-1],flush=True)
    heuristic_best=None;heuristic_grid=[]
    for ka in (4,6,8):
        for ke in (2,4):
            rr=[]
            for g,p in zip(validation,problems):
                masks=heuristic_masks(p,g['utility'],g['eligible'],(ka,ke))
                rr.append(dict(state_id=g['state_id'],seed=g['seed'],**set_metrics(masks,g['target'],g['eligible'])))
            key=calibration_key(rr);heuristic_grid.append(dict(allocated_per_cell=ka,effective_per_cell=ke,loss=-key[1],joint_set_gate=bool(key[0])))
            if heuristic_best is None or key>heuristic_best[0]:heuristic_best=(key,(ka,ke),rr)
    csv_write(out/'set_calibration_grid.csv',grid);csv_write(out/'heuristic_calibration_grid.csv',heuristic_grid)
    csv_write(out/'validation_selected_set_metrics.csv',best[4]);csv_write(out/'validation_heuristic_set_metrics.csv',heuristic_best[2])
    selection=dict(checkpoint=str(best[1].path),sha256=sha(best[1].path),epoch=best[3],
        thresholds=list(best[2]),heuristic_counts=list(heuristic_best[1]),
        calibration_rule='all-state recall>=98% and mean size<=75%; otherwise declared recall/size/precision loss; smaller epoch/threshold-grid order on ties',
        development_not_read=True,training_only_normalization=True,feature_names=FEATURE_NAMES)
    write_json(out/'model_selection.json',selection)
    write_json(out/'set_training_history.json',dict(epochs=60,optimizer_steps=steps,history=history,
        groups=len(arrays),training_ue_rows=len(all_x),elapsed_s=time.perf_counter()-start,
        target='full terminal COUNT>0 and EXPECTED>1B two binary sets; BCE +0.1 cardinality calibration'))
    return best[1],best[2],heuristic_best[1]

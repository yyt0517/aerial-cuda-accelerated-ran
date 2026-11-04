"""One fixed tiny MLP plus leave-seed-out sigmoid calibration, training only."""
import time
import numpy as np
from .terminal_risk import RiskModel,sigmoid,select_index,PROBABILITY_THRESHOLD
from .experiment import write_json
from .bounded_probes import csv_write


def fit(x,y,w):
    import torch
    torch.set_num_threads(1);torch.manual_seed(20261103)
    mean=np.average(x,axis=0,weights=w);scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));scale=np.maximum(scale,.05)
    xx=torch.tensor(np.clip((x-mean)/scale,-8,8),dtype=torch.float64)
    yy=torch.tensor(y,dtype=torch.float64);ww=torch.tensor(w/w.sum(),dtype=torch.float64)
    net=torch.nn.Sequential(torch.nn.Linear(x.shape[1],32),torch.nn.Tanh(),torch.nn.Linear(32,16),torch.nn.Tanh(),torch.nn.Linear(16,1)).double()
    opt=torch.optim.Adam(net.parameters(),lr=.01,weight_decay=.001)
    pos=float(np.average(y,weights=w));balance=torch.where(yy>0,.5/max(pos,.05),.5/max(1-pos,.05))
    for epoch in range(240):
        opt.zero_grad();z=net(xx).ravel();loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,yy,reduction='none')*ww*balance).sum()
        loss.backward();opt.step()
    arrays=dict(mean=mean,scale=scale,calibration=np.array([1.,0.]))
    for i,layer in enumerate((net[0],net[2],net[4])):
        arrays[f'w{i}']=layer.weight.detach().numpy().T.copy();arrays[f'b{i}']=layer.bias.detach().numpy().copy()
    return RiskModel(arrays)


def train(data,out):
    import torch
    started=time.perf_counter();x=data['x'];y=data['coverage'].astype(float);seed=data['seed'];state=data['state'];j=data['j']
    assert set(seed)=={41,42,43},'No validation/development labels allowed'
    w=np.array([1/np.count_nonzero(state==s) for s in state]);oof=np.empty(len(x));folds=[]
    for hold in (41,42,43):
        keep=seed!=hold;model=fit(x[keep],y[keep],w[keep]);oof[~keep]=model.logits(x[~keep])
        folds.append(dict(holdout_seed=hold,training_seeds=sorted(set(seed[keep].tolist())),training_candidates=int(keep.sum()),holdout_candidates=int((~keep).sum())))
    # Monotone Platt calibration on out-of-fold predictions, never on 44--48.
    z=torch.tensor(oof,dtype=torch.float64);yy=torch.tensor(y,dtype=torch.float64);ww=torch.tensor(w/w.sum(),dtype=torch.float64)
    a=torch.tensor(0.,dtype=torch.float64,requires_grad=True);b=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([a,b],lr=.5,max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad();slope=torch.nn.functional.softplus(a)
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(slope*z+b,yy,reduction='none')*ww).sum()+.001*(slope*slope+b*b)
        loss.backward();return loss
    optimizer.step(closure);cal=np.array([float(torch.nn.functional.softplus(a).detach()),float(b.detach())])
    model=fit(x,y,w);model.arrays['calibration']=cal;np.savez_compressed(out/'risk_model.npz',**model.arrays)
    probability=sigmoid(cal[0]*oof+cal[1]);np.savez_compressed(out/'training_oof.npz',logits=oof,probability=probability)
    selected=[]
    for sid in np.unique(state):
        ids=np.flatnonzero(state==sid);local,safe=select_index(j[ids],probability[ids],PROBABILITY_THRESHOLD);i=ids[local]
        selected.append(dict(state_id=str(sid),seed=int(seed[i]),candidate_index=int(i),predicted_probability=float(probability[i]),safe_found=safe,
            coverage_pass=bool(y[i]),quality_pass=bool(data['j_ratio'][i]>=.995),j_ratio=float(data['j_ratio'][i]),
            maxj_coverage_pass=bool(y[ids[np.argmax(j[ids])]])))
    csv_write(out/'training_oof_selection.csv',selected)
    bins=[]
    for lo,hi in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
        ids=(probability>=lo)&(probability<hi if hi<1 else probability<=hi)
        bins.append(dict(low=float(lo),high=float(hi),n=int(ids.sum()),predicted=float(np.average(probability[ids],weights=w[ids])) if ids.any() else None,
            observed=float(np.average(y[ids],weights=w[ids])) if ids.any() else None))
    write_json(out/'training_summary.json',dict(training_seeds=[41,42,43],states=len(np.unique(state)),candidates=len(y),positive_labels=int(y.sum()),
        architecture=[x.shape[1],32,16,1],epochs_per_fit=240,model_fits=4,gradient_steps=960,calibration_lbfgs_max_iter=100,
        label='canonical full-positive P5 and fixed weakest20% >=.98 full, no newly unallocatedUE; offline targets only',
        oof_folds=folds,calibration=cal.tolist(),oof_raw_brier=float(np.average((sigmoid(oof)-y)**2,weights=w)),
        oof_calibrated_brier=float(np.average((probability-y)**2,weights=w)),calibration_bins=bins,
        calibration_diagnostic_not_independent_test=True,selection_threshold=PROBABILITY_THRESHOLD,
        elapsed_s=time.perf_counter()-started,validation_read=False,development_read=False))
    return model

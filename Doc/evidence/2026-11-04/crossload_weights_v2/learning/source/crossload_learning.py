"""Conditional low-dimensional regime imitation, only after the grid entry passes."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from .physics import ROOT
from .crossload_native import FEATURES,RATES,run_key,state_features
from .bounded_native import pin_worker
from .analyze_crossload import read_points
from .analyze_bounded_stress import local
from .analyze_dpls_moves import write_json,write_csv,sha

def learned_run(output,seed,rate,model_file):
    import joblib
    from . import crossload_native as native
    from .ttl_weights import objective_weights
    model=joblib.load(model_file);original=native.weights_for;choices=[]
    def policy(family,base,ema,urgency,queue,coefficient):
        if family!='learned_tree':return original(family,base,ema,urgency,queue,coefficient)
        features=state_features(queue,ema,urgency)
        k=float(model.predict(features[None])[0]);choices.append(k)
        return objective_weights(base,urgency,k)
    native.weights_for=policy
    try:result=native.one(output,seed,rate,0.,4000,4000,4000,400,1000,'learned_tree')
    finally:native.weights_for=original
    d=local(result['directory']);np.save(d/'chosen_coefficients.npy',np.array(choices))
    assert len(choices)==8400
    write_json(d/'policy_identity.json',dict(model_sha256=sha(model_file),features=list(FEATURES),
        choices_measured={str(k):choices[4000:8000].count(k) for k in sorted(set(choices))},
        true_load_input=False,perload_oracle_input=False,future_outcome_input=False))
    return result

def main(args):
    import sklearn
    import scipy
    from sklearn.tree import DecisionTreeClassifier,export_text
    import joblib
    source=Path(args.source).resolve();grid=Path(args.grid).resolve();out=Path(args.output).resolve()
    decision=json.loads((grid/'learning_entry.json').read_text());assert decision['completed']
    out.mkdir(parents=True,exist_ok=args.resume_fit)
    if args.resume_fit:assert json.loads((out/'entry_decision.json').read_text())==decision
    else:write_json(out/'entry_decision.json',decision)
    if not decision['train_low_dimensional_policy']:
        write_json(out/'completion.json',dict(completed=True,trained=False,new_native_runs=0,new_native_steps=0,
            reason=('a common near-optimal fixed coefficient leaves little demonstrated learning room' if decision['universal_near_optimal_coefficients'] else
                'no varying train-only oracle labels; residual loss alone does not identify useful state-to-k labels'),
            learning_signal_present=decision['learning_signal_present'],gradient_steps=0));return
    labels={r['rate']:r['coefficient'] for r in decision['train_only_oracle']};protocol=json.loads((source/'protocol.json').read_text())
    frames=[];inputs={str(grid/'learning_entry.json'):sha(grid/'learning_entry.json')}
    for spec in protocol['jobs']:
        if spec['weight_family']!='ttl' or spec['seed'] not in (41,42):continue
        file=source/run_key(spec['seed'],spec['rate'],spec['coefficient'])/'search.csv'
        f=pd.read_csv(file);f=f[f.tti.between(4001,8000)].copy();assert len(f)==4000
        f['label']=labels[spec['rate']];f['rate']=spec['rate'];f['source_seed']=spec['seed']
        frames.append(f[['tti','rate','source_seed','label',*FEATURES]]);inputs[str(file)]=sha(file)
    frame=pd.concat(frames,ignore_index=True);counts=frame.groupby('rate').size()
    weights=np.array([len(frame)/(3*counts[r]) for r in frame.rate]);x=frame[list(FEATURES)].to_numpy()
    y=frame.label.to_numpy();assert np.isfinite(x).all() and set(frame.source_seed)=={41,42}
    model_file=out/'policy.joblib'
    if args.resume_fit:
        model=joblib.load(model_file)
        write_json(out/'fit_resume.json',dict(refitted=False,fit_calls_total=1,model_sha256=sha(model_file),
            features_sha256=sha(out/'training_features.csv.gz'),
            failure='NumPy int64 tree metadata was not JSON serializable; model/feature artifacts already complete',
            partial_metadata_sha256=sha(out/'training_failed_serialization.json.partial')))
    else:
        model=DecisionTreeClassifier(max_depth=2,min_samples_leaf=256,random_state=20261104)
        model.fit(x,y,sample_weight=weights);joblib.dump(model,model_file)
        (out/'policy.txt').write_text(export_text(model,feature_names=list(FEATURES)))
        frame.to_csv(out/'training_features.csv.gz',index=False)
    write_json(out/'training.json',dict(fit_calls=1,rows=len(frame),features=list(FEATURES),train_seeds=[41,42],
        sklearn_version=sklearn.__version__,numpy_version=np.__version__,scipy_version=scipy.__version__,joblib_version=joblib.__version__,
        development_eval_seed=43,labels=labels,load_balanced=True,train_accuracy=float(model.score(x,y,sample_weight=weights)),
        nodes=int(model.tree_.node_count),leaves=int(model.get_n_leaves()),depth=int(model.get_depth()),gradient_steps=0,
        limitation='imitates per-regime fixed-policy labels; these are not optimal counterfactual per-TTI actions; closed-loop result decides usefulness',
        test='seed43 never participates in oracle labels or fit; already-exposed development seed, no final generalization claim'))
    names=('crossload_learning.py','crossload_requirements.txt','crossload_native.py','crossload_campaign.py','ttl_weights.py','bounded_dpls.py','_bounded.so',
        'bounded_kernel.cpp','rate_native.py','physics_kernel.cpp','persistent_kernel.cpp','no_ctbs_inputs.py')
    files=[Path(__file__).with_name(n) for n in names];sources={str(p.relative_to(ROOT)):sha(p) for p in files}
    (out/'source').mkdir()
    for p in files:shutil.copy2(p,out/'source'/p.name)
    write_json(out/'source_manifest.json',sources);write_json(out/'input_manifest.json',inputs)
    write_json(out/'protocol.json',dict(model_sha256=sha(model_file),entry=str(grid/'learning_entry.json'),
        frozen_from_parent=str(source/'protocol.json'),seeds=[41,42,43],rates=list(RATES),
        jobs=[dict(seed=s,rate=r) for r in RATES for s in (41,42,43)],
        warmup=4000,settling=4000,measured=4000,followup=400,solver='same original _bounded.so',
        policy='predeclared depth2 tree on six causal queue/age/EMA summaries; no rate/seed/future outcome input',
        baselines='same seed/load fixedk2 and fixed perload oracle selected using only train seeds41/42',
        runtime='process-local weights_for callback only; original kernel and input adapter unchanged',
        no_retuning=True,workers=args.workers,cpu_pins=list(range(args.cpu,args.cpu+args.workers))))
    ctx=mp.get_context('spawn');q=ctx.Queue()
    for cpu in range(args.cpu,args.cpu+args.workers):q.put(cpu)
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=ctx,initializer=pin_worker,initargs=(q,)) as pool:
        tasks=[pool.submit(learned_run,str(out),s,r,str(model_file)) for r in RATES for s in (41,42,43)]
        for t in as_completed(tasks):results.append(t.result())
    for p,digest in sources.items():assert sha(ROOT/p)==digest
    for p,digest in inputs.items():assert sha(p)==digest
    write_json(out/'completion.json',dict(completed=True,trained=True,runs=results,new_native_runs=9,new_native_steps=111600,
        model_sha256=sha(model_file),gradient_steps=0,tree_fit_rows=len(frame),default_changed=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--grid',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--cpu',type=int,default=236)
    p.add_argument('--resume-fit',action='store_true');main(p.parse_args())

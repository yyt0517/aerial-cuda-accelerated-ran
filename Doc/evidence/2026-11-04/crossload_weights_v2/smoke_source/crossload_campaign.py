"""New predeclared cross-load grid, independent of the historical B sweep."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing as mp
from pathlib import Path
import shutil
from .physics import ROOT
from .crossload_native import one,run_key,COEFFICIENTS,RATES,FEATURES
from .bounded_native import pin_worker
from .analyze_bounded_stress import local
from .analyze_dpls_moves import write_json,sha

def existing():
    result={}
    for name in ('bounded_load_stress_20261104_v1','B_ttl_decoupled_20261104_v1'):
        d=ROOT/'deliverables'/name;c=json.loads((d/'completion.json').read_text())
        assert c['completed']
        for r in c['runs']+c.get('reused_runs',[]):
            key=(r['seed'],r['rate'],r['coefficient'])
            if key[2] in COEFFICIENTS:result[key]=r
    return result

def main(args):
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    if args.smoke:
        jobs=[(41,.8,2.,'ttl'),(41,.8,0.,'mlwdf_pf'),(41,1.28,0.,'exp_pf')]
        reused=[];warmup,settling,measured,followup,save=8,0,8,0,0
    else:
        old=existing();jobs=[];reused=[]
        for rate in RATES:
            for k in COEFFICIENTS:
                for seed in (41,42,43):
                    key=(seed,rate,k)
                    if key in old:reused.append(old[key])
                    else:jobs.append((seed,rate,k,'ttl'))
        for family in ('mlwdf_pf','exp_pf'):
            jobs.extend((s,r,0.,family) for r in RATES for s in (41,42,43))
        assert len(reused)==13 and len(jobs)==50
        warmup=settling=measured=4000;followup=400;save=1000
    inputs={}
    for r in reused:
        d=local(r['directory']);v=json.loads((d/'validation.json').read_text())
        assert v['passed'] and v['exact_native_actions']==12400
        for n in ('summary.json','validation.json','launch.json','actions.npz','cohort.csv.gz','saved_states.json','search.csv'):
            inputs[str(d/n)]=sha(d/n)
    names=('crossload_campaign.py','crossload_native.py','ttl_weights.py','bounded_native.py','rate_native.py',
        'bounded_dpls.py','_bounded.so','bounded_kernel.cpp','physics_kernel.cpp','persistent_kernel.cpp','no_ctbs_inputs.py')
    files=[Path(__file__).with_name(n) for n in names];sources={str(p.relative_to(ROOT)):sha(p) for p in files}
    (out/'source').mkdir()
    for p in files:shutil.copy2(p,out/'source'/p.name)
    write_json(out/'source_manifest.json',sources);write_json(out/'input_manifest.json',inputs)
    write_json(out/'protocol.json',dict(version='crossload_TTL_literature_v2_20261104',
        new_experiment=True,not_supplement_to_old_grid=True,smoke=args.smoke,
        seeds=[41,42,43],coefficients=list(COEFFICIENTS),rates=list(RATES),load_multipliers=[1,1.4,1.6],
        jobs=[dict(seed=s,rate=r,coefficient=k,weight_family=f) for s,r,k,f in jobs],reused_runs=reused,
        warmup=warmup,settling=settling,measured=measured,followup=followup,save_every=save,
        solver='same frozen _bounded.so for ALL fixed/literature/conditional learning arms; engine optimization kept separate',
        causal_TTL='same old base*(1+k*(1-HOLslack/400)) mean-normalized, k0 keeps bits',
        literature=dict(PLR_target=.01,TTL_TTIs=400,EMA='same .995/.005 successful service bytes',
            age='max(1,400-HOLslack), TTI units; one-slot floor for newly arrived/empty queues',
            PF='1/max(EMA,1e-9), uncapped; replaces clipped deficit base',
            M_LWDF='normalize((a*age)/EMA), a=-ln(.01)/400',
            EXP_PF='normalize(exp((a*age-x)/(1+sqrt(x)))/EMA), x=active mean(a*age)',
            attribution='classic weight forms adapted to the same joint expected-byte objective/floor/local solver; not original RB schedulers or original optimality claims',
            sources=['https://9p.io/who/stolyar/varchannel_commag.pdf',
                'https://link.springer.com/article/10.1007/s11277-013-1308-4',
                'https://link.springer.com/article/10.1186/s13638-015-0429-8']),
        frontier='four objectives: max mean payload; min pooled expiry/success-P95/max(seed,UE)expiry; perseed retained',
        oracle='perload retain mean payload>=99.7% best grid mean; lexicographically minimize worst(seed,UE)expiry, pooledexpiry, successP95, -payload, k; also report full Pareto set',
        oracle_training='for conditional learner choose oracle only on seeds41/42; all3-seed oracle is descriptive and never fed to learner',
        near_optimal='perload payload>=99.7% best mean and perseed>=99.4% best; expiry<=min+.1pp, P95<=min+1ms, worstUE<=min+2pp; report coordinates separately if no common feasible k',
        learning_entry='only after completed grid; different train-only oracle k across loads OR high-load residual expiry/worstUE for every k. Residual alone is not proof of achievable gain; constant labels do not justify fitting a constant classifier',
        conditional_learning=dict(model='DecisionTreeClassifier max_depth2,min_samples_leaf256,random_state20261104',
            features=list(FEATURES),training_seeds=[41,42],development_eval_seed=43,
            corpus='settling TTI4001..8000 of NEW fixed TTL trajectories only; balanced load weights; label=training-only perload oracle k',
            no_leakage='no true offered rate, seed, future expiry or oracle index as model input',
            evaluation='9 native trajectories same three loads/seeds/window; compare frozenk2 and training-only fixed oracle; no final holdout/generalization claim'),
        no_postselection='all63 points kept; no new k, threshold tuning, rerun seed selection or default replacement',
        deployment='40% gate independent, unchanged; one final SoA SIMD full-Gamma experiment only',
        workers=args.workers,cpu_pins=list(range(args.cpu,args.cpu+args.workers)),timing='parallel quality collection, no speed claim'))
    ctx=mp.get_context('spawn');queue=ctx.Queue()
    for cpu in range(args.cpu,args.cpu+args.workers):queue.put(cpu)
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=ctx,initializer=pin_worker,initargs=(queue,)) as pool:
        tasks=[pool.submit(one,str(out),s,r,k,warmup,settling,measured,followup,save,f) for s,r,k,f in jobs]
        for task in as_completed(tasks):results.append(task.result())
    for name,digest in sources.items():assert sha(ROOT/name)==digest
    for name,digest in inputs.items():assert sha(name)==digest
    write_json(out/'completion.json',dict(completed=True,mode='crossload_weights',runs=results,reused_runs=reused,
        new_native_runs=len(results),new_native_steps=len(results)*(warmup+settling+measured+followup),training_steps=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--workers',type=int,default=12);p.add_argument('--cpu',type=int,default=244);main(p.parse_args())

"""Paired complete-search timing after behavior-preserving binding changes."""
from pathlib import Path
import sys,json,time
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import replay
from training.stageb_lns.fast_draft import FastDraftRepair
from training.stageb_lns.experiment import ROOT,sha,write_json,snapshot_sources

METHODS=[('cached_original',0,False),('verified_original',0,False),('physical_direct',0,False),
    ('trace_original',8,False),('trace_original',8,True),('spatial_original',16,False),('spatial_original',16,True),
    ('residual_original',8,False),('residual_original',8,True),
    ('residual_new_actor',8,False),('residual_new_actor',8,True)]

def make(name,quota,fast):
    if not fast:return replay.make(name,quota)
    return FastDraftRepair(HERE/('actor_frozen.npz' if name.endswith('new_actor') else 'actor_original.npz'),
        mode=name.split('_')[0],residual_checkpoint=HERE/'residual_frozen.npz',exact_quota=quota)

def main():
    replay.enable_legacy_types();out=HERE/'optimized';out.mkdir(exist_ok=False)
    sources=snapshot_sources(out)
    write_json(out/'protocol.json',dict(plan=json.loads((HERE/'optimization_plan.json').read_text()),
        source_sha256=sources,residual_sha256=sha(HERE/'residual_frozen.npz'),actor_sha256=sha(HERE/'actor_frozen.npz')))
    plan=json.loads((HERE/'plan.json').read_text());cases=[]
    for split,seed,path,rows in replay.collect.prior.sources(plan):
        if split=='evaluation':cases += [(seed,row) for row in rows if row['tti'] in (2001,3001,4000)]
    assert len(cases)==6
    measurements=[];timings=[]
    for si,(seed,row) in enumerate(cases):
        p=replay.problem_from_row(row);draft=replay.initial_draft(p,row['obs']);expected={}
        for name,quota,fast in METHODS:
            model=make(name,quota,fast)
            try:
                a,v,r=replay.run(name,p,draft,model)
                current=(a,v,r['candidate_evaluations'])
                if fast:
                    for value,reference in zip(current,expected[(name,quota,False)]):np.testing.assert_array_equal(value,reference)
                expected[(name,quota,fast)]=current
                measurements.append(dict(seed=seed,tti=row['tti'],method=name,quota=quota,fast=fast,
                    risk=r['objective_score'],score=r['score'],candidates=r['candidate_evaluations']))
            finally:
                if hasattr(model,'close'):model.close()
        for repeat in range(3):
            shift=(si+repeat)%len(METHODS)
            for name,quota,fast in METHODS[shift:]+METHODS[:shift]:
                model=make(name,quota,fast);begin=time.perf_counter()
                try:a,v,r=replay.run(name,p,draft,model)
                finally:
                    if hasattr(model,'close'):model.close()
                wall=1000*(time.perf_counter()-begin)
                for value,reference in zip((a,v,r['candidate_evaluations']),expected[(name,quota,fast)]):np.testing.assert_array_equal(value,reference)
                timings.append(dict(seed=seed,tti=row['tti'],method=name,quota=quota,fast=fast,repeat=repeat,ms=wall))
            print('OPTIMIZED',seed,row['tti'],repeat+1,flush=True)
            (HERE/'progress.txt').write_text(f'保持动作不变的降本配对：seed{seed},TTI{row["tti"]},repeat{repeat+1}/3。\n')
    d=pd.DataFrame(measurements);t=pd.DataFrame(timings)
    d.to_csv(out/'states.csv',index=False);t.to_csv(out/'timings.csv',index=False)
    med=t.groupby(['seed','tti','method','quota','fast']).ms.median().rename('serial_ms').reset_index()
    paired=d.merge(med,on=['seed','tti','method','quota','fast'],validate='one_to_one')
    paired.to_csv(out/'paired_states.csv',index=False)
    paired.groupby(['seed','method','quota','fast']).mean(numeric_only=True).to_csv(out/'per_seed.csv')
    means=paired.groupby(['seed','method','quota','fast']).mean(numeric_only=True).groupby(['method','quota','fast']).mean()
    means.to_csv(out/'means.csv')
    for path,digest in sources.items():assert sha(ROOT/path)==digest,path
    write_json(out/'checks.json',dict(passed=True,states=6,repeats=3,
        all_optimized_actions_values_phy_counts_exact=True,all_repeats_deterministic=True,
        same_frozen_models=True,no_native_delay_claim=True))
    print(means[['risk','candidates','serial_ms']].to_string(),flush=True)

if __name__=='__main__':main()

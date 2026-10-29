"""Serial complete saved-state searches, including all sparse-draft overhead."""
from pathlib import Path
import sys,json,time
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import collect
from training.stageb_lns.archive import enable_legacy_types,problem_from_row,initial_draft
from training.stageb_lns.physics import NativePhy,DEQUEUE
from training.stageb_lns.cached_repair import CachedDeadlineRepair
from training.stageb_lns.prefix_repair import PrefixRepair
from training.stageb_lns.draft_repair import DraftRepair
from training.stageb_lns.quality import select
from training.stageb_lns.experiment import ROOT,sha,write_json,snapshot_sources

METHODS=[('cached_original',0),('verified_original',0),('physical_direct',0)]
METHODS += [(name,q) for q in (4,8,16) for name in ('trace_original','spatial_original','residual_original','residual_new_actor')]

def make(name,quota):
    old=HERE/'actor_original.npz';new=HERE/'actor_frozen.npz'
    if name=='physical_direct':return None
    if name=='cached_original':return CachedDeadlineRepair(old,16)
    if name=='verified_original':return PrefixRepair(old,16,mode='verified',prefix_budget=48)
    return DraftRepair(new if name.endswith('new_actor') else old,
        mode=name.split('_')[0],residual_checkpoint=HERE/'residual_frozen.npz',exact_quota=quota)

def run(name,p,draft,model):
    return select(p,draft,'risk_physical_direct' if name=='physical_direct' else 'risk_neural',model)

def main():
    enable_legacy_types();plan=json.loads((HERE/'plan.json').read_text())
    out=HERE/'replay';out.mkdir(exist_ok=False)
    sources=snapshot_sources(out)
    write_json(out/'protocol.json',dict(plan=json.loads((HERE/'replay_plan.json').read_text()),
        source_sha256=sources,actor_sha256=sha(HERE/'actor_frozen.npz'),residual_sha256=sha(HERE/'residual_frozen.npz')))
    cases=[]
    for split,seed,path,rows in collect.prior.sources(plan):
        if split!='evaluation':continue
        cases += [(seed,row) for row in rows if row['tti'] in (2001,3001,4000)]
    assert len(cases)==6
    measurements=[];timings=[]
    for si,(seed,row) in enumerate(cases):
        p=problem_from_row(row);draft=initial_draft(p,row['obs']);expected={}
        for name,quota in METHODS:
            model=make(name,quota)
            try:
                a,v,r=run(name,p,draft,model)
                audit=NativePhy(p)
                try:np.testing.assert_array_equal(v,audit.evaluate(a).values[0])
                finally:audit.close()
                assert r['score']>=r['initial_score'] and v[:,DEQUEUE].sum()>=r['floor']
                expected[(name,quota)]=(a,v,r['candidate_evaluations'])
                measurements.append(dict(seed=seed,tti=row['tti'],method=name,quota=quota,
                    score=r['score'],risk=r['objective_score'],candidates=r['candidate_evaluations'],
                    rounds=r['rounds'],actor_rows=getattr(model,'option_scores',0),
                    draft_candidates=getattr(model,'draft_candidates',0),
                    verified_candidates=getattr(model,'verified_candidates',0)))
            finally:
                if hasattr(model,'close'):model.close()
        for repeat in range(3):
            shift=(si+repeat)%len(METHODS);order=METHODS[shift:]+METHODS[:shift]
            for name,quota in order:
                model=make(name,quota) # model loading excluded, all decision caches cold
                begin=time.perf_counter()
                try:a,v,r=run(name,p,draft,model)
                finally:
                    if hasattr(model,'close'):model.close()
                wall=1000*(time.perf_counter()-begin)
                ea,ev,en=expected[(name,quota)]
                np.testing.assert_array_equal(a,ea);np.testing.assert_array_equal(v,ev)
                assert r['candidate_evaluations']==en
                timings.append(dict(seed=seed,tti=row['tti'],method=name,quota=quota,repeat=repeat,ms=wall))
            print('REPLAY',seed,row['tti'],'repeat',repeat+1,flush=True)
            (HERE/'progress.txt').write_text(f'实际搜索串行回放：seed{seed},TTI{row["tti"]},repeat{repeat+1}/3。\n离线召回门槛未通过，不运行最终seed闭环。\n')
        pd.DataFrame(measurements).to_csv(out/'states.csv',index=False)
        pd.DataFrame(timings).to_csv(out/'timings.csv',index=False)
    d=pd.DataFrame(measurements);t=pd.DataFrame(timings)
    median=t.groupby(['seed','tti','method','quota']).ms.median().rename('serial_ms').reset_index()
    merged=d.merge(median,on=['seed','tti','method','quota'],validate='one_to_one')
    merged.to_csv(out/'paired_states.csv',index=False)
    merged.groupby(['seed','method','quota']).mean(numeric_only=True).to_csv(out/'per_seed.csv')
    means=merged.groupby(['seed','method','quota']).mean(numeric_only=True).groupby(['method','quota']).mean()
    means.to_csv(out/'means.csv')
    for path,digest in sources.items():assert sha(ROOT/path)==digest,path
    write_json(out/'checks.json',dict(passed=True,states=6,methods=len(METHODS),timing_repeats=3,
        final_phy_exact=True,all_runs_floor_and_j_guards_passed=True,repeats_deterministic=True,
        no_native_packet_delay_claim=True))
    print(means[['risk','score','candidates','draft_candidates','serial_ms']].to_string(),flush=True)

if __name__=='__main__':main()

"""Validate activity-mask MMSE coverage using existing counters, without PHY."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .physics import ROOT
from .analyze_bounded_stress import local
from .analyze_dpls_moves import sha,write_json,write_csv

def main(args):
    source=ROOT/'deliverables/bounded_load_stress_20261104_v1'
    audit=ROOT/'deliverables/bounded_load_stress_analysis_20261104_v1'
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    # The frozen simulator and kernel specialize this scene to36UE/17PRG/3cell.
    users,prgs,cells=36,17,3;upper=users*prgs*(1<<(cells-1))
    inputs={};native=[]
    c=json.loads((source/'completion.json').read_text())
    for r in c['runs']:
        d=local(r['directory']);file=d/'search.csv';inputs[str(file)]=sha(file)
        allrows=pd.read_csv(file)
        assert (allrows.mmse_misses<=upper).all() and (allrows.mmse_misses>=0).all()
        s=allrows[allrows.tti.between(8001,12000)]
        native.append(dict(seed=r['seed'],rate=r['rate'],measured_TTIs=len(s),
            problems_upper=upper,mean_misses=float(s.mmse_misses.mean()),P95_misses=float(s.mmse_misses.quantile(.95)),
            max_misses=int(s.mmse_misses.max()),mean_bank_coverage=float(s.mmse_misses.mean()/upper),
            P95_bank_coverage=float(s.mmse_misses.quantile(.95)/upper)))
    e=pd.read_csv(audit/'equivalence.csv').set_index('state');rows=[]
    file=audit/'work_ledger.json';inputs[str(file)]=sha(file)
    for r in json.loads(file.read_text()):
        f,b=r['full_reference'],r['bounded'];assert f['mmse_misses']<=upper and b['mmse_misses']<=f['mmse_misses']
        rows.append(dict(rate=float(e.loc[r['state'],'rate']),states=1,full_misses=f['mmse_misses'],
            bounded_misses=b['mmse_misses'],full_candidates=f['candidate_evaluations'],bounded_candidates=b['candidate_evaluations']))
    table=pd.DataFrame(rows).groupby('rate').sum()
    table['full_bank_coverage']=table.full_misses/(table.states*upper)
    table['bounded_bank_coverage']=table.bounded_misses/(table.states*upper)
    table['bounded_over_full_misses']=table.bounded_misses/table.full_misses
    table.to_csv(out/'coverage_by_load.csv');write_csv(out/'native_coverage.csv',native)
    kernel=ROOT/'training/stageb_lns/physics_kernel.cpp'
    inputs[str(kernel)]=sha(kernel);write_json(out/'input_manifest.json',inputs)
    write_json(out/'completion.json',dict(completed=True,new_candidate_PHY=0,new_MMSE=0,
        cells=cells,users=users,PRGs=prgs,mask_states_per_pair=1<<(cells-1),MMSE_problems_upper_per_TTI=upper,
        checked_native_policy_TTIs=sum(len(pd.read_csv(local(r['directory'])/'search.csv')) for r in c['runs']),
        checked_full_replay_states=len(rows),per_key_log_available=False,
        proof='gamma strips own serving-cell bit; cache indexed byUE/PRG and remaining activitymask; miss increments only on new mask key',
        limitation='coverage is tuple count from cache-miss counters, not reconstructed per-pair key frequencies; coincident covariance matrices can reduce distinct problems further',
        small_matrix='kernel already uses Eigen::Matrix<complex<double>,4,4,RowMajor>; no new batch/woodbury arithmetic executed'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args())

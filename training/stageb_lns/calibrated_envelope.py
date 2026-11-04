"""Deduplicate full terminal service patterns; freeze an empirical envelope."""
import argparse
import csv
from pathlib import Path
import numpy as np
from .service_gate_calibration import ENSEMBLE
from .analyze_dpls_moves import write_json,write_csv,sha


def envelope(rows):
    return dict(J_min=.995*min(float(r['J']) for r in rows),
        unallocated_max=max(int(r['unallocated']) for r in rows),
        effective_min=min(int(r['effective']) for r in rows),
        queue_positive_P5_min=.98*min(float(r['queue_positive_P5']) for r in rows),
        deficit_weak_sum_min=.98*min(float(r['deficit_weak_sum']) for r in rows))


def passes(row,limit):
    return (row['J']+1.e-6>=limit['J_min'] and row['unallocated']<=limit['unallocated_max'] and
            row['effective']>=limit['effective_min'] and row['queue_positive_P5']+1.e-6>=limit['queue_positive_P5_min'] and
            row['deficit_weak_sum']+1.e-6>=limit['deficit_weak_sum_min'])


def main(args):
    source,out=Path(args.source).resolve(),Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    with (source/'metrics.csv').open() as f:rows=list(csv.DictReader(f))
    limits=[];unique_rows=[];loo=[]
    for state in sorted({int(r['state']) for r in rows}):
        selected=[r for r in rows if int(r['state'])==state and r['variant'] in ENSEMBLE]
        seen=set();distinct=[]
        for row in selected:
            with np.load(source/f"state{state:02d}_{row['variant']}.npz") as z:
                v=z['values'];key=v[:,0].astype(np.int64).tobytes()+np.rint(v[:,6]*1e6).astype(np.int64).tobytes()
            if key not in seen:seen.add(key);distinct.append(row)
        limit=dict(state=state,distinct_service_patterns=len(distinct),source_J=float(selected[0]['J'])/float(selected[0]['J_ratio']),
                   P5_informative=min(float(r['queue_positive_P5']) for r in distinct)>1.e-6,**envelope(selected))
        limits.append(limit);unique_rows.extend(distinct)
        for index,row in enumerate(distinct):
            if len(distinct)==1:continue
            other=[r for j,r in enumerate(distinct) if j!=index]
            current={k:float(row[k]) for k in ('J','unallocated','effective','queue_positive_P5','deficit_weak_sum')}
            loo.append(dict(state=state,variant=row['variant'],passed=passes(current,envelope(other))))
    write_json(out/'protocol.json',dict(version='full_empirical_envelope_v2',
        input=str(source),members=list(ENSEMBLE),
        correction='72/72 raw leave-one-out comparisons reused duplicate terminal patterns; not independent calibration evidence',
        dedup='COUNT vector plus per-UE expected bytes rounded1e-6B; only1--3 patterns/state',
        rule='J>=99.5% minimum full J; absolute count/effective/P5/causal-deficit-weak metrics inside empirical full min/max envelope',
        rationale='few distinct full optima cannot estimate5/95% tails; use an explicitly descriptive empirical support, no probability guarantee',
        timing='frozen using full-only outputs before new persistent guarded4 is evaluated',
        entry='guarded4 must pass all12 frozen envelope rows; only then two-seed2s native development; long-term service judged there',
        old_gate='source-local J/service checks still reported alongside; not silently overwritten',
        native_scope='development, not final promotion or blind confirmation'))
    write_json(out/'limits.json',limits);write_csv(out/'distinct_full.csv',unique_rows);write_csv(out/'deduplicated_leave_one_out.csv',loo)
    write_json(out/'summary.json',dict(distinct_service_patterns=len(unique_rows),
        states_with_one_pattern=sum(r['distinct_service_patterns']==1 for r in limits),
        deduplicated_loo_passes=sum(r['passed'] for r in loo),deduplicated_loo_comparisons=len(loo),
        informative_P5_states=sum(r['P5_informative'] for r in limits),
        interpretation='empirical envelope screening only; leave-one-pattern-out has very small support'))
    write_json(out/'input_manifest.json',{str(source/'metrics.csv'):sha(source/'metrics.csv')})
    write_json(out/'completion.json',dict(completed=True,new_phy_evaluations=0,new_native_runs=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);main(p.parse_args())

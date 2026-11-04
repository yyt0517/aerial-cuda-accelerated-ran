"""Fixed-model attribution after the three probes; no retraining or selection."""
import argparse
import time
from pathlib import Path
import numpy as np

from .bounded_probes import (Predictor,ProbePhy,certified_search,csv_write,
    seed_mean,write_json,sha,read_rows,identity,problem_from_raw_row,raw_pfq_draft,
    solve_ranked,ReferencePhy,EXPECTED,ROOT)


def main(source):
    import json
    source=Path(source).resolve();out=source/'scalar_attribution';out.mkdir(exist_ok=False)
    selected=json.loads((source/'selection.json').read_text())
    scalar=Predictor(selected['scalar']['checkpoint']);spatial=Predictor(selected['spatial']['checkpoint'])
    protocol=dict(fixed_selection_sha256=sha(source/'selection.json'),retraining=False,
        evaluation_already_exposed=True,seeds=[46,47,48],ttis=[3001,4001,5000],repeats=3,
        methods=['full','bound','scalar','spatial'],gate='all-state exact full action and mean time <=80% full',
        purpose='isolate spatial bank cost vs scalar candidate ranking; no hyperparameter tuning',
        timing='same adapter/fresh PHY/feature/network/search/cleanup boundary as parent pilot',
        registry_sha256=sha(ROOT/'training/stageb_hgraph/current_algorithm.json'))
    write_json(out/'protocol.json',protocol)
    files={}
    for path in (Path(__file__),Path(__file__).with_name('bounded_probes.py')):
        (out/path.name).write_bytes(path.read_bytes());files[path.name]=sha(path)
    write_json(out/'source_manifest.json',files)
    rows,_=read_rows((46,47,48),evaluation=True);rows=[r for r in rows if r['tti'] in protocol['ttis']]
    # First immutable TBS table/library setup is outside every measured call.
    problem_from_raw_row(rows[0])
    actions={};records=[];audits=[]
    for index,row in enumerate(rows):
        for repeat in range(3):
            methods=protocol['methods'];shift=(index+repeat)%4
            for method in methods[shift:]+methods[:shift]:
                started=time.perf_counter();p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs'])
                if method=='full':a,v,r=solve_ranked(p,draft)
                else:a,v,r=certified_search(p,draft,'learned' if method=='spatial' else method,
                    spatial if method=='spatial' else scalar)
                r['e2e_ms']=(time.perf_counter()-started)*1000;r.pop('history',None)
                r.update(state_id=identity(row),seed=row['seed'],arm=method,repeat=repeat);records.append(r)
                key=(identity(row),method)
                if key in actions:np.testing.assert_array_equal(actions[key],a)
                else:actions[key]=a.copy()
                if repeat==0:
                    independent=ReferencePhy(p).evaluate(a)[0]
                    np.testing.assert_allclose(v,independent,rtol=2.e-9,atol=2.e-6)
                    audits.append(dict(state_id=identity(row),arm=method,error=float(np.abs(v-independent).max())))
        for method in protocol['methods']:np.testing.assert_array_equal(actions[identity(row),method],actions[identity(row),'full'])
        print('ATTRIBUTION',identity(row),flush=True)
    csv_write(out/'measurements.csv',records)
    medians=[]
    for row in rows:
        for method in protocol['methods']:
            rr=[r for r in records if r['state_id']==identity(row) and r['arm']==method]
            r=dict(rr[0]);r['e2e_ms']=float(np.median([v['e2e_ms'] for v in rr]));r.pop('repeat')
            medians.append(r)
    csv_write(out/'per_state.csv',medians);full=[r for r in medians if r['arm']=='full']
    summary=[]
    for method in protocol['methods']:
        rr=[r for r in medians if r['arm']==method]
        summary.append(dict(arm=method,mean_ms=seed_mean(rr,'e2e_ms'),time_ratio=seed_mean(rr,'e2e_ms')/seed_mean(full,'e2e_ms'),
            candidate_evaluations=seed_mean(rr,'candidate_evaluations'),mmse_misses=seed_mean(rr,'mmse_misses'),
            feature_ms=seed_mean(rr,'feature_ms') if 'feature_ms' in rr[0] else None,
            network_ms=seed_mean(rr,'network_ms') if 'network_ms' in rr[0] else None,exact_full_actions=9))
    csv_write(out/'summary.csv',summary)
    write_json(out/'checks.json',dict(completed=True,exact_action_parity=True,audits=audits,
        registry_unchanged=sha(ROOT/'training/stageb_hgraph/current_algorithm.json')==protocol['registry_sha256']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True)
    main(parser.parse_args().source)

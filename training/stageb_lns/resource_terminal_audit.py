"""Post-selection paid validation-only joint-terminal neighborhood audit."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from .resource_conditions import ConditionalPhy,satisfies
from .resource_conditions_fast import fast_pool
from .resource_condition_campaign import compressed_condition
from .physics import ReferencePhy,EXPECTED,DEQUEUE
from .search import ProposalGraph
from .no_ctbs_inputs import problem_from_raw_row
from .ranked_dpls_campaign import read_rows,identity
from .bounded_probes import csv_write
from .experiment import write_json,sha


def main(source):
    source=Path(source).resolve();out=source/'terminal_audit';out.mkdir(exist_ok=False)
    write_json(out/'protocol.json',dict(posthoc=True,validation_only=True,
        scope='12 frozen joint212 terminal roots; entire same pair/3-cycle local pool, fixed counts/layers/activity',
        purpose='separate finite decoder/proxy/budget omissions from infeasible conditional domain; not global bound',
        teacher_service='used after queries only to diagnose existence of final quality/coverage passing neighbors; never online',
        online_policy_or_selection_changed=False,source_actions_sha256=sha(source/'validation_actions.npz'),code_sha256=sha(__file__)))
    actions=np.load(source/'validation_actions.npz');values=np.load(source/'validation_values.npz')
    ta=np.load(source/'validation_teacher_actions.npz');tv=np.load(source/'validation_teacher_values.npz')
    state={r['state_id']:r for r in csv.DictReader((source/'validation_per_state.csv').open()) if r['arm']=='joint212'}
    rows,_=read_rows((44,45));rows=[r for r in rows if r['tti'] in (4001,8001,18000)]
    records=[];work=[];audits=[];start_all=time.perf_counter()
    for row in rows:
        p=problem_from_raw_row(row);sid=identity(row);phy=ConditionalPhy(p);start=time.perf_counter()
        try:
            root=phy.evaluate(actions[f'{sid}|joint212']);np.testing.assert_array_equal(root.values[0],values[f'{sid}|joint212'])
            condition=compressed_condition(p,ta[sid],tv[sid],'joint212');graph=ProposalGraph(phy);phy.bank()
            pool=fast_pool(p,graph,root,condition);pool=pool[satisfies(p,pool,condition)]
            prior,_,_,_=phy.proxy(root,pool);exact=phy.evaluate(pool,root)
            js=np.array([float(v[:,EXPECTED]@p.weights) for v in exact.values]);root_j=float(root.values[0,:,EXPECTED]@p.weights)
            eligible=exact.values[:,:,DEQUEUE].sum(1)>=float(state[sid]['floor']);gain=js-root_j
            order=np.flatnonzero(eligible);order=order[np.argsort(-np.round(prior[order],7),kind='stable')]
            positive=np.flatnonzero(gain[order]>1)
            reference=tv[sid];ref_j=float(reference[:,EXPECTED]@p.weights);served=np.flatnonzero(reference[:,EXPECTED]>1)
            if len(served):
                weak=served[np.argsort(reference[served,EXPECTED],kind='stable')[:max(1,int(np.ceil(.2*len(served))))]]
                p5=np.quantile(exact.values[:,served,EXPECTED],.05,axis=1)/np.quantile(reference[served,EXPECTED],.05)
                weak_ratio=exact.values[:,weak,EXPECTED].sum(1)/reference[weak,EXPECTED].sum()
            else:p5=weak_ratio=np.ones(len(pool))
            passing=eligible&(js>=.995*ref_j)&(p5>=.98)&(weak_ratio>=.98)
            best=int(np.argmax(np.where(eligible,js,-np.inf))) if eligible.any() else None
            checked=set()
            for index in [best,*np.flatnonzero(passing)[:1]]:
                if index is None or index in checked:continue
                checked.add(index);independent=ReferencePhy(p).evaluate(pool[index])[0]
                np.testing.assert_allclose(exact.values[index],independent,rtol=2.e-9,atol=2.e-6)
                audits.append(dict(state_id=sid,index=int(index),max_error=float(abs(exact.values[index]-independent).max())))
            records.append(dict(state_id=sid,seed=row['seed'],pool=len(pool),best_gain=None if best is None else float(gain[best]),
                improving_candidates=int((eligible&(gain>1)).sum()),improving_gt16=int((eligible&(gain>16)).sum()),
                first_positive_rank=int(positive[0]+1) if len(positive) else None,
                top8_best_gain=max(0.,float(gain[order[:8]].max())) if len(order) else None,
                any_quality_service_passing_neighbor=bool(passing.any()),
                current_relative_j_pct=100*(root_j/ref_j-1),best_neighbor_relative_j_pct=None if best is None else 100*(js[best]/ref_j-1)))
            work.append(dict(state_id=sid,elapsed_ms=(time.perf_counter()-start)*1000,proxy_schedules=len(pool),**phy.stats))
        finally:phy.close()
    csv_write(out/'neighborhoods.csv',records);csv_write(out/'work.csv',work)
    write_json(out/'summary.json',dict(roots=len(records),positive_roots=sum(r['best_gain'] is not None and r['best_gain']>1 for r in records),
        positive_gt16_roots=sum(r['best_gain'] is not None and r['best_gain']>16 for r in records),
        top8_positive_roots=sum(r['top8_best_gain'] is not None and r['top8_best_gain']>1 for r in records),
        passing_neighbor_roots=sum(r['any_quality_service_passing_neighbor'] for r in records),
        physical_scores=sum(r['candidate_evaluations'] for r in work),independent_best_checks=audits,
        elapsed_s=time.perf_counter()-start_all,policy_or_selection_changed=False,
        note='unvisited global/long-cycle solutions remain unknown; positive local gain does not alone certify the final target'))
    print('TERMINAL AUDIT',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);main(parser.parse_args().source)

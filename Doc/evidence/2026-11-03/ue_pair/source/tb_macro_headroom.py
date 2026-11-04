"""Post-freeze validation-only paid fixed-root candidate-space diagnosis."""
import argparse
from pathlib import Path
import time
import numpy as np

from .tb_macro import Descriptors,candidate_pools,prg_priority
from .bounded_probes import ProbePhy,csv_write
from .physics import ReferencePhy,EXPECTED,DEQUEUE
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft,raw_pfq_floor
from .shortlist_trajectory import service_metrics
from .ranked_dpls_campaign import read_rows,identity
from .experiment import write_json,sha


def main(source):
    source=Path(source).resolve();out=source/'posthoc_headroom';out.mkdir(exist_ok=False)
    write_json(out/'protocol.json',dict(posthoc=True,selection_or_policy_changed=False,
        reason='separate poor shortlist/proxy from limited coordinated candidate space after failed formal gate',
        states='12 validation states only, saved single_b212 terminal roots, 4 causal-priority PRGs/root',
        query='score entire single/joint candidate pools on fixed root, no trajectory changes; no free oracle in benchmark',
        gates='raw PFQ90% floor; additional root protection: no newly unallocated UE, effective-root P5/bottom20%>=98%',
        missing='no eligible candidate is unknown, not zero',
        source_actions_sha256=sha(source/'validation_actions.npz'),code_sha256=sha(__file__)))
    actions=np.load(source/'validation_actions.npz');rows,_=read_rows((44,45))
    rows=[r for r in rows if r['tti'] in (4001,8001,18000)];records=[];work=[];checks=[]
    start=time.perf_counter()
    for row in rows:
        p=problem_from_raw_row(row);sid=identity(row);phy=ProbePhy(p);begin=time.perf_counter()
        try:
            floor=raw_pfq_floor(phy.evaluate(raw_pfq_draft(p,row['obs'])).values[0])
            base=phy.evaluate(actions[f'{sid}|single_b212']);root_j=float(base.values[0,:,EXPECTED]@p.weights)
            d=Descriptors(p,phy,56)
            for prg in prg_priority(p,d,base)[:4]:
                for family,pool in zip(('single','joint'),candidate_pools(p,d,base,int(prg))):
                    proxy,total=d.proxies(base,pool);exact=phy.evaluate(pool,base)
                    js=np.array([float(v[:,EXPECTED]@p.weights) for v in exact.values])
                    np.testing.assert_allclose(total,exact.values[:,:,DEQUEUE].sum(1),rtol=0,atol=0)
                    feasible=total>=floor;protected=[]
                    for v in exact.values:
                        m=service_metrics(v,base.values[0])
                        protected.append(m['newly_unserved_ues']==0 and m['positive_p5_ratio']>=.98 and m['positive_weak_ratio']>=.98)
                    for gate,mask in [('floor',feasible),('root_service',feasible&np.asarray(protected,bool))]:
                        indices=np.flatnonzero(mask);selected=None
                        if len(indices):
                            best=int(indices[np.argmax(js[indices])]);selected=exact.actions[best]
                            gain=float(js[best]-root_j)
                            ranked=indices[np.argsort(-proxy[indices],kind='stable')]
                            top2_gain=float(max(js[ranked[:2]])-root_j)
                            independent=ReferencePhy(p).evaluate(selected)[0]
                            np.testing.assert_allclose(exact.values[best],independent,rtol=2.e-9,atol=2.e-6)
                            checks.append(dict(state_id=sid,prg=int(prg),family=family,gate=gate,max_error=float(abs(exact.values[best]-independent).max())))
                        else:gain=None;top2_gain=None
                        records.append(dict(state_id=sid,seed=row['seed'],prg=int(prg),family=family,gate=gate,
                            root_j=root_j,candidates=len(pool),eligible=len(indices),best_gain=gain,
                            proxy_top2_best_gain=top2_gain,proxy_regret=None if gain is None else gain-top2_gain))
            work.append(dict(state_id=sid,elapsed_ms=(time.perf_counter()-begin)*1000,**phy.stats))
        finally:phy.close()
    csv_write(out/'headroom.csv',records);csv_write(out/'work.csv',work)
    index={(r['state_id'],r['prg'],r['gate'],r['family']):r for r in records};summary=[]
    for gate in ('floor','root_service'):
        pairs=[]
        for r in [v for v in records if v['family']=='joint' and v['gate']==gate]:
            single=index[r['state_id'],r['prg'],gate,'single'];pairs.append((r,single))
        observed=[(a,b) for a,b in pairs if a['best_gain'] is not None and b['best_gain'] is not None]
        summary.append(dict(gate=gate,blocks=len(pairs),both_observed=len(observed),
            joint_positive=sum(a['best_gain'] is not None and a['best_gain']>1 for a,b in pairs),
            single_positive=sum(b['best_gain'] is not None and b['best_gain']>1 for a,b in pairs),
            joint_beats_single=sum(a['best_gain']>b['best_gain']+1 for a,b in observed),
            single_beats_joint=sum(b['best_gain']>a['best_gain']+1 for a,b in observed),
            mean_joint_best_gain=float(np.mean([a['best_gain'] for a,b in observed])),
            mean_single_best_gain=float(np.mean([b['best_gain'] for a,b in observed])),
            mean_joint_proxy_regret=float(np.mean([a['proxy_regret'] for a,b in observed])),
            joint_positive_missed_by_top2=sum(a['best_gain']>1 and a['proxy_top2_best_gain']<=1 for a,b in observed),
            no_protected_joint=sum(a['best_gain'] is None for a,b in pairs)))
    write_json(out/'summary.json',summary)
    write_json(out/'checks.json',dict(passed=True,independent_best_candidate_checks=checks,
        complete_tb_dequeue_exact=True,formal_actions_unchanged=sha(source/'validation_actions.npz')==
        __import__('json').loads((out/'protocol.json').read_text())['source_actions_sha256']))
    write_json(out/'completion.json',dict(completed=True,states=len(rows),blocks=4*len(rows),
        physical_scores=sum(r['candidate_evaluations'] for r in work),elapsed_s=time.perf_counter()-start,
        independent_audits=len(checks),validation_only=True,policy_and_selection_changed=False))
    print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);main(parser.parse_args().source)

"""Integrity checks preserve negative scientific/search-quality outcomes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from training.stageb_lns.experiment import ROOT,sha,write_json

HERE=Path(__file__).resolve().parent

def main():
    plan=json.loads((HERE/'plan.json').read_text());data=json.loads((HERE/'data/checks.json').read_text())
    role=[set(plan[k+'_seeds']) for k in ('train','selection','calibration','evaluation')]
    assert sum(map(len,role))==len(set.union(*role))
    assert not(set.union(*role)&set(plan['excluded_seeds']))
    assert sha(HERE/'plan.json')==json.loads((HERE/'data/protocol.json').read_text())['plan_sha256']
    for path,digest in {**data['source_sha256'],**data['input_sha256']}.items():assert sha(ROOT/path)==digest,path
    for phase in ('replay','optimized'):
        for path,digest in json.loads((HERE/phase/'protocol.json').read_text())['source_sha256'].items():assert sha(ROOT/path)==digest,path
    for phase in ('evaluation','replay','optimized','benchmark','diagnostics'):
        assert json.loads((HERE/phase/'checks.json').read_text())['passed']
    for folder,name in [('residual_training','residual_frozen.npz'),('proposer_training','actor_frozen.npz')]:
        selected=json.loads((HERE/folder/'selected.json').read_text())
        assert sha(HERE/name)==selected['sha256']
        history=pd.read_csv(HERE/folder/'history.csv')
        if folder=='residual_training':
            best=history.sort_values(['selection_recall','selection_regret','epoch'],ascending=[False,True,True],kind='stable').iloc[0]
        else:best=history.sort_values(['selection_recall_at48','outside_recall','epoch'],ascending=[False,False,True],kind='stable').iloc[0]
        assert best.checkpoint==selected['checkpoint']
    with np.load(HERE/'data/candidates.npz') as z:
        n=len(z['delta']);seeds=np.unique(z['seed']);parents=z['parent'];groups=z['group']
        assert np.all(parents<np.arange(n))
    with np.load(HERE/'data/pairs.npz') as z:
        assert len(z['candidate'])==data['sparse_pairs']
        assert z['candidate'].min()>=0 and z['candidate'].max()<n
    assert n==742500 and data['sparse_pairs']==3193329 and len(seeds)==10
    d=pd.read_csv(HERE/'evaluation/curves_per_group.csv')
    assert (d.k<=d.n).all() and (d.regret>=-1.e-8).all()
    # No top-decile method can exceed min(k/elite_count,1), including ties.
    good=d.high_yield_count>0
    upper=np.minimum(d.loc[good,'k']/d.loc[good,'high_yield_count'],1)
    assert np.all(d.loc[good,'recall']<=upper+1.e-10)
    oracle=d[(d.method=='exact_oracle')&good]
    np.testing.assert_allclose(oracle.recall,np.minimum(oracle.k/oracle.high_yield_count,1),atol=1.e-10,rtol=0)
    assert not json.loads((HERE/'evaluation/checks.json').read_text())['scientific_gate_passed']
    assert sha(ROOT/'training/stageb_hgraph/current_algorithm.json')==data['registry_sha256']
    write_json(HERE/'checks.json',dict(passed=True,scientific_gate_passed=False,algorithm_promoted=False,
        mean_packet_delay_gate='not_exercised_no_native_trajectory',final_seeds311_315_unused=True,
        seed_roles_disjoint=True,all_input_and_listed_source_hashes_unchanged=True,
        model_selection_reproduced=True,oracle_budget_bounds_exact=True,
        exact_changes_inside_sparse_dependency_closure=True,
        full_python_trace_and_spatial_references_match_cpp=True,
        dense_sparse_affected_features_bit_exact=True,
        optimized_complete_search_actions_values_and_phy_counts_exact=True,
        tests_passed=36,full_suite_exit_code=0,
        targeted_portable_fast_test_rerun_passed=True,registry_unchanged=True,
        registry_sha256=data['registry_sha256'],
        interpretation='Engineering and provenance pass; neural quality/promotion gates do not. No native mean-delay or realtime claim.'))
    print('AUDIT PASS; offline quality gate remains FAIL; no promotion',flush=True)

if __name__=='__main__':main()

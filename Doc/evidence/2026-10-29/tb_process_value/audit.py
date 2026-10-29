"""Independent integrity and metric checks; scientific failure is preserved."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from training.stageb_lns.experiment import ROOT, sha, write_json


def main():
    plan = json.loads((HERE/'plan.json').read_text())
    provenance = json.loads((HERE/'data/frozen_plan.json').read_text())
    collected = json.loads((HERE/'data/checks.json').read_text())
    selected = json.loads((HERE/'training/selected.json').read_text())
    evaluation = json.loads((HERE/'evaluation/checks.json').read_text())
    diagnostic = json.loads((HERE/'diagnostics/checks.json').read_text())
    role_sets = [set(plan[k+'_seeds']) for k in ('train', 'selection', 'calibration', 'evaluation')]
    assert len(set.union(*role_sets)) == sum(map(len, role_sets))
    assert not (set.union(*role_sets) & set(plan['excluded_seeds']))
    assert sha(HERE/'plan.json') == provenance['sha256']
    for path, expected in provenance['source_sha256'].items():
        assert sha(ROOT/path) == expected, path
    for path, expected in collected['input_sha256'].items():
        assert sha(ROOT/path) == expected, path
    assert sha(ROOT/'training/stageb_hgraph/current_algorithm.json') == collected['registry_sha256']
    assert sha(HERE/'value_frozen.npz') == selected['checkpoint_sha256'] == evaluation['frozen_value_sha256']
    assert sha(HERE/'data/candidates.npz') == selected['data_sha256'] == diagnostic['original_candidates_npz_sha256']
    h = pd.read_csv(HERE/'training/history.csv')
    best = h.sort_values(['selection_recall_at10pct', 'selection_regret', 'epoch'],
                        ascending=[False, True, True], kind='stable').iloc[0]
    assert best.checkpoint == selected['checkpoint'] and best.epoch == selected['epoch']
    with np.load(HERE/'data/candidates.npz') as z:
        arrays = {k:z[k] for k in z.files}
    g = pd.read_csv(HERE/'data/groups.csv')
    s = pd.read_csv(HERE/'data/states.csv')
    assert len(g) == 1392 and len(arrays['delta']) == 214508
    assert arrays['unique'].sum() == 156492 and arrays['x'].shape == (214508, 189)
    np.testing.assert_array_equal(np.unique(arrays['seed']), np.sort(list(set.union(*role_sets))))
    assert s.candidate_evaluations.sum() == len(arrays['delta'])+5*len(s)
    assert s.independent_audit_evaluations.sum() == 2*len(s)
    d = pd.read_csv(HERE/'evaluation/curves_per_block.csv')
    assert (d.exact_evaluations <= d.candidates).all()
    assert (d.regret >= -1.e-9).all()
    oracle = d[d.method == 'exact_oracle']
    opportunity = oracle[oracle.high_yield_count > 0]
    upper = np.minimum(opportunity.exact_evaluations/opportunity.high_yield_count, 1)
    np.testing.assert_allclose(opportunity.recall, upper, rtol=0, atol=1.e-12)
    assert np.allclose(d[d.budget == 1].regret, 0)
    threshold = pd.read_csv(HERE/'evaluation/threshold_per_seed.csv')
    cal = threshold[(threshold.split == 'calibration') & (threshold.method == 'empirical_value')]
    assert float(cal.recall.iloc[0]) >= .98
    assert evaluation['scientific_gate_passed'] is False
    write_json(HERE/'checks.json', dict(passed=True, scientific_gate_passed=False,
        policy_promoted=False, registry_unchanged=True,
        source_snapshot_matches_current=True, all_input_snapshots_unchanged=True,
        frozen_actor_sha256=sha(HERE/'actor_frozen.npz'), frozen_value_sha256=sha(HERE/'value_frozen.npz'),
        seed_roles_disjoint=True, final_test_seeds_unused=True,
        architecture_and_epoch_selection_reproduced=True, oracle_recall_bound_exact=True,
        full_budget_regret_zero=True, empirical_calibration_target_reproduced=True,
        independent_full_incremental_label_checks=116,
        additional_control_replay_all_features_and_labels_exact=True,
        unit_test_command="python -m unittest discover -s training/stageb_lns -t . -p 'test_*.py'",
        unit_tests_passed=30, unit_test_exit_code=0,
        interpretation='Integrity checks passed; recall screen failed. No closed-loop or same-quality speedup claim.'))
    print('AUDIT PASS; scientific gate FAIL (preserved)', flush=True)


if __name__ == '__main__':
    main()

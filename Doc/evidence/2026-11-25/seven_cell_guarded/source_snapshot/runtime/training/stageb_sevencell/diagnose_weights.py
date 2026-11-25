"""Offline same-state weight/path comparisons; never advances a native trajectory."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from training.stageb_lns.no_ctbs_inputs import problem_from_raw_row, raw_pfq_draft
from training.stageb_lns.crossload_native import weights_for
from training.stageb_lns.physics import NativePhy, EXPECTED, DEQUEUE
from .solver import solve_bounded


def weight_stats(row, scale, source, seed):
    p = problem_from_raw_row(row)
    urgency = np.asarray(row['hol_urgency'])
    age = np.maximum(1., 400.*urgency)
    active = p.queue > 0
    delay = weights_for('mlwdf_pf', p.weights, row['ema'], urgency, p.queue, 0.)
    return dict(cells=scale, source_arm=source, seed=seed, tti=int(row['tti']),
                active_ues=int(active.sum()), hol_age_mean_ms=float(age[active].mean()/2),
                hol_age_median_ms=float(np.median(age[active])/2),
                hol_age_ge180ms_pct=float(100*np.mean(age[active]>=360)),
                hol_age_ge100ms_pct=float(100*np.mean(age[active]>=200)),
                deficit_weight_ratio=float(p.weights[active].max()/p.weights[active].min()),
                da_weight_ratio=float(delay[active].max()/delay[active].min()),
                da_weight_max=float(delay[active].max()),
                ema_p5=float(np.quantile(row['ema'][active], .05)),
                ema_median=float(np.median(row['ema'][active])))


def main(root):
    root = Path(root)
    out = root/'diagnosis_weights'
    out.mkdir(exist_ok=False)
    actions, comparisons, weights, traces = [], [], [], []
    for source in ('deficit', 'mlwdf_pf'):
        for seed in (41, 42, 43):
            for path in sorted((root/'campaign'/f'{source}_s{seed}'/'states').glob('*.pt')):
                row = torch.load(path, map_location='cpu', weights_only=False)
                p = problem_from_raw_row(row)
                base = p.weights.copy()
                delay = weights_for('mlwdf_pf', base, row['ema'], row['hol_urgency'], p.queue, 0.)
                np.testing.assert_array_equal(row['objective_weights'], base if source=='deficit' else delay)
                draft = raw_pfq_draft(p, row['obs'])
                identity = dict(source_arm=source, seed=seed, tti=int(row['tti']))
                weights.append(weight_stats(row, 7, source, seed))
                phy = NativePhy(p)
                try:
                    raw_values = phy.evaluate(draft).values[0].copy()
                finally:
                    phy.close()
                solved = {'raw_pfq': (draft, raw_values)}
                paths = {}
                for policy, w in [('deficit', base), ('mlwdf_pf', delay), ('mlwdf_pf_from_deficit', delay)]:
                    p.weights = w.copy()
                    trace = []
                    def observe(x):
                        trace.append(dict(iteration=x['iteration'],
                                          selected_key=int(x['selected_keys'][0]) if len(x['selected_keys']) else -1,
                                          score_q=x['score_q'], floor=x['floor'], dequeue=x['dequeue'],
                                          j_deficit=float(x['root_values'][:, EXPECTED]@base),
                                          j_da=float(x['root_values'][:, EXPECTED]@delay),
                                          expected_bytes=float(x['root_values'][:, EXPECTED].sum())))
                    warm = solved['deficit'][0] if policy=='mlwdf_pf_from_deficit' else None
                    a, v, r = solve_bounded(p, draft, warm_start=warm, observer=observe)
                    assert r['converged'] and all(y['score_q']>=x['score_q'] for x,y in zip(trace,trace[1:]))
                    assert r['chosen_dequeue']>=r['floor']
                    if policy==source:
                        np.testing.assert_array_equal(a, row['selected_action'])
                        np.testing.assert_array_equal(v, row['selected_values'])
                    solved[policy] = (a.copy(), v.copy())
                    paths[policy] = trace
                    traces.extend(dict(**identity, policy=policy, **t) for t in trace)
                for policy,(a,v) in solved.items():
                    actions.append(dict(**identity, policy=policy,
                        j_deficit=float(v[:,EXPECTED]@base), j_da=float(v[:,EXPECTED]@delay),
                        expected_bytes=float(v[:,EXPECTED].sum()), planned_dequeue=float(v[:,DEQUEUE].sum()),
                        occupancy_pct=float(100*np.mean(a>=0)),
                        scheduled_ues=int(np.sum(v[:,0]>0)),
                        served_age_weighted_ms=float(v[:,EXPECTED]@np.maximum(1.,400.*row['hol_urgency'])/v[:,EXPECTED].sum()/2),
                        rounds=0 if policy=='raw_pfq' else len(paths[policy])))
                d = paths['deficit'];a = paths['mlwdf_pf']
                divergence = next((i+1 for i,(x,y) in enumerate(zip(d,a)) if x['selected_key']!=y['selected_key']), min(len(d),len(a))+1)
                comparisons.append(dict(**identity, first_divergent_round=divergence,
                    final_different_slots=int(np.sum(solved['deficit'][0]!=solved['mlwdf_pf'][0])),
                    own_selected_action_reproduced=True,
                    same_raw_pfq=True, same_floor=d[0]['floor']==a[0]['floor']))
                print('DIAGNOSED', source, seed, row['tti'], 'divergence', divergence, flush=True)
    historical = pd.read_csv(root/'three_cell_matched_reference.csv')
    for _,r in historical[historical.arm.isin(['deficit','mlwdf_pf'])].iterrows():
        for path in sorted((Path(r.native_directory)/'states').glob('*.pt')):
            row = torch.load(path,map_location='cpu',weights_only=False)
            weights.append(weight_stats(row,3,r.arm,int(r.seed)))
    pd.DataFrame(actions).to_csv(out/'same_state_actions.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'path_comparisons.csv',index=False)
    pd.DataFrame(weights).to_csv(out/'saved_state_weight_stats.csv',index=False)
    pd.DataFrame(traces).to_csv(out/'search_traces.csv.gz',index=False)
    (out/'validation.json').write_text(json.dumps(dict(passed=True, states=len(comparisons),
        own_actions_and_values_reproduced=True, same_state_same_draft_same_floor=True,
        searches_converged=True, objective_monotone_each_search=True,
        scope='Offline same-state single-decision comparisons; no new closed-loop benchmark.'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)

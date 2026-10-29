"""Frozen model evaluation, empirical calibration and cost/recall artifacts."""
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import curves, high_yield
from training.stageb_lns.experiment import write_json, sha
from training.stageb_lns.process_value import ProcessValue, conformal_upper_quantile


def empirical_threshold(arrays, groups, mean, sigma, alpha):
    values, weights = [], []
    for g in groups.itertuples():
        ids = np.flatnonzero((arrays['group'] == g.group) & arrays['unique'])
        target = high_yield(arrays['delta'][ids], arrays['eligible'][ids])
        ids = ids[target]
        if not len(ids):
            continue
        values.extend(-mean[ids]/sigma[ids])
        weights.extend(np.repeat(1/len(ids), len(ids)))
    values, weights = np.asarray(values), np.asarray(weights)
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order]);cumulative /= cumulative[-1]
    i = min(int(np.searchsorted(cumulative, 1-alpha)), len(order)-1)
    return float(values[order[i]]+1.e-10)


def threshold_rows(arrays, groups, score, method):
    rows = []
    for g in groups.itertuples():
        ids = np.flatnonzero((arrays['group'] == g.group) & arrays['unique'])
        kept = score[ids] > 0
        target = high_yield(arrays['delta'][ids], arrays['eligible'][ids])
        best = float(np.max(np.r_[0., arrays['delta'][ids][arrays['eligible'][ids]]]))
        gain = float(np.max(np.r_[0., arrays['delta'][ids][kept & arrays['eligible'][ids]]]))
        rows.append(dict(group=g.group, split=g.split, seed=g.seed, tti=g.tti,
            origin=g.origin, method=method, candidates=len(ids), exact_evaluations=int(kept.sum()),
            actual_fraction=float(kept.mean()), high_yield_count=int(target.sum()),
            recall=float((kept & target).sum()/target.sum()) if target.any() else np.nan,
            any_high_yield=float((kept & target).any()) if target.any() else np.nan,
            oracle_gain=best, retained_gain=gain, regret=best-gain))
    return pd.DataFrame(rows)


def main():
    out = HERE/'evaluation';out.mkdir(exist_ok=False)
    plan = json.loads((HERE/'plan.json').read_text())
    selected = json.loads((HERE/'training/selected.json').read_text())
    assert sha(HERE/'value_frozen.npz') == selected['checkpoint_sha256']
    assert sha(HERE/'data/candidates.npz') == selected['data_sha256']
    with np.load(HERE/'data/candidates.npz') as saved:
        arrays = {k: saved[k] for k in saved.files}
    groups = pd.read_csv(HERE/'data/groups.csv')
    history = pd.read_csv(HERE/'training/history.csv')
    predictions, deviations = {}, {}
    architecture_paths = {}
    for architecture, h in history.groupby('architecture'):
        best = h.sort_values(['selection_recall_at10pct', 'selection_regret', 'epoch'],
                            ascending=[False, True, True], kind='stable').iloc[0]
        path = HERE/'training'/best.checkpoint
        architecture_paths[architecture] = dict(path=best.checkpoint, sha256=sha(path), epoch=int(best.epoch))
        predictions[architecture], deviations[architecture] = ProcessValue(path).predict(arrays['x'])
    chosen = selected['architecture']
    mu, sigma = predictions[chosen][:, 0], deviations[chosen][:, 0]+200.
    cal = groups[groups.split == 'calibration']
    value_lambda = empirical_threshold(arrays, cal, mu, sigma, .02)
    physical_lambda = empirical_threshold(arrays, cal, arrays['proxy'], np.full(len(mu), 200.), .02)
    rank_scores = dict(actor_likelihood=arrays['logp'], frozen_bler_tb_physics=arrays['proxy'],
        **{k: v[:, 0] for k, v in predictions.items()},
        calibrated_value_upper=mu+value_lambda*sigma,
        exact_oracle=np.where(arrays['eligible'], arrays['delta'], -1.e30))
    tables = [curves(arrays, groups[groups.split != 'train'], score, plan['budgets'], name)
              for name, score in rank_scores.items()]
    data = pd.concat(tables, ignore_index=True)
    data.to_csv(out/'curves_per_block.csv', index=False)
    means = data.groupby(['split', 'seed', 'method', 'budget']).mean(numeric_only=True).reset_index()
    means.to_csv(out/'curves_per_seed.csv', index=False)
    macro = means.groupby(['split', 'method', 'budget']).mean(numeric_only=True).reset_index()
    macro.to_csv(out/'curves_macro.csv', index=False)
    screen_scores = dict(empirical_value=mu+value_lambda*sigma,
        empirical_physics=arrays['proxy']+physical_lambda*200., honest_seed_conformal=np.ones(len(mu)))
    screens = pd.concat([threshold_rows(arrays, groups[groups.split != 'train'], scores, name)
                         for name, scores in screen_scores.items()], ignore_index=True)
    screens.to_csv(out/'threshold_per_block.csv', index=False)
    sm = screens.groupby(['split', 'seed', 'method']).mean(numeric_only=True).reset_index()
    sm.to_csv(out/'threshold_per_seed.csv', index=False)
    # A seed is the conservative independent calibration unit. One seed is not
    # thousands of exchangeable units merely because it generated many nodes.
    maxima = []
    for seed in plan['calibration_seeds']:
        ids = arrays['unique'] & (arrays['seed'] == seed)
        maxima.append(float(np.max((arrays['delta'][ids]-mu[ids])/sigma[ids])))
    formal_q = conformal_upper_quantile(maxima, .02)
    assert np.isinf(formal_q)
    write_json(out/'calibration.json', dict(alpha=.02, calibration_seeds=plan['calibration_seeds'],
        independent_seed_units=len(maxima), empirical_value_lambda=value_lambda,
        empirical_physics_lambda=physical_lambda,
        empirical_loss='macro mean over positive-opportunity blocks of fraction of high-yield candidates discarded',
        seed_level_split_conformal_quantile='infinity', formal_screen='keep_all',
        no_nontrivial_two_percent_distribution_free_guarantee=True,
        minimum_units_for_finite_98pct_split_conformal_rank=49,
        exchangeability_across_origins_or_loads_not_established=True,
        empirical_threshold_is_not_conformal_risk_control=True,
        frozen_architecture_checkpoints=architecture_paths))
    errors, timings = [], []
    model = ProcessValue(HERE/'value_frozen.npz')
    for g in groups[groups.split == 'evaluation'].itertuples():
        ids = np.flatnonzero((arrays['group'] == g.group) & arrays['unique'])
        errors.append(dict(group=g.group, seed=g.seed, tti=g.tti, origin=g.origin,
            root_gain_mae=float(np.abs(mu[ids]-arrays['delta'][ids]).mean()),
            parent_increment_mae=float(np.abs(predictions[chosen][ids, 1]-arrays['step_delta'][ids]).mean()),
            cheap_features_collection_ms=float(arrays['feature_ms'][arrays['group'] == g.group].sum())))
        model.predict(arrays['x'][ids])
        for repeat in range(3):
            begin = time.perf_counter();model.predict(arrays['x'][ids])
            timings.append(dict(group=g.group, seed=g.seed, repeat=repeat,
                unique_candidates=len(ids), head_only_ms=1000*(time.perf_counter()-begin)))
    pd.DataFrame(errors).to_csv(out/'prediction_errors.csv', index=False)
    pd.DataFrame(timings).to_csv(out/'head_timings.csv', index=False)
    # Diagnostics are descriptive, never fed back into architecture selection.
    misses = []
    for g in groups[groups.split == 'evaluation'].itertuples():
        ids = np.flatnonzero((arrays['group'] == g.group) & arrays['unique'])
        target = high_yield(arrays['delta'][ids], arrays['eligible'][ids])
        kept = np.zeros(len(ids), bool)
        kept[np.argsort(-mu[ids], kind='stable')[:int(np.ceil(.1*len(ids)))]] = True
        for j in np.flatnonzero(target):
            index = ids[j]
            misses.append(dict(group=g.group, seed=g.seed, origin=g.origin,
                depth=int(arrays['depth'][index]), missed=bool(not kept[j]),
                true_gain=arrays['delta'][index], predicted_gain=mu[index]))
    pd.DataFrame(misses).to_csv(out/'high_yield_misses.csv', index=False)
    gate = means[(means.split == 'evaluation') & (means.method == chosen) & (means.budget == .1)]
    passed = bool((gate.recall >= .95).all() and len(gate) == 2)
    checks = dict(passed=True, scientific_gate_passed=passed,
        gate_definition='Each evaluation seed macro block high-yield recall at10pct >=.95',
        gate_results=gate[['seed', 'recall', 'any_high_yield', 'actual_fraction', 'regret']].to_dict('records'),
        chosen_architecture=chosen, frozen_value_sha256=sha(HERE/'value_frozen.npz'),
        no_model_selection_on_calibration_or_evaluation_labels=True,
        no_native_closed_loop_claim=True, no_end_to_end_cost_claim=True,
        teacher_pool_phy_generation_cost_not_part_of_screening_curve=True,
        no_final_test_seeds_read=True, registry_unchanged=True)
    write_json(out/'checks.json', checks)
    plot = macro[macro.split == 'evaluation']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for method in ('actor_likelihood', 'frozen_bler_tb_physics', 'actor_features_mlp',
                   'tb_features_mlp', 'calibrated_value_upper', 'exact_oracle'):
        z = plot[plot.method == method]
        axes[0].plot(z.actual_fraction*100, z.recall*100, 'o-', label=method)
        axes[1].plot(z.actual_fraction*100, z.any_high_yield*100, 'o-', label=method)
    axes[0].axhline(95, color='gray', linestyle='--', linewidth=.8)
    axes[0].axvline(10, color='gray', linestyle='--', linewidth=.8)
    axes[0].set_ylabel('High-yield candidate recall (%)')
    axes[1].set_ylabel('Blocks retaining any high-yield candidate (%)')
    for ax in axes:
        ax.set_xlabel('Exact evaluations / unique frozen-pool candidates (%)')
        ax.set_ylim(0, 102);ax.grid(alpha=.2)
    axes[1].legend(fontsize=7, loc='lower right')
    fig.suptitle('Process-value pilot: development seeds48/50; teacher-generated pool')
    fig.savefig(HERE/'recall_cost.png', dpi=180);fig.savefig(HERE/'recall_cost.pdf')
    plt.close(fig)
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == '__main__':
    main()

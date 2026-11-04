"""Dependency unions and stale-exact ranking on frozen logs, with zero PHY."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import numpy as np

from .analyze_dpls_moves import ScalarSir, greedy_pack, write_json, write_csv, sha
from .dpls_operations import (edit_metadata, compact_neighborhood, infer_legal,
                              canonical_order)


def unpack(z, i):
    start, end = z['offsets'][i:i+2]
    root, prg, assignment = z['roots'][i], z['prg'][start:end], z['assignment'][start:end]
    users = len(z['values'])
    keys, affected, footprints = edit_metadata(root, prg, assignment, users)
    return dict(root=root, prg=prg, assignment=assignment, keys=keys, affected=affected,
                footprints=footprints, gain=z['delta'][start:end], eligible=z['eligible'][start:end],
                chosen=int(z['chosen_index'][i]), threshold=float(z['threshold'][i]), users=users)


def stale_rank(previous, current, *, k=16, epsilon=1.e-6, scope='within'):
    """No fresh delta labels enter the ranking. Unknown operators rank last."""
    _, old, new = np.intersect1d(previous['keys'], current['keys'], return_indices=True)
    chosen = current['chosen']
    known = np.zeros(len(current['keys']), bool)
    known[new] = True
    prior = np.full(len(known), np.nan)
    prior[new] = previous['gain'][old]
    def rank(pool):
        ranked = pool[canonical_order(prior[pool], current['keys'][pool], epsilon)]
        where = np.flatnonzero(ranked == chosen)
        return int(where[0]+1) if len(where) else None
    eligible = current['eligible']
    full_rank = rank(np.flatnonzero(known))
    filtered_rank = rank(np.flatnonzero(known & eligible))
    result = dict(known=len(new), total=len(known), chosen_known=bool(known[chosen]),
        rank_all=full_rank, rank_floor=filtered_rank,
        top16_all=full_rank is not None and full_rank <= k,
        top16_floor=filtered_rank is not None and filtered_rank <= k)
    if scope == 'within':
        taken = previous['chosen']
        dirty = np.zeros(len(known), bool)
        if taken >= 0:
            dirty = (current['footprints'] & previous['footprints'][taken]) != 0
        dirty |= ~known
        pool = np.flatnonzero(known & dirty & eligible)
        dirty_rank = rank(pool)
        fresh_new = int(np.sum(~known & eligible))
        result.update(chosen_dirty=bool(dirty[chosen]), chosen_new=not bool(known[chosen]),
            dirty_known_rank=dirty_rank, dirty_known_candidates=len(pool), new_eligible=fresh_new,
            dirty_top16_recovers=(not dirty[chosen]) or (not known[chosen]) or
                               (dirty_rank is not None and dirty_rank <= k),
            dirty_top16_fresh_evaluations=fresh_new+min(k, len(pool)))
    return result


def batch_union(current, legal, packed):
    root = current['root'].copy()
    for index in packed:
        root[:, current['prg'][index]] = current['assignment'][index]
    prg, assignment = compact_neighborhood(root, legal)
    keys, _, footprints = edit_metadata(root, prg, assignment, current['users'])
    _, old, new = np.intersect1d(current['keys'], keys, return_indices=True)
    union = np.bitwise_or.reduce(current['footprints'][packed], initial=np.uint64(0))
    dirty_common = int(np.sum((footprints[new] & union) != 0))
    novel = len(keys)-len(new)
    return dict(k=len(packed), next_candidates=len(keys), common=len(new), new=novel,
        dirty_common=dirty_common, union_common_fraction=dirty_common/max(1,len(new)),
        refresh=novel+dirty_common, refresh_fraction=(novel+dirty_common)/max(1,len(keys)))


def main(args):
    import torch
    from .archive import enable_legacy_types
    from .no_ctbs_inputs import problem_from_raw_row
    from .physics import NativePhy, ReferencePhy
    enable_legacy_types()
    source, out = Path(args.source).resolve(), Path(args.output).resolve()
    if not json.loads((source/'completion.json').read_text())['completed']:
        raise ValueError('Use completed frozen source')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'protocol.json', dict(source=str(source), epsilon_bytes=args.epsilon,
        packing='gain-ordered disjoint affected-UE/PRG moves, logged first move retained; above original1/16B threshold',
        shared_floor='enforced only on 12 saved-input states; structural-only others explicitly marked',
        unions='regenerate logical post-batch neighborhood; match canonical operations; novel + dirty common must refresh',
        work_model='per TTI N_first + sum(observed accepted transition post_batch_refresh/k); no multiply of independent savings',
        work_model_limit='logged-root local rate projection, not a simulated batch trajectory or speed measurement',
        ranking='historical exact deltas only; unknown operations last; max-anchored epsilon bands then canonical key',
        ranking_target='historical accepted operator at the logged root, not a rerun of canonical-tie search',
        dirty_budget='clean common free; eligible novel always paid; at most M dirty common, ordered by previous exact delta',
        floor_filter='current eligibility from logs stands in for cheap logical count/layer/TBS screen; unfiltered ranks also reported',
        cross_tti='prior final scored neighborhood; full arms have labels at executed terminal root; warm_n3 is before last accepted edit',
        new_candidate_phy_evaluations=0, new_mmse_evaluations=0, new_native_runs=0))
    union_rows, within_rows, cross_rows, models, inputs = [], [], [], [], {}
    with ExitStack() as stack:
        for cls in (NativePhy, ReferencePhy):
            stack.enter_context(patch.object(cls, 'evaluate', side_effect=AssertionError('No new PHY permitted')))
        for folder in sorted(source.glob('*_seed*')):
            if not folder.is_dir():
                continue
            arm, seed = folder.name.rsplit('_seed',1)
            preceding, preceding_tti = None, None
            for path in sorted((folder/'traces').glob('*.npz')):
                tti = int(path.stem.removeprefix('tti'))
                inputs[str(path)] = sha(path)
                state_path = folder/'states'/(path.stem+'.pt')
                sir = None
                if state_path.exists():
                    row = torch.load(state_path, map_location='cpu', weights_only=False)
                    sir = ScalarSir(problem_from_raw_row(row))
                    inputs[str(state_path)] = sha(state_path)
                identity = dict(arm=arm, seed=int(seed), tti=tti, floor_certified=sir is not None)
                with np.load(path) as z:
                    roots = [unpack(z, i) for i in range(len(z['roots']))]
                    if not all(z['evaluated']):
                        raise ValueError('Full scored neighborhoods required')
                    if not roots:
                        continue
                    legal = infer_legal(roots[0]['root'], roots[0]['prg'], roots[0]['assignment'], roots[0]['users'])
                    first = roots[0]
                    if preceding is not None and preceding_tti == tti-1 and first['chosen'] >= 0:
                        cross_rows.append(dict(identity, previous_tti=preceding_tti,
                            previous_at_executed_terminal=preceding['at_executed'],
                            **stale_rank(preceding, first, epsilon=args.epsilon, scope='cross')))
                    accum = {k:dict(projected_rounds=1., first_candidates=len(first['keys']),
                        projected_refresh=float(len(first['keys'])), projected_batch_full=float(len(first['keys'])),
                        baseline_candidates=sum(len(r['keys']) for r in roots), baseline_rounds=len(roots)) for k in ('1','2','4','8','all')}
                    for i, current in enumerate(roots):
                        if i > 0 and current['chosen'] >= 0:
                            within_rows.append(dict(identity, round=i+1,
                                **stale_rank(roots[i-1], current, epsilon=args.epsilon)))
                        if current['chosen'] < 0:
                            continue
                        pool = np.flatnonzero(current['eligible'] & (current['gain'] > current['threshold']))
                        kwargs = {}
                        if sir is not None:
                            logical = sir.score(current['root'], current['prg'], current['assignment'])
                            kwargs = dict(delta_d=logical['delta_d'], slack=logical['root_d']-z['floor'][i])
                        packed = greedy_pack(current['footprints'], current['gain'], pool, current['chosen'], **kwargs)
                        by_size = {}
                        for cap in ('1','2','4','8','all'):
                            chosen = packed if cap == 'all' else packed[:int(cap)]
                            size = len(chosen)
                            if not size:
                                raise AssertionError('Original accepted move missing from pack')
                            if size not in by_size:
                                by_size[size] = batch_union(current, legal, chosen)
                            record = dict(identity, round=i+1, cap=cap, has_next_logged_round=i+1<len(roots), **by_size[size])
                            union_rows.append(record)
                            if i+1 < len(roots):
                                if cap == '1':
                                    expected = roots[i+1]
                                    if record['next_candidates'] != len(expected['keys']):
                                        raise AssertionError('k1 logical domain differs from actual next round')
                                accum[cap]['projected_rounds'] += 1/size
                                accum[cap]['projected_refresh'] += record['refresh']/size
                                accum[cap]['projected_batch_full'] += record['next_candidates']/size
                    for cap, model in accum.items():
                        models.append(dict(identity, cap=cap, **model))
                    preceding = dict(roots[-1], at_executed=bool(np.array_equal(roots[-1]['root'], z['action'])))
                    preceding_tti = tti
                print('ANALYZED', folder.name, path.stem, flush=True)
    for name, records in [('union',union_rows),('within',within_rows),('cross_tti',cross_rows),('work_model',models)]:
        write_csv(out/(name+'.csv'), records)
    summaries = {}
    for arm in dict.fromkeys(r['arm'] for r in union_rows):
        a = {}
        for cap in ('1','2','4','8','all'):
            selected = [r for r in union_rows if r['arm']==arm and r['cap']==cap]
            m = [r for r in models if r['arm']==arm and r['cap']==cap]
            baseline = sum(r['baseline_candidates'] for r in m)
            a[cap] = dict(mean_k=float(np.mean([r['k'] for r in selected])),
                union_common_fraction=sum(r['dirty_common'] for r in selected)/sum(r['common'] for r in selected),
                projected_round_ratio=sum(r['projected_rounds'] for r in m)/sum(r['baseline_rounds'] for r in m),
                projected_combined_candidate_ratio=sum(r['projected_refresh'] for r in m)/baseline,
                projected_batch_only_candidate_ratio=sum(r['projected_batch_full'] for r in m)/baseline)
        for cap in a:
            a[cap]['incorrect_product_ratio'] = a['1']['projected_combined_candidate_ratio']*a[cap]['projected_batch_only_candidate_ratio']
        def ranked_summary(records):
            return dict(count=len(records), known_winner=sum(r['chosen_known'] for r in records),
                top16_all=float(np.mean([r['top16_all'] for r in records])) if records else None,
                top16_floor=float(np.mean([r['top16_floor'] for r in records])) if records else None)
        w = [r for r in within_rows if r['arm']==arm]
        d = [r for r in w if r['chosen_known'] and r['chosen_dirty']]
        a['within'] = dict(ranked_summary(w), dirty_known_winners=len(d),
            dirty_known_top16=float(np.mean([r['dirty_known_rank'] is not None and r['dirty_known_rank']<=16 for r in d])) if d else None,
            dirty_top16_best_recovered=float(np.mean([r['dirty_top16_recovers'] for r in w])),
            mean_fresh_candidates=float(np.mean([r['dirty_top16_fresh_evaluations'] for r in w])),
            clean_winners=sum(not r['chosen_dirty'] for r in w), novel_winners=sum(r['chosen_new'] for r in w))
        a['cross_tti'] = ranked_summary([r for r in cross_rows if r['arm']==arm])
        summaries[arm] = a
    write_json(out/'summary.json', summaries)
    write_json(out/'input_manifest.json', inputs)
    for name in ('analyze_dpls_joint_cost.py','dpls_operations.py'):
        shutil.copy2(Path(__file__).with_name(name), out/name)
    write_json(out/'completion.json', dict(completed=True, traces=96,
        new_native_runs=0, new_candidate_phy_evaluations=0, new_mmse_evaluations=0,
        guarded_phy_entrypoints=True, union_rows=len(union_rows), within_predictions=len(within_rows), cross_tti_predictions=len(cross_rows)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epsilon', type=float, default=1.e-6)
    main(parser.parse_args())

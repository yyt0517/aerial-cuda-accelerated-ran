"""Four offline move diagnostics from existing traces; never call a PHY scorer.

Delta changes match edit operators, not row numbers or absolute schedules.
Packing uses a sufficient interference/TB independence rule. SIR and exact
logical dequeue calculations are limited to inputs actually saved on disk.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
from pathlib import Path
import shutil
import warnings
from unittest.mock import patch
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def operators(root, prg, assignment, users):
    """Stable overwrite-operation keys and the native kernel's affected UEs."""
    if users+root.shape[1] > 63:
        raise ValueError('Offline footprint masks support at most 63 UE+PRG resources')
    old = root[:, prg].T
    edited = assignment != old
    targets = np.where(edited, assignment, -2)
    radix = users+3
    if (root.shape[1]*radix**root.shape[0]) >= 2**63:
        raise ValueError('Operator key exceeds int64')
    keys = prg.astype(np.int64).copy()
    for cell in range(root.shape[0]):
        keys = keys*radix+targets[:, cell]+2
    if len(np.unique(keys)) != len(keys):
        raise ValueError('Duplicate edit operators in a round')
    activity_change = np.any((old >= 0) != (assignment >= 0), axis=1)
    affected = np.zeros(len(prg), dtype=np.uint64)
    for cell in range(root.shape[0]):
        for ids in (old[:, cell], assignment[:, cell]):
            active = (ids >= 0) & (edited[:, cell] | activity_change)
            affected |= np.where(active, np.left_shift(np.uint64(1), np.maximum(ids, 0).astype(np.uint64)), 0)
    footprints = affected | np.left_shift(np.uint64(1), (users+prg).astype(np.uint64))
    return keys, affected, footprints


def greedy_pack(footprints, gain, pool, chosen=-1, delta_d=None, slack=None):
    """A constructive set; optionally preserve the logged first move and floor."""
    order = np.lexsort((pool, -gain[pool]))
    order = list(pool[order])
    if chosen >= 0 and chosen in order:
        order.remove(chosen)
        order.insert(0, chosen)
    used, selected, change = 0, [], 0.
    for index in order:
        mask = int(footprints[index])
        if used & mask:
            continue
        if delta_d is not None and change+delta_d[index] < -slack:
            continue
        selected.append(index)
        used |= mask
        if delta_d is not None:
            change += float(delta_d[index])
    return selected


def pack_bounds(footprints, gain, pool, resources, *, delta_d=None, slack=None, seconds=.25):
    """Maximum-cardinality packing, with certified bounds if MILP times out.

    For identical footprints retain max delta-D when a floor is present;
    otherwise any positive-gain representative suffices for cardinality.
    """
    if not len(pool):
        return dict(lower=0, upper=0, optimal=True)
    priority = gain if delta_d is None else delta_d
    order = pool[np.lexsort((pool, -priority[pool]))]
    _, keep = np.unique(footprints[order], return_index=True)
    selected = order[keep]
    masks = footprints[selected]
    matrix = ((masks[None] >> np.arange(resources, dtype=np.uint64)[:, None]) & np.uint64(1)).astype(float)
    lower, upper = np.full(resources, -np.inf), np.ones(resources)
    if delta_d is not None:
        matrix = np.vstack((matrix, delta_d[selected]))
        lower = np.r_[lower, -slack]
        upper = np.r_[upper, np.inf]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message="Unrecognized options detected:.*threads.*", category=RuntimeWarning)
        result = milp(-np.ones(len(selected)), integrality=np.ones(len(selected)),
            bounds=Bounds(0, 1), constraints=LinearConstraint(csc_matrix(matrix), lower, upper),
            options=dict(time_limit=seconds, mip_rel_gap=0., threads=1))
    constructive = len(greedy_pack(footprints, gain, pool, delta_d=delta_d, slack=slack))
    count = constructive if result.x is None else max(constructive, int(np.rint(result.x).sum()))
    dual = getattr(result, 'mip_dual_bound', None)
    bound = resources if dual is None or not np.isfinite(dual) else int(np.floor(-dual+1.e-6))
    if result.status == 0:
        bound = count
    return dict(lower=count, upper=max(count, bound), optimal=result.status == 0)


class ScalarSir:
    """Fixed trace-SIR proxy; no MMSE, BLER lookup or exact candidate J.

    Main proxy = sum weights*min(queue,TBS(count,min-layer))*log1p(mean SIR).
    Secondary proxy uses one-PRG capacity and a local log1p(SIR) difference.
    Both definitions are fixed before reading accepted-move rank outcomes.
    """
    def __init__(self, problem):
        self.p = problem
        self.received = np.sum(np.abs(problem.h)**2, axis=(-1, -2))
        self.signal = self.received[np.arange(problem.U), :, problem.serving]
        self.other = self.received.copy()
        self.other[np.arange(problem.U), :, problem.serving] = 0.

    def score(self, root, prg, assignment):
        p = self.p
        n = len(prg)
        ids = np.arange(p.U)
        assigned = root[p.serving] == ids[:, None]
        count = assigned.sum(1)
        caps = np.where(assigned, p.caps, 5)
        smallest = caps.min(1)
        unique = (caps == smallest[:, None]).sum(1) == 1
        second = np.where(caps > smallest[:, None], caps, 5).min(1)
        layer = np.where(count > 0, smallest, 4)
        root_d = np.minimum(p.queue, p.tbs[ids, count, np.where(count > 0, layer, 0)])
        denominator = 4*p.noise+np.einsum('cp,upc->up', root >= 0, self.other)
        sir = self.signal/denominator
        sir_sum = (sir*assigned).sum(1)
        root_utility = p.weights*root_d*np.log1p(np.divide(sir_sum, count, out=np.zeros(p.U), where=count > 0))
        old = root[:, prg].T
        edited = assignment != old
        dc = np.zeros((n, p.U), dtype=np.int16)
        rows = np.broadcast_to(np.arange(n)[:, None], old.shape)
        removed, added = edited & (old >= 0), edited & (assignment >= 0)
        np.add.at(dc, (rows[removed], old[removed]), -1)
        np.add.at(dc, (rows[added], assignment[added]), 1)
        candidate_count = count+dc
        local_caps = p.caps[:, prg].T
        candidate_layer = np.broadcast_to(layer, (n, p.U)).copy()
        candidate_layer = np.where((dc < 0) & unique & (local_caps == smallest), second, candidate_layer)
        candidate_layer = np.where(dc > 0, np.minimum(candidate_layer, local_caps), candidate_layer)
        candidate_layer = np.where(candidate_count > 0, candidate_layer, 0)
        candidate_d = np.minimum(p.queue, p.tbs[ids[None], candidate_count, candidate_layer])
        new_denominator = 4*p.noise+np.einsum('nc,nuc->nu', assignment >= 0, self.other[:, prg].transpose(1, 0, 2))
        new_sir = self.signal[:, prg].T/new_denominator
        old_assigned = root[p.serving[:, None], prg[None]].T == ids
        new_assigned = assignment[:, p.serving] == ids
        sums = sir_sum+new_assigned*new_sir-old_assigned*sir[:, prg].T
        means = np.divide(sums, candidate_count, out=np.zeros_like(sums), where=candidate_count > 0)
        tb_gain = (p.weights*candidate_d*np.log1p(np.maximum(means, 0.))).sum(1)-root_utility.sum()
        local_capacity = np.minimum(p.queue, p.tbs[ids[None], 1, local_caps])
        prg_gain = (p.weights*local_capacity*(new_assigned*np.log1p(new_sir)-old_assigned*np.log1p(sir[:, prg].T))).sum(1)
        return dict(tb=tb_gain, prg=prg_gain, delta_d=candidate_d.sum(1)-root_d.sum(), root_d=float(root_d.sum()))


def rank_of(scores, chosen, eligible=None):
    pool = np.arange(len(scores)) if eligible is None else np.flatnonzero(eligible)
    order = pool[np.lexsort((pool, -scores[pool]))]
    where = np.flatnonzero(order == chosen)
    return int(where[0]+1) if len(where) else None


def analyze_trace(path, *, sir=None, tolerance=1.e-6, pack_seconds=.25):
    records, transitions = [], []
    with np.load(path) as z:
        users = len(z['values'])
        rounds = len(z['roots'])
        previous = None
        cached = None
        guarded_cached = None
        for number, root in enumerate(z['roots']):
            start, end = z['offsets'][number:number+2]
            prg, assignment = z['prg'][start:end], z['assignment'][start:end]
            gain, eligible = z['delta'][start:end], z['eligible'][start:end]
            if not z['evaluated'][start:end].all():
                raise ValueError('These diagnostics require full scored neighborhoods')
            chosen, threshold = int(z['chosen_index'][number]), float(z['threshold'][number])
            keys, affected, footprints = operators(root, prg, assignment, users)
            pool = np.flatnonzero(eligible & (gain > tolerance))
            qualifying = np.flatnonzero(eligible & (gain > threshold))
            packed = greedy_pack(footprints, gain, pool, chosen)
            record = dict(round=number+1, candidates=len(gain), accepted_moves=int(chosen >= 0),
                edited_slots=int(np.sum(assignment[chosen] != root[:, prg[chosen]])) if chosen >= 0 else 0,
                positive_candidates=int(np.sum(gain > 0)), positive_eligible_candidates=int(np.sum(eligible & (gain > 0))),
                qualifying_candidates=len(qualifying), chosen_index=chosen,
                threshold=threshold, parallel_greedy=len(packed),
                parallel_greedy_gain=float(gain[packed].sum()),
                proxy_available=sir is not None)
            bounds = pack_bounds(footprints, gain, pool, users+root.shape[1], seconds=pack_seconds)
            record.update(parallel_lower=bounds['lower'], parallel_upper=bounds['upper'], parallel_optimal=bounds['optimal'])
            qualifying_bounds = (bounds if np.array_equal(pool, qualifying) else
                pack_bounds(footprints, gain, qualifying, users+root.shape[1], seconds=pack_seconds))
            record.update(qualifying_parallel_lower=qualifying_bounds['lower'],
                          qualifying_parallel_upper=qualifying_bounds['upper'])
            if previous is None:
                cached = gain.copy()
                guarded_cached = gain.copy()
                refresh = np.ones(len(gain), bool)
                record['lazy_refresh_candidates'] = len(gain)
            else:
                old = previous
                if old['chosen'] < 0:
                    if not np.array_equal(root, old['root']):
                        raise AssertionError('Rejected round changed its root')
                _, iold, inew = np.intersect1d(old['keys'], keys, return_indices=True)
                changed = np.abs(gain[inew]-old['gain'][iold]) > tolerance
                taken = old['chosen']
                invalid = np.zeros(len(keys), bool)
                if taken >= 0:
                    expected_root = old['root'].copy()
                    expected_root[:, old['prg'][taken]] = old['assignment'][taken]
                    if not np.array_equal(expected_root, root):
                        raise AssertionError('Logged accepted operator does not produce next root')
                    invalid = ((prg == old['prg'][taken]) |
                               ((affected & old['affected'][taken]) != 0))
                refresh = np.ones(len(keys), bool)
                refresh[inew] = invalid[inew]
                next_cached = gain.copy()
                next_cached[inew] = cached[iold]
                next_cached[refresh] = gain[refresh]
                cached = next_cached
                next_guarded = gain.copy()
                next_guarded[inew] = guarded_cached[iold]
                next_guarded[refresh] = gain[refresh]
                guarded_cached = next_guarded
                record['lazy_refresh_candidates'] = int(refresh.sum())
                if taken >= 0:
                    outside = ~invalid[inew]
                    transitions.append(dict(after_round=number, before_candidates=len(old['keys']),
                        next_candidates=len(keys), common=len(inew), new=len(keys)-len(inew), removed=len(old['keys'])-len(iold),
                        delta_changed=int(changed.sum()), delta_changed_fraction=float(changed.mean()) if len(changed) else None,
                        bitwise_delta_changed=int(np.sum(gain[inew] != old['gain'][iold])),
                        eligibility_flipped=int(np.sum(eligible[inew] != old['eligible'][iold])),
                        invalidated=int(refresh.sum()), reusable=int((~refresh).sum()),
                        outside_dependency=len(inew[outside]), outside_changed=int(np.sum(changed & outside)),
                        outside_max_abs_change=float(np.max(np.abs(gain[inew[outside]]-old['gain'][iold[outside]]), initial=0.)),
                        oracle_refresh=len(keys)-len(inew)+int(changed.sum())))
            candidates = np.flatnonzero(eligible)
            pick = int(candidates[np.argmax(cached[candidates])]) if len(candidates) else -1
            pick = pick if pick >= 0 and cached[pick] > threshold else -1
            record['cached_chosen_matches'] = pick == chosen
            record['cached_chosen_index'] = pick
            record['logged_chosen_gain'] = float(gain[chosen]) if chosen >= 0 else 0.
            record['cached_picked_true_gain'] = float(gain[pick]) if pick >= 0 else 0.
            record['cache_max_abs_error'] = float(np.max(np.abs(cached-gain), initial=0.))
            # Offline counterfactual: logged fresh values stand in for paid
            # tie/threshold rechecks. This adds no PHY calls and is no proof of
            # a production numerical error bound.
            top = float(guarded_cached[candidates].max()) if len(candidates) else -np.inf
            near = eligible & (top-guarded_cached <= tolerance)
            guarded_cached[near] = gain[near]
            record['tie_guard_extra_refresh'] = int(np.sum(near & ~refresh))
            guarded_pick = int(candidates[np.argmax(guarded_cached[candidates])]) if len(candidates) else -1
            guarded_pick = guarded_pick if guarded_pick >= 0 and guarded_cached[guarded_pick] > threshold else -1
            record['guarded_chosen_matches'] = guarded_pick == chosen
            if sir is not None:
                proxy = sir.score(root, prg, assignment)
                logical_eligible = proxy['root_d']+proxy['delta_d'] >= z['floor'][number]
                if not np.array_equal(logical_eligible, eligible):
                    raise AssertionError('Pure logical dequeue differs from logged PHY eligibility')
                if chosen >= 0:
                    for name in ('tb', 'prg'):
                        record[f'sir_{name}_rank_all'] = rank_of(proxy[name], chosen)
                        record[f'sir_{name}_rank_eligible'] = rank_of(proxy[name], chosen, eligible)
                floor_bounds = pack_bounds(footprints, gain, pool, users+root.shape[1],
                    delta_d=proxy['delta_d'], slack=proxy['root_d']-z['floor'][number], seconds=pack_seconds)
                floor_packed = greedy_pack(footprints, gain, pool, chosen,
                    proxy['delta_d'], proxy['root_d']-z['floor'][number])
                record.update(parallel_floor_lower=floor_bounds['lower'], parallel_floor_upper=floor_bounds['upper'],
                    parallel_floor_optimal=floor_bounds['optimal'],
                    parallel_floor_greedy=len(floor_packed),
                    parallel_floor_greedy_gain=float(gain[floor_packed].sum()))
            records.append(record)
            previous = dict(root=root, keys=keys, prg=prg, assignment=assignment, affected=affected, gain=gain,
                            eligible=eligible, chosen=chosen)
        if rounds and z['chosen_index'][-1] >= 0:
            # Final truncated accepted move has no subsequent scored neighborhood.
            records[-1]['next_round_missing'] = True
    return records, transitions


def distribution(values):
    data = np.asarray(values, float)
    if not len(data):
        return None
    return dict(mean=float(data.mean()), p50=float(np.quantile(data, .5)),
                p95=float(np.quantile(data, .95)), maximum=float(data.max()))


def summarize(rounds, transitions):
    result = dict(rounds=len(rounds), accepted_moves=sum(r['accepted_moves'] for r in rounds),
        accepted_per_round=distribution([r['accepted_moves'] for r in rounds]),
        edited_slots_per_accepted=distribution([r['edited_slots'] for r in rounds if r['accepted_moves']]),
        comparable_transitions=len(transitions),
        changed_fraction_per_transition=distribution([t['delta_changed_fraction'] for t in transitions]),
        changed_fraction_candidate_weighted=sum(t['delta_changed'] for t in transitions)/max(1, sum(t['common'] for t in transitions)),
        outside_dependency_changed=sum(t['outside_changed'] for t in transitions),
        outside_dependency_compared=sum(t['outside_dependency'] for t in transitions),
        outside_dependency_max_abs_change=max([t['outside_max_abs_change'] for t in transitions], default=0.),
        lazy_refresh_ratio=sum(r['lazy_refresh_candidates'] for r in rounds)/max(1, sum(r['candidates'] for r in rounds)),
        cached_choice_matches=sum(r['cached_chosen_matches'] for r in rounds),
        guarded_choice_matches=sum(r['guarded_chosen_matches'] for r in rounds),
        tie_guard_extra_refresh=sum(r['tie_guard_extra_refresh'] for r in rounds),
        parallel_count_lower=distribution([r['parallel_lower'] for r in rounds]),
        parallel_count_upper=distribution([r['parallel_upper'] for r in rounds]),
        parallel_greedy=distribution([r['parallel_greedy'] for r in rounds]),
        packing_optimal_rounds=sum(r['parallel_optimal'] for r in rounds))
    for key in ('parallel_floor_lower', 'parallel_floor_upper', 'parallel_floor_greedy'):
        result[key] = distribution([r[key] for r in rounds if key in r])
    for name in ('tb', 'prg'):
        for kind in ('all', 'eligible'):
            key = f'sir_{name}_rank_{kind}'
            values = [r[key] for r in rounds if r.get(key) is not None]
            result[key] = dict(count=len(values), distribution=distribution(values),
                              recall={str(k):float(np.mean(np.array(values) <= k)) for k in (16,32,64,128,256)}) if values else None
    return result


def main(args):
    import torch
    from .archive import enable_legacy_types
    from .no_ctbs_inputs import problem_from_raw_row
    from .physics import NativePhy, ReferencePhy
    if args.pack_seconds <= 0 or args.tolerance <= 0:
        raise ValueError('Positive numerical tolerance and packing time limit required')
    enable_legacy_types()
    source = Path(args.source).resolve()
    out = Path(args.output).resolve()
    if not json.loads((source/'completion.json').read_text())['completed']:
        raise ValueError('Only completed source experiments are accepted')
    out.mkdir(parents=True, exist_ok=False)
    manifest = {}
    write_json(out/'protocol.json', dict(source=str(source), tolerance_bytes=args.tolerance,
        new_native_runs=0, new_candidate_phy_evaluations=0, new_mmse_evaluations=0,
        operation_identity='PRG and overwritten cell targets; unchanged cell is a distinct sentinel',
        changed_delta='compare matching operators across an accepted move; abs difference > tolerance',
        dependency='same PRG OR overlapping affected UE sets; source/target UEs plus all active PRG neighbors on activity changes',
        packing='one move per PRG and disjoint affected UEs; logged eligible delta>1e-6B; original1/16B qualifying subset separate',
        floor='structural packing alone is not a shared-floor certificate; exact logical floor checked on saved-input subset',
        sir='TB-aware weighted dequeue*log1p(mean trace SIR); secondary local one-PRG log1p SIR; no BLER/MMSE',
        packing_seconds=args.pack_seconds,
        sample='96 correlated seed41 transient TTIs; scalar inputs only12 TTIs; no new-seed inference'))
    all_rounds, all_transitions = [], []
    trace_count = 0
    with ExitStack() as stack:
        # A call would be a task-scope violation, not free offline analysis.
        for cls in (NativePhy, ReferencePhy):
            stack.enter_context(patch.object(cls, 'evaluate', side_effect=AssertionError('New PHY evaluation forbidden')))
        for folder in sorted(source.glob('*_seed*')):
            if not folder.is_dir():
                continue
            for trace in sorted((folder/'traces').glob('*.npz')):
                trace_count += 1
                state = folder/'states'/(trace.stem+'.pt')
                sir = None
                if state.exists():
                    row = torch.load(state, map_location='cpu', weights_only=False)
                    sir = ScalarSir(problem_from_raw_row(row))
                    manifest[str(state)] = sha(state)
                records, transitions = analyze_trace(trace, sir=sir, tolerance=args.tolerance,
                                                     pack_seconds=args.pack_seconds)
                identity = dict(arm=folder.name.rsplit('_seed', 1)[0], seed=int(folder.name.rsplit('_seed', 1)[1]),
                                tti=int(trace.stem.removeprefix('tti')))
                all_rounds.extend(dict(identity, **r) for r in records)
                all_transitions.extend(dict(identity, **r) for r in transitions)
                manifest[str(trace)] = sha(trace)
                print('ANALYZED', folder.name, trace.stem, flush=True)
    write_csv(out/'rounds.csv', all_rounds)
    write_csv(out/'transitions.csv', all_transitions)
    summaries = {}
    for arm in dict.fromkeys(r['arm'] for r in all_rounds):
        summaries[arm] = summarize([r for r in all_rounds if r['arm'] == arm],
                                   [t for t in all_transitions if t['arm'] == arm])
    write_json(out/'summary.json', summaries)
    write_json(out/'input_manifest.json', manifest)
    shutil.copy2(__file__, out/'analyze_dpls_moves.py')
    write_json(out/'completion.json', dict(completed=True, traces=trace_count,
        rounds=len(all_rounds), transitions=len(all_transitions),
        new_native_runs=0, new_candidate_phy_evaluations=0, new_mmse_evaluations=0,
        guarded_phy_entrypoints=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tolerance', type=float, default=1.e-6)
    parser.add_argument('--pack-seconds', type=float, default=.25)
    main(parser.parse_args())

"""Paid full/cached canonical baseline gate on twelve captured causal inputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .cached_dpls import CanonicalDplsConfig,solve_canonical
from .analyze_dpls_moves import write_json,write_csv,sha
from .physics import ROOT,ReferencePhy
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft


def main(args):
    import torch
    from .archive import enable_legacy_types
    enable_legacy_types()
    source,out = Path(args.source).resolve(),Path(args.output).resolve()
    paths = sorted(source.glob('*_seed*/states/tti*.pt'))
    if not paths:
        raise ValueError('No saved causal inputs')
    out.mkdir(parents=True,exist_ok=False)
    registry = ROOT/'training/stageb_hgraph/current_algorithm.json'
    registry_sha = sha(registry)
    sources = [Path(__file__).with_name(name) for name in ('cached_dpls.py','cached_dpls_gate.py',
        'dpls_operations.py','physics.py','physics_kernel.cpp','_physics.so','ranked_dpls.py',
        'no_ctbs_inputs.py','archive.py')]
    source_hashes = {str(p.relative_to(ROOT)):sha(p) for p in sources}
    for p in sources:
        target = out/'source'/p.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(p,target)
    write_json(out/'source_manifest.json',source_hashes)
    write_json(out/'input_manifest.json',{str(p):sha(p) for p in paths})
    write_json(out/'protocol.json',dict(source=str(source),baseline_version='canonical_dependency_v1',
        epsilon_bytes=args.epsilon,
        tie='true eligible qualifying maximum anchored epsilon band, then smallest canonical PRG/overwrite-target key',
        threshold='each selected delta strictly >1B in rounds1/2, >16B later; epsilon does not soften acceptance',
        paid_work='one newly scored complete canonical baseline + one cached decision per captured input; separate NumPy terminal audit',
        initializers='source-arm actual predecessor, current legality/current PHY; bootstrap and PFQ evaluation charged',
        gate='bitwise equality of all candidate PHY fields, scores, eligibility, root, chosen key and terminal action each round',
        cache='affected-UE exact values; current-root unaffected fields; full original-layout J/D reductions, no cached global delta',
        timing='observer includes storing/checking all candidate fields; no fair wall-time speedup gate claimed',
        snapshots=len(paths),native_runs=0,confirmation_seeds_used=False,registry_sha256=registry_sha))
    decisions,round_rows = [],[]
    for path in paths:
        row = torch.load(path,map_location='cpu',weights_only=False)
        p = problem_from_raw_row(row)
        draft = raw_pfq_draft(p,row['obs'])
        warm = row['previous_action'] if row['source_arm'].startswith('warm_') else None
        captured = []
        def remember(info):
            item = {k:info[k].copy() for k in ('keys','values','scores','eligible','gain')}
            item.update(root=info['root'].actions.copy(),root_values=info['root'].values.copy(),
                        chosen=info['chosen'],floor=info['floor'])
            captured.append(item)
        begin = time.perf_counter()
        full = solve_canonical(p,draft,CanonicalDplsConfig('full',args.epsilon),warm_start=warm,observer=remember)
        full_elapsed = (time.perf_counter()-begin)*1000
        audited_rounds = audited_candidates = 0
        def audit(info):
            nonlocal audited_rounds,audited_candidates
            index = info['iteration']-1
            if index >= len(captured):
                raise AssertionError('Cached search took an additional round')
            expected = captured[index]
            for k in ('keys','values','scores','eligible','gain'):
                np.testing.assert_array_equal(info[k],expected[k],err_msg=f'{path}:round{index+1}:{k}')
            np.testing.assert_array_equal(info['root'].actions,expected['root'])
            np.testing.assert_array_equal(info['root'].values,expected['root_values'])
            if info['chosen'] != expected['chosen'] or info['floor'] != expected['floor']:
                raise AssertionError('Different canonical choice/floor')
            audited_rounds += 1
            audited_candidates += len(info['keys'])
        begin = time.perf_counter()
        cached = solve_canonical(p,draft,CanonicalDplsConfig('cached',args.epsilon),warm_start=warm,observer=audit)
        cached_elapsed = (time.perf_counter()-begin)*1000
        if len(captured) != audited_rounds or not full[2]['converged'] or not cached[2]['converged']:
            raise AssertionError('Round count/convergence gate failed')
        np.testing.assert_array_equal(full[0],cached[0])
        np.testing.assert_array_equal(full[1],cached[1])
        np.testing.assert_allclose(cached[1],ReferencePhy(p).evaluate(cached[0])[0],rtol=1.e-8,atol=1.e-6)
        identity = dict(source_arm=row['source_arm'],seed=row['seed'],tti=row['tti'])
        result = dict(identity,passed=True,audited_rounds=audited_rounds,
            audited_complete_candidates=audited_candidates,
            full_candidate_evaluations=full[2]['candidate_evaluations'],
            cached_candidate_evaluations=cached[2]['candidate_evaluations'],
            phy_candidate_ratio=cached[2]['candidate_evaluations']/full[2]['candidate_evaluations'],
            full_mmse_requests=full[2]['mmse_requests'],cached_mmse_requests=cached[2]['mmse_requests'],
            full_mmse_misses=full[2]['mmse_misses'],cached_mmse_misses=cached[2]['mmse_misses'],
            full_ue_evaluations=full[2]['ue_evaluations'],cached_ue_evaluations=cached[2]['ue_evaluations'],
            full_observed_ms=full_elapsed,cached_observed_ms=cached_elapsed,
            full_observer_ms=full[2]['observer_ms'],cached_observer_ms=cached[2]['observer_ms'],
            score=full[2]['score'],floor=full[2]['floor'])
        old_path = path.parent.parent/'traces'/(path.stem+'.npz')
        if row['source_arm'].endswith('_full'):
            with np.load(old_path) as old:
                old_score = float(old['values'][:,6]@p.weights)
                result.update(historical_full_terminal_changed=not np.array_equal(full[0],old['action']),
                              score_minus_historical_full=full[2]['score']-old_score)
        for mode,answer in [('full',full),('cached',cached)]:
            for record in answer[2]['history']:
                round_rows.append(dict(identity,mode=mode,**record))
        np.savez_compressed(out/f"{row['source_arm']}_seed{row['seed']}_tti{row['tti']}.npz",
            action=full[0],values=full[1],roots=np.concatenate([x['root'] for x in captured]),
            root_values=np.concatenate([x['root_values'] for x in captured]),
            chosen_keys=np.array([int(x['keys'][x['chosen']]) if x['chosen']>=0 else -1 for x in captured]))
        decisions.append(result)
        print('GATE',row['source_arm'],row['tti'],'rounds',audited_rounds,'PHY ratio',result['phy_candidate_ratio'],flush=True)
    for name,digest in source_hashes.items():
        if sha(ROOT/name) != digest:
            raise AssertionError('Source changed during experiment: '+name)
    if sha(registry) != registry_sha:
        raise AssertionError('Registry changed')
    write_csv(out/'decisions.csv',decisions)
    write_csv(out/'rounds.csv',round_rows)
    write_json(out/'completion.json',dict(completed=True,engineering_equivalence_passed=True,
        snapshots=len(decisions),rounds=sum(r['audited_rounds'] for r in decisions),
        bitwise_candidate_phy_audits=sum(r['audited_complete_candidates'] for r in decisions),
        new_full_candidate_evaluations=sum(r['full_candidate_evaluations'] for r in decisions),
        new_cached_candidate_evaluations=sum(r['cached_candidate_evaluations'] for r in decisions),
        new_full_mmse_misses=sum(r['full_mmse_misses'] for r in decisions),
        new_cached_mmse_misses=sum(r['cached_mmse_misses'] for r in decisions),
        numpy_terminal_audits=len(decisions),new_native_runs=0,registry_unchanged=True,
        wall_speed_gate_evaluated=False,closed_loop_quality_gate_evaluated=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--epsilon',type=float,default=1.e-6)
    main(parser.parse_args())

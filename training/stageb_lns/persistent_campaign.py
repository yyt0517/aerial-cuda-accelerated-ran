"""New integer full baseline, persistent equivalence and guarded cap4 gate."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from .physics import ROOT,NativePhy,ReferencePhy,EXPECTED
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .persistent_dpls import solve_persistent,solve_integer_reference,PersistentConfig,persistent_library
from .cached_dpls import solve_canonical,CanonicalDplsConfig
from .service_gate_calibration import absolute_service
from .calibrated_envelope import passes
from .cached_budget_campaign import service
from .analyze_dpls_moves import write_json,write_csv,sha


def main(args):
    import torch
    from .archive import enable_legacy_types
    enable_legacy_types()
    source,reference,envelope,out=map(lambda x:Path(x).resolve(),(args.source,args.reference,args.envelope,args.output))
    out.mkdir(parents=True,exist_ok=False)
    paths=sorted(source.glob('*_seed*/states/tti*.pt'))
    limits=json.loads((envelope/'limits.json').read_text())
    files=[Path(__file__),*[Path(__file__).with_name(n) for n in ('persistent_dpls.py','persistent_kernel.cpp','_persistent.so',
        'build_persistent.py','physics_kernel.cpp','physics.py','_physics.so','no_ctbs_inputs.py','dpls_operations.py')]]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
    (out/'source').mkdir()
    for p in files:shutil.copy2(p,out/'source'/p.name)
    write_json(out/'source_manifest.json',hashes)
    write_json(out/'input_manifest.json',{str(p):sha(p) for p in paths})
    write_json(out/'protocol.json',dict(source=str(source),reference=str(reference),envelope=str(envelope),
        version='persistent_q1e6_v1',quantization='per UE nearbyint((expected*weight)*1e6), nearest-even, sum int64',
        bounds='positive integer queues, sum ceil(queue*weight*1e6)+1 < INT64_MAX/4; fail outside contract',
        threshold_q='strict >1e6 first2 rounds, >16e6 later; true maximum anchored one-unit tie then canonical key',
        cache='persistent per-PRG edits; regenerate only changed PRGs; invalidate other entries by affected-UE union; sparse payload and integer deltas',
        equivalence='new independent Python full-array/full-integer-sum baseline versus native full-refresh/cache: every round all candidate PHY fields and integer scores/choices exact',
        guard='compare candidate batch to canonical best-single counterfactual; reject whole batch on new unallocated UE or any causal deficit-top20% UE expected loss>1e-6B; fall back to single',
        guard_phy='no extra scoring; all needed per-UE payloads already scored and independent',
        service='report old single-reference gate and pre-frozen full empirical envelope separately; not a probability/fairness certificate',
        entry='all12 guarded4 empirical-envelope passes required for two-seed2s native development',
        native_runs=0,learning=False,dirty_M_paused=True))
    rows=[];rounds=[];ledger=[];gate_candidates=gate_rounds=0
    for state,path in enumerate(paths):
        row=torch.load(path,map_location='cpu',weights_only=False);p=problem_from_raw_row(row)
        draft=raw_pfq_draft(p,row['obs']);warm=row['previous_action'] if row['source_arm'].startswith('warm_') else None
        with np.load(reference/f"{row['source_arm']}_seed{row['seed']}_tti{row['tti']}.npz") as saved:
            oldvalues=saved['values'].copy();oldaction=saved['action'].copy()
        captured=[]
        def capture(info):captured.append({k:v.copy() if isinstance(v,np.ndarray) else v for k,v in info.items()})
        baseline=solve_integer_reference(p,draft,warm_start=warm,observer=capture)
        ledger.append(dict(state=state,arm='integer_reference',kind='online',**baseline[2]))
        for mode in ('full','cached','cap4','guarded4'):
            audited=[0,0]
            root_auditor=NativePhy(p) if mode in ('cap4','guarded4') else None
            def audit(info):
                nonlocal gate_candidates,gate_rounds
                if mode in ('full','cached'):
                    expected=captured[info['iteration']-1]
                    for key in ('root','root_values','keys','values','delta_q','delta_d','selected_keys'):
                        np.testing.assert_array_equal(info[key],expected[key],err_msg=f'{state}:{mode}:{key}')
                    if info['score_q']!=expected['score_q'] or info['floor']!=expected['floor']:
                        raise AssertionError('Integer objective/floor gate mismatch')
                    audited[0]+=1;audited[1]+=len(info['keys'])
                else:
                    actual=root_auditor.evaluate(info['root']).values[0]
                    np.testing.assert_array_equal(info['root_values'],actual)
                    if info['root_values'][:,3].sum()<info['floor']:raise AssertionError('Batch floor failed')
                rounds.append(dict(state=state,source_arm=row['source_arm'],tti=row['tti'],mode=mode,
                    iteration=info['iteration'],candidates=len(info['keys']),scored=info['scored'],
                    selected_keys=info['selected_keys'].tolist(),proposed=info['proposed'],guard_rejected=info['guard_rejected']))
            answer=solve_persistent(p,draft,PersistentConfig(mode),warm_start=warm,observer=audit)
            a,v,r=answer
            if not r['converged']:raise AssertionError('Persistent mode did not converge')
            if mode in ('full','cached'):
                np.testing.assert_array_equal(a,baseline[0]);np.testing.assert_array_equal(v,baseline[1])
                if audited[0]!=len(captured):raise AssertionError('Different canonical path length')
                gate_rounds+=audited[0];gate_candidates+=audited[1]
            if root_auditor is not None:
                np.testing.assert_array_equal(v,root_auditor.evaluate(a).values[0])
                ledger.append(dict(state=state,arm=mode,kind='root_audit',**root_auditor.stats));root_auditor.close()
            np.testing.assert_allclose(v,ReferencePhy(p).evaluate(a)[0],rtol=1.e-8,atol=1.e-6)
            absolute=dict(J=float(v[:,EXPECTED]@p.weights),**absolute_service(p,v))
            oldj=float(oldvalues[:,EXPECTED]@p.weights);s=service(v,oldvalues)
            oldpass=absolute['J']>=.995*oldj and s['newly_unallocated']==0 and s['positive_p5_ratio']>=.98 and s['positive_weak_ratio']>=.98
            result=dict(state=state,source_arm=row['source_arm'],tti=row['tti'],mode=mode,
                old_J_ratio=absolute['J']/oldj,old_single_gate_passed=oldpass,
                empirical_envelope_passed=passes(absolute,limits[state]),
                terminal_changed_from_float=not np.array_equal(a,oldaction),
                rounds=r['rounds'],score_q=r['score_q'],candidate_evaluations=r['candidate_evaluations'],
                generated_edits=r['generated_edits'],sum_neighborhood_sizes=r['sum_neighborhood_sizes'],
                regenerated_prgs=r['regenerated_prgs'],reused_edits=r['reused_edits'],
                guard_rejections=r['guard_rejections'],accepted_batches=r['accepted_batches'],
                accepted_moves=r['accepted_moves'],**absolute,**s)
            rows.append(result)
            ledger.append(dict(state=state,arm=mode,kind='online',**{k:r[k] for k in ('candidate_evaluations','ue_evaluations','mmse_requests','mmse_misses')}))
            np.savez_compressed(out/f'state{state:02d}_{mode}.npz',action=a,values=v)
            print('PERSISTENT',state,mode,'oldJ',round(result['old_J_ratio'],6),'envelope',result['empirical_envelope_passed'],flush=True)
    write_csv(out/'metrics.csv',rows);write_csv(out/'rounds.csv',rounds);write_json(out/'work_ledger.json',ledger)
    for file,digest in hashes.items():
        if sha(ROOT/file)!=digest:raise AssertionError('Source changed: '+file)
    guard=[r for r in rows if r['mode']=='guarded4']
    write_json(out/'completion.json',dict(completed=True,integer_equivalence_passed=True,
        audited_full_and_cache_rounds=gate_rounds,audited_full_and_cache_candidates=gate_candidates,
        guarded_envelope_passes=sum(r['empirical_envelope_passed'] for r in guard),
        guarded_states=len(guard),native_entry_authorized_by_gate=all(r['empirical_envelope_passed'] for r in guard),
        online_candidate_evaluations=sum(r['candidate_evaluations'] for r in ledger if r['kind']=='online'),
        audit_candidate_evaluations=sum(r['candidate_evaluations'] for r in ledger if r['kind']=='root_audit'),
        numpy_terminal_audits=len(rows),new_native_runs=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--envelope',required=True);p.add_argument('--output',required=True);main(p.parse_args())

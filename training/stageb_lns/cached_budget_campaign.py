"""Cap4 and M16/32/64 terminal quality, certified convergence and paid regret.

Regret labels for unscored dirty candidates are queried AFTER each decision
in an independent PHY. They never change the executing policy or its budget.
Exact-cache dirty labels are also saved without additional PHY work for a
future low-dimensional ranker; no model is trained in this experiment.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from .physics import ROOT,NativePhy,ReferencePhy,EXPECTED,DEQUEUE,COUNT
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .dpls_operations import expand_edits,canonical_choice
from .budgeted_cached_dpls import solve_budgeted,BudgetedCacheConfig
from .analyze_dpls_moves import write_json,write_csv,sha


ARMS={'cached':BudgetedCacheConfig('cached'),'cap4':BudgetedCacheConfig('batch',batch_cap=4),
      'M16':BudgetedCacheConfig('dirty',dirty_m=16),'M32':BudgetedCacheConfig('dirty',dirty_m=32),
      'M64':BudgetedCacheConfig('dirty',dirty_m=64)}
FEATURES=('old_exact_gain','affected_ues','sum_weighted_expected_change','sum_abs_weighted_expected_change',
          'max_abs_weighted_expected_change','sum_resource_count_change','sum_layer_change','sum_bler_change')


def service(values,reference):
    target=reference[:,EXPECTED];actual=values[:,EXPECTED]
    positive=np.flatnonzero(target>1.)
    weak=positive[np.argsort(target[positive],kind='stable')[:max(1,int(np.ceil(.2*len(positive))))]]
    p5=float(np.quantile(actual[positive],.05)/np.quantile(target[positive],.05)) if len(positive) else 1.
    wr=float(actual[weak].sum()/target[weak].sum()) if len(weak) else 1.
    return dict(reference_positive_ues=len(positive),positive_p5_ratio=p5,positive_weak_ratio=wr,
        newly_unallocated=int(np.sum((reference[:,COUNT]>0)&(values[:,COUNT]==0))),
        newly_ineffective=int(np.sum((target>1.)&(actual<=1.))),
        service_drops_gt2pct=int(np.sum(actual[positive]<.98*target[positive])))


def main(args):
    import torch
    from .archive import enable_legacy_types
    enable_legacy_types()
    source,reference,out=Path(args.source).resolve(),Path(args.reference).resolve(),Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    paths=sorted(source.glob('*_seed*/states/tti*.pt'))
    files=[Path(__file__),Path(__file__).with_name('budgeted_cached_dpls.py'),
        Path(__file__).with_name('dpls_operations.py'),Path(__file__).with_name('physics.py'),
        Path(__file__).with_name('no_ctbs_inputs.py'),Path(__file__).with_name('_physics.so')]
    sources={str(p.relative_to(ROOT)):sha(p) for p in files}
    (out/'source').mkdir()
    for p in files:shutil.copy2(p,out/'source'/p.name)
    write_json(out/'source_manifest.json',sources)
    write_json(out/'input_manifest.json',{str(p):sha(p) for p in paths})
    write_json(out/'protocol.json',dict(source=str(source),reference=str(reference),arms=list(ARMS),
        mode='same 12 captured causal states; source-arm predecessor provenance retained; no native closed-loop',
        batch='cap4; distinct PRG and affected-UE sets; running aggregate dequeue floor; joint PHY verification charged online',
        dirty='first scan full; clean valid cache retained, novel all paid; dirty-known old exact delta top M; labels may age',
        rank_floor='no free current-eligibility screen before dirty ranking; exact current eligibility only for verified candidates',
        persistent_invalidity='unscored dirty entries never become valid without fresh PHY',
        stop='failed shortlist scans unscored dirty complement; convergence only when all candidates current and no >threshold gain',
        regret='chosen single/first move exact delta / true maximum eligible delta; absolute regret also recorded; stop states separate',
        regret_teacher='independent post-decision PHY scores only unscored dirty actions, plus audit root; charged separately',
        terminal='original floating weighted J; expected-byte snapshot service, not native packets/latency or throughput',
        offline_snapshot_gate='J>=99.5% full, no new unallocated UE, positive-service P5 and weak sum>=98%; report failures',
        quantization='diagnostic only: round each weighted expected UE contribution to 1e-6B, sum int64, compare choice at same root; no quantized policy executed',
        learning='no training; save old exact gain + affected-UE changes against newly evaluated exact dirty labels in cached arm',
        seeds='all12 inputs seed41, correlated exposed development; labels are not independent scenes'))
    metrics,regrets,rounds,ues,quants=[],[],[],[],[]
    label_x=[];label_y=[];label_state=[];label_round=[];label_keys=[]
    work=[]
    for state,path in enumerate(paths):
        row=torch.load(path,map_location='cpu',weights_only=False)
        p=problem_from_raw_row(row)
        draft=raw_pfq_draft(p,row['obs'])
        warm=row['previous_action'] if row['source_arm'].startswith('warm_') else None
        refpath=reference/f"{row['source_arm']}_seed{row['seed']}_tti{row['tti']}.npz"
        with np.load(refpath) as saved:
            ref_values=saved['values'].copy();ref_action=saved['action'].copy()
            ref_roots=saved['roots'].copy();ref_keys=saved['chosen_keys'].copy()
        reference_j=float(ref_values[:,EXPECTED]@p.weights)
        for arm,cfg in ARMS.items():
            visits=[]
            previous_root=None
            def capture(info):
                nonlocal previous_root
                visits.append(dict(root=info['root'].actions[0].copy(),root_score=info['root_score'],
                    prg=info['prg'].copy(),assignment=info['assignment'].copy(),gain=info['gain'].copy(),
                    eligible=info['eligible'].copy(),valid=info['valid'].copy(),chosen=info['chosen'],
                    threshold=info['threshold'],floor=info['floor'],iteration=info['iteration'],
                    selected=info['selected'].copy(),fallback=info['fallback']))
                if arm=='cached':
                    i=info['iteration']-1
                    np.testing.assert_array_equal(info['root'].actions[0],ref_roots[i])
                    key=int(info['keys'][info['chosen']]) if info['chosen']>=0 else -1
                    if key!=ref_keys[i]:raise AssertionError('Exact cached control changed canonical path')
                    # Sensitivity uses already-scored online fields, no new PHY.
                    scale=1_000_000
                    safe=sum(int(np.ceil(q*w*scale))+1 for q,w in zip(p.queue,p.weights))
                    if safe>=np.iinfo(np.int64).max//2:raise OverflowError('Unsafe quantization bound')
                    q=np.rint(info['values'][:,:,EXPECTED]*p.weights*scale).astype(np.int64).sum(1,dtype=np.int64)
                    qroot=int(np.rint(info['root'].values[0,:,EXPECTED]*p.weights*scale).astype(np.int64).sum())
                    choice=canonical_choice(q-qroot,info['eligible'],info['keys'],int(info['threshold']*scale),1)
                    float_j=info['gain']+info['root_score']
                    quants.append(dict(state=state,round=info['iteration'],choice_changed=choice!=info['chosen'],
                        max_absolute_J_quantization_error=float(np.max(np.abs(q/scale-float_j),initial=0.))))
                    if previous_root is not None:
                        indices=np.flatnonzero(info['fresh'] & np.isfinite(info['prior_labels']))
                        support=((info['affected'][indices,None]>>np.arange(p.U,dtype=np.uint64))&1)!=0
                        change=info['root'].values[0]-previous_root
                        de=change[:,EXPECTED]*p.weights
                        weighted=support*de
                        x=np.column_stack((info['prior_labels'][indices],support.sum(1),weighted.sum(1),
                            np.abs(weighted).sum(1),np.abs(weighted).max(1,initial=0.),
                            (support*change[:,COUNT]).sum(1),(support*change[:,1]).sum(1),
                            (support*change[:,5]).sum(1)))
                        label_x.append(x.astype(np.float32));label_y.append(info['gain'][indices].copy())
                        label_state.append(np.full(len(indices),state,np.int16))
                        label_round.append(np.full(len(indices),info['iteration'],np.int16))
                        label_keys.append(info['keys'][indices].copy())
                    previous_root=info['root'].values[0].copy()
            action,values,record=solve_budgeted(p,draft,cfg,warm_start=warm,observer=capture)
            if not record['converged']:raise AssertionError('Round cap, not convergence')
            if arm=='cached':
                np.testing.assert_array_equal(action,ref_action)
                np.testing.assert_array_equal(values,ref_values)
            np.testing.assert_allclose(values,ReferencePhy(p).evaluate(action)[0],rtol=1.e-8,atol=1.e-6)
            identity=dict(state=state,source_arm=row['source_arm'],seed=row['seed'],tti=row['tti'],arm=arm)
            audit_phy=None
            for visit in visits:
                truth=visit['gain'].copy();eligible=visit['eligible'].copy()
                missing=np.flatnonzero(~visit['valid'])
                if len(missing):
                    if audit_phy is None:audit_phy=NativePhy(p)
                    base=audit_phy.evaluate(visit['root'])
                    extra=audit_phy.evaluate(expand_edits(visit['root'],visit['prg'][missing],visit['assignment'][missing]),base)
                    truth[missing]=extra.values[:,:,EXPECTED]@p.weights-visit['root_score']
                    eligible[missing]=extra.values[:,:,DEQUEUE].sum(1)>=visit['floor']
                maximum=max(0.,float(np.max(truth[eligible],initial=0.)))
                chosen=visit['chosen']
                selected_gain=float(truth[chosen]) if chosen>=0 else 0.
                if chosen<0 and maximum>visit['threshold']+1.e-6:
                    raise AssertionError('False convergence/false empty shortlist')
                regrets.append(dict(identity,round=visit['iteration'],accepted=chosen>=0,
                    true_maximum_delta=maximum,selected_delta=selected_gain,
                    gain_retention=selected_gain/maximum if chosen>=0 and maximum>0 else None,
                    regret_bytes=max(0.,maximum-selected_gain) if chosen>=0 else None,
                    unscored_audited=len(missing),selected_moves=len(visit['selected'])))
            audit_stats=dict(candidate_evaluations=0,ue_evaluations=0,mmse_requests=0,mmse_misses=0)
            if audit_phy is not None:audit_stats=audit_phy.stats;audit_phy.close()
            s=service(values,ref_values)
            ratio=float(values[:,EXPECTED]@p.weights/reference_j)
            passed=ratio>=.995 and s['newly_unallocated']==0 and s['positive_p5_ratio']>=.98 and s['positive_weak_ratio']>=.98
            metrics.append(dict(identity,J_ratio=ratio,score=float(values[:,EXPECTED]@p.weights),
                snapshot_quality_service_passed=passed,rounds=record['rounds'],
                candidate_evaluations=record['candidate_evaluations'],fallback_candidates=record['fallback_candidates'],
                fallback_rounds=record['fallback_rounds'],joint_evaluations=record['joint_evaluations'],
                first_scan_candidates=record['first_scan_candidates'],converged=record['converged'],**s))
            rounds.extend(dict(identity,**entry) for entry in record['history'])
            for u in range(p.U):
                ues.append(dict(identity,ue=u,expected=values[u,EXPECTED],reference_expected=ref_values[u,EXPECTED],
                    resources=values[u,COUNT],reference_resources=ref_values[u,COUNT],
                    dequeue=values[u,DEQUEUE],reference_dequeue=ref_values[u,DEQUEUE]))
            work.append(dict(identity,online={k:record[k] for k in ('candidate_evaluations','ue_evaluations','mmse_requests','mmse_misses')},
                             regret_audit=audit_stats,numpy_terminal_audits=1))
            np.savez_compressed(out/f"state{state:02d}_{arm}.npz",action=action,values=values)
            print('QUALITY',state,arm,'J/full',round(ratio,6),'rounds',record['rounds'],flush=True)
    for name,data in [('metrics',metrics),('regret',regrets),('rounds',rounds),('per_ue',ues),('quantization',quants)]:
        write_csv(out/(name+'.csv'),data)
    write_json(out/'work_ledger.json',work)
    np.savez_compressed(out/'exact_dirty_labels.npz',x=np.concatenate(label_x),y=np.concatenate(label_y),
        state=np.concatenate(label_state),round=np.concatenate(label_round),key=np.concatenate(label_keys))
    write_json(out/'label_schema.json',dict(features=list(FEATURES),target='current exact delta J in bytes',
        labels=sum(len(y) for y in label_y),source='fresh scored dirty-known candidates in exact cached control; no added PHY labels',
        captured_inputs=len(paths),independent_states_not_established=True,independent_seeds=1,training_started=False))
    for path,digest in sources.items():
        if sha(ROOT/path)!=digest:raise AssertionError('Source changed during campaign: '+path)
    write_json(out/'completion.json',dict(completed=True,decisions=len(metrics),new_native_runs=0,
        online_candidate_evaluations=sum(r['online']['candidate_evaluations'] for r in work),
        audit_candidate_evaluations=sum(r['regret_audit']['candidate_evaluations'] for r in work),
        online_mmse_misses=sum(r['online']['mmse_misses'] for r in work),
        audit_mmse_misses=sum(r['regret_audit']['mmse_misses'] for r in work),
        numpy_terminal_audits=len(metrics),regret_roots=len(regrets),all_converged=all(r['converged'] for r in metrics)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--output',required=True);main(p.parse_args())

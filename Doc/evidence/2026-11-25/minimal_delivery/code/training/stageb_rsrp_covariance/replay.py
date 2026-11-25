"""Serial cached/refresh timing, explicit amortization, and paired PHY audits."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from training.stageb_ppo.native import write, sha
from training.stage2_closure.revise_metrics import FIELDS
from training.stageb_lns.archive import enable_legacy_types
from training.stageb_lns.no_ctbs_inputs import problem_from_raw_row, raw_pfq_draft
from training.stageb_lns.bounded_dpls import solve_bounded
from training.stageb_lns.physics import ReferencePhy
from training.stageb_rsrp_ablation.model import check_counters
from training.stageb_rsrp_ablation.replay import first_round_audit
from .analyze import local
from .model import (STAGES, SPATIAL_STAGES, SpatialReceiver, from_row, setting,
                    normalized_shapes, unpack_shape)


def inputs(row, stage, receiver=None):
    p=problem_from_raw_row(row);p.weights=np.asarray(row['objective_weights']).copy()
    draft=raw_pfq_draft(p,row['obs'])
    return from_row(p,stage,row,receiver),draft


def evaluate(row, stage, receiver=None, **kwargs):
    p,draft=inputs(row,stage,receiver)
    return solve_bounded(p,draft,**kwargs)


def prepared_receiver(row, stage):
    r=SpatialReceiver()
    if stage in SPATIAL_STAGES:
        report=row['spatial_reports'][stage]
        r.accept(int(report['sample_tti']),report['packet'])
    return r


def timed(row, stage, cached, regime, expected):
    receiver=cached if regime=='cached' else SpatialReceiver()
    start=time.perf_counter();cpu=time.process_time()
    result=evaluate(row,stage,receiver)
    cpu_ms=(time.process_time()-cpu)*1000;wall_ms=(time.perf_counter()-start)*1000
    np.testing.assert_array_equal(result[0],expected[0]);np.testing.assert_array_equal(result[1],expected[1])
    return dict(cpu_ms=cpu_ms,wall_ms=wall_ms)


def summarize_timings(records, output, prefix):
    df=pd.DataFrame(records);df.to_csv(output/(prefix+'_raw.csv'),index=False)
    identifiers=['scene','seed','objective','source_stage','tti','stage']
    rows=[]
    for identity,g in df.groupby(identifiers):
        rec=dict(zip(identifiers,identity));stage=rec['stage']
        period=setting(stage)[1] if stage in SPATIAL_STAGES else 1
        for clock in ('cpu_ms','wall_ms'):
            cache=g[g.regime=='cached'][clock]
            refresh=g[g.regime=='refresh'][clock] if stage in SPATIAL_STAGES else cache
            rec['cached_'+clock]=cache.mean();rec['refresh_'+clock]=refresh.mean()
            rec['amortized_'+clock]=(1-1/period)*cache.mean()+refresh.mean()/period
            rec['cached_p95_'+clock]=cache.quantile(.95);rec['refresh_p95_'+clock]=refresh.quantile(.95)
        rows.append(rec)
    states=pd.DataFrame(rows);states.to_csv(output/(prefix+'_per_state.csv'),index=False)
    cols=[c for c in states if c not in identifiers]
    seed=states.groupby(['scene','seed','objective','stage'],as_index=False)[cols].mean()
    seed.to_csv(output/(prefix+'_per_seed.csv'),index=False)
    seed.groupby(['scene','objective','stage'],as_index=False)[cols].mean().to_csv(output/(prefix+'_summary.csv'),index=False)
    return seed


def shape_diagnostics(p, row, stage):
    current=normalized_shapes(p.h,p.serving)['prg']
    if stage=='exact':predicted=current;age=0
    elif stage=='reported':predicted=np.broadcast_to(np.eye(4),current.shape);age=None
    else:
        packet=row['spatial_reports'][stage]
        predicted=np.broadcast_to(unpack_shape(packet['packet']),current.shape)
        age=int(row['tti'])-1-int(packet['sample_tti'])
        assert 1 <= age <= setting(stage)[1]
    relative=np.linalg.norm(predicted-current,axis=(-2,-1))/np.linalg.norm(current,axis=(-2,-1))
    a=current-np.eye(4);b=predicted-np.eye(4)
    denom=np.linalg.norm(a,axis=(-2,-1))*np.linalg.norm(b,axis=(-2,-1))
    valid=denom>1e-14
    similarity=np.real((a.conj()*b).sum(axis=(-2,-1)))[valid]/denom[valid]
    return dict(covariance_age_tti=age,normalized_shape_relative_error=float(relative.mean()),
        centered_shape_similarity=float(similarity.mean()) if len(similarity) else None)


def chosen_action_truth(p, expected):
    """Untimed original-channel utility of each model's selected action.

    The exact-policy action is a reference, not a global optimum. Ratios use
    the same recorded queue weights and native MCS/layers for every model.
    """
    phy=ReferencePhy(p)
    actual={stage:phy.evaluate(result[0])[0] for stage,result in expected.items()}
    reference=float(np.dot(p.weights,actual['exact'][:,6]))
    rows={}
    for stage,result in expected.items():
        truth=float(np.dot(p.weights,actual[stage][:,6]))
        predicted=float(np.dot(p.weights,result[1][:,6]))
        rows[stage]=dict(selected_true_expected_bytes=float(actual[stage][:,6].sum()),
            selected_true_utility_vs_exact_pct=100*(truth/reference-1) if reference>0 else None,
            selected_predicted_utility_error_pct=100*(predicted/truth-1) if truth>0 else None)
    return rows


def main(args):
    import torch
    torch.set_num_threads(1);enable_legacy_types()
    source=Path(args.source);output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    assert json.loads((source/'completion.json').read_text())['completed']
    cpu=sorted(os.sched_getaffinity(0))[-1];os.sched_setaffinity(0,{cpu})
    all_specs=[json.loads(p.read_text()) for p in sorted(source.glob('*/done.json'))]
    witnesses=[s for s in all_specs if s['stage']=='prg_t1']
    timings=[];mechanisms=[];certificates=[];root_errors=[];checks=[]
    for spec in witnesses:
        d=local(spec['directory'])
        horizon=int(json.loads((d/'launch.json').read_text())['horizon'])
        native_metrics=np.memmap(d/'metrics.bin',dtype='<f8',mode='r').reshape(horizon,36,len(FIELDS))
        samples=json.loads((d/'saved_diagnostic_states.json').read_text())
        assert len(samples)==6
        for sample_index,sample in enumerate(samples):
            path=local(sample['path']);assert sha(path)==sample['sha256']
            row=torch.load(path,map_location='cpu',weights_only=False)
            identity=dict(scene=spec['scene'],seed=spec['seed'],objective=spec['objective'],
                          source_stage=spec['stage'],tti=int(row['tti']))
            original_h=row['snapshot']['h'].copy()
            expected={stage:evaluate(row,stage) for stage in STAGES}
            cached={stage:prepared_receiver(row,stage) for stage in STAGES}
            np.testing.assert_array_equal(expected[spec['stage']][0],row['selected_action'])
            np.testing.assert_array_equal(expected[spec['stage']][1],row['selected_values'])
            for result in expected.values():assert result[2]['converged'];check_counters(result[2])
            for repeat in range(args.repeats):
                shift=repeat%len(STAGES);order=STAGES[shift:]+STAGES[:shift]
                for stage in order:
                    regimes=('cached','refresh') if stage in SPATIAL_STAGES else ('cached',)
                    if repeat%2:regimes=regimes[::-1]
                    for regime in regimes:
                        timings.append(dict(**identity,stage=stage,repeat=repeat,regime=regime,
                            **timed(row,stage,cached[stage],regime,expected[stage])))
            original_problem=problem_from_raw_row(row)
            original_problem.weights=np.asarray(row['objective_weights']).copy()
            selected_truth=chosen_action_truth(original_problem,expected)
            native=native_metrics[int(row['tti'])-1]
            dequeued=native[:,FIELDS.index('scheduled_dequeue_bytes')]
            active_native=dequeued>0
            native_expected=float(np.sum(dequeued[active_native]*(1-native[active_native,FIELDS.index('predicted_bler_cpu')])))
            native_expected_error=abs(selected_truth[spec['stage']]['selected_true_expected_bytes']-native_expected)
            assert native_expected_error<float(dequeued.sum())*.0005+1e-8
            truth=None;common_keys=None;common_eligibility=None
            for stage in STAGES:
                p,draft=inputs(row,stage);first={}
                def observe(info):
                    if not first:first.update({k:v.copy() if isinstance(v,np.ndarray) else v for k,v in info.items()})
                    # One phase per witness: independent exhaustive ALL-round audit.
                    # Remaining phases: independently audit the full first neighborhood.
                    if sample_index==0 or info['iteration']==1:
                        audit,_=first_round_audit(p,info['root'],info)
                        certificates.append(dict(**identity,stage=stage,iteration=info['iteration'],
                            all_rounds_audited=sample_index==0,candidates=audit['first_candidates'],
                            fresh_paid=audit['first_scored'],required=audit['first_required'],
                            bound_violations=audit['first_bound_violation_count']))
                observed=evaluate(row,stage,observer=observe)
                np.testing.assert_array_equal(observed[0],expected[stage][0])
                np.testing.assert_array_equal(observed[1],expected[stage][1])
                audit,values=first_round_audit(p,draft,first)
                eligibility=first['dequeue']+first['delta_d']>=first['floor']
                if truth is None:
                    truth=values;common_keys=first['keys'];common_eligibility=eligibility
                np.testing.assert_array_equal(first['keys'],common_keys)
                np.testing.assert_array_equal(eligibility,common_eligibility)
                active=truth[:,0]>0;error=values[active,4]-truth[active,4]
                for u in np.flatnonzero(active):
                    root_errors.append(dict(**identity,stage=stage,ue_id=int(u),
                        sinr_error_db=float(values[u,4]-truth[u,4]),
                        true_sinr_db=float(truth[u,4]),estimated_sinr_db=float(values[u,4])))
                mechanisms.append(dict(**identity,stage=stage,**audit,
                    **{k:v for k,v in expected[stage][2].items() if k not in ('version','core_ms')},
                    root_sinr_bias_db=float(error.mean()),root_sinr_mae_db=float(np.abs(error).mean()),
                    root_sinr_var_db2=float(error.var()),**shape_diagnostics(original_problem,row,stage),
                    **selected_truth[stage]))
            np.testing.assert_array_equal(row['snapshot']['h'],original_h)
            checks.append(dict(**identity,path=str(path),sha256=sha(path),passed=True,
                chosen_action_native_expected_bytes_error=native_expected_error))
            print('REPLAYED',spec['key'],row['tti'],flush=True)
    timing_seed=summarize_timings(timings,output,'timing')
    mechanism=pd.DataFrame(mechanisms);mechanism.to_csv(output/'mechanism_raw.csv',index=False)
    excluded=['scene','seed','objective','source_stage','tti','stage']
    cols=[c for c in mechanism if c not in excluded and pd.api.types.is_numeric_dtype(mechanism[c])]
    per_seed=mechanism.groupby(['scene','seed','objective','stage'],as_index=False)[cols].mean()
    per_seed.to_csv(output/'mechanism_per_seed.csv',index=False)
    per_seed.groupby(['scene','objective','stage'],as_index=False)[cols].mean().to_csv(output/'mechanism_summary.csv',index=False)
    pd.DataFrame(root_errors).to_csv(output/'common_root_errors.csv',index=False)
    certificate=pd.DataFrame(certificates);certificate.to_csv(output/'certificate_rounds.csv',index=False)
    assert (certificate.fresh_paid==certificate.required).all() and not certificate.bound_violations.any()
    # Each policy's own archived queues/weights are a separate, complementary comparison.
    own=[];own_checks=[]
    for spec in all_specs:
        for sample in json.loads((local(spec['directory'])/'saved_states.json').read_text()):
            path=local(sample['path']);assert sha(path)==sample['sha256']
            row=torch.load(path,map_location='cpu',weights_only=False);stage=spec['stage']
            reference=evaluate(row,stage);cached=prepared_receiver(row,stage)
            np.testing.assert_array_equal(reference[0],row['selected_action'])
            np.testing.assert_array_equal(reference[1],row['selected_values'])
            identity=dict(scene=spec['scene'],seed=spec['seed'],objective=spec['objective'],source_stage=stage,tti=int(row['tti']))
            for repeat in range(args.repeats):
                regimes=('cached','refresh') if stage in SPATIAL_STAGES else ('cached',)
                if repeat%2:regimes=regimes[::-1]
                for regime in regimes:
                    own.append(dict(**identity,stage=stage,repeat=repeat,regime=regime,
                                    **timed(row,stage,cached,regime,reference)))
            own_checks.append(dict(**identity,path=str(path),sha256=sha(path),passed=True))
    summarize_timings(own,output,'own_timing')
    intervals=[]
    for (scene,objective,stage),g in timing_seed.groupby(['scene','objective','stage']):
        for control in ('exact','reported'):
            base=timing_seed[(timing_seed.scene==scene)&(timing_seed.objective==objective)&(timing_seed.stage==control)]
            paired=g.merge(base,on='seed',suffixes=('','_base'),validate='one_to_one')
            for metric in ('cached_cpu_ms','refresh_cpu_ms','amortized_cpu_ms'):
                delta=paired[metric]-paired[metric+'_base'];n=len(delta);mean=float(delta.mean())
                half=float(student_t.ppf(.975,n-1)*delta.std(ddof=1)/np.sqrt(n)) if n>1 else None
                intervals.append(dict(scene=scene,objective=objective,stage=stage,control=control,
                    metric=metric,n=n,mean_delta=mean,ci_low=mean-half if half is not None else None,
                    ci_high=mean+half if half is not None else None))
    pd.DataFrame(intervals).to_csv(output/'paired_intervals.csv',index=False)
    write(output/'validation.json',dict(passed=True,completed=True,cpu=cpu,repeats=args.repeats,
        matched_states=len(checks),own_states=len(own_checks),checks=checks,own_checks=own_checks,
        chosen_action_native_max_expected_bytes_error=max(x['chosen_action_native_expected_bytes_error'] for x in checks),
        certificate_rounds=len(certificate),candidate_occurrences=int(certificate.candidates.sum()),
        fresh_paid=int(certificate.fresh_paid.sum()),required=int(certificate.required.sum()),bound_violations=0,
        audit_scope='all rounds for first diagnostic phase of each witness; complete first round for other phases',
        timing='serial fixed CPU, observer off; fresh PHY contexts; cyclic stage order; alternating cache/refresh order',
        amortization='cached*(1-1/K)+refresh/K; sampled-state estimate, not full online latency or p95 guarantee',
        state_scope='six stratified/dithered report phases on prg_t1 trajectories, all models on identical states; own 3 states separately',
        chosen_action_truth='untimed NumPy PHY on original channel, identical recorded weights/MCS/layers; exact-policy action is a local-search reference, not a global optimum',
        exclusions='native simulator, channel generation, GPU PFQ, measurement emulator, HOL calculation; weights recorded',
        adapter='normalized covariance decoded into cached factor for existing H-based PHY interface; direct covariance API not implemented'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--repeats',type=int,default=12);main(parser.parse_args())

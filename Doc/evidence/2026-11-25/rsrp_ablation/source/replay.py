"""Paired serial timings and an exhaustive all-visited-round certificate audit.

Timing disables observers/profiling. A separate untimed pass independently
scores every candidate in every visited round. First-round comparisons use
the same PFQ root for all six models, separating bound changes from a changed
later search/queue trajectory.
"""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from training.stageb_ppo.native import write, sha
from training.stageb_lns.archive import enable_legacy_types
from training.stageb_lns.no_ctbs_inputs import problem_from_raw_row, raw_pfq_draft
from training.stageb_lns.bounded_dpls import solve_bounded, BoundedConfig, certificate_sets
from training.stageb_lns.physics import NativePhy, DEQUEUE, EXPECTED, SINR
from training.stageb_lns.dpls_operations import compact_neighborhood, expand_edits, edit_metadata, canonical_choice
from training.stageb_lns.persistent_dpls import SCALE, q_contributions
from .analyze import local
from .model import STAGES, from_row, check_counters


def first_round_audit(p, draft, first):
    np.testing.assert_array_equal(first['root'],draft)
    prg,assignment=compact_neighborhood(draft,p.legal)
    keys,affected,_=edit_metadata(draft,prg,assignment,p.U)
    # The C++ table may use its own stable row order; align by canonical key.
    order=np.argsort(keys);positions=order[np.searchsorted(keys[order],first['keys'])]
    np.testing.assert_array_equal(keys[positions],first['keys'])
    prg,assignment,affected=prg[positions],assignment[positions],affected[positions]
    support=(affected[:,None] >> np.arange(p.U,dtype=np.uint64)[None,:])&1
    support=support.astype(bool)
    phy=NativePhy(p)
    try:
        root=phy.evaluate(draft)
        values=phy.evaluate(expand_edits(draft,prg,assignment),root).values
    finally:phy.close()
    np.testing.assert_array_equal(root.values[0],first['root_values'])
    root_q=q_contributions(root.values[0],p.weights)
    result_q=q_contributions(values,p.weights)
    capacity_q=np.rint((values[:,:,DEQUEUE]*p.weights)*SCALE).astype(np.int64)
    upper=((capacity_q-root_q)*support).sum(1,dtype=np.int64)
    gain=(result_q-root_q).sum(1,dtype=np.int64)
    slack=((capacity_q-result_q)*support).sum(1,dtype=np.int64)
    delta_d=(values[:,:,DEQUEUE].sum(1)-root.values[0,:,DEQUEUE].sum()).astype(np.int64)
    np.testing.assert_array_equal(upper,first['upper'])
    np.testing.assert_array_equal(delta_d,first['delta_d'])
    np.testing.assert_array_equal(upper-gain,slack)
    assert np.all(slack>=0)
    valid=(first['status']&1)!=0
    np.testing.assert_array_equal(gain[valid],first['delta_q'][valid])
    eligible=first['dequeue']+delta_d>=first['floor']
    chosen=canonical_choice(gain,eligible,first['keys'],first['threshold'],1)
    expected=first['keys'][chosen:chosen+1] if chosen>=0 else np.empty(0,np.int64)
    np.testing.assert_array_equal(expected,first['selected_keys'])
    required,fresh=certificate_sets(first,dict(delta_q=gain))
    np.testing.assert_array_equal(required,fresh)
    capacity=((values[:,:,DEQUEUE]*p.weights)*support).sum(1)
    failure=((values[:,:,DEQUEUE]-values[:,:,EXPECTED])*p.weights*support).sum(1)
    assert len(keys)>0 and eligible.any()
    prg_upper=[upper[(prg==i)&eligible].max()/SCALE for i in range(p.P) if ((prg==i)&eligible).any()]
    prg_gain=[gain[(prg==i)&eligible].max()/SCALE for i in range(p.P) if ((prg==i)&eligible).any()]
    return dict(first_candidates=len(keys),first_eligible=int(eligible.sum()),
        first_scored=int(fresh.sum()),first_paid_fraction=float(fresh.sum()/eligible.sum()),
        first_slack_weighted_bytes=float(slack[eligible].mean()/SCALE),
        first_failure_fraction=float(failure[eligible].sum()/capacity[eligible].sum()),
        first_upper_weighted_bytes=float(upper[eligible].mean()/SCALE),
        first_best_gain_weighted_bytes=float(gain[eligible].max()/SCALE),
        first_upper_std=float(upper[eligible].std()/SCALE),first_gain_std=float(gain[eligible].std()/SCALE),
        first_prg_max_upper_std=float(np.std(prg_upper)),first_prg_max_gain_std=float(np.std(prg_gain)),
        first_unique_upper_fraction=float(len(np.unique(upper[eligible]))/eligible.sum()),
        first_required=int(required.sum()),first_paid_to_required=float(fresh.sum()/required.sum()) if required.any() else None,
        first_bound_violation_count=int((gain>upper).sum()),
        first_root_failure_fraction=float(1-(root.values[0,:,EXPECTED]@p.weights)/
            (root.values[0,:,DEQUEUE]@p.weights)),
        first_threshold_pruned=int((eligible&(upper<=first['threshold'])).sum()),
        first_certificate_pruned=int((eligible&(upper>first['threshold'])&(~fresh)).sum())), root.values[0]


def main(args):
    import torch
    torch.set_num_threads(1);enable_legacy_types()
    source=Path(args.source);output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    assert json.loads((source/'completion.json').read_text())['completed']
    available=sorted(os.sched_getaffinity(0));cpu=args.cpu if args.cpu is not None else available[-1]
    os.sched_setaffinity(0,{cpu})
    all_specs=[json.loads(p.read_text()) for p in sorted(source.glob('*/done.json'))]
    # Balance both endpoints; do not select states based on observed degradation.
    specs=[s for s in all_specs if s['stage'] in ('exact','reported')]
    reports={(s['scene'],s['seed']):s for s in all_specs if s['stage']=='reported' and s['objective']=='deficit'}
    timings=[];mechanisms=[];profiles=[];checks=[];rounds=[];certificates=[];root_errors=[]
    for spec in specs:
        d=local(spec['directory'])
        for sample in json.loads((d/'saved_states.json').read_text()):
            path=local(sample['path']);assert sha(path)==sample['sha256']
            row=torch.load(path,map_location='cpu',weights_only=False)
            report_spec=reports[spec['scene'],spec['seed']]
            report_row=torch.load(local(report_spec['directory'])/'states'/path.name,map_location='cpu',weights_only=False)
            for field in ('h','w'):np.testing.assert_array_equal(row['snapshot'][field],report_row['snapshot'][field])
            row=dict(row,rsrp_report=report_row['rsrp_report'],rsrp_filtered=report_row['rsrp_filtered'])
            original=row['snapshot']['h'].copy()
            identity=dict(scene=spec['scene'],seed=spec['seed'],objective=spec['objective'],
                          source_stage=spec['stage'],tti=int(row['tti']),
                          report_age_tti=int(row['tti'])-1-int(report_row['rsrp_report_tti']))
            def inputs(stage):
                p=problem_from_raw_row(row);p.weights=np.asarray(row['objective_weights']).copy()
                draft=raw_pfq_draft(p,row['obs']);return from_row(p,stage,row),draft
            def evaluate(stage,**kwargs):
                p,draft=inputs(stage);return solve_bounded(p,draft,**kwargs)
            expected={stage:evaluate(stage) for stage in STAGES}
            for result in expected.values():assert result[2]['converged'];check_counters(result[2])
            np.testing.assert_array_equal(expected[spec['stage']][0],row['selected_action'])
            np.testing.assert_array_equal(expected[spec['stage']][1],row['selected_values'])
            for repeat in range(args.repeats):
                # Six cyclic rotations place each model in every position once.
                rotated=STAGES[repeat%6:]+STAGES[:repeat%6]
                for stage in rotated:
                    start=time.perf_counter();start_cpu=time.process_time()
                    result=evaluate(stage)
                    cpu_ms=(time.process_time()-start_cpu)*1000;wall_ms=(time.perf_counter()-start)*1000
                    np.testing.assert_array_equal(result[0],expected[stage][0])
                    np.testing.assert_array_equal(result[1],expected[stage][1])
                    timings.append(dict(**identity,stage=stage,repeat=repeat,cpu_ms=cpu_ms,wall_ms=wall_ms))
            root_truth=None;common_keys=None;common_eligibility=None
            for stage in STAGES:
                first={}
                p,draft=inputs(stage)
                def observe(info):
                    if not first:first.update({k:v.copy() if isinstance(v,np.ndarray) else v for k,v in info.items()})
                    audit,_=first_round_audit(p,info['root'],info)
                    certificates.append(dict(**identity,stage=stage,iteration=info['iteration'],
                        candidates=audit['first_candidates'],floor_eligible=audit['first_eligible'],
                        fresh_paid=audit['first_scored'],certificate_required=audit['first_required'],
                        paid_to_required=audit['first_paid_to_required'],bound_violations=audit['first_bound_violation_count']))
                    eligible=info['dequeue']+info['delta_d']>=info['floor']
                    rounds.append(dict(**identity,stage=stage,iteration=info['iteration'],
                        candidates=len(info['keys']),paid=info['scored'],known_before=info['known_before'],
                        eligible=int(eligible.sum()),accepted=len(info['selected_keys'])))
                observed=evaluate(stage,observer=observe)
                np.testing.assert_array_equal(observed[0],expected[stage][0])
                np.testing.assert_array_equal(observed[1],expected[stage][1])
                p,draft=inputs(stage);audit,root_values=first_round_audit(p,draft,first)
                eligibility=first['dequeue']+first['delta_d']>=first['floor']
                if root_truth is None:
                    root_truth=root_values;common_keys=first['keys'];common_eligibility=eligibility
                np.testing.assert_array_equal(common_keys,first['keys'])
                np.testing.assert_array_equal(common_eligibility,eligibility)
                active=root_truth[:,0]>0
                bias=root_values[active,SINR]-root_truth[active,SINR]
                for u in np.flatnonzero(active):
                    root_errors.append(dict(**identity,stage=stage,ue_id=int(u),native_mcs=int(p.mcs[u]),
                        sinr_estimate_db=float(root_values[u,SINR]),sinr_original_db=float(root_truth[u,SINR]),
                        sinr_error_db=float(root_values[u,SINR]-root_truth[u,SINR]),
                        bler_estimate=float(root_values[u,5]),bler_original=float(root_truth[u,5])))
                counters={k:v for k,v in expected[stage][2].items() if k not in ('version','core_ms')}
                mechanisms.append(dict(**identity,stage=stage,**audit,**counters,
                    root_sinr_bias_db=float(bias.mean()),root_sinr_mae_db=float(np.abs(bias).mean()),
                    root_sinr_var_db2=float(bias.var()),root_optimistic_fraction=float((bias>0).mean()),
                    changed_positions_vs_exact=int((expected[stage][0]!=expected['exact'][0]).sum())))
                stamp=time.process_time()
                profile_p=problem_from_raw_row(row);profile_p.weights=np.asarray(row['objective_weights']).copy()
                profile_draft=raw_pfq_draft(profile_p,row['obs']);input_end=time.process_time()
                profile_p=from_row(profile_p,stage,row);conversion_end=time.process_time()
                profiled=solve_bounded(profile_p,profile_draft,config=BoundedConfig(diagnostics=True))
                solve_end=time.process_time()
                np.testing.assert_array_equal(profiled[0],expected[stage][0])
                np.testing.assert_array_equal(profiled[1],expected[stage][1])
                profiles.append(dict(**identity,stage=stage,
                    input_reconstruction_cpu_ms=(input_end-stamp)*1000,
                    model_conversion_cpu_ms=(conversion_end-input_end)*1000,
                    profiled_solver_cpu_ms=(solve_end-conversion_end)*1000,
                    **{k:v for k,v in profiled[2].items() if k.endswith('_ms')}))
            np.testing.assert_array_equal(original,row['snapshot']['h'])
            checks.append(dict(**identity,path=str(path),sha256=sha(path),passed=True,
                saved_decision_exact=True,first_round_full_oracle=True,quantized_slack_identity=True,
                certificate_required_equals_paid=True,all_stages_same_root_and_eligibility=True))
            print('REPLAYED',spec['key'],row['tti'],flush=True)
    for name,records in [('timing',timings),('mechanism',mechanisms),('profile',profiles),('rounds',rounds)]:
        df=pd.DataFrame(records);df.to_csv(output/(name+'_raw.csv'),index=False)
        if name=='rounds':continue
        ids=['scene','seed','objective','stage'];exclude=ids+['source_stage','tti','repeat','report_age_tti']
        metrics=[c for c in df if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        per_seed=df.groupby(ids,as_index=False)[metrics].mean();per_seed.to_csv(output/(name+'_per_seed.csv'),index=False)
        per_seed.groupby(['scene','objective','stage'],as_index=False)[metrics].mean().to_csv(output/(name+'_summary.csv'),index=False)
    # Own-trajectory latency includes intermediate policies' visited states;
    # keep it separate from the controlled common-state model comparison.
    own=[r for r in timings if r['stage']==r['source_stage']]
    own_checks=[]
    for spec in all_specs:
        if spec['stage'] in ('exact','reported'):continue
        for sample in json.loads((local(spec['directory'])/'saved_states.json').read_text()):
            path=local(sample['path']);assert sha(path)==sample['sha256']
            row=torch.load(path,map_location='cpu',weights_only=False)
            def own_evaluate():
                p=problem_from_raw_row(row);p.weights=np.asarray(row['objective_weights']).copy()
                draft=raw_pfq_draft(p,row['obs']);return solve_bounded(from_row(p,spec['stage'],row),draft)
            own_evaluate()  # Warm local imports and code/cache pages, not the fresh PHY context.
            for repeat in range(args.repeats):
                start=time.perf_counter();start_cpu=time.process_time();result=own_evaluate()
                cpu_ms=(time.process_time()-start_cpu)*1000;wall_ms=(time.perf_counter()-start)*1000
                np.testing.assert_array_equal(result[0],row['selected_action'])
                np.testing.assert_array_equal(result[1],row['selected_values'])
                own.append(dict(scene=spec['scene'],seed=spec['seed'],objective=spec['objective'],
                    stage=spec['stage'],source_stage=spec['stage'],tti=int(row['tti']),repeat=repeat,
                    cpu_ms=cpu_ms,wall_ms=wall_ms))
            own_checks.append(dict(path=str(path),sha256=sha(path),passed=True))
    own=pd.DataFrame(own);own.to_csv(output/'own_timing_raw.csv',index=False)
    own_seed=own.groupby(['scene','seed','objective','stage'],as_index=False)[['cpu_ms','wall_ms']].mean()
    own_seed.to_csv(output/'own_timing_per_seed.csv',index=False)
    own_seed.groupby(['scene','objective','stage'],as_index=False)[['cpu_ms','wall_ms']].mean().to_csv(output/'own_timing_summary.csv',index=False)
    certificate=pd.DataFrame(certificates);certificate.to_csv(output/'certificate_rounds.csv',index=False)
    root_errors=pd.DataFrame(root_errors);root_errors.to_csv(output/'common_root_sinr_errors.csv',index=False)
    root_errors.groupby(['scene','seed','objective','stage','ue_id']).agg(
        n=('sinr_error_db','size'),sinr_error_mean_db=('sinr_error_db','mean'),
        sinr_error_var_db2=('sinr_error_db',lambda x:x.var(ddof=1)),
        optimistic_fraction=('sinr_error_db',lambda x:(x>0).mean())).reset_index().to_csv(output/'common_root_sinr_per_ue.csv',index=False)
    cs=certificate.groupby(['scene','seed','objective','source_stage','tti','stage'],as_index=False)[
        ['candidates','floor_eligible','fresh_paid','certificate_required','bound_violations']].sum()
    cs['paid_to_required']=cs.fresh_paid.div(cs.certificate_required.replace(0,np.nan))
    np.testing.assert_array_equal(cs.fresh_paid,cs.certificate_required)
    assert not cs.bound_violations.any()
    cs.to_csv(output/'certificate_per_tti.csv',index=False)
    m=pd.DataFrame(mechanisms);decomposition=[]
    for (scene,objective),g in m.groupby(['scene','objective']):
        base=g[g.stage=='exact']
        def factors(x):
            return np.array([x['rounds'].mean(),x.sum_neighborhood_sizes.sum()/x['rounds'].sum(),
                             x.scored_edits.sum()/x.sum_neighborhood_sizes.sum()])
        for stage in STAGES:
            a=g[g.stage==stage];ratio=factors(a)/factors(base)
            observed=a.scored_edits.mean()/base.scored_edits.mean()
            np.testing.assert_allclose(np.prod(ratio),observed,rtol=1e-12)
            decomposition.append(dict(scene=scene,objective=objective,stage=stage,
                paid_scoring_ratio=float(observed),rounds_ratio=float(ratio[0]),
                neighborhood_per_round_ratio=float(ratio[1]),paid_fraction_ratio=float(ratio[2])))
    pd.DataFrame(decomposition).to_csv(output/'cost_decomposition.csv',index=False)
    intervals=[]
    for name,metrics in [('timing',['cpu_ms','wall_ms']),('mechanism',['scored_edits','rounds','mmse_misses',
        'first_slack_weighted_bytes','first_scored','first_paid_fraction','root_sinr_bias_db',
        'root_optimistic_fraction','first_upper_std','first_gain_std','first_prg_max_upper_std',
        'first_prg_max_gain_std','first_unique_upper_fraction'])]:
        frame=pd.read_csv(output/(name+'_per_seed.csv'))
        for (scene,objective,stage),g in frame.groupby(['scene','objective','stage']):
            if stage=='exact':continue
            for control in dict.fromkeys(('exact',STAGES[STAGES.index(stage)-1])):
                base=frame[(frame.scene==scene)&(frame.objective==objective)&(frame.stage==control)]
                paired=g.merge(base,on='seed',suffixes=('','_base'),validate='one_to_one')
                for metric in metrics:
                    delta=paired[metric]-paired[metric+'_base'];n=len(delta);mean=float(delta.mean())
                    half=float(student_t.ppf(.975,n-1)*delta.std(ddof=1)/np.sqrt(n)) if n>1 else None
                    intervals.append(dict(scene=scene,objective=objective,stage=stage,control=control,
                        metric=metric,n=n,mean_delta=mean,ci_low=mean-half if half is not None else None,
                        ci_high=mean+half if half is not None else None))
    pd.DataFrame(intervals).to_csv(output/'paired_intervals.csv',index=False)
    write(output/'validation.json',dict(passed=True,cpu=cpu,states=len(checks),stages=list(STAGES),
        repeats=args.repeats,checks=checks,own_intermediate_states=len(own_checks),own_checks=own_checks,completed=True,
        timing='serial; original campaign finished; observer/profile OFF; rotated order; fixed CPU; balanced endpoint states',
        timing_scope='state reconstruction, fixed recorded weights, PFQ decode, transform, native context/search/destruction',
        timing_exclusions='GPU PFQ, bridge, HOL, measurement emulation and transport; profiling/oracle separate',
        oracle='ALL candidates in EVERY visited round exhaustively evaluated; exact integer UB/gain/slack and required-paid-set identities',
        certificate_rounds=len(certificates),certificate_candidate_occurrences=int(certificate.candidates.sum()),
        certificate_fresh_paid=int(certificate.fresh_paid.sum()),certificate_required=int(certificate.certificate_required.sum()),
        bound_violations=int(certificate.bound_violations.sum()),
        sampling_caveat='inherited 8001/18001/28000 samples have report ages 20/20/19 TTIs; latency is a sparse near-report-boundary sample, not all-phase real-time certification',
        decomposition='multiplicative accounting of rounds x neighborhood per round x paid fraction, not independent causal effects'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--repeats',type=int,default=6);parser.add_argument('--cpu',type=int)
    main(parser.parse_args())

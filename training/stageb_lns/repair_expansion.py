"""Bounded extra training states and qualified-repair-versus-incumbent training."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time
import platform
from unittest.mock import patch
import numpy as np
from .terminal_cached import solve_cached
from .ue_pair_solver import solve_ue_pair
from .ue_pair_risk import UEPairModel,RISK_CAP
from .ue_pair_train import prepare
from .repair_pair_train import train,set_pairs,cohort
from .directed_reorder import DirectedConfig,library as directed_library
from .joint_reorder import library as delta_library
from .resource_conditions import satisfies,library as physical_library
from .resource_condition_campaign import teachers,compressed_condition,REGISTRY
from .weak_service_campaign import RecordedMemo
from .terminal_service_campaign import assert_parity
from .shortlist_trajectory import service_metrics
from .shortlist_campaign import selection_key
from .bounded_probes import csv_write,seed_mean
from .physics import ROOT,EXPECTED,DEQUEUE,ReferencePhy
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .ranked_dpls import solve_ranked
from .ranked_dpls_campaign import read_rows,identity
from .experiment import sha,write_json

NAMES=['full','cached_maxj','previous_guard','expanded_ordinary','repair_guard','repair_unguarded']
CONFIG=DirectedConfig(joint_candidates=False)
PREVIOUS=ROOT/'deliverables/ue_pair_20261103_v1'
PRIOR_ARCHIVE=ROOT/'deliverables/terminal_service_20261103_v1'


def run(row,arm,teacher,model=None):
    RecordedMemo.latest=None;start=time.perf_counter();p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs']);group=None
    if arm=='full':a,v,r=solve_ranked(p,draft)
    else:
        c=compressed_condition(p,*teacher,'joint212')
        if arm=='cached_maxj':a,v,r,group=solve_cached(p,draft,c,CONFIG,mode='maxj',offline_diagnostic=True)
        else:
            mode='maxj' if arm=='new_maxj' else ('ue_pair' if arm=='repair_unguarded' else 'ue_pair_guard')
            chosen=model.get('repair_guard') if arm=='repair_unguarded' else model.get(arm)
            a,v,r,group=solve_ue_pair(p,draft,c,CONFIG,mode=mode,model=dict(pair=chosen),offline_diagnostic=True)
    r['score']=float(v[:,EXPECTED]@p.weights);r['e2e_ms']=(time.perf_counter()-start)*1000;r.pop('history',None)
    r.update(state_id=identity(row),seed=row['seed'],source=row['family'],tti=row['tti'],arm=arm)
    memo=RecordedMemo.latest;observed=visited=None
    if memo is not None:
        observed=np.stack(list(memo.rows.values()));visited=np.frombuffer(b''.join(memo.rows),np.int64).reshape(-1,p.C,p.P).copy()
        r['query_sha256']=hashlib.sha256(b''.join(memo.events)).hexdigest();r['archive_sha256']=hashlib.sha256(b''.join(memo.rows)+observed.tobytes()).hexdigest()
    return a,v,r,group,observed,visited


def collect(rows,files,out):
    dest=out/'collection';dest.mkdir();added=[r for r in rows if r['tti'] in (4000,8000,28000)]
    assert len(added)==18 and {r['seed'] for r in added}=={41,42,43}
    write_json(dest/'added_input_manifest.json',dict(files=files,states=[identity(r) for r in added]))
    refs=teachers(added,dest,'added');work=[];checks=[];observed={};visited={}
    for row in added:
        a,v,r,g,vv,aa=run(row,'cached_maxj',refs[identity(row)]);sid=identity(row);work.append(r);observed[sid]=vv;visited[sid]=aa
        p=problem_from_raw_row(row);ind=ReferencePhy(p).evaluate(a)[0];np.testing.assert_allclose(v,ind,rtol=2e-9,atol=2e-6)
        checks.append(dict(state_id=sid,max_error=float(abs(v-ind).max())))
        print('REPAIR EXPANSION COLLECT',sid,flush=True)
    csv_write(dest/'added_decoder_work.csv',work);write_json(dest/'added_decoder_checks.json',dict(independent_phy=checks))
    mapping={'training_observed_values.npz':observed,'training_observed_actions.npz':visited,
        'training_teacher_values.npz':{sid:v for sid,(a,v) in refs.items()},'training_teacher_actions.npz':{sid:a for sid,(a,v) in refs.items()}}
    oldhash={}
    for name,new in mapping.items():
        oldhash[name]=sha(PRIOR_ARCHIVE/name)
        with np.load(PRIOR_ARCHIVE/name) as old:
            assert not (set(old.files)&set(new));merged={k:old[k].copy() for k in old.files};merged.update(new)
        np.savez_compressed(dest/name,**merged)
    write_json(dest/'training_input_manifest.json',dict(files=files,states=[identity(r) for r in rows]))
    shutil.copyfile(out/'source_manifest.json',dest/'source_manifest.json')
    write_json(dest/'provenance.json',dict(prior=str(PRIOR_ARCHIVE),prior_files=oldhash,reused_states=18,new_states=18,
        merged_archive_is_not_all_new=True,teacher_and_decoder_costs_separately_recorded=True))
    prepared=out/'prepared';prepared.mkdir();data=prepare(rows,dest,prepared)
    coverage=[]
    for d in data:
        base=int(np.argmax(d['j']));allowed=d['j']+1.e-8>=.98*d['j'][base]
        coverage.append(dict(state_id=d['state_id'],seed=d['seed'],new_state=int(d['state_id'].split('tti')[1]) in (4000,8000,28000),
            cohort=cohort(d),baseline_j_ratio=float(d['j_ratio'][base]),baseline_p5=float(d['p5'][base]),baseline_weak=float(d['weak'][base]),
            allowed_joint_candidates=int((d['good']&allowed).sum())))
    per_seed={str(seed):sum(r['cohort']=='repairable' and r['seed']==seed for r in coverage) for seed in (41,42,43)}
    gate=all(v>=1 for v in per_seed.values()) and sum(per_seed.values())>=3
    csv_write(out/'training_state_coverage.csv',coverage)
    write_json(out/'coverage_gate.json',dict(passed=gate,required='at least1 independent repairable failure per trainingseed,and>=3 total',
        unique_states=len(data),per_seed_repairable=per_seed,new_repairable_states=sum(r['new_state'] and r['cohort']=='repairable' for r in coverage),
        training_role='controlled diagnostic continuation only' if not gate else 'expanded repair-training experiment',
        frozen_before_validation=True,no_new_native_trajectory=True))
    return data


def engineering(rows,out,models):
    ta=np.load(out/'collection/training_teacher_actions.npz');tv=np.load(out/'collection/training_teacher_values.npz');work=[];checks=[]
    for row in (rows[0],rows[-1]):
        sid=identity(row);ref=(ta[sid],tv[sid]);p=problem_from_raw_row(row);results={}
        for arm in ['cached_maxj','new_maxj',*NAMES[2:]]:
            result=run(row,arm,ref,models);results[arm]=result;a,v,r,*_=result;work.append(r)
            ind=ReferencePhy(p).evaluate(a)[0];np.testing.assert_allclose(v,ind,rtol=2e-9,atol=2e-6);checks.append(dict(state_id=sid,arm=arm,max_error=float(abs(v-ind).max())))
        for arm in ['new_maxj',*NAMES[2:]]:assert_parity(results['cached_maxj'],results[arm],final=arm=='new_maxj')
    csv_write(out/'engineering_measurements.csv',work);write_json(out/'engineering_checks.json',dict(independent_phy=checks,query_archive_equivalent=True))


def benchmark(rows,refs,out,phase,model):
    raw=[];saved={k:{} for k in ('actions','values','service_groups','observed_values','observed_actions')}
    for n,row in enumerate(rows):
        state_results={}
        for repeat in range(3):
            shift=(n+repeat)%len(NAMES)
            for arm in NAMES[shift:]+NAMES[:shift]:
                result=run(row,arm,refs[identity(row)],model);a,v,r,g,vv,aa=result;r['repeat']=repeat;raw.append(r);key=f'{identity(row)}|{arm}'
                for name,value in zip(saved,(a,v,g,vv,aa)):
                    if value is None:continue
                    if key in saved[name]:np.testing.assert_array_equal(saved[name][key],value)
                    else:saved[name][key]=value
                if repeat==0:state_results[arm]=result
        for arm in NAMES[2:]:assert_parity(state_results['cached_maxj'],state_results[arm],final=False)
        print('REPAIR EXPANSION BENCH',phase,identity(row),flush=True)
    state=[];checks=[];parity=[];previous=np.load(PREVIOUS/f'{phase}_actions.npz');pv=np.load(PREVIOUS/f'{phase}_values.npz')
    for row in rows:
        p=problem_from_raw_row(row);sid=identity(row);ref=saved['values'][f'{sid}|full'];jref=float(ref[:,EXPECTED]@p.weights)
        full_ms=float(np.median([r['e2e_ms'] for r in raw if r['state_id']==sid and r['arm']=='full']))
        np.testing.assert_array_equal(saved['actions'][f'{sid}|full'],refs[sid][0])
        np.testing.assert_array_equal(saved['actions'][f'{sid}|cached_maxj'],previous[f'{sid}|cached_maxj'])
        np.testing.assert_array_equal(saved['values'][f'{sid}|cached_maxj'],pv[f'{sid}|cached_maxj'])
        parity.append(dict(state_id=sid,previous_baseline_equal=True,cached_maxj_previous_final_equal=True,all_terminal_arms_query_archive_equal=True))
        for arm in NAMES:
            key=f'{sid}|{arm}';v=saved['values'][key];a=saved['actions'][key];ind=ReferencePhy(p).evaluate(a)[0]
            np.testing.assert_allclose(v,ind,rtol=2e-9,atol=2e-6)
            rr=[r for r in raw if r['state_id']==sid and r['arm']==arm]
            r={k:v for k,v in rr[0].items() if k!='repeat'};r['e2e_ms']=float(np.median([v['e2e_ms'] for v in rr]))
            r.update(ratio=r['score']/jref,reference_ms=full_ms,execution_time_ratio=r['e2e_ms']/full_ms,**service_metrics(v,ref))
            assert v[:,DEQUEUE].sum()>=r['floor']
            if arm!='full':
                assert r['candidate_evaluations']==212 and r['feature_roots']<=56 and r['condition_feasible'] and not r['fallback_to_raw']
                assert satisfies(p,a,compressed_condition(p,*refs[sid],'joint212'))[0] and r['mmse_misses']==2448
                assert r['selected_j_loss_fraction']<=RISK_CAP+1.e-12
                assert len({z['query_sha256'] for z in rr})==len({z['archive_sha256'] for z in rr})==1
            state.append(r);checks.append(dict(state_id=sid,arm=arm,max_error=float(abs(v-ind).max())))
    csv_write(out/f'{phase}_measurements.csv',raw);csv_write(out/f'{phase}_per_state.csv',state)
    csv_write(out/f'{phase}_summary.csv',summarize(state));csv_write(out/f'{phase}_equivalence.csv',parity)
    for name,data in saved.items():np.savez_compressed(out/f'{phase}_{name}.npz',**data)
    write_json(out/f'{phase}_checks.json',dict(independent_phy=checks,repeats_and_equivalence=True,budget_floor_conditions=True))
    return state


def freeze(out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(date='2026-11-03',version=1,methods=NAMES,search_config=asdict(CONFIG),
        inventory='sixtrainingfiles seeds41-43 each only TTIs4000,4001,8000,8001,18000,28000;allremaining18source states collected,none selected using results',
        old_training_ttis=[4001,8001,18000],added_training_ttis=[4000,8000,28000],max_new_source_states=18,
        coverage_gate='>=1 independent repairable maximumJfailure per41/42/43 and>=3total; if insufficient still one requested controlled diagnostic training round, no further expansion',
        training='36source states merged,immutable input/archive/reference hashdedup;only41-43labels;ordinaryexpanded versus focusedexpanded controls,oldmodel frozen',
        focused_pairs='everyqualifiedrepair vsfailedmaxJ plusatmost7otherhighestJbadcandidates;baselinegood:maximumJvsbadpreferred andmaximumJvsothergoodties;unrepairable:no pair supervision',
        balancing='repairable andbaselinegood cohorts half pair mass each whenpresent,states equal withincohort;unrepairable auxiliaryonly;auxiliary half mass to referencepositive underprotected UEs whenpresent,half others,states equal',
        architecture='unchanged20-24tanh-10 sharedperUE monotone lossmodel;240epochs Adam.01 decay.001;3leave-seed-out calibration fits and3initialization final fits pervariant;2880total gradientsteps',
        confidence='unchanged2%ownJwindow;minimum3-member pairwin probability>=.9 andpositivegain versusmaxJ,elseexactmaxJ;unguardeddiagnostic notselectable',
        conditions='offlinepaid teacher counts/exact min-layer/activity;no teacher positions/J/service/group at runtime;noCTBS/PPO/headroom',
        budget='212exactcompleteactions/56roots/16seeds/8batch;cached existingpool2slots quotaexpfrontier unchanged',
        objective='canonical native BLER causaldeficitweightedJ,ownrawPFQ90%dequeuefloor',
        validation_seeds=[44,45],evaluation_seeds=[46,47,48],train_validation_ttis=[4001,8001,18000],evaluation_ttis=[3001,4001,5000],repeats=3,
        gates='eachJ>=.995full;newunallocated0;fullpositiveP5/fixedweak20>=.98;seed-equalcompletetime<=.8full',
        historical_development=True,final_seeds_not_read=[311,312,313,314,315],native_trajectories=0,
        timing='input/PFQ,freshPHY/MMSE,conditioncompression,allsearch/cache/terminalfeatures/ensemble/calibration/fallback/cleanup;eventcopiesincluded',
        exclusions='teacherprep,training/modelIO,initiallibrary/TBS,archiveIO,digests/independentchecks',
        registry_sha256=sha(REGISTRY),physics_source_sha256=sha(ROOT/'training/stageb_lns/physics_kernel.cpp'),
        old_model_sha256=sha(PREVIOUS/'ue_pair_model.npz'),platform=platform.platform(),
        threads=dict(OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1))
    write_json(out/'protocol.json',protocol);manifest={}
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.cpp','.so','.rst'):
            target=out/'source'/p.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target);manifest[p.name]=sha(p)
    write_json(out/'source_manifest.json',manifest);return protocol


def main(output):
    out=Path(output).resolve();protocol=freeze(out);start=time.perf_counter();rows,files=read_rows((41,42,43));assert len(rows)==36
    write_json(out/'training_input_manifest.json',dict(files=files,states=[identity(r) for r in rows]))
    problem_from_raw_row(rows[0]);physical_library();delta_library();directed_library()
    with patch('training.stageb_lns.terminal_cached.ExactMemo',RecordedMemo),patch('training.stageb_lns.ue_pair_solver.ExactMemo',RecordedMemo):
        data=collect(rows,files,out)
        models={};paths={}
        for name,focused in [('expanded_ordinary',False),('repair_guard',True)]:
            folder=out/name;folder.mkdir();dataset=set_pairs(data,focused,folder);train(dataset,folder,focused)
            paths[name]=folder/'ue_pair_model.npz';models[name]=UEPairModel.load(paths[name])
        shutil.copyfile(PREVIOUS/'ue_pair_model.npz',out/'previous_model.npz');paths['previous_guard']=out/'previous_model.npz';models['previous_guard']=UEPairModel.load(paths['previous_guard'])
        hashes={str(path.relative_to(out)):sha(path) for path in paths.values()};write_json(out/'models_frozen.json',dict(hashes=hashes,validation_not_read=True,development_not_read=True))
        engineering(rows,out,models)
        for phase,seeds,evaluation in [('validation',(44,45),False),('evaluation',(46,47,48),True)]:
            rr,files=read_rows(seeds,evaluation=evaluation);ttis=protocol['evaluation_ttis'] if evaluation else protocol['train_validation_ttis'];rr=[r for r in rr if r['tti'] in ttis]
            write_json(out/f'{phase}_input_manifest.json',dict(files=files,states=[identity(r) for r in rr]));refs=teachers(rr,out,phase);state=benchmark(rr,refs,out,phase,models)
            if not evaluation:
                selection=choose(state);write_json(out/'selection.json',selection);print('REPAIR EXPANSION SELECTION FROZEN',selection,flush=True)
    assert sha(REGISTRY)==protocol['registry_sha256'] and sha(ROOT/'training/stageb_lns/physics_kernel.cpp')==protocol['physics_source_sha256']
    assert all(sha(out/n)==h for n,h in hashes.items());assert all(sha(Path(__file__).with_name(n))==h for n,h in json.loads((out/'source_manifest.json').read_text()).items())
    write_json(out/'completion.json',dict(completed=True,formal_decisions=378,formal_independent_finals=126,new_teacher_decisions=39,
        new_collection_decoders=18,engineering_decoders=12,independent_phy_finals=195,source_model_registry_unchanged=True,
        training_updates=2880,final_seeds_read=False,native_trajectories=0,elapsed_s=time.perf_counter()-start))
    print('REPAIR EXPANSION COMPLETE',out,flush=True)


def summarize(state):
    full=[r for r in state if r['arm']=='full'];out=[]
    for arm in NAMES:
        rr=[r for r in state if r['arm']==arm]
        q=all(r['ratio']>=.995 for r in rr)
        c=all(r['newly_unserved_ues']==0 and r['positive_p5_ratio']>=.98 and r['positive_weak_ratio']>=.98 for r in rr)
        cost=seed_mean(rr,'e2e_ms')/seed_mean(full,'e2e_ms')
        out.append(dict(arm=arm,mean_j=seed_mean(rr,'score'),mean_ms=seed_mean(rr,'e2e_ms'),execution_time_ratio=cost,
            mean_relative_j_pct=100*(seed_mean(rr,'ratio')-1),worst_relative_j_pct=100*(min(r['ratio'] for r in rr)-1),
            mean_newly_unserved=seed_mean(rr,'newly_unserved_ues'),mean_positive_p5=seed_mean(rr,'positive_p5_ratio'),
            worst_positive_p5=min(r['positive_p5_ratio'] for r in rr),mean_positive_weak=seed_mean(rr,'positive_weak_ratio'),
            worst_positive_weak=min(r['positive_weak_ratio'] for r in rr),mean_actual_phy=seed_mean(rr,'candidate_evaluations'),
            mean_feature_roots=seed_mean(rr,'feature_roots') if arm!='full' else None,
            quality_failures=sum(r['ratio']<.995 for r in rr),coverage_failures=sum(not(r['newly_unserved_ues']==0 and r['positive_p5_ratio']>=.98 and r['positive_weak_ratio']>=.98) for r in rr),
            quality_pass=q,coverage_pass=c,execution_cost_pass=cost<=.8,conditional_target_pass=q and c and cost<=.8,
            online_ai_success=False))
    return out


def choose(state):
    choices=[]
    for arm in ['cached_maxj','previous_guard','expanded_ordinary','repair_guard']:
        rr=[r for r in state if r['arm']==arm];key=selection_key(rr)
        q=all(r['ratio']>=.995 for r in rr)
        c=all(r['newly_unserved_ues']==0 and r['positive_p5_ratio']>=.98 and r['positive_weak_ratio']>=.98 for r in rr)
        cost=seed_mean(rr,'e2e_ms')/seed_mean(rr,'reference_ms')
        choices.append(((int(q and c),*key[1:]),arm,cost,seed_mean(rr,'e2e_ms')))
    best=max(choices,key=lambda v:(v[0],-v[3]));feasible=[v for v in choices if v[2]<=.8]
    within=max(feasible,key=lambda v:(v[0],-v[3])) if feasible else min(choices,key=lambda v:v[3])
    return dict(quality_candidate=best[1],execution_cost_candidate=within[1],execution_cost_feasible=bool(feasible),
        validation_only=True,development_not_read=True,historical_exposure=True,teacher_preparation_excluded=True,
        rule='same prior rule: all-state quality/service then mean quality shortfall, service shortfall, mean ratio; execution cost filter, lower time on ties',online_ai_claim=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)

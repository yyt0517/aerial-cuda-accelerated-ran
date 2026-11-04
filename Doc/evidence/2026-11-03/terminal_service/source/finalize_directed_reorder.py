"""Recover the completed experiment's JSON archive, without new native scores.

The original diagnostic finished all native labels and independent checks, then
failed while serializing one NumPy index. Preserve the partial file and exact
executed snapshot; recheck saved values and write the missing metadata only.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .physics import ROOT,ReferencePhy
from .no_ctbs_inputs import problem_from_raw_row
from .ranked_dpls_campaign import read_rows,identity
from .resource_condition_campaign import REGISTRY
from .experiment import sha,write_json


def main(source):
    source=Path(source).resolve();out=source/'recall_diagnosis'
    log=Path('/tmp/directed_reorder_execution_20261102.log').read_text()
    assert 'TypeError: Object of type int64 is not JSON serializable' in log
    work=list(csv.DictReader((out/'work.csv').open()));candidates=list(csv.DictReader((out/'candidates.csv').open()))
    actions=np.load(out/'actions.npz');values=np.load(out/'values.npz');checks=[]
    for phase,seeds,ttis in [('validation',(44,45),(4001,8001,18000)),('evaluation',(46,47,48),(3001,4001,5000))]:
        rows,_=read_rows(seeds,evaluation=phase=='evaluation');rows={identity(r):r for r in rows if r['tti'] in ttis}
        roots=np.load(source/f'{phase}_actions.npz');root_values=np.load(source/f'{phase}_values.npz')
        for w in [w for w in work if w['phase']==phase]:
            sid=w['state_id'];arm=w['arm'];key=f'{sid}|{arm}';label=f'{phase}|{key}';p=problem_from_raw_row(rows[sid])
            rr=[r for r in candidates if (r['phase'],r['state_id'],r['arm'])==(phase,sid,arm)]
            assert len(rr)==int(w['unseen_exact_candidates'])==len(actions[label])==len(values[label])
            assert int(w['candidate_evaluations'])==1+len(rr)
            ids=[]
            if rr:
                ids.extend([int(max(rr,key=lambda r:float(r['exact_j']))['index']),int(max(rr,key=lambda r:float(r['p5_ratio']))['index'])])
                novel=[r for r in rr if r['novel_directed']=='True'];passed=[r for r in rr if r['joint_pass']=='True']
                if novel:ids.append(int(max(novel,key=lambda r:float(r['exact_j']))['index']))
                if passed:ids.append(int(passed[0]['index']))
                ids.extend(int(r['index']) for r in sorted(rr,key=lambda r:int(r['proxy_rank']))[:2])
            for i in [-1,*dict.fromkeys(ids)]:
                a=roots[key] if i<0 else actions[label][i];v=root_values[key] if i<0 else values[label][i]
                independent=ReferencePhy(p).evaluate(a)[0];np.testing.assert_allclose(v,independent,rtol=2e-9,atol=2e-6)
                checks.append(dict(phase=phase,state_id=sid,arm=arm,index=int(i),max_error=float(abs(v-independent).max())))
    partial=out/'checks.json';assert partial.exists();partial.rename(out/'checks.partial.json')
    write_json(out/'checks.json',dict(passed=True,independent_phy=checks,recovery_repeated_original_independent_checks=True))
    selected=json.loads((source/'selection.json').read_text())
    diagnostic=dict(post_selection=True,policy_changed=False,training=False,selected_quality=selected['quality_candidate'],roots=len(work),max_roots=8,
        exact_scores=sum(int(w['candidate_evaluations']) for w in work),unique_mmse=sum(int(w['mmse_misses']) for w in work),
        independent_audits=len(checks),recovery_additional_independent_audits=len(checks),all_finite_pools_complete=True,
        rule='up to2 failing states per phase by combined quality/service shortfall; baseline and validation-selected quality method',
        scores_separate_from_formal212=True,teacher_preparation_reused_from_paid_campaign=True,
        limits='finite unseen one-step pool only; iteration0 synthetic shortlist,not an actual policy continuation; teacher-positive labels offline only')
    write_json(out/'summary.json',diagnostic)
    manifest=json.loads((source/'source_manifest.json').read_text());name='directed_reorder_diagnosis.py'
    old=source/'source'/name;new=ROOT/'training/stageb_lns'/name
    before='index=i,max_error=';after='index=int(i),max_error='
    assert old.read_text().count(before)==1 and old.read_text().replace(before,after)==new.read_text()
    for n,d in manifest.items():
        assert sha(source/'source'/n)==d
        if n!=name:assert sha(ROOT/'training/stageb_lns'/n)==d
    protocol=json.loads((source/'protocol.json').read_text());assert sha(REGISTRY)==protocol['registry_sha256']
    assert sha(ROOT/'training/stageb_lns/physics_kernel.cpp')==protocol['physics_source_sha256']
    patch=dict(reason='NumPy index failed JSON serialization after all native scores and independent checks completed',
        file=name,executed_sha256=sha(old),fixed_sha256=sha(new),only_change='index=i -> index=int(i) in diagnostic audit record',
        policy_or_native_kernel_changed=False,original_partial_preserved='recall_diagnosis/checks.partial.json',
        additional_native_scores=0,additional_independent_reference_checks=len(checks))
    write_json(source/'serialization_recovery.json',patch)
    for phase,count in [('validation',12),('evaluation',9)]:
        rr=list(csv.DictReader((source/f'{phase}_measurements.csv').open()));assert len(rr)==count*6*3
        c=json.loads((source/f'{phase}_checks.json').read_text());assert len(c['full_phy'])==count*6
    write_json(source/'completion.json',dict(completed=True,completed_by_archive_recovery=True,formal_decisions=378,formal_independent_finals=126,
        offline_teacher_decisions=23,engineering_decoder_decisions=12,validation_states=12,development_states=9,
        diagnostic_roots=len(work),diagnostic_exact_scores=diagnostic['exact_scores'],diagnostic_independent_audits=len(checks),
        recovery_additional_independent_audits=len(checks),training_updates=0,registry_and_physics_unchanged=True,
        executed_sources_snapshot_valid=True,diagnostic_serialization_only_patch=True,native_trajectories=0,final_seeds_read=False,
        formal_elapsed_s=None,elapsed_s=None,wall_time_note='final process wall timers were not persisted before the JSON error; per-decision timings are intact'))
    print(json.dumps(dict(recovery=patch,diagnostics=diagnostic),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);main(p.parse_args().source)

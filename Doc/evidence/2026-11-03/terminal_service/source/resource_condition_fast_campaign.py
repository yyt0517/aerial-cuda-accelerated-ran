"""Re-measure identical frozen decisions with equivalent faster enumeration."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .resource_condition_campaign import (REGISTRY,NAMES,freeze,engineering,teachers,benchmark,choose)
from .resource_conditions import library
from .resource_conditions_fast import pool_library,solve_conditional
from .bounded_probes import probe_library
from .selective_tb import feature_library
from .ranked_dpls_campaign import read_rows,identity
from .no_ctbs_inputs import problem_from_raw_row
from .physics import ROOT
from .experiment import sha,write_json

PREVIOUS=ROOT/'deliverables/resource_conditions_20261102_v1'


def audit_parity(out,phase):
    checked=[]
    for suffix in ('actions','values'):
        previous=np.load(PREVIOUS/f'{phase}_{suffix}.npz');current=np.load(out/f'{phase}_{suffix}.npz')
        assert previous.files==current.files
        for key in current.files:
            np.testing.assert_array_equal(current[key],previous[key]);checked.append(f'{key}|{suffix}')
    write_json(out/f'{phase}_v1_parity.json',dict(passed=True,all_final_actions_and_values_identical=True,
        checked=checked,prior_actions_sha256=sha(PREVIOUS/f'{phase}_actions.npz'),prior_values_sha256=sha(PREVIOUS/f'{phase}_values.npz')))


def main(output):
    out=Path(output).resolve();protocol=freeze(out);start=time.perf_counter()
    write_json(out/'backend_protocol.json',dict(version=2,prior_output=str(PREVIOUS),
        prior_protocol_sha256=sha(PREVIOUS/'protocol.json'),prior_source_manifest_sha256=sha(PREVIOUS/'source_manifest.json'),
        changes=['compiled identical ordered local pool','first-occurrence byte dedup instead of equivalent row-wise np.unique'],
        unchanged=['conditions','budgets','proxy','beam','seed','acceptance','matching/global reseeds'],
        parity='all validation/development final actions AND values must match V1 exactly',
        historical_exposure='V1 outcomes were read; this is engineering remeasurement, not independent quality confirmation'))
    training,files=read_rows((41,42,43));training=[r for r in training if r['tti'] in protocol['validation_ttis']]
    erows=[training[0],training[-1]];write_json(out/'engineering_input_manifest.json',dict(files=files,states=[identity(r) for r in erows]))
    problem_from_raw_row(erows[0]);library();pool_library();probe_library();feature_library()
    engineering(erows,out,solver=solve_conditional)
    validation,files=read_rows((44,45));validation=[r for r in validation if r['tti'] in protocol['validation_ttis']]
    write_json(out/'validation_input_manifest.json',dict(files=files,states=[identity(r) for r in validation]))
    reference=teachers(validation,out,'validation');state=benchmark(validation,reference,out,'validation',solver=solve_conditional)
    audit_parity(out,'validation');selection=choose(state);write_json(out/'selection.json',selection)
    print('RESOURCE FAST SELECTION FROZEN',selection,flush=True)
    evaluation,files=read_rows((46,47,48),evaluation=True);evaluation=[r for r in evaluation if r['tti'] in protocol['development_ttis']]
    write_json(out/'evaluation_input_manifest.json',dict(files=files,states=[identity(r) for r in evaluation]))
    reference=teachers(evaluation,out,'evaluation');benchmark(evaluation,reference,out,'evaluation',solver=solve_conditional)
    audit_parity(out,'evaluation')
    assert sha(REGISTRY)==protocol['registry_sha256']
    for name,digest in json.loads((out/'source_manifest.json').read_text()).items():assert sha(Path(__file__).with_name(name))==digest
    write_json(out/'completion.json',dict(completed=True,formal_decisions=(len(validation)+len(evaluation))*len(NAMES)*3,
        formal_independent_finals=(len(validation)+len(evaluation))*len(NAMES),offline_teacher_decisions=23,
        engineering_decoder_decisions=8,validation_states=len(validation),development_states=len(evaluation),
        training_updates=0,all_v1_final_actions_values_identical=True,registry_and_physics_unchanged=True,
        executed_sources_unchanged=True,no_teacher_UE_positions_in_solver=True,native_trajectories=0,
        final_seeds_read=False,elapsed_s=time.perf_counter()-start))
    print('RESOURCE FAST COMPLETE',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);main(parser.parse_args().output)

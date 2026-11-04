"""User-authorized B scan, independent of the unchanged deployment wall gate."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing as mp
from pathlib import Path
import shutil
from .physics import ROOT
from .bounded_native import one,identifier,pin_worker
from .ttl_weights import COEFFICIENTS
from .analyze_dpls_moves import write_json,sha

def main(args):
    gate=Path(args.gate).resolve();stress=Path(args.stress).resolve();out=Path(args.output).resolve()
    engineering=json.loads((gate/'completion.json').read_text())
    costs=json.loads((stress/'completion.json').read_text())
    if not engineering['equivalence_passed'] or not costs['equivalence_passed']:
        raise ValueError('B still requires exact-engine correctness')
    oldplan=ROOT/'deliverables/bounded_B_plan_20261104_v1/protocol.json'
    frozen=json.loads(oldplan.read_text())
    assert frozen['coefficients']==list(COEFFICIENTS) and frozen['seeds']==[41,42,43]
    out.mkdir(parents=True,exist_ok=False)
    specs=[(seed,.8,k) for k in COEFFICIENTS for seed in (41,42,43) if k!=0 or seed==43]
    reuse=[];inputs={str(oldplan):sha(oldplan),str(gate/'completion.json'):sha(gate/'completion.json'),
        str(stress/'completion.json'):sha(stress/'completion.json')}
    stress_source=Path(args.stress_source).resolve()
    for seed in (41,42):
        d=stress_source/identifier(seed,.8,0.);s=json.loads((d/'summary.json').read_text())
        if not s['native_passed'] or s['coefficient']!=0 or s['rate']!=.8:raise ValueError('Invalid fixed-weight control')
        reuse.append(dict(directory=str(d),**s))
        for n in ('summary.json','actions.npz','cohort.csv.gz','validation.json','launch.json'):
            inputs[str(d/n)]=sha(d/n)
    names=('ttl_decoupled_campaign.py','bounded_native.py','rate_native.py','ttl_weights.py','bounded_dpls.py',
        'bounded_kernel.cpp','_bounded.so','persistent_kernel.cpp','physics_kernel.cpp','no_ctbs_inputs.py')
    files=[Path(__file__).with_name(n) for n in names];hashes={str(f.relative_to(ROOT)):sha(f) for f in files}
    (out/'source').mkdir()
    for f in files:shutil.copy2(f,out/'source'/f.name)
    write_json(out/'source_manifest.json',hashes);write_json(out/'input_manifest.json',inputs)
    write_json(out/'protocol.json',dict(version='B_causal_HOL_scan_v1_wall_decoupled',mode='ttl',
        authorization='user explicitly decoupled B scientific comparison from40% deployment wall gate',
        original_plan_sha256=sha(oldplan),changed='entry only; coefficients/formula/solver/seeds/window/metrics fixed',
        coefficients=list(COEFFICIENTS),seeds=[41,42,43],jobs=[dict(seed=s,rate=r,coefficient=k) for s,r,k in specs],
        warmup=4000,settling=4000,measured=4000,followup=400,baseline_rate=.8,
        weights=frozen['formula'],selection=frozen['no_postselection'],B_rule=frozen['B_rule'],
        aggregation=frozen['aggregation'],reused_zero_coefficient=reuse,
        solver='same frozen capacity_bound_q1e6_v1 ELF for every scientific arm',
        deployment_gate_unchanged=True,previous_stress_wall_passed=bool(costs['B_entry_passed']),
        workers=args.workers,cpu_pins=list(range(args.cpu,args.cpu+args.workers)),
        timing='parallel quality trajectories; no fair speed claim from policy wall times',
        role='exposed training/development frontier, parameter scan only; no gradient/ranking fitting or final confirmation'))
    ctx=mp.get_context('spawn');queue=ctx.Queue()
    for cpu in range(args.cpu,args.cpu+args.workers):queue.put(cpu)
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=ctx,initializer=pin_worker,initargs=(queue,)) as pool:
        futures=[pool.submit(one,str(out),s,r,k,4000,4000,4000,400,1000) for s,r,k in specs]
        for f in as_completed(futures):results.append(f.result())
    for f,digest in hashes.items():assert sha(ROOT/f)==digest
    for f,digest in inputs.items():assert sha(f)==digest
    write_json(out/'completion.json',dict(completed=True,mode='ttl',runs=results,reused_runs=reuse,
        new_native_runs=len(results),new_native_steps=len(results)*12400,training_steps=0,
        wall_gate_decoupled_by_user=True,deployment_wall_result_unchanged=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--gate',default='deliverables/bounded_gate_wall_20261104_v1')
    p.add_argument('--stress',default='deliverables/bounded_load_stress_analysis_20261104_v1')
    p.add_argument('--stress-source',default='deliverables/bounded_load_stress_20261104_v1')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--cpu',type=int,default=252);main(p.parse_args())

"""One final full-Gamma SIMD gate; numerical failures never enter scientific arms."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from .physics import ROOT
from .archive import enable_legacy_types
from .no_ctbs_inputs import problem_from_raw_row
from .analyze_dpls_moves import write_json,write_csv,sha

def library():
    lib=ctypes.CDLL(str(Path(__file__).with_name('_gamma_soa.so')))
    i=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    d=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    for name in ('sg_table','sg_reference'):
        f=getattr(lib,name);f.argtypes=[ctypes.c_int]*3+[ctypes.c_double,i,d,d,d];f.restype=ctypes.c_int
    return lib

def inputs(p):
    cov=np.ascontiguousarray(p.h@p.h.conj().swapaxes(-1,-2))
    sig=np.ascontiguousarray(p.h[np.arange(p.U),:,p.serving]@p.w)
    return p.U,p.C,p.P,p.noise,p.serving,cov.view(np.float64),sig.view(np.float64)

def main(args):
    import torch
    enable_legacy_types();os.sched_setaffinity(0,{args.cpu})
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
    files=[Path(__file__).with_name(n) for n in ('gamma_soa.cpp','gamma_bridge.cpp','build_gamma_soa.py','_gamma_soa.so','audit_gamma_soa.py')]
    for p in files:shutil.copy2(p,out/'source'/p.name)
    sources={str(p.relative_to(ROOT)):sha(p) for p in files};write_json(out/'source_manifest.json',sources)
    prior=ROOT/'deliverables/tournament_gate_wall_20261104_v2/states.json'
    states=json.loads(prior.read_text());lib=library();rows=[];work=0
    write_json(out/'protocol.json',dict(single_final_attempt=True,keys='all4 interferer masks per UE/PRG, including unused keys',
        implementation='SoA real/imag arrays, AVX512 eight independent keys per vector, lane-specific LDLT pivots, triangular solve, adjoint product, cofactor inverse',
        arithmetic='same pivoted LDLT/cofactor identities; match scalar expression order where explicit; original Eigen packet/reduction order still requires bitwise gate',
        flags='-O3 -ffp-contract=off -fno-fast-math; SIMD TU avx512f/dq; original Eigen reference TU original architecture',
        acceptance='all Gamma bit patterns must equal frozen-form Eigen reference before solver integration; no tolerance substitution',
        fallback='if numerical gate or wall fails, stop40% engineering effort; retain34/41/46% existing exact curve',
        scientific='new library only, never enters weight grid or literature/learning arms',
        timing='gate only while native quality runs; no fair end-to-end speed claim from concurrent microtiming'))
    for idx,item in enumerate(states):
        path=Path(item['path']);assert sha(path)==item['sha256']
        row=torch.load(path,map_location='cpu',weights_only=False);p=problem_from_raw_row(row);a=inputs(p)
        ref=np.empty((p.U,p.P,4,4));got=np.empty_like(ref)
        assert lib.sg_reference(*a,ref)==0 and lib.sg_table(*a,got)==0
        assert np.isfinite(ref).all() and np.isfinite(got).all()
        equal=ref.view(np.uint64)==got.view(np.uint64);mismatch=np.argwhere(~equal)
        absolute=np.abs(ref-got);relative=absolute/np.maximum(np.abs(ref),1e-300)
        rows.append(dict(state=idx,rate=item['rate'],seed=item['seed'],keys=p.U*p.P*4,scalars=ref.size,
            bitwise_equal=int(equal.sum()),mismatch_scalars=int((~equal).sum()),max_abs=float(absolute.max()),
            max_rel=float(relative.max()),first_mismatch=mismatch[0].tolist() if len(mismatch) else []))
        work+=2*p.U*p.P*4
        if idx==0:np.savez_compressed(out/'first_state.npz',reference=ref,SoA_SIMD=got)
        if idx%40==0:print('GAMMA_SOA_GATE',idx,'mismatches',int((~equal).sum()),flush=True)
    write_csv(out/'gamma_equivalence.csv',rows);passed=all(r['mismatch_scalars']==0 for r in rows)
    write_json(out/'input_manifest.json',{i['path']:i['sha256'] for i in states})
    # Keep generated instructions as evidence that the key axis is vectorized.
    import subprocess
    disasm=subprocess.check_output(['objdump','-d',str(Path(__file__).with_name('_gamma_soa.so'))],text=True)
    instructions=[line for line in disasm.splitlines() if 'zmm' in line]
    (out/'SIMD_instruction_sample.txt').write_text('\n'.join(instructions[:50])+'\n')
    write_json(out/'completion.json',dict(completed=True,bitwise_passed=passed,states=len(rows),
        mismatch_states=sum(r['mismatch_scalars']>0 for r in rows),mismatch_scalars=sum(r['mismatch_scalars'] for r in rows),
        checked_scalars=sum(r['scalars'] for r in rows),max_abs=max(r['max_abs'] for r in rows),
        max_rel=max(r['max_rel'] for r in rows),new_MMSE_problems=work,
        AVX512_instruction_lines=len(instructions),FMA_instructions=sum('vfm' in line for line in disasm.splitlines()),
        integrated_into_solver=False,engineering_stopped=not passed,new_native_runs=0))
    for p,digest in sources.items():assert sha(ROOT/p)==digest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--cpu',type=int,default=243);main(p.parse_args())

"""Freeze local outputs and compact evidence; preserve old experiment files."""
from pathlib import Path
import json,hashlib,shutil

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
    with p.open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def main():
    checks=json.loads((HERE/'checks.json').read_text())
    assert checks['passed'] and not checks['scientific_gate_passed']
    dest=ROOT/'Doc/evidence/2026-10-29/tb_residual';dest.mkdir(parents=True,exist_ok=False)
    selected=['report.txt','progress.txt','plan.json','replay_plan.json','optimization_plan.json','checks.json',
        'results.png','results.pdf','paired_vs_verified.csv',
        'data/checks.json','data/protocol.json','data/groups.csv','data/states.csv','data/roots.csv',
        'residual_training/protocol.json','residual_training/selected.json','residual_training/history.csv',
        'proposer_training/protocol.json','proposer_training/selected.json','proposer_training/history.csv',
        'evaluation/checks.json','evaluation/curves_per_seed.csv','evaluation/curves_macro.csv',
        'evaluation/proposal_per_seed.csv','evaluation/per_ue_mae.csv','evaluation/calibration.json',
        'evaluation/empirical_threshold_seeds.csv',
        'replay/protocol.json','replay/checks.json','replay/means.csv','replay/per_seed.csv','replay/paired_states.csv',
        'benchmark/protocol.json','benchmark/checks.json','benchmark/per_seed.csv','benchmark/setups.csv','benchmark/full_recompute.cpp',
        'optimized/protocol.json','optimized/checks.json','optimized/means.csv','optimized/per_seed.csv','optimized/paired_states.csv',
        'diagnostics/checks.json','diagnostics/runs.csv','diagnostics/self_search_summary.csv',
        'profile/trace_original.txt','profile/residual_original.txt']
    selected += [p.name for p in sorted(HERE.glob('*.py'))]
    for name in selected:
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(HERE/name,target)
    all_files={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'}
    write(HERE/'manifest.json',dict(files=all_files,scientific_gate_passed=False,algorithm_promoted=False))
    write(dest/'source_manifest.json',dict(source_root=str(HERE.relative_to(ROOT)),source_manifest_sha256=sha(HERE/'manifest.json'),
        copied={name:sha(HERE/name) for name in selected},
        local_large_artifacts={name:all_files[name] for name in
            ('data/candidates.npz','data/pairs.npz','actor_original.npz','actor_frozen.npz','residual_frozen.npz')}))
    files={str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}
    write(dest/'manifest.json',dict(files=files))
    for name,digest in all_files.items():assert sha(HERE/name)==digest,name
    for name,digest in files.items():assert sha(dest/name)==digest,name
    print(f'FINALIZED local={len(all_files)} compact={len(files)}; no promotion',flush=True)

if __name__=='__main__':main()

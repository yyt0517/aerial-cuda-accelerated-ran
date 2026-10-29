"""Archive compact evidence and hashes without copying large local datasets."""
from pathlib import Path
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False);f.write('\n')


def main():
    checks = json.loads((HERE/'checks.json').read_text())
    assert checks['passed'] and not checks['scientific_gate_passed']
    evidence = ROOT/'Doc/evidence/2026-10-29/tb_process_value'
    evidence.mkdir(parents=True, exist_ok=False)
    selected = ['report.txt', 'research.txt', 'plan.json', 'checks.json', 'progress.txt',
        'recall_cost_controls.png', 'recall_cost_controls.pdf',
        'data/checks.json', 'data/frozen_plan.json', 'data/groups.csv', 'data/states.csv',
        'training/protocol.json', 'training/selected.json', 'training/history.csv',
        'evaluation/checks.json', 'evaluation/calibration.json',
        'evaluation/curves_per_seed.csv', 'evaluation/curves_macro.csv',
        'evaluation/threshold_per_seed.csv', 'evaluation/prediction_errors.csv',
        'evaluation/head_timings.csv', 'diagnostics/protocol.json', 'diagnostics/checks.json',
        'diagnostics/trace_curves_per_seed.csv', 'diagnostics/depth_curves_per_seed.csv',
        'diagnostics/trace_physics_timings.csv', 'diagnostics/additional_replay_cost.csv']
    selected += [p.name for p in sorted(HERE.glob('*.py'))]
    for name in selected:
        destination = evidence/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE/name, destination)
    paths = sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'manifest.json')
    manifest = {str(p.relative_to(HERE)): digest(p) for p in paths}
    write(HERE/'manifest.json', dict(files=manifest, root=str(HERE.relative_to(ROOT)),
        interpretation='Local experimental outputs; checks passed, scientific gate failed'))
    write(evidence/'source_manifest.json', dict(source_root=str(HERE.relative_to(ROOT)),
        source_manifest_sha256=digest(HERE/'manifest.json'),
        copied={name:digest(HERE/name) for name in selected},
        local_large_artifacts={name:manifest[name] for name in ('data/candidates.npz', 'actor_frozen.npz', 'value_frozen.npz')}))
    small = {str(p.relative_to(evidence)):digest(p) for p in sorted(evidence.rglob('*')) if p.is_file()}
    write(evidence/'manifest.json', dict(files=small))
    index = evidence.parent/'README.md'
    index.write_text('# 2026-10-29 过程价值研究证据\n\n'
        '[过程价值报告](tb_process_value/report.txt)记录最小离线蒸馏试验、强近似PHY和同深度对照。'
        '完整性核验通过，95%召回门槛未通过；未启动新seed闭环，当前算法未替换。\n\n'
        '全部原始候选、模型及源码快照保存在本地`deliverables/tb_process_value_20261029_v1/`；'
        '此处保存小型报告、协议、统计与SHA-256。\n')
    # Verify both the complete local archive and its compact evidence copy.
    for name, expected in manifest.items():
        assert digest(HERE/name) == expected
    for name, expected in small.items():
        assert digest(evidence/name) == expected
    print(f'FINALIZED local_files={len(manifest)} evidence_files={len(small)}', flush=True)


if __name__ == '__main__':
    main()

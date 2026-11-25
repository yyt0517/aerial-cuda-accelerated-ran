from pathlib import Path
import collections
import hashlib
import json
import shutil

ROOT = Path('/data/aerial-cuda-accelerated-ran')
DEST = ROOT / 'Doc/evidence/2026-11-25'
TEXT = {'.json', '.csv', '.rst', '.md', '.py', '.cpp', '.cu', '.h', '.hpp', '.diff', '.sh', '.txt', '.log', '.html'}
FIGURES = {'.png', '.pdf'}
BUNDLES = {
    'stage2_closure': 'stage2_closure_20261104',
    'stage2_metrics_revision': 'stage2_closure_20261105_r2',
    'stage2_generalization': 'stage2_generalization_20261106',
    'seven_cell': 'seven_cell_boundary08_20261105_v1',
    'seven_cell_multistart': 'seven_cell_boundary08_multistart_20261105_v1',
    'seven_cell_guarded': 'seven_cell_boundary08_guarded_l010_b098_20261106_v1',
    'seven_cell_delay': 'seven_cell_delay_diagnosis_20261106_v1',
    'rsrp': 'rsrp_interference_20261109',
    'rsrp_ablation': 'rsrp_ablation_20261110',
    'rsrp_covariance': 'rsrp_covariance_20261110',
}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def selected(p, rel):
    if '__pycache__' in rel.parts or p.is_symlink():
        return False
    # Preserve final aggregates and small execution/provenance records. Raw
    # trajectories, intermediate runs and host binaries remain at their source.
    if any(x.startswith(('aborted_', 'smoke', 'replay_smoke', 'campaign_workers32_initial')) for x in rel.parts):
        return len(rel.parts) == 1 and p.suffix in TEXT
    if p.suffix not in TEXT | FIGURES or p.stat().st_size > 2 * 1024 * 1024:
        return False
    if 'source_documents' in rel.parts or 'concurrent_timing_archive' in rel.parts:
        return False
    if any(x in rel.parts for x in ('campaign', 'parallel_tail', 'parity_disabled_seed41', 'three_cell_parity')):
        return p.suffix == '.json' or p.suffix in {'.py', '.cpp', '.cu', '.h', '.diff', '.rst'}
    if p.suffix == '.csv' and any(x in p.name for x in ('cohort', 'search_trace', 'queue_trace', 'packet_events')):
        return False
    if len(rel.parts) == 1 and p.suffix == '.pdf':
        return False
    return True

def copy_bundle(name, source, complete=False, output=False):
    dest = DEST / name
    entries = []
    omitted = collections.Counter()
    omitted_bytes = 0
    for p in sorted(source.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(source)
        keep = complete or (p.suffix in TEXT and '__pycache__' not in rel.parts if output else selected(p, rel))
        if not keep:
            omitted[p.suffix or '(no suffix)'] += 1
            omitted_bytes += p.stat().st_size
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        digest = sha(p)
        assert sha(target) == digest
        entries.append({'path': str(rel), 'bytes': p.stat().st_size, 'sha256': digest})
    manifest = {'archived_on': '2026-11-25', 'source_root': str(source.relative_to(ROOT)),
                'complete_copy': complete, 'files': entries, 'omitted_files_by_extension': dict(omitted),
                'omitted_bytes': omitted_bytes,
                'scope': 'Frozen source bytes. Omitted raw data, binaries and exports remain local; original manifests may reference omitted files.'}
    # Do not alter the self-contained delivery's own manifest or file set.
    manifest_path = DEST / 'manifests' / (name.replace('/', '__') + '.json')
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(name, len(entries), round(sum(x['bytes'] for x in entries) / 1048576, 2), 'MiB')

if __name__ == '__main__':
    assert not DEST.exists(), 'Archive already exists; never overwrite frozen evidence'
    copy_bundle('minimal_delivery', ROOT / 'deliverables/stage2_minimal_delivery_20261123', complete=True)
    for name, original in BUNDLES.items():
        copy_bundle(name, ROOT / 'deliverables' / original)
    for source in sorted((ROOT / 'outputs').iterdir()):
        if source.is_dir():
            copy_bundle('report_support/' + source.name, source, output=True)

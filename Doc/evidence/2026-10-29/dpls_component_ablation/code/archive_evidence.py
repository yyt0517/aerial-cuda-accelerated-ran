"""Archive reviewable small evidence; keep native binaries/cohorts local."""
import hashlib
import json
from pathlib import Path
import shutil

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert json.loads((OUT/'validation.json').read_text())['passed']
    protocol=json.loads((OUT/'protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():assert sha(OUT/name)==digest
    assert sha(REPO/'training/stageb_hgraph/current_algorithm.json')==protocol['registry_sha256']
    files=[p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json']
    (OUT/'manifest.json').write_text(json.dumps(dict(files={p.name:sha(p) for p in files},
        protocol_sha256=sha(OUT/'protocol.json'),report_sha256=sha(OUT/'report.txt'),
        conclusions_sha256=sha(OUT/'conclusions.txt')),indent=2)+'\n')
    target=REPO/'Doc/evidence/2026-10-29/dpls_component_ablation'
    target.mkdir(parents=True,exist_ok=False)
    for path in OUT.iterdir():
        if path.is_file() and path.suffix in ['.csv','.json','.txt','.png','.pdf','.rst']:
            shutil.copy2(path,target/path.name)
    (target/'code').mkdir()
    for path in OUT.glob('*.py'):shutil.copy2(path,target/'code'/path.name)
    shutil.copytree(OUT/'checks',target/'checks')
    (target/'README.rst').write_text('DPLS component ablation evidence\n===============================\n\n'
        'Read conclusions.txt and report.txt. Eight arms, five paired seeds,\n'
        'thirty new full native trajectories and ten verified references.\n'
        'This directory archives small results, input/physical checks, and\n'
        'exact source snapshots; it does not contain native binaries, full\n'
        'metrics streams, or the complete packet-cohort tables.\n\n'
        'Canonical local experiment: deliverables/dpls_component_ablation_20261029_v1/.\n'
        'The copied code is a source snapshot, not an independently runnable\n'
        'package at this archive location. To reproduce, copy only code/*.py\n'
        'into a fresh deliverables/<experiment_name>/ directory, retain the\n'
        'existing research worktree, native simulator, frozen libraries and\n'
        'reference trajectories, and run setup.py, preflight.py, campaign.py\n'
        'inside the documented CUDA container. Do not copy protocol/output\n'
        'files into the fresh directory before setup. See runtime_notes.rst.\n\n'
        'manifest.json is the canonical local file manifest, so its paths\n'
        'and hashes refer to that local experiment. archive_manifest.json\n'
        'covers exactly the files archived here. Runtime notes below were\n'
        'renamed to preserve this archive introduction.\n')
    shutil.copy2(OUT/'README.rst',target/'runtime_notes.rst')
    (target/'archive_manifest.json').write_text(json.dumps(dict(
        canonical_source=str(OUT),canonical_manifest_sha256=sha(OUT/'manifest.json'),
        files={str(p.relative_to(target)):sha(p) for p in sorted(target.rglob('*')) if p.is_file() and p.name!='archive_manifest.json'},
        excluded=['native/','packet_cohorts/','analysis_parts/','attempts/','logs/','runtime/'],
        excluded_reason='Large native data, transient runtime files and interrupted startup/warmup attempts remain in the canonical local directory.'),indent=2)+'\n')
    print('EVIDENCE_ARCHIVED',target)


if __name__=='__main__':main()

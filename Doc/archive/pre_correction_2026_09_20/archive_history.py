"""One-shot, lossless archival approved on 2026-09-20. No deletion or training."""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOC_ARCHIVE = Path(__file__).resolve().parent
DATA_ARCHIVE = ROOT/'output/pfq_independent_reuse/archive/pre_correction_2026_09_20'
FORMAL = ROOT/'output/pfq_independent_reuse/formal'
VERSION = ROOT/'output/pfq_independent_reuse/versions/stageb_corrected_2026_09_20_v1'
KEEP = {'mcs_capacity_units_audit_v1','phy_slot_invariance_v1','gpu_mcs_sync_v1','seed42_pfq_rrq_corrected_v1'}

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

assert not DATA_ARCHIVE.exists() and not VERSION.exists(), 'Do not overwrite an archive/version'
entries = sorted(FORMAL.iterdir())
assert KEEP <= {p.name for p in entries}
targets = [p for p in entries if p.name not in KEEP]
assert all(not p.is_symlink() for p in targets), 'Resolve pre-existing aliases before archival'
DATA_ARCHIVE.mkdir(parents=True)
(DATA_ARCHIVE/'formal').mkdir()
manifest = {'date':'2026-09-20','status':'preparing','root':str(ROOT),
            'current_formal_entries':sorted(KEEP),'moves':[],'documents':[]}
for src in targets:
    manifest['moves'].append({'old':str(src.relative_to(ROOT)),
        'archive':str((DATA_ARCHIVE/'formal'/src.name).relative_to(ROOT))})
(DATA_ARCHIVE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

# Preserve exact pre-edit documents, including historical parameter JSON.
documents = [ROOT/'README.md', ROOT/'training/stageb_hgraph/README.md']
documents += sorted(p for p in (ROOT/'Doc').iterdir() if p.is_file())
documents += sorted((ROOT/'docs/superpowers').rglob('*.md'))
for src in documents:
    dst = DOC_ARCHIVE/'originals'/src.relative_to(ROOT)
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)
    assert digest(src)==digest(dst)
    manifest['documents'].append({'original':str(src.relative_to(ROOT)),
        'archive':str(dst.relative_to(ROOT)), 'sha256':digest(dst)})
(DOC_ARCHIVE/'originals/output').symlink_to(os.path.relpath(ROOT/'output',DOC_ARCHIVE/'originals'),target_is_directory=True)

# Hash every legacy regular file before moving, including BC data/checkpoints.
# Preserve symlinks as links; never traverse out-of-tree targets.
for move in manifest['moves']:
    src = ROOT/move['old']; dst = ROOT/move['archive']
    files = [src] if src.is_file() else sorted(src.rglob('*'))
    records = []
    for p in files:
        rel = str(p.relative_to(src)) if p != src else '.'
        if p.is_symlink(): records.append({'path':rel,'symlink':os.readlink(p)})
        elif p.is_file():
            s=p.stat(); records.append({'path':rel,'bytes':s.st_size,'sha256':digest(p),'inode':s.st_ino,'device':s.st_dev})
    move['files']=records
    src.rename(dst)
    src.symlink_to(os.path.relpath(dst,src.parent),target_is_directory=dst.is_dir())
    # Same-filesystem rename must preserve each file identity and size.
    for r in records:
        p=dst if r['path']=='.' else dst/r['path']
        if 'symlink' in r: assert p.is_symlink() and os.readlink(p)==r['symlink']
        else:
            s=p.stat(); assert (s.st_ino,s.st_dev,s.st_size)==(r['inode'],r['device'],r['bytes'])
    print('archived',src.name,len(records),'entries',flush=True)
    manifest['status']='moving'
    (DATA_ARCHIVE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

# Pin the tested corrected build plus dirty source reconstruction, not HEAD alone.
VERSION.mkdir(parents=True)
build=ROOT/'build.gpu_mcs_sync_v1'
for src,name,want in [
    (build/'cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection','simulator','b0e56c8b851603a920467d8ca2c2e24e554b4048a207204ee985abedd9a73644'),
    (build/'cuMAC/src/libcumac.so','libcumac.so','079f4c72ca94358a410584fe1a666914a5da4041acbc4085fada50957363692e')]:
    assert digest(src)==want
    shutil.copy2(src,VERSION/name)
    assert digest(VERSION/name)==want
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
patch=subprocess.check_output(['git','diff','--binary','HEAD','--','cuMAC','training'],cwd=ROOT)
(VERSION/'tracked_source.patch').write_bytes(patch)
paths=subprocess.check_output(['git','ls-files','-m','-o','--exclude-standard','--','cuMAC','training'],cwd=ROOT,text=True).splitlines()
source=[]
for name in sorted(set(paths)):
    src=ROOT/name
    if src.is_symlink() or not src.is_file() or src.suffix not in {'.py','.sh','.cu','.cuh','.cpp','.h','.hpp','.txt','.cmake','.json'}: continue
    dst=VERSION/'source_overlay'/name; dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    source.append({'path':name,'sha256':digest(dst)})
version={'version_id':'stageb_corrected_2026_09_20_v1','git_base':head,'dirty_worktree':True,
         'note':'Research environment version, not a Git commit or a validated DPA checkpoint.',
         'source_overlay':source,'tracked_patch_sha256':digest(VERSION/'tracked_source.patch'),
         'simulator_sha256':digest(VERSION/'simulator'),'libcumac_sha256':digest(VERSION/'libcumac.so')}
(VERSION/'manifest.json').write_text(json.dumps(version,indent=2)+'\n')
manifest['status']='complete'
manifest['regular_files']=sum('sha256' in r for m in manifest['moves'] for r in m['files'])
manifest['total_bytes']=sum(r.get('bytes',0) for m in manifest['moves'] for r in m['files'])
(DATA_ARCHIVE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(DOC_ARCHIVE/'manifest.json').write_text(json.dumps({k:v for k,v in manifest.items() if k!='moves'},indent=2)+'\n')
print(json.dumps({k:manifest[k] for k in ['status','regular_files','total_bytes']}),flush=True)

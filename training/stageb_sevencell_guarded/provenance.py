"""Freeze this experiment and the existing search libraries used at runtime."""
from pathlib import Path
from training.stageb_ppo.native import ROOT,sha
from training.stageb_lns.archive import ARCHIVE

def source_hashes():
    paths=[]
    for package in ('stageb_sevencell_guarded','stageb_sevencell','stageb_sevencell_multistart'):
        paths.extend(p for p in (ROOT/'training'/package).rglob('*') if p.is_file() and p.suffix in ('.py','.cpp','.so'))
    for p in (ROOT/'deliverables/seven_cell_boundary08_20261105_v1/source_snapshot/shared').iterdir():
        if p.is_file():paths.append(ROOT/'training/stageb_lns'/p.name)
    for name in ('ttl_weights.py','no_ctbs_inputs.py','physics.py','persistent_dpls.py','dpls_operations.py','ranked_dpls.py','bounded_campaign.py','archive.py'):
        paths.append(ROOT/'training/stageb_lns'/name)
    for name in ('bridge_env.py','phy_tbs_alignment_audit.py'):
        paths.append(ARCHIVE/'training/stageb_hgraph'/name)
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}

def verify_sources(expected):
    for name,value in expected.items():assert sha(ROOT/name)==value,name

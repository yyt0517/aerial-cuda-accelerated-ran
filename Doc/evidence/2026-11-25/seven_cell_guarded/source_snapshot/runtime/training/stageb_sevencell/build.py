"""Isolated size-only simulator build and 128-bit UE dependency masks."""
import argparse
import difflib
import json
from pathlib import Path
import re
import subprocess
from training.stageb_ppo.native import ROOT, ARCHIVE, FORMAL, REFERENCE, sha, write


def prepare(out, cells=7):
    from .boundary import extend
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    old=FORMAL/'boundary08_corrected_physics_20261020_v1'
    manifest=json.loads((REFERENCE/'native_build_manifest.json').read_text())
    network_manifest=json.loads((old/'native_network_build.json').read_text())
    examples=ARCHIVE/'cuMAC/examples'
    sources={'network.cu':examples/'network.cu','network.h':examples/'network.h',
        'parameters.h':examples/'parameters.h','instrumented_main.cpp':REFERENCE/'instrumented_main.cpp',
        'fresh_observation.h':old/'fresh_observation.h',
        'directed_snapshot.h':FORMAL/'boundary08_directed_coordination_20261008_v1/directed_snapshot.h'}
    for p,digest in manifest['source_sha256'].items():
        assert sha(ROOT/Path(p).relative_to('/opt/nvidia/cuBB'))==digest
    assert sha(sources['instrumented_main.cpp'])==manifest['main_source_sha256']
    diffs=[]
    for name,p in sources.items():
        original=p.read_text();text=original
        if name=='parameters.h':
            for define in ('numCellConst','numCoorCellConst'):
                text,n=re.subn(r'(#define\s+'+define+r'\s+)3\b',lambda m:m[1]+str(cells),text);assert n==1
        if name=='network.cu' and cells==7:text=extend(text)
        if name=='directed_snapshot.h' and cells==7:
            assert 'u != 36 || c != 3' in text
            text=text.replace('u != 36 || c != 3','u != 84 || c != 7').replace('3cell/36UE','7cell/84UE')
        (out/name).write_text(text)
        diffs.extend(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(p),tofile=name))
    (out/'size_only.diff').write_text(''.join(diffs))
    cwd=ARCHIVE/'build.corrected_uniform_v1/cuMAC/examples/multiCellSchedulerUeSelection'
    network=list(network_manifest['command']);network[network.index('-c')+1]=str(out/'network.cu');network[network.index('-o')+1]=str(out/'network.o')
    network.insert(1,'-I'+str(examples))
    main=list(manifest['compile_main']);main[main.index('-c')+1]=str(out/'instrumented_main.cpp');main[main.index('-o')+1]=str(out/'main.o')
    main.insert(1,'-I'+str(out))
    link=list(manifest['link_command']);link[3]=str(out/'main.o');link[4]=str(out/'network.o');link[link.index('-o')+1]=str(out/'simulator')
    for cmd in (network,main,link):
        print('BUILD',Path(cmd[0]).name,flush=True);subprocess.run(cmd,cwd=cwd,check=True)
    library=ARCHIVE/'build.corrected_uniform_v1/cuMAC/src/libcumac.so'
    assert sha(library)==manifest['libcumac_sha256']
    write(out/'manifest.json',dict(cells=cells,users=12*cells,changes='Cell count and adjacent shared-boundary neighbor selection; frozen main, PHY and libcumac',
        source_sha256={str(p):sha(p) for p in sources.values()},compiled_sources={n:sha(out/n) for n in sources},
        simulator_sha256=sha(out/'simulator'),libcumac_sha256=sha(library),commands=[network,main,link]))


def solver():
    here=Path(__file__).parent;source=ROOT/'training/stageb_lns';dest=here/'kernel';dest.mkdir(exist_ok=True)
    diffs=[]
    for name in ('physics_kernel.cpp','persistent_kernel.cpp','bounded_kernel.cpp'):
        original=(source/name).read_text();text=original
        if name=='persistent_kernel.cpp':
            for a,b in [('uint64_t support=','__uint128_t support='),('uint64_t weak=0','__uint128_t weak=0'),
                        ('uint64_t used=','__uint128_t used='),('uint64_t changed=','__uint128_t changed='),
                        ('uint64_t(1)<<old','__uint128_t(1)<<old'),('uint64_t(1)<<next','__uint128_t(1)<<next'),
                        ('uint64_t(1)<<(phy.U+p)','__uint128_t(1)<<(phy.U+p)')]:
                assert a in text;text=text.replace(a,b)
        if name=='bounded_kernel.cpp':
            assert 'uint64_t changed=' in text;text=text.replace('uint64_t changed=','__uint128_t changed=')
        (dest/name).write_text(text)
        diffs.extend(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(source/name),tofile=name))
    (dest/'mask_width.diff').write_text(''.join(diffs))
    subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-ffp-contract=off','-fno-fast-math','-shared','-fPIC',
        '-Wl,-Bsymbolic','-I/usr/include/eigen3','-I/usr/local/include/eigen3',str(dest/'bounded_kernel.cpp'),'-o',str(here/'_bounded128.so')],check=True)
    write(dest/'manifest.json',dict(original={n:sha(source/n) for n in ('physics_kernel.cpp','persistent_kernel.cpp','bounded_kernel.cpp')},
        modified={p.name:sha(p) for p in dest.glob('*.cpp')},library_sha256=sha(here/'_bounded128.so'),
        scope='UE support/footprint masks widened to 128 bits; cell activity masks and canonical int64 keys unchanged'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--cells',type=int,default=7);p.add_argument('--solver-only',action='store_true');p.add_argument('--native-only',action='store_true');a=p.parse_args()
    if not a.native_only:solver()
    if not a.solver_only:prepare(a.output,a.cells)

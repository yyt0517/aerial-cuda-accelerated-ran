"""Compile only an instrumented copy of main; link the verified native objects."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
SOURCE = ROOT/'cuMAC/examples/multiCellSchedulerUeSelection/main.cpp'
BUILD = ROOT/'build.corrected_uniform_v1/cuMAC/examples/multiCellSchedulerUeSelection'
TARGET = BUILD/'CMakeFiles/multiCellSchedulerUeSelection.dir'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    source = SOURCE.read_text()
    needle = '        if (!onlineBridge->sendPreActionRsp(snapshot)) {'
    assert source.count(needle) == 1
    source = source.replace('using namespace cumac;', '#include "directed_snapshot.h"\n\nusing namespace cumac;', 1)
    source = source.replace(needle, '        writeDirectedSnapshot(t, net->cellGrpPrmsGpu.get(), net->cellGrpUeStatusCpu.get());\n'+needle)
    generated = OUT/'instrumented_main.cpp'
    generated.write_text(source)
    flags = {}
    for line in (TARGET/'flags.make').read_text().splitlines():
        if ' = ' in line:
            key,value = line.split(' = ',1)
            flags[key] = shlex.split(value)
    compiler = '/usr/local/gnu/bin/g++'
    obj = OUT/'instrumented_main.o'
    compile_cmd = [compiler,*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],
                   '-I'+str(SOURCE.parent),'-c',str(generated),'-o',str(obj)]
    subprocess.run(compile_cmd,cwd=BUILD,check=True)
    link_cmd = shlex.split((TARGET/'link.txt').read_text())
    link_cmd[link_cmd.index('CMakeFiles/multiCellSchedulerUeSelection.dir/main.cpp.o')] = str(obj)
    link_cmd[link_cmd.index('-o')+1] = str(OUT/'simulator')
    subprocess.run(link_cmd,cwd=BUILD,check=True)
    manifest = dict(original_main_sha256=sha(SOURCE),original_simulator_sha256=sha(BUILD/'multiCellSchedulerUeSelection'),
                    original_libcumac_sha256=sha(ROOT/'build.corrected_uniform_v1/cuMAC/src/libcumac.so'),
                    probe_simulator_sha256=sha(OUT/'simulator'),compile_command=compile_cmd,link_command=link_cmd)
    (OUT/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k.endswith('sha256')}),flush=True)

if __name__ == '__main__': main()

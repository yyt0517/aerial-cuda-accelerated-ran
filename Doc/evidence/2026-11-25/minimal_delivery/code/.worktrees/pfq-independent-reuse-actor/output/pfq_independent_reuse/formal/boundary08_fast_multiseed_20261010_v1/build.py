"""Only remove diagnostic output and instrument CPU PFQ; retain native transitions."""
import json
import shlex
import subprocess
from pathlib import Path
from output.pfq_independent_reuse.formal.boundary08_directed_coordination_20261008_v1.build_probe import sha

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
OLD = OUT.parent / 'boundary08_directed_coordination_20261008_v1'
BUILD = ROOT / 'build.corrected_uniform_v1/cuMAC/examples/multiCellSchedulerUeSelection'
TARGET = BUILD / 'CMakeFiles/multiCellSchedulerUeSelection.dir'
# Explicit doubles avoid C++ struct packing and preserve integer bytes exactly.
FIELDS = dict(tti='tti', ue_id='ueId', offered_bytes='offeredBytes', accepted_bytes='acceptedBytes',
    rejected_bytes='rejectedBytes', expired_bytes='expiredBytes', previous_post_queue_bytes='previousPostQueueBytes',
    pre_queue_bytes='preQueueBytes', scheduled_dequeue_bytes='scheduledDequeueBytes', post_queue_bytes='postQueueBytes',
    successful_bytes='successfulBytes', failed_bytes='failedBytes', actual_prg_count='actualPrgCount',
    mcs='mcs', layer='layer', phy_sinr_db_cpu='phySinrDbCpu', predicted_bler_cpu='predictedBlerCpu',
    tb_tx_count='tbTxCount', tb_err_count='tbErrCount')

def main():
    source = (OLD / 'instrumented_main.cpp').read_text()
    changes = {}
    def replace(before, after):
        nonlocal source
        assert source.count(before) == 1, before
        changes[before] = after
        source = source.replace(before, after)
    replace('#include "directed_snapshot.h"', '#include "directed_snapshot.h"\n#include <chrono>')
    replace('  auto captureOnlineTeacherAction = [&]() {',
        '  std::vector<double> pfqCoreUs;\n  auto captureOnlineTeacherAction = [&]() {')
    replace('    net->ueDownSelectCpu();\n    if (rrBaseline)',
        '    const auto pfqStart = std::chrono::steady_clock::now();\n    net->ueDownSelectCpu();\n    if (rrBaseline)')
    replace('    onlineTeacherActionUe = buildOnlineTeacherUeAction',
        '    pfqCoreUs.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-pfqStart).count());\n    onlineTeacherActionUe = buildOnlineTeacherUeAction')
    replace('  if (inlineTtiProgressPrinted) {',
        '  { std::ofstream f("pfq_core_us.bin", std::ios::binary); f.write(reinterpret_cast<const char*>(pfqCoreUs.data()), pfqCoreUs.size()*sizeof(double)); }\n  if (inlineTtiProgressPrinted) {')
    start = source.index('    writeServiceBalanceTraceHeader(serviceBalanceTrace);')
    end = source.index('\n  }', start)
    replace(source[start:end], '')
    replace('      channelFingerprintTrace << (t + 1) << \',\' << net->rawChannelFingerprint() << \'\\n\';', '')
    values = ','.join('static_cast<double>(row.'+v+')' for v in FIELDS.values())
    replace('        if (row.tti >= serviceBalanceTraceFirstTti &&\n            !writeServiceBalanceTraceRow(serviceBalanceTrace, row)) {',
        '        const double metricValues[] = {'+values+'};\n'
        '        if (row.tti >= serviceBalanceTraceFirstTti) serviceBalanceTrace.write(reinterpret_cast<const char*>(metricValues), sizeof(metricValues));\n'
        '        if (!serviceBalanceTrace) {')
    replace('    serviceBalanceTrace.open(serviceBalanceTraceCsv);', '    serviceBalanceTrace.open(serviceBalanceTraceCsv, std::ios::binary);')
    generated = OUT / 'instrumented_main.cpp'
    generated.write_text(source)
    flags = {}
    for line in (TARGET/'flags.make').read_text().splitlines():
        if ' = ' in line:
            k,v = line.split(' = ',1); flags[k] = shlex.split(v)
    obj = OUT/'instrumented_main.o'
    compile_cmd = ['/usr/local/gnu/bin/g++', *flags['CXX_DEFINES'], *flags['CXX_INCLUDES'], *flags['CXX_FLAGS'],
        '-I'+str(ROOT/'cuMAC/examples/multiCellSchedulerUeSelection'), '-I'+str(OLD), '-c', str(generated), '-o', str(obj)]
    subprocess.run(compile_cmd, cwd=BUILD, check=True)
    link = shlex.split((TARGET/'link.txt').read_text())
    link[link.index('CMakeFiles/multiCellSchedulerUeSelection.dir/main.cpp.o')] = str(obj)
    link[link.index('-o')+1] = str(OUT/'simulator')
    subprocess.run(link, cwd=BUILD, check=True)
    (OUT/'build_manifest.json').write_text(json.dumps(dict(changes=changes, fields=list(FIELDS),
        original_source_sha256=sha(OLD/'instrumented_main.cpp'), source_sha256=sha(generated),
        simulator_sha256=sha(OUT/'simulator'), lib_sha256=sha(ROOT/'build.corrected_uniform_v1/cuMAC/src/libcumac.so')) ,indent=2))

if __name__ == '__main__': main()

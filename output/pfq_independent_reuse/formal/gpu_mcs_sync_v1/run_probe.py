"""One fixed-input kernel replay plus paired CPU-service PF/RR scenarios."""
import argparse
import csv
import json
import shutil
import time
import traceback
from pathlib import Path

from training.stageb_hgraph.dpa_ablation_experiment import ROOT, SIM_ARGS, base_env, eval_env, cli, run_command, sha, write_json
from training.stageb_hgraph.pfq_reference_calibration import _extend_ld_library_path
from training.stageb_hgraph.capacity_units_decision_probe import summarize_arm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['before', 'after'], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / args.phase
    root.mkdir(exist_ok=False)
    build = ROOT / 'build.gpu_mcs_sync_v1'
    sim = build / 'cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection'
    lib = build / 'cuMAC/src/libcumac.so'
    # Preserve the actual shared library; binary hash alone misses this change.
    shutil.copy2(sim, root / 'simulator')
    shutil.copy2(lib, root / 'libcumac.so')
    for name in ('mcsSelectionLUT.cu', 'mcsSelectionLUT.cuh'):
        shutil.copy2(ROOT / 'cuMAC/src/4T4R' / name, root / name)
    status = dict(status='running', phase=args.phase, started_at=time.time(),
                  binary_sha256=sha(sim), libcumac_sha256=sha(lib), arms={})
    write_json(root/'status.json', status)
    try:
        for policy, baseline in [('pf',0), ('rr',1)]:
            out = root / policy
            out.mkdir()
            status['current'] = policy
            write_json(root/'status.json', status)
            env = eval_env(base_env(1), out, 61, '/tmp/mcs_sync_unused.sock')
            env.update(CUMAC_PHY_AUDIT_PERMUTE_SLOTS='0', CUMAC_PHY_AUDIT_COMMON_MCS='0',
                       CUMAC_RR_AUDIT_SEED='61', CUMAC_MCS_FIXED_INPUT_AUDIT_CSV=str(out/'gpu_mcs_replay.csv'))
            _extend_ld_library_path(env, sim)
            command = cli('pfq_reference_calibration', {
                'sim-bin':sim, 'sim-args':SIM_ARGS.replace('-b 2', f'-b {baseline}'), 'sim-cwd':out,
                'topology-seed':61, 'traffic-seed':61, 'reference-policy':policy,
                'packet-ttl-ms':200, 'warmup-steps':1000, 'measure-steps':4000,
                'service-balance-trace-first-tti':1, 'out-dir':out}) + ['--service-balance-audit']
            run_command(command, out, env, 3600)
            rows = list(csv.DictReader((out/'gpu_mcs_replay.csv').open()))
            replay = dict(rows=len(rows), repeats=len({r['repeat'] for r in rows}),
                input_sha256=sha(out/'gpu_mcs_replay.csv.inputs'),
                count_mismatches=sum(r['raw_count'] != r['expected_count'] for r in rows),
                mcs_mismatches=sum(r['raw_mcs'] != r['expected_mcs'] for r in rows),
                invalid_scheduled_mcs=sum(int(r['expected_count'])>0 and int(r['raw_mcs'])<0 for r in rows),
                variable_slots=sum(len({(r['raw_count'],r['raw_mcs']) for r in rows if r['slot']==s})>1
                                   for s in {r['slot'] for r in rows}),
                examples=[r for r in rows if r['raw_count']!=r['expected_count'] or r['raw_mcs']!=r['expected_mcs']][:10])
            status['arms'][policy] = dict(replay=replay, service=summarize_arm(out,warmup=1000,measured=4000))
            write_json(root/'status.json',status)
        status.update(status='complete',completed_at=time.time())
        write_json(root/'report.json',status); write_json(root/'status.json',status)
    except BaseException:
        status.update(status='failed',error=traceback.format_exc())
        write_json(root/'status.json',status)
        raise


if __name__ == '__main__':
    main()

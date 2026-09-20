"""Run existing corrected simulator/calibration/aggregation, seed42 PFQ vs RRQ."""
import time
import traceback
from pathlib import Path

from training.stageb_hgraph.dpa_ablation_experiment import (
    ROOT, REFERENCE_SIM_ARGS, base_env, eval_env, cli, run_command, sha, write_json,
)
from training.stageb_hgraph.pfq_reference_calibration import _extend_ld_library_path
from training.stageb_hgraph.capacity_units_decision_probe import summarize_arm

root = Path(__file__).resolve().parent
sim = ROOT / 'build.gpu_mcs_sync_v1/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection'
lib = ROOT / 'build.gpu_mcs_sync_v1/cuMAC/src/libcumac.so'
assert sha(sim) == 'b0e56c8b851603a920467d8ca2c2e24e554b4048a207204ee985abedd9a73644'
assert sha(lib) == '079f4c72ca94358a410584fe1a666914a5da4041acbc4085fada50957363692e'
if (root / 'status.json').exists():
    raise RuntimeError('Existing run: preserve its artifacts, do not overwrite')
status = dict(status='running', seed=42, warmup=1000, measured=4000,
              binary_sha256=sha(sim), libcumac_sha256=sha(lib), started_at=time.time(), arms={})
write_json(root/'status.json',status)
try:
    for policy in ('pfq','rrq'):
        out = root/policy
        out.mkdir(exist_ok=False)
        status['current'] = policy; write_json(root/'status.json',status)
        env = eval_env(base_env(1),out,42,'/tmp/seed42_baselines_unused.sock')
        env.update(CUMAC_PHY_AUDIT_PERMUTE_SLOTS='0', CUMAC_PHY_AUDIT_COMMON_MCS='0', CUMAC_RR_AUDIT_SEED='42')
        _extend_ld_library_path(env,sim)
        cmd = cli('pfq_reference_calibration',{
            'sim-bin':sim, 'sim-args':REFERENCE_SIM_ARGS[policy], 'sim-cwd':out,
            'topology-seed':42, 'traffic-seed':42, 'reference-policy':policy,
            'traffic-profile':'homogeneous', 'packet-ttl-ms':200,
            'warmup-steps':1000, 'measure-steps':4000,
            'service-balance-trace-first-tti':1, 'out-dir':out}) + ['--service-balance-audit','--formal']
        run_command(cmd,out,env,3600)
        assert sha(sim)==status['binary_sha256'] and sha(lib)==status['libcumac_sha256']
        status['arms'][policy]=summarize_arm(out,warmup=1000,measured=4000)
        write_json(root/'status.json',status)
    for key in ('offered_identity','topology_sha256','channel_fingerprint_sha256'):
        assert status['arms']['pfq'][key] == status['arms']['rrq'][key], key
    status.update(status='complete', completed_at=time.time(), paired_inputs_equal=True)
    write_json(root/'final_report.json',status); write_json(root/'status.json',status)
except BaseException:
    status.update(status='failed',error=traceback.format_exc())
    write_json(root/'status.json',status)
    raise

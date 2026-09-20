"""One-seed PF/RR causal PHY check and paired 1000+4000 comparison. No training."""
import argparse
import json
import time
import traceback
from pathlib import Path

from training.stageb_hgraph.dpa_ablation_experiment import ROOT, SIM_ARGS, base_env, eval_env, cli, run_command, read_csv, sha, write_json
from training.stageb_hgraph.pfq_reference_calibration import _extend_ld_library_path
from training.stageb_hgraph.capacity_units_decision_probe import summarize_arm


def run(out, sim, policy, permutation, warmup, measured):
    out.mkdir(parents=True, exist_ok=False)
    env = eval_env(base_env(1), out, 61, '/tmp/slot_probe_unused.sock')
    env['CUMAC_PHY_AUDIT_PERMUTE_SLOTS'] = str(int(permutation))
    env['CUMAC_RR_AUDIT_SEED'] = '61'
    env['CUMAC_PHY_AUDIT_COMMON_MCS'] = str(int(measured == 1))
    _extend_ld_library_path(env, sim)
    args = SIM_ARGS.replace('-b 2', '-b '+str({'pf':0, 'rr':1}[policy]))
    command = cli('pfq_reference_calibration', {'sim-bin': sim, 'sim-args': args,
        'sim-cwd': out, 'topology-seed': 61, 'traffic-seed': 61,
        'reference-policy': policy, 'packet-ttl-ms': 200,
        'warmup-steps': warmup, 'measure-steps': measured,
        'service-balance-trace-first-tti': 1, 'out-dir': out}) + ['--service-balance-audit']
    run_command(command, out, env, 3600)


def compare(first, second):
    a = sorted(read_csv(first/'service_balance_trace.csv'), key=lambda r:int(r['ue_id']))
    b = sorted(read_csv(second/'service_balance_trace.csv'), key=lambda r:int(r['ue_id']))
    assert len(a) == len(b) == 36
    # First TTI: same exogenous/state inputs and fixed canonical action/MCS/layer.
    keys = ['tti','ue_id','serving_cell','offered_bytes','pre_queue_bytes',
            'actual_prg_mask','actual_prg_count','mcs','layer','tbs_capacity_bits','scheduled_dequeue_bytes']
    assert all(x[k] == y[k] for x,y in zip(a,b) for k in keys)
    assert sha(first/'traffic_offered_packets.csv') == sha(second/'traffic_offered_packets.csv')
    assert sha(first/'ue_topology.csv') == sha(second/'ue_topology.csv')
    assert sha(first/'service_balance_trace.csv.channel.csv') == sha(second/'service_balance_trace.csv.channel.csv')
    result = {'physical_inputs_equal': True,
        'changed_slot_count': sum(x['sched_slot'] != y['sched_slot'] for x,y in zip(a,b)),
        'tb_error_changed_ues': sum(x['tb_err_count'] != y['tb_err_count'] for x,y in zip(a,b)),
        'goodput_changed_ues': sum(x['successful_bytes'] != y['successful_bytes'] for x,y in zip(a,b))}
    for side in ('cpu','gpu'):
        for field,label in [(f'phy_sinr_db_{side}',f'max_sinr_delta_{side}_db'),
                            (f'predicted_bler_{side}',f'max_predicted_bler_delta_{side}')]:
            result[label] = max(abs(float(x[field])-float(y[field])) for x,y in zip(a,b))
    result['examples'] = [{k:x[k] for k in ['ue_id','actual_prg_mask','mcs','layer']}
        | {'slot_before': x['sched_slot'], 'slot_after': y['sched_slot'],
           'sinr_before': x['phy_sinr_db_cpu'], 'sinr_after': y['phy_sinr_db_cpu'],
           'bler_before': x['predicted_bler_cpu'], 'bler_after': y['predicted_bler_cpu']}
        for x,y in zip(a,b) if abs(float(x['phy_sinr_db_cpu'])-float(y['phy_sinr_db_cpu'])) > 1e-5][:5]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['before','after'], required=True)
    p.add_argument('--sim-bin', required=True)
    p.add_argument('--label')
    p.add_argument('--audit-only', action='store_true')
    args = p.parse_args()
    root = Path(__file__).resolve().parent / (args.label or args.phase)
    root.mkdir(exist_ok=False)
    sim = Path(args.sim_bin).resolve()
    status = {'status':'running', 'binary_sha256':sha(sim), 'phase':args.phase,
              'started_at':time.time(), 'seed':61, 'arrival_rate':1.0, 'ttl_ms':200,
              'one_tti_common_cpu_mcs_layers': True, 'rr_audit_seed': 61}
    write_json(root/'status.json', status)
    try:
        invariance = {}
        for policy in ('pf','rr'):
            for perm in (False,True):
                out = root/f'{policy}_one_tti_perm{int(perm)}'
                status['current'] = out.name; write_json(root/'status.json',status)
                run(out, sim, policy, perm, 0, 1)
            invariance[policy] = compare(root/f'{policy}_one_tti_perm0', root/f'{policy}_one_tti_perm1')
        write_json(root/'invariance.json',invariance)
        long = {}
        for policy in (() if args.audit_only else ('pf','rr')):
            out = root/f'{policy}_5000'
            status['current'] = out.name; write_json(root/'status.json',status)
            run(out, sim, policy, False, 1000, 4000)
            long[policy] = summarize_arm(out, warmup=1000, measured=4000)
        status.update(status='complete', completed_at=time.time(), invariance=invariance, arms=long)
        write_json(root/'report.json',status); write_json(root/'status.json',status)
    except BaseException:
        status.update(status='failed', error=traceback.format_exc())
        write_json(root/'status.json',status)
        raise


if __name__ == '__main__':
    main()

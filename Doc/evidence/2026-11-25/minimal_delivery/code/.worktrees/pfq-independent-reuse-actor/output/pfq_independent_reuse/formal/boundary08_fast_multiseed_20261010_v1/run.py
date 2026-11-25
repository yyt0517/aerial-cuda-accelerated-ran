"""Frozen-policy evaluation: necessary native evolution, no candidate oracle/audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
# This is a CPU inference benchmark; small NumPy matrices must not create a
# 64-thread BLAS pool. Set this before importing NumPy/Torch in fresh workers.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import torch
from training.stageb_hgraph.directed_reuse import build_directed_features, DirectedReuseCoordinator, choose_blank
from output.pfq_independent_reuse.formal.boundary08_directed_coordination_20261008_v1 import run_probe as old
from .build import OUT, ROOT, sha

p, c = old.p, old.c
CHECKPOINT = OUT.parent/'boundary08_directed_attention_20261008_v1/training/full.pt'

def timing_summary(x):
    x = np.asarray(x, dtype=float)
    return dict(n=len(x), mean_us=float(x.mean()), p50_us=float(np.quantile(x,.5)),
                p95_us=float(np.quantile(x,.95)), p99_us=float(np.quantile(x,.99)))

def run(args):
    out = OUT/args.name
    out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    snapshot_path = out/'pre_action.bin'
    socket = f'/tmp/b08fast_{os.getpid()}.sock'
    horizon = args.warmup + args.measured + args.followup
    env = c.eval_env(c.environment(), out, args.seed, socket)
    env.update(CUMAC_UE_PLACEMENT_MODE='coop_boundary', CUMAC_TRAFFIC_ARRIVAL_RATE='.8',
        CUMAC_PFQ_DEMAND_CAP_SLACK_BYTES='0', CUMAC_DPA_FEATURE_MODE='dimensionless',
        CUMAC_DIRECTED_SNAPSHOT_PATH=str(snapshot_path), OPENBLAS_NUM_THREADS='1',
        CUDA_VISIBLE_DEVICES=str(args.gpu), CUMAC_SERVICE_BALANCE_TRACE_CSV=str(out/'metrics.bin'),
        CUMAC_PACKET_DELAY_SAMPLES_CSV='/dev/null')
    command = [str(OUT/'simulator'), '-d','1','-b','2','-x','0','-f','0','-g','100','-r','3000']
    actor = coordinator = None
    if args.arm == 'full':
        actor,_ = old._load_actor_checkpoint(old.CHECKPOINT, torch.device('cpu'))
        ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        assert sha(old.CHECKPOINT) == ckpt['base_checkpoint_sha256']
        assert sha(ROOT/'training/stageb_hgraph/directed_reuse.py') == ckpt['network_source_sha256']
        assert sha(CHECKPOINT) == '1e259ecef5b58b3cdf70e1f0e2e73dd6b16082505a547725844034d7bf39534f'
        coordinator = DirectedReuseCoordinator(no_directed=False)
        coordinator.load_state_dict(ckpt['model']); coordinator.eval()
    c.write_json(out/'launch.json', dict(args=vars(args), horizon=horizon, command=command,
        env={k:v for k,v in env.items() if k.startswith('CUMAC_') or k=='CUDA_VISIBLE_DEVICES'},
        base_sha256=sha(old.CHECKPOINT), checkpoint_sha256=sha(CHECKPOINT),
        simulator_sha256=sha(OUT/'simulator'), runner_sha256=sha(__file__)))
    builder = p.FeatureSnapshotBuilder(p.SnapshotBuilderConfig(tb_err_ema_alpha=.1))
    edges = p.SparseEdgeGenerator(p.EdgeGeneratorConfig(ue_prg_topk=4, ue_prg_diversity_extra=0,
        candidate_mode='all_demand_ues', include_pfq_score_anchor=False, include_candidate_risk_features=False))
    times = {k:[] for k in ['feature','bc_decode','attention','decision_total']}
    reference = None
    if args.reference:
        with np.load(args.reference+'/prefix.npz') as saved:
            reference = {k:saved[k] for k in ['hashes','actions']}
    interference_sum = interference_weighted = 0.
    tx_count = prg_count = blanks = network_calls = 0
    action_digest = hashlib.sha256()
    client = None; started = time.monotonic()
    with (out/'sim.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            client = p.BridgeEnvClient(p.BridgeEnvConfig(socket_path=socket, connect_timeout_s=60, enable_ctbs=True))
            obs,_,info = client.reset(seed=args.seed, episode_horizon=horizon)
            assert info['harq_enabled'] == 0
            config = p.PhyTbsConfig.from_vector(info['ctbs_config']); phy = old.CausalPhy(config)
            for tti in range(1,horizon+1):
                if reference is not None:
                    assert p.digest_arrays(obs) == reference['hashes'][tti-1], (tti,'observation changed')
                feature_us = decode_us = attention_us = 0.
                if actor is not None:
                    t0 = time.perf_counter_ns()
                    graph = edges.build_graph(builder.build(obs))
                    feature_us += (time.perf_counter_ns()-t0)/1000
                # IPC and simulator state export are deliberately outside decision timers.
                ctbs = p._ctbs_inputs(client, info, config)
                snapshot = old.read_snapshot(snapshot_path, ctbs.tti)
                assert ctbs.tti == tti-1
                if actor is None:
                    ue = obs['teacher_action_ue_select'].copy(); prg = obs['teacher_action_prg_alloc'].copy()
                else:
                    t0 = time.perf_counter_ns()
                    packed = old.pack_prg_candidates(graph,max_candidates=old.resolve_full_candidate_width(graph),fixed_width=True)
                    with torch.no_grad():
                        _,decoded = old.select_independent_action(actor,graph,packed,mode='greedy',ctbs_headroom_inputs=ctbs)
                    ue = decoded.action_ue_select.numpy(); prg = decoded.action_prg_alloc.numpy().copy()
                    decode_us = (time.perf_counter_ns()-t0)/1000
                draft = p.decode_type0_teacher_action(ue,prg,n_cell=3,n_prg=17).reshape(3,17)
                chosen = draft; token = -1
                if coordinator is not None and tti > args.warmup:
                    t0 = time.perf_counter_ns()
                    phy.prepare(snapshot,ctbs)
                    features = build_directed_features(draft,ctbs,snapshot['queue'],phy.bank,phy.received,
                        snapshot['noise'],obs['obs_ue_features'],phy.tbs)
                    feature_us += (time.perf_counter_ns()-t0)/1000
                    t0 = time.perf_counter_ns()
                    with torch.no_grad(): scores = coordinator(**{k:torch.from_numpy(v)[None] for k,v in features.items()})[0].numpy()
                    chosen,token = choose_blank(draft,scores,features['valid'])
                    attention_us = (time.perf_counter_ns()-t0)/1000
                    network_calls += 1
                    if token >= 0:
                        cell,local = divmod(token,17); prg[local*3+cell] = -1; blanks += 1
                if args.warmup < tti <= args.warmup+args.measured:
                    for key,value in [('feature',feature_us),('bc_decode',decode_us),('attention',attention_us),
                                      ('decision_total',feature_us+decode_us+attention_us)]:
                        if actor is not None: times[key].append(value)
                    # Only the selected action's interference metric; no MMSE/oracle for PFQ.
                    received = phy.received if coordinator is not None else np.trace(
                        snapshot['h']@snapshot['h'].conj().swapaxes(-1,-2),axis1=-2,axis2=-1).real
                    serving = np.asarray(ctbs.serving_cell)
                    assigned = chosen[serving,:] == np.arange(36)[:,None]
                    count = assigned.sum(axis=1)
                    cross = received*(np.arange(3)[None,None,:] != serving[:,None,None])
                    power = np.einsum('upc,pc->up',cross,(chosen>=0).T)
                    weighted = (power*assigned).sum(axis=1)
                    interference_sum += float((weighted/np.maximum(count,1))[count>0].sum())
                    interference_weighted += float(weighted.sum()); tx_count += int((count>0).sum()); prg_count += int(count.sum())
                if reference is not None:
                    assert np.array_equal(chosen.reshape(-1),reference['actions'][tti-1]), (tti,'action changed')
                action_digest.update(chosen.astype('<i4').tobytes())
                next_obs,_,_,next_info = client.step(ue,prg)
                actual = p._actual_chosen_from_mask(next_info['actual_ue_prg_mask'],np.asarray(ctbs.serving_cell),n_cell=3,n_prg=17)
                assert np.array_equal(actual,chosen.reshape(-1)), (tti,'native action')
                obs,info = next_obs,next_info
                if tti%1000 == 0 or tti == horizon:
                    c.write_json(out/'progress.json',dict(tti=tti,horizon=horizon,elapsed_s=time.monotonic()-started,blanks=blanks))
                    print(out.name,tti,round(time.monotonic()-started,2),flush=True)
        finally:
            if client: client.close()
            try: code = proc.wait(timeout=60)
            except subprocess.TimeoutExpired: proc.terminate(); code = proc.wait(timeout=15)
            assert code == 0, code
    assert network_calls == (horizon-args.warmup if actor is not None else 0)
    pfq = np.fromfile(out/'pfq_core_us.bin',dtype='<f8')
    # Each TTI exports pre-action and post-action teacher; the original runner uses
    # the preceding StepRsp teacher. Keep that exact policy and time its used calls.
    assert len(pfq) == 2*(horizon+1)-1, len(pfq)
    used = np.r_[pfq[0],pfq[1:2*horizon-1:2]]
    stats = {k:timing_summary(v) for k,v in times.items() if v}
    if actor is None:
        stats['pfq_cpu_schedule'] = timing_summary(used[args.warmup:args.warmup+args.measured])
    result = dict(seed=args.seed,arm=args.arm,measured_ttis=args.measured,warmup_ttis=args.warmup,
        followup_ttis=args.followup,horizon=horizon,elapsed_native_s=time.monotonic()-started,
        additional_blanks=blanks,network_calls=network_calls,actions_exact=horizon,
        reference_prefix_exact=horizon if reference is not None else None,action_sha256=action_digest.hexdigest(),
        interference_tx_mean_scaled=interference_sum/tx_count,
        interference_prg_weighted_scaled=interference_weighted/prg_count,timing=stats,
        exogenous_sha256={k:sha(out/k) for k in ['traffic_offered_packets.csv','ue_topology.csv']})
    c.write_json(out/'execution.json',result)
    if args.summarize:
        from .summarize import summarize
        summarize(out)
    print('COMPLETE',out.name,json.dumps(stats),flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('arm',choices=['pfq0','full']); parser.add_argument('--name',required=True)
    parser.add_argument('--seed',type=int,default=43); parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--warmup',type=int,default=4000); parser.add_argument('--measured',type=int,default=20000)
    parser.add_argument('--followup',type=int,default=400); parser.add_argument('--reference',default='')
    parser.add_argument('--summarize',action='store_true')
    run(parser.parse_args())

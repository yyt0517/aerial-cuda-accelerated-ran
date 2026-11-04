"""Fresh-observation native closed loop, using the corrected steady-state simulator.

The simulator and its ABI are reused without changing any historical runner.
Each arm evolves its own queues, CRC outcomes and causal service-deficit EMA.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from .archive import ARCHIVE, FORMAL, enable_legacy_types, problem_from_row, initial_draft
from .experiment import write_json, snapshot_sources, sha, ROOT
from .neural_repair import JointRepair
from .physics import NativePhy, COUNT, LAYER, CAPACITY, DEQUEUE, SINR, BLER
from .quality import select


def native_modules():
    enable_legacy_types()
    sys.path.insert(0, str(ARCHIVE))
    from output.pfq_independent_reuse.formal.boundary08_steady_state_20261021_v1 import run as native
    from output.pfq_independent_reuse.formal.boundary08_corrected_physics_20261020_v1 import fresh_observation
    return native, fresh_observation


def update_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def summarize(out, native, launch):
    import pandas as pd
    from output.pfq_independent_reuse.formal.boundary08_fast_multiseed_20261010_v1.summarize import packets_from_fifo
    fields = native.FIELDS
    data = np.fromfile(out/'metrics.bin', dtype='<f8').reshape(launch['horizon'],36,len(fields))
    service = pd.DataFrame(data.reshape(-1,len(fields)), columns=fields)
    packets = packets_from_fifo(service, pd.read_csv(out/'traffic_offered_packets.csv'),
                               pd.read_csv(out/'packet_outcome_samples.csv'), launch['horizon'])
    first, last = launch['measurement_first_tti'], launch['measurement_last_tti']
    cohort = packets[packets.arrival_tti.between(first,last)]
    window = service[service.tti.between(first,last)]
    rates = window.groupby('ue_id').successful_bytes.sum()*8/(launch['measured']*.0005)/1.e6
    packets.to_csv(out/'packets.csv', index=False)
    row = dict(seed=launch['seed'], arm=launch['arm'], measured_ttis=launch['measured'],
        goodput_mbps=float(rates.sum()), ue_goodput_p5_mbps=float(rates.quantile(.05)),
        ue_goodput_jain=float(rates.sum()**2/(36*(rates**2).sum())),
        dequeue_bytes=int(window.scheduled_dequeue_bytes.sum()),
        failed_bytes=int(window.failed_bytes.sum()), expired_bytes=int(window.expired_bytes.sum()),
        tb_bler=float(window.tb_err_count.sum()/max(1,window.tb_tx_count.sum())),
        cohort_packets=len(cohort), censored_packets=int(cohort.censored.sum()),
        start_backlog_bytes=int(window[window.tti == first].previous_post_queue_bytes.sum()),
        end_backlog_bytes=int(window[window.tti == last].post_queue_bytes.sum()))
    assert row['start_backlog_bytes']+int(window.accepted_bytes.sum())-row['expired_bytes']-row['dequeue_bytes'] == row['end_backlog_bytes']
    if launch['followup'] >= 400:
        assert not cohort.censored.any(), 'All measurement arrivals must mature'
        success = cohort[cohort.successfully_delivered]
        row.update(complete_success_pct=float(100*cohort.successfully_delivered.mean()),
            expiry_pct=float(100*(cohort.outcome == 'expired').mean()),
            full_payload_mbps=float(success.packet_size_bytes.sum()*8/(launch['measured']*.0005)/1.e6),
            success_delay_mean_ms=float(success.system_time_ms.mean()),
            success_delay_p95_ms=float(success.system_time_ms.quantile(.95)),
            outcome_mean_ms=float(cohort.system_time_ms.mean()))
    write_json(out/'metrics.json', row)
    return data, row


def run(args):
    import torch
    import pandas as pd
    native, fresh = native_modules(); r = native.r
    torch.set_num_threads(1)
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    sources = snapshot_sources(out)
    checkpoint = Path(args.checkpoint).resolve()
    decoder = getattr(args, 'decoder', 'reference')
    if decoder in ('coverage','verified','verified_cover') and args.arm not in ('risk_neural','risk_hybrid','risk_residual'):
        raise ValueError('Prefix decoders require an 87-input risk-neural policy')
    if args.arm in ('risk_neural','risk_hybrid','risk_residual'):
        from .deadline import DeadlineRepair
        if decoder in ('coverage','verified','verified_cover'):
            from .prefix_repair import PrefixRepair
            repairer=PrefixRepair(checkpoint,args.width,mode=decoder,
                prefix_budget=getattr(args,'prefix_budget',48),reserve=getattr(args,'prefix_reserve',4))
        elif decoder == 'cached':
            from .cached_repair import CachedDeadlineRepair
            repairer=CachedDeadlineRepair(checkpoint,args.width)
        else:
            repairer=DeadlineRepair(checkpoint,args.width)
    else:
        repairer = JointRepair(checkpoint, args.width) if args.arm in ('neural','hybrid','neural_extra') else None
        if decoder == 'cached' and repairer is not None:
            from .cached_repair import CachedJointRepair
            repairer=CachedJointRepair(checkpoint,args.width)
    anchor=None
    if args.arm in ('risk_residual','neural_extra'):
        if not args.anchor_checkpoint:raise ValueError('Residual policy requires its frozen neural anchor')
        anchor=JointRepair(args.anchor_checkpoint,args.width)
        if decoder == 'cached':
            from .cached_repair import CachedJointRepair
            anchor=CachedJointRepair(args.anchor_checkpoint,args.width)
    template = FORMAL/'boundary08_steady_state_20261021_v1/seed41_pfq0_20000/launch.json'
    old = json.loads(template.read_text())
    env = {k:v for k,v in os.environ.items() if not k.startswith('CUMAC_')}
    env.update(old['env'])
    origin = str(Path(old['env']['CUMAC_DIRECTED_SNAPSHOT_PATH']).parent)
    for key, value in list(env.items()):
        if key.startswith('CUMAC_') and value.startswith(origin):
            env[key] = str(out)+value[len(origin):]
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), CUMAC_ONLINE_SOCKET=f'/tmp/joint_repair_{os.getpid()}.sock',
        CUMAC_TOPOLOGY_SEED=str(args.seed), CUMAC_TRAFFIC_SEED=str(args.seed),
        CUMAC_FRESH_OBSERVATION_PATH=str(out/'fresh_observation.bin'),
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    assert not any(k.startswith('CUMAC_LAST_SLOT_DROP') for k in env)
    simulator = FORMAL/'boundary08_steady_state_20261021_v1/simulator'
    command = [str(simulator)]+old['command'][1:]
    linked = subprocess.check_output(['ldd',str(simulator)],env=env,text=True)
    library = ARCHIVE/'build.corrected_uniform_v1/cuMAC/src/libcumac.so'
    assert 'libcumac.so => '+str(library)+' (' in linked
    horizon = args.warmup+args.settling+args.measured+args.followup
    launch = dict(seed=args.seed, arm=args.arm, horizon=horizon, warmup=args.warmup,
        settling=args.settling, measured=args.measured, followup=args.followup,
        measurement_first_tti=args.warmup+args.settling+1,
        measurement_last_tti=args.warmup+args.settling+args.measured,
        command=command, env={k:v for k,v in env.items() if k.startswith('CUMAC_') or k=='CUDA_VISIBLE_DEVICES'},
        source_sha256=sources, simulator_sha256=sha(simulator), libcumac_sha256=sha(library),
        checkpoint_sha256=sha(checkpoint), checkpoint=str(checkpoint), width=args.width, decoder=decoder,
        prefix_budget=getattr(args,'prefix_budget',48),prefix_reserve=getattr(args,'prefix_reserve',4),
        checkpoint_used_for_inference=repairer is not None,
        anchor_sha256=None if anchor is None else sha(args.anchor_checkpoint),
        common_pfq_warmup=True, fresh_preaction_all_inputs=True,
        time_cutoff_ms=None, model_selection_seeds=getattr(args,'model_selection_seeds','44,45'),
        training_seeds=getattr(args,'training_seeds','41,42,43'),
        target_success_bytes_per_tti=[2400.]*36, proactive_drop=False)
    write_json(out/'launch.json', launch)
    ema = np.zeros(36); decisions=[]; states=[]
    rows = {key:[] for key in ('input_hashes','channel_fingerprints','precoder_fingerprints',
                              'fresh_input_tti','actions','goodput')}
    sample_ttis = {args.warmup, args.warmup+1, launch['measurement_first_tti'],
                   launch['measurement_last_tti'], horizon}
    sample_ttis |= set(range(max(1,args.warmup+1),horizon+1,getattr(args,'sample_interval',500)))
    sample_ttis = {t for t in sample_ttis if 1 <= t <= horizon}
    client = None; started=time.monotonic()
    legacy = builder = edges = oldphy = None
    if args.arm == 'legacy_current':
        from output.pfq_independent_reuse.formal.boundary08_causal_floor_ratio_seed41_20261027_v1 import policy as legacy
        legacy.RATIO_NUM=90
        builder=r.p.FeatureSnapshotBuilder(r.p.SnapshotBuilderConfig(tb_err_ema_alpha=.1))
        edges=r.p.SparseEdgeGenerator(r.p.EdgeGeneratorConfig(ue_prg_topk=4,ue_prg_diversity_extra=0,
            candidate_mode='all_demand_ues',include_pfq_score_anchor=False,include_candidate_risk_features=False))
    with (out/'sim.log').open('w') as log:
        proc=subprocess.Popen(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            client=r.p.BridgeEnvClient(r.p.BridgeEnvConfig(socket_path=env['CUMAC_ONLINE_SOCKET'],
                                                       connect_timeout_s=60,enable_ctbs=True))
            previous,_,info=client.reset(seed=args.seed,episode_horizon=horizon)
            assert info['harq_enabled'] == 0
            config=r.p.PhyTbsConfig.from_vector(info['ctbs_config'])
            if legacy is not None:
                oldphy=r.optimized.OptimizedPhy(config)
            for tti in range(1,horizon+1):
                ctbs=r.p._ctbs_inputs(client,info,config); assert ctbs.tti == tti-1
                obs=fresh.read(out/'fresh_observation.bin',previous,ctbs)
                snapshot=r.historical.old.read_snapshot(out/'pre_action.bin',ctbs.tti)
                np.testing.assert_array_equal(obs['obs_ue_features'][:,0],snapshot['queue'])
                serving=np.asarray(ctbs.serving_cell)
                np.testing.assert_array_equal(obs['action_mask_cell_ue'],np.arange(3)[:,None] == serving[None,:])
                np.testing.assert_array_equal(obs['action_mask_ue'],snapshot['queue']>0)
                rows['input_hashes'].append(r.p.digest_arrays(obs))
                rows['fresh_input_tti'].append(ctbs.tti)
                rows['channel_fingerprints'].append(hashlib.sha256(snapshot['h'].tobytes()).hexdigest())
                rows['precoder_fingerprints'].append(hashlib.sha256(snapshot['w'].tobytes()).hexdigest())
                begin=time.perf_counter()
                counters=('calls','proposals','option_scores','logit_calls')
                models=tuple(m for m in (repairer,anchor) if m is not None)
                neural_before={key:sum(getattr(m,key) for m in models) for key in counters}
                problem=problem_from_row(dict(ctbs=ctbs,obs=obs,snapshot=snapshot,ema=ema))
                record={}
                if tti <= args.warmup or args.arm == 'pfq0':
                    if builder is not None:
                        builder.build(obs)
                    ue=obs['teacher_action_ue_select'].copy();prg=obs['teacher_action_prg_alloc'].copy()
                    chosen=r.p.decode_type0_teacher_action(ue,prg,n_cell=3,n_prg=17).reshape(3,17)
                    physical=NativePhy(problem)
                    try:
                        values=physical.evaluate(chosen).values[0].copy()
                    finally:
                        physical.close()
                elif legacy is not None:
                    draft,keep,ue,prg,*_=legacy.policy(obs,ctbs,snapshot,builder,edges,None,None,oldphy)
                    np.testing.assert_array_equal(draft, initial_draft(problem,obs))
                    chosen,_,record,_=legacy.select(draft,keep,oldphy,obs,ctbs,problem.weights)
                    assert record['converged'] and not record['cap_reached']
                    ue,prg=r.replacement.prior.encode(chosen,obs)
                    physical=NativePhy(problem)
                    try:
                        values=physical.evaluate(chosen).values[0].copy()
                    finally:
                        physical.close()
                else:
                    draft=initial_draft(problem,obs)
                    if anchor is None:
                        chosen,values,record=select(problem,draft,args.arm,repairer)
                    else:
                        from .residual_deadline import select as residual_select
                        chosen,values,record=residual_select(problem,draft,anchor,repairer)
                        record['arm']=args.arm
                    ue,prg=r.replacement.prior.encode(chosen,obs)
                if tti > args.warmup:
                    record.update({f'neural_{key}':sum(getattr(m,key) for m in models)-neural_before[key]
                                   for key in counters})
                    decisions.append(dict(record,tti=tti,decision_wall_ms=(time.perf_counter()-begin)*1000))
                active=values[:,COUNT]>0
                sample=None
                if tti in sample_ttis:
                    sample=dict(tti=tti,obs=obs,ctbs=ctbs,snapshot=snapshot,chosen=chosen.copy(),
                                ema=ema.copy(),values=values.copy())
                rows['actions'].append(chosen.reshape(-1).copy())
                previous,_,_,info=client.step(ue,prg)
                actual=r.p._actual_chosen_from_mask(info['actual_ue_prg_mask'],serving,n_cell=3,n_prg=17)
                np.testing.assert_array_equal(actual,chosen.reshape(-1))
                for predicted,key in ((problem.mcs,'actual_ue_mcs'),(values[:,LAYER],'actual_ue_layer'),
                                      (values[:,CAPACITY],'actual_ue_tbs_capacity_bytes')):
                    np.testing.assert_array_equal(predicted[active],np.asarray(info[key])[active])
                np.testing.assert_array_equal(values[:,DEQUEUE],info['actual_ue_served_bytes'])
                good=np.asarray(info['actual_ue_goodput_bytes']);ema=.995*ema+.005*good
                rows['goodput'].append(float(good.sum()))
                if sample is not None:
                    sample['native_info']=info;states.append(sample)
                    torch.save(states,out/'states.pt')
                if tti%100 == 0 or tti == horizon:
                    update_json(out/'progress.json',dict(tti=tti,horizon=horizon,elapsed_s=time.monotonic()-started))
                    pd.DataFrame(decisions).to_csv(out/'decisions.csv',index=False)
                    print(out.name,tti,round(time.monotonic()-started,2),flush=True)
        finally:
            if client is not None:
                client.close()
            try:
                code=proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.terminate();code=proc.wait(timeout=15)
            if code != 0:
                raise RuntimeError(f'Native simulator exited {code}; see {out}/sim.log')
    np.savez_compressed(out/'prefix.npz',**{k:np.asarray(v) for k,v in rows.items()})
    data,metrics=summarize(out,native,launch)
    np.testing.assert_array_equal(data[:,:,native.FIELDS.index('successful_bytes')].sum(1),rows['goodput'])
    np.testing.assert_array_equal(rows['fresh_input_tti'],np.arange(horizon))
    audits=[]
    for state in states:
        values=state['values'];active=values[:,COUNT]>0;actual=data[state['tti']-1]
        sinr=float(np.abs(values[active,SINR]-actual[active,native.FIELDS.index('phy_sinr_db_cpu')]).max(initial=0))
        bler=float(np.abs(values[active,BLER]-actual[active,native.FIELDS.index('predicted_bler_cpu')]).max(initial=0))
        assert sinr < .005 and bler < .0005, (state['tti'],sinr,bler)
        audits.append(dict(tti=state['tti'],sinr_max_abs_db=sinr,bler_max_abs=bler))
    for relative,digest in sources.items():
        if sha(ROOT/relative) != digest:
            raise RuntimeError(f'Source changed during native run: {relative}')
    write_json(out/'execution.json',dict(passed=True,horizon=horizon,seed=args.seed,arm=args.arm,
        elapsed_s=time.monotonic()-started,all_tti_action_bytes_exact=True,
        causal_fresh_inputs_verified=True,physical_samples=audits,
        neural_calls=(0 if repairer is None else repairer.calls)+(0 if anchor is None else anchor.calls),
        neural_proposals=(0 if repairer is None else repairer.proposals)+(0 if anchor is None else anchor.proposals),
        neural_option_scores=(0 if repairer is None else repairer.option_scores)+(0 if anchor is None else anchor.option_scores),
        neural_logit_calls=(0 if repairer is None else repairer.logit_calls)+(0 if anchor is None else anchor.logit_calls),
        anchor_calls=0 if anchor is None else anchor.calls,metrics=metrics))
    print('NATIVE_DONE',out.name,metrics,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--arm',choices=('pfq0','greedy','neural','hybrid','physical_lns','legacy_current','risk_neural','risk_hybrid','risk_residual','neural_extra','risk_physical_lns','risk_physical_direct'),required=True)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--warmup',type=int,default=2000);parser.add_argument('--settling',type=int,default=1000)
    parser.add_argument('--measured',type=int,default=2000);parser.add_argument('--followup',type=int,default=400)
    parser.add_argument('--width',type=int,default=16);parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--sample-interval',type=int,default=500)
    parser.add_argument('--training-seeds',default='41,42,43');parser.add_argument('--model-selection-seeds',default='44,45')
    parser.add_argument('--anchor-checkpoint')
    parser.add_argument('--decoder',choices=('reference','cached','coverage','verified','verified_cover'),default='reference')
    parser.add_argument('--prefix-budget',type=int,default=48)
    parser.add_argument('--prefix-reserve',type=int,default=4)
    run(parser.parse_args())

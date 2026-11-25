"""Native causal-feature validation and a bounded sustained-coordination pair."""
import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from output.pfq_independent_reuse.formal.boundary08_blank_probe_20261008_v1 import probe as p
from training.stageb_hgraph.eval_checkpoint_online import _load_actor_checkpoint
from training.stageb_hgraph.graph_ops import pack_prg_candidates
from training.stageb_hgraph.independent_policy import select_independent_action, resolve_full_candidate_width
from .causal_phy import CausalPhy, read_snapshot

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
CHECKPOINT = OUT.parent/'boundary08_blank_probe_20261008_v1/bc_diagnostic/balanced_fit_probe/actor_diagnostic_only.pt'
c=p.c

def run(arm, horizon, warmup):
    assert arm in ['validate','validate_blank','baseline','directed']
    out=OUT/arm
    out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    snapshot_path=out/'pre_action.bin'
    socket=f'/tmp/b08direct_{arm}_{os.getpid()}.sock'
    env=c.eval_env(c.environment(),out,42,socket)
    env.update(CUMAC_UE_PLACEMENT_MODE='coop_boundary',CUMAC_TRAFFIC_ARRIVAL_RATE='.8',
        CUMAC_PFQ_DEMAND_CAP_SLACK_BYTES='3000' if arm.startswith('validate') else '0',
        CUMAC_DPA_FEATURE_MODE='dimensionless',CUMAC_DIRECTED_SNAPSHOT_PATH=str(snapshot_path),
        OPENBLAS_NUM_THREADS='1')
    command=[str(OUT/'simulator'),'-d','1','-b','2','-x','0','-f','0','-g','100','-r','3000']
    c.write_json(out/'launch.json',dict(command=command,env={k:v for k,v in env.items() if k.startswith('CUMAC_')},
        arm=arm,horizon=horizon,warmup=warmup,checkpoint_sha256=c.sha(CHECKPOINT),
        simulator_sha256=c.sha(OUT/'simulator'),protocol_sha256=c.sha(OUT/'protocol.json')))
    actor=None
    if not arm.startswith('validate'): actor,_=_load_actor_checkpoint(CHECKPOINT,torch.device('cpu'))
    builder=p.FeatureSnapshotBuilder(p.SnapshotBuilderConfig(tb_err_ema_alpha=.1))
    edges=p.SparseEdgeGenerator(p.EdgeGeneratorConfig(ue_prg_topk=4,ue_prg_diversity_extra=0,
        candidate_mode='all_demand_ues',include_pfq_score_anchor=False,include_candidate_risk_features=False))
    fields=['tti','ue_id','count','layer','mcs','capacity','dequeue','sinr_db','bler','expected','interference']
    decision_fields=['tti','blank_token','predicted_net_gain','candidate_count','base_expected','executed_expected','snapshot_tti',
        'predicted_source_cell_gain','predicted_victim_gain']
    client=None; started=time.monotonic(); blank_count=0; hashes=[]; actions_log=[]
    reference=None
    if arm == 'validate_blank':
        reference=np.load(OUT/'validate/prefix.npz')
    elif arm == 'directed' and (OUT/'baseline/prefix.npz').exists():
        reference=np.load(OUT/'baseline/prefix.npz')
    with (out/'sim.log').open('w') as log, (out/'predictions.csv').open('w') as prediction_file, (out/'decisions.csv').open('w') as decision_file:
        pw=csv.DictWriter(prediction_file,fieldnames=fields);pw.writeheader()
        dw=csv.DictWriter(decision_file,fieldnames=decision_fields);dw.writeheader()
        proc=subprocess.Popen(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            client=p.BridgeEnvClient(p.BridgeEnvConfig(socket_path=socket,connect_timeout_s=60,enable_ctbs=True))
            obs,_,info=client.reset(seed=42,episode_horizon=horizon)
            assert info['harq_enabled']==0
            config=p.PhyTbsConfig.from_vector(info['ctbs_config'])
            phy=CausalPhy(config)
            for tti in range(1,horizon+1):
                digest=p.digest_arrays(obs);hashes.append(digest)
                if reference is not None and (tti<=warmup or arm=='validate_blank'):
                    assert digest==reference['hashes'][tti-1],(arm,tti,'different prefix observation')
                graph=edges.build_graph(builder.build(obs))
                ctbs=p._ctbs_inputs(client,info,config)
                snapshot=read_snapshot(snapshot_path,ctbs.tti)
                assert ctbs.tti == tti-1
                if actor is None:
                    ue=obs['teacher_action_ue_select'].copy();prg=obs['teacher_action_prg_alloc'].copy()
                else:
                    packed=pack_prg_candidates(graph,max_candidates=resolve_full_candidate_width(graph),fixed_width=True)
                    with torch.no_grad():
                        trace,decoded=select_independent_action(actor,graph,packed,mode='greedy',ctbs_headroom_inputs=ctbs)
                    ue=decoded.action_ue_select.numpy();prg=decoded.action_prg_alloc.numpy().copy()
                chosen=p.decode_type0_teacher_action(ue,prg,n_cell=3,n_prg=17).reshape(3,17)
                # Compute every TTI during validation; for long runs, prediction
                # begins at the shared intervention boundary, with sparse warmup checks.
                calculate=actor is None or tti>warmup or tti%1000==0
                selected=0;token=-1;gain=0.;candidate_count=0
                if calculate:
                    phy.prepare(snapshot,ctbs)
                    candidates,pred,gains,tokens,best=phy.candidates(chosen)
                    candidate_count=len(tokens)
                    if arm=='directed' and tti>warmup:
                        selected=best
                    if arm=='validate_blank' and tti==horizon:
                        target=9  # known TTI1101 / cell0 PRG9
                        selected=int(np.flatnonzero(tokens==target)[0])+1
                    if selected:
                        token=int(tokens[selected-1]);gain=float(gains[selected]);blank_count+=1
                        cell,local=divmod(token,17)
                        prg[local*3+cell]=-1
                        chosen=candidates[selected]
                    for u in range(36):
                        pw.writerow(dict(tti=tti,ue_id=u,**{key:pred[key][selected,u] for key in fields[2:]}))
                    ue_gain=pred['expected'][selected]-pred['expected'][0]
                    source_gain=float(ue_gain[phy.serving==token//17].sum()) if selected else 0.
                    victim_gain=float(ue_gain[phy.serving!=token//17].sum()) if selected else 0.
                    assert abs(source_gain+victim_gain-gain)<1.e-6
                    dw.writerow(dict(tti=tti,blank_token=token,predicted_net_gain=gain,candidate_count=candidate_count,
                        base_expected=float(pred['expected'][0].sum()),executed_expected=float(pred['expected'][selected].sum()),snapshot_tti=ctbs.tti,
                        predicted_source_cell_gain=source_gain,predicted_victim_gain=victim_gain))
                    if tti in [1,1101,warmup+1,warmup+1000,warmup+4000]:
                        np.savez_compressed(out/f'state_{tti}.npz',**snapshot,features=phy.directed_features(),
                            serving=phy.serving,chosen=chosen,base_chosen=candidates[0],bank=phy.bank,
                            candidate_gains=gains,candidate_tokens=tokens)
                        c.write_json(out/f'ctbs_{tti}.json',asdict(ctbs))
                actions_log.append(chosen.reshape(-1).copy())
                obs_next,_,_,ni=client.step(ue,prg)
                actual=p._actual_chosen_from_mask(ni['actual_ue_prg_mask'],np.asarray(ctbs.serving_cell),n_cell=3,n_prg=17)
                assert np.array_equal(actual,chosen.reshape(-1)),(arm,tti,'action mismatch')
                if calculate:
                    for key,label in [('mcs','actual_ue_mcs'),('layer','actual_ue_layer'),('capacity','actual_ue_tbs_capacity_bytes')]:
                        active=pred['count'][selected]>0
                        assert np.array_equal(pred[key][selected,active],ni[label][active]),(arm,tti,key)
                    assert np.array_equal(pred['dequeue'][selected],ni['actual_ue_served_bytes']),(arm,tti,'queue time or TBS mismatch')
                obs,info=obs_next,ni
                if tti%250==0 or tti==horizon:
                    prediction_file.flush();decision_file.flush()
                    print(f'{arm}: {tti}/{horizon}, added_blanks={blank_count}, elapsed={time.monotonic()-started:.1f}s',flush=True)
        finally:
            if client: client.close()
            try: code=proc.wait(timeout=60)
            except subprocess.TimeoutExpired: proc.terminate();code=proc.wait(timeout=15)
            assert code==0,(arm,code)
    np.savez_compressed(out/'prefix.npz',hashes=np.asarray(hashes),actions=np.asarray(actions_log))
    service=pd.read_csv(out/'service_balance_trace.csv')
    prediction=pd.read_csv(out/'predictions.csv')
    merged=prediction.merge(service,on=['tti','ue_id'],validate='one_to_one')
    active=merged['count']>0
    sinr_err=float(np.max(np.abs(merged.loc[active,'sinr_db']-merged.loc[active,'phy_sinr_db_cpu'])))
    bler_err=float(np.max(np.abs(merged.loc[active,'bler']-merged.loc[active,'predicted_bler_cpu'])))
    assert sinr_err<.005,(arm,'SINR parity',sinr_err)
    assert bler_err<.0005,(arm,'BLER parity',bler_err)
    assert np.array_equal(merged.dequeue,merged.scheduled_dequeue_bytes)
    if arm.startswith('validate'):
        old=pd.read_csv(OUT.parent/'boundary08_blank_probe_20261008_v1/reference/service_balance_trace.csv')
        limit=horizon if arm=='validate' else horizon-1
        pd.testing.assert_frame_equal(service[service.tti<=limit].reset_index(drop=True),old[old.tti<=limit].reset_index(drop=True),check_exact=True)
    elif arm=='baseline':
        old=pd.read_csv(OUT.parent/'boundary08_local_explore_20261008_v1/evaluation/bcfit512_seed42/service_balance_trace.csv')
        limit=min(horizon,5000)
        pd.testing.assert_frame_equal(service[service.tti<=limit].reset_index(drop=True),old[old.tti<=limit].reset_index(drop=True),check_exact=True)
    c.write_json(out/'verification.json',dict(passed=True,horizon=horizon,warmup=warmup,additional_blanks=blank_count,
        feature_capture='pre-action current estimated CSI and current queue; CTBS tti-bound',native_action_checks=horizon,
        predicted_ue_ttis=len(merged),sinr_max_abs_error_db=sinr_err,bler_max_abs_error=bler_err,
        mcs_layer_tbs_queue_exact=True,old_prefix_exact=arm!='directed',elapsed_s=time.monotonic()-started))
    print('VERIFIED',arm,sinr_err,bler_err,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('arm');parser.add_argument('--horizon',type=int,default=8400);parser.add_argument('--warmup',type=int,default=4000)
    args=parser.parse_args();run(args.arm,args.horizon,args.warmup)

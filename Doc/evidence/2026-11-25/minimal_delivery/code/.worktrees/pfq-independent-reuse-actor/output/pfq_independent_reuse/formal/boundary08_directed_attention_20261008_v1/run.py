"""Collect pre-action labels or execute a frozen learned coordinator natively."""
import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from training.stageb_hgraph.directed_reuse import build_directed_features,DirectedReuseCoordinator,choose_blank
from output.pfq_independent_reuse.formal.boundary08_directed_coordination_20261008_v1 import run_probe as old

OUT=Path(__file__).resolve().parent
p=old.p;c=old.c

def run(args):
    out=OUT/(args.name or args.arm);out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    snapshot_path=out/'pre_action.bin';socket=f'/tmp/b08learn_{os.getpid()}.sock'
    env=c.eval_env(c.environment(),out,args.seed,socket)
    env.update(CUMAC_UE_PLACEMENT_MODE='coop_boundary',CUMAC_TRAFFIC_ARRIVAL_RATE='.8',
        CUMAC_PFQ_DEMAND_CAP_SLACK_BYTES='3000' if args.arm=='pfq' else '0',
        CUMAC_DPA_FEATURE_MODE='dimensionless',CUMAC_DIRECTED_SNAPSHOT_PATH=str(snapshot_path),
        OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=str(args.gpu))
    command=[str(old.OUT/'simulator'),'-d','1','-b','2','-x','0','-f','0','-g','100','-r','3000']
    actor=None
    if args.arm!='pfq': actor,_=old._load_actor_checkpoint(old.CHECKPOINT,torch.device('cpu'))
    coordinator=None
    if args.arm in ['full','no_directed']:
        ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        assert ckpt['base_checkpoint_sha256']==c.sha(old.CHECKPOINT)
        assert ckpt['network_source_sha256']==c.sha(OUT.parents[3]/'training/stageb_hgraph/directed_reuse.py')
        assert ckpt['no_directed']==(args.arm=='no_directed')
        coordinator=DirectedReuseCoordinator(no_directed=ckpt['no_directed'])
        coordinator.load_state_dict(ckpt['model']);coordinator.eval()
    c.write_json(out/'launch.json',dict(args=vars(args),command=command,
        env={k:v for k,v in env.items() if k.startswith('CUMAC_') or k=='CUDA_VISIBLE_DEVICES'},
        base_checkpoint_sha256=c.sha(old.CHECKPOINT),coordinator_sha256=c.sha(args.checkpoint) if coordinator else None,
        simulator_sha256=c.sha(old.OUT/'simulator'),protocol_sha256=c.sha(OUT/'protocol.md'),
        network_source_sha256=c.sha(OUT.parents[3]/'training/stageb_hgraph/directed_reuse.py')))
    builder=p.FeatureSnapshotBuilder(p.SnapshotBuilderConfig(tb_err_ema_alpha=.1))
    graph_edges=p.SparseEdgeGenerator(p.EdgeGeneratorConfig(ue_prg_topk=4,ue_prg_diversity_extra=0,
        candidate_mode='all_demand_ues',include_pfq_score_anchor=False,include_candidate_risk_features=False))
    feature_rows=[];label_rows=[];identities=[];hashes=[];action_rows=[]
    fields=['tti','ue_id','count','layer','mcs','capacity','dequeue','sinr_db','bler','expected','interference']
    decision_fields=['tti','token','predicted_gain_bytes','oracle_chosen_gain_bytes','oracle_best_gain_bytes','oracle_best_token',
        'oracle_source_gain_bytes','oracle_victim_gain_bytes']
    selected_count=0;client=None;started=time.monotonic();native_checks=0;network_calls=0
    with (out/'sim.log').open('w') as log,(out/'predictions.csv').open('w') as pf,(out/'decisions.csv').open('w') as df:
        pw=csv.DictWriter(pf,fieldnames=fields);pw.writeheader()
        dw=csv.DictWriter(df,fieldnames=decision_fields);dw.writeheader()
        proc=subprocess.Popen(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            client=p.BridgeEnvClient(p.BridgeEnvConfig(socket_path=socket,connect_timeout_s=60,enable_ctbs=True))
            obs,_,info=client.reset(seed=args.seed,episode_horizon=args.horizon)
            assert info['harq_enabled']==0
            config=p.PhyTbsConfig.from_vector(info['ctbs_config']);phy=old.CausalPhy(config)
            for tti in range(1,args.horizon+1):
                hashes.append(p.digest_arrays(obs))
                graph=graph_edges.build_graph(builder.build(obs))
                ctbs=p._ctbs_inputs(client,info,config);snapshot=old.read_snapshot(snapshot_path,ctbs.tti)
                assert ctbs.tti==tti-1
                if actor is None:
                    ue=obs['teacher_action_ue_select'].copy();prg=obs['teacher_action_prg_alloc'].copy()
                else:
                    packed=old.pack_prg_candidates(graph,max_candidates=old.resolve_full_candidate_width(graph),fixed_width=True)
                    with torch.no_grad():
                        _,decoded=old.select_independent_action(actor,graph,packed,mode='greedy',ctbs_headroom_inputs=ctbs)
                    ue=decoded.action_ue_select.numpy();prg=decoded.action_prg_alloc.numpy().copy()
                draft=p.decode_type0_teacher_action(ue,prg,n_cell=3,n_prg=17).reshape(3,17)
                chosen=draft.copy();token=-1;model_gain=0.
                calculate=tti>args.warmup or tti%1000==0 or tti==1
                if calculate:
                    phy.prepare(snapshot,ctbs)
                    if args.arm=='collect' or coordinator is not None:
                        features=build_directed_features(draft,ctbs,snapshot['queue'],phy.bank,phy.received,
                            snapshot['noise'],obs['obs_ue_features'],phy.tbs)
                    if coordinator is not None and tti>args.warmup:
                        inputs={k:torch.from_numpy(v)[None] for k,v in features.items()}
                        with torch.no_grad(): scores=coordinator(**inputs)[0].numpy()
                        network_calls+=1
                        chosen,token=choose_blank(draft,scores,features['valid'])
                        model_gain=float(scores[token+1]*3000) if token>=0 else 0.
                    # Oracle is evaluated AFTER the learned action is fixed.
                    # It provides collection labels / audit only, never overrides NN.
                    if args.arm=='collect' or coordinator is not None:
                        candidates,pred,gains,tokens,best=phy.candidates(draft)
                        if args.arm=='collect' and tti>args.warmup:
                            labels=np.zeros(52,dtype=np.float32);labels[tokens+1]=gains[1:]/3000
                            feature_rows.append(features);label_rows.append(labels);identities.append(tti)
                            chosen=candidates[best].copy();token=int(tokens[best-1]) if best else -1
                            model_gain=float(gains[best])
                        selected=int(np.flatnonzero(tokens==token)[0])+1 if token>=0 else 0
                        assert np.array_equal(chosen,candidates[selected])
                        chosen_prediction={k:v[selected] for k,v in pred.items()}
                        per_ue_gain=pred['expected'][selected]-pred['expected'][0]
                        source_gain=float(per_ue_gain[phy.serving==token//17].sum()) if token>=0 else 0.
                        victim_gain=float(per_ue_gain[phy.serving!=token//17].sum()) if token>=0 else 0.
                        dw.writerow(dict(tti=tti,token=token,predicted_gain_bytes=model_gain,
                            oracle_chosen_gain_bytes=float(gains[selected]),oracle_best_gain_bytes=float(gains[best]),
                            oracle_best_token=int(tokens[best-1]) if best else -1,
                            oracle_source_gain_bytes=source_gain,oracle_victim_gain_bytes=victim_gain))
                    else:
                        chosen_prediction={k:v[0] for k,v in phy.evaluate(chosen).items()}
                    for u in range(36):pw.writerow(dict(tti=tti,ue_id=u,**{k:chosen_prediction[k][u] for k in fields[2:]}))
                    if tti in [1,args.warmup+1,args.warmup+1000,args.horizon-400]:
                        c.write_json(out/f'ctbs_{tti}.json',asdict(ctbs))
                if token>=0:
                    cell,local=divmod(token,17);prg[local*3+cell]=-1;selected_count+=1
                assert (chosen!=draft).sum() in [0,1]
                action_rows.append(chosen.reshape(-1).copy())
                next_obs,_,_,next_info=client.step(ue,prg)
                actual=p._actual_chosen_from_mask(next_info['actual_ue_prg_mask'],np.asarray(ctbs.serving_cell),n_cell=3,n_prg=17)
                assert np.array_equal(actual,chosen.reshape(-1)),(tti,'native action')
                if calculate:
                    active=chosen_prediction['count']>0
                    for key,label in [('mcs','actual_ue_mcs'),('layer','actual_ue_layer'),('capacity','actual_ue_tbs_capacity_bytes')]:
                        assert np.array_equal(chosen_prediction[key][active],next_info[label][active]),(tti,key)
                    assert np.array_equal(chosen_prediction['dequeue'],next_info['actual_ue_served_bytes']),(tti,'queue')
                    native_checks+=1
                obs,info=next_obs,next_info
                if tti%250==0 or tti==args.horizon:
                    pf.flush();df.flush()
                    print(f'{out.name}: {tti}/{args.horizon}, blanks={selected_count}, samples={len(feature_rows)}, elapsed={time.monotonic()-started:.1f}s',flush=True)
        finally:
            if client:client.close()
            try: code=proc.wait(timeout=60)
            except subprocess.TimeoutExpired: proc.terminate();code=proc.wait(timeout=15)
            assert code==0,code
    np.savez_compressed(out/'prefix.npz',hashes=np.asarray(hashes),actions=np.asarray(action_rows))
    if feature_rows:
        data={key:np.stack([f[key] for f in feature_rows]) for key in feature_rows[0]}
        data.update(labels=np.stack(label_rows),tti=np.asarray(identities),seed=np.full(len(identities),args.seed))
        np.savez_compressed(out/'dataset.npz',**data)
    service=pd.read_csv(out/'service_balance_trace.csv');prediction=pd.read_csv(out/'predictions.csv')
    merged=prediction.merge(service,on=['tti','ue_id'],validate='one_to_one');active=merged['count']>0
    sinr_error=float(np.max(np.abs(merged.loc[active,'sinr_db']-merged.loc[active,'phy_sinr_db_cpu'])))
    bler_error=float(np.max(np.abs(merged.loc[active,'bler']-merged.loc[active,'predicted_bler_cpu'])))
    from .precision import audit
    c.write_json(out/'precision_audit.json',audit(merged))
    assert np.array_equal(merged.dequeue,merged.scheduled_dequeue_bytes)
    c.write_json(out/'verification.json',dict(passed=True,arm=args.arm,seed=args.seed,horizon=args.horizon,warmup=args.warmup,
        validity_scope='Native action/CTBS/dequeue accounting; original approximate-PHY precision gate outcomes remain separate in precision_audit.json',
        actions_exact=args.horizon,phy_checked_ttis=native_checks,additional_blanks=selected_count,
        sinr_max_abs_error_db=sinr_error,bler_max_abs_error=bler_error,ctbs_and_dequeue_exact=True,
        collected_states=len(feature_rows),elapsed_s=time.monotonic()-started))
    if coordinator is not None: assert network_calls==args.horizon-args.warmup
    c.write_json(out/'network_execution.json',dict(forward_calls=network_calls,
        decision_order='Neural scores and action fixed before oracle audit; no oracle override',
        checkpoint_sha256=c.sha(args.checkpoint) if coordinator else None))
    print('VERIFIED',out.name,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('arm',choices=['collect','bc','pfq','full','no_directed'])
    parser.add_argument('--name');parser.add_argument('--seed',type=int,default=43)
    parser.add_argument('--horizon',type=int,default=24400);parser.add_argument('--warmup',type=int,default=4000)
    parser.add_argument('--checkpoint',default='');parser.add_argument('--gpu',type=int,default=1)
    run(parser.parse_args())

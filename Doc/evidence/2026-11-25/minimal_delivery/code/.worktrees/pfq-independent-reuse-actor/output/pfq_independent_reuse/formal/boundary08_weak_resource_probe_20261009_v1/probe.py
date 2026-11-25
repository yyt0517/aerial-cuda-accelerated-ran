"""Disposable exact-prefix, one-PRG resource-repair feasibility experiment."""
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
from training.gnnrl.slot_layout import build_type0_slot_layout
from training.stageb_hgraph.directed_reuse import build_directed_features,DirectedReuseCoordinator,choose_blank
from output.pfq_independent_reuse.formal.boundary08_directed_attention_20261008_v1 import run as learned
from output.pfq_independent_reuse.formal.boundary08_directed_attention_20261008_v1.precision import audit

OUT=Path(__file__).resolve().parent
PREVIOUS=learned.OUT
old=learned.old;p=old.p;c=old.c
CASES=[(4001,15),(4101,32),(4201,35)]
CHECKPOINT=PREVIOUS/'training/full.pt'

def candidate_actions(base,serving,target):
    base=np.asarray(base).reshape(3,17)
    actions=[base.copy()];metadata=[dict(arm='keep',token=-1,donor=-1)]
    for token in np.flatnonzero(base.reshape(-1)>=0):
        a=base.copy();donor=int(a.reshape(-1)[token]);a.reshape(-1)[token]=-1
        actions.append(a);metadata.append(dict(arm='blank',token=int(token),donor=donor))
    cell=int(serving[target])
    for prg in range(17):
        donor=int(base[cell,prg])
        if donor<0 or donor==target:continue
        a=base.copy();a[cell,prg]=target
        actions.append(a);metadata.append(dict(arm='swap',token=cell*17+prg,donor=donor))
    return np.stack(actions),metadata

def select_index(indices,weak,net):
    indices=np.asarray(indices)
    positive=indices[weak[indices]>1.]
    joint=positive[net[positive]>=0.]
    if len(joint):return int(max(joint,key=lambda i:(weak[i],net[i],-i)))
    if len(positive):return int(max(positive,key=lambda i:(net[i],weak[i],-i)))
    return int(max(indices,key=lambda i:(weak[i],net[i],-i)))

def encode(chosen,obs,reference_slots=None):
    chosen=np.asarray(chosen).reshape(3,17)
    cell_mask=np.asarray(obs['action_mask_cell_ue'],dtype=bool)
    ue_mask=np.asarray(obs['action_mask_ue'],dtype=bool)
    prg_mask=np.asarray(obs['action_mask_prg_cell'],dtype=bool)
    slots=len(obs['teacher_action_ue_select'])
    layout=build_type0_slot_layout(torch.from_numpy(cell_mask),slots,torch.from_numpy(ue_mask))
    slot_cell=layout.slot_to_cell[0].numpy();valid=layout.slot_valid_mask[0].numpy()
    ue=np.full(slots,-1,dtype=np.int32);prg=np.full(51,-1,dtype=np.int16)
    if reference_slots is not None:
        for u,s in enumerate(reference_slots):
            if s>=0:
                cell=int(np.flatnonzero(cell_mask[:,u])[0])
                assert ue_mask[u] and valid[s] and slot_cell[s]==cell and ue[s]==-1
                ue[s]=u
    for cell in range(3):
        ids=sorted(set(int(x) for x in chosen[cell] if x>=0))
        available=np.flatnonzero(valid&(slot_cell==cell))
        assert len(ids)<=len(available)
        mapping={u:int(reference_slots[u]) if reference_slots is not None else int(available[i]) for i,u in enumerate(ids)}
        assert len(set(mapping.values()))==len(mapping)
        for u,s in mapping.items():
            assert cell_mask[cell,u] and ue_mask[u] and 0<=s<slots and valid[s] and slot_cell[s]==cell
            ue[s]=u
        for local,u in enumerate(chosen[cell]):
            if u>=0:
                assert prg_mask[cell,local]
                prg[local*3+cell]=mapping[int(u)]
    assert np.array_equal(p.decode_type0_teacher_action(ue,prg,n_cell=3,n_prg=17).reshape(3,17),chosen)
    return ue,prg

def verify_saved_capture():
    """Retain the failed raw metadata gate; validate saved physical root states."""
    out=OUT/'capture';total=4201
    saved=np.load(out/'prefix.npz');reference=np.load(PREVIOUS/'full/prefix.npz')
    for key in ['hashes','actions']:assert np.array_equal(saved[key],reference[key][:total])
    a=pd.read_csv(out/'service_balance_trace.csv');b=pd.read_csv(PREVIOUS/'full/service_balance_trace.csv',nrows=total*36)
    cols=[k for k in a if k not in ['layer','sched_slot']]
    pd.testing.assert_frame_equal(a[cols],b[cols],check_exact=True)
    differing=(a.layer!=b.layer)|(a.sched_slot!=b.sched_slot)
    assert differing.sum()==23
    assert ((a.actual_prg_count[differing]==0)&(b.actual_prg_count[differing]==0)).all()
    assert ((a.layer[differing]==-1)&(b.layer[differing]==255)&(a.sched_slot[differing]==-1)&(b.sched_slot[differing]>=0)).all()
    for file,key in [('traffic_offered_packets.csv','arrival_tti'),('service_balance_trace.csv.channel.csv','tti')]:
        x=pd.read_csv(out/file);y=pd.read_csv(PREVIOUS/'full'/file)
        pd.testing.assert_frame_equal(x[x[key]<=total].reset_index(drop=True),y[y[key]<=total].reset_index(drop=True),check_exact=True)
    assert c.sha(out/'ue_topology.csv')==c.sha(PREVIOUS/'full/ue_topology.csv')
    pred=pd.read_csv(out/'predictions.csv');assert len(pred)==3*36
    merged=pred.merge(a,on=['tti','ue_id'],validate='one_to_one');active=merged['count']>0
    for x,y in [('count','actual_prg_count'),('mcs_x','mcs_y'),('layer_x','layer_y'),('capacity','tbs_capacity_bytes')]:
        assert np.array_equal(merged.loc[active,x],merged.loc[active,y])
    assert np.array_equal(merged.dequeue,merged.scheduled_dequeue_bytes)
    c.write_json(out/'precision_audit.json',audit(merged))
    selections=[]
    for t,u in CASES:
        root=json.loads((OUT/f'root_t{t}_u{u}.json').read_text())
        state=torch.load(OUT/f'state_t{t}_u{u}.pt',weights_only=False)
        assert root['observation_digest']==saved['hashes'][t-1]
        assert np.array_equal(state['neural'].reshape(-1),saved['actions'][t-1])
        selections.extend(root['selections'])
    c.write_json(OUT/'selected.json',selections)
    c.write_json(out/'verification.json',dict(passed_native_execution_and_business_accounting=True,
        original_full_service_trace_gate_passed=False,inactive_slot_layer_only_differences=23,
        exact_observation_and_native_action_prefix_ttis=total,exact_business_trace_ttis=total,
        exogenous_inputs_exact=True,nn_forward_calls=3,root_ctbs_dequeue_exact=True,
        reason='Replay omitted selected UE slots when their last PRG was blanked; only zero-TX slot/layer=255 metadata differed. All observations/actions/other trace columns exact. Original failure retained. Encoder corrected before continuations; each continuation requires full raw service prefix equality.'))
    print('VERIFIED SAVED CAPTURE; original inactive metadata mismatch retained',flush=True)

def policy(obs,ctbs,snapshot,builder,edges,actor,coordinator,phy):
    graph=edges.build_graph(builder.build(obs))
    packed=old.pack_prg_candidates(graph,max_candidates=old.resolve_full_candidate_width(graph),fixed_width=True)
    with torch.no_grad():
        _,decoded=old.select_independent_action(actor,graph,packed,mode='greedy',ctbs_headroom_inputs=ctbs)
    ue=decoded.action_ue_select.numpy();prg=decoded.action_prg_alloc.numpy().copy()
    draft=p.decode_type0_teacher_action(ue,prg,n_cell=3,n_prg=17).reshape(3,17)
    phy.prepare(snapshot,ctbs)
    features=build_directed_features(draft,ctbs,snapshot['queue'],phy.bank,phy.received,snapshot['noise'],obs['obs_ue_features'],phy.tbs)
    with torch.no_grad():scores=coordinator(**{k:torch.from_numpy(v)[None] for k,v in features.items()})[0].numpy()
    chosen,token=choose_blank(draft,scores,features['valid'])
    if token>=0:
        cell,local=divmod(token,17);prg[local*3+cell]=-1
    return draft,chosen,ue,prg,scores,token

def capture_candidates(tti,target,obs,ctbs,snapshot,draft,neural,scores,phy):
    root=OUT/f'state_t{tti}_u{target}.pt'
    torch.save(dict(obs=obs,ctbs=ctbs,snapshot=snapshot,draft=draft,neural=neural,scores=scores),root)
    actions,metadata=candidate_actions(draft,phy.serving,target)
    prediction=phy.evaluate(actions)
    gain=prediction['expected']-prediction['expected'][0]
    weak=gain[:,target];net=gain.sum(axis=1)
    rows=[]
    for i,m in enumerate(metadata):
        rows.append(dict(tti=tti,ue=target,index=i,**m,weak_expected_gain_bytes=float(weak[i]),
            network_expected_gain_bytes=float(net[i]),weak_count=int(prediction['count'][i,target]),
            weak_mcs=int(prediction['mcs'][i,target]),weak_layer=int(prediction['layer'][i,target]),
            weak_sinr_db=float(prediction['sinr_db'][i,target]),weak_expected_bytes=float(prediction['expected'][i,target])))
    pd.DataFrame(rows).to_csv(OUT/f'candidates_t{tti}_u{target}.csv',index=False)
    selections=[]
    for arm in ['keep','blank','swap']:
        idx=0 if arm=='keep' else select_index([i for i,m in enumerate(metadata) if m['arm']==arm],weak,net)
        selected=dict(rows[idx],name=f't{tti}_u{target}_{arm}',chosen=actions[idx].tolist(),
            has_joint_candidate=bool(any(weak[i]>1 and net[i]>=0 for i,m in enumerate(metadata) if m['arm']==arm)),
            pre_queue_bytes=float(snapshot['queue'][target]),pre_observation_digest=p.digest_arrays(obs))
        selections.append(selected)
    neural_indices=[i for i,a in enumerate(actions) if np.array_equal(a,neural)]
    assert len(neural_indices)==1
    c.write_json(OUT/f'root_t{tti}_u{target}.json',dict(tti=tti,ue=target,
        observation_digest=p.digest_arrays(obs),snapshot_digest=p.digest_arrays(snapshot),ctbs=asdict(ctbs),
        neural_candidate=rows[neural_indices[0]],selections=selections))
    return selections

def run(args):
    capture=args.mode=='capture'
    spec=None
    if not capture:
        spec=next(s for s in json.loads((OUT/'selected.json').read_text()) if s['name']==args.name)
    first=spec['tti'] if spec else max(t for t,u in CASES)
    horizon=first+(0 if capture else 799)
    out=OUT/('capture' if capture else args.name);out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    prefix=np.load(PREVIOUS/'full/prefix.npz')
    reference_service=pd.read_csv(PREVIOUS/'full/service_balance_trace.csv',nrows=horizon*36)
    reference_slots=reference_service.sched_slot.to_numpy().reshape(horizon,36)
    env=c.eval_env(c.environment(),out,43,f'/tmp/b08weak_{os.getpid()}.sock')
    snapshot_path=out/'pre_action.bin'
    env.update(CUMAC_UE_PLACEMENT_MODE='coop_boundary',CUMAC_TRAFFIC_ARRIVAL_RATE='.8',
        CUMAC_PFQ_DEMAND_CAP_SLACK_BYTES='0',CUMAC_DPA_FEATURE_MODE='dimensionless',
        CUMAC_DIRECTED_SNAPSHOT_PATH=str(snapshot_path),CUDA_VISIBLE_DEVICES=str(args.gpu),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[str(old.OUT/'simulator'),'-d','1','-b','2','-x','0','-f','0','-g','100','-r','3000']
    ckpt=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    assert ckpt['base_checkpoint_sha256']==c.sha(old.CHECKPOINT)
    assert ckpt['network_source_sha256']==c.sha(old.ROOT/'training/stageb_hgraph/directed_reuse.py')
    actor,_=old._load_actor_checkpoint(old.CHECKPOINT,torch.device('cpu'))
    coordinator=DirectedReuseCoordinator();coordinator.load_state_dict(ckpt['model']);coordinator.eval()
    builder=p.FeatureSnapshotBuilder(p.SnapshotBuilderConfig(tb_err_ema_alpha=.1))
    edges=p.SparseEdgeGenerator(p.EdgeGeneratorConfig(ue_prg_topk=4,ue_prg_diversity_extra=0,
        candidate_mode='all_demand_ues',include_pfq_score_anchor=False,include_candidate_risk_features=False))
    c.write_json(out/'launch.json',dict(args=vars(args),horizon=horizon,first=first,command=command,
        env={k:v for k,v in env.items() if k.startswith('CUMAC_') or k=='CUDA_VISIBLE_DEVICES'},
        protocol_sha256=c.sha(OUT/'protocol.json'),coordinator_sha256=c.sha(CHECKPOINT),
        simulator_sha256=c.sha(old.OUT/'simulator'),intervention=spec))
    fields=['tti','ue_id','count','layer','mcs','capacity','dequeue','sinr_db','bler','expected','interference']
    hashes=[];actions=[];selections=[];forward_calls=0;client=None;started=time.monotonic()
    with (out/'sim.log').open('w') as log,(out/'predictions.csv').open('w') as pred_file,(out/'decisions.csv').open('w') as decision_file:
        pw=csv.DictWriter(pred_file,fieldnames=fields);pw.writeheader()
        dw=csv.DictWriter(decision_file,fieldnames=['tti','token','intervention']);dw.writeheader()
        proc=subprocess.Popen(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            client=p.BridgeEnvClient(p.BridgeEnvConfig(socket_path=env['CUMAC_ONLINE_SOCKET'],connect_timeout_s=60,enable_ctbs=True))
            obs,_,info=client.reset(seed=43,episode_horizon=horizon)
            assert info['harq_enabled']==0
            config=p.PhyTbsConfig.from_vector(info['ctbs_config']);phy=old.CausalPhy(config)
            for tti in range(1,horizon+1):
                digest=p.digest_arrays(obs);hashes.append(digest)
                if capture or tti<=first:assert digest==prefix['hashes'][tti-1],(tti,'state prefix')
                at_root=capture and tti in [t for t,u in CASES]
                calculate=at_root or (not capture and tti>=first)
                ctbs=p._ctbs_inputs(client,info,config);assert ctbs.tti==tti-1
                if calculate:
                    snapshot=old.read_snapshot(snapshot_path,ctbs.tti)
                    draft,chosen,ue,prg,scores,token=policy(obs,ctbs,snapshot,builder,edges,actor,coordinator,phy)
                    forward_calls+=1
                    if at_root:
                        assert np.array_equal(chosen.reshape(-1),prefix['actions'][tti-1]),(tti,'NN replay')
                        target=dict(CASES)[tti]
                        selections.extend(capture_candidates(tti,target,obs,ctbs,snapshot,draft,chosen,scores,phy))
                    elif tti==first:
                        saved=torch.load(OUT/f'state_t{first}_u{spec["ue"]}.pt',weights_only=False)
                        assert p.digest_arrays(snapshot)==p.digest_arrays(saved['snapshot'])
                        assert asdict(ctbs)==asdict(saved['ctbs'])
                        assert np.array_equal(draft,saved['draft']) and np.array_equal(chosen,saved['neural'])
                        chosen=np.asarray(spec['chosen']);assert (chosen!=draft).sum()==(0 if spec['arm']=='keep' else 1)
                        ue,prg=encode(chosen,obs)
                        token=spec['token']
                    prediction={k:v[0] for k,v in phy.evaluate(chosen).items()}
                    for u in range(36):pw.writerow(dict(tti=tti,ue_id=u,**{k:prediction[k][u] for k in fields[2:]}))
                    dw.writerow(dict(tti=tti,token=token,intervention=int(not capture and tti==first)))
                    if not capture and tti in [first,first+399,horizon]:c.write_json(out/f'ctbs_{tti}.json',asdict(ctbs))
                else:
                    builder.build(obs) # reproduce EMA state without building unused prefix graphs
                    chosen=prefix['actions'][tti-1].reshape(3,17)
                    ue,prg=encode(chosen,obs,reference_slots[tti-1])
                actions.append(chosen.reshape(-1).copy())
                next_obs,_,_,next_info=client.step(ue,prg)
                serving=np.argmax(obs['action_mask_cell_ue'],axis=0)
                actual=p._actual_chosen_from_mask(next_info['actual_ue_prg_mask'],serving,n_cell=3,n_prg=17)
                assert np.array_equal(actual,chosen.reshape(-1)),(tti,'native action')
                if calculate:
                    active=prediction['count']>0
                    for key,label in [('mcs','actual_ue_mcs'),('layer','actual_ue_layer'),('capacity','actual_ue_tbs_capacity_bytes')]:
                        assert np.array_equal(prediction[key][active],next_info[label][active]),(tti,key)
                    assert np.array_equal(prediction['dequeue'],next_info['actual_ue_served_bytes']),(tti,'dequeue')
                obs,info=next_obs,next_info
                if tti%500==0 or tti==first or tti==horizon:
                    pred_file.flush();decision_file.flush()
                    print(f'{out.name}: {tti}/{horizon}, elapsed={time.monotonic()-started:.1f}s',flush=True)
        finally:
            if client:client.close()
            try:code=proc.wait(timeout=60)
            except subprocess.TimeoutExpired:proc.terminate();code=proc.wait(timeout=15)
            assert code==0,code
    np.savez_compressed(out/'prefix.npz',hashes=np.asarray(hashes),actions=np.asarray(actions))
    service=pd.read_csv(out/'service_balance_trace.csv')
    limit=horizon if capture else first-1
    pd.testing.assert_frame_equal(service.iloc[:limit*36],reference_service.iloc[:limit*36],check_exact=True)
    for file,key in [('traffic_offered_packets.csv','arrival_tti'),('service_balance_trace.csv.channel.csv','tti')]:
        a=pd.read_csv(out/file);b=pd.read_csv(PREVIOUS/'full'/file)
        pd.testing.assert_frame_equal(a[a[key]<=horizon].reset_index(drop=True),b[b[key]<=horizon].reset_index(drop=True),check_exact=True)
    assert c.sha(out/'ue_topology.csv')==c.sha(PREVIOUS/'full/ue_topology.csv')
    merged=pd.read_csv(out/'predictions.csv').merge(service,on=['tti','ue_id'],validate='one_to_one')
    c.write_json(out/'precision_audit.json',audit(merged))
    assert forward_calls==(3 if capture else 800)
    c.write_json(out/'verification.json',dict(passed_native_execution_and_accounting=True,first=first,horizon=horizon,
        exact_observation_and_service_prefix_ttis=limit,exact_native_actions=horizon,nn_forward_calls=forward_calls,
        exogenous_inputs_exact=True,ctbs_dequeue_exact=True,elapsed_s=time.monotonic()-started))
    if capture:
        c.write_json(OUT/'selected.json',selections)
        print('FROZEN CANDIDATES',json.dumps([{k:s[k] for k in ['tti','ue','arm','token','donor','weak_expected_gain_bytes','network_expected_gain_bytes','has_joint_candidate']} for s in selections],indent=2),flush=True)
    print('VERIFIED',out.name,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['capture','verify_capture','run']);parser.add_argument('--name');parser.add_argument('--gpu',type=int,default=1)
    args=parser.parse_args()
    if args.mode=='verify_capture':verify_saved_capture()
    else:run(args)

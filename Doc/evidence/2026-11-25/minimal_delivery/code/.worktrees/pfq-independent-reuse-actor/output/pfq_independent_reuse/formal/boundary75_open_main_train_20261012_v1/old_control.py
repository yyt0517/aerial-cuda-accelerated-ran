"""Research copy of mixed-scene runner: only mean arrival lambda changes to1.0."""
import argparse,csv,json,os,subprocess,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from output.pfq_independent_reuse.formal.boundary08_weak_resource_probe_20261009_v1 import probe as prior
from training.stageb_hgraph.directed_reuse import DirectedReuseCoordinator
from output.pfq_independent_reuse.formal.mixed_center_boundary_coordination_20261010_v1.candidates import choose_candidates
OUT=Path(__file__).resolve().parent
old=prior.old;p=prior.p;c=prior.c
PROTOCOL=json.loads((OUT/'protocol.json').read_text());ROOTS=[]
FIELDS=['tti','ue_id','count','layer','mcs','capacity','dequeue','sinr_db','bler','expected','interference']

def check_scene(out):
    topology=pd.read_csv(out/'ue_topology.csv');ue=topology[topology.node_type=='ue'].copy()
    assert len(ue)==36
    center=ue.ue_local_id<3
    assert center.sum()==9 and (ue.loc[center,'d_serving_m']<=150.001).all()
    assert (ue.loc[~center,'d_serving_m']>=479.99).all()
    service=pd.read_csv(out/'service_balance_trace.csv')
    assoc=service.groupby('ue_id').serving_cell.agg(['first','nunique']);assert (assoc['nunique']==1).all()
    ue['actual_serving_cell']=ue.ue_id.map(assoc['first'])
    ue['class']=np.where(center,'center','boundary')
    ue['configured_rate']=1.0*np.asarray([.6,1,1.4])[ue.ue_id.to_numpy()//12]*np.where(ue.ue_local_id<6,1.25,.75)
    ue.to_csv(out/'scene_ues.csv',index=False)
    s=ue.groupby(['actual_serving_cell','class']).agg(ues=('ue_id','size'),arrival_packets_per_tti=('configured_rate','sum')).reset_index()
    s['expected_mbps']=s.arrival_packets_per_tti*48;s.to_csv(out/'scene_loads.csv',index=False)
    assert abs(ue.configured_rate.sum()*48-1728.0)<1.e-5
    return s.to_dict('records')

def run(args):
    first=args.root
    horizon=first+799 if first else args.horizon
    out=OUT/args.name;out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    snapshot_path=out/'pre_action.bin';socket=f'/tmp/mixed_coord_{os.getpid()}.sock'
    env=c.eval_env(c.environment(),out,42,socket)
    env.update(CUMAC_UE_PLACEMENT_MODE='coop_boundary',CUMAC_TRAFFIC_ARRIVAL_RATE='1.0',
        CUMAC_TRAFFIC_PROFILE='mixed_center_boundary_imbalanced_v1' if args.scene else 'homogeneous',
        CUMAC_MIXED_CENTER_BOUNDARY='1' if args.scene else '0',CUMAC_PFQ_DEMAND_CAP_SLACK_BYTES='0',
        CUMAC_DPA_FEATURE_MODE='dimensionless',CUMAC_DIRECTED_SNAPSHOT_PATH=str(snapshot_path),
        CUMAC_TOPOLOGY_DUMP='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='1')
    sim=old.OUT/'simulator' if args.legacy else OUT.parent/'boundary75_continuous_coordination_20261012_v1/simulator'
    command=[str(sim),'-d','1','-b','2','-x','0','-f','0','-g','100','-r','3000']
    actor=coordinator=None
    if not args.teacher:
        actor,_=old._load_actor_checkpoint(old.CHECKPOINT,torch.device('cpu'));actor.eval()
        ckpt=torch.load(prior.CHECKPOINT,map_location='cpu',weights_only=False)
        coordinator=DirectedReuseCoordinator();coordinator.load_state_dict(ckpt['model']);coordinator.eval()
    builder=p.FeatureSnapshotBuilder(p.SnapshotBuilderConfig(tb_err_ema_alpha=.1))
    edges=p.SparseEdgeGenerator(p.EdgeGeneratorConfig(ue_prg_topk=4,ue_prg_diversity_extra=0,candidate_mode='all_demand_ues',include_pfq_score_anchor=False,include_candidate_risk_features=False))
    reference=np.load(OUT/'reference/prefix.npz') if first else None
    root_path=OUT/(args.root_state or f'root_{first}.pt')
    root=torch.load(root_path,weights_only=False) if first else None
    c.write_json(out/'launch.json',dict(args=vars(args),horizon=horizon,command=command,env={k:v for k,v in env.items() if k.startswith('CUMAC_') or k=='CUDA_VISIBLE_DEVICES'},simulator_sha256=c.sha(sim),protocol_sha256=c.sha(OUT/'protocol.json'),root_state_sha256=c.sha(root_path) if first else None,old_coordinator_sha256=c.sha(prior.CHECKPOINT),bc_sha256=c.sha(old.CHECKPOINT)))
    hashes=[];actions=[];ue_rows=[];prg_rows=[];forward=0;checked=0;client=None;started=time.monotonic();summaries=[]
    with (out/'sim.log').open('w') as log,(out/'predictions.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
        proc=subprocess.Popen(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            client=p.BridgeEnvClient(p.BridgeEnvConfig(socket_path=socket,connect_timeout_s=60,enable_ctbs=True))
            obs,_,info=client.reset(seed=42,episode_horizon=horizon)
            assert info['harq_enabled']==0
            config=p.PhyTbsConfig.from_vector(info['ctbs_config']);phy=old.CausalPhy(config)
            for tti in range(1,horizon+1):
                digest=p.digest_arrays(obs);hashes.append(digest)
                if first and tti<=first:assert digest==reference['hashes'][tti-1],(tti,'prefix observation')
                ctbs=p._ctbs_inputs(client,info,config);assert ctbs.tti==tti-1
                calculate=tti>=first if first else (tti>4000 or tti in [1,horizon] or tti%1000==0)
                if first and tti<first:
                    builder.build(obs)
                    chosen=reference['actions'][tti-1].reshape(3,17);ue=reference['ue'][tti-1];prg=reference['prg'][tti-1]
                elif args.teacher:
                    builder.build(obs);ue=obs['teacher_action_ue_select'].copy();prg=obs['teacher_action_prg_alloc'].copy()
                    chosen=p.decode_type0_teacher_action(ue,prg,n_cell=3,n_prg=17).reshape(3,17)
                elif tti<=4000:
                    builder.build(obs);ue=obs['teacher_action_ue_select'].copy();prg=obs['teacher_action_prg_alloc'].copy()
                    chosen=p.decode_type0_teacher_action(ue,prg,n_cell=3,n_prg=17).reshape(3,17)
                else:
                    snapshot=old.read_snapshot(snapshot_path,ctbs.tti)
                    draft,chosen,ue,prg,scores,token=prior.policy(obs,ctbs,snapshot,builder,edges,actor,coordinator,phy);forward+=1
                    if first and tti==first:
                        assert np.array_equal(chosen,root['arms']['old'])
                        assert p.digest_arrays(snapshot)==p.digest_arrays(root['snapshot'])
                        assert asdict(ctbs)==asdict(root['ctbs'])
                        chosen=root['arms'][args.arm].copy();ue,prg=prior.encode(chosen,obs)
                    elif not first and tti in ROOTS:
                        arms,pred,rows,summary=choose_candidates(chosen,phy,obs,tti)
                        # All selected actions are materializable without reading outcomes.
                        for a in arms.values():prior.encode(a,obs)
                        pd.DataFrame(rows).to_csv(OUT/f'candidates_{tti}.csv',index=False)
                        c.write_json(OUT/f'root_{tti}.json',summary)
                        torch.save(dict(obs=obs,ctbs=ctbs,snapshot=snapshot,arms=arms,predictions=pred,summary=summary),OUT/f'root_{tti}.pt')
                        summaries.append(summary);print('ROOT',json.dumps(summary),flush=True)
                if calculate:
                    snapshot=old.read_snapshot(snapshot_path,ctbs.tti);phy.prepare(snapshot,ctbs)
                    prediction={k:v[0] for k,v in phy.evaluate(chosen).items()}
                    for u in range(36):writer.writerow(dict(tti=tti,ue_id=u,**{k:prediction[k][u] for k in FIELDS[2:]}))
                    if tti in [1,first,first+399,horizon] or tti in ROOTS:c.write_json(out/f'ctbs_{tti}.json',asdict(ctbs))
                actions.append(chosen.reshape(-1).copy());ue_rows.append(np.asarray(ue).copy());prg_rows.append(np.asarray(prg).copy())
                nxt,_,_,ni=client.step(ue,prg)
                actual=p._actual_chosen_from_mask(ni['actual_ue_prg_mask'],np.asarray(ctbs.serving_cell),n_cell=3,n_prg=17)
                assert np.array_equal(actual,chosen.reshape(-1)),(tti,'actual action')
                if calculate:
                    active=prediction['count']>0
                    for key,label in [('mcs','actual_ue_mcs'),('layer','actual_ue_layer'),('capacity','actual_ue_tbs_capacity_bytes')]:assert np.array_equal(prediction[key][active],ni[label][active]),(tti,key)
                    assert np.array_equal(prediction['dequeue'],ni['actual_ue_served_bytes']),(tti,'dequeue');checked+=1
                obs,info=nxt,ni
                if tti%250==0 or tti==horizon:
                    f.flush();print(f'{args.name} {tti}/{horizon} elapsed={time.monotonic()-started:.1f}s',flush=True)
        finally:
            if client:client.close()
            try:code=proc.wait(timeout=60)
            except subprocess.TimeoutExpired:proc.terminate();code=proc.wait(timeout=15)
            assert code==0,code
    np.savez_compressed(out/'prefix.npz',hashes=np.asarray(hashes),actions=np.asarray(actions),ue=np.asarray(ue_rows),prg=np.asarray(prg_rows))
    service=pd.read_csv(out/'service_balance_trace.csv')
    merged=pd.read_csv(out/'predictions.csv').merge(service,on=['tti','ue_id'],validate='one_to_one')
    c.write_json(out/'precision_audit.json',prior.audit(merged))
    if first:
        ref=pd.read_csv(OUT/'reference/service_balance_trace.csv',nrows=(first-1)*36)
        pd.testing.assert_frame_equal(service.iloc[:(first-1)*36],ref,check_exact=True)
        for name,key in [('traffic_offered_packets.csv','arrival_tti'),('service_balance_trace.csv.channel.csv','tti')]:
            a=pd.read_csv(out/name);b=pd.read_csv(OUT/'reference'/name)
            pd.testing.assert_frame_equal(a[a[key]<=horizon].reset_index(drop=True),b[b[key]<=horizon].reset_index(drop=True),check_exact=True)
        assert c.sha(out/'ue_topology.csv')==c.sha(OUT/'reference/ue_topology.csv')
        assert forward==800
    scene=check_scene(out) if args.scene else None
    c.write_json(out/'verification.json',dict(passed=True,horizon=horizon,native_actions=horizon,ctbs_dequeue_checked_ttis=checked,root=first,exact_shared_prefix_ttis=first-1 if first else None,forward_calls=forward,scene=scene,elapsed_s=time.monotonic()-started))
    if summaries:c.write_json(OUT/'selected.json',summaries)
    print('VERIFIED',args.name,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('name');ap.add_argument('--root',type=int,default=0);ap.add_argument('--root-state');ap.add_argument('--arm',default='old',choices=['old','joint','fill','shuffle']);ap.add_argument('--horizon',type=int,default=6400);ap.add_argument('--scene',type=int,default=1);ap.add_argument('--legacy',action='store_true');ap.add_argument('--teacher',action='store_true');run(ap.parse_args())

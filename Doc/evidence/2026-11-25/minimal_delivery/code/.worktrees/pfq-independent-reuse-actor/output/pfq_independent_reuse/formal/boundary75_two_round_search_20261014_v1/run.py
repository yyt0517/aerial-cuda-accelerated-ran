"""Single-seed native two-round search experiment; fixed PHY/MCS and K gate."""
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from output.pfq_independent_reuse.formal.boundary75_joint_long_residual_20261013_v1.policy import old,p,c,prior,materialize
from output.pfq_independent_reuse.formal.boundary75_old_joint_probe_20261013_v1.selector import candidates
from output.pfq_independent_reuse.formal.boundary75_joint_search_diagnostics_20261014_v1.probe import evaluate
from output.pfq_independent_reuse.formal.boundary08_directed_attention_20261008_v1.precision import native_bler
from training.stageb_hgraph.directed_reuse import DirectedReuseCoordinator

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
FROZEN=OUT.parent/'boundary75_joint_search_frozen_20261014_v1'
REF=FROZEN/'snapshot/output/pfq_independent_reuse/formal/boundary75_joint_long_residual_20261013_v1'
DIAG=OUT.parent/'boundary75_joint_search_diagnostics_20261014_v1'


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def two_round(draft,keep,phy,obs,ctbs):
    first=materialize(draft,keep,phy,obs,ctbs)
    incumbent=first['base_action']
    i=next(i for i,a in enumerate(first['actions']) if np.array_equal(a,incumbent))
    keep_dequeue=float(first['pred']['dequeue'][0].sum())
    first_score=float(first['pred']['expected'][i].sum())
    actions,meta=candidates(incumbent,incumbent,obs,phy.snapshot['queue'],True)
    pred=evaluate(phy,actions)
    scores=pred['expected'].sum(1);dequeues=pred['dequeue'].sum(1)
    assert np.isfinite(scores).all() and np.isfinite(dequeues).all()
    # The frozen MCS3 table is nonmonotone: np.interp can depend on preceding
    # batch inputs. Keep the frozen scoring; verify physical equality and isolate
    # the known <=8.1e-5 MCS3 BLER lookup difference instead of asserting bit equality.
    for key in ['count','layer','mcs','capacity','dequeue']:
        np.testing.assert_array_equal(pred[key][0],first['pred'][key][i])
    np.testing.assert_allclose(pred['sinr_db'][0],first['pred']['sinr_db'][i],rtol=0,atol=1e-10)
    difference=pred['bler'][0]-first['pred']['bler'][i]
    mcs3=pred['mcs'][0]==3
    assert np.abs(difference[~mcs3]).max(initial=0)<1e-10
    assert np.abs(difference[mcs3]).max(initial=0)<8.1e-5
    rescore_delta=float(scores[0]-first_score)
    if abs(rescore_delta)>1e-8 and not (OUT/'incumbent_rescore_audit.json').exists():
        write(OUT/'incumbent_rescore_audit.json',dict(tti=ctbs.tti+1,delta_bytes=rescore_delta,
            physical_fields_identical=True,affected_ues=np.flatnonzero(np.abs(difference)>1e-10).tolist(),
            mcs=pred['mcs'][0].tolist(),bler_difference=difference.tolist(),dequeue=pred['dequeue'][0].tolist()))
    eligible=dequeues>=keep_dequeue
    assert eligible[0]
    best=int(np.argmax(np.where(eligible,scores,-np.inf)))
    accepted=bool(scores[best]>first_score+1.)
    if not accepted:best=0
    chosen=actions[best]
    assert dequeues[best]>=keep_dequeue
    chosen_prediction={k:v[best] for k,v in pred.items()} if accepted else {k:v[i] for k,v in first['pred'].items()}
    chosen_score=float(scores[best]) if accepted else first_score
    assert chosen_score>=first_score
    def native_expected(j):
        active=pred['count'][j]>0
        return float((pred['dequeue'][j,active]*(1-native_bler(pred['sinr_db'][j,active],pred['mcs'][j,active]))).sum())
    native_curve_second_gain=native_expected(best)-native_expected(0)
    extra=int(((draft>=0)&(chosen<0)).sum());changed=int((chosen!=draft).sum())
    assert extra<=2 and changed<=4
    info=dict(first_candidates=first['full_candidates'],second_candidates=len(actions),
        keep_expected=float(first['pred']['expected'][0].sum()),first_expected=first_score,
        second_expected=chosen_score,second_gain=chosen_score-first_score,
        incumbent_rescore_delta_bytes=rescore_delta,native_curve_second_gain=native_curve_second_gain,
        keep_dequeue=keep_dequeue,first_dequeue=float(first['pred']['dequeue'][i].sum()),
        chosen_dequeue=float(dequeues[best]),second_accepted=accepted,second_kind=meta[best][0],
        extra_blanks=extra,changed_from_draft=changed)
    return incumbent,chosen,chosen_prediction,info


def run():
    torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False)
    provenance=json.loads((DIAG/'protocol.json').read_text())['source_hashes']
    for file,digest in provenance.items():assert c.sha(ROOT/file)==digest,file
    selection=json.loads((FROZEN/'selection.json').read_text())
    for file,digest in selection['selected_components'].items():assert c.sha(ROOT/file)==digest,file
    out=OUT/'eval_s301_two_round';out.mkdir(exist_ok=False)
    reference=REF/'eval_s301_search';prefix=np.load(reference/'prefix.npz')
    actor,_=old._load_actor_checkpoint(old.CHECKPOINT,torch.device('cpu'));actor.eval()
    coordinator=DirectedReuseCoordinator()
    coordinator.load_state_dict(torch.load(prior.CHECKPOINT,map_location='cpu',weights_only=False)['model']);coordinator.eval()
    builder=p.FeatureSnapshotBuilder(p.SnapshotBuilderConfig(tb_err_ema_alpha=.1))
    edges=p.SparseEdgeGenerator(p.EdgeGeneratorConfig(ue_prg_topk=4,ue_prg_diversity_extra=0,
        candidate_mode='all_demand_ues',include_pfq_score_anchor=False,include_candidate_risk_features=False))
    launch=json.loads((reference/'launch.json').read_text());env=os.environ.copy();env.update(launch['env'])
    original=str(Path(launch['env']['CUMAC_DIRECTED_SNAPSHOT_PATH']).parent)
    for key,value in list(env.items()):
        if key.startswith('CUMAC_') and value.startswith(original):env[key]=str(out)+value[len(original):]
    socket=f'/tmp/b75two_{os.getpid()}.sock'
    env.update(CUMAC_ONLINE_SOCKET=socket,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    simulator=FROZEN/'snapshot/output/pfq_independent_reuse/formal/boundary75_continuous_coordination_20261012_v1/simulator'
    command=[str(simulator)]+launch['command'][1:]
    protocol=dict(seed=301,question='Does a second local search round improve sustained native goodput versus frozen one-round search?',
        warmup_ttis=4000,measurement=[4001,8000],followup_end=8400,
        first_round='Unmodified frozen materialize() full candidate argmax and >K+1B acceptance.',
        second_round='Enumerate the same neighborhood around A1; dequeue>=original K; replace A1 only if S>A1+1B.',
        budget='At most 4 changed positions and 2 new blanks relative to original BC draft D; at most 2 rounds.',
        mcs='Unchanged CTBS/OLLA rule; MCS fixed across candidates within each TTI. OLLA evolves on each arm own trajectory.',
        reference='Archived seed301 search; PFQ retained as context, not the primary comparison.',
        metrics=['native_goodput','mature_packet_expiry','queue_drift','end_queue','UE_p10'],
        stop='Complete one seed with fixed 4000+4000+400 TTIs, irrespective of outcome; no tuning or further rounds/seeds.',
        numerical_check='Physical incumbent fields identical; only legacy nonmonotone MCS3 interpolation may differ by <=8.1e-5 BLER across batches. Frozen scores and >1B acceptance unchanged. Native-curve second gain logged separately.',
        source_hashes=provenance,diagnostic_evaluate_sha256=c.sha(DIAG/'probe.py'),
        runner_sha256=c.sha(__file__),simulator_sha256=c.sha(simulator),
        bc_sha256=c.sha(old.CHECKPOINT),coordinator_sha256=c.sha(prior.CHECKPOINT))
    write(OUT/'protocol.json',protocol)
    write(out/'launch.json',dict(command=command,env={k:v for k,v in env.items() if k.startswith('CUMAC_') or k=='CUDA_VISIBLE_DEVICES'},protocol=protocol))
    hashes=[];actions=[];ue_rows=[];prg_rows=[];goodputs=[];decisions=[]
    started=time.monotonic();client=None
    with (out/'sim.log').open('w') as log:
        proc=subprocess.Popen(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            client=p.BridgeEnvClient(p.BridgeEnvConfig(socket_path=socket,connect_timeout_s=60,enable_ctbs=True))
            obs,_,info=client.reset(seed=301,episode_horizon=8400)
            assert info['harq_enabled']==0
            config=p.PhyTbsConfig.from_vector(info['ctbs_config']);phy=old.CausalPhy(config)
            for tti in range(1,8401):
                hashes.append(p.digest_arrays(obs))
                if tti<=4001:assert hashes[-1]==prefix['hashes'][tti-1],(tti,'common pre-action state')
                ctbs=p._ctbs_inputs(client,info,config);assert ctbs.tti==tti-1
                prediction=None
                if tti<=4000:
                    builder.build(obs)
                    ue=obs['teacher_action_ue_select'].copy();prg=obs['teacher_action_prg_alloc'].copy()
                    chosen=p.decode_type0_teacher_action(ue,prg,n_cell=3,n_prg=17).reshape(3,17)
                else:
                    decision_start=time.monotonic()
                    snapshot=old.read_snapshot(out/'pre_action.bin',ctbs.tti)
                    draft,keep,ue,prg,*_=prior.policy(obs,ctbs,snapshot,builder,edges,actor,coordinator,phy)
                    first,chosen,prediction,record=two_round(draft,keep,phy,obs,ctbs)
                    if not np.array_equal(chosen,keep):ue,prg=prior.encode(chosen,obs)
                    record.update(tti=tti,decision_wall_s=time.monotonic()-decision_start)
                    decisions.append(record)
                    if tti==4001:
                        assert np.array_equal(first.reshape(-1),prefix['actions'][tti-1])
                        check=pd.read_csv(DIAG/'candidate_diagnostics.csv').set_index('tti').loc[tti]
                        np.testing.assert_allclose(record['second_gain'],check.second_gain_original_keep,rtol=0,atol=1e-8)
                        write(OUT/'initial_verification.json',dict(passed=True,first_round_exact_frozen_action=True,
                            second_round_matches_prior_diagnostic=True,root_tti=tti,**record))
                actions.append(chosen.reshape(-1).copy());ue_rows.append(np.asarray(ue).copy());prg_rows.append(np.asarray(prg).copy())
                nxt,_,_,ni=client.step(ue,prg)
                actual=p._actual_chosen_from_mask(ni['actual_ue_prg_mask'],np.asarray(ctbs.serving_cell),n_cell=3,n_prg=17)
                assert np.array_equal(actual,chosen.reshape(-1)),(tti,'native action')
                if prediction is not None:
                    active=prediction['count']>0
                    for key,label in [('mcs','actual_ue_mcs'),('layer','actual_ue_layer'),('capacity','actual_ue_tbs_capacity_bytes')]:
                        np.testing.assert_array_equal(prediction[key][active],ni[label][active],err_msg=f'{tti} {key}')
                    np.testing.assert_array_equal(prediction['dequeue'],ni['actual_ue_served_bytes'],err_msg=f'{tti} dequeue')
                good=float(np.asarray(ni['actual_ue_goodput_bytes']).sum());goodputs.append(good)
                if tti<=4000:
                    for key,value in [('actions',actions[-1]),('ue',ue_rows[-1]),('prg',prg_rows[-1])]:
                        np.testing.assert_array_equal(value,prefix[key][tti-1])
                    assert good==prefix['goodput'][tti-1]
                obs,info=nxt,ni
                if tti%100==0 or tti==4001:
                    pd.DataFrame(decisions).to_csv(out/'decisions.csv',index=False)
                    end=min(tti,8000);n=end-4000
                    progress=dict(phase='running',tti=tti,total=8400,elapsed_s=time.monotonic()-started)
                    if n>0:
                        progress.update(measured_ttis=n,partial_goodput_mbps=sum(goodputs[4000:end])*.016/n,
                            reference_same_window_mbps=float(prefix['goodput'][4000:end].sum()*.016/n),
                            mean_second_gain_bytes=float(np.mean([r['second_gain'] for r in decisions if r['tti']<=end])))
                    write(OUT/'status.json',progress);print(json.dumps(progress),flush=True)
        finally:
            if client:client.close()
            try:code=proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.terminate();code=proc.wait(timeout=15)
            assert code==0,code
    np.savez_compressed(out/'prefix.npz',hashes=np.asarray(hashes),actions=np.asarray(actions),ue=np.asarray(ue_rows),prg=np.asarray(prg_rows),goodput=np.asarray(goodputs))
    service=pd.read_csv(out/'service_balance_trace.csv');ref=pd.read_csv(reference/'service_balance_trace.csv')
    assert len(service)==8400*36 and not service.duplicated(['tti','ue_id']).any()
    pd.testing.assert_frame_equal(service.iloc[:4000*36],ref.iloc[:4000*36],check_exact=True)
    np.testing.assert_array_equal(service.groupby('tti').successful_bytes.sum(),goodputs)
    for name in ['traffic_offered_packets.csv','service_balance_trace.csv.channel.csv','ue_topology.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(out/name),pd.read_csv(reference/name),check_exact=True)
    assert len(decisions)==4400
    write(out/'verification.json',dict(passed=True,total_ttis=8400,native_actions=8400,ctbs_mcs_layer_tbs_dequeue_checked=4400,
        exact_common_prefix_ttis=4000,root_observation_exact=True,exogenous_inputs_exact=True,source_hashes=provenance,
        second_round_calls=4400,elapsed_s=time.monotonic()-started))
    write(OUT/'status.json',dict(phase='native_complete',tti=8400,elapsed_s=time.monotonic()-started))
    print('NATIVE_VERIFIED',flush=True)


if __name__=='__main__':
    try:run()
    except BaseException as exc:
        write(OUT/'FAILED.json',dict(type=type(exc).__name__,message=str(exc)))
        raise

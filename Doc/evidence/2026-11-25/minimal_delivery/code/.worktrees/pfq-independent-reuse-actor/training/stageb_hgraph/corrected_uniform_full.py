"""One fresh BC/Full run with paired argmax checkpoints, fixed seed42/43 protocol."""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import traceback

import numpy as np
import pandas as pd

from .dpa_ablation_experiment import ROOT, SIM_ARGS, base_env, eval_env, cli, run_command, sha, write_json, train_command
from .dpa_reproduction_experiment import bc_collection_command
from .pfq_reference_calibration import _extend_ld_library_path
from .corrected_uniform_metrics import export_arm, export_legacy_catalog

SIM=ROOT/'build.corrected_uniform_v1/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection'
LIB=ROOT/'build.corrected_uniform_v1/cuMAC/src/libcumac.so'
SIM_SHA='9e71e4b959b996251919e20965a76cf1fa2b9c512181b02eaffdc4b8f0d6674d'
LIB_SHA='7b414f13851e50da73c91db3fc1342017113c8180a04148606f24a5adcc08eea'


def environment():
    assert sha(SIM)==SIM_SHA and sha(LIB)==LIB_SHA
    env=base_env(1)
    env.update(CUMAC_UE_PLACEMENT_MODE='uniform',CUMAC_DPA_FEATURE_MODE='dimensionless',
               CUMAC_PHY_AUDIT_PERMUTE_SLOTS='0',CUMAC_PHY_AUDIT_COMMON_MCS='0')
    _extend_ld_library_path(env,SIM)
    return env


def options(command, changes):
    command=list(command)
    for key,value in changes.items():
        flag='--'+key
        if flag in command: command[command.index(flag)+1]=str(value)
        else: command.extend([flag,str(value)])
    return command


def filter_bc_dataset(source,destination,warmup=1000):
    from .online_imitation import IndependentBcDatasetReader,IndependentBcDatasetWriter
    reader=IndependentBcDatasetReader(source)
    m=dict(reader.manifest['metadata'])
    m['expected_steps_per_episode']-=warmup
    m['expected_record_count']=m['expected_steps_per_episode']*m['expected_episode_count']
    m.update(warmup_excluded_steps_per_episode=warmup,source_dataset_sha256=reader.manifest['dataset_sha256'])
    writer=IndependentBcDatasetWriter(destination,initial_actor_sha256=reader.manifest['initial_actor_sha256'],metadata=m)
    identities=[]
    for record in reader:
        metadata=dict(record['metadata'])
        if metadata['step']<warmup: continue
        metadata['source_step']=metadata['step'];metadata['source_global_step']=metadata['global_step']
        metadata['step']-=warmup
        metadata['global_step']=len(identities)
        assert metadata['topology_seed'] in (42,43) and metadata['ctbs_post_action_exact']
        identities.append(dict(index=len(identities),episode=metadata['episode'],seed=metadata['topology_seed'],bridge_tti=metadata['tti'],tti=metadata['tti']+1))
        writer.write(graph=record['graph'],teacher_trace=record['teacher_trace'],teacher_action_ue=record['teacher_action_ue'],
                     teacher_action_prg=record['teacher_action_prg'],next_obs=record['next_obs'],next_info=record['next_info'],step_metadata=metadata)
    manifest=writer.finalize(max_candidates=reader.manifest['max_candidates'])
    pd.DataFrame(identities).to_csv(Path(destination)/'measured_sample_identity.csv',index=False)
    return manifest


def collect(root,steps,*,warmup=1000):
    out=root/'collect';out.mkdir(parents=True,exist_ok=False)
    dataset=root/'bc_raw_dataset'
    cmd=options(bc_collection_command(SIM,out,dataset,steps=warmup+steps,feature_mode='dimensionless'),
                {'episodes':2,'topology-seed':42,'topology-seed-mode':'list_cycle','seed-list':'42,43','independent-bc-max-candidates':16,'device':'cpu'})
    env=environment()
    run_command(cmd,out,env,24*3600)
    return filter_bc_dataset(dataset,root/'bc_measured_dataset',warmup=warmup)


def train_bc(root):
    out=root/'bc_train';out.mkdir(exist_ok=False)
    cmd=options(bc_collection_command(SIM,out,root/'unused',steps=4000,feature_mode='dimensionless'),
                {'episodes':2,'topology-seed':42,'topology-seed-mode':'list_cycle','seed-list':'42,43',
                 'device':'cpu','independent-bc-collect-only':0,'independent-bc-dataset-in':root/'bc_measured_dataset'})
    index=cmd.index('--independent-bc-dataset-out');del cmd[index:index+2]
    run_command(cmd,out,environment(),24*3600)
    summary=json.loads((out/'online_imitation_summary.json').read_text())
    rows=[json.loads(line) for line in (out/'online_imitation_metrics.jsonl').read_text().splitlines()]
    assert len(rows)==8000 and all(np.isfinite(row['loss']) and np.isfinite(row['grad_norm']) for row in rows)
    assert summary['resolved_max_steps']==8000 and summary['episodes']==2
    return out/'hgraph_independent_bc_last.pt'


def evaluate(root,checkpoint,iteration,seed,baseline_dir,*,final=False):
    out=root/('frozen' if final else 'validation')/f'iter{iteration:04d}_seed{seed}'
    reuse_completed=(out/'eval_summary.json').exists()
    if out.exists():
        assert reuse_completed and iteration==0 and not final, 'Only a completed pre-PPO evaluation may be reused'
        original=json.loads((out/'command.json').read_text())['command']
        assert original[original.index('--checkpoint')+1]==str(checkpoint)
        summary=json.loads((out/'eval_summary.json').read_text())
        assert summary['checkpoint_sha256']==sha(checkpoint) and summary['sim_binary_sha256']==sha(SIM)
        assert sum(1 for _ in (out/'eval_steps.jsonl').open())==5000
    else:
        out.mkdir(parents=True,exist_ok=False)
    socket=f'/tmp/corrected_full_eval_{os.getpid()}_{seed}.sock'
    env=eval_env(environment(),out,seed,socket)
    env['CUMAC_RAW_OBSERVATIONS_CSV']=str(out/'ue_tti_observations.csv')
    env['CUMAC_RR_AUDIT_SEED']=str(seed)
    cmd=cli('eval_checkpoint_online',{
        'checkpoint':checkpoint,'socket-path':socket,'episode-horizon':5000,'max-steps':5000,
        'stats-warmup-steps':1000,'decode-mode':'argmax','seed':seed,'sample-seed':seed,
        'device':'cpu','print-every-steps':500,'save-steps':1,'sim-bin':SIM,'out-dir':out,
        'outcome-samples':out/'packet_outcome_samples.csv','independent-policy-diagnostics':1,
        'independent-runtime-candidate-width':0,'independent-headroom-accounting':'ctbs-v1',
        'independent-non-starvation-guard':'off','slot-repair-max-per-cell':0,'safe-fill-max-per-cell':0,
        'candidate-blank-budget-max-per-cell':-1})
    if not reuse_completed:
        with (out/'sim.log').open('w') as log:
            proc=subprocess.Popen([str(SIM),*shlex.split(SIM_ARGS)],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                run_command(cmd,out,env,24*3600)
                assert proc.wait(timeout=60)==0
            finally:
                if proc.poll() is None: proc.terminate();proc.wait(timeout=15)
    write_json(out/'reference_manifest.json',dict(binary_sha256=sha(SIM),libcumac_sha256=sha(LIB),checkpoint_sha256=sha(checkpoint),
               scenario=dict(ue_placement='uniform',topology_seed=seed,traffic_seed=seed),reference_policy='dpa',warmup_steps=1000,measure_steps=4000))
    reference=Path(baseline_dir)/f'seed{seed}_pfq'
    assert sha(out/'ue_topology.csv')==sha(reference/'ue_topology.csv'), 'DPA topology is not paired'
    for filename,timekey in [('traffic_offered_packets.csv','arrival_tti'),('service_balance_trace.csv.channel.csv','tti')]:
        observed=pd.read_csv(out/filename).query(f'{timekey}<=5000').reset_index(drop=True)
        expected=pd.read_csv(reference/filename).query(f'{timekey}<=5000').reset_index(drop=True)
        pd.testing.assert_frame_equal(observed,expected,check_exact=True)
    metrics=export_arm(out,seed,'dpa',1000,4000)[-1]
    export_legacy_catalog(out)
    write_json(out/'validation_metrics.json',dict(iteration=iteration,seed=seed,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),metrics=metrics))
    return metrics


def qualify(pair,baseline):
    checks={}
    for seed,m in pair.items():
        b=baseline[str(seed)]
        checks[str(seed)]={
            'goodput':m['goodput_mbps']>=.99*b['goodput_mbps'],
            'expiry':m['expiry_bytes_over_offered']<=b['expiry_bytes_over_offered']+.005,
            'weak_ue':m['ue_goodput_mbps_p10']>=.9*b['ue_goodput_mbps_p10']}
        for field in ['mac_dequeue_completion_delay_ms_mean','mac_dequeue_completion_delay_ms_p95']:
            checks[str(seed)][field]=m[field] is not None and b[field] is not None and m[field]<=max(1.1*b[field],b[field]+.5)
        for field in ['total_backlog_bytes_mean','total_backlog_bytes_p95']:
            checks[str(seed)][field]=m[field]<=1.1*b[field]
    return all(all(c.values()) for c in checks.values()),checks


def final_report(root,baseroot,evaluations,chosen):
    comparison=pd.read_csv(baseroot/'comparison.csv')
    frozen=pd.concat([pd.read_csv(p) for p in sorted((root/'frozen').glob('*/global_metrics.csv'))])
    comparison=pd.concat([comparison,frozen],ignore_index=True)
    comparison.to_csv(root/'comparison.csv',index=False)
    rows=[]
    for r in evaluations:
        for seed,m in r['metrics'].items():
            rows.append(dict(iteration=r['iteration'],seed=seed,checkpoint_sha256=r['sha256'],eligible=r['eligible'],**{k:v for k,v in m.items() if k not in ['seed','policy']}))
    curves=pd.DataFrame(rows);curves.to_csv(root/'argmax_learning_curve.csv',index=False)
    objectives=[]
    for r in evaluations:
        values=list(r['metrics'].values())
        objectives.append(np.array([-min(v['goodput_mbps'] for v in values),max(v['expiry_bytes_over_offered'] for v in values),max(v['mac_completion_or_expiry_outcome_delay_ms_p95'] for v in values)]))
    pareto=[]
    for i,r in enumerate(evaluations):
        dominated=any(np.all(other<=objectives[i]) and np.any(other<objectives[i]) for j,other in enumerate(objectives) if j!=i)
        pareto.append(dict(iteration=r['iteration'],checkpoint_sha256=r['sha256'],eligible=r['eligible'],pareto=not dominated,
                           worst_goodput_mbps=-objectives[i][0],worst_expiry=objectives[i][1],worst_outcome_p95_ms=objectives[i][2]))
    pd.DataFrame(pareto).to_csv(root/'pareto_checkpoints.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(9,10),sharex=True)
    for ax,key in zip(axes,['goodput_mbps','expiry_bytes_over_offered','mac_completion_or_expiry_outcome_delay_ms_mean']):
        for seed in [42,43]:
            data=curves[curves.seed==seed].sort_values('iteration')
            line=ax.plot(data.iteration,data[key],marker='o',label=f'DPA seed{seed}')[0]
            baseline=comparison[(comparison.seed==seed)&(comparison.policy=='pfq')&(comparison.window=='measured')].iloc[0]
            ax.axhline(baseline[key],color=line.get_color(),linestyle='--',label=f'PFQ seed{seed}')
        ax.set_ylabel(key);ax.grid(alpha=.25);ax.legend()
    axes[-1].set_xlabel('PPO iteration (0 = fresh BC)')
    fig.suptitle('Corrected uniform training scenarios 42/43; independent argmax validation')
    fig.tight_layout();fig.savefig(root/'argmax_learning_curve.png',dpi=160);plt.close(fig)
    keys=['seed','policy','goodput_mbps','expiry_bytes_over_offered','mac_dequeue_completion_delay_ms_mean','successful_delivery_delay_ms_mean','mac_completion_or_expiry_outcome_delay_ms_mean','ue_goodput_mbps_p10']
    measured=comparison[comparison.window=='measured'].sort_values(['seed','policy'])
    text='# Corrected uniform Full experiment\n\n'
    text+=f"Frozen iteration: {chosen['iteration']}; SHA256: `{chosen['sha256']}`. Eligible checkpoints: {sum(r['eligible'] for r in evaluations)}.\n\n"
    text+='Qualified checkpoint selected by frozen two-seed rule.\n\n' if chosen['eligible'] else 'No checkpoint met all frozen per-seed criteria. Last checkpoint is shown for diagnosis; the superiority hypothesis is not supported by this run.\n\n'
    table=measured[keys].to_csv(sep='|',index=False,float_format='%.6f').splitlines()
    text+=table[0]+'\n'+'|'.join(['---']*len(measured[keys].columns))+'\n'+'\n'.join(table[1:])
    text+='\n\nAll policies use the same verified build, 1000 warmup + 4000 measured TTIs, and paired topology/traffic/channel per seed. Both seeds participated in BC/PPO and checkpoint selection: this is training-scenario reproduction, not unseen generalization.\n\n'
    text+='MAC dequeue completion is distinct from fully successful packet delivery. Completion-or-expiry delay includes expired and failed-MAC-completed outcomes. Successful-delivery delays are success-conditioned; pending packets retain right censoring and raw remaining bytes.\n\n'
    text+='Fresh BC uses exactly 8000 measured records, excluding each seed’s first 1000. PPO keeps mean subdecision log-prob, Full attention/deadline/CTBS, and a single continuous optimizer/critic/normalizer state. Checkpoints are retained every 4 iterations; the validation schedule and its user-directed amendments are recorded in validation_schedule.json.\n\n'
    text+='![Independent argmax learning curves](argmax_learning_curve.png)\n\nSee `argmax_learning_curve.csv`, `ppo/online_ppo_metrics.csv`, `pareto_checkpoints.csv`, `checkpoint_evaluations.json`, `comparison.csv`, per-arm raw/aggregate CSVs, metric catalogs and `sha256_manifest.csv`.\n'
    (root/'report.md').write_text(text)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out-dir',required=True)
    parser.add_argument('--baseline-dir',required=True)
    parser.add_argument('--pilot-only',action='store_true')
    parser.add_argument('--continue-from-bc',action='store_true',help='Continue this campaign after a failed pre-PPO evaluation; never retrain BC.')
    args=parser.parse_args()
    root=Path(args.out_dir).resolve();root.mkdir(parents=True,exist_ok=args.continue_from_bc)
    baseroot=Path(args.baseline_dir).resolve()
    gate=json.loads((baseroot/'validation.json').read_text());assert gate['passed']
    baseline={str(seed):pd.read_csv(baseroot/f'seed{seed}_pfq/global_metrics.csv').query("window=='measured'").iloc[0].to_dict() for seed in [42,43]}
    if args.continue_from_bc:
        assert not args.pilot_only and not (root/'ppo').exists()
        protocol=json.loads((root/'protocol.json').read_text())
        status=json.loads((root/'status.json').read_text())
        assert status['status'] in {'failed','pre_ppo_schedule_change'} and status.get('iteration')==0
        assert protocol['binary_sha256']==sha(SIM) and protocol['libcumac_sha256']==sha(LIB)
        assert protocol['baseline']==baseline
        checkpoint=root/'bc_train/hgraph_independent_bc_last.pt'
        rows=[json.loads(line) for line in (root/'bc_train/online_imitation_metrics.jsonl').read_text().splitlines()]
        assert len(rows)==8000 and all(np.isfinite(r['loss']) and np.isfinite(r['grad_norm']) for r in rows)
        assert checkpoint.is_file()
        schedule_change=status['status']=='pre_ppo_schedule_change'
        recovery=root/('pre_ppo_schedule_change' if schedule_change else 'pre_ppo_recovery')
        recovery.mkdir(exist_ok=False)
        (recovery/'previous_status.json').write_bytes((root/'status.json').read_bytes())
        if not schedule_change:
            (root/'validation').rename(recovery/'failed_validation')
        (recovery/'corrected_uniform_full.py').write_bytes(Path(__file__).read_bytes())
        write_json(recovery/'identity.json',dict(reason=('User requested less frequent validation; preserve every4 saves, evaluate8 then20/40/60/80/100.' if schedule_change else 'Evaluator candidate width1 invalid; use full topology auto width0.'),
                   checkpoint_sha256=sha(checkpoint),source_sha256=sha(Path(__file__)),time=time.time()))
        status.pop('error',None)
    else:
        protocol=dict(model_seed=801,model_device='cpu',phy_device='cuda',scene_seeds=[42,43],bc_candidate_width=16,actual_association_counts={'42':[14,13,9],'43':[11,16,9]},baseline=baseline,binary_sha256=sha(SIM),libcumac_sha256=sha(LIB),
                      first_check_iterations=8,max_iterations=100,checkpoint_interval=4,mean_subdecision_log_probability=True,
                      feature_mode='dimensionless',unseen_generalization_claim=False,
                      reward='deadline_tail E; offered=108000 bytes/TTI; weights 1,-1,-2; deadline eta=.25; rate tail weight=.25; both tail=.1; window100',
                      reward_scales_semantics='New baseline reward_scales retained as diagnostics. Existing deadline_tail E is unit-normalized and does not consume empirical PFQ P95 scales.',
                      qualification='Per seed: goodput>=.99 PFQ; expiry<=PFQ+.005; MAC mean/P95<=max(1.1PFQ,PFQ+.5ms); UE P10>=.9PFQ; backlog mean/P95<=1.1PFQ; not relaxed after training.')
        from .pfq_reference_calibration import _git_revision
        source_names=['online_imitation.py','online_ppo.py','independent_actor.py','independent_policy.py','feature_snapshot.py',
                      'bridge_env.py','corrected_raw_export.py','corrected_uniform_metrics.py','corrected_uniform_full.py']
        protocol['source_revision']=_git_revision(ROOT)
        protocol['source_sha256']={name:sha(ROOT/'training/stageb_hgraph'/name) for name in source_names}
        snapshot=root/'source_snapshot';snapshot.mkdir()
        for name in source_names:(snapshot/name).write_bytes((ROOT/'training/stageb_hgraph'/name).read_bytes())
        write_json(root/'protocol.json',protocol)
        status=dict(status='bc_collection',started=time.time(),protocol_sha256=sha(root/'protocol.json'))
        write_json(root/'status.json',status)
    if not (root/'validation_schedule.json').exists():
        write_json(root/'validation_schedule.json',dict(save_interval=4,evaluate_during_ppo=False,reason='User requested continuous PPO; defer independent validation until after training.'))
    try:
        if args.pilot_only:
            manifest=collect(root,64,warmup=0)
            assert manifest['count']==128
            status.update(status='pilot_passed',samples=128)
            write_json(root/'status.json',status);return
        # A passed independent 64-sample-per-seed pilot is required before this entry.
        pilot=json.loads((root.parent/'corrected_uniform_bc_pilot_v3/status.json').read_text())
        assert pilot['status']=='pilot_passed'
        if not args.continue_from_bc:
            collect(root,4000)
            status['status']='bc_training';write_json(root/'status.json',status)
            checkpoint=train_bc(root)
        evaluations=[]
        def paired(cp,iteration):
            status.update(status='argmax_validation',iteration=iteration);write_json(root/'status.json',status)
            pair={seed:evaluate(root,cp,iteration,seed,baseroot) for seed in [42,43]}
            eligible,checks=qualify(pair,baseline)
            row=dict(iteration=iteration,checkpoint=str(cp),sha256=sha(cp),eligible=eligible,checks=checks,metrics=pair)
            evaluations.append(row);write_json(root/'checkpoint_evaluations.json',evaluations)
            return row
        paired(checkpoint,0)
        out=root/'ppo';out.mkdir(exist_ok=False)
        cmd=options(train_command(SIM,out,'full',801,smoke=False),{
            'device':'cpu','init-checkpoint':checkpoint,'independent-random-init':0,'iterations':100,
            'episode-horizon':1512,'topology-seed':42,'topology-seed-mode':'list_cycle','seed-list':'42,43',
            'target-kl':.05,'target-kl-min-updates':4,'candidate-checkpoint-interval':4,
            'candidate-checkpoint-require-gate':0,'early-stop-patience':0})
        env=environment()
        write_json(out/'command.json',dict(command=cmd,env={k:v for k,v in env.items() if k.startswith('CUMAC_')}))
        with (out/'run.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status.update(status='ppo',pid=proc.pid);write_json(root/'status.json',status)
            completed=set()
            try:
                while proc.poll() is None:
                    time.sleep(5)
                    path=out/'online_ppo_metrics.csv'
                    if not path.exists():continue
                    try:rows=list(csv.DictReader(path.open()))
                    except (OSError,csv.Error):continue
                    schedule=json.loads((root/'validation_schedule.json').read_text())
                    due=[r for r in rows if schedule.get('evaluate_during_ppo',False) and r.get('candidate_actor_checkpoint') and int(float(r['iter'])) in {8,20,40,60,80,100} and int(float(r['iter'])) not in completed]
                    for row in due:
                        iteration=int(float(row['iter']))
                        cp=Path(row['candidate_actor_checkpoint'])
                        if not cp.is_file():continue
                        paused=False
                        if proc.poll() is None:
                            try:
                                os.killpg(proc.pid,signal.SIGSTOP)
                                paused=True
                            except ProcessLookupError:
                                proc.wait()
                        try:
                            # CSV is written only after both actor and train checkpoints are saved.
                            import torch
                            saved=torch.load(cp,map_location='cpu',weights_only=False)
                            assert all(torch.isfinite(t).all() for t in saved['actor_state'].values())
                            for r in rows:
                                for key in ['policy_loss','value_loss','approx_kl_mean','rollout_reward_mean']:
                                    assert np.isfinite(float(r[key])),(iteration,key)
                                assert int(float(r['topology_seed']))==[42,43][(int(float(r['iter']))-1)%2]
                            result=paired(cp,iteration)
                            completed.add(iteration)
                            if iteration==8:
                                write_json(root/'first8_review.json',dict(decision='continue_same_live_state_to_budget',
                                    reason='finite weights/metrics; seed alternation verified; paired argmax evidence retained; non-superiority is not invalidity',evaluation=result))
                        finally:
                            if paused and proc.poll() is None:
                                try:os.killpg(proc.pid,signal.SIGCONT)
                                except ProcessLookupError:proc.wait()
                        status.update(status='ppo',iteration=iteration);write_json(root/'status.json',status)
                if proc.returncode:raise RuntimeError(f'PPO exited {proc.returncode}')
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGCONT);os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=30)
        if 100 not in completed:
            paired(out/'candidate_checkpoints/ppo_actor_iter_0100.pt',100)
        eligible=[r for r in evaluations if r['eligible']]
        def rank(r):
            return (-min(r['metrics'][s]['goodput_mbps']/baseline[str(s)]['goodput_mbps'] for s in [42,43]),
                    max(r['metrics'][s]['expiry_bytes_over_offered']-baseline[str(s)]['expiry_bytes_over_offered'] for s in [42,43]),
                    max(r['metrics'][s]['mac_completion_or_expiry_outcome_delay_ms_p95'] for s in [42,43]),r['iteration'])
        chosen=min(eligible,key=rank) if eligible else evaluations[-1]
        write_json(root/'frozen_checkpoint.json',dict(**chosen,selection='qualified' if eligible else 'no_eligible_model_last_for_diagnosis'))
        for seed in [42,43]:
            evaluate(root,Path(chosen['checkpoint']),chosen['iteration'],seed,baseroot,final=True)
            relative=f"iter{chosen['iteration']:04d}_seed{seed}"
            assert sha(root/'frozen'/relative/'service_balance_trace.csv')==sha(root/'validation'/relative/'service_balance_trace.csv'), 'frozen replay changed'
        final_report(root,baseroot,evaluations,chosen)
        from .corrected_uniform_complete_reports import complete_dpa_manifests
        complete_dpa_manifests(root,baseroot)
        from .corrected_uniform_finalize import seal
        seal(root)
        status.update(status='complete',eligible_model_count=len(eligible),frozen_checkpoint_sha256=chosen['sha256'],finished=time.time())
        write_json(root/'status.json',status)
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());write_json(root/'status.json',status);raise

if __name__=='__main__':main()

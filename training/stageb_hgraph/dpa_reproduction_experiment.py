"""Fixed, gated DPA training-path reproduction. No automatic protocol tuning."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import time
import traceback

import numpy as np

from .dpa_ablation_experiment import (
    ROOT, SIM_ARGS, REFERENCE_SIM_ARGS, METRICS, base_env, cli, evaluate,
    read_csv, run_command, sha, summarize_run, train_command, write_json,
)
from .pfq_dpa_aligned_comparison import validate_seed_artifacts
from .pfq_reference_calibration import _extend_ld_library_path

OLD = ROOT / 'output/pfq_independent_reuse/formal/deadline_dimensionless_features_seed45_v1/train_attempt2/ppo_actor_last.pt'
OLD_SHA = '94a2e40f0c0a56632e0c0ca63af4fa54d894d35aea38c2dd34afc0cfeadf2d7a'
ARMS = ('full', 'no_reuse_attention', 'no_dimensionless', 'no_reward')
CONFIG = ROOT / 'Doc/dpa_training_reproduction_ablation_v1_protocol.json'
SOURCE_SUFFIXES={'.py','.cu','.cuh','.cpp','.h','.hpp'}
EXPECTED_E_REWARD_CONTRACT={
    'gamma':.99,'independent_offered_bytes_per_tti':108000.,
    'independent_goodput_weight':1.,'independent_failed_weight':1.,'independent_expired_weight':2.,
    'deadline_tail_eta':.25,'deadline_tail_window_steps':100,'deadline_tail_slot_duration_s':.0005,
    'deadline_tail_epsilon_bytes':1.,'deadline_tail_tail_fraction':.1,
    'deadline_tail_rate_weight':.25,'deadline_tail_rate_tail_fraction':.1,
    'deadline_tail_rate_epsilon_mbps':1e-6}


def max_schedulable_zero_dequeue_streak(rows,first_tti=1001,last_tti=4000):
    """Carry qualifying per-UE streaks through warmup, report maxima in the window."""
    streaks={}; maximum=0
    for row in sorted(rows,key=lambda r:(int(r['tti']),int(r['ue_id']))):
        tti=int(row['tti']); ue=int(row['ue_id'])
        eligible=(int(row['pre_queue_bytes'])>0 and int(row['serving_cell'])>=0 and
                  int(row['demand_valid_prg_count'])>0 and int(row['scheduled_dequeue_bytes'])==0)
        streaks[ue]=streaks.get(ue,0)+1 if eligible else 0
        if first_tti<=tti<=last_tti:maximum=max(maximum,streaks[ue])
    return maximum


def eval_identity(*,seed,ttl_tti,feature_mode,policy,checkpoint_sha256,binary_sha256,
                  source_fingerprint,protocol_sha256,mechanism_snapshots=False,dataset_sha256=None):
    return {'seed':int(seed),'ttl_tti':int(ttl_tti),'feature_mode':feature_mode,'policy':policy,
            'checkpoint_sha256':checkpoint_sha256,'binary_sha256':binary_sha256,
            'source_fingerprint':source_fingerprint,'protocol_sha256':protocol_sha256,
            'common_dataset_sha256':dataset_sha256,
            'window':{'total_tti':4000,'warmup_tti':1000,'measured_tti':3000},
            'decode':'argmax','ctbs_headroom':'ctbs-v1','ctbs_guard':'off',
            'traffic_profile':'homogeneous','mechanism_snapshots':bool(mechanism_snapshots)}


def bc_collection_command(sim,out,dataset,*,steps,feature_mode):
    return cli('online_imitation',{
        'sim-bin':sim,'sim-args':SIM_ARGS,'sim-cwd':out,'out-dir':out,
        'socket-path':f'/tmp/dpa_repro_collect_{os.getpid()}_{feature_mode}.sock',
        'connect-timeout-s':60,'episode-horizon':steps,'max-steps':steps,'episodes':1,
        'online-persistent':1,'topology-seed':41,'topology-seed-mode':'fixed','seed':801,
        'device':'cuda','hidden-dim':128,'message-layers':2,'dropout':0,'ue-prg-topk':4,
        'ue-prg-diversity-extra':0,'tb-err-ema-alpha':.1,'action-head-mode':'independent',
        'candidate-mode':'all_demand_ues','include-pfq-score-anchor':0,
        'teacher-mode':'pfq_passthrough','lr':3e-4,'weight-decay':1e-5,'grad-clip':5,
        'save-every-episodes':1,'print-every':500,'independent-headroom-accounting':'ctbs-v1',
        'independent-ablation':'full','independent-bc-dataset-out':dataset,
        'independent-bc-collect-only':1,'sim-env':f'CUMAC_DPA_FEATURE_MODE={feature_mode}'})


def wait_for_p0(root,poll_seconds=30):
    status_path=Path(root)/'status.json'
    while True:
        if status_path.exists():
            status=json.loads(status_path.read_text())
            if status.get('status')=='p0_references_complete':return status
            if status.get('status')=='failed' or str(status.get('status','')).startswith('stopped'):
                raise RuntimeError('P0 failed; refusing after-p0 continuation: '+str(status.get('error','')))
            pid=int(status.get('pid',-1))
            if pid>0:
                try:os.kill(pid,0)
                except ProcessLookupError:raise RuntimeError('P0 status is stale; recorded process is not alive')
        time.sleep(poll_seconds)


def source_files():
    roots=(ROOT/'training/stageb_hgraph',ROOT/'training/gnnrl',ROOT/'cuMAC/examples/onlineTrainBridge',
           ROOT/'cuMAC/examples/trafficModel',ROOT/'cuMAC/src/4T4R')
    files={p for root in roots for p in root.rglob('*') if p.is_file() and p.suffix in SOURCE_SUFFIXES}
    files.update((ROOT/'cuMAC/examples/network.cu',ROOT/'cuMAC/examples/network.h',
                  ROOT/'cuMAC/examples/multiCellSchedulerUeSelection/main.cpp',
                  ROOT/'cuMAC/scripts/run_stageB_pfq_independent_reuse.sh'))
    return sorted(p for p in files if p.is_file())


def archive_current_source(destination):
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=False)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    diff=subprocess.check_output(['git','diff','--binary','HEAD','--',*[str(p.relative_to(ROOT)) for p in source_files()]],cwd=ROOT)
    (destination/'git_diff.patch').write_bytes(diff)
    manifest=[]
    with tarfile.open(destination/'current_sources.tar.gz','w:gz') as archive:
        for path in source_files():
            rel=path.relative_to(ROOT); archive.add(path,arcname=str(rel)); manifest.append({'path':str(rel),'sha256':sha(path)})
    payload={'git_head':head,'files':manifest,'git_diff_sha256':sha(destination/'git_diff.patch'),
             'historical_dirty_patch_available':False,
             'note':'Current executable source snapshot; not claimed to reproduce unavailable historical dirty runtime.'}
    write_json(destination/'manifest.json',payload)
    return payload


def source_fingerprint(manifest,protocol_sha256,binary_sha256):
    payload={'source':manifest,'protocol_sha256':protocol_sha256,'binary_sha256':binary_sha256}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_feature_records(legacy,dimensionless):
    if len(legacy)!=16 or len(dimensionless)!=16:raise ValueError('feature audit requires exactly 16 records per mode')
    for record in legacy:
        values=record['graph'].snapshot.ue_features.values.cpu().numpy()
        names=record['graph'].snapshot.ue_features.names
        if tuple(names[12:])!=('critical_ttl_backlog_bytes','near_ttl_backlog_bytes','age_weighted_backlog_bytes','service_gap_tti'):
            raise ValueError('legacy feature names mismatch')
        buffer,hol,slack=values[:,0],values[:,8],values[:,9]
        expected=np.column_stack((np.where((slack>=0)&(slack<=20),buffer,0),
                                  np.where((slack>=0)&(slack<=60),buffer,0),
                                  buffer*np.clip(hol/200.,0,2)))
        if not np.allclose(values[:,12:15],expected,rtol=1e-5,atol=1e-4):raise ValueError('legacy feature formula mismatch')
        if np.any(values[:,15]<0) or not np.all(np.isfinite(values[:,15])):raise ValueError('legacy service gap invalid')
    recent=None; last_service=None
    for index,record in enumerate(dimensionless):
        values=record['graph'].snapshot.ue_features.values.cpu().numpy(); n=len(values)
        names=record['graph'].snapshot.ue_features.names
        if tuple(names[12:])!=('log1p_drain_pressure','scheduled_dequeue_adequacy','ttl_normalized_service_age','reserved_zero'):
            raise ValueError('dimensionless feature names mismatch')
        if not np.allclose(values[:,15],0):raise ValueError('dimensionless reserved slot is nonzero')
        if index==0:recent=np.zeros(n); last_service=np.full(n,-1)
        queue=np.maximum(values[:,0],0); slack=np.maximum(values[:,9]/.5,1)
        drain=recent*slack
        expected_pressure=np.where(queue>0,np.log1p(queue/(drain+1)),0)
        expected_adequacy=np.where(queue>0,np.clip(drain/np.maximum(queue,1),0,1),0)
        tti=int(record['metadata']['tti'])
        # RESET tti=0 already follows the first traffic Update (traffic clock 1).
        # Subsequent observation extras are the previous action's post-result
        # snapshot, at the response TTI; dequeue events use that same clock.
        observation_tti=max(tti,1)
        expected_age=np.where(queue>0,np.clip(np.where(last_service>=0,observation_tti-last_service,observation_tti)/400.,0,1),0)
        if not np.allclose(values[:,12],expected_pressure,rtol=1e-4,atol=1e-4):raise ValueError('drain pressure formula mismatch')
        if not np.allclose(values[:,13],expected_adequacy,rtol=1e-4,atol=1e-4):raise ValueError('dequeue adequacy formula mismatch')
        if not np.allclose(values[:,14],expected_age,rtol=1e-4,atol=1e-4):raise ValueError('service age formula mismatch')
        # Carry the exact bridge EWMA and most recent dequeue event into the next pre-action record.
        served=np.asarray(record['next_info']['actual_ue_served_bytes'],dtype=float)
        recent=.95*recent+.05*np.maximum(served,0)
        response_tti=int(record['next_info']['tti'])
        if response_tti!=tti+1:raise ValueError('BC response TTI sequence mismatch')
        last_service=np.where(served>0,response_tti,last_service)
    return {'passed':True,'legacy_records':16,'dimensionless_records':16,
            'legacy_formulas_checked':True,'dimensionless_formulas_checked':True,'slot15_zero':True}


def validate_same_prefix_records(records):
    import torch
    from dataclasses import replace
    from .eval_checkpoint_online import _load_actor_checkpoint
    from .graph_ops import pack_prg_candidates
    from .independent_policy import _run_policy,resolve_full_candidate_width
    actor,_=_load_actor_checkpoint(str(OLD),torch.device('cpu')); actor.eval()
    actors={'full':actor}
    for ablation in ('no_reuse_attention','no_dimensionless'):
        clone=deepcopy(actor); clone.cfg=replace(clone.cfg,ablation=ablation); clone.eval(); actors[ablation]=clone
    maxima={key:0. for key in actors}
    with torch.no_grad():
        for record in records:
            graph=record['graph']; packed=pack_prg_candidates(graph,max_candidates=resolve_full_candidate_width(graph),fixed_width=True)
            trace=record['teacher_trace']; outputs={}
            for name,model in actors.items():
                replay=_run_policy(model,graph,packed,mode='greedy',stored_trace=trace,collect_numerical_trace=True)
                if not torch.equal(replay.choice,trace.choice) or not torch.equal(replay.valid_at_step,trace.valid_at_step):
                    raise ValueError('same-prefix stored choices/masks did not replay exactly')
                outputs[name]=replay.raw_logits_at_step
            valid=trace.valid_at_step
            for name in actors:
                maxima[name]=max(maxima[name],float((outputs[name]-outputs['full'])[valid].abs().max()))
    return {'passed':True,'records':len(records),'effective_logit_max_abs_difference':maxima,
            'action_change_required':False,'online_counterfactual_service_claim':False}


def validate_reward_records(records):
    from .edge_generator import EdgeGeneratorConfig,SparseEdgeGenerator
    from .feature_snapshot import FeatureSnapshotBuilder
    from .independent_objectives import DeadlineTailPressureTracker
    from .independent_reward import IndependentRewardConfig,compute_independent_reward
    builder=FeatureSnapshotBuilder(); edges=SparseEdgeGenerator(EdgeGeneratorConfig(
        candidate_mode='all_demand_ues',include_pfq_score_anchor=False,include_candidate_risk_features=False))
    n=int(records[0]['graph'].snapshot.static_meta.n_ue)
    trackers={flag:DeadlineTailPressureTracker(n) for flag in (False,True)}
    differences=[]; removed_errors=[]; formula_errors=[]; component_errors=[]; all_finite=True
    shared_components=(
        'reward_goodput_bytes','reward_scheduled_bytes','reward_failed_bytes','reward_expired_bytes',
        'reward_goodput_raw','reward_failed_raw','reward_expired_raw',
        'reward_goodput_weighted','reward_failed_weighted','reward_expired_weighted',
        'reward_deadline_tail_phi_before','reward_deadline_tail_phi_after','reward_deadline_tail_phi_delta',
        'reward_deadline_tail_rate_before','reward_deadline_tail_rate_after','reward_deadline_tail_rate_delta',
    )
    for record in records:
        after=edges.build_graph(builder.build(record['next_obs']))
        results=[]
        for flag in (False,True):
            result=compute_independent_reward(record['graph'],after,record['next_info'],trackers[flag],
                                              IndependentRewardConfig(treatment='E',remove_deadline_reward=flag))
            results.append(result)
            all_finite &= all(math.isfinite(float(value)) for value in result.values())
        full,no_reward=results
        difference=full['reward_total']-no_reward['reward_total']
        expected_removed=(full['reward_expired_weighted']+full['reward_deadline_tail_weighted']+
                          full['reward_deadline_tail_rate_weighted'])
        expected_no_reward=(
            (full['reward_goodput_bytes']-full['reward_failed_bytes']) /
            float(IndependentRewardConfig(treatment='E').offered_bytes_per_tti)
        )
        differences.append(difference)
        removed_errors.append(abs(difference-expected_removed))
        formula_errors.append(abs(no_reward['reward_total']-expected_no_reward))
        component_errors.append(max(abs(full[key]-no_reward[key]) for key in shared_components))
    maxima=(max(removed_errors,default=math.inf),max(formula_errors,default=math.inf),
            max(component_errors,default=math.inf))
    if not all_finite or any(error>1e-6 for error in maxima):
        raise ValueError('real-transition reward ablation identity failed: '
                         f'removed={maxima[0]} formula={maxima[1]} shared={maxima[2]} finite={all_finite}')
    return {'passed':True,'real_transition_records':len(records),'full_minus_no_reward':differences,
            'max_removed_terms_error':maxima[0],'max_no_reward_formula_error':maxima[1],
            'max_shared_raw_component_error':maxima[2],'all_finite':all_finite,'tolerance':1e-6,
            'reward_denominator':'fixed 108000 offered_bytes_per_tti; distinct from measured expiry metric actual offered bytes',
            'unit_nonzero_expiry_evidence':'test_remove_deadline_reward_subtracts_all_deadline_terms',
            'online_counterfactual_service_claim':False}


def acceptance(m, old, pfq, ttl_tti=400):
    p10_ref = old['ue_expiry_aware_rate_p10_mbps'] or pfq['ue_expiry_aware_rate_p10_mbps']
    checks = {
        'goodput_vs_pfq': m['goodput_mbps'] >= .99 * pfq['goodput_mbps'],
        'goodput_vs_old': m['goodput_mbps'] >= .99 * old['goodput_mbps'],
        'expiry_absolute': m['expiry_drop_rate'] <= .0005,
        'expiry_vs_pfq': m['expiry_drop_rate'] <= pfq['expiry_drop_rate'] + .0005,
        'mean_delay': m['outcome_delay_mean_ms'] <= max(1.25 * old['outcome_delay_mean_ms'], old['outcome_delay_mean_ms'] + .5),
        'p95_delay': m['outcome_delay_p95_ms'] <= max(1.25 * old['outcome_delay_p95_ms'], old['outcome_delay_p95_ms'] + 1.),
        'ue_p10': m['ue_expiry_aware_rate_p10_mbps'] >= .9 * p10_ref,
        'backlog': m['end_backlog_bytes'] <= old['end_backlog_bytes'] + .01 * m['offered_bytes'],
        'no_full_ttl_starvation': m['max_backlogged_schedulable_zero_dequeue_streak_tti'] < ttl_tti,
    }
    return {'passed': all(checks.values()), 'checks': checks,
            'reference_p10_abnormal': old['ue_expiry_aware_rate_p10_mbps'] == 0}


def choose_checkpoint(rows, old, pfq):
    if not rows:
        raise ValueError('no structurally valid candidates')
    scored = [dict(row, gate=acceptance(row['metrics'], old, pfq)) for row in rows]
    def key(r):
        m=r['metrics']
        return (not r['gate']['passed'], m['outcome_delay_p95_ms'], m['outcome_delay_mean_ms'],
                -m['ue_expiry_aware_rate_p10_mbps'], -m['goodput_mbps'], r['iteration'])
    return min(scored, key=key)


def choose_pressure(rows):
    if not rows:
        raise ValueError('no PFQ calibration data')
    for row in rows:
        if .001 <= statistics.median(row['expiry_rates']) <= .05:
            return dict(row, classification='in_range')
    for row in rows:
        if statistics.median(row['expiry_rates']) > .001:
            return dict(row, classification='high_pressure')
    return dict(rows[-1], classification='insufficient_expiry_resolution')


def may_expand(gates):
    return len(gates) == 2 and all(g['passed'] for g in gates)


def offered_identity(folder):
    rows=read_csv(folder / 'traffic_offered_packets.csv')
    digest=hashlib.sha256()
    for r in rows:
        digest.update((','.join(r[k] for k in ('ue_id','packet_id','arrival_tti','packet_size_bytes'))+'\n').encode())
    return digest.hexdigest(), sorted({int(r['ttl_tti']) for r in rows})


def validate_reference_pair(pfq_dir,rrq_dir,seed,ttl_tti,binary_hash):
    manifests=[]
    for directory,policy,bflag in ((Path(pfq_dir),'pfq','2'),(Path(rrq_dir),'rrq','3')):
        manifest=json.loads((directory/'reference_manifest.json').read_text()); manifests.append(manifest)
        scenario=manifest['scenario']; command=[str(x) for x in manifest['command']]
        checks=(manifest['binary_sha256']==binary_hash and manifest['reference_policy']==policy and
                int(scenario['topology_seed'])==seed and int(scenario['traffic_seed'])==seed and
                float(scenario['packet_ttl_ms'])==ttl_tti*.5 and int(manifest['warmup_steps'])==0 and
                int(manifest['measure_steps'])==4000 and '-b' in command and
                command[command.index('-b')+1]==bflag)
        if not checks:raise ValueError(f'{policy} reference identity mismatch')
        rows=read_csv(directory/'service_balance_trace.csv')
        if len(rows)!=4000*36 or sorted({int(r['tti']) for r in rows})!=list(range(1,4001)):
            raise ValueError(f'{policy} service grid mismatch')
    if sha(Path(pfq_dir)/'ue_topology.csv')!=sha(Path(rrq_dir)/'ue_topology.csv'):
        raise ValueError('PFQ/RRQ topology mismatch')
    if offered_identity(pfq_dir)[0]!=offered_identity(rrq_dir)[0]:raise ValueError('PFQ/RRQ offered stream mismatch')
    return {'passed':True,'policies':['pfq','rrq'],'scheduler_bflags':[2,3]}


def validate_cross_condition_inputs(root,names):
    pairs=[]
    for seed in range(211,216):
        for name in names:
            standard=Path(root)/f'test/standard/{seed}/{name}'
            pressure=Path(root)/f'test/deadline_pressure/{seed}/{name}'
            stream_equal=offered_identity(standard)[0]==offered_identity(pressure)[0]
            topology_equal=sha(standard/'ue_topology.csv')==sha(pressure/'ue_topology.csv')
            row={'scene_seed':seed,'policy':name,'arrival_size_equal':stream_equal,
                 'topology_equal':topology_equal}
            if not stream_equal or not topology_equal:
                raise ValueError(f'cross-condition external inputs changed: seed={seed} policy={name}')
            pairs.append(row)
    return {'passed':True,'pair_count':len(pairs),'pairs':pairs,
            'comparison':'standard vs deadline_pressure; arrival/size hash excludes TTL by construction'}


def enrich_metrics(folder):
    result=summarize_run(folder, False)['steady']
    rows=read_csv(folder / 'service_balance_trace.csv')
    selected=[r for r in rows if 1001 <= int(r['tti']) <= 4000]
    result['max_backlogged_schedulable_zero_dequeue_streak_tti']=max_schedulable_zero_dequeue_streak(rows)
    result['offered_bytes']=sum(int(r['offered_bytes']) for r in selected)
    result['end_pre_action_max_hol_ms']=max(float(r['hol_delay_ms']) for r in selected if int(r['tti'])==4000)
    result['age_note']='HOL is pre-action at final TTI, not fabricated post-action age for pending packets'
    if result['terminal_outcome_count']==0 or result['offered_bytes']==0:
        raise ValueError('undefined service metrics: no terminal outcome or offered bytes')
    for name in (*METRICS, 'offered_bytes'):
        if not math.isfinite(result[name]):
            raise ValueError(f'nonfinite metric {name}')
    return result


class Campaign:
    def __init__(self, root, sim):
        self.root=root.resolve(); self.sim=sim.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.env=base_env(1); _extend_ld_library_path(self.env,self.sim)
        self.binary_hash=sha(self.sim)
        if sha(OLD)!=OLD_SHA: raise ValueError('frozen historical checkpoint changed')
        self.protocol=json.loads(CONFIG.read_text())
        self.protocol_hash=sha(CONFIG); self.source_fingerprint=None
        self.status={'status':'running','phase':'P0','pid':os.getpid(),'started_at':time.time(),'completed_jobs':[]}
        self.cache={}
        lock=self.root/'binary_lock.json'
        if lock.exists() and json.loads(lock.read_text())['sha256']!=self.binary_hash:
            raise ValueError('campaign binary changed')
        if not lock.exists(): write_json(lock,{'path':str(self.sim),'sha256':self.binary_hash})

    def _freeze(self):
        archive=self.root/'source_archive'
        manifest=(json.loads((archive/'manifest.json').read_text()) if archive.exists()
                  else archive_current_source(archive))
        fingerprint=source_fingerprint(manifest,self.protocol_hash,self.binary_hash)
        executable={'binary_sha256':self.binary_hash,'protocol_sha256':self.protocol_hash,
                    'bc_teacher_alignment':'ctbs_prefix_prune_v1',
                    'source_fingerprint':fingerprint,'source_archive_manifest':manifest,
                    'parent_cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                    'required_parent_cuda_visible_devices':'1',
                    'protocol_preexecution_fields_archival_only':{
                        'status':self.protocol.get('status'),'launch_allowed':self.protocol.get('launch_allowed')},
                    'runtime_gate_authority':'implementation_preflight.json passed plus status.json; frozen user thresholds unchanged'}
        if executable['parent_cuda_visible_devices']!='1':
            raise RuntimeError('campaign parent must start with CUDA_VISIBLE_DEVICES=1 before any CUDA use')
        path=self.root/'executable_manifest.json'
        if path.exists() and json.loads(path.read_text())!=executable:raise ValueError('executable manifest changed')
        if not path.exists():write_json(path,executable)
        self.source_fingerprint=fingerprint

    def _assert_frozen(self):
        if self.source_fingerprint is None:raise RuntimeError('executable source is not frozen')
        archive=json.loads((self.root/'source_archive/manifest.json').read_text())
        frozen={row['path']:row['sha256'] for row in archive['files']}
        current={str(path.relative_to(ROOT)):sha(path) for path in source_files()}
        if current!=frozen:raise ValueError('current source files changed after campaign freeze')
        if source_fingerprint(archive,sha(CONFIG),sha(self.sim))!=self.source_fingerprint:
            raise ValueError('source/protocol/binary changed after campaign freeze')

    def _collect_dataset(self,name,steps,feature_mode):
        dataset=self.root/name; out=self.root/'preflight'/('collect_'+name)
        if (dataset/'manifest.json').exists():return
        if dataset.exists() or out.exists():raise ValueError(f'partial collection exists: {name}')
        self.progress('preflight/collect_'+name)
        out.mkdir(parents=True,exist_ok=False)
        env={**self.env,'CUMAC_DPA_FEATURE_MODE':feature_mode}
        run_command(bc_collection_command(self.sim,out,dataset,steps=steps,feature_mode=feature_mode),out,env,7*86400)

    def prepare_common_data(self):
        import torch
        from .online_imitation import IndependentBcDatasetReader,_model_tensor_sha256
        from .independent_actor import IndependentActorConfig,PFQIndependentReuseActor
        reference=self.root/'preflight/standard/61/pfq'
        capture_job='preflight/mechanism_capture/61/pfq'
        self.evaluate(capture_job,61,policy='pfq',feature_mode='dimensionless',mechanism_snapshots=True)
        capture_dir=self.root/capture_job
        unchanged={name:sha(reference/name)==sha(capture_dir/name) for name in (
            'tti_trace.jsonl','service_balance_trace.csv','packet_outcome_samples.csv',
            'traffic_offered_packets.csv','ue_topology.csv')}
        snapshot_manifest=json.loads((capture_dir/'snapshots/manifest.json').read_text())
        if not all(unchanged.values()) or snapshot_manifest['ctbs_post_action_audit']['steps']!=4000:
            raise ValueError('mechanism capture changed native PFQ trajectory or lacks CTBS parity coverage')
        self._collect_dataset('bc_expert_data',4000,'legacy')
        self._collect_dataset('bc_dimensionless_probe',16,'dimensionless')
        checks={'native_pfq_mechanism_capture':{'passed':True,'artifact_hashes_unchanged':unchanged,
                                                'snapshot_manifest':snapshot_manifest}}; manifests={}
        for name,steps,mode in (('bc_expert_data',4000,'legacy'),('bc_dimensionless_probe',16,'dimensionless')):
            reader=IndependentBcDatasetReader(self.root/name,expected_metadata={
                'feature_stage':mode,'topology_seed':41,'model_seed':801,
                'teacher_alignment':'ctbs_prefix_prune_v1',
                'expected_episode_count':1,'expected_steps_per_episode':steps,
                'expected_record_count':steps})
            records=list(reader); manifests[name]=reader.manifest
            ckpt=torch.load(self.root/'preflight'/('collect_'+name)/'hgraph_independent_bc_last.pt',map_location='cpu',weights_only=False)
            config={key:value for key,value in ckpt['actor_config'].items()
                    if key in IndependentActorConfig.__dataclass_fields__}
            restored=PFQIndependentReuseActor(IndependentActorConfig(**config)); restored.load_state_dict(ckpt['actor_state'])
            unchanged=(_model_tensor_sha256(restored)==reader.manifest['initial_actor_sha256']==ckpt['initial_actor_sha256'])
            checks[name]={'records_validated':len(records),'collector_initial_unchanged':unchanged,
                          'teacher_alignment':'ctbs_prefix_prune_v1',
                          'pruned_choices':sum(r['metadata']['teacher_alignment']['dropped_count'] for r in records),
                          'ctbs_post_action_exact':all(r['metadata']['ctbs_post_action_exact'] for r in records),
                          'dataset_sha256':reader.manifest['dataset_sha256']}
            if not unchanged:raise ValueError('collect-only actor changed')
            if not checks[name]['ctbs_post_action_exact']:raise ValueError('BC teacher lacks exact post-action CTBS parity')
        legacy_records=list(IndependentBcDatasetReader(self.root/'bc_expert_data'))[:16]
        dimensionless_records=list(IndependentBcDatasetReader(self.root/'bc_dimensionless_probe'))
        checks['feature_semantics']=validate_feature_records(legacy_records,dimensionless_records)
        checks['architectural_rng_isolation']={
            'passed':True,'fact':'candidate validations are run only by evaluate subprocesses after each training process exits'}
        checks['reward_ablation']=validate_reward_records(legacy_records)
        checks['same_prefix_ablation']=validate_same_prefix_records(dimensionless_records)
        passed=all(row.get('collector_initial_unchanged',True) for row in checks.values())
        write_json(self.root/'implementation_preflight.json',{
            'passed':passed,'checks':checks,'datasets':manifests,
            'binary_sha256':self.binary_hash,'protocol_sha256':self.protocol_hash})
        return checks

    def validate_p0_migration(self):
        results=self.root/'p0_reference_results.json'
        if not results.exists():raise RuntimeError('P0 references are incomplete')
        audits={}
        for seed in (61,62,63):
            audits[str(seed)]=validate_seed_artifacts(
                seed=seed,dpa_dir=self.root/f'preflight/standard/{seed}/old_dpa',
                pfq_dir=self.root/f'preflight/standard/{seed}/pfq',expected_sim_sha256=self.binary_hash)
        write_json(self.root/'p0_provenance_migration.json',{
            'passed':all(a['aligned'] for a in audits.values()),'audits':audits,
            'provenance_strength':'validated artifacts from already-running pre-freeze P0; explicitly weaker source identity',
            'not_claimed_same_source_as_frozen_full':True})

    def progress(self, job):
        if self.source_fingerprint is not None:self._assert_frozen()
        if sha(self.sim)!=self.binary_hash: raise ValueError('binary changed during campaign')
        self.status.update(current_job=job,updated_at=time.time())
        write_json(self.root/'status.json',self.status)
        print(f'[{time.strftime("%F %T")}] {job}',flush=True)

    def evaluate(self, job, seed, ttl_tti=400, checkpoint=None, policy=None, feature_mode='dimensionless',mechanism_snapshots=False):
        self.progress(job); folder=self.root/job
        identity=eval_identity(seed=seed,ttl_tti=ttl_tti,feature_mode=feature_mode,policy=policy,
                  checkpoint_sha256=sha(checkpoint) if checkpoint else None,binary_sha256=self.binary_hash,
                  source_fingerprint=self.source_fingerprint or 'P0_pre_freeze_weaker_provenance',
                  protocol_sha256=self.protocol_hash,mechanism_snapshots=mechanism_snapshots,
                  dataset_sha256=(json.loads((self.root/'bc_expert_data/manifest.json').read_text())['dataset_sha256']
                                  if checkpoint is not None and sha(checkpoint)!=OLD_SHA and
                                  (self.root/'bc_expert_data/manifest.json').exists() else None))
        completion=folder/'result.json'
        if completion.exists():
            result=json.loads(completion.read_text())
            if result['identity']!=identity: raise ValueError(f'cached identity mismatch: {job}')
            return result['metrics']
        env={**self.env,'CUMAC_DPA_FEATURE_MODE':feature_mode,
             'CUMAC_PACKET_TTL_MS':str(ttl_tti*.5),'CUMAC_PACKET_TTL_TTI':'0'}
        if policy:
            folder.mkdir(parents=True,exist_ok=False)
            command=cli('pfq_reference_calibration',{
                'sim-bin':self.sim,'sim-args':REFERENCE_SIM_ARGS[policy],'sim-cwd':folder,
                'topology-seed':seed,'traffic-seed':seed,'traffic-profile':'homogeneous',
                'reference-policy':policy,'packet-ttl-ms':ttl_tti*.5,
                'warmup-steps':0,'measure-steps':4000,'service-balance-trace-first-tti':1,'out-dir':folder})+['--service-balance-audit']
            if mechanism_snapshots:command.append('--mechanism-snapshots')
            run_command(command,folder,env,14400)
        else:
            evaluate(self.sim,folder,checkpoint,seed,env,False)
            evaluated=json.loads((folder/'eval_summary.json').read_text())
            if evaluated['checkpoint_sha256']!=identity['checkpoint_sha256']:
                raise ValueError('saved checkpoint and evaluated checkpoint hash differ')
        offered_hash,ttls=offered_identity(folder)
        if ttls!=[ttl_tti]: raise ValueError(f'actual offered TTL differs: {ttls}')
        metrics=enrich_metrics(folder)
        write_json(completion,{'identity':identity,'metrics':metrics,'arrival_size_stream_sha256':offered_hash})
        self.status['completed_jobs'].append(job)
        return metrics

    def paired(self, actor_job, ref_job, seed):
        result=validate_seed_artifacts(seed=seed,dpa_dir=self.root/actor_job,pfq_dir=self.root/ref_job,
                                      expected_sim_sha256=self.binary_hash)
        a=json.loads((self.root/actor_job/'result.json').read_text())['identity']
        b=json.loads((self.root/ref_job/'result.json').read_text())['identity']
        if a['ttl_tti']!=b['ttl_tti']: raise ValueError('paired TTL mismatch')
        write_json(self.root/actor_job/'paired_pfq.json',result)
        return result

    def p0(self):
        self.status['phase']='P0'
        references={}
        for seed in (61,62,63):
            references[str(seed)]={}
            for policy in ('pfq','old_dpa'):
                job=f'preflight/standard/{seed}/{policy}'
                references[str(seed)][policy]=self.evaluate(job,seed,checkpoint=OLD if policy=='old_dpa' else None,
                                                            policy='pfq' if policy=='pfq' else None)
            self.paired(f'preflight/standard/{seed}/old_dpa',f'preflight/standard/{seed}/pfq',seed)
        rows=[]
        for ttl in (200,100,50,25):
            rates=[]
            for seed in (61,62,63):
                job=f'preflight/pressure_ttl{ttl}/{seed}/pfq'
                m=self.evaluate(job,seed,ttl_tti=ttl,policy='pfq'); rates.append(m['expiry_drop_rate'])
                ref=self.root/f'preflight/standard/{seed}/pfq'
                if offered_identity(self.root/job)[0]!=offered_identity(ref)[0]:
                    raise ValueError('TTL-only intervention changed arrival/size stream')
                if sha(self.root/job/'ue_topology.csv')!=sha(ref/'ue_topology.csv'):
                    raise ValueError('TTL-only intervention changed topology')
            rows.append({'ttl_tti':ttl,'expiry_rates':rates})
            write_json(self.root/'pressure_calibration_progress.json',{'rows':rows})
            if .001<=statistics.median(rates)<=.05: break
        write_json(self.root/'p0_reference_results.json',{'references':references,'pressure':choose_pressure(rows),'all_calibration_rows':rows})
        return references

    def stage_train(self, seed, arm, stage, parent=None):
        import torch
        from .independent_contract import state_dict_sha256
        job=f'train/{seed}/{arm}/{stage}'; self.progress(job); folder=self.root/job
        self._assert_frozen()
        dataset_hash=(json.loads((self.root/'bc_expert_data/manifest.json').read_text())['dataset_sha256']
                      if stage=='S0' else None)
        identity={'seed':seed,'arm':arm,'stage':stage,'feature_mode':'dimensionless' if stage=='S3' else 'legacy',
                  'parent_sha256':sha(parent) if parent else None,'dataset_sha256':dataset_hash,
                  'binary_sha256':self.binary_hash,'protocol_sha256':self.protocol_hash,
                  'source_fingerprint':self.source_fingerprint}
        complete=folder/'training_complete.json'
        if complete.exists():
            r=json.loads(complete.read_text()); path=Path(r['checkpoint'])
            if r.get('identity')!=identity:raise ValueError('cached training identity mismatch')
            if sha(path)!=r['sha256']: raise ValueError('cached training checkpoint changed')
            return path
        folder.mkdir(parents=True,exist_ok=False)
        mode='dimensionless' if stage=='S3' else 'legacy'
        env={**self.env,'CUMAC_DPA_FEATURE_MODE':mode}
        actor_arm=arm if arm!='no_reward' else 'full'
        if arm=='no_dimensionless' and stage!='S3': actor_arm='full'
        if stage=='S0':
            options={
                'sim-bin':self.sim,'sim-args':SIM_ARGS,'sim-cwd':folder,'out-dir':folder,
                'socket-path':f'/tmp/dpa_repro_bc_{os.getpid()}.sock','connect-timeout-s':60,
                'episode-horizon':4000,'max-steps':4000,'episodes':1,'online-persistent':1,
                'topology-seed':41,'topology-seed-mode':'fixed','seed':seed,'device':'cuda',
                'hidden-dim':128,'message-layers':2,'dropout':0,'ue-prg-topk':4,'ue-prg-diversity-extra':0,
                'tb-err-ema-alpha':.1,'action-head-mode':'independent','candidate-mode':'all_demand_ues',
                'include-pfq-score-anchor':0,'teacher-mode':'pfq_passthrough','lr':3e-4,
                'weight-decay':1e-5,'grad-clip':5,'save-every-episodes':1,'print-every':500,
                'independent-headroom-accounting':'ctbs-v1','independent-ablation':actor_arm}
            options['independent-bc-dataset-in']=self.root/'bc_expert_data'
            options['sim-env']='CUMAC_DPA_FEATURE_MODE=legacy'
            command=cli('online_imitation',options)
            checkpoint=folder/'hgraph_independent_bc_last.pt'
        else:
            if parent is None: raise ValueError('PPO stage has no parent')
            source=torch.load(parent,map_location='cpu',weights_only=False)
            initial={k:deepcopy(source[k]) for k in ('actor_state','actor_config','graph_config')}
            initial['actor_config']['ablation']=actor_arm
            initial['derivation']={'parent_sha256':sha(parent),'feature_mode':mode,'stage':stage}
            init=folder/'init_actor.pt'; torch.save(initial,init)
            niter={'S1':4,'S2':100,'S3':8}[stage]
            command=train_command(self.sim,folder,actor_arm,seed,smoke=False)
            changes={'independent-random-init':0,'iterations':niter,'rollout-warmup-once':int(stage=='S1'),
                     'topology-seed':45 if stage=='S3' else 41,'topology-seed-mode':'list_cycle' if stage=='S2' else 'fixed',
                     'seed-list':'41,42' if stage=='S2' else '', 'target-kl':.05,'gamma':.99,
                     'reward-mode':'pfq_independent' if stage=='S1' else 'deadline_tail',
                     'independent-treatment':'B' if stage=='S1' else 'E','safety-prefix-iters':0,
                     'candidate-checkpoint-interval':4 if stage=='S2' else (2 if stage=='S3' else 0)}
            for name,value in changes.items(): command[command.index('--'+name)+1]=str(value)
            command+=['--init-checkpoint',str(init),'--candidate-checkpoint-require-gate','0',
                      '--candidate-checkpoint-iters','21' if stage=='S2' else '',
                      '--target-kl-min-updates','4',
                      '--independent-offered-bytes-per-tti','108000',
                      '--independent-remove-deadline-reward',str(int(arm=='no_reward'))]
            checkpoint=folder/'ppo_actor_last.pt'
        run_command(command,folder,env,7*86400)
        ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if not all(torch.isfinite(t).all() for t in ckpt['actor_state'].values()): raise ValueError('nonfinite actor')
        if stage!='S0':
            train_args=ckpt.get('args',{})
            reward_contract=ckpt.get('reward_cfg',{})
            if any(not math.isclose(float(reward_contract.get(key,math.nan)),value,rel_tol=0,abs_tol=1e-12)
                   for key,value in EXPECTED_E_REWARD_CONTRACT.items()):
                raise ValueError('independent E reward coefficient/scale contract mismatch')
            expected_reward='pfq_independent' if stage=='S1' else 'deadline_tail'
            expected_treatment='B' if stage=='S1' else 'E'
            if (train_args.get('reward_mode')!=expected_reward or
                train_args.get('independent_treatment')!=expected_treatment or
                float(train_args.get('target_kl',-1))!=.05 or
                int(train_args.get('target_kl_min_updates',-1))!=4 or
                int(train_args.get('update_epochs',-1))!=4 or
                int(train_args.get('safety_prefix_iters',-1))!=0):
                raise ValueError('PPO stage hyperparameter contract mismatch')
            rows=read_csv(folder/'online_ppo_metrics.csv')
            if len(rows)!=niter: raise ValueError('PPO sampling budget incomplete')
            for i,row in enumerate(rows):
                warmup=0 if stage=='S1' and i else 1000
                if int(float(row['rollout_steps']))!=512 or int(float(row['rollout_warmup_steps']))!=warmup:
                    raise ValueError('PPO budget mismatch')
                expected_topology=(41 if stage=='S1' else (41,42)[i%2] if stage=='S2' else 45)
                if int(float(row['topology_seed']))!=expected_topology:raise ValueError('PPO topology sequence mismatch')
                updates=int(float(row['num_updates']))
                if not 4<=updates<=32:raise ValueError('PPO target-KL update bounds mismatch')
                if row['rollout_ctbs_reference_label_deletion_equal']!='True': raise ValueError('CTBS deletion mismatch')
                for key in ('rollout_ctbs_mcs_mismatch_count','rollout_ctbs_layer_mismatch_count','rollout_ctbs_tbs_mismatch_count'):
                    if float(row[key])!=0: raise ValueError(key)
        write_json(complete,{'checkpoint':str(checkpoint),'sha256':sha(checkpoint),'identity':identity,
                            'actor_tensor_sha256':state_dict_sha256(ckpt['actor_state']),'feature_mode':mode})
        return checkpoint

    def train_chain(self, seed, arm, full_record=None):
        refs=json.loads((self.root/'p0_reference_results.json').read_text())['references']['61']
        record={'seed':seed,'arm':arm,'stages':{}}
        parent=None
        for stage in ('S0','S1','S2','S3'):
            shared=(arm=='no_dimensionless' and stage!='S3') or (arm=='no_reward' and stage=='S0')
            if shared:
                record['stages'][stage]={**full_record['stages'][stage],'shared_from_full':True}
                parent=Path(record['stages'][stage]['selected_checkpoint']); continue
            checkpoint=self.stage_train(seed,arm,stage,parent)
            if stage in ('S0','S1'):
                candidates=[(0 if stage=='S0' else 4,checkpoint)]
            else:
                candidates=[(i,checkpoint.parent/'candidate_checkpoints'/f'ppo_actor_iter_{i:04d}.pt')
                            for i in self.protocol['selection'][stage+'_candidate_iterations']]
            rows=[]
            for iteration,path in candidates:
                job=f'validation/{seed}/{arm}/{stage}/iter{iteration:04d}'
                m=self.evaluate(job,61,checkpoint=path,feature_mode='dimensionless' if stage=='S3' else 'legacy')
                self.paired(job,'preflight/standard/61/pfq',61)
                rows.append({'iteration':iteration,'checkpoint':str(path),'sha256':sha(path),'metrics':m})
            chosen=choose_checkpoint(rows,refs['old_dpa'],refs['pfq'])
            parent=Path(chosen['checkpoint'])
            record['stages'][stage]={'selected_checkpoint':str(parent),'sha256':sha(parent),
                                     'selected':chosen,'candidates':rows}
            write_json(self.root/f'selection_{seed}_{arm}.json',record)
        record['checkpoint']=str(parent); record['sha256']=sha(parent)
        write_json(self.root/f'selection_{seed}_{arm}.json',record)
        return record

    def train_and_test(self):
        self.validate_p0_migration(); self.prepare_common_data(); self._freeze()
        ready=json.loads((self.root/'implementation_preflight.json').read_text())
        ready.update(source_fingerprint=self.source_fingerprint)
        write_json(self.root/'implementation_preflight.json',ready)
        if not ready.get('passed'):raise RuntimeError('generated real-state checks failed; no new Full training')
        self.status['phase']='P1'
        models={}; full=self.train_chain(801,'full'); models['801/full']=full
        refs=json.loads((self.root/'p0_reference_results.json').read_text())['references']
        gates=[]
        for seed in (62,63):
            job=f'acceptance/full801/{seed}'
            m=self.evaluate(job,seed,checkpoint=Path(full['checkpoint']))
            self.paired(job,f'preflight/standard/{seed}/pfq',seed)
            gates.append({'seed':seed,**acceptance(m,refs[str(seed)]['old_dpa'],refs[str(seed)]['pfq'])})
        write_json(self.root/'full801_acceptance.json',{'scenes':gates,'passed':may_expand(gates)})
        if not may_expand(gates):
            self.status.update(status='stopped_full801_acceptance_failed',current_job=None)
            return
        self.status['phase']='P2'
        for seed in (801,802,803):
            if seed!=801: models[f'{seed}/full']=self.train_chain(seed,'full')
            for arm in ARMS[1:]: models[f'{seed}/{arm}']=self.train_chain(seed,arm,models[f'{seed}/full'])
        write_json(self.root/'frozen_models.json',models)
        self.status['phase']='P3'
        ttl=json.loads((self.root/'p0_reference_results.json').read_text())['pressure']['ttl_tti']
        results={}
        for condition,ttl_tti in (('standard',400),('deadline_pressure',ttl)):
            for seed in (211,212,213,214,215):
                for name in ('pfq','rrq','old_dpa'):
                    job=f'test/{condition}/{seed}/{name}'
                    results[job]=self.evaluate(job,seed,ttl_tti,checkpoint=OLD if name=='old_dpa' else None,
                                               policy=name if name!='old_dpa' else None,
                                               mechanism_snapshots=name=='pfq')
                validate_reference_pair(self.root/f'test/{condition}/{seed}/pfq',
                                        self.root/f'test/{condition}/{seed}/rrq',seed,ttl_tti,self.binary_hash)
                self.paired(f'test/{condition}/{seed}/old_dpa',f'test/{condition}/{seed}/pfq',seed)
                for name,model in models.items():
                    checkpoint=Path(model['checkpoint'])
                    if sha(checkpoint)!=model['sha256']: raise ValueError('frozen model changed')
                    job=f'test/{condition}/{seed}/{name}'
                    results[job]=self.evaluate(job,seed,ttl_tti,checkpoint=checkpoint)
                    self.paired(job,f'test/{condition}/{seed}/pfq',seed)
                stream_hashes={name:offered_identity(self.root/f'test/{condition}/{seed}/{name}')[0]
                               for name in ('pfq','rrq','old_dpa')}
                stream_hashes.update({name:offered_identity(self.root/f'test/{condition}/{seed}/{name}')[0]
                                      for name in models})
                if len(set(stream_hashes.values()))!=1:raise ValueError('paired policies changed arrival/size stream')
                write_json(self.root/'test_results_partial.json',results)
        write_json(self.root/'test_results.json',results)
        cross_condition=validate_cross_condition_inputs(
            self.root,['pfq','rrq','old_dpa',*models.keys()])
        if cross_condition['pair_count']!=75:raise ValueError('cross-condition audit count mismatch')
        write_json(self.root/'cross_condition_external_input_audit.json',cross_condition)
        from .dpa_reproduction_mechanisms import finish_campaign
        finish_campaign(self,models,results)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root',type=Path,required=True); p.add_argument('--sim-bin',type=Path,required=True)
    p.add_argument('--phase',choices=('p0','all','train-test','after-p0'),default='all')
    args=p.parse_args()
    if args.phase!='p0' and os.environ.get('CUDA_VISIBLE_DEVICES')!='1':
        raise RuntimeError('launch training continuation with CUDA_VISIBLE_DEVICES=1 for parent and children')
    if args.phase=='after-p0':wait_for_p0(args.out_root)
    campaign=Campaign(args.out_root,args.sim_bin)
    try:
        if args.phase in ('p0','all'): campaign.p0()
        if args.phase!='p0': campaign.train_and_test()
        else: campaign.status.update(status='p0_references_complete',current_job=None)
    except BaseException:
        campaign.status.update(status='failed',error=traceback.format_exc()); raise
    finally:
        campaign.status['updated_at']=time.time(); write_json(campaign.root/'status.json',campaign.status)


if __name__=='__main__': main()

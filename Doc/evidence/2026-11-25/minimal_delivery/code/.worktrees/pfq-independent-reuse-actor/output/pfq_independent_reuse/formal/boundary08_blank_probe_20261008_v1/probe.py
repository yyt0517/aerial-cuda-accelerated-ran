"""Throwaway same-prefix, actual-PHY blank experiment; no training changes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from training.stageb_hgraph import corrected_uniform_full as c
from training.stageb_hgraph.bridge_env import BridgeEnvClient, BridgeEnvConfig
from training.stageb_hgraph.deadline_tail_observation_probe import _ctbs_inputs
from training.stageb_hgraph.edge_generator import EdgeGeneratorConfig, SparseEdgeGenerator
from training.stageb_hgraph.feature_snapshot import FeatureSnapshotBuilder, SnapshotBuilderConfig
from training.stageb_hgraph.phy_tbs_alignment_audit import PhyTbsConfig
from training.stageb_hgraph.same_state_scheduler_feasibility_spike import (
    decode_type0_teacher_action, _ctbs_action_capacity, _actual_chosen_from_mask)

OUT = Path(__file__).resolve().parent
TARGETS = [1001, 1101, 1201]
torch.set_num_threads(1)

def digest_arrays(values):
    h = hashlib.sha256()
    for k, v in sorted(values.items()):
        a = np.asarray(v)
        h.update(k.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def candidates(chosen, ctbs, backlog, quality, tti):
    base = np.asarray(_ctbs_action_capacity(ctbs, chosen, backlog,
        n_cell=3, n_prg=17)['demand_capped_service_bytes_by_ue'])
    rows = []
    for width in [1, 4]:
        for cell in range(3):
            for start in range(18-width):
                prgs = list(range(start, start+width))
                tokens = [cell*17+p for p in prgs]
                if np.any(chosen[tokens] < 0):
                    continue
                modified = chosen.copy(); modified[tokens] = -1
                after = np.asarray(_ctbs_action_capacity(ctbs, modified, backlog,
                    n_cell=3, n_prg=17)['demand_capped_service_bytes_by_ue'])
                loss = float(np.sum(base-after))
                exposure = 0.
                for p in prgs:
                    for neighbor in range(3):
                        ue = int(chosen[neighbor*17+p])
                        if neighbor != cell and ue >= 0:
                            exposure += float(1/(1+np.maximum(quality[ue,p], 0).mean()))
                rows.append(dict(tti=tti, cell=cell, prgs=prgs, width=width,
                    predicted_dequeue_loss_bytes=loss, weak_neighbor_exposure=exposure))
    selected = []
    for width in [1,4]:
        best = min((r for r in rows if r['width']==width),
            key=lambda r:(r['predicted_dequeue_loss_bytes'], -r['weak_neighbor_exposure'], r['cell'], r['prgs']))
        selected.append(dict(best, kind=f'targeted{width}', name=f't{tti}_targeted{width}'))
    singles = [r for r in rows if r['width']==1 and
        (r['cell'],r['prgs']) != (selected[0]['cell'],selected[0]['prgs'])]
    random = singles[int(np.random.default_rng(10000+tti).integers(len(singles)))]
    selected.append(dict(random, kind='random1', name=f't{tti}_random1'))
    return rows, selected

def run(name, intervention=None):
    out = OUT/name; out.mkdir(exist_ok=False)
    horizon = intervention['tti'] if intervention else max(TARGETS)
    socket = f'/tmp/b08blank_{os.getpid()}.sock'
    env = c.eval_env(c.environment(), out, 42, socket)
    env.update(CUDA_VISIBLE_DEVICES='1', CUMAC_UE_PLACEMENT_MODE='coop_boundary',
        CUMAC_TRAFFIC_ARRIVAL_RATE='0.8', CUMAC_PFQ_DEMAND_CAP='1',
        CUMAC_PFQ_DEMAND_CAP_SLACK_BYTES='3000', CUMAC_DPA_FEATURE_MODE='dimensionless',
        OPENBLAS_NUM_THREADS='1')
    command = [str(c.SIM), '-d','1','-b','2','-x','0','-f','0','-g','100','-r','3000']
    c.write_json(out/'command.json', dict(command=command, env={k:v for k,v in env.items() if k.startswith('CUMAC_')}, intervention=intervention))
    builder = FeatureSnapshotBuilder(SnapshotBuilderConfig(tb_err_ema_alpha=.10))
    edges = SparseEdgeGenerator(EdgeGeneratorConfig(ue_prg_topk=4, ue_prg_diversity_extra=0,
        candidate_mode='all_demand_ues', include_pfq_score_anchor=False, include_candidate_risk_features=False))
    hashes = []; selections = []; all_candidates = []; outcomes = []
    reference = json.loads((OUT/'reference/summary.json').read_text()) if intervention else None
    client = None
    with (out/'sim.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            client = BridgeEnvClient(BridgeEnvConfig(socket_path=socket, connect_timeout_s=60, enable_ctbs=True))
            obs, _, info = client.reset(seed=42, episode_horizon=horizon)
            assert info['harq_enabled']==0
            config = PhyTbsConfig.from_vector(info['ctbs_config'])
            for tti in range(1,horizon+1):
                state_hash = digest_arrays(obs); hashes.append(state_hash)
                if reference:
                    assert state_hash == reference['state_hashes'][tti-1], (name,tti,'pre-state differs')
                snapshot = builder.build(obs)
                ue = obs['teacher_action_ue_select'].copy()
                prg = obs['teacher_action_prg_alloc'].copy()
                chosen = decode_type0_teacher_action(ue,prg,n_cell=3,n_prg=17)
                ctbs = _ctbs_inputs(client, info, config)
                if tti in TARGETS:
                    graph = edges.build_graph(snapshot)
                    backlog = snapshot.ue_features.values[:,0].numpy()
                    ctbs_hash = digest_arrays(dict(wb=ctbs.wb_sinr_first_layer_db,
                        olla=ctbs.olla_delta, previous_tb=ctbs.prior_tb_err, layers=ctbs.prg_layer_cap))
                    if reference:
                        assert ctbs_hash == reference['ctbs_hashes'][str(tti)]
                    else:
                        rows, selected = candidates(chosen,ctbs,backlog,obs['post_eq_sinr'],tti)
                        all_candidates.extend(rows); selections.extend(selected)
                        torch.save(dict(graph=graph,ctbs=ctbs,chosen=chosen,obs=obs),out/f'state_{tti}.pt')
                    outcomes.append(dict(tti=tti,ctbs_hash=ctbs_hash))
                if intervention and tti==horizon:
                    for p in intervention['prgs']:
                        assert prg[p*3+intervention['cell']]>=0
                        prg[p*3+intervention['cell']]=-1
                    chosen = decode_type0_teacher_action(ue,prg,n_cell=3,n_prg=17)
                next_obs, _, done, ni = client.step(ue,prg)
                actual = _actual_chosen_from_mask(ni['actual_ue_prg_mask'],
                    np.argmax(obs['action_mask_cell_ue'],axis=0), n_cell=3,n_prg=17)
                assert np.array_equal(actual,chosen), (name,tti,'action mismatch')
                if tti in TARGETS:
                    outcomes[-1].update({k:np.asarray(ni[k]).tolist() for k in
                        ['actual_ue_goodput_bytes','actual_ue_served_bytes','actual_ue_tbs_capacity_bytes','actual_ue_prg_count']})
                obs,info = next_obs,ni
                if tti%500==0: print(name,tti,flush=True)
        finally:
            if client: client.close()
            try: returncode=proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.terminate();returncode=proc.wait(timeout=15)
            assert returncode==0,(name,returncode)
    summary = dict(name=name, horizon=horizon, state_hashes=hashes, outcomes=outcomes,
        ctbs_hashes={str(r['tti']):r['ctbs_hash'] for r in outcomes}, selections=selections)
    if not intervention:
        c.write_json(OUT/'candidates.json',all_candidates)
        c.write_json(OUT/'selected_interventions.json',selections)
    else:
        ref = OUT/'reference'
        assert c.sha(out/'ue_topology.csv')==c.sha(ref/'ue_topology.csv')
        for filename,key in [('traffic_offered_packets.csv','arrival_tti'),('service_balance_trace.csv.channel.csv','tti')]:
            a=pd.read_csv(out/filename).query(f'{key}<={horizon}').reset_index(drop=True)
            b=pd.read_csv(ref/filename).query(f'{key}<={horizon}').reset_index(drop=True)
            pd.testing.assert_frame_equal(a,b,check_exact=True)
        a=pd.read_csv(out/'service_balance_trace.csv');b=pd.read_csv(ref/'service_balance_trace.csv')
        pd.testing.assert_frame_equal(a.query(f'tti<{horizon}').reset_index(drop=True),
            b.query(f'tti<{horizon}').reset_index(drop=True),check_exact=True)
    c.write_json(out/'summary.json',summary)
    print('COMPLETE',name,flush=True)
    return summary

def analyze():
    baseline=pd.read_csv(OUT/'reference/service_balance_trace.csv')
    prior=OUT.parent/'boundary_load_probe_20260930_v1/lambda_0.8/seed42_pfq'
    previous=pd.read_csv(prior/'service_balance_trace.csv')
    pd.testing.assert_frame_equal(baseline.query('tti<=1201').reset_index(drop=True),
        previous.query('tti<=1201').reset_index(drop=True),check_exact=True)
    result=[]; details=[]
    specs=json.loads((OUT/'selected_interventions.json').read_text())
    if (OUT/'risk_interventions.json').exists():
        extra=json.loads((OUT/'risk_interventions.json').read_text())
        if all((OUT/s['name']/'summary.json').exists() for s in extra):specs+=extra
    for spec in specs:
        a=pd.read_csv(OUT/spec['name']/'service_balance_trace.csv')
        a=a[a.tti==spec['tti']].sort_values('ue_id').reset_index(drop=True)
        b=baseline[baseline.tti==spec['tti']].sort_values('ue_id').reset_index(drop=True)
        assert len(a)==len(b)==36
        own=a.serving_cell==spec['cell']; neighbor=~own
        for frame in [a,b]:
            frame['expected_success']=frame.scheduled_dequeue_bytes*(1-frame.predicted_bler_cpu.clip(lower=0))
            assert np.allclose(frame.successful_bytes+frame.failed_bytes,frame.scheduled_dequeue_bytes)
        assert np.array_equal(a.pre_queue_bytes,b.pre_queue_bytes)
        assert np.array_equal(a.expired_bytes,b.expired_bytes)
        delta=a.successful_bytes-b.successful_bytes
        edelta=a.expected_success-b.expected_success
        active_neighbor=neighbor&(a.tb_tx_count>0)&(b.tb_tx_count>0)
        assert (a.loc[neighbor,'actual_prg_mask'].values==b.loc[neighbor,'actual_prg_mask'].values).all()
        assert (a.loc[neighbor,'tbs_capacity_bytes'].values==b.loc[neighbor,'tbs_capacity_bytes'].values).all()
        assert (a.loc[neighbor,'mcs'].values==b.loc[neighbor,'mcs'].values).all()
        assert (a.loc[neighbor,'layer'].values==b.loc[neighbor,'layer'].values).all()
        assert (a.loc[neighbor,'scheduled_dequeue_bytes'].values==b.loc[neighbor,'scheduled_dequeue_bytes'].values).all()
        row=dict(**spec,neighbor_success_gain_bytes=float(delta[neighbor].sum()),
            own_success_loss_bytes=float(-delta[own].sum()),net_success_gain_bytes=float(delta.sum()),
            neighbor_expected_gain_bytes=float(edelta[neighbor].sum()),own_expected_loss_bytes=float(-edelta[own].sum()),
            net_expected_gain_bytes=float(edelta.sum()),
            neighbor_sinr_delta_db_mean=float((a.phy_sinr_db_cpu-b.phy_sinr_db_cpu)[active_neighbor].mean()),
            own_dequeue_loss_bytes=float((b.scheduled_dequeue_bytes-a.scheduled_dequeue_bytes)[own].sum()),
            net_dequeue_delta_bytes=float((a.scheduled_dequeue_bytes-b.scheduled_dequeue_bytes).sum()),
            base_reward_GFE_delta=float((2*delta.sum()-(a.scheduled_dequeue_bytes-b.scheduled_dequeue_bytes).sum())/86400),
            expected_base_reward_GFE_delta=float((2*edelta.sum()-(a.scheduled_dequeue_bytes-b.scheduled_dequeue_bytes).sum())/86400))
        result.append(row)
        for u in range(36):
            details.append(dict(name=spec['name'],tti=spec['tti'],ue=u,cell=int(a.serving_cell[u]),
                own=bool(own[u]),success_delta_bytes=float(delta[u]),expected_delta_bytes=float(edelta[u]),
                sinr_base=float(b.phy_sinr_db_cpu[u]),sinr_blank=float(a.phy_sinr_db_cpu[u]),
                bler_base=float(b.predicted_bler_cpu[u]),bler_blank=float(a.predicted_bler_cpu[u]),
                dequeue_base=int(b.scheduled_dequeue_bytes[u]),dequeue_blank=int(a.scheduled_dequeue_bytes[u])))
    pd.DataFrame(result).to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(details).to_csv(OUT/'per_ue_comparison.csv',index=False)
    c.write_json(OUT/'verification.json',dict(passed=True,seed=42,targets=TARGETS,treatments=len(result),
        exact_pre_action_observation_prefix=True,exact_pre_action_ctbs=True,exact_pre_intervention_service_trace=True,
        exact_topology_traffic_channel=True,all_actions_applied_exactly=True,neighbor_allocation_tbs_dequeue_unchanged=True,
        original_pfq_reference_trace_exact_through_tti1201=True,actual_pre_queue_equal=True,
        neighbor_mcs_layer_unchanged=True,
        binary_sha256=c.sha(c.SIM),lib_sha256=c.sha(c.LIB)))
    print(pd.DataFrame(result).to_string(index=False),flush=True)

if __name__=='__main__':
    c.write_json(OUT/'protocol.json',dict(scope='throwaway actual-PHY mechanism probe',seed=42,
        placement='coop_boundary',arrival_rate=.8,pfq_slack_bytes=3000,targets=TARGETS,
        selection='Pre-outcome: minimize own CTBS demand-capped dequeue loss, then maximize neighbor 1/(1+pre SINR); contiguous widths1/4. Seeded random single control.',
        comparison='Each run follows identical PFQ prefix; one blank intervention at terminal TTI. Same actual native PHY, MCS/layer policy unchanged.',
        endpoints=['neighbor minus own actual successful bytes','neighbor minus own scheduled_bytes*(1-native predicted BLER)'],
        limits='Immediate mechanism only; not long-run expiry/goodput validation. No selection on realized TB errors.',
        stop='3 states x 3 interventions; separately diagnose frozen BC dataset/checkpoint. No PPO training.'))
    if not (OUT/'reference/summary.json').exists(): run('reference')
    for spec in json.loads((OUT/'selected_interventions.json').read_text()):
        if not (OUT/spec['name']/'summary.json').exists():run(spec['name'],spec)
    analyze()

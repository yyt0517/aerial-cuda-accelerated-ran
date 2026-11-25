"""Disposable seed-301 replay; diagnostics never change the executed action."""
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from output.pfq_independent_reuse.formal.boundary75_joint_long_residual_20261013_v1.policy import old, p, c, prior
from output.pfq_independent_reuse.formal.boundary75_old_joint_probe_20261013_v1.selector import candidates
from training.stageb_hgraph.directed_reuse import DirectedReuseCoordinator

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
FROZEN = OUT.parent / 'boundary75_joint_search_frozen_20261014_v1'
REF = FROZEN / 'snapshot/output/pfq_independent_reuse/formal/boundary75_joint_long_residual_20261013_v1/eval_s301_search'
SAMPLES = set(np.linspace(4001, 8000, 64, dtype=int).tolist())


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def evaluate(phy, actions):
    chunks = [phy.evaluate(actions[i:i+256]) for i in range(0, len(actions), 256)]
    return {k: np.concatenate([x[k] for x in chunks]) for k in chunks[0]}


def diagnose(tti, draft, keep, phy, obs, chosen):
    actions, meta = candidates(draft, keep, obs, phy.snapshot['queue'], True)
    pred = evaluate(phy, actions)
    score = pred['expected'].sum(1)
    deq = pred['dequeue'].sum(1)
    eligible = deq >= deq[0]
    best = int(np.argmax(np.where(eligible, score, -np.inf)))
    if score[best] <= score[0] + 1:
        best = 0
    assert np.array_equal(actions[best], chosen), (tti, 'frozen policy reproduction')
    rejected_better = (~eligible) & (score > score[best] + 1)
    unrestricted = int(score.argmax())
    row = dict(tti=tti, kind=meta[best][0], candidates=len(actions),
               unique_candidates=len({a.tobytes() for a in actions}),
               rejected=int((~eligible).sum()), rejected_better=int(rejected_better.sum()),
               keep_score=float(score[0]), selected_score=float(score[best]),
               selected_gain=float(score[best]-score[0]),
               unrestricted_gain=float(score[unrestricted]-score[best]),
               unrestricted_kind=meta[unrestricted][0],
               unrestricted_dequeue_loss_vs_keep=float(max(0, deq[0]-deq[unrestricted])),
               keep_dequeue=float(deq[0]), selected_dequeue=float(deq[best]))
    for epsilon in [.001, .01, .05]:
        allowed = deq >= (1-epsilon)*deq[0]
        row[f'epsilon_{epsilon}_gain'] = float(score[allowed].max()-score[best])
    # Pure second-round probe: center changes, original K feasibility reference remains.
    second, _ = candidates(chosen, chosen, obs, phy.snapshot['queue'], True)
    p2 = evaluate(phy, second)
    s2, q2 = p2['expected'].sum(1), p2['dequeue'].sum(1)
    for label, threshold in [('original_keep', deq[0]), ('selected', deq[best])]:
        i = int(np.argmax(np.where(q2 >= threshold, s2, -np.inf)))
        prior.encode(second[i], obs)
        row[f'second_gain_{label}'] = float(s2[i]-score[best])
        row[f'second_added_blanks_{label}'] = int(((draft >= 0) & (second[i] < 0)).sum())
    # Selected blank's marginal value with the selected UE reassignment held fixed.
    filled = chosen.copy()
    removed = (draft >= 0) & (chosen < 0)
    filled[removed] = draft[removed]
    fill_pred = phy.evaluate(filled)
    row['blank_marginal_bytes'] = float(score[best]-fill_pred['expected'].sum())
    dominant = np.where(np.arange(3)[None, None, :] != phy.serving[:, None, None], phy.received, -1).argmax(-1)
    ue_rows = []
    for u in range(36):
        used = chosen[phy.serving[u]] == u
        main_off = chosen[dominant[u], np.arange(17)] < 0
        main_new_off = removed[dominant[u], np.arange(17)]
        ue_rows.append(dict(tti=tti, ue_id=u,
            main_interferer_off_any=bool(np.any(used & main_off)),
            main_interferer_newly_blank_any=bool(np.any(used & main_new_off)),
            **{k: float(pred[k][best, u]) for k in ['count','layer','mcs','dequeue','sinr_db','bler','expected']}))
    return row, ue_rows


def run():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    # Audit all archived Python source/table inputs before importing them into evidence.
    provenance = {}
    for archived in (FROZEN/'snapshot').rglob('*'):
        if archived.is_file() and (archived.suffix == '.py' or archived.name == 'cumac.h'):
            rel = archived.relative_to(FROZEN/'snapshot')
            assert c.sha(ROOT/rel) == c.sha(archived), str(rel)
            provenance[str(rel)] = c.sha(archived)
    out = OUT/'replay_s301'
    out.mkdir(exist_ok=False)
    prefix = np.load(REF/'prefix.npz')
    actor, _ = old._load_actor_checkpoint(old.CHECKPOINT, torch.device('cpu'))
    actor.eval()
    coordinator = DirectedReuseCoordinator()
    coordinator.load_state_dict(torch.load(prior.CHECKPOINT, map_location='cpu', weights_only=False)['model'])
    coordinator.eval()
    selection = json.loads((FROZEN/'selection.json').read_text())
    for name, digest in selection['selected_components'].items():
        assert c.sha(ROOT/name) == digest, name
    builder = p.FeatureSnapshotBuilder(p.SnapshotBuilderConfig(tb_err_ema_alpha=.1))
    edges = p.SparseEdgeGenerator(p.EdgeGeneratorConfig(ue_prg_topk=4, ue_prg_diversity_extra=0,
        candidate_mode='all_demand_ues', include_pfq_score_anchor=False, include_candidate_risk_features=False))
    launch = json.loads((REF/'launch.json').read_text())
    env = os.environ.copy()
    env.update(launch['env'])
    original_dir = str(Path(launch['env']['CUMAC_DIRECTED_SNAPSHOT_PATH']).parent)
    for key, value in list(env.items()):
        if key.startswith('CUMAC_') and value.startswith(original_dir):
            env[key] = str(out)+value[len(original_dir):]
    socket = f'/tmp/b75diag_{os.getpid()}.sock'
    env.update(CUMAC_ONLINE_SOCKET=socket, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    simulator = FROZEN/'snapshot/output/pfq_independent_reuse/formal/boundary75_continuous_coordination_20261012_v1/simulator'
    command = [str(simulator)] + launch['command'][1:]
    write('protocol.json', dict(seed=301, horizon=8000, sampled_ttis=sorted(SAMPLES),
        question='Do the dequeue gate and one-round budget exclude predicted improvements?',
        behavior='Replay frozen search actions verbatim; no learning or intervention.',
        d3='Second neighborhood around selected allocation; report original-K and selected-dequeue gates separately.',
        d4='Executed-action calibration only; actual same-state K outcome is unobserved.',
        d5='Main interferer by pre-action received covariance trace per UE/PRG; groups are descriptive.',
        stop='64 predetermined states plus existing trace accounting; no automatic ablations.',
        source_hashes=provenance, simulator_sha256=c.sha(simulator)))
    rows, ue_rows, olla = [], [], []
    started = time.monotonic()
    client = None
    with (out/'sim.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            client = p.BridgeEnvClient(p.BridgeEnvConfig(socket_path=socket, connect_timeout_s=60, enable_ctbs=True))
            obs, _, info = client.reset(seed=301, episode_horizon=8000)
            assert info['harq_enabled'] == 0
            config = p.PhyTbsConfig.from_vector(info['ctbs_config'])
            phy = old.CausalPhy(config)
            for tti in range(1,8001):
                assert p.digest_arrays(obs) == prefix['hashes'][tti-1], (tti, 'observation replay')
                ctbs = p._ctbs_inputs(client, info, config)
                assert ctbs.tti == tti-1
                chosen = prefix['actions'][tti-1].reshape(3,17)
                if tti in SAMPLES:
                    snapshot = old.read_snapshot(out/'pre_action.bin', ctbs.tti)
                    draft, keep, *_ = prior.policy(obs, ctbs, snapshot, builder, edges, actor, coordinator, phy)
                    row, uu = diagnose(tti, draft, keep, phy, obs, chosen)
                    rows.append(row); ue_rows.extend(uu)
                else:
                    builder.build(obs)
                if tti >= 4001:
                    delta = np.asarray(ctbs.olla_delta, np.float32)
                    prior_err = np.asarray(ctbs.prior_tb_err)
                    updated = np.where(prior_err==1, delta+np.asarray(ctbs.olla_delta_up,np.float32),
                        np.where(prior_err==0, delta-np.asarray(ctbs.olla_delta_down,np.float32),delta))
                    olla.extend(dict(tti=tti, ue_id=u, pre_delta=float(delta[u]), applied_delta=float(updated[u]),
                        wb_first_layer_db=float(ctbs.wb_sinr_first_layer_db[u])) for u in range(36))
                nxt, _, _, ni = client.step(prefix['ue'][tti-1], prefix['prg'][tti-1])
                actual = p._actual_chosen_from_mask(ni['actual_ue_prg_mask'],np.asarray(ctbs.serving_cell),n_cell=3,n_prg=17)
                assert np.array_equal(actual,chosen.reshape(-1)), (tti,'action replay')
                assert float(np.asarray(ni['actual_ue_goodput_bytes']).sum()) == prefix['goodput'][tti-1], (tti,'reward replay')
                obs,info = nxt,ni
                if tti%500==0 or tti in SAMPLES:
                    pd.DataFrame(rows).to_csv(OUT/'candidate_diagnostics.csv',index=False)
                    write('status.json',dict(phase='replaying',tti=tti,samples=len(rows),elapsed_s=time.monotonic()-started))
                    print(f'REPLAY {tti}/8000 samples={len(rows)} elapsed={time.monotonic()-started:.1f}s',flush=True)
        finally:
            if client: client.close()
            try: code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.terminate(); code = proc.wait(timeout=15)
            assert code == 0, code
    pd.DataFrame(ue_rows).to_csv(OUT/'sampled_ue_predictions.csv',index=False)
    pd.DataFrame(olla).to_csv(OUT/'olla.csv',index=False)
    actual = pd.read_csv(out/'service_balance_trace.csv')
    ref = pd.read_csv(REF/'service_balance_trace.csv',nrows=8000*36)
    pd.testing.assert_frame_equal(actual,ref,check_exact=True)
    for name,key in [('traffic_offered_packets.csv','arrival_tti'),('service_balance_trace.csv.channel.csv','tti')]:
        a,b = pd.read_csv(out/name),pd.read_csv(REF/name)
        pd.testing.assert_frame_equal(a[a[key]<=8000].reset_index(drop=True),b[b[key]<=8000].reset_index(drop=True),check_exact=True)
    assert c.sha(out/'ue_topology.csv') == c.sha(REF/'ue_topology.csv')
    assert len(rows)==64 and len(olla)==4000*36
    write('verification.json',dict(passed=True,exact_observation_action_reward_ttis=8000,
        exact_service_rows=len(actual),exogenous_inputs_exact=True,sampled_policy_reproductions=64,
        sampled_candidate_legality=True,elapsed_s=time.monotonic()-started))
    write('status.json',dict(phase='replay_complete',tti=8000,samples=64,elapsed_s=time.monotonic()-started))
    print('VERIFIED',flush=True)


if __name__ == '__main__':
    run()

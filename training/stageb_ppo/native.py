"""Adapter to the frozen continuous-channel native simulator.

The legacy CTBS message transports fresh PHY/MCS/queue diagnostics. It does
not invoke the CTBS scheduler, headroom mask, pruning, or action repair.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .reward import PacketReward

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / '.worktrees/pfq-independent-reuse-actor'
FORMAL = ARCHIVE / 'output/pfq_independent_reuse/formal'
REFERENCE = FORMAL / 'boundary08_steady_state_20261021_v1'
ABLATION = ROOT / 'deliverables/dpls_component_ablation_20261029_v1'
EXPECTED_SIMULATOR = 'a68f48668bc623c0bc793c452f1b0cd5d8943929eb26e8eefd25f776d180b615'


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def legacy():
    sys.path.insert(0, str(ARCHIVE)) if str(ARCHIVE) not in sys.path else None
    import training.stageb_hgraph
    hp = str(ARCHIVE / 'training/stageb_hgraph')
    if hp not in training.stageb_hgraph.__path__:
        training.stageb_hgraph.__path__.insert(0, hp)
    from output.pfq_independent_reuse.formal.boundary08_steady_state_20261021_v1 import run
    return run


def teacher():
    legacy()
    # These are frozen, repository-owned experiment modules with verified hashes.
    sys.path.insert(0, str(ABLATION))
    from policy import Ablation
    return Ablation('raw_global_equal')


class NativeEnv:
    def __init__(self, out, seed, horizon):
        self.out, self.seed, self.horizon = Path(out), seed, horizon
        self.out.mkdir(parents=True, exist_ok=False)
        self.b = legacy()
        self.r = self.b.r
        self.tti = 0
        self.started = time.monotonic()
        self.rewards, self.actions, self.input_hashes = [], [], []
        self.fifo = PacketReward()
        original = self.r.OLD / f'seed{seed}_pfq0_2000/launch.json'
        launch = json.loads(original.read_text())
        self.env = {k: v for k, v in os.environ.items() if not k.startswith('CUMAC_')}
        self.env.update(launch['env'])
        origin = str(Path(launch['env']['CUMAC_DIRECTED_SNAPSHOT_PATH']).parent)
        for k, v in list(self.env.items()):
            if k.startswith('CUMAC_') and v.startswith(origin):
                self.env[k] = str(self.out) + v[len(origin):]
        self.socket = f'/tmp/equalppo_{os.getpid()}_{seed}_{time.time_ns()}.sock'
        self.env.update(CUMAC_ONLINE_SOCKET=self.socket, CUDA_VISIBLE_DEVICES='0',
            CUMAC_SHADOW_FADING_STD_DB='3.0', CUMAC_FRESH_OBSERVATION_PATH=str(self.out/'fresh_observation.bin'),
            OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        assert self.env['CUMAC_PFQ_DEMAND_CAP_SLACK_BYTES'] == '0'
        assert self.env['CUMAC_TRAFFIC_PROFILE'] == 'homogeneous'
        assert not any(k.startswith('CUMAC_LAST_SLOT_DROP') for k in self.env)
        command = [str(REFERENCE/'simulator'), *launch['command'][1:]]
        assert sha(command[0]) == EXPECTED_SIMULATOR
        linked = subprocess.check_output(['ldd', command[0]], env=self.env, text=True)
        library = next(x.strip() for x in linked.splitlines() if 'libcumac.so =>' in x)
        assert library.startswith('libcumac.so => ' + str(ARCHIVE/'build.corrected_uniform_v1/cuMAC/src/libcumac.so') + ' (')
        self.sources = {str(p): sha(p) for p in Path(__file__).parent.glob('*.py')}
        self.sources[str(ARCHIVE/'build.corrected_uniform_v1/cuMAC/src/libcumac.so')] = sha(ARCHIVE/'build.corrected_uniform_v1/cuMAC/src/libcumac.so')
        write(self.out/'launch.json', dict(seed=seed, horizon=horizon, command=command,
            env={k:v for k,v in self.env.items() if k.startswith(('CUMAC_', 'CUDA_'))},
            simulator_sha256=EXPECTED_SIMULATOR, source_sha256=self.sources,
            runtime_libcumac_resolution=library, ctbs_action_clipping=False, business_weights=False))
        self.log = (self.out/'sim.log').open('w')
        self.proc = subprocess.Popen(command, cwd=self.out, env=self.env, stdout=self.log, stderr=subprocess.STDOUT)
        self.client = None
        try:
            self.client = self.r.p.BridgeEnvClient(self.r.p.BridgeEnvConfig(socket_path=self.socket, connect_timeout_s=60, enable_ctbs=True))
            self.previous, _, self.info = self.client.reset(seed=seed, episode_horizon=horizon)
            assert self.info['harq_enabled'] == 0
            self.config = self.r.p.PhyTbsConfig.from_vector(self.info['ctbs_config'])
            self.phy = self.r.optimized.OptimizedPhy(self.config)
            self.builder = self.r.p.FeatureSnapshotBuilder(self.r.p.SnapshotBuilderConfig(tb_err_ema_alpha=.1))
            self.edges = self.r.p.SparseEdgeGenerator(self.r.p.EdgeGeneratorConfig(ue_prg_topk=4,
                ue_prg_diversity_extra=0, candidate_mode='all_demand_ues',
                include_pfq_score_anchor=False, include_candidate_risk_features=False))
            self.refresh()
        except BaseException:
            self.close(validate=False)
            raise

    def refresh(self):
        self.inputs = self.r.p._ctbs_inputs(self.client, self.info, self.config)
        assert self.inputs.tti == self.tti
        self.obs = self.b.fresh_observation.read(self.out/'fresh_observation.bin', self.previous, self.inputs)
        self.snapshot = self.r.historical.old.read_snapshot(self.out/'pre_action.bin', self.tti)
        np.testing.assert_array_equal(self.obs['obs_ue_features'][:, 0], self.snapshot['queue'])
        np.testing.assert_array_equal(self.obs['action_mask_cell_ue'],
            np.arange(3)[:, None] == np.asarray(self.inputs.serving_cell)[None])
        np.testing.assert_array_equal(self.obs['action_mask_ue'], self.snapshot['queue'] > 0)

    def raw_pfq(self):
        return self.r.p.decode_type0_teacher_action(self.obs['teacher_action_ue_select'],
            self.obs['teacher_action_prg_alloc'], n_cell=3, n_prg=17).reshape(3, 17)

    def expert(self, study):
        draft, keep, *_ = study.policy(self.obs, self.inputs, self.snapshot, self.builder,
            self.edges, None, None, self.phy)
        weights = study.weights(np.zeros(36), np.full(36, 2400.))
        np.testing.assert_array_equal(weights, np.ones(36))
        np.testing.assert_array_equal(draft, self.raw_pfq())
        action, _, record, _ = study.select(draft, keep, self.phy, self.obs, self.inputs, weights)
        assert record['converged'] and not record['cap_reached']
        return action

    def step(self, action):
        assert self.tti < self.horizon
        action = np.asarray(action, dtype=np.int64)
        assert action.shape == (3, 17)
        self.fifo.before(self.tti + 1, self.snapshot['queue'])
        ue, prg = self.r.replacement.prior.encode(action, self.obs)
        prediction = self.b.discrete_prediction(action, self.inputs, self.snapshot, self.phy.tbs)
        self.input_hashes.append(self.r.p.digest_arrays(self.obs))
        self.previous, _, _, self.info = self.client.step(ue, prg)
        actual = self.r.p._actual_chosen_from_mask(self.info['actual_ue_prg_mask'],
            np.asarray(self.inputs.serving_cell), n_cell=3, n_prg=17)
        np.testing.assert_array_equal(actual, action.reshape(-1))
        active = prediction['count'] > 0
        for key, label in [('mcs','actual_ue_mcs'), ('layer','actual_ue_layer'), ('capacity','actual_ue_tbs_capacity_bytes')]:
            np.testing.assert_array_equal(prediction[key][active], np.asarray(self.info[label])[active])
        np.testing.assert_array_equal(prediction['dequeue'], self.info['actual_ue_served_bytes'])
        record = self.fifo.after(self.info['actual_ue_served_bytes'], self.info['actual_ue_goodput_bytes'])
        self.tti += 1
        record['tti'] = self.tti
        self.rewards.append(record)
        self.actions.append(action.copy())
        if self.tti % 100 == 0 or self.tti == self.horizon:
            write(self.out/'progress.json', dict(tti=self.tti, horizon=self.horizon, elapsed_s=time.monotonic()-self.started))
        if self.tti < self.horizon:
            self.refresh()
        return record

    def close(self, validate=True):
        try:
            if self.client is not None:
                self.client.close()
        finally:
            try:
                code = self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                code = self.proc.wait(timeout=10)
            self.log.close()
        if not validate:
            return
        assert code == 0 and self.tti == self.horizon, (code, self.tti, self.horizon)
        for p, digest in self.sources.items():
            assert sha(p) == digest, p
        data = np.fromfile(self.out/'metrics.bin', dtype='<f8').reshape(self.horizon, 36, len(self.b.FIELDS))
        idx = self.b.FIELDS.index
        np.testing.assert_array_equal(data[:, :, idx('successful_bytes')].sum(1), [r['goodput_bytes'] for r in self.rewards])
        np.testing.assert_array_equal(data[:, :, idx('post_queue_bytes')].sum(1), [r['post_queue_bytes'] for r in self.rewards])
        pd.DataFrame(self.rewards).to_csv(self.out/'rewards.csv', index=False)
        np.savez_compressed(self.out/'actions.npz', actions=self.actions, input_hashes=self.input_hashes)
        write(self.out/'validation.json', dict(passed=True, horizon=self.horizon,
            exact_native_actions=self.tti, fresh_input_checks=self.tti, exact_mcs_layer_tbs_dequeue=self.tti,
            no_ctbs_action_clipping=True, equal_ue_reward=True, sources_unchanged=True,
            elapsed_s=time.monotonic()-self.started))


def packets(out):
    b = legacy()
    from output.pfq_independent_reuse.formal.boundary08_fast_multiseed_20261010_v1.summarize import packets_from_fifo
    out = Path(out)
    horizon = json.loads((out/'launch.json').read_text())['horizon']
    service = pd.DataFrame(np.fromfile(out/'metrics.bin', dtype='<f8').reshape(-1, len(b.FIELDS)), columns=b.FIELDS)
    pk = packets_from_fifo(service, pd.read_csv(out/'traffic_offered_packets.csv'),
        pd.read_csv(out/'packet_outcome_samples.csv'), horizon)
    rewards = pd.read_csv(out/'rewards.csv')
    successful = pk[pk.successfully_delivered].groupby('terminal_tti').packet_size_bytes.sum().reindex(range(1,horizon+1), fill_value=0)
    np.testing.assert_array_equal(successful, rewards.complete_payload_bytes)
    expiry = pk[pk.outcome.eq('expired')].groupby('terminal_tti').packet_size_bytes.sum().reindex(range(horizon),fill_value=0)
    np.testing.assert_array_equal(expiry, rewards.expired_payload_bytes)
    write(out/'reward_validation.json', dict(passed=True, full_payload_reward_exact=True,
        expiry_full_packet_reward_exact=True, tti=horizon))
    return service, pk

"""Closure adapter: frozen native step/validation; explicit rate and placement."""
import json
import os
from pathlib import Path
import subprocess
import time
from training.stageb_ppo import native as base
from training.stageb_ppo.reward import PacketReward
import numpy as np
import pandas as pd
from .io import read_fresh, read_snapshot, encode, discrete_prediction

class SevenCellEnv:
    def __init__(self,out,seed,horizon,build,cells=7):
        rate,placement=.8,"coop_boundary"
        self.C,self.U,self.P=cells,12*cells,17
        build=Path(build).resolve();manifest=json.loads((build/"manifest.json").read_text())
        assert manifest["cells"]==cells and manifest["users"]==self.U
        self.out,self.seed,self.horizon=Path(out),seed,horizon;self.out.mkdir(parents=True,exist_ok=False)
        self.b=base.legacy();self.r=self.b.r;self.tti=0;self.started=time.monotonic()
        self.rewards,self.actions,self.input_hashes=[],[],[];self.fifo=PacketReward(users=self.U)
        original=self.r.OLD/f'seed{seed}_pfq0_2000/launch.json'
        launch=json.loads(original.read_text())
        self.env={k:v for k,v in os.environ.items() if not k.startswith('CUMAC_')};self.env.update(launch['env'])
        self.env.update(CUMAC_UE_PLACEMENT_MODE=placement,CUMAC_TOPOLOGY_SEED=str(seed),CUMAC_TRAFFIC_SEED=str(seed))
        origin=str(Path(launch['env']['CUMAC_DIRECTED_SNAPSHOT_PATH']).parent)
        for k,v in list(self.env.items()):
            if k.startswith('CUMAC_') and v.startswith(origin):self.env[k]=str(self.out)+v[len(origin):]
        self.socket=f'/tmp/bound_{os.getpid()}_{seed}_{time.time_ns()}.sock'
        self.env.update(CUMAC_ONLINE_SOCKET=self.socket,CUDA_VISIBLE_DEVICES='0',CUMAC_SHADOW_FADING_STD_DB='3.0',
            CUMAC_FRESH_OBSERVATION_PATH=str(self.out/'fresh_observation.bin'),
            CUMAC_TRAFFIC_ARRIVAL_RATE=format(float(rate),'.12g'),
            OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        assert self.env['CUMAC_PFQ_DEMAND_CAP_SLACK_BYTES']=='0' and self.env['CUMAC_TRAFFIC_PROFILE']=='homogeneous'
        assert not any(k.startswith('CUMAC_LAST_SLOT_DROP') for k in self.env)
        command=[str(build/'simulator'),*launch['command'][1:]]
        assert base.sha(command[0])==manifest['simulator_sha256']
        linked=subprocess.check_output(['ldd',command[0]],env=self.env,text=True)
        library=next(x.strip() for x in linked.splitlines() if 'libcumac.so =>' in x)
        assert library.startswith('libcumac.so => '+str(base.ARCHIVE/'build.corrected_uniform_v1/cuMAC/src/libcumac.so')+' (')
        self.sources={str(p):base.sha(p) for p in Path(base.__file__).parent.glob('*.py')}
        for n in ('rate_native.py','ttl_weights.py','bounded_dpls.py','bounded_kernel.cpp','_bounded.so',
                  'persistent_kernel.cpp','physics_kernel.cpp','no_ctbs_inputs.py'):
            p=Path(base.ROOT)/'training/stageb_lns'/n;self.sources[str(p)]=base.sha(p)
        library_file=base.ARCHIVE/'build.corrected_uniform_v1/cuMAC/src/libcumac.so'
        self.sources[str(library_file)]=base.sha(library_file)
        self.sources[str(build/'simulator')]=base.sha(build/'simulator')
        self.sources.update({str(p):base.sha(p) for p in [Path(__file__).with_name(n) for n in ('native.py','io.py','campaign.py','solver.py','contract.py')]})
        self.sources[str(Path(__file__).with_name('_bounded128.so'))]=base.sha(Path(__file__).with_name('_bounded128.so'))
        base.write(self.out/'launch.json',dict(seed=seed,horizon=horizon,command=command,
            env={k:v for k,v in self.env.items() if k.startswith(('CUMAC_','CUDA_'))},
            cells=self.C,users=self.U,prgs=self.P,simulator_sha256=manifest["simulator_sha256"],source_sha256=self.sources,runtime_libcumac_resolution=library,
            ctbs_action_clipping=False,business_weights=True,arrival_rate=float(rate),placement=placement,current_teacher_calls=0))
        self.log=(self.out/'sim.log').open('w')
        self.proc=subprocess.Popen(command,cwd=self.out,env=self.env,stdout=self.log,stderr=subprocess.STDOUT)
        self.client=None
        try:
            self.client=self.r.p.BridgeEnvClient(self.r.p.BridgeEnvConfig(socket_path=self.socket,connect_timeout_s=60,enable_ctbs=True))
            self.previous,_,self.info=self.client.reset(seed=seed,episode_horizon=horizon)
            assert self.info['harq_enabled']==0
            self.config=self.r.p.PhyTbsConfig.from_vector(self.info['ctbs_config'])
            self.refresh()
        except BaseException:self.close(validate=False);raise

    def refresh(self):
        self.inputs = self.r.p._ctbs_inputs(self.client, self.info, self.config)
        assert self.inputs.tti == self.tti
        self.obs = read_fresh(self.out/'fresh_observation.bin', self.previous, self.inputs)
        self.snapshot = read_snapshot(self.out/'pre_action.bin', self.tti)
        np.testing.assert_array_equal(self.obs['obs_ue_features'][:, 0], self.snapshot['queue'])
        np.testing.assert_array_equal(self.obs['action_mask_cell_ue'],
            np.arange(self.C)[:, None] == np.asarray(self.inputs.serving_cell)[None])
        np.testing.assert_array_equal(self.obs['action_mask_ue'], self.snapshot['queue'] > 0)

    def raw_pfq(self):
        return self.r.p.decode_type0_teacher_action(self.obs['teacher_action_ue_select'],
            self.obs['teacher_action_prg_alloc'], n_cell=self.C, n_prg=self.P).reshape(self.C, self.P)

    def step(self, action):
        assert self.tti < self.horizon
        action = np.asarray(action, dtype=np.int64)
        assert action.shape == (self.C, self.P)
        self.fifo.before(self.tti + 1, self.snapshot['queue'])
        ue, prg = encode(action, self.obs)
        prediction = discrete_prediction(action, self.inputs, self.snapshot)
        self.input_hashes.append(self.r.p.digest_arrays(self.obs))
        self.previous, _, _, self.info = self.client.step(ue, prg)
        actual = self.r.p._actual_chosen_from_mask(self.info['actual_ue_prg_mask'],
            np.asarray(self.inputs.serving_cell), n_cell=self.C, n_prg=self.P)
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
            base.write(self.out/'progress.json', dict(tti=self.tti, horizon=self.horizon, elapsed_s=time.monotonic()-self.started))
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
            assert base.sha(p) == digest, p
        data = np.fromfile(self.out/'metrics.bin', dtype='<f8').reshape(self.horizon, self.U, len(self.b.FIELDS))
        idx = self.b.FIELDS.index
        np.testing.assert_array_equal(data[:, :, idx('successful_bytes')].sum(1), [r['goodput_bytes'] for r in self.rewards])
        np.testing.assert_array_equal(data[:, :, idx('post_queue_bytes')].sum(1), [r['post_queue_bytes'] for r in self.rewards])
        pd.DataFrame(self.rewards).to_csv(self.out/'rewards.csv', index=False)
        np.savez_compressed(self.out/'actions.npz', actions=self.actions, input_hashes=self.input_hashes)
        base.write(self.out/'validation.json', dict(passed=True, horizon=self.horizon,
            exact_native_actions=self.tti, fresh_input_checks=self.tti, exact_mcs_layer_tbs_dequeue=self.tti,
            no_ctbs_action_clipping=True, equal_ue_reward=True, sources_unchanged=True,
            elapsed_s=time.monotonic()-self.started))

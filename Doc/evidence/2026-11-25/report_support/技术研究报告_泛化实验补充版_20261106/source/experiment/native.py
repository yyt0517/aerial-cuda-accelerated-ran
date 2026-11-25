"""Isolated variable-traffic adapter; native action/PHY checks inherited unchanged."""
import json
import os
from pathlib import Path
import subprocess
import time
from training.stageb_ppo import native as base
from .fifo import HeterogeneousPacketReward
from .protocol import OUT

class GeneralizationEnv(base.NativeEnv):
    def __init__(self,out,seed,horizon,rate=.8,placement="coop_boundary",flow_file=None,mixed=True):
        self.out,self.seed,self.horizon=Path(out),seed,horizon;self.out.mkdir(parents=True,exist_ok=False)
        self.b=base.legacy();self.r=self.b.r;self.tti=0;self.started=time.monotonic()
        self.rewards,self.actions,self.input_hashes=[],[],[];
        sizes=[3000]*36
        if flow_file:
            import pandas as pd
            sizes=pd.read_csv(flow_file).packet_size_bytes.to_numpy()
        self.fifo=HeterogeneousPacketReward(sizes)
        original=self.r.OLD/'seed41_pfq0_2000/launch.json'
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
        assert self.env['CUMAC_PFQ_DEMAND_CAP_SLACK_BYTES']=='0'
        self.env['CUMAC_GENERALIZATION_MIXED_PLACEMENT']='1' if mixed else '0'
        if flow_file:
            self.env['CUMAC_GENERALIZATION_FLOW_CSV']=str(Path(flow_file).resolve())
            self.env['CUMAC_TRAFFIC_PROFILE']='per_ue_generalization_v1'
        assert not any(k.startswith('CUMAC_LAST_SLOT_DROP') for k in self.env)
        build=json.loads((OUT/'native/build_manifest.json').read_text())
        command=[str(OUT/'native/simulator'),*launch['command'][1:]]
        assert base.sha(command[0])==build['simulator_sha256']
        linked=subprocess.check_output(['ldd',command[0]],env=self.env,text=True)
        library=next(x.strip() for x in linked.splitlines() if 'libcumac.so =>' in x)
        assert library.startswith('libcumac.so => '+str(base.ARCHIVE/'build.corrected_uniform_v1/cuMAC/src/libcumac.so')+' (')
        self.sources={str(p):base.sha(p) for p in Path(base.__file__).parent.glob('*.py')}
        for n in ('rate_native.py','ttl_weights.py','bounded_dpls.py','bounded_kernel.cpp','_bounded.so',
                  'persistent_kernel.cpp','physics_kernel.cpp','no_ctbs_inputs.py'):
            p=Path(base.ROOT)/'training/stageb_lns'/n;self.sources[str(p)]=base.sha(p)
        library_file=base.ARCHIVE/'build.corrected_uniform_v1/cuMAC/src/libcumac.so'
        self.sources[str(library_file)]=base.sha(library_file)
        for p in Path(__file__).parent.glob('*.py'):self.sources[str(p)]=base.sha(p)
        self.sources[command[0]]=base.sha(command[0])
        if flow_file:self.sources[str(Path(flow_file).resolve())]=base.sha(flow_file)
        base.write(self.out/'launch.json',dict(seed=seed,horizon=horizon,command=command,
            env={k:v for k,v in self.env.items() if k.startswith(('CUMAC_','CUDA_'))},
            simulator_sha256=build['simulator_sha256'],source_sha256=self.sources,runtime_libcumac_resolution=library,
            ctbs_action_clipping=False,business_weights=True,arrival_rate=float(rate),placement=placement,mixed_placement=mixed,flow_file=str(flow_file) if flow_file else None,current_teacher_calls=0))
        self.log=(self.out/'sim.log').open('w')
        self.proc=subprocess.Popen(command,cwd=self.out,env=self.env,stdout=self.log,stderr=subprocess.STDOUT)
        self.client=None
        try:
            self.client=self.r.p.BridgeEnvClient(self.r.p.BridgeEnvConfig(socket_path=self.socket,connect_timeout_s=60,enable_ctbs=True))
            self.previous,_,self.info=self.client.reset(seed=seed,episode_horizon=horizon)
            assert self.info['harq_enabled']==0
            self.config=self.r.p.PhyTbsConfig.from_vector(self.info['ctbs_config'])
            self.phy=self.r.optimized.OptimizedPhy(self.config)
            self.builder=self.r.p.FeatureSnapshotBuilder(self.r.p.SnapshotBuilderConfig(tb_err_ema_alpha=.1))
            self.edges=self.r.p.SparseEdgeGenerator(self.r.p.EdgeGeneratorConfig(ue_prg_topk=4,ue_prg_diversity_extra=0,
                candidate_mode='all_demand_ues',include_pfq_score_anchor=False,include_candidate_risk_features=False))
            self.refresh()
        except BaseException:self.close(validate=False);raise

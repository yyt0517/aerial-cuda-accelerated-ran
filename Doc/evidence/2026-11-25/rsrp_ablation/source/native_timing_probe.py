"""Separate timing-only native build; quality trajectories remain frozen.

Adds host clocks around existing synchronized channel generation and SVD;
does not add synchronization or change any numerical work. Eight sequential
short runs must reproduce the uninstrumented smoke actions and native metrics.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
from training.stageb_ppo import native as base
from training.stageb_ppo.native import ROOT,write,sha
from training.stage2_closure.native import ReportNativeEnv


def build(root):
    reference=base.REFERENCE;manifest=json.loads((reference/'native_build_manifest.json').read_text())
    source=reference/'instrumented_main.cpp';assert sha(source)==manifest['main_source_sha256']
    if str(source)!=manifest['main_source']:
        raise RuntimeError('Build inside the original GPU container so manifest paths resolve without touching old outputs')
    out=root/'native_timing/build';out.mkdir(parents=True,exist_ok=False)
    text=source.read_text()
    def replace(before,after):
        nonlocal text
        assert text.count(before)==1,before;text=text.replace(before,after)
    replace('  std::vector<double> pfqCoreUs;',
            '  std::vector<double> pfqCoreUs;\n  std::vector<double> ablationChannelUs, ablationSvdUs;')
    replace('      net->genFastFadingGpu(t);  // Continuous channel time; slotIdx only wraps bounded record buffers.\n'
            '      CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));',
            '      const auto ablationChannelStart = std::chrono::steady_clock::now();\n'
            '      net->genFastFadingGpu(t);  // Continuous channel time; slotIdx only wraps bounded record buffers.\n'
            '      CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));\n'
            '      ablationChannelUs.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-ablationChannelStart).count());')
    replace('        svdPrd->run(net->cellGrpPrmsGpu.get());\n        CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));',
            '        const auto ablationSvdStart = std::chrono::steady_clock::now();\n'
            '        svdPrd->run(net->cellGrpPrmsGpu.get());\n        CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));\n'
            '        ablationSvdUs.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-ablationSvdStart).count());')
    marker='  { std::ofstream f("pfq_core_us.bin", std::ios::binary); f.write(reinterpret_cast<const char*>(pfqCoreUs.data()), pfqCoreUs.size()*sizeof(double)); }'
    extra=''
    for filename,variable in [('channel_generation_us.bin','ablationChannelUs'),('svd_us.bin','ablationSvdUs')]:
        extra+=f'\n  {{ std::ofstream f("{filename}", std::ios::binary); f.write(reinterpret_cast<const char*>({variable}.data()), {variable}.size()*sizeof(double)); }}'
    replace(marker,marker+extra)
    newsource=out/'instrumented_main.cpp';newsource.write_text(text)
    original_prefix=str(reference)
    # This build runs inside the existing container where manifest paths resolve.
    commands=[]
    for name in ('compile_main','link_command'):
        command=[x.replace(original_prefix,str(out)) for x in manifest[name]]
        assert Path(command[command.index('-o')+1]).parent==out
        if name=='compile_main':assert str(newsource) in command
        commands.append(command)
    cwd=base.ARCHIVE/'build.corrected_uniform_v1/cuMAC/examples/multiCellSchedulerUeSelection'
    for command in commands:subprocess.run(command,cwd=cwd,check=True)
    write(out/'manifest.json',dict(passed=True,original_simulator_sha256=manifest['simulator_sha256'],
        simulator_sha256=sha(out/'simulator'),original_source_sha256=sha(source),new_source_sha256=sha(newsource),
        commands=commands,cwd=str(cwd),scope='clocks around existing stream-synchronized generation/SVD; original network object and libcumac'))


class TimedEnv(ReportNativeEnv):
    def __init__(self,*args,**kwargs):
        self.roundtrip=[];super().__init__(*args,**kwargs)
    def step(self,action):
        start=time.perf_counter();cpu=time.process_time();result=super().step(action)
        self.roundtrip.append(dict(tti=self.tti,native_roundtrip_wall_ms=(time.perf_counter()-start)*1000,
                                  host_wrapper_cpu_ms=(time.process_time()-cpu)*1000))
        return result
    def close(self,validate=True):
        super().close(validate=validate)
        if validate:pd.DataFrame(self.roundtrip).to_csv(self.out/'roundtrip.csv',index=False)


def run(root,wait):
    if wait:
        path=root/'replay/validation.json'
        while not path.exists():
            status=root/'postprocessing_status.json'
            if status.exists() and json.loads(status.read_text()).get('stage')=='failed':
                raise RuntimeError('Main postprocessing failed before serial timing probe')
            time.sleep(30)
        assert json.loads(path.read_text())['passed']
    from . import campaign
    builddir=root/'native_timing/build';manifest=json.loads((builddir/'manifest.json').read_text())
    base.REFERENCE=builddir;base.EXPECTED_SIMULATOR=manifest['simulator_sha256']
    assert sha(builddir/'simulator')==base.EXPECTED_SIMULATOR
    campaign.ReportNativeEnv=TimedEnv
    out=root/'native_timing/runs';out.mkdir(exist_ok=False)
    checks=[];raw=[];records=[]
    for scene,rate in [('boundary',.8),('uniform',1.)]:
        for objective in ('deficit','mlwdf_pf'):
            for stage in ('exact','reported'):
                spec=dict(scene=scene,rate=rate,seed=41,objective=objective,stage=stage)
                result=campaign.run_one(out,spec,True);key=campaign.key(spec);d=out/key;original=root/'smoke_v1'/key
                for name in ('metrics.bin','traffic_offered_packets.csv','ue_topology.csv'):
                    assert sha(d/name)==sha(original/name),(key,name)
                with np.load(d/'actions.npz') as a,np.load(original/'actions.npz') as b:
                    for name in a.files:np.testing.assert_array_equal(a[name],b[name])
                for name in ('channel_hashes.npy','report_hashes.npy'):
                    np.testing.assert_array_equal(np.load(d/name),np.load(original/name))
                times=pd.read_csv(d/'roundtrip.csv')
                channel=np.fromfile(d/'channel_generation_us.bin',dtype='<f8')/1000
                svd=np.fromfile(d/'svd_us.bin',dtype='<f8')/1000
                pfq=np.fromfile(d/'pfq_core_us.bin',dtype='<f8')/1000
                # Native bridge prepares one final unexecuted state and calls
                # PFQ both before actions and when exporting post-step state.
                assert len(times)==512 and len(channel)==len(svd)==513 and len(pfq)==1025
                times['channel_generation_wall_ms']=channel[:512];times['svd_wall_ms']=svd[:512]
                times['native_pfq_pre_wall_ms']=pfq[:1024:2];times['native_pfq_post_wall_ms']=pfq[1:1024:2]
                times['native_pfq_wall_ms']=times.native_pfq_pre_wall_ms+times.native_pfq_post_wall_ms
                search=pd.read_csv(d/'search.csv.gz').rename(columns={'wall_ms':'scheduler_wall_ms','cpu_ms':'scheduler_cpu_ms'})
                times=times.merge(search[['tti','scheduler_wall_ms','scheduler_cpu_ms','measurement_emulator_ms']],on='tti',how='inner')
                # Sample all 480 post-warmup steps; separate probe, not quality metrics.
                for name,value in spec.items():times[name]=value
                raw.append(times)
                cols=[c for c in times if c.endswith('_ms')]
                records.append(dict(**spec,**{c:float(times[c].mean()) for c in cols}))
                checks.append(dict(key=key,passed=True,all_native_metrics_actions_and_channels_bit_identical=True))
                print('NATIVE_TIMING',key,flush=True)
    pd.concat(raw,ignore_index=True).to_csv(root/'native_timing/raw.csv',index=False)
    pd.DataFrame(records).to_csv(root/'native_timing/summary.csv',index=False)
    write(root/'native_timing/validation.json',dict(passed=True,runs=8,checks=checks,
        scope='sequential 512-TTI smoke-equivalent probes, first 32 excluded; synchronized channel and SVD wall clocks',
        roundtrip='parent step wall includes native simulator, bridge and host validation; final step has no next-state refresh',
        native_invocations='513 input-state generations/SVDs and 1025 PFQ calls for 512 actions; final unexecuted input state excluded; both PFQ calls reported',
        channel_timing='channel/precoder at input state, roundtrip advances to next state; averages are separate categories, not aligned additive slices',
        caveat='shared GPU wall time, short-run diagnostic only; main scheduler CPU study excludes native generation altogether'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--build',action='store_true')
    parser.add_argument('--run',action='store_true');parser.add_argument('--wait-for-replay',action='store_true');args=parser.parse_args()
    root=Path(args.root).resolve()
    if args.build:build(root)
    if args.run:run(root,args.wait_for_replay)

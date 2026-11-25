"""Trace the frozen simulator's instantaneous H to the DPLS evaluator."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.stageb_ppo.native import ROOT, REFERENCE, write, sha, EXPECTED_SIMULATOR
from training.stageb_lns.archive import enable_legacy_types


def main(args):
    import torch
    enable_legacy_types()
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    formal=REFERENCE.parent
    network=formal/'boundary08_corrected_physics_20261020_v1/native_network_source.cu'
    main_source=REFERENCE/'instrumented_main.cpp'
    export=formal/'boundary08_directed_coordination_20261008_v1/directed_snapshot.h'
    manifest=json.loads((REFERENCE/'native_build_manifest.json').read_text())
    assert sha(REFERENCE/'simulator')==EXPECTED_SIMULATOR
    assert sha(network)==next(v for k,v in manifest['source_sha256'].items() if k.endswith('/network.cu'))
    assert sha(main_source)==manifest['main_source_sha256']
    assert sha(export)==next(v for k,v in manifest['main_headers_sha256'].items() if k.endswith('/directed_snapshot.h'))
    assert 'net->genFastFadingGpu(t);' in main_source.read_text()
    assert 'gpu->estH_fr_actUe' in export.read_text()
    assert 'rho*channMatGpu[currChannIdx].x + rhoPrime*sqrtAmpl' in network.read_text()
    records=[]
    for scene,rate in [('boundary',.8),('uniform',1.)]:
        for seed in (41,42,43):
            d=ROOT/f'deliverables/stage2_closure_20261104/campaign/{scene}_r{rate:g}_deficit_s{seed}'
            rows=[torch.load(p,map_location='cpu',weights_only=False) for p in sorted((d/'states').glob('*.pt'))]
            assert len(rows)==3
            h0=rows[0]['snapshot']['h']
            frequency_residual=float(np.linalg.norm(h0-h0.mean(axis=1,keepdims=True))/np.linalg.norm(h0))
            records.append(dict(scene=scene,seed=seed,shape=list(h0.shape),
                prg_residual_over_channel_norm=frequency_residual,
                tti_samples=[int(row['tti']) for row in rows],
                time_relative_differences=[float(np.linalg.norm(row['snapshot']['h']-h0)/np.linalg.norm(h0)) for row in rows[1:]],
                hashes=[sha(p) for p in sorted((d/'states').glob('*.pt'))]))
    evidence=[]
    for path,patterns in [(network,['netData->rho =','netData->rhoPrime =','float PL =',
                                   'netData->rxSigPowDB[cIdx * nActiveUe + uIdx] =','channMatGpu_real = rho*']),
                          (main_source,['net->genFastFadingGpu(t);','writeDirectedSnapshot(t,']),
                          (export,['gpu->estH_fr_actUe']),
                          (ROOT/'training/stageb_lns/no_ctbs_inputs.py',["h = np.asarray(snapshot['h'])"]),
                          (ROOT/'training/stageb_lns/physics.py',['self.covariance = problem.h @'])]:
        lines=path.read_text().splitlines()
        for pattern in patterns:
            matches=[dict(line=i+1,text=line.strip()) for i,line in enumerate(lines) if pattern in line]
            assert matches,(path,pattern)
            evidence.append(dict(path=str(path),sha256=sha(path),pattern=pattern,matches=matches))
    write(output,dict(passed=True,conclusion='instantaneous complex H including path loss, shadowing and correlated Rayleigh small-scale fading',
        formula='H[t] = rho H[t-1] + rhoPrime amplitude_LS (N_real + j N_imag)',
        rho=.9938,rhoPrime=.0786,simulator_sha256=EXPECTED_SIMULATOR,
        note='Inherited rho constants are used as coded; do not reinterpret their 1 ms comment as the actual 0.5 ms experiment TTI.',
        evidence=evidence,snapshot_checks=records))
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(ROOT/'deliverables/rsrp_interference_20261109/channel_audit.json'))
    main(parser.parse_args())

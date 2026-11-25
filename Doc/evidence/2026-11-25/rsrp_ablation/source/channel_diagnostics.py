"""Describe lost channel information on the shared, archived channel samples."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from training.stageb_ppo.native import ROOT,write,sha
from training.stageb_lns.archive import enable_legacy_types
from training.stageb_lns.no_ctbs_inputs import problem_from_raw_row
from .analyze import local


def main(args):
    import torch
    torch.set_num_threads(1);enable_legacy_types()
    root=Path(args.root);prior=ROOT/'deliverables/rsrp_interference_20261109/campaign'
    records=[];checks=[]
    for scene,rate in [('boundary',.8),('uniform',1.)]:
        for seed in (41,42,43):
            spec=json.loads((prior/f'{scene}_r{rate:g}_rsrp_deficit_s{seed}'/'done.json').read_text())
            d=local(spec['directory'])
            for sample in json.loads((d/'saved_states.json').read_text()):
                path=local(sample['path']);assert sha(path)==sample['sha256']
                row=torch.load(path,map_location='cpu',weights_only=False);p=problem_from_raw_row(row)
                neighbor=np.arange(p.C)[None,:]!=p.serving[:,None]
                covariance=p.h@p.h.conj().swapaxes(-1,-2)
                eig=np.linalg.eigvalsh(covariance)
                assert (eig>0).all()
                normalized=eig/eig.mean(-1,keepdims=True)
                shape_error=np.sqrt(np.square(normalized-1).sum(-1)/np.square(normalized).sum(-1))
                select=lambda x:x.transpose(0,2,1)[neighbor]
                per_prg=np.square(np.abs(p.h)).mean(axis=(-1,-2))
                wide=per_prg.mean(1);pilot=np.square(np.abs(p.h[...,0])).mean(axis=(1,3))
                record=dict(scene=scene,seed=seed,tti=int(row['tti']),
                    spatial_covariance_relative_frobenius_error=float(select(shape_error).mean()),
                    weakest_eigenvalue_over_mean=float(select(normalized[...,0]).mean()),
                    strongest_eigenvalue_over_mean=float(select(normalized[...,-1]).mean()),
                    prg_power_cv=float((per_prg.std(1)/wide)[neighbor].mean()),
                    report_age_tti=int(row['tti'])-1-int(row['rsrp_report_tti']))
                for label,a,b in [('frequency',wide[:,None,:],per_prg),('single_port',pilot,wide),
                    ('filter',row['rsrp_filtered'],pilot),('report_hold_delay',row['rsrp_report'],row['rsrp_filtered']),
                    ('filtered_vs_current_all_port',row['rsrp_filtered'],wide),
                    ('reported_vs_current_all_port',row['rsrp_report'],wide)]:
                    errors=10*np.log10(a/b)
                    errors=select(errors).ravel() if errors.ndim==3 else errors[neighbor]
                    record[label+'_power_bias_db']=float(errors.mean())
                    record[label+'_power_mae_db']=float(np.abs(errors).mean())
                records.append(record);checks.append(dict(path=str(path),sha256=sha(path),passed=True))
    df=pd.DataFrame(records);df.to_csv(root/'channel_information_per_state.csv',index=False)
    metrics=[c for c in df if c not in ('scene','seed','tti')]
    seed=df.groupby(['scene','seed'],as_index=False)[metrics].mean()
    seed.groupby('scene',as_index=False)[metrics].mean().to_csv(root/'channel_information_summary.csv',index=False)
    write(root/'channel_information_validation.json',dict(passed=True,states=len(checks),checks=checks,
        scope='same archived exogenous channels; all UE/neighbor pairs equally weighted, not traffic-weighted',
        caveat='different information-error measures are descriptive, not causal performance-loss shares'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);main(parser.parse_args())

"""Causal snapshots paired with observed, delayed native outcomes.

Returns are training labels only. Packet expiry is assigned to terminal_tti,
the last usable slot, rather than the following slot's pre-action removal.
"""
from pathlib import Path
import json
import numpy as np
from .archive import ROOT,enable_legacy_types


def discounted_window(values,horizon=400,gamma=.9975):
    values=np.asarray(values,dtype=float)
    result=np.zeros_like(values);running=np.zeros(values.shape[1:])
    normalizer=(1-gamma**horizon)/(1-gamma)
    for t in range(len(values)-1,-1,-1):
        running=values[t]+gamma*running
        if t+horizon<len(values):running-=gamma**horizon*values[t+horizon]
        result[t]=running/normalizer
    return result


def outcomes(out):
    import pandas as pd
    fields=('tti','ue_id','offered_bytes','accepted_bytes','rejected_bytes','expired_bytes',
        'previous_post_queue_bytes','pre_queue_bytes','scheduled_dequeue_bytes','post_queue_bytes',
        'successful_bytes','failed_bytes','actual_prg_count','mcs','layer','phy_sinr_db_cpu',
        'predicted_bler_cpu','tb_tx_count','tb_err_count')
    launch=json.loads((out/'launch.json').read_text());n=launch['horizon']
    service=np.fromfile(out/'metrics.bin',dtype='<f8').reshape(n,36,len(fields))
    good=service[:,:,fields.index('successful_bytes')]
    ema=np.zeros_like(good);previous=np.zeros(36)
    for t in range(n):
        # Bridge exports float32 goodput. Preserve that multiplication before
        # adding to the float64 EMA, exactly as the native policy loop does.
        previous=.995*previous+.005*good[t].astype(np.float32);ema[t]=previous
    expired=np.zeros_like(good)
    packets=pd.read_csv(out/'packets.csv',usecols=['ue_id','terminal_tti','outcome','packet_size_bytes'])
    failures=packets[(packets.outcome=='expired') & packets.terminal_tti.between(1,n-1)]
    np.add.at(expired,(failures.terminal_tti.to_numpy(int)-1,failures.ue_id.to_numpy(int)),failures.packet_size_bytes)
    per_ue=np.stack((good/2400.,expired/2400.,np.maximum(1-ema/2400.,0)**2),axis=-1)
    returns=discounted_window(per_ue)
    return dict(returns=returns,ema=ema,reward=(per_ue[:,:,0]-2*per_ue[:,:,1]-.5*per_ue[:,:,2]).mean(1),horizon=n)


def dataset(collection,seeds):
    import torch
    enable_legacy_types();rows=[];sources=[]
    roots=[ROOT/'deliverables/tb_joint_repair_20261027_native_v2',
           ROOT/'deliverables/tb_joint_repair_20261027_native_local27_v1',Path(collection).resolve()]
    for campaign in roots:
        completion=json.loads((campaign/'completion.json').read_text());assert completion['passed']
        for out in sorted(p for p in campaign.glob('seed*') if p.is_dir()):
            launch=json.loads((out/'launch.json').read_text());seed=launch['seed']
            if seed not in seeds:continue
            observed=outcomes(out);saved=torch.load(out/'states.pt',weights_only=False,map_location='cpu')
            for row in saved:
                t=row['tti']-1
                # Require all delayed outcomes, including the next expiry event.
                if row['tti']<=launch['warmup'] or row['tti']+400>observed['horizon']:continue
                before=np.zeros(36) if t==0 else observed['ema'][t-1]
                np.testing.assert_allclose(row['ema'],before,rtol=0,atol=1.e-9)
                ret=observed['returns'][t]
                rows.append(dict(row,seed=seed,source=str(out),native_return_components=ret,
                    native_reward=float((ret[:,0]-2*ret[:,1]-.5*ret[:,2]).mean())))
            sources.append(dict(path=str(out),seed=seed,arm=launch['arm'],horizon=observed['horizon']))
    return rows,sources

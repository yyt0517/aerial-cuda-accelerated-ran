"""Prove new dimension-aware adapter preserves the original PFQ trajectory."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from training.stageb_ppo.native import REFERENCE,sha,write
from .campaign import run_one
from .analyze import FIELDS

def main(output, verify_existing=False):
    out=Path(output).resolve()
    if not verify_existing:
        out.mkdir(parents=True,exist_ok=False)
        build=out/'reference_build';build.mkdir();(build/'simulator').symlink_to(REFERENCE/'simulator')
        write(build/'manifest.json',dict(cells=3,users=36,simulator_sha256=sha(REFERENCE/'simulator')))
        run_one(out,build,41,'pfq',True,3)
    old=REFERENCE/'seed41_pfq0_20000';new=out/'pfq_s41';n=448
    with np.load(old/'prefix.npz') as a,np.load(new/'actions.npz') as b:
        np.testing.assert_array_equal(a['actions'][:n].reshape(n,3,17),b['actions'])
        np.testing.assert_array_equal(a['input_hashes'][:n],b['input_hashes'])
    x=np.memmap(old/'metrics.bin',dtype='<f8',mode='r').reshape(-1,36,len(FIELDS))
    y=np.fromfile(new/'metrics.bin',dtype='<f8').reshape(n,36,len(FIELDS));x=x[:n].copy()
    # Original raw-slot PFQ reports unsigned 255 for an unallocated layer;
    # explicit action packing uses signed -1. No transmitted row may differ.
    layer=FIELDS.index('layer');count=FIELDS.index('actual_prg_count')
    differs=x[:,:,layer]!=y[:,:,layer]
    assert np.all(x[:,:,count][differs]==0) and np.all(y[:,:,count][differs]==0)
    assert np.all(x[:,:,layer][differs]==255) and np.all(y[:,:,layer][differs]==-1)
    x[:,:,layer][differs]=-1
    np.testing.assert_array_equal(x,y)
    a=pd.read_csv(old/'traffic_offered_packets.csv');b=pd.read_csv(new/'traffic_offered_packets.csv')
    pd.testing.assert_frame_equal(a[a.arrival_tti<=n].reset_index(drop=True),b[b.arrival_tti<=n].reset_index(drop=True))
    write(out/'parity.json',dict(passed=True,tti=n,exact_actions_inputs_traffic_transmitted_phy_and_byte_metrics=True,unallocated_layer_sentinel_rows=int(differs.sum()),sentinel_normalization="unallocated layer 255→-1 only"))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--verify-existing',action='store_true');a=p.parse_args();main(a.output,a.verify_existing)

"""Audit all short trajectories and reproduce both archived endpoints."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from training.stageb_ppo.native import ROOT,sha,write
from .analyze import main as analyze


def main(args):
    root=Path(args.root);source=root/'smoke_v1';output=root/'smoke_analysis'
    completion=json.loads((source/'completion.json').read_text())
    assert completion['completed'] and completion['total']==48
    if not (output/'completion.json').exists():
        analyze(SimpleNamespace(source=str(source),output=str(output),workers=4))
    assert json.loads((output/'completion.json').read_text())['passed']
    prior=ROOT/'deliverables/rsrp_interference_20261109/smoke_v1';checks=[]
    for path in source.glob('*/done.json'):
        spec=json.loads(path.read_text())
        if spec['stage'] not in ('exact','reported'):continue
        arm=('rsrp_' if spec['stage']=='reported' else '')+spec['objective']
        old=prior/f"{spec['scene']}_r{spec['rate']:g}_{arm}_s{spec['seed']}";current=path.parent
        for name in ('metrics.bin','traffic_offered_packets.csv','ue_topology.csv'):
            assert sha(current/name)==sha(old/name),(spec['key'],name)
        with np.load(current/'actions.npz') as a,np.load(old/'actions.npz') as b:
            for name in a.files:np.testing.assert_array_equal(a[name],b[name])
        checks.append(spec['key'])
    assert len(checks)==8
    write(root/'smoke_endpoint_equivalence.json',dict(passed=True,checks=checks,
        scope='all native metrics, actions and input hashes bit identical to prior exact AND reported smoke'))
    print('48 short trajectories and 8 archived endpoints verified',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);main(parser.parse_args())

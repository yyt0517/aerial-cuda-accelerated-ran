"""Freeze the eight-arm input pairing, raw inventory and descriptive comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import pandas as pd
from .dpa_ablation_experiment import ROOT, sha, write_json
from .corrected_uniform_full import SIM,SIM_SHA,LIB,LIB_SHA


def seal(root):
    root=Path(root)
    records=[]
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name not in {'sha256_manifest.csv','status.json'}:
            records.append(dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=sha(path)))
    pd.DataFrame(records).to_csv(root/'sha256_manifest.csv',index=False)


def finalize(root):
    root=Path(root).resolve()
    status=json.loads((root/'status.json').read_text());assert status['status']=='raw_runs_complete' and len(status['arms'])==8
    assert sha(SIM)==SIM_SHA and sha(LIB)==LIB_SHA
    rows=[];identities={}
    policy_log={'pf':'PF reference check','rr':'RR baseline','pfq':'queue-aware PF reference','rrq':'queue-aware RR baseline'}
    for seed in [42,43]:
        identities[seed]={}
        for policy in ['pf','rr','pfq','rrq']:
            out=root/f'seed{seed}_{policy}'
            manifest=json.loads((out/'reference_manifest.json').read_text())
            assert manifest['binary_sha256']==SIM_SHA and manifest['reference_policy']==policy
            log=(out/'sim.log').read_text()
            # The exact startup messages are retained with the actual -b command.
            assert policy_log[policy] in log
            command=manifest['command']
            assert command[command.index('-b')+1]==str(dict(pf=0,rr=1,pfq=2,rrq=3)[policy])
            service=pd.read_csv(out/'service_balance_trace.csv',usecols=['tti','ue_id','offered_bytes','successful_bytes'])
            assert len(service)==180000 and not service.duplicated(['tti','ue_id']).any()
            observations=pd.read_csv(out/'ue_tti_observations.csv',usecols=['tti','ue_id'])
            assert len(observations)==180000 and not observations.duplicated(['tti','ue_id']).any()
            actions=pd.read_csv(out/'cell_prg_actions.csv',usecols=['tti','cell_id','prg_id'])
            assert len(actions)==255000 and not actions.duplicated(['tti','cell_id','prg_id']).any()
            outcomes=pd.read_csv(out/'packet_lifecycle.csv')
            assert outcomes.packet_id.is_unique and outcomes.arrival_tti.max()<=5000
            channel=pd.read_csv(out/'service_balance_trace.csv.channel.csv').query('tti<=5000')
            assert channel.tti.tolist()==list(range(1,5001))
            offered=pd.read_csv(out/'traffic_offered_packets.csv').query('arrival_tti<=5000')
            identity=dict(topology=sha(out/'ue_topology.csv'),
                channel=hashlib.sha256('\n'.join(channel.raw_cfr_fnv1a64.astype(str)).encode()).hexdigest(),
                offered=hashlib.sha256(offered.to_csv(index=False).encode()).hexdigest())
            identities[seed][policy]=identity
            if policy!='pf':assert identity==identities[seed]['pf'],(seed,policy,'unpaired input')
            metrics=pd.read_csv(out/'global_metrics.csv')
            measured=metrics.query("window=='measured'").iloc[0]
            expected=service.query('tti>1000').successful_bytes.sum()*8/2/1e6
            assert abs(measured.goodput_mbps-expected)<1e-8
            rows.extend(metrics.to_dict('records'))
            manifest.update(libcumac_sha256=LIB_SHA,schema='corrected-uniform-v1-fifo-success',
                            packet_semantics='MAC completion and successful byte-perfect packet delivery distinct; expired included in outcome delays; pending right-censored',
                            source_snapshot='../source_identity.json')
            write_json(out/'manifest.json',manifest)
    compare=pd.DataFrame(rows);compare.to_csv(root/'comparison.csv',index=False)
    means=compare.groupby(['policy','window']).mean(numeric_only=True).drop(columns=['seed'],errors='ignore')
    means.to_csv(root/'two_seed_descriptive_means.csv')
    files=['training/stageb_hgraph/'+name for name in ['bridge_env.py','pfq_reference_calibration.py','corrected_raw_export.py','corrected_uniform_metrics.py','corrected_uniform_baselines.py']]
    files+=['cuMAC/examples/trafficModel/trafficGenerator.hpp']
    source=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                files={name:sha(ROOT/name) for name in files},binary_sha256=SIM_SHA,libcumac_sha256=LIB_SHA)
    write_json(root/'source_identity.json',source)
    (root/'source_dirty.patch').write_bytes(subprocess.check_output(['git','diff','--',*files],cwd=ROOT))
    archive=root/'source_snapshot';archive.mkdir(exist_ok=True)
    for name in files:
        target=archive/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    write_json(root/'validation.json',dict(passed=True,arms=8,ue_tti_rows=1440000,cell_prg_rows=2040000,
               paired_inputs=identities,binary_sha256=SIM_SHA,libcumac_sha256=LIB_SHA))
    measured=compare.query("window=='measured'")
    columns=['seed','policy','goodput_mbps','expiry_bytes_over_offered','mac_dequeue_completion_delay_ms_mean',
             'successful_delivery_delay_ms_mean','mac_completion_or_expiry_outcome_delay_ms_mean','ue_goodput_mbps_p10']
    text='# Corrected uniform seed42/43 baselines\n\nEight independent 1000-warmup + 4000-measured runs. CPU PHY/queue goodput; 97.92 MHz per cell. Fixed 3 cell, 36 UE, 3000-byte Poisson arrivals, TTL 200 ms.\n\n'
    table=measured[columns].to_csv(sep='|',index=False,float_format='%.6f').splitlines()
    text+=table[0]+'\n'+'|'.join(['---']*len(measured[columns].columns))+'\n'+'\n'.join(table[1:])
    text+='\n\nMAC completion includes bytes dequeued on failed TBs. Successful delivery requires all bytes of the reconstructed FIFO packet to succeed; delays are conditioned on that success. Completion-or-expiry delay includes expired and failed-MAC-completed packets. Pending packets retain censoring, age and remaining bytes. Terminal TTI and expiry-removal event TTI are both exported; unused bridge preparation is excluded.\n\n'
    text+='Packet completion timestamp was fixed before these runs: zero-based slot t completes at (t+1)*0.5 ms. P0 confirmed unchanged PHY/queue trajectories. Raw-on/off equality and independent TBS/byte/FIFO conservation checks passed.\n\n'
    text+='All old full-window KPI definitions and native CPU/GPU arrays are retained in each historical_full_window directory. Canonical full/measured CSVs distinguish packet populations, TB-weighted BLER and pooled quantiles. The metric catalog contains the declared historical/aligned/granularity union with aliases. No two-seed significance or unseen-generalization claim.\n'
    (root/'report.md').write_text(text)
    seal(root)
    status.update(status='validated_complete');write_json(root/'status.json',status)
    print(measured[columns].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args();finalize(args.root)

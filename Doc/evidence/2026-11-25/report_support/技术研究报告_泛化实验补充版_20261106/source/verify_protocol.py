"""Independent audit of frozen scene inputs and finished native launch configuration."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'deliverables/stage2_generalization_20261106'
OUT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    protocol=json.loads((EXP/'protocol.json').read_text());plans=[]
    for seed in range(51,56):
        setup=json.loads((EXP/'setup'/f'seed{seed}'/'plan.json').read_text())
        for scene in ['G1','G2','G3','G4']:
            p=EXP/'plans'/f'{scene}_s{seed}.csv';d=pd.read_csv(p)
            assert len(d)==36 and (d.ue.to_numpy()==np.arange(36)).all()
            assert (d.serving_cell.to_numpy()==setup['serving']).all()
            assert dict(d.placement_group.value_counts())==dict(uniform=18,boundary=18)
            for nominal in range(3):
                q=d[d.ue//12==nominal];assert (q.placement_group.value_counts()==6).all()
            amount=d.packet_size_bytes*d.arrival_rate_packets_per_tti
            np.testing.assert_allclose(amount.sum(),86400,atol=1e-7)
            if scene in ['G2','G4']:
                assert dict(d.packet_size_bytes.value_counts())=={1500:18,12000:18}
                for _,q in d.groupby('placement_group'):assert dict(q.packet_size_bytes.value_counts())=={1500:9,12000:9}
            else:assert d.packet_size_bytes.eq(3000).all()
            cells=[float(amount[d.serving_cell==c].sum()*.016) for c in range(3)]
            if scene in ['G3','G4']:
                heavy=(seed-51)%3;target=np.full(3,345.6);target[heavy]=691.2
                np.testing.assert_allclose(cells,target,atol=1e-8)
                for c in range(3):
                    a=amount[d.serving_cell==c];np.testing.assert_allclose(a,a.iloc[0],atol=1e-9)
            else:np.testing.assert_allclose(amount,2400,atol=1e-9)
            plans.append(dict(scenario=scene,seed=seed,cluster_mbps=sum(cells),cell_mbps=cells,sha256=sha(p)))
    # Per-seed geometry is identical across every available scenario and algorithm.
    geometry=[]
    for seed in range(51,56):
        files=list((EXP/'campaign').glob(f'*_s{seed}/ue_topology.csv'))
        if not files:continue
        hashes=[sha(p) for p in files];assert len(set(hashes))==1
        setup=EXP/'setup'/f'seed{seed}'/'ue_topology.csv';assert hashes[0]==sha(setup)
        geometry.append(dict(seed=seed,files=len(files),unique_topologies=1))
    # Verify source freeze using path mapping between the container and host.
    for path,expected in protocol['sources'].items():assert sha(str(path).replace('/opt/nvidia/cuBB',str(ROOT)))==expected
    result=dict(passed=True,configured_plans=plans,geometry=geometry,all_protocol_sources_unchanged=True)
    (OUT/'核验/场景配置核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('20 frozen plans verified; topology shared across scenarios within each available seed.')
if __name__=='__main__':main()

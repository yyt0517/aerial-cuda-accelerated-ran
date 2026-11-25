"""Predeclared scenarios, before outcomes; no per-scenario policy tuning."""
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'deliverables/stage2_generalization_20261106'
SEEDS=tuple(range(51,56))
SCENARIOS={
 'G1':dict(name='混合布点',mixed_packets=False,imbalanced=False),
 'G2':dict(name='混合布点＋异构业务',mixed_packets=True,imbalanced=False),
 'G3':dict(name='混合布点＋小区不平衡',mixed_packets=False,imbalanced=True),
 'G4':dict(name='混合布点＋异构业务＋小区不平衡',mixed_packets=True,imbalanced=True),
}
ARMS=('pfq','deficit','mlwdf_pf')
WARMUP,SETTLING,MEASURED,FOLLOWUP=4000,4000,20000,400
FIRST,LAST=8001,28000
TOTAL_BYTES_PER_TTI=36*2400.
def flows(scenario,seed,serving):
    cfg=SCENARIOS[scenario];serving=np.asarray(serving,int)
    sizes=np.where(np.arange(36)%2==0,1500,12000) if cfg['mixed_packets'] else np.full(36,3000)
    offered=np.full(36,2400.)
    heavy=(seed-SEEDS[0])%3
    if cfg['imbalanced']:
        for c in range(3):
            users=serving==c;assert users.any()
            offered[users]=TOTAL_BYTES_PER_TTI*(.5 if c==heavy else .25)/users.sum()
    rates=offered/sizes
    assert np.isclose(offered.sum(),TOTAL_BYTES_PER_TTI)
    return sizes,rates,heavy

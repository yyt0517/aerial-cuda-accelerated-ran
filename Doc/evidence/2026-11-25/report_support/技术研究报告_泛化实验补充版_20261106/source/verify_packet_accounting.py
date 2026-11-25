"""Targeted checks for variable-size packet segmentation, CRC loss and expiration."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from training.stage2_generalization.fifo import HeterogeneousPacketReward
from training.stageb_ppo.native import PacketReward
OUT=Path(__file__).resolve().parents[1]
def zero():return np.zeros(36,dtype=np.int64)
def main():
    sizes=np.full(36,3000);sizes[0]=1500;sizes[1]=12000
    f=HeterogeneousPacketReward(sizes,ttl=3)
    pre=zero();pre[0]=1500;pre[1]=12000;f.before(1,pre)
    served=zero();served[0]=1500;served[1]=3000
    r=f.after(served,served);assert r['complete_payload_bytes']==1500 and r['post_queue_bytes']==9000
    pre=zero();pre[1]=9000;f.before(2,pre)
    served=zero();served[1]=1000;r=f.after(served,zero());assert r['failed_complete_bytes']==0 and r['post_queue_bytes']==8000
    pre=zero();pre[1]=8000;f.before(3,pre);served=zero();served[1]=8000
    r=f.after(served,served);assert r['complete_payload_bytes']==0 and r['failed_complete_bytes']==12000
    # An expired partially served large packet counts its full offered payload.
    f=HeterogeneousPacketReward(sizes,ttl=3);pre=zero();pre[1]=12000;f.before(1,pre)
    served=zero();served[1]=3000;f.after(served,served)
    for t in [2,3]:
        pre=zero();pre[1]=9000;f.before(t,pre);f.after(zero(),zero())
    f.before(4,zero());r=f.after(zero(),zero());assert r['expired_payload_bytes']==12000 and r['post_queue_bytes']==0
    # The original fixed-size implementation and generalized accounting agree.
    a=PacketReward();b=HeterogeneousPacketReward(np.full(36,3000));rng=np.random.default_rng(8742)
    for t in range(1,35):
        arrival=rng.poisson(.8,36)*3000
        pre=np.array([sum(x[1] for x in q) for q in a.queues])+arrival
        a.before(t,pre);b.before(t,pre)
        served=np.minimum(pre,rng.integers(0,7000,36));good=served*(rng.random(36)>.25)
        assert a.after(served,good)==b.after(served,good)
    (OUT/'核验/异构包统计核验.json').write_text(json.dumps(dict(passed=True,tests=['1500 B completion with 12000 B partial service','Failed segment invalidates the entire large packet','Partial-service expiry counts full offered packet bytes','Fixed 3000 B compatibility: 34 random TTIs, exact equality']),indent=2)+'\n')
    print('Packet segmentation, CRC and expiration checks passed.')
if __name__=='__main__':main()

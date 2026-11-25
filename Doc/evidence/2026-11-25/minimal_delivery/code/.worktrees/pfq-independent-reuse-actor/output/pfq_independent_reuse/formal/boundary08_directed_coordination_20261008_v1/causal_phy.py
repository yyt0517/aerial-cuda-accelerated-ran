"""Small pre-action PHY reconstruction for this 3-cell research probe.

Uses estimated cross-link CSI, serving precoders and the existing CTBS snapshot.
No sampled outcome, future state, teacher action, or realized CRC is an input.
"""
from pathlib import Path
import re
import numpy as np
from training.stageb_hgraph.phy_tbs_alignment_audit import (
    determine_tbs_bytes, MCS_CODE_RATES, MCS_QAM_ORDERS,
)

ROOT = Path(__file__).resolve().parents[4]
_TABLES = (ROOT/'cuMAC/src/cumac.h').read_text()
SNR = []
BLER = []
for m in range(28):
    for key, target in [('snrMcs', SNR), ('blerMcs', BLER)]:
        values = re.search(rf'{key}{m}\[\]\s*=\s*\{{([^}}]+)\}}', _TABLES).group(1)
        target.append(np.asarray([float(v.strip()) for v in values.split(',')], dtype=np.float32))

def read_snapshot(path, expected_tti):
    raw = Path(path).read_bytes()
    head = np.frombuffer(raw, dtype='<u4', count=8)
    magic, tti, u, c, p, nr, nt, version = map(int, head)
    assert (magic, version) == (0x44494331, 1)
    assert tti == expected_tti, ('stale directed snapshot', tti, expected_tti)
    assert (u,c,p,nr,nt) == (36,3,17,4,4)
    noise = float(np.frombuffer(raw,dtype='<f4',count=1,offset=32)[0])
    queue = np.frombuffer(raw,dtype='<f4',count=u,offset=36).copy()
    start = 36+4*u
    h = np.frombuffer(raw,dtype='<c8',count=p*u*c*nr*nt,offset=start)
    h = h.reshape(p,u,c,nt,nr).swapaxes(-1,-2).transpose(1,0,2,3,4).astype(np.complex128)
    start += p*u*c*nr*nt*8
    w = np.frombuffer(raw,dtype='<c8',count=u*p*nt*nt,offset=start)
    w = w.reshape(u,p,nt,nt).swapaxes(-1,-2).astype(np.complex128)
    assert len(raw) == start+u*p*nt*nt*8
    assert noise > 0 and np.isfinite(h).all() and np.isfinite(w).all()
    assert np.isfinite(queue).all() and (queue >= 0).all()
    return dict(tti=tti, noise=noise, queue=queue, h=h, w=w)

def sinr_bank(h, w, serving, noise):
    """[U,P,8,L] native MMSE SINR, plus directed receive covariance traces."""
    u,p,c,nr,nt = h.shape
    covariance = h @ h.conj().swapaxes(-1,-2)
    received = np.trace(covariance,axis1=-2,axis2=-1).real
    signal = h[np.arange(u),:,serving] @ w
    gram_identity = np.eye(nt, dtype=np.complex128)
    results = []
    for mask in range(1 << c):
        interferers = ((mask >> np.arange(c)) & 1)[None,:] * (np.arange(c)[None,:] != serving[:,None])
        cov = noise*np.eye(nr)[None,None,:,:] + np.einsum('upcij,uc->upij',covariance,interferers)
        e = gram_identity + signal.conj().swapaxes(-1,-2) @ np.linalg.solve(cov,signal)
        diagonal = np.diagonal(np.linalg.inv(e),axis1=-2,axis2=-1).real
        results.append(np.maximum(1.0/diagonal-1.0,0.0))
    bank = np.stack(results,axis=2)
    assert np.isfinite(bank).all() and (bank >= 0).all()
    return bank, received

class CausalPhy:
    def __init__(self, config):
        self.config = config
        self.tbs = np.zeros((28,18,5),dtype=np.int64)
        for m in range(28):
            for n in range(1,18):
                for layer in range(1,5):
                    self.tbs[m,n,layer] = determine_tbs_bytes(n*config.prbs_per_prg,
                        config.pdsch_data_symbols,layer,MCS_CODE_RATES[m]/1024,MCS_QAM_ORDERS[m])

    def prepare(self, snapshot, ctbs):
        self.snapshot = snapshot
        self.serving = np.asarray(ctbs.serving_cell)
        self.mcs = np.asarray([ctbs.reconstructed_mcs_for_ue(u) for u in range(36)])
        self.caps = np.asarray(ctbs.prg_layer_cap)
        self.bank,self.received = sinr_bank(snapshot['h'],snapshot['w'],self.serving,snapshot['noise'])

    def evaluate(self, actions):
        """actions [B,C,P] contain active UE IDs or -1; returns [B,U] fields."""
        actions = np.asarray(actions)
        if actions.ndim == 2: actions = actions[None]
        assert actions.shape[1:] == (3,17)
        assigned = actions[:,self.serving,:] == np.arange(36)[None,:,None]
        count = assigned.sum(axis=-1)
        layers = np.where(assigned,self.caps[None],5).min(axis=-1)
        layers = np.where(count>0,layers,1)
        capacity = self.tbs[self.mcs[None,:],count,layers]
        dequeue = np.minimum(capacity,self.snapshot['queue'][None,:])
        masks = ((actions >= 0)*(1 << np.arange(3))[None,:,None]).sum(axis=1)
        gamma = self.bank[np.arange(36)[None,:,None],np.arange(17)[None,None,:],masks[:,None,:],:]
        use_layer = np.arange(4)[None,None,:] < layers[:,:,None]
        linear = (gamma*assigned[:,:,:,None]*use_layer[:,:,None,:]).sum(axis=(-1,-2))/np.maximum(1,count*layers)
        db = 10*np.log10(np.maximum(linear,1.e-30))
        clipped = np.maximum(db,-4.75)
        bler = np.zeros_like(db)
        for m in np.unique(self.mcs):
            use = self.mcs == m
            bler[:,use] = np.interp(clipped[:,use],SNR[m][::-1],BLER[m][::-1],left=1.,right=0.)
        expected = dequeue*(1-bler)
        # An interference measure in the simulator's scaled receive-power units.
        active_sources = ((masks[:,:,None] >> np.arange(3)[None,None,:]) & 1)
        cross = self.received * (np.arange(3)[None,None,:] != self.serving[:,None,None])
        interference = np.einsum('upc,bpc->bup',cross,active_sources)
        interference = (interference*assigned).sum(axis=-1)/np.maximum(count,1)
        return dict(count=count,layer=np.where(count>0,layers,-1),mcs=np.where(count>0,self.mcs[None],-1),
            capacity=capacity,dequeue=dequeue,sinr_db=np.where(count>0,db,-999.),
            bler=np.where(count>0,bler,-1.),expected=expected,interference=interference)

    def candidates(self, chosen):
        base = np.asarray(chosen).reshape(3,17)
        tokens = np.flatnonzero(base.reshape(-1)>=0)
        actions = np.repeat(base[None],len(tokens)+1,axis=0)
        for i,token in enumerate(tokens): actions[i+1].reshape(-1)[token] = -1
        prediction = self.evaluate(actions)
        gain = prediction['expected'].sum(axis=1)-prediction['expected'][0].sum()
        best = int(np.argmax(gain))
        if gain[best] <= 1.: best = 0
        return actions,prediction,gain,tokens,best

    def directed_features(self):
        """[U,P,C,2]: log interference/noise and full-reuse removal gain in dB."""
        cross = self.received/(4*self.snapshot['noise'])
        features = []
        for c in range(3):
            full = self.bank[:,:,7,:].mean(axis=-1)
            removed = self.bank[:,:,7^(1<<c),:].mean(axis=-1)
            gain = 10*np.log10(np.maximum(removed,1.e-30)/np.maximum(full,1.e-30))
            valid = self.serving != c
            features.append(np.stack((np.log1p(cross[:,:,c])*valid[:,None],gain*valid[:,None]),axis=-1))
        result = np.stack(features,axis=2).astype(np.float32)
        assert np.isfinite(result).all() and result.min() >= -1.e-5
        return result

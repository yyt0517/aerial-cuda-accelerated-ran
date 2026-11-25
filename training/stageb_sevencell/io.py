"""Dimension-aware copies of frozen binary readers and action packing."""
from pathlib import Path
import numpy as np
from training.stageb_lns.archive import tbs_table

ARRAYS=[('obs_cell_features','<f4'),('obs_ue_features','<f4'),('obs_prg_features','<f4'),
 ('obs_edge_index','<i2'),('obs_edge_attr','<f4'),('action_mask_ue','u1'),('action_mask_cell_ue','u1'),
 ('action_mask_prg_cell','u1'),('post_eq_sinr','<f4'),('teacher_action_ue_select','<i4'),('teacher_action_prg_alloc','<i2')]

def read_fresh(path,template,ctbs):
    raw=Path(path).read_bytes();head=np.frombuffer(raw,dtype='<u8',count=16)
    assert tuple(head[:5])==(0x46524F31,1,ctbs.tti,ctbs.snapshot_id,int(template['n_sched_ue']))
    offset=128;result={'n_sched_ue':np.int32(head[4])}
    for (key,dtype),size in zip(ARRAYS,head[5:]):
        expected=np.asarray(template[key]);assert int(size)==expected.size
        a=np.frombuffer(raw,dtype=dtype,count=int(size),offset=offset).copy();offset+=int(size)*np.dtype(dtype).itemsize
        result[key]=a.reshape(expected.shape).astype(expected.dtype,copy=False)
    assert offset==len(raw) and np.isfinite(result['obs_ue_features']).all()
    assert result['post_eq_sinr'].shape==(len(ctbs.serving_cell),result['action_mask_prg_cell'].shape[1],4)
    return result

def read_snapshot(path,expected_tti):
    raw=Path(path).read_bytes();magic,tti,u,c,p,nr,nt,version=map(int,np.frombuffer(raw,dtype='<u4',count=8))
    assert (magic,version,tti)==(0x44494331,1,expected_tti)
    assert c in (3,7) and u==12*c and (p,nr,nt)==(17,4,4)
    noise=float(np.frombuffer(raw,dtype='<f4',count=1,offset=32)[0]);queue=np.frombuffer(raw,dtype='<f4',count=u,offset=36).copy();start=36+4*u
    h=np.frombuffer(raw,dtype='<c8',count=p*u*c*nr*nt,offset=start).reshape(p,u,c,nt,nr).swapaxes(-1,-2).transpose(1,0,2,3,4).astype(np.complex128)
    start+=p*u*c*nr*nt*8
    w=np.frombuffer(raw,dtype='<c8',count=u*p*nt*nt,offset=start).reshape(u,p,nt,nt).swapaxes(-1,-2).astype(np.complex128)
    assert len(raw)==start+u*p*nt*nt*8 and noise>0 and np.isfinite(h).all() and np.isfinite(w).all()
    assert np.isfinite(queue).all() and (queue>=0).all()
    return dict(tti=tti,noise=noise,queue=queue,h=h,w=w)

def encode(chosen,obs):
    import torch
    from training.gnnrl.slot_layout import build_type0_slot_layout
    C,P=chosen.shape;cm=np.asarray(obs['action_mask_cell_ue'],bool);um=np.asarray(obs['action_mask_ue'],bool)
    slots=len(obs['teacher_action_ue_select']);layout=build_type0_slot_layout(torch.from_numpy(cm),slots,torch.from_numpy(um))
    sc=layout.slot_to_cell[0].numpy();valid=layout.slot_valid_mask[0].numpy()
    ue=np.full(slots,-1,np.int32);prg=np.full(C*P,-1,np.int16)
    for c in range(C):
        ids=np.unique(chosen[c][chosen[c]>=0]);available=np.flatnonzero(valid&(sc==c));assert len(ids)<=len(available)
        for u,s in zip(ids,available):
            assert cm[c,u] and um[u];ue[s]=u
            for p in np.flatnonzero(chosen[c]==u):
                assert obs['action_mask_prg_cell'][c,p];prg[p*C+c]=s
    return ue,prg

def discrete_prediction(chosen,ctbs,snapshot):
    U=len(ctbs.serving_cell);assigned=chosen[np.asarray(ctbs.serving_cell)]==np.arange(U)[:,None]
    count=assigned.sum(1);layer=np.where(assigned,np.asarray(ctbs.prg_layer_cap),5).min(1);layer=np.where(count>0,layer,1)
    mcs=np.array([ctbs.reconstructed_mcs_for_ue(u) for u in range(U)])
    table=tbs_table(ctbs.config.prbs_per_prg,ctbs.config.pdsch_data_symbols,chosen.shape[1]);capacity=table[mcs,count,layer]
    return dict(count=count,mcs=np.where(count>0,mcs,-1),layer=np.where(count>0,layer,-1),capacity=capacity,dequeue=np.minimum(capacity,snapshot['queue']))

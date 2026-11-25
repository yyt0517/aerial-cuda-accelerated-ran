"""Experimental drop-in draft replacement; shared modules remain unchanged."""
from pathlib import Path
import json
import numpy as np
import torch
from training.stageb_hgraph.online_imitation import _align_ctbs_teacher
from training.stageb_hgraph.directed_reuse import build_directed_features,choose_blank
from output.pfq_independent_reuse.formal.boundary75_learning_route_20261015_v1.initialization import old,p,c,prior,two

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
CAP=OUT.parent/'boundary75_service_capacity_20261014_v1'
DIST=OUT.parent/'boundary75_distillation_20261014_v1'
BASE=CAP/'eval_s311_x12_teacher'

def frozen_check():
    frozen=json.loads((OUT.parent/'boundary75_teacher_weight_20261014_v1/frozen_teacher.json').read_text())
    for name,digest in frozen['sha256'].items():assert c.sha(ROOT/name)==digest,name
    return frozen['sha256']

def pfq_draft(obs,graph,ctbs):
    ue=np.asarray(obs['teacher_action_ue_select']).copy()
    raw=np.asarray(obs['teacher_action_prg_alloc']).copy()
    aligned,audit=_align_ctbs_teacher(graph,ue,raw,ctbs)
    prg=aligned.numpy().astype(raw.dtype,copy=True)
    changed=prg!=raw
    assert int(changed.sum())==audit['dropped_count']
    assert (prg[changed]==-1).all() and (raw[changed]>=0).all()
    draft=p.decode_type0_teacher_action(ue,prg,n_cell=3,n_prg=17).reshape(3,17)
    # Check the native action contract without remapping the original slot IDs.
    prior.encode(draft,obs)
    return draft,ue,prg,audit

def coordinate(draft,ue,prg,obs,ctbs,snapshot,coordinator,phy):
    phy.prepare(snapshot,ctbs)
    features=build_directed_features(draft,ctbs,snapshot['queue'],phy.bank,phy.received,snapshot['noise'],obs['obs_ue_features'],phy.tbs)
    with torch.no_grad():scores=coordinator(**{k:torch.from_numpy(v)[None] for k,v in features.items()})[0].numpy()
    chosen,token=choose_blank(draft,scores,features['valid'])
    if token>=0:
        cell,local=divmod(token,17);prg[local*3+cell]=-1
    np.testing.assert_array_equal(p.decode_type0_teacher_action(ue,prg,n_cell=3,n_prg=17).reshape(3,17),chosen)
    return draft,chosen,ue,prg,scores,token

AUDIT=[]
def policy(obs,ctbs,snapshot,builder,edges,actor,coordinator,phy):
    # actor is accepted only for the original runner signature; never evaluated.
    graph=edges.build_graph(builder.build(obs))
    draft,ue,prg,audit=pfq_draft(obs,graph,ctbs)
    result=coordinate(draft,ue,prg,obs,ctbs,snapshot,coordinator,phy)
    AUDIT.append(dict(tti=int(ctbs.tti)+1,ctbs_pruned_prgs=audit['dropped_count'],
        raw_blank=int((np.asarray(obs['teacher_action_prg_alloc'])<0).sum()),
        aligned_blank=int((draft<0).sum()),coordinator_blank_token=int(result[-1]),
        keep_blank=int((result[1]<0).sum())))
    return result

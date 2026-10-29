"""Full legal next-step supervision and sparse approximate-PHY residual labels."""
from pathlib import Path
import importlib.util
import json
import shutil
import time
from dataclasses import replace
import numpy as np
import pandas as pd

from training.stageb_lns.archive import enable_legacy_types,problem_from_row,initial_draft
from training.stageb_lns.experiment import ROOT,sha,write_json,snapshot_sources
from training.stageb_lns.physics import NativePhy,EXPECTED,DEQUEUE
from training.stageb_lns.deadline import packet_objective
from training.stageb_lns.neural_repair import destroy
from training.stageb_lns.cached_repair import CachedDeadlineRepair
from training.stageb_lns.prefix_repair import shortlist
from training.stageb_lns.search import ProposalGraph
from training.stageb_lns.quality import JOINT
from training.stageb_lns.residual import DraftCache

HERE=Path(__file__).resolve().parent
OLD=ROOT/'deliverables/tb_process_value_20261029_v1'
spec=importlib.util.spec_from_file_location('prior_collect',OLD/'collect.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)


def main():
    enable_legacy_types();plan=json.loads((HERE/'plan.json').read_text())
    assert sha(ROOT/plan['actor_checkpoint'])==plan['actor_sha256']
    out=HERE/'data';out.mkdir(exist_ok=False)
    shutil.copyfile(ROOT/plan['actor_checkpoint'],HERE/'actor_original.npz')
    sources=snapshot_sources(out)
    write_json(out/'protocol.json',dict(plan=plan,plan_sha256=sha(HERE/'plan.json'),source_sha256=sources))
    model=CachedDeadlineRepair(HERE/'actor_original.npz',16)
    chunks=[];pair_chunks=[];groups=[];states=[];roots=[];inputs={}
    gid=bid=offset=0;pair_count=0;started=time.perf_counter()
    for split,seed,path,rows in prior.sources(plan):
        inputs[str(path.relative_to(ROOT))]=sha(path)
        for row in rows:
            begin=time.perf_counter();p=problem_from_row(row);phy=NativePhy(p)
            pair_before=pair_count;candidate_before=offset
            try:
                draft=initial_draft(p,row['obs']);initial=phy.evaluate(draft)
                floor=(90*int(initial.values[0,:,DEQUEUE].sum())+99)//100
                graph=ProposalGraph(phy)
                for origin,action in (('own_pfqdraft_ctbs',draft),('saved_chosen',row['chosen'])):
                    base=phy.evaluate(action);objective=packet_objective(p)
                    root_r=float(objective(base.values)[0]);root_j=float(base.values[0,:,EXPECTED]@p.weights)
                    start_cache=time.perf_counter();dc=DraftCache(graph,base)
                    cache_setup_ms=1000*(time.perf_counter()-start_cache)
                    try:
                        cache=model.cache_for(graph,base)
                        blocks=graph.blocks(base.actions[0],base.values[0],replace(JOINT,repair_width=16,ue_topk=p.U),np.random.default_rng(0))
                        selected=list(range(min(8,len(blocks))))
                        if len(blocks)>8:selected+=list(map(int,np.linspace(8,len(blocks)-1,min(4,len(blocks)-8))))
                        for bi in selected:
                            block=blocks[bi];root=destroy(base.actions[0],block)
                            assigned=root[p.serving,:]==np.arange(p.U)[:,None]
                            count=assigned.sum(-1)[None];layer=np.where(assigned,p.caps,5).min(-1)[None]
                            layer=np.where(count>0,layer,4);active=(root>=0)[None]
                            edits=np.full((1,len(block)),-1,np.int64);logp=np.zeros(1)
                            parent_node=np.full(1,-1,np.int32)
                            for depth,(cell,prg) in enumerate(block):
                                choices=cache.node(cell,prg)[0];features=cache.features(block,depth,count,layer,active)
                                logits=model.logits(features);logits-=logits.max(-1,keepdims=True)
                                logits-=np.log(np.exp(logits).sum(-1,keepdims=True))
                                joint=logp[:,None]+logits;n=joint.size
                                parent,value=np.divmod(np.arange(n),len(choices))
                                prefix=edits[parent].copy();prefix[:,depth]=choices[value]
                                complete=np.broadcast_to(base.actions[0],(n,p.C,p.P)).copy()
                                for step,(c,pr) in enumerate(block[:depth+1]):complete[:,c,pr]=prefix[:,step]
                                old_short=shortlist(joint,choices,int(base.actions[0,cell,prg]),16,48,2)
                                old_mask=np.zeros(n,bool);old_mask[old_short]=True
                                approx=dc.evaluate(complete)
                                exact=phy.evaluate(complete,base)
                                delta=objective(exact.values)-root_r
                                eligible=((exact.values[:,:,EXPECTED]@p.weights>root_j+16)&
                                    (exact.values[:,:,DEQUEUE].sum(1)>=floor)&(delta>1))
                                _,first=np.unique(complete.reshape(n,-1),axis=0,return_index=True)
                                unique=np.zeros(n,bool);unique[first]=True
                                rr,uu=np.nonzero(approx['mask'])
                                # Any exact change must lie inside the sparse dependency closure.
                                changed=np.abs(exact.values[:,:,EXPECTED]-base.values[0,:,EXPECTED])>1.e-9
                                assert not np.any(changed&~approx['mask'])
                                chunks.append(dict(actor_x=features.reshape(n,87),group=np.full(n,gid,np.int32),
                                    block=np.full(n,bid,np.int32),seed=np.full(n,seed,np.int16),depth=np.full(n,depth,np.int8),
                                    parent=parent_node[parent],value=choices[value].astype(np.int16),
                                    old_logp=joint.ravel(),old_short=old_mask,unique=unique,
                                    delta=delta,eligible=eligible,draft_scores=approx['scores']))
                                pair_chunks.append(dict(x=approx['features'][rr,uu],raw=approx['raw'][rr,uu],
                                    exact_expected=exact.values[rr,uu,EXPECTED],
                                    candidate=(rr+offset).astype(np.int32),ue=uu.astype(np.int16),
                                    seed=np.full(len(rr),seed,np.int16)))
                                groups.append(dict(group=gid,block=bid,split=split,seed=seed,tti=row['tti'],origin=origin,
                                    block_index=bi,depth=depth,cell=cell,prg=prg,choices=json.dumps(choices.tolist()),
                                    parents=len(logp),options=len(choices),start=offset,end=offset+n,
                                    root_r=root_r,root_j=root_j,floor=floor,
                                    positives=int(eligible.sum()),outside_positives=int((eligible&~old_mask).sum())))
                                order=np.lexsort((np.arange(n),-joint.ravel(),-delta))[:16]
                                pick=choices[value[order]];previous=parent[order]
                                edits=prefix[order];count,layer,active=count[previous],layer[previous],active[previous]
                                active[:,cell,prg]=pick>=0;rows_valid=np.flatnonzero(pick>=0);users=pick[rows_valid]
                                count[rows_valid,users]+=1
                                layer[rows_valid,users]=np.minimum(layer[rows_valid,users],p.caps[users,prg])
                                logp=joint.ravel()[order];parent_node=(offset+order).astype(np.int32)
                                offset+=n;pair_count+=len(rr);gid+=1
                            bid+=1
                        roots.append(dict(split=split,seed=seed,tti=row['tti'],origin=origin,
                            cache_setup_ms=cache_setup_ms,**dc.stats))
                    finally:dc.close()
                states.append(dict(split=split,seed=seed,tti=row['tti'],candidate_occurrences=offset-candidate_before,
                    sparse_pairs=pair_count-pair_before,wall_ms=1000*(time.perf_counter()-begin),**phy.stats))
            finally:phy.close()
            print('COLLECT',split,seed,row['tti'],offset-candidate_before,pair_count-pair_before,flush=True)
            (HERE/'progress.txt').write_text(f'全合法扩展采集：{len(states)}个快照，{offset}次候选，{pair_count}个受影响UE样本。\n')
    candidate={k:np.concatenate([z[k] for z in chunks]) for k in chunks[0]}
    pairs={k:np.concatenate([z[k] for z in pair_chunks]) for k in pair_chunks[0]}
    np.savez_compressed(out/'candidates.npz',**candidate)
    np.savez_compressed(out/'pairs.npz',**pairs)
    pd.DataFrame(groups).to_csv(out/'groups.csv',index=False);pd.DataFrame(states).to_csv(out/'states.csv',index=False)
    pd.DataFrame(roots).to_csv(out/'roots.csv',index=False)
    for path,digest in sources.items():assert sha(ROOT/path)==digest,path
    for path,digest in inputs.items():assert sha(ROOT/path)==digest,path
    write_json(out/'checks.json',dict(passed=True,snapshots=len(states),roots=len(roots),blocks=bid,groups=gid,
        candidate_occurrences=offset,sparse_pairs=pair_count,all_exact_changes_inside_sparse_closure=True,
        input_sha256=inputs,source_sha256=sources,wall_s=time.perf_counter()-started,
        outside_positives=int(pd.DataFrame(groups).outside_positives.sum()),
        registry_sha256=sha(ROOT/'training/stageb_hgraph/current_algorithm.json')))

if __name__=='__main__':main()

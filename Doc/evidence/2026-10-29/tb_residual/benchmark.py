"""Separate C++ migration from sparse affected-UE recomputation and head cost."""
from pathlib import Path
import sys,json,time,ctypes,subprocess,importlib.util
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from common import load
import collect
from training.stageb_lns.archive import enable_legacy_types,problem_from_row,initial_draft
from training.stageb_lns.physics import NativePhy
from training.stageb_lns.search import ProposalGraph
from training.stageb_lns.residual import DraftCache,ResidualValue
from training.stageb_lns.draft_repair import RebasedDraft
from training.stageb_lns.experiment import ROOT,write_json,sha

def spatial_reference(graph,base,actions):
    p=graph.problem
    signal=base.owner.signal;cov=base.owner.covariance
    norm=(np.abs(signal)**2).sum(-2)
    projected=np.maximum(np.einsum('upil,upcij,upjl->upcl',signal.conj(),cov,signal).real/np.maximum(norm[:,:,None,:],1.e-30),0)
    projected[np.arange(p.U),:,p.serving]=0
    original=p.noise+np.einsum('cp,upcl->upl',base.actions[0]>=0,projected)
    def rough(a):
        assigned=a[:,p.serving,:]==np.arange(p.U)[None,:,None]
        count=assigned.sum(-1);layer=np.where(assigned,p.caps,5).min(-1)
        layer=np.where(count>0,layer,0)
        denom=p.noise+np.einsum('bcp,upcl->bupl',a>=0,projected)
        gamma=p.sinr_hint[None]*original[None]/denom
        include=np.arange(4)[None,None,None]<layer[...,None,None]
        gamma=(gamma*include).sum(-1)/np.maximum(layer[...,None],1)
        gamma=(gamma*assigned).sum(-1)/np.maximum(count,1)
        db=np.where(count>0,10*np.log10(np.maximum(gamma,1.e-12)),0)
        return count,layer,db
    count,layer,db=rough(actions);root_db=rough(base.actions)[2][0]
    served=base.values[0,:,0]>0;db[:,served]+=base.values[0,served,4]-root_db[served]
    from training.stageb_lns.physics import native_bler
    from training.stageb_lns.process_value import risk_from_expected
    e=np.minimum(p.queue,p.tbs[np.arange(p.U)[None],count,layer])*(1-native_bler(db,p.mcs))
    return risk_from_expected(p,e)-risk_from_expected(p,base.values[0,:,6])

def main():
    enable_legacy_types();out=HERE/'benchmark';out.mkdir(exist_ok=False)
    plan=json.loads((HERE/'plan.json').read_text());a,_,g=load()
    spec=importlib.util.spec_from_file_location('old_control',collect.OLD/'controls.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    # Same context/layout/gamma tables/BLER scan, force every UE recomputation.
    # This is a benchmark-only ablation, not a second production PHY kernel.
    source=ROOT/'training/stageb_lns/draft_kernel.cpp'
    text=source.read_text();assert text.count('if(!affected[u])continue;')==1
    text=text.replace('int draft_eval(', 'int full_draft_eval(').replace('if(!affected[u])continue;','/* benchmark: recompute every UE */')
    full_cpp=out/'full_recompute.cpp';full_cpp.write_text(text)
    subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-shared','-fPIC',str(full_cpp),'-o',str(out/'full_recompute.so')],check=True)
    lib=ctypes.CDLL(str(out/'full_recompute.so'))
    write_json(out/'protocol.json',dict(kernel_sha256=sha(source),modified_condition='remove affected-UE skip only',
        model_sha256=sha(HERE/'residual_frozen.npz'),repeats=3,
        methods=['full_trace_py','all_ue_cpp','sparse_cpp','sparse_cpp_residual'],
        geometry_root_setup_reported_separately=True,no_parallel_training_or_replay=True))
    model=ResidualValue(HERE/'residual_frozen.npz');measure=[];setups=[];audit_groups=0
    for split,seed,path,rows in collect.prior.sources(plan):
        if split!='evaluation':continue
        for row in rows:
            if row['tti'] not in (2001,3001,4000):continue
            p=problem_from_row(row);phy=NativePhy(p);graph=ProposalGraph(phy)
            try:
                draft=initial_draft(p,row['obs'])
                for origin,root in (('own_pfqdraft_ctbs',draft),('saved_chosen',row['chosen'])):
                    base=phy.evaluate(root)
                    start=time.perf_counter();cache=DraftCache(graph,base)
                    setup=1000*(time.perf_counter()-start)
                    manager=RebasedDraft();manager.cache=cache
                    try:
                        lib.full_draft_eval.argtypes=cache.lib.draft_eval.argtypes
                        lib.full_draft_eval.restype=ctypes.c_int
                        python=old.TraceDraft(graph,base)
                        selected=g[(g.seed==seed)&(g.tti==row['tti'])&(g.origin==origin)]
                        block_actions={};last_base=base
                        for gr in selected.itertuples():
                            ids=np.arange(gr.start,gr.end)
                            parents=a['parent'][ids]
                            if gr.depth==0:actions=np.broadcast_to(root,(len(ids),p.C,p.P)).copy()
                            else:
                                previous_start,previous=block_actions[gr.block]
                                actions=previous[parents-previous_start].copy()
                            actions[:,gr.cell,gr.prg]=a['value'][ids]
                            block_actions[gr.block]=(gr.start,actions)
                            sparse=cache.evaluate(actions)
                            np.testing.assert_allclose(sparse['scores'][:,0],python.score(actions),rtol=1.e-9,atol=1.e-7)
                            np.testing.assert_allclose(sparse['scores'][:,1],spatial_reference(graph,base,actions),rtol=1.e-9,atol=1.e-7)
                            def full():
                                n=len(actions);scores=np.empty((n,2),np.float64)
                                f=np.empty((n,p.U,32),np.float32);raw=np.empty((n,p.U,7),np.float64);mask=np.empty((n,p.U),np.uint8)
                                rc=lib.full_draft_eval(cache.context,n,actions,p.mcs,scores,f,raw,mask);assert rc==0
                                return dict(scores=scores,features=f,raw=raw,mask=mask.astype(bool))
                            dense=full()
                            np.testing.assert_allclose(dense['scores'],sparse['scores'],rtol=1.e-9,atol=1.e-7)
                            np.testing.assert_array_equal(dense['features'][sparse['mask']],sparse['features'][sparse['mask']])
                            def end_to_end():return model.score(cache.evaluate(actions))
                            funcs=[('full_trace_py',lambda:python.score(actions)),('all_ue_cpp',full),
                                   ('sparse_cpp',lambda:cache.evaluate(actions)),('sparse_cpp_residual',end_to_end)]
                            for repeat in range(3):
                                shift=(gr.group+repeat)%4
                                for method,func in funcs[shift:]+funcs[:shift]:
                                    begin=time.perf_counter();func();elapsed=1000*(time.perf_counter()-begin)
                                    measure.append(dict(seed=seed,tti=row['tti'],origin=origin,group=gr.group,block=gr.block,
                                        depth=gr.depth,method=method,repeat=repeat,ms=elapsed,n=len(actions),
                                        affected_pairs=int(sparse['mask'].sum()),full_pairs=len(actions)*p.U))
                            audit_groups+=1
                        # Existing exact root for same graph, independent of tested candidate scores.
                        another=phy.evaluate(root)
                        start=time.perf_counter();manager.for_base(graph,another)
                        rebase=1000*(time.perf_counter()-start)
                        setups.append(dict(seed=seed,tti=row['tti'],origin=origin,setup_ms=setup,rebase_ms=rebase))
                    finally:manager.close()
                print('BENCH',seed,row['tti'],flush=True)
            finally:phy.close()
    d=pd.DataFrame(measure);d.to_csv(out/'timings.csv',index=False);pd.DataFrame(setups).to_csv(out/'setups.csv',index=False)
    m=d.groupby(['seed','tti','origin','group','block','method']).ms.median().rename('ms').reset_index()
    m.to_csv(out/'group_medians.csv',index=False)
    blocks=m.groupby(['seed','tti','origin','block','method']).ms.sum().reset_index()
    blocks.to_csv(out/'block_costs.csv',index=False)
    blocks.groupby(['seed','method']).ms.mean().to_csv(out/'per_seed.csv')
    write_json(out/'checks.json',dict(passed=True,groups=audit_groups,
        trace_cpp_python_match=True,spatial_cpp_python_match=True,
        dense_sparse_scores_match=True,affected_features_bit_exact=True,
        benchmark_only_dense_code_sha256=sha(full_cpp),source_kernel_sha256=sha(source)))
    print('PER BLOCK\n',blocks.groupby('method').ms.mean().to_string(),flush=True)

if __name__=='__main__':main()

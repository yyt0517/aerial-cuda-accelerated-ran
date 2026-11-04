"""Calibrate a single-terminal service gate against full DPLS itself."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from .physics import ROOT,NativePhy,Evaluation,ReferencePhy,EXPECTED,DEQUEUE,COUNT
from .no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .dpls_operations import compact_neighborhood,expand_edits,edit_metadata,canonical_choice
from .ranked_dpls import repair_warm_start
from .cached_budget_campaign import service
from .analyze_dpls_moves import write_json,write_csv,sha


ENSEMBLE=('cold_canonical','warm_canonical','cold_random17','warm_random17','cold_random29','warm_random29')


def weak_ues(p):
    active=np.flatnonzero(p.queue>0)
    return active[np.lexsort((active,-p.weights[active]))[:max(1,int(np.ceil(.2*len(active))))]]


def absolute_service(p,values):
    active=np.flatnonzero(p.queue>0)
    weak=weak_ues(p)
    return dict(unallocated=int(np.sum(values[active,COUNT]==0)),
        effective=int(np.sum(values[active,EXPECTED]>1.)),
        queue_positive_P5=float(np.quantile(values[active,EXPECTED],.05)) if len(active) else 0.,
        deficit_weak_sum=float(values[weak,EXPECTED].sum()))


def ensemble_limits(rows):
    return dict(unallocated_max=float(np.quantile([r['unallocated'] for r in rows],.95)),
        effective_min=float(np.quantile([r['effective'] for r in rows],.05)),
        queue_positive_P5_min=.98*float(np.quantile([r['queue_positive_P5'] for r in rows],.05)),
        deficit_weak_sum_min=.98*float(np.quantile([r['deficit_weak_sum'] for r in rows],.05)))


def ensemble_service_pass(row,limits):
    return (row['unallocated']<=limits['unallocated_max']+1.e-9 and
            row['effective']>=limits['effective_min']-1.e-9 and
            row['queue_positive_P5']+1.e-6>=limits['queue_positive_P5_min'] and
            row['deficit_weak_sum']+1.e-6>=limits['deficit_weak_sum_min'])


def full_variant(p,draft,previous,*,warm=False,tie='canonical',seed=17,shuffle=False):
    rng=np.random.default_rng(seed)
    phy=NativePhy(p)
    try:
        root=phy.evaluate(draft)
        score=float(root.values[0,:,EXPECTED]@p.weights)
        floor=(90*int(root.values[0,:,DEQUEUE].sum())+99)//100
        if warm:
            other=phy.evaluate(repair_warm_start(p,previous))
            j=float(other.values[0,:,EXPECTED]@p.weights)
            if other.values[0,:,DEQUEUE].sum()>=floor and j>score:root,score=other,j
        for iteration in range(1,257):
            prg,assignment=compact_neighborhood(root.actions[0],p.legal)
            keys,_,_=edit_metadata(root.actions[0],prg,assignment,p.U)
            actions=expand_edits(root.actions[0],prg,assignment)
            order=rng.permutation(len(keys)) if shuffle else np.arange(len(keys))
            result=phy.evaluate(actions[order],root)
            values=np.empty_like(result.values);values[order]=result.values
            gain=values[:,:,EXPECTED]@p.weights-score
            eligible=values[:,:,DEQUEUE].sum(1)>=floor
            threshold=1. if iteration<=2 else 16.
            chosen=canonical_choice(gain,eligible,keys,threshold)
            if tie=='random':
                pool=np.flatnonzero(eligible & (gain>threshold))
                if len(pool):chosen=int(rng.choice(pool[gain[pool]>=gain[pool].max()-1.e-6]))
            if chosen>=0:
                a=actions[chosen:chosen+1].copy();v=values[chosen:chosen+1].copy()
                a.flags.writeable=v.flags.writeable=False
                root=Evaluation(a,v,phy);score=float(v[0,:,EXPECTED]@p.weights)
            if iteration>=3 and chosen<0:break
        else:raise AssertionError('Full variant did not converge')
        return root.actions[0].copy(),root.values[0].copy(),dict(rounds=iteration,J=score,**phy.stats)
    finally:phy.close()


def main(args):
    import torch
    from .archive import enable_legacy_types
    enable_legacy_types()
    source,reference,out=map(lambda x:Path(x).resolve(),(args.source,args.reference,args.output))
    out.mkdir(parents=True,exist_ok=False)
    paths=sorted(source.glob('*_seed*/states/tti*.pt'))
    files=[Path(__file__),*[Path(__file__).with_name(n) for n in ('physics.py','_physics.so','dpls_operations.py','no_ctbs_inputs.py')]]
    (out/'source').mkdir()
    sources={str(f.relative_to(ROOT)):sha(f) for f in files}
    for f in files:shutil.copy2(f,out/'source'/f.name)
    write_json(out/'source_manifest.json',sources)
    write_json(out/'input_manifest.json',{str(p):sha(p) for p in paths})
    write_json(out/'protocol.json',dict(ensemble=list(ENSEMBLE),seed_rule='fixed17/29 plus source-state index',
        source=str(source),reference=str(reference),
        original_gate='J>=single full99.5%, no new allocated-UE loss, positive-service P5 and weak sum>=98%',
        diagnostic='report full-vs-full gate and J-close conditional service gate separately; cold/warm paired both directions',
        ensemble_gate='retain J>=source canonical99.5%; service absolute metrics within fixed full ensemble5/95% bands, 98% lower tolerance on byte metrics',
        service_metrics='current positive-queue universe: unallocated/effective count, P5 expected bytes, top20% causal deficit-weight UE expected sum',
        weak_set='positive-queue highest deficit weights, stable UE-id tie; does not read a teacher terminal',
        degenerate='zero P5 band is explicitly reported as noninformative',
        loo='leave-one-variant-out service false rejection diagnostic; not a confidence or long-term fairness guarantee',
        negative_control='canonical source initializer plus shuffled candidate enumeration must match source canonical',
        source_initializers='same captured input and actual source predecessor, no new native trajectories',
        native_entry='guarded cap4 only if all12 states pass source J and frozen ensemble service gate; native packets decide long-term service'))
    metrics=[];pairs=[];loo=[];bounds=[];work=[]
    for state,path in enumerate(paths):
        row=torch.load(path,map_location='cpu',weights_only=False)
        p=problem_from_raw_row(row);draft=raw_pfq_draft(p,row['obs'])
        refpath=reference/f"{row['source_arm']}_seed{row['seed']}_tti{row['tti']}.npz"
        with np.load(refpath) as z:ref_values=z['values'].copy();ref_action=z['action'].copy()
        ref_j=float(ref_values[:,EXPECTED]@p.weights)
        answers={};state_rows=[]
        variants=('source_canonical',*ENSEMBLE,'source_shuffle')
        for variant in variants:
            warm=(variant.startswith('warm') or variant.startswith('source') and row['source_arm'].startswith('warm'))
            tie='random' if 'random' in variant else 'canonical'
            seed=(29 if '29' in variant else 17)+state*101
            a,v,r=full_variant(p,draft,row['previous_action'],warm=warm,tie=tie,seed=seed,shuffle=variant.endswith('shuffle'))
            if variant in ('source_canonical','source_shuffle'):
                np.testing.assert_array_equal(a,ref_action)
                np.testing.assert_array_equal(v,ref_values)
            np.testing.assert_allclose(v,ReferencePhy(p).evaluate(a)[0],rtol=1.e-8,atol=1.e-6)
            s=service(v,ref_values)
            passed=r['J']>=.995*ref_j and s['newly_unallocated']==0 and s['positive_p5_ratio']>=.98 and s['positive_weak_ratio']>=.98
            item=dict(state=state,source_arm=row['source_arm'],tti=row['tti'],variant=variant,
                J=r['J'],J_ratio=r['J']/ref_j,old_gate_passed=passed,**s,**absolute_service(p,v))
            metrics.append(item);state_rows.append(item);answers[variant]=(a,v,r)
            work.append(dict(state=state,variant=variant,**r))
            np.savez_compressed(out/f'state{state:02d}_{variant}.npz',action=a,values=v)
        distribution=[r for r in state_rows if r['variant'] in ENSEMBLE]
        limit=ensemble_limits(distribution)
        bounds.append(dict(state=state,source_arm=row['source_arm'],tti=row['tti'],reference_J=ref_j,
                           P5_informative=limit['queue_positive_P5_min']>1.e-6,**limit))
        for item in distribution:
            other=[r for r in distribution if r['variant']!=item['variant']]
            loo.append(dict(state=state,variant=item['variant'],service_passed=ensemble_service_pass(item,ensemble_limits(other))))
        for va in ENSEMBLE:
            for vb in ENSEMBLE:
                if va==vb:continue
                actual,rv=answers[va][1],answers[vb][1]
                ja,jb=float(actual[:,EXPECTED]@p.weights),float(rv[:,EXPECTED]@p.weights)
                s=service(actual,rv)
                ok=s['newly_unallocated']==0 and s['positive_p5_ratio']>=.98 and s['positive_weak_ratio']>=.98
                pairs.append(dict(state=state,actual=va,reference=vb,J_ratio=ja/jb,
                    J_close=abs(ja/jb-1.)<=.005,service_passed=ok,old_gate_passed=ja>=.995*jb and ok,
                    **s))
        print('CALIBRATED',state,flush=True)
    for name,data in [('metrics',metrics),('pairs',pairs),('leave_one_out',loo),('ensemble_limits',bounds),('work',work)]:write_csv(out/(name+'.csv'),data)
    write_json(out/'ensemble_limits.json',bounds)
    close=[r for r in pairs if r['J_close']]
    primary=[r for r in pairs if {r['actual'],r['reference']}=={'cold_canonical','warm_canonical'}]
    write_json(out/'summary.json',dict(cold_warm_pairs=len(primary),cold_warm_old_gate_passes=sum(r['old_gate_passed'] for r in primary),
        all_full_pairs=len(pairs),old_gate_passes=sum(r['old_gate_passed'] for r in pairs),
        J_close_pairs=len(close),J_close_service_passes=sum(r['service_passed'] for r in close),
        leave_one_out_service_passes=sum(r['service_passed'] for r in loo),leave_one_out_comparisons=len(loo),
        informative_P5_states=sum(r['P5_informative'] for r in bounds)))
    for name,digest in sources.items():
        if sha(ROOT/name)!=digest:raise AssertionError('Source changed: '+name)
    write_json(out/'completion.json',dict(completed=True,decisions=len(work),new_native_runs=0,
        candidate_evaluations=sum(r['candidate_evaluations'] for r in work),mmse_misses=sum(r['mmse_misses'] for r in work),
        numpy_terminal_audits=len(work),canonical_shuffle_control_passed=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--output',required=True);main(p.parse_args())

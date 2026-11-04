"""Separately paid finite-pool recall audit, after all policy selections."""
import json
import time
import numpy as np
from .directed_reorder import DirectedFeatures,DirectedConfig,target_ues,choose_batch
from .weak_service import ServiceMetric
from .joint_reorder import JointFeatures
from .resource_conditions import ConditionalPhy,satisfies
from .resource_conditions_fast import ordered_unique
from .resource_condition_campaign import compressed_condition
from .physics import ReferencePhy,EXPECTED,DEQUEUE,COUNT
from .no_ctbs_inputs import problem_from_raw_row
from .ranked_dpls_campaign import read_rows,identity
from .bounded_probes import csv_write
from .experiment import write_json


def audit(source,configs):
    source=source.resolve();out=source/'recall_diagnosis';out.mkdir(exist_ok=False)
    selection=json.loads((source/'selection.json').read_text());chosen=selection['quality_candidate']
    arms=list(dict.fromkeys(['baseline',chosen]));rows_out=[];summaries=[];work=[];checks=[];all_actions={};all_values={}
    import csv
    for phase,seeds,ttis in [('validation',(44,45),(4001,8001,18000)),('evaluation',(46,47,48),(3001,4001,5000))]:
        records=list(csv.DictReader((source/f'{phase}_per_state.csv').open()))
        severity={}
        for r in records:
            if r['arm'] not in arms:continue
            short=max(.995-float(r['ratio']),0)*10+max(.98-float(r['positive_p5_ratio']),0)+max(.98-float(r['positive_weak_ratio']),0)
            if short>0:severity[r['state_id']]=max(severity.get(r['state_id'],0),short)
        selected=sorted(severity,key=lambda s:(-severity[s],s))[:2]
        raw,_=read_rows(seeds,evaluation=phase=='evaluation');raw={identity(r):r for r in raw if r['tti'] in ttis}
        actions=np.load(source/f'{phase}_actions.npz');values=np.load(source/f'{phase}_values.npz')
        visited=np.load(source/f'{phase}_observed_actions.npz')
        for sid in selected:
            row=raw[sid];p=problem_from_raw_row(row);reference=values[f'{sid}|full'];full_j=float(reference[:,EXPECTED]@p.weights)
            condition=compressed_condition(p,actions[f'{sid}|full'],reference,'joint212')
            for arm in arms:
                key=f'{sid}|{arm}';root=actions[key];root_values=values[key];start=time.perf_counter();phy=ConditionalPhy(p)
                try:
                    base=phy.evaluate(root);np.testing.assert_allclose(base.values[0],root_values,rtol=2e-9,atol=2e-6)
                    bank=phy.bank();features=JointFeatures(p,bank,condition);directed=DirectedFeatures(features)
                    group=(condition.counts>0)&(root_values[:,DEQUEUE]>0)
                    targets=target_ues(p,root_values,group);wide=features.pool(base,True,20261102);macro=directed.pool(base,targets,4)
                    wide_keys={a.tobytes() for a in wide};macro_keys={a.tobytes() for a in macro};seen={a.tobytes() for a in visited[key]}
                    pool=ordered_unique(np.concatenate([wide,macro]));generated=len(pool)
                    pool=pool[np.array([a.tobytes() not in seen for a in pool],bool)]
                    assert len(pool)<=8192 and satisfies(p,pool,condition).all()
                    cfg=configs.get(arm,DirectedConfig(directed_slots=0,joint_candidates=False))
                    metric=ServiceMetric(p,condition,group,cfg);metric.weights=p.weights
                    if len(pool):
                        scores,pred,valid=features.proxy(base,pool);assert valid.all()
                        exact=phy.evaluate(pool,base).values
                        direct,_=choose_batch(scores,pred,valid,base,metric,targets,8,cfg,0)
                    else:scores=np.empty(0);exact=np.empty((0,p.U,8));direct=[]
                    stats=phy.stats.copy()
                finally:phy.close()
                elapsed=(time.perf_counter()-start)*1000
                j=exact[:,:,EXPECTED]@p.weights;root_j=float(root_values[:,EXPECTED]@p.weights)
                target=np.flatnonzero(reference[:,EXPECTED]>1)
                weak=target[np.argsort(reference[target,EXPECTED],kind='stable')[:max(1,int(np.ceil(.2*len(target))))]]
                if len(pool):
                    p5=np.quantile(exact[:,target,EXPECTED],.05,axis=1)/np.quantile(reference[target,EXPECTED],.05)
                    wr=exact[:,weak,EXPECTED].sum(1)/reference[weak,EXPECTED].sum()
                    coverage=(p5>=.98)&(wr>=.98)&~((reference[None,:,COUNT]>0)&(exact[:,:,COUNT]==0)).any(1)
                else:p5=wr=np.empty(0);coverage=np.empty(0,bool)
                passed=coverage&(j>=.995*full_j);order=np.argsort(-np.round(scores,7),kind='stable');rank=np.empty(len(order),int);rank[order]=np.arange(1,len(order)+1)
                wide_mask=np.array([a.tobytes() in wide_keys for a in pool],bool);macro_mask=np.array([a.tobytes() in macro_keys for a in pool],bool)
                novel=macro_mask&~wide_mask;direct_mask=np.zeros(len(pool),bool);direct_mask[direct]=True
                for family,mask in [('union',np.ones(len(pool),bool)),('wide',wide_mask),('directed',macro_mask),('novel_directed',novel)]:
                    ids=np.flatnonzero(mask);best=int(ids[np.argmax(j[ids])]) if len(ids) else None
                    top=order[mask[order]][:8];picked=np.flatnonzero(mask&direct_mask)
                    summaries.append(dict(phase=phase,state_id=sid,arm=arm,family=family,candidates=len(ids),
                        positive_j_candidates=int((j[ids]>root_j+1).sum()),coverage_candidates=int(coverage[ids].sum()),joint_candidates=int(passed[ids].sum()),
                        best_j_gain=float(j[best]-root_j) if best is not None else None,
                        best_j_relative_pct=float(100*(j[best]/full_j-1)) if best is not None else None,
                        best_j_global_proxy_rank=int(rank[best]) if best is not None else None,
                        proxy_top8_joint=int(passed[top].sum()),directed_top8_joint=int(passed[picked].sum()),
                        proxy_top8_best_gain=float(j[top].max()-root_j) if len(top) else None,
                        directed_top8_best_gain=float(j[picked].max()-root_j) if len(picked) else None,
                        best_p5=float(p5[ids].max()) if len(ids) else None))
                for i in range(len(pool)):
                    rows_out.append(dict(phase=phase,state_id=sid,arm=arm,index=i,in_wide=bool(wide_mask[i]),in_directed=bool(macro_mask[i]),
                        novel_directed=bool(novel[i]),proxy_j=float(scores[i]),exact_j=float(j[i]),j_gain=float(j[i]-root_j),
                        relative_j_pct=float(100*(j[i]/full_j-1)),p5_ratio=float(p5[i]),weak_ratio=float(wr[i]),joint_pass=bool(passed[i]),
                        proxy_rank=int(rank[i]),selected_by_directed8=bool(direct_mask[i]),
                        edited_slots=int((pool[i]!=root).sum()),edited_cells=int(np.any(pool[i]!=root,axis=1).sum())))
                audit_ids=[]
                if len(pool):
                    audit_ids.extend([int(np.argmax(j)),int(np.argmax(p5))])
                    if novel.any():audit_ids.append(int(np.flatnonzero(novel)[np.argmax(j[novel])]))
                    if passed.any():audit_ids.append(int(np.flatnonzero(passed)[0]))
                    audit_ids.extend(direct[:2])
                for i in [-1,*dict.fromkeys(audit_ids)]:
                    a=root if i<0 else pool[i];v=root_values if i<0 else exact[i]
                    independent=ReferencePhy(p).evaluate(a)[0];np.testing.assert_allclose(v,independent,rtol=2e-9,atol=2e-6)
                    checks.append(dict(phase=phase,state_id=sid,arm=arm,index=i,max_error=float(abs(v-independent).max())))
                label=f'{phase}|{key}';all_actions[label]=pool;all_values[label]=exact
                work.append(dict(phase=phase,state_id=sid,arm=arm,generated_unique=generated,previously_paid_skipped=generated-len(pool),
                    unseen_exact_candidates=len(pool),whole_pool_scored=True,ms=elapsed,**stats,
                    proxy_gamma_reads=features.gamma_reads,directed_gamma_reads=directed.gamma_reads,
                    directed_attempts=directed.attempts,directed_combinations=directed.combinations))
                assert stats['candidate_evaluations']==1+len(pool)
                print('DIRECTED_RECALL',phase,sid,arm,len(pool),int(passed.sum()),flush=True)
    csv_write(out/'candidates.csv',rows_out);csv_write(out/'summary.csv',summaries);csv_write(out/'work.csv',work)
    np.savez_compressed(out/'actions.npz',**all_actions);np.savez_compressed(out/'values.npz',**all_values)
    write_json(out/'checks.json',dict(passed=True,independent_phy=checks))
    write_json(out/'summary.json',dict(post_selection=True,policy_changed=False,training=False,selected_quality=chosen,
        roots=len(work),max_roots=8,exact_scores=sum(w['candidate_evaluations'] for w in work),
        unique_mmse=sum(w['mmse_misses'] for w in work),independent_audits=len(checks),all_finite_pools_complete=True,
        rule='up to2 failing states per phase by combined quality/service shortfall; baseline and validation-selected quality method',
        scores_separate_from_formal212=True,teacher_preparation_reused_from_paid_campaign=True,
        limits='posthoc finite unseen candidate pool only; root/proxy diagnostics at iteration0, not actual online continuation; teacher-positive service used only for labels'))

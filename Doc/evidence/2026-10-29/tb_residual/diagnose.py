"""Diagnose model error on its own visited prefixes; never retrain on these labels."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import replay
from training.stageb_lns.fast_draft import FastDraftRepair
from training.stageb_lns.deadline import packet_objective
from training.stageb_lns.experiment import write_json

class AuditRepair(FastDraftRepair):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.rows=[];self.block_index=0

    def repair(self,graph,base,block,evaluate):
        cache=self.drafts.for_base(graph,base);original=cache.evaluate
        ctx={};self.block_index+=1
        def spy(actions):
            value=original(actions);ctx['actions']=actions;ctx['draft']=value
            ctx['depth']=ctx.get('depth',0)+1
            return value
        cache.evaluate=spy
        if self.residual is not None:
            old_score=self.residual.score
            def score(draft):
                result=old_score(draft);ctx['pred']=result[0];return result
            self.residual.score=score
        root_r=float(packet_objective(graph.problem)(base.values)[0])
        def verified(actions,source):
            result=evaluate(actions,source)
            if result is None:return None
            lookup={a.tobytes():i for i,a in enumerate(ctx['actions'])}
            true=packet_objective(graph.problem)(result.values)-root_r
            for action,value in zip(result.actions,true):
                i=lookup[action.tobytes()];draft=ctx['draft']['scores'][i,0]
                pred=ctx['pred'][i] if self.residual is not None else draft
                self.rows.append(dict(block=self.block_index,depth=ctx['depth'],true_delta=value,
                    trace_delta=draft,predicted_delta=pred,trace_error=draft-value,predicted_error=pred-value))
            return result
        try:return super().repair(graph,base,block,verified)
        finally:
            cache.evaluate=original
            if self.residual is not None:self.residual.score=old_score

def main():
    replay.enable_legacy_types();out=HERE/'diagnostics';out.mkdir(exist_ok=False)
    plan=json.loads((HERE/'plan.json').read_text());reference=pd.read_csv(HERE/'optimized/states.csv')
    errors=[];runs=[]
    for split,seed,path,rows in replay.collect.prior.sources(plan):
        if split!='evaluation':continue
        for row in rows:
            if row['tti'] not in (2001,3001,4000):continue
            p=replay.problem_from_row(row);draft=replay.initial_draft(p,row['obs'])
            for mode in ('trace','residual'):
                model=AuditRepair(HERE/'actor_original.npz',mode=mode,
                    residual_checkpoint=HERE/'residual_frozen.npz',exact_quota=8)
                try:a,v,r=replay.run(mode+'_original',p,draft,model)
                finally:model.close()
                expected=reference[(reference.seed==seed)&(reference.tti==row['tti'])&
                    (reference.method==mode+'_original')&(reference.quota==8)&(~reference.fast)].iloc[0]
                np.testing.assert_allclose(r['objective_score'],expected.risk,rtol=0,atol=1.e-8)
                assert r['candidate_evaluations']==expected.candidates
                errors += [dict(seed=seed,tti=row['tti'],method=mode,**e) for e in model.rows]
                runs.append(dict(seed=seed,tti=row['tti'],method=mode,candidates=r['candidate_evaluations'],risk=r['objective_score']))
    d=pd.DataFrame(errors);d.to_csv(out/'self_search_verified_errors.csv',index=False)
    pd.DataFrame(runs).to_csv(out/'runs.csv',index=False)
    summaries=[]
    for (seed,method),frame in d.groupby(['seed','method']):
        summaries.append(dict(seed=seed,method=method,n=len(frame),
            trace_mae=float(frame.trace_error.abs().mean()),residual_mae=float(frame.predicted_error.abs().mean()),
            trace_bias=float(frame.trace_error.mean()),residual_bias=float(frame.predicted_error.mean()),
            overestimate_gt1000=float((frame.predicted_error>1000).mean())))
    pd.DataFrame(summaries).to_csv(out/'self_search_summary.csv',index=False)
    write_json(out/'checks.json',dict(passed=True,unchanged_final_scores_and_counts=True,
        models_not_retrained=True,evaluation_labels_not_added_to_training=True,
        interpretation='Selected exact-verified candidates from each policy own saved-state search. Different visited distributions, not unbiased error or native on-policy packet outcomes'))
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)

if __name__=='__main__':main()

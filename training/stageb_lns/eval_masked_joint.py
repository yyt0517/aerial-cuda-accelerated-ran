"""Frozen-state quality, escape witnesses and serial end-to-end timing.

This runner never updates a model. Escape preparation is explicitly separated
from direct-from-draft timing. A physical beam stop is not an exact optimum.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import time
import numpy as np

from .archive import ROOT, enable_legacy_types, problem_from_row, initial_draft
from .experiment import write_json, sha
from .physics import NativePhy, ReferencePhy, EXPECTED, DEQUEUE
from .search import solve, greedy_candidates
from .quality import FULL, JOINT
from .cached_repair import CachedJointRepair
from .masked_joint import MaskedJointRepair
from .masked_search import solve_masked, MaskedConfig

ARMS=('greedy','neural79_cached','physical4','physical16',
      'masked_joint','neural79_cap73','physical4_cap73','masked_no_global')


def evaluation_rows(protocol):
    import torch
    enable_legacy_types();rows=[];files={}
    for seed in protocol['evaluation_seeds']:
        for family in protocol['evaluation_sources']:
            path=ROOT/f'deliverables/tb_joint_repair_20261027_native_v2/seed{seed}_{family}/states.pt'
            saved=torch.load(path,map_location='cpu',weights_only=False)
            for tti in protocol['evaluation_ttis']:
                matches=[r for r in saved if r['tti']==tti]
                if len(matches)!=1:raise ValueError(f'Missing/duplicate prescribed state: {path} {tti}')
                rows.append(dict(matches[0],seed=seed,family=family,source=str(path),
                                 state_id=f'{family}:seed{seed}:tti{tti}'))
            files[str(path)]=sha(path)
    return rows,files


def strict_greedy(problem,draft,root):
    """Certify a strict local optimum of the enumerated original neighborhood."""
    start=time.perf_counter();phy=NativePhy(problem)
    try:
        original=phy.evaluate(draft)
        floor=(90*int(original.values[0,:,DEQUEUE].sum())+99)//100
        incumbent=phy.evaluate(root);score=float(incumbent.values[0,:,EXPECTED]@problem.weights)
        for iteration in range(256):
            actions=greedy_candidates(problem,incumbent.actions[0])
            if not len(actions):
                best_delta=0.;break
            values=phy.evaluate(actions,incumbent)
            eligible=values.values[:,:,DEQUEUE].sum(1)>=floor
            scores=values.values[:,:,EXPECTED]@problem.weights
            if not eligible.any():best_delta=0.;break
            best=int(np.argmax(np.where(eligible,scores,-np.inf)))
            best_delta=float(scores[best]-score)
            if best_delta<=1.e-7:break
            incumbent=phy.take(values,best);score=float(scores[best])
        else:raise RuntimeError('Strict greedy certificate failed to converge')
        return incumbent.actions[0].copy(),dict(score=score,floor=floor,
                 best_neighbor_delta=best_delta,strict_local_certificate=True,
                 rounds=iteration+1,preparation_ms=(time.perf_counter()-start)*1000,**phy.stats)
    finally:phy.close()


def run_arm(row,arm,neural,new,initial_action=None):
    started=time.perf_counter()
    problem=problem_from_row(row);draft=initial_draft(problem,row['obs'])
    converted=time.perf_counter()
    if arm=='greedy':config=FULL
    elif arm in ('neural79_cached','neural79_cap73'):
        config=JOINT if arm=='neural79_cached' else replace(JOINT,max_candidates=73)
    elif arm in ('physical4','physical16','physical4_cap73'):
        config=replace(JOINT,repair_width=16,ue_topk=problem.U,
                       block_size=16 if arm=='physical16' else 4,
                       max_candidates=73 if arm=='physical4_cap73' else JOINT.max_candidates)
    elif arm in ('masked_joint','masked_no_global'):
        action,values,record=solve_masked(problem,draft,new,
                    MaskedConfig(global_reconstruction=arm=='masked_joint'),initial_action=initial_action)
        record.update(e2e_ms=(time.perf_counter()-started)*1000,
                      conversion_ms=(converted-started)*1000)
        return action,values,record
    else:raise ValueError(arm)
    action,values,record=solve(problem,draft,config,
                    repairer=neural if arm.startswith('neural79') else None,initial_action=initial_action)
    record['draft_score']=record['initial_score']
    record.update(e2e_ms=(time.perf_counter()-started)*1000,
                  conversion_ms=(converted-started)*1000)
    return action,values,record


def main(args):
    import torch
    import pandas as pd
    torch.set_num_threads(1)
    source=Path(args.training).resolve();protocol=json.loads((source/'protocol.json').read_text())
    out=Path(args.output).resolve() if args.output else source/'evaluation'
    out.mkdir(exist_ok=False)
    checkpoint=source/'repair.npz';frozen79=Path(protocol['frozen79'])
    if sha(frozen79)!=protocol['frozen79_sha256']:raise ValueError('79-input baseline checkpoint changed')
    new=MaskedJointRepair(checkpoint);neural=CachedJointRepair(frozen79,width=16)
    rows,files=evaluation_rows(protocol)
    write_json(out/'protocol.json',dict(training_protocol=protocol,checkpoint_sha256=sha(checkpoint),
                input_files=files,states=[r['state_id'] for r in rows],arms=ARMS,repeats=args.repeats,
                order_seed=9917,host=platform.node(),python=platform.python_version(),
                numpy=np.__version__,torch=torch.__version__,cpu=platform.processor(),
                timing='serial; decoded snapshot to action; includes conversion, PHY setup, features, model, generation, verification, cleanup',
                excluded='archive reading, model/library loading, one-time immutable TBS tables, IPC and simulator',
                escape='strict original-neighborhood local certificate and converged bounded physical4 beam stop; preparation excluded from continuation timing'))
    # Warm one decision per implementation, outside measured repetitions. Every
    # measured decision creates a new PHY context and therefore cold gamma cache.
    for arm in ARMS:run_arm(rows[0],arm,neural,new)
    rng=np.random.default_rng(9917);measurements=[];outputs={};direct_records={}
    with (out/'decisions.jsonl').open('x') as stream:
        for row_index,row in enumerate(rows):
            for repetition in range(args.repeats):
                for arm in rng.permutation(ARMS):
                    action,values,record=run_arm(row,str(arm),neural,new)
                    key=(row['state_id'],str(arm))
                    if key in outputs:
                        np.testing.assert_array_equal(action,outputs[key][0])
                        np.testing.assert_array_equal(values,outputs[key][1])
                        if record['candidate_evaluations']!=direct_records[key]['candidate_evaluations']:
                            raise AssertionError('Repeated decision changed work count')
                    else:outputs[key]=(action.copy(),values.copy());direct_records[key]=record
                    item=dict(state_id=row['state_id'],seed=row['seed'],family=row['family'],tti=row['tti'],
                              arm=str(arm),repetition=repetition,**record)
                    stream.write(json.dumps(item)+'\n');stream.flush()
                    flat={k:v for k,v in item.items() if isinstance(v,(str,int,float,bool))}
                    if 'model' in record:flat.update({f'model_{k}':v for k,v in record['model'].items()})
                    measurements.append(flat)
            print('TIMED',row_index+1,len(rows),row['state_id'],flush=True)
    pd.DataFrame(measurements).to_csv(out/'measurements.csv',index=False)

    escape=[];certificates=[];witnesses=[];audits=[];escape_roots={}
    for row_index,row in enumerate(rows):
        problem=problem_from_row(row);draft=initial_draft(problem,row['obs'])
        greedy=outputs[(row['state_id'],'greedy')][0]
        strict,certificate=strict_greedy(problem,draft,greedy)
        certificates.append(dict(state_id=row['state_id'],seed=row['seed'],**certificate))
        strong,_,stop=solve(problem,draft,replace(JOINT,repair_width=16,ue_topk=problem.U,max_rounds=32),initial_action=strict)
        if stop['stop_reason']!='no_improvement_in_visited_neighborhood':
            raise RuntimeError('Strong physical4 preparation hit a limit, not a search stop')
        for root_name,root,root_score in [('strict_greedy',strict,certificate['score']),
                                         ('physical4_stop',strong,stop['score'])]:
            escape_roots[f'{row_index}_{root_name}']=root.copy()
            for arm in ('masked_joint','neural79_cached','physical16'):
                action,values,record=run_arm(row,arm,neural,new,initial_action=root)
                delta=record['score']-root_score
                escape.append(dict(state_id=row['state_id'],seed=row['seed'],family=row['family'],tti=row['tti'],
                              root=root_name,arm=arm,root_score=root_score,score=record['score'],gain=delta,
                              escaped=delta>1.,additional_ms=record['e2e_ms'],
                              candidates=record['candidate_evaluations'],floor=record['floor'],
                              preparation_ms=certificate['preparation_ms']+(stop['core_ms'] if root_name=='physical4_stop' else 0.)))
                # Independent NumPy evaluation is outside all measured runs.
                independent=ReferencePhy(problem).evaluate(action)[0]
                np.testing.assert_allclose(values,independent,rtol=1.e-8,atol=1.e-6)
                if arm=='masked_joint' and delta>1.:
                    reference=ReferencePhy(problem).evaluate(np.stack((root,action)))
                    independent_gain=float((reference[1,:,EXPECTED]-reference[0,:,EXPECTED])@problem.weights)
                    if abs(delta-independent_gain)>1.e-4:raise AssertionError('Escape witness gain mismatch')
                    witnesses.append(dict(state_id=row['state_id'],root_kind=root_name,root=root.tolist(),
                                          action=action.tolist(),gain=delta,independent_gain=independent_gain,
                                          floor=record['floor'],dequeue=float(values[:,DEQUEUE].sum()),
                                          changed_positions=int((root!=action).sum()),
                                          best_original_neighbor_delta=certificate['best_neighbor_delta'] if root_name=='strict_greedy' else None))
        for arm in ARMS:
            action,values=outputs[(row['state_id'],arm)]
            independent=ReferencePhy(problem).evaluate(action)[0]
            np.testing.assert_allclose(values,independent,rtol=1.e-8,atol=1.e-6)
            audits.append(dict(state_id=row['state_id'],arm=arm,max_phy_error=float(np.abs(values-independent).max())))
        print('ESCAPE',row_index+1,len(rows),row['state_id'],flush=True)
    pd.DataFrame(escape).to_csv(out/'escape.csv',index=False)
    write_json(out/'certificates.json',certificates)
    write_json(out/'witnesses.json',witnesses)
    write_json(out/'checks.json',dict(passed=True,repeated_actions_and_values_exact=True,
                unchanged_79_checkpoint=sha(frozen79)==protocol['frozen79_sha256'],
                independent_final_audits=audits,strict_certificates=len(certificates),
                caveat='A physical4 beam stop is not a proof of exact local or global optimality'))
    np.savez_compressed(out/'actions.npz',**{
        f'{i}_{arm}':outputs[(r['state_id'],arm)][0] for i,r in enumerate(rows) for arm in ARMS})
    np.savez_compressed(out/'escape_roots.npz',**escape_roots)
    summarize(out)


def summarize(out):
    import pandas as pd
    f=pd.read_csv(out/'measurements.csv')
    keys=['state_id','seed','family','tti','arm']
    numeric=f.select_dtypes(include='number').columns.difference(['seed','tti','repetition'])
    per=f.groupby(keys)[numeric].median().reset_index()
    per.to_csv(out/'per_state.csv',index=False)
    ref=per[per.arm=='physical4'].set_index('state_id')
    records=[]
    for arm,g in per.groupby('arm'):
        g=g.set_index('state_id');base=ref.loc[g.index]
        score_gap=(g.score-base.score)/base.score*100
        gain_ratio=(g.score-g.draft_score)/(base.score-base.draft_score).replace(0,np.nan)
        by_seed=g.groupby('seed').mean(numeric_only=True)
        records.append(dict(arm=arm,states=len(g),score=float(by_seed.score.mean()),
             mean_e2e_ms=float(by_seed.e2e_ms.mean()),p95_state_median_ms=float(g.e2e_ms.quantile(.95)),
             candidates=float(by_seed.candidate_evaluations.mean()),
             mean_relative_J_pct=float(score_gap.groupby(g.seed).mean().mean()),
             p05_relative_J_pct=float(score_gap.quantile(.05)),
             retained_gain_pct=float(gain_ratio.groupby(g.seed).mean().mean()*100),
             wins=int((g.score>base.score+1).sum()),losses=int((g.score<base.score-1).sum()),
             ties=int((np.abs(g.score-base.score)<=1).sum()),
             ue_evaluations=float(by_seed.ue_evaluations.mean()),mmse_misses=float(by_seed.mmse_misses.mean())))
    summary=pd.DataFrame(records);summary.to_csv(out/'summary.csv',index=False)
    escape=pd.read_csv(out/'escape.csv')
    escaped=escape.groupby(['root','arm']).agg(states=('gain','size'),escapes=('escaped','sum'),
                      mean_gain=('gain','mean'),max_gain=('gain','max'),mean_additional_ms=('additional_ms','mean')).reset_index()
    escaped.to_csv(out/'escape_summary.csv',index=False)
    print(summary.to_string(index=False),flush=True)
    print(escaped.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--training',required=True)
    parser.add_argument('--output')
    parser.add_argument('--repeats',type=int,default=3)
    main(parser.parse_args())

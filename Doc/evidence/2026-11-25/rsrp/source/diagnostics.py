"""Saved-state mechanisms; these are not extra closed-loop treatment arms."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
from training.stageb_ppo.native import write
from training.stageb_lns.archive import enable_legacy_types
from training.stageb_lns.no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from training.stageb_lns.physics import ReferencePhy
from training.stageb_lns.bounded_dpls import solve_bounded
from .analyze import local
from .model import approximate_problem


def main(args):
    import torch
    enable_legacy_types()
    root=Path(args.root);source=root/'campaign';output=root/'diagnostics'
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for done in sorted(source.glob('*/done.json')):
        spec=json.loads(done.read_text())
        if not spec['arm'].startswith('rsrp_'):continue
        d=local(spec['directory'])
        for entry in json.loads((d/'saved_states.json').read_text()):
            row=torch.load(local(entry['path']),map_location='cpu',weights_only=False)
            p=problem_from_raw_row(row);p.weights=np.asarray(row['objective_weights'])
            true_phy=ReferencePhy(p)
            action=row['selected_action'];draft=raw_pfq_draft(p,row['obs'])
            truth=true_phy.evaluate(action)[0];root_values=true_phy.evaluate(draft)[0]
            exact_action,exact_values,info=solve_bounded(p,draft);assert info['converged']
            active=truth[:,0]>0
            # Diagnostic 1 keeps instantaneous per-PRG total received power,
            # but removes only spatial covariance structure. It uses oracle H
            # and must never be represented as deployable RSRP telemetry.
            per_prg=np.square(np.abs(p.h)).sum(axis=(-1,-2))/4.0
            diag=np.arange(4);isotropic=np.zeros_like(p.h)
            isotropic[...,diag,diag]=np.sqrt(per_prg)[...,None]
            isotropic[np.arange(p.U),:,p.serving]=p.h[np.arange(p.U),:,p.serving]
            spatial=ReferencePhy(replace(p,h=isotropic)).evaluate(action)[0]
            # Diagnostic 2 additionally removes frequency detail, still using
            # an undelayed all-port scalar oracle rather than the actual proxy.
            wideband=np.square(np.abs(p.h)).mean(axis=(1,3,4))
            wide=ReferencePhy(approximate_problem(p,wideband)).evaluate(action)[0]
            approx=ReferencePhy(approximate_problem(p,row['rsrp_report'])).evaluate(action)[0]
            values=dict(scene=spec['scene'],seed=spec['seed'],arm=spec['arm'],tti=int(row['tti']),
                true_selected_weighted_bytes=float(truth[:,6]@p.weights),
                true_raw_pfq_weighted_bytes=float(root_values[:,6]@p.weights),
                true_exact_local_search_weighted_bytes=float(exact_values[:,6]@p.weights),
                approximate_predicted_weighted_bytes=float(approx[:,6]@p.weights),
                changed_positions_vs_exact=int(np.sum(action!=exact_action)))
            for name,prediction in [('spatial_only',spatial),('wideband',wide),('reported_rsrp',approx)]:
                error=prediction[active,4]-truth[active,4]
                values[name+'_sinr_bias_db']=float(error.mean())
                values[name+'_sinr_mae_db']=float(np.abs(error).mean())
            rows.append(values)
    assert len(rows)==36
    df=pd.DataFrame(rows);df.to_csv(output/'saved_state_mechanisms.csv',index=False)
    metric=[c for c in df if c not in ('scene','seed','arm','tti')]
    seed=df.groupby(['scene','seed','arm'],as_index=False)[metric].mean()
    seed.to_csv(output/'per_seed.csv',index=False)
    seed.groupby(['scene','arm'],as_index=False)[metric].mean().to_csv(output/'summary.csv',index=False)
    write(output/'validation.json',dict(passed=True,states=len(rows),
        scope='same selected actions; spatial/frequency diagnostics use oracle instantaneous H and are NOT closed-loop treatment arms',
        caveat='differences between diagnostic MAEs are not an additive causal decomposition; exact local search is not a global optimum'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);main(parser.parse_args())

"""Pay a separate full-neighborhood audit on captured native states."""
import argparse
import json
from pathlib import Path
import torch
from training.stageb_ppo.native import write
from training.stageb_lns.no_ctbs_inputs import problem_from_raw_row,raw_pfq_draft
from .analyze import local
from .oracle import audit

def main(source,output,all_states=False):
    source=Path(source);checks=[]
    for run in sorted(source.glob('*_s*')):
        if not (run/'done.json').exists() or run.name.startswith('pfq_'):continue
        files=sorted((run/'states').glob('*.pt'))
        for path in (files if all_states else files[:1]):
            row=torch.load(path,map_location='cpu',weights_only=False);p=problem_from_raw_row(row)
            p.weights=row['objective_weights'];result=audit(p,raw_pfq_draft(p,row['obs']))
            checks.append(dict(run=run.name,tti=row['tti'],**result));print('AUDIT',run.name,row['tti'],result,flush=True)
    assert checks
    write(output,dict(passed=True,states=len(checks),checks=checks))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--all',action='store_true');a=p.parse_args();main(a.source,a.output,a.all)

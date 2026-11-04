"""Frozen-model paired tests of direct neural and physical risk-aware repair."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .experiment import ROOT, sha, snapshot_sources, write_json


def progress(output, plan, completed):
    runs=[]
    horizon=sum(plan[k] for k in ('warmup','settling','measured','followup'))
    for seed in plan['seeds']:
        for arm in plan['arms']:
            directory=output/f'seed{seed}_{arm}'
            path=directory/'progress.json'
            value=json.loads(path.read_text()) if path.exists() else dict(tti=0,horizon=horizon,elapsed_s=0.)
            runs.append(dict(seed=seed,arm=arm,complete=(directory/'execution.json').exists(),**value))
    timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    lines=[f'风险感知直接神经修复配对实验（{timestamp}）',
           f'已退出 {len(completed)}/{len(runs)} 条轨迹；失败 {sum(r["returncode"]!=0 for r in completed)}。',
           'seeds104–106；2000预热＋2000稳定化＋10000测量＋400随访；测量为5秒。',
           '模型、阈值和候选流程按protocol.json冻结；完成后自动核验并生成final_report.txt。',
           '', 'seed arm tti/horizon native_verified']
    lines += [f'{r["seed"]} {r["arm"]} {r["tti"]}/{r["horizon"]} {r["complete"]}' for r in runs]
    temporary=output/'progress.tmp'
    temporary.write_text('\n'.join(lines)+'\n');temporary.replace(output/'progress.txt')


def main(args):
    plan_path=Path(args.plan).resolve();plan=json.loads(plan_path.read_text())
    seeds=plan['seeds'];arms=plan['arms']
    assert len(seeds)==len(set(seeds)) and len(arms)==len(set(arms))
    assert not set(seeds)&set(plan['training_seeds']+plan['selection_seeds']+plan['previous_test_seeds'])
    assert set(arms)=={'legacy_current','physical_lns','risk_physical_lns','risk_physical_direct','risk_neural'}
    assert plan['width']==16 and plan['followup']>=400
    checkpoint=ROOT/plan['checkpoint']
    assert sha(checkpoint)==plan['checkpoint_sha256']
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(plan_path,output/'plan_frozen.json')
    frozen=output/'repair_frozen.npz';shutil.copyfile(checkpoint,frozen)
    sources=snapshot_sources(output)
    protocol=dict(plan,workers=args.workers,plan_sha256=sha(plan_path),source_sha256=sources,
                  frozen_checkpoint=str(frozen),active_registry_sha256=sha(ROOT/'training/stageb_hgraph/current_algorithm.json'))
    write_json(output/'protocol.json',protocol)
    completed=[];progress(output,plan,completed)

    def one(seed,arm):
        name=f'seed{seed}_{arm}'
        command=[sys.executable,'-u','-m','training.stageb_lns.native_repair',
                 '--output',str(output/name),'--checkpoint',str(frozen),
                 '--seed',str(seed),'--arm',arm,'--width',str(plan['width']),
                 '--sample-interval','500',
                 '--training-seeds',','.join(map(str,plan['training_seeds'])),
                 '--model-selection-seeds',','.join(map(str,plan['selection_seeds']))]
        for key in ('warmup','settling','measured','followup'):
            command+=['--'+key,str(plan[key])]
        started=time.monotonic();print('START',name,flush=True)
        with (output/(name+'.log')).open('x') as log:
            code=subprocess.call(command,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
        result=dict(seed=seed,arm=arm,returncode=code,elapsed_s=time.monotonic()-started)
        print('DONE',result,flush=True)
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(one,seed,arm) for seed in seeds for arm in arms}
        while pending:
            done,pending=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            completed.extend(f.result() for f in done)
            progress(output,plan,completed)
    write_json(output/'completion.json',dict(passed=all(r['returncode']==0 for r in completed),runs=completed))
    if any(r['returncode']!=0 for r in completed):
        raise RuntimeError('Native trajectories failed; logs retained and no final comparison emitted')
    from .summarize_direct_risk import summarize
    summarize(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=15)
    main(parser.parse_args())

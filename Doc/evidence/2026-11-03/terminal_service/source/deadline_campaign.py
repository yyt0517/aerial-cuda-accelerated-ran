"""Versioned collection, validation and longer unseen-seed native campaigns."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import os
from pathlib import Path
import subprocess
import sys
import time
from .experiment import write_json,sha


def main(args):
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    old=Path(args.initial).resolve();new=Path(args.checkpoint).resolve() if args.checkpoint else None
    if args.stage=='collect':
        seeds=[49,50];arms=['neural','physical_lns'];warmup,settling,measured,followup=1000,1000,4000,400
    elif args.stage=='validation':
        seeds=[48,50];arms=['greedy','neural','risk_neural','risk_hybrid'];warmup,settling,measured,followup=1000,1000,2000,400
    else:
        seeds=[101,102,103];arms=['legacy_current','greedy','neural','risk_neural','risk_hybrid','risk_residual','neural_extra'];warmup,settling,measured,followup=2000,2000,10000,400
    if args.stage!='collect' and new is None:raise ValueError('A frozen checkpoint is required')
    write_json(out/'protocol.json',dict(stage=args.stage,seeds=seeds,arms=arms,warmup=warmup,
        settling=settling,measured=measured,followup=followup,initial_sha256=sha(old),
        checkpoint_sha256=None if new is None else sha(new),training_seeds=[41,42,43,46,47,49],
        selection_seeds=[44,45,48,50],test_seeds=[101,102,103],workers=args.workers,
        reward='goodput/target - 2*expired_packet_payload/target - 0.5*mean squared post-service deficit',
        wall_time_scope='Concurrent exploratory execution, not production latency',time_cutoff_ms=None))
    def one(seed,arm):
        name=f'seed{seed}_{arm}';checkpoint=new if arm.startswith('risk_') else old
        command=[sys.executable,'-u','-m','training.stageb_lns.native_repair','--output',str(out/name),
            '--checkpoint',str(checkpoint),'--seed',str(seed),'--arm',arm,'--warmup',str(warmup),
            '--settling',str(settling),'--measured',str(measured),'--followup',str(followup),
            '--sample-interval','100' if args.stage=='collect' else '500',
            '--training-seeds','41,42,43,46,47,49' if arm.startswith('risk_') else '41,42,43',
            '--model-selection-seeds','44,45,48,50' if arm.startswith('risk_') else '44,45']
        if arm in ('risk_residual','neural_extra'):command+=['--anchor-checkpoint',str(old)]
        started=time.monotonic();print('START',name,flush=True)
        with (out/(name+'.log')).open('x') as log:
            code=subprocess.call(command,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
        result=dict(seed=seed,arm=arm,returncode=code,elapsed_s=time.monotonic()-started)
        print('DONE',result,flush=True);return result
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(one,s,a) for s in seeds for a in arms]
        for future in as_completed(futures):results.append(future.result())
    write_json(out/'completion.json',dict(passed=all(r['returncode']==0 for r in results),runs=results))
    if not all(r['returncode']==0 for r in results):raise RuntimeError('Failed native runs; see retained logs')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--initial',required=True);p.add_argument('--checkpoint')
    p.add_argument('--stage',choices=('collect','validation','test'),required=True)
    p.add_argument('--workers',type=int,default=15);main(p.parse_args())

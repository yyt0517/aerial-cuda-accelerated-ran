"""Run a predeclared paired native campaign; failures stay visible."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .experiment import write_json, sha


def main(args):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    checkpoint=Path(args.checkpoint).resolve()
    seeds=[int(s) for s in args.seeds.split(',')]
    arms=args.arms.split(',')
    if set(seeds) & {41,42,43,44,45}:
        raise ValueError('Final native tests must not use training or selection seeds')
    protocol=dict(seeds=seeds, arms=arms, warmup=args.warmup, settling=args.settling,
        measured=args.measured,followup=args.followup, checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),width=16,workers=args.workers,
        selection='Frozen before native outcomes; no native-result tuning',
        timing_scope='Concurrent exploratory runs; not a real-time latency benchmark',
        inference='Finite-horizon paired native experiment; no steady-state or statistical-significance claim')
    write_json(output/'protocol.json',protocol)
    def one(seed,arm):
        name=f'seed{seed}_{arm}';out=output/name
        command=[sys.executable,'-u','-m','training.stageb_lns.native_repair','--output',str(out),
            '--checkpoint',str(checkpoint),'--seed',str(seed),'--arm',arm,
            '--warmup',str(args.warmup),'--settling',str(args.settling),
            '--measured',str(args.measured),'--followup',str(args.followup)]
        print('START',name,flush=True);start=time.monotonic()
        with (output/(name+'.log')).open('x') as log:
            code=subprocess.call(command,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
        result=dict(seed=seed,arm=arm,returncode=code,elapsed_s=time.monotonic()-start)
        print('FINISH',json.dumps(result),flush=True)
        return result
    # Interleave arms and seeds so no entire comparison family runs at one time.
    tasks=[(s,a) for s in seeds for a in arms]
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(one,s,a) for s,a in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    write_json(output/'completion.json',dict(passed=all(r['returncode']==0 for r in results),runs=results))
    if any(r['returncode'] for r in results):
        raise RuntimeError('Native campaign has failed runs; inspect retained logs')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--seeds',default='46,47,48')
    parser.add_argument('--arms',default='hybrid,neural,physical_lns,greedy,legacy_current,pfq0')
    parser.add_argument('--warmup',type=int,default=2000);parser.add_argument('--settling',type=int,default=1000)
    parser.add_argument('--measured',type=int,default=2000);parser.add_argument('--followup',type=int,default=400)
    parser.add_argument('--workers',type=int,default=6)
    main(parser.parse_args())

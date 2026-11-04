"""Thirty independent simulator processes; no learned agents or model tuning."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
import subprocess
import sys
import time
from common import ARMS,OUT,ROOT,SEEDS,path_for,sha,write


def worker(arm,seed):
    name=f'{arm}_seed{seed}'
    with (OUT/'logs'/f'{name}.log').open('x') as stream:
        subprocess.run([sys.executable,'-u',str(OUT/'run.py'),'--arm',arm,'--seed',str(seed)],
            cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    return dict(arm=arm,seed=seed,path=str(path_for(arm,seed)))


def main():
    assert json.loads((OUT/'preflight.json').read_text())['passed']
    protocol=json.loads((OUT/'protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():assert sha(OUT/name)==digest
    (OUT/'logs').mkdir(exist_ok=True)
    (OUT/'checks').mkdir(exist_ok=True)
    jobs=[(arm,seed) for seed in SEEDS for arm,config in ARMS.items() if not config['reused']]
    assert len(jobs)==30
    launch=dict(jobs=jobs,workers=30,pid=os.getpid(),protocol_sha256=sha(OUT/'protocol.json'),
        runtime_environment={key:value for key,value in os.environ.items() if key.startswith('CUDA_MPS_') or key=='CUDA_DEVICE_MAX_CONNECTIONS'},
        sources={name:sha(OUT/name) for name in ['campaign.py','validate.py']})
    with (OUT/'campaign_launch.json').open('x') as stream:json.dump(launch,stream,indent=2)
    started=time.monotonic();completed=[];errors=[]
    with ThreadPoolExecutor(max_workers=30) as pool:
        tasks={pool.submit(worker,*job):job for job in jobs}
        for future in as_completed(tasks):
            arm,seed=tasks[future]
            try:
                completed.append(future.result())
                print('NATIVE_FINISHED',arm,seed,flush=True)
            except Exception as error:
                errors.append(dict(arm=arm,seed=seed,error=str(error)))
                print('NATIVE_FAILED',arm,seed,str(error),flush=True)
            write(OUT/'campaign_status.json',dict(completed=completed,errors=errors,elapsed_s=time.monotonic()-started))
    assert not errors,errors
    write(OUT/'native_campaign_complete.json',dict(passed=True,new_runs=30,new_ttis=30*28400,
        reused_runs=10,completed=completed,elapsed_s=time.monotonic()-started))
    subprocess.run([sys.executable,'-u',str(OUT/'analyze.py')],cwd=ROOT,check=True)


if __name__=='__main__':main()

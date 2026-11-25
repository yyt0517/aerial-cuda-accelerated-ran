"""Process each completed simulation while remaining independent jobs are running."""
import json,time,sys,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from training.stage2_generalization.protocol import OUT
from training.stage2_generalization.analyze import one,aggregate

def main():
    dest=OUT/'analysis';dest.mkdir(exist_ok=True)
    known=set();pending={};failed=[];completed=0
    with ProcessPoolExecutor(max_workers=3) as pool:
        while True:
            for p in sorted((OUT/'campaign').glob('*/done.json')):
                if str(p) not in known:
                    known.add(str(p));pending[pool.submit(one,p,dest,True)]=str(p)
            if pending:
                ready,_=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
                for f in ready:
                    p=pending.pop(f)
                    try:
                        r=f.result();completed+=1
                        print('ANALYZED',completed,r['key'],'payload',round(r['payload_mbps'],3),'expiry',round(r['expiry_pct'],3),flush=True)
                    except BaseException as exc:
                        import traceback
                        detail=dict(path=p,error=traceback.format_exc());failed.append(detail);print('FAILED',detail,flush=True)
                    (dest/'watch_progress.json').write_text(json.dumps(dict(completed=completed,failed=failed,queued=len(pending)),indent=2))
            else:time.sleep(20)
            if len(known)==60 and not pending:break
            if (OUT/'completion.json').exists():
                campaign=json.loads((OUT/'completion.json').read_text())
                if campaign.get('failed') and not pending:raise RuntimeError('Campaign has failures; inspect completion.json')
    if failed:raise RuntimeError('Analysis failures: '+str(failed))
    aggregate(dest,smoke=False,partial=False)
    print('ALL 60 VERIFIED',flush=True)
if __name__=='__main__':main()
